#!/usr/bin/env python3
"""
Generate Poisson arrival schedule from all queries in a directory.

Arrivals follow a Poisson process at a fixed rate. Each arrival randomly
selects a query uniformly from all available queries in the specified directory.

Usage:
    # Single schedule via CLI:
    python poisson_schedule_generator.py --query-dir ./queries --rate 0.5 --length 3600

    # Batch mode using SCHEDULES config:
    python poisson_schedule_generator.py --batch
"""

import argparse
import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import List

# =============================================================================
# CONFIGURATION - Define your schedules here (used with --batch flag)
# =============================================================================

@dataclass
class ScheduleConfig:
    arrival_rate: float   # Queries per second (QPS)
    length: float         # Duration of schedule in seconds
    query_dir: str        # Directory containing query SQL files
    seed: int = 123       # RNG seed

# Define all schedules to generate in batch mode
SCHEDULES: List[ScheduleConfig] = [
    ScheduleConfig(arrival_rate=0.18, length=300, query_dir="queries/extreme_aws/greedy"), 
    ScheduleConfig(arrival_rate=0.24, length=300, query_dir="queries/extreme_aws/greedy"),
    ScheduleConfig(arrival_rate=0.27, length=300, query_dir="queries/extreme_aws/greedy"),
    ScheduleConfig(arrival_rate=0.32, length=300, query_dir="queries/extreme_aws/greedy"),
    # Add more schedules as needed:
    # ScheduleConfig(arrival_rate=0.2, length=3600, query_dir="./queries"),
    # ScheduleConfig(arrival_rate=0.5, length=1800, query_dir="./queries/fast"),
]

# Output directory for generated schedules
SCHEDULES_DIR = "./schedules"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class Event:
    at_ms: int
    qid: str


def load_queries(query_dir: str) -> List[str]:
    """
    Load all query names from a directory.
    
    Returns list of query names (filenames without .sql extension).
    """
    pattern = os.path.join(query_dir, "*.sql")
    queries = []
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        query_name = filename[:-4] if filename.endswith(".sql") else filename
        queries.append(query_name)
    return queries


def gen_poisson_schedule(
    arrival_rate: float,
    length: float,
    queries: List[str],
    rng: random.Random
) -> List[Event]:
    """
    Generate Poisson arrivals with uniform query selection.
    
    Args:
        arrival_rate: Arrival rate in queries per second (QPS)
        length: Total duration in seconds
        queries: List of query IDs to choose from
        rng: Random number generator
    
    Returns:
        List of Events sorted by arrival time
    """
    events = []
    t = 0.0
    
    if arrival_rate <= 0:
        return events
    
    while True:
        t += rng.expovariate(arrival_rate)
        if t > length:
            break
        qid = rng.choice(queries)
        at_ms = int(round(t * 1000.0))
        events.append(Event(at_ms=at_ms, qid=qid))
    
    # Sort by timestamp; tie-break by qid for stable output
    events.sort(key=lambda e: (e.at_ms, e.qid))
    return events


def write_schedule(events: List[Event], output_path: str):
    """Write schedule to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["at_ms", "qid"])
        for e in events:
            w.writerow([e.at_ms, e.qid])


def generate_output_filename(
    config: ScheduleConfig, 
    index: int = None,
    schedules_dir: str = SCHEDULES_DIR
) -> str:
    """Generate output filename from config parameters."""
    rate_str = f"{config.arrival_rate:.6f}".rstrip('0').rstrip('.')
    length_str = f"{config.length:.0f}"
    dir_name = os.path.basename(os.path.normpath(config.query_dir))
    
    if index is not None:
        filename = f"{index:03d}_{dir_name}_rate{rate_str}_len{length_str}_s{config.seed}.csv"
    else:
        filename = f"{dir_name}_rate{rate_str}_len{length_str}_s{config.seed}.csv"
    
    return os.path.join(schedules_dir, filename)


def generate_from_config(config: ScheduleConfig, index: int = None) -> bool:
    """
    Generate schedule from config.
    
    Args:
        config: Schedule configuration
        index: Optional index for filename prefix (used in batch mode)
    
    Returns:
        True on success, False on failure
    """
    queries = load_queries(config.query_dir)
    if not queries:
        print(f"  Error: No queries found in {config.query_dir}")
        return False
    
    print(f"  Query directory: {config.query_dir}")
    print(f"  Queries found: {len(queries)}")
    print(f"  Arrival rate: {config.arrival_rate} QPS")
    print(f"  Duration: {config.length}s")
    
    rng = random.Random(config.seed)
    events = gen_poisson_schedule(config.arrival_rate, config.length, queries, rng)
    
    output_path = generate_output_filename(config, index)
    write_schedule(events, output_path)
    
    print(f"  Generated {len(events)} events -> {output_path}")
    return True


def run_batch_mode():
    """Generate all schedules defined in SCHEDULES config."""
    print("=" * 70)
    print("Poisson Schedule Generator - Batch Mode")
    print("=" * 70)
    
    if not SCHEDULES:
        print("No schedules defined in SCHEDULES config.")
        return
    
    # Clear existing schedules
    if os.path.exists(SCHEDULES_DIR):
        for f in glob.glob(os.path.join(SCHEDULES_DIR, "*.csv")):
            os.remove(f)
    os.makedirs(SCHEDULES_DIR, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(SCHEDULES, 1):
        print(f"\n[{i}/{len(SCHEDULES)}] rate={config.arrival_rate} QPS, "
              f"length={config.length}s, dir={config.query_dir}")
        print("-" * 50)
        
        if generate_from_config(config, index=i):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 70)
    print(f"Done! Generated {success_count} schedule(s), {fail_count} failed")
    print("=" * 70)
    
    return fail_count == 0


def run_single_mode(args):
    """Generate a single schedule from CLI arguments."""
    queries = load_queries(args.query_dir)
    if not queries:
        print(f"Error: No queries found in {args.query_dir}")
        return False
    
    print(f"Loaded {len(queries)} queries from {args.query_dir}")
    for q in queries:
        print(f"  - {q}")
    
    rng = random.Random(args.seed)
    events = gen_poisson_schedule(args.rate, args.length, queries, rng)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        config = ScheduleConfig(
            arrival_rate=args.rate,
            length=args.length,
            query_dir=args.query_dir,
            seed=args.seed
        )
        output_path = generate_output_filename(config)
    
    write_schedule(events, output_path)
    print(f"\nGenerated {len(events)} events -> {output_path}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Generate Poisson schedule with uniform query selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single schedule:
  python poisson_schedule_generator.py --query-dir ./queries --rate 0.5 --length 3600

  # Generate all schedules from SCHEDULES config:
  python poisson_schedule_generator.py --batch

  # Specify custom output path:
  python poisson_schedule_generator.py --query-dir ./queries --rate 0.5 --length 3600 --output my_schedule.csv
"""
    )
    ap.add_argument("--query-dir", type=str, 
                    help="Directory containing query SQL files")
    ap.add_argument("--rate", type=float, 
                    help="Arrival rate in queries per second (QPS)")
    ap.add_argument("--length", type=float, 
                    help="Duration of schedule in seconds")
    ap.add_argument("--seed", type=int, default=123, 
                    help="RNG seed (default: 123)")
    ap.add_argument("--output", type=str, 
                    help="Output CSV path (auto-generated if not specified)")
    ap.add_argument("--batch", action="store_true",
                    help="Generate all schedules from SCHEDULES config")
    args = ap.parse_args()

    # Batch mode: generate from SCHEDULES config
    if args.batch:
        success = run_batch_mode()
        return 0 if success else 1

    # If no arguments provided, default to batch mode
    if not args.query_dir and not args.rate and not args.length:
        print("No arguments provided. Running in batch mode...")
        print("(Use --help to see CLI options)\n")
        success = run_batch_mode()
        return 0 if success else 1

    # Single schedule mode - validate required args
    if not all([args.query_dir, args.rate, args.length]):
        ap.error("--query-dir, --rate, and --length are all required for single mode")

    success = run_single_mode(args)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
