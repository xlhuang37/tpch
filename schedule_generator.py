#!/usr/bin/env python3
"""
Generate multiple Poisson schedules by specifying configurations at the top of the file.

Load is defined as: arrival_rate × CPU_seconds
where CPU_seconds is extracted from the query name (e.g., np_greedy_300 → 300).

Query files are discovered from ./queries directory. Filenames must follow
the pattern: {np|p}_{type}_{cpu_seconds}.sql (e.g., np_greedy_300.sql)
"""

import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List

# =============================================================================
# CONFIGURATION - Define your schedules here
# =============================================================================

@dataclass
class ScheduleConfig:
    load: float          # Total load = lam_np * cpu_np + lam_p * cpu_p
    ratio: float         # Ratio of arrival rates = lam_np / lam_p
    length: float        # Duration of schedule in seconds
    query_type: str      # "greedy" or "default"
    p_size: int          # CPU seconds of p query (e.g., 1200 or 2400)
    np_size: int = 300   # CPU seconds of np query (default: 300)
    seed: int = 123      # RNG seed

# Define all schedules to generate
SCHEDULES: List[ScheduleConfig] = [
    # Load Config
    ScheduleConfig(load=72, ratio=1.0, length=900, query_type="greedy", p_size=94, np_size=113),
    ScheduleConfig(load=72, ratio=1.0, length=900, query_type="default", p_size=94, np_size=113),
    ScheduleConfig(load=48, ratio=1.0, length=900, query_type="greedy", p_size=94, np_size=113),
    ScheduleConfig(load=48, ratio=1.0, length=900, query_type="default", p_size=94, np_size=113),
    ScheduleConfig(load=24, ratio=1.0, length=900, query_type="greedy", p_size=94, np_size=113),
    ScheduleConfig(load=24, ratio=1.0, length=900, query_type="default", p_size=94, np_size=113),
    # # Load Config
    # ScheduleConfig(load=36, ratio=2.0, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=24, ratio=2.0, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=12, ratio=2.0, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=36, ratio=2.0, length=120, query_type="default", p_size=1200, np_size=300),
    # ScheduleConfig(load=24, ratio=2.0, length=120, query_type="default", p_size=1200, np_size=300),
    # ScheduleConfig(load=12, ratio=2.0, length=120, query_type="default", p_size=1200, np_size=300),
    # # 
    # ScheduleConfig(load=36, ratio=2.0, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=24, ratio=2.0, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=12, ratio=2.0, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=36, ratio=2.0, length=120, query_type="default", p_size=2400, np_size=300),
    # ScheduleConfig(load=24, ratio=2.0, length=120, query_type="default", p_size=2400, np_size=300),
    # ScheduleConfig(load=12, ratio=2.0, length=120, query_type="default", p_size=2400, np_size=300),
    # # Load Config
    # ScheduleConfig(load=36, ratio=4.0, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=24, ratio=4.0, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=12, ratio=4.0, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=36, ratio=4.0, length=120, query_type="default", p_size=1200, np_size=300),
    # ScheduleConfig(load=24, ratio=4.0, length=120, query_type="default", p_size=1200, np_size=300),
    # ScheduleConfig(load=12, ratio=4.0, length=120, query_type="default", p_size=1200, np_size=300),
    # # 
    # ScheduleConfig(load=36, ratio=4.0, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=24, ratio=4.0, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=12, ratio=4.0, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=36, ratio=4.0, length=120, query_type="default", p_size=2400, np_size=300),
    # ScheduleConfig(load=24, ratio=4.0, length=120, query_type="default", p_size=2400, np_size=300),
    # ScheduleConfig(load=12, ratio=4.0, length=120, query_type="default", p_size=2400, np_size=300),
    # # Load Config
    # ScheduleConfig(load=36, ratio=0.5, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=24, ratio=0.5, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=12, ratio=0.5, length=120, query_type="greedy", p_size=1200, np_size=300),
    # ScheduleConfig(load=36, ratio=0.5, length=120, query_type="default", p_size=1200, np_size=300),
    # ScheduleConfig(load=24, ratio=0.5, length=120, query_type="default", p_size=1200, np_size=300),
    # ScheduleConfig(load=12, ratio=0.5, length=120, query_type="default", p_size=1200, np_size=300),
    # # 
    # ScheduleConfig(load=36, ratio=0.5, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=24, ratio=0.5, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=12, ratio=0.5, length=120, query_type="greedy", p_size=2400, np_size=300),
    # ScheduleConfig(load=36, ratio=0.5, length=120, query_type="default", p_size=2400, np_size=300),
    # ScheduleConfig(load=24, ratio=0.5, length=120, query_type="default", p_size=2400, np_size=300),
    # ScheduleConfig(load=12, ratio=0.5, length=120, query_type="default", p_size=2400, np_size=300),
]

# Directory containing query SQL files
QUERIES_DIR = "./queries"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def load_query_cpu_seconds(queries_dir: str = "./queries") -> dict[str, int]:
    """
    Scan the queries directory and parse filenames to extract CPU seconds.
    
    Filename pattern: {np|p}_{greedy|default}_{cpu_seconds}.sql
    Example: np_greedy_300.sql → {"np_greedy_300": 300}
    """
    query_cpu_seconds = {}
    pattern = os.path.join(queries_dir, "*.sql")
    
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        # Remove .sql extension
        query_name = filename[:-4] if filename.endswith(".sql") else filename
        
        # Parse the filename: expected format is {prefix}_{type}_{cpu_seconds}
        # e.g., np_greedy_300, p_default_1200
        match = re.match(r"^(.+)_(\d+)$", query_name)
        if match:
            cpu_seconds = int(match.group(2))
            query_cpu_seconds[query_name] = cpu_seconds
    
    return query_cpu_seconds


def compute_arrival_rates(total_load: float, ratio: float, cpu1: int, cpu2: int) -> tuple[float, float]:
    """
    Compute arrival rates from total load and ratio.
    
    Given:
        total_load = lam1 * cpu1 + lam2 * cpu2
        ratio = lam1 / lam2
    
    Solve:
        lam2 = total_load / (ratio * cpu1 + cpu2)
        lam1 = ratio * lam2
    """
    lam2 = total_load / (ratio * cpu1 + cpu2)
    lam1 = ratio * lam2
    return lam1, lam2


def generate_schedule(config: ScheduleConfig, query_cpu_seconds: dict[str, int], index: int) -> bool:
    """Generate a single schedule. Returns True on success."""
    # Build query IDs
    qid_np = f"np_{config.query_type}_{config.np_size}"
    qid_p = f"p_{config.query_type}_{config.p_size}"

    # Validate queries exist
    if qid_np not in query_cpu_seconds:
        print(f"  Error: Query '{qid_np}' not found")
        print(f"  Available: {', '.join(sorted(query_cpu_seconds.keys()))}")
        return False
    if qid_p not in query_cpu_seconds:
        print(f"  Error: Query '{qid_p}' not found")
        print(f"  Available: {', '.join(sorted(query_cpu_seconds.keys()))}")
        return False

    # Get CPU seconds
    cpu_np = query_cpu_seconds[qid_np]
    cpu_p = query_cpu_seconds[qid_p]

    # Compute arrival rates
    lam_np, lam_p = compute_arrival_rates(config.load, config.ratio, cpu_np, cpu_p)

    # Validate rates are positive
    if lam_np <= 0 or lam_p <= 0:
        print(f"  Error: Invalid arrival rates (lam_np={lam_np}, lam_p={lam_p})")
        return False

    # Print summary
    print(f"  NP: {qid_np} @ {lam_np:.6f} QPS (load: {lam_np * cpu_np:.4f})")
    print(f"  P:  {qid_p} @ {lam_p:.6f} QPS (load: {lam_p * cpu_p:.4f})")
    print(f"  Total load: {lam_np * cpu_np + lam_p * cpu_p:.4f}")

    # Build command with prefix for ordering (e.g., "001_", "002_", etc.)
    prefix = f"{index:03d}_"
    cmd = [
        sys.executable, "poisson_schedule_generator.py",
        "--qid1", qid_np,
        "--qid2", qid_p,
        "--lam1", str(lam_np),
        "--lam2", str(lam_p),
        "--length", str(config.length),
        "--seed", str(config.seed),
        "--prefix", prefix,
    ]

    result = subprocess.run(cmd, cwd=".")
    return result.returncode == 0


def main():
    print("=" * 70)
    print("Schedule Generator")
    print("=" * 70)
    
    # Load query CPU seconds from directory
    query_cpu_seconds = load_query_cpu_seconds(QUERIES_DIR)
    
    if not query_cpu_seconds:
        print(f"Error: No query files found in {QUERIES_DIR}")
        sys.exit(1)
    
    print(f"Found {len(query_cpu_seconds)} queries in {QUERIES_DIR}:")
    for qid, cpu in sorted(query_cpu_seconds.items()):
        print(f"  {qid}: {cpu} CPU seconds")
    print()
    
    print(f"Generating {len(SCHEDULES)} schedule(s)...")
    print("=" * 70)
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(SCHEDULES, 1):
        print(f"\n[{i}/{len(SCHEDULES)}] load={config.load}, ratio={config.ratio}, "
              f"length={config.length}s, type={config.query_type}, p_size={config.p_size}")
        print("-" * 50)
        
        if generate_schedule(config, query_cpu_seconds, index=i):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 70)
    print(f"Done! Generated {success_count} schedule(s), {fail_count} failed")
    print("=" * 70)
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
