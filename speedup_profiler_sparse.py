#!/usr/bin/env python3
"""
Sparse Speedup Profiler for ClickHouse Queries

Similar to speedup_profiler.py but samples at scattered thread counts (e.g., 1,2,4,8,12,16,...)
and uses speedup_fitcurve to fit a scalability model and interpolate the full speedup curve.

Output is saved in a timestamped folder containing:
- speedup.csv: The fitted speedup curve
- runtime.txt: Average runtimes for all sampled core counts
- metadata.txt: Run configuration (model, sample points, fitted parameters, etc.)

Supported models:
- amdahl: Amdahl's Law (default)
- usl: Universal Scalability Law

Usage:
    # Test a single query file
    python speedup_profiler_sparse.py --query queries/p_greedy_108.sql --max-threads 64 --repeat 5
    
    # Test all .sql files in a directory
    python speedup_profiler_sparse.py --dir queries/ --max-threads 64 --repeat 5
    
    # Test with inline query string (no output file saved)
    python speedup_profiler_sparse.py --query-string "SELECT count(*) FROM lineitem" --max-threads 64
    
    # Custom sample points
    python speedup_profiler_sparse.py --query q.sql --sample-points 1,2,4,8,16,32,64
    
    # Use USL model instead of Amdahl
    python speedup_profiler_sparse.py --query q.sql --model usl
    
    # Adjust plateau threshold (default 0.001 = 0.1% improvement threshold)
    python speedup_profiler_sparse.py --query q.sql --plateau-threshold 0.1
"""

import argparse
import glob
import os
import time
import sys
from datetime import datetime
from typing import List, Tuple
import urllib.request
import urllib.parse

# Import curve fitting functionality
from speedup_fitcurve import fit_speedup_curve


# Default sample points for sparse profiling
DEFAULT_SAMPLE_POINTS = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]


def read_query_file(filepath: str) -> str:
    """Read SQL query from file."""
    with open(filepath, 'r') as f:
        return f.read().strip()


def run_query_with_threads(host: str, port: int, query: str, max_threads: int) -> float:
    """
    Run a query with specified max_threads setting and return execution time in seconds.
    Uses ClickHouse HTTP interface.
    """
    params = urllib.parse.urlencode({'max_threads': max_threads})
    url = f"http://{host}:{port}/?{params}"
    
    start_time = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=query.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req) as response:
            response.read()
    except Exception as e:
        print(f"  Error running query with {max_threads} threads: {e}", file=sys.stderr)
        return float('inf')
    end_time = time.perf_counter()
    
    return end_time - start_time


def measure_speedup_sparse(
    host: str,
    port: int,
    query: str,
    sample_points: List[int],
    repeat: int,
    warmup: int = 3,
    verbose: bool = True
) -> Tuple[List[int], List[float], List[float]]:
    """
    Measure query execution times at sparse sample points.
    
    Returns:
        thread_counts: The thread counts actually sampled
        speedups: Speedup values at each sample point
        avg_times: Average execution times at each sample point
    """
    avg_times = []
    
    # Warmup runs
    if warmup > 0 and verbose:
        print(f"Running {warmup} warmup iteration(s)...")
        max_sample = max(sample_points)
        for _ in range(warmup):
            run_query_with_threads(host, port, query, max_sample)
    
    # Measure at each sample point
    for num_threads in sample_points:
        times = []
        if verbose:
            print(f"Testing with {num_threads} thread(s)...", end=" ", flush=True)
        
        for rep in range(repeat):
            elapsed = run_query_with_threads(host, port, query, num_threads)
            times.append(elapsed)
            if verbose:
                print(f"{elapsed:.3f}s", end=" ", flush=True)
        
        avg_time = sum(times) / len(times)
        avg_times.append(avg_time)
        
        if verbose:
            print(f"-> avg: {avg_time:.3f}s")
    
    # Calculate speedups (relative to 1 thread)
    baseline = avg_times[0]  # average time with 1 thread
    speedups = [baseline / t if t > 0 else 0.0 for t in avg_times]
    
    return sample_points, speedups, avg_times


def create_output_folder(base_path: str, model: str) -> str:
    """
    Create a timestamped output folder for sparse profiler results.
    
    Returns the path to the created folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_path}_sparse_{model}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def process_query_file(
    query_file: str, 
    host: str, 
    port: int,
    sample_points: List[int],
    repeat: int, 
    warmup: int,
    plateau_threshold: float,
    model: str,
    verbose: bool
) -> List[float]:
    """Process a single query file and save speedup results to timestamped folder."""
    query = read_query_file(query_file)
    
    # Filter sample points to be within max_threads and include 1
    valid_samples = sorted(set([1] + [p for p in sample_points]))
    max_threads = max(valid_samples)
    
    if verbose:
        print(f"Query: {query_file}")
        print(f"ClickHouse: {host}:{port}")
        print(f"Max threads: {max_threads}")
        print(f"Sample points: {valid_samples}")
        print(f"Repetitions per sample: {repeat}")
        print(f"Warmup iterations: {warmup}")
        print(f"Plateau threshold: {plateau_threshold:.1%}")
        print(f"Model: {model}")
        print("-" * 60)
    
    # Run sparse measurements
    thread_counts, speedups_sampled, avg_times = measure_speedup_sparse(
        host, port, query, valid_samples, repeat, warmup, verbose
    )
    
    if verbose:
        print("-" * 60)
        print("SAMPLED DATA")
        print("-" * 60)
        print(f"{'Threads':<10} {'Avg Time (s)':<15} {'Speedup':<10}")
        print("-" * 35)
        for tc, t, s in zip(thread_counts, avg_times, speedups_sampled):
            print(f"{tc:<10} {t:<15.4f} {s:<10.2f}")
        print()
    
    # Use speedup_fitcurve to fit and generate full curve
    full_speedups, metadata = fit_speedup_curve(
        thread_counts, speedups_sampled,
        max_threads, plateau_threshold, model
    )
    
    if verbose:
        if metadata['has_plateau']:
            print(f"Plateau detected at {metadata['plateau_thread']} threads (speedup: {metadata['plateau_speedup']:.2f})")
        else:
            print("No plateau detected")
        
        # Print model-specific parameters
        params = metadata['params']
        if model == 'amdahl':
            print(f"Fitted serial fraction (s): {params['s']:.4f}")
            theoretical_max = 1.0 / params['s'] if params['s'] > 0 else float('inf')
            print(f"Theoretical max speedup: {theoretical_max:.2f}")
        elif model == 'usl':
            print(f"Fitted sigma (contention): {params['sigma']:.6f}")
            print(f"Fitted kappa (coherency): {params['kappa']:.6f}")
    
    # Print fitted results
    if verbose:
        print("-" * 60)
        print("FITTED SPEEDUP CURVE (1 to {})".format(max_threads))
        print("-" * 60)
        # Print a subset for readability
        display_points = [1, 2, 4, 8, 16, 32, 64]
        display_points = [p for p in display_points if p <= max_threads]
        print(f"{'Threads':<10} {'Fitted Speedup':<15}")
        print("-" * 25)
        for p in display_points:
            print(f"{p:<10} {full_speedups[p-1]:<15.2f}")
        print()
    
    # Output speedup list
    speedup_str = ", ".join(f"{s:.2f}" for s in full_speedups)
    print(f"Speedup list: [{speedup_str}]")
    
    # Create timestamped output folder
    base_path = os.path.splitext(query_file)[0]
    output_folder = create_output_folder(base_path, model)
    
    if verbose:
        print(f"\nOutput folder: {output_folder}")
    
    # Save speedup values
    speedup_path = os.path.join(output_folder, "speedup.csv")
    with open(speedup_path, 'w') as f:
        for s in full_speedups:
            f.write(f"{s:.4f}\n")
    
    if verbose:
        print(f"Speedup values saved to: {speedup_path}")
    
    # Save ALL sampled runtimes (cores and avg_time for each sampled point)
    runtime_path = os.path.join(output_folder, "runtime.txt")
    with open(runtime_path, 'w') as f:
        f.write("# cores avg_runtime_seconds\n")
        for tc, avg_time in zip(thread_counts, avg_times):
            f.write(f"{tc} {avg_time:.4f}\n")
    
    if verbose:
        print(f"All sampled runtimes saved to: {runtime_path}")
    
    # Save comprehensive metadata
    meta_path = os.path.join(output_folder, "metadata.txt")
    with open(meta_path, 'w') as f:
        f.write(f"# Sparse Speedup Profiler Run Metadata\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
        
        f.write(f"[configuration]\n")
        f.write(f"query_file: {query_file}\n")
        f.write(f"host: {host}\n")
        f.write(f"port: {port}\n")
        f.write(f"repeat: {repeat}\n")
        f.write(f"warmup: {warmup}\n")
        f.write(f"plateau_threshold: {plateau_threshold}\n\n")
        
        f.write(f"[sampling]\n")
        f.write(f"sample_points: {thread_counts}\n")
        f.write(f"max_threads: {max_threads}\n")
        f.write(f"sampled_speedups: {[round(s, 4) for s in speedups_sampled]}\n")
        f.write(f"sampled_runtimes: {[round(t, 4) for t in avg_times]}\n\n")
        
        f.write(f"[model]\n")
        f.write(f"model: {metadata['model']}\n")
        for key, value in metadata['params'].items():
            f.write(f"{key}: {value:.6f}\n")
        f.write(f"\n")
        
        f.write(f"[plateau]\n")
        f.write(f"has_plateau: {metadata['has_plateau']}\n")
        f.write(f"plateau_thread: {metadata['plateau_thread']}\n")
        f.write(f"plateau_speedup: {metadata['plateau_speedup']:.4f}\n")
    
    if verbose:
        print(f"Metadata saved to: {meta_path}")
    
    return full_speedups


def process_query_string(
    query: str, 
    host: str, 
    port: int,
    sample_points: List[int],
    repeat: int, 
    warmup: int,
    plateau_threshold: float,
    model: str,
    verbose: bool
) -> List[float]:
    """Process an inline query string (no output file saved)."""
    # Filter sample points
    valid_samples = sorted(set([1] + [p for p in sample_points]))
    max_threads = max(valid_samples)
    
    if verbose:
        print(f"Query: inline query")
        print(f"ClickHouse: {host}:{port}")
        print(f"Max threads: {max_threads}")
        print(f"Sample points: {valid_samples}")
        print(f"Repetitions per sample: {repeat}")
        print(f"Warmup iterations: {warmup}")
        print(f"Plateau threshold: {plateau_threshold:.1%}")
        print(f"Model: {model}")
        print("-" * 60)
    
    # Run sparse measurements
    thread_counts, speedups_sampled, avg_times = measure_speedup_sparse(
        host, port, query, valid_samples, repeat, warmup, verbose
    )
    
    if verbose:
        print("-" * 60)
        print("SAMPLED DATA")
        print("-" * 60)
        print(f"{'Threads':<10} {'Avg Time (s)':<15} {'Speedup':<10}")
        print("-" * 35)
        for tc, t, s in zip(thread_counts, avg_times, speedups_sampled):
            print(f"{tc:<10} {t:<15.4f} {s:<10.2f}")
        print()
    
    # Use speedup_fitcurve to fit and generate full curve
    full_speedups, metadata = fit_speedup_curve(
        thread_counts, speedups_sampled,
        max_threads, plateau_threshold, model
    )
    
    if verbose:
        if metadata['has_plateau']:
            print(f"Plateau detected at {metadata['plateau_thread']} threads (speedup: {metadata['plateau_speedup']:.2f})")
        else:
            print("No plateau detected")
        
        # Print model-specific parameters
        params = metadata['params']
        if model == 'amdahl':
            print(f"Fitted serial fraction (s): {params['s']:.4f}")
            theoretical_max = 1.0 / params['s'] if params['s'] > 0 else float('inf')
            print(f"Theoretical max speedup: {theoretical_max:.2f}")
        elif model == 'usl':
            print(f"Fitted sigma (contention): {params['sigma']:.6f}")
            print(f"Fitted kappa (coherency): {params['kappa']:.6f}")
    
    # Print fitted results
    if verbose:
        print("-" * 60)
        print("FITTED SPEEDUP CURVE (1 to {})".format(max_threads))
        print("-" * 60)
        display_points = [1, 2, 4, 8, 16, 32, 64]
        display_points = [p for p in display_points if p <= max_threads]
        print(f"{'Threads':<10} {'Fitted Speedup':<15}")
        print("-" * 25)
        for p in display_points:
            print(f"{p:<10} {full_speedups[p-1]:<15.2f}")
        print()
    
    # Output speedup list
    speedup_str = ", ".join(f"{s:.2f}" for s in full_speedups)
    print(f"Speedup list: [{speedup_str}]")
    
    return full_speedups


def parse_sample_points(s: str) -> List[int]:
    """Parse comma-separated sample points string."""
    return sorted(set(int(x.strip()) for x in s.split(',')))


def main():
    parser = argparse.ArgumentParser(
        description="Sparse speedup profiler with scalability model curve fitting"
    )
    
    # Query specification (one of these required)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", "-q", help="Path to SQL query file")
    query_group.add_argument("--dir", "-d", help="Directory containing .sql query files (tests all)")
    query_group.add_argument("--query-string", "-Q", help="SQL query string directly")
    
    # Test parameters
    parser.add_argument("--sample-points", "-s", type=str, 
                        default="1,2,4,8,12,16,24,32,48,64",
                        help="Comma-separated thread counts to sample (default: 1,2,4,8,12,16,24,32,48,64)")
    parser.add_argument("--repeat", "-r", type=int, default=8,
                        help="Number of repetitions per sample point (default: 8)")
    parser.add_argument("--warmup", "-w", type=int, default=1,
                        help="Number of warmup runs before measurement (default: 1)")
    
    # Fitting parameters
    parser.add_argument("--plateau-threshold", "-p", type=float, default=0.001,
                        help="Relative speedup improvement threshold for plateau detection (default: 0.001)")
    parser.add_argument("--model", "-m", choices=['amdahl', 'usl'], default='usl',
                        help="Scalability model to fit: amdahl or usl (default: usl)")
    
    # ClickHouse connection
    parser.add_argument("--host", default="localhost", help="ClickHouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port (default: 8123)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Parse sample points
    sample_points = parse_sample_points(args.sample_points)
    
    # Collect query files to process
    if args.dir:
        pattern = os.path.join(args.dir, "*.sql")
        query_files = sorted(glob.glob(pattern))
        if not query_files:
            print(f"No .sql files found in {args.dir}", file=sys.stderr)
            sys.exit(1)
        if verbose:
            print(f"Found {len(query_files)} query files in {args.dir}")
            print("=" * 60)
    elif args.query:
        query_files = [args.query]
    else:
        query_files = None  # inline query string
    
    # Process each query file
    if query_files:
        for query_file in query_files:
            process_query_file(
                query_file, args.host, args.port, 
                sample_points, args.repeat, args.warmup,
                args.plateau_threshold, args.model, verbose
            )
            if verbose and len(query_files) > 1:
                print()
    else:
        # Inline query string
        process_query_string(
            args.query_string, args.host, args.port,
            sample_points, args.repeat, args.warmup,
            args.plateau_threshold, args.model, verbose
        )


if __name__ == "__main__":
    main()
