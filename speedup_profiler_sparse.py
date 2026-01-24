#!/usr/bin/env python3
"""
Sparse Speedup Profiler for ClickHouse Queries

Similar to speedup_profiler.py but samples at scattered thread counts (e.g., 1,2,4,8,12,16,...)
and fits Amdahl's law to interpolate the full speedup curve.

Amdahl's Law: Speedup(N) = 1 / (s + (1-s)/N)
where s is the serial fraction of the workload.

If the speedup increment becomes too low (plateau detected), the fitted curve
is flattened to that plateau value.

Usage:
    # Test a single query file
    python speedup_profiler_sparse.py --query queries/p_greedy_108.sql --max-threads 64 --repeat 5
    
    # Test all .sql files in a directory
    python speedup_profiler_sparse.py --dir queries/ --max-threads 64 --repeat 5
    
    # Test with inline query string (no output file saved)
    python speedup_profiler_sparse.py --query-string "SELECT count(*) FROM lineitem" --max-threads 64
    
    # Custom sample points
    python speedup_profiler_sparse.py --query q.sql --sample-points 1,2,4,8,16,32,64
    
    # Adjust plateau threshold (default 0.05 = 5% improvement threshold)
    python speedup_profiler_sparse.py --query q.sql --plateau-threshold 0.1
"""

import argparse
import glob
import os
import time
import sys
from typing import List, Tuple, Optional
import urllib.request
import urllib.parse

try:
    import numpy as np
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy/numpy not available. Install with: pip install scipy numpy", file=sys.stderr)


# Default sample points for sparse profiling
DEFAULT_SAMPLE_POINTS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]


def amdahl_law(n: np.ndarray, s: float) -> np.ndarray:
    """
    Amdahl's Law speedup function.
    
    Args:
        n: Number of processors/threads
        s: Serial fraction (0 <= s <= 1)
    
    Returns:
        Speedup = 1 / (s + (1-s)/n)
    """
    return 1.0 / (s + (1.0 - s) / n)


def amdahl_law_with_overhead(n: np.ndarray, s: float, overhead: float) -> np.ndarray:
    """
    Modified Amdahl's Law with parallelization overhead.
    
    Args:
        n: Number of processors/threads
        s: Serial fraction (0 <= s <= 1)
        overhead: Per-thread overhead factor
    
    Returns:
        Speedup accounting for overhead
    """
    base_speedup = 1.0 / (s + (1.0 - s) / n)
    overhead_factor = 1.0 + overhead * (n - 1)
    return base_speedup / overhead_factor


def fit_amdahl(thread_counts: List[int], speedups: List[float], 
               use_overhead: bool = False) -> Tuple[float, Optional[float]]:
    """
    Fit Amdahl's law to observed speedup data.
    
    Args:
        thread_counts: List of thread counts sampled
        speedups: Corresponding observed speedups
        use_overhead: If True, fit model with overhead term
    
    Returns:
        (serial_fraction, overhead) - overhead is None if use_overhead=False
    """
    if not HAS_SCIPY:
        # Fallback: estimate s from asymptotic speedup
        max_speedup = max(speedups)
        s = 1.0 / max_speedup if max_speedup > 0 else 1.0
        return s, None
    
    x = np.array(thread_counts, dtype=float)
    y = np.array(speedups, dtype=float)
    
    try:
        if use_overhead:
            # Fit with overhead parameter
            popt, _ = curve_fit(
                amdahl_law_with_overhead, x, y,
                p0=[0.1, 0.001],  # Initial guess: 10% serial, small overhead
                bounds=([0.0, 0.0], [1.0, 0.1]),  # s in [0,1], overhead in [0, 0.1]
                maxfev=5000
            )
            return popt[0], popt[1]
        else:
            # Fit standard Amdahl's law
            popt, _ = curve_fit(
                amdahl_law, x, y,
                p0=[0.1],  # Initial guess: 10% serial fraction
                bounds=([0.0], [1.0]),  # s must be in [0, 1]
                maxfev=5000
            )
            return popt[0], None
    except Exception as e:
        print(f"  Warning: Curve fitting failed ({e}), using fallback estimation", file=sys.stderr)
        # Fallback estimation
        max_speedup = max(speedups)
        s = 1.0 / max_speedup if max_speedup > 0 else 1.0
        return min(max(s, 0.0), 1.0), None


def detect_plateau(thread_counts: List[int], speedups: List[float], 
                   threshold: float = 0.05) -> Tuple[bool, int, float]:
    """
    Detect if speedup has plateaued (incremental gain below threshold).
    
    Args:
        thread_counts: List of thread counts
        speedups: Corresponding speedups
        threshold: Relative improvement threshold (e.g., 0.05 = 5%)
    
    Returns:
        (has_plateau, plateau_thread_count, plateau_speedup)
    """
    if len(speedups) < 2:
        return False, thread_counts[-1], speedups[-1]
    
    for i in range(1, len(speedups)):
        prev_speedup = speedups[i - 1]
        curr_speedup = speedups[i]
        
        if prev_speedup > 0:
            relative_gain = (curr_speedup - prev_speedup) / prev_speedup
            
            # If gain is below threshold (or negative), we've hit plateau
            if relative_gain < threshold:
                return True, thread_counts[i - 1], prev_speedup
    
    return False, thread_counts[-1], speedups[-1]


def generate_full_speedup_curve(
    max_threads: int,
    serial_fraction: float,
    overhead: Optional[float],
    plateau_thread: int,
    plateau_speedup: float,
    has_plateau: bool
) -> List[float]:
    """
    Generate full speedup curve from 1 to max_threads using fitted model.
    
    If plateau is detected, the curve flattens at the plateau point.
    """
    speedups = []
    
    for n in range(1, max_threads + 1):
        if has_plateau and n >= plateau_thread:
            # Flatten at plateau value
            speedup = plateau_speedup
        else:
            # Use fitted Amdahl's law
            if overhead is not None:
                speedup = float(amdahl_law_with_overhead(np.array([n]), serial_fraction, overhead)[0])
            else:
                speedup = float(amdahl_law(np.array([n]), serial_fraction)[0])
        
        speedups.append(speedup)
    
    return speedups


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


def process_query_file(
    query_file: str, 
    host: str, 
    port: int,
    max_threads: int,
    sample_points: List[int],
    repeat: int, 
    warmup: int,
    plateau_threshold: float,
    use_overhead: bool,
    verbose: bool
) -> List[float]:
    """Process a single query file and save speedup results."""
    query = read_query_file(query_file)
    
    # Filter sample points to be within max_threads and include 1
    valid_samples = sorted(set([1] + [p for p in sample_points if p <= max_threads]))
    
    if verbose:
        print(f"Query: {query_file}")
        print(f"ClickHouse: {host}:{port}")
        print(f"Max threads: {max_threads}")
        print(f"Sample points: {valid_samples}")
        print(f"Repetitions per sample: {repeat}")
        print(f"Warmup iterations: {warmup}")
        print(f"Plateau threshold: {plateau_threshold:.1%}")
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
    
    # Detect plateau
    has_plateau, plateau_thread, plateau_speedup = detect_plateau(
        thread_counts, speedups_sampled, plateau_threshold
    )
    
    if verbose:
        if has_plateau:
            print(f"Plateau detected at {plateau_thread} threads (speedup: {plateau_speedup:.2f})")
        else:
            print("No plateau detected")
    
    # Fit Amdahl's law
    serial_fraction, overhead = fit_amdahl(thread_counts, speedups_sampled, use_overhead)
    
    if verbose:
        print(f"Fitted serial fraction: {serial_fraction:.4f}")
        if overhead is not None:
            print(f"Fitted overhead: {overhead:.6f}")
        theoretical_max = 1.0 / serial_fraction if serial_fraction > 0 else float('inf')
        print(f"Theoretical max speedup (Amdahl): {theoretical_max:.2f}")
    
    # Generate full speedup curve
    full_speedups = generate_full_speedup_curve(
        max_threads, serial_fraction, overhead,
        plateau_thread, plateau_speedup, has_plateau
    )
    
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
    
    # Save to file next to the original query file
    base_path = os.path.splitext(query_file)[0]
    output_path = f"{base_path}_speedup.csv"
    
    with open(output_path, 'w') as f:
        for s in full_speedups:
            f.write(f"{s:.4f}\n")
        # Save 1-core average runtime as indicator of query size
        f.write(f"{avg_times[0]:.4f}\n")
    
    if verbose:
        print(f"Speedup values saved to: {output_path}")
    
    # Also save fitting metadata
    meta_path = f"{base_path}_speedup_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"serial_fraction: {serial_fraction:.6f}\n")
        if overhead is not None:
            f.write(f"overhead: {overhead:.6f}\n")
        f.write(f"has_plateau: {has_plateau}\n")
        f.write(f"plateau_thread: {plateau_thread}\n")
        f.write(f"plateau_speedup: {plateau_speedup:.4f}\n")
        f.write(f"sample_points: {thread_counts}\n")
        f.write(f"sampled_speedups: {speedups_sampled}\n")
    
    if verbose:
        print(f"Fitting metadata saved to: {meta_path}")
    
    return full_speedups


def process_query_string(
    query: str, 
    host: str, 
    port: int,
    max_threads: int,
    sample_points: List[int],
    repeat: int, 
    warmup: int,
    plateau_threshold: float,
    use_overhead: bool,
    verbose: bool
) -> List[float]:
    """Process an inline query string (no output file saved)."""
    # Filter sample points
    valid_samples = sorted(set([1] + [p for p in sample_points if p <= max_threads]))
    
    if verbose:
        print(f"Query: inline query")
        print(f"ClickHouse: {host}:{port}")
        print(f"Max threads: {max_threads}")
        print(f"Sample points: {valid_samples}")
        print(f"Repetitions per sample: {repeat}")
        print(f"Warmup iterations: {warmup}")
        print(f"Plateau threshold: {plateau_threshold:.1%}")
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
    
    # Detect plateau
    has_plateau, plateau_thread, plateau_speedup = detect_plateau(
        thread_counts, speedups_sampled, plateau_threshold
    )
    
    if verbose:
        if has_plateau:
            print(f"Plateau detected at {plateau_thread} threads (speedup: {plateau_speedup:.2f})")
        else:
            print("No plateau detected")
    
    # Fit Amdahl's law
    serial_fraction, overhead = fit_amdahl(thread_counts, speedups_sampled, use_overhead)
    
    if verbose:
        print(f"Fitted serial fraction: {serial_fraction:.4f}")
        if overhead is not None:
            print(f"Fitted overhead: {overhead:.6f}")
        theoretical_max = 1.0 / serial_fraction if serial_fraction > 0 else float('inf')
        print(f"Theoretical max speedup (Amdahl): {theoretical_max:.2f}")
    
    # Generate full speedup curve
    full_speedups = generate_full_speedup_curve(
        max_threads, serial_fraction, overhead,
        plateau_thread, plateau_speedup, has_plateau
    )
    
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
        description="Sparse speedup profiler with Amdahl's law curve fitting"
    )
    
    # Query specification (one of these required)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", "-q", help="Path to SQL query file")
    query_group.add_argument("--dir", "-d", help="Directory containing .sql query files (tests all)")
    query_group.add_argument("--query-string", "-Q", help="SQL query string directly")
    
    # Test parameters
    parser.add_argument("--max-threads", "-t", type=int, default=64,
                        help="Maximum number of threads for output curve (default: 64)")
    parser.add_argument("--sample-points", "-s", type=str, 
                        default="1,2,4,8,12,16,24,32,48,64",
                        help="Comma-separated thread counts to sample (default: 1,2,4,8,12,16,24,32,48,64)")
    parser.add_argument("--repeat", "-r", type=int, default=5,
                        help="Number of repetitions per sample point (default: 5)")
    parser.add_argument("--warmup", "-w", type=int, default=5,
                        help="Number of warmup runs before measurement (default: 5)")
    
    # Fitting parameters
    parser.add_argument("--plateau-threshold", "-p", type=float, default=0.001,
                        help="Relative speedup improvement threshold for plateau detection (default: 0.001)")
    parser.add_argument("--with-overhead", action="store_true",
                        help="Use Amdahl's law model with parallelization overhead term")
    
    # ClickHouse connection
    parser.add_argument("--host", default="localhost", help="ClickHouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port (default: 8123)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Parse sample points
    sample_points = parse_sample_points(args.sample_points)
    
    if not HAS_SCIPY:
        print("Warning: scipy not installed. Curve fitting will use fallback estimation.", file=sys.stderr)
        print("Install scipy for better fitting: pip install scipy numpy", file=sys.stderr)
    
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
                args.max_threads, sample_points, args.repeat, args.warmup,
                args.plateau_threshold, args.with_overhead, verbose
            )
            if verbose and len(query_files) > 1:
                print()
    else:
        # Inline query string
        process_query_string(
            args.query_string, args.host, args.port,
            args.max_threads, sample_points, args.repeat, args.warmup,
            args.plateau_threshold, args.with_overhead, verbose
        )


if __name__ == "__main__":
    main()
