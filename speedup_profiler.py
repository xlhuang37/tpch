#!/usr/bin/env python3
"""
Speedup Profiler for ClickHouse Queries

Measures query execution time across different thread counts to generate
a speedup curve. Speedup is calculated as: runtime(1 thread) / runtime(N threads)

Usage:
    # Test a single query file
    python speedup_profiler.py --query queries/p_greedy_108.sql --max-threads 8 --repeat 5
    
    # Test all .sql files in a directory
    python speedup_profiler.py --dir queries/ --max-threads 8 --repeat 5
    
    # Test with inline query string (no output file saved)
    python speedup_profiler.py --query-string "SELECT count(*) FROM lineitem" --max-threads 4
    
    # Custom ClickHouse host/port
    python speedup_profiler.py --query queries/p_greedy_108.sql -t 8 -r 5 --host myserver --port 8123
"""

import argparse
import glob
import os
import time
import sys
from typing import List, Tuple
import urllib.request
import urllib.parse


def read_query_file(filepath: str) -> str:
    """Read SQL query from file."""
    with open(filepath, 'r') as f:
        return f.read().strip()


def run_query_with_threads(host: str, port: int, query: str, max_threads: int) -> float:
    """
    Run a query with specified max_threads setting and return execution time in seconds.
    Uses ClickHouse HTTP interface.
    """
    # Build URL with max_threads setting
    params = urllib.parse.urlencode({'max_threads': max_threads})
    url = f"http://{host}:{port}/?{params}"
    
    start_time = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=query.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req) as response:
            response.read()  # Consume the response
    except Exception as e:
        print(f"  Error running query with {max_threads} threads: {e}", file=sys.stderr)
        return float('inf')
    end_time = time.perf_counter()
    
    return end_time - start_time


def measure_speedup(
    host: str,
    port: int,
    query: str,
    max_threads: int,
    repeat: int,
    warmup: int = 3,
    verbose: bool = True
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Measure query execution times for thread counts 1 to max_threads.
    
    Returns:
        speedups: List of speedup values [1.0, speedup_2, speedup_3, ...]
        avg_times: List of average execution times for each thread count
        all_times: List of all individual measurements for each thread count
    """
    all_times = []  # all_times[thread_idx] = list of measurements
    avg_times = []
    
    # Warmup runs (not counted)
    if warmup > 0 and verbose:
        print(f"Running {warmup} warmup iteration(s)...")
        for _ in range(warmup):
            run_query_with_threads(host, port, query, max_threads)
    
    # Measure for each thread count
    for num_threads in range(1, max_threads + 1):
        times = []
        if verbose:
            print(f"Testing with {num_threads} thread(s)...", end=" ", flush=True)
        
        for rep in range(repeat):
            elapsed = run_query_with_threads(host, port, query, num_threads)
            times.append(elapsed)
            if verbose:
                print(f"{elapsed:.3f}s", end=" ", flush=True)
        
        avg_time = sum(times) / len(times)
        all_times.append(times)
        avg_times.append(avg_time)
        
        if verbose:
            print(f"-> avg: {avg_time:.3f}s")
    
    # Calculate speedups (relative to 1 thread)
    baseline = avg_times[0]  # average time with 1 thread
    speedups = [baseline / t if t > 0 else 0.0 for t in avg_times]
    
    return speedups, avg_times, all_times


def process_query_file(query_file: str, host: str, port: int, 
                       max_threads: int, repeat: int, warmup: int, verbose: bool):
    """Process a single query file and save speedup results."""
    query = read_query_file(query_file)
    
    if verbose:
        print(f"Query: {query_file}")
        print(f"ClickHouse: {host}:{port}")
        print(f"Max threads: {max_threads}")
        print(f"Repetitions per thread count: {repeat}")
        print(f"Warmup iterations: {warmup}")
        print("-" * 60)
    
    # Run measurements
    speedups, avg_times, all_times = measure_speedup(
        host, port, query, max_threads, repeat, warmup, verbose
    )
    
    # Print results
    if verbose:
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"{'Threads':<10} {'Avg Time (s)':<15} {'Speedup':<10}")
        print("-" * 35)
        for i, (t, s) in enumerate(zip(avg_times, speedups), 1):
            print(f"{i:<10} {t:<15.4f} {s:<10.2f}")
        print()
    
    # Output speedup list
    speedup_str = ", ".join(f"{s:.2f}" for s in speedups)
    print(f"Speedup list: [{speedup_str}]")
    
    # Save to file next to the original query file
    base_path = os.path.splitext(query_file)[0]
    output_path = f"{base_path}_speedup.csv"
    
    with open(output_path, 'w') as f:
        for s in speedups:
            f.write(f"{s:.4f}\n")
        # Save 1-core average runtime as indicator of query size
        f.write(f"{avg_times[0]:.4f}\n")
    
    if verbose:
        print(f"Speedup values saved to: {output_path}")
    
    return speedups


def process_query_string(query: str, host: str, port: int,
                         max_threads: int, repeat: int, warmup: int, verbose: bool):
    """Process an inline query string (no output file saved)."""
    if verbose:
        print(f"Query: inline query")
        print(f"ClickHouse: {host}:{port}")
        print(f"Max threads: {max_threads}")
        print(f"Repetitions per thread count: {repeat}")
        print(f"Warmup iterations: {warmup}")
        print("-" * 60)
    
    # Run measurements
    speedups, avg_times, all_times = measure_speedup(
        host, port, query, max_threads, repeat, warmup, verbose
    )
    
    # Print results
    if verbose:
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)
        print(f"{'Threads':<10} {'Avg Time (s)':<15} {'Speedup':<10}")
        print("-" * 35)
        for i, (t, s) in enumerate(zip(avg_times, speedups), 1):
            print(f"{i:<10} {t:<15.4f} {s:<10.2f}")
        print()
    
    # Output speedup list
    speedup_str = ", ".join(f"{s:.2f}" for s in speedups)
    print(f"Speedup list: [{speedup_str}]")
    
    return speedups


def main():
    parser = argparse.ArgumentParser(
        description="Profile query speedup across different thread counts in ClickHouse"
    )
    
    # Query specification (one of these required)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", "-q", help="Path to SQL query file")
    query_group.add_argument("--dir", "-d", help="Directory containing .sql query files (tests all)")
    query_group.add_argument("--query-string", "-Q", help="SQL query string directly")
    
    # Test parameters
    parser.add_argument("--max-threads", "-t", type=int, default=8,
                       help="Maximum number of threads to test (default: 8)")
    parser.add_argument("--repeat", "-r", type=int, default=5,
                       help="Number of repetitions per thread count (default: 5)")
    parser.add_argument("--warmup", "-w", type=int, default=5,
                       help="Number of warmup runs before measurement (default: 1)")
    
    # ClickHouse connection
    parser.add_argument("--host", default="localhost", help="ClickHouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port (default: 8123)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Collect query files to process
    if args.dir:
        # Find all .sql files in directory
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
                args.max_threads, args.repeat, args.warmup, verbose
            )
            if verbose and len(query_files) > 1:
                print()
    else:
        # Inline query string
        process_query_string(
            args.query_string, args.host, args.port,
            args.max_threads, args.repeat, args.warmup, verbose
        )


if __name__ == "__main__":
    main()
