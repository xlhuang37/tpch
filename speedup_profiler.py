#!/usr/bin/env python3
"""
Speedup Profiler for ClickHouse Queries

Measures query execution time across different thread counts to generate
a speedup curve. Speedup is calculated as: runtime(1 thread) / runtime(N threads)

Usage:
    python speedup_profiler.py --query queries/p_greedy_108.sql --max-threads 8 --repeat 5
    python speedup_profiler.py --query-string "SELECT count(*) FROM lineitem" --max-threads 4
"""

import argparse
import os
import time
import sys
from typing import List, Tuple
import clickhouse_connect


def read_query_file(filepath: str) -> str:
    """Read SQL query from file."""
    with open(filepath, 'r') as f:
        return f.read().strip()


def run_query_with_threads(client, query: str, max_threads: int, timeout: int = 300) -> float:
    """
    Run a query with specified max_threads setting and return execution time in seconds.
    """
    # Set max_threads for this query
    settings = {'max_threads': max_threads}
    
    start_time = time.perf_counter()
    try:
        client.query(query, settings=settings)
    except Exception as e:
        print(f"  Error running query with {max_threads} threads: {e}", file=sys.stderr)
        return float('inf')
    end_time = time.perf_counter()
    
    return end_time - start_time


def measure_speedup(
    client,
    query: str,
    max_threads: int,
    repeat: int,
    warmup: int = 1,
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
            run_query_with_threads(client, query, max_threads)
    
    # Measure for each thread count
    for num_threads in range(1, max_threads + 1):
        times = []
        if verbose:
            print(f"Testing with {num_threads} thread(s)...", end=" ", flush=True)
        
        for rep in range(repeat):
            elapsed = run_query_with_threads(client, query, num_threads)
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


def main():
    parser = argparse.ArgumentParser(
        description="Profile query speedup across different thread counts in ClickHouse"
    )
    
    # Query specification (one of these required)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", "-q", help="Path to SQL query file")
    query_group.add_argument("--query-string", "-Q", help="SQL query string directly")
    
    # Test parameters
    parser.add_argument("--max-threads", "-t", type=int, default=8,
                       help="Maximum number of threads to test (default: 8)")
    parser.add_argument("--repeat", "-r", type=int, default=5,
                       help="Number of repetitions per thread count (default: 5)")
    parser.add_argument("--warmup", "-w", type=int, default=1,
                       help="Number of warmup runs before measurement (default: 1)")
    
    # ClickHouse connection
    parser.add_argument("--host", default="localhost", help="ClickHouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port (default: 8123)")
    
    # Output options
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Get query
    if args.query:
        query = read_query_file(args.query)
        query_name = args.query
    else:
        query = args.query_string
        query_name = "inline_query"
    
    verbose = not args.quiet
    
    if verbose:
        print(f"Query: {query_name}")
        print(f"Max threads: {args.max_threads}")
        print(f"Repetitions per thread count: {args.repeat}")
        print(f"Warmup iterations: {args.warmup}")
        print("-" * 60)
    
    # Connect to ClickHouse
    try:
        client = clickhouse_connect.get_client(
            host=args.host,
            port=args.port
        )
        if verbose:
            print(f"Connected to ClickHouse at {args.host}:{args.port}")
    except Exception as e:
        print(f"Error connecting to ClickHouse: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run measurements
    if verbose:
        print("-" * 60)
    
    speedups, avg_times, all_times = measure_speedup(
        client, query, args.max_threads, args.repeat, args.warmup, verbose
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
    if args.query:
        # Generate output filename: queries/p_greedy_108.sql -> queries/p_greedy_108_speedup.csv
        base_path = os.path.splitext(args.query)[0]
        output_path = f"{base_path}_speedup.csv"
        
        with open(output_path, 'w') as f:
            for s in speedups:
                f.write(f"{s:.4f}\n")
        
        if verbose:
            print(f"Speedup values saved to: {output_path}")
    
    # Return speedups for programmatic use
    return speedups


if __name__ == "__main__":
    main()
