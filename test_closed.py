#!/usr/bin/env python3
"""
Minimal closed system benchmark wrapper for clickhouse-benchmark.
Reads SQL files, formats them with clickhouse-format, and pipes to clickhouse-benchmark.
Output saved to ./output/closed/<timestamp>.txt
"""
import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime


def main():
    ap = argparse.ArgumentParser(
        description="Run closed system ClickHouse benchmark using clickhouse-benchmark."
    )
    ap.add_argument(
        "--queries-dir",
        required=True,
        help="Directory containing SQL files (default: ./queries/)"
    )
    ap.add_argument(
        "--clickhouse-dir",
        default="../ClickHouse/build/programs",
        help="Directory containing clickhouse-format and clickhouse-benchmark (default: ../ClickHouse/build/programs)"
    )
    ap.add_argument(
        "-c", "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent connections (default: 20)"
    )
    ap.add_argument(
        "-i", "--iterations",
        type=int,
        default=200,
        help="Number of query iterations (default: 200)"
    )
    ap.add_argument(
        "--host",
        default="localhost",
        help="ClickHouse server host (default: localhost)"
    )
    ap.add_argument(
        "--port",
        type=int,
        default=9000,
        help="ClickHouse native port (default: 9000)"
    )

    args = ap.parse_args()

    # Find SQL files
    sql_pattern = os.path.join(args.queries_dir, "*.sql")
    sql_files = sorted(glob.glob(sql_pattern))

    if not sql_files:
        print(f"No SQL files found in {args.queries_dir}")
        sys.exit(1)

    print(f"Found {len(sql_files)} SQL files in {args.queries_dir}")

    # Concatenate all SQL files
    all_sql = ""
    for sql_file in sql_files:
        with open(sql_file, "r", encoding="utf-8") as f:
            all_sql += f.read() + "\n"

    # Create output directory
    output_dir = "./output/closed"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{timestamp}.txt")

    # Build paths
    format_path = os.path.join(args.clickhouse_dir, "clickhouse-format")
    benchmark_path = os.path.join(args.clickhouse_dir, "clickhouse-benchmark")

    # Pipeline: SQL -> clickhouse-format -> clickhouse-benchmark -> output file
    # Equivalent to: cat *.sql | clickhouse-format -n --oneline | clickhouse-benchmark -c 20 -i 200 &> output.txt

    print(f"Running benchmark with -c {args.concurrency} -i {args.iterations}")
    print(f"Output: {output_file}")

    # Step 1: Format SQL with clickhouse-format
    format_cmd = [format_path, "-n", "--oneline"]
    format_proc = subprocess.Popen(
        format_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    formatted_sql, format_err = format_proc.communicate(input=all_sql)

    if format_proc.returncode != 0:
        print(f"clickhouse-format failed: {format_err}")
        sys.exit(1)

    # Step 2: Run clickhouse-benchmark with formatted SQL
    benchmark_cmd = [
        benchmark_path,
        "-h", args.host,
        "--port", str(args.port),
        "-c", str(args.concurrency),
        "-i", str(args.iterations),
    ]

    with open(output_file, "w", encoding="utf-8") as out_f:
        benchmark_proc = subprocess.Popen(
            benchmark_cmd,
            stdin=subprocess.PIPE,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            text=True
        )
        benchmark_proc.communicate(input=formatted_sql)

    print(f"Benchmark complete. Results saved to: {output_file}")

    # Print results to console as well
    with open(output_file, "r", encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
