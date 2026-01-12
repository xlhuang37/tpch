#!/usr/bin/env python3
#python3 open_test.py
from __future__ import annotations
from typing import Optional
import argparse
import asyncio
import csv
import glob
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict

import aiohttp


@dataclass
class Event:
    at_ms: int
    qid: str


@dataclass
class LatencyRecord:
    arrival_ms: float     # Actual arrival time (relative to t0)
    start_ms: float       # Actual query start time (when server accepted)
    end_ms: float         # Query completion time
    latency_ms: float     # end_ms - start_ms (execution time)
    qid: str


def compute_percentile(sorted_values: List[float], p: float) -> float:
    """Compute the p-th percentile of sorted values (p in 0-100)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (p / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def save_raw_data(output_dir: str, schedule_name: str, records: List[LatencyRecord]) -> str:
    """Save raw data (arrival_ms, start_ms, end_ms, latency_ms, qid) sorted by arrival_ms to a CSV file."""
    sorted_records = sorted(records, key=lambda r: (r.arrival_ms, r.qid))
    filename = f"{schedule_name}_raw.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_ms", "start_ms", "end_ms", "latency_ms", "qid"])
        for r in sorted_records:
            w.writerow([f"{r.arrival_ms:.4f}", f"{r.start_ms:.4f}", 
                       f"{r.end_ms:.4f}", f"{r.latency_ms:.4f}", r.qid])
    
    return filepath


def save_statistics(output_dir: str, schedule_name: str, records: List[LatencyRecord], qid_list: List[str]) -> str:
    """Save statistics (average, percentiles) to a text file."""
    filename = f"{schedule_name}_stats.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Group records by qid
    by_qid: Dict[str, List[float]] = {qid: [] for qid in qid_list}
    all_latencies: List[float] = []
    
    for r in records:
        by_qid[r.qid].append(r.latency_ms)
        all_latencies.append(r.latency_ms)
    
    with open(filepath, "w") as f:
        f.write(f"Statistics for schedule: {schedule_name}\n")
        f.write(f"Total queries: {len(records)}\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics
        if all_latencies:
            sorted_all = sorted(all_latencies)
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Count:    {len(sorted_all)}\n")
            f.write(f"  Average:  {sum(sorted_all) / len(sorted_all):.4f} ms\n")
            f.write(f"  Min:      {sorted_all[0]:.4f} ms\n")
            f.write(f"  Max:      {sorted_all[-1]:.4f} ms\n")
            f.write(f"  P50:      {compute_percentile(sorted_all, 50):.4f} ms\n")
            f.write(f"  P90:      {compute_percentile(sorted_all, 90):.4f} ms\n")
            f.write(f"  P95:      {compute_percentile(sorted_all, 95):.4f} ms\n")
            f.write(f"  P99:      {compute_percentile(sorted_all, 99):.4f} ms\n")
            f.write("\n")
        
        # Per-qid statistics
        f.write("PER-QUERY STATISTICS\n")
        f.write("-" * 40 + "\n")
        for qid in qid_list:
            latencies = by_qid[qid]
            if latencies:
                sorted_lat = sorted(latencies)
                f.write(f"\n  {qid}:\n")
                f.write(f"    Count:    {len(sorted_lat)}\n")
                f.write(f"    Average:  {sum(sorted_lat) / len(sorted_lat):.4f} ms\n")
                f.write(f"    Min:      {sorted_lat[0]:.4f} ms\n")
                f.write(f"    Max:      {sorted_lat[-1]:.4f} ms\n")
                f.write(f"    P50:      {compute_percentile(sorted_lat, 50):.4f} ms\n")
                f.write(f"    P90:      {compute_percentile(sorted_lat, 90):.4f} ms\n")
                f.write(f"    P95:      {compute_percentile(sorted_lat, 95):.4f} ms\n")
                f.write(f"    P99:      {compute_percentile(sorted_lat, 99):.4f} ms\n")
    
    return filepath


def read_schedule_csv(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        # expects columns: at_ms,qid
        for row in r:
            at_ms = int(row["at_ms"])
            qid = row["qid"].strip()
            events.append(Event(at_ms=at_ms, qid=qid))
    events.sort(key=lambda e: e.at_ms)
    return events


def load_query(queries_dir: str, qid: str) -> bytes:
    qpath = os.path.join(queries_dir, f"{qid}.sql")
    if not os.path.exists(qpath):
        raise FileNotFoundError(f"Missing query file: {qpath}")
    with open(qpath, "rb") as f:
        sql = f.read().strip()
    # ClickHouse accepts query in request body; ensure newline at end
    return sql + b"\n"


async def wait_until_ns(target_ns: int, spin_ns: int) -> None:
    """
    Coarse sleep then spin near the target for better ms accuracy.
    spin_ns: final busy-wait window
    """
    while True:
        now = time.perf_counter_ns()
        remaining = target_ns - now
        if remaining <= 0:
            return
        if remaining > spin_ns:
            await asyncio.sleep((remaining - spin_ns) / 1e9)
        else:
            while time.perf_counter_ns() < target_ns:
                pass
            return


async def send_one(
    session: aiohttp.ClientSession,
    url: str,
    event: Event,
    sql_bytes: bytes,
    t0_ns: int,
    spin_ns: int,
    sem: asyncio.Semaphore,
    latency_records: List[LatencyRecord],
    records_lock: asyncio.Lock,
    max_retries: int = 1000,
    retry_delay_ms: float = 10000,
) -> None:
    target_ns = t0_ns + event.at_ms * 1_000_000
    await wait_until_ns(target_ns, spin_ns)

    # Record actual arrival time (relative to t0)
    arrival_ns = time.perf_counter_ns()
    arrival_ms = (arrival_ns - t0_ns) / 1e6

    for attempt in range(max_retries):
        async with sem:
            req_start_ns = time.perf_counter_ns()
            start_ms = (req_start_ns - t0_ns) / 1e6
            
            try:
                async with session.post(url, data=sql_bytes) as resp:
                    body = await resp.read()
                    req_end_ns = time.perf_counter_ns()
                    end_ms = (req_end_ns - t0_ns) / 1e6
                    latency_ms = (req_end_ns - req_start_ns) / 1e6

                    if resp.status == 200:
                        # Success - record and return
                        async with records_lock:
                            latency_records.append(LatencyRecord(
                                arrival_ms=arrival_ms,
                                start_ms=start_ms,
                                end_ms=end_ms,
                                latency_ms=latency_ms,
                                qid=event.qid
                            ))
                        wait_ms = start_ms - arrival_ms
                        print(f"[{event.at_ms:>6} ms] {event.qid}: OK "
                              f"lat={latency_ms:.2f}ms wait={wait_ms:.2f}ms")
                        return
                    else:
                        # Server rejected (e.g., overloaded) - retry
                        snippet = body[:200].decode("utf-8", errors="replace")
                        print(f"[{event.at_ms:>6} ms] {event.qid}: HTTP {resp.status} "
                              f"(attempt {attempt+1}) - retrying... resp='{snippet}'")
                        
            except Exception as e:
                print(f"[{event.at_ms:>6} ms] {event.qid}: ERROR {e} "
                      f"(attempt {attempt+1}) - retrying...")
        
        # Wait before retry (outside sem to not block others)
        print(f"[{event.at_ms:>6} ms] {event.qid}: waiting {retry_delay_ms/1000:.1f}s before retry...")
        await asyncio.sleep(retry_delay_ms / 1000.0)
    
    # Max retries exceeded - record as failed with last attempt times
    req_end_ns = time.perf_counter_ns()
    end_ms = (req_end_ns - t0_ns) / 1e6
    print(f"[{event.at_ms:>6} ms] {event.qid}: FAILED after {max_retries} attempts")


async def run_schedule(
    schedule_path: str,
    queries_dir: str,
    url: str,
    max_concurrency: int,
    spin_ns: int,
    output_dir: str,
) -> None:
    """Run a single schedule and save results."""
    schedule_name = os.path.splitext(os.path.basename(schedule_path))[0]
    print(f"\n{'='*60}")
    print(f"Running schedule: {schedule_path}")
    print(f"{'='*60}\n")
    
    events = read_schedule_csv(schedule_path)

    # Preload SQL bytes for each qid (so send path is lightweight)
    sql_cache = {}
    qid_list = []
    for e in events:
        if e.qid not in sql_cache:
            sql_cache[e.qid] = load_query(queries_dir, e.qid)
            qid_list.append(e.qid)

    connector = aiohttp.TCPConnector(limit=max_concurrency, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=None)  # let queries run; adjust if desired
    sem = asyncio.Semaphore(max_concurrency)
    
    latency_records: List[LatencyRecord] = []
    records_lock = asyncio.Lock()

    t0_ns = time.perf_counter_ns()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                send_one(
                    session=session,
                    url=url,
                    event=e,
                    sql_bytes=sql_cache[e.qid],
                    t0_ns=t0_ns,
                    spin_ns=spin_ns,
                    sem=sem,
                    latency_records=latency_records,
                    records_lock=records_lock,
                )
            )
            for e in events
        ]
        await asyncio.gather(*tasks)

    # Save results
    raw_path = save_raw_data(output_dir, schedule_name, latency_records)
    stats_path = save_statistics(output_dir, schedule_name, latency_records, qid_list)
    
    print(f"\nSchedule '{schedule_name}' completed.")
    print(f"  Raw data saved to: {raw_path}")
    print(f"  Statistics saved to: {stats_path}")
    
    # Print summary to console
    if latency_records:
        all_latencies = [r.latency_ms for r in latency_records]
        avg_latency = sum(all_latencies) / len(all_latencies)
        print(f"  Average latency: {avg_latency:.2f} ms")


async def main():
    ap = argparse.ArgumentParser(description="Replay ClickHouse HTTP workload by timestamp (ms).")
    ap.add_argument("--schedules-dir", default="./schedules/", 
                    help="Directory containing schedule CSV files (default: ./schedules/)")
    ap.add_argument("--queries-dir", default="./queries/", help="Directory containing <qid>.sql files")
    ap.add_argument("--url", default="http://localhost:8123/", help="ClickHouse HTTP endpoint URL")
    ap.add_argument("--max-concurrency", type=int, default=50,
                    help="Max in-flight HTTP requests from the client")
    ap.add_argument("--spin-ns", type=int, default=100000,
                    help="Final busy-wait window for timing accuracy (microseconds)")
    args = ap.parse_args()

    # Discover all CSV files in the schedules directory
    schedule_pattern = os.path.join(args.schedules_dir, "*.csv")
    schedule_paths = sorted(glob.glob(schedule_pattern))
    
    if not schedule_paths:
        print(f"No schedule CSV files found in {args.schedules_dir}")
        return
    
    print(f"Found {len(schedule_paths)} schedule(s) in {args.schedules_dir}:")
    for p in schedule_paths:
        print(f"  - {os.path.basename(p)}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Run each schedule sequentially
    for schedule_path in schedule_paths:
        await run_schedule(
            schedule_path=schedule_path,
            queries_dir=args.queries_dir,
            url=args.url,
            max_concurrency=args.max_concurrency,
            spin_ns=args.spin_ns,
            output_dir=output_dir,
        )

    print(f"\n{'='*60}")
    print(f"All schedules completed. Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
