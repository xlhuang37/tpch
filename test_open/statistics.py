"""
Statistics computation and data saving functions.
"""

import csv
import os
from typing import List, Dict

from .models import LatencyRecord, DroppedRecord


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


def save_raw_data(output_dir: str, schedule_name: str, records: List[LatencyRecord], 
                  dropped_records: List[DroppedRecord]) -> str:
    """Save raw data (arrival_ms, start_ms, end_ms, latency_ms, wait_ms, qid) sorted by arrival_ms to a CSV file.
    Dropped queries are saved with just arrival_ms, a dropped message, and qid."""
    # Combine and sort all records by arrival_ms
    all_entries = []
    for r in records:
        all_entries.append(('record', r.arrival_ms, r))
    for d in dropped_records:
        all_entries.append(('dropped', d.arrival_ms, d))
    all_entries.sort(key=lambda x: (x[1], x[2].qid))
    
    filename = f"{schedule_name}_raw.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arrival_ms", "start_ms", "end_ms", "latency_ms", "wait_ms", "qid"])
        for entry_type, _, entry in all_entries:
            if entry_type == 'record':
                r = entry
                w.writerow([f"{r.arrival_ms:.4f}", f"{r.start_ms:.4f}", 
                           f"{r.end_ms:.4f}", f"{r.latency_ms:.4f}", f"{r.wait_ms:.4f}", r.qid])
            else:
                d = entry
                w.writerow([f"{d.arrival_ms:.4f}", "DROPPED", "DROPPED", "DROPPED", "DROPPED", d.qid])
    
    return filepath


def save_statistics(output_dir: str, schedule_name: str, records: List[LatencyRecord], 
                    dropped_records: List[DroppedRecord], qid_list: List[str]) -> str:
    """Save statistics (average, percentiles) for latency and wait time to a text file.
    Dropped queries are excluded from latency/wait stats but reported separately."""
    filename = f"{schedule_name}_stats.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Group records by qid
    latencies_by_qid: Dict[str, List[float]] = {qid: [] for qid in qid_list}
    waits_by_qid: Dict[str, List[float]] = {qid: [] for qid in qid_list}
    all_latencies: List[float] = []
    all_waits: List[float] = []
    
    for r in records:
        latencies_by_qid[r.qid].append(r.latency_ms)
        waits_by_qid[r.qid].append(r.wait_ms)
        all_latencies.append(r.latency_ms)
        all_waits.append(r.wait_ms)
    
    # Count dropped queries by qid
    dropped_by_qid: Dict[str, int] = {qid: 0 for qid in qid_list}
    for d in dropped_records:
        if d.qid in dropped_by_qid:
            dropped_by_qid[d.qid] += 1
        else:
            dropped_by_qid[d.qid] = 1
    total_dropped = len(dropped_records)
    
    with open(filepath, "w") as f:
        f.write(f"Statistics for schedule: {schedule_name}\n")
        f.write(f"Total completed queries: {len(records)}\n")
        f.write(f"Total dropped queries: {total_dropped}\n")
        f.write("=" * 60 + "\n\n")
        
        # Dropped queries report
        if total_dropped > 0:
            f.write("DROPPED QUERIES\n")
            f.write("-" * 40 + "\n")
            for qid in qid_list:
                if dropped_by_qid[qid] > 0:
                    f.write(f"  {qid}: {dropped_by_qid[qid]}\n")
            # Include any qids not in qid_list
            for qid, count in dropped_by_qid.items():
                if qid not in qid_list and count > 0:
                    f.write(f"  {qid}: {count}\n")
            f.write("\n")
        
        # Overall latency statistics
        if all_latencies:
            sorted_all = sorted(all_latencies)
            f.write("OVERALL LATENCY STATISTICS\n")
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
        
        # Overall wait time statistics
        if all_waits:
            sorted_waits = sorted(all_waits)
            f.write("OVERALL WAIT TIME STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Count:    {len(sorted_waits)}\n")
            f.write(f"  Average:  {sum(sorted_waits) / len(sorted_waits):.4f} ms\n")
            f.write(f"  Min:      {sorted_waits[0]:.4f} ms\n")
            f.write(f"  Max:      {sorted_waits[-1]:.4f} ms\n")
            f.write(f"  P50:      {compute_percentile(sorted_waits, 50):.4f} ms\n")
            f.write(f"  P90:      {compute_percentile(sorted_waits, 90):.4f} ms\n")
            f.write(f"  P95:      {compute_percentile(sorted_waits, 95):.4f} ms\n")
            f.write(f"  P99:      {compute_percentile(sorted_waits, 99):.4f} ms\n")
            f.write("\n")
        
        # Per-qid statistics
        f.write("PER-QUERY STATISTICS\n")
        f.write("-" * 40 + "\n")
        for qid in qid_list:
            latencies = latencies_by_qid[qid]
            waits = waits_by_qid[qid]
            dropped_count = dropped_by_qid[qid]
            
            if latencies or dropped_count > 0:
                f.write(f"\n  {qid}:\n")
                if dropped_count > 0:
                    f.write(f"    Dropped:  {dropped_count}\n")
                if latencies:
                    sorted_lat = sorted(latencies)
                    sorted_wait = sorted(waits)
                    f.write(f"    Completed: {len(sorted_lat)}\n")
                    f.write(f"    Latency:\n")
                    f.write(f"      Average:  {sum(sorted_lat) / len(sorted_lat):.4f} ms\n")
                    f.write(f"      Min:      {sorted_lat[0]:.4f} ms\n")
                    f.write(f"      Max:      {sorted_lat[-1]:.4f} ms\n")
                    f.write(f"      P50:      {compute_percentile(sorted_lat, 50):.4f} ms\n")
                    f.write(f"      P90:      {compute_percentile(sorted_lat, 90):.4f} ms\n")
                    f.write(f"      P95:      {compute_percentile(sorted_lat, 95):.4f} ms\n")
                    f.write(f"      P99:      {compute_percentile(sorted_lat, 99):.4f} ms\n")
                    f.write(f"    Wait Time:\n")
                    f.write(f"      Average:  {sum(sorted_wait) / len(sorted_wait):.4f} ms\n")
                    f.write(f"      Min:      {sorted_wait[0]:.4f} ms\n")
                    f.write(f"      Max:      {sorted_wait[-1]:.4f} ms\n")
                    f.write(f"      P50:      {compute_percentile(sorted_wait, 50):.4f} ms\n")
                    f.write(f"      P90:      {compute_percentile(sorted_wait, 90):.4f} ms\n")
                    f.write(f"      P95:      {compute_percentile(sorted_wait, 95):.4f} ms\n")
                    f.write(f"      P99:      {compute_percentile(sorted_wait, 99):.4f} ms\n")
    
    return filepath
