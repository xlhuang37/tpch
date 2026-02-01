"""
Utility functions for schedule/query loading and timing.
"""

import asyncio
import csv
import os
import re
import threading
import time
from typing import List, Optional, Dict, Tuple

from .models import Event


def read_schedule_csv(path: str) -> List[Event]:
    """Read schedule events from a CSV file."""
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


def extract_workload_from_sql(sql: bytes) -> Optional[str]:
    """Extract workload setting from SQL if present."""
    # Match workload='...' or workload="..."
    match = re.search(rb"workload\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
    if match:
        return match.group(1).decode('utf-8')
    return None


def load_query(queries_dir: str, qid: str) -> Tuple[bytes, Optional[str]]:
    """Load query and extract workload. Returns (sql_bytes, workload_name or None)."""
    qpath = os.path.join(queries_dir, f"{qid}.sql")
    if not os.path.exists(qpath):
        raise FileNotFoundError(f"Missing query file: {qpath}")
    with open(qpath, "rb") as f:
        sql = f.read().strip()
    workload = extract_workload_from_sql(sql)
    # ClickHouse accepts query in request body; ensure newline at end
    return (sql + b"\n", workload)


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


def wait_until_ns_sync(target_ns: int, spin_ns: int, stop_event: Optional[threading.Event] = None) -> bool:
    """
    Synchronous version: Coarse sleep then spin near the target for better ms accuracy.
    spin_ns: final busy-wait window
    stop_event: if provided, checked periodically to allow early termination
    Returns: True if wait completed normally, False if interrupted by stop_event
    """
    CHECK_INTERVAL_NS = 100_000_000  # Check stop_event every 100ms
    
    while True:
        # Check for stop signal
        if stop_event and stop_event.is_set():
            return False
        
        now = time.perf_counter_ns()
        remaining = target_ns - now
        if remaining <= 0:
            return True
        if remaining > spin_ns:
            # Sleep in chunks to allow stop_event checking
            sleep_ns = min(remaining - spin_ns, CHECK_INTERVAL_NS)
            time.sleep(sleep_ns / 1e9)
        else:
            while time.perf_counter_ns() < target_ns:
                if stop_event and stop_event.is_set():
                    return False
                pass
            return True
