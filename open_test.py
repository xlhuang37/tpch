#!/usr/bin/env python3
#python3 open_test.py --schedule=./schedules/test.csv
from __future__ import annotations
from typing import Sequence, Optional, Tuple
import argparse
import asyncio
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import matplotlib.pyplot as plt


@dataclass
class Event:
    at_ms: int
    qid: str

def array_to_bar_chart(
    values: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot `values` as a bar chart. Returns (fig, ax).
    """
    if values is None:
        raise ValueError("values is None")
    n = len(values)
    if n == 0:
        raise ValueError("values is empty")

    if labels is not None and len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) != values length ({n})")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = range(n)
    ax.bar(x, values)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels if labels is not None else [str(i) for i in x])

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax

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
    latency_ms_dict: dict,
) -> None:
    target_ns = t0_ns + event.at_ms * 1_000_000
    await wait_until_ns(target_ns, spin_ns)

    scheduled_ns = target_ns
    actual_send_start_ns = time.perf_counter_ns()
    sched_error_ms = (actual_send_start_ns - scheduled_ns) / 1e6

    async with sem:
        req_start_ns = time.perf_counter_ns()
        try:
            async with session.post(url, data=sql_bytes) as resp:
                # If query returns rows, reading response fully avoids connection pool issues.
                body = await resp.read()
                req_end_ns = time.perf_counter_ns()
                latency_ms = (req_end_ns - req_start_ns) / 1e6
                latency_ms_dict[event.qid].append(latency_ms)

                if resp.status != 200:
                    snippet = body[:200].decode("utf-8", errors="replace")
                    print(f"[{event.at_ms:>6} ms] {event.qid}: HTTP {resp.status} "
                          f"lat={latency_ms:.2f}ms sched_err={sched_error_ms:.2f}ms "
                          f"resp='{snippet}'")
                else:
                    print(f"[{event.at_ms:>6} ms] {event.qid}: OK "
                          f"lat={latency_ms:.2f}ms sched_err={sched_error_ms:.2f}ms")
        except Exception as e:
            req_end_ns = time.perf_counter_ns()
            latency_ms = (req_end_ns - req_start_ns) / 1e6
            print(f"[{event.at_ms:>6} ms] {event.qid}: ERROR {e} "
                  f"lat={latency_ms:.2f}ms sched_err={sched_error_ms:.2f}ms")

async def main():
    ap = argparse.ArgumentParser(description="Replay ClickHouse HTTP workload by timestamp (ms).")
    ap.add_argument("--schedule", required=True, help="Path to schedule CSV (columns: at_ms,qid)")
    ap.add_argument("--queries-dir", default="./queries/", help="Directory containing <qid>.sql files")
    ap.add_argument("--url", default="http://localhost:8123/", help="ClickHouse HTTP endpoint URL")

    ap.add_argument("--max-concurrency", type=int, default=50,
                    help="Max in-flight HTTP requests from the client")
    ap.add_argument("--spin-ns", type=int, default=100000,
                    help="Final busy-wait window for timing accuracy (microseconds)")
    args = ap.parse_args()

    events = read_schedule_csv(args.schedule)

    # Preload SQL bytes for each qid (so send path is lightweight)
    sql_cache = {}
    latency_ms_dict= {}
    qid_list = []
    for e in events:
        if e.qid not in sql_cache:
            latency_ms_dict[e.qid] = []
            sql_cache[e.qid] = load_query(args.queries_dir, e.qid)
            qid_list.append(e.qid)

    connector = aiohttp.TCPConnector(limit=args.max_concurrency, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=None)  # let queries run; adjust if desired
    sem = asyncio.Semaphore(args.max_concurrency)
    spin_ns = args.spin_ns

    t0_ns = time.perf_counter_ns()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            asyncio.create_task(
                send_one(
                    session=session,
                    url=args.url,
                    event=e,
                    sql_bytes=sql_cache[e.qid],
                    t0_ns=t0_ns,
                    spin_ns=spin_ns,
                    sem=sem,
                    latency_ms_dict=latency_ms_dict,
                )
            )
            for e in events
        ]
        await asyncio.gather(*tasks)

    for qid in qid_list:
        print("Class 1 Query Averege Latency: ", sum(latency_ms_dict[qid]) / 1000 / latency_ms_dict[qid].count, " seconds")
        array_to_bar_chart(sorted(latency_ms_dict[qid]))


if __name__ == "__main__":
    asyncio.run(main())
