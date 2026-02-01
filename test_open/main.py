#!/usr/bin/env python3
"""
Main entry point for open-loop ClickHouse workload testing.

Usage:
    python test_open.py --trace-processes "--query-trace-period-ms 1000 --trace-metrics --schedule-trace-period-ms 1000
"""

import argparse
import asyncio
import glob
import os
import threading
import time
from datetime import datetime
from queue import PriorityQueue
from typing import List, Dict, Optional

from .models import (
    LatencyRecord, DroppedRecord, 
    ScheduleEventsTrace, ScheduleMetricsTrace, QueryProfileTrace
)
from .system_info import save_hardware_specs, save_pre_run_state
from .statistics import save_raw_data, save_statistics
from .metrics_tracing import (
    query_system_events, save_system_events, compute_events_diff,
    save_schedule_events_traces, save_schedule_metrics_traces,
    save_query_profile_traces, periodic_system_events_collector,
    periodic_system_metrics_collector
)
from .query_execution import producer_thread, consumer_thread
from .utils import read_schedule_csv, load_query
from .scheduler import (
    WorkloadTracker, scheduler_thread, DEFAULT_TOTAL_CORES
)


async def run_schedule(
    schedule_path: str,
    queries_dir: str,
    url: str,
    max_concurrency: int,
    spin_ns: int,
    output_dir: str,
    schedule_trace_period_ms: int,
    query_trace_period_ms: int,
    trace_events: bool = False,
    trace_processes: bool = False,
    trace_metrics: bool = False,
    continue_upon_drop: bool = False,
    pause_seconds: float = 10.0,
    enable_scheduler: bool = False,
    scheduler_interval_ms: int = 500,
    total_cores: int = DEFAULT_TOTAL_CORES,
) -> None:
    """Run a single schedule and save results using producer/consumer threads."""
    schedule_name = os.path.splitext(os.path.basename(schedule_path))[0]
    print(f"\n{'='*60}")
    print(f"Running schedule: {schedule_path}")
    print(f"Schedule trace period: {schedule_trace_period_ms}ms, Query trace period: {query_trace_period_ms}ms")
    print(f"Pause on failure: {pause_seconds}s")
    if enable_scheduler:
        print(f"Scheduler: enabled (interval={scheduler_interval_ms}ms, cores={total_cores})")
    print(f"{'='*60}\n")
    
    # Record pre-run system state (CPU/memory usage to detect interference)
    print("Recording pre-run system state...")
    pre_run_path = save_pre_run_state(output_dir, schedule_name)
    print(f"  Pre-run state saved to: {pre_run_path}\n")
    
    # Create output folder for query profile traces if enabled
    query_traces_dir = None
    if trace_processes:
        query_traces_dir = os.path.join(output_dir, f"{schedule_name}_query_traces")
        os.makedirs(query_traces_dir, exist_ok=True)
        print(f"Query profile traces will be saved to: {query_traces_dir}")
    
    # Collect system.events at start
    print("Collecting system.events (start)...")
    start_events = query_system_events(url)
    
    events = read_schedule_csv(schedule_path)

    # Preload SQL bytes and workload for each qid (so send path is lightweight)
    sql_cache = {}  # qid -> (sql_bytes, workload_name or None)
    qid_list = []
    for e in events:
        if e.qid not in sql_cache:
            sql_cache[e.qid] = load_query(queries_dir, e.qid)
            qid_list.append(e.qid)
    
    # Thread-safe data structures
    latency_records: List[LatencyRecord] = []
    dropped_records: List[DroppedRecord] = []
    query_profile_traces: Dict[str, List[QueryProfileTrace]] = {}  # query_id -> traces
    records_lock = threading.Lock()
    
    # Workload tracker for scheduler (shared between consumer and scheduler)
    workload_tracker = WorkloadTracker() if enable_scheduler else None
    
    # Priority queue for producer/consumer communication
    priority_queue: PriorityQueue = PriorityQueue()
    
    # Threading events for coordination
    stop_event = threading.Event()           # Signal early termination (e.g., on error)
    producer_done_event = threading.Event()  # Producer signals it's done producing
    termination_event = threading.Event()    # Consumer signals producer can exit
    
    # Initialize periodic collection for async trace collectors (conditional based on flags)
    schedule_events_traces: List[ScheduleEventsTrace] = []
    schedule_events_traces_lock = asyncio.Lock()
    schedule_metrics_traces: List[ScheduleMetricsTrace] = []
    schedule_metrics_traces_lock = asyncio.Lock()
    schedule_stop_event = asyncio.Event()

    t0_ns = time.perf_counter_ns()
    
    # Start periodic collectors (conditional) - these run in asyncio
    events_collector_task = None
    metrics_collector_task = None
    
    if trace_events:
        events_collector_task = asyncio.create_task(
            periodic_system_events_collector(
                url=url,
                period_ms=schedule_trace_period_ms,
                traces=schedule_events_traces,
                traces_lock=schedule_events_traces_lock,
                t0_ns=t0_ns,
                stop_event=schedule_stop_event,
            )
        )
    
    if trace_metrics:
        metrics_collector_task = asyncio.create_task(
            periodic_system_metrics_collector(
                url=url,
                period_ms=schedule_trace_period_ms,
                traces=schedule_metrics_traces,
                traces_lock=schedule_metrics_traces_lock,
                t0_ns=t0_ns,
                stop_event=schedule_stop_event,
            )
        )
    
    # Create and start producer thread
    producer = threading.Thread(
        target=producer_thread,
        args=(events, sql_cache, priority_queue, t0_ns, spin_ns, stop_event,
              producer_done_event, termination_event),
        name="producer"
    )
    
    # Create and start consumer thread
    consumer = threading.Thread(
        target=consumer_thread,
        args=(priority_queue, url, t0_ns, latency_records, dropped_records, 
              records_lock, pause_seconds, max_concurrency, stop_event,
              producer_done_event, termination_event, trace_processes, 
              query_trace_period_ms, query_profile_traces, workload_tracker),
        name="consumer"
    )
    
    # Create scheduler thread (optional)
    scheduler = None
    if enable_scheduler and workload_tracker:
        scheduler = threading.Thread(
            target=scheduler_thread,
            args=(url, workload_tracker, scheduler_interval_ms, total_cores, stop_event),
            name="scheduler"
        )
    
    print("Starting producer and consumer threads...")
    producer.start()
    consumer.start()
    if scheduler:
        scheduler.start()
    
    # Wait for threads to complete (run in executor to not block asyncio)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, producer.join)
    await loop.run_in_executor(None, consumer.join)
    
    # Stop scheduler thread (stop_event is already set by consumer when done)
    if scheduler:
        stop_event.set()  # Ensure scheduler stops
        await loop.run_in_executor(None, scheduler.join)
    
    print("Producer, consumer, and scheduler threads completed.")
    
    # Stop the periodic collectors
    schedule_stop_event.set()
    if events_collector_task:
        await events_collector_task
    if metrics_collector_task:
        await metrics_collector_task

    # Collect system.events at end
    print("\nCollecting system.events (end)...")
    end_events = query_system_events(url)

    # Save results
    raw_path = save_raw_data(output_dir, schedule_name, latency_records, dropped_records)
    stats_path = save_statistics(output_dir, schedule_name, latency_records, dropped_records, qid_list)
    
    # Save system.events data (always saved for start/end/diff)
    start_events_path = save_system_events(output_dir, schedule_name, "start", start_events)
    end_events_path = save_system_events(output_dir, schedule_name, "end", end_events)
    diff_events_path = compute_events_diff(start_events, end_events, output_dir, schedule_name)
    
    # Save periodic traces (conditional)
    periodic_events_path = None
    periodic_metrics_path = None
    
    if trace_events:
        periodic_events_path = save_schedule_events_traces(output_dir, schedule_name, schedule_events_traces)
    
    if trace_metrics:
        periodic_metrics_path = save_schedule_metrics_traces(output_dir, schedule_name, schedule_metrics_traces)
    
    # Save query profile traces (conditional)
    saved_query_traces_count = 0
    if trace_processes and query_traces_dir and query_profile_traces:
        for query_id, traces in query_profile_traces.items():
            if traces:
                save_query_profile_traces(query_traces_dir, query_id, traces)
                saved_query_traces_count += 1
    
    print(f"\nSchedule '{schedule_name}' completed.")
    print(f"  Raw data saved to: {raw_path}")
    print(f"  Statistics saved to: {stats_path}")
    print(f"  Events (start) saved to: {start_events_path}")
    print(f"  Events (end) saved to: {end_events_path}")
    print(f"  Events (diff) saved to: {diff_events_path}")
    if periodic_events_path:
        print(f"  Periodic events ({len(schedule_events_traces)} traces) saved to: {periodic_events_path}")
    if periodic_metrics_path:
        print(f"  Periodic metrics ({len(schedule_metrics_traces)} traces) saved to: {periodic_metrics_path}")
    if trace_processes and query_traces_dir:
        print(f"  Query profile traces ({saved_query_traces_count} queries) saved to: {query_traces_dir}")
    
    # Print summary to console
    if latency_records:
        all_latencies = [r.latency_ms for r in latency_records]
        avg_latency = sum(all_latencies) / len(all_latencies)
        print(f"  Completed queries: {len(latency_records)}")
        print(f"  Average latency: {avg_latency:.2f} ms")
    if dropped_records:
        print(f"  Dropped queries: {len(dropped_records)}")


async def main():
    """Main entry point for the open-loop workload tester."""
    ap = argparse.ArgumentParser(description="Replay ClickHouse HTTP workload by timestamp (ms).")
    ap.add_argument("--schedules-dir", default="./schedules/", 
                    help="Directory containing schedule CSV files (default: ./schedules/)")
    ap.add_argument("--queries-dir", default="./queries/", help="Directory containing <qid>.sql files")
    ap.add_argument("--url", default="http://localhost:8123/", help="ClickHouse HTTP endpoint URL")
    ap.add_argument("--max-concurrency", type=int, default=0,
                    help="Max in-flight HTTP requests from the client (0 means no limit)")
    ap.add_argument("--spin-ns", type=int, default=100000,
                    help="Final busy-wait window for timing accuracy (microseconds)")
    ap.add_argument("--schedule-trace-period-ms", type=int, default=1000,
                    help="Period in ms for collecting system.events/metrics during schedule execution (default: 1000)")
    ap.add_argument("--query-trace-period-ms", type=int, default=5000,
                    help="Period in ms for collecting ProfileEvents per query (default: 5000)")
    ap.add_argument("--trace-events", action="store_true", default=False,
                    help="Enable periodic tracing of system.events (default: False)")
    ap.add_argument("--trace-processes", action="store_true", default=False,
                    help="Enable periodic tracing of system.processes ProfileEvents per query (default: False)")
    ap.add_argument("--trace-metrics", action="store_true", default=False,
                    help="Enable periodic tracing of system.metrics (default: False)")
    ap.add_argument("--continue-upon-drop", action="store_true", default=False,
                    help="If set to true, will not terminate the run immediately if any query is rejected by server (default: False)")
    ap.add_argument("--pause-seconds", type=float, default=10.0,
                    help="Seconds to pause on failure before checking responses and reinserting (default: 10)")
    ap.add_argument("--enable-scheduler", action="store_true", default=False,
                    help="Enable scheduler thread for dynamic workload settings (default: False)")
    ap.add_argument("--scheduler-interval-ms", type=int, default=500,
                    help="Scheduler update interval in milliseconds (default: 500)")
    ap.add_argument("--total-cores", type=int, default=DEFAULT_TOTAL_CORES,
                    help=f"Total cores to distribute among workloads (default: {DEFAULT_TOTAL_CORES})")
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
    print(f"\nConfiguration:")
    print(f"  Max concurrency: {args.max_concurrency} (0 = unlimited)")
    print(f"  Dispatch mode: Producer/Consumer with Priority Queue")
    print(f"  Pause on failure: {args.pause_seconds}s")
    print(f"  Trace system.events: {args.trace_events}")
    print(f"  Trace system.processes: {args.trace_processes}")
    print(f"  Trace system.metrics: {args.trace_metrics}")
    print(f"  Schedule trace period: {args.schedule_trace_period_ms}ms")
    print(f"  Query trace period: {args.query_trace_period_ms}ms")
    print(f"  Continue upon drop: {args.continue_upon_drop}")
    print(f"  Scheduler enabled: {args.enable_scheduler}")
    if args.enable_scheduler:
        print(f"  Scheduler interval: {args.scheduler_interval_ms}ms")
        print(f"  Total cores: {args.total_cores}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output/open", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Save hardware specifications (once per run)
    print("\nRecording hardware specifications...")
    hw_specs_path = save_hardware_specs(output_dir)
    print(f"  Hardware specs saved to: {hw_specs_path}")

    # Run each schedule sequentially
    for schedule_path in schedule_paths:
        await run_schedule(
            schedule_path=schedule_path,
            queries_dir=args.queries_dir,
            url=args.url,
            max_concurrency=args.max_concurrency,
            spin_ns=args.spin_ns,
            output_dir=output_dir,
            schedule_trace_period_ms=args.schedule_trace_period_ms,
            query_trace_period_ms=args.query_trace_period_ms,
            trace_events=args.trace_events,
            trace_processes=args.trace_processes,
            trace_metrics=args.trace_metrics,
            continue_upon_drop=args.continue_upon_drop,
            pause_seconds=args.pause_seconds,
            enable_scheduler=args.enable_scheduler,
            scheduler_interval_ms=args.scheduler_interval_ms,
            total_cores=args.total_cores,
        )

    print(f"\n{'='*60}")
    print(f"All schedules completed. Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
