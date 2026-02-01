"""
Core query execution logic with producer/consumer pattern.
"""

import threading
import time
import uuid
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import requests

from .models import (
    Event, LatencyRecord, DroppedRecord, QueryProfileTrace,
    PriorityQueueEntry, InFlightQuery, QueryResult
)
from .metrics_tracing import profile_events_tracer_thread
from .utils import wait_until_ns_sync


def generate_query_id(qid: str, at_ms: int) -> str:
    """Generate a unique query ID for ClickHouse."""
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{qid}_{at_ms}_{unique_suffix}"


def producer_thread(
    events: List[Event],
    sql_cache: Dict[str, tuple],
    priority_queue: PriorityQueue,
    t0_ns: int,
    spin_ns: int,
    stop_event: threading.Event,
    producer_done_event: threading.Event,
    termination_event: threading.Event,
) -> None:
    """
    Producer thread: enqueues queries at scheduled times.
    
    Iterates through schedule events in order, waits until the scheduled
    arrival time for each event, then places it into the priority queue.
    After producing all events, signals done and waits for termination.
    """
    entry_counter = 0
    for event in events:
        if stop_event.is_set():
            print(f"[Producer] Stop signal received, exiting...")
            break
        
        # Wait until scheduled arrival time (with stop_event check)
        target_ns = t0_ns + event.at_ms * 1_000_000
        if not wait_until_ns_sync(target_ns, spin_ns, stop_event):
            print(f"[Producer] Stop signal received during wait, exiting...")
            break
        
        # Record actual arrival time
        arrival_ns = time.perf_counter_ns()
        
        # Get SQL and workload from cache
        sql_bytes, workload = sql_cache[event.qid]
        
        # Create priority queue entry
        entry = PriorityQueueEntry(
            priority=event.at_ms,
            entry_id=entry_counter,
            event=event,
            sql_bytes=sql_bytes,
            workload=workload,
            arrival_ns=arrival_ns,
        )
        
        priority_queue.put(entry)
        entry_counter += 1
    
    # Signal that producer is done producing
    producer_done_event.set()
    print(f"[Producer] Finished enqueueing {entry_counter} events, waiting for termination...")
    
    # Wait for termination signal from consumer
    termination_event.wait()
    print(f"[Producer] Termination received, exiting.")


def execute_query_sync(
    session: requests.Session,
    entry: PriorityQueueEntry,
    url: str,
    t0_ns: int,
    trace_processes: bool = False,
    query_trace_period_ms: int = 100,
) -> QueryResult:
    """
    Execute a single query synchronously using requests.
    Returns QueryResult with success/failure status and timing data.
    
    If trace_processes is True, spawns a background thread to sample
    ProfileEvents from system.processes during query execution.
    """
    event = entry.event
    arrival_ms = (entry.arrival_ns - t0_ns) / 1e6
    
    # Generate unique query ID
    unique_suffix = uuid.uuid4().hex[:8]
    query_id = f"{event.qid}_{event.at_ms}_{unique_suffix}"
    
    query_url = f"{url.rstrip('/')}/?query_id={query_id}"
    if entry.workload:
        query_url += f"&workload={entry.workload}"
    
    # Set up profile tracing if enabled
    profile_traces: List[QueryProfileTrace] = []
    traces_lock = threading.Lock()
    tracer_stop_event = threading.Event()
    tracer_thread = None
    
    try:
        req_start_ns = time.perf_counter_ns()
        start_ms = (req_start_ns - t0_ns) / 1e6
        
        # Start profile tracer thread if enabled
        if trace_processes:
            tracer_thread = threading.Thread(
                target=profile_events_tracer_thread,
                args=(url, query_id, query_trace_period_ms, profile_traces, 
                      traces_lock, t0_ns, tracer_stop_event),
                daemon=True,
            )
            tracer_thread.start()
        
        response = session.post(query_url, data=entry.sql_bytes, timeout=None)
        
        req_end_ns = time.perf_counter_ns()
        end_ms = (req_end_ns - t0_ns) / 1e6
        wait_ms = start_ms - arrival_ms
        latency_ms = end_ms - arrival_ms
        exec_ms = end_ms - start_ms
        
        # Stop tracer thread and wait for it to finish
        if tracer_thread:
            tracer_stop_event.set()
            tracer_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        # Get final traces (thread-safe copy)
        final_traces = None
        if trace_processes:
            with traces_lock:
                final_traces = list(profile_traces) if profile_traces else None
        
        if response.status_code == 200:
            # Success
            latency_record = LatencyRecord(
                arrival_ms=arrival_ms,
                start_ms=start_ms,
                end_ms=end_ms,
                latency_ms=latency_ms,
                wait_ms=wait_ms,
                qid=event.qid
            )
            trace_info = f" traces={len(final_traces)}" if final_traces else ""
            print(f"[{event.at_ms:>6} ms] {event.qid} ({query_id}): OK "
                  f"lat={latency_ms:.2f}ms wait={wait_ms:.2f}ms exec={exec_ms:.2f}ms{trace_info}")
            return QueryResult(
                success=True,
                should_terminate=False,
                entry=entry,
                event=event,
                latency_record=latency_record,
                profile_traces=final_traces,
                query_id=query_id,
            )
        else:
            # Server rejected the query
            snippet = response.text[:200]
            print(f"[{event.at_ms:>6} ms] {event.qid} ({query_id}): HTTP {response.status_code} "
                  f"- FAILED resp='{snippet}'")
            return QueryResult(
                success=False,
                should_terminate=False,
                entry=entry,
                event=event,
                error_message=f"HTTP {response.status_code}: {snippet}",
                profile_traces=None,  # No traces on failure
                query_id=query_id,
            )
            
    except requests.exceptions.ConnectionError as e:
        # Stop tracer thread on exception
        if tracer_thread:
            tracer_stop_event.set()
            tracer_thread.join(timeout=1.0)
        print(f"[{event.at_ms:>6} ms] {event.qid}: CONNECTION ERROR - {e}")
        return QueryResult(
            success=False,
            should_terminate=True,  # Connection error should terminate
            entry=entry,
            event=event,
            error_message=str(e),
            profile_traces=None,
            query_id=query_id,
        )
    except requests.exceptions.RequestException as e:
        if tracer_thread:
            tracer_stop_event.set()
            tracer_thread.join(timeout=1.0)
        print(f"[{event.at_ms:>6} ms] {event.qid}: REQUEST ERROR - {e}")
        return QueryResult(
            success=False,
            should_terminate=False,
            entry=entry,
            event=event,
            error_message=str(e),
            profile_traces=None,
            query_id=query_id,
        )
    except Exception as e:
        if tracer_thread:
            tracer_stop_event.set()
            tracer_thread.join(timeout=1.0)
        print(f"[{event.at_ms:>6} ms] {event.qid}: UNEXPECTED ERROR - {e}")
        return QueryResult(
            success=False,
            should_terminate=True,  # Unexpected errors should terminate
            entry=entry,
            event=event,
            error_message=str(e),
            profile_traces=None,
            query_id=query_id,
        )


def consumer_thread(
    priority_queue: PriorityQueue,
    url: str,
    t0_ns: int,
    latency_records: List[LatencyRecord],
    dropped_records: List[DroppedRecord],
    records_lock: threading.Lock,
    pause_seconds: float,
    max_concurrency: int,
    stop_event: threading.Event,
    producer_done_event: threading.Event,
    termination_event: threading.Event,
    trace_processes: bool = False,
    query_trace_period_ms: int = 100,
    query_profile_traces: Optional[Dict[str, List[QueryProfileTrace]]] = None,
    workload_tracker: Optional['WorkloadTracker'] = None,
) -> None:
    """
    Consumer thread: dispatches queries and handles failures.
    
    Dequeues from priority queue, dispatches queries using a thread pool,
    and handles failures by pausing and reinserting failed queries.
    When queue is exhausted and producer is done, waits for in-flight queries
    then signals termination to the producer.
    
    If trace_processes is True, profile events are collected for each query
    and stored in query_profile_traces dict (keyed by query_id).
    
    If workload_tracker is provided, tracks running query counts per workload
    class for the scheduler thread.
    """
    # Use a thread pool for concurrent query execution
    pool_size = max_concurrency if max_concurrency > 0 else 100
    executor = ThreadPoolExecutor(max_workers=pool_size)
    
    in_flight: Dict[int, InFlightQuery] = {}  # entry_id -> InFlightQuery
    session = requests.Session()
    
    # Configure session for connection pooling
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,  # We handle retries ourselves
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    def process_completed_queries() -> List[QueryResult]:
        """Check and process all completed queries."""
        completed_results = []
        completed_ids = []
        
        for entry_id, in_flight_query in list(in_flight.items()):
            if in_flight_query.future.done():
                try:
                    result = in_flight_query.future.result()
                    completed_results.append(result)
                except Exception as e:
                    # Future raised an exception
                    completed_results.append(QueryResult(
                        success=False,
                        should_terminate=True,
                        entry=in_flight_query.entry,
                        event=in_flight_query.entry.event,
                        error_message=str(e),
                    ))
                completed_ids.append(entry_id)
                
                # Decrement workload count when query completes
                if workload_tracker and in_flight_query.entry.workload:
                    workload_tracker.decrement(in_flight_query.entry.workload)
        
        # Remove completed from in_flight
        for entry_id in completed_ids:
            del in_flight[entry_id]
        
        return completed_results
    
    def check_should_terminate(results: List[QueryResult]) -> bool:
        """Check if any result indicates termination is required."""
        return any(r.should_terminate for r in results)
    
    def record_results(results: List[QueryResult]) -> List[PriorityQueueEntry]:
        """Record successful results and return failed entries for reinsertion."""
        failed_entries = []
        
        with records_lock:
            for result in results:
                if result.success and result.latency_record:
                    latency_records.append(result.latency_record)
                    # Collect profile traces if available
                    if trace_processes and query_profile_traces is not None:
                        if result.query_id and result.profile_traces:
                            query_profile_traces[result.query_id] = result.profile_traces
                elif result.entry:
                    failed_entries.append(result.entry)
        
        return failed_entries
    
    def handle_failure(initial_failed: List[PriorityQueueEntry], should_terminate: bool) -> bool:
        """
        Handle failure: pause, check responses, reinsert failed queries.
        Returns True if the program should terminate.
        """
        if should_terminate:
            print(f"\n*** TERMINATION REQUESTED - recording dropped queries ***")
            # Record all failed entries as dropped
            with records_lock:
                for entry in initial_failed:
                    arrival_ms = (entry.arrival_ns - t0_ns) / 1e6
                    dropped_records.append(DroppedRecord(arrival_ms=arrival_ms, qid=entry.event.qid))
            return True
        
        print(f"\n*** Failure detected - pausing for {pause_seconds:.0f}s ***")
        time.sleep(pause_seconds)
        
        # Collect all completed responses after pause
        completed_results = process_completed_queries()
        
        # Check if any completed results require termination
        if check_should_terminate(completed_results):
            print(f"*** TERMINATION REQUESTED during pause ***")
            # Record all as dropped
            with records_lock:
                for entry in initial_failed:
                    arrival_ms = (entry.arrival_ns - t0_ns) / 1e6
                    dropped_records.append(DroppedRecord(arrival_ms=arrival_ms, qid=entry.event.qid))
                for result in completed_results:
                    if not result.success and result.entry:
                        arrival_ms = (result.entry.arrival_ns - t0_ns) / 1e6
                        dropped_records.append(DroppedRecord(arrival_ms=arrival_ms, qid=result.entry.event.qid))
            return True
        
        # Record results and collect any additional failures
        additional_failed = record_results(completed_results)
        
        # Combine all failed entries
        all_failed = initial_failed + additional_failed
        
        # Reinsert failed queries with original priority (maintains order)
        for entry in all_failed:
            print(f"[Consumer] Reinserting failed query: {entry.event.qid} at_ms={entry.event.at_ms}")
            priority_queue.put(entry)
        
        print(f"*** Resuming dispatch - reinserted {len(all_failed)} failed queries ***\n")
        return False
    
    terminated = False
    
    try:
        while True:
            if stop_event.is_set():
                print("[Consumer] Stop signal received")
                break
            
            # Check completed queries before getting next
            completed_results = process_completed_queries()
            if completed_results:
                should_terminate = check_should_terminate(completed_results)
                failed_entries = record_results(completed_results)
                if failed_entries or should_terminate:
                    if handle_failure(failed_entries, should_terminate):
                        terminated = True
                        stop_event.set()  # Signal producer to stop
                        break
                    continue  # Re-check queue after handling failure
            
            # Wait for concurrency slot if needed
            while max_concurrency > 0 and len(in_flight) >= max_concurrency:
                time.sleep(0.001)  # Small sleep to avoid busy waiting
                completed_results = process_completed_queries()
                if completed_results:
                    should_terminate = check_should_terminate(completed_results)
                    failed_entries = record_results(completed_results)
                    if failed_entries or should_terminate:
                        if handle_failure(failed_entries, should_terminate):
                            terminated = True
                            stop_event.set()  # Signal producer to stop
                            break
            
            if terminated:
                break
            
            # Get next entry from queue (with timeout to allow stop_event check)
            try:
                entry = priority_queue.get(timeout=0.1)
            except:
                # Queue is empty or timeout - check termination conditions
                # Only terminate when: producer done AND queue empty AND no in-flight queries
                # (in-flight queries might fail and need reinsertion, so keep looping)
                if producer_done_event.is_set() and priority_queue.empty() and not in_flight:
                    print("[Consumer] Queue exhausted, producer done, no in-flight - terminating")
                    break
                continue
            
            # Increment workload count before dispatching
            if workload_tracker and entry.workload:
                workload_tracker.increment(entry.workload)
            
            # Dispatch query using thread pool
            future = executor.submit(
                execute_query_sync,
                session,
                entry,
                url,
                t0_ns,
                trace_processes,
                query_trace_period_ms,
            )
            in_flight[entry.entry_id] = InFlightQuery(entry=entry, future=future)

            time.sleep(0.002)

           
        
        # Handle termination: record all in-flight as dropped and signal producer
        if terminated:
            stop_event.set()  # Ensure producer is signaled to stop
            print("[Consumer] Terminating - signaling producer and recording remaining queries as dropped")
            with records_lock:
                for in_flight_query in in_flight.values():
                    entry = in_flight_query.entry
                    arrival_ms = (entry.arrival_ns - t0_ns) / 1e6
                    dropped_records.append(DroppedRecord(arrival_ms=arrival_ms, qid=entry.event.qid))
                    # Decrement workload count for dropped in-flight queries
                    if workload_tracker and entry.workload:
                        workload_tracker.decrement(entry.workload)
            # Drain remaining queue items as dropped
            while True:
                try:
                    entry = priority_queue.get_nowait()
                    arrival_ms = (entry.arrival_ns - t0_ns) / 1e6
                    with records_lock:
                        dropped_records.append(DroppedRecord(arrival_ms=arrival_ms, qid=entry.event.qid))
                except:
                    break
        
        # Note: Normal exit (not terminated) guarantees in_flight is empty
        # because we only break when: producer_done AND queue empty AND not in_flight
        print(f"[Consumer] Finished processing all queries")
        
    finally:
        # Signal termination to producer so it can exit
        termination_event.set()
        executor.shutdown(wait=True)
        session.close()
