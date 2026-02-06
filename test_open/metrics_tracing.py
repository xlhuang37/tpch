"""
ClickHouse metrics querying and periodic collection functions.
"""

import asyncio
import csv
import io
import json
import os
import threading
import time
import urllib.request
from typing import List, Dict, Any, Optional

from .models import ScheduleEventsTrace, ScheduleMetricsTrace, QueryProfileTrace


def query_system_events(url: str) -> List[List[str]]:
    """
    Query system.events from ClickHouse and return parsed CSV rows.
    Each row is [event_name, value, description].
    """
    query = "SELECT event, value, description FROM system.events FORMAT CSV"
    
    try:
        req = urllib.request.Request(url, data=query.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode('utf-8')
    except Exception as e:
        print(f"Warning: Failed to query system.events: {e}")
        return []
    
    # Parse CSV response
    rows = []
    for line in body.strip().split('\n'):
        if not line:
            continue
        # CSV format: "event_name",value,"description"
        # Use csv reader for proper parsing
        reader = csv.reader(io.StringIO(line))
        for row in reader:
            if len(row) >= 2:
                rows.append(row)
    return rows


def query_system_metrics(url: str) -> List[List[str]]:
    """
    Query system.metrics from ClickHouse and return parsed CSV rows.
    Each row is [metric_name, value, description].
    """
    query = "SELECT metric, value, description FROM system.metrics FORMAT CSV"
    
    try:
        req = urllib.request.Request(url, data=query.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode('utf-8')
    except Exception as e:
        print(f"Warning: Failed to query system.metrics: {e}")
        return []
    
    # Parse CSV response
    rows = []
    for line in body.strip().split('\n'):
        if not line:
            continue
        reader = csv.reader(io.StringIO(line))
        for row in reader:
            if len(row) >= 2:
                rows.append(row)
    return rows


def save_system_events(output_dir: str, schedule_name: str, suffix: str, 
                       events_data: List[List[str]]) -> str:
    """Save system.events data to a CSV file."""
    filename = f"{schedule_name}_events_{suffix}.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event", "value", "description"])
        for row in events_data:
            w.writerow(row)
    
    return filepath


def compute_events_diff(start_events: List[List[str]], end_events: List[List[str]], 
                        output_dir: str, schedule_name: str) -> str:
    """
    Compute the difference (end - start) for each event and save to CSV.
    """
    # Build lookup from event name to value for start events
    start_map: Dict[str, int] = {}
    desc_map: Dict[str, str] = {}
    for row in start_events:
        if len(row) >= 2:
            event_name = row[0]
            try:
                start_map[event_name] = int(row[1])
            except ValueError:
                start_map[event_name] = 0
            if len(row) >= 3:
                desc_map[event_name] = row[2]
    
    # Build lookup for end events
    end_map: Dict[str, int] = {}
    for row in end_events:
        if len(row) >= 2:
            event_name = row[0]
            try:
                end_map[event_name] = int(row[1])
            except ValueError:
                end_map[event_name] = 0
            if len(row) >= 3 and event_name not in desc_map:
                desc_map[event_name] = row[2]
    
    # Compute differences
    all_events = set(start_map.keys()) | set(end_map.keys())
    diff_rows = []
    for event_name in sorted(all_events):
        start_val = start_map.get(event_name, 0)
        end_val = end_map.get(event_name, 0)
        diff_val = end_val - start_val
        desc = desc_map.get(event_name, "")
        diff_rows.append([event_name, start_val, end_val, diff_val, desc])
    
    # Save to file
    filename = f"{schedule_name}_events_diff.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event", "start_value", "end_value", "diff", "description"])
        for row in diff_rows:
            w.writerow(row)
    
    return filepath


def save_schedule_events_traces(output_dir: str, schedule_name: str, 
                                 traces: List[ScheduleEventsTrace]) -> str:
    """
    Save periodic system.events traces collected during schedule execution.
    Each row is one trace: timestamp_ms, followed by event data.
    Format: timestamp_ms,event1,value1,desc1,event2,value2,desc2,...
    """
    filename = f"{schedule_name}_periodic_events.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Header: timestamp_ms followed by flattened event data
        w.writerow(["timestamp_ms", "events_data"])
        for trace in traces:
            # Flatten events into a single comma-separated string for the row
            events_str = ";".join([
                f"{row[0]}={row[1]}" + (f"({row[2]})" if len(row) > 2 else "")
                for row in trace.events
            ])
            w.writerow([f"{trace.timestamp_ms:.2f}", events_str])
    
    return filepath


def save_schedule_metrics_traces(output_dir: str, schedule_name: str, 
                                 traces: List[ScheduleMetricsTrace]) -> str:
    """
    Save periodic system.metrics traces collected during schedule execution.
    Each row is one trace: timestamp_ms, followed by metric data.
    Format: timestamp_ms,metrics_data
    """
    filename = f"{schedule_name}_periodic_metrics.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "metrics_data"])
        for trace in traces:
            metrics_str = ";".join([
                f"{row[0]}={row[1]}" + (f"({row[2]})" if len(row) > 2 else "")
                for row in trace.metrics
            ])
            w.writerow([f"{trace.timestamp_ms:.2f}", metrics_str])
    
    return filepath


def query_profile_events(url: str, query_id: str) -> Dict[str, Any]:
    """
    Query ProfileEvents from system.processes for a specific query_id.
    Returns the ProfileEvents map or empty dict if not found.
    """
    query = f"SELECT ProfileEvents FROM system.processes WHERE query_id='{query_id}' FORMAT JSON"
    
    try:
        req = urllib.request.Request(url, data=query.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=5) as response:
            body = response.read().decode('utf-8')
    except Exception as e:
        # Query might not be running anymore or connection issue
        return {}
    
    try:
        data = json.loads(body)
        if data.get("data") and len(data["data"]) > 0:
            profile_events = data["data"][0].get("ProfileEvents", {})
            return profile_events if isinstance(profile_events, dict) else {}
    except Exception:
        pass
    
    return {}


def profile_events_tracer_thread(
    url: str,
    query_id: str,
    period_ms: int,
    traces: List[QueryProfileTrace],
    traces_lock: threading.Lock,
    query_start_ns: int,
    stop_event: threading.Event,
) -> None:
    """
    Background thread to periodically sample ProfileEvents from system.processes
    for a specific query while it's running.
    
    Args:
        url: ClickHouse HTTP endpoint
        query_id: The query_id to trace
        period_ms: Sampling period in milliseconds
        traces: List to append traces to (shared with caller)
        traces_lock: Lock for thread-safe access to traces list
        query_start_ns: Start time of the query in nanoseconds (for relative timestamps)
        stop_event: Event to signal when query has completed
    """
    period_seconds = period_ms / 1000.0
    
    while not stop_event.is_set():
        # Sample ProfileEvents
        profile_events = query_profile_events(url, query_id)
        timestamp_ms = (time.perf_counter_ns() - query_start_ns) / 1e6
        
        if profile_events:
            with traces_lock:
                traces.append(QueryProfileTrace(
                    timestamp_ms=timestamp_ms,
                    profile_events=profile_events
                ))
        
        # Wait for next sample period or until query completes
        if stop_event.wait(timeout=period_seconds):
            break  # Query completed, exit loop
    
    # One final sample after query completes to capture final state
    profile_events = query_profile_events(url, query_id)
    timestamp_ms = (time.perf_counter_ns() - query_start_ns) / 1e6
    if profile_events:
        with traces_lock:
            traces.append(QueryProfileTrace(
                timestamp_ms=timestamp_ms,
                profile_events=profile_events
            ))


def save_query_profile_traces(per_query_dir: str, query_id: str, 
                               traces: List[QueryProfileTrace]) -> str:
    """
    Save ProfileEvents traces for a single query to its own file.
    """
    # Sanitize query_id for filename
    safe_query_id = query_id.replace("/", "_").replace("\\", "_")
    filename = f"{safe_query_id}.csv"
    filepath = os.path.join(per_query_dir, filename)
    
    if not traces:
        # Skip creating file if no traces were collected
        return ""
    
    # Collect all unique event names across all traces
    all_event_names = set()
    for trace in traces:
        all_event_names.update(trace.profile_events.keys())
    sorted_event_names = sorted(all_event_names)
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Header: timestamp_ms followed by each event name
        w.writerow(["timestamp_ms"] + sorted_event_names)
        for trace in traces:
            row = [f"{trace.timestamp_ms:.2f}"]
            for event_name in sorted_event_names:
                row.append(trace.profile_events.get(event_name, 0))
            w.writerow(row)
    
    return filepath


async def periodic_system_events_collector(
    url: str,
    period_ms: int,
    traces: List[ScheduleEventsTrace],
    traces_lock: asyncio.Lock,
    t0_ns: int,
    stop_event: asyncio.Event,
) -> None:
    """
    Background task to periodically collect system.events during schedule execution.
    """
    while not stop_event.is_set():
        # Collect events
        events = query_system_events(url)
        timestamp_ms = (time.perf_counter_ns() - t0_ns) / 1e6
        
        async with traces_lock:
            traces.append(ScheduleEventsTrace(
                timestamp_ms=timestamp_ms,
                events=events
            ))
        
        # Wait for next collection period
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=period_ms / 1000.0)
        except asyncio.TimeoutError:
            pass  # Continue collecting


async def periodic_system_metrics_collector(
    url: str,
    period_ms: int,
    traces: List[ScheduleMetricsTrace],
    traces_lock: asyncio.Lock,
    t0_ns: int,
    stop_event: asyncio.Event,
) -> None:
    """
    Background task to periodically collect system.metrics during schedule execution.
    """
    while not stop_event.is_set():
        # Collect metrics
        metrics = query_system_metrics(url)
        timestamp_ms = (time.perf_counter_ns() - t0_ns) / 1e6
        
        async with traces_lock:
            traces.append(ScheduleMetricsTrace(
                timestamp_ms=timestamp_ms,
                metrics=metrics
            ))
        
        # Wait for next collection period
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=period_ms / 1000.0)
        except asyncio.TimeoutError:
            pass  # Continue collecting


async def query_profile_collector(
    url: str,
    query_id: str,
    period_ms: int,
    traces: List[QueryProfileTrace],
    traces_lock: asyncio.Lock,
    query_start_ns: int,
    stop_event: asyncio.Event,
) -> None:
    """
    Background task to periodically collect ProfileEvents for a specific query.
    """
    while not stop_event.is_set():
        # Collect ProfileEvents
        profile_events = query_profile_events(url, query_id)
        timestamp_ms = (time.perf_counter_ns() - query_start_ns) / 1e6
        
        if profile_events:  # Only record if we got data
            async with traces_lock:
                traces.append(QueryProfileTrace(
                    timestamp_ms=timestamp_ms,
                    profile_events=profile_events
                ))
        
        # Wait for next collection period
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=period_ms / 1000.0)
        except asyncio.TimeoutError:
            pass  # Continue collecting
