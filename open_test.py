#!/usr/bin/env python3
#python3 open_test.py
from __future__ import annotations
from typing import Optional
import argparse
import asyncio
import csv
import glob
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import aiohttp
import io
import platform
import urllib.request
import threading
import requests
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor, Future

# Optional: psutil for detailed system info
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with 'pip install psutil' for detailed system info.")


@dataclass
class Event:
    at_ms: int
    qid: str


@dataclass
class LatencyRecord:
    arrival_ms: float     # Actual arrival time (relative to t0)
    start_ms: float       # Actual query start time (when server accepted)
    end_ms: float         # Query completion time
    latency_ms: float     # end_ms - arrival_ms (total latency including wait)
    wait_ms: float        # start_ms - arrival_ms (time waiting before execution)
    qid: str


@dataclass
class DroppedRecord:
    arrival_ms: float     # Actual arrival time (relative to t0)
    qid: str              # Query type


@dataclass
class ScheduleEventsTrace:
    """A single trace of system.events during schedule execution."""
    timestamp_ms: float   # Relative to schedule start
    events: List[List[str]]  # List of [event, value, description]


@dataclass
class ScheduleMetricsTrace:
    """A single trace of system.metrics during schedule execution."""
    timestamp_ms: float   # Relative to schedule start
    metrics: List[List[str]]  # List of [metric, value, description]


@dataclass
class QueryProfileTrace:
    """A single trace of ProfileEvents for a query."""
    timestamp_ms: float   # Relative to query start
    profile_events: Dict[str, Any]  # ProfileEvents map


@dataclass(order=True)
class PriorityQueueEntry:
    """Entry for the priority queue, ordered by timestamp."""
    priority: int  # at_ms - lower value = higher priority
    entry_id: int = field(compare=False)  # Tie-breaker for FIFO on same timestamp
    event: Event = field(compare=False)
    sql_bytes: bytes = field(compare=False)
    workload: Optional[str] = field(compare=False)
    arrival_ns: int = field(compare=False)  # When it was enqueued


@dataclass
class InFlightQuery:
    """Tracks a dispatched query awaiting response."""
    entry: PriorityQueueEntry
    future: Future


@dataclass
class QueryResult:
    """Unified result of a query execution (sync or async)."""
    success: bool
    should_terminate: bool = False  # If True, program should terminate
    entry: Optional[PriorityQueueEntry] = None  # For sync thread-based execution
    event: Optional[Event] = None  # For async execution
    latency_record: Optional[LatencyRecord] = None
    error_message: Optional[str] = None
    profile_traces: Optional[List[QueryProfileTrace]] = None  # ProfileEvents traces during query execution
    query_id: Optional[str] = None  # The query_id used for this execution


def parse_proc_cpuinfo() -> Dict[str, Any]:
    """Parse /proc/cpuinfo for detailed CPU information (Linux only)."""
    cpuinfo = {}
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
        
        # Extract key fields from first processor entry
        lines = content.split("\n")
        processors = []
        current_proc = {}
        
        for line in lines:
            if line.strip() == "":
                if current_proc:
                    processors.append(current_proc)
                    current_proc = {}
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                current_proc[key.strip()] = value.strip()
        
        if current_proc:
            processors.append(current_proc)
        
        if processors:
            first_proc = processors[0]
            cpuinfo["model_name"] = first_proc.get("model name", "Unknown")
            cpuinfo["vendor_id"] = first_proc.get("vendor_id", "Unknown")
            cpuinfo["cpu_family"] = first_proc.get("cpu family", "Unknown")
            cpuinfo["model"] = first_proc.get("model", "Unknown")
            cpuinfo["stepping"] = first_proc.get("stepping", "Unknown")
            cpuinfo["microcode"] = first_proc.get("microcode", "Unknown")
            cpuinfo["cpu_mhz"] = first_proc.get("cpu MHz", "Unknown")
            cpuinfo["cache_size"] = first_proc.get("cache size", "Unknown")
            cpuinfo["flags"] = first_proc.get("flags", "")[:200] + "..."  # Truncate flags
            cpuinfo["processor_count"] = len(processors)
    except FileNotFoundError:
        cpuinfo["note"] = "/proc/cpuinfo not found (not Linux?)"
    except Exception as e:
        cpuinfo["error"] = str(e)
    
    return cpuinfo


def parse_proc_meminfo() -> Dict[str, Any]:
    """Parse /proc/meminfo for detailed memory information (Linux only)."""
    meminfo = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    # Store key memory fields
                    if key in ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached",
                               "SwapTotal", "SwapFree", "HugePages_Total", "Hugepagesize"]:
                        meminfo[key] = value
    except FileNotFoundError:
        meminfo["note"] = "/proc/meminfo not found (not Linux?)"
    except Exception as e:
        meminfo["error"] = str(e)
    
    return meminfo


def get_dmi_info() -> Dict[str, Any]:
    """Get system DMI information (motherboard, BIOS, etc.) from /sys/class/dmi (Linux only)."""
    dmi = {}
    dmi_path = "/sys/class/dmi/id"
    
    try:
        if not os.path.exists(dmi_path):
            return {"note": "/sys/class/dmi/id not found"}
        
        fields = ["sys_vendor", "product_name", "product_version", "board_vendor", 
                  "board_name", "board_version", "bios_vendor", "bios_version", "bios_date"]
        
        for field in fields:
            field_path = os.path.join(dmi_path, field)
            if os.path.exists(field_path):
                try:
                    with open(field_path, "r") as f:
                        dmi[field] = f.read().strip()
                except PermissionError:
                    dmi[field] = "(permission denied)"
                except Exception:
                    pass
    except Exception as e:
        dmi["error"] = str(e)
    
    return dmi


def get_hardware_specs() -> Dict[str, Any]:
    """Collect CPU and memory specifications."""
    specs = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
    }
    
    # Parse /proc files for detailed Linux info
    specs["proc_cpuinfo"] = parse_proc_cpuinfo()
    specs["proc_meminfo"] = parse_proc_meminfo()
    specs["dmi_info"] = get_dmi_info()
    
    if HAS_PSUTIL:
        # CPU info
        try:
            cpu_freq = psutil.cpu_freq()
            specs["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency_mhz": cpu_freq.max if cpu_freq else None,
                "min_frequency_mhz": cpu_freq.min if cpu_freq else None,
            }
        except Exception as e:
            specs["cpu"] = {"error": str(e)}
        
        # Memory info
        try:
            mem = psutil.virtual_memory()
            specs["memory"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
            }
        except Exception as e:
            specs["memory"] = {"error": str(e)}

    else:
        specs["cpu"] = {"note": "psutil not installed"}
        specs["memory"] = {"note": "psutil not installed"}
    
    return specs


def get_system_state() -> Dict[str, Any]:
    """Collect current system state (CPU usage, memory usage, etc.) to detect interference."""
    state = {
        "timestamp": datetime.now().isoformat(),
    }
    
    if HAS_PSUTIL:
        # CPU usage (sample over 1 second for accuracy)
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            state["cpu"] = {
                "usage_percent": cpu_percent,
                "usage_per_core_percent": cpu_percent_per_core,
            }
        except Exception as e:
            state["cpu"] = {"error": str(e)}
        
        # Memory usage
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            state["memory"] = {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "usage_percent": mem.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "swap_percent": swap.percent,
            }
        except Exception as e:
            state["memory"] = {"error": str(e)}
    
        
        # Network I/O counters (baseline)
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                state["network_io"] = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                }
        except Exception as e:
            state["network_io"] = {"error": str(e)}
        
        # Top processes by memory/CPU (potential interference sources)
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['cpu_percent'] > 1.0 or pinfo['memory_percent'] > 1.0:
                        processes.append({
                            "pid": pinfo['pid'],
                            "name": pinfo['name'],
                            "cpu_percent": round(pinfo['cpu_percent'], 1),
                            "memory_percent": round(pinfo['memory_percent'], 1),
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            # Sort by CPU usage and take top 10
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            state["top_processes"] = processes[:10]
        except Exception as e:
            state["top_processes"] = {"error": str(e)}
        
        # Load average (Unix only)
        try:
            load_avg = psutil.getloadavg()
            state["load_average"] = {
                "1min": round(load_avg[0], 2),
                "5min": round(load_avg[1], 2),
                "15min": round(load_avg[2], 2),
            }
        except (AttributeError, OSError):
            # Windows doesn't have getloadavg
            state["load_average"] = {"note": "not available on this platform"}
    else:
        state["note"] = "psutil not installed - limited system state info"
    
    return state


def save_hardware_specs(output_dir: str) -> str:
    """Save hardware specifications to a file in the output directory."""
    import json
    
    system_info_dir = os.path.join(output_dir, "system_info")
    os.makedirs(system_info_dir, exist_ok=True)
    
    specs = get_hardware_specs()
    filepath = os.path.join(system_info_dir, "hardware_specs.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)
    
    # Also save a human-readable text version
    txt_filepath = os.path.join(system_info_dir, "hardware_specs.txt")
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("HARDWARE SPECIFICATIONS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Recorded at: {specs['timestamp']}\n\n")
        
        # System/DMI Info
        dmi = specs.get("dmi_info", {})
        if dmi and "note" not in dmi and "error" not in dmi:
            f.write("SYSTEM INFO (DMI)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Vendor:      {dmi.get('sys_vendor', 'N/A')}\n")
            f.write(f"  Product:     {dmi.get('product_name', 'N/A')}\n")
            f.write(f"  Version:     {dmi.get('product_version', 'N/A')}\n")
            f.write(f"  Board:       {dmi.get('board_vendor', 'N/A')} {dmi.get('board_name', 'N/A')}\n")
            f.write(f"  BIOS:        {dmi.get('bios_vendor', 'N/A')} {dmi.get('bios_version', 'N/A')} ({dmi.get('bios_date', 'N/A')})\n")
            f.write("\n")
        
        f.write("PLATFORM\n")
        f.write("-" * 40 + "\n")
        for k, v in specs.get("platform", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # CPU Info from /proc/cpuinfo
        proc_cpu = specs.get("proc_cpuinfo", {})
        if proc_cpu and "note" not in proc_cpu and "error" not in proc_cpu:
            f.write("CPU (from /proc/cpuinfo)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Model:       {proc_cpu.get('model_name', 'N/A')}\n")
            f.write(f"  Vendor:      {proc_cpu.get('vendor_id', 'N/A')}\n")
            f.write(f"  CPU Family:  {proc_cpu.get('cpu_family', 'N/A')}\n")
            f.write(f"  Stepping:    {proc_cpu.get('stepping', 'N/A')}\n")
            f.write(f"  Microcode:   {proc_cpu.get('microcode', 'N/A')}\n")
            f.write(f"  MHz:         {proc_cpu.get('cpu_mhz', 'N/A')}\n")
            f.write(f"  Cache:       {proc_cpu.get('cache_size', 'N/A')}\n")
            f.write(f"  Processors:  {proc_cpu.get('processor_count', 'N/A')}\n")
            f.write("\n")
        
        f.write("CPU (from psutil)\n")
        f.write("-" * 40 + "\n")
        for k, v in specs.get("cpu", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # Memory Info from /proc/meminfo
        proc_mem = specs.get("proc_meminfo", {})
        if proc_mem and "note" not in proc_mem and "error" not in proc_mem:
            f.write("MEMORY (from /proc/meminfo)\n")
            f.write("-" * 40 + "\n")
            for k, v in proc_mem.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        
        f.write("MEMORY (from psutil)\n")
        f.write("-" * 40 + "\n")
        for k, v in specs.get("memory", {}).items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
    return filepath


def save_pre_run_state(output_dir: str, schedule_name: str) -> str:
    """Save system state before a schedule run to detect interference."""
    import json
    
    system_info_dir = os.path.join(output_dir, "system_info")
    os.makedirs(system_info_dir, exist_ok=True)
    
    state = get_system_state()
    filepath = os.path.join(system_info_dir, f"{schedule_name}_pre_run_state.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    
    # Also save a human-readable text version
    txt_filepath = os.path.join(system_info_dir, f"{schedule_name}_pre_run_state.txt")
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"PRE-RUN SYSTEM STATE: {schedule_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Recorded at: {state['timestamp']}\n\n")
        
        f.write("CPU USAGE\n")
        f.write("-" * 40 + "\n")
        cpu_info = state.get("cpu", {})
        if "usage_percent" in cpu_info:
            f.write(f"  Overall: {cpu_info['usage_percent']}%\n")
        if "usage_per_core_percent" in cpu_info:
            f.write(f"  Per-core: {cpu_info['usage_per_core_percent']}\n")
        f.write("\n")
        
        f.write("MEMORY USAGE\n")
        f.write("-" * 40 + "\n")
        mem_info = state.get("memory", {})
        for k, v in mem_info.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        if "load_average" in state:
            f.write("LOAD AVERAGE\n")
            f.write("-" * 40 + "\n")
            load_info = state.get("load_average", {})
            for k, v in load_info.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
        
        if "top_processes" in state and isinstance(state["top_processes"], list):
            f.write("TOP PROCESSES (potential interference)\n")
            f.write("-" * 40 + "\n")
            for proc in state["top_processes"]:
                f.write(f"  PID {proc['pid']}: {proc['name']} "
                       f"(CPU: {proc['cpu_percent']}%, MEM: {proc['memory_percent']}%)\n")
            f.write("\n")
    
    return filepath


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


def query_system_events(url: str) -> List[List[str]]:
    """
    Query system.events from ClickHouse and return parsed CSV rows.
    Each row is [event_name, value, description].
    """
    # Extract host:port from URL for urllib
    # URL is like "http://localhost:8123/" 
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
        import json
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


def extract_workload_from_sql(sql: bytes) -> Optional[str]:
    """Extract workload setting from SQL if present."""
    # Match workload='...' or workload="..."
    match = re.search(rb"workload\s*=\s*['\"]([^'\"]+)['\"]", sql, re.IGNORECASE)
    if match:
        return match.group(1).decode('utf-8')
    return None


def load_query(queries_dir: str, qid: str) -> tuple:
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
                      traces_lock, req_start_ns, tracer_stop_event),
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
) -> None:
    """
    Consumer thread: dispatches queries and handles failures.
    
    Dequeues from priority queue, dispatches queries using a thread pool,
    and handles failures by pausing and reinserting failed queries.
    When queue is exhausted and producer is done, waits for in-flight queries
    then signals termination to the producer.
    
    If trace_processes is True, profile events are collected for each query
    and stored in query_profile_traces dict (keyed by query_id).
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


def generate_query_id(qid: str, at_ms: int) -> str:
    """Generate a unique query ID for ClickHouse."""
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{qid}_{at_ms}_{unique_suffix}"


class ConnectionLostError(Exception):
    """Raised when TCP connection to ClickHouse is lost."""
    pass


async def _cleanup_collector(stop_event: asyncio.Event, collector_task: Optional[asyncio.Task]) -> None:
    """Stop and await the collector task, ignoring any errors."""
    stop_event.set()
    if collector_task:
        try:
            await collector_task
        except Exception:
            pass

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
) -> None:
    """Run a single schedule and save results using producer/consumer threads."""
    schedule_name = os.path.splitext(os.path.basename(schedule_path))[0]
    print(f"\n{'='*60}")
    print(f"Running schedule: {schedule_path}")
    print(f"Schedule trace period: {schedule_trace_period_ms}ms, Query trace period: {query_trace_period_ms}ms")
    print(f"Pause on failure: {pause_seconds}s")
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
              query_trace_period_ms, query_profile_traces),
        name="consumer"
    )
    
    print("Starting producer and consumer threads...")
    producer.start()
    consumer.start()
    
    # Wait for threads to complete (run in executor to not block asyncio)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, producer.join)
    await loop.run_in_executor(None, consumer.join)
    
    print("Producer and consumer threads completed.")
    
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
        )

    print(f"\n{'='*60}")
    print(f"All schedules completed. Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())