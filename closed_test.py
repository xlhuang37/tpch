#!/usr/bin/env python3
#python3 closed_test.py
"""
Closed system testing script for ClickHouse using clickhouse-benchmark.

This script is symmetric to open_test.py but uses clickhouse-benchmark for 
closed-loop testing (fixed concurrency and iteration count). It:
1. Reads SQL files from a queries directory
2. Aggregates them into a query list
3. Passes the query list to clickhouse-benchmark
4. Saves results to ./output/closed/<timestamp>/
"""
from __future__ import annotations
import argparse
import glob
import os
import platform
import re
import subprocess
import sys
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

# Optional: psutil for detailed system info
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with 'pip install psutil' for detailed system info.")


@dataclass
class BenchmarkResult:
    """Parsed results from clickhouse-benchmark output."""
    queries_executed: int = 0
    qps: float = 0.0
    rps: float = 0.0  # Rows per second
    mibps: float = 0.0  # MiB per second
    result_rps: float = 0.0  # Result rows per second
    result_mibps: float = 0.0  # Result MiB per second
    
    # Latency percentiles (in seconds)
    latency_min: float = 0.0
    latency_max: float = 0.0
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p99: float = 0.0
    latency_p999: float = 0.0
    latency_p9999: float = 0.0


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


def save_pre_run_state(output_dir: str, run_name: str) -> str:
    """Save system state before a benchmark run to detect interference."""
    system_info_dir = os.path.join(output_dir, "system_info")
    os.makedirs(system_info_dir, exist_ok=True)
    
    state = get_system_state()
    filepath = os.path.join(system_info_dir, f"{run_name}_pre_run_state.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    
    # Also save a human-readable text version
    txt_filepath = os.path.join(system_info_dir, f"{run_name}_pre_run_state.txt")
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"PRE-RUN SYSTEM STATE: {run_name}\n")
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


def read_sql_files(queries_dir: str, pattern: str = "*.sql") -> List[tuple]:
    """
    Read all SQL files from a directory.
    Returns list of (filename, sql_content) tuples, sorted by filename.
    """
    sql_pattern = os.path.join(queries_dir, pattern)
    sql_files = sorted(glob.glob(sql_pattern))
    
    if not sql_files:
        raise FileNotFoundError(f"No SQL files matching '{pattern}' found in {queries_dir}")
    
    queries = []
    for sql_path in sql_files:
        filename = os.path.basename(sql_path)
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_content = f.read().strip()
        queries.append((filename, sql_content))
    
    return queries


def create_query_list(queries: List[tuple], oneline: bool = True) -> str:
    """
    Create aggregated query list from SQL queries.
    Each query is separated by a newline.
    
    Args:
        queries: List of (filename, sql_content) tuples
        oneline: If True, format each query on a single line
    
    Returns:
        Combined query string suitable for clickhouse-benchmark
    """
    query_lines = []
    for filename, sql_content in queries:
        if oneline:
            # Convert multi-line SQL to single line
            # Remove comments, collapse whitespace
            sql_oneline = re.sub(r'--.*$', '', sql_content, flags=re.MULTILINE)  # Remove -- comments
            sql_oneline = re.sub(r'/\*.*?\*/', '', sql_oneline, flags=re.DOTALL)  # Remove /* */ comments
            sql_oneline = ' '.join(sql_oneline.split())  # Collapse whitespace
            query_lines.append(sql_oneline)
        else:
            query_lines.append(sql_content)
    
    return '\n'.join(query_lines)


def parse_benchmark_output(output: str) -> BenchmarkResult:
    """
    Parse clickhouse-benchmark output and extract metrics.
    
    Example output format:
    Queries executed: 200
    
    localhost:9000, queries 200, QPS: 10.234, RPS: 1234567.890, MiB/s: 123.456, result RPS: 9876.543, result MiB/s: 0.987.
    
    0.000%      0.012 sec.
    10.000%     0.015 sec.
    ...
    """
    result = BenchmarkResult()
    
    # Parse "Queries executed: N"
    match = re.search(r'Queries executed:\s*(\d+)', output)
    if match:
        result.queries_executed = int(match.group(1))
    
    # Parse main metrics line
    # Example: localhost:9000, queries 200, QPS: 10.234, RPS: 1234567.890, MiB/s: 123.456, result RPS: 9876.543, result MiB/s: 0.987.
    metrics_match = re.search(
        r'QPS:\s*([\d.]+).*?RPS:\s*([\d.]+).*?MiB/s:\s*([\d.]+).*?result RPS:\s*([\d.]+).*?result MiB/s:\s*([\d.]+)',
        output
    )
    if metrics_match:
        result.qps = float(metrics_match.group(1))
        result.rps = float(metrics_match.group(2))
        result.mibps = float(metrics_match.group(3))
        result.result_rps = float(metrics_match.group(4))
        result.result_mibps = float(metrics_match.group(5))
    
    # Parse percentile latencies
    percentile_patterns = [
        (r'0\.000%\s+([\d.]+)\s+sec', 'latency_min'),
        (r'50\.000%\s+([\d.]+)\s+sec', 'latency_p50'),
        (r'90\.000%\s+([\d.]+)\s+sec', 'latency_p90'),
        (r'99\.000%\s+([\d.]+)\s+sec', 'latency_p99'),
        (r'99\.900%\s+([\d.]+)\s+sec', 'latency_p999'),
        (r'99\.990%\s+([\d.]+)\s+sec', 'latency_p9999'),
        (r'100\.000%\s+([\d.]+)\s+sec', 'latency_max'),
    ]
    
    for pattern, attr in percentile_patterns:
        match = re.search(pattern, output)
        if match:
            setattr(result, attr, float(match.group(1)))
    
    # Calculate mean if we have min and max (rough estimate)
    # clickhouse-benchmark doesn't directly output mean, but we can try to extract it
    # from other percentiles or leave it as 0
    
    return result


def save_benchmark_results(output_dir: str, run_name: str, 
                           result: BenchmarkResult, raw_output: str,
                           queries: List[tuple], config: Dict[str, Any]) -> str:
    """Save benchmark results to files."""
    
    # Save raw output
    raw_path = os.path.join(output_dir, f"{run_name}_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_output)
    
    # Save parsed results as JSON
    json_path = os.path.join(output_dir, f"{run_name}_results.json")
    results_dict = {
        "config": config,
        "queries_executed": result.queries_executed,
        "qps": result.qps,
        "rps": result.rps,
        "mibps": result.mibps,
        "result_rps": result.result_rps,
        "result_mibps": result.result_mibps,
        "latency": {
            "min_sec": result.latency_min,
            "max_sec": result.latency_max,
            "p50_sec": result.latency_p50,
            "p90_sec": result.latency_p90,
            "p99_sec": result.latency_p99,
            "p999_sec": result.latency_p999,
            "p9999_sec": result.latency_p9999,
        },
        "query_files": [q[0] for q in queries],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    
    # Save human-readable statistics
    stats_path = os.path.join(output_dir, f"{run_name}_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Closed System Benchmark Results: {run_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Concurrency (-c):     {config.get('concurrency', 'N/A')}\n")
        f.write(f"  Iterations (-i):      {config.get('iterations', 'N/A')}\n")
        f.write(f"  Host:                 {config.get('host', 'N/A')}\n")
        f.write(f"  Port:                 {config.get('port', 'N/A')}\n")
        f.write(f"  Queries directory:    {config.get('queries_dir', 'N/A')}\n")
        f.write(f"  Query count:          {len(queries)}\n")
        f.write("\n")
        
        f.write("THROUGHPUT\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Queries executed:     {result.queries_executed}\n")
        f.write(f"  QPS:                  {result.qps:.4f}\n")
        f.write(f"  RPS (rows/sec):       {result.rps:.4f}\n")
        f.write(f"  MiB/s:                {result.mibps:.4f}\n")
        f.write(f"  Result RPS:           {result.result_rps:.4f}\n")
        f.write(f"  Result MiB/s:         {result.result_mibps:.4f}\n")
        f.write("\n")
        
        f.write("LATENCY PERCENTILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Min (0%):             {result.latency_min * 1000:.4f} ms\n")
        f.write(f"  P50:                  {result.latency_p50 * 1000:.4f} ms\n")
        f.write(f"  P90:                  {result.latency_p90 * 1000:.4f} ms\n")
        f.write(f"  P99:                  {result.latency_p99 * 1000:.4f} ms\n")
        f.write(f"  P99.9:                {result.latency_p999 * 1000:.4f} ms\n")
        f.write(f"  P99.99:               {result.latency_p9999 * 1000:.4f} ms\n")
        f.write(f"  Max (100%):           {result.latency_max * 1000:.4f} ms\n")
        f.write("\n")
        
        f.write("QUERY FILES\n")
        f.write("-" * 40 + "\n")
        for filename, _ in queries:
            f.write(f"  - {filename}\n")
    
    # Save query list that was sent to benchmark
    query_list_path = os.path.join(output_dir, f"{run_name}_query_list.sql")
    query_list = create_query_list(queries, oneline=True)
    with open(query_list_path, "w", encoding="utf-8") as f:
        f.write(query_list)
    
    return stats_path


def run_benchmark(
    benchmark_path: str,
    queries: List[tuple],
    host: str,
    port: int,
    concurrency: int,
    iterations: int,
    extra_args: List[str],
    output_dir: str,
    run_name: str,
) -> BenchmarkResult:
    """
    Run clickhouse-benchmark with the given queries.
    
    Args:
        benchmark_path: Path to clickhouse-benchmark executable
        queries: List of (filename, sql_content) tuples
        host: ClickHouse host
        port: ClickHouse native port
        concurrency: Number of concurrent connections (-c)
        iterations: Number of iterations (-i)
        extra_args: Additional arguments to pass to clickhouse-benchmark
        output_dir: Directory to save results
        run_name: Name for this benchmark run
    
    Returns:
        BenchmarkResult with parsed metrics
    """
    # Create query list
    query_list = create_query_list(queries, oneline=True)
    
    # Build command
    cmd = [
        benchmark_path,
        "-h", host,
        "--port", str(port),
        "-c", str(concurrency),
        "-i", str(iterations),
    ]
    cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Queries: {len(queries)} SQL files")
    print(f"Concurrency: {concurrency}, Iterations: {iterations}")
    print(f"{'='*60}\n")
    
    # Run benchmark
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        stdout, _ = process.communicate(input=query_list)
        
        if process.returncode != 0:
            print(f"Warning: clickhouse-benchmark exited with code {process.returncode}")
        
        print("Benchmark output:")
        print("-" * 40)
        print(stdout)
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"Error: clickhouse-benchmark not found at {benchmark_path}")
        print("Please specify the correct path using --benchmark-path")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        sys.exit(1)
    
    # Parse results
    result = parse_benchmark_output(stdout)
    
    # Save results
    config = {
        "benchmark_path": benchmark_path,
        "host": host,
        "port": port,
        "concurrency": concurrency,
        "iterations": iterations,
        "extra_args": extra_args,
        "queries_dir": "",  # Will be set by caller
    }
    
    stats_path = save_benchmark_results(output_dir, run_name, result, stdout, queries, config)
    print(f"\nResults saved to: {stats_path}")
    
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Run closed system ClickHouse benchmark using clickhouse-benchmark."
    )
    ap.add_argument(
        "--queries-dir", 
        default="./queries/",
        help="Directory containing SQL files to benchmark (default: ./queries/)"
    )
    ap.add_argument(
        "--pattern",
        default="*.sql",
        help="Glob pattern for SQL files (default: *.sql)"
    )
    ap.add_argument(
        "--benchmark-path",
        default="../ClickHouse/build/programs/clickhouse-benchmark",
        help="Path to clickhouse-benchmark executable (default: ../ClickHouse/build/programs/clickhouse-benchmark)"
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
        "--extra-args",
        nargs="*",
        default=[],
        help="Additional arguments to pass to clickhouse-benchmark"
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Name for this benchmark run (default: derived from queries directory)"
    )
    
    args = ap.parse_args()
    
    # Determine run name
    if args.run_name:
        run_name = args.run_name
    else:
        # Derive from queries directory name
        queries_dirname = os.path.basename(os.path.normpath(args.queries_dir))
        run_name = f"bench_{queries_dirname}"
    
    print(f"Closed System Benchmark")
    print(f"=" * 60)
    print(f"Queries directory: {args.queries_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Benchmark path: {args.benchmark_path}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Iterations: {args.iterations}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output/closed", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save hardware specifications
    print("\nRecording hardware specifications...")
    hw_specs_path = save_hardware_specs(output_dir)
    print(f"  Hardware specs saved to: {hw_specs_path}")
    
    # Record pre-run system state
    print("Recording pre-run system state...")
    pre_run_path = save_pre_run_state(output_dir, run_name)
    print(f"  Pre-run state saved to: {pre_run_path}")
    
    # Read SQL files
    try:
        queries = read_sql_files(args.queries_dir, args.pattern)
        print(f"\nFound {len(queries)} SQL files:")
        for filename, _ in queries:
            print(f"  - {filename}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Run benchmark
    result = run_benchmark(
        benchmark_path=args.benchmark_path,
        queries=queries,
        host=args.host,
        port=args.port,
        concurrency=args.concurrency,
        iterations=args.iterations,
        extra_args=args.extra_args,
        output_dir=output_dir,
        run_name=run_name,
    )
    
    # Update config with queries_dir for saved results
    # (already done in run_benchmark via config dict)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Benchmark completed: {run_name}")
    print(f"{'='*60}")
    print(f"  Queries executed: {result.queries_executed}")
    print(f"  QPS: {result.qps:.4f}")
    print(f"  Latency P50: {result.latency_p50 * 1000:.2f} ms")
    print(f"  Latency P99: {result.latency_p99 * 1000:.2f} ms")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
