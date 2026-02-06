"""
System information collection for hardware specs and pre-run state.
"""

import json
import os
import platform
from datetime import datetime
from typing import Dict, Any

# Optional: psutil for detailed system info
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with 'pip install psutil' for detailed system info.")


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


def save_pre_run_state(output_dir: str, schedule_name: str) -> str:
    """Save system state before a schedule run to detect interference."""
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
