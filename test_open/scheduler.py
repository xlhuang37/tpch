"""
Scheduler thread for dynamic workload settings based on running query counts.

This module provides:
- Hardcoded speedup curves for different workload types
- Thread-safe workload count tracking
- Greedy allocation algorithm for computing max_concurrent_threads
- Scheduler thread that periodically updates ClickHouse workload settings
"""

import threading
import time
import urllib.request
from typing import Dict, List, Optional

# =============================================================================
# Hardcoded Speedup Curves (from ISchedulerNode.h)
# =============================================================================

# Speedup vs. core count for parallel workloads
# - Almost linear until 32 cores
# - After 32 cores, still increases but with smaller per-core gain
# Index i corresponds to i cores (index 0 = 0 cores)
PARALLEL_SPEEDUP = [
    0.0000,   # 0 cores
    1.0000,   # 1 core
    1.9836,   # 2 cores
    2.9424,   # 3 cores
    3.7905,   # 4 cores
    4.6265,   # 5 cores
    5.4625,   # 6 cores
    6.2985,   # 7 cores
    7.1345,   # 8 cores
    7.9705,   # 9 cores
    8.8065,   # 10 cores
    9.6425,   # 11 cores
    10.4785,  # 12 cores
    11.3138,  # 13 cores
    12.1491,  # 14 cores
    12.9844,  # 15 cores
    13.7840,  # 16 cores
    14.5836,  # 17 cores
    15.3832,  # 18 cores
    16.1828,  # 19 cores
    16.9294,  # 20 cores
    17.6760,  # 21 cores
    18.4226,  # 22 cores
    19.1692,  # 23 cores
    19.9158,  # 24 cores
    20.6624,  # 25 cores
    21.4090,  # 26 cores
    22.1556,  # 27 cores
    22.8429,  # 28 cores
    23.5302,  # 29 cores
    24.2175,  # 30 cores
    24.7708,  # 31 cores
    25.3241,  # 32 cores
    25.8774,  # 33 cores
    26.4307,  # 34 cores
    26.9059,  # 35 cores
    27.3811,  # 36 cores
    27.8563,  # 37 cores
    28.3315,  # 38 cores
    28.8067,  # 39 cores
    29.2320,  # 40 cores
    29.6573,  # 41 cores
    30.0826,  # 42 cores
    30.4246,  # 43 cores
    30.7666,  # 44 cores
    30.9294,  # 45 cores
    31.0922,  # 46 cores
]
# Fill remaining up to 120 with last value
PARALLEL_SPEEDUP.extend([31.0922] * (120 - len(PARALLEL_SPEEDUP)))

# Speedup vs. core count for almost-serial (non-parallel) workloads
# - Almost linear up to 2 cores
# - Completely flat after 2 cores
NONPARALLEL_SPEEDUP = [
    0.0000,   # 0 cores
    1.0000,   # 1 core
    1.9136,   # 2 cores
]
# Fill remaining up to 120 with last value
NONPARALLEL_SPEEDUP.extend([1.9136] * (120 - len(NONPARALLEL_SPEEDUP)))

# Dictionary mapping workload names to their speedup curves
# Workload name = speedup curve name
SPEEDUP_CURVES: Dict[str, List[float]] = {
    "SpeedUpOne": PARALLEL_SPEEDUP,
    "SpeedUpTwo": NONPARALLEL_SPEEDUP,
}

# Default total cores to distribute
DEFAULT_TOTAL_CORES = 60
DEFAULT_CORE_TO_THREAD_RATIO = 2

# =============================================================================
# Workload Tracker (Thread-Safe)
# =============================================================================

class WorkloadTracker:
    """
    Thread-safe tracker for running query counts per workload class.
    
    Used by the consumer thread to track active queries and by the scheduler
    thread to read current counts for allocation decisions.
    """
    
    def __init__(self):
        self._counts: Dict[str, int] = {}  # workload_name -> running count
        self._lock = threading.Lock()
    
    def increment(self, workload: str) -> None:
        """Increment the running count for a workload (called when query starts)."""
        if not workload:
            return
        with self._lock:
            self._counts[workload] = self._counts.get(workload, 0) + 1
    
    def decrement(self, workload: str) -> None:
        """Decrement the running count for a workload (called when query completes)."""
        if not workload:
            return
        with self._lock:
            current = self._counts.get(workload, 0)
            if current > 0:
                self._counts[workload] = current - 1
                # Clean up zero counts to avoid memory growth
                if self._counts[workload] == 0:
                    del self._counts[workload]
    
    def get_snapshot(self) -> Dict[str, int]:
        """
        Get a snapshot of current workload counts.
        
        Returns a copy to avoid race conditions while the scheduler processes it.
        """
        with self._lock:
            return dict(self._counts)
    
    def get_count(self, workload: str) -> int:
        """Get the current count for a specific workload."""
        with self._lock:
            return self._counts.get(workload, 0)


# =============================================================================
# Greedy Allocation Algorithm
# =============================================================================

def compute_thread_allocation(
    workload_counts: Dict[str, int],
    total_cores: int,
    speedup_curves: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, int]:
    """
    Compute optimal max_concurrent_threads for each workload using greedy allocation.
    
    This algorithm is similar to FairPolicy.h::recomputeWeights:
    1. Build candidate list of workloads with running queries > 0
    2. For each core, find the workload with highest marginal speedup gain
    3. Assign the core to that workload
    4. Return allocated cores per workload (minimum 1)
    
    Args:
        workload_counts: Dict mapping workload name to running query count
        total_cores: Total number of cores to distribute
        speedup_curves: Dict mapping workload name to speedup array (uses SPEEDUP_CURVES if None)
    
    Returns:
        Dict mapping workload name to allocated max_concurrent_threads
    """
    if speedup_curves is None:
        speedup_curves = SPEEDUP_CURVES
    
    # Build candidate list: workloads with running > 0
    candidates = []
    for workload, running in workload_counts.items():
        if running > 0:
            # Get speedup curve for this workload (default to parallel if not found)
            speedup = speedup_curves.get(workload, PARALLEL_SPEEDUP)
            candidates.append({
                'workload': workload,
                'running': running,
                'speedup': speedup,
                'allocated': 0,  # cores allocated so far
            })
    
    if not candidates:
        return {}
    
    # Greedily assign cores one by one
    for _ in range(total_cores):
        best_candidate = None
        best_marginal = 0.0
        
        for c in candidates:
            # k = cores per query = allocated / running
            # We use integer division similar to FairPolicy.h
            k = c['allocated'] // c['running']
            k = min(k, len(c['speedup']) - 2)  # Ensure we don't go out of bounds
            
            # Marginal speedup for adding one more core
            s0 = c['speedup'][k]
            s1 = c['speedup'][min(k + 1, len(c['speedup']) - 1)]
            marginal = s1 - s0
            
            if marginal > best_marginal:
                best_marginal = marginal
                best_candidate = c
        
        if best_candidate is None:
            break
        
        # Assign this core to the best workload
        best_candidate['allocated'] += 1
    
    # Build result with minimum of 1 thread per active workload
    result = {}
    for c in candidates:
        allocated = c['allocated']
        if allocated < 1:
            allocated = 1
        result[c['workload']] = allocated * DEFAULT_CORE_TO_THREAD_RATIO
    
    return result


# =============================================================================
# ClickHouse Workload Update
# =============================================================================

def update_workload_settings(url: str, workload: str, max_threads: int) -> bool:
    """
    Send CREATE OR REPLACE WORKLOAD command to ClickHouse.
    
    Args:
        url: ClickHouse HTTP endpoint URL
        workload: Workload name
        max_threads: max_concurrent_threads setting value
    
    Returns:
        True if successful, False otherwise
    """
    # SQL command to update workload settings
    sql = f'CREATE OR REPLACE WORKLOAD "{workload}" IN `all` SETTINGS weight = {max_threads}'
    
    try:
        req = urllib.request.Request(url, data=sql.encode('utf-8'), method='POST')
        with urllib.request.urlopen(req, timeout=5) as response:
            response.read()  # Consume response
        return True
    except Exception as e:
        print(f"[Scheduler] Failed to update workload '{workload}': {e}")
        return False


# =============================================================================
# Scheduler Thread
# =============================================================================

def scheduler_thread(
    url: str,
    workload_tracker: WorkloadTracker,
    interval_ms: int,
    total_cores: int,
    stop_event: threading.Event,
    speedup_curves: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    Scheduler thread that periodically computes and updates workload settings.
    
    This thread:
    1. Reads current workload counts from the tracker
    2. Computes optimal max_concurrent_threads using greedy allocation
    3. Sends CREATE OR REPLACE WORKLOAD commands to ClickHouse
    
    Args:
        url: ClickHouse HTTP endpoint URL
        workload_tracker: Shared WorkloadTracker instance
        interval_ms: Update interval in milliseconds
        total_cores: Total number of cores to distribute
        stop_event: Event to signal thread termination
        speedup_curves: Optional custom speedup curves (uses SPEEDUP_CURVES if None)
    """
    if speedup_curves is None:
        speedup_curves = SPEEDUP_CURVES
    
    interval_seconds = interval_ms / 1000.0
    last_allocation: Dict[str, int] = {}
    
    print(f"[Scheduler] Started with interval={interval_ms}ms, total_cores={total_cores}")
    
    while not stop_event.is_set():
        # Get current workload counts
        workload_counts = workload_tracker.get_snapshot()
        
        if workload_counts:
            # Compute optimal allocation
            allocation = compute_thread_allocation(
                workload_counts, 
                total_cores, 
                speedup_curves
            )
            
            # Update ClickHouse for each workload that changed
            for workload, max_threads in allocation.items():
                if last_allocation.get(workload) != max_threads:
                    print(f"[Scheduler] Updating '{workload}': weight={max_threads} "
                          f"(running={workload_counts.get(workload, 0)})")
                    if update_workload_settings(url, workload, max_threads):
                        print(f"[Scheduler] Updated '{workload}': weight={max_threads} "
                              f"(running={workload_counts.get(workload, 0)})")
            
            last_allocation = allocation
        
        # Wait for next interval or stop signal
        if stop_event.wait(timeout=interval_seconds):
            break
    
    print("[Scheduler] Stopped")
