"""
Data structures and exceptions for the test_open package.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from concurrent.futures import Future


@dataclass
class Event:
    """A scheduled query event from the schedule CSV."""
    at_ms: int
    qid: str


@dataclass
class LatencyRecord:
    """Timing data for a completed query."""
    arrival_ms: float     # Actual arrival time (relative to t0)
    start_ms: float       # Actual query start time (when server accepted)
    end_ms: float         # Query completion time
    latency_ms: float     # end_ms - arrival_ms (total latency including wait)
    wait_ms: float        # start_ms - arrival_ms (time waiting before execution)
    qid: str


@dataclass
class DroppedRecord:
    """Record of a dropped/failed query."""
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


class ConnectionLostError(Exception):
    """Raised when TCP connection to ClickHouse is lost."""
    pass
