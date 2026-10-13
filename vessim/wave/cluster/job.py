"""
Job model used by WAVE cluster scheduling.

   `Job` is the shared workload record passed between the queue, scheduler,
   resource manager, runtime model, and metrics exporter. Put lifecycle fields
   that must travel with the job here; keep scheduling algorithms and node-state
   transitions outside this module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Job:

    job_id: int 
    cpi: float  # Cycles per instruction.
    num_instructions: float
    arrival_time: datetime
    num_nodes: int
    requested_walltime_s: float
    memory_gb: float = 0.0  # Used to estimate memory overhead.
    priority: int = 2
    submission_time: Optional[datetime] = None
    status: str = "pending"  # One of: pending, running, completed, failed.
    failure_reason: Optional[str] = None
    start_time: Optional[datetime] = None  # Time when execution starts.
    completion_time: Optional[datetime] = None  # Time when execution finishes.

    # Runtime prediction and progress state.
    predicted_t_exec: float = 0.0  # Scheduler-facing runtime estimate.
    actual_t_exec: float = 0.0  # Runtime used by the execution model.
    prediction_error_factor: float = 1.0  # Uncertainty applied to predictions.
    runtime_error_factor: float = 1.0  # Uncertainty applied to actual runtime.
    actual_compute_cycles: float = 0.0  # Immutable compute work estimate.
    actual_runtime_overhead: float = 0.0  # Immutable non-compute overhead.
    remaining_compute_cycles: float = 0.0  # Compute work left after each step.
    remaining_runtime_overhead: float = 0.0  # Runtime overhead left after each step.
    remaining_time: float = 0.0  # Time left according to the runtime model.
    assigned_subcluster: Optional[str] = None
    assigned_frequency: Optional[float] = None
    frequency_history: list[tuple[datetime, float]] = field(default_factory=list)

    # Logical queue metadata.
    queue_name: Optional[str] = None
    queue_classification_runtime_s: Optional[float] = None
    queue_enter_time: Optional[datetime] = None
    queue_max_wait_s: Optional[float] = None
    queue_max_nodes: Optional[int] = None
    queue_priority_weight: int = 0
    queue_deadline_missed: bool = False

    def __post_init__(self):
        if self.submission_time is None:
            self.submission_time = self.arrival_time
        if self.actual_compute_cycles <= 0.0 and self.actual_t_exec > 0.0:
            self.actual_compute_cycles = self.cpi * self.num_instructions
        if self.remaining_compute_cycles <= 0.0:
            self.remaining_compute_cycles = self.actual_compute_cycles
        if self.remaining_runtime_overhead <= 0.0:
            self.remaining_runtime_overhead = self.actual_runtime_overhead
        self.remaining_time = self.actual_t_exec

    def start_execution(
        self,
        current_time: datetime,
        assigned_nodes: list[str],
        frequency: float,
        predicted_t_exec: float,
        actual_work,
        prediction_error_factor: float,
    ) -> None:
        self.predicted_t_exec = predicted_t_exec
        self.actual_t_exec = actual_work.total_time
        self.prediction_error_factor = prediction_error_factor
        self.runtime_error_factor = actual_work.runtime_error_factor
        self.actual_compute_cycles = actual_work.compute_cycles
        self.actual_runtime_overhead = actual_work.runtime_overhead
        self.remaining_compute_cycles = actual_work.compute_cycles
        self.remaining_runtime_overhead = actual_work.runtime_overhead
        self.remaining_time = actual_work.total_time
        self.assigned_frequency = frequency
        self.frequency_history = [(current_time, frequency)]
        self.status = "running"
        self.start_time = current_time
        self.assigned_subcluster = (
            assigned_nodes[0]
            if len(assigned_nodes) == 1
            else ",".join(assigned_nodes)
        )

    def mark_completed(self, completion_time: datetime) -> None:
        self.status = "completed"
        self.completion_time = completion_time

    def mark_failed(self, reason: Optional[str] = None) -> None:
        self.status = "failed"
        self.failure_reason = reason


__all__ = ["Job"]
