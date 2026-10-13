"""
Runtime prediction and progress helpers for WAVE schedulers.

   Runtime modules define how predicted execution time, actual work, contention,
   and remaining progress are calculated. They should not decide which jobs run
   or which nodes are allocated.
"""

from vessim.wave.scheduler.runtime.execution_models import (
    ActualExecutionWork,
    compute_actual_execution_work,
    compute_network_overhead,
    compute_runtime_components,
    predict_execution_time,
    sample_prediction_error_factor,
)
from vessim.wave.scheduler.runtime.progress import (
    advance_job_work,
    estimate_remaining_time,
    update_running_jobs,
)

__all__ = [
    "ActualExecutionWork",
    "advance_job_work",
    "compute_actual_execution_work",
    "compute_network_overhead",
    "compute_runtime_components",
    "estimate_remaining_time",
    "predict_execution_time",
    "sample_prediction_error_factor",
    "update_running_jobs",
]
