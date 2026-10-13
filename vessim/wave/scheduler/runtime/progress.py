"""
Per-step runtime progress updates for running WAVE jobs.

   Progress helpers consume scheduler/resource-manager state and decrement the
   actual remaining work of running jobs. They should not select new jobs or
   change node allocation state directly.
"""

from __future__ import annotations

from vessim.wave.cluster.job_queue import Job


def update_running_jobs(scheduler, step_size_seconds: int) -> None:
    utilization = scheduler.rm.get_cluster_utilization()
    contention_factor = 1.0

    if utilization > 0.8:
        contention_factor = 1.0 - ((utilization - 0.8) * 0.5)

    for allocation in scheduler.rm.get_running_allocations():
        if allocation.job.status == "running":
            advance_job_work(
                scheduler,
                allocation.job,
                step_size_seconds,
                contention_factor,
            )


def advance_job_work(
    scheduler,
    job: Job,
    step_size_seconds: int,
    contention_factor: float,
) -> None:
    progress_seconds = step_size_seconds * contention_factor

    if job.remaining_runtime_overhead > 0.0:
        overhead_progress = min(job.remaining_runtime_overhead, progress_seconds)
        job.remaining_runtime_overhead -= overhead_progress
        progress_seconds -= overhead_progress
        if job.remaining_runtime_overhead < 1e-9:
            job.remaining_runtime_overhead = 0.0

    if progress_seconds > 0.0 and job.remaining_compute_cycles > 0.0:
        frequency = job.assigned_frequency or scheduler.dvfs_policy.base_frequency_ghz
        cycles_progress = frequency * 1e9 * progress_seconds
        job.remaining_compute_cycles = max(
            0.0,
            job.remaining_compute_cycles - cycles_progress,
        )
        if job.remaining_compute_cycles < 1.0:
            job.remaining_compute_cycles = 0.0

    job.remaining_time = estimate_remaining_time(scheduler, job)


def estimate_remaining_time(scheduler, job: Job) -> float:
    frequency = job.assigned_frequency or scheduler.dvfs_policy.base_frequency_ghz
    if frequency <= 0.0:
        return float("inf")
    compute_remaining = job.remaining_compute_cycles / (frequency * 1e9)
    remaining = max(0.0, job.remaining_runtime_overhead + compute_remaining)
    return 0.0 if remaining < 1e-9 else remaining
