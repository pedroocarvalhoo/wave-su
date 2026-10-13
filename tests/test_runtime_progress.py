from __future__ import annotations

from types import SimpleNamespace

import pytest

from vessim.wave.resource_manager import RunningAllocation
from vessim.wave.scheduler.runtime import advance_job_work, estimate_remaining_time, update_running_jobs


def test_advance_job_work_consumes_overhead_before_compute(make_job, scheduler_stub):
    job = make_job()
    job.assigned_frequency = 2.0
    job.remaining_runtime_overhead = 12.0
    job.remaining_compute_cycles = 240e9

    advance_job_work(
        scheduler=scheduler_stub,
        job=job,
        step_size_seconds=60,
        contention_factor=1.0,
    )

    assert job.remaining_runtime_overhead == 0.0
    assert job.remaining_compute_cycles == 144e9
    assert job.remaining_time == 72.0


def test_estimate_remaining_time_uses_current_assigned_frequency(make_job, scheduler_stub):
    job = make_job()
    job.remaining_runtime_overhead = 2.0
    job.remaining_compute_cycles = 4e9

    job.assigned_frequency = 2.0
    assert estimate_remaining_time(scheduler_stub, job) == 4.0

    job.assigned_frequency = 1.0
    assert estimate_remaining_time(scheduler_stub, job) == 6.0


def test_update_running_jobs_applies_contention_from_cluster_utilization(make_job, scheduler_stub):
    job = make_job()
    job.status = "running"
    job.assigned_frequency = 1.0
    job.remaining_compute_cycles = 100e9
    job.remaining_runtime_overhead = 0.0

    rm = SimpleNamespace(
        get_cluster_utilization=lambda: 1.0,
        get_running_allocations=lambda: [
            RunningAllocation(job=job, node_names=["n0"], frequency=1.0)
        ],
    )
    scheduler_stub.rm = rm

    update_running_jobs(scheduler_stub, step_size_seconds=10)

    assert job.remaining_compute_cycles == pytest.approx(91e9)
    assert job.remaining_time == pytest.approx(91.0)
