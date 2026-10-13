from __future__ import annotations

from datetime import timedelta

from vessim.wave.scheduler.runtime import ActualExecutionWork


def test_job_initializes_explicit_runtime_state(make_job):
    job = make_job(
        cpi=2.0,
        num_instructions=100.0,
        predicted_t_exec=6.0,
        actual_t_exec=7.0,
    )

    assert job.predicted_t_exec == 6.0
    assert job.actual_t_exec == 7.0
    assert job.actual_compute_cycles == 200.0
    assert job.remaining_compute_cycles == 200.0
    assert job.remaining_time == 7.0


def test_start_execution_centralizes_runtime_and_metric_initialization(make_job, sim_time):
    job = make_job(num_nodes=2)
    actual_work = ActualExecutionWork(
        compute_cycles=4e9,
        runtime_overhead=2.0,
        total_time=6.0,
        runtime_error_factor=1.0,
    )

    job.start_execution(
        current_time=sim_time,
        assigned_nodes=["n0", "n1"],
        frequency=2.0,
        predicted_t_exec=5.0,
        actual_work=actual_work,
        prediction_error_factor=0.8,
    )

    assert job.status == "running"
    assert job.start_time == sim_time
    assert job.assigned_subcluster == "n0,n1"
    assert job.assigned_frequency == 2.0
    assert job.frequency_history == [(sim_time, 2.0)]
    assert job.predicted_t_exec == 5.0
    assert job.actual_t_exec == 6.0
    assert job.prediction_error_factor == 0.8
    assert job.actual_compute_cycles == 4e9
    assert job.actual_runtime_overhead == 2.0
    assert job.remaining_compute_cycles == 4e9
    assert job.remaining_runtime_overhead == 2.0
    assert job.remaining_time == 6.0


def test_lifecycle_helpers_only_update_lifecycle_state(make_job, sim_time):
    job = make_job()

    completion_time = sim_time + timedelta(seconds=10)
    job.mark_completed(completion_time)
    assert job.status == "completed"
    assert job.completion_time == completion_time

    job.mark_failed()
    assert job.status == "failed"
