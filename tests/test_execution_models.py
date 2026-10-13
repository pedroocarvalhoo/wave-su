from __future__ import annotations

import pytest

from vessim.wave.scheduler.runtime import (
    compute_actual_execution_work,
    predict_execution_time,
)


def test_actual_work_uses_dataset_fields_without_prediction_error(make_job):
    job = make_job(
        job_id=7,
        cpi=2.0,
        num_instructions=3e9,
        memory_gb=10.0,
    )

    actual = compute_actual_execution_work(
        job=job,
        frequency=2.0,
        num_assigned_nodes=2,
        cluster_mem_bw_gb_s=5.0,
    )

    assert actual.compute_cycles == 6e9
    assert actual.runtime_overhead == pytest.approx(2.2)
    assert actual.total_time == pytest.approx(5.2)
    assert actual.runtime_error_factor == 1.0


def test_prediction_error_changes_prediction_but_not_actual_work(make_job):
    job = make_job(job_id=7, cpi=1.0, num_instructions=2e9, memory_gb=0.0)

    predicted = predict_execution_time(
        job=job,
        frequency=2.0,
        num_assigned_nodes=1,
        prediction_error_rate=0.5,
        prediction_seed=123,
    )
    actual = compute_actual_execution_work(
        job=job,
        frequency=2.0,
        num_assigned_nodes=1,
    )

    assert predicted != pytest.approx(actual.total_time)
    assert actual.compute_cycles == 2e9
    assert actual.runtime_error_factor == 1.0
