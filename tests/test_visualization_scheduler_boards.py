from types import SimpleNamespace

import pandas as pd
import pytest

from vessim.visualization import (
    _job_average_power_w,
    _pack_real_jobs_to_power,
    _stitch_puzzle_green_board,
)


def test_stitch_puzzle_green_board_keeps_latest_plan_per_slot():
    slots = pd.DataFrame(
        {
            "plan_id": [0, 0, 1, 0],
            "cluster_type": ["arm", "arm", "arm", "__total__"],
            "start_time": [
                "2023-06-01 00:00:00",
                "2023-06-01 00:05:00",
                "2023-06-01 00:05:00",
                "2023-06-01 00:05:00",
            ],
            "end_time": [
                "2023-06-01 00:05:00",
                "2023-06-01 00:10:00",
                "2023-06-01 00:10:00",
                "2023-06-01 00:10:00",
            ],
            "green_available_w": [100.0, 110.0, 90.0, 200.0],
        }
    )

    stitched = _stitch_puzzle_green_board(slots)

    assert stitched["plan_id"].tolist() == [0, 1]
    assert stitched["green_available_w"].tolist() == [100.0, 90.0]
    assert set(stitched["cluster_type"]) == {"arm"}


def test_job_average_power_weights_dvfs_segments_and_nodes():
    start = pd.Timestamp("2023-06-01 00:00:00")
    job = SimpleNamespace(
        num_nodes=2,
        assigned_frequency=2.0,
        frequency_history=[
            (start, 1.0),
            (start + pd.Timedelta(hours=1), 2.0),
        ],
    )
    node_cfg = {
        "idle_power_w": 10.0,
        "frequencies": {"1.0": 100.0, "2.0": 200.0},
    }

    average_w = _job_average_power_w(
        job, node_cfg, start, start + pd.Timedelta(hours=2)
    )

    assert average_w == pytest.approx(320.0)


def test_real_job_packing_reuses_vertical_space_after_completion():
    start = pd.Timestamp("2023-06-01 00:00:00")

    def job(job_id, offset_h, duration_h):
        job_start = start + pd.Timedelta(hours=offset_h)
        return SimpleNamespace(
            job_id=job_id,
            num_nodes=1,
            start_time=job_start,
            completion_time=job_start + pd.Timedelta(hours=duration_h),
            assigned_frequency=1.0,
            frequency_history=[(job_start, 1.0)],
        )

    scheduler = SimpleNamespace(
        job_queue=SimpleNamespace(
            jobs=[job(1, 0, 2), job(2, 1, 2), job(3, 2, 1)]
        )
    )
    cfg = {"idle_power_w": 0.0, "frequencies": {"1.0": 1000.0}}

    placements = _pack_real_jobs_to_power(
        scheduler, cfg, start, start + pd.Timedelta(hours=4)
    )
    by_id = {placement["job"].job_id: placement for placement in placements}

    assert by_id[1]["y_kw"] == pytest.approx(0.0)
    assert by_id[2]["y_kw"] == pytest.approx(1.0)
    assert by_id[3]["y_kw"] == pytest.approx(0.0)
