from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from gen_jobs import generate_system_datasets, job_counts_for_clusters


def _queues():
    return [
        {"name": "short", "min_runtime_s": 0, "max_runtime_s": 24 * 3600},
        {"name": "medium", "min_runtime_s": 24 * 3600, "max_runtime_s": 7 * 24 * 3600},
        {"name": "long", "min_runtime_s": 7 * 24 * 3600, "max_runtime_s": None},
    ]


def test_balanced_job_counts_scale_by_cluster_capacity():
    clusters = {
        "large_cpu": 1000,
        "small_gpu": 50,
    }
    node_config = {
        "large_cpu": {"base_frequency_ghz": 2.0},
        "small_gpu": {"base_frequency_ghz": 1.5},
    }

    counts = job_counts_for_clusters(
        clusters,
        min_jobs=100,
        max_jobs=500,
        strategy="balanced",
        node_config=node_config,
    )

    assert counts["large_cpu"] == 500
    assert 100 <= counts["small_gpu"] < counts["large_cpu"]


def test_cluster_load_weights_shift_balanced_counts():
    clusters = {
        "large_cpu": 1000,
        "small_gpu": 50,
    }

    base_counts = job_counts_for_clusters(
        clusters,
        min_jobs=100,
        max_jobs=500,
        strategy="balanced",
    )
    weighted_counts = job_counts_for_clusters(
        clusters,
        min_jobs=100,
        max_jobs=500,
        strategy="balanced",
        cluster_load_weights={"small_gpu": 5.0},
    )

    assert weighted_counts["small_gpu"] > base_counts["small_gpu"]


def test_generate_system_datasets_writes_all_cluster_csvs(tmp_path):
    supercomputers = {
        "test_system": {
            "clusters": {
                "large_cpu": 100,
                "small_gpu": 20,
            },
            "queues": _queues(),
        }
    }
    node_config = {
        "large_cpu": {"base_frequency_ghz": 2.0},
        "small_gpu": {"base_frequency_ghz": 1.5},
    }

    summary = generate_system_datasets(
        "test_system",
        supercomputers_config=supercomputers,
        output_dir=tmp_path,
        node_config=node_config,
        min_jobs=2,
        max_jobs=5,
        seed=1,
        progress=None,
    )

    assert set(summary["job_counts"]) == {"large_cpu", "small_gpu"}
    for cluster, count in summary["job_counts"].items():
        queue_rows = 0
        for queue in ["short", "medium", "long"]:
            path = tmp_path / "jobs" / "test_system" / f"jobs_{cluster}_{queue}.csv"
            assert path.exists()
            df = pd.read_csv(path)
            queue_rows += len(df)
            assert "Requested_Walltime_s" in df.columns
            if not df.empty:
                assert df["Arrival_Offset_s"].max() <= summary["arrival_window_s"]
        assert queue_rows == count
