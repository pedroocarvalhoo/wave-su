"""Safeguards for the five-day Chapter 5 experiment; no simulations run."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import evaluation_campaign as base
from scripts import evaluation_campaign_v5 as v5
from scripts import evaluation_campaign_v7 as v7


def test_derived_workload_preserves_job_identity_and_queue_fields(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source = root / "datasets" / "jobs" / "deucalion_test"
    source.mkdir(parents=True)
    for index, size in enumerate((150, 100, 74)):
        frame = pd.DataFrame({
            "job_id": range(sum((150, 100, 74)[:index]), sum((150, 100, 74)[:index + 1])),
            "#I": [105 + index] * size,
            "Arrival_Offset_s": range(size),
            "Priority": [index] * size,
            "Queue_Name": ["medium"] * size,
            "Walltime_s": [3600] * size,
            "Nodes": [2] * size,
        })
        frame.to_csv(source / f"jobs_{index}.csv", index=False)
    monkeypatch.setattr(v7, "ROOT", root)
    campaign = root / "results" / "evaluation_campaign" / "v7"
    target = v7.prepare_workload(campaign)[v7.WORKLOAD]
    assert len(list(target.glob("*.csv"))) == 3
    for original in source.glob("*.csv"):
        before = pd.read_csv(original)
        after = pd.read_csv(target / original.name)
        pd.testing.assert_frame_equal(before.drop(columns=["#I"]), after.drop(columns=["#I"]))
        assert after["#I"].tolist() == (before["#I"] * 0.2).round().astype("int64").tolist()
    assert v7.prepare_workload(campaign)[v7.WORKLOAD] == target
    changed = target / "jobs_0.csv"
    changed.write_text(changed.read_text().replace("medium", "high", 1))
    with pytest.raises(ValueError, match="Derived workload changed"):
        v7.prepare_workload(campaign)


def test_five_day_matrix_is_matched_and_adaptive_except_controls():
    matrix = v7.cases()
    assert len(matrix) == 18
    assert len({case.name for case in matrix.values()}) == 18
    for profile in ("grid", "carbon", "solar", "solar_wind"):
        rows = [matrix[f"{profile}_{policy}"] for policy in ("fifo", "priority", "eft", "puzzle")]
        assert {case.workload for case in rows} == {v7.WORKLOAD}
        assert len({case.energy for case in rows}) == 1
        assert len({case.system for case in rows}) == 1
        for case in rows:
            assert v5.POLICIES[case.policy][-1] == "adaptive" or case.policy == "puzzle"
            if profile == "grid":
                assert case.energy == "zero"
            else:
                assert case.extra[-2:] == ("--carbon-grid-threshold", str(v7.THRESHOLD))
    assert matrix["grid_puzzle"].extra == ("--carbon-grid-threshold", "2000.0")
    assert matrix["solar_fifo"].energy == matrix["dvfs_fifo_fixed"].energy
    assert matrix["solar_priority"].energy == matrix["dvfs_priority_fixed"].energy
    for policy in ("fifo", "priority"):
        fixed = matrix[f"dvfs_{policy}_fixed"]
        adaptive = matrix[f"solar_{policy}"]
        assert (fixed.system, fixed.energy, fixed.workload, fixed.extra) == (
            adaptive.system, adaptive.energy, adaptive.workload, adaptive.extra
        )
        assert v5.POLICIES[fixed.policy][:-1] == v5.POLICIES[adaptive.policy][:-1]
        assert v5.POLICIES[fixed.policy][-1] == "fixed"


def test_runner_configuration_sets_bounded_energy_inputs(monkeypatch):
    original = {name: getattr(base, name) for name in (
        "POLICIES", "ENERGY", "COMMON", "SOURCE_FILES", "prepare_workloads",
        "jobs_dir_for", "campaign_cases",
    )}
    with monkeypatch.context() as patch:
        for name, value in original.items():
            patch.setattr(base, name, value)
        v7.configure()
        common = list(base.COMMON)
        assert common[common.index("--grid-power-limit") + 1] == "200000"
        assert common[common.index("--battery-capacity-kwh") + 1] == "200"
        assert "--no-snapshot" not in common
        assert "--sla-grid-override" not in common
        assert "pv=20000" in base.ENERGY["pv20"]
        assert "wind=20000" in base.ENERGY["pv20_wind20"]
        assert base.campaign_cases is v7.cases
