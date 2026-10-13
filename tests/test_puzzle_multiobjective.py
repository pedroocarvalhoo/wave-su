from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from vessim.controller import Monitor
from vessim.wave.energy.puzzle_multiobjective import (
    MultiObjectiveLimits,
    PuzzleRunMetrics,
    constraint_violations,
    extract_run_metrics,
    select_multi_scenario,
    write_selection_outputs,
)


def metrics(**changes) -> PuzzleRunMetrics:
    base = PuzzleRunMetrics(
        results_dir="/tmp/run",
        run_tag="candidate",
        supercomputer="test",
        completed_jobs=10,
        deadline_misses=0,
        co2_kg=100.0,
        it_energy_kwh=1000.0,
        physical_grid_import_kwh=100.0,
        physical_grid_export_kwh=10.0,
        attributed_grid_kwh=80.0,
        mean_wait_h=2.0,
        p95_wait_h=5.0,
        makespan_h=20.0,
        workload_end="2023-01-01 20:00:00",
        simulation_end="2023-01-01 20:05:00",
        green_dispatches=8,
        grid_dispatches=2,
        fallback_dispatches=0,
        planned_misses=0,
        budget_violation_samples=0,
        budget_max_excess_w=0.0,
        source_identity_max_error_w=1e-9,
    )
    return replace(base, **changes)


def test_constraints_reject_service_and_correctness_regressions():
    reference = metrics()
    candidate = metrics(
        completed_jobs=9,
        deadline_misses=1,
        p95_wait_h=5.1,
        makespan_h=20.2,
        budget_violation_samples=2,
        source_identity_max_error_w=0.01,
    )

    violations = constraint_violations(candidate, reference, MultiObjectiveLimits())

    assert any("completed_jobs" in item for item in violations)
    assert any("deadline_misses" in item for item in violations)
    assert any("p95_regression" in item for item in violations)
    assert any("makespan_regression" in item for item in violations)
    assert any("budget_violation" in item for item in violations)
    assert any("source_identity" in item for item in violations)


def test_constraints_honour_explicit_service_slack():
    reference = metrics()
    candidate = metrics(mean_wait_h=2.2, p95_wait_h=5.5, makespan_h=21.0)
    limits = MultiObjectiveLimits(
        max_makespan_regression_h=1.0,
        max_p95_regression_h=0.5,
        max_mean_wait_regression_h=0.2,
    )

    assert constraint_violations(candidate, reference, limits) == ()


def test_selector_minimises_co2_only_among_feasible_candidates():
    reference = metrics(co2_kg=100.0)
    runs = {
        "one": {
            "reference": reference,
            "clean-feasible": metrics(co2_kg=90.0),
            "cleaner-but-slow": metrics(co2_kg=80.0, makespan_h=20.01),
        }
    }

    selection = select_multi_scenario(runs, "reference")

    assert selection.selected_candidate == "clean-feasible"
    rejected = next(
        item for item in selection.candidates if item.candidate == "cleaner-but-slow"
    )
    assert rejected.feasible is False


def test_selector_requires_candidate_coverage_in_every_scenario():
    runs = {
        "one": {"reference": metrics(), "candidate": metrics()},
        "two": {"reference": metrics()},
    }

    with pytest.raises(ValueError, match="same candidates"):
        select_multi_scenario(runs, "reference")


def test_selector_uses_mean_relative_co2_across_scenarios():
    runs = {
        "large": {
            "reference": metrics(co2_kg=1000.0),
            "balanced": metrics(co2_kg=900.0),
            "large-only": metrics(co2_kg=800.0),
        },
        "small": {
            "reference": metrics(co2_kg=10.0),
            "balanced": metrics(co2_kg=9.0),
            "large-only": metrics(co2_kg=12.0),
        },
    }

    selection = select_multi_scenario(runs, "reference")

    assert selection.selected_candidate == "balanced"


def test_limit_validation_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        MultiObjectiveLimits(max_p95_regression_h=-0.1)


def _write_result_fixture(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "run_tag": "fixture",
                "supercomputer": "test",
                "sim_start": "2023-01-01 00:00:00",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "Submission_Time": ["2023-01-01 00:00:00", "2023-01-01 01:00:00.500"],
            "Wait_Time_s": [3600.0, 7200.0],
            "Real_Exec_Time_s": [1800.0, 3600.0],
            "Queue_Deadline_Missed": [False, True],
        }
    ).to_csv(root / "jobs_metrics_cluster.csv", index=False)
    energy = pd.DataFrame(
        {
            "e_delta": [-3_600_000.0, 7_200_000.0],
            "cluster_power": [50.0, 120.0],
            "budget_scheduler_w": [100.0, 100.0],
            "budget_renewable_w": [130.0, 130.0],
            "budget_battery_w": [0.0, 0.0],
            "budget_grid_w": [0.0, 0.0],
            "budget_pue": [1.3, 1.3],
        },
        index=pd.to_datetime(["2023-01-01 00:00", "2023-01-01 04:01"]),
    )
    carbon_g_per_wh = dict(zip(energy.index, [0.1, 0.2]))
    monitor = Monitor(
        step_size=60,
        co2_signal=SimpleNamespace(
            atco2=lambda time, column: carbon_g_per_wh[time],
        ),
    )
    for time, e_delta in energy["e_delta"].items():
        monitor.step(time, p_delta=0.0, e_delta=e_delta, actor_states={})
    energy["co2"] = [monitor.monitor_log[time]["co2"] for time in energy.index]
    energy.to_csv(root / "energy_simulation_result.csv")
    pd.DataFrame(
        {
            "co2_total_kg_cumulative": energy["co2"].cumsum().to_numpy() / 1000.0,
            "total_energy_kwh": [10.0, 25.0],
            "cluster_grid_energy_kwh_cumulative": [2.0, 7.0],
        }
    ).to_csv(root / "carbon_metrics.csv", index=False)
    pd.DataFrame(
        {
            "green_dispatched": [1],
            "grid_dispatched": [1],
            "fallback_dispatched": [0],
            "planned_missed": [0],
        }
    ).to_csv(root / "puzzle_dispatch_summary.csv", index=False)


def test_extract_run_metrics_reads_physical_service_and_invariant_values(tmp_path):
    root = tmp_path / "run"
    _write_result_fixture(root)

    result = extract_run_metrics(root)

    assert result.completed_jobs == 2
    assert result.deadline_misses == 1
    # The import is 1 kWh at 100 g/kWh; exporting has no operational CO2.
    # The monitor's additional term is reported separately in both records.
    assert result.co2_kg == pytest.approx(0.1)
    assert result.monitor_baseline_co2_kg == pytest.approx(3e-7)
    assert result.it_energy_kwh == 25.0
    assert result.physical_grid_import_kwh == 1.0
    assert result.physical_grid_export_kwh == 2.0
    assert result.attributed_grid_kwh == 7.0
    assert result.mean_wait_h == 1.5
    assert result.p95_wait_h == pytest.approx(1.95)
    assert result.makespan_h == pytest.approx(4.0001388889)
    assert result.budget_violation_samples == 1
    assert result.budget_max_excess_w == 20.0
    assert result.source_identity_max_error_w == pytest.approx(0.0)


def test_extract_run_metrics_supports_no_plots_results(tmp_path):
    root = tmp_path / "run"
    _write_result_fixture(root)
    (root / "carbon_metrics.csv").unlink()
    energy = pd.read_csv(root / "energy_simulation_result.csv", index_col=0)
    energy["cluster_energy_kwh"] = [1.0, 3.0]
    energy["budget_renewable_w"] = [130.0, 65.0]
    energy["budget_grid_w"] = [0.0, 65.0]
    energy.to_csv(root / "energy_simulation_result.csv")

    result = extract_run_metrics(root)

    assert result.co2_kg == pytest.approx(0.1)
    assert result.monitor_baseline_co2_kg == pytest.approx(3e-7)
    assert result.it_energy_kwh == pytest.approx((50.0 + 120.0) / 60_000.0)
    assert result.attributed_grid_kwh == pytest.approx(1.0)


def test_write_selection_outputs_records_selected_and_rejected_candidates(tmp_path):
    runs = {
        "scenario": {
            "reference": metrics(),
            "clean": metrics(co2_kg=90.0),
            "slow": metrics(co2_kg=80.0, makespan_h=21.0),
        }
    }
    limits = MultiObjectiveLimits()
    selection = select_multi_scenario(runs, "reference", limits)

    csv_path, json_path = write_selection_outputs(
        tmp_path, runs, selection, limits
    )

    table = pd.read_csv(csv_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["selected_candidate"] == "clean"
    assert table.loc[table["candidate"] == "clean", "selected"].item()
    assert not table.loc[table["candidate"] == "slow", "scenario_feasible"].item()
