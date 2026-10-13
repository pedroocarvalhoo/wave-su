"""Static safeguards for the proposed Chapter 5 matrix; no simulations run."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import evaluation_campaign as base
from scripts import evaluation_campaign_v5 as v5


def test_main_matrix_is_energy_first_and_uses_adaptive_dvfs():
    cases = v5.campaign_cases()
    stages = ("grid", "carbon", "solar", "solar_wind")
    assert len(cases) == 21  # 16 main, 2 fixed-frequency controls, 3 pilots
    for stage in stages:
        rows = [cases[f"{stage}_{name}"] for name in ("fifo", "priority", "eft", "puzzle")]
        assert len({c.energy for c in rows}) == 1
        assert len({c.workload for c in rows}) == 1
        assert len({c.system for c in rows}) == 1
        for case in rows:
            assert v5.POLICIES[case.policy][-1] == "adaptive" or case.policy == "puzzle"
            if stage == "grid":
                assert case.energy == "zero"
            else:
                assert case.extra[-2:] == ("--carbon-grid-threshold", str(v5.THRESHOLD))
        assert cases[f"{stage}_priority"].reference == f"{stage}_fifo"
        assert cases[f"{stage}_eft"].reference == f"{stage}_fifo"
        assert cases[f"{stage}_puzzle"].reference == f"{stage}_eft"
    assert v5.ENERGY["zero"][-1] == "pv=0"
    assert v5.ENERGY["pv40"][-1] == "pv=40000"
    assert "wind=40000" in v5.ENERGY["pv40_wind40"]
    assert "eft-dvfs" in v5.POLICIES["eft_carbon-minimizing"]


def test_fixed_frequency_controls_change_only_dvfs_setting():
    cases = v5.campaign_cases()
    for name in ("fifo", "priority"):
        fixed = cases[f"dvfs_{name}_fixed"]
        adaptive = cases[f"solar_{name}"]
        assert (fixed.energy, fixed.workload, fixed.system, fixed.extra) == (
            adaptive.energy, adaptive.workload, adaptive.system, adaptive.extra
        )
        args_fixed = v5.POLICIES[fixed.policy]
        args_adaptive = v5.POLICIES[adaptive.policy]
        assert args_fixed[:-1] == args_adaptive[:-1]
        assert args_fixed[-1] == "fixed" and args_adaptive[-1] == "adaptive"


def test_execution_requires_explicit_approval_flag(monkeypatch):
    started = []
    monkeypatch.setattr(base, "run_case", lambda *args, **kwargs: started.append(args))
    with pytest.raises(SystemExit) as exc:
        v5.main(["run", "grid_fifo"])
    assert exc.value.code == 2
    assert not started


def test_threshold_has_early_grid_windows_unlike_170():
    series = pd.read_csv(v5.ROOT / "datasets" / "historical_data.csv", parse_dates=["datetime"])
    first_72h = series.set_index("datetime")["carbon_intensity"].loc[
        "2023-06-01 12:00:00+00:00":
    ].iloc[:72]
    assert not (first_72h <= 170).any()
    assert (first_72h <= v5.THRESHOLD).any()
    assert (first_72h > v5.THRESHOLD).any()


def test_queue_order_figure_uses_observed_start_times(tmp_path, monkeypatch):
    import json
    from scripts import evaluation_campaign_v5_visuals as visuals

    root = tmp_path / "repo"
    source = root / "datasets" / "jobs" / "deucalion_test" / "jobs_arm_a64fx_fujitsu_medium.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame({
        "job_id": [11, 12, 13], "Arrival_Offset_s": [0, 60, 120],
        "Priority": [1, 3, 2],
    }).to_csv(source, index=False)
    campaign = root / "results" / "evaluation_campaign" / "v5"
    campaign.mkdir(parents=True)
    rows = []
    for policy, waits in (
        ("fifo", [10, 20, 30]),
        ("priority", [200, 20, 30]),
        ("eft", [200, 180, 30]),
        ("puzzle", [200, 20, 30]),
    ):
        case = f"grid_{policy}"
        run = campaign / case
        run.mkdir()
        pd.DataFrame({
            "Job_ID": [11, 12, 13], "Priority": [1, 3, 2],
            "Queue_Name": ["medium"] * 3,
            "Submission_Time": ["2023-06-01 12:00:00", "2023-06-01 12:01:00", "2023-06-01 12:02:00"],
            "Wait_Time_s": waits,
        }).to_csv(run / "jobs_metrics_arm_a64fx_fujitsu.csv", index=False)
        rows.append({
            "case": case, "complete": True, "run_status": "complete",
            "system": "same", "energy": "zero", "workload_hash": "same",
            "source_hash": "same", "result_dir": str(run),
        })
    pd.DataFrame(rows).to_csv(campaign / "summary.csv", index=False)
    monkeypatch.setattr(visuals, "ROOT", root)
    monkeypatch.setattr(visuals, "MEDIUM_QUEUE", source)
    output = visuals.queue_order(campaign, "grid", n=3)
    evidence = json.loads((output.parent / "queue_order_grid_evidence.json").read_text())
    assert output.is_file()
    assert evidence["selected_job_ids"] == [11, 12, 13]
    assert evidence["actual_orders"]["grid_fifo"] == [11, 12, 13]
    assert evidence["actual_orders"]["grid_priority"] == [12, 13, 11]


def test_synthesis_and_dvfs_tables_keep_physical_import_distinct(tmp_path, monkeypatch):
    from scripts import evaluation_campaign_v5_visuals as visuals

    root = tmp_path / "repo"
    campaign = root / "results" / "evaluation_campaign" / "v5"
    campaign.mkdir(parents=True)
    rows = []
    names = [f"{stage}_{policy}" for stage in visuals.STAGES
             for policy in ("fifo", "priority", "eft", "puzzle")]
    names += ["dvfs_fifo_fixed", "dvfs_priority_fixed"]
    for i, name in enumerate(names):
        run = campaign / name
        run.mkdir()
        pd.DataFrame({"Frequency_Changes": [i, 1]}).to_csv(
            run / "jobs_metrics_arm_a64fx_fujitsu.csv", index=False
        )
        rows.append({
            "case": name, "complete": True, "run_status": "complete",
            "system": "same", "source_hash": "same", "workload_hash": "same",
            "energy": ("pv40" if name.startswith(("dvfs", "solar_"))
                       else "pv40_wind40" if name.startswith("solar_wind_")
                       else "zero"),
            "result_dir": str(run), "completed_jobs": 324, "deadline_misses": 0,
            "mean_wait_h": i + 1, "p95_wait_h": i + 2, "makespan_h": i + 3,
            "it_energy_kwh": 100 + i, "physical_grid_import_kwh": 50 + i,
            "co2_kg": 10 + i, "battery_first_record_kwh": 1000,
            "battery_final_kwh": 900, "budget_violation_samples": 0,
            "budget_max_excess_w": 0,
        })
    pd.DataFrame(rows).to_csv(campaign / "summary.csv", index=False)
    monkeypatch.setattr(visuals, "ROOT", root)
    table, plot = visuals.synthesis(campaign)
    aggregate = pd.read_csv(table)
    assert len(aggregate) == 16 and plot.is_file()
    assert aggregate.loc[0, "effective_grid_g_per_kwh"] == 200
    dvfs_table, dvfs_plot = visuals.dvfs_effect(campaign)
    comparison = pd.read_csv(dvfs_table)
    assert len(comparison) == 4 and dvfs_plot.is_file()
    assert comparison["frequency"].tolist() == ["Fixed", "Adaptive", "Fixed", "Adaptive"]
