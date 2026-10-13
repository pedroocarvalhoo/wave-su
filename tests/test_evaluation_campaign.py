"""Integrity checks for the dissertation evaluation campaign."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import evaluation_campaign as campaign


def test_campaign_pairs_use_matching_workload_and_energy():
    cases = campaign.campaign_cases()
    assert len(cases) == 32
    for case in cases.values():
        if case.reference is None:
            continue
        reference = cases[case.reference]
        assert case.system == reference.system
        assert case.energy == reference.energy
        assert case.workload == reference.workload
        assert case.name != reference.name


def test_derived_workloads_preserve_jobs_and_compress_only_arrivals(tmp_path, monkeypatch):
    source = tmp_path / "datasets" / "jobs" / "deucalion_test"
    source.mkdir(parents=True)
    original = pd.DataFrame({
        "job_id": [1, 2, 3, 4],
        "Arrival_Offset_s": [0, 10, 20, 30],
        "Requested_Nodes": [1, 2, 3, 4],
    })
    original.to_csv(source / "jobs_arm_short.csv", index=False)
    monkeypatch.setattr(campaign, "DATASETS", tmp_path / "datasets")
    prepared = campaign.prepare_workloads(tmp_path / "campaign")
    half = pd.read_csv(prepared["half"] / "jobs_arm_short.csv")
    dense = pd.read_csv(prepared["dense"] / "jobs_arm_short.csv")
    assert half["job_id"].tolist() == [1, 3]
    assert dense["job_id"].tolist() == original["job_id"].tolist()
    assert dense["Arrival_Offset_s"].tolist() == [0, 5, 10, 15]
    assert dense["Requested_Nodes"].tolist() == original["Requested_Nodes"].tolist()
    (prepared["half"] / "jobs_arm_short.csv").write_text("changed")
    with pytest.raises(ValueError, match="differs from the reproducible derivation"):
        campaign.prepare_workloads(tmp_path / "campaign")


def test_collect_rejects_incompatible_paired_runs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    for name in ("eft_carbon_test", "puzzle_test"):
        (runs / f"{name}.json").write_text(json.dumps({"case": name}))

    def analyze(record):
        name = record["case"]
        return {
            "case": name,
            "group": "main_test",
            "system": "deucalion_test",
            "workload_hash": "same-workload",
            "energy": "pv490" if name == "eft_carbon_test" else "pv200",
            "source_hash": "same-source",
            "reference": "eft_carbon_test" if name == "puzzle_test" else None,
            "result_dir": name,
            "complete": True,
        }

    monkeypatch.setattr(campaign, "analyze_run", analyze)
    with pytest.raises(ValueError, match="Incompatible comparison"):
        campaign.collect(tmp_path)


def test_collect_rejects_duplicate_completed_cases(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    for index in range(2):
        (runs / f"{index}.json").write_text(json.dumps({"index": index}))
    monkeypatch.setattr(campaign, "analyze_run", lambda record: {
        "case": "puzzle_test", "group": "main_test", "result_dir": str(record["index"]),
        "complete": True,
    })
    with pytest.raises(ValueError, match="Multiple completed runs"):
        campaign.collect(tmp_path)


def test_analyze_run_rejects_missing_completed_job(tmp_path, monkeypatch):
    from dataclasses import dataclass

    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    pd.DataFrame({"job_id": [1, 2]}).to_csv(jobs_dir / "jobs_arm_short.csv", index=False)
    run_dir = tmp_path / "result"
    run_dir.mkdir()
    pd.DataFrame({"Job_ID": [1]}).to_csv(run_dir / "jobs_metrics_arm.csv", index=False)
    pd.DataFrame({"charge_level": [1000]}).to_csv(
        run_dir / "energy_simulation_result.csv", index=False
    )

    @dataclass
    class Metrics:
        co2_kg: float = 1.0

    monkeypatch.setattr(campaign, "data_hashes", lambda case, path: {"input": "same"})
    monkeypatch.setattr(campaign, "source_hash", lambda: "same")
    monkeypatch.setattr(campaign, "extract_run_metrics", lambda path: Metrics())
    record = {
        "case": campaign.campaign_cases()["eft_carbon_test"].__dict__,
        "result_dir": str(run_dir),
        "command": ["simulation_energy.py", "--jobs-dir", str(jobs_dir)],
        "expected_jobs": 2,
        "input_hashes": {"input": "same"},
        "source_hash": "same",
        "exit_code": 0,
        "wall_seconds": 1.0,
    }
    row = campaign.analyze_run(record)
    assert row["completed_jobs"] == 1
    assert row["missing_jobs"] == 1
    assert not row["complete"]


def test_watchdog_stops_a_hung_process_and_retains_log(tmp_path, monkeypatch):
    import sys
    import time

    monkeypatch.setattr(campaign, "RESULTS", tmp_path)
    log = tmp_path / "hung.log"
    started = time.monotonic()
    exit_code, reason, run_dir, elapsed = campaign.execute_guarded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        log, "never-created", tmp_path / "campaign",
        timeout_s=0.4, stall_s=10,
        max_run_bytes=10**9, max_campaign_bytes=10**9,
        min_free_bytes=0, poll_s=0.1,
    )
    assert reason == "wall_timeout"
    assert exit_code != 0
    assert run_dir is None
    assert elapsed < 5
    assert time.monotonic() - started < 5
    assert log.is_file()


def test_collect_records_failed_run_without_outputs(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    case = campaign.campaign_cases()["eft_carbon_test"]
    record = {
        "case": campaign.asdict(case), "result_dir": None,
        "exit_code": -15, "expected_jobs": 324,
        "wall_seconds": 0.5, "status": "incomplete",
        "termination_reason": "wall_timeout",
    }
    (runs / "stopped.json").write_text(json.dumps(record))
    monkeypatch.setattr(campaign, "analyze_run", lambda rec: (_ for _ in ()).throw(
        FileNotFoundError("no result directory")
    ))
    summary, _ = campaign.collect(tmp_path)
    rows = pd.read_csv(summary)
    assert rows.loc[0, "complete"] == False
    assert rows.loc[0, "termination_reason"] == "wall_timeout"
    assert rows.loc[0, "expected_jobs"] == 324


def test_source_snapshot_preserves_dirty_code_and_rejects_drift(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "vessim").mkdir(parents=True)
    (root / "scripts").mkdir()
    source = root / "vessim" / "model.py"
    runner = root / "scripts" / "simulation_energy.py"
    campaign_script = root / "scripts" / "evaluation_campaign.py"
    source.write_text("version = 1\n")
    runner.write_text("runner = 1\n")
    campaign_script.write_text("campaign = 1\n")
    monkeypatch.setattr(campaign, "ROOT", root)
    monkeypatch.setattr(campaign, "SOURCE_FILES", (runner, campaign_script))
    output = tmp_path / "campaign"
    digest = campaign.snapshot_source(output)
    assert (output / "source_snapshot" / "vessim" / "model.py").read_text() == "version = 1\n"
    assert campaign.snapshot_source(output) == digest
    source.write_text("version = 2\n")
    with pytest.raises(ValueError, match="Source changed"):
        campaign.snapshot_source(output)


def test_watchdog_stops_run_without_log_progress(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(campaign, "RESULTS", tmp_path)
    code, reason, _, elapsed = campaign.execute_guarded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "silent.log", "no-result", tmp_path / "campaign",
        timeout_s=10, stall_s=0.3,
        max_run_bytes=10**9, max_campaign_bytes=10**9,
        min_free_bytes=0, poll_s=0.1,
    )
    assert code != 0
    assert reason == "no_log_progress"
    assert elapsed < 5


def test_watchdog_stops_run_at_output_limit(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(campaign, "RESULTS", tmp_path)
    output = tmp_path / "result_eval-size-test"
    script = (
        "from pathlib import Path; import time; "
        f"p=Path({str(output)!r}); p.mkdir(); "
        "(p/'large.bin').open('wb').truncate(1048576); time.sleep(30)"
    )
    code, reason, result_dir, elapsed = campaign.execute_guarded(
        [sys.executable, "-c", script],
        tmp_path / "large.log", "eval-size-test", tmp_path / "campaign",
        timeout_s=10, stall_s=10,
        max_run_bytes=100000, max_campaign_bytes=10**9,
        min_free_bytes=0, poll_s=0.1,
    )
    assert code != 0
    assert reason == "run_size_limit"
    assert result_dir == output
    assert elapsed < 5
