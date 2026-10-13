"""Reproducible WAVE and Puzzle experiments for dissertation Chapter 5.

Prepare derived workloads, run one case at a time, and collect comparable
measurements. This script never rewrites the source datasets or archived runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vessim.wave.energy.puzzle_multiobjective import extract_run_metrics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "outputs"
DEFAULT_CAMPAIGN = RESULTS / "evaluation_campaign" / "v4"
SOURCE_FILES = (
    ROOT / "scripts" / "simulation_energy.py",
    ROOT / "scripts" / "evaluation_campaign.py",
)


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    system: str
    policy: str
    energy: str = "pv20"
    workload: str = "original"
    reference: str | None = None
    extra: tuple[str, ...] = ()


COMMON = (
    "--no-plots",
    "--no-snapshot",
    "--no-scheduler-debug-log",
    "--no-grid-charging",
    "--sla-grid-override",
    "--pue", "1.3",
    "--grid-power-limit", "200000",
    "--renewable-charge-fraction", "0.3",
    "--energy-scheduler-step-size", "300",
    "--microgrid-step-size", "60",
    "--battery-capacity-kwh", "2000",
)

POLICIES = {
    "fifo_grid": (
        "--budget-policy", "grid",
        "--allocation-strategy", "default",
        "--scheduling-policy", "fifo",
        "--dvfs-policy", "fixed",
    ),
    "fifo_carbon": (
        "--budget-policy", "carbon-minimizing",
        "--allocation-strategy", "compute-prioritized",
        "--scheduling-policy", "fifo",
        "--dvfs-policy", "fixed",
    ),
    "priority_carbon": (
        "--budget-policy", "carbon-minimizing",
        "--allocation-strategy", "compute-prioritized",
        "--scheduling-policy", "priority",
        "--dvfs-policy", "fixed",
    ),
    "eft_carbon": (
        "--budget-policy", "carbon-minimizing",
        "--allocation-strategy", "compute-prioritized",
        "--scheduling-policy", "eft-dvfs",
        "--dvfs-policy", "adaptive",
    ),
    "puzzle": (
        "--budget-policy", "puzzle",
        "--allocation-strategy", "puzzle-plan",
        "--scheduling-policy", "puzzle",
        "--dvfs-policy", "adaptive",
        "--no-puzzle-fallback-dispatch",
    ),
}

ENERGY = {
    "pv10": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=10000"),
    "pv20": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=20000"),
    "pv40": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=40000"),
    "pv_wind": (
        "--renewable-sources", "pv,wind",
        "--renewable-source-scale", "pv=15000",
        "--renewable-source-scale", "wind=15000",
    ),
    "wind20": ("--renewable-sources", "wind", "--renewable-source-scale", "wind=20000"),
    "pv490": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=490000"),
}


def campaign_cases() -> dict[str, Case]:
    rows = [
        Case("fifo_grid_test", "main_test", "deucalion_test", "fifo_grid"),
        Case("fifo_carbon_test", "main_test", "deucalion_test", "fifo_carbon", reference="eft_carbon_test"),
        Case("priority_carbon_test", "main_test", "deucalion_test", "priority_carbon", reference="eft_carbon_test"),
        Case("eft_carbon_test", "main_test", "deucalion_test", "eft_carbon"),
        Case("puzzle_test", "main_test", "deucalion_test", "puzzle", reference="eft_carbon_test"),
        Case("fifo_grid_macc", "main_macc", "deucalion_macc", "fifo_grid", energy="pv490"),
        Case("eft_carbon_macc", "main_macc", "deucalion_macc", "eft_carbon", energy="pv490"),
        Case("puzzle_macc", "main_macc", "deucalion_macc", "puzzle", energy="pv490", reference="eft_carbon_macc"),
        Case("puzzle_local_board", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--no-puzzle-global-green-board",)),
        Case("puzzle_legacy_grid", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--no-puzzle-carbon-aware-placement",)),
        Case("puzzle_no_regret", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--no-puzzle-carbon-regret",)),
        Case("puzzle_no_tail_reserve", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--puzzle-tail-battery-reserve-fraction", "0")),
        Case("puzzle_no_battery_priority", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--no-puzzle-battery-carbon-aware-dispatch",)),
        Case("puzzle_repair", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--puzzle-repack-repair",)),
        Case("puzzle_front_run", "ablation", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--puzzle-allow-front-run",)),
        Case("puzzle_pv10", "energy", "deucalion_test", "puzzle", energy="pv10", reference="eft_pv10"),
        Case("eft_pv10", "energy", "deucalion_test", "eft_carbon", energy="pv10"),
        Case("puzzle_pv40", "energy", "deucalion_test", "puzzle", energy="pv40", reference="eft_pv40"),
        Case("eft_pv40", "energy", "deucalion_test", "eft_carbon", energy="pv40"),
        Case("puzzle_pv_wind", "energy", "deucalion_test", "puzzle", energy="pv_wind", reference="eft_pv_wind"),
        Case("eft_pv_wind", "energy", "deucalion_test", "eft_carbon", energy="pv_wind"),
        Case("puzzle_wind20", "energy", "deucalion_test", "puzzle", energy="wind20", reference="eft_wind20"),
        Case("eft_wind20", "energy", "deucalion_test", "eft_carbon", energy="wind20"),
        Case("puzzle_no_battery", "energy", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--battery-p-charge-max", "0", "--battery-p-discharge-max", "0")),
        Case("puzzle_half", "workload", "deucalion_test", "puzzle", workload="half", reference="eft_half"),
        Case("eft_half", "workload", "deucalion_test", "eft_carbon", workload="half"),
        Case("puzzle_dense", "workload", "deucalion_test", "puzzle", workload="dense", reference="eft_dense"),
        Case("eft_dense", "workload", "deucalion_test", "eft_carbon", workload="dense"),
        Case("puzzle_tail_robust", "information", "deucalion_test", "puzzle", reference="puzzle_test",
             extra=("--puzzle-tail-renewable-robustness", "1",
                    "--puzzle-tail-renewable-calibration-days", "0")),
        Case("eft_error_seed11", "information", "deucalion_test", "eft_carbon", reference="eft_carbon_test",
             extra=("--runtime-prediction-error-rate", "0.25", "--runtime-prediction-seed", "11")),
        Case("eft_error_seed22", "information", "deucalion_test", "eft_carbon", reference="eft_carbon_test",
             extra=("--runtime-prediction-error-rate", "0.25", "--runtime-prediction-seed", "22")),
        Case("eft_error_seed33", "information", "deucalion_test", "eft_carbon", reference="eft_carbon_test",
             extra=("--runtime-prediction-error-rate", "0.25", "--runtime-prediction-seed", "33")),
    ]
    result = {case.name: case for case in rows}
    assert len(result) == len(rows)
    for case in rows:
        assert case.policy in POLICIES and case.energy in ENERGY
        assert case.reference is None or case.reference in result
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(path_label(path).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def workload_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("jobs_*.csv"))
    if not files:
        raise FileNotFoundError(f"No jobs_*.csv files in {directory}")
    return files


def prepare_workloads(campaign_dir: Path) -> dict[str, Path]:
    source = DATASETS / "jobs" / "deucalion_test"
    outputs = {
        "half": campaign_dir / "inputs" / "deucalion_test_half",
        "dense": campaign_dir / "inputs" / "deucalion_test_dense",
    }
    for name, directory in outputs.items():
        directory.mkdir(parents=True, exist_ok=True)
        for input_file in workload_files(source):
            frame = pd.read_csv(input_file)
            if name == "half":
                derived = frame.iloc[::2].copy()
            else:
                derived = frame.copy()
                derived["Arrival_Offset_s"] = (
                    pd.to_numeric(derived["Arrival_Offset_s"], errors="raise") / 2.0
                )
            target = directory / input_file.name
            generated = derived.to_csv(index=False).encode()
            if target.exists():
                if target.read_bytes() != generated:
                    raise ValueError(
                        f"{target} differs from the reproducible derivation; use a new campaign directory"
                    )
            else:
                target.write_bytes(generated)
    return outputs


def jobs_dir_for(case: Case, campaign_dir: Path) -> Path:
    if case.workload == "original":
        return DATASETS / "jobs" / case.system
    if case.system != "deucalion_test":
        raise ValueError("Derived workloads are defined only for deucalion_test")
    return campaign_dir / "inputs" / f"deucalion_test_{case.workload}"


def expected_job_keys(jobs_dir: Path) -> set[tuple[str, int]]:
    expected: set[tuple[str, int]] = set()
    for path in workload_files(jobs_dir):
        cluster = path.stem.rsplit("_", 1)[0].removeprefix("jobs_")
        frame = pd.read_csv(path, usecols=["job_id"])
        keys = {(cluster, int(job_id)) for job_id in frame["job_id"]}
        if len(keys) != len(frame) or expected.intersection(keys):
            raise ValueError(f"Duplicate job IDs in {path}")
        expected.update(keys)
    return expected


def output_job_keys(run_dir: Path) -> set[tuple[str, int]]:
    observed: set[tuple[str, int]] = set()
    for path in sorted(run_dir.glob("jobs_metrics_*.csv")):
        cluster = path.stem.removeprefix("jobs_metrics_")
        frame = pd.read_csv(path, usecols=["Job_ID"])
        keys = {(cluster, int(job_id)) for job_id in frame["Job_ID"]}
        if len(keys) != len(frame) or observed.intersection(keys):
            raise ValueError(f"Duplicate completed job IDs in {path}")
        observed.update(keys)
    return observed


def data_hashes(case: Case, jobs_dir: Path) -> dict[str, str]:
    sources = json.loads((DATASETS / "renewable_sources.json").read_text())
    selected = ENERGY[case.energy][1].split(",")
    files = workload_files(jobs_dir) + [
        DATASETS / "historical_data.csv",
        DATASETS / "node_config.json",
        DATASETS / "supercomputers" / f"{case.system}.json",
        DATASETS / "renewable_sources.json",
    ]
    files += [DATASETS / sources[name]["dataset"] for name in selected]
    return {path_label(path): sha256(path) for path in sorted(set(files))}


def source_hash() -> str:
    paths = list((ROOT / "vessim").rglob("*.py")) + list(SOURCE_FILES)
    return hash_files(paths)


def snapshot_source(campaign_dir: Path) -> str:
    """Retain the exact Python sources behind a campaign with a dirty checkout."""
    digest = source_hash()
    snapshot = campaign_dir / "source_snapshot"
    marker = snapshot / "source_hash.txt"
    if marker.exists():
        if marker.read_text(encoding="utf-8").strip() != digest:
            raise ValueError("Source changed within this campaign; use a new directory")
        return digest
    if snapshot.exists():
        raise ValueError(f"Incomplete source snapshot at {snapshot}")
    files = list((ROOT / "vessim").rglob("*.py")) + list(SOURCE_FILES)
    for path in files:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    marker.write_text(digest + "\n", encoding="utf-8")
    return digest


def case_args(case: Case, jobs_dir: Path, run_tag: str) -> list[str]:
    args = [
        str(ROOT / "scripts" / "simulation_energy.py"),
        case.system,
        *COMMON,
        *POLICIES[case.policy],
        *ENERGY[case.energy],
        *case.extra,
        "--jobs-dir", str(jobs_dir.resolve()),
        "--run-tag", run_tag,
    ]
    return args


def directory_bytes(directory: Path) -> int:
    if not directory.exists():
        return 0
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def campaign_bytes(campaign_dir: Path, current_dir: Path | None = None) -> int:
    total = directory_bytes(campaign_dir)
    for metadata in (campaign_dir / "runs").glob("*.json"):
        try:
            run_dir = Path(json.loads(metadata.read_text())["result_dir"])
        except (KeyError, TypeError, ValueError):
            continue
        if current_dir is None or run_dir != current_dir:
            total += directory_bytes(run_dir)
    if current_dir is not None:
        total += directory_bytes(current_dir)
    return total


def terminate_group(process: subprocess.Popen, grace_s: float = 15.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute_guarded(
    command: list[str], log_path: Path, tag: str, campaign_dir: Path,
    *, timeout_s: float, stall_s: float, max_run_bytes: int,
    max_campaign_bytes: int, min_free_bytes: int, poll_s: float = 5.0,
    env: dict[str, str] | None = None,
) -> tuple[int, str | None, Path | None, float]:
    """Run a simulation with hard time, progress and storage bounds."""
    if shutil.disk_usage(RESULTS).free < min_free_bytes:
        raise RuntimeError("Insufficient free disk space to start another run")
    if campaign_bytes(campaign_dir) >= max_campaign_bytes:
        raise RuntimeError("Campaign storage limit reached")
    clock = time.monotonic()
    run_dir = None
    reason = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        last_activity = clock
        last_log_size = 0
        while process.poll() is None:
            try:
                process.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            log_size = log_path.stat().st_size
            if log_size > last_log_size:
                last_activity = now
                last_log_size = log_size
            candidates = list(RESULTS.glob(f"*_{tag}"))
            if len(candidates) == 1:
                run_dir = candidates[0]
            if now - clock > timeout_s:
                reason = "wall_timeout"
            elif now - last_activity > stall_s:
                reason = "no_log_progress"
            elif shutil.disk_usage(RESULTS).free < min_free_bytes:
                reason = "low_disk_space"
            elif run_dir is not None and directory_bytes(run_dir) > max_run_bytes:
                reason = "run_size_limit"
            elif campaign_bytes(campaign_dir, run_dir) > max_campaign_bytes:
                reason = "campaign_size_limit"
            if reason:
                terminate_group(process)
                break
    candidates = list(RESULTS.glob(f"*_{tag}"))
    if len(candidates) == 1:
        run_dir = candidates[0]
    return process.returncode, reason, run_dir, time.monotonic() - clock


def run_case(
    case: Case, campaign_dir: Path, *, timeout_s: float | None = None,
    stall_s: float = 1800, max_run_gb: float = 3,
    max_campaign_gb: float = 10, min_free_gb: float = 10,
) -> Path:
    if case.workload != "original":
        prepare_workloads(campaign_dir)
    jobs_dir = jobs_dir_for(case, campaign_dir)
    source_digest = snapshot_source(campaign_dir)
    expected = expected_job_keys(jobs_dir)
    for path in (campaign_dir / "runs").glob("*.json"):
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior["case"]["name"] == case.name and prior.get("status") == "complete":
            raise ValueError(f"{case.name} already completed in {campaign_dir}")
    tag = ("eval-" + campaign_dir.name + "-" + case.name + "-"
           + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    command = [sys.executable, "-u", *case_args(case, jobs_dir, tag)]
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "WAVE_PUZZLE_FORECAST_MAX_PLANS": "8",
    })
    runs_dir = campaign_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{tag}.log"
    started = datetime.now(timezone.utc)
    limits = {
        "timeout_s": timeout_s if timeout_s is not None else (
            10800 if case.system == "deucalion_macc" else 3600
        ),
        "stall_s": stall_s,
        "max_run_bytes": int(max_run_gb * 1024**3),
        "max_campaign_bytes": int(max_campaign_gb * 1024**3),
        "min_free_bytes": int(min_free_gb * 1024**3),
    }
    exit_code, reason, run_dir, wall_seconds = execute_guarded(
        command, log_path, tag, campaign_dir, env=env, **limits
    )
    record = {
        "case": asdict(case),
        "run_tag": tag,
        "command": command,
        "environment": {key: env[key] for key in (
            "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "MPLBACKEND",
            "WAVE_PUZZLE_FORECAST_MAX_PLANS"
        )},
        "limits": limits,
        "started_utc": started.isoformat(),
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "exit_code": exit_code,
        "termination_reason": reason,
        "result_dir": str(run_dir) if run_dir is not None else None,
        "log": str(log_path),
        "expected_jobs": len(expected),
        "input_hashes": data_hashes(case, jobs_dir),
        "source_hash": source_digest,
        "source_snapshot": str(campaign_dir / "source_snapshot"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "host": platform.node(),
        "python": platform.python_version(),
    }
    metadata = runs_dir / f"{tag}.json"
    if run_dir is not None:
        (run_dir / "evaluation_run.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    complete = False
    if run_dir is not None and reason is None and exit_code == 0:
        try:
            complete = bool(analyze_run(record)["complete"])
        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
            record["analysis_error"] = str(exc)
    record["status"] = "complete" if complete else "incomplete"
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if run_dir is not None:
        (run_dir / "evaluation_run.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    print(f"{case.name}: {record['status']}, reason={reason}, result={run_dir}, log={log_path}")
    if record["status"] != "complete":
        raise RuntimeError(f"Simulation incomplete for {case.name}; inspect {metadata}")
    return metadata


def analyze_run(record: dict) -> dict:
    case = Case(**record["case"])
    run_dir = Path(record["result_dir"])
    # Use the exact workload directory recorded for this run.
    command = record["command"]
    jobs_dir = Path(command[command.index("--jobs-dir") + 1])
    expected = expected_job_keys(jobs_dir)
    observed = output_job_keys(run_dir)
    if len(expected) != record["expected_jobs"]:
        raise ValueError(f"Input workload changed since {case.name} ran")
    current_hashes = data_hashes(case, jobs_dir)
    if current_hashes != record["input_hashes"]:
        raise ValueError(f"Inputs changed since {case.name} ran")
    if record["source_hash"] != source_hash():
        raise ValueError(f"Simulation source changed since {case.name} ran")
    metrics = extract_run_metrics(run_dir)
    energy = pd.read_csv(run_dir / "energy_simulation_result.csv")
    missing = expected - observed
    extra = observed - expected
    row = {
        "case": case.name,
        "group": case.group,
        "system": case.system,
        "policy": case.policy,
        "energy": case.energy,
        "workload": case.workload,
        "reference": case.reference,
        "result_dir": str(run_dir),
        "exit_code": record["exit_code"],
        "expected_jobs": len(expected),
        "completed_jobs": len(observed),
        "missing_jobs": len(missing),
        "extra_jobs": len(extra),
        "source_hash": record["source_hash"],
        "workload_hash": hash_files(workload_files(jobs_dir)),
        "wall_seconds": record["wall_seconds"],
        **asdict(metrics),
    }
    if "charge_level" in energy:
        charge = pd.to_numeric(energy["charge_level"], errors="coerce").dropna()
        row["battery_first_record_kwh"] = float(charge.iloc[0] / 1000)
        row["battery_final_kwh"] = float(charge.iloc[-1] / 1000)
    for column in energy:
        if column.endswith("_util"):
            row[f"mean_{column}"] = float(pd.to_numeric(energy[column], errors="coerce").mean())
    summary_path = run_dir / "puzzle_plan_summary.csv"
    if case.policy == "puzzle":
        if not summary_path.is_file():
            raise ValueError(f"Missing Puzzle plan summary in {run_dir}")
        plans = pd.read_csv(summary_path)
        ids = plans["plan_id"].astype(int).tolist()
        row["plan_count"] = len(plans)
        row["plan_ids_contiguous"] = ids == list(range(len(ids)))
        row["future_job_rows"] = int(plans["num_future_job_rows"].sum())
        durations = pd.to_numeric(plans["build_duration_s"], errors="raise")
        row["plan_build_total_s"] = float(durations.sum())
        row["plan_build_p95_s"] = float(durations.quantile(0.95))
        row["plan_build_max_s"] = float(durations.max())
    else:
        row["plan_count"] = 0
        row["plan_ids_contiguous"] = True
        row["future_job_rows"] = 0
        row["plan_build_total_s"] = 0.0
        row["plan_build_p95_s"] = 0.0
        row["plan_build_max_s"] = 0.0
    row["complete"] = (
        record["exit_code"] == 0
        and not missing
        and not extra
        and row["plan_ids_contiguous"]
        and row["future_job_rows"] == 0
    )
    return row


def collect(campaign_dir: Path) -> tuple[Path, Path]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((campaign_dir / "runs").glob("*.json"))
    ]
    if not records:
        raise ValueError(f"No campaign runs in {campaign_dir}")
    rows = []
    for record in records:
        try:
            row = analyze_run(record)
        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
            if record.get("status") != "incomplete":
                raise
            case = record["case"]
            row = {
                "case": case["name"], "group": case["group"],
                "system": case["system"], "policy": case["policy"],
                "energy": case["energy"], "workload": case["workload"],
                "reference": case["reference"],
                "result_dir": record.get("result_dir"),
                "exit_code": record["exit_code"],
                "expected_jobs": record["expected_jobs"],
                "wall_seconds": record["wall_seconds"],
                "complete": False,
                "analysis_error": str(exc),
            }
        row["termination_reason"] = record.get("termination_reason")
        row["run_status"] = record.get("status", "legacy")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["group", "case", "result_dir"])
    summary_path = campaign_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    by_case = {}
    for row in rows:
        if row["complete"]:
            if row["case"] in by_case:
                raise ValueError(f"Multiple completed runs for {row['case']}; use separate campaign directories")
            by_case[row["case"]] = row
    pairs = []
    for row in rows:
        reference = row["reference"]
        if not row["complete"] or not reference or reference not in by_case:
            continue
        base = by_case[reference]
        if (row["system"] != base["system"]
                or row["workload_hash"] != base["workload_hash"]
                or row["energy"] != base["energy"]
                or row["source_hash"] != base["source_hash"]):
            raise ValueError(f"Incompatible comparison: {row['case']} and {reference}")
        pairs.append({
            "case": row["case"],
            "reference": reference,
            "group": row["group"],
            "system": row["system"],
            "co2_delta_kg": row["co2_kg"] - base["co2_kg"],
            "co2_ratio": row["co2_kg"] / base["co2_kg"] if base["co2_kg"] else None,
            "physical_import_delta_kwh": (
                row["physical_grid_import_kwh"] - base["physical_grid_import_kwh"]
            ),
            "wait_p95_delta_h": row["p95_wait_h"] - base["p95_wait_h"],
            "makespan_delta_h": row["makespan_h"] - base["makespan_h"],
            "deadline_miss_delta": row["deadline_misses"] - base["deadline_misses"],
        })
    pairs_path = campaign_dir / "comparisons.csv"
    pd.DataFrame(pairs).to_csv(pairs_path, index=False)
    print(f"{len(rows)} runs, {len(pairs)} paired comparisons: {summary_path}, {pairs_path}")
    return summary_path, pairs_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("cases", nargs="+")
    run.add_argument("--timeout-s", type=float, default=None,
                     help="Maximum run time; default 1h for test, 3h for MACC")
    run.add_argument("--stall-s", type=float, default=1800,
                     help="Stop if the run writes no log output for this long")
    run.add_argument("--max-run-gb", type=float, default=3)
    run.add_argument("--max-campaign-gb", type=float, default=10)
    run.add_argument("--min-free-gb", type=float, default=10)
    sub.add_parser("collect")
    args = parser.parse_args(argv)
    cases = campaign_cases()
    if args.action == "list":
        for case in cases.values():
            print(f"{case.group:12} {case.name:28} {case.system:16} {case.reference or '-'}")
    elif args.action == "prepare":
        for name, path in prepare_workloads(args.campaign_dir).items():
            print(f"{name}: {path}, {len(expected_job_keys(path))} jobs")
    elif args.action == "run":
        for name in args.cases:
            if name not in cases:
                parser.error(f"Unknown case: {name}")
        for name in args.cases:
            run_case(
                cases[name], args.campaign_dir,
                timeout_s=args.timeout_s, stall_s=args.stall_s,
                max_run_gb=args.max_run_gb,
                max_campaign_gb=args.max_campaign_gb,
                min_free_gb=args.min_free_gb,
            )
    else:
        collect(args.campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
