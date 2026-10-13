"""Run Chapter 5 carbon-threshold and energy-policy sensitivity experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs" / "thesis_sensitivity_2026-09-28"
FINAL = ROOT / "results" / "results_thesis_final"
WORKLOAD_NAME = "macc_sparse_early_tail"
TIMEOUT_S = 900.0
MAX_RUN_BYTES = 2 * 1024**3
MAX_CAMPAIGN_BYTES = 8 * 1024**3
MIN_FREE_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class Case:
    case_id: str
    experiment: str
    scheduler: str
    budget_policy: str
    threshold_g_per_kwh: float | None = None
    energy_alias: str | None = None


def matrix() -> list[Case]:
    cases = []
    for threshold in (200, 225, 250, 275, 300):
        for scheduler in ("fifo", "priority", "eft", "puzzle"):
            cases.append(Case(
                f"ct{threshold}_{scheduler}", "carbon_threshold", scheduler,
                "puzzle" if scheduler == "puzzle" else "carbon-minimizing",
                float(threshold),
                f"ep_carbon_{scheduler}" if threshold == 250 and scheduler != "puzzle" else None,
            ))
    for policy, prefix in (("grid", "grid"), ("daytime-renewable", "daytime")):
        for scheduler in ("fifo", "priority", "eft"):
            cases.append(Case(
                f"ep_{prefix}_{scheduler}", "energy_policy", scheduler, policy,
            ))
    if len(cases) != 26 or len({case.case_id for case in cases}) != 26:
        raise AssertionError("The campaign matrix must contain exactly 26 unique cases")
    by_id = {case.case_id: case for case in cases}
    execution_order = ["ct250_fifo", "ct250_priority", "ct250_eft", "ct250_puzzle"]
    execution_order += [
        f"ep_{policy}_{scheduler}"
        for policy in ("grid", "daytime")
        for scheduler in ("fifo", "priority", "eft")
    ]
    execution_order += [
        f"ct{threshold}_{scheduler}"
        for threshold in (275, 300, 225, 200)
        for scheduler in ("fifo", "priority", "eft", "puzzle")
    ]
    return [by_id[case_id] for case_id in execution_order]


CASES = matrix()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def physical_inputs() -> list[Path]:
    paths = [
        ROOT / "datasets" / name
        for name in (
            "historical_data.csv", "pv_gen.csv", "wind_gen.csv",
            "node_config.json", "renewable_sources.json", "supercomputer_config.py",
        )
    ]
    paths.append(ROOT / "datasets" / "supercomputers" / "deucalion_macc.json")
    return paths


def workload_files(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("jobs_*.csv"))
    if len(paths) != 9:
        raise ValueError(f"Expected nine workload files in {directory}, found {len(paths)}")
    return paths


def job_keys(directory: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for path in workload_files(directory):
        cluster = path.stem.removeprefix("jobs_").rsplit("_", 1)[0]
        jobs = pd.read_csv(path, usecols=["job_id"])
        local = {(cluster, int(job_id)) for job_id in jobs["job_id"]}
        if len(local) != len(jobs) or keys.intersection(local):
            raise ValueError(f"Duplicate job key in {path}")
        keys.update(local)
    if len(keys) != 2523:
        raise ValueError(f"Expected 2,523 distinct jobs, found {len(keys)}")
    return keys


def hash_files(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve().relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def source_files() -> list[Path]:
    snapshot = FINAL / "source_snapshot"
    return sorted([* (snapshot / "vessim").rglob("*.py"),
                   snapshot / "results" / "simulation_energy.py"])


def prepare() -> None:
    if CAMPAIGN.exists():
        raise FileExistsError(f"Campaign directory already exists: {CAMPAIGN}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"At least 10 GiB free space required; available {free_bytes / 1024**3:.2f} GiB")

    source_jobs = FINAL / "inputs" / WORKLOAD_NAME
    workload_files(source_jobs)
    expected = job_keys(source_jobs)
    sources = physical_inputs()
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("A physical input required by the campaign is missing")

    final_records = [json.loads(path.read_text(encoding="utf-8"))
                     for path in (FINAL / "runs").glob("*.json")]
    reference = next(row for row in final_records
                     if row.get("case", {}).get("name") == "solar_wind_fifo")
    for label, recorded_digest in reference["input_hashes"].items():
        if label.startswith("datasets/"):
            path = ROOT / label
            if not path.is_file() or sha256(path) != recorded_digest:
                raise ValueError(f"Physical input differs from final reference: {label}")

    snapshot = FINAL / "source_snapshot"
    reference_source_hash = snapshot.joinpath("source_hash.txt").read_text().strip()
    if reference["source_hash"] != reference_source_hash:
        raise ValueError("Final workload run and source snapshot hashes differ")
    frozen_python = source_files()
    if len(frozen_python) != 50:
        raise ValueError(f"Expected 50 Python runtime files, found {len(frozen_python)}")
    for path in frozen_python:
        relative = path.relative_to(snapshot)
        if relative == Path("results/simulation_energy.py"):
            relative = Path("scripts/simulation_energy.py")
        active = ROOT / relative
        if not active.is_file() or sha256(path) != sha256(active):
            raise ValueError(f"Active simulator differs from final snapshot: {path}")

    CAMPAIGN.mkdir(parents=True)
    shutil.copytree(source_jobs, CAMPAIGN / "inputs" / WORKLOAD_NAME)
    derivation = FINAL / "inputs" / "derivation.json"
    shutil.copy2(derivation, CAMPAIGN / "inputs" / "derivation.json")
    datasets = CAMPAIGN / "inputs" / "datasets"
    for source in sources:
        target = datasets / source.relative_to(ROOT / "datasets")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(snapshot, CAMPAIGN / "source_snapshot")

    # These files control cluster/queue construction. Freeze copies beside the
    # runtime. Time-series profiles are loaded from the campaign input copy.
    runtime_datasets = CAMPAIGN / "source_snapshot" / "datasets"
    for source in (
        ROOT / "datasets" / "node_config.json",
        ROOT / "datasets" / "renewable_sources.json",
        ROOT / "datasets" / "supercomputer_config.py",
        ROOT / "datasets" / "supercomputers" / "deucalion_macc.json",
    ):
        target = runtime_datasets / source.relative_to(ROOT / "datasets")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    runner = CAMPAIGN / "runner"
    runner.mkdir()
    shutil.copy2(Path(__file__), runner / Path(__file__).name)
    (runner / "simulation_entry.py").write_text(
        '"""Run from the frozen campaign source tree."""\n'
        "from __future__ import annotations\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "CAMPAIGN = Path(__file__).resolve().parents[1]\n"
        "FROZEN = CAMPAIGN / 'source_snapshot'\n"
        "sys.path.insert(0, str(FROZEN))\n"
        "from results import simulation_energy\n"
        "_build_original = simulation_energy.build_environment\n"
        "def _build(args, config, selected_system, **kwargs):\n"
        "    return _build_original(args, config, selected_system,\n"
        "        root_dir=os.environ['CAMPAIGN_SIM_ROOT'],\n"
        "        datasets_dir=str(CAMPAIGN / 'inputs' / 'datasets'), **kwargs)\n"
        "simulation_energy.build_environment = _build\n"
        "raise SystemExit(simulation_energy.main())\n",
        encoding="utf-8",
    )

    hashes = hash_files([*sources, derivation, *workload_files(CAMPAIGN / "inputs" / WORKLOAD_NAME)])
    runtime_hashes = {
        str(path.relative_to(snapshot)): sha256(path)
        for path in frozen_python
    }
    atomic_json(CAMPAIGN / "source_snapshot" / "runtime_hashes.json", {
        "source_hash": reference_source_hash, "files": runtime_hashes,
    })
    case_rows = [asdict(case) for case in CASES]
    atomic_json(CAMPAIGN / "manifest.json", {
        "created_utc": utc_now(),
        "timeout_s": TIMEOUT_S,
        "expected_job_count": len(expected),
        "input_hashes": hashes,
        "runtime_source_hash": reference_source_hash,
        "cases": case_rows,
        "shared_energy_policy_aliases": {
            case.energy_alias: case.case_id for case in CASES if case.energy_alias
        },
    })
    pd.DataFrame(case_rows).to_csv(CAMPAIGN / "manifest.csv", index=False)
    atomic_json(CAMPAIGN / "progress.json", {
        case.case_id: {"status": "pending"} for case in CASES
    })
    (CAMPAIGN / "analysis").mkdir()
    (CAMPAIGN / "figures").mkdir()
    execution_plan = ROOT / "docs" / "research-notes" / "thesis-sensitivity-execution-plan.md"
    if execution_plan.is_file():
        shutil.copy2(execution_plan, CAMPAIGN / "execution_plan.md")
    (CAMPAIGN / "README.md").write_text(
        "# Thesis sensitivity campaign\n\n"
        "Twenty-six unique carbon-threshold and energy-policy cases for Chapter 5. "
        "The simulation process group is killed at 900 wall seconds. "
        "See manifest.json, progress.json, and analysis/ for provenance and outputs.\n",
        encoding="utf-8",
    )
    atomic_json(CAMPAIGN / "preflight.json", {
        "status": "passed", "checked_utc": utc_now(),
        "job_count": len(expected), "workload_files": len(workload_files(source_jobs)),
        "physical_input_hashes_match_final_reference": True,
        "runtime_files_match_final_snapshot": len(frozen_python),
        "runtime_source_hash": reference_source_hash,
        "free_disk_bytes_at_prepare": free_bytes,
        "python": sys.version, "platform": platform.platform(),
    })


def command_for(case: Case, attempt: Path) -> list[str]:
    entry = CAMPAIGN / "runner" / "simulation_entry.py"
    command = [
        sys.executable, "-u", str(entry), "deucalion_macc",
        "--no-plots", "--no-snapshot", "--scheduler-debug-log",
        "--no-grid-charging", "--no-sla-grid-override", "--sla-policy", "enforce",
        "--pue", "1.3", "--grid-power-limit", "350000",
        "--renewable-charge-fraction", "0.3",
        "--renewable-sources", "pv,wind",
        "--renewable-source-scale", "pv=400000",
        "--renewable-source-scale", "wind=400000",
        "--battery-capacity-kwh", "2000",
        "--battery-soc-min", "0.2", "--battery-soc-max", "0.8",
        "--battery-p-charge-max", "500", "--battery-p-discharge-max", "500",
        "--energy-scheduler-step-size", "300", "--microgrid-step-size", "60",
        "--runtime-prediction-error-rate", "0", "--runtime-prediction-seed", "42",
        "--start-nodes-on", "--keep-idle-nodes-on",
        "--jobs-dir", str((CAMPAIGN / "inputs" / WORKLOAD_NAME).resolve()),
        "--run-tag", f"thesis_sensitivity_{case.case_id}_{attempt.name}",
    ]
    if case.budget_policy == "grid":
        command += ["--budget-policy", "grid", "--allocation-strategy", "compute-prioritized"]
    elif case.budget_policy == "daytime-renewable":
        command += [
            "--budget-policy", "daytime-renewable",
            "--allocation-strategy", "compute-prioritized",
            "--day-grid-fraction", "0.5", "--night-grid-fraction", "1.0",
        ]
    elif case.budget_policy == "carbon-minimizing":
        command += [
            "--budget-policy", "carbon-minimizing",
            "--allocation-strategy", "compute-prioritized",
            "--carbon-grid-threshold", str(case.threshold_g_per_kwh),
            "--low-carbon-grid-fraction", "1.0", "--high-carbon-grid-fraction", "0.0",
        ]
    else:
        command += [
            "--budget-policy", "puzzle", "--allocation-strategy", "puzzle-plan",
            "--carbon-grid-threshold", str(case.threshold_g_per_kwh),
        ]
    if case.scheduler == "eft":
        command += [
            "--scheduling-policy", "eft-dvfs", "--dvfs-policy", "adaptive",
            "--eft-dvfs-alpha", "0.1", "--eft-dvfs-beta", "0.9",
        ]
    else:
        command += ["--scheduling-policy", case.scheduler, "--dvfs-policy", "adaptive"]
    return command


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc_root = Path("/proc")
    if proc_root.is_dir():
        try:
            for stat_path in proc_root.glob("[0-9]*/stat"):
                try:
                    fields = stat_path.read_text(encoding="ascii").rsplit(")", 1)[1].split()
                    state, process_group = fields[0], int(fields[2])
                except (IndexError, OSError, ValueError):
                    continue
                if process_group == pgid and state not in {"Z", "X"}:
                    return True
            return False
        except OSError:
            pass
    return True


def kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def guarded_process(command: list[str], log_path: Path, env: dict[str, str],
                    sim_root: Path, timeout_s: float = TIMEOUT_S) -> dict:
    if timeout_s <= 0 or timeout_s > TIMEOUT_S:
        raise ValueError(f"Timeout must be in (0, {TIMEOUT_S}] seconds")
    sim_root.mkdir(parents=True, exist_ok=True)
    child_env = dict(env)
    child_env["CAMPAIGN_SIM_ROOT"] = str(sim_root.resolve())
    started_wall = datetime.now(timezone.utc)
    start_mono = time.monotonic()
    deadline = start_mono + timeout_s
    reason = None
    signal_mono = None
    stop_watchdog = threading.Event()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def enforce_deadline() -> None:
            nonlocal reason, signal_mono
            if stop_watchdog.wait(max(0.0, deadline - time.monotonic())):
                return
            if group_alive(process.pid):
                reason = "wall_timeout"
                signal_mono = time.monotonic()
                kill_group(process.pid)

        watchdog = threading.Thread(target=enforce_deadline, daemon=True)
        watchdog.start()
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if group_alive(process.pid):
                    reason = "wall_timeout"
                    signal_mono = time.monotonic()
                    kill_group(process.pid)
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            process.wait()
        stop_watchdog.set()
        watchdog.join()
        # The leader exiting must not leave any child in the simulation group.
        if group_alive(process.pid):
            reason = reason or "orphan_processes"
            signal_mono = signal_mono or time.monotonic()
            kill_group(process.pid)
        # SIGKILL delivery is asynchronous. Give descendants time to exit
        # before reporting whether any executable process remains in the group.
        cleanup_deadline = time.monotonic() + 1.0
        while group_alive(process.pid) and time.monotonic() < cleanup_deadline:
            time.sleep(0.01)
    end_mono = time.monotonic()
    if end_mono - start_mono >= timeout_s:
        reason = reason or "wall_timeout"
    return {
        "started_utc": started_wall.isoformat(),
        "ended_utc": utc_now(),
        "wall_seconds": end_mono - start_mono,
        "exit_code": process.returncode,
        "termination_reason": reason,
        "watchdog_signal_latency_s": (
            max(0.0, signal_mono - deadline) if signal_mono is not None else None
        ),
        "process_group_empty": not group_alive(process.pid),
        "wall_timeout_s": timeout_s,
    }


def directory_bytes(path: Path) -> int:
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except FileNotFoundError:
                continue
    return total


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "WAVE_PUZZLE_FORECAST_MAX_PLANS": "8",
    })
    return env


def update_progress(case_id: str, record: dict) -> None:
    path = CAMPAIGN / "progress.json"
    progress = json.loads(path.read_text(encoding="utf-8"))
    progress[case_id] = record
    atomic_json(path, progress)


def validate_run(record: dict, result: Path, case: Case) -> None:
    config = json.loads((result / "config.json").read_text(encoding="utf-8"))
    if config["renewable_sources"] != ["pv", "wind"]:
        raise ValueError("Run did not use PV and wind")
    if config["renewable_source_scales_w"] != {"pv": 400000.0, "wind": 400000.0}:
        raise ValueError("Renewable scales differ from manifest")
    if config["sla_grid_override"] or config["runtime_prediction_error_rate"] != 0:
        raise ValueError("SLA override or runtime prediction settings differ from manifest")
    if config["budget_policy_key"] != case.budget_policy:
        raise ValueError("Budget policy differs from manifest")
    expected = job_keys(CAMPAIGN / "inputs" / WORKLOAD_NAME)
    observed: set[tuple[str, int]] = set()
    frames = []
    paths = sorted(result.glob("jobs_metrics_*.csv"))
    if len(paths) != 3:
        raise ValueError(f"Expected three jobs metrics files, found {len(paths)}")
    for path in paths:
        cluster = path.stem.removeprefix("jobs_metrics_")
        frame = pd.read_csv(path)
        keys = {(cluster, int(value)) for value in frame["Job_ID"]}
        if observed.intersection(keys):
            raise ValueError(f"Duplicate job output in {path}")
        observed.update(keys)
        frames.append(frame)
    if observed != expected:
        raise ValueError(f"Job mismatch: missing {len(expected-observed)}, extra {len(observed-expected)}")
    energy = pd.read_csv(result / "energy_simulation_result.csv")
    if energy.empty or not {"e_delta", "co2"}.issubset(energy.columns):
        raise ValueError("Missing microgrid monitor data")
    jobs = pd.concat(frames, ignore_index=True)
    record["expected_jobs"] = len(expected)
    record["completed_jobs"] = len(observed)
    record["missing_jobs"] = len(expected - observed)
    record["extra_jobs"] = len(observed - expected)
    record["deadline_misses"] = int(
        jobs["Queue_Deadline_Missed"].astype(str).str.lower().isin({"true", "1"}).sum()
    )
    record["mean_wait_h"] = float(pd.to_numeric(jobs["Wait_Time_s"]).mean() / 3600)
    record["p95_wait_h"] = float(pd.to_numeric(jobs["Wait_Time_s"]).quantile(.95) / 3600)
    if case.scheduler == "puzzle":
        plans = pd.read_csv(result / "puzzle_plan_summary.csv")
        if int(plans["num_future_job_rows"].sum()) != 0:
            raise ValueError("Puzzle plan export contains future job rows")
        record["plan_count"] = len(plans)
        for name, column in (
            ("puzzle_low_carbon_plan_occurrences", "num_low_carbon_grid"),
            ("puzzle_carbon_regret_plan_occurrences", "num_carbon_regret_grid"),
            ("puzzle_sla_emergency_plan_occurrences", "num_sla_emergency_grid"),
        ):
            record[name] = int(plans[column].sum())
    record["config"] = config


def run_one(case: Case) -> dict:
    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("Less than 10 GiB free; stopping before next run")
    if directory_bytes(CAMPAIGN) >= MAX_CAMPAIGN_BYTES:
        raise RuntimeError("Campaign reached its 8 GiB size limit")
    case_root = CAMPAIGN / "runs" / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    attempt = case_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt.mkdir()
    (attempt / "results").mkdir()
    argv = command_for(case, attempt)
    env = environment()
    record = {
        "case": asdict(case), "status": "running", "command": argv,
        "environment": {key: env[key] for key in (
            "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "MPLBACKEND",
            "WAVE_PUZZLE_FORECAST_MAX_PLANS",
        )},
        "input_hashes": json.loads((CAMPAIGN / "manifest.json").read_text())["input_hashes"],
        "runtime_source_hash": json.loads((CAMPAIGN / "manifest.json").read_text())["runtime_source_hash"],
        "started_utc": utc_now(), "attempt_dir": str(attempt.resolve()),
        "result_dir": None, "wall_timeout_s": TIMEOUT_S,
    }
    atomic_json(attempt / "command.json", {
        "argv": argv, "cwd": str(ROOT), "environment": record["environment"],
        "timeout_s": TIMEOUT_S,
    })
    atomic_json(attempt / "run.json", record)
    update_progress(case.case_id, {"status": "running", "attempt_dir": str(attempt.resolve())})
    outcome = guarded_process(argv, attempt / "stdout.log", env, attempt / "simulator_root")
    generated = attempt / "simulator_root" / "results"
    children = sorted(path for path in generated.iterdir() if path.is_dir()) if generated.exists() else []
    result = None
    if children:
        if len(children) != 1:
            outcome["termination_reason"] = outcome["termination_reason"] or "multiple_result_directories"
        else:
            result = attempt / "results" / children[0].name
            shutil.move(str(children[0]), result)
            record["result_dir"] = str(result.resolve())
    record.update(outcome)
    record["result_bytes"] = directory_bytes(result) if result else 0
    record["attempt_bytes"] = directory_bytes(attempt)
    record["status"] = (
        "complete" if outcome["exit_code"] == 0
        and outcome["termination_reason"] is None
        and outcome["process_group_empty"] else "incomplete"
    )
    if record["result_bytes"] > MAX_RUN_BYTES:
        record["status"] = "incomplete"
        record["termination_reason"] = "run_size_limit"
    if record["status"] == "complete" and result:
        try:
            validate_run(record, result, case)
        except Exception as exc:
            record["status"] = "incomplete"
            record["validation_error"] = f"{type(exc).__name__}: {exc}"
    record["ended_utc"] = utc_now()
    atomic_json(attempt / "run.json", record)
    if result:
        atomic_json(result / "evaluation_run.json", record)
    update_progress(case.case_id, {
        "status": record["status"], "attempt_dir": str(attempt.resolve()),
        "result_dir": record["result_dir"], "ended_utc": record["ended_utc"],
        "wall_seconds": record["wall_seconds"],
        "termination_reason": record["termination_reason"],
        "completed_jobs": record.get("completed_jobs", 0),
        "deadline_misses": record.get("deadline_misses"),
    })
    return record


def run_campaign() -> None:
    failed = []
    progress_path = CAMPAIGN / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    for number, case in enumerate(CASES, start=1):
        if progress[case.case_id]["status"] != "pending":
            continue
        print(f"[{number}/26] {case.case_id}", flush=True)
        try:
            record = run_one(case)
            print(
                f"  {record['status']} in {record['wall_seconds']:.1f}s; "
                f"jobs={record.get('completed_jobs', 0)}; "
                f"reason={record['termination_reason']}",
                flush=True,
            )
            if record["status"] != "complete":
                failed.append(case.case_id)
        except RuntimeError as exc:
            if "10 GiB" in str(exc) or "8 GiB" in str(exc):
                raise
            update_progress(case.case_id, {"status": "failed", "reason": str(exc)})
            failed.append(case.case_id)
    print(f"Campaign cases incomplete or failed: {failed}", flush=True)


def collect() -> list[dict]:
    rows = []
    for case in CASES:
        case_root = CAMPAIGN / "runs" / case.case_id
        attempts = sorted(case_root.glob("*/run.json")) if case_root.exists() else []
        if not attempts:
            progress = json.loads((CAMPAIGN / "progress.json").read_text(encoding="utf-8"))
            state = progress[case.case_id]
            rows.append({
                "case": case.case_id, "experiment": case.experiment,
                "scheduler": case.scheduler, "budget_policy": case.budget_policy,
                "threshold_g_per_kwh": case.threshold_g_per_kwh,
                "status": state.get("status", "missing"),
                "termination_reason": state.get("reason"),
                "completed_jobs": 0, "expected_jobs": 2523,
                "missing_jobs": None, "extra_jobs": None,
                "result_dir": state.get("result_dir"),
            })
        for metadata in attempts:
            record = json.loads(metadata.read_text(encoding="utf-8"))
            row = {
                "case": case.case_id, "experiment": case.experiment,
                "scheduler": case.scheduler, "budget_policy": case.budget_policy,
                "threshold_g_per_kwh": case.threshold_g_per_kwh,
                "status": record.get("status"),
                "wall_seconds": record.get("wall_seconds"),
                "exit_code": record.get("exit_code"),
                "termination_reason": record.get("termination_reason"),
                "expected_jobs": record.get("expected_jobs", 2523),
                "completed_jobs": record.get("completed_jobs", 0),
                "missing_jobs": record.get("missing_jobs"),
                "extra_jobs": record.get("extra_jobs"),
                "deadline_misses": record.get("deadline_misses"),
                "result_dir": record.get("result_dir"),
            }
            if record.get("status") == "complete" and record.get("result_dir"):
                result = Path(record["result_dir"])
                energy = pd.read_csv(result / "energy_simulation_result.csv")
                e_delta = pd.to_numeric(energy["e_delta"], errors="raise")
                monitor_co2 = pd.to_numeric(energy["co2"], errors="raise")
                import_ws = -e_delta.clip(upper=0)
                import_wh = import_ws / 3600.0
                grid_fraction = import_wh / (import_wh + 0.001)
                operational_co2_kg = float((monitor_co2 * grid_fraction).sum() / 1000.0)
                total_import_ws = float(import_ws.sum())
                jobs = pd.concat(
                    [pd.read_csv(path) for path in sorted(result.glob("jobs_metrics_*.csv"))],
                    ignore_index=True,
                )
                wait = pd.to_numeric(jobs["Wait_Time_s"], errors="raise")
                runtime = pd.to_numeric(jobs["Real_Exec_Time_s"], errors="raise")
                submitted = pd.to_datetime(jobs["Submission_Time"], format="mixed")
                config = json.loads((result / "config.json").read_text(encoding="utf-8"))
                finish = submitted + pd.to_timedelta(wait + runtime, unit="s")
                row.update({
                    "co2_kg": operational_co2_kg,
                    "physical_grid_import_kwh": float(total_import_ws / 3_600_000),
                    "effective_grid_g_per_kwh": (
                        operational_co2_kg * 1_000_000 / (total_import_ws / 3_600_000)
                        if total_import_ws > 0 else None
                    ),
                    "it_energy_kwh": float(
                        energy.filter(regex="_energy_kwh$").iloc[-1].sum()
                    ),
                    "mean_wait_h": float(wait.mean() / 3600),
                    "p95_wait_h": float(wait.quantile(.95) / 3600),
                    "makespan_h": float(
                        (finish.max() - pd.Timestamp(config["sim_start"])).total_seconds() / 3600
                    ),
                })
                if case.scheduler == "puzzle":
                    plans = pd.read_csv(result / "puzzle_plan_summary.csv")
                    for key, column in (
                        ("puzzle_low_carbon_plan_occurrences", "num_low_carbon_grid"),
                        ("puzzle_carbon_regret_plan_occurrences", "num_carbon_regret_grid"),
                        ("puzzle_sla_emergency_plan_occurrences", "num_sla_emergency_grid"),
                    ):
                        row[key] = int(plans[column].sum())
            rows.append(row)
    analysis = CAMPAIGN / "analysis"
    frame = pd.DataFrame(rows)
    frame.to_csv(analysis / "run_status.csv", index=False)
    complete = frame.loc[frame["status"] == "complete"].copy()
    complete.to_csv(analysis / "summary.csv", index=False)
    complete.loc[complete["experiment"] == "carbon_threshold"].to_csv(
        analysis / "carbon_threshold_sweep.csv", index=False
    )
    by_id = {row["case"]: row for row in complete.to_dict("records")}
    energy_rows = []
    for case in CASES:
        if case.experiment == "energy_policy":
            energy_rows.append(by_id.get(case.case_id, {"case": case.case_id, "status": "missing"}))
        elif case.energy_alias:
            row = dict(by_id.get(case.case_id, {"case": case.energy_alias, "status": "missing"}))
            row["case"] = case.energy_alias
            row["source_case"] = case.case_id
            energy_rows.append(row)
    pd.DataFrame(energy_rows).to_csv(analysis / "energy_policy_comparison.csv", index=False)
    paired = []
    for case in CASES:
        if case.experiment != "carbon_threshold" or case.threshold_g_per_kwh == 250:
            continue
        left, reference = by_id.get(case.case_id), by_id.get(f"ct250_{case.scheduler}")
        if left and reference:
            paired.append({
                "case": case.case_id, "reference": f"ct250_{case.scheduler}",
                "co2_kg_delta": left["co2_kg"] - reference["co2_kg"],
                "grid_import_kwh_delta": left["physical_grid_import_kwh"] - reference["physical_grid_import_kwh"],
                "p95_wait_h_delta": left["p95_wait_h"] - reference["p95_wait_h"],
                "makespan_h_delta": left["makespan_h"] - reference["makespan_h"],
            })
    pd.DataFrame(paired).to_csv(analysis / "paired_comparisons.csv", index=False)
    complete.loc[complete["scheduler"] == "puzzle"].to_csv(
        analysis / "puzzle_grid_reasons.csv", index=False
    )
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text(encoding="utf-8"))
    atomic_json(analysis / "analysis_evidence.json", {
        "created_utc": utc_now(),
        "input_hashes": manifest["input_hashes"],
        "runtime_source_hash": manifest["runtime_source_hash"],
        "runner_hashes": json.loads((CAMPAIGN / "runner" / "runner_hashes.json").read_text(encoding="utf-8")),
        "co2_formula": "sum each monitor row co2 * import_Wh/(import_Wh+0.001), then convert g to kg",
        "puzzle_reason_columns_are": "plan occurrences across replans, not unique jobs or measured grid energy",
        "grid_import_formula": "sum(max(-e_delta, 0)) / 3,600,000, in kWh",
        "effective_intensity": "operational co2 kg * 1,000,000 / grid import kWh",
        "wall_timeout_s": TIMEOUT_S,
        "complete_cases": len(complete),
        "incomplete_cases": sorted(frame.loc[frame["status"] != "complete", "case"].tolist()),
    })
    return rows


def make_figure() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-thesis")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for face in ("NewsGotT.ttf", "NewsGotTBold.ttf", "NewsGotTLight.ttf"):
        font_manager.fontManager.addfont(str(ROOT / "thesis" / "fonts" / face))
    plt.rcParams.update({"font.family": "NewsGotT", "font.size": 9, "pdf.fonttype": 42})
    summary = pd.read_csv(CAMPAIGN / "analysis" / "summary.csv")
    palette = {"fifo": "#2f70a9", "priority": "#d18a23",
               "eft": "#39885f", "puzzle": "#98509a"}
    labels = {"fifo": "FIFO", "priority": "Priority", "eft": "EFT", "puzzle": "Puzzle"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for scheduler in ("fifo", "priority", "eft", "puzzle"):
        rows = summary.loc[
            (summary["experiment"] == "carbon_threshold")
            & (summary["scheduler"] == scheduler)
        ].sort_values("threshold_g_per_kwh")
        if rows.empty:
            continue
        axes[0].plot(rows["threshold_g_per_kwh"], rows["co2_kg"], marker="o",
                     color=palette[scheduler], label=labels[scheduler])
        axes[1].plot(rows["threshold_g_per_kwh"], rows["p95_wait_h"], marker="o",
                     color=palette[scheduler], label=labels[scheduler])
    axes[0].set(xlabel="Carbon threshold (gCO$_2$/kWh)", ylabel="Operational CO$_2$ (kg)")
    axes[1].set(xlabel="Carbon threshold (gCO$_2$/kWh)", ylabel="P95 wait (h)")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(frameon=False, fontsize=8)
        axis.tick_params(labelsize=8)
    output = CAMPAIGN / "figures" / "carbon_threshold_sensitivity.pdf"
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    thesis_output = ROOT / "thesis" / "figures" / "carbon_threshold_sensitivity.pdf"
    shutil.copy2(output, thesis_output)
    atomic_json(ROOT / "thesis" / "figures" / "carbon_threshold_sensitivity_evidence.json", {
        "data": str((CAMPAIGN / "analysis" / "carbon_threshold_sweep.csv").relative_to(ROOT)),
        "generation_command": "venv/bin/python -B scripts/evaluation_sensitivity.py figures",
        "campaign_figure": str(output.relative_to(ROOT)),
        "sha256": sha256(thesis_output),
        "thresholds_g_per_kwh": [200, 225, 250, 275, 300],
        "profile": {"pv_w": 400000, "wind_w": 400000},
        "missing_points_are_not_interpolated": True,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "collect", "figures", "all"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "run":
        run_campaign()
    elif action == "collect":
        collect()
    elif action == "figures":
        make_figure()
    else:
        prepare()
        run_campaign()
        collect()
        make_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
