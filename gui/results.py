"""Read generated runs and the published thesis CSV/PNG collection."""

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd


def locations(outputs: Path, thesis: Path) -> dict[str, tuple[Path, dict]]:
    found = {}
    if outputs.exists():
        for directory in outputs.glob("results_*"):
            config = directory / "config.json"
            if config.is_file():
                try:
                    found[directory.name] = (directory, json.loads(config.read_text()))
                except (ValueError, OSError):
                    continue
    data = thesis / "data"
    for monitor in sorted(data.rglob("energy_simulation_result.csv")):
        directory = monitor.parent
        parts = directory.relative_to(data).parts
        policy = parts[-1].rsplit("_", 1)[-1]
        if policy == "fixed":
            policy = parts[-1].split("_")[1]
        if parts[-1].startswith("battery_"):
            policy = "fifo"
        config = {
            "supercomputer": "deucalion_macc",
            "scheduling_policy": policy,
            "run_tag": " / ".join(parts),
            "published_result": True,
        }
        found["thesis-" + "--".join(parts)] = (directory, config)
    return found


def image_locations(directory: Path, thesis: Path) -> dict[str, Path]:
    data = thesis / "data"
    if not directory.is_relative_to(data):
        return {path.name: path for path in sorted(directory.glob("*.png"))}
    relative = directory.relative_to(data)
    plots = thesis / "plots"
    if relative.parts[0] == "workloads":
        picture = plots / relative.with_suffix(".png")
        candidates = [picture, plots / "workloads" / "overview.png"]
    elif relative.parts[0] == "carbon_threshold":
        candidates = [plots / "carbon_threshold.png"]
    else:
        candidates = sorted((plots / relative).glob("*.png"))
        if relative.parts[0] == "main":
            candidates += sorted((plots / "main").glob("*.png"))
            if relative.name in {"solar_wind_eft", "solar_wind_puzzle"}:
                candidates += sorted((plots / "main" / "eft_puzzle_solar_wind_comparison").glob("*.png"))
    return {path.name: path for path in candidates if path.is_file()}


def read_result(run_id: str, outputs: Path, thesis: Path) -> dict | None:
    match = locations(outputs, thesis).get(run_id)
    if match is None:
        return None
    return _read_result(run_id, match, thesis)


def _read_result(run_id: str, match: tuple[Path, dict], thesis: Path) -> dict:
    directory, config = match
    files = sorted(directory.glob("*.csv"))
    stamp = tuple((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files)
    result = _read_cached(str(directory), json.dumps(config, sort_keys=True), stamp)
    return {
        **result,
        "id": run_id,
        "label": ("Thesis / " + config["run_tag"]) if config.get("published_result") else config.get("run_tag", run_id),
        "source": "thesis" if config.get("published_result") else "simulation",
        "images": list(image_locations(directory, thesis)),
    }


def list_results(outputs: Path, thesis: Path) -> list[dict]:
    found = locations(outputs, thesis)
    return [_read_result(run_id, found[run_id], thesis)
            for run_id in sorted(found, key=lambda key: (not found[key][1].get("published_result", False), key), reverse=True)]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@lru_cache(maxsize=128)
def _read_cached(directory: str, config_json: str, stamp: tuple) -> dict:
    root = Path(directory)
    config = json.loads(config_json)
    summary = {}
    frames = [_read_csv(path) for path in sorted(root.glob("jobs_metrics_*.csv"))]
    jobs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary["total_jobs_completed"] = len(jobs)
    if not jobs.empty:
        waits = pd.to_numeric(jobs["Wait_Time_s"], errors="coerce")
        summary["avg_wait_time_s"] = float(waits.mean())
        summary["p95_wait_time_s"] = float(waits.quantile(0.95))
        summary["avg_stretch_factor"] = float(pd.to_numeric(jobs["Stretch_Factor"], errors="coerce").mean())
        if "Queue_Deadline_Missed" in jobs:
            summary["deadline_misses"] = int(jobs["Queue_Deadline_Missed"].astype(str).str.lower().isin({"true", "1"}).sum())
    monitor = root / "energy_simulation_result.csv"
    if monitor.is_file():
        energy = _read_csv(monitor, index_col=0)
        if not energy.empty:
            exchange = pd.to_numeric(energy["e_delta"], errors="coerce").fillna(0)
            imports_wh = -exchange.clip(upper=0) / 3600
            grid_fraction = imports_wh / (imports_wh + 0.001)
            summary["total_co2_kg"] = float((pd.to_numeric(energy["co2"], errors="coerce").fillna(0) * grid_fraction).sum() / 1000)
            summary["physical_grid_import_kwh"] = float(imports_wh.sum() / 1000)
            columns = [name for name in energy if name.endswith("_power")]
            times = pd.to_datetime(energy.index, format="mixed")
            default_step = float((times[1] - times[0]).total_seconds()) if len(times) > 1 else 60.0
            step = float(config.get("microgrid_step_size_s", default_step))
            if columns:
                summary["total_energy_kwh"] = float(energy[columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum() * step / 3_600_000)
    carbon = root / "carbon_metrics.csv"
    if carbon.is_file():
        frame = _read_csv(carbon)
        if not frame.empty and "total_energy_kwh" in frame:
            summary["total_energy_kwh"] = float(frame["total_energy_kwh"].iloc[-1])
    return {
        "timestamp": config.get("timestamp", ""),
        "supercomputer": config.get("supercomputer", "unknown"),
        "scheduling_policy": config.get("scheduling_policy", ""),
        "dvfs_policy": config.get("dvfs_policy", ""),
        "budget_policy": config.get("budget_policy", ""),
        "alloc_strategy": config.get("alloc_strategy", ""),
        "config": config,
        "csvs": sorted(path.name for path in root.glob("*.csv")),
        "summary_metrics": {key: value if math.isfinite(value) else None
                            for key, value in summary.items()},
    }
