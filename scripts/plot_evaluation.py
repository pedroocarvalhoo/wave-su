"""Create Chapter 5 figures from a collected evaluation campaign.

Usage:
    python scripts/plot_evaluation.py overview results/evaluation_campaign/v4
    python scripts/plot_evaluation.py trace RUN_DIR [RUN_DIR ...] --output trace.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(path)


def overview(campaign_dir: Path) -> None:
    summary = pd.read_csv(campaign_dir / "summary.csv")
    summary = summary[summary["complete"].astype(str).str.lower() == "true"]
    if summary.empty:
        raise ValueError("No complete runs in summary.csv")
    output = campaign_dir / "figures"
    for system, rows in summary.groupby("system"):
        rows = rows[rows["group"].isin(["main_test", "main_macc"])]
        if rows.empty:
            continue
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.scatter(rows["co2_kg"], rows["p95_wait_h"], s=65)
        for _, row in rows.iterrows():
            axis.annotate(row["policy"], (row["co2_kg"], row["p95_wait_h"]),
                          xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.set(xlabel="Physical grid-import emissions (kg CO2)",
                 ylabel="95th-percentile queue wait (h)",
                 title=f"{system}: complete policy configurations")
        axis.grid(alpha=0.25)
        _save(figure, output / f"service_carbon_{system}.png")
    profiles = summary[
        (summary["system"] == "deucalion_test")
        & (summary["policy"].isin(["eft_carbon", "puzzle"]))
        & (summary["workload"] == "original")
        & (summary["group"].isin(["main_test", "energy"]))
        & (summary["case"] != "puzzle_no_battery")
    ]
    order = ["pv10", "pv20", "pv40", "pv_wind", "wind20"]
    if set(order).issubset(set(profiles["energy"])):
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for policy, label in (("eft_carbon", "EFT"), ("puzzle", "Puzzle")):
            rows = profiles[profiles["policy"] == policy].set_index("energy").reindex(order)
            if rows["co2_kg"].isna().any():
                continue
            axes[0].plot(order, rows["co2_kg"], marker="o", label=label)
            axes[1].plot(order, rows["p95_wait_h"], marker="o", label=label)
        axes[0].set_ylabel("Grid-import emissions (kg CO2)")
        axes[1].set_ylabel("95th-percentile wait (h)")
        axes[1].set_xlabel("Renewable scenario")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        _save(figure, output / "energy_profile_sensitivity.png")
    comparisons_path = campaign_dir / "comparisons.csv"
    if comparisons_path.stat().st_size == 0:
        return
    try:
        pairs = pd.read_csv(comparisons_path)
    except pd.errors.EmptyDataError:
        return
    if pairs.empty:
        return
    for group, rows in pairs.groupby("group"):
        figure, axis = plt.subplots(figsize=(7, max(3, 0.42 * len(rows) + 1.5)))
        rows = rows.sort_values("co2_delta_kg")
        axis.barh(rows["case"], rows["co2_delta_kg"], color="#247a8a")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Change in physical grid-import emissions (kg CO2)",
                 title=f"{group}: change against each paired reference")
        axis.grid(axis="x", alpha=0.25)
        _save(figure, output / f"carbon_deltas_{group}.png")


def diagnostic_row(run_dir: Path, window_days: int = 7) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run_dir / "energy_simulation_result.csv", index_col=0)
    frame.index = pd.to_datetime(frame.index, format="mixed", errors="raise")
    step_s = float(config["microgrid_step_size_s"])
    if step_s <= 0:
        raise ValueError("Microgrid step must be positive")
    import_w = (-pd.to_numeric(frame["e_delta"]).clip(upper=0)) / step_s
    allowance_w = float(config["budget_policy_details"]["effective_grid_power_limit"])
    excess_w = (import_w - allowance_w).clip(lower=0)
    start = pd.Timestamp(config["sim_start"])
    within_window = (frame.index >= start) & (
        frame.index < start + pd.Timedelta(days=window_days)
    )
    generator_columns = [
        column for column in frame
        if column.startswith("actor_states.Generator-") and column.endswith(".p")
    ]
    if not generator_columns:
        raise ValueError(f"No physical generator measurements in {run_dir}")
    renewable_w = frame[generator_columns].apply(pd.to_numeric, errors="raise").sum(axis=1)
    budget_renewable_w = pd.to_numeric(frame["renewable_generation_w"], errors="raise")
    signal_mismatch = (budget_renewable_w - renewable_w).abs()
    return {
        "peak_grid_import_kw": float(import_w.max() / 1000),
        "samples_above_scheduler_grid_allowance": int((excess_w > 1e-6).sum()),
        "max_excess_above_scheduler_grid_allowance_kw": float(excess_w.max() / 1000),
        f"physical_generation_first_{window_days}d_kwh": float(
            renewable_w.loc[within_window].sum() * step_s / 3_600_000
        ),
        "budget_signal_mismatch_samples": int((signal_mismatch > 1e-6).sum()),
        "budget_signal_max_mismatch_kw": float(signal_mismatch.max() / 1000),
        "monitor_samples": len(frame),
    }


def diagnostics(campaign_dir: Path) -> Path:
    summary = pd.read_csv(campaign_dir / "summary.csv")
    rows = []
    for _, run in summary.iterrows():
        if str(run["complete"]).lower() != "true":
            continue
        rows.append({"case": run["case"], "system": run["system"],
                     "energy": run["energy"],
                     **diagnostic_row(Path(run["result_dir"]))})
    output = campaign_dir / "diagnostics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(output)
    return output


def _trace(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "energy_simulation_result.csv", index_col=0)
    frame.index = pd.to_datetime(frame.index, format="mixed")
    interval_s = frame.index.to_series().diff().dt.total_seconds().median()
    if pd.isna(interval_s) or interval_s <= 0:
        raise ValueError(f"Cannot determine monitor interval in {run_dir}")
    power_columns = [column for column in frame if column.endswith("_power")]
    result = pd.DataFrame(index=frame.index)
    result["IT power"] = frame[power_columns].sum(axis=1) / 1000
    generator_columns = [
        column for column in frame
        if column.startswith("actor_states.Generator-") and column.endswith(".p")
    ]
    if not generator_columns:
        raise ValueError(f"No physical generator measurements in {run_dir}")
    result["Renewable generation"] = frame[generator_columns].sum(axis=1) / 1000
    result["Grid import"] = (-frame["e_delta"]).clip(lower=0) / interval_s / 1000
    result["Grid export"] = frame["e_delta"].clip(lower=0) / interval_s / 1000
    result["Battery"] = frame["charge_level"] / 1000
    return result.resample("1h").mean()


def trace(run_dirs: list[Path], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for index, run_dir in enumerate(run_dirs):
        data = _trace(run_dir)
        label = run_dir.name.split("_eval-")[-1].split("-")[0]
        axes[0].plot(data.index, data["IT power"], label=f"{label} IT")
        if index == 0:
            axes[0].plot(data.index, data["Renewable generation"],
                         color="forestgreen", linestyle="--", label="renewables")
        axes[1].plot(data.index, data["Grid import"], label=f"{label} import")
        axes[1].plot(data.index, data["Grid export"], linestyle="--", label=f"{label} export")
        axes[2].plot(data.index, data["Battery"], label=label)
    axes[0].set_ylabel("IT power (kW)")
    axes[1].set_ylabel("Physical grid power (kW)")
    axes[2].set_ylabel("Stored energy (kWh)")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    axes[2].set_xlabel("Simulation time")
    figure.autofmt_xdate()
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p_overview = commands.add_parser("overview")
    p_overview.add_argument("campaign_dir", type=Path)
    p_diagnostics = commands.add_parser("diagnostics")
    p_diagnostics.add_argument("campaign_dir", type=Path)
    p_trace = commands.add_parser("trace")
    p_trace.add_argument("run_dirs", type=Path, nargs="+")
    p_trace.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "overview":
        overview(args.campaign_dir)
    elif args.action == "diagnostics":
        diagnostics(args.campaign_dir)
    else:
        trace(args.run_dirs, args.output)


if __name__ == "__main__":
    main()
