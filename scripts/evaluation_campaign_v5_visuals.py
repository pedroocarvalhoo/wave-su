"""Create v5 input and observed job-order figures from documented artefacts.

No simulation is started by this script. Each order figure requires complete runs from the same energy profile and uses
the same input jobs in every row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs" / "evaluation_campaign" / "v5"
MEDIUM_QUEUE = ROOT / "datasets" / "jobs" / "deucalion_test" / "jobs_arm_a64fx_fujitsu_medium.csv"
CLUSTER = "arm_a64fx_fujitsu"
COLORS = {1: "#5086bf", 2: "#e5a44b", 3: "#b45c8c"}
STAGES = ("grid", "carbon", "solar", "solar_wind")
SIM_START = pd.Timestamp("2023-06-01 12:00:00+00:00")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_dir(campaign: Path) -> Path:
    path = campaign / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def inputs(campaign: Path) -> Path:
    carbon = pd.read_csv(ROOT / "datasets" / "historical_data.csv", parse_dates=["datetime"])
    pv = pd.read_csv(ROOT / "datasets" / "pv_gen.csv", parse_dates=["time"])
    wind = pd.read_csv(ROOT / "datasets" / "wind_gen.csv", parse_dates=["time"])
    end = SIM_START + pd.Timedelta(days=7)
    c = carbon.set_index("datetime").loc[SIM_START:end, "carbon_intensity"]
    p = pv.set_index("time").loc[SIM_START:end, "P"] * 40
    w = wind.set_index("time").loc[SIM_START:end, "P"] * 40
    if len(c) < 168 or len(p) < 168 or len(w) < 168:
        raise ValueError("Input profiles do not cover the first seven days")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.4), sharex=True)
    x = (c.index - SIM_START).total_seconds() / 3600
    top.plot((p.index - SIM_START).total_seconds() / 3600, p, color="#e5a44b", label="PV 40 kW")
    top.plot((w.index - SIM_START).total_seconds() / 3600, w, color="#5086bf", label="Wind 40 kW")
    top.set_ylabel("Available power (kW)")
    top.legend(loc="upper right", frameon=False, ncol=2)
    bottom.plot(x, c, color="#444e69", label="Grid carbon intensity")
    bottom.axhline(250, color="#c75f50", linestyle="--", label="250 gCO₂/kWh allowance")
    bottom.set_ylabel("Grid carbon intensity\n(gCO₂/kWh)")
    bottom.set_xlabel("Hours since simulation start")
    bottom.legend(loc="upper right", frameon=False)
    for ax in (top, bottom):
        ax.set_xlim(0, 168)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = output_dir(campaign) / "energy_inputs.pdf"
    fig.savefig(out)
    plt.close(fig)
    provenance = {
        "method": "Bundled hourly profiles from simulation start, shown at configured peak scales",
        "sources": {path_label(p): digest(p) for p in (
            ROOT / "datasets" / "historical_data.csv",
            ROOT / "datasets" / "pv_gen.csv", ROOT / "datasets" / "wind_gen.csv",
            Path(__file__).resolve(),
        )},
        "output": path_label(out),
    }
    (out.parent / "energy_inputs_evidence.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return out


def _complete_cases(campaign: Path, stage: str, policies: tuple[str, ...]) -> tuple[dict[str, Path], Path]:
    summary = campaign / "summary.csv"
    frame = pd.read_csv(summary).set_index("case")
    names = [f"{stage}_{policy}" for policy in policies]
    selected = frame.loc[names]
    if not selected["complete"].all() or not selected["run_status"].eq("complete").all():
        raise ValueError(f"{stage} has incomplete runs")
    for column in ("system", "energy", "workload_hash", "source_hash"):
        if selected[column].nunique() != 1:
            raise ValueError(f"{stage} has incompatible {column}")
    return {name: Path(selected.loc[name, "result_dir"]) for name in names}, summary


def queue_order(
    campaign: Path, stage: str, n: int | None = None,
    policies: tuple[str, ...] = ("fifo", "priority", "eft"),
) -> Path:
    if stage not in STAGES or not policies or set(policies) - {"fifo", "priority", "eft", "puzzle"}:
        raise ValueError((stage, policies))
    run_dirs, summary = _complete_cases(campaign, stage, policies)
    source = pd.read_csv(MEDIUM_QUEUE).sort_values(["Arrival_Offset_s", "job_id"])
    if n is not None:
        source = source.head(n)
    n = len(source)
    ids = source["job_id"].astype(int).tolist()
    priority = dict(zip(source["job_id"].astype(int), source["Priority"].astype(int)))
    records = {}
    inputs = [summary, MEDIUM_QUEUE, Path(__file__).resolve()]
    for name, run_dir in run_dirs.items():
        path = run_dir / f"jobs_metrics_{CLUSTER}.csv"
        inputs.append(path)
        jobs = pd.read_csv(path).set_index("Job_ID", verify_integrity=True).loc[ids]
        if not jobs["Priority"].astype(int).eq([priority[j] for j in ids]).all():
            raise ValueError(f"Priority differs in {name}")
        if not jobs["Queue_Name"].eq("medium").all():
            raise ValueError(f"Wrong queue in {name}")
        start = pd.to_datetime(jobs["Submission_Time"], format="mixed") + pd.to_timedelta(jobs["Wait_Time_s"], unit="s")
        records[name] = sorted(ids, key=lambda j: (start.loc[j], j))
    rows = [("Input queue", ids), *((name.removeprefix(stage + "_").upper(), order) for name, order in records.items())]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11.6, 1.05 * len(rows) + 0.4), sharex=True)
    for ax, (label, order) in zip(np.atleast_1d(axes), rows):
        for i, job_id in enumerate(order):
            ax.add_patch(Rectangle((i, 0.1), .94, .8, facecolor=COLORS[priority[job_id]], edgecolor="white"))
            ax.text(i + .47, .5, str(job_id), va="center", ha="center", fontsize=8, color="#182230")
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=5)
        ax.spines[:].set_visible(False)
    axes = np.atleast_1d(axes)
    axes[-1].set_xticks([i + .47 for i in range(n)], [str(i + 1) for i in range(n)])
    axes[-1].set_xlabel("Position among the same selected jobs")
    fig.legend(handles=[Patch(facecolor=COLORS[k], label=f"Priority {k}") for k in COLORS],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.55, 1.01))
    fig.subplots_adjust(left=.17, right=.99, top=.89, bottom=.12, hspace=.12)
    scope = "wave" if set(policies) != {"puzzle"} else "puzzle"
    suffix = "" if scope == "wave" else "_puzzle"
    out = output_dir(campaign) / f"queue_order_{stage}{suffix}.pdf"
    fig.savefig(out)
    plt.close(fig)
    evidence = {
        "method": "All ARM medium-queue jobs; input order by arrival; actual execution order by Submission_Time plus Wait_Time_s, ties by job ID",
        "selected_job_ids": ids,
        "actual_orders": records,
        "sources": {path_label(path): digest(path) for path in inputs},
        "output": path_label(out),
    }
    (out.parent / f"queue_order_{stage}{suffix}_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return out


def _summary_cases(campaign: Path, names: list[str]) -> tuple[pd.DataFrame, Path]:
    summary_path = campaign / "summary.csv"
    summary = pd.read_csv(summary_path).set_index("case", verify_integrity=True)
    rows = summary.loc[names].copy()
    if not rows["complete"].all() or not rows["run_status"].eq("complete").all():
        raise ValueError("All requested cases must have complete runs")
    for column in ("system", "workload_hash", "source_hash"):
        if rows[column].nunique() != 1:
            raise ValueError(f"Incompatible {column} in requested cases")
    return rows, summary_path


def synthesis(campaign: Path) -> list[Path]:
    """Build the full same-workload table and two aggregate figures."""
    policies = ("fifo", "priority", "eft", "puzzle")
    names = [f"{stage}_{policy}" for stage in STAGES for policy in policies]
    rows, summary_path = _summary_cases(campaign, names)
    rows.insert(0, "profile", [stage for stage in STAGES for _ in policies])
    rows.insert(1, "scheduler", list(policies) * len(STAGES))
    rows["effective_grid_g_per_kwh"] = (
        rows["co2_kg"] * 1000 / rows["physical_grid_import_kwh"].replace(0, float("nan"))
    )
    columns = [
        "profile", "scheduler", "energy", "completed_jobs", "deadline_misses",
        "mean_wait_h", "p95_wait_h", "makespan_h", "it_energy_kwh",
        "physical_grid_import_kwh", "effective_grid_g_per_kwh", "co2_kg",
        "battery_first_record_kwh", "battery_final_kwh",
        "budget_violation_samples", "budget_max_excess_w",
    ]
    out_dir = output_dir(campaign)
    table = out_dir / "super_table.csv"
    rows[columns].to_csv(table, index_label="case")
    metrics = (
        ("co2_kg", "Operational CO₂ (kg)"),
        ("p95_wait_h", "P95 queue wait (h)"),
        ("physical_grid_import_kwh", "Physical grid import (kWh)"),
        ("it_energy_kwh", "IT energy (kWh)"),
    )
    policy_colors = {"fifo": "#5086bf", "priority": "#b45c8c", "eft": "#e5a44b", "puzzle": "#499572"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for policy in policies:
            values = rows[rows["scheduler"] == policy][metric].to_numpy()
            ax.plot(range(len(STAGES)), values, marker="o", color=policy_colors[policy], label=policy.upper())
        ax.set_xticks(range(len(STAGES)), ["Grid", "Carbon", "PV", "PV + wind"])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=.91, bottom=.10, left=.11, right=.98, hspace=.42, wspace=.32)
    plot = out_dir / "profile_effects.pdf"
    fig.savefig(plot)
    plt.close(fig)
    source_paths = [summary_path, Path(__file__).resolve()]
    evidence = {
        "method": "All 16 complete main cases; one common workload and simulation source; effective intensity from operational CO2 divided by physical grid import",
        "sources": {path_label(path): digest(path) for path in source_paths},
        "outputs": [path_label(table), path_label(plot)],
    }
    (out_dir / "synthesis_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return [table, plot]


def dvfs_effect(campaign: Path) -> list[Path]:
    """Compare adaptive and genuinely fixed dispatch for FIFO and Priority."""
    names = ["dvfs_fifo_fixed", "solar_fifo", "dvfs_priority_fixed", "solar_priority"]
    rows, summary_path = _summary_cases(campaign, names)
    if rows["energy"].nunique() != 1:
        raise ValueError("DVFS controls use different energy profiles")
    frame = rows[["co2_kg", "it_energy_kwh", "physical_grid_import_kwh",
                  "mean_wait_h", "p95_wait_h", "makespan_h", "deadline_misses"]].copy()
    source_paths = [summary_path, Path(__file__).resolve()]
    changes = []
    for name in names:
        run = Path(rows.loc[name, "result_dir"])
        paths = sorted(run.glob("jobs_metrics_*.csv"))
        if not paths:
            raise FileNotFoundError(f"No job metrics for {name}")
        source_paths.extend(paths)
        changes.append(sum(pd.read_csv(path)["Frequency_Changes"].sum() for path in paths))
    frame["frequency_changes"] = changes
    frame["scheduler"] = ["FIFO", "FIFO", "Priority", "Priority"]
    frame["frequency"] = ["Fixed", "Adaptive", "Fixed", "Adaptive"]
    out_dir = output_dir(campaign)
    table = out_dir / "dvfs_effect.csv"
    frame.to_csv(table, index_label="case")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    measures = (("it_energy_kwh", "IT energy (kWh)"), ("co2_kg", "Operational CO₂ (kg)"),
                ("p95_wait_h", "P95 queue wait (h)"))
    for ax, (metric, ylabel) in zip(axes, measures):
        for i, policy in enumerate(("FIFO", "Priority")):
            pair = frame[frame["scheduler"] == policy]
            ax.bar([i * 3, i * 3 + 1], pair[metric],
                   color=["#89939e", "#499572"], width=.8)
        ax.set_xticks([.5, 3.5], ["FIFO", "Priority"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(facecolor="#89939e", label="Fixed"),
                        Patch(facecolor="#499572", label="Adaptive")],
               loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=.84, bottom=.16, left=.08, right=.99, wspace=.38)
    plot = out_dir / "dvfs_effect.pdf"
    fig.savefig(plot)
    plt.close(fig)
    evidence = {
        "method": "Same PV/carbon profile and workload; FIFO/Priority adaptive versus fixed controller",
        "sources": {path_label(path): digest(path) for path in source_paths},
        "outputs": [path_label(table), path_label(plot)],
    }
    (out_dir / "dvfs_effect_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return [table, plot]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("inputs")
    sub.add_parser("synthesis")
    sub.add_parser("dvfs")
    order = sub.add_parser("queue-order")
    order.add_argument("stage", choices=STAGES)
    order.add_argument("--scope", choices=("wave", "puzzle"), default="wave")
    args = parser.parse_args()
    if args.action == "inputs":
        print(inputs(args.campaign_dir))
    elif args.action == "synthesis":
        for output in synthesis(args.campaign_dir):
            print(output)
    elif args.action == "dvfs":
        for output in dvfs_effect(args.campaign_dir):
            print(output)
    else:
        policies = ("puzzle",) if args.scope == "puzzle" else ("fifo", "priority", "eft")
        print(queue_order(args.campaign_dir, args.stage, policies=policies))


if __name__ == "__main__":
    main()
