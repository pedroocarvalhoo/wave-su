"""
Puzzle forecast visualization.

Plots the plans produced by PuzzleBudgetPolicy during the simulation.  This is
the planned/forecast puzzle, not the post-hoc reconstruction of actual job
execution.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pandas as pd


def load_forecast(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    slots_path = os.path.join(results_dir, "puzzle_forecast_slots.csv")
    jobs_path = os.path.join(results_dir, "puzzle_forecast_jobs.csv")
    if not os.path.exists(slots_path) or not os.path.exists(jobs_path):
        raise FileNotFoundError(
            "Puzzle forecast CSVs not found. Run a simulation with "
            "--budget-policy puzzle first."
        )

    slots = pd.read_csv(
        slots_path,
        parse_dates=["start_time", "end_time"],
    )
    jobs = pd.read_csv(
        jobs_path,
        parse_dates=["plan_start", "plan_end", "arrival_time", "start_time", "end_time"],
    )
    return slots, jobs


def resolve_plan_id(slots: pd.DataFrame, plan_id: str | int) -> int:
    plan_ids = sorted(int(pid) for pid in slots["plan_id"].dropna().unique())
    if not plan_ids:
        raise ValueError("No puzzle forecast plans found.")
    if plan_id == "first":
        return plan_ids[0]
    if plan_id == "latest":
        return plan_ids[-1]

    selected = int(plan_id)
    if selected not in plan_ids:
        raise ValueError(f"Plan {selected} not found. Available plans: {plan_ids}")
    return selected


def forecast_clusters(slots: pd.DataFrame) -> list[str]:
    clusters = [
        str(cluster)
        for cluster in slots["cluster_type"].dropna().unique()
        if str(cluster) != "__total__"
    ]
    return sorted(clusters)


def plot_cluster_forecast(
    slots: pd.DataFrame,
    jobs: pd.DataFrame,
    cluster: str,
    plan_id: int,
    output_path: str,
    show: bool = False,
    figsize: tuple[float, float] = (22, 9),
    dpi: int = 150,
) -> None:
    slot_df = slots[
        (slots["plan_id"] == plan_id)
        & (slots["cluster_type"] == cluster)
    ].sort_values("slot_idx")
    if slot_df.empty:
        print(f"  No forecast slots found for {cluster}; skipping.")
        return

    job_df = jobs[
        (jobs["plan_id"] == plan_id)
        & (jobs["cluster_type"] == cluster)
        & (jobs["placed"].astype(str).str.lower().isin(["true", "1"]))
    ].copy()
    job_df = job_df.dropna(subset=["start_time", "end_time"])

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    times = list(slot_df["start_time"]) + [slot_df["end_time"].iloc[-1]]
    envelope_kw = list(slot_df["total_available_w"] / 1000.0) + [0.0]
    allocated_kw = list(slot_df["allocated_w"] / 1000.0) + [0.0]

    ax.fill_between(
        times,
        envelope_kw,
        step="post",
        alpha=0.20,
        color="#1a3a5c",
        label="Forecast budget envelope",
    )
    ax.step(
        times,
        envelope_kw,
        where="post",
        color="#4a9eff",
        linewidth=1.8,
    )
    ax.step(
        times,
        allocated_kw,
        where="post",
        color="#f2cc60",
        linewidth=1.2,
        alpha=0.9,
        label="Planned allocated power",
    )

    for idx, row in enumerate(job_df.itertuples(index=False)):
        start = row.start_time
        end = row.end_time
        if pd.isna(start) or pd.isna(end):
            continue

        x = mdates.date2num(start)
        width = mdates.date2num(end) - x
        y = float(row.stack_base_w) / 1000.0
        height = float(row.power_w) / 1000.0

        hue = (idx * 0.618033988749895) % 1.0
        color = hsv_to_rgb((hue, 0.62, 0.84))

        rect = mpatches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=0.5,
            edgecolor=(*color, 0.85),
            facecolor=(*color, 0.65),
        )
        ax.add_patch(rect)

        width_hours = (end - start).total_seconds() / 3600.0
        if height > 0.2 and width_hours > 0.25:
            ax.text(
                x + width / 2,
                y + height / 2,
                f"J{int(row.job_id)}",
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                fontweight="bold",
                alpha=0.9,
            )

    plan_start = slot_df["start_time"].iloc[0]
    plan_end = slot_df["end_time"].iloc[-1]
    ax.set_title(
        f"Puzzle Forecast Plan #{plan_id} - {cluster}\n"
        f"{plan_start:%Y-%m-%d %H:%M} to {plan_end:%Y-%m-%d %H:%M} | "
        f"{len(job_df)} planned jobs",
        fontsize=12,
        color="#f0f6fc",
    )
    ax.set_xlabel("Time", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Power (kW)", fontsize=11, color="#c9d1d9")
    ax.set_xlim(mdates.date2num(plan_start), mdates.date2num(plan_end))
    max_y = max(envelope_kw) if envelope_kw else 0.0
    ax.set_ylim(0, max_y * 1.08 if max_y > 0 else 1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.grid(axis="y", alpha=0.18, linestyle="--", color="#484f58")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    print(f"  Saved {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_all_clusters(
    results_dir: str,
    plan_id: str | int = "first",
    show: bool = False,
) -> None:
    slots, jobs = load_forecast(results_dir)
    selected_plan_id = resolve_plan_id(slots, plan_id)

    print(f"Generating Puzzle forecast charts for plan #{selected_plan_id}...")
    for cluster in forecast_clusters(slots):
        output_path = os.path.join(
            results_dir,
            f"puzzle_forecast_{cluster}.png",
        )
        plot_cluster_forecast(
            slots,
            jobs,
            cluster,
            selected_plan_id,
            output_path,
            show=show,
        )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot Puzzle Scheduler forecasts.")
    parser.add_argument("results_dir")
    parser.add_argument(
        "--plan-id",
        default="first",
        help="Plan id to plot, or 'first'/'latest'. Default: first.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    plot_all_clusters(args.results_dir, plan_id=args.plan_id, show=args.show)


if __name__ == "__main__":
    main()
