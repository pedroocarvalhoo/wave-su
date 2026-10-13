"""
Gantt-Puzzle Visualization: Combines Gantt chart (node allocation) and Puzzle packing.

Reads the schedulers from schedulers.pkl and energy results from
energy_simulation_result.csv to plot a 2D job block Gantt chart where:
- X-axis is Time.
- Y-axis is Power (kW).
- Each job is a solid 2D block of width = duration and height = estimated power.
- Overlaid with the real scheduler budget and measured cluster power.

Usage::

    python scripts/plot_gantt_puzzle.py <results_dir> [--show]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.colors import hsv_to_rgb
import numpy as np
import pandas as pd


ARM_ZOOM_TOP_KW = 130.0
X86_ZOOM_TOP_KW = 220.0


def load_schedulers(results_dir: str):
    path = os.path.join(results_dir, "schedulers.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_results_df(results_dir: str):
    path = os.path.join(results_dir, "energy_simulation_result.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_node_config(results_dir: str) -> dict:
    """Find node profiles in the run or an ancestor workspace."""
    run_dir = Path(results_dir).resolve()
    config_path = run_dir / "config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            embedded = json.load(f).get("node_config")
        if isinstance(embedded, dict) and embedded:
            return embedded

    for parent in (run_dir, *run_dir.parents):
        candidate = parent / "datasets" / "node_config.json"
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No node_config.json available for {run_dir}")


def node_power_at_frequency(cfg: dict, frequency) -> float:
    idle_power = float(cfg.get("idle_power_w", 0.0))
    freq_table = {
        float(f): float(p)
        for f, p in cfg.get("frequencies", {}).items()
    }
    if freq_table:
        if frequency is None:
            frequency = cfg.get("base_frequency_ghz")
        if frequency is not None:
            frequency = float(frequency)
            selected = min(freq_table, key=lambda f: abs(f - frequency))
            dynamic_power = freq_table[selected]
        else:
            dynamic_power = max(freq_table.values())
    else:
        dynamic_power = float(cfg.get("max_power_w", 0.0))
    return idle_power + dynamic_power


def job_power_w(job, cfg: dict) -> float:
    frequency = None
    if getattr(job, "frequency_history", None):
        frequency = job.frequency_history[0][1]
    if frequency is None:
        frequency = getattr(job, "assigned_frequency", None)
    return node_power_at_frequency(cfg, frequency) * job.num_nodes


def pack_jobs_to_power(
    jobs_to_plot: list[tuple[datetime, datetime, any]],
    cfg: dict,
    *,
    dynamic_only: bool = False,
) -> list[tuple[float, datetime, datetime, any, float]]:
    """
    Greedy 2D interval packer.
    Assigns each job a contiguous range of y-coordinates of height = power kW
    such that overlapping jobs (in time) do not overlap in y-range.
    """
    def plotted_power(job) -> float:
        total = job_power_w(job, cfg)
        return total - float(cfg.get("idle_power_w", 0.0)) * job.num_nodes if dynamic_only else total

    sorted_jobs = sorted(
        jobs_to_plot,
        key=lambda x: (
            x[0],
            -plotted_power(x[2]),
            -(x[1] - x[0]).total_seconds()
        )
    )

    placed: list[tuple[datetime, float, float]] = []
    job_placements = []

    for start, end, job in sorted_jobs:
        h = plotted_power(job) / 1000.0
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"Cannot draw job {job.job_id}: nonpositive estimated power")
        active = [p for p in placed if p[0] > start]

        y_start = 0.0
        while True:
            overlap = False
            for _, ay_start, ay_end in active:
                if not (y_start + h <= ay_start or y_start >= ay_end):
                    overlap = True
                    y_start = ay_end
                    break
            if not overlap:
                break

        placed.append((end, y_start, y_start + h))
        job_placements.append((y_start, start, end, job, h))

    return job_placements


def plot_gantt_puzzle(
    results_dir: str,
    show: bool = False,
    figsize: tuple[float, float] = (20, 10),
    dpi: int = 150,
    *,
    output_dir: str | None = None,
    policy_label: str = "Puzzle",
    arm_zoom_top_kw: float = ARM_ZOOM_TOP_KW,
    x86_zoom_top_kw: float = X86_ZOOM_TOP_KW,
    clusters: tuple[str, ...] | None = None,
    time_limits: tuple[datetime, datetime] | None = None,
) -> None:
    target_dir = Path(output_dir or results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    schedulers = load_schedulers(results_dir)
    energy_df = load_results_df(results_dir)
    node_config = load_node_config(results_dir)

    print(f"Generating Gantt-{policy_label} charts...")

    # Dark background theme matching plot_puzzle.py
    plt.style.use("dark_background")

    for scheduler in schedulers:
        ct = scheduler.cluster_type
        if clusters is not None and ct not in clusters:
            continue
        print(f"\n[{ct}]")

        # Collect executed jobs
        jobs_to_plot = []
        for job in scheduler.job_queue.jobs:
            if job.status in ["running", "completed"] and job.start_time is not None:
                start = job.start_time
                if getattr(job, "completion_time", None) is not None:
                    end = job.completion_time
                else:
                    duration = job.actual_t_exec or job.remaining_time
                    end = start + timedelta(seconds=duration)
                jobs_to_plot.append((start, end, job))

        if not jobs_to_plot:
            print(f"  No executed jobs found for {ct}; skipping.")
            continue

        total_nodes = len(scheduler.timeline)
        print(f"  Packing {len(jobs_to_plot)} jobs for a cluster of {total_nodes} nodes...")
        cfg = node_config.get(ct)
        if not cfg:
            raise ValueError(f"Missing node power profile for {ct} in {results_dir}")
        idle_base = ct in {"arm_a64fx_fujitsu", "x86_epyc_rome"}
        zoom_top_kw = {"arm_a64fx_fujitsu": arm_zoom_top_kw,
                       "x86_epyc_rome": x86_zoom_top_kw}.get(ct)
        idle_kw = total_nodes * float(cfg["idle_power_w"]) / 1000.0 if idle_base else 0.0
        placements = pack_jobs_to_power(jobs_to_plot, cfg, dynamic_only=idle_base)

        # Zoom CPU partitions from their always-on idle demand so the dynamic
        # job pieces fill the plot. GPU retains its original total-power view.
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        if idle_base:
            ax.axhline(
                idle_kw, color="#aebbc7", linewidth=1.4, alpha=0.9,
                label=f"Idle baseline ({idle_kw:.1f} kW)", clip_on=False,
            )

        # -- Draw Budget and Measured Power --------------------------------
        budget_col = f"{ct}_budget"
        if budget_col in energy_df.columns:
            times = energy_df.index
            budget_kw = energy_df[budget_col].values / 1000.0
            ax.step(
                times,
                budget_kw,
                where="post",
                color="#4a9eff",
                linewidth=1.8,
                alpha=0.9,
                label="Real budget",
            )
            ax.fill_between(
                times,
                budget_kw,
                step="post",
                alpha=0.15,
                color="#1a3a5c",
                linewidth=0,
            )
            if idle_base and np.nanmax(budget_kw) > zoom_top_kw:
                # Show the periods in which the budget leaves the zoomed range.
                ax.step(
                    times,
                    np.where(budget_kw > zoom_top_kw, zoom_top_kw - 0.6, np.nan),
                    where="post", color="#4a9eff", linewidth=3.0, alpha=0.95,
                )
                ax.text(
                    0.015, 0.96,
                    f"Budget above axis; peak {np.nanmax(budget_kw):.1f} kW",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=14, color="#8fc1ff",
                    bbox={"facecolor": "#0d1117", "edgecolor": "none", "alpha": 0.88},
                )

        power_col = f"{ct}_power"
        if power_col in energy_df.columns:
            ax.step(
                energy_df.index,
                energy_df[power_col].values / 1000.0,
                where="post",
                color="#f2cc60",
                linewidth=1.1,
                alpha=0.85,
                label="Measured power",
            )

        # -- Draw Jobs as 2D Blocks ----------------------------------------
        for i, (y_start, start, end, job, height_kw) in enumerate(placements):
            start_num = mdates.date2num(start)
            width = mdates.date2num(end) - start_num
            display_y = y_start + idle_kw

            hue = (i * 0.618033988749895) % 1.0
            rgb = hsv_to_rgb((hue, 0.65, 0.80))
            r = mpatches.FancyBboxPatch(
                (start_num, display_y), width, height_kw,
                boxstyle="round,pad=0,rounding_size=0.001",
                linewidth=0.5, edgecolor=(*rgb, 0.7), facecolor=(*rgb, 0.65),
            )
            ax.add_patch(r)

            width_hours = (end - start).total_seconds() / 3600.0
            if height_kw > 0.2 and width_hours > 0.5:
                ax.text(
                    start_num + width / 2, display_y + height_kw / 2,
                    f"J{job.job_id}", ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold", alpha=0.9,
                )

        # -- Formatting ---------------------------------------------------
        ax.set_xlabel("Time", fontsize=12, color="#c9d1d9", labelpad=10)
        ax.set_ylabel("Power (kW)", fontsize=12, color="#c9d1d9", labelpad=10)
        packed_peak_kw = max((idle_kw + y + h for y, _, _, _, h in placements), default=0.0)
        measured_peak_kw = (float(energy_df[power_col].max()) / 1000.0
                            if power_col in energy_df.columns else 0.0)
        if idle_base:
            if max(packed_peak_kw, measured_peak_kw) > zoom_top_kw:
                raise ValueError(f"{ct} zoom would hide measured power or job pieces")
            ax.set_ylim(idle_kw, zoom_top_kw)
            tick_step = 10 if zoom_top_kw - idle_kw <= 80 else 20
            first_tick = int(np.ceil(idle_kw / tick_step) * tick_step)
            zoom_ticks = [idle_kw, *range(first_tick, int(zoom_top_kw) + 1, tick_step)]
            if zoom_ticks[-1] < zoom_top_kw:
                zoom_ticks.append(zoom_top_kw)
            if len(zoom_ticks) > 1 and zoom_ticks[1] == idle_kw:
                zoom_ticks.pop(1)
            ax.set_yticks(zoom_ticks)
            ax.set_yticklabels([f"{idle_kw:.1f}", *(f"{tick:g}" for tick in zoom_ticks[1:])])
            for y_break in (0.0, 0.025):
                ax.plot(
                    (-0.012, 0.012), (y_break - 0.015, y_break + 0.015),
                    transform=ax.transAxes, color="#c9d1d9", linewidth=1.4,
                    clip_on=False, zorder=8,
                )
        else:
            max_y = max(1.0, packed_peak_kw, measured_peak_kw)
            if budget_col in energy_df.columns:
                max_y = max(max_y, float(energy_df[budget_col].max()) / 1000.0)
            ax.set_ylim(0, max_y * 1.05)
        time_start, time_end = time_limits or (energy_df.index[0], energy_df.index[-1])
        ax.set_xlim(mdates.date2num(time_start), mdates.date2num(time_end))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        fig.autofmt_xdate(rotation=0, ha="center")
        ax.grid(axis="y", alpha=0.15, linestyle="--", color="#484f58")
        ax.grid(axis="x", alpha=0.08, linestyle=":", color="#484f58")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
            spine.set_linewidth(0.5)
        ax.set_title(
            f"Gantt-{policy_label} Packing Chart  ·  {ct}\n"
            f"{len(jobs_to_plot)} jobs executed  |  Cluster Capacity: {total_nodes} nodes",
            fontsize=13, color="#e6edf3", pad=15, fontweight="bold",
        )
        ax.legend(
            loc="upper right", fontsize=9,
            facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9",
        )
        fig.tight_layout(pad=2.0)

        output_path = target_dir / f"gantt_{policy_label.lower()}_{ct}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Gantt-{policy_label} plot saved to {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Gantt-Puzzle packed job block visualization "
            "from simulation results."
        ),
    )
    parser.add_argument(
        "results_dir",
        help="Path to the simulation results directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    parser.add_argument("--output-dir", help="Write charts to this directory.")
    parser.add_argument("--policy-label", default="Puzzle", help="Policy name in the chart title.")
    parser.add_argument("--arm-zoom-top-kw", type=float, default=ARM_ZOOM_TOP_KW)
    parser.add_argument("--x86-zoom-top-kw", type=float, default=X86_ZOOM_TOP_KW)
    parser.add_argument("--clusters", nargs="+", help="Only draw these cluster identifiers.")
    args = parser.parse_args()

    plot_gantt_puzzle(
        args.results_dir, show=args.show, output_dir=args.output_dir,
        policy_label=args.policy_label, arm_zoom_top_kw=args.arm_zoom_top_kw,
        x86_zoom_top_kw=args.x86_zoom_top_kw,
        clusters=tuple(args.clusters) if args.clusters else None,
    )


if __name__ == "__main__":
    main()
