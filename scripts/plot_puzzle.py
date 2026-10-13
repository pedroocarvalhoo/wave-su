"""
Puzzle Scheduler Visualization  --  post-hoc from simulation CSVs.

Reads the energy_simulation_result.csv (budget envelope) and the
scheduler_debug_events_<cluster>.csv + jobs_metrics_<cluster>.csv files
to reconstruct the puzzle view: a stacked area chart where the budget
envelope is the "container" and actual job executions are the "pieces".

Usage::

    python scripts/plot_puzzle.py <results_dir> [--cluster <name>] [--show]

If --cluster is omitted, one plot per cluster is generated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend by default.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.colors import hsv_to_rgb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_energy_results(results_dir: str) -> pd.DataFrame:
    """Load the monitor CSV with budget/power time series."""
    path = os.path.join(results_dir, "energy_simulation_result.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def load_node_config(results_dir: str) -> dict:
    """Load node_config from the saved config.json inside results."""
    config_path = os.path.join(results_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("node_config", cfg)
    # Fallback: try the datasets directory.
    fallback = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(results_dir))),
        "datasets", "node_config.json",
    )
    if os.path.exists(fallback):
        with open(fallback) as f:
            return json.load(f)
    return {}


def detect_clusters(results_dir: str) -> list[str]:
    """Detect cluster names from jobs_metrics_*.csv files."""
    clusters = []
    for fname in os.listdir(results_dir):
        if fname.startswith("jobs_metrics_") and fname.endswith(".csv"):
            name = fname[len("jobs_metrics_"):-len(".csv")]
            clusters.append(name)
    return sorted(clusters)


def load_job_events(
    results_dir: str, cluster: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load job metrics and debug events for a cluster.

    Returns (jobs_df, events_df).
    """
    jobs_path = os.path.join(results_dir, f"jobs_metrics_{cluster}.csv")
    events_path = os.path.join(
        results_dir, f"scheduler_debug_events_{cluster}.csv",
    )
    jobs_df = pd.read_csv(jobs_path)
    events_df = pd.read_csv(events_path)
    return jobs_df, events_df


# ---------------------------------------------------------------------------
# Job rectangle reconstruction
# ---------------------------------------------------------------------------

def build_job_rectangles(
    jobs_df: pd.DataFrame,
    events_df: pd.DataFrame,
    cluster: str,
    node_config: dict,
) -> list[dict]:
    """
    Build a list of job rectangles with:
      - start_time, end_time (datetime)
      - power_w (total power of all assigned nodes)
      - job_id, num_nodes, cluster
    """
    cfg = node_config.get(cluster, {})
    idle_power = cfg.get("idle_power_w", 0)
    freq_table = {
        float(f): float(p) for f, p in cfg.get("frequencies", {}).items()
    }

    # Get start events from debug log.
    starts = events_df[events_df["Event"] == "job_started"][
        ["Time", "Job_ID", "Frequency_After_GHz"]
    ].copy()
    starts["Time"] = pd.to_datetime(starts["Time"])
    starts = starts.set_index("Job_ID")

    rectangles = []

    for _, row in jobs_df.iterrows():
        job_id = row["Job_ID"]
        real_exec = row.get("Real_Exec_Time_s", 0)
        if real_exec <= 0:
            continue  # Skip zero-duration or failed-at-start jobs.
        num_nodes = int(row.get("Nodes", 1))

        # Determine start time from debug events.
        if job_id in starts.index:
            entry = starts.loc[job_id]
            if isinstance(entry, pd.DataFrame):
                entry = entry.iloc[0]
            start_time = entry["Time"]
            freq = float(entry.get("Frequency_After_GHz", 0))
        else:
            # Fallback: submission + wait.
            sub_time = pd.to_datetime(row["Submission_Time"])
            wait = float(row.get("Wait_Time_s", 0))
            start_time = sub_time + timedelta(seconds=wait)
            freq = float(row.get("Initial_Frequency_GHz", 0))

        if isinstance(start_time, pd.Timestamp):
            start_time = start_time.to_pydatetime()

        end_time = start_time + timedelta(seconds=float(real_exec))

        # Use the final frequency for power estimation (more accurate for
        # the actual execution, since DVFS may have upscaled).
        final_freq = float(row.get("Final_Frequency_GHz", freq))
        dyn_power = freq_table.get(final_freq, cfg.get("max_power_w", 0) - idle_power)
        power_per_node = idle_power + dyn_power
        total_power = power_per_node * num_nodes

        rectangles.append({
            "job_id": job_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": float(real_exec),
            "power_w": total_power,
            "num_nodes": num_nodes,
            "frequency_ghz": final_freq,
            "cluster": cluster,
        })

    return rectangles


# ---------------------------------------------------------------------------
# Stacking algorithm
# ---------------------------------------------------------------------------

def stack_rectangles(
    rectangles: list[dict], time_index: pd.DatetimeIndex,
) -> list[dict]:
    """
    Assign a vertical base (stack_base_w) to each rectangle so they don't
    overlap visually.  Uses a sweep-line approach for correctness.
    """
    # Sort by start time, then by power (larger first for visual stability).
    rects = sorted(
        rectangles,
        key=lambda r: (r["start_time"], -r["power_w"]),
    )

    # Active rectangles: list of (end_time, base + height).
    active: list[tuple[datetime, float, float]] = []

    for rect in rects:
        t_start = rect["start_time"]
        power = rect["power_w"]

        # Remove expired rectangles.
        active = [
            (end, base, height)
            for end, base, height in active
            if end > t_start
        ]

        # Find the lowest position where this rectangle fits (no overlap).
        # Sort active by base to find gaps.
        active_sorted = sorted(active, key=lambda x: x[1])

        base = 0.0
        for end, a_base, a_height in active_sorted:
            # Check if there's a gap below this active rect.
            if base + power <= a_base:
                break  # Fits in this gap.
            base = max(base, a_base + a_height)

        rect["stack_base_w"] = base
        active.append((rect["end_time"], base, power))

    return rects


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_puzzle(
    energy_df: pd.DataFrame,
    rectangles: list[dict],
    cluster: str,
    output_path: str,
    show: bool = False,
    figsize: tuple[float, float] = (22, 9),
    dpi: int = 150,
) -> None:
    """Generate the puzzle visualization for one cluster."""

    # -- Budget envelope ---------------------------------------------------
    budget_col = f"{cluster}_budget"
    power_col = f"{cluster}_power"

    if budget_col not in energy_df.columns:
        print(f"  Warning: column '{budget_col}' not found; skipping {cluster}.")
        return

    times = energy_df.index
    budget_kw = energy_df[budget_col].values / 1000.0
    actual_power_kw = energy_df[power_col].values / 1000.0 if power_col in energy_df.columns else None

    # -- Setup figure with dark theme ------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # -- Envelope: gradient fill -----------------------------------------
    # Create a subtle gradient effect for the envelope.
    envelope_color = "#1a3a5c"
    envelope_edge = "#4a9eff"

    ax.fill_between(
        times,
        budget_kw,
        step="post",
        alpha=0.20,
        color=envelope_color,
        linewidth=0,
    )
    ax.step(
        times,
        budget_kw,
        where="post",
        color=envelope_edge,
        linewidth=1.8,
        alpha=0.9,
        label="Budget envelope",
    )

    # -- Renewable generation overlay (lighter) --------------------------
    if "renewable_generation_w" in energy_df.columns:
        ren_kw = energy_df["renewable_generation_w"].values / 1000.0
        ax.fill_between(
            times, ren_kw, step="post",
            alpha=0.07, color="#22c55e", linewidth=0,
        )
        ax.step(
            times, ren_kw, where="post",
            color="#22c55e", linewidth=0.8, alpha=0.4,
            label="Renewable generation", linestyle="--",
        )

    # -- Job rectangles ---------------------------------------------------
    if not rectangles:
        print(f"  No job rectangles for {cluster}.")
    else:
        # Stack them.
        rectangles = stack_rectangles(rectangles, times)
        n_jobs = len(rectangles)

        # Color palette: use a perceptually uniform hue cycle.
        for i, rect in enumerate(rectangles):
            x_start = rect["start_time"]
            x_end = rect["end_time"]
            y_base = rect["stack_base_w"] / 1000.0
            height = rect["power_w"] / 1000.0

            # Color: golden-ratio hue spread with warm saturation.
            hue = (i * 0.618033988749895) % 1.0
            rgb = hsv_to_rgb((hue, 0.65, 0.80))

            r = mpatches.FancyBboxPatch(
                (mdates.date2num(x_start), y_base),
                mdates.date2num(x_end) - mdates.date2num(x_start),
                height,
                boxstyle="round,pad=0,rounding_size=0.001",
                linewidth=0.5,
                edgecolor=(*rgb, 0.7),
                facecolor=(*rgb, 0.70),
            )
            ax.add_patch(r)

            # Label inside if large enough.
            width_hours = (x_end - x_start).total_seconds() / 3600.0
            max_budget_kw = float(np.nanmax(budget_kw)) if len(budget_kw) > 0 else 1
            if height > (max_budget_kw * 0.025) and width_hours > 0.5:
                cx = mdates.date2num(x_start) + (
                    mdates.date2num(x_end) - mdates.date2num(x_start)
                ) / 2
                cy = y_base + height / 2
                label = f"J{rect['job_id']}"
                ax.text(
                    cx, cy, label,
                    ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold",
                    alpha=0.9,
                )

    # -- Actual power overlay -------------------------------------------
    if actual_power_kw is not None:
        ax.step(
            times, actual_power_kw, where="post",
            color="#ff6b6b", linewidth=1.2, alpha=0.8,
            label="Actual cluster power",
        )

    # -- Formatting -------------------------------------------------------
    ax.set_xlabel("Time", fontsize=12, color="#c9d1d9", labelpad=10)
    ax.set_ylabel("Power (kW)", fontsize=12, color="#c9d1d9", labelpad=10)

    # Compute stats.
    n_jobs = len(rectangles)
    total_budget_kwh = float(np.nansum(budget_kw)) * (
        (times[1] - times[0]).total_seconds() / 3600.0 if len(times) > 1 else 0
    )
    total_used_kwh = sum(
        r["power_w"] * r["duration_s"] / 3600000.0 for r in rectangles
    )
    utilization_pct = (
        total_used_kwh / total_budget_kwh * 100 if total_budget_kwh > 0 else 0
    )

    title = (
        f"Puzzle Scheduler  ·  {cluster}\n"
        f"{n_jobs} jobs  |  "
        f"Budget {total_budget_kwh:,.0f} kWh  ·  "
        f"Used {total_used_kwh:,.0f} kWh  ·  "
        f"Utilization {utilization_pct:.1f}%"
    )
    ax.set_title(title, fontsize=13, color="#e6edf3", pad=15, fontweight="bold")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate(rotation=0, ha="center")

    max_y = float(np.nanmax(budget_kw)) if len(budget_kw) > 0 else 1
    # Also consider stacked job height.
    if rectangles:
        max_job_top = max(
            (r["stack_base_w"] + r["power_w"]) / 1000.0 for r in rectangles
        )
        max_y = max(max_y, max_job_top)
    ax.set_ylim(0, max_y * 1.08)

    # Grid styling.
    ax.grid(axis="y", alpha=0.15, linestyle="--", color="#484f58")
    ax.grid(axis="x", alpha=0.08, linestyle=":", color="#484f58")
    ax.tick_params(colors="#8b949e", labelsize=9)

    for spine in ax.spines.values():
        spine.set_color("#30363d")
        spine.set_linewidth(0.5)

    # Legend.
    legend = ax.legend(
        loc="upper right", fontsize=9,
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#c9d1d9",
    )

    fig.tight_layout(pad=2.0)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Puzzle plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Combined multi-cluster plot
# ---------------------------------------------------------------------------

def plot_all_clusters(
    results_dir: str,
    clusters: Optional[list[str]] = None,
    show: bool = False,
) -> None:
    """Generate puzzle plots for all (or specified) clusters."""
    energy_df = load_energy_results(results_dir)
    node_config = load_node_config(results_dir)

    if clusters is None:
        clusters = detect_clusters(results_dir)

    if not clusters:
        print("No cluster data found.")
        return

    print(f"Generating puzzle plots for: {', '.join(clusters)}")

    for cluster in clusters:
        print(f"\n[{cluster}]")
        try:
            jobs_df, events_df = load_job_events(results_dir, cluster)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        rects = build_job_rectangles(jobs_df, events_df, cluster, node_config)
        print(f"  {len(rects)} job rectangles built.")

        output_path = os.path.join(results_dir, f"puzzle_{cluster}.png")
        plot_puzzle(
            energy_df, rects, cluster,
            output_path=output_path, show=show,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate puzzle scheduler visualization from simulation results.",
    )
    parser.add_argument(
        "results_dir",
        help="Path to the simulation results directory.",
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="Specific cluster to plot. If omitted, all clusters are plotted.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively instead of just saving.",
    )
    args = parser.parse_args()

    clusters = [args.cluster] if args.cluster else None
    plot_all_clusters(args.results_dir, clusters=clusters, show=args.show)


if __name__ == "__main__":
    main()
