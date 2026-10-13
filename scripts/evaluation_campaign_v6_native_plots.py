"""Generate the requested native simulation plots for a short run."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.plot_energy import load_results_df, load_schedulers
from vessim.visualization import (
    plot_budget_distribution,
    plot_carbon_analysis,
    plot_frequency_timeline,
    plot_result_df,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    data = load_results_df(str(run_dir))
    schedulers = load_schedulers(str(run_dir))
    plot_budget_distribution(data, schedulers, str(run_dir / "budget_distribution.png"))
    plot_result_df(data, str(run_dir / "energy_results.png"), min_soc=0.2)
    plot_carbon_analysis(
        data, schedulers, str(run_dir / "carbon_analysis.png"),
        str(run_dir / "carbon_metrics.csv"),
    )
    plot_frequency_timeline(schedulers, str(run_dir / "frequency_timeline.png"))
    if (run_dir / "puzzle_plan_summary.csv").is_file():
        from scripts.plot_gantt_puzzle import plot_gantt_puzzle
        plot_gantt_puzzle(str(run_dir))


if __name__ == "__main__":
    main()
