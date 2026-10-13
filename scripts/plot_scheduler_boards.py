#!/usr/bin/env python3
"""Compare real scheduler executions against one real Puzzle green board."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from vessim.visualization import plot_scheduler_board_comparison


def _policy_label(results_dir: str) -> str:
    config_path = Path(results_dir).expanduser().resolve() / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    policy = str(
        config.get("scheduling_policy_key")
        or config.get("scheduling_policy")
        or config_path.parent.name
    )
    budget = str(config.get("budget_policy_key") or "unknown-budget")
    return f"{policy} ({budget})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the jobs actually executed by several schedulers against "
            "the same stitched green board from a real Puzzle run."
        )
    )
    parser.add_argument(
        "reference_results_dir",
        help="Puzzle results directory providing puzzle_forecast_slots.csv.",
    )
    parser.add_argument(
        "results_dirs",
        nargs="+",
        help="Result directories to compare, in desired row order.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Optional labels matching results_dirs. Defaults to the scheduling "
            "and budget policy keys stored in each config.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Default: <reference>/scheduler_board_comparison."
        ),
    )
    parser.add_argument(
        "--window-start",
        help=(
            "Start timestamp understood by pandas. Default: first reference "
            "board slot."
        ),
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=12.0,
        help="Common window length in hours. Default: 12.",
    )
    parser.add_argument(
        "--cluster",
        dest="clusters",
        action="append",
        help="Cluster to render. Repeat to select several; default renders all.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.labels and len(args.labels) != len(args.results_dirs):
        raise SystemExit(
            "--labels must contain exactly one label per results directory"
        )

    labels = args.labels or [
        _policy_label(results_dir) for results_dir in args.results_dirs
    ]
    scheduler_results = list(zip(labels, args.results_dirs))
    output_dir = args.output_dir or os.path.join(
        args.reference_results_dir,
        "scheduler_board_comparison",
    )

    outputs = plot_scheduler_board_comparison(
        args.reference_results_dir,
        scheduler_results,
        output_dir,
        window_start=args.window_start,
        window_hours=args.window_hours,
        clusters=args.clusters,
        dpi=args.dpi,
    )
    for output in outputs:
        print(f"Scheduler board comparison saved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
