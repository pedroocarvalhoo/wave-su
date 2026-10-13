"""Five-day, reproducible Chapter 5 campaign with guarded execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts import evaluation_campaign as base
from scripts import evaluation_campaign_v5 as previous

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs" / "evaluation_campaign" / "v7"
THRESHOLD = 250.0
WORKLOAD = "five_day"


def prepare_workload(campaign_dir: Path) -> dict[str, Path]:
    """Keep every job and arrival; reduce modeled instruction counts to 20%."""
    source = ROOT / "datasets" / "jobs" / "deucalion_test"
    target = campaign_dir / "inputs" / "deucalion_test_five_day"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for input_file in base.workload_files(source):
        frame = pd.read_csv(input_file)
        frame = frame.copy()
        frame["#I"] = (frame["#I"] * 0.2).round().astype("int64")
        path = target / input_file.name
        generated = frame.to_csv(index=False).encode()
        if path.exists() and path.read_bytes() != generated:
            raise ValueError(f"Derived workload changed: {path}; use a new campaign directory")
        if not path.exists():
            path.write_bytes(generated)
        count += len(frame)
    if count != 324:
        raise ValueError(f"Expected 324 derived jobs, found {count}")
    return {WORKLOAD: target}


def jobs_dir_for(case: base.Case, campaign_dir: Path) -> Path:
    if case.workload != WORKLOAD:
        raise ValueError(f"Unexpected workload: {case.workload}")
    return campaign_dir / "inputs" / "deucalion_test_five_day"


def cases() -> dict[str, base.Case]:
    matrix: list[base.Case] = []
    profiles = (
        ("grid", "zero", "grid", None),
        ("carbon", "zero", "carbon-minimizing", THRESHOLD),
        ("solar", "pv20", "carbon-minimizing", THRESHOLD),
        ("solar_wind", "pv20_wind20", "carbon-minimizing", THRESHOLD),
    )
    for stage, energy, budget, threshold in profiles:
        for policy in ("fifo", "priority", "eft", "puzzle"):
            name = f"{stage}_{policy}"
            selected_threshold = 2000.0 if stage == "grid" and policy == "puzzle" else threshold
            extra = (() if selected_threshold is None else
                     ("--carbon-grid-threshold", str(selected_threshold)))
            matrix.append(base.Case(
                name=name, group=stage, system="deucalion_test",
                policy=("puzzle" if policy == "puzzle" else f"{policy}_{budget}"),
                energy=energy, workload=WORKLOAD,
                reference=(f"{stage}_fifo" if policy not in ("fifo", "puzzle") else
                           f"{stage}_eft" if policy == "puzzle" else None),
                extra=extra,
            ))
    for policy in ("fifo", "priority"):
        matrix.append(base.Case(
            name=f"dvfs_{policy}_fixed", group="dvfs", system="deucalion_test",
            policy=f"{policy}_carbon_fixed", energy="pv20", workload=WORKLOAD,
            reference=f"solar_{policy}",
            extra=("--carbon-grid-threshold", str(THRESHOLD)),
        ))
    return {case.name: case for case in matrix}


def configure() -> None:
    base.POLICIES = previous.POLICIES
    base.ENERGY = {
        "zero": previous.ENERGY["zero"],
        "pv20": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=20000"),
        "pv20_wind20": (
            "--renewable-sources", "pv,wind", "--renewable-source-scale", "pv=20000",
            "--renewable-source-scale", "wind=20000",
        ),
    }
    common = list(base.COMMON)
    common.remove("--no-snapshot")
    common.remove("--sla-grid-override")
    for key, value in (("--grid-power-limit", "200000"),
                       ("--battery-capacity-kwh", "200")):
        common[common.index(key) + 1] = value
    base.COMMON = tuple(common)
    base.SOURCE_FILES = (
        ROOT / "scripts" / "simulation_energy.py",
        ROOT / "scripts" / "evaluation_campaign.py",
        ROOT / "scripts" / "evaluation_campaign_v5.py",
        Path(__file__).resolve(),
    )
    base.prepare_workloads = prepare_workload
    base.jobs_dir_for = jobs_dir_for
    base.campaign_cases = cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("cases", nargs="+")
    run.add_argument("--execute-approved", action="store_true")
    run.add_argument("--timeout-s", type=float, default=900)
    run.add_argument("--stall-s", type=float, default=300)
    run.add_argument("--max-run-gb", type=float, default=1.0)
    run.add_argument("--max-campaign-gb", type=float, default=5)
    run.add_argument("--min-free-gb", type=float, default=10)
    sub.add_parser("collect")
    args = parser.parse_args()
    configure()
    if args.action == "list":
        for case in cases().values():
            print(case.name, case.policy, case.energy)
    elif args.action == "prepare":
        paths = prepare_workload(args.campaign_dir)
        for name, path in paths.items():
            print(name, path, len(base.expected_job_keys(path)))
    elif args.action == "run":
        if not args.execute_approved:
            parser.error("No simulations started without --execute-approved")
        unknown = set(args.cases) - cases().keys()
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        for name in args.cases:
            metadata = base.run_case(
                cases()[name], args.campaign_dir, timeout_s=args.timeout_s,
                stall_s=args.stall_s, max_run_gb=args.max_run_gb,
                max_campaign_gb=args.max_campaign_gb, min_free_gb=args.min_free_gb,
            )
            record = json.loads(metadata.read_text())
            subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts" / "evaluation_campaign_v6_native_plots.py"),
                 record["result_dir"]], cwd=ROOT, check=True, timeout=120,
            )
    else:
        base.collect(args.campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
