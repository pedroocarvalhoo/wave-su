"""Chapter 5 energy-first campaign. Listing and preflight never run simulations.

The shared runner supplies source snapshots, complete-job checks, watchdogs and
storage limits. No case is started without an explicit --execute-approved flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import evaluation_campaign as base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "outputs" / "evaluation_campaign" / "v5"
THRESHOLD = 250.0

ENERGY = {
    "zero": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=0"),
    "pv40": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=40000"),
    "pv40_wind40": (
        "--renewable-sources", "pv,wind",
        "--renewable-source-scale", "pv=40000",
        "--renewable-source-scale", "wind=40000",
    ),
}


def policy_args(scheduling: str, budget: str, *, fixed: bool = False) -> tuple[str, ...]:
    if scheduling == "puzzle":
        return (
            "--budget-policy", "puzzle", "--allocation-strategy", "puzzle-plan",
            "--scheduling-policy", "puzzle", "--dvfs-policy", "adaptive",
            "--no-puzzle-fallback-dispatch",
        )
    return (
        "--budget-policy", budget,
        "--allocation-strategy", "compute-prioritized",
        "--scheduling-policy", "eft-dvfs" if scheduling == "eft" else scheduling,
        "--dvfs-policy", "fixed" if fixed else "adaptive",
    )


POLICIES = {
    f"{scheduling}_{budget}": policy_args(scheduling, budget)
    for budget in ("grid", "carbon-minimizing")
    for scheduling in ("fifo", "priority", "eft")
}
POLICIES["puzzle"] = policy_args("puzzle", "puzzle")
for scheduling in ("fifo", "priority"):
    POLICIES[f"{scheduling}_carbon_fixed"] = policy_args(
        scheduling, "carbon-minimizing", fixed=True
    )


def campaign_cases() -> dict[str, base.Case]:
    cases: list[base.Case] = []
    profiles = (
        ("grid", "zero", "grid", None),
        ("carbon", "zero", "carbon-minimizing", THRESHOLD),
        ("solar", "pv40", "carbon-minimizing", THRESHOLD),
        ("solar_wind", "pv40_wind40", "carbon-minimizing", THRESHOLD),
    )
    for stage, energy, budget, threshold in profiles:
        threshold_args = () if threshold is None else ("--carbon-grid-threshold", str(threshold))
        for scheduling in ("fifo", "priority", "eft"):
            name = f"{stage}_{scheduling}"
            cases.append(base.Case(
                name=name, group=stage, system="deucalion_test",
                policy=f"{scheduling}_{budget}", energy=energy,
                reference=f"{stage}_fifo" if scheduling != "fifo" else None,
                extra=threshold_args,
            ))
        # Puzzle uses its own budget and plan allocation. A very high carbon
        # threshold makes grid windows unrestricted in the grid-only profile.
        puzzle_threshold = 2000.0 if stage == "grid" else THRESHOLD
        cases.append(base.Case(
            name=f"{stage}_puzzle", group="puzzle", system="deucalion_test",
            policy="puzzle", energy=energy, reference=f"{stage}_eft",
            extra=("--carbon-grid-threshold", str(puzzle_threshold)),
        ))
    for scheduling in ("fifo", "priority"):
        cases.append(base.Case(
            name=f"dvfs_{scheduling}_fixed", group="dvfs",
            system="deucalion_test", policy=f"{scheduling}_carbon_fixed",
            energy="pv40", reference=f"solar_{scheduling}",
            extra=("--carbon-grid-threshold", str(THRESHOLD)),
        ))
    for scheduling in ("fifo", "eft", "puzzle"):
        cases.append(base.Case(
            name=f"pilot_carbon_{scheduling}", group="pilot",
            system="deucalion_test",
            policy=("puzzle" if scheduling == "puzzle" else f"{scheduling}_carbon-minimizing"),
            energy="zero", workload="half", reference=None,
            extra=("--carbon-grid-threshold", str(THRESHOLD)),
        ))
    result = {case.name: case for case in cases}
    if len(result) != len(cases):
        raise ValueError("Duplicate case names")
    for case in cases:
        if case.policy not in POLICIES or case.energy not in ENERGY:
            raise ValueError(f"Unknown policy or energy in {case.name}")
        if case.reference:
            other = result[case.reference]
            if (case.system, case.energy, case.workload) != (other.system, other.energy, other.workload):
                raise ValueError(f"Incompatible reference for {case.name}")
    return result


def configure_runner() -> None:
    """Bind the existing guarded runner to this campaign's immutable matrix."""
    base.POLICIES = POLICIES
    base.ENERGY = ENERGY
    base.SOURCE_FILES = (
        ROOT / "scripts" / "simulation_energy.py",
        ROOT / "scripts" / "evaluation_campaign.py",
        Path(__file__).resolve(),
    )
    base.campaign_cases = campaign_cases


def preflight() -> None:
    """Inspect input coverage and commands without writing or running a case."""
    cases = campaign_cases()
    series = pd.read_csv(ROOT / "datasets" / "historical_data.csv", parse_dates=["datetime"])
    carbon = series.set_index("datetime")["carbon_intensity"].loc["2023-06-01 12:00:00+00:00":]
    for hours in (24, 72, 168, 336):
        window = carbon.iloc[:hours]
        print(f"first {hours} h: <=170 {100 * (window <= 170).mean():.1f}%; "
              f"<=250 {100 * (window <= THRESHOLD).mean():.1f}%")
    for case in cases.values():
        jobs_dir = base.jobs_dir_for(case, DEFAULT_CAMPAIGN)
        if case.workload != "original":
            continue  # derived inputs are checked when prepare is explicitly run
        print(f"{case.name:23} jobs={len(base.expected_job_keys(jobs_dir)):3} "
              f"energy={case.energy:11} policy={case.policy}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    sub.add_parser("preflight")
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("cases", nargs="+")
    run.add_argument("--execute-approved", action="store_true",
                     help="Explicit authorization gate after the author approves this matrix")
    run.add_argument("--timeout-s", type=float, default=None)
    run.add_argument("--stall-s", type=float, default=1800)
    run.add_argument("--max-run-gb", type=float, default=3)
    run.add_argument("--max-campaign-gb", type=float, default=10)
    run.add_argument("--min-free-gb", type=float, default=10)
    sub.add_parser("collect")
    args = parser.parse_args(argv)
    cases = campaign_cases()
    if args.action == "list":
        for case in cases.values():
            print(f"{case.group:12} {case.name:23} {case.energy:11} {case.reference or '-'}")
    elif args.action == "preflight":
        preflight()
    elif args.action == "prepare":
        for name, path in base.prepare_workloads(args.campaign_dir).items():
            print(f"{name}: {path}, {len(base.expected_job_keys(path))} jobs")
    elif args.action == "run":
        for name in args.cases:
            if name not in cases:
                parser.error(f"Unknown case: {name}")
        if not args.execute_approved:
            parser.error("No simulation started: author approval requires --execute-approved")
        configure_runner()
        for name in args.cases:
            base.run_case(
                cases[name], args.campaign_dir,
                timeout_s=args.timeout_s, stall_s=args.stall_s,
                max_run_gb=args.max_run_gb,
                max_campaign_gb=args.max_campaign_gb,
                min_free_gb=args.min_free_gb,
            )
    else:
        configure_runner()
        base.collect(args.campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
