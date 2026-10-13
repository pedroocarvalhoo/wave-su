"""Five-day full-Deucalion evaluation with powered-on idle nodes and guarded runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluation_campaign as base
from scripts import evaluation_campaign_v5 as previous

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs" / "evaluation_campaign" / "v8"
SOURCE = ROOT / "datasets" / "jobs" / "deucalion_macc"
WORKLOAD = "macc_five_day"
ARRIVAL_END_S = 96 * 3600
MAX_BASE_RUNTIME_S = 6 * 3600
THRESHOLD = 250.0
FREQUENCIES = {
    name: config["base_frequency_ghz"]
    for name, config in json.loads((ROOT / "datasets" / "node_config.json").read_text()).items()
}


def prepare_workload(campaign_dir: Path) -> dict[str, Path]:
    """Preserve jobs/queues, compress arrivals and cap base-frequency work."""
    target = campaign_dir / "inputs" / "deucalion_macc_five_day"
    target.mkdir(parents=True, exist_ok=True)
    files = base.workload_files(SOURCE)
    frames = {path: pd.read_csv(path) for path in files}
    final_arrival = max(int(frame["Arrival_Offset_s"].max()) for frame in frames.values())
    if final_arrival <= 0:
        raise ValueError("Source workload has no arrival span")
    count = 0
    clipped = 0
    for input_file, frame in frames.items():
        cluster = input_file.stem.split("jobs_", 1)[1].rsplit("_", 1)[0]
        frequency = FREQUENCIES[cluster]
        output = frame.copy()
        output["Arrival_Offset_s"] = (
            output["Arrival_Offset_s"] * ARRIVAL_END_S / final_arrival
        ).round().astype("int64")
        max_instructions = (MAX_BASE_RUNTIME_S * frequency * 1e9 / output["CPI"])
        clipped += int((output["#I"] > max_instructions).sum())
        output["#I"] = np.minimum(output["#I"], max_instructions).round().astype("int64")
        path = target / input_file.name
        generated = output.to_csv(index=False).encode()
        if path.exists() and path.read_bytes() != generated:
            raise ValueError(f"Derived workload changed: {path}; use a new campaign directory")
        if not path.exists():
            path.write_bytes(generated)
        count += len(output)
    if count != 2523:
        raise ValueError(f"Expected 2523 jobs, found {count}")
    provenance = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_job_count": count,
        "arrival_end_s": ARRIVAL_END_S,
        "source_arrival_end_s": final_arrival,
        "max_base_runtime_s": MAX_BASE_RUNTIME_S,
        "clipped_instruction_rows": clipped,
        "source_hash": base.hash_files(files),
        "derived_hash": base.hash_files(base.workload_files(target)),
    }
    record = campaign_dir / "inputs" / "derivation.json"
    payload = json.dumps(provenance, indent=2) + "\n"
    if record.exists() and record.read_text() != payload:
        raise ValueError("Derived workload provenance changed; use a new campaign directory")
    record.write_text(payload)
    return {WORKLOAD: target}


def jobs_dir_for(case: base.Case, campaign_dir: Path) -> Path:
    if case.workload != WORKLOAD:
        raise ValueError(f"Unexpected workload: {case.workload}")
    return campaign_dir / "inputs" / "deucalion_macc_five_day"


def cases() -> dict[str, base.Case]:
    matrix = []
    profiles = (
        ("grid", "zero", "grid", None),
        ("solar", "pv400", "carbon-minimizing", THRESHOLD),
        ("solar_wind", "pv400_wind400", "carbon-minimizing", THRESHOLD),
    )
    for stage, energy, budget, threshold in profiles:
        for policy in ("fifo", "priority", "eft", "puzzle"):
            selected_threshold = 2000.0 if stage == "grid" and policy == "puzzle" else threshold
            extra = []
            if selected_threshold is not None:
                extra.extend(("--carbon-grid-threshold", str(selected_threshold)))
            if stage == "grid" and policy == "puzzle":
                # Grid is available at every time in this baseline. Keep the
                # Puzzle placement on the first node-feasible slot instead of
                # searching the carbon trace for a later start.
                extra.append("--no-puzzle-carbon-aware-placement")
            # The grid-only baseline must be unconstrained by the experimental
            # 350 kW cap used by renewable/carbon-limited profiles. The modeled
            # Deucalion peak is 688.06 kW IT, or 894.48 kW at PUE 1.3.
            if stage == "grid":
                extra.extend(("--grid-power-limit", "900000"))
            matrix.append(base.Case(
                name=f"{stage}_{policy}", group=stage, system="deucalion_macc",
                policy=("puzzle" if policy == "puzzle" else f"{policy}_{budget}"),
                energy=energy, workload=WORKLOAD,
                reference=(f"{stage}_fifo" if policy not in ("fifo", "puzzle") else
                           f"{stage}_eft" if policy == "puzzle" else None),
                extra=tuple(extra),
            ))
    for policy in ("fifo", "priority"):
        matrix.append(base.Case(
            name=f"dvfs_{policy}_fixed", group="dvfs", system="deucalion_macc",
            policy=f"{policy}_carbon_fixed", energy="pv400", workload=WORKLOAD,
            reference=f"solar_{policy}",
            extra=("--carbon-grid-threshold", str(THRESHOLD)),
        ))
    matrix.append(base.Case(
        name="battery_grid_charge", group="battery", system="deucalion_macc",
        policy="fifo_carbon-minimizing", energy="pv400", workload=WORKLOAD,
        reference="solar_fifo",
        extra=("--carbon-grid-threshold", str(THRESHOLD), "--allow-grid-charging"),
    ))
    return {case.name: case for case in matrix}


def configure() -> None:
    base.POLICIES = previous.POLICIES
    base.ENERGY = {
        "zero": previous.ENERGY["zero"],
        "pv400": ("--renewable-sources", "pv", "--renewable-source-scale", "pv=400000"),
        "pv400_wind400": (
            "--renewable-sources", "pv,wind",
            "--renewable-source-scale", "pv=400000",
            "--renewable-source-scale", "wind=400000",
        ),
    }
    common = list(base.COMMON)
    for flag in ("--no-snapshot", "--sla-grid-override"):
        common.remove(flag)
    common += ["--start-nodes-on", "--keep-idle-nodes-on"]
    for key, value in (("--grid-power-limit", "350000"),
                       ("--battery-capacity-kwh", "2000")):
        common[common.index(key) + 1] = value
    base.COMMON = tuple(common)
    base.SOURCE_FILES = (
        ROOT / "scripts" / "simulation_energy.py",
        ROOT / "vessim" / "wave" / "resource_manager" / "manager.py",
        ROOT / "vessim" / "wave" / "scheduler" / "cluster_scheduler.py",
        ROOT / "scripts" / "evaluation_campaign.py",
        ROOT / "scripts" / "evaluation_campaign_v5.py",
        Path(__file__).resolve(),
    )
    base.prepare_workloads = prepare_workload
    base.jobs_dir_for = jobs_dir_for
    base.campaign_cases = cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("cases", nargs="+")
    run.add_argument("--execute-approved", action="store_true")
    run.add_argument("--timeout-s", type=float, default=1800)
    run.add_argument("--stall-s", type=float, default=300)
    run.add_argument("--max-run-gb", type=float, default=2)
    run.add_argument("--max-campaign-gb", type=float, default=8)
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
            base.run_case(
                cases()[name], args.campaign_dir, timeout_s=args.timeout_s,
                stall_s=args.stall_s, max_run_gb=args.max_run_gb,
                max_campaign_gb=args.max_campaign_gb, min_free_gb=args.min_free_gb,
            )
    else:
        base.collect(args.campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
