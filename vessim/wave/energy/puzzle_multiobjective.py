"""Constrained multi-scenario selection for Puzzle Scheduler experiments.

The scheduler itself cannot know a workload's final makespan or P95 wait while
it is constructing one forecast board.  Treating those global service metrics
as another per-slot greedy weight would therefore hide the actual trade-off.
This module keeps the online planner unchanged and selects a configuration from
completed, reproducible simulations instead:

1. compare every candidate with the reference configuration in every scenario;
2. reject candidates that violate correctness, makespan, or P95 constraints;
3. minimise mean relative CO2 only across the candidates that remain feasible.

The command-line interface consumes a JSON manifest and writes a detailed CSV
plus a machine-readable selection JSON.  See ``--help`` for the manifest shape.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class PuzzleRunMetrics:
    """Service, carbon, energy, and correctness metrics for one simulation."""

    results_dir: str
    run_tag: str | None
    supercomputer: str
    completed_jobs: int
    deadline_misses: int
    co2_kg: float
    it_energy_kwh: float
    physical_grid_import_kwh: float
    physical_grid_export_kwh: float
    attributed_grid_kwh: float
    mean_wait_h: float
    p95_wait_h: float
    makespan_h: float
    workload_end: str
    simulation_end: str
    green_dispatches: int
    grid_dispatches: int
    fallback_dispatches: int
    planned_misses: int
    budget_violation_samples: int
    budget_max_excess_w: float
    source_identity_max_error_w: float
    monitor_baseline_co2_kg: float = 0.0


@dataclass(frozen=True)
class MultiObjectiveLimits:
    """Hard limits applied relative to the reference in each scenario."""

    max_makespan_regression_h: float = 0.0
    max_p95_regression_h: float = 0.0
    max_mean_wait_regression_h: float | None = None
    source_identity_tolerance_w: float = 1e-6
    require_no_deadline_misses: bool = True
    require_no_budget_violations: bool = True
    require_all_reference_jobs: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_makespan_regression_h",
            "max_p95_regression_h",
            "source_identity_tolerance_w",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.max_mean_wait_regression_h is not None
            and self.max_mean_wait_regression_h < 0
        ):
            raise ValueError("max_mean_wait_regression_h must be non-negative")


@dataclass(frozen=True)
class CandidateSelection:
    """Aggregate result for one candidate across all benchmark scenarios."""

    candidate: str
    feasible: bool
    mean_co2_ratio: float
    worst_co2_ratio: float
    mean_makespan_ratio: float
    mean_p95_ratio: float
    violations: tuple[str, ...]


@dataclass(frozen=True)
class MultiScenarioSelection:
    """Selected candidate and the complete aggregate comparison."""

    reference_candidate: str
    selected_candidate: str
    candidates: tuple[CandidateSelection, ...]


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return pd.read_csv(path, **kwargs)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _job_metric_files(results_dir: Path) -> list[Path]:
    files = sorted(results_dir.glob("jobs_metrics_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No jobs_metrics_*.csv files found in {results_dir}"
        )
    return files


def extract_run_metrics(results_dir: str | Path) -> PuzzleRunMetrics:
    """Extract comparable Puzzle metrics from one completed result directory."""

    root = Path(results_dir).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Required result file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    jobs = pd.concat(
        [_read_csv(path) for path in _job_metric_files(root)],
        ignore_index=True,
    )
    wait_s = _numeric(jobs["Wait_Time_s"])
    runtime_s = _numeric(jobs["Real_Exec_Time_s"])
    submissions = pd.to_datetime(
        jobs["Submission_Time"], format="mixed", errors="raise"
    )
    completion_times = submissions + pd.to_timedelta(wait_s + runtime_s, unit="s")
    workload_end = completion_times.max()
    sim_start = pd.Timestamp(config["sim_start"])

    missed = jobs["Queue_Deadline_Missed"] if "Queue_Deadline_Missed" in jobs else False
    if isinstance(missed, pd.Series):
        deadline_misses = int(
            missed.astype(str).str.strip().str.lower().isin({"true", "1"}).sum()
        )
    else:
        deadline_misses = 0

    energy = _read_csv(root / "energy_simulation_result.csv", index_col=0)
    if energy.empty:
        raise ValueError(f"energy_simulation_result.csv is empty in {root}")
    energy.index = pd.to_datetime(energy.index, format="mixed", errors="raise")
    simulation_end = energy.index[-1]

    # e_delta is the physical energy exchanged with the grid during a monitor
    # step (J): negative means import, positive means export.
    e_delta = _numeric(energy["e_delta"])
    physical_import_kwh = float(-e_delta.clip(upper=0.0).sum() / 3_600_000.0)
    physical_export_kwh = float(e_delta.clip(lower=0.0).sum() / 3_600_000.0)

    # Monitor.step adds a fixed 0.001 Wh-equivalent carbon term per record.
    # Separate that term from physical grid-import emissions in both plotted
    # and --no-plots runs, including archived monitor files.
    monitor_co2_g = _numeric(energy["co2"])
    import_wh = -e_delta.clip(upper=0.0) / 3600.0
    grid_carbon_fraction = import_wh / (import_wh + 0.001)
    grid_co2_g = monitor_co2_g * grid_carbon_fraction
    baseline_co2_g = monitor_co2_g - grid_co2_g
    co2_kg = float(grid_co2_g.sum() / 1000.0)
    monitor_baseline_co2_kg = float(baseline_co2_g.sum() / 1000.0)

    power_columns = [
        column for column in energy.columns if column.endswith("_power")
    ]
    actual_it_w = (
        energy[power_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        if power_columns
        else pd.Series(0.0, index=energy.index)
    )
    scheduler_budget_w = _numeric(energy["budget_scheduler_w"])
    budget_excess_w = (actual_it_w - scheduler_budget_w).clip(lower=0.0)
    budget_epsilon_w = 1e-6

    pue = _numeric(energy["budget_pue"]).where(lambda values: values > 0.0)
    pue = pue.ffill().bfill().fillna(1.0)
    source_columns = (
        "budget_renewable_w",
        "budget_battery_w",
        "budget_grid_w",
    )
    source_facility_w = sum(_numeric(energy[column]) for column in source_columns)
    source_identity_error_w = (source_facility_w / pue - scheduler_budget_w).abs()

    carbon_path = root / "carbon_metrics.csv"
    if carbon_path.is_file():
        # Its cumulative CO2 includes the monitor baseline, so use only its
        # energy and source-attribution columns here.
        carbon = _read_csv(carbon_path)
        if carbon.empty:
            raise ValueError(f"carbon_metrics.csv is empty in {root}")
        carbon_last = carbon.iloc[-1]
        attributed_columns = [
            column
            for column in carbon.columns
            if column.endswith("_grid_energy_kwh_cumulative")
        ]
        it_energy_kwh = float(carbon_last["total_energy_kwh"])
        attributed_grid_kwh = float(carbon_last[attributed_columns].sum())
    else:
        # ``--no-plots`` intentionally skips carbon_metrics.csv.  Reconstruct
        # the same values from the monitor so tuning does not depend on plot
        # generation.  Scheduler energy columns are cumulative kWh; the source
        # attribution follows the budget mix used by plot_carbon_analysis().
        cumulative_energy_columns = [
            column for column in energy.columns if column.endswith("_energy_kwh")
        ]
        cumulative_it_energy = energy[cumulative_energy_columns].apply(
            pd.to_numeric, errors="coerce"
        ).ffill().fillna(0.0)
        monitor_step_size_s = float(config.get("microgrid_step_size_s", 60.0))
        it_energy_kwh = float(
            actual_it_w.sum() * monitor_step_size_s / 3_600_000.0
        )
        step_it_energy_kwh = cumulative_it_energy.diff().fillna(
            cumulative_it_energy.iloc[0]
        ).clip(lower=0.0).sum(axis=1)
        source_total_w = source_facility_w.where(source_facility_w > 0.0)
        grid_fraction = (
            _numeric(energy["budget_grid_w"]) / source_total_w
        ).fillna(0.0)
        attributed_grid_kwh = float((step_it_energy_kwh * grid_fraction).sum())

    dispatch_path = root / "puzzle_dispatch_summary.csv"
    if dispatch_path.is_file():
        dispatch = _read_csv(dispatch_path)
        dispatch_sum = dispatch.select_dtypes(include="number").sum()
    else:
        dispatch_sum = pd.Series(dtype=float)

    return PuzzleRunMetrics(
        results_dir=str(root),
        run_tag=config.get("run_tag"),
        supercomputer=str(config.get("supercomputer", "")),
        completed_jobs=len(jobs),
        deadline_misses=deadline_misses,
        co2_kg=co2_kg,
        it_energy_kwh=it_energy_kwh,
        physical_grid_import_kwh=physical_import_kwh,
        physical_grid_export_kwh=physical_export_kwh,
        attributed_grid_kwh=attributed_grid_kwh,
        mean_wait_h=float(wait_s.mean() / 3600.0),
        p95_wait_h=float(wait_s.quantile(0.95) / 3600.0),
        makespan_h=float((workload_end - sim_start).total_seconds() / 3600.0),
        workload_end=workload_end.isoformat(sep=" "),
        simulation_end=simulation_end.isoformat(sep=" "),
        green_dispatches=int(dispatch_sum.get("green_dispatched", 0)),
        grid_dispatches=int(dispatch_sum.get("grid_dispatched", 0)),
        fallback_dispatches=int(dispatch_sum.get("fallback_dispatched", 0)),
        planned_misses=int(dispatch_sum.get("planned_missed", 0)),
        budget_violation_samples=int((budget_excess_w > budget_epsilon_w).sum()),
        budget_max_excess_w=float(budget_excess_w.max()),
        source_identity_max_error_w=float(source_identity_error_w.max()),
        monitor_baseline_co2_kg=monitor_baseline_co2_kg,
    )


def constraint_violations(
    candidate: PuzzleRunMetrics,
    reference: PuzzleRunMetrics,
    limits: MultiObjectiveLimits,
) -> tuple[str, ...]:
    """Return hard-constraint violations for one scenario comparison."""

    violations: list[str] = []
    if (
        limits.require_all_reference_jobs
        and candidate.completed_jobs < reference.completed_jobs
    ):
        violations.append(
            f"completed_jobs={candidate.completed_jobs} < {reference.completed_jobs}"
        )
    if limits.require_no_deadline_misses and candidate.deadline_misses > 0:
        violations.append(f"deadline_misses={candidate.deadline_misses}")
    if candidate.planned_misses > 0:
        violations.append(f"planned_misses={candidate.planned_misses}")
    if (
        candidate.makespan_h
        > reference.makespan_h + limits.max_makespan_regression_h + 1e-12
    ):
        violations.append(
            "makespan_regression_h="
            f"{candidate.makespan_h - reference.makespan_h:.6f}"
        )
    if candidate.p95_wait_h > reference.p95_wait_h + limits.max_p95_regression_h + 1e-12:
        violations.append(
            f"p95_regression_h={candidate.p95_wait_h - reference.p95_wait_h:.6f}"
        )
    if (
        limits.max_mean_wait_regression_h is not None
        and candidate.mean_wait_h
        > reference.mean_wait_h + limits.max_mean_wait_regression_h + 1e-12
    ):
        violations.append(
            f"mean_wait_regression_h={candidate.mean_wait_h - reference.mean_wait_h:.6f}"
        )
    if limits.require_no_budget_violations and candidate.budget_violation_samples > 0:
        violations.append(
            f"budget_violation_samples={candidate.budget_violation_samples}"
        )
    if candidate.source_identity_max_error_w > limits.source_identity_tolerance_w:
        violations.append(
            "source_identity_max_error_w="
            f"{candidate.source_identity_max_error_w:.12g}"
        )
    return tuple(violations)


def _safe_ratio(value: float, reference: float) -> float:
    if reference > 0.0:
        return value / reference
    return 1.0 if value == 0.0 else math.inf


def select_multi_scenario(
    runs: Mapping[str, Mapping[str, PuzzleRunMetrics]],
    reference_candidate: str,
    limits: MultiObjectiveLimits | None = None,
) -> MultiScenarioSelection:
    """Select the lowest-carbon feasible candidate across every scenario.

    ``runs`` maps ``scenario -> candidate -> metrics``.  Every scenario must
    contain the reference and every candidate must cover the same scenarios;
    silently comparing different validation matrices would bias the mean.
    """

    limits = limits or MultiObjectiveLimits()
    if not runs:
        raise ValueError("at least one scenario is required")
    scenarios = tuple(sorted(runs))
    for scenario in scenarios:
        if reference_candidate not in runs[scenario]:
            raise ValueError(
                f"reference candidate '{reference_candidate}' missing from scenario '{scenario}'"
            )

    candidate_names = set(runs[scenarios[0]])
    for scenario in scenarios[1:]:
        if set(runs[scenario]) != candidate_names:
            raise ValueError("every scenario must contain the same candidates")

    summaries: list[CandidateSelection] = []
    for candidate_name in sorted(candidate_names):
        co2_ratios: list[float] = []
        makespan_ratios: list[float] = []
        p95_ratios: list[float] = []
        violations: list[str] = []
        for scenario in scenarios:
            candidate = runs[scenario][candidate_name]
            reference = runs[scenario][reference_candidate]
            co2_ratios.append(_safe_ratio(candidate.co2_kg, reference.co2_kg))
            makespan_ratios.append(
                _safe_ratio(candidate.makespan_h, reference.makespan_h)
            )
            p95_ratios.append(_safe_ratio(candidate.p95_wait_h, reference.p95_wait_h))
            violations.extend(
                f"{scenario}: {violation}"
                for violation in constraint_violations(candidate, reference, limits)
            )
        summaries.append(
            CandidateSelection(
                candidate=candidate_name,
                feasible=not violations,
                mean_co2_ratio=sum(co2_ratios) / len(co2_ratios),
                worst_co2_ratio=max(co2_ratios),
                mean_makespan_ratio=sum(makespan_ratios) / len(makespan_ratios),
                mean_p95_ratio=sum(p95_ratios) / len(p95_ratios),
                violations=tuple(violations),
            )
        )

    feasible = [summary for summary in summaries if summary.feasible]
    if not feasible:
        raise ValueError("no candidate satisfies the configured hard constraints")
    selected = min(
        feasible,
        key=lambda summary: (
            summary.mean_co2_ratio,
            summary.worst_co2_ratio,
            summary.mean_makespan_ratio,
            summary.mean_p95_ratio,
            summary.candidate,
        ),
    )
    return MultiScenarioSelection(
        reference_candidate=reference_candidate,
        selected_candidate=selected.candidate,
        candidates=tuple(summaries),
    )


def _load_manifest(path: Path) -> tuple[
    dict[str, dict[str, PuzzleRunMetrics]],
    str,
    MultiObjectiveLimits,
]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    reference_candidate = str(raw["reference_candidate"])
    limits = MultiObjectiveLimits(**raw.get("limits", {}))
    rows = raw.get("runs", [])
    if not rows:
        raise ValueError("manifest must contain at least one run")

    runs: dict[str, dict[str, PuzzleRunMetrics]] = {}
    for row in rows:
        scenario = str(row["scenario"])
        candidate = str(row["candidate"])
        scenario_runs = runs.setdefault(scenario, {})
        if candidate in scenario_runs:
            raise ValueError(
                f"duplicate candidate '{candidate}' in scenario '{scenario}'"
            )
        result_path = Path(row["results_dir"])
        if not result_path.is_absolute():
            result_path = (path.parent / result_path).resolve()
        scenario_runs[candidate] = extract_run_metrics(result_path)
    return runs, reference_candidate, limits


def _comparison_rows(
    runs: Mapping[str, Mapping[str, PuzzleRunMetrics]],
    selection: MultiScenarioSelection,
    limits: MultiObjectiveLimits,
) -> list[dict[str, Any]]:
    aggregate = {summary.candidate: summary for summary in selection.candidates}
    rows: list[dict[str, Any]] = []
    for scenario in sorted(runs):
        reference = runs[scenario][selection.reference_candidate]
        for candidate in sorted(runs[scenario]):
            metrics = runs[scenario][candidate]
            violations = constraint_violations(metrics, reference, limits)
            rows.append(
                {
                    "scenario": scenario,
                    "candidate": candidate,
                    "reference_candidate": selection.reference_candidate,
                    "selected": candidate == selection.selected_candidate,
                    "feasible_all_scenarios": aggregate[candidate].feasible,
                    "scenario_feasible": not violations,
                    "constraint_violations": "; ".join(violations),
                    "co2_delta_kg": metrics.co2_kg - reference.co2_kg,
                    "co2_ratio": _safe_ratio(metrics.co2_kg, reference.co2_kg),
                    "makespan_delta_h": metrics.makespan_h - reference.makespan_h,
                    "p95_delta_h": metrics.p95_wait_h - reference.p95_wait_h,
                    "mean_wait_delta_h": metrics.mean_wait_h - reference.mean_wait_h,
                    **asdict(metrics),
                }
            )
    return rows


def write_selection_outputs(
    output_dir: str | Path,
    runs: Mapping[str, Mapping[str, PuzzleRunMetrics]],
    selection: MultiScenarioSelection,
    limits: MultiObjectiveLimits,
) -> tuple[Path, Path]:
    """Write the detailed comparison CSV and aggregate selection JSON."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "puzzle_multiobjective_runs.csv"
    json_path = root / "puzzle_multiobjective_selection.json"
    pd.DataFrame(_comparison_rows(runs, selection, limits)).to_csv(
        csv_path, index=False
    )
    json_path.write_text(
        json.dumps(
            {
                "reference_candidate": selection.reference_candidate,
                "selected_candidate": selection.selected_candidate,
                "limits": asdict(limits),
                "candidates": [asdict(summary) for summary in selection.candidates],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a Puzzle configuration by minimum CO2 subject to explicit "
            "makespan, P95, deadline, and budget constraints."
        ),
        epilog=(
            'Manifest: {"reference_candidate":"protected", "limits":{}, '
            '"runs":[{"scenario":"small-pv", "candidate":"protected", '
            '"results_dir":"results/..."}]}.'
        ),
    )
    parser.add_argument("manifest", type=Path, help="JSON experiment manifest")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to a multiobjective/ sibling of the manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    runs, reference_candidate, limits = _load_manifest(manifest)
    selection = select_multi_scenario(runs, reference_candidate, limits)
    output_dir = args.output_dir or manifest.parent / "multiobjective"
    csv_path, json_path = write_selection_outputs(
        output_dir, runs, selection, limits
    )
    print(f"Selected candidate: {selection.selected_candidate}")
    print(f"Detailed comparison: {csv_path}")
    print(f"Selection summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
