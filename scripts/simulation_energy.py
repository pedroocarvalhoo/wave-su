from __future__ import annotations

import os
import json
import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import time as _time
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datetime import datetime
from vessim.actor import ComputingSystem, Generator
from vessim.controller import Monitor
from vessim.cosim import Environment
from vessim.policy import DefaultMicrogridPolicy
from vessim.power_meter import ComputeNode
from vessim.signal import HistoricalSignal
from vessim.storage import NewBattery
from vessim.wave.cluster import JobQueue
from vessim.wave.energy import EnergyScheduler
from vessim.wave.energy.budget_policy import (
    CarbonMinimizingBudgetPolicy,
    DaytimeRenewableBudgetPolicy,
    GridBudgetPolicy,
)
from vessim.wave.energy.puzzle_budget_policy import PuzzleBudgetPolicy
from vessim.wave.energy.dvfs import *
from vessim.wave.energy.strategy import (
    DefaultAllocationStrategy,
    ComputePrioritizedStrategy,
    PuzzlePlanAllocationStrategy,
)
from vessim.wave.resource_manager import ResourceManager
from vessim.wave.scheduler import JobQueueScheduler
from vessim.wave.scheduler.cluster_scheduler import DEFAULT_SLA_GRID_LEAD_S
from vessim.wave.scheduler.policies import *
from vessim.wave.scheduler.policies.puzzle import PuzzleSchedulingPolicy
from vessim.wave.scheduler.queues import queue_definitions_from_config

import vessim.wave.scheduler.cluster_scheduler as scheduler_module

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
if DATASETS_DIR not in sys.path:
    sys.path.insert(0, DATASETS_DIR)

from supercomputer_config import (
    jobs_dataset_path,
    load_node_config as load_node_config_from_dir,
    load_supercomputers_config,
)

SIM_START = "2023-06-01 12:00:00"
SCHEDULERS_FILE = "schedulers.pkl"
RENEWABLE_SOURCES_FILE = "renewable_sources.json"
DEFAULT_PUE = 1.3
DEFAULT_PUZZLE_FORECAST_MAX_EXPORT_PLANS = 64
PUZZLE_ACTIVE_FORECAST_FILE = "puzzle_active_forecast.csv"

ProgressFn = Callable[[str], None]


@dataclass
class SimulationConfig:
    node_config: dict[str, Any]
    supercomputers: dict[str, Any]
    renewable_sources_config: dict[str, Any]


@dataclass
class SchedulerBuild:
    clusters: list[dict[str, Any]]
    schedulers: list[JobQueueScheduler]
    compute_nodes: list[ComputeNode]


@dataclass
class SimulationContext:
    environment: Environment
    schedulers: list[JobQueueScheduler]
    monitor: Monitor
    results_dir: str
    selected_system: str
    battery: NewBattery
    signal: HistoricalSignal
    energy_scheduler: EnergyScheduler
    config: SimulationConfig
    clusters: list[dict[str, Any]]
    selected_renewable_sources: list[str]
    renewable_source_scales_w: dict[str, float]


def log_elapsed(progress: ProgressFn, started_at: float, message: str) -> None:
    progress(f"[{_time.time() - started_at:.2f}s] {message}")


def load_config(
    datasets_dir: str = DATASETS_DIR,
    progress: ProgressFn | None = None,
    started_at: float | None = None,
) -> SimulationConfig:
    node_config = load_node_config_from_dir(datasets_dir)
    if progress is not None and started_at is not None:
        log_elapsed(progress, started_at, f"Node config loaded: {list(node_config.keys())}")

    supercomputers = load_supercomputers_config(datasets_dir)
    if progress is not None and started_at is not None:
        log_elapsed(progress, started_at, f"Supercomputers loaded: {list(supercomputers.keys())}")

    with open(os.path.join(datasets_dir, RENEWABLE_SOURCES_FILE), encoding="utf-8") as f:
        renewable_sources_config = json.load(f)
    if progress is not None and started_at is not None:
        log_elapsed(
            progress,
            started_at,
            f"Renewable sources loaded: {list(renewable_sources_config.keys())}",
        )

    return SimulationConfig(
        node_config=node_config,
        supercomputers=supercomputers,
        renewable_sources_config=renewable_sources_config,
    )


def load_clusters(supercomputers_config, system_name, sla_policy_override=None):
    if system_name not in supercomputers_config:
        raise ValueError(
            f"Supercomputer '{system_name}' not found. Available: {list(supercomputers_config.keys())}"
        )

    system_config = supercomputers_config[system_name]
    if "clusters" in system_config:
        clusters_config = system_config["clusters"]
        queue_config = system_config.get("queues", [])
        sla_policy = system_config.get("sla_policy", "enforce")
        cluster_queue_overrides = system_config.get("cluster_queue_overrides", {})
    else:
        clusters_config = system_config
        queue_config = []
        sla_policy = "enforce"
        cluster_queue_overrides = {}

    if sla_policy_override is not None:
        sla_policy = sla_policy_override

    return [
        {
            "type": cluster_type,
            "num_nodes": num_nodes,
            "queues": queue_config,
            "queue_overrides": cluster_queue_overrides.get(cluster_type, {}),
            "sla_policy": sla_policy,
        }
        for cluster_type, num_nodes in clusters_config.items()
    ]

def build_dvfs_policy(policy_name, freq_table, base_frequency):
    if policy_name == "fixed":
        return FixedFrequencyDVFSPolicy(
            frequencies=freq_table,
            base_frequency_ghz=base_frequency,
        )
    if policy_name == "max-throughput":
        return MaxThroughputDVFSPolicy(
            frequencies=freq_table,
            base_frequency_ghz=base_frequency,
        )
    if policy_name == "min-power":
        return MinPowerDVFSPolicy(
            frequencies=freq_table,
            base_frequency_ghz=base_frequency,
        )
    if policy_name == "adaptive":
        return AdaptiveDVFSPolicy(
            frequencies=freq_table,
            base_frequency_ghz=base_frequency,
        )
    raise ValueError(f"Unknown DVFS policy: {policy_name}")


def build_scheduling_policy(policy_name, cluster_mem_bw_gb_s, eft_dvfs_alpha, eft_dvfs_beta):
    if policy_name == "fifo":
        return FIFOPolicy()
    if policy_name == "priority":
        return PriorityPolicy()
    if policy_name == "sjf":
        return SJFPolicy(cluster_mem_bw_gb_s=cluster_mem_bw_gb_s)
    if policy_name == "ljf":
        return LJFPolicy(cluster_mem_bw_gb_s=cluster_mem_bw_gb_s)
    if policy_name == "eft":
        return EFTPolicy(cluster_mem_bw_gb_s=cluster_mem_bw_gb_s)
    if policy_name == "eft-dvfs":
        return EFT_DVFSPolicy(
            alpha=eft_dvfs_alpha,
            beta=eft_dvfs_beta,
            cluster_mem_bw_gb_s=cluster_mem_bw_gb_s,
        )
    if policy_name == "puzzle":
        # Placeholder: PuzzleSchedulingPolicy needs a reference to the
        # PuzzleBudgetPolicy which is created later.  build_environment()
        # replaces this placeholder with the real policy after wiring.
        return PriorityPolicy()
    raise ValueError(f"Unknown scheduling policy: {policy_name}")


def build_allocation_strategy(strategy_name, budget_policy=None):
    if strategy_name == "default":
        return DefaultAllocationStrategy()
    if strategy_name == "compute-prioritized":
        return ComputePrioritizedStrategy()
    if strategy_name == "puzzle-plan":
        if not isinstance(budget_policy, PuzzleBudgetPolicy):
            raise ValueError(
                "--allocation-strategy puzzle-plan requires --budget-policy puzzle"
            )
        return PuzzlePlanAllocationStrategy(budget_policy)
    raise ValueError(f"Unknown allocation strategy: {strategy_name}")


def effective_grid_power_limit(args) -> float:
    if args.microgrid_mode == "islanded":
        return 0.0
    return args.grid_power_limit


def build_budget_policy(args, node_config=None):
    grid_power_limit = effective_grid_power_limit(args)
    def charge_fraction(policy_default: float) -> float:
        # ``--renewable-charge-fraction`` overrides every policy; when unset
        # each policy keeps its own default (the puzzle board loses more to a
        # large reserve, so its default is lower).
        if args.renewable_charge_fraction is None:
            return policy_default
        return args.renewable_charge_fraction

    if args.budget_policy == "grid":
        return GridBudgetPolicy(grid_power_limit=grid_power_limit)
    if args.budget_policy == "daytime-renewable":
        return DaytimeRenewableBudgetPolicy(
            grid_power_limit=grid_power_limit,
            day_hours=(8, 20),
            renewable_charge_fraction=charge_fraction(0.7),
            day_grid_fraction=args.day_grid_fraction,
            night_grid_fraction=args.night_grid_fraction,
        )
    if args.budget_policy == "carbon-minimizing":
        return CarbonMinimizingBudgetPolicy(
            grid_power_limit=grid_power_limit,
            carbon_intensity_threshold_g_per_kwh=args.carbon_grid_threshold,
            day_hours=(8, 20),
            renewable_charge_fraction=charge_fraction(0.7),
            low_carbon_grid_fraction=args.low_carbon_grid_fraction,
            high_carbon_grid_fraction=args.high_carbon_grid_fraction,
        )
    if args.budget_policy == "puzzle":
        return PuzzleBudgetPolicy(
            grid_power_limit=grid_power_limit,
            carbon_intensity_threshold_g_per_kwh=args.carbon_grid_threshold,
            plan_horizon_h=24.0,
            slot_duration_s=args.energy_scheduler_step_size,
            replan_interval_h=6.0,
            renewable_charge_fraction=charge_fraction(0.3),
            day_hours=(8, 20),
            node_configs=node_config or {},
            repack_repair=getattr(args, "puzzle_repack_repair", False),
            pue=args.pue,
            carbon_aware_placement=getattr(
                args, "puzzle_carbon_aware_placement", True
            ),
            forecast_tail_carbon_multiplier=getattr(
                args, "puzzle_forecast_tail_carbon_multiplier", 1.25
            ),
            sla_start_safety_margin_h=getattr(
                args, "puzzle_sla_start_safety_margin_h", 6.0
            ),
            global_green_board=getattr(
                args, "puzzle_global_green_board", True
            ),
            carbon_regret_threshold_g_per_h=getattr(
                args, "puzzle_carbon_regret_g_per_h", 1.0
            ),
            tail_battery_reserve_fraction=getattr(
                args, "puzzle_tail_battery_reserve_fraction", 0.10
            ),
            tail_renewable_uncertainty_fraction=getattr(
                args, "puzzle_tail_renewable_uncertainty_fraction", 0.20
            ),
            tail_renewable_robustness=getattr(
                args, "puzzle_tail_renewable_robustness", 0.0
            ),
            tail_renewable_calibration_days=getattr(
                args, "puzzle_tail_renewable_calibration_days", 90.0
            ),
            battery_carbon_aware_dispatch=getattr(
                args, "puzzle_battery_carbon_aware_dispatch", True
            ),
            plan_history_limit=_puzzle_forecast_max_export_plans(),
            battery_carbon_priority_fraction=getattr(
                args, "puzzle_battery_carbon_priority_fraction", 0.0
            ),
        )
    raise ValueError(f"Unknown budget policy: {args.budget_policy}")


def resolve_allow_grid_charging(args) -> bool:
    if args.microgrid_mode == "islanded":
        return False
    if args.allow_grid_charging is not None:
        return args.allow_grid_charging
    # Puzzle treats the battery as renewable recourse.  Keep its command-line
    # default aligned with the GUI and require an explicit opt-in before grid
    # imports may be used to charge it.
    if args.budget_policy == "puzzle":
        return False
    return not (
        args.budget_policy == "carbon-minimizing"
        and args.grid_power_limit <= 0
    )


def default_renewable_sources(renewable_sources_config: dict[str, Any]) -> list[str]:
    defaults = [
        name
        for name, config in renewable_sources_config.items()
        if config.get("enabled_by_default", False)
    ]
    return defaults or list(renewable_sources_config.keys())


def parse_renewable_sources(
    raw_sources: str | None,
    renewable_sources_config: dict[str, Any],
) -> list[str]:
    sources = default_renewable_sources(renewable_sources_config) if raw_sources is None else [
        source.strip()
        for source in raw_sources.split(",")
        if source.strip()
    ]
    unknown_sources = [source for source in sources if source not in renewable_sources_config]
    if unknown_sources:
        raise ValueError(
            "Unknown renewable source(s): "
            + ", ".join(unknown_sources)
            + f". Available: {', '.join(renewable_sources_config)}"
        )
    if not sources:
        raise ValueError("At least one renewable source must be selected.")
    return sources


def parse_renewable_source_scales(
    scale_args: list[str] | None,
    pv_scale: float | None,
    renewable_sources_config: dict[str, Any],
) -> dict[str, float]:
    scales = {
        name: float(config.get("default_scale_w", 0.0))
        for name, config in renewable_sources_config.items()
    }
    if pv_scale is not None:
        scales["pv"] = float(pv_scale)

    for raw_scale in scale_args or []:
        if "=" not in raw_scale:
            raise ValueError(
                f"Invalid renewable source scale '{raw_scale}'. Use SOURCE=W, e.g. pv=400000."
            )
        source, value = raw_scale.split("=", 1)
        source = source.strip()
        if source not in renewable_sources_config:
            raise ValueError(f"Unknown renewable source in scale override: {source}")
        scales[source] = float(value)
    return scales


def build_renewable_signal(
    selected_sources: list[str],
    source_scales_w: dict[str, float],
    datasets_dir: str,
    renewable_sources_config: dict[str, Any],
) -> HistoricalSignal:
    base = pd.read_csv(
        os.path.join(datasets_dir, "historical_data.csv"),
        parse_dates=["datetime"],
    )
    actual = base[["datetime", "carbon_intensity"]].copy()
    actual = actual.set_index("datetime").sort_index()

    renewable_total = pd.Series(0.0, index=actual.index)
    for source_name in selected_sources:
        source_config = renewable_sources_config[source_name]
        dataset_path = os.path.join(datasets_dir, source_config["dataset"])
        source_df = pd.read_csv(dataset_path, parse_dates=["time"])
        source_df = source_df.set_index("time").sort_index()

        source_column = source_config.get("source_column", "P")
        signal_column = source_config.get("signal_column", f"{source_name}_gen")
        scale_w = float(source_scales_w[source_name])

        profile = source_df[source_column].reindex(actual.index, method="ffill").fillna(0.0)
        actual[signal_column] = profile * scale_w
        renewable_total = renewable_total + actual[signal_column]

    actual["renewable_gen"] = renewable_total
    return HistoricalSignal(actual=actual, fill_method="ffill")


def parse_args(
    systems: list[str],
    renewable_sources_config: dict[str, Any],
    argv: list[str] | None = None,
):
    parser = argparse.ArgumentParser(description="Run the Vessim HPC energy simulation.")
    parser.add_argument(
        "system",
        nargs="?",
        choices=systems,
        help="Supercomputer to simulate. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--jobs-dir",
        default=None,
        help="Directory containing jobs_<cluster>_<queue>.csv workload files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Run only the simulation and skip plot generation.",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip the plot-only schedulers.pkl snapshot after exporting CSV results.",
    )
    parser.add_argument(
        "--plot-workers",
        type=int,
        default=None,
        help="Number of ProcessPoolExecutor workers used for plot generation.",
    )
    parser.add_argument(
        "--adaptive-step",
        dest="adaptive_step",
        action="store_true",
        default=True,
        help="Allow schedulers to use sub-60s steps for job completions. Enabled by default.",
    )
    parser.add_argument(
        "--no-adaptive-step",
        dest="adaptive_step",
        action="store_false",
        help="Disable adaptive scheduler steps and keep the scheduler at its fixed base step.",
    )
    parser.add_argument(
        "--microgrid-step-size",
        type=int,
        default=60,
        help="Microgrid step size in seconds. Use 1 with --adaptive-step for high-resolution energy accounting.",
    )
    parser.add_argument(
        "--node-power-on-gate",
        choices=["eager", "runnable"],
        default="runnable",
        help=(
            "When the resource manager wakes idle nodes. 'runnable' (default) "
            "only powers nodes on if a job the scheduler would dispatch this "
            "step can actually run on them within the energy budget, avoiding "
            "waking the cluster for jobs a plan withholds for a greener slot. "
            "'eager' restores the legacy behaviour that powers on toward total "
            "pending node demand."
        ),
    )
    parser.add_argument(
        "--microgrid-mode",
        choices=["grid-connected", "islanded"],
        default="grid-connected",
        help=(
            "Physical microgrid mode. grid-connected permits import/export with the grid; "
            "islanded forbids grid exchange and raises if local supply cannot serve demand."
        ),
    )
    parser.add_argument(
        "--pue",
        type=float,
        default=DEFAULT_PUE,
        help=(
            "Power Usage Effectiveness applied between IT power and facility power. "
            "Energy budgets are divided by this value before reaching cluster schedulers."
        ),
    )
    parser.add_argument(
        "--allow-grid-charging",
        dest="allow_grid_charging",
        action="store_true",
        default=None,
        help="Allow battery charge commands to import power from the grid in grid-connected mode.",
    )
    parser.add_argument(
        "--no-grid-charging",
        dest="allow_grid_charging",
        action="store_false",
        help="Constrain battery charging to local renewable surplus.",
    )
    parser.add_argument(
        "--energy-scheduler-step-size",
        type=int,
        default=300,
        help="Energy budget redistribution step size in seconds.",
    )
    parser.add_argument(
        "--budget-policy",
        choices=["grid", "daytime-renewable", "carbon-minimizing", "puzzle"],
        default="carbon-minimizing",
        help="Energy budget policy used to expose compute power.",
    )
    parser.add_argument(
        "--allocation-strategy",
        choices=["default", "compute-prioritized", "puzzle-plan"],
        default="compute-prioritized",
        help=(
            "Cross-cluster energy budget allocation strategy. 'puzzle-plan' "
            "gives each cluster the share the puzzle plan assigned it "
            "(requires --budget-policy puzzle)."
        ),
    )
    parser.add_argument(
        "--scheduling-policy",
        choices=["fifo", "priority", "sjf", "ljf", "eft", "eft-dvfs", "puzzle"],
        default="eft-dvfs",
        help="Per-cluster job scheduling policy.",
    )
    parser.add_argument(
        "--dvfs-policy",
        choices=["fixed", "max-throughput", "adaptive", "min-power"],
        default="adaptive",
        help="Node frequency-selection policy.",
    )
    parser.add_argument(
        "--puzzle-allow-front-run",
        dest="puzzle_allow_front_run",
        action="store_true",
        help=(
            "For --scheduling-policy puzzle: let the dispatcher fill free "
            "capacity with jobs the plan scheduled for a future slot "
            "(greedy, higher utilization) instead of steering strictly by "
            "the plan's timing. Default: plan-steered."
        ),
    )
    parser.add_argument(
        "--puzzle-fallback-dispatch",
        dest="puzzle_fallback_dispatch",
        action="store_true",
        default=None,
        help=(
            "For --scheduling-policy puzzle: allow jobs deliberately left "
            "unplaced by the plan to backfill the current power budget. The "
            "legacy planner enables this by default; carbon-aware placement "
            "disables it because a long fallback job can create a dirty live "
            "power floor after the current green slot."
        ),
    )
    parser.add_argument(
        "--no-puzzle-fallback-dispatch",
        dest="puzzle_fallback_dispatch",
        action="store_false",
        help="Disable unplanned Puzzle fallback dispatch explicitly.",
    )
    parser.add_argument(
        "--sla-grid-override",
        dest="sla_grid_override",
        action="store_true",
        default=None,
        help=(
            "SLA floor: let a job whose start deadline is approaching draw grid "
            "beyond the (carbon-limited) energy budget instead of being held "
            "until it misses. Trades some carbon for deadline safety under "
            "scarce/carbon-minimizing budgets. Unset: enabled automatically for "
            "budget policies with no SLA mechanism of their own, whenever the "
            "system declares sla_policy=enforce — PuzzleBudgetPolicy is left "
            "alone because it already plans an sla_emergency grid escape."
        ),
    )
    parser.add_argument(
        "--no-sla-grid-override",
        dest="sla_grid_override",
        action="store_false",
        help=(
            "Disable the SLA grid floor everywhere, including for budget "
            "policies that would otherwise get it automatically. Use to "
            "reproduce SLA-blind baseline behaviour."
        ),
    )
    parser.add_argument(
        "--sla-policy",
        dest="sla_policy",
        choices=["enforce", "report_only"],
        default=None,
        help=(
            "Override the system's sla_policy. 'report_only' makes every "
            "scheduling policy ignore start deadlines when ordering jobs, "
            "reproducing the SLA-blind baselines that predate queue deadlines; "
            "misses are still measured. Defaults to the system JSON."
        ),
    )
    parser.add_argument(
        "--sla-grid-lead-s",
        dest="sla_grid_lead_s",
        type=float,
        default=DEFAULT_SLA_GRID_LEAD_S,
        help=(
            "Lead time before a start deadline at which the SLA grid floor "
            "begins releasing a job, so critical jobs drain gradually instead "
            "of stampeding at the last step. Defaults to Puzzle's own 6h "
            "sla_start_safety_margin_h so baselines get the same lead time."
        ),
    )
    parser.add_argument(
        "--run-tag",
        dest="run_tag",
        default=None,
        help=(
            "Optional label appended to the results directory name and recorded "
            "in config.json. Guarantees a unique output directory for runs that "
            "share the same policy signature (e.g. puzzle front-run vs repair), "
            "which otherwise collide when started in the same second."
        ),
    )
    parser.add_argument(
        "--puzzle-repack-repair",
        dest="puzzle_repack_repair",
        action="store_true",
        help=(
            "For --budget-policy puzzle: after the greedy packer, run a "
            "local-search repair that pulls grid-spilled jobs back onto the "
            "green board, relocating already-placed jobs within their SLA slack "
            "to make room (a 1-opt hill-climb). Reduces grid spill without "
            "risking deadlines. Default: off."
        ),
    )
    parser.add_argument(
        "--puzzle-carbon-aware-placement",
        dest="puzzle_carbon_aware_placement",
        action="store_true",
        default=True,
        help=(
            "For --budget-policy puzzle: score partial-green placements by "
            "forecast grid CO2 across every DVFS variant and start, enforce "
            "the per-slot low-carbon grid cap, and classify deliberate "
            "high-carbon exceptions as carbon-regret or SLA emergency. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-puzzle-carbon-aware-placement",
        dest="puzzle_carbon_aware_placement",
        action="store_false",
        help=(
            "Restore the legacy Puzzle spill heuristic (lowest-energy variant "
            "at the earliest node-feasible start) for A/B comparisons."
        ),
    )
    parser.add_argument(
        "--puzzle-global-green-board",
        dest="puzzle_global_green_board",
        action="store_true",
        default=True,
        help=(
            "For --budget-policy puzzle: pack all clusters against one shared "
            "green envelope while retaining independent per-cluster node "
            "ledgers. Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-puzzle-global-green-board",
        dest="puzzle_global_green_board",
        action="store_false",
        help=(
            "Restore fixed demand-weighted cluster slices before packing for "
            "A/B comparisons with Puzzle version 0.1."
        ),
    )
    parser.add_argument(
        "--puzzle-forecast-tail-carbon-multiplier",
        dest="puzzle_forecast_tail_carbon_multiplier",
        type=float,
        default=1.25,
        help=(
            "Risk multiplier applied when scoring grid energy in forecast slots "
            "extrapolated beyond the real window. Must be >= 1. Default: 1.25."
        ),
    )
    parser.add_argument(
        "--puzzle-tail-renewable-uncertainty-fraction",
        dest="puzzle_tail_renewable_uncertainty_fraction",
        type=float,
        default=0.20,
        help=(
            "For --budget-policy puzzle: build pessimistic/expected/optimistic "
            "renewable scenarios in extrapolated slots as expected times "
            "(1-fraction), 1, and (1+fraction). Commitment is selected by "
            "--puzzle-tail-renewable-robustness. Range 0-1. Default: 0.20."
        ),
    )
    parser.add_argument(
        "--puzzle-tail-renewable-robustness",
        dest="puzzle_tail_renewable_robustness",
        type=float,
        default=0.0,
        help=(
            "For --budget-policy puzzle: interpolate the extrapolated planning "
            "commitment between expected renewable (0, default) and the "
            "pessimistic scenario (1). Intermediate values provide partial "
            "robustness. Range 0-1."
        ),
    )
    parser.add_argument(
        "--puzzle-tail-renewable-calibration-days",
        dest="puzzle_tail_renewable_calibration_days",
        type=float,
        default=90.0,
        help=(
            "For --budget-policy puzzle: calibrate repeated-window P10/P90 "
            "renewable scenarios from this many days of realised history. "
            "Zero disables calibration and retains the fixed uncertainty "
            "fraction. Default: 90."
        ),
    )
    parser.add_argument(
        "--puzzle-battery-carbon-aware-dispatch",
        dest="puzzle_battery_carbon_aware_dispatch",
        action="store_true",
        default=True,
        help=(
            "For --budget-policy puzzle: allocate finite battery recourse to "
            "protected and ordinary slots by descending marginal grid CO2 "
            "(default)."
        ),
    )
    parser.add_argument(
        "--no-puzzle-battery-carbon-aware-dispatch",
        dest="puzzle_battery_carbon_aware_dispatch",
        action="store_false",
        help="Restore the version-0.5 chronological/equal battery allocation.",
    )
    parser.add_argument(
        "--puzzle-battery-carbon-priority-fraction",
        dest="puzzle_battery_carbon_priority_fraction",
        type=float,
        default=0.0,
        help=(
            "For carbon-aware Puzzle battery dispatch: fraction of ordinary "
            "recourse allocated greedily by marginal grid CO2 before the "
            "remaining energy is water-filled across time. Range 0-1. "
            "Default: 0 (protected deficits only)."
        ),
    )
    parser.add_argument(
        "--puzzle-tail-battery-reserve-fraction",
        dest="puzzle_tail_battery_reserve_fraction",
        type=float,
        default=0.10,
        help=(
            "For --budget-policy puzzle: hold this fraction of the smaller of "
            "deliverable battery energy and estimated extrapolated-job tail "
            "exposure after the protected job has physically started. The "
            "reserve is released when a real forecast shows its deficit. "
            "Range 0-1; 0 "
            "disables the reserve while retaining physical tail accounting. "
            "Default: 0.10."
        ),
    )
    parser.add_argument(
        "--puzzle-carbon-regret-g-per-h",
        dest="puzzle_carbon_regret_g_per_h",
        type=float,
        default=1.0,
        help=(
            "For --budget-policy puzzle: start the earliest node-feasible "
            "partial-green job when deferring it saves at most this many "
            "forecast grams of CO2 per extra waiting hour. The effective "
            "threshold grows with job age. Default: 1 gCO2/h."
        ),
    )
    parser.add_argument(
        "--no-puzzle-carbon-regret",
        dest="puzzle_carbon_regret_g_per_h",
        action="store_const",
        const=None,
        help=(
            "Disable carbon-regret advancement and restore the v0.2 rule: "
            "high-carbon grid is admitted only as an SLA emergency."
        ),
    )
    parser.add_argument(
        "--puzzle-sla-start-safety-margin-h",
        dest="puzzle_sla_start_safety_margin_h",
        type=float,
        default=6.0,
        help=(
            "Hours reserved before the formal queue start deadline when the "
            "Puzzle planner computes latest safe starts. Covers one default "
            "replan interval and dispatcher/node slippage. Default: 6."
        ),
    )
    parser.add_argument(
        "--puzzle-defer-carbon-margin",
        dest="puzzle_defer_carbon_margin",
        type=float,
        default=None,
        help=(
            "For --scheduling-policy puzzle: enable selective carbon-aware "
            "deferral with this margin in gCO2/kWh. The dispatcher front-runs "
            "future-planned jobs by default but holds a job for its planned "
            "slot when running now is dirtier than that slot by more than the "
            "margin (and its deadline is not at risk). Targets the Pareto point "
            "between front-run (never defers) and steering (always defers). "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--eft-dvfs-alpha",
        type=float,
        default=0.1,
        help="Runtime weight for --scheduling-policy eft-dvfs.",
    )
    parser.add_argument(
        "--eft-dvfs-beta",
        type=float,
        default=0.9,
        help="Energy weight for --scheduling-policy eft-dvfs.",
    )
    parser.add_argument(
        "--grid-power-limit",
        type=float,
        default=200_000,
        help="Maximum grid power available in low-carbon windows, in W.",
    )
    parser.add_argument(
        "--carbon-grid-threshold",
        type=float,
        default=120.0,
        help=(
            "Only expose grid budget when carbon intensity is at or below this "
            "gCO2/kWh threshold. Use 0 for a pure renewable/battery run, which "
            "can take a very long simulated time."
        ),
    )
    parser.add_argument(
        "--low-carbon-grid-fraction",
        type=float,
        default=1.0,
        help="Fraction of --grid-power-limit exposed when carbon intensity is below threshold.",
    )
    parser.add_argument(
        "--high-carbon-grid-fraction",
        type=float,
        default=0.0,
        help="Fraction of --grid-power-limit exposed when carbon intensity is above threshold.",
    )
    parser.add_argument(
        "--renewable-charge-fraction",
        "--pv-charge-fraction",
        dest="renewable_charge_fraction",
        type=float,
        default=None,
        help=(
            "Fraction of renewable generation reserved for battery charging "
            "by carbon/daytime-aware policies. Default: policy-specific "
            "(0.7 for daytime/carbon-minimizing, 0.3 for puzzle — the puzzle "
            "board loses more to a large reserve because it also plans "
            "night battery discharge from the same store)."
        ),
    )
    parser.add_argument(
        "--pv-scale",
        type=float,
        default=None,
        help="Backward-compatible scale override for PV generation, in Watts.",
    )
    parser.add_argument(
        "--renewable-sources",
        default=None,
        help=(
            "Comma-separated renewable source ids to enable. "
            f"Available: {', '.join(renewable_sources_config)}. "
            f"Default: {', '.join(default_renewable_sources(renewable_sources_config))}."
        ),
    )
    parser.add_argument(
        "--renewable-source-scale",
        action="append",
        default=[],
        metavar="SOURCE=W",
        help="Override a renewable source scale in W. Can be repeated, e.g. pv=400000 wind=300000.",
    )
    parser.add_argument(
        "--battery-soc-min",
        type=float,
        default=0.2,
        help="Minimum state of charge for the battery.",
    )
    parser.add_argument(
        "--battery-soc-max",
        type=float,
        default=0.8,
        help="Maximum state of charge for the battery.",
    )
    parser.add_argument(
        "--battery-capacity-kwh",
        type=float,
        default=2000.0,
        help="Maximum battery capacity in kWh.",
    )
    parser.add_argument(
        "--battery-p-charge-max",
        type=float,
        default=500.0,
        help="Maximum battery charge power in kW.",
    )
    parser.add_argument(
        "--battery-p-discharge-max",
        type=float,
        default=500.0,
        help="Maximum battery discharge power in kW.",
    )
    parser.add_argument(
        "--day-grid-fraction",
        type=float,
        default=0.5,
        help="Fraction of --grid-power-limit exposed during daytime by --budget-policy daytime-renewable.",
    )
    parser.add_argument(
        "--night-grid-fraction",
        type=float,
        default=1.0,
        help="Fraction of --grid-power-limit exposed at night by --budget-policy daytime-renewable.",
    )
    parser.add_argument(
        "--keep-idle-nodes-on",
        action="store_true",
        help="Leave unused compute nodes powered on throughout the simulation.",
    )
    parser.add_argument(
        "--start-nodes-on",
        dest="start_nodes_off",
        action="store_false",
        default=True,
        help="Keep all nodes powered on at simulation start. Default starts nodes off.",
    )
    parser.add_argument(
        "--runtime-prediction-error-rate",
        type=float,
        default=0.0,
        help=(
            "Relative error applied to scheduler-visible runtime predictions; "
            "physical execution work is unchanged. Default 0.0: exact model "
            "predictions (2026-07-17 decision)."
        ),
    )
    parser.add_argument(
        "--runtime-prediction-seed",
        type=int,
        default=42,
        help="Base seed for deterministic runtime prediction errors.",
    )
    parser.add_argument(
        "--scheduler-debug-log",
        dest="scheduler_debug_log",
        action="store_true",
        default=True,
        help="Export scheduler_debug_events_<cluster>.csv files. Enabled by default.",
    )
    parser.add_argument(
        "--no-scheduler-debug-log",
        dest="scheduler_debug_log",
        action="store_false",
        help="Disable scheduler debug event CSV export.",
    )
    parser.add_argument(
        "--verbose-scheduler",
        dest="scheduler_silent",
        action="store_false",
        default=True,
        help="Print periodic scheduler progress and job submission/completion messages.",
    )
    parser.add_argument(
        "--silent-scheduler",
        dest="scheduler_silent",
        action="store_true",
        help="Suppress scheduler progress messages. Enabled by default.",
    )
    return parser.parse_args(argv)


def select_system(args, systems: list[str], progress: ProgressFn = print) -> str:
    if args.system:
        progress(f"Auto-selected via CLI: {args.system}")
        return args.system

    while True:
        try:
            choice = input(f"Enter your choice (1-{len(systems)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(systems):
                return systems[idx]
            progress("Invalid number.")
        except ValueError:
            progress("Please enter a valid integer.")


def build_schedulers(
    args,
    selected_system: str,
    clusters: list[dict[str, Any]],
    node_config: dict[str, Any],
    datasets_dir: str = DATASETS_DIR,
    progress: ProgressFn | None = None,
    started_at: float | None = None,
) -> SchedulerBuild:
    schedulers = []
    all_compute_nodes = []

    for idx, cluster_def in enumerate(clusters):
        ctype = cluster_def["type"]
        n_nodes = cluster_def["num_nodes"]
        cfg = node_config[ctype]
        freq = cfg["base_frequency_ghz"]
        freq_table = {float(f): float(p) for f, p in cfg["frequencies"].items()}
        queue_definitions = queue_definitions_from_config(
            cluster_def.get("queues", []),
            overrides=cluster_def.get("queue_overrides", {}),
        )
        if not queue_definitions:
            raise ValueError(
                f"Cluster '{ctype}' in system '{selected_system}' must define queues."
            )

        job_queue = JobQueue()
        for queue_definition in queue_definitions:
            jobs_file_path = jobs_dataset_path(
                datasets_dir,
                selected_system,
                ctype,
                queue_definition.name,
            )
            if getattr(args, "jobs_dir", None) is not None:
                jobs_file_path = (
                    Path(args.jobs_dir).expanduser().resolve() / jobs_file_path.name
                )
            if not jobs_file_path.exists():
                raise FileNotFoundError(
                    f"Dataset {jobs_file_path} not found. "
                    f"Run 'python datasets/gen_jobs.py --system {selected_system}' first."
                )
            job_queue.extend_from_csv(
                filepath=str(jobs_file_path),
                start_time=SIM_START,
                queue_definition=queue_definition,
            )
        job_queue.jobs.sort(key=lambda job: (job.arrival_time, job.job_id))
        if len({job.job_id for job in job_queue.jobs}) != len(job_queue.jobs):
            raise ValueError(
                f"Duplicate job_id values found across queue datasets for cluster '{ctype}'."
            )
        if progress is not None and started_at is not None:
            log_elapsed(progress, started_at, f"[{ctype}] Loaded {job_queue.total_jobs()} jobs")

        rm = ResourceManager(
            nodes=[], power_on_gate=args.node_power_on_gate,
            keep_idle_nodes_on=args.keep_idle_nodes_on,
        )
        nodes = [
            ComputeNode(
                max_p=cfg["max_power_w"],
                idle_p=cfg["idle_power_w"],
                name=f"{ctype}-{i}",
                base_frequency_ghz=freq,
            )
            for i in range(n_nodes)
        ]

        rm.nodes = {node.name: node for node in nodes}
        rm._idle_nodes = set(node.name for node in nodes)

        if args.start_nodes_off:
            sim_start_dt = datetime.fromisoformat(SIM_START)
            for node in nodes:
                node.power_off(sim_start_dt)
            rm._idle_nodes = set()
            rm._off_nodes = set(node.name for node in nodes)

        dvfs_policy = build_dvfs_policy(args.dvfs_policy, freq_table, freq)
        scheduling_policy = build_scheduling_policy(
            args.scheduling_policy,
            cfg.get("memory_bandwidth_gb_s", 100.0),
            args.eft_dvfs_alpha,
            args.eft_dvfs_beta,
        )
        if queue_definitions:
            queue_policies = {}
            for queue_definition in queue_definitions:
                if not queue_definition.policy:
                    continue
                queue_policies[queue_definition.name] = build_scheduling_policy(
                    queue_definition.policy,
                    cfg.get("memory_bandwidth_gb_s", 100.0),
                    args.eft_dvfs_alpha,
                    args.eft_dvfs_beta,
                )
            scheduling_policy = MultiQueuePolicy(
                queue_definitions=queue_definitions,
                default_policy=scheduling_policy,
                queue_policies=queue_policies,
                sla_policy=cluster_def.get("sla_policy", "enforce"),
            )

        scheduler = JobQueueScheduler(
            job_queue=job_queue,
            resource_manager=rm,
            node_failure_rate=0.0,
            node_downtime_range=(120.0, 600.0),
            failure_seed=42 + idx,
            runtime_prediction_error_rate=args.runtime_prediction_error_rate,
            runtime_prediction_seed=args.runtime_prediction_seed + idx,
            queue_definitions=queue_definitions,
            sla_policy=cluster_def.get("sla_policy", "enforce"),
            cluster_type=ctype,
            scheduling_policy=scheduling_policy,
            dvfs_policy=dvfs_policy,
            node_config=node_config,
            step_size=60,
            auto_stop=True,
            adaptive_step=args.adaptive_step,
            debug_logging_enabled=args.scheduler_debug_log,
        )

        schedulers.append(scheduler)
        all_compute_nodes.extend(nodes)
        if progress is not None and started_at is not None:
            log_elapsed(
                progress,
                started_at,
                (
                    f"[{ctype}] Cluster ready: {n_nodes} nodes "
                    f"({cfg['idle_power_w']}W idle, {cfg['max_power_w']}W max)"
                ),
            )

    return SchedulerBuild(
        clusters=clusters,
        schedulers=schedulers,
        compute_nodes=all_compute_nodes,
    )


def build_battery(args) -> tuple[dict[str, float], NewBattery]:
    input_battery = {
        "state_of_charge_min": args.battery_soc_min,
        "state_of_charge_max": args.battery_soc_max,
        "battery_energy_max": args.battery_capacity_kwh * 1000,  # kWh to Wh
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "p_charge_max": args.battery_p_charge_max * 1000,  # kW to W
        "p_discharge_max": args.battery_p_discharge_max * 1000,  # kW to W
        "initial_battery_energy": (args.battery_capacity_kwh / 2) * 1000,
        "final_battery_energy": (args.battery_capacity_kwh / 2) * 1000,
    }
    return input_battery, NewBattery(input_battery)


def prepare_results_dir(
    args,
    root_dir: str,
    selected_system: str,
    clusters: list[dict[str, Any]],
    schedulers: list[JobQueueScheduler],
    budget_policy,
    alloc_strategy,
    policy: DefaultMicrogridPolicy,
    input_battery: dict[str, float],
    selected_renewable_sources: list[str],
    renewable_source_scales_w: dict[str, float],
    renewable_sources_config: dict[str, Any],
    alloc_strategy_key: str = None,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.scheduling_policy == "puzzle":
        sched_policy_name = "PuzzleSchedulingPolicy"
    else:
        sched_policy_name = (
            schedulers[0].scheduling_policy.__class__.__name__
            if schedulers else "UnknownPolicy"
        )
    dvfs_policy_name = schedulers[0].dvfs_policy.__class__.__name__ if schedulers else "UnknownDVFS"
    run_tag = getattr(args, "run_tag", None)
    tag_suffix = f"_{run_tag}" if run_tag else ""
    results_dir = os.path.join(
        root_dir,
        "outputs",
        (
            f"results_{timestamp}_{selected_system}_{sched_policy_name}_{dvfs_policy_name}_"
            f"{budget_policy.__class__.__name__}_{alloc_strategy.__class__.__name__}{tag_suffix}"
        ),
    )
    os.makedirs(results_dir, exist_ok=True)

    for scheduler in schedulers:
        scheduler.out_dir = results_dir

    config_dict = {
        "timestamp": timestamp,
        "run_tag": run_tag,
        "sim_start": SIM_START,
        "supercomputer": selected_system,
        "jobs_dir": (
            os.path.abspath(args.jobs_dir)
            if getattr(args, "jobs_dir", None) is not None
            else os.path.join(DATASETS_DIR, "jobs", selected_system)
        ),
        "clusters": clusters,
        "scheduling_policy": sched_policy_name,
        "scheduling_policy_key": args.scheduling_policy,
        "puzzle_plan_history_limit": (
            budget_policy.history.limit if isinstance(budget_policy, PuzzleBudgetPolicy) else None
        ),
        "puzzle_job_visibility": (
            "arrived_only" if args.budget_policy == "puzzle" else None
        ),
        "puzzle_allow_front_run": getattr(args, "puzzle_allow_front_run", False),
        "puzzle_fallback_dispatch": getattr(
            args, "puzzle_fallback_dispatch", None
        ),
        "puzzle_repack_repair": getattr(args, "puzzle_repack_repair", False),
        "puzzle_carbon_aware_placement": getattr(
            args, "puzzle_carbon_aware_placement", True
        ),
        "puzzle_global_green_board": getattr(
            args, "puzzle_global_green_board", True
        ),
        "puzzle_forecast_tail_carbon_multiplier": getattr(
            args, "puzzle_forecast_tail_carbon_multiplier", 1.25
        ),
        "puzzle_tail_battery_reserve_fraction": getattr(
            args, "puzzle_tail_battery_reserve_fraction", 0.10
        ),
        "puzzle_tail_renewable_uncertainty_fraction": getattr(
            args, "puzzle_tail_renewable_uncertainty_fraction", 0.20
        ),
        "puzzle_tail_renewable_robustness": getattr(
            args, "puzzle_tail_renewable_robustness", 0.0
        ),
        "puzzle_tail_renewable_calibration_days": getattr(
            args, "puzzle_tail_renewable_calibration_days", 90.0
        ),
        "puzzle_battery_carbon_aware_dispatch": getattr(
            args, "puzzle_battery_carbon_aware_dispatch", True
        ),
        "puzzle_battery_carbon_priority_fraction": getattr(
            args, "puzzle_battery_carbon_priority_fraction", 0.0
        ),
        "puzzle_carbon_regret_g_per_h": getattr(
            args, "puzzle_carbon_regret_g_per_h", 1.0
        ),
        "puzzle_sla_start_safety_margin_h": getattr(
            args, "puzzle_sla_start_safety_margin_h", 6.0
        ),
        "puzzle_defer_carbon_margin": getattr(args, "puzzle_defer_carbon_margin", None),
        # Effective values read off the wired schedulers, not the raw args: the
        # SLA grid floor resolves from the budget policy when the flag is unset,
        # and a run's config must say what actually ran.
        "sla_grid_override": bool(
            schedulers and getattr(schedulers[0], "sla_grid_override", False)
        ),
        "sla_grid_lead_s": (
            getattr(schedulers[0], "sla_grid_lead_s", None) if schedulers else None
        ),
        "sla_policy": (
            getattr(schedulers[0], "sla_policy", None) if schedulers else None
        ),
        "dvfs_policy": dvfs_policy_name,
        "dvfs_policy_key": args.dvfs_policy,
        "alloc_strategy": alloc_strategy.__class__.__name__,
        "alloc_strategy_key": alloc_strategy_key if alloc_strategy_key is not None else args.allocation_strategy,
        "budget_policy": budget_policy.__class__.__name__,
        "budget_policy_key": args.budget_policy,
        "budget_policy_details": {
            "grid_power_limit": getattr(budget_policy, "grid_power_limit", None),
            "requested_grid_power_limit": args.grid_power_limit,
            "effective_grid_power_limit": effective_grid_power_limit(args),
            "carbon_intensity_threshold_g_per_kwh": getattr(
                budget_policy,
                "carbon_intensity_threshold_g_per_kwh",
                None,
            ),
            "day_hours": getattr(budget_policy, "day_hours", None),
            "renewable_charge_fraction": getattr(
                budget_policy,
                "renewable_charge_fraction",
                None,
            ),
            "carbon_aware_placement": getattr(
                budget_policy, "carbon_aware_placement", None
            ),
            "global_green_board": getattr(
                budget_policy, "global_green_board", None
            ),
            "forecast_tail_carbon_multiplier": getattr(
                budget_policy, "forecast_tail_carbon_multiplier", None
            ),
            "tail_battery_reserve_fraction": getattr(
                budget_policy, "tail_battery_reserve_fraction", None
            ),
            "tail_renewable_uncertainty_fraction": getattr(
                budget_policy, "tail_renewable_uncertainty_fraction", None
            ),
            "tail_renewable_robustness": getattr(
                budget_policy, "tail_renewable_robustness", None
            ),
            "tail_renewable_calibration_days": getattr(
                budget_policy, "tail_renewable_calibration_days", None
            ),
            "battery_carbon_aware_dispatch": getattr(
                budget_policy, "battery_carbon_aware_dispatch", None
            ),
            "battery_carbon_priority_fraction": getattr(
                budget_policy, "battery_carbon_priority_fraction", None
            ),
            "carbon_regret_threshold_g_per_h": getattr(
                budget_policy, "carbon_regret_threshold_g_per_h", None
            ),
            "sla_start_safety_margin_h": getattr(
                budget_policy, "sla_start_safety_margin_h", None
            ),
            "day_grid_fraction": getattr(budget_policy, "day_grid_fraction", None),
            "night_grid_fraction": getattr(budget_policy, "night_grid_fraction", None),
            "low_carbon_grid_fraction": getattr(
                budget_policy,
                "low_carbon_grid_fraction",
                None,
            ),
            "high_carbon_grid_fraction": getattr(
                budget_policy,
                "high_carbon_grid_fraction",
                None,
            ),
        },
        "renewable_sources": selected_renewable_sources,
        "renewable_source_scales_w": {
            source: renewable_source_scales_w[source]
            for source in selected_renewable_sources
        },
        "renewable_source_config": {
            source: renewable_sources_config[source]
            for source in selected_renewable_sources
        },
        "battery": input_battery,
        "node_failure_rate": 0.0,
        "node_downtime_range": (120.0, 600.0),
        "actual_runtime_model": "work_based_cycles_plus_overhead",
        "runtime_prediction_error_rate": args.runtime_prediction_error_rate,
        "runtime_prediction_seed": args.runtime_prediction_seed,
        "scheduler_step_size_s": 60,
        "scheduler_adaptive_step": args.adaptive_step,
        "scheduler_debug_log": args.scheduler_debug_log,
        "start_nodes_off": args.start_nodes_off,
        "keep_idle_nodes_on": args.keep_idle_nodes_on,
        "node_power_on_gate": args.node_power_on_gate,
        "queue_config": {
            scheduler.cluster_type: {
                "queues": [
                    definition.to_dict()
                    for definition in getattr(scheduler, "queue_definitions", [])
                ],
                "sla_policy": getattr(scheduler, "sla_policy", "enforce"),
            }
            for scheduler in schedulers
        },
        "energy_scheduler_step_size_s": args.energy_scheduler_step_size,
        "microgrid_step_size_s": args.microgrid_step_size,
        "microgrid_mode": args.microgrid_mode,
        "pue": args.pue,
        "scheduler_budget_pue_adjusted": True,
        "microgrid_allow_grid_charging": not policy.constrain_charge_power_to_delta,
        "microgrid_charge_power_constrained_to_surplus": policy.constrain_charge_power_to_delta,
        "eft_dvfs_alpha": args.eft_dvfs_alpha,
        "eft_dvfs_beta": args.eft_dvfs_beta,
    }
    with open(os.path.join(results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)

    return results_dir


def build_environment(
    args,
    config: SimulationConfig,
    selected_system: str,
    root_dir: str = ROOT_DIR,
    datasets_dir: str = DATASETS_DIR,
    progress: ProgressFn = print,
    started_at: float | None = None,
) -> SimulationContext:
    if args.pue <= 0:
        raise ValueError("--pue must be positive.")

    started_at = _time.time() if started_at is None else started_at
    clusters = load_clusters(
        config.supercomputers,
        selected_system,
        sla_policy_override=getattr(args, "sla_policy", None),
    )
    environment = Environment(sim_start=SIM_START)

    scheduler_build = build_schedulers(
        args,
        selected_system,
        clusters,
        config.node_config,
        datasets_dir=datasets_dir,
        progress=progress,
        started_at=started_at,
    )
    schedulers = scheduler_build.schedulers

    selected_renewable_sources = parse_renewable_sources(
        args.renewable_sources,
        config.renewable_sources_config,
    )
    renewable_source_scales_w = parse_renewable_source_scales(
        args.renewable_source_scale,
        args.pv_scale,
        config.renewable_sources_config,
    )
    signal = build_renewable_signal(
        selected_renewable_sources,
        renewable_source_scales_w,
        datasets_dir=datasets_dir,
        renewable_sources_config=config.renewable_sources_config,
    )
    renewable_summary = ", ".join(
        f"{source}={renewable_source_scales_w[source]:.0f}W"
        for source in selected_renewable_sources
    )
    log_elapsed(progress, started_at, f"Renewable signal loaded ({renewable_summary})")

    input_battery, battery = build_battery(args)
    log_elapsed(
        progress,
        started_at,
        (
            f"Battery created ({args.battery_capacity_kwh}kWh, "
            f"{args.battery_p_charge_max}kW max charge/discharge)"
        ),
    )

    if args.keep_idle_nodes_on and args.start_nodes_off:
        raise ValueError("--keep-idle-nodes-on requires --start-nodes-on")
    allow_grid_charging = resolve_allow_grid_charging(args)
    policy = DefaultMicrogridPolicy(
        mode=args.microgrid_mode,
        # WAVE budget policies may reserve renewable power for storage before
        # exposing the remaining budget to compute. When this is unconstrained,
        # grid-connected runs may import power to satisfy that charge command.
        constrain_charge_power_to_delta=not allow_grid_charging,
    )

    budget_policy = build_budget_policy(args, node_config=config.node_config)
    # When the puzzle budget policy is active and the user didn't explicitly
    # choose a different allocation strategy, default to puzzle-plan so the
    # three puzzle flags always travel together.
    effective_alloc_strategy = args.allocation_strategy
    if isinstance(budget_policy, PuzzleBudgetPolicy) and args.allocation_strategy == "compute-prioritized":
        effective_alloc_strategy = "puzzle-plan"
        progress(
            "  [note] --allocation-strategy compute-prioritized is upgraded to "
            "puzzle-plan for the puzzle budget; pass --allocation-strategy "
            "default to get a non-plan split."
        )
    alloc_strategy = build_allocation_strategy(effective_alloc_strategy, budget_policy)
    # Late-bind cluster schedulers for policies that need job access.
    if isinstance(budget_policy, PuzzleBudgetPolicy):
        budget_policy.set_cluster_schedulers(schedulers)

    # Wire PuzzleSchedulingPolicy into cluster schedulers when requested.
    if args.scheduling_policy == "puzzle":
        if not isinstance(budget_policy, PuzzleBudgetPolicy):
            raise ValueError(
                "--scheduling-policy puzzle requires --budget-policy puzzle"
            )
        fallback_dispatch = getattr(args, "puzzle_fallback_dispatch", None)
        if fallback_dispatch is None:
            # An unplaced job is a deliberate carbon deferral in the new
            # planner.  Starting it with the current slot's spare budget would
            # make its whole runtime an unplanned live-power floor, often on
            # dirty grid after the green slot ends.  Keep legacy fallback only
            # when reproducing the old placement heuristic.
            fallback_dispatch = not budget_policy.carbon_aware_placement
        args.puzzle_fallback_dispatch = fallback_dispatch
        for idx, scheduler in enumerate(schedulers):
            puzzle_sched = PuzzleSchedulingPolicy(
                budget_policy,
                cluster_type=scheduler.cluster_type,
                enable_fallback_dispatch=fallback_dispatch,
                allow_front_run=getattr(args, "puzzle_allow_front_run", False),
                defer_carbon_margin=getattr(
                    args, "puzzle_defer_carbon_margin", None
                ),
            )
            puzzle_sched.set_prediction_error_model(
                args.runtime_prediction_error_rate,
                args.runtime_prediction_seed + idx,
            )
            if isinstance(scheduler.scheduling_policy, MultiQueuePolicy):
                scheduler.scheduling_policy.default_policy = puzzle_sched
            else:
                scheduler.scheduling_policy = puzzle_sched
        log_elapsed(
            progress, started_at,
            "PuzzleSchedulingPolicy wired as per-cluster default scheduling policy",
        )

    # SLA grid floor applies to every cluster scheduler regardless of policy.
    # Unset on the CLI means "give it to whoever has no SLA mechanism already":
    # PuzzleBudgetPolicy plans its own sla_emergency grid escape, so stacking a
    # second floor on top would hand it a boost the baselines cannot match,
    # while leaving the baselines without one hands the advantage the other way.
    sla_grid_override = getattr(args, "sla_grid_override", None)
    if sla_grid_override is None:
        sla_grid_override = not isinstance(budget_policy, PuzzleBudgetPolicy) and any(
            getattr(scheduler, "sla_policy", "enforce") == "enforce"
            for scheduler in schedulers
        )
    for scheduler in schedulers:
        scheduler.sla_grid_override = bool(sla_grid_override)
        scheduler.sla_grid_lead_s = float(
            os.environ.get(
                "SLA_GRID_LEAD_S",
                str(getattr(args, "sla_grid_lead_s", DEFAULT_SLA_GRID_LEAD_S)),
            )
        )
    if args.microgrid_mode == "islanded" and args.grid_power_limit > 0:
        log_elapsed(
            progress,
            started_at,
            (
                "Islanded mode selected: ignoring configured grid power limit "
                f"({args.grid_power_limit:.0f}W) for scheduler budgets."
            ),
        )
    results_dir = prepare_results_dir(
        args,
        root_dir,
        selected_system,
        clusters,
        schedulers,
        budget_policy,
        alloc_strategy,
        policy,
        input_battery,
        selected_renewable_sources,
        renewable_source_scales_w,
        config.renewable_sources_config,
        alloc_strategy_key=effective_alloc_strategy,
    )

    energy_scheduler = EnergyScheduler(
        renewable_signal=signal,
        battery=battery,
        cluster_schedulers=schedulers,
        node_config=config.node_config,
        alloc_strategy=alloc_strategy,
        budget_policy=budget_policy,
        policy=policy,
        step_size=args.energy_scheduler_step_size,
        pue=args.pue,
    )
    log_elapsed(progress, started_at, "EnergyScheduler created")

    computing_system = ComputingSystem(
        scheduler_build.compute_nodes,
        pue=args.pue,
        detailed_state=False,
    )
    monitor = Monitor(co2_signal=signal, storage=battery, step_size=60)
    for scheduler in schedulers:
        rm = scheduler.rm
        ctype = scheduler.cluster_type
        _s = scheduler
        monitor.add_monitor_fn(lambda time, _rm=rm, _ct=ctype, _sched=_s: {
            f"{_ct}_busy": len(_rm.get_busy_nodes()),
            f"{_ct}_completed": _sched.jobs_completed,
            f"{_ct}_pending": len(_sched._get_pending_jobs(time)),
            f"{_ct}_running": _rm.get_running_jobs_count(),
            f"{_ct}_power": _sched.get_current_power(),
            f"{_ct}_budget": _sched.energy_budget or 0,
            f"{_ct}_util": _rm.get_cluster_utilization(),
            f"{_ct}_energy_kwh": _sched.total_energy_wh / 1000,
        })

    monitor.add_monitor_fn(lambda time: {
        "renewable_generation_w": energy_scheduler.last_budget_info.get(
            "renewable_power_w",
            0,
        ),
        "budget_renewable_w": energy_scheduler.last_budget_info.get(
            "budget_renewable_w",
            0,
        ),
        "budget_battery_w": energy_scheduler.last_budget_info.get(
            "battery_discharge_power_w",
            0,
        ),
        "budget_grid_w": energy_scheduler.last_budget_info.get("grid_power_w", 0),
        "budget_scheduler_w": energy_scheduler.last_budget_info.get(
            "scheduler_total_budget_w",
            0,
        ),
        "budget_pue": energy_scheduler.last_budget_info.get("pue", args.pue),
        "charge_reserve_w": energy_scheduler.last_budget_info.get("charge_reserve_w", 0),
        "budget_carbon_intensity_g_per_kwh": energy_scheduler.last_budget_info.get(
            "carbon_intensity_g_per_kwh",
            0,
        ),
        "budget_low_carbon_grid_allowed": energy_scheduler.last_budget_info.get(
            "low_carbon_grid_allowed",
            False,
        ),
        **{
            config.renewable_sources_config[source].get("signal_column", f"{source}_gen"): signal.at(
                time,
                column=config.renewable_sources_config[source].get(
                    "signal_column",
                    f"{source}_gen",
                ),
            )
            for source in selected_renewable_sources
        },
    })

    environment.add_microgrid(
        actors=[
            computing_system,
            *[
                Generator(
                    signal,
                    name=f"Generator-{source}",
                    column=config.renewable_sources_config[source].get(
                        "signal_column",
                        f"{source}_gen",
                    ),
                )
                for source in selected_renewable_sources
            ],
        ],
        controllers=[monitor, energy_scheduler] + schedulers,
        storage=battery,
        step_size=args.microgrid_step_size,
        policy=policy,
    )
    log_elapsed(progress, started_at, "Microgrid added")

    return SimulationContext(
        environment=environment,
        schedulers=schedulers,
        monitor=monitor,
        results_dir=results_dir,
        selected_system=selected_system,
        battery=battery,
        signal=signal,
        energy_scheduler=energy_scheduler,
        config=config,
        clusters=clusters,
        selected_renewable_sources=selected_renewable_sources,
        renewable_source_scales_w=renewable_source_scales_w,
    )


def run_simulation(
    context: SimulationContext,
    progress: ProgressFn = print,
    started_at: float | None = None,
) -> tuple[bool, RuntimeError | None]:
    started_at = _time.time() if started_at is None else started_at
    progress(f"\n[{_time.time() - started_at:.2f}s] Starting simulation...")

    simulation_interrupted = False
    simulation_error = None
    try:
        context.environment.run()
    except KeyboardInterrupt:
        simulation_interrupted = True
        progress(f"\n[{_time.time() - started_at:.2f}s] Simulation interrupted; saving partial results.")
    except RuntimeError as exc:
        if "islanded mode" not in str(exc):
            raise
        simulation_error = exc
        progress(
            f"\n[{_time.time() - started_at:.2f}s] Simulation failed; "
            f"saving partial results. Reason: {exc}"
        )
    finally:
        for scheduler in context.schedulers:
            scheduler.finalize()

    return simulation_interrupted, simulation_error


def print_scheduler_stats(
    schedulers: list[JobQueueScheduler],
    interrupted: bool,
    failed: bool = False,
    progress: ProgressFn = print,
    started_at: float | None = None,
) -> None:
    started_at = _time.time() if started_at is None else started_at
    if failed:
        status_label = "Failed"
    elif interrupted:
        status_label = "Interrupted"
    else:
        status_label = "Complete"
    progress(f"\n[{_time.time() - started_at:.2f}s] Simulation {status_label}")
    for scheduler in schedulers:
        stats = scheduler.get_stats()
        ctype = scheduler.cluster_type.upper()
        progress(f"\n[{ctype}] SIMULATION STATISTICS")
        progress(
            f"  |- Jobs:  {stats['completed']} completed, "
            f"{stats['failed']} failed, {stats['total']} total"
        )
        progress(
            f"  |- Nodes: {stats['busy_nodes']} busy, {stats['idle_nodes']} idle, "
            f"{stats['down_nodes']} down, {stats['total_nodes']} total"
        )
        progress(
            f"  +- Usage: {stats.get('utilization_mean', 0.0):.1%} mean "
            f"({stats.get('utilization_peak', 0.0):.1%} peak) node utilization, "
            f"{stats['total_jobs_executed']} node-jobs executed"
        )


def export_puzzle_forecast(
    context: SimulationContext,
    progress: ProgressFn = print,
) -> None:
    budget_policy = context.energy_scheduler.budget_policy
    if not isinstance(budget_policy, PuzzleBudgetPolicy):
        return

    selected_plans = budget_policy.history.indexed_plans
    if not selected_plans:
        return
    plan_count = budget_policy.history.count
    sampled = len(selected_plans) < plan_count
    pue = context.energy_scheduler.pue
    active_rows = budget_policy.history.active_timeline()
    slot_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []

    def add_slot_rows(plan_id: int, cluster_type: str, slots) -> None:
        for slot_idx, slot in enumerate(slots):
            slot_rows.append({
                "plan_id": plan_id,
                "cluster_type": cluster_type,
                "slot_idx": slot_idx,
                "start_time": slot.start_time,
                "end_time": slot.end_time,
                "duration_s": slot.duration_s,
                "renewable_power_w": slot.renewable_power_w,
                "renewable_pessimistic_power_w": getattr(
                    slot, "renewable_pessimistic_power_w", slot.renewable_power_w
                ),
                "renewable_expected_power_w": getattr(
                    slot, "renewable_expected_power_w", slot.renewable_power_w
                ),
                "renewable_optimistic_power_w": getattr(
                    slot, "renewable_optimistic_power_w", slot.renewable_power_w
                ),
                "is_extrapolated": bool(getattr(slot, "is_extrapolated", False)),
                "renewable_pessimistic_fraction": (
                    max(
                        0.0,
                        1.0
                        - getattr(
                            slot,
                            "renewable_pessimistic_power_w",
                            slot.renewable_power_w,
                        )
                        / getattr(
                            slot,
                            "renewable_expected_power_w",
                            slot.renewable_power_w,
                        ),
                    )
                    if getattr(
                        slot,
                        "renewable_expected_power_w",
                        slot.renewable_power_w,
                    ) > 0
                    else 0.0
                ),
                "renewable_optimistic_fraction": (
                    max(
                        0.0,
                        getattr(
                            slot,
                            "renewable_optimistic_power_w",
                            slot.renewable_power_w,
                        )
                        / getattr(
                            slot,
                            "renewable_expected_power_w",
                            slot.renewable_power_w,
                        )
                        - 1.0,
                    )
                    if getattr(
                        slot,
                        "renewable_expected_power_w",
                        slot.renewable_power_w,
                    ) > 0
                    else 0.0
                ),
                "battery_power_w": slot.battery_power_w,
                "battery_correction_power_w": getattr(
                    slot, "battery_correction_power_w", 0.0
                ),
                "grid_power_w": slot.grid_power_w,
                "green_available_w": slot.green_available_w,
                "grid_allocated_w": slot.grid_allocated_w,
                "total_available_w": slot.total_available_w,
                "scheduler_total_available_w": slot.total_available_w / pue,
                "allocated_w": slot.allocated_w,
                "idle_reserved_w": getattr(slot, "idle_reserved_w", 0.0),
                "carbon_intensity": slot.carbon_intensity,
                "charge_reserve_w": slot.charge_reserve_w,
            })

    def add_job_rows(plan_id: int, plan, rects, placed: bool) -> None:
        for rect in rects:
            start_time = ""
            end_time = ""
            if (
                placed
                and rect.start_slot_idx is not None
                and rect.end_slot_idx is not None
                and 0 <= rect.start_slot_idx < len(plan.slots)
                and 0 <= rect.end_slot_idx < len(plan.slots)
            ):
                start_time = plan.slots[rect.start_slot_idx].start_time
                end_time = plan.slots[rect.end_slot_idx].end_time

            job_rows.append({
                "plan_id": plan_id,
                "plan_start": plan.plan_start,
                "plan_end": plan.plan_end,
                "cluster_type": rect.cluster_type,
                "job_id": rect.job_id,
                "placed": placed,
                "arrival_time": rect.arrival_time or "",
                "earliest_start_slot_idx": rect.earliest_start_slot_idx,
                "latest_start_slot_idx": getattr(rect, "latest_start_slot_idx", None),
                "start_slot_idx": rect.start_slot_idx,
                "end_slot_idx": rect.end_slot_idx,
                "start_time": start_time,
                "end_time": end_time,
                "power_w": rect.power_w,
                "duration_s": rect.duration_s,
                "num_nodes": rect.num_nodes,
                "frequency_ghz": rect.frequency_ghz,
                "energy_wh": rect.energy_wh,
                "stack_base_w": rect.stack_base_w,
                "used_grid": bool(getattr(rect, "used_grid", False)),
                "grid_reason": getattr(rect, "grid_reason", None),
                "planned_grid_carbon_g": getattr(
                    rect, "planned_grid_carbon_g", 0.0
                ),
            })

    for plan_id, plan in selected_plans:
        add_slot_rows(plan_id, "__total__", plan.slots)
        for cluster_type, slots in sorted(plan.cluster_slots.items()):
            add_slot_rows(plan_id, cluster_type, slots)
        add_job_rows(plan_id, plan, plan.placed_jobs, True)
        add_job_rows(plan_id, plan, plan.unplaced_jobs, False)

    slots_path = os.path.join(context.results_dir, "puzzle_forecast_slots.csv")
    jobs_path = os.path.join(context.results_dir, "puzzle_forecast_jobs.csv")
    active_path = os.path.join(
        context.results_dir, PUZZLE_ACTIVE_FORECAST_FILE,
    )
    pd.DataFrame(slot_rows).to_csv(slots_path, index=False)
    pd.DataFrame(job_rows).to_csv(jobs_path, index=False)
    pd.DataFrame(active_rows).to_csv(active_path, index=False)
    if sampled:
        progress(
            f"Puzzle forecast exported to {slots_path} and {jobs_path} "
            f"({len(selected_plans)} detailed plans sampled from {plan_count}; "
            f"all active-plan slots exported to {active_path}; "
            "set WAVE_PUZZLE_FORECAST_MAX_PLANS=0 before running to retain all)"
        )
    else:
        progress(
            f"Puzzle forecast exported to {slots_path} and {jobs_path} "
            f"({plan_count} plans; active timeline in {active_path})"
        )


def _puzzle_forecast_max_export_plans() -> int | None:
    raw_value = os.environ.get("WAVE_PUZZLE_FORECAST_MAX_PLANS")
    if raw_value is None:
        return DEFAULT_PUZZLE_FORECAST_MAX_EXPORT_PLANS

    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_PUZZLE_FORECAST_MAX_EXPORT_PLANS

    if value <= 0:
        return None
    return value


def export_puzzle_plan_summary(
    context: SimulationContext,
    progress: ProgressFn = print,
) -> None:
    """Write small per-plan and per-cluster dispatch summaries.

    ``puzzle_plan_summary.csv`` holds one row per plan (budget vs planned
    area, jobs placed/unplaced) for scheduling analysis.
    ``puzzle_dispatch_summary.csv`` holds one row per cluster with the
    ``PuzzleSchedulingPolicy`` dispatch counters (how many planned jobs
    were dispatched, how many failed their slot, and how many fallback
    jobs backfilled the slack).
    """
    budget_policy = context.energy_scheduler.budget_policy
    if not isinstance(budget_policy, PuzzleBudgetPolicy):
        return

    plan_rows = budget_policy.history.summaries
    if not plan_rows:
        return

    plan_summary_path = os.path.join(
        context.results_dir, "puzzle_plan_summary.csv"
    )
    pd.DataFrame(plan_rows).to_csv(plan_summary_path, index=False)

    dispatch_rows: list[dict[str, Any]] = []
    for scheduler in context.schedulers:
        policy = getattr(scheduler, "scheduling_policy", None)
        if isinstance(policy, MultiQueuePolicy):
            policy = policy.default_policy
        if not isinstance(policy, PuzzleSchedulingPolicy):
            continue
        stats = policy.dispatch_stats
        dispatch_rows.append({
            "cluster_type": scheduler.cluster_type,
            "planned_dispatched": stats.get("planned_dispatched", 0),
            "planned_missed": stats.get("planned_missed", 0),
            "fallback_dispatched": stats.get("fallback_dispatched", 0),
            "green_dispatched": stats.get("green_dispatched", 0),
            "grid_dispatched": stats.get("grid_dispatched", 0),
        })

    if dispatch_rows:
        dispatch_summary_path = os.path.join(
            context.results_dir, "puzzle_dispatch_summary.csv"
        )
        pd.DataFrame(dispatch_rows).to_csv(dispatch_summary_path, index=False)
        progress(
            f"Puzzle plan summary exported to {plan_summary_path} "
            f"({len(plan_rows)} plans) and dispatch summary "
            f"({len(dispatch_rows)} clusters)"
        )
    else:
        progress(
            f"Puzzle plan summary exported to {plan_summary_path} "
            f"({len(plan_rows)} plans)"
        )


class _PostprocessingPickler(pickle.Pickler):
    """Save plot inputs without the live planner graph; not a resume checkpoint."""

    def reducer_override(self, obj):
        if isinstance(obj, PuzzleBudgetPolicy):
            from vessim.wave.energy.puzzle_history import PuzzlePlanHistory

            state = obj.__dict__.copy()
            state["history"] = PuzzlePlanHistory(obj.history.limit, pue=obj.pue)
            state["_current_plan"] = None
            state["cluster_schedulers"] = []
            state["_plan_index_cache"] = {}
            return (type(obj), (), state)
        if isinstance(obj, JobQueueScheduler):
            state = obj.__dict__.copy()
            state["energy_scheduler"] = None
            return (object.__new__, (type(obj),), state)
        return NotImplemented


def save_outputs(
    context: SimulationContext,
    progress: ProgressFn = print,
    *,
    save_snapshot: bool = True,
) -> None:
    started_at = _time.perf_counter()
    progress("Saving monitor and Puzzle exports...")
    results_file = os.path.join(context.results_dir, "energy_simulation_result.csv")
    context.monitor.to_csv(results_file)
    progress(f"\nResults saved to {results_file}")

    export_puzzle_forecast(context, progress)
    export_puzzle_plan_summary(context, progress)

    if save_snapshot:
        progress(f"Exports saved in {_time.perf_counter() - started_at:.2f}s; saving plot snapshot...")
        snapshot_started = _time.perf_counter()
        with open(os.path.join(context.results_dir, SCHEDULERS_FILE), "wb") as f:
            _PostprocessingPickler(f, protocol=pickle.HIGHEST_PROTOCOL).dump(context.schedulers)
        progress(f"Plot snapshot saved in {_time.perf_counter() - snapshot_started:.2f}s")
    else:
        progress(f"Exports saved in {_time.perf_counter() - started_at:.2f}s; plot snapshot skipped")


def maybe_generate_plots(
    args,
    context: SimulationContext,
    progress: ProgressFn = print,
    started_at: float | None = None,
) -> None:
    started_at = _time.time() if started_at is None else started_at
    if args.no_plots:
        progress(f"\n[{_time.time() - started_at:.2f}s] Plot generation skipped.")
        progress(f"Run plots later with:")
        progress(f"  py scripts/plot_energy.py {context.results_dir}")
        if args.budget_policy == "puzzle":
            progress(f"  py scripts/plot_puzzle_forecast.py {context.results_dir}")
            progress(f"  py scripts/plot_puzzle.py {context.results_dir}")
            progress(f"  py scripts/plot_gantt_puzzle.py {context.results_dir}")
        return

    from scripts.plot_energy import generate_plots

    progress(f"\n[{_time.time() - started_at:.2f}s] Generating plots...")
    generate_plots(
        context.results_dir,
        context.schedulers,
        plot_workers=args.plot_workers,
    )

    # The combined puzzle boards are no longer rendered by default (the
    # default puzzle view is the budget-vs-forecast chart, generated above).
    # plot_combined_puzzle stays available via scripts/plot_energy.py
    # --puzzle-plan / --real-puzzle for on-demand inspection.



def suppress_asyncio_warnings() -> None:
    sys.stderr = open(os.devnull, "w")


def install_asyncio_warning_suppression() -> None:
    import atexit

    atexit.register(suppress_asyncio_warnings)


def main(argv: list[str] | None = None, progress: ProgressFn = print) -> int:
    started_at = _time.time()

    config = load_config()
    systems = list(config.supercomputers.keys())

    args = parse_args(systems, config.renewable_sources_config, argv)
    log_elapsed(progress, started_at, "Imports done")
    log_elapsed(progress, started_at, f"Node config loaded: {list(config.node_config.keys())}")
    log_elapsed(progress, started_at, f"Supercomputers loaded: {systems}")
    log_elapsed(
        progress,
        started_at,
        f"Renewable sources loaded: {list(config.renewable_sources_config.keys())}",
    )

    if not args.system:
        progress("=== Vessim HPC Simulation ===")
        progress("Select a supercomputer to simulate:")
        for i, sys_name in enumerate(systems):
            progress(f"  {i + 1}. {sys_name}")

    scheduler_module.SILENT = args.scheduler_silent
    selected_system = select_system(args, systems, progress)

    try:
        context = build_environment(
            args,
            config,
            selected_system,
            progress=progress,
            started_at=started_at,
        )
    except FileNotFoundError as exc:
        progress(f"\nERROR: {exc}\n")
        return 1

    interrupted, simulation_error = run_simulation(
        context,
        progress=progress,
        started_at=started_at,
    )
    print_scheduler_stats(
        context.schedulers,
        interrupted,
        failed=simulation_error is not None,
        progress=progress,
        started_at=started_at,
    )
    save_outputs(context, progress=progress, save_snapshot=not args.no_snapshot)
    if simulation_error is not None:
        progress(f"\nERROR: {simulation_error}")
        progress("Partial outputs were saved; plot generation skipped for the failed run.")
        install_asyncio_warning_suppression()
        return 1
    maybe_generate_plots(args, context, progress=progress, started_at=started_at)
    install_asyncio_warning_suppression()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
