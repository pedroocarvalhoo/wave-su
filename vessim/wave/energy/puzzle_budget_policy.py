"""
Puzzle budget policy: 24-hour lookahead envelope with job bin-packing.

   Builds a power envelope from renewable forecasts (PV, wind), battery state,
   and grid carbon constraints over a configurable horizon (default 24h).
   Already-arrived pending jobs are represented as power x time rectangles and
   packed into a shared cross-cluster envelope using an SLA-slack greedy pass.
   Placements that require grid power are evaluated separately.  The earlier
   cluster-local slicing remains available for reproducible comparisons.

   The policy implements the standard BudgetGenerationPolicy interface:
   generate_budget() returns the BudgetInfo for the current time slot
   from the pre-computed plan.  No changes to EnergyScheduler are needed.

   Keep policy-specific scoring logic here; keep cross-cluster budget
   distribution in strategy.py.
"""

from __future__ import annotations

import math
from itertools import accumulate
from time import perf_counter
from bisect import bisect_right, insort
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING

from vessim.wave.energy.budget import BudgetInfo
from vessim.wave.energy.puzzle_history import PuzzlePlanHistory
from vessim.wave.energy.budget_policy import (
    BudgetGenerationPolicy,
    renewable_power_at,
    battery_energy_bounds,
    free_battery_energy_wh,
    is_daytime,
    remaining_night_hours,
)
from vessim.wave.energy.puzzle_battery import (
    deliverable_battery_energy_wh,
    plan_tail_battery_dispatch,
)
from vessim.wave.energy.puzzle_forecast import (
    TailRenewableCalibration,
    calibrate_tail_renewable_uncertainty,
)

if TYPE_CHECKING:
    from vessim.signal import HistoricalSignal
    from vessim.storage import Storage
    from vessim.wave.scheduler import JobQueueScheduler


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PowerSlot:
    """A facility-side time interval of the green board plus grid overflow.

    The puzzle board is built from green energy only: ``green_available_w`` is
    the ceiling the packer fills (renewable budget + planned battery).  The
    grid is not part of the board — ``grid_power_w`` is the carbon-filtered
    overflow capacity, and ``grid_allocated_w`` records the grid power the plan
    deliberately spent on jobs that could not be placed green (the penalty for
    a badly-built puzzle).
    """

    start_time: datetime
    end_time: datetime
    duration_s: float
    renewable_power_w: float       # Facility PV + wind forecast for this slot.
    battery_power_w: float         # Planned facility battery contribution.
    grid_power_w: float            # Facility grid overflow (carbon-filtered).
    green_available_w: float       # Facility green ceiling = renewable + battery.
    battery_correction_power_w: float = 0.0  # Tail-risk subset of battery power.
    # Renewable scenarios are identical inside the real forecast window.  In
    # extrapolated slots ``tail_renewable_robustness`` selects a commitment
    # between expected and pessimistic; the optimistic path is retained for
    # observability.
    renewable_pessimistic_power_w: float = 0.0
    renewable_expected_power_w: float = 0.0
    renewable_optimistic_power_w: float = 0.0
    is_extrapolated: bool = False
    carbon_intensity: float = 0.0  # gCO2/kWh at this slot.
    charge_reserve_w: float = 0.0  # Renewable power reserved for charging.
    allocated_w: float = 0.0       # Facility green allocated to idle/running/placed jobs.
    idle_reserved_w: float = 0.0   # Facility idle draw outside the job footprint.
    idle_allocated_w: float = 0.0  # Idle already included in green/grid allocations.
    grid_allocated_w: float = 0.0  # Facility grid allocated to idle/spills/running jobs.

    @property
    def remaining_w(self) -> float:
        """Free capacity on the green board."""
        return max(0.0, self.green_available_w - self.allocated_w)

    @property
    def total_available_w(self) -> float:
        """Full envelope (green board + grid overflow), for plots/back-compat."""
        return self.green_available_w + self.grid_power_w


@dataclass
class JobRectangle:
    """A job represented as a power × time rectangle for the puzzle packer."""

    job_id: int
    power_w: float                 # Facility draw beyond any reserved static idle floor.
    duration_s: float              # Estimated execution time.
    num_nodes: int
    frequency_ghz: float
    energy_wh: float               # = power_w × duration_s / 3600.
    cluster_type: str = ""
    arrival_time: Optional[datetime] = None
    earliest_start_slot_idx: int = 0
    # Latest slot the job may start in to satisfy its SLA (start deadline).
    # ``None`` means no SLA bound — the piece may shift anywhere in the horizon.
    latest_start_slot_idx: Optional[int] = None
    # DVFS variants: (frequency_ghz, board_power_w, duration_s) the packer may
    # deform this piece into to fit the green envelope.  Empty → fixed shape.
    dvfs_variants: list = field(default_factory=list)
    # Position in the plan (filled by the packer):
    start_slot_idx: Optional[int] = None
    end_slot_idx: Optional[int] = None
    # True when the packer could not place this job on the green board and
    # routed it to the grid-overflow lane (high-carbon penalty).
    used_grid: bool = False
    # Stacking position for visualization:
    stack_base_w: float = 0.0
    # Exact per-slot source split for grid-spilled jobs.  A spill can consume
    # different green/grid amounts in every slot, so a single ``used_grid``
    # flag is not enough to undo its footprint during the optional repair pass.
    green_allocation_w_by_slot: dict[int, float] = field(default_factory=dict)
    grid_allocation_w_by_slot: dict[int, float] = field(default_factory=dict)
    # Why grid was admitted for this rectangle.  ``low_carbon`` means every
    # per-slot deficit fitted the carbon-filtered grid allowance;
    # ``carbon_regret`` admits an earlier start when the forecast CO2 saved by
    # waiting is too small per hour; ``sla_emergency`` is the final escape hatch
    # once the job's latest safe start enters the real forecast window.
    grid_reason: Optional[str] = None
    # Forecast emissions of the exact per-slot grid split.  This is a planning
    # metric (the physical monitor remains the source of truth for actual CO2).
    planned_grid_carbon_g: float = 0.0


@dataclass
class PuzzlePlan:
    """Complete plan for the lookahead horizon."""

    plan_start: datetime
    plan_end: datetime
    slots: list[PowerSlot]
    placed_jobs: list[JobRectangle]
    unplaced_jobs: list[JobRectangle]
    total_budget_kwh: float
    total_planned_kwh: float
    cluster_slots: dict[str, list[PowerSlot]] = field(default_factory=dict)
    # Green vs grid attribution: ``total_planned_kwh`` is the green-board area;
    # ``grid_planned_kwh`` is the carbon penalty paid for spilled jobs.
    grid_planned_kwh: float = 0.0
    grid_planned_co2_g: float = 0.0
    # Battery energy deliberately held outside the compute envelope until a
    # later real forecast can correct an extrapolation-backed running job.
    battery_tail_reserve_kwh: float = 0.0
    battery_tail_risk_kwh: float = 0.0
    battery_correction_planned_kwh: float = 0.0
    battery_avoided_co2_g: float = 0.0
    battery_correction_avoided_co2_g: float = 0.0
    battery_carbon_aware_dispatch: bool = False
    battery_carbon_priority_fraction: float = 0.0
    tail_protected_job_keys: tuple[tuple[str, int], ...] = ()
    tail_renewable_uncertainty_fraction: float = 0.0
    tail_renewable_pessimistic_kwh: float = 0.0
    tail_renewable_expected_kwh: float = 0.0
    tail_renewable_optimistic_kwh: float = 0.0
    tail_renewable_planning_kwh: float = 0.0
    tail_renewable_robustness: float = 0.0
    tail_renewable_calibration_days: float = 0.0
    tail_renewable_calibration_samples: int = 0
    tail_renewable_calibrated: bool = False
    tail_renewable_pessimistic_fraction_max: float = 0.0
    tail_renewable_optimistic_fraction_max: float = 0.0
    tail_battery_effective_reserve_fraction: float = 0.0

    @property
    def slack_kwh(self) -> float:
        return self.total_budget_kwh - self.total_planned_kwh

    @property
    def green_planned_kwh(self) -> float:
        """Green-board energy planned (alias for ``total_planned_kwh``)."""
        return self.total_planned_kwh

    @property
    def green_fraction(self) -> float:
        """Share of planned compute energy served from the green board.

        1.0 means a perfectly-built puzzle (no grid spill); lower values mean
        more high-carbon grid energy was needed.
        """
        total = self.total_planned_kwh + self.grid_planned_kwh
        if total <= 0:
            return 1.0
        return self.total_planned_kwh / total

    @property
    def num_grid_spilled(self) -> int:
        """Number of placed jobs routed to the grid-overflow lane."""
        return sum(1 for rect in self.placed_jobs if rect.used_grid)

    @property
    def num_low_carbon_grid(self) -> int:
        """Placed jobs whose forecast grid deficit stayed inside the clean cap."""
        return sum(
            1 for rect in self.placed_jobs
            if rect.grid_reason == "low_carbon"
        )

    @property
    def num_sla_emergency_grid(self) -> int:
        """Placed jobs admitted to high-carbon grid to protect their SLA."""
        return sum(
            1 for rect in self.placed_jobs
            if rect.grid_reason == "sla_emergency"
        )

    @property
    def num_carbon_regret_grid(self) -> int:
        """Placed jobs advanced because waiting saved too little CO2 per hour."""
        return sum(
            1 for rect in self.placed_jobs
            if rect.grid_reason == "carbon_regret"
        )

    @property
    def utilization(self) -> float:
        if self.total_budget_kwh <= 0:
            return 0.0
        return self.total_planned_kwh / self.total_budget_kwh

    @property
    def num_placed(self) -> int:
        return len(self.placed_jobs)

    @property
    def num_unplaced(self) -> int:
        return len(self.unplaced_jobs)

@dataclass(frozen=True)
class _PlanIndex:
    """Dispatcher-facing lookup tables derived once per plan, per cluster.

    ``by_start`` holds the cluster's placed rectangles that have a start slot,
    ordered by it, with ``start_slots`` the matching key list — so "every
    rectangle due at or before slot i" is a ``bisect`` and a slice instead of a
    scan of every placed job on every dispatch query.
    """

    job_ids: frozenset[int]
    rects_by_id: dict[int, JobRectangle]
    by_start: list[JobRectangle]
    start_slots: list[int]


@dataclass(frozen=True)
class _GridPlacement:
    """A scored partial-green/grid candidate for one DVFS variant and start."""

    frequency_ghz: float
    power_w: float
    duration_s: float
    start_slot_idx: int
    n_slots: int
    reason: str
    forecast_carbon_g: float


def _max_free_gap(
    obstacles: list[tuple[float, float]],
    floor_w: float,
    ceiling_w: float,
) -> float:
    """Free power capacity in one slot, between floor and ceiling.

    Power is fungible: a piece of power ``p`` fits a window iff every slot it
    spans has free capacity ``>= p``.  Each obstacle interval belongs to one
    placed piece and its height ``hi - lo`` is that piece's power draw, so
    summing heights gives the power already committed in this slot — where
    the pieces sit vertically is display-only (``stack_base_w``).

    (Until 2026-07-16 this returned the tallest *contiguous* free band —
    treating pieces as rigid rectangles that must occupy one fixed band
    ``[base, base + p)`` across their whole span.  That is stricter than the
    physics and rejected placements that fit fine.  NOTE: the fungible test
    packs denser plans but measured WORSE campaign CO2 on the 2523-job
    workload (+1.2..1.8% PV, +8.5% pv+wind — A/B tags nb-* vs nb2-* in
    results/); adopted anyway as the physically correct base model, the
    regression is an open investigation.  See the explaining_the_puzzle
    notebook, section 9.)
    """
    used = floor_w
    for lo, hi in obstacles:
        used += hi - lo
    free = ceiling_w - used
    return free if free > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class PuzzleBudgetPolicy(BudgetGenerationPolicy):
    """
    24-hour lookahead budget policy with job bin-packing.

    Builds a power envelope from renewable forecasts (PV, wind), battery
    state, and grid carbon constraints.  Only pending jobs whose arrival time
    is at or before the current simulation time are converted to power × time
    rectangles and packed into a shared cross-cluster envelope using an
    SLA-slack greedy pass.
    Pieces that do not fit wholly on green capacity are evaluated across their
    DVFS variants and feasible starts by their exact per-slot grid-carbon cost.

    Integration contract
    --------------------
    This policy implements the standard ``BudgetGenerationPolicy`` interface.
    ``generate_budget()`` returns an instantaneous ``BudgetInfo`` for the
    current simulation time by looking up the corresponding slot in the
    pre-computed plan.  The ``EnergyScheduler`` does not need any changes.

    Cluster schedulers are injected via the constructor or via
    ``set_cluster_schedulers()`` for late binding.

    Replanning triggers
    -------------------
    * Newly arrived pending jobs, at the next budget update (no cooldown).
    * Plan expired (time >= plan_end).
    * Periodic (every ``replan_interval_h`` hours).
    * External request via ``request_replan()`` (e.g. after a job failure),
      rate-limited by ``replan_request_cooldown_h``.
    * Average planned utilization below ``replan_on_underutilization``
      threshold, when explicitly enabled with a positive threshold.
    """

    def __init__(
        self,
        grid_power_limit: float = 0.0,
        carbon_intensity_threshold_g_per_kwh: float = 120.0,
        plan_horizon_h: float = 24.0,
        slot_duration_s: int = 300,
        replan_interval_h: float = 6.0,
        renewable_charge_fraction: float = 0.3,
        day_hours: tuple[int, int] = (8, 20),
        node_configs: Optional[dict] = None,
        cluster_schedulers: Optional[list["JobQueueScheduler"]] = None,
        replan_on_underutilization: float = 0.0,
        replan_request_cooldown_h: float = 1.0,
        repack_repair: bool = False,
        pue: float = 1.0,
        carbon_aware_placement: bool = True,
        forecast_tail_carbon_multiplier: float = 1.25,
        sla_start_safety_margin_h: Optional[float] = None,
        global_green_board: bool = True,
        carbon_regret_threshold_g_per_h: Optional[float] = 1.0,
        tail_battery_reserve_fraction: float = 0.10,
        tail_renewable_uncertainty_fraction: float = 0.20,
        tail_renewable_robustness: float = 0.0,
        tail_renewable_calibration_days: float = 90.0,
        battery_carbon_aware_dispatch: bool = True,
        battery_carbon_priority_fraction: float = 0.0,
        plan_history_limit: Optional[int] = 64,
    ):
        if pue <= 0:
            raise ValueError("pue must be positive")
        if forecast_tail_carbon_multiplier < 1.0:
            raise ValueError("forecast_tail_carbon_multiplier must be >= 1")
        if sla_start_safety_margin_h is not None and sla_start_safety_margin_h < 0:
            raise ValueError("sla_start_safety_margin_h must be >= 0")
        if (
            carbon_regret_threshold_g_per_h is not None
            and carbon_regret_threshold_g_per_h < 0
        ):
            raise ValueError("carbon_regret_threshold_g_per_h must be >= 0")
        if not 0.0 <= tail_battery_reserve_fraction <= 1.0:
            raise ValueError("tail_battery_reserve_fraction must be between 0 and 1")
        if not 0.0 <= tail_renewable_uncertainty_fraction <= 1.0:
            raise ValueError(
                "tail_renewable_uncertainty_fraction must be between 0 and 1"
            )
        if not 0.0 <= tail_renewable_robustness <= 1.0:
            raise ValueError(
                "tail_renewable_robustness must be between 0 and 1"
            )
        if tail_renewable_calibration_days < 0:
            raise ValueError("tail_renewable_calibration_days must be >= 0")
        if not 0.0 <= battery_carbon_priority_fraction <= 1.0:
            raise ValueError(
                "battery_carbon_priority_fraction must be between 0 and 1"
            )
        self.grid_power_limit = grid_power_limit
        self.carbon_intensity_threshold_g_per_kwh = carbon_intensity_threshold_g_per_kwh
        self.plan_horizon_h = plan_horizon_h
        self.slot_duration_s = slot_duration_s
        self.replan_interval_h = replan_interval_h
        self.renewable_charge_fraction = renewable_charge_fraction
        self.day_hours = day_hours
        self.node_configs = node_configs or {}
        self.cluster_schedulers: list["JobQueueScheduler"] = cluster_schedulers or []
        self.replan_on_underutilization = replan_on_underutilization
        self.replan_request_cooldown_h = replan_request_cooldown_h
        self.pue = float(pue)
        self.carbon_aware_placement = bool(carbon_aware_placement)
        self.global_green_board = bool(global_green_board)
        self.forecast_tail_carbon_multiplier = float(
            forecast_tail_carbon_multiplier
        )
        self.carbon_regret_threshold_g_per_h = (
            None
            if carbon_regret_threshold_g_per_h is None
            else float(carbon_regret_threshold_g_per_h)
        )
        self.tail_battery_reserve_fraction = float(
            tail_battery_reserve_fraction
        )
        self.tail_renewable_uncertainty_fraction = float(
            tail_renewable_uncertainty_fraction
        )
        self.tail_renewable_robustness = float(tail_renewable_robustness)
        self.tail_renewable_calibration_days = float(
            tail_renewable_calibration_days
        )
        self.battery_carbon_aware_dispatch = bool(
            battery_carbon_aware_dispatch
        )
        self.battery_carbon_priority_fraction = float(
            battery_carbon_priority_fraction
        )
        # Reserve one replan interval before the formal queue deadline.  A plan
        # that targets the exact last 5-minute slot can miss by seconds at the
        # dispatcher boundary, or by hours if an earlier running job slips.
        # Keeping this configurable makes the carbon/SLA trade-off explicit.
        self.sla_start_safety_margin_h = float(
            replan_interval_h
            if sla_start_safety_margin_h is None
            else sla_start_safety_margin_h
        )
        # When True, run a local-search repair pass after the greedy packer:
        # try to pull grid-spilled jobs back onto the green board, relocating
        # already-placed green jobs within their SLA slack to make room.  Turns
        # the one-pass heuristic into a hill-climb that rearranges pieces to
        # admit a newcomer instead of spilling it (opt-in; default off).
        self._repack_repair = repack_repair

        self._current_plan: Optional[PuzzlePlan] = None
        self.history = PuzzlePlanHistory(plan_history_limit, pue=pue)
        self._last_plan_time: Optional[datetime] = None
        self._replan_requested: bool = False
        # Include unplaced and rejected jobs so they do not trigger a fresh
        # plan at every budget update. Keys are scoped to the last plan.
        self._pending_job_keys_at_last_plan: set[tuple[str, int]] = set()
        self._utilization_tracker: list[float] = []
        # Per-plan lookup tables for the dispatcher, rebuilt on every replan.
        # ``MultiQueuePolicy`` re-invokes ``select_jobs`` once per job it
        # dispatches in a step, and each invocation asks which rectangles are
        # planned/due — a full scan of ``placed_jobs`` every time.  Deriving
        # them once per plan makes those queries O(log n) or O(1).
        self._plan_index_cache: dict[Optional[str], _PlanIndex] = {}
        # Jobs first admitted through an extrapolated forecast tail remain
        # protected while pending/running.  Later replans can then spend the
        # held battery reserve against their newly visible renewable deficit.
        self._tail_protected_job_keys: set[tuple[str, int]] = set()
        self._last_tail_battery_reserve_wh = 0.0
        self._last_tail_battery_risk_wh = 0.0
        self._last_tail_renewable_calibration: tuple[
            TailRenewableCalibration, ...
        ] = ()
        self._renewable_history_cache_signal_id: Optional[int] = None
        self._renewable_history_cache_end: Optional[datetime] = None
        self._renewable_history_cache_values: list[float] = []

    # -- Public helpers for external integration ---------------------------

    def set_cluster_schedulers(self, schedulers: list["JobQueueScheduler"]) -> None:
        """Allow late binding of cluster schedulers after construction."""
        self.cluster_schedulers = list(schedulers)

    def request_replan(self, time: Optional[datetime] = None) -> None:
        """External trigger for replanning (e.g. after a job failure)."""
        self._replan_requested = True

    @property
    def current_plan(self) -> Optional[PuzzlePlan]:
        """Expose the current plan for visualization and debugging."""
        return self._current_plan

    @property
    def plan_history(self) -> list[PuzzlePlan]:
        """Expose the retained sample of detailed plans, in creation order."""
        return [plan for _, plan in self.history.indexed_plans]

    def get_due_planned_jobs(
        self,
        time: datetime,
        cluster_type: Optional[str] = None,
    ) -> list[JobRectangle]:
        """Return placed rectangles whose planned start slot has arrived.

        This is the interface between the budget policy and the
        ``PuzzleSchedulingPolicy``, which calls it every step to decide
        which jobs to dispatch.  When ``cluster_type`` is provided, only
        jobs planned for that cluster are returned.

        A rectangle is *due* once ``time`` reaches its ``start_slot_idx``
        (``start_slot_idx <= current_slot_idx``).  This is not limited to a
        single slot: a job that could not start exactly at its planned slot
        (e.g. nodes were busy) stays due on later steps instead of being
        orphaned, while jobs planned for a *future* slot are withheld so
        dispatch follows the plan's timing rather than front-running it.

        The result is ordered by planned start slot rather than by position in
        ``placed_jobs``.  Callers re-sort on a key that ends in ``job_id``, so
        the ordering is total and this does not change which jobs they pick.
        """
        plan = self._current_plan
        if plan is None or not plan.slots:
            return []

        offset_s = (time - plan.plan_start).total_seconds()
        current_slot_idx = int(offset_s // self.slot_duration_s)
        if current_slot_idx < 0:
            return []
        current_slot_idx = min(current_slot_idx, len(plan.slots) - 1)

        index = self._plan_index(cluster_type)
        due = bisect_right(index.start_slots, current_slot_idx)
        return index.by_start[:due]

    def _plan_index(self, cluster_type: Optional[str]) -> _PlanIndex:
        """Cached per-cluster lookup tables for the current plan."""
        index = self._plan_index_cache.get(cluster_type)
        if index is not None:
            return index

        plan = self._current_plan
        rects = [
            rect for rect in plan.placed_jobs
            if cluster_type is None or rect.cluster_type == cluster_type
        ] if plan is not None else []
        by_start = sorted(
            (r for r in rects if r.start_slot_idx is not None),
            key=lambda r: r.start_slot_idx,
        )
        index = _PlanIndex(
            job_ids=frozenset(r.job_id for r in rects),
            rects_by_id={r.job_id: r for r in rects},
            by_start=by_start,
            start_slots=[r.start_slot_idx for r in by_start],
        )
        self._plan_index_cache[cluster_type] = index
        return index

    def get_planned_job_ids(
        self,
        cluster_type: Optional[str] = None,
    ) -> frozenset[int]:
        """Return the ids of every job placed in the current plan.

        Lets the scheduling policy tell *planned* jobs (whose timing the
        plan owns) apart from *unplaced* jobs (which the plan could not
        fit and which a safety-net fallback may dispatch).
        """
        if self._current_plan is None:
            return frozenset()
        return self._plan_index(cluster_type).job_ids

    def current_slot_idx(self, time: datetime) -> Optional[int]:
        """Index of the plan slot covering ``time`` (clamped), or ``None``."""
        plan = self._current_plan
        if plan is None or not plan.slots:
            return None
        offset_s = (time - plan.plan_start).total_seconds()
        idx = int(offset_s // self.slot_duration_s)
        if idx < 0:
            return None
        return min(idx, len(plan.slots) - 1)

    def slot_carbon_intensity(self, time: datetime) -> Optional[float]:
        """Grid carbon intensity (gCO2/kWh) of the slot covering ``time``.

        The forecast signal a carbon-aware dispatcher compares against when
        deciding whether running a job *now* is dirtier than at its planned
        slot.  Returns ``None`` when no plan covers ``time``.
        """
        idx = self.current_slot_idx(time)
        if idx is None:
            return None
        return self._current_plan.slots[idx].carbon_intensity

    def rect_planned_carbon(self, rect: JobRectangle) -> Optional[float]:
        """Grid carbon intensity (gCO2/kWh) at ``rect``'s planned start slot."""
        plan = self._current_plan
        if plan is None or not plan.slots or rect.start_slot_idx is None:
            return None
        idx = min(max(rect.start_slot_idx, 0), len(plan.slots) - 1)
        return plan.slots[idx].carbon_intensity

    def planned_rects_by_id(
        self,
        cluster_type: Optional[str] = None,
    ) -> dict[int, JobRectangle]:
        """Map job_id -> its placed rectangle in the current plan.

        Lets the scheduling policy look up a pending job's planned slot and
        greenness so it can decide whether to front-run it now or hold it for
        the greener slot the plan found.

        The returned mapping is the plan's own cached index — treat it as
        read-only; it is rebuilt on the next replan.
        """
        if self._current_plan is None:
            return {}
        return self._plan_index(cluster_type).rects_by_id

    def get_cluster_compute_budgets(self, time: datetime) -> dict[str, float]:
        """Return each cluster's planned compute budget (W) for ``time``.

        Reads the per-cluster steering footprint in
        ``PuzzlePlan.cluster_slots`` at the slot covering ``time``.  On the
        global board this is reconstructed from the jobs actually placed; in
        legacy mode it is the fixed slice used during packing.  The allocation
        strategy uses it to keep runtime distribution aligned with the plan.
        Returns ``{}`` when no plan covers ``time``.
        """
        plan = self._current_plan
        if plan is None or not plan.cluster_slots:
            return {}

        offset_s = (time - plan.plan_start).total_seconds()
        idx = int(offset_s // self.slot_duration_s)

        budgets: dict[str, float] = {}
        for cluster_type, slots in plan.cluster_slots.items():
            if not slots:
                continue
            i = min(max(idx, 0), len(slots) - 1)
            # The cluster's runtime budget is its green board ceiling plus the
            # grid allocated to jobs. Idle on the other powered nodes must
            # also be covered; spare green within the share counts first.
            slot = slots[i]
            budgets[cluster_type] = max(
                slot.green_available_w + slot.grid_allocated_w,
                slot.allocated_w + slot.grid_allocated_w
                + max(0.0, slot.idle_reserved_w - slot.idle_allocated_w),
            )
        return budgets

    # -- BudgetGenerationPolicy interface ----------------------------------

    def generate_budget(
        self,
        time: datetime,
        renewable_signal: "HistoricalSignal",
        battery: "Storage",
        step_size: int = 3600,
    ) -> BudgetInfo:
        if self._should_replan(time):
            self._build_plan(time, renewable_signal, battery, step_size)
        return self._budget_at(time, renewable_signal, battery)

    # -- Internal: replanning triggers -------------------------------------

    def _arrived_pending_job_keys(self, time: datetime) -> set[tuple[str, int]]:
        """Jobs submitted by now, including those a plan cannot place."""
        return {
            (scheduler.cluster_type, job.job_id)
            for scheduler in self.cluster_schedulers
            if not scheduler.done
            for job in scheduler.job_queue.jobs
            if job.status == "pending" and job.arrival_time <= time
        }

    def _should_replan(self, time: datetime) -> bool:
        if self._current_plan is None:
            return True

        # A new submission must not wait for the periodic refresh or for a
        # dispatch-failure cooldown. All arrivals since the preceding budget
        # update are handled together by one plan.
        if not self._arrived_pending_job_keys(time).issubset(
            self._pending_job_keys_at_last_plan
        ):
            return True

        if self._replan_requested:
            cooldown_h = max(0.0, float(self.replan_request_cooldown_h))
            if self._last_plan_time is None or cooldown_h <= 0.0:
                self._replan_requested = False
                return True

            hours_since = (time - self._last_plan_time).total_seconds() / 3600
            if hours_since >= cooldown_h:
                self._replan_requested = False
                return True

        if time >= self._current_plan.plan_end:
            return True

        if self._last_plan_time is not None:
            hours_since = (time - self._last_plan_time).total_seconds() / 3600
            if hours_since >= self.replan_interval_h:
                return True

        # Replan when planned utilization is consistently low, if enabled.
        # The tracker measures planned area usage, not actual cluster
        # consumption, so keeping it enabled by default causes churn when the
        # workload is simply smaller than the green envelope.
        if self.replan_on_underutilization > 0 and len(self._utilization_tracker) >= 20:
            avg_util = sum(self._utilization_tracker) / len(self._utilization_tracker)
            if avg_util < self.replan_on_underutilization:
                self._utilization_tracker.clear()
                return True

        return False

    # -- Internal: plan construction ---------------------------------------

    def _build_plan(
        self,
        time: datetime,
        renewable_signal: "HistoricalSignal",
        battery: "Storage",
        step_size: int,
    ) -> None:
        """Build the protected plan for the lookahead horizon."""
        started_at = perf_counter()
        self._refresh_tail_protected_job_keys()
        job_rects = self._build_job_rectangles(time)
        visible_slots = int(
            self.plan_horizon_h * 3600 / self.slot_duration_s
        )
        slots = self._build_envelope(
            time, renewable_signal, battery, job_rects,
        )
        placed, unplaced, cluster_slots = self._pack_jobs_by_cluster(
            slots,
            job_rects,
            time,
            visible_slots=visible_slots,
        )
        protected_keys = set(self._tail_protected_job_keys)
        protected_keys.update(
            (rect.cluster_type, rect.job_id)
            for rect in placed
            if rect.end_slot_idx is not None and rect.end_slot_idx >= visible_slots
        )
        self._current_plan = self._assemble_plan(
            time=time,
            visible_slots=visible_slots,
            slots=slots,
            placed=placed,
            unplaced=unplaced,
            cluster_slots=cluster_slots,
            protected_keys=protected_keys,
        )
        self._tail_protected_job_keys = protected_keys
        self._plan_index_cache.clear()
        self.history.record(self._current_plan, perf_counter() - started_at)
        self._last_plan_time = time
        self._pending_job_keys_at_last_plan = self._arrived_pending_job_keys(time)
        self._replan_requested = False
        self._utilization_tracker.clear()

    def _assemble_plan(
        self,
        *,
        time: datetime,
        visible_slots: int,
        slots: list[PowerSlot],
        placed: list[JobRectangle],
        unplaced: list[JobRectangle],
        cluster_slots: dict[str, list[PowerSlot]],
        protected_keys: set[tuple[str, int]],
    ) -> PuzzlePlan:
        """Calculate aggregate metrics and materialise the active plan."""

        self._record_always_on_idle_power(slots, placed, cluster_slots)

        # Budget and utilization are measured against the green board only.
        # ``slots`` may extend past the real forecast window when a job longer
        # than the window forced extrapolation (see ``_build_envelope``).
        total_budget_wh = sum(
            s.green_available_w * s.duration_s / 3600.0 for s in slots
        )
        total_planned_wh = sum(
            s.allocated_w * s.duration_s / 3600.0 for s in slots
        )
        # Grid spill: the carbon penalty for jobs that did not fit green.
        grid_planned_wh = sum(
            s.grid_allocated_w * s.duration_s / 3600.0 for s in slots
        )
        grid_planned_co2_g = sum(
            s.grid_allocated_w
            * s.duration_s
            / 3_600_000.0
            * max(0.0, s.carbon_intensity)
            for s in slots
        )
        correction_planned_wh = sum(
            s.battery_correction_power_w * s.duration_s / 3600.0
            for s in slots[:visible_slots]
        )
        battery_avoided_co2_g = sum(
            s.battery_power_w
            * s.duration_s
            / 3_600_000.0
            * max(0.0, s.carbon_intensity)
            for s in slots[:visible_slots]
        )
        battery_correction_avoided_co2_g = sum(
            s.battery_correction_power_w
            * s.duration_s
            / 3_600_000.0
            * max(0.0, s.carbon_intensity)
            for s in slots[:visible_slots]
        )
        tail_slots = slots[visible_slots:]
        tail_pessimistic_wh = sum(
            s.renewable_pessimistic_power_w * s.duration_s / 3600.0
            for s in tail_slots
        )
        tail_expected_wh = sum(
            s.renewable_expected_power_w * s.duration_s / 3600.0
            for s in tail_slots
        )
        tail_optimistic_wh = sum(
            s.renewable_optimistic_power_w * s.duration_s / 3600.0
            for s in tail_slots
        )
        tail_planning_wh = sum(
            s.renewable_power_w * s.duration_s / 3600.0
            for s in tail_slots
        )

        plan = PuzzlePlan(
            plan_start=time,
            # The plan expires at the end of the REAL forecast window, even when
            # ``slots`` was extended past it to host a job longer than the window
            # (extrapolated slots repeat the current forecast — fictional future
            # we must replan over with real data): expiry drives the
            # ``time >= plan_end`` replan trigger in ``_should_replan``.
            plan_end=time + timedelta(hours=self.plan_horizon_h),
            slots=slots,
            placed_jobs=placed,
            unplaced_jobs=unplaced,
            total_budget_kwh=total_budget_wh / 1000.0,
            total_planned_kwh=total_planned_wh / 1000.0,
            grid_planned_kwh=grid_planned_wh / 1000.0,
            grid_planned_co2_g=grid_planned_co2_g,
            battery_tail_reserve_kwh=(
                self._last_tail_battery_reserve_wh / 1000.0
            ),
            battery_tail_risk_kwh=self._last_tail_battery_risk_wh / 1000.0,
            battery_correction_planned_kwh=correction_planned_wh / 1000.0,
            battery_avoided_co2_g=battery_avoided_co2_g,
            battery_correction_avoided_co2_g=(
                battery_correction_avoided_co2_g
            ),
            battery_carbon_aware_dispatch=self.battery_carbon_aware_dispatch,
            battery_carbon_priority_fraction=(
                self.battery_carbon_priority_fraction
            ),
            tail_protected_job_keys=tuple(sorted(protected_keys)),
            tail_renewable_uncertainty_fraction=(
                self.tail_renewable_uncertainty_fraction
            ),
            tail_renewable_pessimistic_kwh=tail_pessimistic_wh / 1000.0,
            tail_renewable_expected_kwh=tail_expected_wh / 1000.0,
            tail_renewable_optimistic_kwh=tail_optimistic_wh / 1000.0,
            tail_renewable_planning_kwh=tail_planning_wh / 1000.0,
            tail_renewable_robustness=self.tail_renewable_robustness,
            tail_renewable_calibration_days=(
                self.tail_renewable_calibration_days
            ),
            tail_renewable_calibration_samples=sum(
                point.sample_count
                for point in self._last_tail_renewable_calibration
            ),
            tail_renewable_calibrated=any(
                not point.used_fallback
                for point in self._last_tail_renewable_calibration
            ),
            tail_renewable_pessimistic_fraction_max=max(
                (
                    point.pessimistic_fraction
                    for point in self._last_tail_renewable_calibration
                ),
                default=0.0,
            ),
            tail_renewable_optimistic_fraction_max=max(
                (
                    point.optimistic_fraction
                    for point in self._last_tail_renewable_calibration
                ),
                default=0.0,
            ),
            tail_battery_effective_reserve_fraction=(
                self._effective_tail_battery_reserve_fraction()
            ),
            cluster_slots=cluster_slots,
        )
        return plan

    def _refresh_tail_protected_job_keys(self) -> None:
        """Forget extrapolation-backed jobs once they are no longer active."""
        active_keys: set[tuple[str, int]] = set()
        for scheduler in self.cluster_schedulers:
            cluster_type = str(getattr(scheduler, "cluster_type", ""))
            queue = getattr(scheduler, "job_queue", None)
            for job in getattr(queue, "jobs", []) or []:
                if getattr(job, "status", None) in {"pending", "running"}:
                    active_keys.add((cluster_type, int(job.job_id)))
        self._tail_protected_job_keys.intersection_update(active_keys)

    def _effective_tail_battery_reserve_fraction(self) -> float:
        """Physical recourse reserve used once a protected job is committed.

        Scenario uncertainty does not automatically increase this value.  A/B
        comparisons remain explicit: robustness changes the renewable tail
        commitment, while the independently configured battery reserve remains
        the physical recourse pool.
        """
        return self.tail_battery_reserve_fraction

    def _tail_battery_risk_wh(
        self,
        job_rects: list[JobRectangle],
        visible_slots: int,
    ) -> float:
        """Output-energy exposure that lies beyond the real forecast window.

        Pending rectangles contribute the unavoidable tail of their
        lowest-energy DVFS shape.  A protected running job contributes only
        the part of its remaining measured span that is still beyond this new
        real window.  This total is an observability metric; only the committed
        running subset is allowed to hold battery energy out of the board.
        """
        slot_h = self.slot_duration_s / 3600.0
        risk_wh = 0.0

        for rect in job_rects:
            if self._window_extension_need(rect, visible_slots) <= 0:
                continue
            variants = [
                variant for variant in (
                    rect.dvfs_variants
                    or [(rect.frequency_ghz, rect.power_w, rect.duration_s)]
                )
                if variant[1] > 0 and variant[2] > 0
            ]
            if not variants:
                continue
            _freq, power_w, duration_s = min(
                variants, key=lambda variant: (variant[1] * variant[2], variant[2])
            )
            needed = max(1, math.ceil(duration_s / self.slot_duration_s))
            tail_slots = max(
                0,
                max(0, rect.earliest_start_slot_idx) + needed - visible_slots,
            )
            risk_wh += tail_slots * power_w * slot_h

        risk_wh += self._protected_running_tail_risk_wh(visible_slots)
        return risk_wh

    def _protected_running_tail_risk_wh(self, visible_slots: int) -> float:
        """Tail exposure of protected work that has physically started."""
        slot_h = self.slot_duration_s / 3600.0
        risk_wh = 0.0
        for scheduler in self.cluster_schedulers:
            cluster_type = str(getattr(scheduler, "cluster_type", ""))
            rm = getattr(scheduler, "rm", None)
            if rm is None or not hasattr(rm, "get_running_allocations"):
                continue
            for allocation in rm.get_running_allocations():
                job = getattr(allocation, "job", None)
                if (
                    job is None
                    or getattr(job, "status", None) != "running"
                    or (
                        cluster_type,
                        int(job.job_id),
                    ) not in self._tail_protected_job_keys
                ):
                    continue
                remaining_s = max(
                    0.0, float(getattr(job, "remaining_time", 0.0) or 0.0)
                )
                needed = math.ceil(remaining_s / self.slot_duration_s)
                tail_slots = max(0, needed - visible_slots)
                if tail_slots <= 0:
                    continue
                power_w = self._running_allocation_board_power_w(
                    scheduler, allocation
                )
                risk_wh += tail_slots * power_w * slot_h

        return risk_wh

    def _protected_running_demand_by_slot(
        self, visible_slots: int,
    ) -> list[float]:
        """Facility power of running jobs previously backed by extrapolation."""
        demand_w = [0.0] * visible_slots
        if not self._tail_protected_job_keys:
            return demand_w

        for scheduler in self.cluster_schedulers:
            cluster_type = str(getattr(scheduler, "cluster_type", ""))
            rm = getattr(scheduler, "rm", None)
            if rm is None or not hasattr(rm, "get_running_allocations"):
                continue
            for allocation in rm.get_running_allocations():
                job = getattr(allocation, "job", None)
                if (
                    job is None
                    or getattr(job, "status", None) != "running"
                    or (
                        cluster_type,
                        int(job.job_id),
                    ) not in self._tail_protected_job_keys
                ):
                    continue
                remaining_s = max(
                    0.0, float(getattr(job, "remaining_time", 0.0) or 0.0)
                )
                if remaining_s <= 0:
                    continue
                span = min(
                    visible_slots,
                    max(1, math.ceil(remaining_s / self.slot_duration_s)),
                )
                power_w = self._running_allocation_board_power_w(
                    scheduler, allocation
                )
                for idx in range(span):
                    demand_w[idx] += power_w

        return demand_w

    # -- Envelope construction ---------------------------------------------

    def _renewable_history_values(
        self,
        time: datetime,
        signal: "HistoricalSignal",
        history_slots: int,
    ) -> list[float]:
        """Return a rolling realised-history window with incremental refresh."""
        if history_slots <= 0:
            return []

        signal_id = id(signal)
        cache_valid = (
            self._renewable_history_cache_signal_id == signal_id
            and self._renewable_history_cache_end is not None
            and len(self._renewable_history_cache_values) == history_slots
        )
        if cache_valid:
            elapsed_s = (
                time - self._renewable_history_cache_end
            ).total_seconds()
            delta_slots = round(elapsed_s / self.slot_duration_s)
            aligned = abs(
                elapsed_s - delta_slots * self.slot_duration_s
            ) <= 1e-6
            if aligned and 0 <= delta_slots <= history_slots:
                try:
                    additions = [
                        renewable_power_at(
                            signal,
                            self._renewable_history_cache_end
                            + timedelta(
                                seconds=idx * self.slot_duration_s
                            ),
                        )
                        for idx in range(delta_slots)
                    ]
                except (AttributeError, KeyError, ValueError):
                    additions = []
                    cache_valid = False
                if cache_valid:
                    if additions:
                        self._renewable_history_cache_values = (
                            self._renewable_history_cache_values + additions
                        )[-history_slots:]
                    self._renewable_history_cache_end = time
                    return list(self._renewable_history_cache_values)

        history_start = time - timedelta(
            seconds=history_slots * self.slot_duration_s
        )
        try:
            history = [
                renewable_power_at(
                    signal,
                    history_start
                    + timedelta(seconds=idx * self.slot_duration_s),
                )
                for idx in range(history_slots)
            ]
        except (AttributeError, KeyError, ValueError):
            history = []

        if len(history) == history_slots:
            self._renewable_history_cache_signal_id = signal_id
            self._renewable_history_cache_end = time
            self._renewable_history_cache_values = list(history)
        else:
            self._renewable_history_cache_signal_id = None
            self._renewable_history_cache_end = None
            self._renewable_history_cache_values = []
        return history

    def _calibrate_tail_renewable_scenarios(
        self,
        time: datetime,
        signal: "HistoricalSignal",
        *,
        n_slots: int,
        repeats: int,
    ) -> tuple[TailRenewableCalibration, ...]:
        """Calibrate repeated-window P10/P90 paths using only realised history."""
        max_lead = max(0, repeats - 1)
        if max_lead == 0:
            return ()

        history_slots = int(
            self.tail_renewable_calibration_days
            * 24.0
            * 3600.0
            / self.slot_duration_s
        )
        history_w = self._renewable_history_values(
            time, signal, history_slots
        )

        return calibrate_tail_renewable_uncertainty(
            history_w,
            slots_per_window=n_slots,
            max_lead_windows=max_lead,
            fallback_fraction=self.tail_renewable_uncertainty_fraction,
        )

    def _build_envelope(
        self,
        time: datetime,
        renewable_signal: "HistoricalSignal",
        battery: "Storage",
        job_rects: Optional[list[JobRectangle]] = None,
    ) -> list[PowerSlot]:
        """
        Build a discretized power envelope for the lookahead horizon.

        For each slot the envelope computes:
        * Forecasted renewable generation (PV + wind).
        * Charge reserve deducted from renewable budget — demand-aware: the
          reserve yields to queued/running compute, claiming only the PV the
          work cannot absorb.  Direct use beats store-then-discharge (no
          round-trip loss, no export while jobs wait), so when demand covers
          a slot's renewable the whole slot goes to the green board.
        * Battery contribution (a flexible SoC-bounded reservoir spread across
          PV-short slots — night or dim daytime).
        * Grid overflow capacity (only when carbon intensity is below threshold).
        """
        n_slots = int(self.plan_horizon_h * 3600 / self.slot_duration_s)
        horizon = timedelta(hours=self.plan_horizon_h)
        end_time = time + horizon

        # -- Renewable forecast for the full horizon -----------------------
        renewable_forecast = self._get_renewable_forecast(
            time, end_time, renewable_signal, n_slots,
        )
        # -- Carbon intensity forecast ------------------------------------
        carbon_forecast = self._get_carbon_forecast(
            time, end_time, renewable_signal, n_slots,
        )
        self._last_tail_renewable_calibration = ()
        # -- Compute demand curve (ASAP estimate) ---------------------------
        idle_floor_w = sum(
            self._static_idle_floor_w(scheduler)
            for scheduler in self.cluster_schedulers
        )
        demand_w = [
            power_w + idle_floor_w
            for power_w in self._asap_demand_by_slot(job_rects or [], n_slots)
        ]

        # -- Battery planning ---------------------------------------------
        # Battery is finite and is planned only over this REAL forecast window.
        # A fraction of its deliverable energy is held outside the compute board
        # only after a protected extrapolation-backed job has physically started
        # and still crosses this real window.  Pending risk remains observable
        # but cannot distort today's board speculatively.  Protected running-job
        # deficits receive first claim on the spendable pool.  The extrapolated
        # copies below never repeat SOC.
        deliverable_wh = deliverable_battery_energy_wh(battery)
        discharge_max = getattr(battery, "_p_discharge_max", float("inf"))
        free_wh = free_battery_energy_wh(battery)
        charge_max = getattr(battery, "_p_charge_max", float("inf"))
        charge_efficiency = max(
            1e-12, min(1.0, float(getattr(battery, "_charge_efficiency", 1.0)))
        )
        discharge_efficiency = max(
            1e-12,
            min(1.0, float(getattr(battery, "_discharge_efficiency", 1.0))),
        )

        # A slot charges when it is daytime and PV is abundant (>= a fraction of
        # the horizon peak); every other slot may draw the battery to fill green.
        peak_pv = max((max(0.0, v) for v in renewable_forecast), default=0.0)
        charge_pv_threshold = peak_pv * self.renewable_charge_fraction

        def _is_charging_slot(idx: int) -> bool:
            slot_time = time + timedelta(seconds=idx * self.slot_duration_s)
            return (
                renewable_forecast[idx] > 0
                and renewable_forecast[idx] >= charge_pv_threshold
                and is_daytime(slot_time, self.day_hours)
            )

        charging_slots = {i for i in range(n_slots) if _is_charging_slot(i)}
        protected_demand_w = [
            power_w + idle_floor_w
            for power_w in self._protected_running_demand_by_slot(n_slots)
        ]
        tail_risk_wh = self._tail_battery_risk_wh(job_rects or [], n_slots)
        committed_tail_risk_wh = self._protected_running_tail_risk_wh(n_slots)
        battery_dispatch = plan_tail_battery_dispatch(
            renewable_forecast_w=renewable_forecast,
            protected_demand_w=protected_demand_w,
            charging_slots=charging_slots,
            deliverable_energy_wh=deliverable_wh,
            future_tail_risk_wh=committed_tail_risk_wh,
            reserve_fraction=self._effective_tail_battery_reserve_fraction(),
            discharge_power_max_w=discharge_max,
            slot_duration_s=self.slot_duration_s,
            carbon_intensity_g_per_kwh=carbon_forecast,
            carbon_aware=self.battery_carbon_aware_dispatch,
            carbon_priority_fraction=self.battery_carbon_priority_fraction,
        )
        self._last_tail_battery_reserve_wh = battery_dispatch.held_reserve_wh
        self._last_tail_battery_risk_wh = tail_risk_wh

        # -- Build slots ---------------------------------------------------
        slots: list[PowerSlot] = []

        # Planned-SoC tracking for the charge reserve: the battery can only
        # absorb ``free_wh`` before it is full, so reserving PV beyond that
        # just fills the battery early and exports the rest of the reserved
        # power.  ``reserved_charge_wh`` accumulates planned charging and is
        # credited back by planned discharge (which frees storage room).
        reserved_charge_wh = 0.0  # Internal battery energy, after efficiency.

        for i in range(n_slots):
            slot_start = time + timedelta(seconds=i * self.slot_duration_s)
            slot_end = slot_start + timedelta(seconds=self.slot_duration_s)

            renewable_w = max(0.0, renewable_forecast[i])
            carbon_intensity = carbon_forecast[i]

            # A correction may discharge even in a nominal PV charging slot
            # when the protected running load exceeds its now-visible forecast.
            slot_battery_w = battery_dispatch.power_w[i]
            correction_battery_w = battery_dispatch.correction_power_w[i]
            if i in charging_slots and slot_battery_w <= 0.0:
                # Reserve a fraction of renewable power for charging, but
                # yield to demonstrable compute demand: charging only claims
                # the renewable the queued/running work cannot use directly.
                remaining_room_wh = max(0.0, free_wh - reserved_charge_wh)
                charge_limit = (
                    remaining_room_wh
                    / charge_efficiency
                    * 3600.0
                    / self.slot_duration_s
                )
                surplus_w = max(0.0, renewable_w - demand_w[i])
                charge_reserve = min(
                    renewable_w * self.renewable_charge_fraction,
                    surplus_w,
                    charge_limit,
                    charge_max,
                )
                reserved_charge_wh += (
                    charge_reserve
                    * self.slot_duration_s
                    / 3600.0
                    * charge_efficiency
                )
            else:
                charge_reserve = 0.0
                # Planned discharge frees room for later charging slots.
                reserved_charge_wh -= (
                    slot_battery_w
                    * self.slot_duration_s
                    / 3600.0
                    / discharge_efficiency
                )

            budget_renewable = max(0.0, renewable_w - charge_reserve)

            # Grid: only expose as overflow when carbon intensity is below
            # threshold.  Grid is NOT part of the green board — it is the
            # penalty channel for jobs the puzzle could not place green.
            low_carbon = (
                self.carbon_intensity_threshold_g_per_kwh > 0
                and carbon_intensity <= self.carbon_intensity_threshold_g_per_kwh
            )
            grid_w = self.grid_power_limit if low_carbon else 0.0

            # Green board ceiling = renewable budget + planned battery only.
            green_available = budget_renewable + slot_battery_w

            slots.append(PowerSlot(
                start_time=slot_start,
                end_time=slot_end,
                duration_s=float(self.slot_duration_s),
                renewable_power_w=renewable_w,
                battery_power_w=slot_battery_w,
                battery_correction_power_w=correction_battery_w,
                renewable_pessimistic_power_w=renewable_w,
                renewable_expected_power_w=renewable_w,
                renewable_optimistic_power_w=renewable_w,
                is_extrapolated=False,
                grid_power_w=grid_w,
                green_available_w=green_available,
                carbon_intensity=carbon_intensity,
                charge_reserve_w=charge_reserve,
            ))

        # -- Horizon extrapolation for jobs longer than the window ----------
        # A job none of whose variants finishes inside the real forecast
        # window would stay unplaced in every replan.  Extend the horizon with
        # explicit pessimistic, expected, and optimistic renewable scenarios.
        # Robustness=0 (default) commits to expected; robustness=1 commits to
        # pessimistic, and intermediate values interpolate between them.
        # Uncertainty=0 reproduces the version 0.4 forecast exactly.  Finite
        # battery SOC and charge reservation are not copied: doing so would
        # invent the same stored energy per window.
        # The held battery reserve can only enter a later plan once a real
        # forecast exposes a protected running job's deficit.  The
        # packer still forces every job to *start* inside the real window and
        # keeps window-fitting jobs out of the extrapolated zone entirely
        # (see ``_pack_jobs``), so short jobs never plan against repeated
        # forecast data.
        repeats = self._forecast_repeats(job_rects or [], n_slots)
        if repeats > 1:
            base = list(slots)
            calibration = self._calibrate_tail_renewable_scenarios(
                time,
                renewable_signal,
                n_slots=n_slots,
                repeats=repeats,
            )
            self._last_tail_renewable_calibration = calibration
            for k in range(1, repeats):
                point = calibration[k - 1]
                shift = timedelta(seconds=k * n_slots * self.slot_duration_s)
                for slot in base:
                    expected_w = max(0.0, slot.renewable_expected_power_w)
                    pessimistic_w = expected_w * (
                        1.0 - point.pessimistic_fraction
                    )
                    optimistic_w = expected_w * (
                        1.0 + point.optimistic_fraction
                    )
                    planning_w = expected_w - self.tail_renewable_robustness * (
                        expected_w - pessimistic_w
                    )
                    slots.append(replace(
                        slot,
                        start_time=slot.start_time + shift,
                        end_time=slot.end_time + shift,
                        renewable_power_w=planning_w,
                        renewable_pessimistic_power_w=pessimistic_w,
                        renewable_expected_power_w=expected_w,
                        renewable_optimistic_power_w=optimistic_w,
                        is_extrapolated=True,
                        battery_power_w=0.0,
                        battery_correction_power_w=0.0,
                        green_available_w=planning_w,
                        charge_reserve_w=0.0,
                    ))

        return slots

    def _window_extension_need(self, rect: JobRectangle, n_visible: int) -> int:
        """Slots this job needs beyond the real forecast window, or 0.

        A job warrants forecast extrapolation only when it could start inside
        the real window (``n_visible`` slots) and yet no DVFS variant finishes
        inside it.  Returns the total slot count such a job needs from slot 0
        (earliest start offset + its longest variant, so every variant stays
        searchable), and 0 for every job the real window can handle — or that
        cannot start in it, which is the next window's problem as before.
        """
        min_start = max(0, rect.earliest_start_slot_idx)
        if min_start >= n_visible:
            return 0
        variants = rect.dvfs_variants or [
            (rect.frequency_ghz, rect.power_w, rect.duration_s)
        ]
        longest = 0
        for _freq, power, duration in variants:
            if power <= 0:
                continue
            n_needed = max(1, math.ceil(duration / self.slot_duration_s))
            if min_start + n_needed <= n_visible:
                return 0
            longest = max(longest, n_needed)
        if longest == 0:
            return 0
        return min_start + longest

    def _forecast_repeats(
        self, job_rects: list[JobRectangle], n_visible: int,
    ) -> int:
        """How many copies of the forecast window the plan must cover (>= 1)."""
        repeats = 1
        for rect in job_rects:
            need = self._window_extension_need(rect, n_visible)
            if need > 0:
                repeats = max(repeats, math.ceil(need / n_visible))
        return repeats

    def _get_renewable_forecast(
        self,
        start: datetime,
        end: datetime,
        signal: "HistoricalSignal",
        n_slots: int,
    ) -> list[float]:
        """Retrieve renewable generation forecast as a list of per-slot values."""
        try:
            forecast_dict = signal.forecast(
                start_time=start,
                end_time=end,
                column="renewable_gen",
                frequency=timedelta(seconds=self.slot_duration_s),
                resample_method="ffill",
            )
            values = [float(v) for v in forecast_dict.values()]
        except (ValueError, KeyError):
            # Fallback: repeat current value.
            try:
                current = renewable_power_at(signal, start)
            except (ValueError, KeyError):
                current = 0.0
            values = [current] * n_slots

        # Pad or trim to exactly n_slots.
        if len(values) < n_slots:
            pad = values[-1] if values else 0.0
            values.extend([pad] * (n_slots - len(values)))
        return values[:n_slots]

    def _get_carbon_forecast(
        self,
        start: datetime,
        end: datetime,
        signal: "HistoricalSignal",
        n_slots: int,
    ) -> list[float]:
        """Retrieve carbon intensity forecast as a list of per-slot values."""
        try:
            forecast_dict = signal.forecast(
                start_time=start,
                end_time=end,
                column="carbon_intensity",
                frequency=timedelta(seconds=self.slot_duration_s),
                resample_method="ffill",
            )
            values = [float(v) for v in forecast_dict.values()]
        except (ValueError, KeyError):
            values = [0.0] * n_slots

        if len(values) < n_slots:
            pad = values[-1] if values else 0.0
            values.extend([pad] * (n_slots - len(values)))
        return values[:n_slots]

    def _asap_demand_by_slot(
        self,
        job_rects: list[JobRectangle],
        n_slots: int,
    ) -> list[float]:
        """Power (W) the queued+running work would draw per slot if run ASAP.

        Front-loads each pending rectangle at its earliest feasible slot and
        adds running jobs' total facility power over their remaining time.  Node
        capacity is deliberately ignored — this is an optimistic "how much PV
        could compute absorb right now" curve, used only to size how much of
        the renewable the battery charge reserve may claim.  Optimism errs
        toward direct PV use, which is the cheaper direction (no battery
        round-trip, no export while jobs wait); the curve refreshes on every
        replan.
        """
        demand = [0.0] * n_slots
        for rect in job_rects:
            if rect.power_w <= 0:
                continue
            start = max(0, rect.earliest_start_slot_idx)
            if start >= n_slots:
                continue
            span = max(1, math.ceil(rect.duration_s / self.slot_duration_s))
            for s in range(start, min(n_slots, start + span)):
                demand[s] += rect.power_w

        for scheduler in self.cluster_schedulers:
            rm = getattr(scheduler, "rm", None)
            if rm is None or not hasattr(rm, "get_running_allocations"):
                continue
            for allocation in rm.get_running_allocations():
                job = getattr(allocation, "job", None)
                if job is None or getattr(job, "status", None) != "running":
                    continue
                remaining_s = max(
                    0.0, float(getattr(job, "remaining_time", 0.0) or 0.0)
                )
                if remaining_s <= 0:
                    continue
                power_w = self._running_allocation_board_power_w(
                    scheduler, allocation,
                )
                if power_w <= 0:
                    continue
                span = min(
                    n_slots,
                    max(1, math.ceil(remaining_s / self.slot_duration_s)),
                )
                for s in range(span):
                    demand[s] += power_w

        return demand

    # -- Job rectangle construction ----------------------------------------

    def _build_job_rectangles(self, time: datetime) -> list[JobRectangle]:
        """
        Convert pending jobs from all cluster schedulers into rectangles.

        With a fully powered, static-on cluster, its exact idle floor is
        reserved before packing and rectangles contain only incremental dynamic
        facility power. Otherwise each rectangle includes its nodes' idle draw.
        Both modes account for idle exactly once.
        """
        from vessim.wave.scheduler.runtime.execution_models import (
            compute_runtime_components,
        )

        rectangles: list[JobRectangle] = []

        for scheduler in self.cluster_schedulers:
            if scheduler.done:
                continue

            cluster_type = scheduler.cluster_type
            cfg = self.node_configs.get(cluster_type, {})
            idle_power = cfg.get("idle_power_w", 0)
            job_idle_power = 0.0 if self._static_idle_floor_w(scheduler) > 0 else idle_power
            base_freq = cfg.get("base_frequency_ghz", 2.0)
            mem_bw = cfg.get("memory_bandwidth_gb_s", 100.0)
            freq_table = {
                float(f): float(p)
                for f, p in cfg.get("frequencies", {}).items()
            }

            # Workload files contain future submissions for the simulator.
            # They are not information available to this online planner.
            pending = [
                j for j in scheduler.job_queue.jobs
                if j.status == "pending" and j.arrival_time <= time
            ]

            for job in pending:
                # Skip jobs that violate their queue's node limit: they
                # are guaranteed to be rejected at dispatch
                # (``requested_nodes_exceeds_queue_limit``), so packing
                # them only wastes planned budget area.
                queue_max_nodes = getattr(job, "queue_max_nodes", None)
                if queue_max_nodes is not None and job.num_nodes > queue_max_nodes:
                    continue

                earliest_start_slot_idx = 0

                # SLA start deadline: the job must start by
                # queue_enter_time + queue_max_wait_s.  No max_wait → unbounded.
                max_wait_s = getattr(job, "queue_max_wait_s", None)
                if max_wait_s is None:
                    latest_start_slot_idx: Optional[int] = None
                else:
                    queue_enter = getattr(job, "queue_enter_time", None) or job.arrival_time
                    latest_offset_s = (
                        queue_enter + timedelta(seconds=float(max_wait_s)) - time
                    ).total_seconds()
                    latest_offset_s -= self.sla_start_safety_margin_h * 3600.0
                    latest_start_slot_idx = max(
                        earliest_start_slot_idx,
                        int(latest_offset_s // self.slot_duration_s),
                    )

                # Use base frequency for rectangle estimation.
                frequency = base_freq
                dynamic_power = freq_table.get(
                    frequency,
                    cfg.get("max_power_w", 0) - idle_power,
                )

                components = compute_runtime_components(
                    job, frequency, job.num_nodes, mem_bw,
                )
                duration_s = components.total_time

                # A static-on floor covers every configured node, including
                # occupied nodes. Only its incremental dynamic draw remains
                # on a job rectangle; other modes include the node's idle draw.
                total_power = (
                    (job_idle_power + dynamic_power) * job.num_nodes * self.pue
                )
                energy_wh = total_power * duration_s / 3600.0

                # DVFS variants: one (frequency, total facility power, duration)
                # per available frequency.  Lower frequency = flatter+longer,
                # higher = taller+shorter.  The packer deforms the piece into
                # whichever variant fits the green envelope, raising frequency
                # to fit a narrow PV peak before spilling to grid.
                dvfs_variants: list[tuple[float, float, float]] = []
                for var_freq, var_dyn in sorted(freq_table.items()):
                    var_comp = compute_runtime_components(
                        job, var_freq, job.num_nodes, mem_bw,
                    )
                    dvfs_variants.append((
                        float(var_freq),
                        (job_idle_power + var_dyn) * job.num_nodes * self.pue,
                        var_comp.total_time,
                    ))

                rectangles.append(JobRectangle(
                    job_id=job.job_id,
                    power_w=total_power,
                    duration_s=duration_s,
                    num_nodes=job.num_nodes,
                    frequency_ghz=frequency,
                    energy_wh=energy_wh,
                    cluster_type=cluster_type,
                    arrival_time=job.arrival_time,
                    earliest_start_slot_idx=earliest_start_slot_idx,
                    latest_start_slot_idx=latest_start_slot_idx,
                    dvfs_variants=dvfs_variants,
                ))

        return rectangles

    # -- Cluster-local bin-packing ----------------------------------------

    def _pack_jobs_by_cluster(
        self,
        slots: list[PowerSlot],
        job_rects: list[JobRectangle],
        time: datetime,
        visible_slots: Optional[int] = None,
    ) -> tuple[list[JobRectangle], list[JobRectangle], dict[str, list[PowerSlot]]]:
        """Pack jobs globally or in reproducible legacy cluster-local slices.

        ``visible_slots`` is the length of the real forecast window; slots
        beyond it are extrapolated copies (see ``_build_envelope``).  ``None``
        means the whole board is real.
        """
        if self.global_green_board:
            return self._pack_jobs_global(
                slots, job_rects, time, visible_slots=visible_slots,
            )

        rects_by_cluster: dict[str, list[JobRectangle]] = {}
        for rect in job_rects:
            rects_by_cluster.setdefault(rect.cluster_type, []).append(rect)

        cluster_slots = self._build_cluster_slot_plans(slots, job_rects, time)
        placed: list[JobRectangle] = []
        unplaced: list[JobRectangle] = []

        for cluster_type in sorted(cluster_slots):
            rects = rects_by_cluster.get(cluster_type, [])
            slot_plan = cluster_slots.get(cluster_type)
            if slot_plan is None:
                slot_plan = [replace(slot, allocated_w=0.0) for slot in slots]
                cluster_slots[cluster_type] = slot_plan

            scheduler = next(
                (
                    scheduler for scheduler in self.cluster_schedulers
                    if scheduler.cluster_type == cluster_type
                ),
                None,
            )
            if scheduler is not None:
                self._reserve_running_jobs(slot_plan, scheduler)

            if rects:
                node_capacity, node_reserved = self._cluster_node_ledger(
                    scheduler, len(slot_plan),
                )
                cluster_placed, cluster_unplaced = self._pack_jobs(
                    slot_plan,
                    rects,
                    repair=self._repack_repair,
                    node_capacity=node_capacity,
                    node_reserved=node_reserved,
                    visible_slots=visible_slots,
                )
                placed.extend(cluster_placed)
                unplaced.extend(cluster_unplaced)

        missing_rects = [
            rect for rect in job_rects
            if rect.cluster_type not in cluster_slots
        ]
        unplaced.extend(missing_rects)

        for idx, slot in enumerate(slots):
            slot.allocated_w = sum(
                cluster_plan[idx].allocated_w
                for cluster_plan in cluster_slots.values()
            )
            slot.idle_reserved_w = sum(
                cluster_plan[idx].idle_reserved_w
                for cluster_plan in cluster_slots.values()
            )
            slot.grid_allocated_w = sum(
                cluster_plan[idx].grid_allocated_w
                for cluster_plan in cluster_slots.values()
            )

        return placed, unplaced, cluster_slots

    def _pack_jobs_global(
        self,
        slots: list[PowerSlot],
        job_rects: list[JobRectangle],
        time: datetime,
        visible_slots: Optional[int] = None,
    ) -> tuple[list[JobRectangle], list[JobRectangle], dict[str, list[PowerSlot]]]:
        """Pack every cluster against one shared green and grid envelope.

        Power is globally fungible, but compute nodes are not: each rectangle
        is checked against the independent node ledger of its own cluster.
        Running jobs reserve the shared board first.  Pending rectangles are
        then considered in global SLA order, so capacity that one cluster does
        not use remains immediately available to another instead of becoming
        stranded inside a fixed slice.

        ``cluster_slots`` is reconstructed after packing from the exact green
        and grid footprint assigned to each cluster.  These per-cluster slots
        are runtime steering weights, not pre-packing quotas.
        """
        idle_reservations = self._reserve_static_idle_floor(slots)
        schedulers_by_type = {
            scheduler.cluster_type: scheduler
            for scheduler in self.cluster_schedulers
            if not getattr(scheduler, "done", False)
            or scheduler.cluster_type in idle_reservations
        }
        cluster_types = set(schedulers_by_type)
        cluster_types.update(rect.cluster_type for rect in job_rects)
        ordered_cluster_types = sorted(cluster_types)
        n_slots = len(slots)

        node_capacities: dict[str, Optional[int]] = {}
        running_node_reserved: dict[str, list[int]] = {}
        running_green: dict[str, list[float]] = {
            cluster_type: list(idle_reservations.get(
                cluster_type, ([0.0] * n_slots, [0.0] * n_slots),
            )[0])
            for cluster_type in ordered_cluster_types
        }
        running_grid: dict[str, list[float]] = {
            cluster_type: list(idle_reservations.get(
                cluster_type, ([0.0] * n_slots, [0.0] * n_slots),
            )[1])
            for cluster_type in ordered_cluster_types
        }

        # Running work is unavoidable and gets first claim on the global
        # board.  Capture each cluster's exact delta while reserving it so the
        # runtime steering slices can later reproduce the aggregate plan.
        for cluster_type in ordered_cluster_types:
            scheduler = schedulers_by_type.get(cluster_type)
            capacity, reserved = self._cluster_node_ledger(scheduler, n_slots)
            node_capacities[cluster_type] = capacity
            running_node_reserved[cluster_type] = list(reserved)
            if scheduler is None:
                continue

            green_before = [slot.allocated_w for slot in slots]
            grid_before = [slot.grid_allocated_w for slot in slots]
            self._reserve_running_jobs(slots, scheduler)
            for idx, slot in enumerate(slots):
                running_green[cluster_type][idx] += max(
                    0.0, slot.allocated_w - green_before[idx],
                )
                running_grid[cluster_type][idx] += max(
                    0.0, slot.grid_allocated_w - grid_before[idx],
                )

        # One global pass is both faster and more faithful to the shared-board
        # model.  The v0.2 implementation called ``_pack_jobs`` once per
        # rectangle; every call rebuilt the 24-hour gap/obstacle caches and
        # collapsed the previous placements back into a bottom reservation.
        # ``_pack_jobs`` now owns one reusable board index while selecting the
        # appropriate independent node ledger for each rectangle's cluster.
        placed, unplaced = self._pack_jobs(
            slots,
            job_rects,
            repair=False,
            node_capacity_by_cluster=node_capacities,
            node_reserved_by_cluster=running_node_reserved,
            visible_slots=visible_slots,
        )

        if self._repack_repair and placed:
            # The repair helper expects the immutable bottom reservation: live
            # green plus the partial-green footprint of jobs still on grid.
            reserved_stack = [
                sum(running_green[c][idx] for c in ordered_cluster_types)
                for idx in range(n_slots)
            ]
            for rect in placed:
                if not rect.used_grid:
                    continue
                for idx, green_w in rect.green_allocation_w_by_slot.items():
                    if 0 <= idx < n_slots:
                        reserved_stack[idx] += green_w

            for cluster_type in ordered_cluster_types:
                self._repair_relocate(
                    slots,
                    placed,
                    reserved_stack,
                    node_capacity=node_capacities.get(cluster_type),
                    node_reserved=running_node_reserved.get(cluster_type),
                    visible_slots=visible_slots,
                    repair_cluster_type=cluster_type,
                )
            self._sync_allocated_from_placed(slots, placed, reserved_stack)

        cluster_slots = self._build_global_cluster_slot_plans(
            slots,
            placed,
            ordered_cluster_types,
            running_green,
            running_grid,
        )
        return placed, unplaced, cluster_slots

    @staticmethod
    def _build_global_cluster_slot_plans(
        slots: list[PowerSlot],
        placed: list[JobRectangle],
        cluster_types: list[str],
        running_green: dict[str, list[float]],
        running_grid: dict[str, list[float]],
    ) -> dict[str, list[PowerSlot]]:
        """Attribute the shared board's exact planned footprint by cluster."""
        n_slots = len(slots)
        green_by_cluster = {
            cluster_type: list(running_green.get(cluster_type, [0.0] * n_slots))
            for cluster_type in cluster_types
        }
        grid_by_cluster = {
            cluster_type: list(running_grid.get(cluster_type, [0.0] * n_slots))
            for cluster_type in cluster_types
        }

        for rect in placed:
            if rect.start_slot_idx is None or rect.end_slot_idx is None:
                continue
            green = green_by_cluster.setdefault(rect.cluster_type, [0.0] * n_slots)
            grid = grid_by_cluster.setdefault(rect.cluster_type, [0.0] * n_slots)
            last = min(rect.end_slot_idx, n_slots - 1)
            if rect.used_grid:
                for idx in range(rect.start_slot_idx, last + 1):
                    green[idx] += rect.green_allocation_w_by_slot.get(idx, 0.0)
                    grid[idx] += rect.grid_allocation_w_by_slot.get(idx, 0.0)
            else:
                for idx in range(rect.start_slot_idx, last + 1):
                    green[idx] += rect.power_w

        cluster_slots: dict[str, list[PowerSlot]] = {}
        for cluster_type in sorted(green_by_cluster):
            attributed: list[PowerSlot] = []
            for idx, slot in enumerate(slots):
                green_w = green_by_cluster[cluster_type][idx]
                grid_w = grid_by_cluster[cluster_type][idx]
                if slot.green_available_w > 0:
                    scenario_share = green_w / slot.green_available_w
                    renewable_budget_w = max(
                        0.0, slot.renewable_power_w - slot.charge_reserve_w,
                    )
                    renewable_w = (
                        green_w * renewable_budget_w / slot.green_available_w
                    )
                    battery_w = (
                        green_w * slot.battery_power_w / slot.green_available_w
                    )
                    correction_battery_w = (
                        green_w
                        * slot.battery_correction_power_w
                        / slot.green_available_w
                    )
                else:
                    scenario_share = 0.0
                    renewable_w = 0.0
                    battery_w = 0.0
                    correction_battery_w = 0.0
                attributed.append(replace(
                    slot,
                    renewable_power_w=renewable_w,
                    renewable_pessimistic_power_w=(
                        slot.renewable_pessimistic_power_w * scenario_share
                    ),
                    renewable_expected_power_w=(
                        slot.renewable_expected_power_w * scenario_share
                    ),
                    renewable_optimistic_power_w=(
                        slot.renewable_optimistic_power_w * scenario_share
                    ),
                    battery_power_w=battery_w,
                    battery_correction_power_w=correction_battery_w,
                    grid_power_w=grid_w,
                    green_available_w=green_w,
                    charge_reserve_w=0.0,
                    allocated_w=green_w,
                    idle_reserved_w=0.0,
                    grid_allocated_w=grid_w,
                ))
            cluster_slots[cluster_type] = attributed

        return cluster_slots

    def _build_cluster_slot_plans(
        self,
        slots: list[PowerSlot],
        job_rects: list[JobRectangle],
        time: datetime,
    ) -> dict[str, list[PowerSlot]]:
        """Build one scaled copy of the power envelope per active cluster."""
        idle_reservations = self._reserve_static_idle_floor(slots)
        weights = self._cluster_budget_weights(job_rects, time)
        total_weight = sum(weights.values())
        if total_weight <= 0 and not idle_reservations:
            return {}

        cluster_slots: dict[str, list[PowerSlot]] = {}
        for cluster_type in sorted(set(weights) | set(idle_reservations)):
            share = weights.get(cluster_type, 0.0) / total_weight if total_weight > 0 else 0.0
            idle_green, idle_grid = idle_reservations.get(
                cluster_type, ([0.0] * len(slots), [0.0] * len(slots)),
            )
            attributed = []
            for idx, slot in enumerate(slots):
                green_w = idle_green[idx] + slot.remaining_w * share
                source_share = green_w / slot.green_available_w if slot.green_available_w > 0 else 0.0
                attributed.append(replace(
                    slot,
                    renewable_power_w=slot.renewable_power_w * source_share,
                    renewable_pessimistic_power_w=slot.renewable_pessimistic_power_w * source_share,
                    renewable_expected_power_w=slot.renewable_expected_power_w * source_share,
                    renewable_optimistic_power_w=slot.renewable_optimistic_power_w * source_share,
                    battery_power_w=slot.battery_power_w * source_share,
                    battery_correction_power_w=slot.battery_correction_power_w * source_share,
                    grid_power_w=idle_grid[idx] + max(
                        0.0, slot.grid_power_w - slot.grid_allocated_w,
                    ) * share,
                    green_available_w=green_w,
                    charge_reserve_w=slot.charge_reserve_w * source_share,
                    allocated_w=idle_green[idx],
                    grid_allocated_w=idle_grid[idx],
                    idle_reserved_w=idle_green[idx] + idle_grid[idx],
                    idle_allocated_w=idle_green[idx] + idle_grid[idx],
                ))
            cluster_slots[cluster_type] = attributed

        return cluster_slots

    def _cluster_node_ledger(
        self,
        scheduler,
        n_slots: int,
    ) -> tuple[Optional[int], list[int]]:
        """Physical node capacity and per-slot nodes held by running jobs.

        The packer stacks *power*, but a slot can only run as many rectangles
        as the cluster has nodes: without this ledger the plan happily packs
        40 concurrent single-node jobs onto a 33-node cluster whenever the
        power fits, and the dispatcher then slips them at runtime (due jobs
        retrying slot after slot).  ``None`` capacity means unknown (e.g.
        synthetic tests without a scheduler) and disables the check.
        """
        node_reserved = [0] * n_slots
        if scheduler is None:
            return None, node_reserved

        rm = getattr(scheduler, "rm", None)
        node_count = len(getattr(rm, "nodes", {}) or {})
        if node_count <= 0:
            return None, node_reserved

        if hasattr(rm, "get_running_allocations"):
            for allocation in rm.get_running_allocations():
                job = getattr(allocation, "job", None)
                if job is None or getattr(job, "status", None) != "running":
                    continue
                remaining_s = max(
                    0.0, float(getattr(job, "remaining_time", 0.0) or 0.0)
                )
                if remaining_s <= 0:
                    continue
                held = len(getattr(allocation, "node_names", []) or [])
                if held <= 0:
                    held = int(getattr(job, "num_nodes", 0) or 0)
                if held <= 0:
                    continue
                span = min(
                    n_slots,
                    max(1, math.ceil(remaining_s / self.slot_duration_s)),
                )
                for idx in range(span):
                    node_reserved[idx] += held

        return node_count, node_reserved

    def _static_idle_floor_w(self, scheduler) -> float:
        """Exact facility idle floor when every node is kept powered on."""
        rm = getattr(scheduler, "rm", None)
        if not getattr(rm, "keep_idle_nodes_on", False):
            return 0.0
        nodes = getattr(rm, "nodes", {}) or {}
        powered_count = rm.get_powered_on_node_count()
        if not nodes or powered_count != len(nodes):
            return 0.0
        idle_per_node = self.node_configs.get(scheduler.cluster_type, {}).get(
            "idle_power_w", 0.0,
        )
        return max(0.0, float(idle_per_node)) * powered_count * self.pue

    def _reserve_static_idle_floor(
        self, slots: list[PowerSlot],
    ) -> dict[str, tuple[list[float], list[float]]]:
        """Give static idle first claim on green supply before any job is packed."""
        reservations = {}
        for scheduler in sorted(self.cluster_schedulers, key=lambda item: item.cluster_type):
            idle_w = self._static_idle_floor_w(scheduler)
            if idle_w <= 0:
                continue
            green = []
            grid = []
            for slot in slots:
                green_w = min(idle_w, slot.remaining_w)
                grid_w = idle_w - green_w
                slot.allocated_w += green_w
                slot.grid_allocated_w += grid_w
                slot.idle_reserved_w += idle_w
                slot.idle_allocated_w += idle_w
                green.append(green_w)
                grid.append(grid_w)
            reservations[scheduler.cluster_type] = (green, grid)
        return reservations

    def _running_allocation_board_power_w(self, scheduler, allocation) -> float:
        """Running draw beyond idle already reserved on the static board."""
        if self._static_idle_floor_w(scheduler) > 0:
            return self._running_allocation_dynamic_power_w(scheduler, allocation)
        return self._running_allocation_total_power_w(scheduler, allocation)

    def _reserve_running_jobs(self, slots: list[PowerSlot], scheduler) -> None:
        """Carry unavoidable running power into a new plan.

        Running jobs cannot be paused when a later replan lowers the green
        envelope. Reserve their measured facility draw for their remaining
        span, excluding idle already covered by a static floor. Each slot consumes
        its remaining green first and records the unavoidable deficit as grid,
        so the runtime budget never drops below work that is already running.
        """
        rm = getattr(scheduler, "rm", None)
        if rm is None or not hasattr(rm, "get_running_allocations"):
            return

        for allocation in rm.get_running_allocations():
            job = getattr(allocation, "job", None)
            if job is None or getattr(job, "status", None) != "running":
                continue

            remaining_s = max(0.0, float(getattr(job, "remaining_time", 0.0) or 0.0))
            if remaining_s <= 0:
                continue

            power_w = self._running_allocation_board_power_w(scheduler, allocation)
            if power_w <= 0:
                continue

            n_slots = min(
                len(slots),
                max(1, math.ceil(remaining_s / self.slot_duration_s)),
            )
            for idx in range(n_slots):
                slot = slots[idx]
                remaining_green = max(
                    0.0, slot.green_available_w - slot.allocated_w,
                )
                green_take = min(power_w, remaining_green)
                slot.allocated_w += green_take
                slot.grid_allocated_w += power_w - green_take

    def _running_allocation_total_power_w(self, scheduler, allocation) -> float:
        """Return measured facility power (idle + dynamic) for an allocation."""
        rm = getattr(scheduler, "rm", None)
        total_it_power = 0.0
        for node_name in getattr(allocation, "node_names", []) or []:
            node = getattr(rm, "nodes", {}).get(node_name) if rm is not None else None
            if node is None or not hasattr(node, "measure"):
                continue
            total_it_power += max(0.0, float(node.measure()))

        if total_it_power > 0:
            return total_it_power * self.pue

        cluster_type = getattr(scheduler, "cluster_type", "")
        cfg = self.node_configs.get(cluster_type, {})
        idle_power = float(cfg.get("idle_power_w", 0.0))
        dynamic_power = self._running_allocation_dynamic_power_w(
            scheduler, allocation,
        )
        job = getattr(allocation, "job", None)
        num_nodes = len(getattr(allocation, "node_names", []) or [])
        if num_nodes <= 0 and job is not None:
            num_nodes = int(getattr(job, "num_nodes", 0) or 0)
        return dynamic_power + idle_power * num_nodes * self.pue

    def _running_allocation_dynamic_power_w(self, scheduler, allocation) -> float:
        """Return measured dynamic facility power for a running allocation."""
        rm = getattr(scheduler, "rm", None)
        cluster_type = getattr(scheduler, "cluster_type", "")
        cfg = self.node_configs.get(cluster_type, {})
        idle_power = float(cfg.get("idle_power_w", 0.0))
        total = 0.0
        for node_name in getattr(allocation, "node_names", []) or []:
            node = getattr(rm, "nodes", {}).get(node_name) if rm is not None else None
            if node is None or not hasattr(node, "measure"):
                continue
            total += max(0.0, float(node.measure()) - idle_power)

        if total > 0:
            return total * self.pue

        job = getattr(allocation, "job", None)
        if job is None:
            return 0.0

        freq_table = {
            float(f): float(p)
            for f, p in cfg.get("frequencies", {}).items()
        }
        frequency = float(
            getattr(job, "assigned_frequency", 0.0)
            or cfg.get("base_frequency_ghz", 0.0)
            or 0.0
        )
        dynamic_power = freq_table.get(
            frequency,
            float(cfg.get("max_power_w", 0.0)),
        )
        num_nodes = int(getattr(job, "num_nodes", 0) or 0)
        return dynamic_power * num_nodes * self.pue

    def _cluster_budget_weights(
        self,
        job_rects: list[JobRectangle],
        time: datetime,
    ) -> dict[str, float]:
        """Return demand-aware, capacity-capped cluster weights for the horizon.

        The envelope is a shared green supply carved into per-cluster slices.
        Weighting by raw hardware capacity over-funds clusters whose node count
        dwarfs their workload: their slice sits half-empty (green ``blank
        space``) while busier clusters spill jobs to grid.  Instead each cluster
        is weighted by the compute energy it actually demands over the horizon
        (pending job rectangles + energy still owed by running jobs), capped by
        the energy its nodes could physically absorb if saturated for the whole
        horizon.  When demand is below capacity — the common oversupply regime —
        the slice tracks demand, so an under-loaded cluster's green is freed for
        the clusters that can use it; when demand exceeds capacity the cap keeps
        a cluster from claiming more green than it can ever run.
        """
        schedulers_by_type = {
            scheduler.cluster_type: scheduler
            for scheduler in self.cluster_schedulers
            if not scheduler.done
        }
        cluster_types = set(schedulers_by_type)
        cluster_types.update(rect.cluster_type for rect in job_rects)

        weights: dict[str, float] = {}
        for cluster_type in cluster_types:
            scheduler = schedulers_by_type.get(cluster_type)
            if scheduler is None:
                weights[cluster_type] = sum(
                    rect.energy_wh for rect in job_rects
                    if rect.cluster_type == cluster_type
                )
                continue

            has_pending_work = any(
                job.status == "pending" and job.arrival_time <= time
                for job in scheduler.job_queue.jobs
            )
            has_running_work = self._scheduler_has_running_work(scheduler)
            has_work = has_pending_work or has_running_work
            if not has_work:
                weights[cluster_type] = 0.0
                continue

            capacity_energy_wh = (
                self._cluster_power_capacity_w(cluster_type, scheduler)
                * self.plan_horizon_h
            )
            demand_energy_wh = self._cluster_demand_energy_wh(
                cluster_type, job_rects, scheduler,
            )
            # A cluster with work but no measurable demand (e.g. its jobs were
            # filtered out of the rectangle set) still needs a slice: fall back
            # to its full capacity so it is not starved to zero.
            if demand_energy_wh <= 0:
                weights[cluster_type] = capacity_energy_wh
            else:
                weights[cluster_type] = min(demand_energy_wh, capacity_energy_wh)

        active_weights = {
            cluster_type: weight
            for cluster_type, weight in weights.items()
            if weight > 0
        }
        if active_weights:
            return active_weights

        # Fallback for synthetic tests without schedulers/node configs.
        demand_weights = {
            cluster_type: sum(
                rect.energy_wh for rect in job_rects
                if rect.cluster_type == cluster_type
            )
            for cluster_type in cluster_types
        }
        return {
            cluster_type: weight
            for cluster_type, weight in demand_weights.items()
            if weight > 0
        }

    def _cluster_power_capacity_w(self, cluster_type: str, scheduler) -> float:
        cfg = self.node_configs.get(cluster_type, {})
        idle_power = float(cfg.get("idle_power_w", 0.0))
        freq_table = {
            float(f): float(p)
            for f, p in cfg.get("frequencies", {}).items()
        }
        if freq_table:
            max_dynamic_power = max(freq_table.values())
        else:
            max_dynamic_power = float(cfg.get("max_power_w", 0.0))

        node_count = len(getattr(scheduler.rm, "nodes", {}))
        return max(
            0.0,
            (idle_power + max_dynamic_power) * node_count * self.pue,
        )

    def _cluster_demand_energy_wh(
        self,
        cluster_type: str,
        job_rects: list[JobRectangle],
        scheduler,
    ) -> float:
        """Compute energy a cluster actually demands over the horizon (Wh).

        Sums facility energy for pending rectangles and the total facility
        energy still owed by running jobs.  This is the "how much green can
        this cluster actually use" signal that ``_cluster_budget_weights``
        splits the envelope by.
        """
        pending_wh = sum(
            rect.energy_wh for rect in job_rects
            if rect.cluster_type == cluster_type
        )

        running_wh = 0.0
        rm = getattr(scheduler, "rm", None)
        if rm is not None and hasattr(rm, "get_running_allocations"):
            for allocation in rm.get_running_allocations():
                job = getattr(allocation, "job", None)
                if job is None or getattr(job, "status", None) != "running":
                    continue
                remaining_s = max(
                    0.0, float(getattr(job, "remaining_time", 0.0) or 0.0)
                )
                if remaining_s <= 0:
                    continue
                power_w = self._running_allocation_board_power_w(
                    scheduler, allocation,
                )
                running_wh += max(0.0, power_w) * remaining_s / 3600.0

        return pending_wh + running_wh

    @staticmethod
    def _scheduler_has_running_work(scheduler) -> bool:
        rm = getattr(scheduler, "rm", None)
        if rm is None or not hasattr(rm, "get_running_allocations"):
            return False
        return any(
            getattr(getattr(allocation, "job", None), "status", None) == "running"
            for allocation in rm.get_running_allocations()
        )

    # -- Bin-packing -------------------------------------------------------

    @staticmethod
    def _packing_order_key(rect: JobRectangle) -> tuple[float, float]:
        """Global/local greedy order: SLA urgency, then decreasing energy."""
        if rect.latest_start_slot_idx is None:
            slack = math.inf
        else:
            slack = float(
                rect.latest_start_slot_idx
                - max(0, rect.earliest_start_slot_idx)
            )
        return (slack, -rect.energy_wh)

    def _pack_jobs(
        self,
        slots: list[PowerSlot],
        job_rects: list[JobRectangle],
        repair: bool = False,
        node_capacity: Optional[int] = None,
        node_reserved: Optional[list[int]] = None,
        node_capacity_by_cluster: Optional[dict[str, Optional[int]]] = None,
        node_reserved_by_cluster: Optional[dict[str, list[int]]] = None,
        placement: str = "earliest",
        visible_slots: Optional[int] = None,
    ) -> tuple[list[JobRectangle], list[JobRectangle]]:
        """
        Pack jobs into green capacity plus carbon-permitted grid capacity.

        Algorithm: **SLA-slack-ordered greedy** on the admissible envelope

        1. Order jobs by SLA slack ascending (tightest start-deadline first) so
           deadline-critical jobs claim green capacity before flexible ones;
           ties break by decreasing area so big, hard-to-place pieces go early.
        2. For each job, scan starting slots within its SLA window
           ``[earliest_start, latest_start]``.  A job "fits" if its power is
           within the remaining green plus carbon-permitted grid capacity of
           every slot it would span, with enough nodes in its own cluster.
           Green supply is consumed first within the chosen slot; an earlier
           admissible mixed placement is not postponed for a later green one.
           Where the job lands depends
           on ``placement``:

           * ``"earliest"`` (default) — take the first feasible start.  Cheap
             (double early-break) and it dispatches work as soon as possible.
           * ``"best"`` — scan every feasible start in the window and take the
             one whose spanned slots have the least leftover admissible power
             after placement (tightest window), keeping roomier slots for later
             pieces.  Classic best-fit; costs a full window scan per job.

        3. If no normal placement fits the admissible envelope, carbon-aware
           mode evaluates the exact per-slot green deficit for the exceptional
           grid lanes. Carbon-regret mode may
           admit an earlier unbounded-grid candidate when waiting saves too
           little CO2 per hour; otherwise dirtier grid remains restricted to
           the job's safe latest start as an SLA emergency.
           Before deferring, carbon-regret mode compares the forecast grams of
           CO2 saved with the hours of extra wait.  It advances the earliest
           node-feasible candidate when the saving rate is below the configured
           threshold; the threshold grows with a job's age so repeated replans
           cannot postpone low-value savings forever.  When carbon-aware mode
           is disabled, the legacy grid-spill behaviour is retained for A/B
           comparisons.

        Complexity: the greedy pass is ~O(J × S × V) where J = jobs, S = slots,
        V = DVFS variants.  The fungible board stores aggregate free power per
        slot, so candidate starts jump past insufficient slots without scanning
        previously placed rectangles.  The optional ``repair`` pass adds an
        outer loop of spilled jobs × up-to-``max_candidates`` movers and keeps
        its own temporary obstacle index (see ``_repair_relocate``).

        ``visible_slots`` (default: the whole board) marks where the real
        forecast window ends; slots past it are extrapolated copies of the
        window (``_build_envelope``).  Jobs the real window can host are
        confined to it, exactly as before.  Only a job none of whose variants
        finishes inside the real window may run past ``visible_slots`` — and
        it must still *start* inside the real window, so the extrapolated
        zone only ever hosts the tail of an oversized job.
        """
        sorted_rects = sorted(job_rects, key=self._packing_order_key)

        placed: list[JobRectangle] = []
        unplaced: list[JobRectangle] = []

        slot_remaining = [s.remaining_w for s in slots]
        # Bottom-anchored capacity already occupied by running jobs.
        reserved_stack = [s.allocated_w for s in slots]
        ceilings = [s.green_available_w for s in slots]
        n_total = len(slots)
        # The current board model is fungible: only aggregate committed power
        # per slot is physical.  Keeping a vertical interval for every earlier
        # piece made a long single-pass replan quadratic on congested boards,
        # even though those intervals are display-only.  ``slot_remaining`` is
        # therefore the complete feasibility index for the greedy pass.
        # Physical node ledger: the rectangles spanning a slot (green AND grid
        # lane — both run on real nodes) cannot together use more nodes than
        # their cluster has.  The single-ledger arguments retain the local
        # packer API; global mode supplies one independent ledger per cluster.
        node_free_by_cluster: Optional[dict[str, Optional[list[int]]]] = None
        if node_capacity_by_cluster is not None:
            reservations = node_reserved_by_cluster or {}
            node_free_by_cluster = {}
            for cluster_type, capacity in node_capacity_by_cluster.items():
                if capacity is None:
                    node_free_by_cluster[cluster_type] = None
                    continue
                reserved_nodes = reservations.get(cluster_type, [0] * len(slots))
                node_free_by_cluster[cluster_type] = [
                    max(0, capacity - reserved_nodes[i])
                    for i in range(len(slots))
                ]

        if node_capacity is None:
            node_free: Optional[list[int]] = None
        else:
            reserved_nodes = node_reserved or [0] * len(slots)
            node_free = [
                max(0, node_capacity - reserved_nodes[i])
                for i in range(len(slots))
            ]

        n_visible = visible_slots if visible_slots is not None else n_total

        for rect in sorted_rects:
            rect_node_free = (
                node_free_by_cluster.get(rect.cluster_type)
                if node_free_by_cluster is not None
                else node_free
            )
            min_start_idx = max(0, rect.earliest_start_slot_idx)
            variants = rect.dvfs_variants or [
                (rect.frequency_ghz, rect.power_w, rect.duration_s)
            ]
            # A job too long for the real forecast window may run past it into
            # the extrapolated zone; everything else is confined to the window.
            extendable = self._window_extension_need(rect, n_visible) > 0
            # Prefer the lowest-energy (least green area) variant, then the one
            # spanning fewer slots.  Deforming to a higher frequency — taller,
            # shorter — is only chosen when the cheaper variants do not fit, so
            # the packer raises frequency to fit a narrow PV peak before spilling.
            ordered = sorted(variants, key=lambda v: (v[1] * v[2], v[2]))

            placed_green = False
            normal_grid_placement: Optional[_GridPlacement] = None
            admissible_remaining = [
                slot_remaining[i] + (
                    max(0.0, slot.grid_power_w - slot.grid_allocated_w)
                    if self.carbon_aware_placement else 0.0
                )
                for i, slot in enumerate(slots)
            ]
            for freq, power, duration in ordered:
                if power <= 0:
                    continue
                n_slots_needed = max(1, math.ceil(duration / self.slot_duration_s))
                if extendable:
                    # Must start inside the real window; only the tail may
                    # rest on extrapolated forecast.
                    last_feasible_start = min(
                        n_visible - 1, len(slots) - n_slots_needed,
                    )
                else:
                    last_feasible_start = n_visible - n_slots_needed
                if rect.latest_start_slot_idx is not None:
                    last_feasible_start = min(
                        last_feasible_start, rect.latest_start_slot_idx
                    )

                # next_bad[s] = first slot >= s this variant cannot use, because
                # its free capacity is smaller than the piece or it has too
                # few free nodes.  Any window covering such a slot is hopeless at
                # every stack base, so jumping past it skips whole runs of
                # candidate starts without probing them -- and preserves
                # earliest-fit order, since only infeasible starts are skipped.
                tolerance = 1e-6
                next_bad = [n_total] * (n_total + 1)
                nxt = n_total
                for s in range(n_total - 1, -1, -1):
                    if power > admissible_remaining[s] + tolerance or (
                        rect_node_free is not None
                        and rect.num_nodes > rect_node_free[s]
                    ):
                        nxt = s
                    next_bad[s] = nxt

                # Scan feasible starts within this variant's window.
                # "earliest": first feasible start (cheap, double early-break).
                # "best": tightest feasible window — least leftover green over
                # the spanned slots after placement.
                chosen: Optional[tuple[int, float]] = None
                best_leftover = math.inf
                start_idx = min_start_idx
                while start_idx <= last_feasible_start:
                    end_idx = start_idx + n_slots_needed
                    bad = next_bad[start_idx]
                    if bad < end_idx:
                        start_idx = bad + 1
                        continue
                    stack_base = max(
                        ceilings[s] - slot_remaining[s]
                        for s in range(start_idx, end_idx)
                    )
                    if placement != "best":
                        chosen = (start_idx, stack_base)
                        break
                    leftover = sum(
                        admissible_remaining[s] - power
                        for s in range(start_idx, end_idx)
                    )
                    if leftover < best_leftover - 1e-9:
                        best_leftover = leftover
                        chosen = (start_idx, stack_base)
                    start_idx += 1

                if chosen is None:
                    continue

                start_idx, stack_base = chosen
                if any(
                    power > slot_remaining[s] + tolerance
                    for s in range(start_idx, start_idx + n_slots_needed)
                ):
                    normal_grid_placement = _GridPlacement(
                        frequency_ghz=freq,
                        power_w=power,
                        duration_s=duration,
                        start_slot_idx=start_idx,
                        n_slots=n_slots_needed,
                        reason="low_carbon",
                        forecast_carbon_g=sum(
                            max(0.0, power - slot_remaining[s])
                            * slots[s].duration_s / 3_600_000.0
                            * max(0.0, slots[s].carbon_intensity)
                            for s in range(start_idx, start_idx + n_slots_needed)
                        ),
                    )
                    break
                self._apply_variant(rect, freq, power, duration, start_idx, n_slots_needed)
                rect.used_grid = False
                rect.green_allocation_w_by_slot.clear()
                rect.grid_allocation_w_by_slot.clear()
                rect.grid_reason = None
                rect.planned_grid_carbon_g = 0.0
                rect.stack_base_w = stack_base
                for s in range(start_idx, start_idx + n_slots_needed):
                    slot_remaining[s] -= power
                    if rect_node_free is not None:
                        rect_node_free[s] -= rect.num_nodes
                placed.append(rect)
                placed_green = True
                break

            if not placed_green:
                # Commit the normal mixed placement selected by the same
                # greedy scan as green jobs. Only if that scan found no normal
                # placement do the regret/emergency lanes become candidates.
                grid_placement: Optional[_GridPlacement]
                if normal_grid_placement is not None:
                    grid_placement = normal_grid_placement
                elif self.carbon_aware_placement:
                    grid_placement = self._choose_carbon_aware_grid_placement(
                        rect=rect,
                        variants=variants,
                        slots=slots,
                        slot_remaining=slot_remaining,
                        node_free=rect_node_free,
                        min_start_idx=min_start_idx,
                        visible_slots=n_visible,
                        extendable=extendable,
                    )
                else:
                    # Reproducibility switch for comparisons with the previous
                    # heuristic: lowest-energy variant, earliest node-feasible
                    # start, and an unbounded grid lane.
                    spill_total = len(slots) if extendable else n_visible
                    spill = self._choose_spill_variant(
                        rect, variants, spill_total, min_start_idx,
                    )
                    grid_start: Optional[int] = None
                    if spill is not None:
                        freq, power, duration, n_slots_needed = spill
                        grid_start = self._earliest_node_feasible_start(
                            rect, n_slots_needed, min_start_idx, spill_total,
                            rect_node_free,
                            max_start_idx=n_visible - 1,
                        )
                    grid_placement = (
                        _GridPlacement(
                            frequency_ghz=freq,
                            power_w=power,
                            duration_s=duration,
                            start_slot_idx=grid_start,
                            n_slots=n_slots_needed,
                            reason="legacy_grid",
                            forecast_carbon_g=0.0,
                        )
                        if spill is not None and grid_start is not None
                        else None
                    )

                if grid_placement is not None:
                    freq = grid_placement.frequency_ghz
                    power = grid_placement.power_w
                    duration = grid_placement.duration_s
                    grid_start = grid_placement.start_slot_idx
                    n_slots_needed = grid_placement.n_slots
                    self._apply_variant(
                        rect, freq, power, duration, grid_start, n_slots_needed,
                    )
                    rect.used_grid = True
                    rect.green_allocation_w_by_slot.clear()
                    rect.grid_allocation_w_by_slot.clear()
                    rect.grid_reason = grid_placement.reason
                    rect.planned_grid_carbon_g = 0.0
                    # Stack grid rectangles above the green ceiling for viz.
                    rect.stack_base_w = (
                        slots[grid_start].green_available_w
                        + slots[grid_start].grid_allocated_w
                    )
                    # Power is fungible per slot: a job that cannot fit green in
                    # *every* slot it spans (e.g. a long job crossing a low-green
                    # night) still draws the green each slot HAS, and only the
                    # per-slot deficit falls to grid.  Charging the whole
                    # rectangle to grid would waste the PV the job runs over
                    # during the day and inflate the plan's grid budget, letting
                    # more work run dirty.  The green it takes is folded into
                    # ``reserved_stack`` so later green placements — and the
                    # repair pass / allocated_w sync — see the reduced capacity.
                    for s in range(grid_start, grid_start + n_slots_needed):
                        green_take = max(0.0, min(power, slot_remaining[s]))
                        grid_take = power - green_take
                        if grid_take > 0.0:
                            slots[s].grid_allocated_w += grid_take
                            rect.grid_allocation_w_by_slot[s] = grid_take
                            rect.planned_grid_carbon_g += (
                                grid_take
                                * slots[s].duration_s
                                / 3_600_000.0
                                * max(0.0, slots[s].carbon_intensity)
                            )
                        if green_take > 0.0:
                            slot_remaining[s] -= green_take
                            reserved_stack[s] += green_take
                            rect.green_allocation_w_by_slot[s] = green_take
                        if rect_node_free is not None:
                            rect_node_free[s] -= rect.num_nodes
                    placed.append(rect)
                else:
                    unplaced.append(rect)

            # ---------------------------------------------------------------
            # Preemptive scheduling / job fragmentation (future):
            #
            # If a job does not fit contiguously, split it into fragments
            # that fit in non-adjacent slots.  This models a preemptible
            # checkpoint-and-resume execution model.
            #
            # if best_start is None and allow_preemption:
            #     fragments = self._fragment_job(rect, slots, slot_remaining)
            #     if fragments:
            #         for frag in fragments:
            #             placed.append(frag)
            #         continue
            #
            # def _fragment_job(self, rect, slots, slot_remaining):
            #     """Split a job into fragments that fit available gaps."""
            #     remaining_duration = rect.duration_s
            #     fragments = []
            #     for s_idx in range(len(slots)):
            #         if remaining_duration <= 0:
            #             break
            #         if slot_remaining[s_idx] >= rect.power_w:
            #             frag_duration = min(
            #                 remaining_duration,
            #                 slots[s_idx].duration_s,
            #             )
            #             fragments.append(JobRectangle(
            #                 job_id=rect.job_id,
            #                 power_w=rect.power_w,
            #                 duration_s=frag_duration,
            #                 num_nodes=rect.num_nodes,
            #                 frequency_ghz=rect.frequency_ghz,
            #                 energy_wh=rect.power_w * frag_duration / 3600.0,
            #                 cluster_type=rect.cluster_type,
            #                 start_slot_idx=s_idx,
            #                 end_slot_idx=s_idx,
            #             ))
            #             slot_remaining[s_idx] -= rect.power_w
            #             remaining_duration -= frag_duration
            #     if remaining_duration > 0:
            #         # Could not place all fragments; rollback.
            #         for frag in fragments:
            #             slot_remaining[frag.start_slot_idx] += frag.power_w
            #         return None
            #     return fragments
            # ---------------------------------------------------------------

        # Local-search repair: rearrange placed pieces to pull grid-spilled
        # jobs back onto the green board (opt-in).  Recomputes green occupancy
        # from the placed rectangles afterwards, so it does not rely on the
        # incremental ``slot_remaining`` bookkeeping above.
        if repair:
            if node_free_by_cluster is not None:
                raise ValueError(
                    "repair with per-cluster node ledgers must be run per cluster"
                )
            self._repair_relocate(
                slots,
                placed,
                reserved_stack,
                node_capacity=node_capacity,
                node_reserved=node_reserved,
                visible_slots=n_visible,
            )
            self._sync_allocated_from_placed(slots, placed, reserved_stack)
        else:
            # Sync slot allocated_w (green) from remaining green capacity.
            for i, slot in enumerate(slots):
                slot.allocated_w = slot.green_available_w - slot_remaining[i]

        return placed, unplaced

    def _sync_allocated_from_placed(
        self,
        slots: list[PowerSlot],
        placed: list[JobRectangle],
        reserved_stack: list[float],
    ) -> None:
        """Recompute each slot's green ``allocated_w`` from the placed rects.

        ``allocated_w`` = the bottom-anchored running reservation plus the
        power of every green (non-grid) job spanning the slot.  Used after the
        repair pass, which moves rectangles without touching ``slot_remaining``.
        """
        for i, slot in enumerate(slots):
            used = reserved_stack[i]
            for rect in placed:
                if rect.used_grid or rect.start_slot_idx is None:
                    continue
                if rect.start_slot_idx <= i <= rect.end_slot_idx:
                    used += rect.power_w
            slot.allocated_w = used

    @staticmethod
    def _obstacle_index(
        obstacle_rects: list[JobRectangle],
        n_slots: int,
    ) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
        """Per-slot stack intervals for a set of placed rectangles.

        Builds the same ``(slot_obstacles, slot_starts)`` pair the greedy packer
        maintains incrementally, so ``_find_stack_base_for_slot_obstacles`` can
        be reused on a hypothetical board during repair instead of re-scanning
        the full rectangle list once per candidate placement.

        * ``slot_obstacles[s]`` — every interval covering slot ``s``, sorted by
          lower edge (``_max_free_gap`` sweeps it in that order).
        * ``slot_starts[s]`` — the intervals of rectangles that *begin* at slot
          ``s``.  A rectangle overlapping a window ``[a, b)`` either covers
          ``a`` or starts inside ``(a, b)``, so the base search can assemble a
          window's obstacles as ``slot_obstacles[a] + starts in (a, b)`` and
          touch each rectangle exactly once, instead of once per slot it spans.
        """
        index: list[list[tuple[float, float]]] = [[] for _ in range(n_slots)]
        starts: list[list[tuple[float, float]]] = [[] for _ in range(n_slots)]
        for r in obstacle_rects:
            if r.start_slot_idx is None or r.end_slot_idx is None:
                continue
            interval = (r.stack_base_w, r.stack_base_w + r.power_w)
            last = min(r.end_slot_idx, n_slots - 1)
            if r.start_slot_idx < n_slots:
                starts[r.start_slot_idx].append(interval)
            for s in range(r.start_slot_idx, last + 1):
                index[s].append(interval)
        for intervals in index:
            intervals.sort()
        return index, starts

    @staticmethod
    def _slot_gaps(
        ceilings: list[float],
        slot_obstacles: list[list[tuple[float, float]]],
        reserved_stack: list[float],
    ) -> list[float]:
        """Free power capacity per slot (fungible model), for the skip lists."""
        return [
            _max_free_gap(slot_obstacles[s], reserved_stack[s], ceilings[s])
            for s in range(len(ceilings))
        ]

    @staticmethod
    def _gaps_patched(
        base_gaps: list[float],
        ceilings: list[float],
        slot_obstacles: list[list[tuple[float, float]]],
        reserved_stack: list[float],
        *spans: Optional[tuple[int, int]],
    ) -> list[float]:
        """Copy of ``base_gaps`` with only the given slot spans recomputed.

        Removing a mover (or adding the newcomer's ghost) can only change the
        free bands of the slots that rectangle spans; every other slot keeps the
        gap it had on the base board.  Recomputing all ``S`` slots per probe is
        what made repair quadratic in the spilled-job count on a scarce board.
        """
        gaps = list(base_gaps)
        last = len(ceilings) - 1
        for span in spans:
            if span is None:
                continue
            lo, hi = span
            for s in range(max(0, lo), min(hi, last) + 1):
                gaps[s] = _max_free_gap(
                    slot_obstacles[s], reserved_stack[s], ceilings[s],
                )
        return gaps

    def _try_place_green_variant(
        self,
        rect: JobRectangle,
        obstacle_rects: list[JobRectangle],
        slots: list[PowerSlot],
        reserved_stack: list[float],
        node_free: Optional[list[int]] = None,
        slot_obstacles: Optional[list[list[tuple[float, float]]]] = None,
        slot_starts: Optional[list[list[tuple[float, float]]]] = None,
        ceilings: Optional[list[float]] = None,
        gaps: Optional[list[float]] = None,
        visible_slots: Optional[int] = None,
    ) -> Optional[tuple[float, float, float, int, int, float]]:
        """Earliest-fit green placement for ``rect`` avoiding ``obstacle_rects``.

        Mirrors the greedy packer's per-job search (DVFS variants ordered by
        energy, earliest feasible start, obstacle-aware stack base) but reads
        occupancy from the given rectangle list instead of the incremental
        index, so it can be called during repair on a hypothetical board.
        ``slot_obstacles`` lets the caller pass a prebuilt (or incrementally
        patched) per-slot index; when given, ``obstacle_rects`` is not read
        at all — the index alone defines the board.  ``node_free`` (free
        node count per slot, excluding ``rect`` itself) enforces the physical
        node ledger; ``None`` disables the check.  ``visible_slots`` applies
        the same real-window confinement as ``_pack_jobs``.  Returns
        ``(freq, power, duration, start_idx, n_slots, stack_base)`` or
        ``None``.

        Candidate starts are pruned with a skip list before the stack search:
        a window containing a slot with less free power capacity than the
        variant draws (power is fungible — see ``_max_free_gap``), or with
        fewer free nodes than the job needs, cannot host it.  Jumping past
        the blocking slot preserves the earliest-fit order while probing only
        windows that can possibly fit.
        """
        n = len(slots)
        if slot_obstacles is None or slot_starts is None:
            slot_obstacles, slot_starts = self._obstacle_index(obstacle_rects, n)
        if ceilings is None:
            ceilings = [s.green_available_w for s in slots]
        tolerance = 1e-6
        if gaps is None:
            gaps = self._slot_gaps(ceilings, slot_obstacles, reserved_stack)
        variants = rect.dvfs_variants or [
            (rect.frequency_ghz, rect.power_w, rect.duration_s)
        ]
        ordered = sorted(variants, key=lambda v: (v[1] * v[2], v[2]))
        min_start = max(0, rect.earliest_start_slot_idx)
        n_visible = visible_slots if visible_slots is not None else n
        extendable = self._window_extension_need(rect, n_visible) > 0
        for freq, power, duration in ordered:
            if power <= 0:
                continue
            n_slots = max(1, math.ceil(duration / self.slot_duration_s))
            if extendable:
                last_start = min(n_visible - 1, n - n_slots)
            else:
                last_start = n_visible - n_slots
            if rect.latest_start_slot_idx is not None:
                last_start = min(last_start, rect.latest_start_slot_idx)
            # next_bad[s] = first slot >= s this variant cannot use.
            next_bad = [n] * (n + 1)
            nxt = n
            for s in range(n - 1, -1, -1):
                if power > gaps[s] + tolerance or (
                    node_free is not None and rect.num_nodes > node_free[s]
                ):
                    nxt = s
                next_bad[s] = nxt
            start = min_start
            while start <= last_start:
                bad = next_bad[start]
                if bad < start + n_slots:
                    start = bad + 1
                    continue
                base = self._find_stack_base_for_slot_obstacles(
                    ceilings, slot_obstacles, slot_starts, power,
                    start, start + n_slots, reserved_stack,
                )
                if base is not None:
                    return (freq, power, duration, start, n_slots, base)
                start += 1
        return None

    def _repair_relocate(
        self,
        slots: list[PowerSlot],
        placed: list[JobRectangle],
        reserved_stack: list[float],
        max_candidates: int = 48,
        node_capacity: Optional[int] = None,
        node_reserved: Optional[list[int]] = None,
        visible_slots: Optional[int] = None,
        repair_cluster_type: Optional[str] = None,
    ) -> int:
        """Pull grid-spilled jobs onto the green board, relocating if needed.

        Two moves, each of which *strictly* lowers grid spill (so the carbon
        objective monotonically improves and no rollback is needed):

        1. **Direct re-placement (0-opt):** once an earlier repair has shifted a
           mover, the freed room may let a still-spilled job fit green with no
           further move.  Retry each spilled job on green as-is (a no-op on the
           first spill, which the greedy already proved cannot fit unaided).
        2. **Relocation (1-opt):** if the newcomer only fits where a green job
           already sits, try shifting *one* green job — within its own SLA slack
           (unbounded for jobs with no start deadline) — to free the window,
           then place the newcomer.  Both stay green, so grid drops by the
           newcomer's energy.

        Both moves keep every job inside its SLA window, so deadlines are not
        put at risk.  Returns the number of jobs pulled back to green.
        """
        recovered = 0
        ceilings = [s.green_available_w for s in slots]

        def _node_free_all() -> Optional[list[int]]:
            """Free nodes per slot after subtracting *every* placed rect.

            Green and grid rects both hold physical nodes over their span, so
            the ledger counts them all.  ``_node_free_add_back`` then re-credits
            the nodes of the specific rects a probe excludes — cheap span-local
            edits instead of re-scanning ``placed`` once per probe.
            """
            if node_capacity is None:
                return None
            reserved_nodes = node_reserved or [0] * len(slots)
            free = [
                node_capacity - reserved_nodes[i] for i in range(len(slots))
            ]
            for r in placed:
                if (
                    repair_cluster_type is not None
                    and r.cluster_type != repair_cluster_type
                ):
                    continue
                if r.start_slot_idx is None:
                    continue
                last = min(r.end_slot_idx, len(slots) - 1)
                for s in range(r.start_slot_idx, last + 1):
                    free[s] -= r.num_nodes
            return free

        def _node_free_add_back(
            base_free: Optional[list[int]],
            *excluded: JobRectangle,
        ) -> Optional[list[int]]:
            """Re-credit the nodes of ``excluded`` rects to a full-ledger copy.

            ``base_free`` counts every placed rect; adding a rect's nodes back
            over its span yields the ledger that rect is excluded from — the
            free-node count a probe sees when it may reuse that rect's nodes.
            """
            if base_free is None:
                return None
            free = list(base_free)
            for r in excluded:
                if r.start_slot_idx is None:
                    continue
                last = min(r.end_slot_idx, len(slots) - 1)
                for s in range(r.start_slot_idx, last + 1):
                    free[s] += r.num_nodes
            return free

        def _index_patched(
            index: list[list[tuple[float, float]]],
            starts: list[list[tuple[float, float]]],
            remove: Optional[JobRectangle] = None,
            add: Optional[JobRectangle] = None,
        ) -> tuple[
            list[list[tuple[float, float]]], list[list[tuple[float, float]]]
        ]:
            """Copy-on-write edit of the per-slot obstacle and start indexes.

            Only the slots in the edited rects' spans get fresh lists, so a
            mover probe costs O(span) instead of rebuilding the whole index
            from the green list once per candidate — the difference between
            minutes and milliseconds per replan on saturated boards.  Interval
            content matches a full rebuild, and each edited slot of ``index``
            stays sorted by lower edge, which ``_max_free_gap`` relies on.
            """
            out = list(index)
            out_starts = list(starts)
            n = len(out)
            for rect, adding in ((remove, False), (add, True)):
                if rect is None or rect.start_slot_idx is None:
                    continue
                interval = (rect.stack_base_w, rect.stack_base_w + rect.power_w)
                last = min(rect.end_slot_idx, n - 1)
                for s in range(rect.start_slot_idx, last + 1):
                    edited = list(out[s])
                    if adding:
                        insort(edited, interval)
                    else:
                        try:
                            edited.remove(interval)
                        except ValueError:
                            pass
                    out[s] = edited
                if rect.start_slot_idx < n:
                    s = rect.start_slot_idx
                    edited = list(out_starts[s])
                    if adding:
                        edited.append(interval)
                    else:
                        try:
                            edited.remove(interval)
                        except ValueError:
                            pass
                    out_starts[s] = edited
            return out, out_starts

        # Biggest carbon first: recovering a large spilled rect saves most grid.
        spilled = sorted(
            (
                r for r in placed
                if r.used_grid
                and (
                    repair_cluster_type is None
                    or r.cluster_type == repair_cluster_type
                )
            ),
            key=lambda r: -r.energy_wh,
        )
        # The green list, node ledger, obstacle index and free-band gaps are a
        # pure function of ``placed``, which only changes when a move commits.
        # Rebuilding them per spilled job re-derived an identical board for
        # every rejected candidate; rebuild them only after a commit dirties it.
        green: list[JobRectangle] = []
        base_free: Optional[list[int]] = None
        green_index: list[list[tuple[float, float]]] = []
        green_starts: list[list[tuple[float, float]]] = []
        board_dirty = True
        for spill in spilled:
            if board_dirty:
                green = [r for r in placed if not r.used_grid]
                base_free = _node_free_all()
                green_index, green_starts = self._obstacle_index(green, len(slots))
                board_dirty = False

            # The spill already consumes some green in ``reserved_stack``.
            # Exclude its own exact footprint while asking whether it can be
            # recovered onto the green board; otherwise partial-green spills
            # block themselves from a feasible repair.
            spill_reserved = list(reserved_stack)
            for s, green_w in spill.green_allocation_w_by_slot.items():
                if 0 <= s < len(spill_reserved):
                    spill_reserved[s] = max(0.0, spill_reserved[s] - green_w)
            spill_gaps = self._slot_gaps(
                ceilings, green_index, spill_reserved,
            )

            # Move 1: direct re-placement.
            place = self._try_place_green_variant(
                spill, green, slots, spill_reserved,
                node_free=_node_free_add_back(base_free, spill),
                slot_obstacles=green_index,
                slot_starts=green_starts,
                ceilings=ceilings,
                gaps=spill_gaps,
                visible_slots=visible_slots,
            )
            if place is not None:
                self._commit_green_placement(
                    spill, place, slots, was_grid=True,
                    reserved_stack=reserved_stack,
                )
                recovered += 1
                board_dirty = True
                continue

            # Move 2: relocate one overlapping green job to make room.
            win_lo = max(0, spill.earliest_start_slot_idx)
            win_hi = (
                spill.latest_start_slot_idx
                if spill.latest_start_slot_idx is not None
                else len(slots) - 1
            )
            # A mover without a start deadline (``latest_start_slot_idx`` is
            # ``None``) has unbounded slack — the most relocatable piece on
            # the board — so ``None`` means "may shift anywhere in the
            # horizon", not "fixed in place".
            candidates = [
                m for m in green
                if m.start_slot_idx is not None
                and m.start_slot_idx <= win_hi
                and m.end_slot_idx >= win_lo
                and (
                    repair_cluster_type is None
                    or m.cluster_type == repair_cluster_type
                )
            ]
            # Try the most flexible movers first: they have the most room to
            # be re-placed elsewhere, and tight-deadline pieces are disturbed
            # only as a last resort.
            candidates.sort(
                key=lambda m: (
                    math.inf if m.latest_start_slot_idx is None
                    else m.latest_start_slot_idx
                    - max(0, m.earliest_start_slot_idx)
                ),
                reverse=True,
            )
            for mover in candidates[:max_candidates]:
                # The patched index alone defines the board for the probes
                # (obstacle_rects is not read when an index is passed), so no
                # per-mover copies of the green list are built.
                index_wo_mover, starts_wo_mover = _index_patched(
                    green_index, green_starts, remove=mover,
                )
                node_free_wo_pair = _node_free_add_back(base_free, spill, mover)
                # Lifting the mover only widens the free bands of the slots it
                # spanned, so patch those and reuse the base board's gaps.
                gaps_wo_mover = self._gaps_patched(
                    spill_gaps, ceilings, index_wo_mover, spill_reserved,
                    (mover.start_slot_idx, mover.end_slot_idx),
                )
                place_spill = self._try_place_green_variant(
                    spill, green, slots, spill_reserved,
                    node_free=node_free_wo_pair,
                    slot_obstacles=index_wo_mover,
                    slot_starts=starts_wo_mover,
                    ceilings=ceilings,
                    gaps=gaps_wo_mover,
                    visible_slots=visible_slots,
                )
                if place_spill is None:
                    continue
                # Represent the newcomer's tentative footprint as an obstacle,
                # then check the mover can be re-placed green around it.
                ghost = JobRectangle(
                    job_id=spill.job_id,
                    power_w=place_spill[1],
                    duration_s=place_spill[2],
                    num_nodes=spill.num_nodes,
                    frequency_ghz=place_spill[0],
                    energy_wh=place_spill[1] * place_spill[2] / 3600.0,
                    cluster_type=spill.cluster_type,
                    start_slot_idx=place_spill[3],
                    end_slot_idx=place_spill[3] + place_spill[4] - 1,
                    stack_base_w=place_spill[5],
                )
                # The mover's node budget must also account for the ghost's
                # tentative footprint.
                node_free_mover = node_free_wo_pair
                if node_free_mover is not None:
                    node_free_mover = list(node_free_mover)
                    ghost_last = min(ghost.end_slot_idx, len(slots) - 1)
                    for s in range(ghost.start_slot_idx, ghost_last + 1):
                        node_free_mover[s] -= ghost.num_nodes
                index_with_ghost, starts_with_ghost = _index_patched(
                    index_wo_mover, starts_wo_mover, add=ghost,
                )
                place_mover = self._try_place_green_variant(
                    mover, green, slots, spill_reserved,
                    node_free=node_free_mover,
                    slot_obstacles=index_with_ghost,
                    slot_starts=starts_with_ghost,
                    ceilings=ceilings,
                    gaps=self._gaps_patched(
                        gaps_wo_mover, ceilings, index_with_ghost,
                        spill_reserved,
                        (ghost.start_slot_idx, ghost.end_slot_idx),
                    ),
                    visible_slots=visible_slots,
                )
                if place_mover is None:
                    continue
                # Both fit: commit the newcomer (grid->green) and the shift.
                self._commit_green_placement(
                    spill, place_spill, slots, was_grid=True,
                    reserved_stack=reserved_stack,
                )
                self._commit_green_placement(
                    mover, place_mover, slots, was_grid=False,
                )
                recovered += 1
                board_dirty = True
                break

        return recovered

    def _commit_green_placement(
        self,
        rect: JobRectangle,
        placement: tuple[float, float, float, int, int, float],
        slots: list[PowerSlot],
        was_grid: bool,
        reserved_stack: Optional[list[float]] = None,
    ) -> None:
        """Apply a repair placement, clearing the grid footprint if needed."""
        freq, power, duration, start, n_slots, base = placement
        if was_grid:
            has_exact_split = bool(
                rect.green_allocation_w_by_slot
                or rect.grid_allocation_w_by_slot
            )
            for s in range(rect.start_slot_idx, rect.end_slot_idx + 1):
                grid_w = (
                    rect.grid_allocation_w_by_slot.get(s, 0.0)
                    if has_exact_split else rect.power_w
                )
                slots[s].grid_allocated_w = max(
                    0.0, slots[s].grid_allocated_w - grid_w
                )
                if reserved_stack is not None:
                    green_w = rect.green_allocation_w_by_slot.get(s, 0.0)
                    reserved_stack[s] = max(0.0, reserved_stack[s] - green_w)
            rect.green_allocation_w_by_slot.clear()
            rect.grid_allocation_w_by_slot.clear()
            rect.used_grid = False
            rect.grid_reason = None
            rect.planned_grid_carbon_g = 0.0
        self._apply_variant(rect, freq, power, duration, start, n_slots)
        rect.stack_base_w = base

    @staticmethod
    def _apply_variant(
        rect: JobRectangle,
        freq: float,
        power: float,
        duration: float,
        start_idx: int,
        n_slots_needed: int,
    ) -> None:
        """Stamp the chosen DVFS variant and plan position onto the rectangle."""
        rect.frequency_ghz = freq
        rect.power_w = power
        rect.duration_s = duration
        rect.energy_wh = power * duration / 3600.0
        rect.start_slot_idx = start_idx
        rect.end_slot_idx = start_idx + n_slots_needed - 1  # Inclusive.

    @staticmethod
    def _earliest_node_feasible_start(
        rect: JobRectangle,
        n_slots_needed: int,
        min_start_idx: int,
        n_total_slots: int,
        node_free: Optional[list[int]],
        max_start_idx: Optional[int] = None,
    ) -> Optional[int]:
        """Earliest start whose spanned slots all have enough free nodes.

        Used for grid-lane spills: without a node ledger this degenerates to
        the previous behaviour (start at ``min_start_idx``).  Returns ``None``
        when no window in the horizon has capacity.  ``max_start_idx`` caps
        the start (e.g. to the real forecast window on an extrapolated board).
        """
        last_start = n_total_slots - n_slots_needed
        if max_start_idx is not None:
            last_start = min(last_start, max_start_idx)
        if min_start_idx > last_start:
            return None
        if node_free is None:
            return min_start_idx
        for start in range(min_start_idx, last_start + 1):
            if all(
                rect.num_nodes <= node_free[s]
                for s in range(start, start + n_slots_needed)
            ):
                return start
        return None

    def _choose_carbon_aware_grid_placement(
        self,
        rect: JobRectangle,
        variants: list[tuple[float, float, float]],
        slots: list[PowerSlot],
        slot_remaining: list[float],
        node_free: Optional[list[int]],
        min_start_idx: int,
        visible_slots: int,
        extendable: bool,
    ) -> Optional[_GridPlacement]:
        """Choose the lowest-CO2 partial-green placement allowed by policy.

        For each DVFS shape this builds prefix sums of its exact per-slot grid
        deficit, forecast emissions, and infeasibility flags.  Candidate
        windows are therefore scored in O(1), avoiding a duration factor for
        long jobs and keeping a replan near O(variants * slots).

        The normal lane is capped by each slot's remaining carbon-filtered grid
        allowance.  Carbon-regret mode separately compares the earliest
        node-feasible candidate with the cleanest candidate and may advance it
        when the forecast saving per waiting hour is too small.  Otherwise
        high-carbon/unbounded grid is considered only when the job's latest
        safe start lies inside the real forecast window.  On-time emergency
        candidates beat late ones; a late emergency is kept only as the
        physical-node fallback used by the old packer when the SLA window
        itself has no node-feasible start.
        """
        n_total = len(slots)
        if n_total == 0 or visible_slots <= 0:
            return None

        latest = rect.latest_start_slot_idx
        emergency_allowed = latest is not None and latest < visible_slots
        best_low: Optional[tuple[tuple[float, float, float, int, int], _GridPlacement]] = None
        best_emergency: Optional[
            tuple[tuple[float, float, float, int, int], _GridPlacement]
        ] = None
        best_late_emergency: Optional[
            tuple[tuple[float, float, float, int, int], _GridPlacement]
        ] = None
        earliest_any: Optional[
            tuple[tuple[int, float, float, float, int], _GridPlacement]
        ] = None
        best_any: Optional[
            tuple[tuple[float, float, float, int, int], _GridPlacement]
        ] = None
        tolerance = 1e-6

        # A short job cannot use the extrapolated board. Node feasibility and
        # available green/grid power are shared by all of its DVFS variants.
        horizon_slots = n_total if extendable else min(n_total, visible_slots)
        green = [max(0.0, slot_remaining[s]) for s in range(horizon_slots)]
        allowance = [
            max(0.0, slots[s].grid_power_w - slots[s].grid_allocated_w) + tolerance
            for s in range(horizon_slots)
        ]
        node_bad = [
            node_free is not None and rect.num_nodes > node_free[s]
            for s in range(horizon_slots)
        ]
        node_bad_prefix = list(accumulate(node_bad, initial=0))

        for freq, power, duration in variants:
            if power <= 0:
                continue
            n_slots_needed = max(1, math.ceil(duration / self.slot_duration_s))
            last_start = horizon_slots - n_slots_needed
            if extendable:
                last_start = min(last_start, visible_slots - 1)
            if min_start_idx > last_start:
                continue

            reachable_slots = last_start + n_slots_needed
            grid_take_w = [
                max(0.0, power - green[s]) for s in range(reachable_slots)
            ]
            grid_wh = [
                grid_take_w[s] * slots[s].duration_s / 3600.0
                for s in range(reachable_slots)
            ]
            carbon_g = [
                grid_take_w[s]
                * slots[s].duration_s
                / 3_600_000.0
                * max(0.0, slots[s].carbon_intensity)
                * (
                    self.forecast_tail_carbon_multiplier
                    if s >= visible_slots else 1.0
                )
                for s in range(reachable_slots)
            ]
            low_bad = [
                node_bad[s]
                or grid_take_w[s] > allowance[s]
                for s in range(reachable_slots)
            ]

            grid_prefix = list(accumulate(grid_wh, initial=0.0))
            carbon_prefix = list(accumulate(carbon_g, initial=0.0))
            low_bad_prefix = list(accumulate(low_bad, initial=0))
            energy_wh = power * duration / 3600.0
            on_time_last = last_start
            if latest is not None:
                on_time_last = min(on_time_last, latest)

            def _placement(
                start: int, reason: str, candidate_carbon: float,
            ) -> _GridPlacement:
                return _GridPlacement(
                    frequency_ghz=freq,
                    power_w=power,
                    duration_s=duration,
                    start_slot_idx=start,
                    n_slots=n_slots_needed,
                    reason=reason,
                    forecast_carbon_g=candidate_carbon,
                )

            def _candidate(start: int, reason: str) -> tuple[
                tuple[float, float, float, int, int], _GridPlacement
            ]:
                end = start + n_slots_needed
                candidate_carbon = carbon_prefix[end] - carbon_prefix[start]
                candidate_grid_wh = grid_prefix[end] - grid_prefix[start]
                score = (
                    candidate_carbon,
                    candidate_grid_wh,
                    energy_wh,
                    start,
                    n_slots_needed,
                )
                return score, _placement(start, reason, candidate_carbon)

            for start in range(min_start_idx, on_time_last + 1):
                end = start + n_slots_needed
                node_feasible = node_bad_prefix[end] == node_bad_prefix[start]
                low_feasible = low_bad_prefix[end] == low_bad_prefix[start]
                if not node_feasible and not low_feasible:
                    continue

                # All placement classes use the same per-window carbon/grid
                # score. Compute it once; construct a placement only when this
                # window improves one of the best candidates.
                candidate_carbon = carbon_prefix[end] - carbon_prefix[start]
                candidate_grid_wh = grid_prefix[end] - grid_prefix[start]
                score = (
                    candidate_carbon,
                    candidate_grid_wh,
                    energy_wh,
                    start,
                    n_slots_needed,
                )

                if node_feasible:
                    earliest_key = (
                        start,
                        candidate_carbon,
                        candidate_grid_wh,
                        energy_wh,
                        n_slots_needed,
                    )
                    if earliest_any is None or earliest_key < earliest_any[0]:
                        earliest_any = (
                            earliest_key,
                            _placement(start, "carbon_regret", candidate_carbon),
                        )
                    if best_any is None or score < best_any[0]:
                        best_any = (
                            score,
                            _placement(start, "carbon_regret", candidate_carbon),
                        )
                if low_feasible and (best_low is None or score < best_low[0]):
                    best_low = (
                        score,
                        _placement(start, "low_carbon", candidate_carbon),
                    )
                if emergency_allowed and node_feasible and (
                    best_emergency is None or score < best_emergency[0]
                ):
                    best_emergency = (
                        score,
                        _placement(start, "sla_emergency", candidate_carbon),
                    )

            # Preserve progress when the SLA window itself is physically
            # impossible: schedule at the cleanest later node-feasible start,
            # but classify it as an SLA emergency rather than ordinary grid.
            if emergency_allowed and on_time_last < last_start:
                late_start = max(min_start_idx, on_time_last + 1)
                for start in range(late_start, last_start + 1):
                    end = start + n_slots_needed
                    if node_bad_prefix[end] != node_bad_prefix[start]:
                        continue
                    candidate = _candidate(start, "sla_emergency")
                    if (
                        best_late_emergency is None
                        or candidate[0] < best_late_emergency[0]
                    ):
                        best_late_emergency = candidate

        regret_threshold = self.carbon_regret_threshold_g_per_h
        if (
            regret_threshold is not None
            and earliest_any is not None
            and best_any is not None
        ):
            earliest = earliest_any[1]
            cleanest = best_any[1]
            extra_wait_h = max(
                0.0,
                (cleanest.start_slot_idx - earliest.start_slot_idx)
                * self.slot_duration_s
                / 3600.0,
            )
            carbon_saved_g = max(
                0.0,
                earliest.forecast_carbon_g - cleanest.forecast_carbon_g,
            )
            # Aging raises the price of another deferral.  One full planning
            # horizon of accumulated wait doubles the threshold, ensuring that
            # small forecast savings do not win indefinitely at every replan.
            waited_h = 0.0
            if rect.arrival_time is not None and slots:
                waited_h = max(
                    0.0,
                    (slots[0].start_time - rect.arrival_time).total_seconds()
                    / 3600.0,
                )
            aging_factor = 1.0 + waited_h / max(self.plan_horizon_h, 1e-9)
            effective_threshold = regret_threshold * aging_factor
            saving_rate = (
                carbon_saved_g / extra_wait_h
                if extra_wait_h > 0.0 else 0.0
            )
            earliest_is_normal_low_carbon_choice = (
                best_low is not None
                and earliest.start_slot_idx == best_low[1].start_slot_idx
                and earliest.frequency_ghz == best_low[1].frequency_ghz
                and math.isclose(
                    earliest.forecast_carbon_g,
                    best_low[1].forecast_carbon_g,
                    abs_tol=1e-9,
                )
            )
            if (
                not earliest_is_normal_low_carbon_choice
                and saving_rate <= effective_threshold
            ):
                return earliest

        if best_low is not None:
            return best_low[1]
        if best_emergency is not None:
            return best_emergency[1]
        if best_late_emergency is not None:
            return best_late_emergency[1]
        return None

    def _choose_spill_variant(
        self,
        rect: JobRectangle,
        variants: list[tuple[float, float, float]],
        n_total_slots: int,
        min_start_idx: int,
    ) -> Optional[tuple[float, float, float, int]]:
        """Pick the lowest-energy variant that fits the horizon for grid spill.

        Returns ``(frequency, power_w, duration_s, n_slots)`` or ``None`` when
        no variant fits the remaining horizon time-wise from the job's earliest
        slot (a job that genuinely cannot run in this plan window).
        """
        best: Optional[tuple] = None
        for freq, power, duration in variants:
            if power <= 0:
                continue
            n_slots_needed = max(1, math.ceil(duration / self.slot_duration_s))
            if min_start_idx + n_slots_needed > n_total_slots:
                continue
            key = (power * duration / 3600.0, n_slots_needed)
            if best is None or key < best[0]:
                best = (key, freq, power, duration, n_slots_needed)
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]

    @staticmethod
    def _find_stack_base_for_slot_obstacles(
        ceilings: list[float],
        slot_obstacles: list[list[tuple[float, float]]],
        slot_starts: list[list[tuple[float, float]]],
        power_w: float,
        start_idx: int,
        end_idx: int,
        reserved_stack: list[float],
    ) -> Optional[float]:
        """Per-slot capacity check over the window; returns a display base.

        Power is fungible (see ``_max_free_gap``): the piece fits iff every
        spanned slot has ``reserved + committed + power_w <= ceiling``.  There
        is no physical y-position — the returned base is the high-water mark
        of committed power across the span, used only to stamp
        ``stack_base_w`` for plots, and may poke above the ceiling of an
        individual slot in the drawing.  ``slot_starts`` is kept in the
        signature for call-site compatibility (the rigid-band predecessor
        needed it to enumerate window obstacles once per rectangle).
        """
        if start_idx >= end_idx:
            return None

        tolerance = 1e-6
        base = 0.0
        for s in range(start_idx, end_idx):
            used = reserved_stack[s]
            for lo, hi in slot_obstacles[s]:
                used += hi - lo
            if used + power_w > ceilings[s] + tolerance:
                return None
            if used > base:
                base = used
        return base

    # -- Instantaneous budget lookup ---------------------------------------

    def _budget_at(
        self,
        time: datetime,
        renewable_signal: "HistoricalSignal",
        battery: "Storage",
    ) -> BudgetInfo:
        """Look up the current slot in the plan and return BudgetInfo."""
        plan = self._current_plan
        if plan is None:
            return self._fallback_budget(time, battery)

        # Binary-style lookup: slots are contiguous and sorted.
        slot = self._find_slot(time)
        if slot is None:
            return self._fallback_budget(time, battery)

        # Track green-board utilization for the replan trigger.
        if slot.green_available_w > 0:
            util = slot.allocated_w / slot.green_available_w
            self._utilization_tracker.append(util)
            if len(self._utilization_tracker) > 200:
                self._utilization_tracker = self._utilization_tracker[-100:]

        # Storage dispatch decisions.  Cap the static plan by the *current*
        # deliverable SoC as well as the device power limit.  If an earlier
        # dispatch used more energy than forecast, the missing battery power is
        # removed from compute budget instead of surviving as a fictitious
        # source; an unavoidable running load is then exposed explicitly as
        # grid by the live floor below.
        deliverable_wh = deliverable_battery_energy_wh(battery)
        duration_s = max(1e-12, float(slot.duration_s))
        battery_power_w = min(
            max(0.0, slot.battery_power_w),
            max(0.0, float(getattr(battery, "_p_discharge_max", float("inf")))),
            deliverable_wh * 3600.0 / duration_s,
        )
        allow_discharge = battery_power_w > 0
        if allow_discharge:
            charge_power: Optional[float] = -battery_power_w
        elif slot.charge_reserve_w > 0:
            charge_power = slot.charge_reserve_w
        else:
            charge_power = None

        budget_renewable = max(
            0.0, slot.renewable_power_w - slot.charge_reserve_w,
        )

        low_carbon = (
            self.carbon_intensity_threshold_g_per_kwh > 0
            and slot.carbon_intensity <= self.carbon_intensity_threshold_g_per_kwh
        )

        # Budget exposed = the planned green ceiling plus planned grid spill.
        # A static plan can nevertheless become stale between replans (a job
        # starts late, or all running jobs reach minimum DVFS before a falling
        # slot budget).  Already-running power cannot be shed.  Make that live
        # facility draw an explicit grid floor instead of reporting physical
        # consumption for which no budget source exists.
        physically_backed_green_w = budget_renewable + battery_power_w
        planned_runtime_grid_w = slot.grid_allocated_w
        planned_budget_w = physically_backed_green_w + planned_runtime_grid_w
        # Each partition needs its planned work AND the idle nodes outside
        # that work. max(total planned, total live) loses dispatch headroom
        # when idle draw exceeds the job footprint, causing endless replans.
        cluster_budgets = self.get_cluster_compute_budgets(time)
        # The stored shares include forecast battery power. Reduce those
        # shares when SOC no longer backs it; idle reservations must not turn
        # missing battery into permission to start new work on grid.
        missing_green_fraction = (
            max(0.0, 1.0 - physically_backed_green_w / slot.green_available_w)
            if slot.green_available_w > 0.0 else 0.0
        )
        if missing_green_fraction > 0.0:
            idx = self.current_slot_idx(time) or 0
            for cluster_type, per_cluster in plan.cluster_slots.items():
                if not per_cluster:
                    continue
                cluster_slot = per_cluster[min(idx, len(per_cluster) - 1)]
                cluster_budgets[cluster_type] = max(
                    0.0, cluster_budgets[cluster_type]
                    - cluster_slot.green_available_w * missing_green_fraction,
                )
        required_w = sum(cluster_budgets.values())
        for scheduler in self.cluster_schedulers:
            get_power = getattr(scheduler, "get_current_power", None)
            if callable(get_power):
                live_w = max(0.0, float(get_power())) * self.pue
                planned_w = cluster_budgets.get(
                    getattr(scheduler, "cluster_type", None), 0.0,
                )
                required_w += max(0.0, live_w - planned_w)
        unavoidable_grid_w = max(0.0, required_w - planned_budget_w)
        runtime_grid_w = planned_runtime_grid_w + unavoidable_grid_w
        total_budget_w = planned_budget_w + unavoidable_grid_w
        return BudgetInfo(
            time=str(time),
            renewable_power_w=slot.renewable_power_w,
            budget_renewable_w=budget_renewable,
            charge_reserve_w=slot.charge_reserve_w,
            battery_soc=battery.soc(),
            battery_discharge_power_w=battery_power_w,
            grid_power_w=runtime_grid_w,
            total_budget_w=total_budget_w,
            allow_discharge=allow_discharge,
            charge_power=charge_power,
            carbon_intensity_g_per_kwh=slot.carbon_intensity,
            low_carbon_grid_allowed=low_carbon,
        )

    def _record_always_on_idle_power(
        self,
        slots: list[PowerSlot],
        placed: list[JobRectangle],
        cluster_slots: dict[str, list[PowerSlot]],
    ) -> None:
        """Record background idle draw without double-counting job rectangles.

        Fully static clusters reserved their whole idle floor before packing;
        their rectangles contain incremental power. Other modes retain total
        job rectangles and record only powered nodes outside those rectangles.
        """
        for slot in slots:
            slot.idle_reserved_w = 0.0
            slot.idle_allocated_w = 0.0
        for per_cluster in cluster_slots.values():
            for slot in per_cluster:
                slot.idle_reserved_w = 0.0
                slot.idle_allocated_w = 0.0
        placed_by_cluster: dict[str, list[JobRectangle]] = {}
        for rect in placed:
            placed_by_cluster.setdefault(rect.cluster_type, []).append(rect)
        for scheduler in self.cluster_schedulers:
            rm = getattr(scheduler, "rm", None)
            if not getattr(rm, "keep_idle_nodes_on", False):
                continue
            online_nodes = rm.get_powered_on_node_count()
            if online_nodes <= 0:
                continue
            cluster_type = scheduler.cluster_type
            idle_per_node = self.node_configs[cluster_type]["idle_power_w"] * self.pue
            static_floor_w = self._static_idle_floor_w(scheduler)
            _, occupied = self._cluster_node_ledger(scheduler, len(slots))
            for rect in placed_by_cluster.get(cluster_type, []):
                if rect.start_slot_idx is None or rect.end_slot_idx is None:
                    continue
                for idx in range(max(0, rect.start_slot_idx),
                                 min(len(slots), rect.end_slot_idx + 1)):
                    occupied[idx] += rect.num_nodes
            if cluster_type not in cluster_slots:
                # Finished or currently empty clusters still consume idle.
                cluster_slots[cluster_type] = [
                    PowerSlot(
                        start_time=slot.start_time, end_time=slot.end_time,
                        duration_s=slot.duration_s, renewable_power_w=0.0,
                        battery_power_w=0.0, grid_power_w=0.0,
                        green_available_w=0.0,
                        carbon_intensity=slot.carbon_intensity,
                    ) for slot in slots
                ]
            for idx, slot in enumerate(cluster_slots[cluster_type]):
                idle_w = static_floor_w or max(0, online_nodes - occupied[idx]) * idle_per_node
                slot.idle_reserved_w = idle_w
                slot.idle_allocated_w = static_floor_w
                slots[idx].idle_reserved_w += idle_w
                slots[idx].idle_allocated_w += static_floor_w

    def _find_slot(self, time: datetime) -> Optional[PowerSlot]:
        """Find the slot that contains ``time`` in the current plan."""
        plan = self._current_plan
        if plan is None or not plan.slots:
            return None

        # Fast path: compute index directly from time offset.
        offset_s = (time - plan.plan_start).total_seconds()
        idx = int(offset_s // self.slot_duration_s)
        if 0 <= idx < len(plan.slots):
            return plan.slots[idx]

        # Fallback: clamp to boundaries.
        if idx < 0:
            return plan.slots[0]
        return plan.slots[-1]

    @staticmethod
    def _fallback_budget(time: datetime, battery: "Storage") -> BudgetInfo:
        """Return a zero-budget BudgetInfo when no plan is available."""
        return BudgetInfo(
            time=str(time),
            renewable_power_w=0.0,
            budget_renewable_w=0.0,
            charge_reserve_w=0.0,
            battery_soc=battery.soc(),
            battery_discharge_power_w=0.0,
            grid_power_w=0.0,
            total_budget_w=0.0,
            allow_discharge=False,
            charge_power=None,
        )


__all__ = [
    "JobRectangle",
    "PowerSlot",
    "PuzzleBudgetPolicy",
    "PuzzlePlan",
]
