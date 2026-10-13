"""
Plan-driven scheduling policy for the Puzzle Scheduler.

    Consults the ``PuzzleBudgetPolicy`` pre-computed plan to dispatch
    exactly the jobs that were packed into the current time slot.

    Designed to be used as the ``default_policy`` inside a
    ``MultiQueuePolicy`` wrapper so that SLA enforcement, queue priority,
    and deadline tracking are preserved.

    Does not access the plan internals — only calls
    ``PuzzleBudgetPolicy.get_due_planned_jobs(time)`` /
    ``get_planned_job_ids()`` to learn which jobs the plan scheduled.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from datetime import datetime
from math import inf
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.policies.base import (
    SchedulingDecision,
    SchedulingPolicy,
)
from vessim.wave.scheduler.queues import queue_deadline_s

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy
    from vessim.wave.energy.puzzle_budget_policy import PuzzleBudgetPolicy


class PuzzleSchedulingPolicy(SchedulingPolicy):
    """Dispatch jobs according to the Puzzle Scheduler plan.

    The plan steers dispatch.  On each step the policy:

    1. Queries ``get_due_planned_jobs(current_time, cluster_type)`` for the
       rectangles whose planned start slot has arrived
       (``start_slot_idx <= current slot``).  Jobs the plan scheduled for a
       *future* slot are withheld, so dispatch follows the plan's timing
       instead of front-running it into earlier (possibly browner) slots.
    2. Dispatches those due jobs in plan order (earliest planned slot first,
       then queue urgency), respecting ready nodes and the energy budget and
       using the frequency from the plan.  Because the window is
       ``[start_slot, ...)`` rather than a single slot, a job that could not
       start exactly at its slot (nodes busy) stays due and retries later
       instead of being orphaned.
    3. If any due job still cannot be dispatched, ``request_replan()`` asks
       the budget policy to re-pack (rate-limited).

    A job the plan could not place at all (*unplaced*) has no slot for the
    plan to steer.  When ``enable_fallback_dispatch`` is set, a
    safety-net pass dispatches such unplaced pending jobs into any leftover
    nodes/budget, ordered by urgency and bounded by ``energy_budget`` (the
    green envelope).  It never touches jobs the plan scheduled for the
    future, so it cannot override the plan's timing.  Set it to ``False``
    for strict plan-only dispatch.
    """

    def __init__(
        self,
        puzzle_budget_policy: "PuzzleBudgetPolicy",
        cluster_type: str,
        enable_fallback_dispatch: bool = True,
        allow_front_run: bool = False,
        defer_carbon_margin: Optional[float] = None,
    ):
        if not cluster_type:
            raise ValueError("PuzzleSchedulingPolicy requires a cluster_type")

        self.puzzle_policy = puzzle_budget_policy
        self.cluster_type = cluster_type
        self.enable_fallback_dispatch = enable_fallback_dispatch
        # When True the safety-net pass may also dispatch jobs the plan
        # scheduled for a *future* slot (front-running), trading the plan's
        # timing for maximum utilization. Default False keeps dispatch
        # steered by the plan.
        self.allow_front_run = allow_front_run
        # Selective carbon-aware deferral (gCO2/kWh margin).  When set, the
        # dispatcher front-runs future-planned jobs *by default* (like
        # ``allow_front_run``) but HOLDS a job for the slot the plan chose when
        # running it now would be dirtier than that slot by more than this
        # margin — and only while its start deadline is not at risk.  This
        # targets the carbon/latency Pareto point between pure front-run (never
        # defers) and pure steering (always defers): it defers exactly the jobs
        # whose wait buys real carbon, running the rest immediately.  ``None``
        # disables it (behaviour falls back to ``allow_front_run``).
        self.defer_carbon_margin = defer_carbon_margin
        # Distinct job ids dispatched/missed per path, for the plan-summary
        # export. Tracked as sets because ``MultiQueuePolicy`` re-invokes
        # ``select_jobs`` many times per step (offering the same pending jobs
        # until one is picked), so a plain counter would massively overcount.
        self._planned_ids: set[int] = set()
        self._missed_ids: set[int] = set()
        self._fallback_ids: set[int] = set()
        # Planned jobs dispatched *ahead* of their planned slot (front-run).
        self._front_run_ids: set[int] = set()
        # Of the plan-placed dispatches (due or front-run), how many the plan
        # attributed to the green board vs the grid-overflow lane (the carbon
        # penalty for a badly-built puzzle).
        self._green_ids: set[int] = set()
        self._grid_ids: set[int] = set()
        # ``MultiQueuePolicy`` re-invokes ``select_jobs`` once per job it
        # dispatches in a step, always at the same ``current_time``, so the
        # urgency ordering it sorts by is recomputed identically every call —
        # datetime arithmetic per job per invocation.  Memoize it for the step.
        self._urgency_time: Optional[datetime] = None
        self._urgency_cache: dict[int, tuple[float, float, int]] = {}
        # Plan frequency -> nearest available DVFS frequency.  The plan reuses a
        # handful of distinct frequencies across hundreds of rectangles.
        self._freq_snap: dict[float, float] = {}

    @property
    def dispatch_stats(self) -> dict[str, int]:
        """Whole-run distinct-job dispatch counts for the summary export.

        The tracking sets accumulate for the lifetime of the policy (they are
        never cleared per replan), because the consumer
        (``puzzle_dispatch_summary.csv``) is written once at end of run and
        wants run totals, not the counts for the current plan.  Each counter is
        a distinct-``job_id`` set, so re-offers of the same pending job within a
        step are not double-counted.
        """
        return {
            "planned_dispatched": len(self._planned_ids),
            "planned_missed": len(self._missed_ids),
            "fallback_dispatched": len(self._fallback_ids),
            "front_run_dispatched": len(self._front_run_ids),
            "green_dispatched": len(self._green_ids),
            "grid_dispatched": len(self._grid_ids),
        }

    def dispatchable_pending_jobs(
        self,
        pending_jobs: list["Job"],
        current_time: Optional[datetime] = None,
    ) -> list["Job"]:
        """Only jobs the plan would dispatch now steer node power-on.

        These are the jobs ``select_jobs`` can actually place this step: the
        plan's *due* rectangles (planned start slot has arrived), plus — when
        the safety-net fallback is enabled — jobs the plan never placed
        (*unplaced*), which the fallback pass may still dispatch.  Jobs the
        plan scheduled for a *future* slot are withheld, so the cluster is not
        powered up early just to sit idle until their slot arrives.
        """
        if current_time is None:
            return list(pending_jobs)

        due_ids = {
            rect.job_id
            for rect in self.puzzle_policy.get_due_planned_jobs(
                current_time, cluster_type=self.cluster_type
            )
        }
        planned_ids = self.puzzle_policy.get_planned_job_ids(
            cluster_type=self.cluster_type,
        )
        selective = self.defer_carbon_margin is not None
        rects = (
            self.puzzle_policy.planned_rects_by_id(self.cluster_type)
            if selective else {}
        )
        eligible = []
        for job in pending_jobs:
            is_due = job.job_id in due_ids
            # Unplaced jobs (never planned) are eligible only when the
            # safety-net fallback is allowed to dispatch them.
            is_fallback = (
                self.enable_fallback_dispatch
                and job.job_id not in planned_ids
            )
            # Selective deferral: a future-planned job is eligible now (so its
            # nodes may power on) unless we are holding it for a cleaner slot.
            is_front = (
                selective
                and not is_due
                and job.job_id in planned_ids
                and not self._hold_for_greener_slot(
                    rects.get(job.job_id), current_time
                )
            )
            if is_due or is_fallback or is_front:
                eligible.append(job)
        return eligible

    def select_jobs(
        self,
        pending_jobs: list["Job"],
        ready_nodes: list[str],
        dvfs_policy: "DVFSPolicy",
        energy_budget: Optional[float] = None,
        current_power: float = 0.0,
        node_idle_p: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> list[SchedulingDecision]:
        if current_time is None or not pending_jobs or not ready_nodes:
            return []

        pending_by_key = {
            (self.cluster_type, job.job_id): job
            for job in pending_jobs
        }

        # 1. Dispatch the jobs the plan scheduled for now or earlier.
        due_rects = self.puzzle_policy.get_due_planned_jobs(
            current_time,
            cluster_type=self.cluster_type,
        )
        # Keep only rects whose job is still pending, then order by the
        # plan's timeline (earliest planned slot first) and queue urgency.
        due_rects = [
            rect for rect in due_rects
            if (rect.cluster_type, rect.job_id) in pending_by_key
        ]
        due_rects.sort(
            key=lambda rect: (
                rect.start_slot_idx if rect.start_slot_idx is not None else 0,
                self._urgency_key(
                    pending_by_key[(rect.cluster_type, rect.job_id)],
                    current_time,
                ),
                -rect.energy_wh,
            )
        )

        decisions: list[SchedulingDecision] = []
        available_nodes = list(ready_nodes)
        sim_power = current_power
        missed_any = False
        selected_keys: set[tuple[str, int]] = set()

        for rect in due_rects:
            job_key = (rect.cluster_type, rect.job_id)
            if job_key in selected_keys:
                continue
            job = pending_by_key[job_key]

            if job.num_nodes > len(available_nodes):
                # Not enough ready nodes now — keep it due, ask to re-pack.
                missed_any = True
                self._missed_ids.add(job.job_id)
                continue

            # Use the plan's frequency, snapped to the nearest available
            # DVFS frequency to absorb node_config vs DVFS-table rounding.
            frequency = self._snap_frequency(dvfs_policy, rect.frequency_ghz)
            dyn_power = dvfs_policy.power_at_frequency(frequency)

            if energy_budget is not None:
                projected = sim_power + dyn_power * job.num_nodes
                if projected > energy_budget:
                    missed_any = True
                    self._missed_ids.add(job.job_id)
                    continue

            assigned = available_nodes[: job.num_nodes]
            available_nodes = available_nodes[job.num_nodes:]
            sim_power += dyn_power * job.num_nodes

            selected_keys.add(job_key)
            self._planned_ids.add(job.job_id)
            # ``planned_missed`` is a plan-DISPATCH diagnostic (planned jobs the
            # dispatcher could not place), NOT an SLA outcome — real deadline
            # misses are tracked per job as ``Queue_Deadline_Missed``. A job
            # skipped on an earlier offer but now dispatched was not undispatched,
            # so drop it here; the counter then reflects jobs never placed rather
            # than every transient skip.
            self._missed_ids.discard(job.job_id)
            # Record how the plan intended to power this job, for per-job
            # energy-source attribution in the metrics export.
            if rect.used_grid:
                job.planned_energy_source = "grid"
                self._grid_ids.add(job.job_id)
            else:
                job.planned_energy_source = "green"
                self._green_ids.add(job.job_id)

            decisions.append(
                SchedulingDecision(
                    job=job,
                    assigned_nodes=assigned,
                    frequency=frequency,
                    dynamic_power_w=dyn_power,
                )
            )

        # 2. Ask for a re-pack if a due job could not be dispatched.
        if missed_any:
            self.puzzle_policy.request_replan(current_time)

        # 3. Safety net: dispatch jobs the plan could not place at all.
        #    Never touches jobs scheduled for a future slot, so the plan's
        #    timing is preserved; bounded by ``energy_budget``.
        if self.enable_fallback_dispatch and available_nodes:
            fallback = self._fallback_dispatch(
                pending_jobs=pending_jobs,
                selected_keys=selected_keys,
                available_nodes=available_nodes,
                dvfs_policy=dvfs_policy,
                energy_budget=energy_budget,
                current_power=sim_power,
                node_idle_p=node_idle_p,
                current_time=current_time,
            )
            if fallback:
                planned_rects = self.puzzle_policy.planned_rects_by_id(
                    self.cluster_type,
                )
                cur_idx = self.puzzle_policy.current_slot_idx(current_time)
                for decision in fallback:
                    job = decision.job
                    self._missed_ids.discard(job.job_id)
                    rect = planned_rects.get(job.job_id)
                    if rect is None:
                        # True safety net: the plan never placed this job.
                        self._fallback_ids.add(job.job_id)
                        job.planned_energy_source = "fallback"
                        continue
                    # Plan-placed job dispatched through the fallback lane:
                    # keep the plan's green/grid attribution instead of
                    # collapsing it into "fallback".  A job pulled ahead of
                    # its planned slot is tagged ``front_run_*`` so
                    # anticipated dispatches stay distinguishable from
                    # dispatches at the planned slot.
                    front_run = (
                        rect.start_slot_idx is not None
                        and cur_idx is not None
                        and rect.start_slot_idx > cur_idx
                    )
                    source = "grid" if rect.used_grid else "green"
                    if front_run:
                        self._front_run_ids.add(job.job_id)
                        job.planned_energy_source = f"front_run_{source}"
                    else:
                        self._planned_ids.add(job.job_id)
                        job.planned_energy_source = source
                    if rect.used_grid:
                        self._grid_ids.add(job.job_id)
                    else:
                        self._green_ids.add(job.job_id)
                decisions.extend(fallback)

        return decisions

    def _fallback_dispatch(
        self,
        pending_jobs: list["Job"],
        selected_keys: set[tuple[str, int]],
        available_nodes: list[str],
        dvfs_policy: "DVFSPolicy",
        energy_budget: Optional[float],
        current_power: float,
        node_idle_p: float,
        current_time: datetime,
    ) -> list[SchedulingDecision]:
        """Dispatch pending jobs beyond the due-planned set.

        By default only jobs the plan never placed (*unplaced*) are
        eligible — jobs the plan scheduled for now or the future are
        excluded via ``get_planned_job_ids`` so this never front-runs the
        plan.  When ``allow_front_run`` is set, future-planned jobs are
        also eligible, trading the plan's timing for utilization.  Reuses
        the inherited ``_attempt_scheduling`` helper, which assigns nodes,
        snaps DVFS frequencies, and respects ``energy_budget``.
        """
        selective = self.defer_carbon_margin is not None
        if self.allow_front_run or selective:
            # Future-planned jobs are eligible to be front-run; selective mode
            # then holds back the subset worth deferring for a cleaner slot.
            excluded_ids: AbstractSet[int] = frozenset()
        else:
            excluded_ids = self.puzzle_policy.get_planned_job_ids(
                cluster_type=self.cluster_type,
            )
        remaining = [
            job for job in pending_jobs
            if (self.cluster_type, job.job_id) not in selected_keys
            and job.job_id not in excluded_ids
        ]
        if selective and remaining:
            rects = self.puzzle_policy.planned_rects_by_id(self.cluster_type)
            remaining = [
                job for job in remaining
                if not self._hold_for_greener_slot(
                    rects.get(job.job_id), current_time
                )
            ]
        if not remaining:
            return []

        remaining.sort(key=lambda job: self._urgency_key(job, current_time))

        return self._attempt_scheduling(
            sorted_jobs=remaining,
            ready_nodes=available_nodes,
            dvfs_policy=dvfs_policy,
            energy_budget=energy_budget,
            current_power=current_power,
            node_idle_p=node_idle_p,
            allow_backfill=True,
        )

    def _hold_for_greener_slot(
        self,
        rect,
        current_time: datetime,
    ) -> bool:
        """Selective deferral: should this future-planned job wait for a slot?

        Returns ``True`` only when holding the job for the slot the plan chose
        buys real carbon, and is deadline-safe.  All must hold:

        * selective deferral is enabled (``defer_carbon_margin`` is set) and the
          job has a placed rectangle;
        * the job is planned for a *future* slot (not yet due);
        * running now is dirtier than its planned slot by more than
          ``defer_carbon_margin`` gCO2/kWh — i.e. the plan deferred it to a
          genuinely cleaner window (a job the plan parked at a slot no cleaner
          than now, e.g. a grid-lane spill, gains nothing from waiting and is
          front-run instead);
        * the deadline is not at risk — the current slot is still before the
          job's latest feasible start.

        Everything else is front-run (dispatched now), so only the jobs whose
        wait pays a carbon dividend are deferred.
        """
        if self.defer_carbon_margin is None or rect is None:
            return False
        cur_idx = self.puzzle_policy.current_slot_idx(current_time)
        if cur_idx is None or rect.start_slot_idx is None:
            return False
        if rect.start_slot_idx <= cur_idx:
            return False  # already due — not a deferral decision
        latest = getattr(rect, "latest_start_slot_idx", None)
        if latest is not None and cur_idx >= latest:
            return False  # deadline safety: must start now or miss it
        ci_now = self.puzzle_policy.slot_carbon_intensity(current_time)
        ci_planned = self.puzzle_policy.rect_planned_carbon(rect)
        if ci_now is None or ci_planned is None:
            return False
        return ci_now > ci_planned + self.defer_carbon_margin

    def _urgency_key(
        self,
        job: "Job",
        current_time: datetime,
    ) -> tuple[float, float, int]:
        """Order pending jobs most-urgent first for fallback dispatch.

        Earliest queue deadline wins; jobs without a deadline sort last.
        Ties break by arrival time (FIFO) then job id for determinism — so the
        ordering is total, and callers may feed it jobs in any order.

        Memoized for the current step: the key is a pure function of the job's
        deadline/arrival and ``current_time``, both fixed within a step.
        """
        if current_time != self._urgency_time:
            self._urgency_time = current_time
            self._urgency_cache = {}
        cached = self._urgency_cache.get(job.job_id)
        if cached is not None:
            return cached

        deadline_remaining = queue_deadline_s(job, current_time)
        deadline_key = inf if deadline_remaining is None else deadline_remaining

        arrival_time = getattr(job, "arrival_time", None)
        if arrival_time is None:
            arrival_key = 0.0
        else:
            arrival_key = arrival_time.timestamp()

        key = (deadline_key, arrival_key, job.job_id)
        self._urgency_cache[job.job_id] = key
        return key

    def _snap_frequency(
        self,
        dvfs_policy: "DVFSPolicy",
        plan_frequency: float,
    ) -> float:
        """Nearest available DVFS frequency to the one the plan chose.

        Absorbs node_config vs DVFS-table rounding.  Cached per plan frequency:
        ties resolve to the first minimum in ``available_frequencies``, exactly
        as the underlying ``min`` would.
        """
        snapped = self._freq_snap.get(plan_frequency)
        if snapped is None:
            snapped = min(
                dvfs_policy.available_frequencies,
                key=lambda f: abs(f - plan_frequency),
            )
            self._freq_snap[plan_frequency] = snapped
        return snapped


__all__ = ["PuzzleSchedulingPolicy"]
