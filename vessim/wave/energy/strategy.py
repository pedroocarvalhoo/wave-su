"""
Energy allocation strategies across cluster schedulers.

   Strategies receive live `ClusterSnapshot` objects and split the total compute
   budget among clusters. Edit this file when changing inter-cluster budget
   allocation priorities; edit `budget_policy.py` when changing how the total
   budget is produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vessim.wave.scheduler import JobQueueScheduler
    from vessim.wave.energy.puzzle_budget_policy import PuzzleBudgetPolicy


@dataclass
class ClusterSnapshot:
    scheduler: JobQueueScheduler
    cluster_type: str
    total_nodes: int
    powered_on_nodes: int       # Nodes that are IDLE or BUSY.
    busy_nodes: int
    idle_power_per_node: float  # Watts.
    max_power_per_node: float   # Dynamic watts at max frequency.
    current_power: float        # Measured watts.
    is_done: bool               # Cluster has finished all jobs.
    has_pending_jobs: bool      # Queue still contains pending jobs.


class EnergyAllocationStrategy(ABC):

    @abstractmethod
    def allocate(
        self,
        total_budget: float,
        snapshots: list[ClusterSnapshot],
        time: Optional[datetime] = None,
    ) -> dict[JobQueueScheduler, float]:
        """
        Distribute total budget across clusters.

           ``time`` is the current simulation time, passed so plan-aware
           strategies can look up the active plan; snapshot-only strategies
           ignore it.

           Returns:
              Mapping from each scheduler to its allocated budget in watts.
        """


class DefaultAllocationStrategy(EnergyAllocationStrategy):
    """
    Default two-phase allocation strategy.

       Phase 1 assigns each active cluster an idle-power floor. Phase 2
       distributes the remaining budget proportionally to max dynamic demand.
    """

    def allocate(
        self,
        total_budget: float,
        snapshots: list[ClusterSnapshot],
        time: Optional[datetime] = None,
    ) -> dict[JobQueueScheduler, float]:

        allocations: dict[JobQueueScheduler, float] = {}

        # Phase 1: idle power floor for powered-on nodes.
        idle_demands: dict[JobQueueScheduler, float] = {}
        total_idle = 0.0
        for snap in snapshots:
            if snap.is_done:
                idle_demands[snap.scheduler] = 0.0
            else:
                idle = snap.idle_power_per_node * snap.powered_on_nodes
                idle_demands[snap.scheduler] = idle
                total_idle += idle

        # If budget cannot cover idle demand, distribute proportionally.
        if total_budget <= total_idle:
            for snap in snapshots:
                if total_idle > 0:
                    proportion = idle_demands[snap.scheduler] / total_idle
                else:
                    proportion = 0.0
                allocations[snap.scheduler] = total_budget * proportion
            return allocations

        # Phase 2: distribute remaining budget by dynamic demand.
        remaining_budget = total_budget - total_idle

        dynamic_demands: dict[JobQueueScheduler, float] = {}
        total_dynamic = 0.0
        for snap in snapshots:
            if snap.is_done:
                dynamic_demands[snap.scheduler] = 0.0
            else:
                dyn = snap.max_power_per_node * snap.total_nodes
                dynamic_demands[snap.scheduler] = dyn
                total_dynamic += dyn

        for snap in snapshots:
            base = idle_demands[snap.scheduler]
            if total_dynamic > 0 and remaining_budget > 0:
                dyn_share = remaining_budget * (dynamic_demands[snap.scheduler] / total_dynamic)
            else:
                dyn_share = 0.0
            allocations[snap.scheduler] = base + dyn_share

        return allocations


class ComputePrioritizedStrategy(EnergyAllocationStrategy):
    """
    Prioritizes computation over idle power.

       First, cover currently busy nodes. Then allocate remaining budget to
       clusters with pending work. Idle nodes that cannot be funded receive no
       budget, so the cluster scheduler can power them off.
    """

    def allocate(
        self,
        total_budget: float,
        snapshots: list[ClusterSnapshot],
        time: Optional[datetime] = None,
    ) -> dict[JobQueueScheduler, float]:

        allocations: dict[JobQueueScheduler, float] = {}

        # Phase 1: cover currently running computation.
        busy_demands: dict[JobQueueScheduler, float] = {}
        total_busy_demand = 0.0
        for snap in snapshots:
            if snap.is_done:
                busy_demands[snap.scheduler] = 0.0
            else:
                # Use measured power for busy nodes already running.
                busy_demands[snap.scheduler] = snap.current_power
                total_busy_demand += snap.current_power

        # If budget cannot cover running computation, distribute proportionally.
        if total_budget <= total_busy_demand:
            for snap in snapshots:
                if total_busy_demand > 0:
                    proportion = busy_demands[snap.scheduler] / total_busy_demand
                else:
                    proportion = 0.0
                allocations[snap.scheduler] = total_budget * proportion
            return allocations

        # Phase 2: send remaining budget to clusters with pending jobs.
        remaining = total_budget - total_busy_demand

        growth_demands: dict[JobQueueScheduler, float] = {}
        total_growth = 0.0
        for snap in snapshots:
            if snap.is_done or not snap.has_pending_jobs:
                growth_demands[snap.scheduler] = 0.0
            else:
                # Extra power this cluster could use if all available nodes ran jobs.
                available_nodes = snap.total_nodes - snap.busy_nodes
                growth = (snap.idle_power_per_node + snap.max_power_per_node) * available_nodes
                growth_demands[snap.scheduler] = growth
                total_growth += growth

        for snap in snapshots:
            base = busy_demands[snap.scheduler]
            if total_growth > 0 and remaining > 0:
                growth_share = remaining * (growth_demands[snap.scheduler] / total_growth)
            else:
                growth_share = 0.0
            allocations[snap.scheduler] = base + growth_share

        return allocations


class PuzzlePlanAllocationStrategy(EnergyAllocationStrategy):
    """Split the budget across clusters the way the puzzle plan did.

       ``PuzzlePlan.cluster_slots`` records the per-cluster power footprint of
       the jobs placed on the shared board (or the fixed slice in legacy
       cluster-local mode).  This strategy hands each cluster its planned
       share at runtime, while first protecting the measured power of work
       already running.  This keeps a falling plan slot from assigning a
       cluster less power than it can physically shed.

       Finished clusters reserve their measured power if their nodes remain
       on; the remaining budget goes to the still-active clusters. When no plan
       covers ``time`` (e.g. before the first plan, or a cluster the plan
       did not fund), it falls back to ``fallback_strategy``.
    """

    def __init__(
        self,
        budget_policy: "PuzzleBudgetPolicy",
        fallback_strategy: Optional[EnergyAllocationStrategy] = None,
    ):
        self.budget_policy = budget_policy
        self.fallback_strategy = fallback_strategy or ComputePrioritizedStrategy()

    def allocate(
        self,
        total_budget: float,
        snapshots: list[ClusterSnapshot],
        time: Optional[datetime] = None,
    ) -> dict[JobQueueScheduler, float]:

        plan_budgets: dict[str, float] = {}
        if time is not None and hasattr(self.budget_policy, "get_cluster_compute_budgets"):
            plan_budgets = self.budget_policy.get_cluster_compute_budgets(time)

        # Completed schedulers cannot use compute budget, but powered-on nodes
        # still draw power. Reserve that draw before allocating to active work.
        finished_power = {
            snap.scheduler: (
                max(0.0, float(snap.current_power)) if snap.is_done else 0.0
            )
            for snap in snapshots
        }
        finished_total = sum(finished_power.values())
        source_budget = max(0.0, float(total_budget))
        reserve_fraction = (
            min(1.0, source_budget / finished_total)
            if finished_total > 0 else 0.0
        )
        finished_reservations = {
            scheduler: power * reserve_fraction
            for scheduler, power in finished_power.items()
        }
        active_budget = source_budget - sum(finished_reservations.values())

        def include_finished(active_allocations):
            return {
                snap.scheduler: (
                    finished_reservations[snap.scheduler]
                    if snap.is_done else active_allocations.get(snap.scheduler, 0.0)
                )
                for snap in snapshots
            }

        # Weight each active cluster by its planned compute share.
        weights: dict[JobQueueScheduler, float] = {}
        total_weight = 0.0
        for snap in snapshots:
            if snap.is_done:
                weights[snap.scheduler] = 0.0
                continue
            share = float(plan_budgets.get(snap.cluster_type, 0.0))
            weights[snap.scheduler] = max(0.0, share)
            total_weight += weights[snap.scheduler]

        if total_weight <= 0:
            # No plan share for the active clusters — defer to the fallback.
            fallback = self.fallback_strategy.allocate(active_budget, snapshots, time)
            if finished_total == 0:
                return fallback
            return include_finished(fallback)

        # Phase 1: preserve measured live power.  PuzzleBudgetPolicy raises the
        # aggregate budget to at least this floor; the proportional branch is a
        # defensive fallback for custom/fake policies that do not.
        floors = {
            snap.scheduler: (
                0.0 if snap.is_done else max(0.0, float(snap.current_power))
            )
            for snap in snapshots
        }
        floor_total = sum(floors.values())
        if active_budget <= floor_total:
            return include_finished({
                snap.scheduler: (
                    active_budget * floors[snap.scheduler] / floor_total
                    if floor_total > 0 else 0.0
                )
                for snap in snapshots
            })

        allocations = dict(floors)
        remaining = active_budget - floor_total
        pue = max(1e-12, float(getattr(self.budget_policy, "pue", 1.0)))
        headroom = {
            snap.scheduler: max(
                0.0, weights[snap.scheduler] / pue - floors[snap.scheduler],
            )
            for snap in snapshots
        }
        total_headroom = sum(headroom.values())

        # First move each cluster toward its planned IT share.
        toward_plan = min(remaining, total_headroom)
        if total_headroom > 0:
            for snap in snapshots:
                allocations[snap.scheduler] += (
                    toward_plan * headroom[snap.scheduler] / total_headroom
                )
        remaining -= toward_plan

        # Any surplus beyond all planned shares retains the plan proportions.
        if remaining > 0:
            for snap in snapshots:
                allocations[snap.scheduler] += (
                    remaining * weights[snap.scheduler] / total_weight
                )

        return include_finished(allocations)
