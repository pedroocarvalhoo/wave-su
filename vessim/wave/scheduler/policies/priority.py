# Priority-based scheduling policy.

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.policies.base import SchedulingDecision, SchedulingPolicy

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy


class PriorityPolicy(SchedulingPolicy):
    # Greedy priority-based scheduling with backfilling, deadline-first.

    def select_jobs(
        self,
        pending_jobs: list[Job],
        ready_nodes: list[str],
        dvfs_policy: DVFSPolicy,
        energy_budget: Optional[float] = None,
        current_power: float = 0.0,
        node_idle_p: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> list[SchedulingDecision]:
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (*self._sla_sort_prefix(j, current_time), -j.priority),
        )
        return self._attempt_scheduling(
            sorted_jobs=sorted_jobs,
            ready_nodes=ready_nodes,
            dvfs_policy=dvfs_policy,
            energy_budget=energy_budget,
            current_power=current_power,
            node_idle_p=node_idle_p,
            allow_backfill=True,
        )


__all__ = ["PriorityPolicy"]
