# Earliest-finish-time scheduling policy.

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.policies.base import (
    SchedulingCandidate,
    SchedulingDecision,
    SchedulingPolicy,
)

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy


class EFTPolicy(SchedulingPolicy):
    """
    Earliest Finish Time scheduling policy.

       Evaluates each pending job at every available DVFS frequency on the
       currently idle nodes, then greedily selects feasible assignments with
       the smallest predicted finish time. Start-deadline pressure outranks
       finish time, so a job near its SLA is not starved by shorter work.
    """

    def __init__(self, cluster_mem_bw_gb_s: float = 100.0):
        self.cluster_mem_bw_gb_s = cluster_mem_bw_gb_s

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
        if not pending_jobs or not ready_nodes:
            return []

        ranked_candidates: list[
            tuple[tuple[float, float, float, int, int, int], SchedulingCandidate]
        ] = []

        for job in pending_jobs:
            if job.num_nodes > len(ready_nodes):
                continue

            sla_prefix = self._sla_sort_prefix(job, current_time)

            for freq in dvfs_policy.available_frequencies:
                dyn_power = dvfs_policy.power_at_frequency(freq)

                if energy_budget is not None:
                    projected = current_power + dyn_power * job.num_nodes
                    if projected > energy_budget:
                        continue

                finish_time = self._predict_execution_time(
                    job=job,
                    frequency=freq,
                    num_assigned_nodes=job.num_nodes,
                    cluster_mem_bw_gb_s=self.cluster_mem_bw_gb_s,
                )
                ranked_candidates.append(
                    (
                        (
                            *sla_prefix,
                            finish_time,
                            -job.priority,
                            job.num_nodes,
                            job.job_id,
                        ),
                        SchedulingCandidate(
                            job=job,
                            frequency=freq,
                            dynamic_power_w=dyn_power,
                        ),
                    )
                )

        if not ranked_candidates:
            return []

        ranked_candidates.sort(key=lambda c: c[0])
        return self._pack_candidates(
            sorted_candidates=[candidate for _, candidate in ranked_candidates],
            ready_nodes=ready_nodes,
            energy_budget=energy_budget,
            current_power=current_power,
        )


__all__ = ["EFTPolicy"]
