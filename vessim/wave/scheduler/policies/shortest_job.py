# Shortest/longest predicted-runtime scheduling policies.

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.policies.base import SchedulingDecision, SchedulingPolicy

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy


class SJFPolicy(SchedulingPolicy):
    # Shortest Job First on predicted execution time, deadline-first.

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
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (
                *self._sla_sort_prefix(j, current_time),
                self._predict_execution_time(
                    job=j,
                    frequency=dvfs_policy.base_frequency_ghz,
                    num_assigned_nodes=j.num_nodes,
                    cluster_mem_bw_gb_s=self.cluster_mem_bw_gb_s,
                ),
            ),
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


class LJFPolicy(SchedulingPolicy):
    # Longest Job First on predicted execution time, deadline-first.

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
        sorted_jobs = sorted(
            pending_jobs,
            key=lambda j: (
                *self._sla_sort_prefix(j, current_time),
                -self._predict_execution_time(
                    job=j,
                    frequency=dvfs_policy.base_frequency_ghz,
                    num_assigned_nodes=j.num_nodes,
                    cluster_mem_bw_gb_s=self.cluster_mem_bw_gb_s,
                ),
            ),
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


__all__ = ["LJFPolicy", "SJFPolicy"]
