# Logical multi-queue scheduling policy.

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.policies.base import (
    SchedulingDecision,
    SchedulingPolicy,
    select_jobs_with_optional_time,
)
from vessim.wave.scheduler.policies.priority import PriorityPolicy
from vessim.wave.scheduler.queues import QueueDefinition

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy


class MultiQueuePolicy(SchedulingPolicy):
    """
    Coordinates logical queues that share the same physical cluster nodes.

       Each queue can use its own local scheduling policy. The coordinator then
       merges the best local candidates into a global order that respects queue
       SLA pressure before queue and job priority.
    """

    def __init__(
        self,
        queue_definitions: list[QueueDefinition],
        default_policy: Optional[SchedulingPolicy] = None,
        queue_policies: Optional[dict[str, SchedulingPolicy]] = None,
        sla_policy: str = "enforce",
    ):
        if sla_policy not in {"enforce", "report_only"}:
            raise ValueError("sla_policy must be 'enforce' or 'report_only'")

        self.queue_definitions = list(queue_definitions)
        self.definition_by_name = {
            definition.name: definition
            for definition in self.queue_definitions
        }
        self.default_policy = default_policy or PriorityPolicy()
        self.queue_policies = queue_policies or {}
        self.set_sla_policy(sla_policy)

    def set_prediction_error_model(self, error_rate: float, seed: int) -> None:
        super().set_prediction_error_model(error_rate, seed)
        self.default_policy.set_prediction_error_model(error_rate, seed)
        for policy in self.queue_policies.values():
            policy.set_prediction_error_model(error_rate, seed)

    def set_sla_policy(self, sla_policy: str) -> None:
        # Cross-queue merging and each queue's local ordering must agree on the
        # SLA setting, otherwise a queue hands up a candidate chosen without
        # deadline pressure and the merge can only rank what it was given.
        super().set_sla_policy(sla_policy)
        self.default_policy.set_sla_policy(sla_policy)
        for policy in self.queue_policies.values():
            policy.set_sla_policy(sla_policy)

    def dispatchable_pending_jobs(
        self,
        pending_jobs: list[Job],
        current_time: Optional[datetime] = None,
    ) -> list[Job]:
        """Delegate to the wrapped default policy for power-on gating.

        The default policy owns dispatch timing (e.g. the Puzzle plan), so it
        decides which pending jobs are eligible to run now; the multi-queue
        wrapper only layers SLA/priority ordering on top of that.
        """
        return self.default_policy.dispatchable_pending_jobs(
            pending_jobs, current_time
        )

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

        decisions: list[SchedulingDecision] = []
        remaining_jobs = list(pending_jobs)
        available_nodes = list(ready_nodes)
        sim_power = current_power

        while remaining_jobs and available_nodes:
            candidates = self._queue_candidates(
                remaining_jobs=remaining_jobs,
                available_nodes=available_nodes,
                dvfs_policy=dvfs_policy,
                energy_budget=energy_budget,
                current_power=sim_power,
                node_idle_p=node_idle_p,
                current_time=current_time,
            )
            if not candidates:
                break

            candidates.sort(key=lambda item: item[0])
            selected = candidates[0][1]
            power_delta = (
                selected.dynamic_power_w * selected.job.num_nodes
                if selected.dynamic_power_w is not None
                else 0.0
            )
            if energy_budget is not None and sim_power + power_delta > energy_budget:
                remaining_jobs = [
                    job for job in remaining_jobs
                    if job.job_id != selected.job.job_id
                ]
                continue

            decisions.append(selected)
            selected_nodes = set(selected.assigned_nodes)
            available_nodes = [
                node for node in available_nodes
                if node not in selected_nodes
            ]
            sim_power += power_delta
            remaining_jobs = [
                job for job in remaining_jobs
                if job.job_id != selected.job.job_id
            ]

        return decisions

    def _queue_candidates(
        self,
        remaining_jobs: list[Job],
        available_nodes: list[str],
        dvfs_policy: DVFSPolicy,
        energy_budget: Optional[float],
        current_power: float,
        node_idle_p: float,
        current_time: Optional[datetime],
    ) -> list[tuple[tuple[float, float, int, int, int], SchedulingDecision]]:
        candidates: list[tuple[tuple[float, float, int, int, int], SchedulingDecision]] = []
        jobs_by_queue = self._jobs_by_queue(remaining_jobs)

        for queue_name, queue_jobs in jobs_by_queue.items():
            policy = self.queue_policies.get(queue_name, self.default_policy)
            decisions = select_jobs_with_optional_time(
                policy=policy,
                pending_jobs=queue_jobs,
                ready_nodes=available_nodes,
                dvfs_policy=dvfs_policy,
                energy_budget=energy_budget,
                current_power=current_power,
                node_idle_p=node_idle_p,
                current_time=current_time,
            )
            if not decisions:
                continue

            decision = decisions[0]
            candidates.append(
                (
                    self._candidate_key(decision.job, current_time),
                    decision,
                )
            )

        return candidates

    def _jobs_by_queue(self, jobs: list[Job]) -> dict[str, list[Job]]:
        grouped: dict[str, list[Job]] = {}
        for job in jobs:
            queue_name = job.queue_name or ""
            grouped.setdefault(queue_name, []).append(job)
        return grouped

    def _candidate_key(
        self,
        job: Job,
        current_time: Optional[datetime],
    ) -> tuple[float, float, int, int, int]:
        sla_bucket, deadline_key = self._sla_sort_prefix(job, current_time)

        definition = self.definition_by_name.get(job.queue_name or "")
        priority_weight = (
            definition.priority_weight
            if definition is not None
            else getattr(job, "queue_priority_weight", 0)
        )

        return (
            sla_bucket,
            deadline_key,
            -priority_weight,
            -getattr(job, "priority", 0),
            job.job_id,
        )


__all__ = ["MultiQueuePolicy"]
