"""
Shared scheduling policy abstractions.

   This module owns the policy interface, scheduling decision dataclasses, and
   packing helpers used by concrete policies. Concrete policy behavior belongs
   in sibling modules such as `priority.py`, `fifo.py`, and `dvfs_aware.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import inspect
from math import inf
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.queues import queue_deadline_s
from vessim.wave.scheduler.runtime.execution_models import predict_execution_time

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy


@dataclass
class SchedulingDecision:
    # Single job-to-nodes assignment proposed by a scheduling policy.

    job: Job
    assigned_nodes: list[str]
    frequency: float
    dynamic_power_w: Optional[float] = None


@dataclass
class SchedulingCandidate:
    job: Job
    frequency: float
    dynamic_power_w: Optional[float]


class SchedulingPolicy(ABC):
    prediction_error_rate: float = 0.0
    prediction_seed: int = 0
    sla_policy: str = "enforce"

    def set_prediction_error_model(self, error_rate: float, seed: int) -> None:
        self.prediction_error_rate = error_rate
        self.prediction_seed = seed

    def set_sla_policy(self, sla_policy: str) -> None:
        """Adopt the cluster's SLA setting (`enforce` or `report_only`).

        Wrapper policies override this to propagate the setting to the policies
        they delegate to, so one system-level `sla_policy` reaches every layer
        that orders jobs.
        """
        if sla_policy not in {"enforce", "report_only"}:
            raise ValueError("sla_policy must be 'enforce' or 'report_only'")
        self.sla_policy = sla_policy

    def _sla_sort_prefix(
        self,
        job: Job,
        current_time: Optional[datetime],
    ) -> tuple[float, float]:
        """Leading sort terms that put start-deadline pressure ahead of a
        policy's own ordering.

        Returns ``(bucket, slack)``: bucket 0.0 for a job that already missed
        its start deadline (dispatch it first so the queue drains instead of
        accumulating misses), 0.5 for a job still inside its window ordered by
        remaining slack (earliest deadline first), and 1.0 for a job with no
        deadline at all.  Under `report_only`, or with no queue deadline in
        play, every job gets the same neutral prefix and the policy's own key
        decides — which is the pre-SLA behaviour.

        Shared with `MultiQueuePolicy`, which applies the same two terms when
        merging queues, so ordering within a queue and across queues agree.
        """
        if self.sla_policy != "enforce" or current_time is None:
            return (1.0, inf)
        remaining = queue_deadline_s(job, current_time)
        if remaining is None:
            return (1.0, inf)
        return (0.0 if remaining <= 0 else 0.5, max(0.0, remaining))

    def dispatchable_pending_jobs(
        self,
        pending_jobs: list[Job],
        current_time: Optional[datetime] = None,
    ) -> list[Job]:
        """Pending jobs this policy would actually consider dispatching now.

        Used by node power management to decide whether powering nodes on can
        pay off: waking the cluster for jobs the policy will not dispatch this
        step (e.g. a plan withholding them for a greener slot) just burns idle
        energy.  The base policy is work-conserving, so every pending job is a
        candidate; plan-driven policies (e.g. the Puzzle scheduler) narrow this
        to the jobs that are actually eligible right now.
        """
        return list(pending_jobs)

    def _predict_execution_time(
        self,
        job: Job,
        frequency: float,
        num_assigned_nodes: int,
        cluster_mem_bw_gb_s: float,
    ) -> float:
        return predict_execution_time(
            job=job,
            frequency=frequency,
            num_assigned_nodes=num_assigned_nodes,
            cluster_mem_bw_gb_s=cluster_mem_bw_gb_s,
            prediction_error_rate=self.prediction_error_rate,
            prediction_seed=self.prediction_seed,
        )

    @abstractmethod
    def select_jobs(
        self,
        pending_jobs: list[Job],
        ready_nodes: list[str],
        dvfs_policy: DVFSPolicy,
        energy_budget: Optional[float] = None,
        current_power: float = 0.0,
        node_idle_p: float = 0.0,
    ) -> list[SchedulingDecision]:
        # Select and allocate jobs to run in the current step.
        ...

    def _attempt_scheduling(
        self,
        sorted_jobs: list[Job],
        ready_nodes: list[str],
        dvfs_policy: DVFSPolicy,
        energy_budget: Optional[float],
        current_power: float,
        node_idle_p: float,
        allow_backfill: bool,
    ) -> list[SchedulingDecision]:
        decisions: list[SchedulingDecision] = []
        available_nodes = list(ready_nodes)
        sim_power = current_power

        for job in sorted_jobs:
            if not available_nodes:
                break

            if job.num_nodes > len(available_nodes):
                if allow_backfill:
                    continue
                break

            assigned = available_nodes[: job.num_nodes]

            if energy_budget is not None:
                frequency = dvfs_policy.select_frequency(
                    node_idle_p=node_idle_p,
                    budget_remaining=energy_budget,
                    current_power=sim_power,
                    num_nodes_for_job=job.num_nodes,
                )
                dyn_power = dvfs_policy.power_at_frequency(frequency)
                projected = sim_power + dyn_power * job.num_nodes

                if projected > energy_budget:
                    if allow_backfill:
                        continue
                    break
            else:
                frequency = dvfs_policy.base_frequency_ghz
                dyn_power = None

            available_nodes = available_nodes[job.num_nodes :]
            if dyn_power is not None:
                sim_power += dyn_power * job.num_nodes

            decisions.append(
                SchedulingDecision(
                    job=job,
                    assigned_nodes=assigned,
                    frequency=frequency,
                    dynamic_power_w=dyn_power,
                )
            )

        return decisions

    def _pack_candidates(
        self,
        sorted_candidates: list[SchedulingCandidate],
        ready_nodes: list[str],
        energy_budget: Optional[float],
        current_power: float,
    ) -> list[SchedulingDecision]:
        decisions: list[SchedulingDecision] = []
        available_nodes = list(ready_nodes)
        sim_power = current_power
        scheduled_job_ids: set[int] = set()

        for candidate in sorted_candidates:
            job = candidate.job
            if job.job_id in scheduled_job_ids:
                continue
            if job.num_nodes > len(available_nodes):
                continue

            power_delta = (
                candidate.dynamic_power_w * job.num_nodes
                if candidate.dynamic_power_w is not None
                else 0.0
            )
            if energy_budget is not None and sim_power + power_delta > energy_budget:
                continue

            assigned = available_nodes[: job.num_nodes]
            available_nodes = available_nodes[job.num_nodes :]
            sim_power += power_delta
            scheduled_job_ids.add(job.job_id)

            decisions.append(
                SchedulingDecision(
                    job=job,
                    assigned_nodes=assigned,
                    frequency=candidate.frequency,
                    dynamic_power_w=candidate.dynamic_power_w,
                )
            )

        return decisions


def select_jobs_with_optional_time(
    policy: SchedulingPolicy,
    pending_jobs: list[Job],
    ready_nodes: list[str],
    dvfs_policy: DVFSPolicy,
    energy_budget: Optional[float] = None,
    current_power: float = 0.0,
    node_idle_p: float = 0.0,
    current_time: Optional[datetime] = None,
) -> list[SchedulingDecision]:
    kwargs = {
        "pending_jobs": pending_jobs,
        "ready_nodes": ready_nodes,
        "dvfs_policy": dvfs_policy,
        "energy_budget": energy_budget,
        "current_power": current_power,
        "node_idle_p": node_idle_p,
    }
    if "current_time" in inspect.signature(policy.select_jobs).parameters:
        kwargs["current_time"] = current_time
    return policy.select_jobs(**kwargs)


__all__ = [
    "SchedulingCandidate",
    "SchedulingDecision",
    "SchedulingPolicy",
    "select_jobs_with_optional_time",
]
