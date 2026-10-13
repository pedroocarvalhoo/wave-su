"""
Failure behavior for the WAVE resource manager.

   Failure events are returned to the scheduler so it can update metrics and
   timeline output without owning node-failure mechanics. `FailureMixin`
   implements node failure, job failure, and down-node recovery methods for
   `ResourceManager`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random

from vessim.wave.cluster.job import Job


@dataclass
class NodeFailureEvent:
    job: Job
    selected_node: str
    failed_nodes: list[str]


class FailureMixin:
    # ResourceManager methods that own failure and recovery state.

    def fail_job(
        self,
        job_id: int,
        node_downtimes: dict[str, float],
        time,
    ) -> list[str]:
        failed_nodes = []
        for node_name, downtime in node_downtimes.items():
            node = self.nodes[node_name]
            if not node.is_busy() or self.node_to_job_id.get(node_name) != job_id:
                continue
            node.fail_job(time, downtime)
            self._record_node_failure(node_name)
            failed_nodes.append(node_name)
        return failed_nodes

    def inject_node_failure(
        self,
        time,
        step_size_seconds: int,
        base_step_seconds: int,
        node_failure_rate: float,
        node_downtime_range: tuple[float, float],
        rng: random.Random,
    ) -> Optional[NodeFailureEvent]:
        if node_failure_rate <= 0.0:
            return None

        if rng.random() >= _step_failure_probability(
            node_failure_rate,
            step_size_seconds,
            base_step_seconds,
        ):
            return None

        running = self._running_node_jobs()
        if not running:
            return None

        selected_node, job = rng.choice(running)
        job_node_names = self.get_job_node_names(job) or [selected_node]
        node_downtimes = {
            node_name: rng.uniform(*node_downtime_range)
            for node_name in job_node_names
            if self.nodes[node_name].is_busy()
        }
        failed_nodes = self.fail_job(job.job_id, node_downtimes, time)
        if not failed_nodes or job.status != "running":
            return None

        job.mark_failed()
        return NodeFailureEvent(
            job=job,
            selected_node=selected_node,
            failed_nodes=failed_nodes,
        )

    def update_down_nodes(self, step_size: float, time=None) -> list[str]:
        recovered = []
        for node_name in list(self._down_nodes):
            node = self.nodes[node_name]
            if node.update_downtime(step_size, time):
                self._record_node_recovery(node_name)
                recovered.append(node_name)
        return recovered

    def _record_node_failure(self, node_name: str) -> None:
        self._remove_node_from_allocation(node_name)
        self._busy_nodes.discard(node_name)
        self._idle_nodes.discard(node_name)
        self._down_nodes.add(node_name)

    def _record_node_recovery(self, node_name: str) -> None:
        self._down_nodes.discard(node_name)
        self._idle_nodes.add(node_name)

    def _running_node_jobs(self) -> list[tuple[str, Job]]:
        running: list[tuple[str, Job]] = []
        for allocation in self.get_running_allocations():
            job = allocation.job
            if job.status != "running":
                continue
            for node_name in allocation.node_names:
                if self.nodes[node_name].is_busy():
                    running.append((node_name, job))
        return running


def _step_failure_probability(
    node_failure_rate: float,
    step_size_seconds: int,
    base_step_seconds: int,
) -> float:
    return 1.0 - (1.0 - node_failure_rate) ** (
        step_size_seconds / base_step_seconds
    )


__all__ = ["FailureMixin", "NodeFailureEvent"]
