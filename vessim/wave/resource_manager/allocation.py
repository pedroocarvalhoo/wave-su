"""
Running allocation behavior for the WAVE resource manager.

   `RunningAllocation` is the canonical per-job view of a running job and its
   assigned nodes. `AllocationMixin` implements the ResourceManager methods that
   maintain the running-allocation index and the node-to-job reverse index.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, TYPE_CHECKING

from vessim.wave.cluster.job import Job

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class RunningAllocation:
    job: Job
    node_names: list[str]
    frequency: float


class AllocationMixin:
    # ResourceManager methods that own running-job allocation state.

    def get_idle_nodes(self) -> list[str]:
        return list(self._idle_nodes)

    def get_busy_nodes(self) -> list[str]:
        return list(self._busy_nodes)

    def has_running_jobs(self) -> bool:
        return len(self.running_allocations) > 0

    def get_running_jobs_count(self) -> int:
        return len(self.running_allocations)

    def get_next_job_completion_time(self) -> Optional[int]:
        if not self.running_allocations:
            return None

        running_times = [
            allocation.job.remaining_time
            for allocation in self.running_allocations.values()
            if allocation.job.remaining_time > 0
        ]
        if not running_times:
            return None

        return max(1, math.ceil(min(running_times)))

    def get_running_allocations(self) -> list[RunningAllocation]:
        return list(self.running_allocations.values())

    def get_job_node_names(self, job: Job) -> list[str]:
        allocation = self.running_allocations.get(job.job_id)
        if allocation is not None:
            return list(allocation.node_names)

        if not job.assigned_subcluster:
            return []
        return [
            node_name.strip()
            for node_name in job.assigned_subcluster.split(",")
            if node_name.strip()
        ]

    def start_job(
        self,
        job: Job,
        node_names: list[str],
        time: datetime,
        frequency: Optional[float] = None,
        dynamic_power_w: Optional[float] = None,
    ) -> list[str]:
        started_nodes = []
        for node_name in node_names:
            node = self.nodes[node_name]
            node.start_job(
                job,
                time,
                frequency=frequency,
                dynamic_power_w=dynamic_power_w,
            )
            self._record_job_start(node_name, job)
            started_nodes.append(node_name)
        return started_nodes

    def complete_job(self, job_id: int, time: datetime) -> tuple[Optional[Job], list[str]]:
        allocation = self.running_allocations.get(job_id)
        if allocation is None:
            return None, []

        job = allocation.job
        completed_nodes = []
        for node_name in list(allocation.node_names):
            node = self.nodes[node_name]
            if node.is_busy():
                node.complete_job(time)
            self._record_job_complete(node_name)
            completed_nodes.append(node_name)

        return job, completed_nodes

    def set_job_frequency(
        self,
        job: Job,
        frequency: float,
        dynamic_power_w: float,
        time: datetime,
    ) -> int:
        changed_nodes = 0
        for node_name in self.get_job_node_names(job):
            node = self.nodes[node_name]
            current_job = node.current_job
            if not node.is_busy() or (
                current_job is not job
                and getattr(current_job, "job_id", None) != job.job_id
            ):
                continue

            node.set_frequency(frequency, dynamic_power_w, time=time)
            changed_nodes += 1

        if changed_nodes:
            job.assigned_frequency = frequency
            job.frequency_history.append((time, frequency))
            self._set_running_allocation_frequency(job.job_id, frequency)

        return changed_nodes

    def check_nodes_ready_for_new_job(self, time: datetime, step_size: float) -> list[str]:
        ready_nodes = list(self._idle_nodes)

        for node_name in list(self._busy_nodes):
            node = self.nodes[node_name]
            if node.is_idle():
                self._record_job_complete(node_name)
                if node_name not in ready_nodes:
                    ready_nodes.append(node_name)

        return ready_nodes

    def _record_job_start(self, node_name: str, job: Job) -> None:
        self.node_to_job_id[node_name] = job.job_id
        allocation = self.running_allocations.get(job.job_id)
        if allocation is None:
            allocation = RunningAllocation(
                job=job,
                node_names=[],
                frequency=job.assigned_frequency or 0.0,
            )
            self.running_allocations[job.job_id] = allocation
        if node_name not in allocation.node_names:
            allocation.node_names.append(node_name)

        self._idle_nodes.discard(node_name)
        self._busy_nodes.add(node_name)

    def _record_job_complete(self, node_name: str) -> None:
        self._remove_node_from_allocation(node_name)
        self._busy_nodes.discard(node_name)
        if self.nodes[node_name].is_idle():
            self._idle_nodes.add(node_name)

    def _remove_node_from_allocation(self, node_name: str) -> Optional[int]:
        job_id = self.node_to_job_id.pop(node_name, None)
        if job_id is None:
            return None

        allocation = self.running_allocations.get(job_id)
        if allocation is not None:
            if node_name in allocation.node_names:
                allocation.node_names.remove(node_name)
            if not allocation.node_names:
                self.running_allocations.pop(job_id, None)

        return job_id

    def _set_running_allocation_frequency(self, job_id: int, frequency: float) -> None:
        allocation = self.running_allocations.get(job_id)
        if allocation is not None:
            allocation.frequency = frequency


__all__ = ["AllocationMixin", "RunningAllocation"]
