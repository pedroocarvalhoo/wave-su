"""
Logical queue definitions for WAVE cluster schedulers.

   This module owns queue configuration, validation, and job classification.
   It deliberately avoids resource-manager and scheduling-policy decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job


@dataclass(frozen=True)
class QueueDefinition:
    name: str
    min_runtime_s: float = 0.0
    max_runtime_s: Optional[float] = None
    max_wait_s: Optional[float] = None
    priority_weight: int = 0
    policy: Optional[str] = None
    max_nodes: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "QueueDefinition":
        max_runtime_s = data.get("max_runtime_s")
        max_wait_s = data.get("max_wait_s")
        max_nodes = data.get("max_nodes")
        return cls(
            name=str(data["name"]),
            min_runtime_s=float(data.get("min_runtime_s", 0.0)),
            max_runtime_s=None if max_runtime_s is None else float(max_runtime_s),
            max_wait_s=None if max_wait_s is None else float(max_wait_s),
            priority_weight=int(data.get("priority_weight", 0)),
            policy=data.get("policy"),
            max_nodes=None if max_nodes is None else int(max_nodes),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "min_runtime_s": self.min_runtime_s,
            "max_runtime_s": self.max_runtime_s,
            "max_wait_s": self.max_wait_s,
            "max_nodes": self.max_nodes,
            "priority_weight": self.priority_weight,
            "policy": self.policy,
        }

    def matches_runtime(self, runtime_s: float) -> bool:
        if runtime_s < self.min_runtime_s:
            return False
        if self.max_runtime_s is not None and runtime_s >= self.max_runtime_s:
            return False
        return True


def queue_definitions_from_config(
    queue_config: Optional[list[dict]],
    overrides: Optional[dict] = None,
) -> list[QueueDefinition]:
    if not queue_config:
        return []

    override_map = overrides or {}
    definitions = []
    for raw_definition in queue_config:
        data = dict(raw_definition)
        queue_name = str(data["name"])
        if queue_name in override_map:
            data.update(override_map[queue_name])
        definitions.append(QueueDefinition.from_dict(data))

    validate_queue_definitions(definitions)
    return definitions


def validate_queue_definitions(definitions: list[QueueDefinition]) -> None:
    if not definitions:
        return

    ordered = sorted(definitions, key=lambda q: q.min_runtime_s)
    expected_min = 0.0
    for definition in ordered:
        if definition.min_runtime_s != expected_min:
            raise ValueError(
                "Queue runtime ranges must be contiguous. "
                f"Expected min_runtime_s={expected_min}, got "
                f"{definition.min_runtime_s} for queue '{definition.name}'."
            )
        if (
            definition.max_runtime_s is not None
            and definition.max_runtime_s <= definition.min_runtime_s
        ):
            raise ValueError(
                f"Queue '{definition.name}' max_runtime_s must be greater than min_runtime_s."
            )
        if definition.max_wait_s is not None and definition.max_wait_s < 0:
            raise ValueError(f"Queue '{definition.name}' max_wait_s must be non-negative.")
        if definition.max_nodes is not None and definition.max_nodes <= 0:
            raise ValueError(f"Queue '{definition.name}' max_nodes must be positive.")
        if definition.max_runtime_s is None:
            expected_min = float("inf")
        else:
            expected_min = definition.max_runtime_s

    if ordered[-1].max_runtime_s is not None:
        raise ValueError("Queue runtime ranges must include an open-ended final queue.")


def classify_job(
    job: Job,
    definitions: list[QueueDefinition],
    runtime_s: float,
    queue_enter_time: datetime,
) -> QueueDefinition:
    for definition in definitions:
        if not definition.matches_runtime(runtime_s):
            continue

        job.queue_name = definition.name
        job.queue_classification_runtime_s = runtime_s
        job.queue_enter_time = queue_enter_time
        job.queue_max_wait_s = definition.max_wait_s
        job.queue_max_nodes = definition.max_nodes
        job.queue_priority_weight = definition.priority_weight
        return definition

    raise ValueError(
        f"No queue definition matches runtime {runtime_s:.2f}s for job {job.job_id}."
    )


def queue_wait_time_s(job: Job, current_time: datetime) -> float:
    queue_enter_time = job.queue_enter_time or job.arrival_time
    return max(0.0, (current_time - queue_enter_time).total_seconds())


def queue_deadline_s(job: Job, current_time: datetime) -> Optional[float]:
    if job.queue_max_wait_s is None:
        return None
    return job.queue_max_wait_s - queue_wait_time_s(job, current_time)


def update_deadline_status(job: Job, current_time: datetime) -> None:
    remaining = queue_deadline_s(job, current_time)
    if remaining is not None and remaining <= 0:
        job.queue_deadline_missed = True


def pending_counts_by_queue(jobs: list[Job]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        queue_name = job.queue_name or "unclassified"
        counts[queue_name] = counts.get(queue_name, 0) + 1
    return counts


def format_pending_counts_by_queue(jobs: list[Job]) -> str:
    counts = pending_counts_by_queue(jobs)
    return ";".join(f"{name}={counts[name]}" for name in sorted(counts))


__all__ = [
    "QueueDefinition",
    "classify_job",
    "format_pending_counts_by_queue",
    "pending_counts_by_queue",
    "queue_deadline_s",
    "queue_definitions_from_config",
    "queue_wait_time_s",
    "update_deadline_status",
    "validate_queue_definitions",
]
