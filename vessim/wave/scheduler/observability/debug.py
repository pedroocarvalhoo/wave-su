"""
Scheduler debug-event logging.

   This module defines the event schema and CSV export for step-level scheduler
   debug traces. Use it when adding fields that explain scheduler/resource
   manager decisions; avoid changing scheduling behavior here.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.queues import (
    format_pending_counts_by_queue,
    queue_deadline_s,
    queue_wait_time_s,
)

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job


DEBUG_EVENT_COLUMNS = [
    "Time",
    "Cluster",
    "Event",
    "Reason",
    "Step_Size_s",
    "Next_Step_s",
    "Energy_Budget_W",
    "Power_Before_W",
    "Power_After_W",
    "Power_Delta_W",
    "Pending_Jobs",
    "Queue_Pending_Counts",
    "Queue_Oldest_Wait_s",
    "Queue_Min_Deadline_s",
    "Queue_Missed_Deadlines",
    "Running_Jobs",
    "Busy_Nodes",
    "Idle_Nodes",
    "Off_Nodes",
    "Down_Nodes",
    "Job_ID",
    "Queue_Name",
    "Queue_Classification_Runtime_s",
    "Queue_Max_Wait_s",
    "Queue_Wait_Time_s",
    "Queue_Deadline_Missed",
    "Priority",
    "Nodes",
    "Assigned_Nodes",
    "Frequency_Before_GHz",
    "Frequency_After_GHz",
    "Frequency_Changes",
    "Remaining_Time_Before_s",
    "Remaining_Time_After_s",
    "Remaining_Compute_Cycles",
    "Remaining_Runtime_Overhead_s",
    "Predicted_Exec_Time_s",
    "Actual_Model_Exec_Time_s",
    "Prediction_Error_Factor",
    "Runtime_Error_Factor",
    "Real_Exec_Time_s",
    "Wait_Time_s",
    "Contention_Overhead_s",
    "Current_Step_Progress_s",
    "Contention_Factor",
    "Notes",
]


def _unique_running_job_count(scheduler) -> int:
    return sum(
        1
        for allocation in scheduler.rm.get_running_allocations()
        if allocation.job.status == "running"
    )


def _format_queue_map(values: dict[str, float | int]) -> str:
    parts = []
    for queue_name in sorted(values):
        value = values[queue_name]
        if isinstance(value, int):
            formatted = str(value)
        else:
            formatted = f"{value:.2f}"
        parts.append(f"{queue_name}={formatted}")
    return ";".join(parts)


def _queue_pressure_fields(pending_jobs: list[Job], time: datetime) -> dict[str, str]:
    oldest_wait_s: dict[str, float] = {}
    min_deadline_s: dict[str, float] = {}
    missed_deadlines: dict[str, int] = {}

    for job in pending_jobs:
        queue_name = getattr(job, "queue_name", None) or "unclassified"
        wait_s = queue_wait_time_s(job, time)
        oldest_wait_s[queue_name] = max(oldest_wait_s.get(queue_name, 0.0), wait_s)

        deadline_s = queue_deadline_s(job, time)
        if deadline_s is not None:
            min_deadline_s[queue_name] = min(
                min_deadline_s.get(queue_name, deadline_s),
                deadline_s,
            )

        missed = bool(getattr(job, "queue_deadline_missed", False))
        if deadline_s is not None and deadline_s <= 0:
            missed = True
        missed_deadlines[queue_name] = missed_deadlines.get(queue_name, 0) + int(missed)

    return {
        "Queue_Oldest_Wait_s": _format_queue_map(oldest_wait_s),
        "Queue_Min_Deadline_s": _format_queue_map(min_deadline_s),
        "Queue_Missed_Deadlines": _format_queue_map(missed_deadlines),
    }


def log_scheduler_event(
    scheduler,
    time: datetime,
    event: str,
    job: Optional[Job] = None,
    **fields,
) -> None:
    if not scheduler.debug_logging_enabled:
        return

    stats = scheduler.rm.get_stats()
    pending_jobs = scheduler._get_pending_jobs(time)
    queue_pressure_fields = _queue_pressure_fields(pending_jobs, time)
    row = {column: "" for column in DEBUG_EVENT_COLUMNS}
    row.update(
        {
            "Time": time.isoformat(sep=" "),
            "Cluster": scheduler.cluster_type,
            "Event": event,
            "Energy_Budget_W": (
                scheduler.energy_budget if scheduler.energy_budget is not None else ""
            ),
            "Pending_Jobs": len(pending_jobs),
            "Queue_Pending_Counts": format_pending_counts_by_queue(pending_jobs),
            **queue_pressure_fields,
            "Running_Jobs": _unique_running_job_count(scheduler),
            "Busy_Nodes": stats.get("busy_nodes", ""),
            "Idle_Nodes": stats.get("idle_nodes", ""),
            "Off_Nodes": stats.get("off_nodes", ""),
            "Down_Nodes": stats.get("down_nodes", ""),
        }
    )

    if job is not None:
        frequency_history = getattr(job, "frequency_history", [])
        queue_wait = ""
        if getattr(job, "queue_enter_time", None) is not None:
            queue_wait = queue_wait_time_s(job, time)
        row.update(
            {
                "Job_ID": job.job_id,
                "Queue_Name": getattr(job, "queue_name", "") or "",
                "Queue_Classification_Runtime_s": (
                    getattr(job, "queue_classification_runtime_s", "") or ""
                ),
                "Queue_Max_Wait_s": getattr(job, "queue_max_wait_s", "") or "",
                "Queue_Wait_Time_s": queue_wait,
                "Queue_Deadline_Missed": getattr(job, "queue_deadline_missed", ""),
                "Priority": getattr(job, "priority", ""),
                "Nodes": job.num_nodes,
                "Assigned_Nodes": job.assigned_subcluster or "",
                "Frequency_After_GHz": job.assigned_frequency or "",
                "Frequency_Changes": max(0, len(frequency_history) - 1),
                "Remaining_Time_After_s": job.remaining_time,
                "Remaining_Compute_Cycles": job.remaining_compute_cycles,
                "Remaining_Runtime_Overhead_s": job.remaining_runtime_overhead,
                "Predicted_Exec_Time_s": job.predicted_t_exec,
                "Actual_Model_Exec_Time_s": job.actual_t_exec,
                "Prediction_Error_Factor": getattr(job, "prediction_error_factor", ""),
                "Runtime_Error_Factor": getattr(
                    job,
                    "runtime_error_factor",
                    getattr(job, "prediction_error_factor", ""),
                ),
            }
        )

    for key, value in fields.items():
        if key in row:
            row[key] = value

    if row["Power_Before_W"] != "" and row["Power_After_W"] != "":
        row["Power_Delta_W"] = row["Power_After_W"] - row["Power_Before_W"]

    scheduler.debug_events.append(row)


def export_scheduler_debug_events(scheduler) -> None:
    if not scheduler.debug_logging_enabled or not scheduler.debug_events:
        return

    out_file = (
        os.path.join(
            scheduler.out_dir,
            f"scheduler_debug_events_{scheduler.cluster_type}.csv",
        )
        if scheduler.out_dir
        else f"scheduler_debug_events_{scheduler.cluster_type}.csv"
    )
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DEBUG_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(scheduler.debug_events)

    print(f"[{datetime.now()}][SCHEDULER] Exported Scheduler Debug Events to {out_file}")
