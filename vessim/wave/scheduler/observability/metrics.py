"""
Completed-job metrics export for WAVE schedulers.

   The metrics exporter converts final job state into CSV rows for analysis and
   plotting. It should not advance simulation state or make scheduling
   decisions.
"""

from __future__ import annotations

import os
from datetime import datetime

from vessim.wave.scheduler.queues import queue_wait_time_s


def export_job_metrics(scheduler) -> None:
    import pandas as pd

    data = []
    for job in scheduler.job_queue.jobs:
        if not (
            job.status == "completed"
            and job.start_time
            and hasattr(job, "completion_time")
            and job.completion_time
        ):
            continue

        wait_time = (job.start_time - job.arrival_time).total_seconds()
        real_time = (job.completion_time - job.start_time).total_seconds()
        predicted_time = job.predicted_t_exec
        actual_model_time = job.actual_t_exec
        prediction_error = actual_model_time - predicted_time
        prediction_error_pct = (
            prediction_error / predicted_time
            if predicted_time > 0
            else 0.0
        )
        stretch_factor = (
            (wait_time + real_time) / predicted_time
            if predicted_time > 0
            else 1.0
        )
        frequency_history = getattr(job, "frequency_history", [])
        initial_frequency = (
            frequency_history[0][1]
            if frequency_history
            else getattr(job, "assigned_frequency", 0.0)
        )
        final_frequency = (
            frequency_history[-1][1]
            if frequency_history
            else getattr(job, "assigned_frequency", 0.0)
        )

        data.append(
            {
                "Job_ID": job.job_id,
                "Queue_Name": getattr(job, "queue_name", "") or "",
                "Queue_Classification_Runtime_s": round(
                    getattr(job, "queue_classification_runtime_s", 0.0) or 0.0,
                    2,
                ),
                "Queue_Max_Wait_s": (
                    round(getattr(job, "queue_max_wait_s", 0.0) or 0.0, 2)
                ),
                "Queue_Wait_Time_s": round(queue_wait_time_s(job, job.start_time), 2),
                "Queue_Deadline_Missed": getattr(job, "queue_deadline_missed", False),
                "Submission_Time": (
                    job.submission_time.isoformat(sep=" ")
                    if getattr(job, "submission_time", None)
                    else ""
                ),
                "Priority": getattr(job, "priority", 2),
                "Nodes": job.num_nodes,
                "Initial_Frequency_GHz": round(initial_frequency or 0.0, 3),
                "Final_Frequency_GHz": round(final_frequency or 0.0, 3),
                "Frequency_Changes": max(0, len(frequency_history) - 1),
                "Wait_Time_s": round(wait_time, 2),
                "Predicted_Exec_Time_s": round(predicted_time, 2),
                "Actual_Model_Exec_Time_s": round(actual_model_time, 2),
                "Actual_Compute_Cycles": round(
                    getattr(job, "actual_compute_cycles", 0.0),
                    2,
                ),
                "Actual_Runtime_Overhead_s": round(
                    getattr(job, "actual_runtime_overhead", 0.0),
                    2,
                ),
                "Real_Exec_Time_s": round(real_time, 2),
                "Prediction_Error_s": round(prediction_error, 2),
                "Prediction_Error_Pct": round(prediction_error_pct, 4),
                "Prediction_Error_Factor": round(
                    getattr(job, "prediction_error_factor", 1.0),
                    4,
                ),
                "Runtime_Error_Factor": round(
                    getattr(
                        job,
                        "runtime_error_factor",
                        getattr(job, "prediction_error_factor", 1.0),
                    ),
                    4,
                ),
                "Stretch_Factor": round(stretch_factor, 3),
                "Contention_Overhead_s": round(real_time - actual_model_time, 2),
                "Energy_Source": getattr(job, "planned_energy_source", "") or "",
            }
        )

    if not data:
        return

    df = pd.DataFrame(data)
    out_file = (
        os.path.join(scheduler.out_dir, f"jobs_metrics_{scheduler.cluster_type}.csv")
        if scheduler.out_dir
        else f"jobs_metrics_{scheduler.cluster_type}.csv"
    )
    df.to_csv(out_file, index=False)
    print(f"[{datetime.now()}][SCHEDULER] Exported Detailed Metrics to {out_file}")
