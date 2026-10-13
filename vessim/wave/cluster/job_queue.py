"""
Job queue loading and storage for WAVE simulations.

   `JobQueue` owns the ordered collection of jobs and knows how to load
   workloads from CSV datasets. It should stay close to input parsing and queue
   storage; runtime prediction, scheduling decisions, and resource allocation
   belong to the scheduler and resource manager packages.
"""

from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

from vessim.wave.cluster.job import Job
from vessim.wave.scheduler.queues import QueueDefinition


class JobQueue:

    def __init__(self):
        self.jobs: list[Job] = []

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        start_time: datetime,
        queue_definition: QueueDefinition,
    ):
        queue = cls()
        queue.extend_from_csv(
            filepath=filepath,
            start_time=start_time,
            queue_definition=queue_definition,
        )
        return queue

    def extend_from_csv(
        self,
        filepath: str,
        start_time: datetime,
        queue_definition: QueueDefinition,
    ) -> None:
        df = pd.read_csv(filepath)

        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading jobs"):
            arrival_time = start_time + timedelta(seconds=float(row["Arrival_Offset_s"]))
            job = Job(
                job_id=int(row["job_id"]),
                cpi=row['CPI'],
                num_instructions=row['#I'],
                arrival_time=arrival_time,
                num_nodes=int(row['#nodes']),
                requested_walltime_s=float(row["Requested_Walltime_s"]),
                memory_gb=float(row['Memory_GB']),
                priority=int(row['Priority']),
            )
            if not queue_definition.matches_runtime(job.requested_walltime_s):
                raise ValueError(
                    f"Job {job.job_id} in {filepath} has Requested_Walltime_s="
                    f"{job.requested_walltime_s:.2f}, which does not match queue "
                    f"'{queue_definition.name}'."
                )
            job.queue_name = queue_definition.name
            job.queue_classification_runtime_s = job.requested_walltime_s
            job.queue_enter_time = job.arrival_time
            job.queue_max_wait_s = queue_definition.max_wait_s
            job.queue_max_nodes = queue_definition.max_nodes
            job.queue_priority_weight = queue_definition.priority_weight
            self.jobs.append(job)

    def total_jobs(self) -> int:
        return len(self.jobs)


__all__ = ["Job", "JobQueue"]
