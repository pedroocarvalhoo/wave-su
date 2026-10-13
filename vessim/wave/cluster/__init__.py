"""
Cluster domain entities used by WAVE.

   This package contains workload and cluster-side data models that are not tied
   to a particular scheduling policy or energy budget algorithm. Keep scheduler
   decisions and resource-state transitions in their own packages.
"""

from vessim.wave.cluster.job_queue import Job, JobQueue

__all__ = ["Job", "JobQueue"]
