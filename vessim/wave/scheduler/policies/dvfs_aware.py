# DVFS-aware scheduling policies.

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.scheduler.policies.base import (
    SchedulingCandidate,
    SchedulingDecision,
    SchedulingPolicy,
)

if TYPE_CHECKING:
    from vessim.wave.cluster.job_queue import Job
    from vessim.wave.energy.dvfs import DVFSPolicy


class EFT_DVFSPolicy(SchedulingPolicy):
    """
    Energy-aware Earliest Finish Time policy using DVFS.

       For each job, the policy evaluates available DVFS frequencies and scores
       each valid option with a weighted execution-time and energy objective.

       The frequency choice stays purely energy/time driven; start-deadline
       pressure enters only in the dispatch order, ahead of predicted time.
       Without it this policy is actively SLA-hostile — it is shortest-first,
       so a job aging toward its deadline is overtaken by every shorter job
       that arrives behind it.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, cluster_mem_bw_gb_s: float = 100.0):
        self.alpha = alpha
        self.beta = beta
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
        if not pending_jobs or not ready_nodes:
            return []

        best_per_job: list[
            tuple[tuple[float, float, float, int, int, int], SchedulingCandidate]
        ] = []
        candidate_jobs = [
            job for job in pending_jobs
            if job.num_nodes <= len(ready_nodes)
        ]
        freq_power = {
            freq: dvfs_policy.power_at_frequency(freq)
            for freq in dvfs_policy.available_frequencies
        }
        exec_time_cache: dict[tuple[int, float, float, float, float, int, float], float] = {}

        for job in candidate_jobs:
            options: list[tuple[float, float, float, float]] = []

            for freq, dyn_power in freq_power.items():
                if energy_budget is not None:
                    projected = current_power + dyn_power * job.num_nodes
                    if projected > energy_budget:
                        continue

                exec_time_key = (
                    job.job_id,
                    job.cpi,
                    job.num_instructions,
                    job.memory_gb,
                    freq,
                    job.num_nodes,
                    self.cluster_mem_bw_gb_s,
                )
                predicted_time = exec_time_cache.get(exec_time_key)
                if predicted_time is None:
                    predicted_time = self._predict_execution_time(
                        job=job,
                        frequency=freq,
                        num_assigned_nodes=job.num_nodes,
                        cluster_mem_bw_gb_s=self.cluster_mem_bw_gb_s,
                    )
                    exec_time_cache[exec_time_key] = predicted_time

                energy = (dyn_power + node_idle_p) * predicted_time
                options.append((freq, predicted_time, dyn_power, energy))

            if not options:
                continue

            times = [o[1] for o in options]
            energies = [o[3] for o in options]

            t_min, t_max = min(times), max(times)
            e_min, e_max = min(energies), max(energies)

            t_range = t_max - t_min if t_max > t_min else 1.0
            e_range = e_max - e_min if e_max > e_min else 1.0

            best_score = float("inf")
            best_option = None
            sla_prefix = self._sla_sort_prefix(job, current_time)

            for freq, predicted_time, dyn_power, energy in options:
                t_norm = (predicted_time - t_min) / t_range
                e_norm = (energy - e_min) / e_range
                score = self.alpha * t_norm + self.beta * e_norm

                if score < best_score:
                    best_score = score
                    best_option = (
                        (
                            *sla_prefix,
                            predicted_time,
                            -job.priority,
                            job.num_nodes,
                            job.job_id,
                        ),
                        SchedulingCandidate(
                            job=job,
                            frequency=freq,
                            dynamic_power_w=dyn_power,
                        ),
                    )

            if best_option:
                best_per_job.append(best_option)

        if not best_per_job:
            return []

        best_per_job.sort(key=lambda c: c[0])
        return self._pack_candidates(
            sorted_candidates=[candidate for _, candidate in best_per_job],
            ready_nodes=ready_nodes,
            energy_budget=energy_budget,
            current_power=current_power,
        )


__all__ = ["EFT_DVFSPolicy"]
