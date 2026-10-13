"""
Execution-time and work models for WAVE jobs.

   This module separates scheduler-visible runtime predictions from simulated
   actual work. Edit it when changing CPI/instruction/memory/network formulas or
   prediction-error sampling; keep per-step progress updates in `progress.py`.
"""

from dataclasses import dataclass
import random

from vessim.wave.cluster.job_queue import Job


@dataclass(frozen=True)
class ExecutionTimeComponents:
    compute_time: float
    memory_time: float
    network_time: float
    total_time: float


@dataclass(frozen=True)
class ActualExecutionWork:
    compute_cycles: float
    runtime_overhead: float
    total_time: float
    runtime_error_factor: float = 1.0


def compute_network_overhead(num_assigned_nodes: int) -> float:
    if num_assigned_nodes > 1:
        return 0.1 * num_assigned_nodes  # 100 ms penalty per allocated node.
    return 0.0


def compute_runtime_components(
    job: Job,
    frequency: float,
    num_assigned_nodes: int,
    cluster_mem_bw_gb_s: float = 100.0,
) -> ExecutionTimeComponents:
    memory_time = job.memory_gb / cluster_mem_bw_gb_s
    compute_time = (job.cpi * job.num_instructions) / (frequency * 1e9)
    network_time = compute_network_overhead(num_assigned_nodes)
    total_time = compute_time + memory_time + network_time
    return ExecutionTimeComponents(compute_time, memory_time, network_time, total_time)


def predict_execution_time(
    job: Job,
    frequency: float,
    num_assigned_nodes: int,
    cluster_mem_bw_gb_s: float = 100.0,
    prediction_error_rate: float = 0.0,
    prediction_seed: int = 0,
) -> float:
    # Return the scheduler's runtime estimate, including forecast error.
    ideal_time = compute_runtime_components(
        job,
        frequency,
        num_assigned_nodes,
        cluster_mem_bw_gb_s,
    ).total_time
    prediction_error_factor = sample_prediction_error_factor(
        job_id=job.job_id,
        error_rate=prediction_error_rate,
        seed=prediction_seed,
    )
    return ideal_time * prediction_error_factor


def sample_prediction_error_factor(
    job_id: int,
    error_rate: float = 0.0,
    seed: int = 0,
) -> float:
    if error_rate < 0:
        raise ValueError("error_rate must be non-negative")

    if error_rate == 0.0:
        return 1.0

    rng = random.Random(seed + job_id)
    return max(0.0, 1.0 + rng.uniform(-error_rate, error_rate))


def compute_actual_execution_work(
    job: Job,
    frequency: float,
    num_assigned_nodes: int,
    cluster_mem_bw_gb_s: float = 100.0,
) -> ActualExecutionWork:
    # Return physical work from dataset fields without forecast error.
    components = compute_runtime_components(
        job,
        frequency,
        num_assigned_nodes,
        cluster_mem_bw_gb_s,
    )

    compute_cycles = max(0.0, job.cpi * job.num_instructions)
    runtime_overhead = max(0.0, components.memory_time + components.network_time)
    total_time = (compute_cycles / (frequency * 1e9)) + runtime_overhead

    return ActualExecutionWork(
        compute_cycles=compute_cycles,
        runtime_overhead=runtime_overhead,
        total_time=total_time,
        runtime_error_factor=1.0,
    )
