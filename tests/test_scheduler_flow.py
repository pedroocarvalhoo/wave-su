from __future__ import annotations

from datetime import timedelta

import pandas as pd

from vessim.wave.cluster import JobQueue
from vessim.wave.resource_manager import ResourceManager
from vessim.wave.scheduler import JobQueueScheduler
from vessim.wave.scheduler.policies import SchedulingDecision, SchedulingPolicy


NODE_CONFIG = {
    "test_cluster": {
        "idle_power_w": 10.0,
        "max_power_w": 100.0,
        "memory_bandwidth_gb_s": 100.0,
    }
}


class ForcedHighFrequencyPolicy(SchedulingPolicy):
    def select_jobs(
        self,
        pending_jobs,
        ready_nodes,
        dvfs_policy,
        energy_budget=None,
        current_power=0.0,
        node_idle_p=0.0,
    ):
        if not pending_jobs or not ready_nodes:
            return []

        job = pending_jobs[0]
        return [
            SchedulingDecision(
                job=job,
                assigned_nodes=ready_nodes[: job.num_nodes],
                frequency=2.0,
                dynamic_power_w=100.0,
            )
        ]


class FakeEnergyScheduler:
    def __init__(self):
        self.notifications = []

    def notification(self, notification, scheduler):
        self.notifications.append((notification, scheduler))


def _scheduler(
    *,
    jobs,
    nodes,
    dvfs_policy,
    step_size=60,
    auto_stop=False,
    adaptive_step=True,
    scheduling_policy=None,
    node_failure_rate=0.0,
    out_dir="",
):
    queue = JobQueue()
    queue.jobs.extend(jobs)

    return JobQueueScheduler(
        resource_manager=ResourceManager(nodes),
        job_queue=queue,
        cluster_type="test_cluster",
        dvfs_policy=dvfs_policy,
        node_config=NODE_CONFIG,
        scheduling_policy=scheduling_policy,
        step_size=step_size,
        auto_stop=auto_stop,
        adaptive_step=adaptive_step,
        node_failure_rate=node_failure_rate,
        node_downtime_range=(120.0, 120.0),
        debug_logging_enabled=True,
        out_dir=str(out_dir),
    )


def _events(scheduler):
    return [event["Event"] for event in scheduler.debug_events]


def test_scheduler_flow_starts_progresses_completes_and_logs_job(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    job = make_job(job_id=1, cpi=1.0, num_instructions=100e6, memory_gb=1.0)
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node("n0")],
        dvfs_policy=dvfs_policy,
    )

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "running"
    assert job.start_time == sim_time
    assert scheduler.rm.get_running_jobs_count() == 1
    assert scheduler.timeline["n0"] == [(sim_time, None, job.job_id)]
    assert scheduler.step_size_seconds == 1
    assert _events(scheduler) == ["job_started", "step_summary"]

    completion_time = sim_time + timedelta(seconds=scheduler.step_size_seconds)
    scheduler.step(completion_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "completed"
    assert job.completion_time == completion_time
    assert job.remaining_time == 0.0
    assert scheduler.jobs_completed == 1
    assert scheduler.rm.get_running_jobs_count() == 0
    assert scheduler.timeline["n0"] == [(sim_time, completion_time, job.job_id)]
    assert "job_completed" in _events(scheduler)


def test_scheduler_does_not_start_jobs_before_energy_budget_is_assigned(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    job = make_job(job_id=2)
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node("n0")],
        dvfs_policy=dvfs_policy,
    )
    scheduler.energy_scheduler = FakeEnergyScheduler()

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "pending"
    assert scheduler.rm.get_running_jobs_count() == 0
    assert _events(scheduler) == ["step_summary"]


def test_scheduler_auto_stop_powers_off_idle_nodes_and_releases_budget(
    make_node,
    sim_time,
    dvfs_policy,
):
    energy_scheduler = FakeEnergyScheduler()
    scheduler = _scheduler(
        jobs=[],
        nodes=[make_node("n0"), make_node("n1")],
        dvfs_policy=dvfs_policy,
        auto_stop=True,
    )
    scheduler.energy_scheduler = energy_scheduler
    scheduler.set_energy_budget(123.0)

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert scheduler.done is True
    assert set(scheduler.rm.get_off_nodes()) == {"n0", "n1"}
    assert energy_scheduler.notifications == [
        (
            {
                "type": "ClusterComplete",
                "cluster_type": "test_cluster",
                "released_budget": 123.0,
            },
            scheduler,
        )
    ]


def test_scheduler_applies_runtime_dvfs_after_scheduling_to_fit_budget(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    job = make_job(job_id=3, cpi=1.0, num_instructions=4e9, memory_gb=0.0)
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node("n0")],
        dvfs_policy=dvfs_policy,
        scheduling_policy=ForcedHighFrequencyPolicy(),
    )
    scheduler.set_energy_budget(60.0)

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "running"
    assert job.assigned_frequency == 1.0
    assert job.frequency_history == [(sim_time, 2.0), (sim_time, 1.0)]
    assert scheduler.rm.get_node("n0").measure() == 60.0
    assert "runtime_dvfs_downscale" in _events(scheduler)


def test_scheduler_failure_step_marks_job_failed_and_closes_timeline(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    job = make_job(job_id=4, cpi=1.0, num_instructions=1e12, memory_gb=0.0)
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node("n0")],
        dvfs_policy=dvfs_policy,
        node_failure_rate=1.0,
    )

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})
    assert job.status == "running"

    failure_time = sim_time + timedelta(seconds=scheduler.step_size_seconds)
    scheduler.step(failure_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "failed"
    assert scheduler.jobs_failed == 1
    assert scheduler.rm.get_running_jobs_count() == 0
    assert scheduler.rm.get_down_nodes() == ["n0"]
    assert scheduler.timeline["n0"] == [(sim_time, failure_time, job.job_id)]
    assert "job_failed" in _events(scheduler)


def test_scheduler_finalize_exports_debug_events_and_completed_job_metrics(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
    tmp_path,
):
    job = make_job(job_id=5, cpi=1.0, num_instructions=100e6, memory_gb=1.0)
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node("n0")],
        dvfs_policy=dvfs_policy,
        out_dir=tmp_path,
    )

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})
    scheduler.step(
        sim_time + timedelta(seconds=scheduler.step_size_seconds),
        p_delta=0.0,
        e_delta=0.0,
        actor_infos={},
    )
    scheduler.finalize()
    scheduler.finalize()

    debug_file = tmp_path / "scheduler_debug_events_test_cluster.csv"
    metrics_file = tmp_path / "jobs_metrics_test_cluster.csv"

    assert debug_file.exists()
    assert metrics_file.exists()

    debug_events = pd.read_csv(debug_file)
    metrics = pd.read_csv(metrics_file)

    assert set(debug_events["Event"]) >= {"job_started", "job_completed", "step_summary"}
    assert metrics.loc[0, "Job_ID"] == job.job_id
    assert metrics.loc[0, "Frequency_Changes"] == 0
    assert metrics.loc[0, "Runtime_Error_Factor"] == 1.0
