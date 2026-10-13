from __future__ import annotations

from datetime import timedelta

import pandas as pd

from vessim.wave.cluster import JobQueue
from vessim.wave.resource_manager import ResourceManager
from vessim.wave.scheduler import JobQueueScheduler
from vessim.wave.scheduler.policies import MultiQueuePolicy
from vessim.wave.scheduler.queues import QueueDefinition, classify_job, validate_queue_definitions


NODE_CONFIG = {
    "test_cluster": {
        "idle_power_w": 10.0,
        "max_power_w": 100.0,
        "memory_bandwidth_gb_s": 100.0,
    }
}


def _queue_definitions():
    return [
        QueueDefinition(
            name="short",
            min_runtime_s=0.0,
            max_runtime_s=24 * 3600,
            max_nodes=2,
            max_wait_s=48 * 3600,
            priority_weight=3,
        ),
        QueueDefinition(
            name="medium",
            min_runtime_s=24 * 3600,
            max_runtime_s=7 * 24 * 3600,
            max_nodes=4,
            priority_weight=2,
        ),
        QueueDefinition(
            name="long",
            min_runtime_s=7 * 24 * 3600,
            max_runtime_s=None,
            max_nodes=8,
            priority_weight=1,
        ),
    ]


def _scheduler(*, jobs, nodes, dvfs_policy, queue_definitions=None, scheduling_policy=None):
    queue = JobQueue()
    queue.jobs.extend(jobs)
    return JobQueueScheduler(
        resource_manager=ResourceManager(nodes),
        job_queue=queue,
        cluster_type="test_cluster",
        dvfs_policy=dvfs_policy,
        node_config=NODE_CONFIG,
        scheduling_policy=scheduling_policy,
        queue_definitions=queue_definitions,
        auto_stop=False,
        adaptive_step=False,
        debug_logging_enabled=True,
    )


def test_queue_classification_uses_expected_runtime_boundaries(make_job, sim_time):
    definitions = _queue_definitions()

    short = make_job(job_id=1)
    medium = make_job(job_id=2)
    long = make_job(job_id=3)

    classify_job(short, definitions, runtime_s=(24 * 3600) - 1, queue_enter_time=sim_time)
    classify_job(medium, definitions, runtime_s=24 * 3600, queue_enter_time=sim_time)
    classify_job(long, definitions, runtime_s=7 * 24 * 3600, queue_enter_time=sim_time)

    assert short.queue_name == "short"
    assert medium.queue_name == "medium"
    assert long.queue_name == "long"
    assert short.queue_max_nodes == 2
    assert medium.queue_max_nodes == 4
    assert long.queue_max_nodes == 8


def test_queue_definition_serializes_max_nodes():
    definition = QueueDefinition.from_dict({
        "name": "short",
        "min_runtime_s": 0,
        "max_runtime_s": 3600,
        "max_wait_s": 7200,
        "max_nodes": 32,
        "priority_weight": 3,
    })

    assert definition.max_nodes == 32
    assert definition.to_dict()["max_nodes"] == 32


def test_queue_definition_rejects_non_positive_max_nodes():
    definition = QueueDefinition(
        name="short",
        min_runtime_s=0,
        max_runtime_s=None,
        max_nodes=0,
    )

    try:
        validate_queue_definitions([definition])
    except ValueError as exc:
        assert "max_nodes" in str(exc)
    else:
        raise AssertionError("QueueDefinition accepted max_nodes=0")


def test_scheduler_fails_jobs_exceeding_queue_node_limit(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    job = make_job(
        job_id=1,
        num_nodes=3,
        requested_walltime_s=3600.0,
    )
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node(f"n{i}") for i in range(4)],
        dvfs_policy=dvfs_policy,
        queue_definitions=_queue_definitions(),
    )

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "failed"
    assert job.failure_reason == "requested_nodes_exceeds_queue_limit"
    assert job.queue_name == "short"
    assert job.queue_max_nodes == 2
    assert scheduler.jobs_failed == 1
    assert scheduler.rm.get_running_jobs_count() == 0
    assert scheduler.get_stats()["pending"] == 0
    assert any(
        event["Event"] == "job_failed"
        and event["Reason"] == "requested_nodes_exceeds_queue_limit"
        for event in scheduler.debug_events
    )


def test_scheduler_only_submits_jobs_after_arrival(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    job = make_job(
        job_id=1,
        cpi=1.0,
        num_instructions=100e6,
        memory_gb=0.0,
        requested_walltime_s=3600.0,
    )
    job.arrival_time = sim_time + timedelta(minutes=5)
    job.submission_time = job.arrival_time
    scheduler = _scheduler(
        jobs=[job],
        nodes=[make_node("n0")],
        dvfs_policy=dvfs_policy,
        queue_definitions=_queue_definitions(),
    )

    scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "pending"
    assert job.queue_name is None
    assert scheduler.rm.get_running_jobs_count() == 0

    scheduler.step(job.arrival_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert job.status == "running"
    assert job.queue_name == "short"
    assert job.queue_enter_time == job.arrival_time


def test_multi_queue_policy_prioritizes_missed_sla_over_job_priority(
    make_job,
    sim_time,
    dvfs_policy,
):
    definitions = _queue_definitions()
    short = make_job(job_id=1, num_nodes=1, priority=1)
    long = make_job(job_id=2, num_nodes=1, priority=3)
    classify_job(
        short,
        definitions,
        runtime_s=3600.0,
        queue_enter_time=sim_time - timedelta(hours=49),
    )
    classify_job(
        long,
        definitions,
        runtime_s=8 * 24 * 3600,
        queue_enter_time=sim_time,
    )

    decisions = MultiQueuePolicy(definitions, sla_policy="enforce").select_jobs(
        pending_jobs=[long, short],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [1]


def test_multi_queue_policy_report_only_uses_queue_weight_without_sla_enforcement(
    make_job,
    sim_time,
    dvfs_policy,
):
    definitions = [
        QueueDefinition("short", 0.0, 24 * 3600, 48 * 3600, priority_weight=1),
        QueueDefinition("long", 24 * 3600, None, None, priority_weight=3),
    ]
    short = make_job(job_id=1, num_nodes=1, priority=1)
    long = make_job(job_id=2, num_nodes=1, priority=1)
    classify_job(
        short,
        definitions,
        runtime_s=3600.0,
        queue_enter_time=sim_time - timedelta(hours=49),
    )
    classify_job(
        long,
        definitions,
        runtime_s=48 * 3600,
        queue_enter_time=sim_time,
    )

    decisions = MultiQueuePolicy(definitions, sla_policy="report_only").select_jobs(
        pending_jobs=[short, long],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [2]


def test_job_queue_from_csv_can_assign_arrivals_from_dataset_order(tmp_path, sim_time):
    csv_path = tmp_path / "jobs.csv"
    pd.DataFrame(
        {
            "job_id": [10, 11],
            "CPI": [1.0, 1.0],
            "#I": [1e9, 1e9],
            "#nodes": [1, 1],
            "Memory_GB": [1.0, 1.0],
            "Priority": [2, 2],
            "Arrival_Offset_s": [0.0, 30.0],
            "Requested_Walltime_s": [3600.0, 7200.0],
        }
    ).to_csv(csv_path, index=False)

    queue_definition = QueueDefinition(
        name="short",
        min_runtime_s=0.0,
        max_runtime_s=24 * 3600,
        max_wait_s=48 * 3600,
        max_nodes=32,
        priority_weight=3,
    )
    queue = JobQueue.from_csv(
        filepath=str(csv_path),
        start_time=sim_time,
        queue_definition=queue_definition,
    )

    assert [job.job_id for job in queue.jobs] == [10, 11]
    assert [job.arrival_time for job in queue.jobs] == [
        sim_time,
        sim_time + timedelta(seconds=30),
    ]
    assert [job.requested_walltime_s for job in queue.jobs] == [3600.0, 7200.0]


def test_job_queue_from_csv_can_preassign_queue_metadata(tmp_path, sim_time):
    csv_path = tmp_path / "jobs_short.csv"
    pd.DataFrame(
        {
            "job_id": [10],
            "CPI": [1.0],
            "#I": [1e9],
            "#nodes": [1],
            "Memory_GB": [1.0],
            "Priority": [2],
            "Arrival_Offset_s": [0.0],
            "Requested_Walltime_s": [3600.0],
        }
    ).to_csv(csv_path, index=False)

    queue_definition = QueueDefinition(
        name="short",
        min_runtime_s=0.0,
        max_runtime_s=24 * 3600,
        max_wait_s=48 * 3600,
        max_nodes=32,
        priority_weight=3,
    )
    queue = JobQueue.from_csv(
        filepath=str(csv_path),
        start_time=sim_time,
        queue_definition=queue_definition,
    )

    job = queue.jobs[0]
    assert job.queue_name == "short"
    assert job.queue_enter_time == sim_time
    assert job.queue_max_wait_s == 48 * 3600
    assert job.queue_max_nodes == 32
    assert job.queue_priority_weight == 3
