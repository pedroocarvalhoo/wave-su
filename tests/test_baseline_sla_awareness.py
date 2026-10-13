"""SLA retrofit for the baseline scheduling policies.

The baselines predate queue start deadlines, so they ordered jobs purely by
arrival, priority, or predicted runtime.  Against a system that declares
``sla_policy: enforce`` that made them structurally unable to meet deadlines
they were nonetheless measured against.  These tests pin the retrofit: deadline
pressure leads every baseline's ordering under ``enforce``, and ``report_only``
still reproduces the original SLA-blind behaviour.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from vessim.wave.cluster import JobQueue
from vessim.wave.resource_manager import ResourceManager
from vessim.wave.scheduler import JobQueueScheduler
from vessim.wave.scheduler.policies import (
    EFT_DVFSPolicy,
    EFTPolicy,
    FIFOPolicy,
    MultiQueuePolicy,
    PriorityPolicy,
    SJFPolicy,
)
from vessim.wave.scheduler.queues import QueueDefinition


NODE_CONFIG = {
    "test_cluster": {
        "idle_power_w": 10.0,
        "max_power_w": 100.0,
        "memory_bandwidth_gb_s": 100.0,
    }
}


def _with_deadline(job, *, entered, max_wait_s, queue_name="short"):
    """Give a job the queue metadata `queue_deadline_s` reads."""
    job.queue_name = queue_name
    job.queue_enter_time = entered
    job.queue_max_wait_s = max_wait_s
    return job


def test_eft_dvfs_dispatches_deadline_critical_job_before_shorter_work(
    make_job,
    sim_time,
    dvfs_policy,
):
    # EFT-DVFS is shortest-first, so without SLA ordering the long job is
    # overtaken forever. This is the baseline's main miss generator.
    quick = make_job(job_id=1, num_instructions=1e9, num_nodes=1)
    slow_but_due = make_job(job_id=2, num_instructions=5e10, num_nodes=1)
    _with_deadline(quick, entered=sim_time, max_wait_s=48 * 3600)
    _with_deadline(
        slow_but_due,
        entered=sim_time - timedelta(hours=47),
        max_wait_s=48 * 3600,
    )

    policy = EFT_DVFSPolicy(cluster_mem_bw_gb_s=100.0)
    decisions = policy.select_jobs(
        pending_jobs=[quick, slow_but_due],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [2]


def test_eft_dvfs_report_only_keeps_shortest_first_ordering(
    make_job,
    sim_time,
    dvfs_policy,
):
    quick = make_job(job_id=1, num_instructions=1e9, num_nodes=1)
    slow_but_due = make_job(job_id=2, num_instructions=5e10, num_nodes=1)
    _with_deadline(quick, entered=sim_time, max_wait_s=48 * 3600)
    _with_deadline(
        slow_but_due,
        entered=sim_time - timedelta(hours=47),
        max_wait_s=48 * 3600,
    )

    policy = EFT_DVFSPolicy(cluster_mem_bw_gb_s=100.0)
    policy.set_sla_policy("report_only")
    decisions = policy.select_jobs(
        pending_jobs=[quick, slow_but_due],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [1]


def test_fifo_lets_a_later_arrival_with_tighter_deadline_go_first(
    make_job,
    sim_time,
    dvfs_policy,
):
    # Same queue means same max_wait, so arrival order already tracks deadline
    # order; deadlines only reorder FIFO across queues with different windows.
    early_arrival_slack_queue = make_job(job_id=1, num_nodes=1)
    late_arrival_tight_queue = make_job(job_id=2, num_nodes=1)
    late_arrival_tight_queue.arrival_time = sim_time + timedelta(hours=1)
    _with_deadline(
        early_arrival_slack_queue,
        entered=sim_time,
        max_wait_s=14 * 24 * 3600,
        queue_name="long",
    )
    _with_deadline(
        late_arrival_tight_queue,
        entered=sim_time,
        max_wait_s=2 * 3600,
        queue_name="short",
    )

    decisions = FIFOPolicy().select_jobs(
        pending_jobs=[early_arrival_slack_queue, late_arrival_tight_queue],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [2]


def test_already_missed_job_outranks_a_job_still_inside_its_window(
    make_job,
    sim_time,
    dvfs_policy,
):
    # Missed jobs go first so the queue drains instead of accumulating misses,
    # matching how MultiQueuePolicy has always bucketed them.
    missed = make_job(job_id=1, num_nodes=1, priority=1)
    on_time = make_job(job_id=2, num_nodes=1, priority=3)
    _with_deadline(missed, entered=sim_time - timedelta(hours=49), max_wait_s=48 * 3600)
    _with_deadline(on_time, entered=sim_time, max_wait_s=48 * 3600)

    decisions = PriorityPolicy().select_jobs(
        pending_jobs=[on_time, missed],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [1]


@pytest.mark.parametrize(
    "policy_factory",
    [
        lambda: FIFOPolicy(),
        lambda: PriorityPolicy(),
        lambda: SJFPolicy(cluster_mem_bw_gb_s=100.0),
        lambda: EFTPolicy(cluster_mem_bw_gb_s=100.0),
        lambda: EFT_DVFSPolicy(cluster_mem_bw_gb_s=100.0),
    ],
)
def test_jobs_without_a_queue_deadline_keep_the_original_ordering(
    policy_factory,
    make_job,
    sim_time,
    dvfs_policy,
):
    # No max_wait_s means no deadline, so every job gets the same neutral SLA
    # prefix and the policy's own key decides — unchanged pre-SLA behaviour.
    first = make_job(job_id=1, num_nodes=1, priority=9, num_instructions=1e9)
    second = make_job(job_id=2, num_nodes=1, priority=1, num_instructions=1e9)
    second.arrival_time = sim_time + timedelta(seconds=1)

    decisions = policy_factory().select_jobs(
        pending_jobs=[first, second],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        current_time=sim_time,
    )

    assert [decision.job.job_id for decision in decisions] == [1]


def test_multi_queue_policy_propagates_sla_setting_to_its_local_policies():
    definitions = [QueueDefinition("short", 0.0, None, 48 * 3600, priority_weight=1)]
    default_policy = EFT_DVFSPolicy(cluster_mem_bw_gb_s=100.0)
    queue_policy = FIFOPolicy()

    wrapper = MultiQueuePolicy(
        definitions,
        default_policy=default_policy,
        queue_policies={"short": queue_policy},
        sla_policy="report_only",
    )

    assert default_policy.sla_policy == "report_only"
    assert queue_policy.sla_policy == "report_only"

    wrapper.set_sla_policy("enforce")

    assert default_policy.sla_policy == "enforce"
    assert queue_policy.sla_policy == "enforce"


def test_scheduler_propagates_sla_policy_to_a_policy_without_queue_definitions(
    dvfs_policy,
    make_node,
):
    # With no queue definitions there is no MultiQueuePolicy wrapper to carry
    # the system setting, so the scheduler must hand it to the bare policy.
    policy = EFT_DVFSPolicy(cluster_mem_bw_gb_s=100.0)

    JobQueueScheduler(
        resource_manager=ResourceManager([make_node("n0")]),
        job_queue=JobQueue(),
        cluster_type="test_cluster",
        dvfs_policy=dvfs_policy,
        node_config=NODE_CONFIG,
        scheduling_policy=policy,
        sla_policy="report_only",
        auto_stop=False,
        adaptive_step=False,
        debug_logging_enabled=False,
    )

    assert policy.sla_policy == "report_only"


def test_set_sla_policy_rejects_an_unknown_setting():
    with pytest.raises(ValueError, match="sla_policy"):
        FIFOPolicy().set_sla_policy("best_effort")
