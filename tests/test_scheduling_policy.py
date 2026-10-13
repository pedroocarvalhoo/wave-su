from __future__ import annotations

from datetime import timedelta

from vessim.wave.scheduler.policies import EFTPolicy, FIFOPolicy, PriorityPolicy


def test_priority_policy_backfills_when_high_priority_job_needs_too_many_nodes(
    make_job,
    sim_time,
    dvfs_policy,
):
    large_high_priority = make_job(job_id=1, num_nodes=3, priority=3)
    small_low_priority = make_job(job_id=2, num_nodes=1, priority=1)

    decisions = PriorityPolicy().select_jobs(
        pending_jobs=[large_high_priority, small_low_priority],
        ready_nodes=["n0", "n1"],
        dvfs_policy=dvfs_policy,
        energy_budget=None,
    )

    assert [decision.job.job_id for decision in decisions] == [2]
    assert decisions[0].assigned_nodes == ["n0"]


def test_fifo_policy_blocks_when_first_job_cannot_fit(make_job, sim_time, dvfs_policy):
    first = make_job(job_id=1, num_nodes=3, priority=1)
    second = make_job(job_id=2, num_nodes=1, priority=3)
    second.arrival_time = sim_time + timedelta(seconds=1)

    decisions = FIFOPolicy().select_jobs(
        pending_jobs=[first, second],
        ready_nodes=["n0", "n1"],
        dvfs_policy=dvfs_policy,
        energy_budget=None,
    )

    assert decisions == []


def test_eft_policy_schedules_each_job_once_even_with_multiple_frequency_candidates(
    make_job,
    dvfs_policy,
):
    job = make_job(job_id=1, cpi=1.0, num_instructions=2e9, num_nodes=1)

    decisions = EFTPolicy(cluster_mem_bw_gb_s=100.0).select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0"],
        dvfs_policy=dvfs_policy,
        energy_budget=None,
    )

    assert len(decisions) == 1
    assert decisions[0].job is job
    assert decisions[0].frequency == 2.0


def test_eft_policy_candidate_packing_respects_cumulative_budget(
    make_job,
    dvfs_policy,
):
    jobs = [
        make_job(job_id=1, cpi=1.0, num_instructions=2e9, num_nodes=1),
        make_job(job_id=2, cpi=1.0, num_instructions=2e9, num_nodes=1),
    ]

    decisions = EFTPolicy(cluster_mem_bw_gb_s=100.0).select_jobs(
        pending_jobs=jobs,
        ready_nodes=["n0", "n1"],
        dvfs_policy=dvfs_policy,
        energy_budget=150.0,
        current_power=0.0,
    )

    assert len(decisions) == 2
    assert sum(decision.dynamic_power_w for decision in decisions) <= 150.0
    assert {decision.job.job_id for decision in decisions} == {1, 2}
