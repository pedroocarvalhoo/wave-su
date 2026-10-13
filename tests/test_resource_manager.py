from __future__ import annotations

import random

from vessim.wave.resource_manager import ResourceManager


def _running_job(job, sim_time, frequency=2.0, nodes=None):
    nodes = nodes or ["n0"]
    job.start_execution(
        current_time=sim_time,
        assigned_nodes=nodes,
        frequency=frequency,
        predicted_t_exec=10.0,
        actual_work=type(
            "ActualWork",
            (),
            {
                "compute_cycles": 10e9,
                "runtime_overhead": 0.0,
                "total_time": 5.0,
                "runtime_error_factor": 1.0,
            },
        )(),
        prediction_error_factor=1.0,
    )
    return job


def test_start_and_complete_job_updates_canonical_allocation_indexes(
    make_job,
    make_node,
    sim_time,
):
    nodes = [make_node("n0"), make_node("n1")]
    rm = ResourceManager(nodes)
    job = _running_job(make_job(job_id=1, num_nodes=2), sim_time, nodes=["n0", "n1"])

    started_nodes = rm.start_job(
        job,
        ["n0", "n1"],
        sim_time,
        frequency=2.0,
        dynamic_power_w=100.0,
    )

    assert started_nodes == ["n0", "n1"]
    assert rm.get_running_jobs_count() == 1
    assert rm.get_job_node_names(job) == ["n0", "n1"]
    assert rm.node_to_job_id == {"n0": 1, "n1": 1}
    assert set(rm.get_busy_nodes()) == {"n0", "n1"}

    completed_job, completed_nodes = rm.complete_job(job.job_id, sim_time)

    assert completed_job is job
    assert set(completed_nodes) == {"n0", "n1"}
    assert rm.get_running_jobs_count() == 0
    assert rm.node_to_job_id == {}
    assert set(rm.get_idle_nodes()) == {"n0", "n1"}


def test_inject_node_failure_fails_entire_job_allocation(make_job, make_node, sim_time):
    nodes = [make_node("n0"), make_node("n1")]
    rm = ResourceManager(nodes)
    job = _running_job(make_job(job_id=2, num_nodes=2), sim_time, nodes=["n0", "n1"])
    rm.start_job(job, ["n0", "n1"], sim_time, frequency=2.0, dynamic_power_w=100.0)

    failure = rm.inject_node_failure(
        time=sim_time,
        step_size_seconds=60,
        base_step_seconds=60,
        node_failure_rate=1.0,
        node_downtime_range=(120.0, 120.0),
        rng=random.Random(7),
    )

    assert failure is not None
    assert failure.job is job
    assert set(failure.failed_nodes) == {"n0", "n1"}
    assert failure.selected_node in {"n0", "n1"}
    assert job.status == "failed"
    assert rm.get_running_jobs_count() == 0
    assert set(rm.get_down_nodes()) == {"n0", "n1"}


def test_manage_node_power_powers_idle_nodes_down_and_back_on(make_node, sim_time):
    rm = ResourceManager([make_node("n0"), make_node("n1"), make_node("n2")])

    changed = rm.manage_node_power(
        time=sim_time,
        energy_budget=15.0,
        current_power=30.0,
        node_idle_p=10.0,
        has_pending_jobs=False,
    )

    assert changed is True
    assert len(rm.get_off_nodes()) == 2
    assert len(rm.get_idle_nodes()) == 1

    changed = rm.manage_node_power(
        time=sim_time,
        energy_budget=30.0,
        current_power=10.0,
        node_idle_p=10.0,
        has_pending_jobs=True,
    )

    assert changed is True
    assert rm.get_off_nodes() == []
    assert len(rm.get_idle_nodes()) == 3


def test_manage_node_power_caps_power_on_by_pending_demand(make_node, sim_time):
    rm = ResourceManager([make_node(f"n{i}") for i in range(5)])

    # Shed everything to off first (over budget, no pending work).
    rm.manage_node_power(
        time=sim_time,
        energy_budget=0.0,
        current_power=50.0,
        node_idle_p=10.0,
        has_pending_jobs=False,
    )
    assert len(rm.get_off_nodes()) == 5

    # Budget could power all 5 nodes, but pending jobs only demand 2, so
    # only 2 are powered on instead of filling the budget with idle nodes.
    changed = rm.manage_node_power(
        time=sim_time,
        energy_budget=1000.0,
        current_power=0.0,
        node_idle_p=10.0,
        has_pending_jobs=True,
        pending_node_demand=2,
    )

    assert changed is True
    assert len(rm.get_idle_nodes()) == 2
    assert len(rm.get_off_nodes()) == 3


def test_manage_node_power_sheds_idle_nodes_beyond_demand(make_node, sim_time):
    rm = ResourceManager([make_node(f"n{i}") for i in range(5)])  # all idle

    # Well within budget, but pending demand is only 1 node, so the four
    # surplus idle nodes are powered off to stop burning idle power.
    changed = rm.manage_node_power(
        time=sim_time,
        energy_budget=1000.0,
        current_power=50.0,
        node_idle_p=10.0,
        has_pending_jobs=True,
        pending_node_demand=1,
    )

    assert changed is True
    assert len(rm.get_idle_nodes()) == 1
    assert len(rm.get_off_nodes()) == 4


def test_runnable_gate_powers_on_only_for_a_runnable_job(make_node, sim_time):
    rm = ResourceManager(
        [make_node(f"n{i}") for i in range(5)], power_on_gate="runnable"
    )
    # Shed everything off first.
    rm.manage_node_power(
        time=sim_time, energy_budget=0.0, current_power=50.0,
        node_idle_p=10.0, has_pending_jobs=False,
    )
    assert len(rm.get_off_nodes()) == 5

    # Pending demand is 5, but only a single 2-node job is runnable now, so the
    # runnable gate wakes exactly 2 nodes instead of filling toward demand.
    changed = rm.manage_node_power(
        time=sim_time, energy_budget=1000.0, current_power=0.0, node_idle_p=10.0,
        has_pending_jobs=True, pending_node_demand=5, runnable_node_counts=[2],
    )

    assert changed is True
    assert len(rm.get_idle_nodes()) == 2
    assert len(rm.get_off_nodes()) == 3


def test_runnable_gate_keeps_nodes_off_when_nothing_runnable(make_node, sim_time):
    rm = ResourceManager(
        [make_node(f"n{i}") for i in range(3)], power_on_gate="runnable"
    )
    # Jobs are pending but none are dispatchable now (empty runnable list), e.g.
    # a plan withholding them for a greener slot: do not power on, and shed the
    # warm idle pool rather than burning idle power for work that will not run.
    changed = rm.manage_node_power(
        time=sim_time, energy_budget=1000.0, current_power=30.0, node_idle_p=10.0,
        has_pending_jobs=True, pending_node_demand=9, runnable_node_counts=[],
    )

    assert changed is True
    assert len(rm.get_idle_nodes()) == 0
    assert len(rm.get_off_nodes()) == 3


def test_runnable_gate_skips_when_job_larger_than_powerable(make_node, sim_time):
    rm = ResourceManager(
        [make_node(f"n{i}") for i in range(5)], power_on_gate="runnable"
    )
    rm.manage_node_power(
        time=sim_time, energy_budget=0.0, current_power=50.0,
        node_idle_p=10.0, has_pending_jobs=False,
    )
    assert len(rm.get_off_nodes()) == 5

    # Budget headroom can only afford to wake 2 nodes, but the sole runnable job
    # needs 4: no single job fits the nodes we could turn on, so wake nothing.
    changed = rm.manage_node_power(
        time=sim_time, energy_budget=20.0, current_power=0.0, node_idle_p=10.0,
        has_pending_jobs=True, pending_node_demand=4, runnable_node_counts=[4],
    )

    assert changed is False
    assert len(rm.get_idle_nodes()) == 0
    assert len(rm.get_off_nodes()) == 5


def test_eager_gate_matches_legacy_demand_behavior(make_node, sim_time):
    # The eager gate (library default) preserves the pending-demand fill even
    # when runnable counts are supplied, so existing callers are unaffected.
    rm = ResourceManager(
        [make_node(f"n{i}") for i in range(5)], power_on_gate="eager"
    )
    rm.manage_node_power(
        time=sim_time, energy_budget=0.0, current_power=50.0,
        node_idle_p=10.0, has_pending_jobs=False,
    )
    assert len(rm.get_off_nodes()) == 5

    changed = rm.manage_node_power(
        time=sim_time, energy_budget=1000.0, current_power=0.0, node_idle_p=10.0,
        has_pending_jobs=True, pending_node_demand=2, runnable_node_counts=[],
    )

    assert changed is True
    assert len(rm.get_idle_nodes()) == 2
    assert len(rm.get_off_nodes()) == 3


def test_manage_runtime_dvfs_downscales_to_meet_budget(
    make_job,
    make_node,
    sim_time,
    dvfs_policy,
):
    nodes = [make_node("n0"), make_node("n1")]
    rm = ResourceManager(nodes)
    job = _running_job(make_job(job_id=3, num_nodes=2), sim_time, nodes=["n0", "n1"])
    job.remaining_compute_cycles = 4e9
    rm.start_job(job, ["n0", "n1"], sim_time, frequency=2.0, dynamic_power_w=100.0)
    events = []
    invalidations = []

    rm.manage_runtime_dvfs(
        time=sim_time,
        energy_budget=120.0,
        dvfs_policy=dvfs_policy,
        get_current_power=lambda: sum(node.measure() for node in nodes),
        invalidate_power_cache=lambda: invalidations.append(True),
        estimate_remaining_time=lambda j: j.remaining_compute_cycles / (j.assigned_frequency * 1e9),
        has_pending_jobs=False,
        log_event=lambda *args, **kwargs: events.append((args, kwargs)),
        silent=True,
    )

    assert job.assigned_frequency == 1.0
    assert all(node.current_frequency == 1.0 for node in nodes)
    assert sum(node.measure() for node in nodes) == 120.0
    assert invalidations
    assert events[0][0][1] == "runtime_dvfs_downscale"
    assert job.remaining_time == 4.0
