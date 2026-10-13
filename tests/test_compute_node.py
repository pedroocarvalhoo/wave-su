from __future__ import annotations


def test_compute_node_start_and_complete_job_updates_physical_state(
    make_job,
    make_node,
    sim_time,
):
    node = make_node("n0", max_p=100.0, idle_p=10.0, base_frequency_ghz=2.0)
    job = make_job()

    node.start_job(job, sim_time, frequency=2.0, dynamic_power_w=100.0)

    assert node.is_busy()
    assert node.current_job is job
    assert node.current_frequency == 2.0
    assert node.measure() == 110.0

    completed_job = node.complete_job(sim_time)

    assert completed_job is job
    assert node.is_idle()
    assert node.current_job is None
    assert node.current_frequency == 1.0
    assert node.measure() == 10.0
    assert node.total_jobs_executed == 1


def test_compute_node_failure_and_recovery_updates_physical_state(
    make_job,
    make_node,
    sim_time,
):
    node = make_node("n0", max_p=100.0, idle_p=10.0, base_frequency_ghz=2.0)
    job = make_job()
    node.start_job(job, sim_time, frequency=2.0, dynamic_power_w=100.0)

    failed_job = node.fail_job(sim_time, downtime=120.0)

    assert failed_job is job
    assert node.is_down()
    assert node.current_job is None
    assert node.measure() == 0.0
    assert node.total_failures == 1
    assert node.update_downtime(60.0, sim_time) is False

    assert node.update_downtime(60.0, sim_time) is True
    assert node.is_idle()
    assert node.current_frequency == 1.0
    assert node.measure() == 10.0


def test_compute_node_power_off_and_on_only_affects_idle_physical_state(make_node, sim_time):
    node = make_node("n0", idle_p=10.0, base_frequency_ghz=2.0)

    node.power_off(sim_time)
    assert node.status == "OFF"
    assert node.measure() == 0.0
    assert node.current_frequency == 0

    node.power_on(sim_time)
    assert node.is_idle()
    assert node.current_frequency == 1.0
    assert node.measure() == 10.0
