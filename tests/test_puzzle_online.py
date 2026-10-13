"""Online planning: future workload records must not affect current decisions."""

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vessim.power_meter import ComputeNode
from vessim.wave.cluster import Job, JobQueue
from vessim.wave.energy import AdaptiveDVFSPolicy, EnergyScheduler
from vessim.wave.energy.puzzle_budget_policy import PuzzleBudgetPolicy
from vessim.wave.energy.strategy import PuzzlePlanAllocationStrategy
from vessim.wave.resource_manager import ResourceManager
from vessim.wave.scheduler import JobQueueScheduler, SimulationComplete
from vessim.wave.scheduler.policies.puzzle import PuzzleSchedulingPolicy

NOW = datetime(2023, 1, 1, 12)
CLUSTER = "x86_epyc_rome"
CONFIGS = json.loads(
    (Path(__file__).resolve().parents[1] / "datasets/node_config.json").read_text()
)


class ConstantForecast:
    def forecast(self, start_time, end_time, column, frequency, resample_method):
        value = 4000.0 if column == "renewable_gen" else 100.0
        return {
            start_time + i * frequency: value
            for i in range(int((end_time - start_time) / frequency))
        }


def battery():
    return SimpleNamespace(
        _battery_energy=0.0, _battery_energy_max=1000.0,
        _state_of_charge_min=0.0, _state_of_charge_max=1.0,
        _p_discharge_max=0.0, _p_charge_max=0.0, soc=lambda: 0.0,
    )


def job(job_id, arrival=NOW, duration_s=600, num_nodes=1):
    return Job(
        job_id=job_id, cpi=1.0,
        num_instructions=duration_s * CONFIGS[CLUSTER]["base_frequency_ghz"] * 1e9,
        arrival_time=arrival, num_nodes=num_nodes,
        requested_walltime_s=duration_s, memory_gb=0.0,
    )


def setup(jobs, *, cluster=CLUSTER, nodes=2, global_board=True):
    cfg = CONFIGS[cluster]
    queue = JobQueue()
    queue.jobs = jobs
    planner = PuzzleBudgetPolicy(
        node_configs=CONFIGS, slot_duration_s=300,
        renewable_charge_fraction=0.0, tail_renewable_calibration_days=0,
        global_green_board=global_board,
    )
    scheduler = JobQueueScheduler(
        resource_manager=ResourceManager([
            ComputeNode(
                name=f"{cluster}-{i}", max_p=cfg["max_power_w"],
                idle_p=cfg["idle_power_w"],
                base_frequency_ghz=cfg["base_frequency_ghz"],
            ) for i in range(nodes)
        ]),
        job_queue=queue, cluster_type=cluster, node_config=CONFIGS,
        dvfs_policy=AdaptiveDVFSPolicy(
            frequencies={float(f): p for f, p in cfg["frequencies"].items()},
            base_frequency_ghz=cfg["base_frequency_ghz"],
        ),
        scheduling_policy=PuzzleSchedulingPolicy(
            planner, cluster, enable_fallback_dispatch=False,
        ),
        auto_stop=False, debug_logging_enabled=False,
    )
    planner.set_cluster_schedulers([scheduler])
    return planner, scheduler


@pytest.mark.parametrize("global_board", [False, True])
@pytest.mark.parametrize("future_kind", ["short", "long", "invalid"])
def test_future_workload_cannot_change_plan_or_budget(global_board, future_kind):
    known = job(1)
    future = job(2, NOW + timedelta(hours=1),
                 duration_s=30 * 3600 if future_kind == "long" else 600)
    if future_kind == "invalid":
        future.queue_max_nodes = 0
    baseline, _ = setup([deepcopy(known)], global_board=global_board)
    candidate, _ = setup([deepcopy(known), future], global_board=global_board)
    # Also cover a different cluster containing only future demand. Its queue
    # must not divert the green supply through the capacity fallback.
    _, future_cluster = setup(
        [job(1, NOW + timedelta(hours=2))], cluster="arm_a64fx_fujitsu",
    )
    _, empty_cluster = setup([], cluster="arm_a64fx_fujitsu")
    baseline.set_cluster_schedulers(baseline.cluster_schedulers + [empty_cluster])
    candidate.set_cluster_schedulers(candidate.cluster_schedulers + [future_cluster])
    a = baseline.generate_budget(NOW, ConstantForecast(), battery())
    b = candidate.generate_budget(NOW, ConstantForecast(), battery())
    assert a == b
    assert baseline._current_plan == candidate._current_plan
    assert candidate._cluster_budget_weights(
        candidate._build_job_rectangles(NOW), NOW,
    ) == baseline._cluster_budget_weights(baseline._build_job_rectangles(NOW), NOW)
    assert candidate._should_replan(NOW + timedelta(minutes=5)) is False


def test_arrival_boundary_triggers_one_plan_without_waiting_for_cooldown():
    arrival = NOW + timedelta(minutes=5)
    planner, _ = setup([job(1, arrival), job(2, arrival)])
    planner.generate_budget(NOW, ConstantForecast(), battery())
    assert planner._current_plan.placed_jobs == []
    before = arrival - timedelta(microseconds=1)
    assert planner._build_job_rectangles(before) == []
    assert planner._should_replan(before) is False
    assert planner._should_replan(arrival) is True
    planner.generate_budget(arrival, ConstantForecast(), battery())
    assert {r.job_id for r in planner._current_plan.placed_jobs} == {1, 2}
    assert planner.history.count == 2
    planner.generate_budget(arrival, ConstantForecast(), battery())
    assert planner.history.count == 2
    assert planner._should_replan(arrival + timedelta(minutes=5)) is False


def test_unplaceable_and_rejected_arrivals_do_not_cause_replan_loop():
    invalid = job(1)
    invalid.queue_max_nodes = 0
    too_large = job(2, num_nodes=3)
    planner, _ = setup([invalid, too_large])
    planner.generate_budget(NOW, ConstantForecast(), battery())
    assert [r.job_id for r in planner._current_plan.unplaced_jobs] == [2]
    assert planner._should_replan(NOW + timedelta(minutes=5)) is False
    # Genuine dispatch failure requests still obey their own cooldown.
    planner.request_replan()
    assert planner._should_replan(NOW + timedelta(minutes=5)) is False
    assert planner._should_replan(NOW + timedelta(hours=1)) is True


def test_live_submission_replans_and_preserves_running_job():
    planner, scheduler = setup([job(1, duration_s=3600)], nodes=1)
    energy = EnergyScheduler(
        ConstantForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
    )
    energy.step(NOW, 0.0, 0.0, {})
    scheduler.step(NOW, 0.0, 0.0, {})
    running = scheduler.job_queue.jobs[0]
    assert running.status == "running"
    arrival = NOW + timedelta(minutes=5)
    newcomer = job(2, arrival)
    scheduler.job_queue.jobs.append(newcomer)
    energy.step(arrival, 0.0, 0.0, {})
    assert planner._current_plan.plan_start == arrival
    assert running.status == "running"
    rect = next(r for r in planner._current_plan.placed_jobs if r.job_id == 2)
    assert rect.start_slot_idx > 0
    scheduler.step(arrival, 0.0, 0.0, {})
    assert newcomer.status == "pending"
    assert running.status == "running"


def test_staggered_arrivals_execute_without_fallback_or_future_knowledge():
    jobs = [job(i, NOW + timedelta(minutes=5 * i), duration_s=120) for i in range(5)]
    planner, scheduler = setup(jobs)
    energy = EnergyScheduler(
        ConstantForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
        step_size=60,
    )
    for minute in range(40):
        time = NOW + timedelta(minutes=minute)
        energy.step(time, 0.0, 0.0, {})
        assert all(r.arrival_time <= time for r in planner._current_plan.placed_jobs)
        try:
            scheduler.step(time, 0.0, 0.0, {})
        except SimulationComplete:
            break
    assert all(j.status == "completed" for j in jobs)
    assert all(j.start_time == j.arrival_time for j in jobs)
    assert planner.history.count == len(jobs)
    for plan in planner.plan_history:
        assert all(r.arrival_time <= plan.plan_start for r in plan.placed_jobs)


def test_long_job_extends_plan_only_after_submission():
    arrival = NOW + timedelta(minutes=5)
    planner, _ = setup([job(1, arrival, duration_s=30 * 3600)])
    planner.generate_budget(NOW, ConstantForecast(), battery())
    visible_slots = int(planner.plan_horizon_h * 3600 / planner.slot_duration_s)
    assert len(planner._current_plan.slots) == visible_slots
    planner.generate_budget(arrival, ConstantForecast(), battery())
    assert len(planner._current_plan.slots) > visible_slots
    assert [r.job_id for r in planner._current_plan.placed_jobs] == [1]
    assert planner._current_plan.plan_end == arrival + timedelta(hours=24)


def test_arrival_detection_distinguishes_same_job_id_in_different_clusters():
    planner, first = setup([job(1)])
    arrival = NOW + timedelta(minutes=5)
    _, second = setup([job(1, arrival)], cluster="arm_a64fx_fujitsu")
    planner.set_cluster_schedulers([first, second])
    planner.generate_budget(NOW, ConstantForecast(), battery())
    planner.generate_budget(arrival, ConstantForecast(), battery())
    assert planner._current_plan.plan_start == arrival
    assert {(r.cluster_type, r.job_id) for r in planner._current_plan.placed_jobs} == {
        (CLUSTER, 1), ("arm_a64fx_fujitsu", 1),
    }


class GridOnlyForecast(ConstantForecast):
    def forecast(self, start_time, end_time, column, frequency, resample_method):
        value = 0.0 if column == "renewable_gen" else 100.0
        return {
            start_time + i * frequency: value
            for i in range(int((end_time - start_time) / frequency))
        }


@pytest.mark.parametrize("carbon_aware", [False, True])
@pytest.mark.parametrize("global_board", [False, True])
def test_grid_only_dispatch_keeps_headroom_above_always_on_idle_power(
    carbon_aware, global_board,
):
    submitted = job(1, duration_s=120)
    planner, scheduler = setup([submitted], nodes=10, global_board=global_board)
    planner.grid_power_limit = 100_000.0
    planner.carbon_intensity_threshold_g_per_kwh = 2000.0
    planner.carbon_aware_placement = carbon_aware
    planner.pue = 1.3
    scheduler.rm.keep_idle_nodes_on = True
    energy = EnergyScheduler(
        GridOnlyForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
        pue=planner.pue,
    )
    energy.step(NOW, 0.0, 0.0, {})
    rect = planner._current_plan.placed_jobs[0]
    assert rect.used_grid
    assert rect.start_slot_idx == 0
    required = (
        scheduler.get_current_power()
        + scheduler.dvfs_policy.power_at_frequency(rect.frequency_ghz)
    )
    assert scheduler.energy_budget == pytest.approx(required)
    assert energy.last_budget_info["grid_power_w"] == pytest.approx(required * 1.3)
    expected_idle = 10 * CONFIGS[CLUSTER]["idle_power_w"] * planner.pue
    assert planner.current_plan.slots[0].idle_reserved_w == pytest.approx(expected_idle)
    scheduler.step(NOW, 0.0, 0.0, {})
    assert submitted.status == "running"

    # Replanning a running job must count its node idle exactly once as well.
    planner.replan_request_cooldown_h = 0.0
    planner.request_replan()
    energy.step(NOW + timedelta(minutes=1), 0.0, 0.0, {})
    assert scheduler.energy_budget == pytest.approx(required)
    assert planner.current_plan.slots[0].idle_reserved_w == pytest.approx(expected_idle)
    for minute in range(1, 11):
        time = NOW + timedelta(minutes=minute)
        try:
            energy.step(time, 0.0, 0.0, {})
            scheduler.step(time, 0.0, 0.0, {})
        except SimulationComplete:
            break
        if submitted.status == "completed":
            break
    assert submitted.status == "completed"
    assert scheduler.rm.get_powered_on_node_count() == 10
    planner.request_replan()
    info = planner.generate_budget(time + timedelta(minutes=1), GridOnlyForecast(), battery())
    assert info.total_budget_w == pytest.approx(10 * CONFIGS[CLUSTER]["idle_power_w"] * 1.3)


@pytest.mark.parametrize("global_board", [False, True])
def test_grid_dispatch_budget_also_covers_finished_partition_idle(global_board):
    submitted = job(1)
    planner, active = setup([submitted], nodes=10, global_board=global_board)
    _, finished = setup([], nodes=4, cluster="arm_a64fx_fujitsu")
    finished.done = True
    for scheduler in (active, finished):
        scheduler.rm.keep_idle_nodes_on = True
    planner.set_cluster_schedulers([active, finished])
    planner.grid_power_limit = 100_000.0
    planner.carbon_intensity_threshold_g_per_kwh = 2000.0
    planner.pue = 1.3
    energy = EnergyScheduler(
        GridOnlyForecast(), battery(), [active, finished], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
        pue=planner.pue,
    )
    energy.step(NOW, 0.0, 0.0, {})
    rect = planner.current_plan.placed_jobs[0]
    required = active.get_current_power() + active.dvfs_policy.power_at_frequency(rect.frequency_ghz)
    assert active.energy_budget == pytest.approx(required)
    assert finished.energy_budget == pytest.approx(finished.get_current_power())
    assert energy.last_budget_info["total_budget_w"] == pytest.approx(
        (required + finished.get_current_power()) * planner.pue
    )
    active.step(NOW, 0.0, 0.0, {})
    assert submitted.status == "running"


@pytest.mark.parametrize("global_board", [False, True])
def test_spare_renewable_supply_covers_background_idle_without_extra_grid(global_board):
    submitted = job(1)
    planner, scheduler = setup([submitted], nodes=10, global_board=global_board)
    scheduler.rm.keep_idle_nodes_on = True
    planner.pue = 1.3
    energy = EnergyScheduler(
        ConstantForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
        pue=planner.pue,
    )
    energy.step(NOW, 0.0, 0.0, {})
    assert energy.last_budget_info["grid_power_w"] == 0.0
    assert energy.last_budget_info["total_budget_w"] == 4000.0
    scheduler.step(NOW, 0.0, 0.0, {})
    assert submitted.status == "running"


def test_idle_grid_floor_does_not_admit_unplanned_high_carbon_work():
    submitted = job(1)
    planner, scheduler = setup([submitted], nodes=10)
    scheduler.rm.keep_idle_nodes_on = True
    planner.grid_power_limit = 100_000.0
    planner.carbon_intensity_threshold_g_per_kwh = 50.0
    planner.carbon_regret_threshold_g_per_h = None
    energy = EnergyScheduler(
        GridOnlyForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
    )
    energy.step(NOW, 0.0, 0.0, {})
    assert planner.current_plan.placed_jobs == []
    assert energy.last_budget_info["total_budget_w"] == pytest.approx(scheduler.get_current_power())
    scheduler.step(NOW, 0.0, 0.0, {})
    assert submitted.status == "pending"


@pytest.mark.parametrize("global_board", [False, True])
@pytest.mark.parametrize("threshold,admitted", [(250.0, False), (300.0, True)])
def test_static_idle_consumes_renewables_before_threshold_controls_new_work(
    global_board, threshold, admitted,
):
    submitted = job(1, duration_s=120)
    submitted.queue_max_wait_s = 48 * 3600
    planner, scheduler = setup([submitted], nodes=10, global_board=global_board)
    scheduler.rm.keep_idle_nodes_on = True
    planner.pue = 1.3
    planner.grid_power_limit = 100_000.0
    planner.carbon_intensity_threshold_g_per_kwh = threshold
    planner.carbon_regret_threshold_g_per_h = None
    idle_w = 10 * CONFIGS[CLUSTER]["idle_power_w"] * planner.pue

    class IdleOnlyRenewableForecast:
        def forecast(self, start_time, end_time, column, frequency, resample_method):
            value = idle_w if column == "renewable_gen" else 260.0
            return {
                start_time + i * frequency: (
                    value if column == "renewable_gen" or i == 0 else 310.0
                )
                for i in range(int((end_time - start_time) / frequency))
            }

    energy = EnergyScheduler(
        IdleOnlyRenewableForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
        pue=planner.pue,
    )
    energy.step(NOW, 0.0, 0.0, {})
    plan = planner.current_plan
    slot = plan.slots[0]
    assert slot.allocated_w == pytest.approx(idle_w)
    assert slot.idle_allocated_w == pytest.approx(idle_w)
    assert len(plan.placed_jobs) == int(admitted)
    assert len(plan.unplaced_jobs) == int(not admitted)
    if admitted:
        rect = plan.placed_jobs[0]
        assert rect.grid_reason == "low_carbon"
        expected_dynamic_w = (
            scheduler.dvfs_policy.power_at_frequency(rect.frequency_ghz) * planner.pue
        )
        assert rect.power_w == pytest.approx(expected_dynamic_w)
        assert slot.grid_allocated_w == pytest.approx(expected_dynamic_w)
        assert energy.last_budget_info["total_budget_w"] == pytest.approx(idle_w + expected_dynamic_w)
    else:
        assert slot.grid_allocated_w == 0.0
        assert energy.last_budget_info["total_budget_w"] == pytest.approx(idle_w)
    scheduler.step(NOW, 0.0, 0.0, {})
    assert submitted.status == ("running" if admitted else "pending")


@pytest.mark.parametrize("global_board", [False, True])
def test_static_idle_is_included_before_battery_charge_reserve(global_board):
    planner, scheduler = setup([], nodes=10, global_board=global_board)
    scheduler.rm.keep_idle_nodes_on = True
    idle_w = 10 * CONFIGS[CLUSTER]["idle_power_w"]
    planner.renewable_charge_fraction = 0.7

    class IdleOnlyForecast:
        def forecast(self, start_time, end_time, column, frequency, resample_method):
            value = idle_w if column == "renewable_gen" else 260.0
            return {
                start_time + i * frequency: value
                for i in range(int((end_time - start_time) / frequency))
            }

    storage = battery()
    storage._p_charge_max = 10000.0
    info = planner.generate_budget(NOW, IdleOnlyForecast(), storage)
    slot = planner.current_plan.slots[0]
    assert slot.allocated_w == pytest.approx(idle_w)
    assert slot.grid_allocated_w == 0.0
    assert slot.charge_reserve_w == 0.0
    assert info.total_budget_w == pytest.approx(idle_w)


@pytest.mark.parametrize("global_board", [False, True])
@pytest.mark.parametrize("threshold,start_now", [(250.0, False), (300.0, True)])
def test_threshold_selects_now_or_future_green_inside_the_plan(global_board, threshold, start_now):
    submitted = job(1, duration_s=120)
    submitted.queue_max_wait_s = 48 * 3600
    planner, scheduler = setup([submitted], nodes=10, global_board=global_board)
    scheduler.rm.keep_idle_nodes_on = True
    planner.pue = 1.3
    planner.grid_power_limit = 100_000.0
    planner.carbon_intensity_threshold_g_per_kwh = threshold
    planner.carbon_regret_threshold_g_per_h = None
    idle_w = 10 * CONFIGS[CLUSTER]["idle_power_w"] * planner.pue

    class LaterGreenForecast:
        def forecast(self, start_time, end_time, column, frequency, resample_method):
            return {
                start_time + i * frequency: (
                    (idle_w if i < 12 else idle_w + 10_000.0)
                    if column == "renewable_gen" else 260.0
                )
                for i in range(int((end_time - start_time) / frequency))
            }

    energy = EnergyScheduler(
        LaterGreenForecast(), battery(), [scheduler], CONFIGS,
        budget_policy=planner, alloc_strategy=PuzzlePlanAllocationStrategy(planner),
        pue=planner.pue,
    )
    energy.step(NOW, 0.0, 0.0, {})
    rect = planner.current_plan.placed_jobs[0]
    assert rect.start_slot_idx == (0 if start_now else 12)
    assert rect.used_grid is start_now
    assert rect.grid_reason == ("low_carbon" if start_now else None)
    scheduler.step(NOW, 0.0, 0.0, {})
    assert submitted.status == ("running" if start_now else "pending")
