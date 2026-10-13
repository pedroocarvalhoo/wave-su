from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from vessim.wave.energy.budget import BudgetInfo
from vessim.wave.energy.budget_policy import (
    CarbonMinimizingBudgetPolicy,
    DaytimeRenewableBudgetPolicy,
    capped_charge_reserve_w,
    is_daytime,
    rate_limited_battery_discharge_power_w,
    remaining_night_hours,
)
from vessim.wave.energy.puzzle_budget_policy import (
    JobRectangle,
    PowerSlot,
    PuzzleBudgetPolicy,
    PuzzlePlan,
)


class FakeSignal:
    def __init__(
        self,
        pv_gen: float,
        carbon_intensity: float = 100.0,
        renewable_gen: float | None = None,
    ):
        self._values = {
            "pv_gen": pv_gen,
            "carbon_intensity": carbon_intensity,
        }
        if renewable_gen is not None:
            self._values["renewable_gen"] = renewable_gen

    def at(self, time, column: str) -> float:
        return self._values[column]

    def columns(self) -> list[str]:
        return list(self._values)


class FakeBattery:
    _state_of_charge_min = 0.2
    _state_of_charge_max = 0.8
    _battery_energy_max = 2000.0
    _p_charge_max = 500.0
    _p_discharge_max = 300.0

    def __init__(self, energy: float = 1000.0):
        self._battery_energy = energy

    def soc(self) -> float:
        return self._battery_energy / self._battery_energy_max


def test_daytime_and_remaining_night_helpers_support_wrapping_windows():
    assert is_daytime(datetime(2023, 1, 1, 12), (8, 20)) is True
    assert is_daytime(datetime(2023, 1, 1, 21), (8, 20)) is False
    assert remaining_night_hours(datetime(2023, 1, 1, 21), (8, 20)) == 11.0

    assert is_daytime(datetime(2023, 1, 1, 23), (20, 8)) is True
    assert remaining_night_hours(datetime(2023, 1, 1, 12), (20, 8)) == 8.0


def test_budget_info_exposes_compute_budget_and_storage_dispatch_views():
    info = BudgetInfo(
        time="2023-01-01 12:00:00",
        renewable_power_w=1000.0,
        budget_renewable_w=700.0,
        charge_reserve_w=300.0,
        battery_soc=0.5,
        battery_discharge_power_w=100.0,
        grid_power_w=200.0,
        total_budget_w=1000.0,
        allow_discharge=False,
        charge_power=300.0,
        carbon_intensity_g_per_kwh=80.0,
        low_carbon_grid_allowed=True,
    )

    assert info.compute_budget.total_budget_w == 1000.0
    assert info.compute_budget.grid_power_w == 200.0
    assert info.compute_budget.low_carbon_grid_allowed is True
    assert info.storage_dispatch.allow_discharge is False
    assert info.storage_dispatch.charge_power == 300.0
    assert info.storage_dispatch.charge_reserve_w == 300.0


def test_battery_charge_and_discharge_helpers_respect_soc_and_power_limits():
    battery = FakeBattery(energy=1000.0)

    assert capped_charge_reserve_w(renewable_power_w=2000.0, fraction=0.5, battery=battery, step_size=3600) == 500.0
    assert rate_limited_battery_discharge_power_w(battery, remaining_hours=2.0, step_size=3600) == 300.0
    assert rate_limited_battery_discharge_power_w(battery, remaining_hours=12.0, step_size=3600) == 50.0


def test_daytime_policy_reserves_pv_for_charge_and_blocks_discharge():
    policy = DaytimeRenewableBudgetPolicy(
        grid_power_limit=1000.0,
        renewable_charge_fraction=0.5,
        day_grid_fraction=0.2,
    )

    budget = policy.generate_budget(
        datetime(2023, 1, 1, 12),
        FakeSignal(pv_gen=1000.0),
        FakeBattery(energy=1000.0),
        step_size=3600,
    )

    assert budget.charge_reserve_w == 500.0
    assert budget.budget_renewable_w == 500.0
    assert budget.battery_discharge_power_w == 0.0
    assert budget.grid_power_w == 200.0
    assert budget.total_budget_w == 700.0
    assert budget.storage_dispatch.allow_discharge is False
    assert budget.storage_dispatch.charge_power == 500.0


def test_daytime_policy_prefers_aggregate_renewable_generation_column():
    policy = DaytimeRenewableBudgetPolicy(
        grid_power_limit=0.0,
        renewable_charge_fraction=0.5,
    )

    budget = policy.generate_budget(
        datetime(2023, 1, 1, 12),
        FakeSignal(pv_gen=1000.0, renewable_gen=1800.0),
        FakeBattery(energy=1000.0),
        step_size=3600,
    )

    assert budget.renewable_power_w == 1800.0
    assert budget.charge_reserve_w == 500.0
    assert budget.budget_renewable_w == 1300.0


def test_daytime_policy_rate_limits_night_battery_discharge():
    policy = DaytimeRenewableBudgetPolicy(
        grid_power_limit=1000.0,
        night_grid_fraction=1.0,
    )

    budget = policy.generate_budget(
        datetime(2023, 1, 1, 21),
        FakeSignal(pv_gen=0.0),
        FakeBattery(energy=1000.0),
        step_size=3600,
    )

    assert budget.battery_discharge_power_w == pytest.approx(600.0 / 11.0)
    assert budget.grid_power_w == 1000.0
    assert budget.storage_dispatch.allow_discharge is True
    assert budget.storage_dispatch.charge_power == pytest.approx(-600.0 / 11.0)


def test_carbon_policy_exposes_grid_only_in_low_carbon_windows():
    policy = CarbonMinimizingBudgetPolicy(
        grid_power_limit=1000.0,
        carbon_intensity_threshold_g_per_kwh=120.0,
        low_carbon_grid_fraction=0.5,
        high_carbon_grid_fraction=0.0,
    )

    low_carbon = policy.generate_budget(
        datetime(2023, 1, 1, 12),
        FakeSignal(pv_gen=1000.0, carbon_intensity=80.0),
        FakeBattery(energy=1000.0),
        step_size=3600,
    )
    high_carbon = policy.generate_budget(
        datetime(2023, 1, 1, 12),
        FakeSignal(pv_gen=1000.0, carbon_intensity=200.0),
        FakeBattery(energy=1000.0),
        step_size=3600,
    )

    assert low_carbon.low_carbon_grid_allowed is True
    assert low_carbon.grid_power_w == 500.0
    assert low_carbon.storage_dispatch.allow_discharge is False

    assert high_carbon.low_carbon_grid_allowed is False
    assert high_carbon.grid_power_w == 0.0
    assert high_carbon.battery_discharge_power_w == 0.0


def test_puzzle_packer_reserves_capacity_for_running_jobs():
    now = datetime(2023, 1, 1, 12)
    slots = [
        PowerSlot(
            start_time=now,
            end_time=now + timedelta(minutes=5),
            duration_s=300.0,
            renewable_power_w=100.0,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=100.0,
        ),
        PowerSlot(
            start_time=now + timedelta(minutes=5),
            end_time=now + timedelta(minutes=10),
            duration_s=300.0,
            renewable_power_w=100.0,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=100.0,
        ),
    ]
    pending_job = SimpleNamespace(
        status="pending",
        arrival_time=now,
    )
    running_job = SimpleNamespace(
        status="running",
        remaining_time=300.0,
        assigned_frequency=1.0,
        num_nodes=1,
    )
    running_node = SimpleNamespace(measure=lambda: 80.0)
    scheduler = SimpleNamespace(
        cluster_type="cluster_a",
        done=False,
        job_queue=SimpleNamespace(jobs=[pending_job, running_job]),
        rm=SimpleNamespace(
            nodes={"node-0": running_node},
            get_running_allocations=lambda: [
                SimpleNamespace(job=running_job, node_names=["node-0"])
            ],
        ),
    )
    rect = JobRectangle(
        job_id=1,
        power_w=50.0,
        duration_s=300.0,
        num_nodes=1,
        frequency_ghz=1.0,
        energy_wh=50.0 * 300.0 / 3600.0,
        cluster_type="cluster_a",
        arrival_time=now,
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        node_configs={
            "cluster_a": {
                "idle_power_w": 0.0,
                "max_power_w": 100.0,
                "frequencies": {"1.0": 100.0},
            }
        },
        cluster_schedulers=[scheduler],
    )

    placed, unplaced, cluster_slots = policy._pack_jobs_by_cluster(
        slots,
        [rect],
        now,
    )

    assert unplaced == []
    assert len(placed) == 1
    assert placed[0].start_slot_idx == 1
    assert placed[0].stack_base_w == 0.0
    assert cluster_slots["cluster_a"][0].allocated_w == 80.0
    assert cluster_slots["cluster_a"][1].allocated_w == 50.0


def test_running_job_replan_carries_total_power_and_grid_deficit_with_pue():
    now = datetime(2023, 1, 1, 20)
    slots = [
        PowerSlot(
            start_time=now + timedelta(minutes=5 * idx),
            end_time=now + timedelta(minutes=5 * (idx + 1)),
            duration_s=300.0,
            renewable_power_w=50.0,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=50.0,
        )
        for idx in range(2)
    ]
    running_job = SimpleNamespace(
        status="running", remaining_time=600.0,
        assigned_frequency=1.0, num_nodes=1,
    )
    running_node = SimpleNamespace(measure=lambda: 80.0)
    scheduler = SimpleNamespace(
        cluster_type="cluster_a",
        rm=SimpleNamespace(
            nodes={"node-0": running_node},
            get_running_allocations=lambda: [
                SimpleNamespace(job=running_job, node_names=["node-0"]),
            ],
        ),
    )
    policy = PuzzleBudgetPolicy(slot_duration_s=300, pue=1.25)

    policy._reserve_running_jobs(slots, scheduler)

    # 80 W IT × 1.25 PUE = 100 W facility power in both remaining slots.
    assert [slot.allocated_w for slot in slots] == [50.0, 50.0]
    assert [slot.grid_allocated_w for slot in slots] == [50.0, 50.0]
    assert [
        (slot.green_available_w + slot.grid_allocated_w) / policy.pue
        for slot in slots
    ] == [80.0, 80.0]


def test_runtime_budget_has_live_power_floor_and_explicit_grid_source():
    now = datetime(2023, 1, 1, 20)
    slot = PowerSlot(
        start_time=now,
        end_time=now + timedelta(minutes=5),
        duration_s=300.0,
        renewable_power_w=50.0,
        battery_power_w=0.0,
        grid_power_w=0.0,
        green_available_w=50.0,
        grid_allocated_w=10.0,
    )
    scheduler = SimpleNamespace(done=False, get_current_power=lambda: 100.0)
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        cluster_schedulers=[scheduler],
        pue=1.3,
    )
    policy._current_plan = PuzzlePlan(
        plan_start=now,
        plan_end=now + timedelta(hours=24),
        slots=[slot],
        placed_jobs=[],
        unplaced_jobs=[],
        total_budget_kwh=0.0,
        total_planned_kwh=0.0,
    )

    info = policy._budget_at(
        now,
        renewable_signal=SimpleNamespace(),
        battery=FakeBattery(),
    )

    # Planned facility budget is 60 W, but live IT power is 100 W = 130 W
    # facility-side.  The unavoidable 70 W deficit becomes explicit grid.
    assert info.compute_budget.total_budget_w == pytest.approx(130.0)
    assert info.compute_budget.grid_power_w == pytest.approx(80.0)


def test_puzzle_packer_does_not_add_an_approximate_idle_floor():
    now = datetime(2023, 1, 1, 12)
    slots = [
        PowerSlot(
            start_time=now,
            end_time=now + timedelta(minutes=5),
            duration_s=300.0,
            renewable_power_w=100.0,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=100.0,
        )
    ]
    # Rectangles now own their exact total node power (idle + dynamic).  The
    # packer must not add a queue-wide idle estimate based on all pending nodes.
    pending_job = SimpleNamespace(
        status="pending",
        arrival_time=now,
        num_nodes=8,
    )
    scheduler = SimpleNamespace(
        cluster_type="cluster_a",
        done=False,
        job_queue=SimpleNamespace(jobs=[pending_job]),
        rm=SimpleNamespace(
            nodes={f"node-{idx}": object() for idx in range(10)},
            get_running_allocations=lambda: [],
        ),
    )
    rect = JobRectangle(
        job_id=1,
        power_w=45.0,
        duration_s=300.0,
        num_nodes=1,
        frequency_ghz=1.0,
        energy_wh=45.0 * 300.0 / 3600.0,
        cluster_type="cluster_a",
        arrival_time=now,
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        node_configs={
            "cluster_a": {
                "idle_power_w": 5.0,
                "max_power_w": 40.0,
                "frequencies": {"1.0": 40.0},
            }
        },
        cluster_schedulers=[scheduler],
    )

    placed, unplaced, cluster_slots = policy._pack_jobs_by_cluster(
        slots,
        [rect],
        now,
    )

    assert unplaced == []
    assert len(placed) == 1
    assert placed[0].stack_base_w == 0.0
    assert cluster_slots["cluster_a"][0].idle_reserved_w == 0.0
    assert cluster_slots["cluster_a"][0].allocated_w == 45.0


def test_puzzle_packer_places_fungible_power_across_uneven_ceilings():
    now = datetime(2023, 1, 1, 12)
    slots = [
        PowerSlot(
            start_time=now + timedelta(minutes=5 * idx),
            end_time=now + timedelta(minutes=5 * (idx + 1)),
            duration_s=300.0,
            renewable_power_w=available_w,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=available_w,
        )
        for idx, available_w in enumerate([100.0, 100.0, 40.0])
    ]
    first = JobRectangle(
        job_id=1,
        power_w=60.0,
        duration_s=600.0,
        num_nodes=1,
        frequency_ghz=1.0,
        energy_wh=60.0 * 600.0 / 3600.0,
        cluster_type="cluster_a",
    )
    second = JobRectangle(
        job_id=2,
        power_w=40.0,
        duration_s=600.0,
        num_nodes=1,
        frequency_ghz=1.0,
        energy_wh=40.0 * 600.0 / 3600.0,
        cluster_type="cluster_a",
        earliest_start_slot_idx=1,
    )
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    placed, unplaced = policy._pack_jobs(slots, [first, second])

    # Power is fungible (2026-07-16 model change): job 2 fits green because
    # every slot it spans has capacity for it (slot 1: 60 + 40 <= 100;
    # slot 2: 40 <= 40), even though no single contiguous band could host it
    # under the old rigid-rectangle model (base 60 would poke above slot 2's
    # 40 W ceiling — a display-only concern).  The physical invariant is the
    # per-slot power sum, checked below.
    green = [rect for rect in placed if not rect.used_grid]
    grid = [rect for rect in placed if rect.used_grid]
    assert sorted(rect.job_id for rect in green) == [1, 2]
    assert grid == []
    assert unplaced == []
    for idx, slot in enumerate(slots):
        total = sum(
            rect.power_w for rect in green
            if rect.start_slot_idx <= idx <= rect.end_slot_idx
        )
        assert total <= slot.green_available_w + 1e-6


def _repair_slot(idx, now, green_available_w, grid_allocated_w=0.0):
    return PowerSlot(
        start_time=now + timedelta(minutes=5 * idx),
        end_time=now + timedelta(minutes=5 * (idx + 1)),
        duration_s=300.0,
        renewable_power_w=green_available_w,
        battery_power_w=0.0,
        grid_power_w=1_000.0,
        green_available_w=green_available_w,
        grid_allocated_w=grid_allocated_w,
    )


def test_repair_relocates_green_job_to_recover_grid_spill():
    # Two full slots. A movable green job sits in slot 0; a rigid job that can
    # only start in slot 0 was spilled to grid. Relocating the movable job to
    # slot 1 frees slot 0 so the rigid job runs green instead.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 100.0, grid_allocated_w=100.0),
             _repair_slot(1, now, 100.0)]

    mover = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0, stack_base_w=0.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=1,  # may shift to slot 1
    )
    spilled = JobRectangle(
        job_id=2, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0,
        stack_base_w=100.0, used_grid=True,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,  # rigid: slot 0 only
    )
    placed = [mover, spilled]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(slots, placed, reserved_stack=[0.0, 0.0])

    assert recovered == 1
    assert spilled.used_grid is False        # pulled back onto the green board
    assert spilled.start_slot_idx == 0       # in the only slot it may start
    assert mover.start_slot_idx == 1         # shifted out of the way
    assert slots[0].grid_allocated_w == 0.0  # grid footprint cleared


def test_repair_handles_a_partially_green_spill_exactly():
    # The spilled job already consumes 40 W green + 40 W grid in slot 0.
    # Removing the movable 60 W green job makes the spill fully green; its own
    # 40 W reservation must be excluded during the probe and removed exactly
    # when the repair commits.
    now = datetime(2023, 1, 1, 12)
    slots = [
        _repair_slot(0, now, 100.0, grid_allocated_w=40.0),
        _repair_slot(1, now, 100.0),
    ]
    mover = JobRectangle(
        job_id=1, power_w=60.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=5.0, cluster_type="c",
        start_slot_idx=0, end_slot_idx=0, stack_base_w=0.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=1,
    )
    spill = JobRectangle(
        job_id=2, power_w=80.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=8.0, cluster_type="c",
        start_slot_idx=0, end_slot_idx=0, stack_base_w=100.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=0, used_grid=True,
        green_allocation_w_by_slot={0: 40.0},
        grid_allocation_w_by_slot={0: 40.0},
    )
    placed = [mover, spill]
    reserved_stack = [40.0, 0.0]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(
        slots, placed, reserved_stack=reserved_stack,
    )
    policy._sync_allocated_from_placed(slots, placed, reserved_stack)

    assert recovered == 1
    assert spill.used_grid is False
    assert spill.start_slot_idx == 0
    assert mover.start_slot_idx == 1
    assert slots[0].grid_allocated_w == 0.0
    assert reserved_stack == [0.0, 0.0]
    assert [slot.allocated_w for slot in slots] == [80.0, 60.0]


def test_repair_respects_sla_window_and_leaves_spill():
    # Same setup but the movable job may NOT leave slot 0 (latest_start=0), so
    # there is no legal rearrangement: the spill must stay on grid.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 100.0, grid_allocated_w=100.0),
             _repair_slot(1, now, 100.0)]

    stuck = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0, stack_base_w=0.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,  # cannot move
    )
    spilled = JobRectangle(
        job_id=2, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0,
        stack_base_w=100.0, used_grid=True,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,
    )
    placed = [stuck, spilled]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(slots, placed, reserved_stack=[0.0, 0.0])

    assert recovered == 0
    assert spilled.used_grid is True
    assert stuck.start_slot_idx == 0


def test_repair_relocates_mover_without_start_deadline():
    # The green job blocking slot 0 has no SLA bound (latest_start_slot_idx is
    # None): it is the most flexible piece on the board and must be eligible
    # for relocation so the rigid spilled job can be pulled back green.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 100.0, grid_allocated_w=100.0),
             _repair_slot(1, now, 100.0)]

    mover = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0, stack_base_w=0.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=None,  # no SLA bound
    )
    spilled = JobRectangle(
        job_id=2, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0,
        stack_base_w=100.0, used_grid=True,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,  # rigid: slot 0 only
    )
    placed = [mover, spilled]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(slots, placed, reserved_stack=[0.0, 0.0])

    assert recovered == 1
    assert spilled.used_grid is False        # pulled back onto the green board
    assert spilled.start_slot_idx == 0       # in the only slot it may start
    assert mover.start_slot_idx == 1         # unbounded mover shifted aside
    assert slots[0].grid_allocated_w == 0.0  # grid footprint cleared


def test_repair_recovery_respects_node_ledger_when_capacity_allows():
    # Same relocation as the basic recovery test, but with the physical node
    # ledger active (node_capacity=1). Exercises the once-per-spill node
    # accounting (_node_free_all + re-credit via _node_free_add_back): the mover
    # holds slot 0's only node, and freeing it must let the rigid spill claim
    # that node in slot 0 while the mover shifts to slot 1.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 100.0, grid_allocated_w=100.0),
             _repair_slot(1, now, 100.0)]

    mover = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0, stack_base_w=0.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=1,
    )
    spilled = JobRectangle(
        job_id=2, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0,
        stack_base_w=100.0, used_grid=True,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,
    )
    placed = [mover, spilled]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(
        slots, placed, reserved_stack=[0.0, 0.0],
        node_capacity=1, node_reserved=[0, 0],
    )

    assert recovered == 1
    assert spilled.used_grid is False
    assert spilled.start_slot_idx == 0
    assert mover.start_slot_idx == 1


def test_repair_leaves_spill_when_node_ledger_blocks_relocation():
    # Node-bound, not power-bound: powers are tiny but each slot has a single
    # node. A blocker permanently pins slot 1's only node, so the mover has
    # nowhere to shift and the rigid spill cannot be recovered. The re-credited
    # ledger must correctly see slot 1 as full for the mover.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 1000.0, grid_allocated_w=10.0),
             _repair_slot(1, now, 1000.0)]

    blocker = JobRectangle(
        job_id=3, power_w=10.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=10.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=1, end_slot_idx=1, stack_base_w=0.0,
        earliest_start_slot_idx=1, latest_start_slot_idx=1,  # pinned to slot 1
    )
    mover = JobRectangle(
        job_id=1, power_w=10.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=10.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0, stack_base_w=0.0,
        earliest_start_slot_idx=0, latest_start_slot_idx=1,  # could move to slot 1
    )
    spilled = JobRectangle(
        job_id=2, power_w=10.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=10.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0,
        stack_base_w=1000.0, used_grid=True,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,  # rigid: slot 0 only
    )
    placed = [blocker, mover, spilled]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(
        slots, placed, reserved_stack=[0.0, 0.0],
        node_capacity=1, node_reserved=[0, 0],
    )

    assert recovered == 0
    assert spilled.used_grid is True
    assert mover.start_slot_idx == 0  # nowhere to move; stays put


def test_repair_relocates_over_a_crowded_multi_slot_board():
    # Larger board (4 slots, several stacked green rects) so the repair search
    # gathers many obstacles per candidate window. A flexible mover blocks the
    # only slot a rigid spill may use; relocating it past the filler stacks into
    # slot 1 frees slot 0 for the spill. Also asserts the recovered board is
    # physically valid: nothing exceeds the green ceiling and no two green rects
    # overlap in both time and power band.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(i, now, 100.0) for i in range(4)]
    slots[0] = _repair_slot(0, now, 100.0, grid_allocated_w=70.0)

    def green_rect(job_id, start, end, base, power, latest):
        return JobRectangle(
            job_id=job_id, power_w=power, duration_s=300.0 * (end - start + 1),
            num_nodes=1, frequency_ghz=1.0,
            energy_wh=power * 300.0 * (end - start + 1) / 3600.0,
            cluster_type="c", start_slot_idx=start, end_slot_idx=end,
            stack_base_w=base, earliest_start_slot_idx=0,
            latest_start_slot_idx=latest,
        )

    mover = green_rect(1, 0, 0, base=0.0, power=40.0, latest=3)   # flexible
    f1 = green_rect(11, 1, 2, base=0.0, power=30.0, latest=1)
    f2 = green_rect(12, 1, 3, base=30.0, power=20.0, latest=1)
    f3 = green_rect(13, 2, 2, base=50.0, power=40.0, latest=2)
    f4 = green_rect(14, 3, 3, base=50.0, power=50.0, latest=3)  # above f2's band
    spilled = JobRectangle(
        job_id=2, power_w=70.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=70.0 * 300.0 / 3600.0,
        cluster_type="c", start_slot_idx=0, end_slot_idx=0,
        stack_base_w=100.0, used_grid=True,
        earliest_start_slot_idx=0, latest_start_slot_idx=0,  # rigid: slot 0 only
    )
    placed = [mover, f1, f2, f3, f4, spilled]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    recovered = policy._repair_relocate(
        slots, placed, reserved_stack=[0.0] * len(slots),
    )

    assert recovered == 1
    assert spilled.used_grid is False
    assert spilled.start_slot_idx == 0   # taken over the freed slot 0
    assert mover.start_slot_idx == 1     # relocated past the filler stacks

    # The repaired board must be physically valid.
    green = [r for r in placed if not r.used_grid]
    tol = 1e-6
    for r in green:
        for s in range(r.start_slot_idx, r.end_slot_idx + 1):
            assert r.stack_base_w + r.power_w <= slots[s].green_available_w + tol
    for i, a in enumerate(green):
        for b in green[i + 1:]:
            time_overlap = (
                a.start_slot_idx <= b.end_slot_idx
                and b.start_slot_idx <= a.end_slot_idx
            )
            if not time_overlap:
                continue
            band_overlap = (
                a.stack_base_w < b.stack_base_w + b.power_w - tol
                and b.stack_base_w < a.stack_base_w + a.power_w - tol
            )
            assert not band_overlap, (a.job_id, b.job_id)


class _FlatForecastSignal:
    """Signal with a flat forecast for renewable_gen and carbon_intensity."""

    def __init__(self, renewable_w: float, carbon: float = 100.0):
        self._renewable = renewable_w
        self._carbon = carbon

    def forecast(self, start_time, end_time, column, frequency, resample_method):
        n = int((end_time - start_time) / frequency)
        val = self._renewable if column == "renewable_gen" else self._carbon
        return {i: val for i in range(n)}


class _HistoricalFlatForecastSignal(_FlatForecastSignal):
    """Flat forecast with realised history available to the v0.6 calibrator."""

    def __init__(self, renewable_w: float, carbon: float = 100.0):
        super().__init__(renewable_w, carbon)
        self.at_calls = 0

    def columns(self):
        return ["renewable_gen", "carbon_intensity"]

    def at(self, time, column):
        self.at_calls += 1
        return self._renewable if column == "renewable_gen" else self._carbon


def test_puzzle_charge_reserve_yields_to_pending_demand():
    # Daytime PV-rich slot. With no queue the plan reserves the charge
    # fraction of PV for the battery; with pending work the reserve must
    # yield so the PV goes to the compute board instead of being stored
    # (or exported) while jobs wait.
    noon = datetime(2023, 1, 1, 12)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        renewable_charge_fraction=0.5,
    )
    signal = _FlatForecastSignal(renewable_w=1000.0)
    battery = FakeBattery(energy=1000.0)

    def rect(power_w):
        return JobRectangle(
            job_id=1, power_w=power_w, duration_s=3600.0, num_nodes=4,
            frequency_ghz=2.0, energy_wh=power_w, cluster_type="c",
            earliest_start_slot_idx=0,
        )

    idle = policy._build_envelope(noon, signal, battery, job_rects=[])
    # No demand: reserve = 50% of 1000W, capped by the 500W charge rate.
    assert idle[0].charge_reserve_w == 500.0
    assert idle[0].green_available_w == 500.0

    busy = policy._build_envelope(noon, signal, battery, job_rects=[rect(1500.0)])
    # Demand covers the slot's whole renewable: nothing to spare for
    # charging — the full 1000W goes to the green board.
    assert busy[0].charge_reserve_w == 0.0
    assert busy[0].green_available_w == 1000.0

    partial = policy._build_envelope(noon, signal, battery, job_rects=[rect(700.0)])
    # Demand 700W of 1000W: charging claims only the true 300W surplus.
    assert partial[0].charge_reserve_w == 300.0
    assert partial[0].green_available_w == 700.0


def test_puzzle_charge_reserve_stops_when_battery_planned_full():
    # The battery has 10Wh of room: the plan may reserve exactly that much
    # energy and must stop claiming PV afterwards — reserving beyond the
    # battery's absorption just exports the excess.
    noon = datetime(2023, 1, 1, 12)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        renewable_charge_fraction=0.5,
    )
    signal = _FlatForecastSignal(renewable_w=1000.0)
    battery = FakeBattery(energy=1590.0)  # free = 0.8*2000 - 1590 = 10 Wh

    slots = policy._build_envelope(noon, signal, battery, job_rects=[])

    # Slot 0: capped by the 10Wh of room (10Wh over 300s = 120W).
    assert slots[0].charge_reserve_w == 120.0
    # Battery planned full: no further reserve, PV goes to the board.
    assert slots[1].charge_reserve_w == 0.0
    assert slots[1].green_available_w == 1000.0
    reserved_wh = sum(s.charge_reserve_w * 300.0 / 3600.0 for s in slots)
    assert reserved_wh == 10.0


def test_packer_respects_cluster_node_capacity():
    # Both jobs fit slot 0's power envelope side by side, but together they
    # need 40 nodes and the cluster has 33: the packer must serialize them
    # instead of planning a physically impossible slot.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 1000.0), _repair_slot(1, now, 1000.0)]

    def gpu_rect(job_id):
        return JobRectangle(
            job_id=job_id, power_w=200.0, duration_s=300.0, num_nodes=20,
            frequency_ghz=1.0, energy_wh=200.0 * 300.0 / 3600.0,
            cluster_type="gpu",
        )

    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    placed, unplaced = policy._pack_jobs(
        slots, [gpu_rect(1), gpu_rect(2)],
        node_capacity=33, node_reserved=[0, 0],
    )

    assert unplaced == []
    green = [r for r in placed if not r.used_grid]
    assert len(green) == 2
    assert sorted(r.start_slot_idx for r in green) == [0, 1]


def test_packer_grid_spill_waits_for_free_nodes():
    # Job 1 saturates all 33 nodes for slots 0-1. Job 2 must start by slot 0
    # (SLA) so it cannot go green; its grid spill still runs on physical
    # nodes, so it is planned at slot 2 — the first window with node
    # capacity — instead of stacking impossibly on top of job 1 at slot 0.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(idx, now, 100.0) for idx in range(3)]

    saturator = JobRectangle(
        job_id=1, power_w=100.0, duration_s=600.0, num_nodes=33,
        frequency_ghz=1.0, energy_wh=100.0 * 600.0 / 3600.0,
        cluster_type="gpu", earliest_start_slot_idx=0, latest_start_slot_idx=0,
    )
    tight = JobRectangle(
        job_id=2, power_w=50.0, duration_s=300.0, num_nodes=10,
        frequency_ghz=1.0, energy_wh=50.0 * 300.0 / 3600.0,
        cluster_type="gpu", earliest_start_slot_idx=0, latest_start_slot_idx=0,
    )
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    placed, unplaced = policy._pack_jobs(
        slots, [saturator, tight],
        node_capacity=33, node_reserved=[0, 0, 0],
    )

    assert unplaced == []
    assert saturator.used_grid is False
    assert saturator.start_slot_idx == 0
    assert tight.used_grid is True
    assert tight.start_slot_idx == 2  # first node-feasible window


def test_repair_relocation_respects_node_ledger():
    # Power-wise the mover could shift to slot 1, but a small green job
    # already holds 20 of the 33 nodes there: with the node ledger the
    # relocation must be rejected and the spill stays on grid; without the
    # ledger the same board is (wrongly) repaired.
    now = datetime(2023, 1, 1, 12)

    def build_board():
        slots = [_repair_slot(0, now, 300.0, grid_allocated_w=100.0),
                 _repair_slot(1, now, 300.0)]
        mover = JobRectangle(
            job_id=1, power_w=250.0, duration_s=300.0, num_nodes=20,
            frequency_ghz=1.0, energy_wh=250.0 * 300.0 / 3600.0,
            cluster_type="gpu", start_slot_idx=0, end_slot_idx=0,
            stack_base_w=0.0, earliest_start_slot_idx=0,
            latest_start_slot_idx=None,
        )
        blocker = JobRectangle(
            job_id=2, power_w=10.0, duration_s=300.0, num_nodes=20,
            frequency_ghz=1.0, energy_wh=10.0 * 300.0 / 3600.0,
            cluster_type="gpu", start_slot_idx=1, end_slot_idx=1,
            stack_base_w=0.0, earliest_start_slot_idx=1,
            latest_start_slot_idx=1,
        )
        spilled = JobRectangle(
            job_id=3, power_w=100.0, duration_s=300.0, num_nodes=5,
            frequency_ghz=1.0, energy_wh=100.0 * 300.0 / 3600.0,
            cluster_type="gpu", start_slot_idx=0, end_slot_idx=0,
            stack_base_w=300.0, used_grid=True,
            earliest_start_slot_idx=0, latest_start_slot_idx=0,
        )
        return slots, [mover, blocker, spilled], spilled

    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    slots, placed, spilled = build_board()
    recovered = policy._repair_relocate(
        slots, placed, reserved_stack=[0.0, 0.0],
        node_capacity=33, node_reserved=[0, 0],
    )
    assert recovered == 0
    assert spilled.used_grid is True  # no node-feasible rearrangement

    slots, placed, spilled = build_board()
    recovered = policy._repair_relocate(
        slots, placed, reserved_stack=[0.0, 0.0],
    )
    assert recovered == 1             # power alone would have allowed it
    assert spilled.used_grid is False


def test_cluster_budget_weights_follow_demand_not_capacity():
    # A big cluster with tiny demand must not hog the green board over a small
    # cluster that is heavily loaded: the slice follows demand, capped by what
    # each cluster's nodes could physically absorb over the horizon.
    now = datetime(2023, 1, 1, 12)

    def pjob(job_id):
        return SimpleNamespace(
            job_id=job_id, status="pending", arrival_time=now, num_nodes=1,
        )

    def sched(ctype, n_nodes, jobs):
        return SimpleNamespace(
            cluster_type=ctype, done=False,
            job_queue=SimpleNamespace(jobs=jobs),
            rm=SimpleNamespace(
                nodes={f"{ctype}-{i}": object() for i in range(n_nodes)},
                get_running_allocations=lambda: [],
            ),
        )

    big_idle = sched("big", 1000, [pjob(1)])          # huge capacity, one job
    small_busy = sched("small", 50, [pjob(10 + i) for i in range(10)])

    policy = PuzzleBudgetPolicy(
        plan_horizon_h=24.0,
        slot_duration_s=300,
        node_configs={
            "big": {"idle_power_w": 10.0, "max_power_w": 100.0,
                    "frequencies": {"2.0": 100.0}},
            "small": {"idle_power_w": 20.0, "max_power_w": 400.0,
                      "frequencies": {"2.0": 400.0}},
        },
        cluster_schedulers=[big_idle, small_busy],
    )

    def rect(job_id, ctype, power_w, duration_s):
        return JobRectangle(
            job_id=job_id, power_w=power_w, duration_s=duration_s, num_nodes=1,
            frequency_ghz=2.0, energy_wh=power_w * duration_s / 3600.0,
            cluster_type=ctype, arrival_time=now,
        )

    rects = [rect(1, "big", 100, 600)]
    rects += [rect(10 + i, "small", 400, 3600) for i in range(10)]

    weights = policy._cluster_budget_weights(rects, now)

    # Small/busy cluster claims the dominant share despite far less capacity.
    assert weights["small"] > weights["big"]
    total = weights["small"] + weights["big"]
    assert weights["small"] / total > 0.9
    # The tiny-demand cluster is not starved to zero; it still gets its slice.
    assert weights["big"] > 0


def test_global_green_board_reclaims_capacity_stranded_by_cluster_slices():
    now = datetime(2023, 1, 1, 12)

    def board():
        return [_repair_slot(0, now, 100.0), _repair_slot(1, now, 100.0)]

    def rectangles():
        return [
            JobRectangle(
                job_id=1, power_w=50.0, duration_s=300.0, num_nodes=1,
                frequency_ghz=1.0, energy_wh=50.0 / 12.0,
                cluster_type="a", earliest_start_slot_idx=0,
                latest_start_slot_idx=0,
            ),
            JobRectangle(
                job_id=2, power_w=75.0, duration_s=300.0, num_nodes=1,
                frequency_ghz=1.0, energy_wh=75.0 / 12.0,
                cluster_type="b", earliest_start_slot_idx=1,
                latest_start_slot_idx=1,
            ),
        ]

    # Fixed demand-weighted slices reserve 40 W for A and 60 W for B in every
    # slot.  Each job therefore spills despite the other cluster leaving that
    # slot empty.
    local = PuzzleBudgetPolicy(
        slot_duration_s=300, global_green_board=False,
    )
    local_slots = board()
    local_placed, local_unplaced, _ = local._pack_jobs_by_cluster(
        local_slots, rectangles(), now,
    )
    assert local_unplaced == []
    assert all(rect.used_grid for rect in local_placed)
    assert sum(slot.grid_allocated_w for slot in local_slots) == pytest.approx(25.0)

    # The global board lets A use slot 0 and B use slot 1 directly.  No green
    # capacity is stranded in a cluster quota, and the reconstructed cluster
    # steering slots sum exactly to the shared board.
    global_policy = PuzzleBudgetPolicy(
        slot_duration_s=300, global_green_board=True,
    )
    global_slots = board()
    placed, unplaced, cluster_slots = global_policy._pack_jobs_by_cluster(
        global_slots, rectangles(), now,
    )

    assert unplaced == []
    assert all(not rect.used_grid for rect in placed)
    assert [slot.allocated_w for slot in global_slots] == [50.0, 75.0]
    assert [slot.grid_allocated_w for slot in global_slots] == [0.0, 0.0]
    for idx, slot in enumerate(global_slots):
        assert sum(
            plan[idx].allocated_w for plan in cluster_slots.values()
        ) == pytest.approx(slot.allocated_w)
        assert sum(
            plan[idx].grid_allocated_w for plan in cluster_slots.values()
        ) == pytest.approx(slot.grid_allocated_w)


def test_global_green_board_builds_packing_indexes_once_per_plan():
    class CountingPolicy(PuzzleBudgetPolicy):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.pack_calls = 0

        def _pack_jobs(self, *args, **kwargs):
            self.pack_calls += 1
            return super()._pack_jobs(*args, **kwargs)

    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 200.0)]
    rects = [
        JobRectangle(
            job_id=idx, power_w=50.0, duration_s=300.0, num_nodes=1,
            frequency_ghz=1.0, energy_wh=50.0 / 12.0,
            cluster_type=cluster_type, latest_start_slot_idx=0,
        )
        for idx, cluster_type in enumerate(("a", "b"), start=1)
    ]
    policy = CountingPolicy(slot_duration_s=300, global_green_board=True)

    placed, unplaced, _ = policy._pack_jobs_by_cluster(slots, rects, now)

    assert len(placed) == 2
    assert unplaced == []
    assert policy.pack_calls == 1


def test_global_green_board_keeps_node_ledgers_independent_by_cluster():
    now = datetime(2023, 1, 1, 12)

    def scheduler(cluster_type):
        return SimpleNamespace(
            cluster_type=cluster_type,
            done=False,
            job_queue=SimpleNamespace(jobs=[]),
            rm=SimpleNamespace(
                nodes={f"{cluster_type}-0": object()},
                get_running_allocations=lambda: [],
            ),
        )

    rects = [
        JobRectangle(
            job_id=idx, power_w=100.0, duration_s=300.0, num_nodes=1,
            frequency_ghz=1.0, energy_wh=100.0 / 12.0,
            cluster_type=cluster_type, latest_start_slot_idx=0,
        )
        for idx, cluster_type in enumerate(("a", "b"), start=1)
    ]
    slots = [_repair_slot(0, now, 200.0)]
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        cluster_schedulers=[scheduler("a"), scheduler("b")],
        global_green_board=True,
    )

    placed, unplaced, cluster_slots = policy._pack_jobs_by_cluster(
        slots, rects, now,
    )

    assert unplaced == []
    assert len(placed) == 2
    assert all(rect.start_slot_idx == 0 for rect in placed)
    assert cluster_slots["a"][0].allocated_w == 100.0
    assert cluster_slots["b"][0].allocated_w == 100.0
    assert slots[0].allocated_w == 200.0


def test_global_cluster_slots_reproduce_partial_green_grid_split():
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 100.0)]
    rects = [
        JobRectangle(
            job_id=idx, power_w=60.0, duration_s=300.0, num_nodes=1,
            frequency_ghz=1.0, energy_wh=5.0,
            cluster_type=cluster_type, latest_start_slot_idx=0,
        )
        for idx, cluster_type in enumerate(("a", "b"), start=1)
    ]
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300, global_green_board=True,
    )

    placed, unplaced, cluster_slots = policy._pack_jobs_by_cluster(
        slots, rects, now,
    )

    assert unplaced == []
    assert len(placed) == 2
    assert slots[0].allocated_w == pytest.approx(100.0)
    assert slots[0].grid_allocated_w == pytest.approx(20.0)
    assert sum(
        plan[0].allocated_w for plan in cluster_slots.values()
    ) == pytest.approx(100.0)
    assert sum(
        plan[0].grid_allocated_w for plan in cluster_slots.values()
    ) == pytest.approx(20.0)
    assert sum(
        plan[0].green_available_w + plan[0].grid_allocated_w
        for plan in cluster_slots.values()
    ) == pytest.approx(120.0)


def test_build_job_rectangles_skips_queue_node_limit_violations():
    now = datetime(2023, 1, 1, 12)

    def make_job(job_id: int, num_nodes: int, queue_max_nodes):
        return SimpleNamespace(
            job_id=job_id,
            num_nodes=num_nodes,
            status="pending",
            arrival_time=now,
            queue_max_nodes=queue_max_nodes,
            memory_gb=10.0,
            cpi=1.0,
            num_instructions=1e9,
        )

    valid_job = make_job(job_id=1, num_nodes=4, queue_max_nodes=32)
    invalid_job = make_job(job_id=2, num_nodes=84, queue_max_nodes=32)
    scheduler = SimpleNamespace(
        cluster_type="cluster_a",
        done=False,
        job_queue=SimpleNamespace(jobs=[valid_job, invalid_job]),
        rm=SimpleNamespace(nodes={}, get_running_allocations=lambda: []),
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        node_configs={
            "cluster_a": {
                "idle_power_w": 5.0,
                "max_power_w": 100.0,
                "base_frequency_ghz": 2.0,
                "memory_bandwidth_gb_s": 100.0,
                "frequencies": {"2.0": 50.0},
            }
        },
        cluster_schedulers=[scheduler],
        pue=1.3,
    )

    rects = policy._build_job_rectangles(now)

    assert [rect.job_id for rect in rects] == [1]
    assert rects[0].power_w == pytest.approx((5.0 + 50.0) * 4 * 1.3)
    assert rects[0].dvfs_variants[0][1] == pytest.approx(
        (5.0 + 50.0) * 4 * 1.3
    )


def test_job_rectangle_reserves_replan_margin_before_sla_deadline():
    now = datetime(2023, 1, 1, 12)
    job = SimpleNamespace(
        job_id=1,
        num_nodes=1,
        status="pending",
        arrival_time=now,
        queue_enter_time=now,
        queue_max_wait_s=8 * 3600.0,
        queue_max_nodes=32,
        memory_gb=10.0,
        cpi=1.0,
        num_instructions=1e9,
    )
    scheduler = SimpleNamespace(
        cluster_type="cluster_a",
        done=False,
        job_queue=SimpleNamespace(jobs=[job]),
        rm=SimpleNamespace(nodes={}, get_running_allocations=lambda: []),
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        replan_interval_h=6.0,
        node_configs={
            "cluster_a": {
                "idle_power_w": 5.0,
                "max_power_w": 100.0,
                "base_frequency_ghz": 2.0,
                "memory_bandwidth_gb_s": 100.0,
                "frequencies": {"2.0": 50.0},
            }
        },
        cluster_schedulers=[scheduler],
    )

    rect = policy._build_job_rectangles(now)[0]

    # Formal max-wait is 8h, but one 6h replan interval is reserved as
    # dispatcher/contention safety: latest safe start is 2h = 24 slots.
    assert rect.latest_start_slot_idx == 24


def _slot(start: datetime, available_w: float) -> PowerSlot:
    return PowerSlot(
        start_time=start,
        end_time=start + timedelta(minutes=5),
        duration_s=300.0,
        renewable_power_w=available_w,
        battery_power_w=0.0,
        grid_power_w=0.0,
        green_available_w=available_w,
    )


def test_get_cluster_compute_budgets_reads_plan_slice():
    now = datetime(2023, 1, 1, 12)
    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    policy._current_plan = PuzzlePlan(
        plan_start=now,
        plan_end=now + timedelta(hours=24),
        slots=[_slot(now, 120.0), _slot(now + timedelta(minutes=5), 100.0)],
        placed_jobs=[],
        unplaced_jobs=[],
        total_budget_kwh=0.0,
        total_planned_kwh=0.0,
        cluster_slots={
            "arm": [_slot(now, 100.0), _slot(now + timedelta(minutes=5), 80.0)],
            "gpu": [_slot(now, 20.0), _slot(now + timedelta(minutes=5), 20.0)],
        },
    )

    assert policy.get_cluster_compute_budgets(now) == {"arm": 100.0, "gpu": 20.0}
    # Slot 1 (5 min later).
    later = policy.get_cluster_compute_budgets(now + timedelta(minutes=5))
    assert later["arm"] == 80.0


def _snap(scheduler, cluster_type, is_done=False, current_power=0.0):
    from vessim.wave.energy.strategy import ClusterSnapshot
    return ClusterSnapshot(
        scheduler=scheduler,
        cluster_type=cluster_type,
        total_nodes=1,
        powered_on_nodes=0,
        busy_nodes=0,
        idle_power_per_node=0.0,
        max_power_per_node=0.0,
        current_power=current_power,
        is_done=is_done,
        has_pending_jobs=True,
    )


def test_puzzle_plan_allocation_follows_plan_shares():
    from vessim.wave.energy.strategy import PuzzlePlanAllocationStrategy
    now = datetime(2023, 1, 1, 12)

    class FakeBP:
        def get_cluster_compute_budgets(self, time):
            return {"arm": 75.0, "gpu": 25.0}

    arm, gpu = object(), object()
    strat = PuzzlePlanAllocationStrategy(FakeBP())
    alloc = strat.allocate(1000.0, [_snap(arm, "arm"), _snap(gpu, "gpu")], now)
    assert alloc[arm] == 750.0
    assert alloc[gpu] == 250.0

    # A finished cluster gets nothing; its share goes to the active one.
    alloc2 = strat.allocate(
        1000.0, [_snap(arm, "arm"), _snap(gpu, "gpu", is_done=True)], now
    )
    assert alloc2[arm] == 1000.0
    assert alloc2[gpu] == 0.0


def test_puzzle_plan_allocation_preserves_live_cluster_power_first():
    from vessim.wave.energy.strategy import PuzzlePlanAllocationStrategy
    now = datetime(2023, 1, 1, 12)

    class FakeBP:
        pue = 1.0

        def get_cluster_compute_budgets(self, time):
            return {"arm": 50.0, "gpu": 50.0}

    arm, gpu = object(), object()
    strat = PuzzlePlanAllocationStrategy(FakeBP())
    alloc = strat.allocate(
        100.0,
        [
            _snap(arm, "arm", current_power=80.0),
            _snap(gpu, "gpu", current_power=20.0),
        ],
        now,
    )

    assert alloc[arm] == 80.0
    assert alloc[gpu] == 20.0


def test_puzzle_plan_allocation_falls_back_without_plan():
    from vessim.wave.energy.strategy import PuzzlePlanAllocationStrategy
    now = datetime(2023, 1, 1, 12)

    class EmptyBP:
        def get_cluster_compute_budgets(self, time):
            return {}

    arm = object()
    sentinel = {arm: 42.0}

    class FakeFallback:
        def allocate(self, total_budget, snapshots, time=None):
            return sentinel

    strat = PuzzlePlanAllocationStrategy(EmptyBP(), fallback_strategy=FakeFallback())
    assert strat.allocate(1000.0, [_snap(arm, "arm")], now) is sentinel


def test_puzzle_replan_requests_are_rate_limited():
    now = datetime(2023, 1, 1, 12)
    slots = [
        PowerSlot(
            start_time=now,
            end_time=now + timedelta(minutes=5),
            duration_s=300.0,
            renewable_power_w=100.0,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=100.0,
        )
    ]
    policy = PuzzleBudgetPolicy(
        replan_interval_h=6.0,
        replan_request_cooldown_h=1.0,
    )
    policy._current_plan = PuzzlePlan(
        plan_start=now,
        plan_end=now + timedelta(hours=24),
        slots=slots,
        placed_jobs=[],
        unplaced_jobs=[],
        total_budget_kwh=0.0,
        total_planned_kwh=0.0,
    )
    policy._last_plan_time = now

    policy.request_replan(now + timedelta(minutes=5))

    assert policy._should_replan(now + timedelta(minutes=5)) is False
    assert policy._replan_requested is True
    assert policy._should_replan(now + timedelta(hours=1)) is True
    assert policy._replan_requested is False


def test_window_extension_need_and_forecast_repeats():
    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    n_visible = 12  # 1h window of 5-minute slots

    def rect(duration_s, earliest=0, variants=()):
        return JobRectangle(
            job_id=1, power_w=40.0, duration_s=duration_s, num_nodes=1,
            frequency_ghz=1.0, energy_wh=40.0 * duration_s / 3600.0,
            cluster_type="c", earliest_start_slot_idx=earliest,
            dvfs_variants=list(variants),
        )

    # Fits the window: no extension.
    assert policy._window_extension_need(rect(3600.0), n_visible) == 0
    # 2.5h from slot 0 -> needs 30 slots.
    assert policy._window_extension_need(rect(9000.0), n_visible) == 30
    # Any variant fitting the window disables extension for the job.
    assert policy._window_extension_need(
        rect(9000.0, variants=[(1.0, 40.0, 9000.0), (2.0, 80.0, 3000.0)]),
        n_visible,
    ) == 0
    # A job that cannot even start inside the window is the next window's
    # problem, exactly as before extrapolation existed.
    assert policy._window_extension_need(rect(9000.0, earliest=12), n_visible) == 0

    assert policy._forecast_repeats([rect(3600.0)], n_visible) == 1
    assert policy._forecast_repeats([rect(9000.0)], n_visible) == 3
    assert policy._forecast_repeats(
        [rect(9000.0), rect(3600.0)], n_visible,
    ) == 3


def test_envelope_tiles_forecast_for_oversized_job():
    midnight = datetime(2023, 1, 1, 0)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_renewable_uncertainty_fraction=0.25,
    )
    signal = _FlatForecastSignal(renewable_w=1000.0)
    battery = FakeBattery(energy=1000.0)

    long_job = JobRectangle(
        job_id=1, power_w=40.0, duration_s=9000.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    base = policy._build_envelope(midnight, signal, battery, job_rects=[])
    assert len(base) == 12

    tiled = policy._build_envelope(midnight, signal, battery, job_rects=[long_job])
    # 30 slots needed -> 3 copies of the 12-slot window.
    assert len(tiled) == 36
    assert all(slot.battery_power_w > 0 for slot in tiled[:12])
    assert all(not slot.is_extrapolated for slot in tiled[:12])
    assert all(
        slot.renewable_pessimistic_power_w
        == slot.renewable_expected_power_w
        == slot.renewable_optimistic_power_w
        == 1000.0
        for slot in tiled[:12]
    )
    for k in (12, 24):
        for i in range(12):
            copy, orig = tiled[k + i], tiled[i]
            # The repeated forecast becomes an explicit scenario set.  The
            # robust board uses the pessimistic renewable value, while the
            # expected and optimistic paths remain observable.
            assert copy.is_extrapolated is True
            assert copy.renewable_expected_power_w == orig.renewable_power_w
            assert copy.renewable_pessimistic_power_w == 750.0
            assert copy.renewable_optimistic_power_w == 1250.0
            assert copy.renewable_power_w == 1000.0
            assert copy.battery_power_w == 0.0
            assert copy.battery_correction_power_w == 0.0
            assert copy.green_available_w == 1000.0
            assert copy.charge_reserve_w == 0.0
            assert copy.carbon_intensity == orig.carbon_intensity
            assert copy.start_time == orig.start_time + timedelta(hours=k // 12)


def test_zero_tail_uncertainty_reproduces_expected_forecast_board():
    midnight = datetime(2023, 1, 1, 0)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_renewable_uncertainty_fraction=0.0,
    )
    long_job = JobRectangle(
        job_id=1, power_w=40.0, duration_s=9000.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    slots = policy._build_envelope(
        midnight,
        _FlatForecastSignal(renewable_w=1000.0),
        FakeBattery(energy=1000.0),
        job_rects=[long_job],
    )

    for slot in slots[12:]:
        assert slot.is_extrapolated is True
        assert slot.renewable_pessimistic_power_w == 1000.0
        assert slot.renewable_expected_power_w == 1000.0
        assert slot.renewable_optimistic_power_w == 1000.0
        assert slot.green_available_w == 1000.0


def test_tail_scenarios_use_historical_calibration_when_available():
    midnight = datetime(2023, 6, 1, 0)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_renewable_uncertainty_fraction=0.25,
        tail_renewable_calibration_days=1.0,
    )
    long_job = JobRectangle(
        job_id=1, power_w=40.0, duration_s=5400.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=60.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    slots = policy._build_envelope(
        midnight,
        _HistoricalFlatForecastSignal(renewable_w=1000.0),
        FakeBattery(energy=400.0),
        job_rects=[long_job],
    )

    assert policy._last_tail_renewable_calibration
    point = policy._last_tail_renewable_calibration[0]
    assert point.used_fallback is False
    assert point.sample_count > 0
    assert point.pessimistic_fraction == 0.0
    assert point.optimistic_fraction == 0.0
    assert all(slot.renewable_pessimistic_power_w == 1000.0 for slot in slots[12:])
    assert all(slot.renewable_optimistic_power_w == 1000.0 for slot in slots[12:])


def test_renewable_history_cache_reads_only_new_slots_after_first_plan():
    start = datetime(2023, 6, 1, 0)
    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    signal = _HistoricalFlatForecastSignal(renewable_w=1000.0)

    first = policy._renewable_history_values(start, signal, history_slots=288)
    assert len(first) == 288
    assert signal.at_calls == 288

    second = policy._renewable_history_values(
        start + timedelta(hours=6), signal, history_slots=288
    )
    assert len(second) == 288
    assert signal.at_calls == 288 + 72


def test_tail_scenarios_keep_expected_board_and_physical_recourse_independent():
    midnight = datetime(2023, 1, 1, 0)

    def long_job():
        return JobRectangle(
            job_id=1, power_w=750.0, duration_s=5400.0, num_nodes=1,
            frequency_ghz=1.0, energy_wh=1125.0, cluster_type="c",
            earliest_start_slot_idx=0,
        )

    expected_policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_renewable_uncertainty_fraction=0.0,
    )
    expected_rect = long_job()
    expected_slots = expected_policy._build_envelope(
        midnight,
        _FlatForecastSignal(renewable_w=1000.0),
        FakeBattery(energy=400.0),
        job_rects=[expected_rect],
    )
    expected_placed, expected_unplaced = expected_policy._pack_jobs(
        expected_slots, [expected_rect], visible_slots=12,
    )

    robust_policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_renewable_uncertainty_fraction=0.50,
    )
    robust_rect = long_job()
    robust_slots = robust_policy._build_envelope(
        midnight,
        _FlatForecastSignal(renewable_w=1000.0),
        FakeBattery(energy=400.0),
        job_rects=[robust_rect],
    )
    robust_placed, robust_unplaced = robust_policy._pack_jobs(
        robust_slots, [robust_rect], visible_slots=12,
    )

    assert [rect.job_id for rect in expected_placed] == [1]
    assert expected_unplaced == []
    assert expected_placed[0].used_grid is False
    assert [rect.job_id for rect in robust_placed] == [1]
    assert robust_unplaced == []
    assert robust_placed[0].used_grid is False
    assert expected_policy._effective_tail_battery_reserve_fraction() == 0.10
    assert robust_policy._effective_tail_battery_reserve_fraction() == 0.10


def test_tail_uncertainty_does_not_speculatively_increase_recourse_reserve():
    policy = PuzzleBudgetPolicy(
        tail_battery_reserve_fraction=0.10,
        tail_renewable_uncertainty_fraction=1.0,
    )

    assert policy._effective_tail_battery_reserve_fraction() == 0.10


def test_tail_robustness_one_commits_to_the_pessimistic_scenario():
    midnight = datetime(2023, 1, 1, 0)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_renewable_uncertainty_fraction=0.25,
        tail_renewable_robustness=1.0,
    )
    long_job = JobRectangle(
        job_id=1, power_w=40.0, duration_s=9000.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    slots = policy._build_envelope(
        midnight,
        _FlatForecastSignal(renewable_w=1000.0),
        FakeBattery(energy=400.0),
        job_rects=[long_job],
    )

    assert all(slot.renewable_power_w == 750.0 for slot in slots[12:])
    assert all(slot.green_available_w == 750.0 for slot in slots[12:])


@pytest.mark.parametrize("robustness", [-0.01, 1.01])
def test_tail_renewable_robustness_is_bounded(robustness):
    with pytest.raises(
        ValueError,
        match="tail_renewable_robustness must be between 0 and 1",
    ):
        PuzzleBudgetPolicy(tail_renewable_robustness=robustness)


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_tail_renewable_uncertainty_fraction_is_bounded(fraction):
    with pytest.raises(
        ValueError,
        match="tail_renewable_uncertainty_fraction must be between 0 and 1",
    ):
        PuzzleBudgetPolicy(tail_renewable_uncertainty_fraction=fraction)


def test_tail_renewable_calibration_days_must_be_non_negative():
    with pytest.raises(
        ValueError,
        match="tail_renewable_calibration_days must be >= 0",
    ):
        PuzzleBudgetPolicy(tail_renewable_calibration_days=-1)


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_battery_carbon_priority_fraction_is_bounded(fraction):
    with pytest.raises(
        ValueError,
        match="battery_carbon_priority_fraction must be between 0 and 1",
    ):
        PuzzleBudgetPolicy(battery_carbon_priority_fraction=fraction)


def test_pending_tail_risk_is_observed_without_speculative_reserve():
    midnight = datetime(2023, 1, 1, 0)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        tail_battery_reserve_fraction=0.25,
        tail_renewable_uncertainty_fraction=0.0,
    )
    signal = _FlatForecastSignal(renewable_w=0.0)
    battery = FakeBattery(energy=1000.0)  # 600 Wh usable/deliverable.
    battery._p_discharge_max = 500.0
    long_job = JobRectangle(
        job_id=1, power_w=1000.0, duration_s=9000.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=2500.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    slots = policy._build_envelope(
        midnight, signal, battery, job_rects=[long_job]
    )

    # Tail exposure is 18 × 5min × 1000W = 1500Wh, but the job has not started.
    # Its risk is reported without withholding energy from today's board.  The
    # 500Wh visible result is the 500W device cap over this one-hour window.
    visible_battery_wh = sum(
        slot.battery_power_w * slot.duration_s / 3600.0
        for slot in slots[:12]
    )
    assert policy._last_tail_battery_risk_wh == pytest.approx(1500.0)
    assert policy._last_tail_battery_reserve_wh == 0.0
    assert visible_battery_wh == pytest.approx(500.0)
    assert all(slot.battery_power_w == 0.0 for slot in slots[12:])


def test_running_tail_risk_holds_physical_battery_reserve():
    midnight = datetime(2023, 1, 1, 0)
    running_job = SimpleNamespace(
        job_id=7,
        status="running",
        remaining_time=9000.0,
        num_nodes=1,
    )
    allocation = SimpleNamespace(job=running_job, node_names=["n0"])
    scheduler = SimpleNamespace(
        done=False,
        cluster_type="c",
        job_queue=SimpleNamespace(jobs=[running_job]),
        rm=SimpleNamespace(
            nodes={"n0": SimpleNamespace(measure=lambda: 1000.0)},
            get_running_allocations=lambda: [allocation],
        ),
    )
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        cluster_schedulers=[scheduler],
        tail_battery_reserve_fraction=0.25,
        tail_renewable_uncertainty_fraction=0.0,
    )
    policy._tail_protected_job_keys.add(("c", 7))
    battery = FakeBattery(energy=1000.0)
    battery._p_discharge_max = 500.0

    slots = policy._build_envelope(
        midnight,
        _FlatForecastSignal(renewable_w=0.0),
        battery,
        job_rects=[],
    )

    correction_wh = sum(
        slot.battery_correction_power_w * slot.duration_s / 3600.0
        for slot in slots[:12]
    )
    assert policy._last_tail_battery_risk_wh == pytest.approx(1500.0)
    assert policy._last_tail_battery_reserve_wh == pytest.approx(150.0)
    assert correction_wh == pytest.approx(450.0)


def test_tail_battery_reserve_corrects_protected_running_job_first():
    noon = datetime(2023, 1, 1, 12)
    running_job = SimpleNamespace(
        job_id=7,
        status="running",
        remaining_time=1800.0,
        num_nodes=1,
    )
    node = SimpleNamespace(measure=lambda: 200.0)
    allocation = SimpleNamespace(job=running_job, node_names=["n0"])
    scheduler = SimpleNamespace(
        done=False,
        cluster_type="c",
        job_queue=SimpleNamespace(jobs=[running_job]),
        rm=SimpleNamespace(
            nodes={"n0": node},
            get_running_allocations=lambda: [allocation],
        ),
    )
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        cluster_schedulers=[scheduler],
        tail_battery_reserve_fraction=0.25,
        tail_renewable_uncertainty_fraction=0.0,
    )
    policy._tail_protected_job_keys.add(("c", 7))

    slots = policy._build_envelope(
        noon,
        _FlatForecastSignal(renewable_w=100.0),
        FakeBattery(energy=1000.0),
        job_rects=[],
    )

    # The protected job draws 200W facility-side against 100W PV for 30min.
    # Its 100W deficit is served before ordinary spreading, even though noon is
    # otherwise classified as a charging window.
    assert [slot.battery_correction_power_w for slot in slots[:6]] == [100.0] * 6
    assert [slot.battery_power_w for slot in slots[:6]] == [100.0] * 6
    assert all(slot.charge_reserve_w == 0.0 for slot in slots[:6])


def test_tail_protection_identity_includes_cluster_type():
    def scheduler(cluster_type, measured_power_w):
        job = SimpleNamespace(
            job_id=7,
            status="running",
            remaining_time=1800.0,
            num_nodes=1,
        )
        allocation = SimpleNamespace(job=job, node_names=["n0"])
        return SimpleNamespace(
            done=False,
            cluster_type=cluster_type,
            job_queue=SimpleNamespace(jobs=[job]),
            rm=SimpleNamespace(
                nodes={
                    "n0": SimpleNamespace(
                        measure=lambda: measured_power_w
                    )
                },
                get_running_allocations=lambda: [allocation],
            ),
        )

    protected = scheduler("arm", 200.0)
    same_id_other_cluster = scheduler("gpu", 500.0)
    policy = PuzzleBudgetPolicy(
        plan_horizon_h=1.0,
        slot_duration_s=300,
        cluster_schedulers=[protected, same_id_other_cluster],
    )
    policy._tail_protected_job_keys.add(("arm", 7))

    policy._refresh_tail_protected_job_keys()
    demand_w = policy._protected_running_demand_by_slot(12)

    assert policy._tail_protected_job_keys == {("arm", 7)}
    assert demand_w[:6] == [200.0] * 6
    assert demand_w[6:] == [0.0] * 6


def test_runtime_budget_removes_battery_power_not_backed_by_current_soc():
    now = datetime(2023, 1, 1, 20)
    slot = PowerSlot(
        start_time=now,
        end_time=now + timedelta(minutes=5),
        duration_s=300.0,
        renewable_power_w=50.0,
        battery_power_w=100.0,
        grid_power_w=0.0,
        green_available_w=150.0,
        grid_allocated_w=10.0,
    )
    battery = SimpleNamespace(
        _battery_energy=401.0,
        _battery_energy_max=2000.0,
        _state_of_charge_min=0.2,
        _state_of_charge_max=0.8,
        _discharge_efficiency=0.9,
        _p_discharge_max=300.0,
        soc=lambda: 401.0 / 2000.0,
    )
    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    policy._current_plan = PuzzlePlan(
        plan_start=now,
        plan_end=now + timedelta(hours=1),
        slots=[slot],
        placed_jobs=[],
        unplaced_jobs=[],
        total_budget_kwh=0.0,
        total_planned_kwh=0.0,
    )

    info = policy._budget_at(now, SimpleNamespace(), battery)

    # One internal Wh above minimum at 90% efficiency can supply only 0.9Wh,
    # i.e. 10.8W for five minutes.  The remaining planned 89.2W disappears
    # from both battery source and total budget instead of becoming imaginary.
    assert info.battery_discharge_power_w == pytest.approx(10.8)
    assert info.total_budget_w == pytest.approx(70.8)
    assert (
        info.budget_renewable_w
        + info.battery_discharge_power_w
        + info.grid_power_w
    ) == pytest.approx(info.total_budget_w)


def test_packer_extends_only_oversized_jobs_past_visible_window():
    now = datetime(2023, 1, 1, 12)
    # 4 real slots + 4 extrapolated copies, all 100 W of green.
    slots = [_repair_slot(i, now, 100.0) for i in range(8)]

    oversized = JobRectangle(
        job_id=1, power_w=40.0, duration_s=6 * 300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )
    normal = JobRectangle(
        job_id=2, power_w=80.0, duration_s=2 * 300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=13.3, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    placed, unplaced = policy._pack_jobs(
        slots, [oversized, normal], visible_slots=4,
    )

    assert unplaced == []
    by_id = {r.job_id: r for r in placed}

    # The oversized job (6 slots > 4 visible) starts inside the real window
    # and runs its tail into the extrapolated zone.
    assert not by_id[1].used_grid
    assert by_id[1].start_slot_idx <= 3
    assert by_id[1].end_slot_idx > 3

    # The normal job no longer fits green inside the real window next to the
    # oversized one (40 + 80 > 100), and slots 6-7 of the extrapolated zone
    # are off-limits to it: it must spill to grid inside the real window
    # rather than be planned against repeated forecast data.
    assert by_id[2].used_grid
    assert by_id[2].start_slot_idx <= 3


def test_spill_draws_green_per_slot_and_grid_only_for_deficit():
    # A job that spans a green-rich slot and a green-poor slot cannot fit green
    # in every slot, so it spills — but it must still draw the green each slot
    # has and charge grid only for the per-slot deficit, instead of billing the
    # whole rectangle to grid (which would waste the PV it runs over).
    now = datetime(2023, 1, 1, 12)
    # slot 0: 100 W green (daytime PV), slot 1: 10 W green (night).
    slots = [_repair_slot(0, now, 100.0), _repair_slot(1, now, 10.0)]

    job = JobRectangle(
        job_id=1, power_w=80.0, duration_s=2 * 300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=80.0 * 600.0 / 3600.0, cluster_type="c",
        earliest_start_slot_idx=0,
    )

    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    placed, unplaced = policy._pack_jobs(slots, [job])

    assert unplaced == []
    assert len(placed) == 1
    assert placed[0].used_grid  # crossed a slot it could not fully green

    # Slot 0 (100 W green): the whole 80 W is green, nothing on grid.
    assert slots[0].allocated_w == pytest.approx(80.0)
    assert slots[0].grid_allocated_w == pytest.approx(0.0)
    # Slot 1 (10 W green): 10 W green + 70 W grid deficit — NOT 80 W grid.
    assert slots[1].allocated_w == pytest.approx(10.0)
    assert slots[1].grid_allocated_w == pytest.approx(70.0)


def test_spill_green_reduces_capacity_for_later_jobs():
    # The green a spilled job draws must be visible to jobs packed after it, so
    # green is never handed out twice.
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 100.0), _repair_slot(1, now, 10.0)]

    # Tightest slack first: the spanning job is packed before the green-only one.
    spanning = JobRectangle(
        job_id=1, power_w=80.0, duration_s=2 * 300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=1.0, cluster_type="c",
        earliest_start_slot_idx=0, latest_start_slot_idx=0,
    )
    later = JobRectangle(
        job_id=2, power_w=40.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=1.0, cluster_type="c",
        earliest_start_slot_idx=0, latest_start_slot_idx=1,
    )

    policy = PuzzleBudgetPolicy(slot_duration_s=300)
    placed, unplaced = policy._pack_jobs(slots, [spanning, later])

    assert unplaced == []
    by_id = {r.job_id: r for r in placed}
    # Spanning job took 80 W of slot 0's green; only 20 W remains there.  The
    # 40 W later job must therefore see just 20 W free (not the full 100 W —
    # that would be double-counting), so it spills, drawing the 20 W leftover
    # green and 20 W grid.  If green were not reduced it would fit fully green.
    assert by_id[1].start_slot_idx == 0
    assert by_id[2].used_grid
    # Slot 0 green is fully used (80 + 20) but never over its 100 W ceiling.
    assert slots[0].allocated_w == pytest.approx(100.0)
    assert slots[0].grid_allocated_w == pytest.approx(20.0)


def test_normal_grid_placement_uses_first_admissible_start():
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 0.0), _repair_slot(1, now, 0.0)]
    slots[0].carbon_intensity = 100.0
    slots[1].carbon_intensity = 40.0

    job = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 / 12.0, cluster_type="c",
    )

    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        carbon_regret_threshold_g_per_h=None,
    )
    placed, unplaced = policy._pack_jobs(slots, [job])

    assert unplaced == []
    # Both starts have carbon-permitted capacity. Normal earliest-fit packing
    # does not postpone an admissible start for a later, cleaner one.
    assert placed[0].start_slot_idx == 0
    assert placed[0].grid_reason == "low_carbon"
    assert placed[0].planned_grid_carbon_g == pytest.approx(100 / 12 / 1000 * 100)
    assert slots[0].grid_allocated_w == 100.0
    assert slots[1].grid_allocated_w == 0.0


def test_carbon_aware_spill_can_choose_faster_higher_power_dvfs():
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 0.0), _repair_slot(1, now, 0.0)]
    slots[0].carbon_intensity = 20.0
    slots[1].carbon_intensity = 300.0
    job = JobRectangle(
        job_id=1, power_w=50.0, duration_s=600.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=50.0 / 6.0, cluster_type="c",
        earliest_start_slot_idx=0, latest_start_slot_idx=0,
        dvfs_variants=[
            (1.0, 50.0, 600.0),
            (2.0, 80.0, 300.0),
        ],
    )

    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        carbon_regret_threshold_g_per_h=None,
    )
    placed, unplaced = policy._pack_jobs(slots, [job])

    assert unplaced == []
    assert placed[0].frequency_ghz == 2.0
    assert placed[0].end_slot_idx == 0
    assert placed[0].grid_reason == "low_carbon"


def test_high_carbon_grid_waits_until_sla_enters_real_window():
    now = datetime(2023, 1, 1, 12)

    def high_carbon_slots():
        out = [_repair_slot(0, now, 0.0), _repair_slot(1, now, 0.0)]
        for slot in out:
            slot.grid_power_w = 0.0
            slot.carbon_intensity = 300.0
        return out

    optional = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 / 12.0, cluster_type="c",
        latest_start_slot_idx=2,
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        carbon_regret_threshold_g_per_h=None,
    )
    placed, unplaced = policy._pack_jobs(
        high_carbon_slots(), [optional], visible_slots=2,
    )
    assert placed == []
    assert unplaced == [optional]

    urgent = replace(optional, job_id=2, latest_start_slot_idx=1)
    placed, unplaced = policy._pack_jobs(
        high_carbon_slots(), [urgent], visible_slots=2,
    )
    assert unplaced == []
    assert placed[0].used_grid is True
    assert placed[0].grid_reason == "sla_emergency"


def test_carbon_regret_advances_job_when_waiting_saves_too_little():
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 0.0), _repair_slot(1, now, 0.0)]
    for slot in slots:
        slot.grid_power_w = 0.0
        slot.carbon_intensity = 300.0

    job = JobRectangle(
        job_id=1, power_w=100.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=100.0 / 12.0, cluster_type="c",
        latest_start_slot_idx=2,
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        carbon_regret_threshold_g_per_h=50.0,
    )

    placed, unplaced = policy._pack_jobs(slots, [job], visible_slots=2)

    assert unplaced == []
    assert placed[0].start_slot_idx == 0
    assert placed[0].grid_reason == "carbon_regret"


@pytest.mark.parametrize("regret_threshold, advances", [(50.0, False), (2000.0, True)])
def test_carbon_regret_compares_saving_rate_before_admitting_unbounded_grid(
    regret_threshold, advances,
):
    now = datetime(2023, 1, 1, 12)
    slots = [_repair_slot(0, now, 0.0), _repair_slot(1, now, 0.0)]
    for slot in slots:
        slot.grid_power_w = 0.0
    slots[0].carbon_intensity = 300.0
    slots[1].carbon_intensity = 200.0

    job = JobRectangle(
        job_id=1, power_w=10_000.0, duration_s=300.0, num_nodes=1,
        frequency_ghz=1.0, energy_wh=10_000.0 / 12.0, cluster_type="c",
    )
    policy = PuzzleBudgetPolicy(
        slot_duration_s=300,
        carbon_regret_threshold_g_per_h=regret_threshold,
    )

    placed, unplaced = policy._pack_jobs(slots, [job])

    # Both slots exceed the normal carbon threshold. Waiting five minutes
    # saves 83 1/3 g, or 1000 g/h; only the larger regret threshold admits an
    # early exceptional-grid start. Without an imminent SLA, the job waits.
    if advances:
        assert unplaced == []
        assert placed[0].start_slot_idx == 0
        assert placed[0].grid_reason == "carbon_regret"
        assert placed[0].planned_grid_carbon_g == pytest.approx(250.0)
        assert slots[0].grid_allocated_w == 10_000.0
    else:
        assert placed == []
        assert unplaced == [job]
        assert slots[0].grid_allocated_w == 0.0
    assert slots[1].grid_allocated_w == 0.0
