"""Tests for the green-board redesign of the Puzzle Scheduler.

Covers the green-only packing board, the grid-overflow lane, SLA start-deadline
windows, DVFS-deformable pieces, and the SoC-bounded battery reservoir.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from vessim.wave.energy.puzzle_budget_policy import (
    JobRectangle,
    PowerSlot,
    PuzzleBudgetPolicy,
)

NOW = datetime(2023, 1, 1, 12)


def _slot(idx: int, green_w: float, grid_w: float = 0.0) -> PowerSlot:
    return PowerSlot(
        start_time=NOW + timedelta(minutes=5 * idx),
        end_time=NOW + timedelta(minutes=5 * (idx + 1)),
        duration_s=300.0,
        renewable_power_w=green_w,
        battery_power_w=0.0,
        grid_power_w=grid_w,
        green_available_w=green_w,
    )


def _rect(power_w: float, duration_s: float = 300.0, **kwargs) -> JobRectangle:
    return JobRectangle(
        job_id=kwargs.pop("job_id", 1),
        power_w=power_w,
        duration_s=duration_s,
        num_nodes=kwargs.pop("num_nodes", 1),
        frequency_ghz=kwargs.pop("frequency_ghz", 1.0),
        energy_wh=power_w * duration_s / 3600.0,
        cluster_type=kwargs.pop("cluster_type", "c"),
        arrival_time=kwargs.pop("arrival_time", NOW),
        **kwargs,
    )


def test_grid_is_excluded_from_the_packing_ceiling():
    # green=100, grid=1000; a 150W job exceeds green but not green+grid.  The
    # green board caps at its 100W ceiling — proving grid is not part of it —
    # so the job fills the 100W of green it can and only the 50W deficit spills
    # to the grid lane (power is fungible per slot; a job is never billed whole
    # to grid when green is available).
    slots = [_slot(0, green_w=100.0, grid_w=1000.0)]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    placed, unplaced = policy._pack_jobs(slots, [_rect(150.0)])

    assert unplaced == []
    assert len(placed) == 1
    assert placed[0].used_grid is True
    assert slots[0].allocated_w == 100.0        # green board filled to ceiling
    assert slots[0].grid_allocated_w == 50.0     # only the deficit on grid


def test_job_that_fits_green_stays_green():
    slots = [_slot(0, green_w=100.0)]
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    placed, unplaced = policy._pack_jobs(slots, [_rect(50.0)])

    assert unplaced == []
    assert placed[0].used_grid is False
    assert slots[0].allocated_w == 50.0
    assert slots[0].grid_allocated_w == 0.0


def test_sla_window_forces_grid_spill():
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    # Without an SLA bound the 50W job slides to the roomy second slot (green).
    slots = [_slot(0, green_w=40.0), _slot(1, green_w=100.0)]
    placed, _ = policy._pack_jobs(slots, [_rect(50.0, latest_start_slot_idx=None)])
    assert placed[0].used_grid is False
    assert placed[0].start_slot_idx == 1

    # With a start-deadline of slot 0 it can only start in the 40W slot, which
    # cannot hold it, so it spills to grid rather than violating the SLA.
    slots = [_slot(0, green_w=40.0), _slot(1, green_w=100.0)]
    placed, _ = policy._pack_jobs(slots, [_rect(50.0, latest_start_slot_idx=0)])
    assert placed[0].used_grid is True
    assert placed[0].start_slot_idx == 0


def test_dvfs_variant_fits_a_peak_the_base_shape_cannot():
    # Base 150W shape exceeds the 100W board, but a deformed 80W/2-slot variant
    # fits.  The packer must choose the fitting variant and keep the job green.
    slots = [_slot(0, green_w=100.0), _slot(1, green_w=100.0)]
    rect = _rect(
        150.0,
        duration_s=300.0,
        dvfs_variants=[(2.0, 150.0, 300.0), (1.0, 80.0, 600.0)],
    )
    policy = PuzzleBudgetPolicy(slot_duration_s=300)

    placed, unplaced = policy._pack_jobs(slots, [rect])

    assert unplaced == []
    assert placed[0].used_grid is False
    assert placed[0].frequency_ghz == 1.0
    assert placed[0].power_w == 80.0
    assert placed[0].end_slot_idx == 1


def test_battery_reservoir_never_exceeds_usable_soc():
    # Zero PV everywhere -> every slot is discharge-eligible.  Total planned
    # battery energy over the horizon must not exceed the usable SoC energy.
    usable_wh = 1_000_000.0
    battery = SimpleNamespace(
        _battery_energy=usable_wh,
        _battery_energy_max=2_000_000.0,
        _state_of_charge_min=0.0,
        _state_of_charge_max=1.0,
        _p_discharge_max=500_000.0,
        _p_charge_max=500_000.0,
        soc=lambda: 0.5,
    )

    def forecast(start_time, end_time, column, frequency, resample_method):
        return {i: 0.0 for i in range(300)}

    signal = SimpleNamespace(forecast=forecast)
    policy = PuzzleBudgetPolicy(slot_duration_s=300, plan_horizon_h=24.0)

    slots = policy._build_envelope(NOW, signal, battery)

    total_battery_wh = sum(
        s.battery_power_w * s.duration_s / 3600.0 for s in slots
    )
    assert total_battery_wh <= usable_wh + 1e-6
    assert all(s.battery_power_w <= battery._p_discharge_max for s in slots)
