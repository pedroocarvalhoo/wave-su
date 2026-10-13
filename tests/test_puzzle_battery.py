from __future__ import annotations

from vessim.wave.energy.puzzle_battery import plan_tail_battery_dispatch


def _dispatch(*, protected, carbon, energy_wh, carbon_aware=True):
    return plan_tail_battery_dispatch(
        renewable_forecast_w=[0.0] * len(protected),
        protected_demand_w=protected,
        charging_slots=set(),
        deliverable_energy_wh=energy_wh,
        future_tail_risk_wh=0.0,
        reserve_fraction=0.0,
        discharge_power_max_w=100.0,
        slot_duration_s=3600.0,
        carbon_intensity_g_per_kwh=carbon,
        carbon_aware=carbon_aware,
    )


def test_carbon_aware_battery_corrects_highest_carbon_protected_slot_first():
    dispatch = _dispatch(
        protected=[100.0, 100.0, 100.0],
        carbon=[50.0, 300.0, 150.0],
        energy_wh=100.0,
    )

    assert dispatch.correction_power_w == (0.0, 100.0, 0.0)
    assert dispatch.power_w == (0.0, 100.0, 0.0)


def test_protected_recourse_precedes_higher_carbon_ordinary_slot():
    dispatch = _dispatch(
        protected=[100.0, 0.0],
        carbon=[50.0, 300.0],
        energy_wh=100.0,
    )

    assert dispatch.correction_power_w == (100.0, 0.0)
    assert dispatch.power_w == (100.0, 0.0)


def test_remaining_battery_targets_highest_marginal_carbon_slots():
    dispatch = _dispatch(
        protected=[0.0, 0.0, 0.0],
        carbon=[50.0, 300.0, 150.0],
        energy_wh=200.0,
    )

    assert dispatch.correction_power_w == (0.0, 0.0, 0.0)
    assert dispatch.power_w == (0.0, 100.0, 100.0)


def test_compatibility_mode_retains_equal_water_fill():
    dispatch = _dispatch(
        protected=[0.0, 0.0],
        carbon=[50.0, 300.0],
        energy_wh=100.0,
        carbon_aware=False,
    )

    assert dispatch.power_w == (50.0, 50.0)


def test_partial_carbon_priority_preserves_temporal_coverage():
    dispatch = plan_tail_battery_dispatch(
        renewable_forecast_w=[0.0, 0.0],
        protected_demand_w=[0.0, 0.0],
        charging_slots=set(),
        deliverable_energy_wh=100.0,
        future_tail_risk_wh=0.0,
        reserve_fraction=0.0,
        discharge_power_max_w=100.0,
        slot_duration_s=3600.0,
        carbon_intensity_g_per_kwh=[50.0, 300.0],
        carbon_aware=True,
        carbon_priority_fraction=0.25,
    )

    assert dispatch.power_w == (37.5, 62.5)
