from __future__ import annotations

from types import SimpleNamespace

from datetime import datetime, timedelta

import pandas as pd
import pytest

from scripts.simulation_energy import (
    PUZZLE_ACTIVE_FORECAST_FILE,
    build_budget_policy,
    effective_grid_power_limit,
    export_puzzle_forecast,
    parse_args,
    resolve_allow_grid_charging,
)
from vessim.wave.energy.puzzle_budget_policy import PowerSlot, PuzzleBudgetPolicy, PuzzlePlan


def _args(**overrides):
    defaults = {
        "microgrid_mode": "grid-connected",
        "grid_power_limit": 200_000.0,
        "budget_policy": "carbon-minimizing",
        "carbon_grid_threshold": 160.0,
        "renewable_charge_fraction": 0.7,
        "low_carbon_grid_fraction": 1.0,
        "high_carbon_grid_fraction": 0.0,
        "day_grid_fraction": 0.5,
        "night_grid_fraction": 1.0,
        "energy_scheduler_step_size": 300,
        "pue": 1.3,
        "puzzle_repack_repair": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_islanded_mode_forces_effective_grid_limit_to_zero():
    args = _args(microgrid_mode="islanded", grid_power_limit=200_000.0)

    assert effective_grid_power_limit(args) == 0.0
    assert build_budget_policy(args).grid_power_limit == 0.0


def test_grid_connected_mode_keeps_configured_grid_limit():
    args = _args(microgrid_mode="grid-connected", grid_power_limit=200_000.0)

    assert effective_grid_power_limit(args) == 200_000.0
    assert build_budget_policy(args).grid_power_limit == 200_000.0


def test_puzzle_defaults_to_renewable_only_battery_charging():
    args = _args(budget_policy="puzzle", allow_grid_charging=None)

    assert resolve_allow_grid_charging(args) is False


def test_puzzle_can_explicitly_enable_grid_charging():
    args = _args(budget_policy="puzzle", allow_grid_charging=True)

    assert resolve_allow_grid_charging(args) is True


def test_puzzle_policy_receives_facility_pue():
    policy = build_budget_policy(_args(budget_policy="puzzle"))

    assert policy.pue == 1.3
    assert policy.carbon_aware_placement is True
    assert policy.global_green_board is True
    assert policy.forecast_tail_carbon_multiplier == 1.25
    assert policy.sla_start_safety_margin_h == 6.0
    assert policy.carbon_regret_threshold_g_per_h == 1.0
    assert policy.tail_battery_reserve_fraction == 0.10
    assert policy.tail_renewable_uncertainty_fraction == 0.20
    assert policy.tail_renewable_robustness == 0.0
    assert policy.tail_renewable_calibration_days == 90.0
    assert policy.battery_carbon_aware_dispatch is True
    assert policy.battery_carbon_priority_fraction == 0.0


@pytest.mark.parametrize("replan_minutes", [1.5, 5])
def test_active_puzzle_forecast_is_unsampled_and_stops_at_next_replan(
    tmp_path, replan_minutes,
):
    start = datetime(2023, 1, 1, 12)

    def slot(offset_minutes, allocated, grid, green):
        slot_start = start + timedelta(minutes=offset_minutes)
        return PowerSlot(
            start_time=slot_start,
            end_time=slot_start + timedelta(minutes=5),
            duration_s=300.0,
            renewable_power_w=green,
            battery_power_w=0.0,
            grid_power_w=0.0,
            green_available_w=green,
            allocated_w=allocated,
            grid_allocated_w=grid,
        )

    plan0 = PuzzlePlan(
        plan_start=start,
        plan_end=start + timedelta(hours=1),
        slots=[slot(0, 65.0, 0.0, 130.0), slot(5, 65.0, 0.0, 130.0)],
        placed_jobs=[], unplaced_jobs=[],
        total_budget_kwh=0.0, total_planned_kwh=0.0,
    )
    plan1_start = start + timedelta(minutes=replan_minutes)
    plan1 = PuzzlePlan(
        plan_start=plan1_start,
        plan_end=plan1_start + timedelta(minutes=5),
        slots=[
            PowerSlot(
                start_time=plan1_start,
                end_time=plan1_start + timedelta(minutes=5),
                duration_s=300.0,
                renewable_power_w=65.0,
                battery_power_w=0.0,
                grid_power_w=0.0,
                green_available_w=65.0,
                allocated_w=52.0,
                grid_allocated_w=13.0,
            ),
        ],
        placed_jobs=[], unplaced_jobs=[],
        total_budget_kwh=0.0, total_planned_kwh=0.0,
    )

    policy = PuzzleBudgetPolicy(pue=1.3, plan_history_limit=1)
    policy.history.record(plan0)
    policy.history.record(plan1)
    context = SimpleNamespace(
        results_dir=str(tmp_path),
        energy_scheduler=SimpleNamespace(budget_policy=policy, pue=1.3),
    )
    export_puzzle_forecast(context, progress=lambda message: None)

    rows = pd.read_csv(tmp_path / PUZZLE_ACTIVE_FORECAST_FILE)
    detailed = pd.read_csv(tmp_path / "puzzle_forecast_slots.csv")
    assert detailed["plan_id"].tolist() == [1]
    assert list(zip(rows["plan_id"], rows["slot_idx"])) == [(0, 0), (1, 0)]
    assert pd.to_datetime(rows.loc[0, "end_time"]) == plan1_start
    assert pd.to_datetime(rows.loc[1, "start_time"]) == plan1_start
    assert rows["planned_it_power_w"].tolist() == [50.0, 50.0]
    assert rows.loc[1, "runtime_budget_it_w"] == 60.0

    export_puzzle_forecast(context, progress=lambda message: None)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / PUZZLE_ACTIVE_FORECAST_FILE), rows,
    )


def test_parse_args_accepts_renewable_charge_fraction_and_legacy_alias():
    renewable_sources_config = {"pv": {"enabled_by_default": True}}

    new_args = parse_args(
        ["test_system"],
        renewable_sources_config,
        ["--renewable-charge-fraction", "0.4"],
    )
    legacy_args = parse_args(
        ["test_system"],
        renewable_sources_config,
        ["--pv-charge-fraction", "0.5"],
    )

    assert new_args.renewable_charge_fraction == 0.4
    assert legacy_args.renewable_charge_fraction == 0.5


def test_parse_args_can_disable_carbon_aware_puzzle_for_ab_comparison():
    args = parse_args(
        ["test_system"],
        {"pv": {"enabled_by_default": True}},
        [
            "--no-puzzle-carbon-aware-placement",
            "--puzzle-forecast-tail-carbon-multiplier", "1.5",
            "--puzzle-sla-start-safety-margin-h", "4",
        ],
    )

    assert args.puzzle_carbon_aware_placement is False
    assert args.puzzle_forecast_tail_carbon_multiplier == 1.5
    assert args.puzzle_sla_start_safety_margin_h == 4.0


def test_parse_args_can_restore_cluster_local_puzzle_board():
    args = parse_args(
        ["test_system"],
        {"pv": {"enabled_by_default": True}},
        ["--no-puzzle-global-green-board"],
    )

    assert args.puzzle_global_green_board is False


def test_parse_args_can_tune_or_disable_puzzle_carbon_regret():
    sources = {"pv": {"enabled_by_default": True}}
    tuned = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-carbon-regret-g-per-h", "75"],
    )
    disabled = parse_args(
        ["test_system"],
        sources,
        ["--no-puzzle-carbon-regret"],
    )

    assert tuned.puzzle_carbon_regret_g_per_h == 75.0
    assert disabled.puzzle_carbon_regret_g_per_h is None


def test_parse_args_can_tune_or_disable_puzzle_tail_battery_reserve():
    sources = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], sources, [])
    tuned = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-battery-reserve-fraction", "0.4"],
    )
    disabled = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-battery-reserve-fraction", "0"],
    )

    assert default.puzzle_tail_battery_reserve_fraction == 0.10
    assert tuned.puzzle_tail_battery_reserve_fraction == 0.4
    assert disabled.puzzle_tail_battery_reserve_fraction == 0.0


def test_parse_args_can_tune_or_disable_puzzle_tail_renewable_uncertainty():
    sources = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], sources, [])
    tuned = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-renewable-uncertainty-fraction", "0.35"],
    )
    disabled = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-renewable-uncertainty-fraction", "0"],
    )

    assert default.puzzle_tail_renewable_uncertainty_fraction == 0.20
    assert tuned.puzzle_tail_renewable_uncertainty_fraction == 0.35
    assert disabled.puzzle_tail_renewable_uncertainty_fraction == 0.0


def test_parse_args_can_tune_puzzle_tail_renewable_robustness():
    sources = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], sources, [])
    pessimistic = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-renewable-robustness", "1"],
    )

    assert default.puzzle_tail_renewable_robustness == 0.0
    assert pessimistic.puzzle_tail_renewable_robustness == 1.0


def test_parse_args_can_tune_or_disable_tail_renewable_calibration():
    sources = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], sources, [])
    tuned = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-renewable-calibration-days", "180"],
    )
    disabled = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-tail-renewable-calibration-days", "0"],
    )

    assert default.puzzle_tail_renewable_calibration_days == 90.0
    assert tuned.puzzle_tail_renewable_calibration_days == 180.0
    assert disabled.puzzle_tail_renewable_calibration_days == 0.0


def test_parse_args_can_disable_carbon_aware_battery_dispatch():
    sources = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], sources, [])
    disabled = parse_args(
        ["test_system"],
        sources,
        ["--no-puzzle-battery-carbon-aware-dispatch"],
    )

    assert default.puzzle_battery_carbon_aware_dispatch is True
    assert disabled.puzzle_battery_carbon_aware_dispatch is False


def test_parse_args_can_tune_battery_carbon_priority_fraction():
    sources = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], sources, [])
    tuned = parse_args(
        ["test_system"],
        sources,
        ["--puzzle-battery-carbon-priority-fraction", "0.4"],
    )

    assert default.puzzle_battery_carbon_priority_fraction == 0.0
    assert tuned.puzzle_battery_carbon_priority_fraction == 0.4


def test_parse_args_keeps_puzzle_fallback_contextual_unless_explicit():
    source_config = {"pv": {"enabled_by_default": True}}
    default = parse_args(["test_system"], source_config, [])
    enabled = parse_args(
        ["test_system"], source_config, ["--puzzle-fallback-dispatch"]
    )
    disabled = parse_args(
        ["test_system"], source_config, ["--no-puzzle-fallback-dispatch"]
    )

    assert default.puzzle_fallback_dispatch is None
    assert enabled.puzzle_fallback_dispatch is True
    assert disabled.puzzle_fallback_dispatch is False
