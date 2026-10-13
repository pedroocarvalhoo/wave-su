"""Translate dashboard requests into simulator arguments without changing policies."""

from pathlib import Path


def build_command(req, python, script):
    cmd = [python, "-u", str(script), req.system]
    budget_policy = req.budget_policy

    # Map request fields to CLI args
    cmd += ["--scheduling-policy", req.scheduling_policy]
    cmd += ["--dvfs-policy", req.dvfs_policy]
    cmd += ["--budget-policy", budget_policy]
    cmd += ["--allocation-strategy", req.allocation_strategy]
    cmd += ["--carbon-grid-threshold", str(req.carbon_grid_threshold)]
    cmd += ["--grid-power-limit", str(req.grid_power_limit)]
    cmd += ["--microgrid-mode", req.microgrid_mode]
    cmd += ["--pue", str(req.pue)]
    if req.allow_grid_charging:
        cmd.append("--allow-grid-charging")
    else:
        cmd.append("--no-grid-charging")
    if req.renewable_charge_fraction is not None:
        cmd += ["--renewable-charge-fraction", str(req.renewable_charge_fraction)]
    if req.pv_scale is not None:
        cmd += ["--pv-scale", str(req.pv_scale)]
    cmd += ["--renewable-sources", ",".join(req.renewable_sources)]
    for source, scale in sorted(req.renewable_source_scales.items()):
        cmd += ["--renewable-source-scale", f"{source}={scale}"]
    cmd += ["--battery-soc-min", str(req.battery_soc_min)]
    cmd += ["--battery-soc-max", str(req.battery_soc_max)]
    cmd += ["--battery-capacity-kwh", str(req.battery_capacity_kwh)]
    cmd += ["--battery-p-charge-max", str(req.battery_p_charge_max)]
    cmd += ["--battery-p-discharge-max", str(req.battery_p_discharge_max)]
    cmd += ["--low-carbon-grid-fraction", str(req.low_carbon_grid_fraction)]
    cmd += ["--high-carbon-grid-fraction", str(req.high_carbon_grid_fraction)]
    cmd += ["--day-grid-fraction", str(req.day_grid_fraction)]
    cmd += ["--night-grid-fraction", str(req.night_grid_fraction)]
    cmd += ["--eft-dvfs-alpha", str(req.eft_dvfs_alpha)]
    cmd += ["--eft-dvfs-beta", str(req.eft_dvfs_beta)]
    if req.puzzle_allow_front_run:
        cmd.append("--puzzle-allow-front-run")
    if req.puzzle_fallback_dispatch is not None:
        cmd.append("--puzzle-fallback-dispatch" if req.puzzle_fallback_dispatch else "--no-puzzle-fallback-dispatch")
    if req.puzzle_repack_repair:
        cmd.append("--puzzle-repack-repair")
    if not req.puzzle_carbon_aware_placement:
        cmd.append("--no-puzzle-carbon-aware-placement")
    if not req.puzzle_global_green_board:
        cmd.append("--no-puzzle-global-green-board")
    cmd += [
        "--puzzle-forecast-tail-carbon-multiplier",
        str(req.puzzle_forecast_tail_carbon_multiplier),
    ]
    cmd += [
        "--puzzle-tail-renewable-uncertainty-fraction",
        str(req.puzzle_tail_renewable_uncertainty_fraction),
    ]
    cmd += [
        "--puzzle-tail-renewable-robustness",
        str(req.puzzle_tail_renewable_robustness),
    ]
    cmd += [
        "--puzzle-tail-renewable-calibration-days",
        str(req.puzzle_tail_renewable_calibration_days),
    ]
    if req.puzzle_battery_carbon_aware_dispatch:
        cmd.append("--puzzle-battery-carbon-aware-dispatch")
    else:
        cmd.append("--no-puzzle-battery-carbon-aware-dispatch")
    cmd += [
        "--puzzle-battery-carbon-priority-fraction",
        str(req.puzzle_battery_carbon_priority_fraction),
    ]
    cmd += [
        "--puzzle-tail-battery-reserve-fraction",
        str(req.puzzle_tail_battery_reserve_fraction),
    ]
    if req.puzzle_carbon_regret_g_per_h is None:
        cmd.append("--no-puzzle-carbon-regret")
    else:
        cmd += [
            "--puzzle-carbon-regret-g-per-h",
            str(req.puzzle_carbon_regret_g_per_h),
        ]
    cmd += [
        "--puzzle-sla-start-safety-margin-h",
        str(req.puzzle_sla_start_safety_margin_h),
    ]
    if req.puzzle_defer_carbon_margin is not None:
        cmd += ["--puzzle-defer-carbon-margin", str(req.puzzle_defer_carbon_margin)]
    if req.sla_grid_override is not None:
        cmd.append("--sla-grid-override" if req.sla_grid_override else "--no-sla-grid-override")
    cmd += ["--microgrid-step-size", str(req.microgrid_step_size)]
    cmd += ["--energy-scheduler-step-size", str(req.energy_scheduler_step_size)]
    cmd += ["--runtime-prediction-error-rate", str(req.runtime_prediction_error_rate)]
    cmd += ["--runtime-prediction-seed", str(req.runtime_prediction_seed)]

    if req.adaptive_step:
        cmd.append("--adaptive-step")
    else:
        cmd.append("--no-adaptive-step")

    if not req.start_nodes_off:
        cmd.append("--start-nodes-on")

    cmd += ["--node-power-on-gate", req.node_power_on_gate]

    if req.no_plots:
        cmd.append("--no-plots")

    if req.plot_workers is not None:
        cmd += ["--plot-workers", str(req.plot_workers)]

    if req.scheduler_debug_log:
        cmd.append("--scheduler-debug-log")
    else:
        cmd.append("--no-scheduler-debug-log")

    if req.verbose_scheduler:
        cmd.append("--verbose-scheduler")
    else:
        cmd.append("--silent-scheduler")

    if req.jobs_dir:
        cmd += ["--jobs-dir", str(Path(req.jobs_dir).expanduser())]
    if req.run_tag:
        cmd += ["--run-tag", req.run_tag]
    if req.sla_policy:
        cmd += ["--sla-policy", req.sla_policy]
    if req.keep_idle_nodes_on:
        cmd.append("--keep-idle-nodes-on")
    if req.no_snapshot:
        cmd.append("--no-snapshot")
    return cmd
