"""Unit conversion checks for evaluation monitor diagnostics."""

import json

import pandas as pd

from scripts.plot_evaluation import diagnostic_row


def test_diagnostic_row_separates_grid_budget_from_physical_import(tmp_path):
    config = {
        "sim_start": "2023-06-01 12:00:00",
        "microgrid_step_size_s": 60,
        "budget_policy_details": {"effective_grid_power_limit": 200},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    frame = pd.DataFrame({
        "e_delta": [-12000, -6000, -18000],
        "renewable_generation_w": [1000, 2000, 3000],
        "actor_states.Generator-pv.p": [1000, 2000, 3000],
    }, index=["2023-06-01 12:00:00", "2023-06-02 12:00:00", "2023-06-09 12:00:00"])
    frame.to_csv(tmp_path / "energy_simulation_result.csv")
    row = diagnostic_row(tmp_path)
    assert row["peak_grid_import_kw"] == 0.3
    assert row["samples_above_scheduler_grid_allowance"] == 1
    assert row["max_excess_above_scheduler_grid_allowance_kw"] == 0.1
    assert row["physical_generation_first_7d_kwh"] == 0.05
