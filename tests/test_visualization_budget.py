from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from vessim.visualization import (
    _budget_source_it_series,
    plot_budget_distribution,
    plot_budget_vs_forecast,
)


def test_budget_sources_are_pue_normalized_to_the_scheduler_budget():
    df = pd.DataFrame({
        "budget_renewable_w": [130.0, 0.0],
        "budget_battery_w": [0.0, 13.0],
        "budget_grid_w": [0.0, 52.0],
        "budget_scheduler_w": [100.0, 50.0],
        "budget_pue": [1.3, 1.3],
    })

    sources = _budget_source_it_series(df)
    total = sources["renewable"] + sources["battery"] + sources["grid"]

    assert total.tolist() == [100.0, 50.0]
    assert total.tolist() == df["budget_scheduler_w"].tolist()


def test_budget_distribution_plot_uses_consistent_it_series(tmp_path):
    index = pd.date_range("2023-01-01 12:00:00", periods=2, freq="5min")
    df = pd.DataFrame({
        "cluster_a_budget": [100.0, 50.0],
        "cluster_a_power": [80.0, 60.0],
        "budget_renewable_w": [130.0, 0.0],
        "budget_battery_w": [0.0, 13.0],
        "budget_grid_w": [0.0, 52.0],
        "budget_scheduler_w": [100.0, 50.0],
        "budget_pue": [1.3, 1.3],
    }, index=index)
    output = tmp_path / "budget_distribution.png"

    plot_budget_distribution(
        df, [SimpleNamespace(cluster_type="cluster_a")], str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_budget_vs_forecast_prefers_unsampled_active_timeline(tmp_path):
    index = pd.date_range("2023-01-01 12:00:00", periods=2, freq="5min")
    pd.DataFrame({
        "plan_id": [0, 0, 0, 0],
        "cluster_type": ["__total__", "__total__", "cluster_a", "cluster_a"],
        "slot_idx": [0, 1, 0, 1],
        "start_time": [index[0], index[1], index[0], index[1]],
        "end_time": [index[1], index[1] + pd.Timedelta(minutes=5)] * 2,
        "allocated_w": [104.0, 65.0, 104.0, 65.0],
        "grid_allocated_w": [0.0, 13.0, 0.0, 13.0],
    }).to_csv(tmp_path / "puzzle_forecast_slots.csv", index=False)
    pd.DataFrame({
        "plan_id": [0, 1],
        "slot_idx": [0, 0],
        "start_time": index,
        "end_time": [index[1], index[1] + pd.Timedelta(minutes=5)],
        "planned_it_power_w": [80.0, 60.0],
    }).to_csv(tmp_path / "puzzle_active_forecast.csv", index=False)
    df = pd.DataFrame({
        "cluster_a_power": [75.0, 65.0],
        "budget_scheduler_w": [100.0, 60.0],
        "budget_pue": [1.3, 1.3],
    }, index=index)
    output = tmp_path / "budget_vs_forecast.png"

    plot_budget_vs_forecast(str(tmp_path), df=df, output_file=str(output))

    assert output.exists()
    assert output.stat().st_size > 0
