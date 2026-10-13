import importlib
from pathlib import Path

import pytest

from scripts import evaluation_campaign as base
from scripts import evaluation_campaign_v8 as campaign


@pytest.fixture
def configured_campaign():
    importlib.reload(base)
    campaign.configure()
    yield
    importlib.reload(base)


def _grid_limit(case_name: str) -> float:
    case = campaign.cases()[case_name]
    args = base.case_args(
        case,
        Path("results/evaluation_campaign/v8_grid_supply/inputs/deucalion_macc_five_day"),
        "test-run",
    )
    occurrences = [
        float(args[index + 1])
        for index, value in enumerate(args[:-1])
        if value == "--grid-power-limit"
    ]
    return occurrences[-1]


def test_grid_baseline_covers_full_deucalion_macc_power(configured_campaign):
    for case_name in ("grid_fifo", "grid_priority", "grid_eft", "grid_puzzle"):
        assert _grid_limit(case_name) == 900_000


def test_renewable_profiles_keep_experimental_grid_cap(configured_campaign):
    for case_name in ("solar_fifo", "solar_puzzle", "solar_wind_eft"):
        assert _grid_limit(case_name) == 350_000


def test_grid_puzzle_uses_grid_at_first_feasible_slot(configured_campaign):
    cases = campaign.cases()
    grid_puzzle = cases["grid_puzzle"]
    solar_puzzle = cases["solar_puzzle"]

    assert "--no-puzzle-carbon-aware-placement" in grid_puzzle.extra
    assert "--no-puzzle-carbon-aware-placement" not in solar_puzzle.extra
    assert grid_puzzle.energy == "zero"
    assert "--carbon-grid-threshold" in grid_puzzle.extra
    assert "2000.0" in grid_puzzle.extra

    jobs_dir = Path("results/evaluation_campaign/v8_grid_supply/inputs/deucalion_macc_five_day")
    args = base.case_args(grid_puzzle, jobs_dir, "grid-puzzle-test")
    assert "--renewable-source-scale" in args
    assert "pv=0" in args
    assert "--no-puzzle-carbon-aware-placement" in args
