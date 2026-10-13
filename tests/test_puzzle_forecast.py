from __future__ import annotations

import pytest

from vessim.wave.energy.puzzle_forecast import (
    calibrate_tail_renewable_uncertainty,
)


def test_tail_calibration_learns_shortfall_by_lead_window():
    history = [100.0] * 24 + [80.0] * 24 + [64.0] * 24

    points = calibrate_tail_renewable_uncertainty(
        history,
        slots_per_window=24,
        max_lead_windows=2,
    )

    assert points[0].pessimistic_fraction == pytest.approx(0.20)
    assert points[1].pessimistic_fraction == pytest.approx(0.36)
    assert points[0].optimistic_fraction == 0.0
    assert points[1].optimistic_fraction == 0.0
    assert all(point.used_fallback is False for point in points)


def test_tail_calibration_learns_asymmetric_upside():
    history = [100.0] * 24 + [120.0] * 24 + [144.0] * 24

    points = calibrate_tail_renewable_uncertainty(
        history,
        slots_per_window=24,
        max_lead_windows=2,
    )

    assert points[0].pessimistic_fraction == 0.0
    assert points[1].pessimistic_fraction == 0.0
    assert points[0].optimistic_fraction == pytest.approx(0.20)
    assert points[1].optimistic_fraction == pytest.approx(0.44)


def test_tail_calibration_uses_v05_fallback_without_enough_history():
    points = calibrate_tail_renewable_uncertainty(
        [100.0] * 12,
        slots_per_window=24,
        max_lead_windows=2,
        fallback_fraction=0.25,
    )

    assert [point.pessimistic_fraction for point in points] == [0.25, 0.25]
    assert [point.optimistic_fraction for point in points] == [0.25, 0.25]
    assert all(point.used_fallback is True for point in points)


def test_tail_calibration_ignores_zero_expected_slots():
    history = ([0.0] * 12 + [100.0] * 12) * 3

    point = calibrate_tail_renewable_uncertainty(
        history,
        slots_per_window=24,
        max_lead_windows=1,
    )[0]

    assert point.sample_count == 24
    assert point.pessimistic_fraction == 0.0
    assert point.optimistic_fraction == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"slots_per_window": 0}, "slots_per_window must be positive"),
        ({"max_lead_windows": -1}, "max_lead_windows must be >= 0"),
        ({"fallback_fraction": 1.1}, "fallback_fraction must be between 0 and 1"),
    ],
)
def test_tail_calibration_validates_inputs(kwargs, message):
    params = {
        "slots_per_window": 24,
        "max_lead_windows": 1,
        "fallback_fraction": 0.20,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        calibrate_tail_renewable_uncertainty([100.0] * 48, **params)
