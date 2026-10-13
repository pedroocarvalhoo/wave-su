"""Historical calibration for Puzzle's extrapolated renewable tail.

The simulation exposes an exact renewable trace inside the real forecast
window.  Uncertainty is introduced only when Puzzle repeats that window for a
job that runs beyond it.  These helpers therefore calibrate the operation that
actually needs a forecast: compare historical windows with their realised
counterparts one or more windows later.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TailRenewableCalibration:
    """Relative P10/P90 uncertainty for one extrapolated window."""

    lead_windows: int
    pessimistic_fraction: float
    optimistic_fraction: float
    sample_count: int
    used_fallback: bool


def calibrate_tail_renewable_uncertainty(
    history_w: list[float],
    *,
    slots_per_window: int,
    max_lead_windows: int,
    fallback_fraction: float = 0.20,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
    min_samples: int = 24,
    min_power_fraction: float = 0.01,
) -> tuple[TailRenewableCalibration, ...]:
    """Estimate asymmetric tail uncertainty from realised historical windows.

    For lead ``k``, every complete historical window is treated as the repeated
    expectation and compared slot-by-slot with the realised window ``k``
    horizons later.  Near-zero expected values are omitted because ratios at
    sunrise/sunset are unstable and cannot create usable power from a repeated
    zero slot anyway.

    Fractions are monotone with lead time: an additional repeated window cannot
    appear more certain merely because of finite-sample noise.  If history is
    insufficient, the symmetric version-0.5 fraction is retained explicitly.
    """
    if slots_per_window <= 0:
        raise ValueError("slots_per_window must be positive")
    if max_lead_windows < 0:
        raise ValueError("max_lead_windows must be >= 0")
    if not 0.0 <= fallback_fraction <= 1.0:
        raise ValueError("fallback_fraction must be between 0 and 1")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if not 0.0 <= min_power_fraction <= 1.0:
        raise ValueError("min_power_fraction must be between 0 and 1")
    if max_lead_windows == 0:
        return ()

    finite_history = [
        max(0.0, float(value))
        for value in history_w
        if math.isfinite(float(value))
    ]
    n_windows = len(finite_history) // slots_per_window
    if n_windows:
        aligned = finite_history[-n_windows * slots_per_window :]
        windows = [
            aligned[i * slots_per_window : (i + 1) * slots_per_window]
            for i in range(n_windows)
        ]
    else:
        windows = []

    peak_w = max(finite_history, default=0.0)
    min_expected_w = peak_w * min_power_fraction
    results: list[TailRenewableCalibration] = []
    pessimistic_floor = 0.0
    optimistic_floor = 0.0

    for lead in range(1, max_lead_windows + 1):
        ratios: list[float] = []
        for base_idx in range(max(0, len(windows) - lead)):
            expected = windows[base_idx]
            realised = windows[base_idx + lead]
            for expected_w, realised_w in zip(expected, realised):
                if expected_w <= min_expected_w or expected_w <= 0.0:
                    continue
                ratios.append(realised_w / expected_w)

        if len(ratios) >= min_samples:
            lower_factor = float(np.quantile(ratios, lower_quantile))
            upper_factor = float(np.quantile(ratios, upper_quantile))
            pessimistic = min(1.0, max(0.0, 1.0 - lower_factor))
            optimistic = min(1.0, max(0.0, upper_factor - 1.0))
            used_fallback = False
        else:
            pessimistic = fallback_fraction
            optimistic = fallback_fraction
            used_fallback = True

        pessimistic_floor = max(pessimistic_floor, pessimistic)
        optimistic_floor = max(optimistic_floor, optimistic)
        results.append(TailRenewableCalibration(
            lead_windows=lead,
            pessimistic_fraction=pessimistic_floor,
            optimistic_fraction=optimistic_floor,
            sample_count=len(ratios),
            used_fallback=used_fallback,
        ))

    return tuple(results)


__all__ = [
    "TailRenewableCalibration",
    "calibrate_tail_renewable_uncertainty",
]
