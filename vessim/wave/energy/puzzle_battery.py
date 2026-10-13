"""Battery planning helpers for the Puzzle Scheduler forecast tail.

The Puzzle planner may repeat renewable forecasts to place jobs longer than
the real prediction window.  Battery energy is different: it is a finite
stateful resource and must never be repeated with the forecast.  This module
keeps the finite-energy dispatch calculation separate from the puzzle packer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vessim.wave.energy.budget_policy import usable_battery_energy_wh

if TYPE_CHECKING:
    from vessim.storage import Storage


@dataclass(frozen=True)
class TailBatteryDispatch:
    """Finite battery schedule for one real forecast window.

    ``power_w`` is the total planned facility-side discharge.  The
    ``correction_power_w`` subset is assigned first to deficits of running jobs
    that originally depended on an extrapolated tail.  ``held_reserve_wh`` is
    deliberately absent from both arrays and therefore cannot become compute
    budget before a later replan sees a real forecast.
    """

    power_w: tuple[float, ...]
    correction_power_w: tuple[float, ...]
    held_reserve_wh: float
    deliverable_energy_wh: float


def deliverable_battery_energy_wh(battery: "Storage") -> float:
    """Facility-side energy the current usable SoC can physically deliver."""
    efficiency = max(
        0.0, min(1.0, float(getattr(battery, "_discharge_efficiency", 1.0)))
    )
    return usable_battery_energy_wh(battery) * efficiency


def plan_tail_battery_dispatch(
    *,
    renewable_forecast_w: list[float],
    protected_demand_w: list[float],
    charging_slots: set[int],
    deliverable_energy_wh: float,
    future_tail_risk_wh: float,
    reserve_fraction: float,
    discharge_power_max_w: float,
    slot_duration_s: float,
    carbon_intensity_g_per_kwh: list[float] | None = None,
    carbon_aware: bool = False,
    carbon_priority_fraction: float = 1.0,
) -> TailBatteryDispatch:
    """Plan finite battery output while preserving forecast-tail insurance.

    A fraction of the smaller of usable battery energy and estimated future
    tail exposure is held back.  The spendable energy first covers renewable
    deficits of already-running protected jobs, including during a nominal PV
    charging slot.  With ``carbon_aware=True``, protected deficits are served
    from highest to lowest marginal grid intensity.  Up to
    ``carbon_priority_fraction`` of the remaining ordinary recourse receives
    the same greedy priority before the rest is water-filled across time.
    Protected work always remains the hard first tier, so carbon optimisation
    cannot take recourse away from an already-running job.
    """
    if len(renewable_forecast_w) != len(protected_demand_w):
        raise ValueError("renewable forecast and protected demand must align")
    if not 0.0 <= reserve_fraction <= 1.0:
        raise ValueError("reserve_fraction must be between 0 and 1")
    if slot_duration_s <= 0:
        raise ValueError("slot_duration_s must be positive")
    if not 0.0 <= carbon_priority_fraction <= 1.0:
        raise ValueError("carbon_priority_fraction must be between 0 and 1")

    n_slots = len(renewable_forecast_w)
    if carbon_intensity_g_per_kwh is not None and (
        len(carbon_intensity_g_per_kwh) != n_slots
    ):
        raise ValueError("carbon intensity and renewable forecast must align")
    carbon = (
        [max(0.0, float(value)) for value in carbon_intensity_g_per_kwh]
        if carbon_intensity_g_per_kwh is not None
        else [0.0] * n_slots
    )
    deliverable_wh = max(0.0, float(deliverable_energy_wh))
    risk_wh = max(0.0, float(future_tail_risk_wh))
    power_cap_w = max(0.0, float(discharge_power_max_w))
    held_wh = reserve_fraction * min(deliverable_wh, risk_wh)
    remaining_wh = max(0.0, deliverable_wh - held_wh)

    correction = [0.0] * n_slots
    total = [0.0] * n_slots
    slot_h = slot_duration_s / 3600.0

    # A protected job is already running and cannot be delayed.  Its visible
    # renewable deficit has first claim on the spendable battery energy.
    correction_slots = list(range(n_slots))
    if carbon_aware:
        correction_slots.sort(key=lambda idx: (-carbon[idx], idx))
    for idx in correction_slots:
        if remaining_wh <= 1e-12:
            break
        renewable_w = renewable_forecast_w[idx]
        demand_w = protected_demand_w[idx]
        deficit_w = max(0.0, float(demand_w) - max(0.0, float(renewable_w)))
        power_w = min(deficit_w, power_cap_w, remaining_wh / slot_h)
        correction[idx] = power_w
        total[idx] = power_w
        remaining_wh -= power_w * slot_h

    # Allocate ordinary recourse only after every protected deficit.  Carbon
    # mode spends it greedily where a kWh avoids the most grid CO2.  Compatibility
    # mode retains the original equal water-fill behaviour.
    active = {
        idx for idx in range(n_slots)
        if idx not in charging_slots and total[idx] < power_cap_w - 1e-12
    }
    if carbon_aware and carbon_priority_fraction > 0.0:
        priority_wh = remaining_wh * carbon_priority_fraction
        for idx in sorted(active, key=lambda slot: (-carbon[slot], slot)):
            if priority_wh <= 1e-12:
                break
            headroom_w = max(0.0, power_cap_w - total[idx])
            add_w = min(headroom_w, priority_wh / slot_h)
            total[idx] += add_w
            used_wh = add_w * slot_h
            priority_wh = max(0.0, priority_wh - used_wh)
            remaining_wh = max(0.0, remaining_wh - used_wh)
            if total[idx] >= power_cap_w - 1e-12:
                active.discard(idx)
    while remaining_wh > 1e-12 and active:
        equal_power_w = remaining_wh / (slot_h * len(active))
        used_wh = 0.0
        saturated: set[int] = set()
        for idx in active:
            headroom_w = max(0.0, power_cap_w - total[idx])
            add_w = min(equal_power_w, headroom_w)
            total[idx] += add_w
            used_wh += add_w * slot_h
            if add_w >= headroom_w - 1e-12:
                saturated.add(idx)
        if used_wh <= 1e-12:
            break
        remaining_wh = max(0.0, remaining_wh - used_wh)
        active.difference_update(saturated)
        if not saturated:
            break

    return TailBatteryDispatch(
        power_w=tuple(total),
        correction_power_w=tuple(correction),
        held_reserve_wh=held_wh,
        deliverable_energy_wh=deliverable_wh,
    )


__all__ = [
    "TailBatteryDispatch",
    "deliverable_battery_energy_wh",
    "plan_tail_battery_dispatch",
]
