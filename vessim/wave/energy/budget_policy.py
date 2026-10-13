"""
Energy budget generation policies.

   Budget policies convert renewable generation, battery state, and optional
   grid carbon constraints into a compute power budget plus a storage dispatch
   command. Edit this module when changing how much power is exposed to compute;
   keep cross-cluster budget distribution in `strategy.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from vessim.wave.energy.budget import BudgetInfo

if TYPE_CHECKING:
    from vessim.signal import HistoricalSignal
    from vessim.storage import Storage


def renewable_power_at(renewable_signal: "HistoricalSignal", time: datetime) -> float:
    columns = renewable_signal.columns() if hasattr(renewable_signal, "columns") else []
    column = "renewable_gen" if "renewable_gen" in columns else "pv_gen"
    return renewable_signal.at(time, column=column)


def is_daytime(time: datetime, day_hours: tuple[int, int]) -> bool:
    start, end = day_hours
    if start <= end:
        return start <= time.hour < end
    return time.hour >= start or time.hour < end


def remaining_night_hours(time: datetime, day_hours: tuple[int, int]) -> float:
    # Estimate how many hours remain until the next daytime start.
    day_start, day_end = day_hours
    hour = time.hour + time.minute / 60.0

    if is_daytime(time, day_hours):
        return 0.0

    if day_start <= day_end:
        if hour >= day_end:
            return (24.0 - hour) + day_start
        return day_start - hour

    if day_end <= hour < day_start:
        return day_start - hour
    return 0.0


def battery_energy_bounds(battery: "Storage") -> tuple[float, float, float]:
    battery_energy = battery._battery_energy
    min_energy = battery._state_of_charge_min * battery._battery_energy_max
    max_energy = battery._state_of_charge_max * battery._battery_energy_max
    return battery_energy, min_energy, max_energy


def usable_battery_energy_wh(battery: "Storage") -> float:
    battery_energy, min_energy, _ = battery_energy_bounds(battery)
    return max(0.0, battery_energy - min_energy)


def free_battery_energy_wh(battery: "Storage") -> float:
    battery_energy, _, max_energy = battery_energy_bounds(battery)
    return max(0.0, max_energy - battery_energy)


def step_limited_power_w(energy_wh: float, step_size: int) -> float:
    if step_size <= 0:
        return 0.0
    return max(0.0, energy_wh) * 3600.0 / step_size


def capped_charge_reserve_w(
    renewable_power_w: float,
    fraction: float,
    battery: "Storage",
    step_size: int,
) -> float:
    charge_reserve = max(0.0, renewable_power_w * fraction)
    free_power = step_limited_power_w(free_battery_energy_wh(battery), step_size)
    charge_limit = getattr(battery, "_p_charge_max", charge_reserve)
    return min(charge_reserve, free_power, charge_limit)


def rate_limited_battery_discharge_power_w(
    battery: "Storage",
    remaining_hours: float,
    step_size: int,
) -> float:
    usable_energy = usable_battery_energy_wh(battery)
    if usable_energy <= 0.0:
        return 0.0

    remaining_hours = max(remaining_hours, 1.0)
    spread_rate = usable_energy / remaining_hours
    step_limited_rate = step_limited_power_w(usable_energy, step_size)
    discharge_limit = getattr(battery, "_p_discharge_max", spread_rate)
    return min(spread_rate, step_limited_rate, discharge_limit)


class BudgetGenerationPolicy(ABC):
    @abstractmethod
    def generate_budget(
        self,
        time: datetime,
        renewable_signal: "HistoricalSignal",
        battery: "Storage",
        step_size: int = 3600,
    ) -> BudgetInfo:
        """
        Decide how much source power goes into the HPC budget.

           Args:
              time: Current simulation time.
              renewable_signal: Signal with renewable and carbon data.
              battery: Microgrid battery state.
              step_size: Scheduler timestep in seconds.

           Returns:
              Budget information with compute budget and storage dispatch.
        """


class DaytimeRenewableBudgetPolicy(BudgetGenerationPolicy):
    """
    Day/night aware budget policy.

       Day:
          - Reserve a fraction of renewable power for battery charging.
          - Expose the remaining renewable power to compute.
          - Use reduced grid support.
          - Block battery discharge so surplus power can charge storage.

       Night:
          - Prioritize battery discharge with a night-long rate limit.
          - Use grid power to fill the remaining budget gap.
    """

    def __init__(
        self,
        grid_power_limit: float = 0.0,
        day_hours: tuple[int, int] = (8, 20),
        renewable_charge_fraction: float = 0.3,
        day_grid_fraction: float = 0.5,
        night_grid_fraction: float = 1.0,
        pv_charge_fraction: Optional[float] = None,
    ):
        if pv_charge_fraction is not None:
            renewable_charge_fraction = pv_charge_fraction
        self.grid_power_limit = grid_power_limit
        self.day_hours = day_hours
        self.renewable_charge_fraction = renewable_charge_fraction
        self.day_grid_fraction = day_grid_fraction
        self.night_grid_fraction = night_grid_fraction

    def generate_budget(self, time, renewable_signal, battery, step_size: int = 3600) -> BudgetInfo:
        renewable_power = renewable_power_at(renewable_signal, time)

        if is_daytime(time, self.day_hours):
            # Daytime: reserve renewable power for charging and expose the rest to compute.
            charge_reserve = capped_charge_reserve_w(
                renewable_power,
                self.renewable_charge_fraction,
                battery,
                step_size,
            )
            budget_renewable = renewable_power - charge_reserve
            battery_discharge_power = 0.0
            grid_power = self.grid_power_limit * self.day_grid_fraction
            allow_discharge = False
            charge_power = charge_reserve

        else:
            # Nighttime: spread usable battery energy across the remaining night.
            charge_reserve = 0.0
            budget_renewable = renewable_power

            remaining_h = remaining_night_hours(time, self.day_hours)
            battery_discharge_power = rate_limited_battery_discharge_power_w(
                battery,
                remaining_h,
                step_size,
            )

            grid_power = self.grid_power_limit * self.night_grid_fraction
            allow_discharge = True
            # Negative charge_power forces controlled discharge at this rate.
            # Without this, the microgrid policy lets the battery absorb the full
            # deficit, draining it much faster than the rate limit intends.
            charge_power = -battery_discharge_power if battery_discharge_power > 0 else None

        total_budget = budget_renewable + battery_discharge_power + grid_power

        return BudgetInfo(
            time=str(time),
            renewable_power_w=renewable_power,
            budget_renewable_w=budget_renewable,
            charge_reserve_w=charge_reserve,
            battery_soc=battery.soc(),
            battery_discharge_power_w=battery_discharge_power,
            grid_power_w=grid_power,
            total_budget_w=total_budget,
            allow_discharge=allow_discharge,
            charge_power=charge_power,
        )


class GridBudgetPolicy(BudgetGenerationPolicy):
    """
    Non-carbon-aware grid budget policy.

       This is intended for the baseline: compute receives a fixed grid budget
       at every timestep, independent of carbon intensity, daylight, or battery
       state. Renewable generation is still present in the physical microgrid and can offset grid
       draw, but the scheduler does not defer or throttle work for carbon
       reasons.
    """

    def __init__(self, grid_power_limit: float):
        self.grid_power_limit = grid_power_limit

    def generate_budget(self, time, renewable_signal, battery, step_size: int = 3600) -> BudgetInfo:
        renewable_power = renewable_power_at(renewable_signal, time)
        carbon_intensity = renewable_signal.at(time, column="carbon_intensity")

        return BudgetInfo(
            time=str(time),
            renewable_power_w=renewable_power,
            budget_renewable_w=renewable_power,
            charge_reserve_w=0.0,
            battery_soc=battery.soc(),
            battery_discharge_power_w=0.0,
            grid_power_w=self.grid_power_limit,
            total_budget_w=renewable_power + self.grid_power_limit,
            allow_discharge=True,
            charge_power=None,
            carbon_intensity_g_per_kwh=carbon_intensity,
            low_carbon_grid_allowed=True,
        )


class CarbonMinimizingBudgetPolicy(BudgetGenerationPolicy):
    """
    Carbon-first budget policy.

       The policy exposes renewable power and controlled battery discharge as
       the normal compute budget. Grid power is only exposed when the current
       carbon intensity is below a configured threshold. With the default
       threshold of 0 gCO2/kWh, no grid budget is intentionally exposed.
    """

    def __init__(
        self,
        grid_power_limit: float = 0.0,
        carbon_intensity_threshold_g_per_kwh: float = 0.0,
        day_hours: tuple[int, int] = (8, 20),
        renewable_charge_fraction: float = 0.7,
        low_carbon_grid_fraction: float = 1.0,
        high_carbon_grid_fraction: float = 0.0,
        pv_charge_fraction: Optional[float] = None,
    ):
        if pv_charge_fraction is not None:
            renewable_charge_fraction = pv_charge_fraction
        self.grid_power_limit = grid_power_limit
        self.carbon_intensity_threshold_g_per_kwh = carbon_intensity_threshold_g_per_kwh
        self.day_hours = day_hours
        self.renewable_charge_fraction = renewable_charge_fraction
        self.low_carbon_grid_fraction = low_carbon_grid_fraction
        self.high_carbon_grid_fraction = high_carbon_grid_fraction

    def generate_budget(self, time, renewable_signal, battery, step_size: int = 3600) -> BudgetInfo:
        renewable_power = renewable_power_at(renewable_signal, time)
        carbon_intensity = renewable_signal.at(time, column="carbon_intensity")

        low_carbon_grid_allowed = (
            self.carbon_intensity_threshold_g_per_kwh > 0
            and carbon_intensity <= self.carbon_intensity_threshold_g_per_kwh
        )

        charge_reserve = capped_charge_reserve_w(
            renewable_power,
            self.renewable_charge_fraction,
            battery,
            step_size,
        )
        budget_renewable = max(0.0, renewable_power - charge_reserve)

        if low_carbon_grid_allowed:
            # Prefer low-carbon grid now and preserve stored energy for dirtier periods.
            battery_discharge_power = 0.0
            grid_power = self.grid_power_limit * self.low_carbon_grid_fraction
            allow_discharge = False
            charge_power = charge_reserve if charge_reserve > 0 else None
        elif is_daytime(time, self.day_hours):
            # During high-carbon daylight, run only on direct renewables and reserve surplus for later.
            battery_discharge_power = 0.0
            grid_power = self.grid_power_limit * self.high_carbon_grid_fraction
            allow_discharge = False
            charge_power = charge_reserve if charge_reserve > 0 else None
        else:
            # At night/high-carbon periods, spend the battery gradually until next daylight.
            remaining_h = remaining_night_hours(time, self.day_hours)
            battery_discharge_power = rate_limited_battery_discharge_power_w(
                battery,
                remaining_h,
                step_size,
            )
            grid_power = self.grid_power_limit * self.high_carbon_grid_fraction
            allow_discharge = True
            charge_power = -battery_discharge_power if battery_discharge_power > 0 else None

        total_budget = budget_renewable + battery_discharge_power + grid_power

        return BudgetInfo(
            time=str(time),
            renewable_power_w=renewable_power,
            budget_renewable_w=budget_renewable,
            charge_reserve_w=charge_reserve,
            battery_soc=battery.soc(),
            battery_discharge_power_w=battery_discharge_power,
            grid_power_w=grid_power,
            total_budget_w=total_budget,
            allow_discharge=allow_discharge,
            charge_power=charge_power,
            carbon_intensity_g_per_kwh=carbon_intensity,
            low_carbon_grid_allowed=low_carbon_grid_allowed,
        )
