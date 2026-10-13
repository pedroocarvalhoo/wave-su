"""
Energy budget value objects.

   These dataclasses are the data contract between budget-generation policies,
   the energy scheduler, allocation strategies, and the microgrid storage
   policy. Keep policy decisions in `budget_policy.py`; keep shared budget
   shapes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ComputeBudget:
    renewable_power_w: float
    budget_renewable_w: float
    battery_discharge_power_w: float
    grid_power_w: float
    total_budget_w: float
    carbon_intensity_g_per_kwh: Optional[float] = None
    low_carbon_grid_allowed: bool = False


@dataclass(frozen=True)
class StorageDispatch:
    allow_discharge: bool
    charge_power: Optional[float] = None
    charge_reserve_w: float = 0.0
    battery_soc: float = 0.0


@dataclass
class BudgetInfo:
    time: str
    renewable_power_w: float
    budget_renewable_w: float
    charge_reserve_w: float
    battery_soc: float
    battery_discharge_power_w: float
    grid_power_w: float
    total_budget_w: float
    allow_discharge: bool
    charge_power: Optional[float] = None
    carbon_intensity_g_per_kwh: Optional[float] = None
    low_carbon_grid_allowed: bool = False

    @property
    def compute_budget(self) -> ComputeBudget:
        return ComputeBudget(
            renewable_power_w=self.renewable_power_w,
            budget_renewable_w=self.budget_renewable_w,
            battery_discharge_power_w=self.battery_discharge_power_w,
            grid_power_w=self.grid_power_w,
            total_budget_w=self.total_budget_w,
            carbon_intensity_g_per_kwh=self.carbon_intensity_g_per_kwh,
            low_carbon_grid_allowed=self.low_carbon_grid_allowed,
        )

    @property
    def storage_dispatch(self) -> StorageDispatch:
        return StorageDispatch(
            allow_discharge=self.allow_discharge,
            charge_power=self.charge_power,
            charge_reserve_w=self.charge_reserve_w,
            battery_soc=self.battery_soc,
        )


__all__ = ["BudgetInfo", "ComputeBudget", "StorageDispatch"]
