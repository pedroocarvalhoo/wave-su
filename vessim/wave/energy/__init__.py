"""
Energy management, budgets, strategies, and DVFS policies for WAVE.

   Use this package when working on power-budget generation, battery dispatch,
   budget allocation across cluster schedulers, or frequency/power policy logic.
   The scheduler consumes these decisions but should not own the energy model.
"""

from vessim.wave.energy.dvfs import (
    AdaptiveDVFSPolicy,
    DVFSPolicy,
    FixedFrequencyDVFSPolicy,
    MaxThroughputDVFSPolicy,
    MinPowerDVFSPolicy,
)
from vessim.wave.energy.budget import BudgetInfo, ComputeBudget, StorageDispatch
from vessim.wave.energy.budget_policy import BudgetGenerationPolicy
from vessim.wave.energy.puzzle_budget_policy import PuzzleBudgetPolicy
from vessim.wave.energy.strategy import (
    ClusterSnapshot,
    DefaultAllocationStrategy,
    EnergyAllocationStrategy,
)

__all__ = [
    "AdaptiveDVFSPolicy",
    "BudgetGenerationPolicy",
    "BudgetInfo",
    "ClusterSnapshot",
    "ComputeBudget",
    "DefaultAllocationStrategy",
    "DVFSPolicy",
    "EnergyScheduler",
    "EnergyAllocationStrategy",
    "FixedFrequencyDVFSPolicy",
    "MaxThroughputDVFSPolicy",
    "MinPowerDVFSPolicy",
    "PuzzleBudgetPolicy",
    "StorageDispatch",
]


def __getattr__(name: str):
    if name == "EnergyScheduler":
        from vessim.wave.energy.energy_scheduler import EnergyScheduler

        return EnergyScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
