"""
DVFS frequency and dynamic-power policies.

   These policies choose CPU/GPU frequency levels and expose the dynamic power
   cost of each level. They are used both when starting jobs and during runtime
   DVFS adjustments in the resource manager. Do not put job ordering logic here;
   that belongs in scheduler policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class DVFSPolicy(ABC):
    def __init__(self, frequencies: dict[float, float], base_frequency_ghz: float):
        if not frequencies:
            raise ValueError("frequencies dict must have at least one entry")
        
        self._freq_table: list[tuple[float, float]] = sorted(
            ((float(f), float(p)) for f, p in frequencies.items()),
            key=lambda x: x[0],
        ) 
        # Frequency and dynamic power pairs.
        self.base_frequency_ghz = base_frequency_ghz
        self._min_freq = self._freq_table[0][0]
        self._max_freq = self._freq_table[-1][0]
        # Frequencies are sorted from min to max.

    @property
    def available_frequencies(self) -> list[float]:
        return [f for f, _ in self._freq_table]

    @property
    def runtime_dvfs_enabled(self) -> bool:
        return True

    @property
    def allow_runtime_upscale(self) -> bool:
        return True

    def power_at_frequency(self, freq_ghz: float) -> float:
        for f, p in self._freq_table:
            if abs(f - freq_ghz) < 1e-6:
                return p
        raise ValueError(
            f"Frequency {freq_ghz} GHz not in available frequencies: "
            f"{self.available_frequencies}"
        )

    @abstractmethod
    def select_frequency(
        self,
        node_idle_p: float,
        budget_remaining: float,
        current_power: float,
        num_nodes_for_job: int,
    ) -> float:
        """
        Select the frequency for a job about to be submitted.

           Args:
              node_idle_p: Idle power of each node in watts.
              budget_remaining: Remaining cluster energy budget in watts.
              current_power: Current cluster power consumption in watts.
              num_nodes_for_job: Number of nodes this job will occupy.

           Returns:
              Selected frequency in GHz.
        """


class AdaptiveDVFSPolicy(DVFSPolicy):
    """
    Adapts frequency based on available budget headroom.

       - Use the highest frequency when budget headroom is generous.
       - Use the lowest frequency when budget headroom is tight.
       - Otherwise, select the highest frequency that fits the budget.
    """

    def __init__(
        self,
        frequencies: dict[float, float],
        base_frequency_ghz: float,
        high_threshold: float = 0.6,
        low_threshold: float = 0.2,
    ):
        super().__init__(frequencies, base_frequency_ghz)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def select_frequency(
        self,
        node_idle_p: float,
        budget_remaining: float,
        current_power: float,
        num_nodes_for_job: int,
    ) -> float:
        if budget_remaining <= 0:
            return self._min_freq

        if budget_remaining == float('inf'):
            return self._max_freq

        headroom = budget_remaining - current_power
        budget_ratio = headroom / budget_remaining if budget_remaining > 0 else 0.0

        if budget_ratio >= self.high_threshold:
            # High budget: prefer throughput.
            candidates = list(reversed(self._freq_table))
        elif budget_ratio <= self.low_threshold:
            # Low budget: prefer energy conservation.
            candidates = list(self._freq_table)
        else:
            # Mid range: graduate the frequency ceiling by how much headroom
            # remains, so a tightening budget down-clocks instead of always
            # running flat out (which made this branch identical to
            # max-throughput). budget_ratio near ``high_threshold`` caps near
            # the top frequency; near ``low_threshold`` caps near the bottom.
            span = self.high_threshold - self.low_threshold
            frac = (budget_ratio - self.low_threshold) / span if span > 0 else 1.0
            n = len(self._freq_table)
            ceil_idx = min(n - 1, max(0, int(round(frac * (n - 1)))))
            candidates = list(reversed(self._freq_table[: ceil_idx + 1]))

        # Pick the highest frequency whose additional power fits in headroom.
        for freq, dyn_power in candidates:
            additional_power = dyn_power * num_nodes_for_job
            if additional_power <= headroom:
                return freq

        # If nothing fits, still return the lowest supported frequency.
        return self._min_freq


class MaxThroughputDVFSPolicy(DVFSPolicy):
    """
    Always selects the highest frequency that fits within the energy budget.

       Minimizes job execution time at the cost of higher power consumption.
    """

    def select_frequency(
        self,
        node_idle_p: float,
        budget_remaining: float,
        current_power: float,
        num_nodes_for_job: int,
    ) -> float:

        return self._max_freq


class MinPowerDVFSPolicy(DVFSPolicy):
    """
    Always selects the lowest available frequency.
    
       Minimizes power consumption per node, allowing more simultaneous jobs at
       the cost of longer execution times.
    """

    def select_frequency(
        self,
        node_idle_p: float,
        budget_remaining: float,
        current_power: float,
        num_nodes_for_job: int,
    ) -> float:
        return self._min_freq

    @property
    def allow_runtime_upscale(self) -> bool:
        return False


class FixedFrequencyDVFSPolicy(DVFSPolicy):
    """
    Always uses a specific fixed frequency (equivalent to no DVFS).

       Useful as a baseline for comparison.
    """

    def __init__(
        self,
        frequencies: dict[float, float],
        base_frequency_ghz: float,
        fixed_frequency_ghz: Optional[float] = None,
    ):
        super().__init__(frequencies, base_frequency_ghz)
        self.fixed_freq = fixed_frequency_ghz or base_frequency_ghz
        self.power_at_frequency(self.fixed_freq)

    def select_frequency(
        self,
        node_idle_p: float,
        budget_remaining: float,
        current_power: float,
        num_nodes_for_job: int,
    ) -> float:
        return self.fixed_freq

    @property
    def runtime_dvfs_enabled(self) -> bool:
        return False

    @property
    def allow_runtime_upscale(self) -> bool:
        return False
