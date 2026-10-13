"""
Energy scheduler controller for WAVE simulations.

   `EnergyScheduler` is the bridge between the microgrid state and the cluster
   schedulers. It asks a budget policy for available compute power, delegates
   cross-cluster allocation to an energy strategy, and forwards storage commands
   to the microgrid policy. It should coordinate energy decisions, not schedule
   individual jobs.
"""

from datetime import datetime
from vessim.signal import HistoricalSignal
from vessim.storage import Storage
from vessim.controller import Controller
from vessim.policy import DefaultMicrogridPolicy
from vessim.wave.scheduler import JobQueueScheduler, SimulationComplete
from vessim.wave.energy.strategy import (
    EnergyAllocationStrategy,
    DefaultAllocationStrategy,
    ClusterSnapshot,
)
from vessim.wave.energy.budget_policy import BudgetGenerationPolicy


class EnergyScheduler(Controller):
    def __init__(
        self,
        renewable_signal: HistoricalSignal, 
        battery: Storage,
        cluster_schedulers: list[JobQueueScheduler],
        node_config: dict,
        alloc_strategy: EnergyAllocationStrategy = None,
        budget_policy: BudgetGenerationPolicy = None,
        policy: DefaultMicrogridPolicy = None,
        start_time: datetime = None,
        step_size: int = 900,
        pue: float = 1.0,
    ):
        super().__init__(step_size=step_size)
        if pue <= 0:
            raise ValueError("pue must be positive.")
        self.renewable_signal = renewable_signal
        self.battery = battery
        self.cluster_schedulers = cluster_schedulers
        self.node_config = node_config
        self.alloc_strategy = alloc_strategy or DefaultAllocationStrategy()
        self.policy = policy
        self.start_time = start_time
        self.current_time = start_time
        self.pue = float(pue)
        self.last_budget_info = {}  # Cached for monitoring and plotting.

        if budget_policy is None:
            raise ValueError("No budget_policy provided to EnergyScheduler.")
        self.budget_policy = budget_policy

        self.clusters = {}
        for scheduler in cluster_schedulers:
            node_type = scheduler.cluster_type
            self.clusters[scheduler] = {
                "type": node_type,
                "num_nodes": len(scheduler.rm.nodes),
                "config": self.node_config[node_type],
            }
            scheduler.energy_scheduler = self

    def _build_snapshots(self, time: datetime = None) -> list[ClusterSnapshot]:
        # Build live cluster snapshots for the allocation strategy.
        snapshots = []
        visibility_time = time or self.current_time
        for scheduler in self.cluster_schedulers:
            cluster_info = self.clusters[scheduler]
            cfg = cluster_info["config"]
            rm = scheduler.rm
            pending_jobs = _get_scheduler_pending_jobs(scheduler, visibility_time)

            snapshots.append(ClusterSnapshot(
                scheduler=scheduler,
                cluster_type=cluster_info["type"],
                total_nodes=cluster_info["num_nodes"],
                powered_on_nodes=rm.get_powered_on_node_count(),
                busy_nodes=len(rm.get_busy_nodes()),
                idle_power_per_node=cfg["idle_power_w"],
                max_power_per_node=cfg["max_power_w"],
                current_power=scheduler.get_current_power(),
                is_done=scheduler.done,
                has_pending_jobs=len(pending_jobs) > 0,
            ))
        return snapshots
    
    def step(
        self, 
        time: datetime, 
        p_delta: float, 
        e_delta: float, 
        actor_infos: dict
    ) -> None:

        if self.start_time is None:
            self.start_time = time
        self.current_time = time

        # Delegate budget calculation to the injected policy.
        budget_info = self.budget_policy.generate_budget(
            time, self.renewable_signal, self.battery, self.step_size
        )
        compute_budget = budget_info.compute_budget
        storage_dispatch = budget_info.storage_dispatch

        total_budget = compute_budget.total_budget_w
        scheduler_total_budget = total_budget / self.pue

        self.last_budget_info = {
            'time': budget_info.time,
            'renewable_power_w': compute_budget.renewable_power_w,
            'budget_renewable_w': compute_budget.budget_renewable_w,
            'charge_reserve_w': storage_dispatch.charge_reserve_w,
            'battery_soc': storage_dispatch.battery_soc,
            'battery_discharge_power_w': compute_budget.battery_discharge_power_w,
            'grid_power_w': compute_budget.grid_power_w,
            'total_budget_w': compute_budget.total_budget_w,
            'scheduler_total_budget_w': scheduler_total_budget,
            'pue': self.pue,
            'carbon_intensity_g_per_kwh': compute_budget.carbon_intensity_g_per_kwh,
            'low_carbon_grid_allowed': compute_budget.low_carbon_grid_allowed,
        }

        # Forward physical discharge/charge decisions to the microgrid policy.
        if self.policy is not None:
            self.policy.allow_discharge = storage_dispatch.allow_discharge
            self.policy.charge_power = storage_dispatch.charge_power

        snapshots = self._build_snapshots(time)
        allocations = self.alloc_strategy.allocate(scheduler_total_budget, snapshots, time)
        for scheduler, allocation in allocations.items():
            scheduler.set_energy_budget(allocation)

        if all(s.done for s in self.cluster_schedulers):
            raise SimulationComplete()

    def notification(self, notification: dict, scheduler: JobQueueScheduler):

        match notification["type"]:
            case "ClusterComplete":
                total_budget = sum(
                    s.energy_budget for s in self.cluster_schedulers
                    if s.energy_budget is not None
                )
                snapshots = self._build_snapshots(self.current_time)
                allocations = self.alloc_strategy.allocate(total_budget, snapshots, self.current_time)
                for s, allocation in allocations.items():
                    s.set_energy_budget(allocation)


def _get_scheduler_pending_jobs(scheduler, time: datetime):
    try:
        return scheduler._get_pending_jobs(time)
    except TypeError:
        return scheduler._get_pending_jobs()
