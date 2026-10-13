from __future__ import annotations

from vessim.wave.energy import EnergyScheduler
from vessim.wave.energy.budget import BudgetInfo
from vessim.wave.scheduler import SimulationComplete
from vessim.policy import DefaultMicrogridPolicy


class FakeBudgetPolicy:
    def __init__(self, budget_info):
        self.budget_info = budget_info
        self.calls = []

    def generate_budget(self, time, renewable_signal, battery, step_size):
        self.calls.append(
            {
                "time": time,
                "renewable_signal": renewable_signal,
                "battery": battery,
                "step_size": step_size,
            }
        )
        return self.budget_info


class FakeAllocationStrategy:
    def __init__(self, allocations=None):
        self.allocations = allocations or {}
        self.calls = []

    def allocate(self, total_budget, snapshots, time=None):
        self.calls.append((total_budget, snapshots))
        if self.allocations:
            return {
                snapshot.scheduler: self.allocations.get(snapshot.scheduler, 0.0)
                for snapshot in snapshots
            }
        share = total_budget / len(snapshots) if snapshots else 0.0
        return {snapshot.scheduler: share for snapshot in snapshots}


class FakeResourceManager:
    def __init__(self, total_nodes=2, busy_nodes=1, powered_on_nodes=2):
        self.nodes = {f"n{i}": object() for i in range(total_nodes)}
        self._busy_nodes = [f"n{i}" for i in range(busy_nodes)]
        self._powered_on_nodes = powered_on_nodes

    def get_powered_on_node_count(self):
        return self._powered_on_nodes

    def get_busy_nodes(self):
        return list(self._busy_nodes)


class FakeClusterScheduler:
    def __init__(
        self,
        cluster_type,
        *,
        current_power=50.0,
        pending_jobs=1,
        done=False,
        total_nodes=2,
        busy_nodes=1,
        powered_on_nodes=2,
    ):
        self.cluster_type = cluster_type
        self.rm = FakeResourceManager(total_nodes, busy_nodes, powered_on_nodes)
        self.current_power = current_power
        self.pending_jobs = [object()] * pending_jobs
        self.done = done
        self.energy_budget = None
        self.energy_scheduler = None
        self.budget_history = []

    def get_current_power(self):
        return self.current_power

    def _get_pending_jobs(self):
        return list(self.pending_jobs)

    def set_energy_budget(self, budget):
        self.energy_budget = budget
        self.budget_history.append(budget)


NODE_CONFIG = {
    "cluster_a": {
        "idle_power_w": 10.0,
        "max_power_w": 100.0,
    },
    "cluster_b": {
        "idle_power_w": 20.0,
        "max_power_w": 200.0,
    },
}


def _budget_info(total_budget_w=1000.0):
    return BudgetInfo(
        time="2023-01-01 12:00:00",
        renewable_power_w=700.0,
        budget_renewable_w=600.0,
        charge_reserve_w=100.0,
        battery_soc=0.5,
        battery_discharge_power_w=300.0,
        grid_power_w=100.0,
        total_budget_w=total_budget_w,
        allow_discharge=False,
        charge_power=100.0,
        carbon_intensity_g_per_kwh=80.0,
        low_carbon_grid_allowed=True,
    )


def test_energy_scheduler_generates_budget_builds_snapshots_and_allocates(
    sim_time,
):
    scheduler_a = FakeClusterScheduler("cluster_a", current_power=50.0, pending_jobs=1)
    scheduler_b = FakeClusterScheduler(
        "cluster_b",
        current_power=80.0,
        pending_jobs=0,
        busy_nodes=0,
    )
    budget_policy = FakeBudgetPolicy(_budget_info(total_budget_w=1000.0))
    alloc_strategy = FakeAllocationStrategy()
    microgrid_policy = DefaultMicrogridPolicy()

    energy_scheduler = EnergyScheduler(
        renewable_signal=object(),
        battery=object(),
        cluster_schedulers=[scheduler_a, scheduler_b],
        node_config=NODE_CONFIG,
        alloc_strategy=alloc_strategy,
        budget_policy=budget_policy,
        policy=microgrid_policy,
        step_size=300,
    )

    energy_scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    assert budget_policy.calls[0]["step_size"] == 300
    assert energy_scheduler.last_budget_info["total_budget_w"] == 1000.0
    assert energy_scheduler.last_budget_info["carbon_intensity_g_per_kwh"] == 80.0
    assert microgrid_policy.allow_discharge is False
    assert microgrid_policy.charge_power == 100.0
    assert scheduler_a.energy_scheduler is energy_scheduler
    assert scheduler_b.energy_scheduler is energy_scheduler
    assert scheduler_a.energy_budget == 500.0
    assert scheduler_b.energy_budget == 500.0

    total_budget, snapshots = alloc_strategy.calls[0]
    assert total_budget == 1000.0
    assert [snapshot.cluster_type for snapshot in snapshots] == ["cluster_a", "cluster_b"]
    assert snapshots[0].powered_on_nodes == 2
    assert snapshots[0].busy_nodes == 1
    assert snapshots[0].current_power == 50.0
    assert snapshots[0].has_pending_jobs is True
    assert snapshots[1].has_pending_jobs is False


def test_energy_scheduler_converts_facility_budget_to_it_budget_with_pue(
    sim_time,
):
    scheduler_a = FakeClusterScheduler("cluster_a")
    scheduler_b = FakeClusterScheduler("cluster_b")
    budget_policy = FakeBudgetPolicy(_budget_info(total_budget_w=1300.0))
    alloc_strategy = FakeAllocationStrategy()

    energy_scheduler = EnergyScheduler(
        renewable_signal=object(),
        battery=object(),
        cluster_schedulers=[scheduler_a, scheduler_b],
        node_config=NODE_CONFIG,
        alloc_strategy=alloc_strategy,
        budget_policy=budget_policy,
        policy=None,
        step_size=300,
        pue=1.3,
    )

    energy_scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})

    total_budget, _snapshots = alloc_strategy.calls[0]
    assert abs(total_budget - 1000.0) < 1e-9
    assert energy_scheduler.last_budget_info["total_budget_w"] == 1300.0
    assert abs(
        energy_scheduler.last_budget_info["scheduler_total_budget_w"] - 1000.0
    ) < 1e-9
    assert energy_scheduler.last_budget_info["pue"] == 1.3
    assert scheduler_a.energy_budget == 500.0
    assert scheduler_b.energy_budget == 500.0


def test_energy_scheduler_rejects_non_positive_pue():
    try:
        EnergyScheduler(
            renewable_signal=object(),
            battery=object(),
            cluster_schedulers=[],
            node_config=NODE_CONFIG,
            alloc_strategy=FakeAllocationStrategy(),
            budget_policy=FakeBudgetPolicy(_budget_info()),
            policy=None,
            pue=0,
        )
    except ValueError as exc:
        assert "pue" in str(exc)
    else:
        raise AssertionError("EnergyScheduler accepted pue=0")


def test_energy_scheduler_raises_simulation_complete_after_all_clusters_done(
    sim_time,
):
    scheduler = FakeClusterScheduler("cluster_a", done=True, pending_jobs=0, busy_nodes=0)
    energy_scheduler = EnergyScheduler(
        renewable_signal=object(),
        battery=object(),
        cluster_schedulers=[scheduler],
        node_config=NODE_CONFIG,
        alloc_strategy=FakeAllocationStrategy(),
        budget_policy=FakeBudgetPolicy(_budget_info(total_budget_w=100.0)),
        policy=None,
        step_size=60,
    )

    try:
        energy_scheduler.step(sim_time, p_delta=0.0, e_delta=0.0, actor_infos={})
    except SimulationComplete:
        raised = True
    else:
        raised = False

    assert raised is True


def test_energy_scheduler_reallocates_existing_budget_on_cluster_completion():
    scheduler_a = FakeClusterScheduler("cluster_a")
    scheduler_b = FakeClusterScheduler("cluster_b")
    scheduler_a.set_energy_budget(100.0)
    scheduler_b.set_energy_budget(200.0)

    alloc_strategy = FakeAllocationStrategy(
        allocations={
            scheduler_a: 0.0,
            scheduler_b: 300.0,
        }
    )
    energy_scheduler = EnergyScheduler(
        renewable_signal=object(),
        battery=object(),
        cluster_schedulers=[scheduler_a, scheduler_b],
        node_config=NODE_CONFIG,
        alloc_strategy=alloc_strategy,
        budget_policy=FakeBudgetPolicy(_budget_info(total_budget_w=300.0)),
        policy=None,
        step_size=60,
    )

    energy_scheduler.notification({"type": "ClusterComplete"}, scheduler_a)

    total_budget, snapshots = alloc_strategy.calls[0]
    assert total_budget == 300.0
    assert [snapshot.scheduler for snapshot in snapshots] == [scheduler_a, scheduler_b]
    assert scheduler_a.energy_budget == 0.0
    assert scheduler_b.energy_budget == 300.0
