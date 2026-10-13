from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from vessim.wave.energy.puzzle_budget_policy import JobRectangle
from vessim.wave.scheduler.policies.puzzle import PuzzleSchedulingPolicy

CLUSTER = "cluster_a"
NOW = datetime(2023, 1, 1, 12)


class FakeDVFS:
    """Minimal DVFS policy covering the calls the policy makes."""

    def __init__(self, frequency: float = 2.0, power_per_node: float = 10.0):
        self.available_frequencies = [frequency]
        self.base_frequency_ghz = frequency
        self._frequency = frequency
        self._power_per_node = power_per_node

    def power_at_frequency(self, frequency: float) -> float:
        return self._power_per_node

    def select_frequency(
        self,
        node_idle_p: float,
        budget_remaining: float,
        current_power: float,
        num_nodes_for_job: int,
    ) -> float:
        return self._frequency


class FakePuzzlePolicy:
    """Stands in for ``PuzzleBudgetPolicy``.

    ``due_rects`` are placed jobs whose planned slot has arrived;
    ``future_rects`` are placed jobs scheduled for a later slot (still
    part of the plan, but not yet due).  Both count as "planned".
    """

    def __init__(
        self,
        due_rects=None,
        future_rects=None,
        carbon_by_slot=None,
        current_slot=0,
    ):
        self._due = list(due_rects or [])
        self._future = list(future_rects or [])
        # slot_idx -> grid carbon intensity (gCO2/kWh); None disables the signal.
        self._carbon = dict(carbon_by_slot or {})
        self._current_slot = current_slot
        self.replan_calls: list[datetime] = []

    def get_due_planned_jobs(self, time, cluster_type=None):
        rects = self._due
        if cluster_type is not None:
            rects = [r for r in rects if r.cluster_type == cluster_type]
        return list(rects)

    def get_planned_job_ids(self, cluster_type=None):
        rects = self._due + self._future
        return {
            r.job_id for r in rects
            if cluster_type is None or r.cluster_type == cluster_type
        }

    def request_replan(self, time):
        self.replan_calls.append(time)

    # -- carbon helpers used by selective deferral -----------------------
    def current_slot_idx(self, time):
        return self._current_slot

    def slot_carbon_intensity(self, time):
        return self._carbon.get(self._current_slot)

    def rect_planned_carbon(self, rect):
        if rect.start_slot_idx is None:
            return None
        return self._carbon.get(rect.start_slot_idx)

    def planned_rects_by_id(self, cluster_type=None):
        rects = self._due + self._future
        return {
            r.job_id: r for r in rects
            if cluster_type is None or r.cluster_type == cluster_type
        }


def make_job(
    job_id: int,
    num_nodes: int,
    arrival: datetime = NOW,
    queue_max_wait_s: float | None = None,
    queue_enter_time: datetime | None = None,
):
    return SimpleNamespace(
        job_id=job_id,
        num_nodes=num_nodes,
        status="pending",
        priority=2,
        arrival_time=arrival,
        queue_max_wait_s=queue_max_wait_s,
        queue_enter_time=queue_enter_time or arrival,
    )


def make_rect(job_id: int, frequency: float = 2.0, start_slot: int = 0) -> JobRectangle:
    return JobRectangle(
        job_id=job_id,
        power_w=0.0,
        duration_s=300.0,
        num_nodes=1,
        frequency_ghz=frequency,
        energy_wh=0.0,
        cluster_type=CLUSTER,
        start_slot_idx=start_slot,
        end_slot_idx=start_slot,
    )


def make_policy(due_rects=None, future_rects=None, enable_fallback=True, allow_front_run=False):
    puzzle = FakePuzzlePolicy(due_rects=due_rects, future_rects=future_rects)
    policy = PuzzleSchedulingPolicy(
        puzzle,
        cluster_type=CLUSTER,
        enable_fallback_dispatch=enable_fallback,
        allow_front_run=allow_front_run,
    )
    return policy, puzzle


def test_dispatches_due_planned_job():
    policy, _ = make_policy(due_rects=[make_rect(job_id=1)])
    job = make_job(job_id=1, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1", "n2"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=100.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [1]
    assert policy.dispatch_stats["planned_dispatched"] == 1
    assert policy.dispatch_stats["fallback_dispatched"] == 0


def test_due_jobs_dispatch_in_plan_order():
    # job 2 is planned for an earlier slot than job 1; with only one node
    # free, the earlier-planned job goes first.
    policy, _ = make_policy(
        due_rects=[make_rect(job_id=1, start_slot=5), make_rect(job_id=2, start_slot=2)],
    )
    jobs = [make_job(job_id=1, num_nodes=1), make_job(job_id=2, num_nodes=1)]

    decisions = policy.select_jobs(
        pending_jobs=jobs,
        ready_nodes=["n0"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [2]


def test_does_not_front_run_future_planned_job():
    # job 7 is planned for a future slot. It must NOT be dispatched now,
    # even though a node and budget are free.
    policy, _ = make_policy(future_rects=[make_rect(job_id=7)])
    job = make_job(job_id=7, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert decisions == []
    assert policy.dispatch_stats["fallback_dispatched"] == 0


def test_front_run_allows_future_planned_job():
    # With allow_front_run, a future-planned job may fill free capacity.
    policy, _ = make_policy(future_rects=[make_rect(job_id=7)], allow_front_run=True)
    job = make_job(job_id=7, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [7]


def test_front_run_preserves_green_plan_attribution():
    # A green-planned job anticipated by front-run keeps its plan attribution
    # (front_run_green) instead of collapsing into "fallback", and is counted
    # under front_run_dispatched / green_dispatched.
    rect = make_rect(job_id=7, start_slot=4)  # future slot (current is 0)
    policy, _ = make_policy(future_rects=[rect], allow_front_run=True)
    job = make_job(job_id=7, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [7]
    assert job.planned_energy_source == "front_run_green"
    assert policy.dispatch_stats["front_run_dispatched"] == 1
    assert policy.dispatch_stats["fallback_dispatched"] == 0
    assert policy.dispatch_stats["green_dispatched"] == 1
    assert policy.dispatch_stats["grid_dispatched"] == 0


def test_front_run_preserves_grid_plan_attribution():
    # Same, but the plan parked this job on the grid-overflow lane: the
    # anticipated dispatch is tagged front_run_grid and counted as grid.
    rect = make_rect(job_id=8, start_slot=4)
    rect.used_grid = True
    policy, _ = make_policy(future_rects=[rect], allow_front_run=True)
    job = make_job(job_id=8, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [8]
    assert job.planned_energy_source == "front_run_grid"
    assert policy.dispatch_stats["front_run_dispatched"] == 1
    assert policy.dispatch_stats["fallback_dispatched"] == 0
    assert policy.dispatch_stats["grid_dispatched"] == 1


def make_defer_policy(future_rects, carbon_by_slot, current_slot, margin):
    puzzle = FakePuzzlePolicy(
        future_rects=future_rects,
        carbon_by_slot=carbon_by_slot,
        current_slot=current_slot,
    )
    policy = PuzzleSchedulingPolicy(
        puzzle,
        cluster_type=CLUSTER,
        defer_carbon_margin=margin,
    )
    return policy, puzzle


def test_selective_defer_holds_job_when_now_is_dirtier():
    # Job 7 is planned for slot 6 (clean, 200 gCO2/kWh); now is slot 0
    # (dirty, 320). 320 > 200 + margin(50) -> hold it for the greener slot.
    rect = make_rect(job_id=7, start_slot=6)
    policy, _ = make_defer_policy(
        future_rects=[rect],
        carbon_by_slot={0: 320.0, 6: 200.0},
        current_slot=0,
        margin=50.0,
    )
    job = make_job(job_id=7, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert decisions == []  # held for its greener slot
    # And it is not offered to the power-on gate, so nodes stay off for it.
    assert policy.dispatchable_pending_jobs([job], NOW) == []


def test_selective_defer_front_runs_when_now_is_clean_enough():
    # Same job, but now (slot 0) is only 230 vs planned 200: 230 <= 200 + 50,
    # so deferring buys no meaningful carbon -> run it now.
    rect = make_rect(job_id=7, start_slot=6)
    policy, _ = make_defer_policy(
        future_rects=[rect],
        carbon_by_slot={0: 230.0, 6: 200.0},
        current_slot=0,
        margin=50.0,
    )
    job = make_job(job_id=7, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [7]


def test_selective_defer_respects_start_deadline():
    # Now is dirty and the planned slot is clean, but the job's latest
    # feasible start (slot 1) is here — holding would risk a miss, so run now.
    rect = make_rect(job_id=7, start_slot=6)
    rect.latest_start_slot_idx = 1
    policy, _ = make_defer_policy(
        future_rects=[rect],
        carbon_by_slot={0: 320.0, 6: 200.0},
        current_slot=1,
        margin=50.0,
    )
    job = make_job(job_id=7, num_nodes=1)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [7]  # deadline wins over carbon
    # A planned job pulled ahead of its slot is a front-run dispatch, not a
    # safety-net fallback: its plan attribution (green) is preserved.
    assert policy.dispatch_stats["front_run_dispatched"] == 1
    assert policy.dispatch_stats["fallback_dispatched"] == 0
    assert decisions[0].job.planned_energy_source == "front_run_green"


def test_fallback_dispatches_unplaced_job():
    # No plan placed this job, so the safety-net fallback runs it.
    policy, _ = make_policy()
    job = make_job(job_id=1, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1", "n2"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=100.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [1]
    assert policy.dispatch_stats["fallback_dispatched"] == 1
    assert policy.dispatch_stats["planned_dispatched"] == 0


def test_fallback_runs_when_due_planned_job_is_gone():
    # The slot planned job 99, but it is no longer pending; an unplaced
    # job backfills instead of idling.
    policy, _ = make_policy(due_rects=[make_rect(job_id=99)])
    job = make_job(job_id=5, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=100.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [5]
    assert policy.dispatch_stats["planned_dispatched"] == 0
    assert policy.dispatch_stats["fallback_dispatched"] == 1


def test_fallback_respects_energy_budget():
    # big needs 100W (10 nodes x 10W) > budget; small needs 20W and fits.
    policy, _ = make_policy()
    big = make_job(job_id=1, num_nodes=10)
    small = make_job(job_id=2, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[big, small],
        ready_nodes=[f"n{i}" for i in range(12)],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=50.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [2]
    assert policy.dispatch_stats["fallback_dispatched"] == 1


def test_fallback_orders_by_urgency():
    # Only one ready node, so the more urgent unplaced job (smaller
    # deadline remaining) must be the one dispatched.
    policy, _ = make_policy()
    urgent = make_job(
        job_id=1,
        num_nodes=1,
        queue_max_wait_s=1000.0,
        queue_enter_time=NOW - timedelta(seconds=900),  # 100s left
    )
    relaxed = make_job(
        job_id=2,
        num_nodes=1,
        queue_max_wait_s=1000.0,
        queue_enter_time=NOW - timedelta(seconds=100),  # 900s left
    )

    decisions = policy.select_jobs(
        pending_jobs=[relaxed, urgent],
        ready_nodes=["n0"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert [d.job.job_id for d in decisions] == [1]


def test_strict_mode_disables_fallback():
    policy, _ = make_policy(due_rects=[make_rect(job_id=99)], enable_fallback=False)
    job = make_job(job_id=5, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[job],
        ready_nodes=["n0", "n1"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=100.0,
        current_time=NOW,
    )

    assert decisions == []
    assert policy.dispatch_stats["fallback_dispatched"] == 0


def test_planned_and_fallback_counters_increment_together():
    # Job 1 is due (planned); job 2 is unplaced and backfills.
    policy, _ = make_policy(due_rects=[make_rect(job_id=1)])
    planned = make_job(job_id=1, num_nodes=2)
    extra = make_job(job_id=2, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[planned, extra],
        ready_nodes=[f"n{i}" for i in range(4)],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert sorted(d.job.job_id for d in decisions) == [1, 2]
    assert policy.dispatch_stats["planned_dispatched"] == 1
    assert policy.dispatch_stats["fallback_dispatched"] == 1


def test_missed_due_job_triggers_replan():
    # Job 1 is due but only 1 node is ready (job needs 2), so it is
    # missed; fallback is disabled to isolate the planned pass.
    policy, puzzle = make_policy(due_rects=[make_rect(job_id=1)], enable_fallback=False)
    planned = make_job(job_id=1, num_nodes=2)

    decisions = policy.select_jobs(
        pending_jobs=[planned],
        ready_nodes=["n0"],
        dvfs_policy=FakeDVFS(power_per_node=10.0),
        energy_budget=1000.0,
        current_time=NOW,
    )

    assert decisions == []
    assert policy.dispatch_stats["planned_missed"] == 1
    assert puzzle.replan_calls == [NOW]
