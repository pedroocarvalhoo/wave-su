"""History retention must not lose active intervals or retain the live graph."""

from datetime import datetime, timedelta
from io import BytesIO
import pickle

import pytest

from scripts.simulation_energy import _PostprocessingPickler
from vessim.wave.energy.puzzle_budget_policy import (
    JobRectangle, PowerSlot, PuzzleBudgetPolicy, PuzzlePlan,
)
from vessim.wave.energy.puzzle_history import PuzzlePlanHistory
from vessim.wave.scheduler import JobQueueScheduler


START = datetime(2023, 1, 1)


def make_plan(index):
    start = START + timedelta(minutes=5 * index)
    slots = [PowerSlot(
        start_time=start + timedelta(minutes=5 * offset),
        end_time=start + timedelta(minutes=5 * (offset + 1)),
        duration_s=300, renewable_power_w=130, battery_power_w=0,
        grid_power_w=100, green_available_w=130, allocated_w=65,
        carbon_intensity=100,
    ) for offset in range(3)]
    return PuzzlePlan(
        plan_start=start, plan_end=slots[-1].end_time, slots=slots,
        placed_jobs=[], unplaced_jobs=[], total_budget_kwh=0.0325,
        total_planned_kwh=0.01625,
    )


@pytest.mark.parametrize("limit", [1, 2, 3, 64, None])
def test_retention_preserves_summaries_and_every_active_interval(limit):
    history = PuzzlePlanHistory(limit, pue=1.3)
    for index in range(257):
        history.record(make_plan(index), build_duration_s=0.25)
        assert len(history.indexed_plans) <= (limit or index + 1)
    assert history.count == 257
    assert [r["plan_id"] for r in history.summaries] == list(range(257))
    assert sum(r["build_duration_s"] for r in history.summaries) == 64.25
    assert history.indexed_plans[-1][0] == 256
    if limit != 1:
        assert history.indexed_plans[0][0] == 0
    rows = history.active_timeline()
    assert len(rows) == 259
    assert [r["plan_id"] for r in rows[:257]] == list(range(257))
    assert all(a["end_time"] == b["start_time"] for a, b in zip(rows, rows[1:]))
    assert all(r["planned_it_power_w"] == 50 for r in rows)
    assert history.active_timeline() == rows  # Reading must not append twice.


def test_history_clips_partial_slots_and_rejects_backwards_time():
    history = PuzzlePlanHistory(1)
    history.record(make_plan(0))
    newer = make_plan(1)
    newer.plan_start = START + timedelta(seconds=90)
    history.record(newer)
    assert history.active_timeline()[0]["end_time"] == newer.plan_start
    with pytest.raises(ValueError, match="chronological"):
        history.record(make_plan(0))
    assert history.count == 2


def test_plot_snapshot_drops_history_without_mutating_live_policy():
    policy = PuzzleBudgetPolicy()
    plan = make_plan(0)
    policy.history.record(plan)
    policy._current_plan = plan
    policy._plan_index_cache["sentinel"] = plan
    # Deliberately unpicklable: live scheduler references must not be traversed.
    policy.cluster_schedulers = [lambda: None]
    stream = BytesIO()
    _PostprocessingPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(policy)
    restored = pickle.loads(stream.getvalue())
    assert restored.history.count == 0
    assert restored.current_plan is None
    assert restored.cluster_schedulers == []
    assert restored._plan_index_cache == {}
    assert policy.history.count == 1
    assert policy.current_plan is plan
    assert policy._plan_index_cache["sentinel"] is plan
    assert len(policy.cluster_schedulers) == 1


def test_plot_snapshot_preserves_scheduler_data_and_shared_policy():
    # Exercise serialization independently of simulation setup, including a
    # cycle and a shared planner reference as found in multi-cluster runs.
    scheduler = object.__new__(JobQueueScheduler)
    scheduler.energy_scheduler = lambda: None
    scheduler.timeline = [{"time": START, "power": 130}]
    scheduler.scheduling_policy = PuzzleBudgetPolicy()
    scheduler.scheduling_policy.cluster_schedulers = [scheduler]
    scheduler.self_reference = scheduler
    stream = BytesIO()
    _PostprocessingPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(
        [scheduler, scheduler.scheduling_policy],
    )
    restored, planner = pickle.loads(stream.getvalue())
    assert restored.timeline == scheduler.timeline
    assert restored.self_reference is restored
    assert restored.scheduling_policy is planner
    assert restored.energy_scheduler is None
    assert callable(scheduler.energy_scheduler)
    assert scheduler.scheduling_policy.cluster_schedulers == [scheduler]


@pytest.mark.parametrize("latest", [None, 0, 1])
def test_short_grid_candidate_never_reads_unreachable_tail(latest):
    policy = PuzzleBudgetPolicy(carbon_regret_threshold_g_per_h=None)
    rect = JobRectangle(
        job_id=1, power_w=200, duration_s=300, num_nodes=1,
        frequency_ghz=2, energy_wh=200 / 12,
        latest_start_slot_idx=latest,
    )
    slots = make_plan(0).slots
    variants = [(2, 200, 300), (1, 150, 600)]
    kwargs = dict(rect=rect, variants=variants, min_start_idx=0,
                  visible_slots=3, extendable=False)
    expected = policy._choose_carbon_aware_grid_placement(
        slots=slots, slot_remaining=[130] * 3, node_free=[1, 0, 1], **kwargs,
    )
    # Bare objects fail on any access to power, CI, duration, or node count.
    actual = policy._choose_carbon_aware_grid_placement(
        slots=slots + [object()] * 100,
        slot_remaining=[130] * 3 + [object()] * 100,
        node_free=[1, 0, 1] + [object()] * 100, **kwargs,
    )
    assert actual == expected
    assert actual is not None
