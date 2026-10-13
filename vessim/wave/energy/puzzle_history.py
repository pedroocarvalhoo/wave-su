"""Bounded detailed Puzzle history with complete summaries and active timeline."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .puzzle_budget_policy import PuzzlePlan


def plan_summary(plan: PuzzlePlan, plan_id: int, build_duration_s: float) -> dict:
    """Copy scalar metrics only, never the slot boards or job allocations."""
    excluded = {"slots", "placed_jobs", "unplaced_jobs", "cluster_slots",
                "tail_protected_job_keys"}
    row = {field.name: getattr(plan, field.name)
           for field in fields(plan) if field.name not in excluded}
    row.update(
        plan_id=plan_id,
        num_slots=len(plan.slots),
        green_planned_kwh=plan.green_planned_kwh,
        green_fraction=plan.green_fraction,
        slack_kwh=plan.slack_kwh,
        utilization=plan.utilization,
        num_placed=plan.num_placed,
        num_unplaced=plan.num_unplaced,
        num_grid_spilled=plan.num_grid_spilled,
        num_low_carbon_grid=plan.num_low_carbon_grid,
        num_carbon_regret_grid=plan.num_carbon_regret_grid,
        num_sla_emergency_grid=plan.num_sla_emergency_grid,
        num_tail_protected_jobs=len(plan.tail_protected_job_keys),
        build_duration_s=build_duration_s,
        num_future_job_rows=sum(
            rect.arrival_time is not None and rect.arrival_time > plan.plan_start
            for rect in (*plan.placed_jobs, *plan.unplaced_jobs)
        ),
    )
    return row


def active_rows(plan: PuzzlePlan, plan_id: int, end: datetime, pue: float) -> list[dict]:
    """Materialise only the interval during which this plan was in force."""
    if pue <= 0:
        raise ValueError("pue must be positive")
    rows = []
    end = min(plan.plan_end, end)
    for idx, slot in enumerate(plan.slots):
        if slot.start_time >= end:
            break
        planned_w = slot.allocated_w + slot.grid_allocated_w
        budget_w = slot.green_available_w + slot.grid_allocated_w
        rows.append(dict(
            plan_id=plan_id, slot_idx=idx,
            start_time=slot.start_time, end_time=min(slot.end_time, end),
            planned_facility_power_w=planned_w,
            planned_it_power_w=planned_w / pue,
            runtime_budget_facility_w=budget_w,
            runtime_budget_it_w=budget_w / pue,
            battery_power_w=slot.battery_power_w,
            battery_correction_power_w=slot.battery_correction_power_w,
            pue=pue,
        ))
    return rows


class PuzzlePlanHistory:
    """Keep every summary but at most ``limit`` full plans (None keeps all).

    Sampling thins old plans on a doubling stride. Original plan IDs remain
    stable, and the first and latest plans survive when the limit is >= 2.
    The active timeline is closed before the preceding plan can be discarded.
    """

    def __init__(self, limit: int | None = 64, pue: float = 1.0):
        if limit is not None and limit < 1:
            raise ValueError("history limit must be positive or None")
        if pue <= 0:
            raise ValueError("pue must be positive")
        self.limit = limit
        self.pue = pue
        self.summaries: list[dict] = []
        self._samples: dict[int, PuzzlePlan] = {}
        self._active_rows: list[dict] = []
        self._last_plan: PuzzlePlan | None = None
        self._stride = 1

    @property
    def count(self) -> int:
        return len(self.summaries)

    @property
    def indexed_plans(self) -> list[tuple[int, PuzzlePlan]]:
        return sorted(self._samples.items())

    def record(self, plan: PuzzlePlan, build_duration_s: float = 0.0) -> None:
        plan_id = self.count
        if self._last_plan is not None:
            if plan.plan_start < self._last_plan.plan_start:
                raise ValueError("plan history must be chronological")
            self._active_rows.extend(active_rows(
                self._last_plan, plan_id - 1, plan.plan_start, self.pue,
            ))
        self.summaries.append(plan_summary(plan, plan_id, build_duration_s))
        self._last_plan = plan
        self._samples[plan_id] = plan
        if self.limit == 1:
            self._samples = {plan_id: plan}
        elif self.limit is not None:
            # The previous latest was retained regardless of stride. Remove
            # it now if it is not part of the historical sample.
            self._samples = {i: p for i, p in self._samples.items()
                             if i == plan_id or i % self._stride == 0}
            while len(self._samples) > self.limit:
                self._stride *= 2
                self._samples = {i: p for i, p in self._samples.items()
                                 if i == plan_id or i % self._stride == 0}

    def active_timeline(self) -> list[dict]:
        rows = list(self._active_rows)
        if self._last_plan is not None:
            rows.extend(active_rows(
                self._last_plan, self.count - 1, self._last_plan.plan_end, self.pue,
            ))
        return rows
