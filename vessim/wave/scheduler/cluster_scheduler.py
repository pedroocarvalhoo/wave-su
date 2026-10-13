"""
Cluster scheduler controller for WAVE job queues.

   `JobQueueScheduler` is the main per-cluster controller. It advances running
   jobs, asks a scheduling policy for dispatch decisions, requests resource
   manager transitions, reacts to energy budgets, and exports scheduler metrics.
   Avoid placing policy-specific scoring or node-state mutations here.
"""

from __future__ import annotations

import os
import random
from datetime import datetime
from functools import partial
from typing import Optional, TYPE_CHECKING

from vessim.controller import Controller
from vessim.wave.cluster.job_queue import JobQueue, Job
from vessim.wave.resource_manager import ResourceManager
from vessim.wave.energy.dvfs import DVFSPolicy
from vessim.wave.scheduler.runtime.execution_models import (
    compute_actual_execution_work,
    predict_execution_time,
    sample_prediction_error_factor,
)
from vessim.wave.scheduler.policies import MultiQueuePolicy, SchedulingPolicy, PriorityPolicy
from vessim.wave.scheduler.policies.base import select_jobs_with_optional_time
from vessim.wave.scheduler.queues import (
    QueueDefinition,
    classify_job,
    queue_deadline_s,
    update_deadline_status,
)
from vessim.wave.scheduler.runtime.progress import (
    estimate_remaining_time,
    update_running_jobs,
)
from vessim.wave.scheduler.observability.debug import (
    DEBUG_EVENT_COLUMNS,
    export_scheduler_debug_events,
    log_scheduler_event,
)
from vessim.wave.scheduler.observability.metrics import export_job_metrics


if TYPE_CHECKING:
    from vessim.cosim import Microgrid

# Controls verbose output.
SILENT = True

# Default lead time (s) for the SLA grid floor, mirroring
# ``PuzzleBudgetPolicy.sla_start_safety_margin_h`` (6h) so schedulers without a
# planner get the same deadline lead time the planner reserves for itself.
DEFAULT_SLA_GRID_LEAD_S = 6 * 3600.0


class SimulationComplete(Exception):
    pass 

class JobQueueScheduler(Controller): 
    _DEBUG_EVENT_COLUMNS = DEBUG_EVENT_COLUMNS

    def __init__(
        self, 
        resource_manager: ResourceManager, 
        job_queue: JobQueue, 
        cluster_type: str,
        dvfs_policy: DVFSPolicy,
        node_config: dict,
        scheduling_policy: Optional[SchedulingPolicy] = None,
        step_size: int = 60, 
        auto_stop: bool = True,
        adaptive_step: bool = True,
        node_failure_rate: float = 0.0,
        node_downtime_range: tuple[float, float] = (120.0, 600.0),
        failure_seed: int = 42,
        runtime_prediction_error_rate: float = 0.0,
        runtime_prediction_seed: int = 0,
        queue_definitions: Optional[list[QueueDefinition]] = None,
        sla_policy: str = "enforce",
        debug_logging_enabled: bool = True,
        out_dir: str = "",
    ):
        super().__init__(step_size=step_size)
        self.rm = resource_manager
        self.job_queue = job_queue
        self.cluster_type = cluster_type
        self.dvfs_policy = dvfs_policy
        self.node_config = node_config
        self.queue_definitions = list(queue_definitions or [])
        self.queue_definition_by_name = {
            definition.name: definition
            for definition in self.queue_definitions
        }
        base_policy = scheduling_policy or PriorityPolicy()
        if self.queue_definitions and not isinstance(base_policy, MultiQueuePolicy):
            self.scheduling_policy = MultiQueuePolicy(
                queue_definitions=self.queue_definitions,
                default_policy=base_policy,
                sla_policy=sla_policy,
            )
        else:
            self.scheduling_policy = base_policy
        # Reaches the bare policy too: without queue definitions there is no
        # MultiQueuePolicy wrapper to carry the system's SLA setting down.
        self.scheduling_policy.set_sla_policy(sla_policy)
        self.sla_policy = sla_policy
        self.step_size_seconds = step_size
        self.energy_budget = None
        self.energy_scheduler = None
        # SLA grid floor: when enabled, a job whose start deadline is within the
        # next step is allowed to draw grid beyond the (carbon-limited) energy
        # budget instead of being held until it misses. Off by default so the
        # budget's carbon behaviour is unchanged unless explicitly opted in.
        self.sla_grid_override = False
        # Lead time (s) before a start-deadline at which the SLA grid boost
        # begins releasing a job, so critical jobs drain gradually instead of
        # stampeding at the last step (which the physical node pool cannot
        # absorb, causing misses despite the boost). Defaults to 6h to match
        # PuzzleBudgetPolicy's ``sla_start_safety_margin_h``, so a budget-side
        # baseline gets the same deadline lead time the planner gives itself.
        # Env override lets a sweep tune it without touching the CLI wiring.
        self.sla_grid_lead_s = float(
            os.environ.get("SLA_GRID_LEAD_S", str(DEFAULT_SLA_GRID_LEAD_S))
        )
        # Time-weighted node utilization accumulators, so end-of-run stats can
        # report the utilization the cluster actually sustained rather than the
        # final-instant snapshot (which is ~0 once every job has finished).
        self._util_weighted_sum = 0.0
        self._util_time_total = 0.0
        self._util_peak = 0.0

        self.base_step_seconds = step_size
        self.auto_stop = auto_stop
        self.adaptive_step = adaptive_step
        
        self.node_failure_rate = node_failure_rate
        self.node_downtime_range = node_downtime_range
        self._failure_rng = random.Random(failure_seed)
        self.runtime_prediction_error_rate = runtime_prediction_error_rate
        self.runtime_prediction_seed = runtime_prediction_seed
        self.scheduling_policy.set_prediction_error_model(
            runtime_prediction_error_rate,
            runtime_prediction_seed,
        )
        self.debug_logging_enabled = debug_logging_enabled
        self.debug_events: list[dict[str, object]] = []
        self.out_dir = out_dir
        
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.done = False  # Scheduler has no more work.
        self._finalized = False
        
        self.total_energy_wh: float = 0.0  # Cumulative cluster energy in Wh.
        
        self.timeline: dict[str, list[tuple[datetime, Optional[datetime], Optional[int]]]] = {
            name: [] for name in self.rm.get_node_names()
        }
        
        self._cached_pending_jobs = None
        self._cached_pending_jobs_time = None
        self._cached_stats = None
        self._cached_power = None
        self._current_time = None
        self.node_idle_p = self.node_config[self.cluster_type]["idle_power_w"]
        self.cluster_mem_bw_gb_s = self.node_config[self.cluster_type].get("memory_bandwidth_gb_s", 100.0)
        
        self._last_log_time = None
        self._log_interval_seconds = 3600 
        
        if not SILENT:
            print(f"[{datetime.now()}][SCHEDULER] JobQueueScheduler initialized with step size {step_size}s")
    
    def _get_pending_jobs(self, current_time: Optional[datetime] = None) -> list[Job]:
        visibility_time = current_time or self._current_time
        if (
            self._cached_pending_jobs is None
            or self._cached_pending_jobs_time != visibility_time
        ):
            pending_jobs = [
                j for j in self.job_queue.jobs
                if j.status == "pending"
                and (visibility_time is None or j.arrival_time <= visibility_time)
            ]
            if visibility_time is not None:
                self._classify_visible_jobs(pending_jobs, visibility_time)
                self._update_queue_deadlines(pending_jobs, visibility_time)
            self._cached_pending_jobs = pending_jobs
            self._cached_pending_jobs_time = visibility_time
        return self._cached_pending_jobs

    def _has_pending_or_future_jobs(self) -> bool:
        return any(j.status == "pending" for j in self.job_queue.jobs)

    def _runnable_node_counts(
        self,
        pending_jobs: list[Job],
        time: Optional[datetime],
        extra_budget: float = 0.0,
    ) -> list[int]:
        """Per-job node demand of jobs that could actually run this step.

        Feeds the resource manager's ``runnable`` power-on gate: a node is only
        worth waking if some job would run on it now. A job is counted when the
        policy would dispatch it this step (``dispatchable_pending_jobs``) *and*
        its minimum running power (idle + lowest DVFS frequency, over all its
        nodes) fits the current energy budget. Jobs the plan withholds for a
        future/greener slot, or that no budget can afford, are excluded so the
        cluster is not powered up just to sit idle.

        ``extra_budget`` is the SLA grid boost (0 unless ``sla_grid_override``
        found a deadline-critical job): adding it here powers on nodes for a job
        the carbon-limited budget could not otherwise afford, so it is not
        starved into a miss while its nodes stay off.
        """
        dispatchable = self.scheduling_policy.dispatchable_pending_jobs(
            pending_jobs, time
        )
        if not dispatchable:
            return []

        budget = self.energy_budget
        if budget is None:
            return [job.num_nodes for job in dispatchable]
        budget = budget + max(0.0, extra_budget)

        frequencies = getattr(self.dvfs_policy, "available_frequencies", None)
        if frequencies:
            min_dyn = self.dvfs_policy.power_at_frequency(min(frequencies))
        else:
            min_dyn = 0.0
        min_run_power_per_node = self.node_idle_p + min_dyn

        return [
            job.num_nodes
            for job in dispatchable
            if job.num_nodes * min_run_power_per_node <= budget
        ]

    def _classify_visible_jobs(self, pending_jobs: list[Job], current_time: datetime) -> None:
        if not self.queue_definitions:
            return

        for job in pending_jobs:
            if job.queue_name is not None:
                continue

            classify_job(
                job=job,
                definitions=self.queue_definitions,
                runtime_s=job.requested_walltime_s,
                queue_enter_time=job.arrival_time,
            )

    def _update_queue_deadlines(self, pending_jobs: list[Job], current_time: datetime) -> None:
        if not self.queue_definitions:
            return
        for job in pending_jobs:
            update_deadline_status(job, current_time)

    def _reject_queue_node_limit_violations(
        self,
        pending_jobs: list[Job],
        current_time: datetime,
        step_size: int,
    ) -> list[Job]:
        if not self.queue_definitions:
            return pending_jobs

        rejected: list[tuple[Job, QueueDefinition]] = []
        for job in pending_jobs:
            definition = self.queue_definition_by_name.get(job.queue_name or "")
            if definition is None or definition.max_nodes is None:
                continue

            job.queue_max_nodes = definition.max_nodes
            if job.num_nodes > definition.max_nodes:
                job.mark_failed("requested_nodes_exceeds_queue_limit")
                rejected.append((job, definition))

        if not rejected:
            return pending_jobs

        self.jobs_failed += len(rejected)
        self._cached_pending_jobs = None
        self._cached_pending_jobs_time = None
        self._cached_stats = None

        for job, definition in rejected:
            log_scheduler_event(
                self,
                current_time,
                "job_failed",
                job=job,
                Reason=job.failure_reason,
                Step_Size_s=step_size,
                Notes=(
                    f"requested_nodes={job.num_nodes}; "
                    f"queue_max_nodes={definition.max_nodes}"
                ),
            )

        return self._get_pending_jobs(current_time)

    @property
    def silent(self) -> bool:
        return SILENT

    def _invalidate_power_cache(self) -> None:
        self._cached_power = None
    
    def _should_stop_simulation(self) -> bool:
        if not self.auto_stop:
            return False
        
        return (
            not self._has_pending_or_future_jobs()
            and not self.rm.has_running_jobs()
            and not self.rm.has_down_nodes()
        )
    
    def _compute_next_step(self) -> int:
        
        if not self.adaptive_step:
            return self.base_step_seconds

        next_completion = self.rm.get_next_job_completion_time()
        
        if next_completion is None:
            return self.base_step_seconds

        next_step = max(1, min(next_completion, self.base_step_seconds))

        return next_step
    
    def _submit_job(self, job: Job, current_time: datetime, assigned_nodes: list[str], frequency: float):
        if job.status != "pending":
            raise ValueError(f"Job {job.job_id} não está pending (status: {job.status})")
        
        predicted_t_exec = predict_execution_time(
            job,
            frequency,
            len(assigned_nodes),
            self.cluster_mem_bw_gb_s,
            prediction_error_rate=self.runtime_prediction_error_rate,
            prediction_seed=self.runtime_prediction_seed,
        )
        actual_work = compute_actual_execution_work(
            job=job,
            frequency=frequency,
            num_assigned_nodes=len(assigned_nodes),
            cluster_mem_bw_gb_s=self.cluster_mem_bw_gb_s,
        )
        prediction_error_factor = sample_prediction_error_factor(
            job_id=job.job_id,
            error_rate=self.runtime_prediction_error_rate,
            seed=self.runtime_prediction_seed,
        )

        job.start_execution(
            current_time=current_time,
            assigned_nodes=assigned_nodes,
            frequency=frequency,
            predicted_t_exec=predicted_t_exec,
            actual_work=actual_work,
            prediction_error_factor=prediction_error_factor,
        )
        
        nodes_str = ", ".join(assigned_nodes)
        if not SILENT:
            print(
                f"[{current_time}][SCHEDULER] Submitting job {job.job_id} to node(s): "
                f"{nodes_str} @ {frequency}GHz "
                f"(predicted={job.predicted_t_exec:.1f}s, actual={job.actual_t_exec:.1f}s)"
            )
        

    def step(self, time: datetime, p_delta: float, e_delta: float, actor_infos: dict) -> None:
        executed_step_size = self.step_size_seconds
        self._current_time = time
        self._cached_pending_jobs = None
        self._cached_pending_jobs_time = None
        self._cached_stats = None
        self._cached_power = None
        
        self.total_energy_wh += self.get_current_power() * executed_step_size / 3600
        
        if self._should_stop_simulation():
            if not self.done:
                self.done = True
                if not getattr(self.rm, "keep_idle_nodes_on", False):
                    for node_name in list(self.rm.get_idle_nodes()):
                        self.rm.power_off_node(node_name, time)
                self._invalidate_power_cache()
                
                if self.energy_scheduler is not None:
                    self.energy_scheduler.notification(
                        {
                            'type': 'ClusterComplete',
                            'cluster_type': self.cluster_type,
                            'released_budget': self.energy_budget,
                        },
                        self
                    )
            
            return

        update_running_jobs(self, executed_step_size)
        
        recovered = self.rm.update_down_nodes(executed_step_size, time)
        if recovered:
            self._invalidate_power_cache()
        
        completed_jobs: list[Job] = []
        for allocation in list(self.rm.get_running_allocations()):
            job = allocation.job
            if job.remaining_time > 0:
                continue

            completed_job, completed_node_names = self.rm.complete_job(job.job_id, time)
            job = completed_job or job
            for node_name in completed_node_names:
                if self.timeline[node_name] and self.timeline[node_name][-1][1] is None:
                    start_time, _, job_id = self.timeline[node_name][-1]
                    self.timeline[node_name][-1] = (start_time, time, job_id)

            if job.status == 'running':
                job.mark_completed(time)
                self.jobs_completed += 1
                completed_jobs.append(job)

        if completed_jobs:
            self._invalidate_power_cache()
            power_after_completion = self.get_current_power()
            for job in completed_jobs:
                wait_time = (job.start_time - job.arrival_time).total_seconds() if job.start_time else ""
                real_time = (job.completion_time - job.start_time).total_seconds() if job.start_time and job.completion_time else ""
                actual_model_time = job.actual_t_exec
                contention_overhead = (
                    real_time - actual_model_time
                    if real_time != "" and actual_model_time != ""
                    else ""
                )
                log_scheduler_event(
                    self,
                    time,
                    "job_completed",
                    job=job,
                    Step_Size_s=executed_step_size,
                    Power_After_W=power_after_completion,
                    Real_Exec_Time_s=real_time,
                    Wait_Time_s=wait_time,
                    Contention_Overhead_s=contention_overhead,
                    Notes="job reached zero remaining work",
                )

        failure = self.rm.inject_node_failure(
            time=time,
            step_size_seconds=executed_step_size,
            base_step_seconds=self.base_step_seconds,
            node_failure_rate=self.node_failure_rate,
            node_downtime_range=self.node_downtime_range,
            rng=self._failure_rng,
        )
        if failure is not None:
            for node_name in failure.failed_nodes:
                if self.timeline[node_name] and self.timeline[node_name][-1][1] is None:
                    start_time, _, job_id = self.timeline[node_name][-1]
                    self.timeline[node_name][-1] = (start_time, time, job_id)

            self.jobs_failed += 1
            self._invalidate_power_cache()
            log_scheduler_event(
                self,
                time,
                "job_failed",
                job=failure.job,
                Step_Size_s=executed_step_size,
                Power_After_W=self.get_current_power(),
                Notes=f"failed_node={failure.selected_node}",
            )

            if not self.silent:
                print(
                    f"[{time}][SCHEDULER] Node {failure.selected_node} FAILED - "
                    f"Job {failure.job.job_id} marked as failed"
                )
        
        ready_nodes = self.rm.check_nodes_ready_for_new_job(time, executed_step_size)
        pending_jobs = self._get_pending_jobs(time)
        pending_jobs = self._reject_queue_node_limit_violations(
            pending_jobs,
            time,
            executed_step_size,
        )
        
        # Active power management: the RM owns node power-state transitions.
        # ``pending_node_demand`` bounds the eager gate; ``runnable_node_counts``
        # drives the runnable gate — the node demand of only the jobs the policy
        # would dispatch this step (so we don't wake the cluster for jobs a plan
        # withholds for a greener slot), filtered to those that can actually run
        # within the current energy budget.
        pending_node_demand = sum(job.num_nodes for job in pending_jobs)
        # SLA grid floor (opt-in): a deadline-critical job must get its nodes
        # powered on and its budget lifted, or a carbon-limited budget starves
        # it into a miss while its nodes stay off.  The same boost is applied to
        # node power management (max_powerable + power-on headroom), the runnable
        # gate, and the dispatch budget below, so all three see the deadline.
        sla_boost = 0.0
        if self.energy_budget is not None and self.sla_grid_override:
            sla_boost = self._sla_grid_boost(pending_jobs, time)
        power_mgmt_budget = self.energy_budget
        if power_mgmt_budget is not None and sla_boost > 0:
            power_mgmt_budget = power_mgmt_budget + sla_boost
        runnable_node_counts = self._runnable_node_counts(
            pending_jobs, time, extra_budget=sla_boost
        )
        power_state_changed = self.rm.manage_node_power(
            time=time,
            energy_budget=power_mgmt_budget,
            current_power=self.get_current_power(),
            node_idle_p=self.node_idle_p,
            has_pending_jobs=bool(pending_jobs),
            pending_node_demand=pending_node_demand,
            runnable_node_counts=runnable_node_counts,
        )
        if power_state_changed:
            self._invalidate_power_cache()
        
        ready_nodes = self.rm.check_nodes_ready_for_new_job(time, executed_step_size)
        pending_jobs = self._get_pending_jobs(time)

        if ready_nodes and pending_jobs and not SILENT:
            top_jobs_str = ", ".join([f"ID:{j.job_id}(P{j.priority})" for j in pending_jobs[:5]])
            print(f"[{time}][SCHEDULER] Next in Queue (Top 5): [{top_jobs_str}]")

        # Wait for the EnergyScheduler to assign the first budget.
        if self.energy_scheduler is not None and self.energy_budget is None:
            decisions = []
        else:
            # SLA grid floor: lift the budget by the power needed to start jobs
            # whose deadline is imminent, so a carbon-limited budget can't starve
            # them into a miss. No-op unless enabled and such jobs exist.
            effective_budget = self.energy_budget
            if effective_budget is not None and self.sla_grid_override:
                boost = self._sla_grid_boost(pending_jobs, time)
                if boost > 0:
                    effective_budget = effective_budget + boost
            # Delegate scheduling decisions to the policy.
            decisions = select_jobs_with_optional_time(
                policy=self.scheduling_policy,
                pending_jobs=pending_jobs,
                ready_nodes=ready_nodes,
                dvfs_policy=self.dvfs_policy,
                energy_budget=effective_budget,
                current_power=self.get_current_power(),
                node_idle_p=self.node_idle_p,
                current_time=time,
            )

        scheduled_any = False
        for decision in decisions:
            job = decision.job
            assigned_nodes = decision.assigned_nodes
            frequency = decision.frequency
            node_dynamic_power = decision.dynamic_power_w
            node_frequency = frequency if node_dynamic_power is not None else None
            power_before_start = self.get_current_power()

            self._submit_job(job, time, assigned_nodes, frequency)

            started_nodes = self.rm.start_job(
                job,
                assigned_nodes,
                time,
                frequency=node_frequency,
                dynamic_power_w=node_dynamic_power,
            )
            for node_name in started_nodes:
                self.timeline[node_name].append((time, None, job.job_id))

            self._invalidate_power_cache()
            scheduled_any = True
            log_scheduler_event(
                self,
                time,
                "job_started",
                job=job,
                Step_Size_s=executed_step_size,
                Power_Before_W=power_before_start,
                Power_After_W=self.get_current_power(),
                Frequency_Before_GHz="",
                Frequency_After_GHz=frequency,
                Notes=f"dynamic_power_w={node_dynamic_power}; assigned_nodes={','.join(assigned_nodes)}",
            )
        
        if scheduled_any:
            self._cached_pending_jobs = None
            self._cached_pending_jobs_time = None

        self.rm.manage_runtime_dvfs(
            time=time,
            energy_budget=self.energy_budget,
            dvfs_policy=self.dvfs_policy,
            get_current_power=self.get_current_power,
            invalidate_power_cache=self._invalidate_power_cache,
            estimate_remaining_time=partial(estimate_remaining_time, self),
            has_pending_jobs=bool(self._get_pending_jobs(time)),
            log_event=partial(log_scheduler_event, self),
            silent=self.silent,
        )

        # Sample node utilization for this step (time-weighted mean + peak).
        step_util = self.rm.get_cluster_utilization()
        self._util_weighted_sum += step_util * executed_step_size
        self._util_time_total += executed_step_size
        if step_util > self._util_peak:
            self._util_peak = step_util

        next_step = self._compute_next_step()
        self.step_size_seconds = next_step
        log_scheduler_event(
            self,
            time,
            "step_summary",
            Step_Size_s=executed_step_size,
            Next_Step_s=next_step,
            Power_After_W=self.get_current_power(),
            Notes="end-of-step state after scheduling and runtime DVFS",
        )
        
        if not SILENT and self.step_size_seconds != 60:
            print(f"[{time}][SCHEDULER] Next step size: {self.step_size_seconds}s")

        if not SILENT:
            if self._last_log_time is None or (time - self._last_log_time).total_seconds() >= self._log_interval_seconds:
                self._last_log_time = time
                stats = self.get_stats()
                power = self.get_current_power()
                budget_str = f"{self.energy_budget:.0f}W" if self.energy_budget else "N/A"
                print(
                    f"\n | [{time}][{self.cluster_type}]\n"
                    f" | Jobs: {stats['completed']}/{stats['total']} done, {stats['failed']} failed \n"
                    f" | {stats['running']} running, {stats['pending']} pending \n"
                    f" | Nodes: {stats['busy_nodes']}/{stats['total_nodes']} busy, {stats['down_nodes']} down \n"
                    f" | Power: {power:.0f}W / {budget_str} \n"
                    f" | Step: {self.step_size_seconds}s \n"
                )
    
    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True

        last_time = None
        for entries in self.timeline.values():
            if entries:
                for start, end, _ in entries:
                    if end is not None:
                        last_time = max(last_time, end) if last_time else end
        
        if last_time:
            for node_name, entries in self.timeline.items():
                if entries and entries[-1][1] is None:
                    start_time, _, job_id = self.timeline[node_name][-1]
                    self.timeline[node_name][-1] = (start_time, last_time, job_id)
                    
        export_scheduler_debug_events(self)
        export_job_metrics(self)

    def get_current_power(self) -> float:
        if self._cached_power is None:
            self._cached_power = sum(node.measure() for node in self.rm.nodes.values())
        return self._cached_power

    def set_energy_budget(self, budget: float):
        self.energy_budget = budget

    def _sla_grid_boost(self, pending_jobs: list[Job], time: datetime) -> float:
        """Extra watts to add to the budget so deadline-imminent jobs can start.

        A pending job is deadline-critical when its remaining start-deadline
        slack is at most ``sla_grid_lead_s`` — it must start soon or miss. We add
        the maximum dynamic power those jobs would draw (over all their nodes),
        so the policy can dispatch them at whatever frequency it picks, spilling
        to grid rather than holding them under a carbon-limited budget. Returns
        0 when nothing is critical, so the carbon behaviour is untouched in the
        common case.

        ``sla_grid_lead_s`` gives lead time before the deadline so jobs release
        gradually instead of stampeding at the last step (which the physical
        node pool cannot drain, causing misses despite the boost).
        """
        if not pending_jobs:
            return 0.0
        freqs = getattr(self.dvfs_policy, "available_frequencies", None)
        if not freqs:
            return 0.0
        max_dyn = self.dvfs_policy.power_at_frequency(max(freqs))
        buffer_s = max(float(self.step_size_seconds), float(getattr(self, "sla_grid_lead_s", 0.0)))
        boost = 0.0
        for job in pending_jobs:
            slack = queue_deadline_s(job, time)
            if slack is not None and slack <= buffer_s:
                boost += max_dyn * int(getattr(job, "num_nodes", 0) or 0)
        return boost

    def get_stats(self) -> dict:
        if self._cached_stats is not None:
            return self._cached_stats
        rm_stats = self.rm.get_stats()
        
        mean_util = (
            self._util_weighted_sum / self._util_time_total
            if self._util_time_total > 0 else 0.0
        )
        self._cached_stats = {
            'total': self.job_queue.total_jobs(),
            'pending': len(self._get_pending_jobs()),
            'running': self.rm.get_running_jobs_count(),
            'completed': self.jobs_completed,
            'failed': self.jobs_failed,
            **rm_stats,
            # ``utilization`` (from rm_stats) is the instantaneous snapshot;
            # these two are the run-wide time-weighted mean and peak.
            'utilization_mean': mean_util,
            'utilization_peak': self._util_peak,
        }
        return self._cached_stats
