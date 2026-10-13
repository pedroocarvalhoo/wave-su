"""
Resource manager implementation for WAVE clusters.

   `ResourceManager` owns node-level execution state and provides the operations
   used by schedulers: start, complete, fail, recover, power on/off, and runtime
   DVFS changes. Keep direct `ComputeNode` mutations centralized here so
   scheduler code remains orchestration rather than state management.
"""

from datetime import datetime
from typing import Callable, Optional
from vessim.power_meter import ComputeNode
from vessim.wave.cluster.job_queue import Job
from vessim.wave.resource_manager.allocation import (
    AllocationMixin,
    RunningAllocation,
)
from vessim.wave.resource_manager.failures import FailureMixin


# Node power-on gating modes (see ``manage_node_power``). Researchers can add
# their own strategy by handling a new mode string in ``_power_on_target``.
POWER_ON_GATE_EAGER = "eager"        # legacy: wake toward all pending demand
POWER_ON_GATE_RUNNABLE = "runnable"  # only wake if a job can actually run now
VALID_POWER_ON_GATES = (POWER_ON_GATE_EAGER, POWER_ON_GATE_RUNNABLE)


class ResourceManager(AllocationMixin, FailureMixin):

    def __init__(
        self,
        nodes: list[ComputeNode],
        power_on_gate: str = POWER_ON_GATE_EAGER,
        keep_idle_nodes_on: bool = False,
    ):

        self.nodes: dict[str, ComputeNode] = {
            node.name: node
            for node in nodes
        }

        # Canonical running state: one allocation per running job.
        self.running_allocations: dict[int, RunningAllocation] = {}
        self.node_to_job_id: dict[str, int] = {}

        self._idle_nodes: set[str] = set(self.nodes.keys())
        self._busy_nodes: set[str] = set()
        self._down_nodes: set[str] = set()
        self._off_nodes: set[str] = set()

        # How aggressively idle nodes are switched on ahead of work. See the
        # module-level POWER_ON_GATE_* constants and ``manage_node_power``.
        if power_on_gate not in VALID_POWER_ON_GATES:
            raise ValueError(
                f"power_on_gate must be one of {VALID_POWER_ON_GATES}, "
                f"got {power_on_gate!r}"
            )
        self.power_on_gate = power_on_gate
        self.keep_idle_nodes_on = keep_idle_nodes_on
        
    
    def get_node_names(self) -> list[str]:
        return list(self.nodes.keys())
    
    def get_down_nodes(self) -> list[str]:
        return list(self._down_nodes)
    
    def has_down_nodes(self) -> bool:
        return len(self._down_nodes) > 0
    
    def get_off_nodes(self) -> list[str]:
        return list(self._off_nodes)
    
    def has_off_nodes(self) -> bool:
        return len(self._off_nodes) > 0
    
    def get_powered_on_node_count(self) -> int:
        return len(self._idle_nodes) + len(self._busy_nodes)

    def get_node(self, node_name: str) -> ComputeNode:
        return self.nodes[node_name]

    def _power_on_target(
        self,
        pending_node_demand: Optional[int],
        runnable_node_counts: Optional[list[int]],
        max_powerable: int,
    ) -> int:
        """How many *powered-on* nodes to aim for, given the gating mode.

        ``max_powerable`` is the most nodes that could be online this step
        (already-idle plus what the budget can afford to wake). Returns the
        target number of online nodes; the caller wakes the difference.

        Edit here to add a custom power-on strategy — this is the single
        decision point researchers are expected to tune.

        - ``eager``: wake toward the pending node demand (legacy). ``None``
          demand means the greedy budget fill used by direct callers/tests.
        - ``runnable``: only wake nodes if at least one job could *actually
          run* on them now. ``runnable_node_counts`` is the per-job node
          demand of the jobs the scheduler would dispatch this step, already
          filtered to those that fit the energy budget. If none fit within
          ``max_powerable`` there is no point waking anything.
        """
        if self.power_on_gate == POWER_ON_GATE_EAGER:
            if pending_node_demand is None:
                return max_powerable  # legacy greedy fill
            return min(max_powerable, max(0, pending_node_demand))

        # POWER_ON_GATE_RUNNABLE
        if not runnable_node_counts:
            return 0
        feasible = [n for n in runnable_node_counts if 0 < n <= max_powerable]
        if not feasible:
            # No single runnable job fits the nodes we could turn on: waking
            # any would just burn idle power, so keep them off.
            return 0
        # Wake enough for the runnable jobs that fit, capped at what we can
        # afford. ``sum`` lets several small jobs share the pool.
        return min(max_powerable, sum(feasible))

    def manage_node_power(
        self,
        time: datetime,
        energy_budget: Optional[float],
        current_power: float,
        node_idle_p: float,
        has_pending_jobs: bool,
        pending_node_demand: Optional[int] = None,
        runnable_node_counts: Optional[list[int]] = None,
    ) -> bool:
        # Static-on operation leaves idle nodes powered and charges their idle draw.
        if self.keep_idle_nodes_on:
            return False
        # Power idle nodes off/on while keeping node state changes inside the RM.
        #
        # A running node costs idle+dynamic power, so ``budget // idle_p`` is
        # far more nodes than can ever actually run; lighting up the whole
        # cluster just to keep it idle wastes most of the budget. The
        # ``power_on_gate`` mode (see ``_power_on_target``) decides how many
        # nodes to wake:
        #   * ``eager``    — wake toward ``pending_node_demand`` (legacy).
        #   * ``runnable`` — wake only if a job in ``runnable_node_counts``
        #                    (the jobs the scheduler would dispatch now, that
        #                    fit the budget) can actually run on them.
        if energy_budget is None:
            return False

        changed = False

        if current_power > energy_budget:
            for node_name in sorted(self.get_idle_nodes()):
                if current_power <= energy_budget:
                    break
                if self.power_off_node(node_name, time):
                    current_power -= node_idle_p
                    changed = True

        # Nodes we could have online this step: those already idle plus the
        # off-nodes the remaining budget headroom can afford to wake.
        idle_count = len(self.get_idle_nodes())
        affordable_wakes = (
            int((energy_budget - current_power) // node_idle_p)
            if node_idle_p > 0 else len(self.get_off_nodes())
        )
        max_powerable = idle_count + max(0, min(affordable_wakes, len(self.get_off_nodes())))

        target_online = self._power_on_target(
            pending_node_demand=pending_node_demand,
            runnable_node_counts=runnable_node_counts,
            max_powerable=max_powerable,
        )

        # Shed idle nodes beyond the target so we don't hold a warm pool no
        # runnable work can use.
        surplus = idle_count - max(0, target_online)
        if surplus > 0:
            for node_name in sorted(self.get_idle_nodes()):
                if surplus <= 0:
                    break
                if self.power_off_node(node_name, time):
                    current_power -= node_idle_p
                    surplus -= 1
                    changed = True

        if has_pending_jobs and self.has_off_nodes():
            headroom = energy_budget - current_power
            deficit = max(0, target_online - len(self.get_idle_nodes()))
            for node_name in sorted(self.get_off_nodes()):
                if headroom < node_idle_p or deficit <= 0:
                    break
                if self.power_on_node(node_name, time):
                    headroom -= node_idle_p
                    current_power += node_idle_p
                    deficit -= 1
                    changed = True

        return changed

    def manage_runtime_dvfs(
        self,
        time: datetime,
        energy_budget: Optional[float],
        dvfs_policy,
        get_current_power: Callable[[], float],
        invalidate_power_cache: Callable[[], None],
        estimate_remaining_time: Callable[[Job], float],
        has_pending_jobs: bool,
        log_event: Callable[..., None],
        silent: bool = True,
    ) -> None:
        if energy_budget is None:
            return
        if not getattr(dvfs_policy, "runtime_dvfs_enabled", True):
            return

        running_jobs = [
            allocation.job
            for allocation in self.get_running_allocations()
            if allocation.job.status == "running"
        ]
        if not running_jobs:
            return

        tolerance = 1e-6
        current_power = get_current_power()
        downscale_order = sorted(
            running_jobs,
            key=lambda job: (
                job.priority,
                -job.remaining_time,
                job.job_id,
            ),
        )

        while current_power > energy_budget + tolerance:
            changed = False

            for job in downscale_order:
                lower_frequency = self._next_lower_frequency(
                    dvfs_policy,
                    self._current_job_frequency(job, dvfs_policy),
                )
                if lower_frequency is None:
                    continue

                if self._set_runtime_frequency(
                    job=job,
                    time=time,
                    new_frequency=lower_frequency,
                    reason="runtime_dvfs_downscale",
                    dvfs_policy=dvfs_policy,
                    get_current_power=get_current_power,
                    invalidate_power_cache=invalidate_power_cache,
                    estimate_remaining_time=estimate_remaining_time,
                    log_event=log_event,
                    silent=silent,
                ):
                    changed = True
                    current_power = get_current_power()
                    if current_power <= energy_budget + tolerance:
                        break

            if not changed:
                break

        if current_power > energy_budget + tolerance:
            log_event(
                time,
                "runtime_dvfs_budget_unreachable",
                Power_After_W=current_power,
                Notes="all running jobs are at minimum frequency or no valid downscale remained",
            )
            return

        if not getattr(dvfs_policy, "allow_runtime_upscale", True):
            return
        if has_pending_jobs:
            return

        upscale_order = sorted(
            running_jobs,
            key=lambda job: (
                -job.priority,
                job.remaining_time,
                job.job_id,
            ),
        )

        while True:
            changed = False

            for job in upscale_order:
                higher_frequency = self._next_higher_frequency(
                    dvfs_policy,
                    self._current_job_frequency(job, dvfs_policy),
                )
                if higher_frequency is None:
                    continue

                projected_power = current_power + self._dynamic_power_delta_for_frequency(
                    job,
                    dvfs_policy,
                    higher_frequency,
                )
                if projected_power > energy_budget + tolerance:
                    continue

                if self._set_runtime_frequency(
                    job=job,
                    time=time,
                    new_frequency=higher_frequency,
                    reason="runtime_dvfs_upscale",
                    dvfs_policy=dvfs_policy,
                    get_current_power=get_current_power,
                    invalidate_power_cache=invalidate_power_cache,
                    estimate_remaining_time=estimate_remaining_time,
                    log_event=log_event,
                    silent=silent,
                ):
                    changed = True
                    current_power = get_current_power()

            if not changed:
                break

    def _current_job_frequency(self, job: Job, dvfs_policy) -> float:
        if job.assigned_frequency and job.assigned_frequency > 0.0:
            return job.assigned_frequency

        for node_name in self.get_job_node_names(job):
            node = self.get_node(node_name)
            if node.current_frequency and node.current_frequency > 0.0:
                return node.current_frequency

        return dvfs_policy.base_frequency_ghz

    def _next_lower_frequency(self, dvfs_policy, current_frequency: float) -> Optional[float]:
        lower = [
            freq
            for freq in dvfs_policy.available_frequencies
            if freq < current_frequency - 1e-6
        ]
        return lower[-1] if lower else None

    def _next_higher_frequency(self, dvfs_policy, current_frequency: float) -> Optional[float]:
        higher = [
            freq
            for freq in dvfs_policy.available_frequencies
            if freq > current_frequency + 1e-6
        ]
        return higher[0] if higher else None

    def _dynamic_power_delta_for_frequency(self, job: Job, dvfs_policy, new_frequency: float) -> float:
        new_dynamic_power = dvfs_policy.power_at_frequency(new_frequency)
        delta = 0.0

        for node_name in self.get_job_node_names(job):
            node = self.get_node(node_name)
            current_job = node.current_job
            if not node.is_busy() or (
                current_job is not job
                and getattr(current_job, "job_id", None) != job.job_id
            ):
                continue
            delta += new_dynamic_power - node.get_dynamic_power()

        return delta

    def _set_runtime_frequency(
        self,
        job: Job,
        time: datetime,
        new_frequency: float,
        reason: str,
        dvfs_policy,
        get_current_power: Callable[[], float],
        invalidate_power_cache: Callable[[], None],
        estimate_remaining_time: Callable[[Job], float],
        log_event: Callable[..., None],
        silent: bool,
    ) -> bool:
        current_frequency = self._current_job_frequency(job, dvfs_policy)
        if abs(current_frequency - new_frequency) < 1e-6:
            return False

        power_before = get_current_power()
        remaining_time_before = job.remaining_time
        dynamic_power = dvfs_policy.power_at_frequency(new_frequency)
        changed_nodes = self.set_job_frequency(
            job,
            new_frequency,
            dynamic_power,
            time,
        )

        if changed_nodes == 0:
            return False

        job.remaining_time = estimate_remaining_time(job)
        invalidate_power_cache()
        power_after = get_current_power()

        log_event(
            time,
            reason,
            job=job,
            Reason=reason,
            Power_Before_W=power_before,
            Power_After_W=power_after,
            Frequency_Before_GHz=current_frequency,
            Frequency_After_GHz=new_frequency,
            Remaining_Time_Before_s=remaining_time_before,
            Remaining_Time_After_s=job.remaining_time,
            Notes=f"changed_nodes={changed_nodes}",
        )

        if not silent:
            print(
                f"[{time}][RESOURCE_MANAGER] Runtime DVFS: job {job.job_id} "
                f"{current_frequency:.2f}GHz -> {new_frequency:.2f}GHz "
                f"on {changed_nodes} node(s)"
            )

        return True
    
    def power_off_node(self, node_name: str, time) -> bool:
        if node_name not in self._idle_nodes:
            return False
        node = self.nodes[node_name]
        node.power_off(time)
        self._idle_nodes.discard(node_name)
        self._off_nodes.add(node_name)
        return True
    
    def power_on_node(self, node_name: str, time) -> bool:
        if node_name not in self._off_nodes:
            return False
        node = self.nodes[node_name]
        node.power_on(time)
        self._off_nodes.discard(node_name)
        self._idle_nodes.add(node_name)
        return True
    
    def get_cluster_utilization(self) -> float:
        if not self.nodes:
            return 0.0
        
        return len(self._busy_nodes) / len(self.nodes)
    
    def get_stats(self) -> dict:
        return {
            'total_nodes': len(self.nodes),
            'idle_nodes': len(self._idle_nodes),
            'busy_nodes': len(self._busy_nodes),
            'down_nodes': len(self._down_nodes),
            'off_nodes': len(self._off_nodes),
            'powered_on': self.get_powered_on_node_count(),
            'utilization': self.get_cluster_utilization(),
            'total_jobs_executed': sum(
                node.total_jobs_executed for node in self.nodes.values()
            ),
        }
