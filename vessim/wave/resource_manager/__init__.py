"""
Resource manager state and operations for WAVE clusters.

   The resource manager is the execution-side owner of cluster state: node
   availability, running allocations, failures, power state, and runtime DVFS.
   Schedulers request transitions through this package instead of mutating nodes
   directly.
"""

from vessim.wave.resource_manager.allocation import RunningAllocation
from vessim.wave.resource_manager.failures import NodeFailureEvent
from vessim.wave.resource_manager.manager import ResourceManager

__all__ = ["NodeFailureEvent", "ResourceManager", "RunningAllocation"]
