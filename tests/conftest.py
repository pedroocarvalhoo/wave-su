from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from vessim.power_meter import ComputeNode
from vessim.wave.cluster import Job
from vessim.wave.energy import AdaptiveDVFSPolicy


@pytest.fixture
def sim_time() -> datetime:
    return datetime(2023, 1, 1, 0, 0, 0)


@pytest.fixture
def make_job(sim_time):
    def _make_job(
        job_id: int = 1,
        *,
        cpi: float = 1.0,
        num_instructions: float = 1e9,
        num_nodes: int = 1,
        memory_gb: float = 1.0,
        priority: int = 2,
        requested_walltime_s: float = 3600.0,
        predicted_t_exec: float = 0.0,
        actual_t_exec: float = 0.0,
    ) -> Job:
        return Job(
            job_id=job_id,
            cpi=cpi,
            num_instructions=num_instructions,
            arrival_time=sim_time,
            num_nodes=num_nodes,
            requested_walltime_s=requested_walltime_s,
            memory_gb=memory_gb,
            priority=priority,
            predicted_t_exec=predicted_t_exec,
            actual_t_exec=actual_t_exec,
        )

    return _make_job


@pytest.fixture
def make_node():
    def _make_node(
        name: str,
        *,
        max_p: float = 100.0,
        idle_p: float = 10.0,
        base_frequency_ghz: float = 2.0,
    ) -> ComputeNode:
        return ComputeNode(
            max_p=max_p,
            idle_p=idle_p,
            name=name,
            base_frequency_ghz=base_frequency_ghz,
        )

    return _make_node


@pytest.fixture
def dvfs_policy() -> AdaptiveDVFSPolicy:
    return AdaptiveDVFSPolicy(
        frequencies={1.0: 50.0, 2.0: 100.0},
        base_frequency_ghz=2.0,
    )


@pytest.fixture
def scheduler_stub(dvfs_policy):
    return SimpleNamespace(dvfs_policy=dvfs_policy)
