from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from scripts.evaluation_sensitivity import (
    TIMEOUT_S,
    command_for,
    guarded_process,
    matrix,
)


def test_matrix_has_26_unique_runs_and_three_shared_energy_aliases(tmp_path):
    cases = matrix()
    assert len(cases) == 26
    assert len({case.case_id for case in cases}) == 26
    assert sum(case.experiment == "carbon_threshold" for case in cases) == 20
    assert sum(case.experiment == "energy_policy" for case in cases) == 6
    aliases = {case.energy_alias: case.case_id for case in cases if case.energy_alias}
    assert aliases == {
        "ep_carbon_fifo": "ct250_fifo",
        "ep_carbon_priority": "ct250_priority",
        "ep_carbon_eft": "ct250_eft",
    }
    assert TIMEOUT_S == 900


def test_command_keeps_pv_and_wind_on_every_case(tmp_path):
    case = next(row for row in matrix() if row.case_id == "ct225_priority")
    command = command_for(case, tmp_path)
    assert command[command.index("--renewable-sources") + 1] == "pv,wind"
    assert command.count("--renewable-source-scale") == 2
    assert "--no-sla-grid-override" in command
    assert "--carbon-grid-threshold" in command


def test_guard_allows_normal_exit(tmp_path):
    log = tmp_path / "normal.log"
    outcome = guarded_process(
        [sys.executable, "-c", "print('done')"],
        log,
        os.environ.copy(),
        tmp_path / "normal",
        timeout_s=2,
    )
    assert outcome["exit_code"] == 0
    assert outcome["termination_reason"] is None
    assert outcome["process_group_empty"]
    assert "done" in log.read_text()


def test_guard_kills_simulation_group_at_deadline(tmp_path):
    log = tmp_path / "timeout.log"
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('started', flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    outcome = guarded_process(
        [sys.executable, "-c", parent_code],
        log,
        os.environ.copy(),
        tmp_path / "timeout",
        timeout_s=0.25,
    )
    elapsed = time.monotonic() - started
    assert outcome["termination_reason"] == "wall_timeout"
    assert outcome["exit_code"] != 0
    assert outcome["process_group_empty"]
    assert elapsed < 3
    assert "started" in log.read_text()


def test_guard_rejects_timeout_above_user_limit(tmp_path):
    with pytest.raises(ValueError, match="900"):
        guarded_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            tmp_path / "invalid.log",
            os.environ.copy(),
            tmp_path / "invalid",
            timeout_s=900.01,
        )


def test_guard_kills_child_that_outlives_parent(tmp_path):
    marker = tmp_path / "child_finished"
    child_code = (
        "import time; from pathlib import Path; "
        "time.sleep(0.8); Path(" + repr(str(marker)) + ").write_text('still running')"
    )
    parent_code = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('parent done', flush=True)"
    )
    outcome = guarded_process(
        [sys.executable, "-c", parent_code],
        tmp_path / "orphan.log",
        os.environ.copy(),
        tmp_path / "orphan",
        timeout_s=2,
    )
    assert outcome["termination_reason"] == "orphan_processes"
    assert outcome["process_group_empty"]
    time.sleep(0.9)
    assert not marker.exists()
