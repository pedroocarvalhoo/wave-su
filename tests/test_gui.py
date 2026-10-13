"""Dashboard regressions for experiment selection, log lifetime and published data."""

import asyncio
import io
import json
import signal
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from gui import app as dashboard
from gui import results
from gui.simulation import build_command
from scripts.simulation_energy import parse_args


def parsed_request(**options):
    request = dashboard.SimulationRequest(system="test", **options)
    return parse_args(["test"], {"pv": {}, "wind": {}},
                      build_command(request, "python", "script")[3:])


@pytest.mark.parametrize("budget", ["grid", "carbon-minimizing", "daytime-renewable", "puzzle"])
@pytest.mark.parametrize("scheduler", ["fifo", "priority", "eft-dvfs", "puzzle"])
def test_supported_choices_are_retained_and_incompatible_puzzle_is_rejected(scheduler, budget):
    if scheduler == "puzzle" and budget != "puzzle":
        with pytest.raises(ValueError, match="Puzzle scheduling requires"):
            parsed_request(scheduling_policy=scheduler, budget_policy=budget)
        return
    args = parsed_request(scheduling_policy=scheduler, budget_policy=budget)
    assert args.scheduling_policy == scheduler
    assert args.budget_policy == budget


@pytest.mark.parametrize("override", [None, False, True])
def test_contextual_dispatch_and_sla_defaults_survive_gui_translation(override):
    args = parsed_request(sla_grid_override=override, puzzle_fallback_dispatch=override,
                          renewable_charge_fraction=None, plot_workers=None)
    assert args.sla_grid_override is override
    assert args.puzzle_fallback_dispatch is override
    assert args.renewable_charge_fraction is None
    assert args.plot_workers is None


def test_thesis_control_options_are_passed_to_the_simulator(tmp_path):
    args = parsed_request(
        jobs_dir=str(tmp_path), run_tag="control", sla_policy="report_only",
        start_nodes_off=False, keep_idle_nodes_on=True, no_snapshot=True,
        no_plots=True, renewable_sources=["pv", "wind"],
        renewable_source_scales={"pv": 180000, "wind": 260000},
        allow_grid_charging=True, dvfs_policy="fixed", carbon_grid_threshold=200,
        puzzle_carbon_aware_placement=False, puzzle_carbon_regret_g_per_h=None,
    )
    assert args.jobs_dir == str(tmp_path)
    assert args.run_tag == "control"
    assert args.sla_policy == "report_only"
    assert args.keep_idle_nodes_on and not args.start_nodes_off
    assert args.no_plots and args.no_snapshot
    assert args.renewable_sources == "pv,wind"
    assert set(args.renewable_source_scale) == {"pv=180000.0", "wind=260000.0"}
    assert args.allow_grid_charging and args.dvfs_policy == "fixed"
    assert args.carbon_grid_threshold == 200
    assert not args.puzzle_carbon_aware_placement
    assert args.puzzle_carbon_regret_g_per_h is None


def test_conflicting_node_power_options_are_rejected_before_launch():
    with pytest.raises(ValueError, match="Start Nodes Off"):
        dashboard.SimulationRequest(system="test", keep_idle_nodes_on=True)


def test_start_endpoint_validates_cli_and_runs_without_sse(monkeypatch):
    proc = SimpleNamespace(pid=123, stdout=io.StringIO("done\n"), wait=lambda: 0)
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return proc

    monkeypatch.setattr(dashboard, "_load_supercomputers", lambda: {"test": {}})
    monkeypatch.setattr(dashboard, "_load_json", lambda _: {"pv": {}, "wind": {}})
    monkeypatch.setattr(dashboard, "_find_python", lambda: "python")
    monkeypatch.setattr(dashboard.subprocess, "Popen", launch)
    for name, value in [("_sim_process", None), ("_sim_log", []),
                        ("_sim_config", {}), ("_sim_running", False), ("_sim_output_task", None)]:
        monkeypatch.setattr(dashboard, name, value)

    async def exercise():
        response = await dashboard.start_simulation(dashboard.SimulationRequest(
            system="test", scheduling_policy="fifo", budget_policy="grid"))
        assert response["status"] == "started"
        await dashboard._sim_output_task
        assert dashboard._sim_config["budget_policy"] == "grid"
        assert dashboard._sim_config["returncode"] == 0
        assert dashboard._sim_log == ["done"]
        assert commands[0][3] == "test"

    asyncio.run(exercise())


@pytest.mark.skipif(dashboard.os.name == "nt", reason="POSIX process-group cancellation")
def test_cancel_signals_entire_run_and_waits_for_partial_export(monkeypatch):
    signals = []
    monkeypatch.setattr(dashboard, "_sim_process", SimpleNamespace(pid=123))
    monkeypatch.setattr(dashboard, "_sim_running", True)
    monkeypatch.setattr(dashboard, "_sim_config", {})
    monkeypatch.setattr(dashboard.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    response = asyncio.run(dashboard.cancel_simulation())
    assert response["status"] == "cancelling"
    assert signals == [(123, signal.SIGINT)]
    assert dashboard._sim_running
    assert dashboard._sim_config["cancelled"]


def test_output_is_drained_and_replayed_without_a_subscriber(monkeypatch):
    proc = SimpleNamespace(stdout=io.StringIO("progress\nfinished\n"), wait=lambda: 0,
                           poll=lambda: 0)
    monkeypatch.setattr(dashboard, "_sim_process", proc)
    monkeypatch.setattr(dashboard, "_sim_log", [])
    monkeypatch.setattr(dashboard, "_sim_config", {})
    monkeypatch.setattr(dashboard, "_sim_running", True)

    async def exercise():
        await dashboard._collect_simulation_output(proc)
        assert not dashboard._sim_running
        assert dashboard._sim_config["returncode"] == 0
        assert dashboard._sim_log == ["progress", "finished"]

        class ConnectedRequest:
            async def is_disconnected(self):
                return False

        response = await dashboard.simulation_status_sse(ConnectedRequest())
        events = [json.loads(event[6:]) async for event in response.body_iterator]
        assert [event["line"] for event in events if event["type"] == "log"] == ["progress", "finished"]
        assert events[-1] == {"type": "complete", "returncode": 0}

    asyncio.run(exercise())


def test_disconnect_does_not_change_simulation_state(monkeypatch):
    monkeypatch.setattr(dashboard, "_sim_process", SimpleNamespace())
    monkeypatch.setattr(dashboard, "_sim_log", [])
    monkeypatch.setattr(dashboard, "_sim_running", True)
    monkeypatch.setattr(dashboard, "_sim_config", {})

    class DisconnectedRequest:
        async def is_disconnected(self):
            return True

    async def exercise():
        response = await dashboard.simulation_status_sse(DisconnectedRequest())
        assert [event async for event in response.body_iterator] == []
        assert dashboard._sim_running

    asyncio.run(exercise())


def test_failed_process_releases_gui_and_records_exit_code(monkeypatch):
    proc = SimpleNamespace(stdout=io.StringIO("failure\n"), wait=lambda: 1)
    monkeypatch.setattr(dashboard, "_sim_process", proc)
    monkeypatch.setattr(dashboard, "_sim_log", [])
    monkeypatch.setattr(dashboard, "_sim_config", {})
    monkeypatch.setattr(dashboard, "_sim_running", True)
    asyncio.run(dashboard._collect_simulation_output(proc))
    assert not dashboard._sim_running
    assert dashboard._sim_config["returncode"] == 1


def write_metrics(directory):
    directory.mkdir(parents=True)
    pd.DataFrame({"Wait_Time_s": [0, 100], "Stretch_Factor": [1, 3],
                  "Queue_Deadline_Missed": [False, True]}).to_csv(
        directory / "jobs_metrics_test.csv", index=False)
    # Monitor stores a 0.001 Wh baseline in addition to operational emissions.
    pd.DataFrame({"e_delta": [-3600000, 3600000], "co2": [200.0002, 0.0002],
                  "test_power": [60000, 60000]},
                 index=["2023-01-01 00:00:00", "2023-01-01 00:01:00"]).to_csv(
        directory / "energy_simulation_result.csv")


def test_no_plots_run_still_reports_operational_carbon_energy_and_waits(tmp_path):
    outputs = tmp_path / "outputs"
    directory = outputs / "results_test"
    write_metrics(directory)
    (directory / "config.json").write_text(json.dumps({"microgrid_step_size_s": 60}))
    run = results.read_result(directory.name, outputs, tmp_path / "thesis")
    metrics = run["summary_metrics"]
    assert metrics["total_co2_kg"] == pytest.approx(0.2)
    assert metrics["total_energy_kwh"] == pytest.approx(2)
    assert metrics["physical_grid_import_kwh"] == pytest.approx(1)
    assert metrics["total_jobs_completed"] == 2
    assert metrics["avg_wait_time_s"] == 50
    assert metrics["p95_wait_time_s"] == 95
    assert metrics["deadline_misses"] == 1


def test_published_thesis_runs_need_only_csvs_and_pngs(tmp_path):
    thesis = tmp_path / "thesis"
    directory = thesis / "data" / "workloads" / "500" / "puzzle"
    write_metrics(directory)
    picture = thesis / "plots" / "workloads" / "500" / "puzzle.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"figure")
    run = results.list_results(tmp_path / "outputs", thesis)[0]
    assert run["id"] == "thesis-workloads--500--puzzle"
    assert run["source"] == "thesis"
    assert run["images"] == ["puzzle.png"]
    assert run["summary_metrics"]["total_jobs_completed"] == 2
    assert results.read_result("../other", tmp_path / "outputs", thesis) is None


def test_results_written_in_stages_do_not_fail_or_return_nan(tmp_path):
    outputs = tmp_path / "outputs"
    directory = outputs / "results_partial"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    for name in ["jobs_metrics_test.csv", "energy_simulation_result.csv", "carbon_metrics.csv"]:
        (directory / name).touch()
    assert results.list_results(outputs, tmp_path / "thesis")[0]["summary_metrics"] == {"total_jobs_completed": 0}
    (directory / "jobs_metrics_test.csv").write_text("Wait_Time_s,Stretch_Factor\n,\n")
    run = results.list_results(outputs, tmp_path / "thesis")[0]
    assert run["summary_metrics"]["avg_wait_time_s"] is None
    json.dumps(run, allow_nan=False)
