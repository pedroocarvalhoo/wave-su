"""
WAVE GUI — FastAPI Backend
Serves the SPA and provides REST APIs for simulation management,
result browsing, dataset generation, and run comparison.
"""
import os
import sys
import json
import signal
import asyncio
import subprocess
import csv
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gui.simulation import build_command
from gui import results as result_store

# ── Paths ────────────────────────────────────────────────────────────────────
GUI_DIR = Path(__file__).resolve().parent
ROOT_DIR = GUI_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"
RESULTS_DIR = ROOT_DIR / "outputs"
THESIS_RESULTS_DIR = ROOT_DIR / "results" / "thesis"
SIMULATION_SCRIPT = ROOT_DIR / "scripts" / "simulation_energy.py"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from gen_jobs import DEFAULT_LOAD_STRATEGY, generate_system_datasets
from supercomputer_config import (
    jobs_dataset_path,
    load_node_config as load_node_config_from_dir,
    load_supercomputers_config,
    sanitize_identifier,
    save_node_config,
    save_supercomputer_config,
)

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="WAVE GUI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(GUI_DIR / "static")), name="static")
app.mount("/brand", StaticFiles(directory=str(ROOT_DIR / "docs" / "assets")), name="brand")

# ── Global simulation process state ──────────────────────────────────────────
_sim_process: Optional[subprocess.Popen] = None
_sim_log: list[str] = []
_sim_running: bool = False
_sim_config: dict = {}
_sim_output_task: Optional[asyncio.Task] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_python() -> str:
    """Find a working Python interpreter for subprocess calls."""
    if os.name == "nt":
        # Check for Windows venv with Scripts/
        venv_python_win = ROOT_DIR / "venv" / "Scripts" / "python.exe"
        if venv_python_win.exists():
            return str(venv_python_win)
        dot_venv_win = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
        if dot_venv_win.exists():
            return str(dot_venv_win)
    else:
        # Check for Unix-style venv with bin/
        venv_python_unix = ROOT_DIR / "venv" / "bin" / "python"
        if venv_python_unix.exists():
            return str(venv_python_unix)
    # Fall back to the same Python running this server
    return sys.executable


def _result_directory(run_id: str) -> Path:
    location = result_store.locations(RESULTS_DIR, THESIS_RESULTS_DIR).get(run_id)
    if location is None:
        raise HTTPException(404, "Result not found")
    return location[0]


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = GUI_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ── Supercomputers & Node Config ─────────────────────────────────────────

DEFAULT_QUEUE_CONFIG = [
    {
        "name": "short",
        "min_runtime_s": 0,
        "max_runtime_s": 86400,
        "max_nodes": 32,
        "max_wait_s": 172800,
        "priority_weight": 3,
    },
    {
        "name": "medium",
        "min_runtime_s": 86400,
        "max_runtime_s": 604800,
        "max_nodes": 64,
        "max_wait_s": 604800,
        "priority_weight": 2,
    },
    {
        "name": "long",
        "min_runtime_s": 604800,
        "max_runtime_s": None,
        "max_nodes": 128,
        "max_wait_s": 1209600,
        "priority_weight": 1,
    },
]


class CustomClusterRequest(BaseModel):
    node_type: str
    num_nodes: int = Field(..., ge=1)


class CustomNodeTypeRequest(BaseModel):
    id: str
    base_frequency_ghz: float = Field(..., gt=0)
    idle_power_w: float = Field(..., ge=0)
    max_power_w: float = Field(..., gt=0)
    memory_bandwidth_gb_s: float = Field(100.0, gt=0)
    frequencies: dict[str, float] = Field(default_factory=dict)


class CustomSupercomputerRequest(BaseModel):
    id: str
    label: Optional[str] = None
    clusters: list[CustomClusterRequest]
    node_types: list[CustomNodeTypeRequest] = Field(default_factory=list)
    sla_policy: str = "enforce"


def _load_supercomputers() -> dict:
    return load_supercomputers_config(DATASETS_DIR)


def _load_node_config() -> dict:
    return load_node_config_from_dir(DATASETS_DIR)


def _normalize_node_type_config(req: CustomNodeTypeRequest) -> tuple[str, dict]:
    node_id = sanitize_identifier(req.id)
    if req.max_power_w < req.idle_power_w:
        raise ValueError("max_power_w must be greater than or equal to idle_power_w")

    frequencies = {}
    for raw_freq, raw_power in req.frequencies.items():
        freq = float(raw_freq)
        power = float(raw_power)
        if freq <= 0 or power <= 0:
            raise ValueError("Frequency table values must be positive")
        frequencies[str(freq)] = power

    if not frequencies:
        frequencies[str(float(req.base_frequency_ghz))] = float(req.max_power_w)

    base_key = str(float(req.base_frequency_ghz))
    frequencies.setdefault(base_key, float(req.max_power_w))

    return node_id, {
        "base_frequency_ghz": float(req.base_frequency_ghz),
        "idle_power_w": float(req.idle_power_w),
        "max_power_w": float(req.max_power_w),
        "memory_bandwidth_gb_s": float(req.memory_bandwidth_gb_s),
        "frequencies": frequencies,
    }


@app.get("/api/supercomputers")
async def get_supercomputers():
    sc = _load_supercomputers()
    nc = _load_node_config()
    return {"supercomputers": sc, "node_config": nc}


@app.post("/api/supercomputers")
async def create_supercomputer(req: CustomSupercomputerRequest):
    if not req.clusters:
        raise HTTPException(400, "At least one cluster is required")

    try:
        system_id = sanitize_identifier(req.id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    sc = _load_supercomputers()
    if system_id in sc:
        raise HTTPException(409, f"Supercomputer '{system_id}' already exists")

    node_config = _load_node_config()
    pending_node_types = {}
    try:
        for node_type_req in req.node_types:
            node_id, config = _normalize_node_type_config(node_type_req)
            if node_id in node_config or node_id in pending_node_types:
                raise ValueError(f"Node type '{node_id}' already exists")
            pending_node_types[node_id] = config

        available_node_types = set(node_config) | set(pending_node_types)
        clusters = {}
        for cluster_req in req.clusters:
            node_type = sanitize_identifier(cluster_req.node_type)
            if node_type not in available_node_types:
                raise ValueError(f"Unknown node type '{node_type}'")
            if node_type in clusters:
                raise ValueError(f"Duplicate cluster node type '{node_type}'")
            clusters[node_type] = int(cluster_req.num_nodes)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    node_config.update(pending_node_types)
    save_node_config(DATASETS_DIR, node_config)

    system_config = {
        "id": system_id,
        "label": req.label or system_id,
        "clusters": clusters,
        "queues": DEFAULT_QUEUE_CONFIG,
        "sla_policy": req.sla_policy,
    }
    save_supercomputer_config(DATASETS_DIR, system_id, system_config)

    return {
        "status": "ok",
        "system": system_id,
        "supercomputers": _load_supercomputers(),
        "node_config": _load_node_config(),
    }


# ── Datasets ─────────────────────────────────────────────────────────────

@app.get("/api/datasets")
async def get_datasets(system: Optional[str] = None, jobs_dir: Optional[str] = None):
    """List available job CSVs grouped by supercomputer."""
    sc = _load_supercomputers()
    if system and system not in sc:
        raise HTTPException(404, "Unknown supercomputer")
    if jobs_dir and not Path(jobs_dir).expanduser().is_dir():
        raise HTTPException(422, "The selected workload directory does not exist")
    datasets = {}
    for system_name, system_info in sc.items():
        system_datasets = {}
        clusters = system_info.get("clusters", {})
        queues = system_info.get("queues", [])
        for ctype in clusters:
            queue_datasets = {}
            for queue in queues:
                queue_name = str(queue["name"])
                filepath = (
                    Path(jobs_dir).expanduser() / f"jobs_{ctype}_{queue_name}.csv"
                    if jobs_dir and system_name == system
                    else jobs_dataset_path(DATASETS_DIR, system_name, ctype, queue_name)
                )
                filename = filepath.name
                exists = filepath.exists()
                row_count = 0
                size_kb = 0
                if exists:
                    # Count rows.
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            row_count = sum(1 for _ in f) - 1  # exclude header
                    except Exception:
                        row_count = -1
                    size_kb = round(filepath.stat().st_size / 1024, 1)
                queue_datasets[queue_name] = {
                    "filename": filename,
                    "path": str(filepath.relative_to(DATASETS_DIR)) if not (jobs_dir and system_name == system) else filepath.name,
                    "exists": exists,
                    "rows": row_count,
                    "size_kb": size_kb,
                }

            queue_values = list(queue_datasets.values())
            all_ready = bool(queue_values) and all(item["exists"] for item in queue_values)
            total_rows = sum(
                item["rows"]
                for item in queue_values
                if item["exists"] and item["rows"] > 0
            )
            total_size_kb = round(
                sum(item["size_kb"] for item in queue_values if item["exists"]),
                1,
            )
            system_datasets[ctype] = {
                "exists": all_ready,
                "rows": total_rows,
                "size_kb": total_size_kb,
                "queues": queue_datasets,
            }
            if not queues:
                system_datasets[ctype] = {
                    "exists": False,
                    "rows": 0,
                    "size_kb": 0,
                    "queues": {},
                }
        datasets[system_name] = system_datasets
    return {"datasets": datasets}


class GenerateDatasetRequest(BaseModel):
    system: str
    num_jobs: int = Field(500, ge=1, le=10000)
    min_jobs: Optional[int] = Field(default=None, ge=1, le=10000)
    max_jobs: Optional[int] = Field(default=None, ge=1, le=10000)
    load_strategy: str = DEFAULT_LOAD_STRATEGY
    cluster_load_weights: dict[str, float] = Field(default_factory=dict)
    arrival_interval_s: float = Field(300.0, gt=0)
    arrival_window_hours: Optional[float] = Field(default=None, gt=0)
    seed: Optional[int] = None


@app.post("/api/datasets/generate")
async def generate_dataset(req: GenerateDatasetRequest):
    """Generate job datasets for a supercomputer using the official generator."""
    sc = _load_supercomputers()
    if req.system not in sc:
        raise HTTPException(404, f"System '{req.system}' not found")

    max_jobs = req.max_jobs or req.num_jobs
    if req.min_jobs is not None:
        min_jobs = req.min_jobs
    elif req.load_strategy == "equal":
        min_jobs = max_jobs
    else:
        min_jobs = max(1, min(max_jobs, int(round(max_jobs * 0.2))))

    if max_jobs < min_jobs:
        raise HTTPException(400, "max_jobs must be greater than or equal to min_jobs")

    try:
        output_lines = []
        summary = generate_system_datasets(
            req.system,
            supercomputers_config=sc,
            output_dir=str(DATASETS_DIR),
            node_config=_load_node_config(),
            min_jobs=min_jobs,
            max_jobs=max_jobs,
            arrival_interval_s=req.arrival_interval_s,
            arrival_window_hours=req.arrival_window_hours,
            seed=req.seed,
            load_strategy=req.load_strategy,
            cluster_load_weights=req.cluster_load_weights,
            progress=output_lines.append,
        )
        return {
            "status": "ok",
            "summary": summary,
            "output": "\n".join(output_lines),
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ── Simulation ───────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    system: str
    jobs_dir: Optional[str] = None
    run_tag: Optional[str] = None
    sla_policy: Optional[Literal["enforce", "report_only"]] = None
    keep_idle_nodes_on: bool = False
    no_snapshot: bool = False
    scheduling_policy: str = "eft-dvfs"
    dvfs_policy: str = "adaptive"
    budget_policy: str = "carbon-minimizing"
    allocation_strategy: str = "compute-prioritized"
    carbon_grid_threshold: float = 120.0
    grid_power_limit: float = 200000
    microgrid_mode: str = "grid-connected"
    pue: float = Field(1.3, gt=0)
    allow_grid_charging: bool = False
    # None = policy-specific default (0.7 daytime/carbon-minimizing,
    # 0.3 puzzle); an explicit value overrides every policy.
    renewable_charge_fraction: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("renewable_charge_fraction", "pv_charge_fraction"),
    )
    pv_scale: Optional[float] = None
    renewable_sources: list[str] = Field(default_factory=lambda: ["pv"])
    renewable_source_scales: dict[str, float] = Field(
        default_factory=lambda: {"pv": 400000.0, "wind": 400000.0}
    )
    battery_soc_min: float = 0.2
    battery_soc_max: float = 0.8
    battery_capacity_kwh: float = 2000.0
    battery_p_charge_max: float = 500.0
    battery_p_discharge_max: float = 500.0
    low_carbon_grid_fraction: float = 1.0
    high_carbon_grid_fraction: float = 0.0
    day_grid_fraction: float = 0.5
    night_grid_fraction: float = 1.0
    eft_dvfs_alpha: float = 0.1
    eft_dvfs_beta: float = 0.9
    # Puzzle scheduler dispatch options (only used with scheduling_policy=puzzle).
    puzzle_allow_front_run: bool = False
    puzzle_fallback_dispatch: Optional[bool] = None
    puzzle_repack_repair: bool = False
    puzzle_carbon_aware_placement: bool = True
    puzzle_global_green_board: bool = True
    puzzle_forecast_tail_carbon_multiplier: float = 1.25
    puzzle_tail_renewable_uncertainty_fraction: float = 0.20
    puzzle_tail_renewable_robustness: float = 0.0
    puzzle_tail_renewable_calibration_days: float = 90.0
    puzzle_battery_carbon_aware_dispatch: bool = True
    puzzle_battery_carbon_priority_fraction: float = 0.0
    puzzle_tail_battery_reserve_fraction: float = 0.10
    puzzle_carbon_regret_g_per_h: Optional[float] = 1.0
    puzzle_sla_start_safety_margin_h: float = 6.0
    puzzle_defer_carbon_margin: Optional[float] = None
    # SLA grid floor: let deadline-imminent jobs draw grid over a carbon-limited
    # budget instead of missing. Applies to any scheduling policy.
    sla_grid_override: Optional[bool] = None
    microgrid_step_size: int = 60
    energy_scheduler_step_size: int = 300
    runtime_prediction_error_rate: float = 0.0
    runtime_prediction_seed: int = 42
    adaptive_step: bool = True
    start_nodes_off: bool = True
    node_power_on_gate: str = "runnable"
    no_plots: bool = False
    plot_workers: Optional[int] = Field(12, ge=1)
    scheduler_debug_log: bool = True
    verbose_scheduler: bool = False


    @model_validator(mode="after")
    def validate_combination(self):
        if not 0 <= self.battery_soc_min < self.battery_soc_max <= 1:
            raise ValueError("Battery SOC must satisfy 0 <= minimum < maximum <= 1")
        if self.keep_idle_nodes_on and self.start_nodes_off:
            raise ValueError("Disable Start Nodes Off before enabling Keep Idle Nodes On")
        if not self.renewable_sources:
            raise ValueError("Select at least one renewable source")
        if self.scheduling_policy == "puzzle" and self.budget_policy != "puzzle":
            raise ValueError("Puzzle scheduling requires the Puzzle energy budget")
        if self.allocation_strategy == "puzzle-plan" and self.budget_policy != "puzzle":
            raise ValueError("Puzzle Plan allocation requires the Puzzle energy budget")
        return self


@app.post("/api/simulation/start")
async def start_simulation(req: SimulationRequest):
    global _sim_process, _sim_log, _sim_running, _sim_config, _sim_output_task

    if _sim_running:
        raise HTTPException(409, "A simulation is already running")

    cmd = build_command(req, _find_python(), SIMULATION_SCRIPT)
    budget_policy = req.budget_policy
    from scripts.simulation_energy import parse_args
    try:
        parse_args(list(_load_supercomputers()), _load_json(DATASETS_DIR / "renewable_sources.json"), cmd[3:])
    except SystemExit:
        raise HTTPException(422, "Invalid simulation options; check the selected policies and inputs")
    if req.jobs_dir and not Path(req.jobs_dir).expanduser().is_dir():
        raise HTTPException(422, "The selected workload directory does not exist")

    _sim_log = []
    _sim_config = req.model_dump()
    _sim_config["budget_policy"] = budget_policy
    _sim_config["returncode"] = None
    _sim_config["cancelled"] = False

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    _sim_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(ROOT_DIR),
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    _sim_running = True
    _sim_output_task = asyncio.create_task(_collect_simulation_output(_sim_process))

    return {"status": "started", "pid": _sim_process.pid, "command": " ".join(cmd)}


async def _collect_simulation_output(proc):
    """Drain output even when the browser closes or no SSE client connects."""
    global _sim_running
    try:
        while True:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:
                break
            _sim_log.append(line.rstrip("\n\r"))
        returncode = await asyncio.to_thread(proc.wait)
        if proc is _sim_process:
            _sim_config["returncode"] = returncode
    finally:
        if proc is _sim_process:
            _sim_running = False


@app.get("/api/simulation/status")
async def simulation_status_sse(request: Request):
    """Replay buffered logs and subscribe without owning the simulation process."""
    async def event_generator():
        if _sim_process is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No simulation running'})}\n\n"
            return
        proc = _sim_process
        position = 0
        while True:
            if await request.is_disconnected():
                return
            while position < len(_sim_log):
                line = _sim_log[position]
                position += 1
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"
            if proc is not _sim_process or not _sim_running:
                event = {"type": "complete", "returncode": proc.poll()}
                if _sim_config.get("cancelled") and proc is _sim_process:
                    event["cancelled"] = True
                yield f"data: {json.dumps(event)}\n\n"
                return
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/simulation/cancel")
async def cancel_simulation():
    global _sim_process, _sim_running
    if not _sim_running or _sim_process is None:
        raise HTTPException(404, "No simulation running")

    _sim_config["cancelled"] = True
    try:
        if os.name == "nt":
            _sim_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(_sim_process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    except OSError:
        _sim_process.kill()

    return {"status": "cancelling"}


@app.get("/api/simulation/running")
async def simulation_running():
    global _sim_running, _sim_process
    return {"running": _sim_running, "log_lines": len(_sim_log)}


# ── Results ──────────────────────────────────────────────────────────────

@app.get("/api/results")
async def list_results():
    results = await asyncio.to_thread(result_store.list_results, RESULTS_DIR, THESIS_RESULTS_DIR)
    return {"results": results}


@app.get("/api/results/{run_id}")
async def get_result(run_id: str):
    result = await asyncio.to_thread(result_store.read_result, run_id, RESULTS_DIR, THESIS_RESULTS_DIR)
    if result is None:
        raise HTTPException(404, "Result not found")
    return result


@app.get("/api/results/{run_id}/csv/{filename}")
async def get_result_csv(
    run_id: str,
    filename: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=5000),
    sort_col: Optional[str] = None,
    sort_asc: bool = True,
):
    """Serve CSV data with pagination. Returns JSON with headers and rows."""
    if Path(filename).name != filename:
        raise HTTPException(404, "CSV file not found")
    filepath = _result_directory(run_id) / filename
    if not filepath.exists() or not filename.endswith(".csv"):
        raise HTTPException(404, "CSV file not found")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            all_rows = list(reader)
    except Exception as e:
        raise HTTPException(500, f"Error reading CSV: {e}")

    total_rows = len(all_rows)

    # Sort if requested
    if sort_col and sort_col in headers:
        col_idx = headers.index(sort_col)
        try:
            all_rows.sort(
                key=lambda r: float(r[col_idx]) if r[col_idx] else float("inf"),
                reverse=not sort_asc,
            )
        except (ValueError, IndexError):
            all_rows.sort(
                key=lambda r: r[col_idx] if col_idx < len(r) else "",
                reverse=not sort_asc,
            )

    page_rows = all_rows[offset : offset + limit]

    return {
        "headers": headers,
        "rows": page_rows,
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/results/{run_id}/images")
async def list_result_images(run_id: str):
    return {"images": list(result_store.image_locations(_result_directory(run_id), THESIS_RESULTS_DIR))}


@app.get("/api/results/{run_id}/images/{image_name}")
async def serve_result_image(run_id: str, image_name: str):
    filepath = result_store.image_locations(_result_directory(run_id), THESIS_RESULTS_DIR).get(image_name)
    if filepath is None:
        raise HTTPException(404, "Image not found")
    return FileResponse(str(filepath), media_type="image/png")


# ── Comparison ───────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    run_ids: list[str]


@app.post("/api/results/compare")
async def compare_results(req: CompareRequest):
    """Compare multiple runs. Returns structured comparison data."""
    comparisons = []
    for run_id in req.run_ids:
        parsed = await asyncio.to_thread(result_store.read_result, run_id, RESULTS_DIR, THESIS_RESULTS_DIR)
        if parsed:
            comparisons.append(parsed)
    return {"comparisons": comparisons}


# ── Simulation Parameters Metadata ───────────────────────────────────────

@app.get("/api/parameters")
async def get_parameter_metadata():
    """Return metadata about all simulation parameters with descriptions."""
    renewable_config = _load_json(DATASETS_DIR / "renewable_sources.json")
    renewable_options = [
        {
            "value": source,
            "label": config.get("label", source),
            "description": config.get("description", ""),
        }
        for source, config in renewable_config.items()
    ]
    renewable_defaults = [
        source
        for source, config in renewable_config.items()
        if config.get("enabled_by_default", False)
    ] or list(renewable_config.keys())
    renewable_scale_defaults = {
        source: float(config.get("default_scale_w", 0.0))
        for source, config in renewable_config.items()
    }

    return {
        "parameters": {
            "system": {
                "label": "Supercomputer",
                "type": "select",
                "description": "Select the supercomputer architecture to simulate. Each defines a set of node clusters with different hardware.",
            },
            "scheduling_policy": {
                "label": "Scheduling Policy",
                "type": "select",
                "options": [
                    {"value": "fifo", "label": "FIFO", "description": "First-In-First-Out: jobs are scheduled in arrival order."},
                    {"value": "priority", "label": "Priority", "description": "Jobs are scheduled by priority level (3=High, 2=Normal, 1=Low)."},
                    {"value": "sjf", "label": "SJF", "description": "Shortest Job First: prioritizes jobs with shortest predicted runtime."},
                    {"value": "ljf", "label": "LJF", "description": "Longest Job First: prioritizes jobs with longest predicted runtime."},
                    {"value": "eft", "label": "EFT", "description": "Earliest Finish Time: optimizes for earliest predicted completion time."},
                    {"value": "eft-dvfs", "label": "EFT-DVFS", "description": "Energy-aware EFT: balances runtime (alpha) and energy (beta) weights for frequency selection."},
                    {"value": "puzzle", "label": "Puzzle", "description": "Plan-driven puzzle packing dispatcher (requires Puzzle budget policy)."},
                ],
                "description": "Job ordering strategy used by the cluster scheduler to pick which pending job to run next.",
                "default": "eft-dvfs",
            },
            "dvfs_policy": {
                "label": "DVFS Policy",
                "type": "select",
                "options": [
                    {"value": "fixed", "label": "Fixed Frequency", "description": "Always run at base frequency. No dynamic scaling."},
                    {"value": "max-throughput", "label": "Max Throughput", "description": "Always run at highest frequency for maximum performance."},
                    {"value": "adaptive", "label": "Adaptive", "description": "Graduates frequency by remaining budget headroom: full speed when budget is generous, stepping down through the frequency table as headroom tightens, lowest speed when scarce."},
                    {"value": "min-power", "label": "Min Power", "description": "Always run at lowest frequency for minimum power consumption."},
                ],
                "description": "Dynamic Voltage and Frequency Scaling policy. Controls how node CPU frequency is selected during job execution.",
                "default": "adaptive",
            },
            "budget_policy": {
                "label": "Budget Policy",
                "type": "select",
                "options": [
                    {"value": "grid", "label": "Grid budget", "description": "Exposes full grid power. No renewable or carbon awareness."},
                    {"value": "daytime-renewable", "label": "Daytime Renewable", "description": "Renewable-aware: adjusts grid exposure based on time of day, reserving renewable power for battery charging."},
                    {"value": "carbon-minimizing", "label": "Carbon Minimizing", "description": "Carbon-intensity-aware: only exposes grid power when carbon intensity is below threshold."},
                    {"value": "puzzle", "label": "Puzzle", "description": "Carbon-aware 24h forecast envelope with SLA-ordered job packing."},
                ],
                "description": "Energy budget exposure strategy. Controls how much compute power is made available based on energy source and carbon intensity.",
                "default": "carbon-minimizing",
            },
            "allocation_strategy": {
                "label": "Allocation Strategy",
                "type": "select",
                "options": [
                    {"value": "default", "label": "Default (Idle floor + demand)", "description": "Two-phase: give each active cluster an idle-power floor, then split the rest by max dynamic demand."},
                    {"value": "compute-prioritized", "label": "Compute Prioritized", "description": "Cover running compute first, then send the rest to clusters with pending work. With the Puzzle budget this is upgraded to Puzzle Plan."},
                    {"value": "puzzle-plan", "label": "Puzzle Plan", "description": "Distribute runtime power from each cluster's planned footprint on the Puzzle board (requires Puzzle budget policy)."},
                ],
                "description": "Cross-cluster energy budget allocation strategy. Decides how total budget is split between different cluster types.",
                "default": "compute-prioritized",
            },
            "puzzle_allow_front_run": {
                "label": "Puzzle: Front-run",
                "type": "boolean",
                "description": "Puzzle scheduler only. Dispatch future-planned jobs greedily as soon as nodes and budget allow, instead of holding them for their planned slot. Lower waits; the recommended dispatch mode.",
                "default": False,
            },
            "puzzle_fallback_dispatch": {
                "label": "Puzzle: Unplanned Fallback",
                "type": "select",
                "description": "Let unplaced jobs backfill the current budget. The policy default disables this with carbon-aware placement and enables it when that placement is disabled.",
                "default": "",
                "options": [
                    {"value": "", "label": "Policy default"},
                    {"value": "true", "label": "Enabled"},
                    {"value": "false", "label": "Disabled"},
                ],
            },
            "puzzle_repack_repair": {
                "label": "Puzzle: Repack Repair",
                "type": "boolean",
                "description": "Puzzle scheduler only. After the greedy packing, run a local-search pass that pulls grid-spilled jobs back onto the green board by relocating already-placed jobs within their SLA slack. Cuts CO₂ without risking deadlines.",
                "default": False,
            },
            "puzzle_carbon_aware_placement": {
                "label": "Puzzle: Carbon-aware Placement",
                "type": "boolean",
                "description": "Score every partial-green DVFS/start candidate by forecast CO2, enforce the low-carbon grid cap per slot, and classify deliberate high-carbon exceptions as carbon-regret or SLA emergency.",
                "default": True,
            },
            "puzzle_global_green_board": {
                "label": "Puzzle: Global Green Board",
                "type": "boolean",
                "description": "Pack all clusters against one shared green envelope while preserving independent node limits. Prevents unused green capacity from being stranded in a fixed cluster slice.",
                "default": True,
            },
            "puzzle_forecast_tail_carbon_multiplier": {
                "label": "Puzzle: Forecast Tail Risk",
                "type": "number",
                "description": "Carbon-risk multiplier for grid use in extrapolated slots beyond the real forecast window. Values above 1 discourage relying on repeated forecast data.",
                "default": 1.25,
                "min": 1,
                "max": 10,
                "step": 0.05,
            },
            "puzzle_tail_renewable_uncertainty_fraction": {
                "label": "Puzzle: Tail Renewable Uncertainty",
                "type": "number",
                "description": "Build pessimistic, expected, and optimistic renewable scenarios beyond the real forecast window as expected × (1 − fraction), expected, and expected × (1 + fraction).",
                "default": 0.20,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "puzzle_tail_renewable_robustness": {
                "label": "Puzzle: Tail Renewable Robustness",
                "type": "number",
                "description": "Choose the extrapolated planning commitment between expected renewable (0, default) and the pessimistic scenario (1). Intermediate values interpolate between them; battery recourse remains physically independent.",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "puzzle_tail_renewable_calibration_days": {
                "label": "Puzzle: Tail Calibration History (days)",
                "type": "number",
                "description": "Calibrate P10/P90 repeated-window renewable scenarios from this many days of realised history before each replan. Set 0 to retain the fixed version-0.5 uncertainty fraction.",
                "default": 90.0,
                "min": 0,
                "max": 365,
                "step": 1,
            },
            "puzzle_battery_carbon_aware_dispatch": {
                "label": "Puzzle: Carbon-aware Battery Recourse",
                "type": "boolean",
                "description": "Keep protected running jobs as the hard first tier, then allocate finite battery energy within each tier to the slots where it avoids the most marginal grid CO2.",
                "default": True,
            },
            "puzzle_battery_carbon_priority_fraction": {
                "label": "Puzzle: Battery Carbon Priority Share",
                "type": "number",
                "description": "Share of ordinary battery recourse placed greedily in the highest-carbon slots before water-filling the remainder across time. Protected running deficits remain the hard first tier.",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "puzzle_tail_battery_reserve_fraction": {
                "label": "Puzzle: Tail Battery Reserve",
                "type": "number",
                "description": "Once an extrapolation-backed job is physically running, hold this fraction of deliverable battery energy for the part that still crosses the real window. Pending risk is observed but never reserves energy speculatively. Set 0 to disable only the reserve; physical tail accounting remains enabled.",
                "default": 0.10,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "puzzle_carbon_regret_g_per_h": {
                "label": "Puzzle: Carbon Regret (gCO₂/h)",
                "type": "number",
                "description": "Advance a partial-green job when another hour of waiting saves no more than this many forecast grams of CO₂. The effective threshold rises as the job ages; leave blank to reproduce version 0.2.",
                "default": 1.0,
                "min": 0,
                "max": 10000,
                "step": 0.5,
            },
            "puzzle_sla_start_safety_margin_h": {
                "label": "Puzzle: SLA Start Margin (h)",
                "type": "number",
                "description": "Reserve this many hours before the formal queue start deadline so replans, dispatcher granularity, and node contention do not turn an exact-boundary plan into a miss.",
                "default": 6.0,
                "min": 0,
                "max": 48,
                "step": 0.5,
            },
            "puzzle_defer_carbon_margin": {
                "label": "Puzzle: Defer Carbon Margin (gCO₂/kWh)",
                "type": "number",
                "description": "Puzzle scheduler only. When set, hold a job for its planned slot only if running now is dirtier than that slot by more than this margin. Leave blank to disable selective deferral.",
                "default": None,
                "min": 0,
                "max": 500,
            },
            "sla_grid_override": {
                "label": "SLA Grid Floor",
                "type": "select",
                "description": "Let a job whose start deadline falls within the next scheduler step draw grid beyond a carbon-limited budget instead of missing. Trades some carbon for deadline safety under scarce/carbon-minimizing budgets.",
                "default": "",
                "options": [
                    {"value": "", "label": "Policy default"},
                    {"value": "true", "label": "Enabled"},
                    {"value": "false", "label": "Disabled"},
                ],
            },
            "carbon_grid_threshold": {
                "label": "Carbon Grid Threshold (gCO₂/kWh)",
                "type": "number",
                "description": "Carbon Minimizing and Puzzle use this threshold to admit low-carbon grid power. Use 0 for a renewable/battery budget.",
                "default": 120.0,
                "min": 0,
                "max": 2000,
            },
            "grid_power_limit": {
                "label": "Grid Power Limit (W)",
                "type": "number",
                "description": "Physical grid power cap made available to the energy budget policies, in Watts. Default 200,000W (200kW).",
                "default": 200000,
                "min": 0,
            },
            "microgrid_mode": {
                "label": "Microgrid Mode",
                "type": "select",
                "options": [
                    {
                        "value": "grid-connected",
                        "label": "Grid Connected",
                        "description": "Allows physical import/export with the grid. Scheduler grid exposure is still controlled by the selected budget policy.",
                    },
                    {
                        "value": "islanded",
                        "label": "Islanded",
                        "description": "Forbids grid exchange. The run fails if renewable and battery power cannot serve physical demand.",
                    },
                ],
                "description": "Physical grid connection mode for the microgrid simulation.",
                "default": "grid-connected",
            },
            "allow_grid_charging": {
                "label": "Allow Grid Charging",
                "type": "boolean",
                "description": "Allow battery charge commands to import power from the grid in grid-connected mode. Disable for renewable-only charging.",
                "default": False,
            },
            "pue": {
                "label": "PUE",
                "type": "number",
                "description": "Power Usage Effectiveness. Facility power = IT compute power × PUE, and scheduler budgets are divided by PUE.",
                "default": 1.3,
                "min": 1.0,
                "max": 3.0,
                "step": 0.01,
            },
            "renewable_charge_fraction": {
                "label": "Renewable Charge Fraction",
                "type": "number",
                "description": "Fraction of renewable generation reserved for battery charging. Leave unset for the policy-specific default (0.7 for Carbon Minimizing / Daytime Renewable, 0.3 for Puzzle). Range 0-1.",
                "default": None,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "renewable_sources": {
                "label": "Renewable Sources",
                "type": "multiselect",
                "description": "Renewable generation sources enabled in the microgrid and energy budget.",
                "default": renewable_defaults,
                "options": renewable_options,
            },
            "renewable_source_scales": {
                "label": "Source Capacity Scale (W)",
                "type": "source_scales",
                "description": "Power scale in Watts applied to each selected normalized renewable profile.",
                "default": renewable_scale_defaults,
                "options": renewable_options,
                "min": 0,
            },
            "battery_soc_min": {
                "label": "Min State of Charge (SOC)",
                "type": "number",
                "description": "Minimum allowed battery state of charge fraction (battery will stop discharging below this). Range 0-1.",
                "default": 0.2,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "battery_soc_max": {
                "label": "Max State of Charge (SOC)",
                "type": "number",
                "description": "Maximum allowed battery state of charge fraction (battery will stop charging above this). Range 0-1.",
                "default": 0.8,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "battery_capacity_kwh": {
                "label": "Battery Capacity (kWh)",
                "type": "number",
                "description": "Maximum energy storage capacity of the battery in kWh.",
                "default": 2000.0,
                "min": 1,
            },
            "battery_p_charge_max": {
                "label": "Max Charge Power (kW)",
                "type": "number",
                "description": "Maximum power at which the battery can be charged in kW.",
                "default": 500.0,
                "min": 0,
            },
            "battery_p_discharge_max": {
                "label": "Max Discharge Power (kW)",
                "type": "number",
                "description": "Maximum power at which the battery can discharge in kW.",
                "default": 500.0,
                "min": 0,
            },
            "low_carbon_grid_fraction": {
                "label": "Low Carbon Grid Fraction",
                "type": "number",
                "description": "Used only by Carbon Minimizing. Fraction of grid power limit exposed when carbon intensity is below threshold.",
                "default": 1.0,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "high_carbon_grid_fraction": {
                "label": "High Carbon Grid Fraction",
                "type": "number",
                "description": "Used only by Carbon Minimizing. Fraction of grid power limit exposed when carbon intensity is above threshold.",
                "default": 0.0,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "day_grid_fraction": {
                "label": "Day Grid Fraction",
                "type": "number",
                "description": "Used only by Daytime Renewable. Fraction of grid power exposed during daytime hours.",
                "default": 0.5,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "night_grid_fraction": {
                "label": "Night Grid Fraction",
                "type": "number",
                "description": "Used only by Daytime Renewable. Fraction of grid power exposed at night.",
                "default": 1.0,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "eft_dvfs_alpha": {
                "label": "EFT-DVFS Alpha (Runtime Weight)",
                "type": "number",
                "description": "Runtime weight for EFT-DVFS scheduling policy. Higher = more emphasis on job speed. Default 0.1.",
                "default": 0.1,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "eft_dvfs_beta": {
                "label": "EFT-DVFS Beta (Energy Weight)",
                "type": "number",
                "description": "Energy weight for EFT-DVFS scheduling policy. Higher = more emphasis on energy savings. Default 0.9.",
                "default": 0.9,
                "min": 0,
                "max": 1,
                "step": 0.05,
            },
            "microgrid_step_size": {
                "label": "Microgrid Step Size (s)",
                "type": "number",
                "description": "Microgrid simulation step size in seconds. Use 1 with adaptive stepping for high-resolution energy accounting. Default 60.",
                "default": 60,
                "min": 1,
                "max": 3600,
            },
            "energy_scheduler_step_size": {
                "label": "Energy Scheduler Step Size (s)",
                "type": "number",
                "description": "Energy budget redistribution step size in seconds. How often the energy scheduler recalculates budgets. Default 300 (5 min).",
                "default": 300,
                "min": 1,
                "max": 3600,
            },
            "runtime_prediction_error_rate": {
                "label": "Runtime Prediction Error Rate",
                "type": "number",
                "description": "Relative error applied only to scheduler-visible runtime predictions; physical job work is unchanged. Default 0.0 (exact model predictions).",
                "default": 0.0,
                "min": 0,
                "max": 2,
                "step": 0.05,
            },
            "runtime_prediction_seed": {
                "label": "Prediction Error Seed",
                "type": "number",
                "description": "Base seed for deterministic runtime prediction errors. Change to get different random noise. Default 42.",
                "default": 42,
                "min": 0,
            },
            "adaptive_step": {
                "label": "Adaptive Stepping",
                "type": "boolean",
                "description": "Allow schedulers to use sub-60s steps for job completions. Improves timing accuracy and performance. Enabled by default.",
                "default": True,
            },
            "start_nodes_off": {
                "label": "Start Nodes Off",
                "type": "boolean",
                "description": "Start all nodes powered off at simulation start. Avoids initial idle grid consumption. Enabled by default for carbon-aware runs.",
                "default": True,
            },
            "node_power_on_gate": {
                "label": "Node Power-On Gate",
                "type": "select",
                "options": [
                    {
                        "value": "runnable",
                        "label": "Runnable only",
                        "description": "Only power idle nodes on when a job the scheduler would dispatch this step can actually run on them within the energy budget. Avoids waking the cluster for jobs a plan withholds for a greener slot.",
                    },
                    {
                        "value": "eager",
                        "label": "Eager (legacy)",
                        "description": "Power nodes on toward total pending node demand, even if those jobs will not be dispatched yet. Restores the previous behaviour.",
                    },
                ],
                "description": "When the resource manager wakes idle nodes ahead of work.",
                "default": "runnable",
            },
            "jobs_dir": {"label": "Workload directory (optional)", "type": "text", "description": "Use separate queue CSVs instead of the selected system's default workload.", "default": ""},
            "run_tag": {"label": "Run name (optional)", "type": "text", "description": "A short label to identify this experiment.", "default": ""},
            "keep_idle_nodes_on": {"label": "Keep idle nodes on", "type": "boolean", "description": "Retain idle power between jobs. Disable Start Nodes Off to match the thesis evaluation setup.", "default": False},
            "no_snapshot": {"label": "Skip plotting snapshot", "type": "boolean", "description": "Export CSVs without saving the scheduler pickle. Enable together with Skip Plot Generation.", "default": False},
            "sla_policy": {"label": "Waiting-bound policy", "type": "select", "description": "Use the scenario default, enforce waiting bounds or only report them.", "default": "", "options": [{"value": "", "label": "Scenario default"}, {"value": "enforce", "label": "Enforce"}, {"value": "report_only", "label": "Report only"}]},
            "no_plots": {
                "label": "Skip Plot Generation",
                "type": "boolean",
                "description": "Skip generating matplotlib plots at the end of the simulation. Saves time; plots can be generated later from saved data.",
                "default": False,
            },
            "plot_workers": {
                "label": "Plot Workers",
                "type": "number",
                "description": "Number of ProcessPoolExecutor workers used to generate result plots. Higher is faster but uses more memory.",
                "default": 12,
                "min": 1,
                "max": 16,
            },
            "scheduler_debug_log": {
                "label": "Scheduler Debug Log",
                "type": "boolean",
                "description": "Export scheduler_debug_events CSV files with detailed event logs (starts, completions, failures, DVFS changes).",
                "default": True,
            },
            "verbose_scheduler": {
                "label": "Verbose Scheduler",
                "type": "boolean",
                "description": "Print periodic scheduler progress and job submission/completion messages during simulation.",
                "default": False,
            },
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
