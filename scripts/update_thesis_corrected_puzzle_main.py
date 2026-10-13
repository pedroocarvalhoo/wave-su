"""Update the thesis main PV+wind Puzzle figures from the audited corrected run.

Reads saved outputs only and launches no simulator runs.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'results/thesis_puzzle_admissible_grid_2026-09-28/analysis/main_run_reproduction_audit.json'
CORRECTED_RECORD = ROOT / 'results/thesis_puzzle_admissible_grid_2026-09-28/runs/ct250_puzzle/20260928T203532337340Z/run.json'
OUT = ROOT / 'thesis/figures/evaluation_final'


def schedulers_from_csv(run_dir: Path):
    config = json.loads((run_dir / 'config.json').read_text())
    schedulers = []
    for cluster in ('arm_a64fx_fujitsu', 'x86_epyc_rome', 'gpu_node_a100'):
        jobs = pd.read_csv(run_dir / f'jobs_metrics_{cluster}.csv')
        events = pd.read_csv(run_dir / f'scheduler_debug_events_{cluster}.csv')
        starts = events[events.Event == 'job_started'].set_index('Job_ID').Time
        ends = events[events.Event == 'job_completed'].set_index('Job_ID').Time
        objects = []
        for row in jobs.itertuples():
            start = pd.Timestamp(starts.loc[row.Job_ID]).to_pydatetime()
            end = pd.Timestamp(ends.loc[row.Job_ID]).to_pydatetime()
            objects.append(SimpleNamespace(job_id=int(row.Job_ID), num_nodes=int(row.Nodes),
                status='completed', start_time=start, completion_time=end,
                assigned_frequency=float(row.Initial_Frequency_GHz),
                actual_t_exec=float(row.Real_Exec_Time_s), remaining_time=0))
        node_count = next(c['num_nodes'] for c in config['clusters'] if c['type'] == cluster)
        schedulers.append(SimpleNamespace(cluster_type=cluster,
            job_queue=SimpleNamespace(jobs=objects), timeline=range(node_count)))
    return schedulers


def main():
    import thesis.figures.make_evaluation_final as gen
    audit = json.loads(AUDIT.read_text())
    corrected = audit['metrics'][1]
    run_dir = Path(json.loads(CORRECTED_RECORD.read_text())['result_dir'])
    old_runs = ROOT / 'results/results_thesis_final'
    summary, _ = gen.main_summary()
    idx = summary.index[summary.case == 'solar_wind_puzzle'][0]
    mapping = {
        'completed_jobs':'completed_jobs', 'deadline_misses':'deadline_misses',
        'co2_kg':'co2_kg', 'it_energy_kwh':'it_energy_kwh',
        'physical_grid_import_kwh':'physical_grid_import_kwh',
        'physical_grid_export_kwh':'physical_grid_export_kwh',
        'attributed_grid_kwh':'attributed_grid_kwh', 'mean_wait_h':'mean_wait_h',
        'p95_wait_h':'p95_wait_h', 'makespan_h':'makespan_h',
        'green_dispatches':'green_dispatches','grid_dispatches':'grid_dispatches',
        'fallback_dispatches':'fallback_dispatches','planned_misses':'planned_misses',
        'budget_violation_samples':'budget_violation_samples',
        'budget_max_excess_w':'budget_max_excess_w',
        'source_identity_max_error_w':'source_identity_max_error_w',
    }
    for column, key in mapping.items():
        summary.loc[idx, column] = corrected[key]
    summary.loc[idx, 'source_hash'] = json.loads(CORRECTED_RECORD.read_text())['runtime_source_hash']
    summary.loc[idx, 'result_dir'] = str(run_dir)
    summary.loc[idx, 'results_dir'] = str(run_dir)
    summary.loc[idx, 'run_tag'] = corrected['run_tag']
    summary.loc[idx, 'wall_seconds'] = audit['corrected_run_wall_seconds']
    summary.loc[idx, 'effective_grid_g_per_kwh'] = corrected['co2_kg'] * 1000 / corrected['physical_grid_import_kwh']

    # The normal plot generation remains the single source for the aggregate
    # table and profile plot; the archived campaign summary is not modified.
    gen.summarize_plots(summary)

    # The corrected run was captured without pickles. These plots require only
    # the list of cluster labels, and this adapter supplies those from the saved
    # monitor/job CSV outputs.
    gen.load_schedulers = lambda _directory: schedulers_from_csv(run_dir)
    puzzle_out = OUT / 'solar_wind_puzzle'
    gen.extract_panels('solar_wind_puzzle', run_dir, puzzle_out)

    # These native-thesis renderer Gantts were already exported from corrected
    # CSV event timelines, with common profile axes and all 2,523 job blocks.
    gantt_dir = ROOT / 'results/thesis_puzzle_admissible_grid_2026-09-28/figures/gantt_thesis/ct250'
    for name in ('gantt_puzzle_arm_a64fx_fujitsu.png','gantt_puzzle_gpu_node_a100.png'):
        shutil.copy2(gantt_dir / name, puzzle_out / name)

    # Rebuild the x86 pair using the same renderer and common 0-290 kW axis.
    import thesis.figures.plot_gantt_final as plotter
    original_load_schedulers = plotter.load_schedulers
    def loader(directory):
        return schedulers_from_csv(run_dir) if Path(directory).resolve() == run_dir.resolve() else original_load_schedulers(directory)
    plotter.load_schedulers = loader
    base_summary = summary.copy()
    base_summary.loc[base_summary.case == 'solar_wind_puzzle', list(mapping)] = summary.loc[idx, list(mapping)].values
    base_summary.loc[base_summary.case == 'solar_wind_puzzle', 'source_hash'] = summary.loc[idx, 'source_hash']
    gen.make_eft_puzzle_comparison({
        'solar_wind_eft': old_runs / 'results' / Path(base_summary.loc[base_summary.case == 'solar_wind_eft','result_dir'].iloc[0]).name,
        'solar_wind_puzzle': run_dir,
    }, base_summary.set_index('case', drop=False))

    # Evidence for this derived update records both the historical and corrected
    # source records, the audit, and generated output digests.
    import hashlib
    output_files = [OUT/'super_table.csv', OUT/'profile_effects.pdf',
        puzzle_out/'partition_budget.pdf', puzzle_out/'budget_sources.pdf',
        puzzle_out/'actor_power.pdf', puzzle_out/'energy.pdf', puzzle_out/'carbon.pdf',
        puzzle_out/'gantt_puzzle_arm_a64fx_fujitsu.png', puzzle_out/'gantt_puzzle_gpu_node_a100.png',
        OUT/'eft_puzzle_solar_wind_comparison/gantt_eft_x86_epyc_rome.png',
        OUT/'eft_puzzle_solar_wind_comparison/gantt_puzzle_x86_epyc_rome.png',
        OUT/'eft_puzzle_solar_wind_comparison/comparison_metrics.csv',
        OUT/'eft_puzzle_solar_wind_comparison/comparison_deltas.csv']
    sources = [AUDIT, CORRECTED_RECORD, run_dir/'energy_simulation_result.csv',
        run_dir/'config.json', old_runs/'summary.csv',
        old_runs/'runs/eval-results_thesis_final-solar_wind_eft-20260927T011334323823.json']
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {
        'command':'venv/bin/python -B scripts/update_thesis_corrected_puzzle_main.py',
        'simulations_launched':0,'corrected_case':'solar_wind_puzzle',
        'corrected_run':str(CORRECTED_RECORD.relative_to(ROOT)),
        'reproduction_audit':str(AUDIT.relative_to(ROOT)),
        'workload_jobs':2523,'workload_hash_matches_historical_case':True,
        'runtime_source_hash_differs_from_historical_case':True,
        'comparison_source_hashes_may_differ_but_workload_hash_is_equal':True,
        'sources':{str(p.relative_to(ROOT)):digest(p) for p in sources if p.is_file()},
        'outputs':{str(p.relative_to(ROOT)):digest(p) for p in output_files if p.is_file()}}
    (OUT/'corrected_puzzle_main_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Updated corrected Puzzle table row, energy panels and comparison Gantts.')

if __name__ == '__main__': main()
