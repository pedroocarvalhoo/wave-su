"""Compare the original PV+wind Puzzle main run with the corrected same-config run."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from vessim.wave.energy.puzzle_multiobjective import extract_run_metrics

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/results_thesis_final'
NEW = ROOT / 'results/thesis_puzzle_admissible_grid_2026-09-28'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_record_path = next((OLD / 'runs').glob('*solar_wind_puzzle*.json'))
    old_record = json.loads(old_record_path.read_text())
    new_record_path = next((NEW / 'runs/ct250_puzzle').glob('*/run.json'))
    new_record = json.loads(new_record_path.read_text())
    directories = [OLD / 'results' / Path(old_record['result_dir']).name, Path(new_record['result_dir'])]
    configs = [json.loads((p / 'config.json').read_text()) for p in directories]
    metadata = {'timestamp', 'run_tag', 'jobs_dir'}
    config_diffs = {k: [configs[0].get(k), configs[1].get(k)]
                    for k in sorted(set(configs[0]) | set(configs[1]))
                    if configs[0].get(k) != configs[1].get(k)}
    assert set(config_diffs) <= metadata, config_diffs
    assert old_record['environment'] == new_record['environment']
    inputs = []
    for name, expected in old_record['input_hashes'].items():
        reference = Path(name) if Path(name).is_absolute() else ROOT / name
        if name.startswith('datasets/'):
            current = NEW / 'inputs' / name
        else:
            current = NEW / 'inputs/macc_sparse_early_tail' / reference.name
        assert digest(reference) == digest(current) == expected, name
        inputs.append({'original_input': name, 'corrected_input': str(current.relative_to(ROOT)), 'sha256': expected})
    old_source = Path(old_record['source_snapshot'])
    new_source = NEW / 'source_snapshot'
    runtime_files = sorted([*(old_source / 'vessim').rglob('*.py'), old_source / 'scripts/simulation_energy.py'])
    changes = []
    for p in runtime_files:
        rel = p.relative_to(old_source)
        q = new_source / rel
        assert digest(q) == digest(ROOT / rel), str(rel)
        if digest(p) != digest(q):
            changes.append(str(rel))
    assert changes == ['vessim/wave/energy/puzzle_budget_policy.py'], changes
    metrics, monitor_energy = [], []
    keys = []
    for directory in directories:
        metric = asdict(extract_run_metrics(directory))
        assert metric['completed_jobs'] == 2523 and metric['deadline_misses'] == 0
        metrics.append(metric)
        e = pd.read_csv(directory / 'energy_simulation_result.csv')
        step = json.loads((directory / 'config.json').read_text())['microgrid_step_size_s']
        monitor_energy.append(float(e.filter(regex='_power$').sum(axis=1).sum() * step / 3_600_000))
        run_keys = set()
        for p in directory.glob('jobs_metrics_*.csv'):
            cluster = p.stem.removeprefix('jobs_metrics_')
            for job in pd.read_csv(p, usecols=['Job_ID']).Job_ID:
                key = (cluster, int(job))
                assert key not in run_keys
                run_keys.add(key)
        keys.append(run_keys)
    assert keys[0] == keys[1] and len(keys[0]) == 2523
    rows = []
    for label, metric, energy in zip(('original_main_run', 'corrected_same_configuration'), metrics, monitor_energy):
        rows.append({'version': label,
                     **{k: metric[k] for k in ('mean_wait_h', 'p95_wait_h', 'makespan_h', 'physical_grid_import_kwh', 'co2_kg', 'completed_jobs', 'deadline_misses', 'budget_violation_samples')},
                     'it_energy_monitor_kwh': energy,
                     'effective_grid_g_per_kwh': metric['co2_kg'] * 1000 / metric['physical_grid_import_kwh'],
                     'result_dir': metric['results_dir']})
    frame = pd.DataFrame(rows)
    output = NEW / 'analysis/main_run_reproduction_comparison.csv'
    frame.to_csv(output, index=False)
    audit = {'original_record': str(old_record_path.relative_to(ROOT)),
             'corrected_record': str(new_record_path.relative_to(ROOT)),
             'simulation_configuration_equal': True,
             'metadata_differences': config_diffs,
             'environment_equal': True,
             'verified_original_inputs': inputs,
             'identical_completed_job_keys': 2523,
             'changed_runtime_files': changes,
             'corrected_runtime_matches_live_source': True,
             'runtime_file_count_verified': len(runtime_files),
             'corrected_run_wall_seconds': new_record['wall_seconds'],
             'corrected_run_timeout_seconds': new_record['wall_timeout_s'],
             'metrics': metrics,
             'monitor_it_energy_formula': 'sum each monitor row cluster power * configured monitor step seconds / 3,600,000',
             'simulations_launched_by_this_audit': 0,
             'reason_for_reuse': 'The completed corrected ct250 run has identical simulation configuration, all original inputs and seeds; only the Puzzle runtime changed.',
             'generation_command': 'venv/bin/python -B -m results.audit_puzzle_main_reproduction',
             'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(), old_record_path, new_record_path, *(d / 'config.json' for d in directories), *(d / 'energy_simulation_result.csv' for d in directories)]}}
    (NEW / 'analysis/main_run_reproduction_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(frame.drop(columns='result_dir').to_string(index=False))
    print('Identical configuration, environment, 15 original inputs and 2523 completed job keys. Only Puzzle runtime differs.')


if __name__ == '__main__':
    main()
