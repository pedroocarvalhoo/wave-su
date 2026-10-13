import numpy as np
import pandas as pd
import os
import sys
import argparse

from supercomputer_config import (
    LEGACY_SUPERCOMPUTERS_FILENAME,
    SUPERCOMPUTERS_DIRNAME,
    jobs_dataset_dir,
    jobs_dataset_path,
    load_node_config as load_node_config_from_dir,
    load_supercomputers_config as load_supercomputers_config_from_dir,
)

#WAVE
DEFAULT_MIN_JOBS = 200
DEFAULT_MAX_JOBS = 1000
DEFAULT_ARRIVAL_INTERVAL_S = 300.0
DEFAULT_LOAD_STRATEGY = "balanced"


DEFAULT_MIN_RUNTIME_S = 60.0


def generate_jobs(
    n_jobs,
    mode,
    cluster_nodes=None,
    base_frequency_ghz=2.0,
    min_runtime_s=DEFAULT_MIN_RUNTIME_S,
    max_runtime_s=None,
    max_job_nodes=None,
    arrival_interval_s=300.0,
    arrival_window_s=None,
):
    # CPI (cycles per instruction): a few, clustered around 5.
    if mode == 'normal':
        cpi = np.random.normal(loc=5, scale=1.8, size=n_jobs)
    else:
        cpi = 5 + np.random.standard_cauchy(size=n_jobs) * 0.5
    cpi = np.clip(cpi, 1.0, 10)

    # Requested walltime drives the queue class + SLA. Real HPC users heavily
    # over-request, so the ACTUAL runtime is a fraction of it (Beta-shaped
    # utilisation, median ~0.4). Floor the actual runtime at ``min_runtime_s``
    # so no sub-minute "flash" job produces the insane one-step power spikes.
    requested_walltime_s = generate_requested_walltimes(n_jobs)
    utilisation = np.random.beta(2.0, 6.0, size=n_jobs)  # median ~0.24: users over-request heavily
    actual_runtime_s = np.clip(
        requested_walltime_s * utilisation,
        min_runtime_s,
        requested_walltime_s.astype(float),
    )
    # A tight SLA short job could have requested < min_runtime; keep runtime >=
    # floor regardless (still realistic: it just fully uses its small request).
    actual_runtime_s = np.maximum(actual_runtime_s, min_runtime_s)
    # Optional cap: a handful of week-long jobs dominate the makespan on small
    # partitions without adding new dynamics, so clamp the tail if requested.
    if max_runtime_s is not None:
        actual_runtime_s = np.minimum(actual_runtime_s, float(max_runtime_s))

    # Back out instruction count from the target COMPUTE time at the cluster's
    # base (== max here) frequency: compute_time = CPI*#I/(f*1e9). Runtime only
    # grows under DVFS-down, so this floors the executed runtime at the target.
    freq_hz = float(base_frequency_ghz) * 1e9
    num_instructions = actual_runtime_s * freq_hz / cpi

    num_nodes = sample_node_counts(n_jobs, cluster_nodes, max_job_nodes)

    # Memory in GB (0.1 to 100 GB), log-uniform.
    memory_gb = 10 ** np.random.uniform(np.log10(0.1), np.log10(100.0), n_jobs)

    # Priority (1=Low, 2=Normal, 3=High)
    priority = np.random.choice([1, 2, 3], size=n_jobs, p=[0.2, 0.6, 0.2])

    arrival_offset_s = generate_arrival_offsets(
        n_jobs,
        arrival_interval_s=arrival_interval_s,
        arrival_window_s=arrival_window_s,
    )

    df = pd.DataFrame({
        'job_id': range(n_jobs),
        'CPI': np.round(cpi, 4),
        '#I': np.round(num_instructions).astype('int64'),
        '#nodes': num_nodes,
        'Memory_GB': np.round(memory_gb, 2),
        'Priority': priority,
        'Arrival_Offset_s': np.round(arrival_offset_s, 2),
        'Requested_Walltime_s': requested_walltime_s,
    })

    return df


def sample_node_counts(n_jobs, cluster_nodes, max_job_nodes=None):
    """Realistic HPC node counts: clustered on powers of two, skewed to small,
    with a tail up to ~20% of the cluster, capped by the queue node limit."""
    # A job takes up to ~20% of the cluster, but never more than the per-job
    # queue limit (else the scheduler rejects it as a node-limit violation).
    cap = max(1, int((cluster_nodes or 1) * 0.2))
    if max_job_nodes:
        cap = min(cap, int(max_job_nodes))
    cap = max(cap, 1)
    if cap <= 9:
        choices = list(range(1, cap + 1))
        weights = np.array([1.0 / (c ** 0.7) for c in choices])
        weights /= weights.sum()
        return np.random.choice(choices, size=n_jobs, p=weights).astype(int)
    powers = [1 << k for k in range(0, int(np.floor(np.log2(cap))) + 1)]
    # Weight ~ 1/size**0.7: many 1-4 node jobs, a thinning tail to large ones.
    weights = np.array([1.0 / (p ** 0.7) for p in powers])
    weights /= weights.sum()
    return np.random.choice(powers, size=n_jobs, p=weights).astype(int)


def generate_arrival_offsets(n_jobs, arrival_interval_s=300.0, arrival_window_s=None):
    if arrival_window_s is None:
        return np.arange(n_jobs, dtype=float) * float(arrival_interval_s)
    return np.linspace(0.0, float(arrival_window_s), n_jobs)


def generate_requested_walltimes(n_jobs):
    queue_class = np.random.choice(
        ["short", "medium", "long"],
        size=n_jobs,
        p=[0.82, 0.15, 0.03],
    )
    requested = np.empty(n_jobs)

    def log_uniform(lo, hi, n):
        # Log-uniform => skewed toward the short end of each band, which matches
        # real submission mixes (far more short jobs than long ones).
        return 10 ** np.random.uniform(np.log10(lo), np.log10(hi), n)

    short_mask = queue_class == "short"
    medium_mask = queue_class == "medium"
    long_mask = queue_class == "long"

    requested[short_mask] = log_uniform(5 * 60, 24 * 3600, short_mask.sum())
    requested[medium_mask] = log_uniform(24 * 3600, 7 * 24 * 3600, medium_mask.sum())
    requested[long_mask] = log_uniform(7 * 24 * 3600, 21 * 24 * 3600, long_mask.sum())
    return np.ceil(requested).astype(int)

script_dir = os.path.dirname(os.path.abspath(__file__))
supercomputers_dir = os.path.join(script_dir, SUPERCOMPUTERS_DIRNAME)
supercomputers_file = os.path.join(script_dir, LEGACY_SUPERCOMPUTERS_FILENAME)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate WAVE job datasets.")
    parser.add_argument("--system", choices=None, help="Supercomputer to generate.")
    parser.add_argument("--all", action="store_true", help="Generate datasets for all systems.")
    parser.add_argument("--min-jobs", type=int, default=DEFAULT_MIN_JOBS)
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    parser.add_argument(
        "--arrival-interval-s",
        type=float,
        default=DEFAULT_ARRIVAL_INTERVAL_S,
        help="Baseline interval for the largest cluster when the arrival window is automatic.",
    )
    parser.add_argument(
        "--arrival-window-hours",
        type=float,
        default=None,
        help="Spread every cluster in a system over this many hours.",
    )
    parser.add_argument(
        "--load-strategy",
        choices=["balanced", "node-proportional", "equal"],
        default=DEFAULT_LOAD_STRATEGY,
        help=(
            "How to distribute jobs across clusters. 'balanced' aims for similar "
            "normalized cluster load, 'node-proportional' scales by node count, "
            "and 'equal' gives every cluster the same count."
        ),
    )
    parser.add_argument(
        "--cluster-load-weight",
        action="append",
        default=[],
        metavar="CLUSTER=WEIGHT",
        help=(
            "Multiplier for a cluster's generated load. Can be repeated, e.g. "
            "gpu_node_a100=1.5 x86_epyc_rome=0.8."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "cauchy"],
        default="normal",
        help="Instruction/CPI distribution mode for generated jobs.",
    )
    parser.add_argument(
        "--min-runtime-s",
        type=float,
        default=DEFAULT_MIN_RUNTIME_S,
        help=(
            "Floor for each job's actual runtime in seconds (default 60). "
            "Prevents sub-minute jobs that render as one-step power spikes."
        ),
    )
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=None,
        help=(
            "Optional cap on each job's actual runtime in seconds. Clamps the "
            "week-long tail that drags makespan on small partitions without "
            "adding new dynamics (e.g. 259200 = 3 days)."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def clusters_for_system(system_config):
    return system_config["clusters"] if "clusters" in system_config else system_config


def queues_for_system(system_config):
    return list(system_config.get("queues", [])) if "clusters" in system_config else []


def queue_matches_runtime(queue_definition, runtime_s):
    min_runtime_s = float(queue_definition.get("min_runtime_s", 0.0))
    max_runtime_s = queue_definition.get("max_runtime_s")
    if runtime_s < min_runtime_s:
        return False
    if max_runtime_s is not None and runtime_s >= float(max_runtime_s):
        return False
    return True


def queue_name_for_runtime(queue_definitions, runtime_s):
    for definition in queue_definitions:
        if queue_matches_runtime(definition, runtime_s):
            return str(definition["name"])
    raise ValueError(f"No queue definition matches requested walltime {runtime_s:.2f}s.")


def split_jobs_by_queue(df_jobs, queue_definitions):
    if not queue_definitions:
        raise ValueError("Supercomputer config must define queues to generate queue datasets.")

    queue_names = [
        queue_name_for_runtime(queue_definitions, runtime_s)
        for runtime_s in df_jobs["Requested_Walltime_s"]
    ]
    split_jobs = {}
    for queue_definition in queue_definitions:
        queue_name = str(queue_definition["name"])
        queue_df = df_jobs[[name == queue_name for name in queue_names]].copy()
        split_jobs[queue_name] = queue_df
    return split_jobs


def parse_cluster_load_weights(raw_weights, clusters):
    weights = {ctype: 1.0 for ctype in clusters}
    for raw_weight in raw_weights or []:
        if "=" not in raw_weight:
            raise ValueError(
                f"Invalid cluster load weight '{raw_weight}'. Use CLUSTER=WEIGHT."
            )
        cluster_name, value = raw_weight.split("=", 1)
        cluster_name = cluster_name.strip()
        if cluster_name not in clusters:
            raise ValueError(
                f"Unknown cluster '{cluster_name}'. Available: {', '.join(clusters)}"
            )
        weight = float(value)
        if weight <= 0:
            raise ValueError("Cluster load weights must be positive.")
        weights[cluster_name] = weight
    return weights


def expected_job_node_count(cluster_nodes):
    if cluster_nodes is not None and cluster_nodes > 9:
        max_nodes = int(cluster_nodes * 0.2)
        max_nodes = max(max_nodes, 9)
        return (max_nodes - 1.0) / np.log(max_nodes)
    return float(np.dot(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        np.array([0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]),
    ))


def cluster_balance_score(ctype, num_nodes, node_config=None):
    base_frequency = 1.0
    if node_config and ctype in node_config:
        base_frequency = float(node_config[ctype].get("base_frequency_ghz", 1.0))
    return (float(num_nodes) * base_frequency) / expected_job_node_count(num_nodes)


def job_counts_for_clusters(
    clusters,
    min_jobs,
    max_jobs,
    strategy=DEFAULT_LOAD_STRATEGY,
    cluster_load_weights=None,
    node_config=None,
):
    if min_jobs <= 0 or max_jobs <= 0:
        raise ValueError("min_jobs and max_jobs must be positive.")
    if max_jobs < min_jobs:
        raise ValueError("max_jobs must be greater than or equal to min_jobs.")
    if strategy not in {"balanced", "node-proportional", "equal"}:
        raise ValueError(f"Unknown load strategy: {strategy}")

    weights = {ctype: 1.0 for ctype in clusters}
    if cluster_load_weights:
        weights = validate_cluster_load_weights(cluster_load_weights, clusters)

    if strategy == "equal":
        max_weight = max(weights.get(ctype, 1.0) for ctype in clusters)
        return {
            ctype: max(min_jobs, int(round(max_jobs * (weights.get(ctype, 1.0) / max_weight))))
            for ctype in clusters
        }

    if strategy == "node-proportional":
        scores = {
            ctype: float(num_nodes) * weights.get(ctype, 1.0)
            for ctype, num_nodes in clusters.items()
        }
    else:
        scores = {
            ctype: cluster_balance_score(ctype, num_nodes, node_config) * weights.get(ctype, 1.0)
            for ctype, num_nodes in clusters.items()
        }

    max_score = max(scores.values())
    if max_score <= 0:
        return {ctype: min_jobs for ctype in clusters}

    return {
        ctype: max(min_jobs, int(round(max_jobs * (score / max_score))))
        for ctype, score in scores.items()
    }


def system_arrival_window_s(job_counts, arrival_interval_s, arrival_window_hours):
    if arrival_window_hours is not None:
        return arrival_window_hours * 3600
    return (max(job_counts.values()) - 1) * arrival_interval_s


def load_supercomputers_config(path=script_dir):
    return load_supercomputers_config_from_dir(path)


def load_node_config(path=script_dir):
    return load_node_config_from_dir(path)


def validate_cluster_load_weights(weights, clusters):
    normalized = {ctype: 1.0 for ctype in clusters}
    for ctype, weight in (weights or {}).items():
        if ctype not in clusters:
            raise ValueError(f"Unknown cluster '{ctype}'. Available: {', '.join(clusters)}")
        weight = float(weight)
        if weight <= 0:
            raise ValueError("Cluster load weights must be positive.")
        normalized[ctype] = weight
    return normalized


def generate_system_datasets(
    system_name,
    supercomputers_config=None,
    output_dir=script_dir,
    node_config=None,
    min_jobs=DEFAULT_MIN_JOBS,
    max_jobs=DEFAULT_MAX_JOBS,
    arrival_interval_s=DEFAULT_ARRIVAL_INTERVAL_S,
    arrival_window_hours=None,
    seed=None,
    mode="normal",
    load_strategy=DEFAULT_LOAD_STRATEGY,
    cluster_load_weights=None,
    min_runtime_s=DEFAULT_MIN_RUNTIME_S,
    max_runtime_s=None,
    progress=print,
):
    if seed is not None:
        np.random.seed(seed)

    if supercomputers_config is None:
        supercomputers_config = load_supercomputers_config()
    if node_config is None:
        node_config = load_node_config()

    if system_name not in supercomputers_config:
        raise ValueError(
            f"Unknown system '{system_name}'. Available: {list(supercomputers_config.keys())}"
        )

    system_config = supercomputers_config[system_name]
    clusters = clusters_for_system(system_config)
    queue_definitions = queues_for_system(system_config)
    if not queue_definitions:
        raise ValueError(
            f"System '{system_name}' must define queues to generate queue datasets."
        )
    cluster_load_weights = validate_cluster_load_weights(cluster_load_weights, clusters)
    job_counts = job_counts_for_clusters(
        clusters,
        min_jobs,
        max_jobs,
        strategy=load_strategy,
        cluster_load_weights=cluster_load_weights,
        node_config=node_config,
    )
    arrival_window_s = system_arrival_window_s(
        job_counts,
        arrival_interval_s=arrival_interval_s,
        arrival_window_hours=arrival_window_hours,
    )

    if progress:
        progress(
            f"\nGenerating {load_strategy} datasets for '{system_name}' "
            f"({min_jobs}-{max_jobs} jobs, arrival window {arrival_window_s / 3600:.2f}h)..."
        )

    system_jobs_dir = jobs_dataset_dir(output_dir, system_name)
    os.makedirs(system_jobs_dir, exist_ok=True)
    summaries = []
    for ctype, num_nodes in clusters.items():
        n_jobs_target = job_counts[ctype]
        avg_arrival_interval_s = arrival_window_s / max(n_jobs_target - 1, 1)
        if progress:
            progress(
                f"  -> [{ctype}]: {num_nodes} nodes, load weight "
                f"{cluster_load_weights[ctype]:.2f} -> {n_jobs_target} jobs "
                f"(avg arrival interval {avg_arrival_interval_s:.2f}s)"
            )

        base_frequency_ghz = float(
            (node_config.get(ctype, {}) or {}).get("base_frequency_ghz", 2.0)
        )
        # Never generate jobs bigger than the queue node limit (they'd be
        # rejected as node-limit violations), nor bigger than the cluster.
        queue_node_caps = [
            int(q["max_nodes"]) for q in queue_definitions
            if q.get("max_nodes") is not None
        ]
        max_job_nodes = min([num_nodes] + queue_node_caps) if queue_node_caps else num_nodes
        df_jobs = generate_jobs(
            n_jobs_target,
            mode=mode,
            cluster_nodes=num_nodes,
            base_frequency_ghz=base_frequency_ghz,
            min_runtime_s=min_runtime_s,
            max_runtime_s=max_runtime_s,
            max_job_nodes=max_job_nodes,
            arrival_interval_s=arrival_interval_s,
            arrival_window_s=arrival_window_s,
        )

        queue_summaries = []
        for queue_name, queue_df in split_jobs_by_queue(df_jobs, queue_definitions).items():
            out_path = jobs_dataset_path(output_dir, system_name, ctype, queue_name)
            queue_df.to_csv(out_path, index=False)
            queue_summaries.append({
                "queue": queue_name,
                "jobs": len(queue_df),
                "filename": os.path.basename(out_path),
                "path": str(out_path),
            })
            if progress:
                progress(
                    f"     Saved {queue_name:6s}: {len(queue_df):5d} jobs -> {out_path}"
                )

        summaries.append({
            "cluster": ctype,
            "nodes": num_nodes,
            "jobs": n_jobs_target,
            "load_weight": cluster_load_weights[ctype],
            "avg_arrival_interval_s": avg_arrival_interval_s,
            "queues": queue_summaries,
        })

    return {
        "system": system_name,
        "load_strategy": load_strategy,
        "min_jobs": min_jobs,
        "max_jobs": max_jobs,
        "arrival_window_s": arrival_window_s,
        "job_counts": job_counts,
        "cluster_load_weights": cluster_load_weights,
        "clusters": summaries,
    }


def main():
    args = parse_args()

    if not os.path.isdir(supercomputers_dir) and not os.path.exists(supercomputers_file):
        print(f"Error: {supercomputers_dir} not found.")
        sys.exit(1)

    sc_data = load_supercomputers_config()
    node_config = load_node_config()
    if args.seed is not None:
        np.random.seed(args.seed)

    systems = list(sc_data.keys())
    if args.all:
        selected_systems = systems
    elif args.system:
        if args.system not in systems:
            print(f"Error: unknown system '{args.system}'. Available: {systems}")
            sys.exit(1)
        selected_systems = [args.system]
    else:
        print("=== Supercomputer Job Generator ===")
        print("Select a supercomputer to generate jobs for:")
        for i, sys_name in enumerate(systems):
            print(f"  {i + 1}. {sys_name}")
        
        while True:
            try:
                choice = input(f"Enter your choice (1-{len(systems)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(systems):
                    selected_systems = [systems[idx]]
                    break
                print("Invalid number.")
            except ValueError:
                print("Please enter a valid integer.")

    min_jobs = args.min_jobs
    max_jobs = args.max_jobs
    if not args.all and not args.system:
        while True:
            try:
                min_input = input("Enter minimum jobs for the smallest cluster (default 200): ")
                min_jobs = int(min_input) if min_input.strip() else min_jobs
            
                max_input = input("Enter maximum jobs for the largest cluster (default 1000): ")
                max_jobs = int(max_input) if max_input.strip() else max_jobs
            
                if min_jobs > 0 and max_jobs >= min_jobs:
                    break
                print("Values must be positive and max >= min.")
            except ValueError:
                print("Please enter valid integers.")

    for selected_system in selected_systems:
        clusters = clusters_for_system(sc_data[selected_system])
        try:
            cluster_load_weights = parse_cluster_load_weights(
                args.cluster_load_weight,
                clusters,
            )
            generate_system_datasets(
                selected_system,
                supercomputers_config=sc_data,
                output_dir=script_dir,
                node_config=node_config,
                min_jobs=min_jobs,
                max_jobs=max_jobs,
                arrival_interval_s=args.arrival_interval_s,
                arrival_window_hours=args.arrival_window_hours,
                seed=None,
                mode=args.mode,
                load_strategy=args.load_strategy,
                cluster_load_weights=cluster_load_weights,
                min_runtime_s=args.min_runtime_s,
                max_runtime_s=args.max_runtime_s,
                progress=print,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        
    print("\nJob generation complete.")

if __name__ == "__main__":
    main()
