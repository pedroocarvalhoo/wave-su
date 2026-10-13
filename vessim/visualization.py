from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from vessim.wave.scheduler import JobQueueScheduler


def plot_gantt_chart(
    scheduler: 'JobQueueScheduler',
    output_file: str = "gantt_chart.png",
    max_nodes: int = 20
):
    """
    Create a Gantt chart showing job execution timeline.
    Auto-limits to max_nodes (default 20) for readability.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    # Only show nodes that actually ran jobs, limited to max_nodes
    nodes_with_work = [
        name for name in sorted(scheduler.timeline.keys())
        if len(scheduler.timeline[name]) > 0
    ]
    nodes_to_plot = nodes_with_work[:max_nodes]
    
    all_times = []
    for node_name in nodes_to_plot:
        for start_time, end_time, job_id in scheduler.timeline[node_name]:
            all_times.append(start_time)
            if end_time:
                all_times.append(end_time)
    
    if not all_times:
        print(f"No timeline data to plot for {scheduler.cluster_type}")
        return
    
    ref_time = min(all_times)
    
    fig, ax = plt.subplots(figsize=(16, max(6, len(nodes_to_plot) * 0.4)))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for idx, node_name in enumerate(nodes_to_plot):
        for start_time, end_time, job_id in scheduler.timeline[node_name]:
            if end_time is None:
                continue
            start_hours = (start_time - ref_time).total_seconds() / 3600
            end_hours = (end_time - ref_time).total_seconds() / 3600
            duration_hours = end_hours - start_hours
            
            if job_id is None:
                ax.barh(idx, duration_hours, left=start_hours, height=0.8, 
                       color='lightgray', edgecolor='black', linewidth=0.3)
            else:
                color = colors[job_id % len(colors)]
                ax.barh(idx, duration_hours, left=start_hours, height=0.8,
                       color=color, edgecolor='black', linewidth=0.3)
    
    ax.set_yticks(range(len(nodes_to_plot)))
    ax.set_yticklabels(nodes_to_plot, fontsize=8)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Nodes', fontsize=12)
    
    total_nodes = len(scheduler.timeline)
    shown = len(nodes_to_plot)
    title = f'{scheduler.cluster_type} — Gantt Chart ({shown}/{total_nodes} nodes shown)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    idle_patch = mpatches.Patch(color='lightgray', label='Idle')
    job_patch = mpatches.Patch(color=colors[0], label='Job Execution')
    ax.legend(handles=[idle_patch, job_patch], loc='upper right')
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Gantt chart saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def plot_cluster_dashboard(df, schedulers, output_file: str = "cluster_dashboard.png", min_soc: float = 0.2):
    """
    Comprehensive 5-panel dashboard:
      1. Job Progress (completed jobs over time, per cluster)
      2. Cluster Utilization (% busy nodes over time)
      3. Power vs Budget (actual power vs budget ceiling)
      4. Battery State of Charge
      5. Cumulative CO₂ Emissions
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    cluster_types = [s.cluster_type for s in schedulers]
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    cluster_colors = {ct: palette[i % len(palette)] for i, ct in enumerate(cluster_types)}
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    fig.suptitle('Multi-Cluster HPC Simulation Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    # ====== Panel 1: Job Progress ======
    ax = axes[0]
    for ct in cluster_types:
        col = f'{ct}_completed'
        if col in df.columns:
            ax.plot(df.index, df[col].ffill().fillna(0),
                    label=ct, color=cluster_colors[ct], linewidth=2)
    
    # Total completed (stacked)
    total_cols = [f'{ct}_completed' for ct in cluster_types if f'{ct}_completed' in df.columns]
    if total_cols:
        total = sum(df[c].ffill().fillna(0) for c in total_cols)
        ax.plot(df.index, total, label='Total', color='black', linewidth=2, linestyle='--', alpha=0.6)
    
    ax.set_ylabel('Jobs Completed')
    ax.set_title('Job Progress Over Time', fontweight='bold')
    ax.legend(loc='upper left', ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ====== Panel 2: Cluster Utilization ======
    ax = axes[1]
    for ct in cluster_types:
        col = f'{ct}_util'
        if col in df.columns:
            util_pct = df[col].ffill().fillna(0) * 100
            ax.plot(df.index, util_pct, label=ct, color=cluster_colors[ct], linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=100, color='red', linestyle=':', alpha=0.4, label='100%')
    ax.set_ylabel('Utilization (%)')
    ax.set_ylim(-5, 110)
    ax.set_title('Cluster Node Utilization', fontweight='bold')
    ax.legend(loc='upper right', ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ====== Panel 3: Power vs Budget ======
    ax = axes[2]
    for ct in cluster_types:
        power_col = f'{ct}_power'
        budget_col = f'{ct}_budget'
        color = cluster_colors[ct]
        
        if power_col in df.columns:
            power_kw = df[power_col].ffill().fillna(0) / 1000
            ax.plot(df.index, power_kw, label=f'{ct} power', color=color, linewidth=1.5)
        
        if budget_col in df.columns:
            budget_kw = df[budget_col].ffill().fillna(0) / 1000
            ax.plot(df.index, budget_kw, label=f'{ct} budget', color=color, 
                    linewidth=1, linestyle='--', alpha=0.6)
    
    ax.set_ylabel('Power (kW)')
    ax.set_title('Cluster Power Draw vs Energy Budget', fontweight='bold')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ====== Panel 4: Battery SoC ======
    ax = axes[3]
    soc_col = None
    for col in df.columns:
        if 'soc' in col.lower() or 'state_of_charge' in col.lower():
            soc_col = col
            break
    
    if soc_col:
        soc = df[soc_col].ffill() * 100
        ax.fill_between(df.index, soc, alpha=0.3, color='#2E8B57')
        ax.plot(df.index, soc, color='#2E8B57', linewidth=2, label='Battery SoC')
        ax.axhline(y=min_soc * 100, color='red', linestyle='--', alpha=0.7, label=f'Min ({min_soc*100:.0f}%)')
        ax.set_ylim(0, 105)
    else:
        ax.text(0.5, 0.5, 'No battery SoC data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, color='gray')
    
    ax.set_ylabel('SoC (%)')
    ax.set_title('Battery State of Charge', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ====== Panel 5: CO₂ Emissions ======
    ax = axes[4]
    if 'co2' in df.columns:
        # CO₂ is stored as string tuples like "(123.45,)" in the CSV.
        co2_raw = df['co2'].fillna(0)
        co2 = pd.to_numeric(
            co2_raw.astype(str).str.strip('(),').str.split(',').str[0],
            errors='coerce'
        ).fillna(0)
        total_co2_kg = co2.cumsum() / 1000

        partition_power = {
            ct: (
                pd.to_numeric(df[f'{ct}_power'], errors='coerce').ffill().fillna(0)
                if f'{ct}_power' in df.columns
                else pd.Series(0.0, index=df.index)
            )
            for ct in cluster_types
        }
        total_power = sum(partition_power.values(), pd.Series(0.0, index=df.index))
        for ct in cluster_types:
            power = partition_power[ct]
            fraction = power.where(total_power > 0, 0) / total_power.where(total_power > 0, 1)
            cluster_co2_kg = (co2 * fraction).cumsum() / 1000
            ax.plot(df.index, cluster_co2_kg, label=f'{ct}',
                    color=cluster_colors[ct], linewidth=1.5, zorder=3)

        # Shade only the area between zero and the measured system total.
        ax.fill_between(df.index, 0, total_co2_kg, alpha=0.28,
                        color='#d9c7f2', zorder=1)
        ax.plot(df.index, total_co2_kg, color='#8e44ad', linewidth=2.6,
                label='Total CO₂', zorder=4)
        ax.set_ylabel('CO₂ (kg)')
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, 'No CO₂ data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.set_title('Cumulative CO₂ Emissions', fontweight='bold')
    ax.set_xlabel('Time')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # X-axis formatting — auto-pick ticks for the time span
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for a in axes:
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Cluster dashboard saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def plot_job_analysis(schedulers, output_file: str = "job_analysis.png"):
    """
    Job-level statistical analysis:
      1. Job duration distribution (histogram per cluster)
      2. Node productivity (top N nodes by jobs completed)
      3. Top 20 nodes — busy vs idle time (stacked horizontal bars)
      4. Per-cluster summary — total busy vs idle hours
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    cluster_types = [s.cluster_type for s in schedulers]
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    cluster_colors = {ct: palette[i % len(palette)] for i, ct in enumerate(cluster_types)}

    # ── helper: compute busy seconds per node from timeline ──────────────────
    def _node_busy_idle(scheduler):
        """Returns {node_name: (busy_h, idle_h)} for each node in the scheduler."""
        # measure total simulation span from the timeline
        all_times = []
        for entries in scheduler.timeline.values():
            for start, end, _ in entries:
                if start:
                    all_times.append(start)
                if end:
                    all_times.append(end)
        if not all_times:
            return {}
        sim_start = min(all_times)
        sim_end = max(all_times)
        sim_total_h = (sim_end - sim_start).total_seconds() / 3600

        result = {}
        for node_name, entries in scheduler.timeline.items():
            busy_secs = sum(
                (end - start).total_seconds()
                for start, end, _ in entries
                if start is not None and end is not None
            )
            busy_h = busy_secs / 3600
            idle_h = max(0.0, sim_total_h - busy_h)
            result[node_name] = (busy_h, idle_h)
        return result

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Job Execution Analysis', fontsize=16, fontweight='bold')

    # ====== Panel 1: Job Duration Distribution ======
    ax = axes[0, 0]
    all_durations = {}
    for scheduler in schedulers:
        ct = scheduler.cluster_type
        durations = [
            j.actual_t_exec
            for j in scheduler.job_queue.jobs
            if j.status == 'completed'
        ]
        all_durations[ct] = durations

    max_dur = max((max(d) if d else 0) for d in all_durations.values()) if all_durations else 1
    flat = [d for ds in all_durations.values() for d in ds]
    bins = np.linspace(0, min(max_dur, np.percentile(flat, 95) if flat else 1), 40)

    for ct in cluster_types:
        if all_durations.get(ct):
            ax.hist(all_durations[ct], bins=bins, alpha=0.5,
                    label=f'{ct} ({len(all_durations[ct])} jobs)',
                    color=cluster_colors[ct], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Actual Model Execution Time (seconds)')
    ax.set_ylabel('Number of Jobs')
    ax.set_title('Job Duration Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ====== Panel 2: Node Productivity ======
    ax = axes[0, 1]

    node_data = []
    for scheduler in schedulers:
        ct = scheduler.cluster_type
        for node_name, node in scheduler.rm.nodes.items():
            if node.total_jobs_executed > 0:
                node_data.append((node_name, node.total_jobs_executed, ct))

    node_data.sort(key=lambda x: x[1], reverse=True)
    top_nodes = node_data[:20]

    if top_nodes:
        names   = [n[0] for n in top_nodes]
        counts  = [n[1] for n in top_nodes]
        colors_list = [cluster_colors[n[2]] for n in top_nodes]

        ax.barh(range(len(top_nodes)), counts, color=colors_list, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_nodes)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Jobs Completed')

        legend_handles = [mpatches.Patch(color=cluster_colors[ct], label=ct) for ct in cluster_types]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No completed jobs', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.set_title('Top 20 Most Productive Nodes', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # ====== Panel 3: Random sample of nodes — Busy vs Idle time ======
    # Proportional to number of clusters: total_sample / n_clusters per cluster
    ax = axes[1, 0]
    import random as _random

    total_sample = 21
    n_clusters = len(schedulers)
    per_cluster = max(1, total_sample // n_clusters)

    sampled_nodes = []   # (node_name, busy_h, idle_h, cluster_type)
    for scheduler in schedulers:
        ct = scheduler.cluster_type
        busy_idle = _node_busy_idle(scheduler)
        candidates = [
            (name, bh, ih, ct)
            for name, (bh, ih) in busy_idle.items()
            if bh > 0
        ]
        k = min(per_cluster, len(candidates))
        sampled_nodes.extend(_random.sample(candidates, k))

    # Sort by cluster then by name for a tidy display
    sampled_nodes.sort(key=lambda x: (x[3], x[0]))

    if sampled_nodes:
        names_t     = [n[0] for n in sampled_nodes]
        busy_h      = [n[1] for n in sampled_nodes]
        idle_h      = [n[2] for n in sampled_nodes]
        ct_list     = [n[3] for n in sampled_nodes]
        busy_colors = [cluster_colors[ct] for ct in ct_list]

        ys = list(range(len(sampled_nodes)))
        ax.barh(ys, busy_h, color=busy_colors, edgecolor='black', linewidth=0.4, label='Busy')
        ax.barh(ys, idle_h, left=busy_h, color='lightgray', edgecolor='black', linewidth=0.4, label='Idle')

        # Labels inside bars: busy% in colored bar, idle% right-aligned in gray bar
        for yi, (bh, ih) in enumerate(zip(busy_h, idle_h)):
            total = bh + ih
            if total > 0:
                busy_pct = bh / total * 100
                idle_pct = ih / total * 100
                # busy %: right-aligned inside the colored segment
                if bh > 0:
                    ax.text(bh - total * 0.01, yi, f'{busy_pct:.0f}%',
                            ha='right', va='center', fontsize=7,
                            color='white', fontweight='bold')
                # idle %: right-aligned inside the gray segment
                if ih > 0:
                    ax.text(bh + ih - total * 0.01, yi, f'{idle_pct:.0f}%',
                            ha='right', va='center', fontsize=7,
                            color='#555555', fontweight='bold')

        ax.set_yticks(ys)
        ax.set_yticklabels(names_t, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Time (hours)')

        idle_patch = mpatches.Patch(color='lightgray', label='Idle')
        cluster_handles = [mpatches.Patch(color=cluster_colors[ct], label=f'{ct} (busy)') for ct in cluster_types]
        ax.legend(handles=cluster_handles + [idle_patch], loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No timeline data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.set_title(f'{per_cluster} Random Nodes per Cluster — Busy vs Idle Time', fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # ====== Panel 4: Per-cluster summary — total busy vs idle hours ======
    ax = axes[1, 1]

    cluster_summary = {}   # ct -> (total_busy_h, total_idle_h)
    for scheduler in schedulers:
        ct = scheduler.cluster_type
        busy_idle = _node_busy_idle(scheduler)
        total_busy = sum(b for b, _ in busy_idle.values())
        total_idle = sum(i for _, i in busy_idle.values())
        cluster_summary[ct] = (total_busy, total_idle)

    if cluster_summary:
        cts     = list(cluster_summary.keys())
        busy_v  = [cluster_summary[ct][0] for ct in cts]
        idle_v  = [cluster_summary[ct][1] for ct in cts]
        x       = np.arange(len(cts))
        w       = 0.5

        bars_b = ax.bar(x, busy_v, width=w, label='Busy',
                        color=[cluster_colors[ct] for ct in cts], edgecolor='black', linewidth=0.5)
        bars_i = ax.bar(x, idle_v, width=w, bottom=busy_v, label='Idle',
                        color='lightgray', edgecolor='black', linewidth=0.5)

        # Annotate utilisation % on each stacked bar
        for xi, ct in enumerate(cts):
            total = busy_v[xi] + idle_v[xi]
            if total > 0:
                pct = busy_v[xi] / total * 100
                ax.text(xi, total + total * 0.01, f'{pct:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(cts, fontsize=10)
        ax.set_ylabel('Node-Hours')
        ax.set_title('Per-Cluster Total Busy vs Idle Node-Hours', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No timeline data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Job analysis saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def plot_job_arrivals(schedulers, output_file: str = "job_arrivals.png"):
    """
    Workload arrival analysis based on each job's arrival time and walltime.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    cluster_types = [s.cluster_type for s in schedulers]
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    cluster_colors = {ct: palette[i % len(palette)] for i, ct in enumerate(cluster_types)}

    arrivals = []
    for scheduler in schedulers:
        for job in scheduler.job_queue.jobs:
            arrivals.append(
                {
                    "cluster": scheduler.cluster_type,
                    "arrival_time": job.arrival_time,
                    "requested_walltime_s": job.requested_walltime_s,
                    "queue": _job_queue_label(scheduler, job),
                }
            )

    if not arrivals:
        print(f"⚠️ No jobs available to plot {os.path.basename(output_file)}")
        return

    sim_start = min(row["arrival_time"] for row in arrivals)
    for row in arrivals:
        row["arrival_hour"] = (row["arrival_time"] - sim_start).total_seconds() / 3600
        row["requested_walltime_h"] = row["requested_walltime_s"] / 3600

    max_arrival_h = max(row["arrival_hour"] for row in arrivals)
    arrival_bins = np.linspace(0, max(1.0, max_arrival_h), 40)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Job Arrival Distribution', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    hist_values = []
    hist_labels = []
    hist_colors = []
    for ct in cluster_types:
        values = [row["arrival_hour"] for row in arrivals if row["cluster"] == ct]
        if values:
            hist_values.append(values)
            hist_labels.append(f'{ct} ({len(values)} jobs)')
            hist_colors.append(cluster_colors[ct])
    if hist_values:
        ax.hist(
            hist_values,
            bins=arrival_bins,
            label=hist_labels,
            color=hist_colors,
            edgecolor='black',
            linewidth=0.4,
            rwidth=0.9,
        )
    ax.set_xlabel('Arrival Time Since Simulation Start (hours)')
    ax.set_ylabel('Jobs')
    ax.set_title('Arrival Histogram', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for ct in cluster_types:
        values = sorted(row["arrival_hour"] for row in arrivals if row["cluster"] == ct)
        if values:
            ax.step(
                values,
                np.arange(1, len(values) + 1),
                where='post',
                label=ct,
                color=cluster_colors[ct],
                linewidth=1.8,
            )
    ax.set_xlabel('Arrival Time Since Simulation Start (hours)')
    ax.set_ylabel('Cumulative Jobs Submitted')
    ax.set_title('Cumulative Arrival Stream', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for ct in cluster_types:
        subset = [row for row in arrivals if row["cluster"] == ct]
        if subset:
            ax.scatter(
                [row["arrival_hour"] for row in subset],
                [row["requested_walltime_h"] for row in subset],
                s=10,
                alpha=0.35,
                color=cluster_colors[ct],
                label=ct,
            )
    ax.axhline(24, color='black', linestyle='--', linewidth=1, alpha=0.5, label='24h')
    ax.axhline(7 * 24, color='black', linestyle=':', linewidth=1.2, alpha=0.5, label='1 week')
    ax.set_xlabel('Arrival Time Since Simulation Start (hours)')
    ax.set_ylabel('Requested Walltime (hours)')
    ax.set_title('Requested Walltime by Arrival Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    queue_order = ["short", "medium", "long", "unclassified"]
    queue_colors = {
        "short": "#2ecc71",
        "medium": "#f1c40f",
        "long": "#e74c3c",
        "unclassified": "#95a5a6",
    }
    x = np.arange(len(cluster_types))
    bottom = np.zeros(len(cluster_types))
    for queue_name in queue_order:
        counts = []
        for ct in cluster_types:
            counts.append(
                sum(
                    1
                    for row in arrivals
                    if row["cluster"] == ct and row["queue"] == queue_name
                )
            )
        if any(counts):
            ax.bar(
                x,
                counts,
                bottom=bottom,
                label=queue_name,
                color=queue_colors[queue_name],
                edgecolor='black',
                linewidth=0.4,
            )
            bottom += np.array(counts)
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_types, rotation=20, ha='right')
    ax.set_ylabel('Jobs')
    ax.set_title('Requested Walltime Queue Mix', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Job arrivals saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def _job_queue_label(scheduler, job) -> str:
    if getattr(job, "queue_name", None):
        return job.queue_name

    requested_walltime_s = getattr(job, "requested_walltime_s", None)
    if requested_walltime_s is None:
        return "unclassified"

    for definition in getattr(scheduler, "queue_definitions", []):
        if definition.matches_runtime(requested_walltime_s):
            return definition.name

    return "unclassified"


def plot_queue_evolution(
    results_dir: str,
    schedulers=None,
    output_file: str = "queue_evolution.png",
):
    """
    Plot queue arrivals, backlog, service, and wait pressure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    debug_df = _load_scheduler_debug_events(results_dir)
    metrics_df = _load_jobs_metrics(results_dir)
    arrivals_df = _load_scheduler_arrivals(results_dir, schedulers)

    if debug_df.empty:
        print(f"No scheduler debug CSVs found to plot {os.path.basename(output_file)}")
        return

    step_df = debug_df[debug_df["Event"] == "step_summary"].copy()
    if step_df.empty:
        step_df = debug_df.copy()

    clusters = sorted(str(cluster) for cluster in debug_df["Cluster"].dropna().unique())
    if not clusters:
        print(f"No cluster data found to plot {os.path.basename(output_file)}")
        return

    queue_names = _queue_names_from_debug_and_metrics(debug_df, metrics_df, arrivals_df)
    if not queue_names:
        queue_names = ["unclassified"]

    fig, axes = plt.subplots(
        len(clusters),
        4,
        figsize=(28, max(5, 4.5 * len(clusters))),
        squeeze=False,
    )
    fig.suptitle("Queue Evolution", fontsize=16, fontweight="bold")

    for row_idx, cluster in enumerate(clusters):
        cluster_steps = step_df[step_df["Cluster"] == cluster].sort_values("Time")
        cluster_events = debug_df[debug_df["Cluster"] == cluster].sort_values("Time")
        cluster_metrics = metrics_df[metrics_df["Cluster"] == cluster] if not metrics_df.empty else pd.DataFrame()
        cluster_arrivals = arrivals_df[arrivals_df["Cluster"] == cluster] if not arrivals_df.empty else pd.DataFrame()

        _plot_arrival_queue_cumulative(
            axes[row_idx, 0],
            cluster_arrivals,
            cluster_steps,
            queue_names,
            title=f"{cluster} jobs submitted",
        )
        _plot_pending_queue_stack(
            axes[row_idx, 1],
            cluster_steps,
            queue_names,
            title=f"{cluster} pending after scheduling",
        )
        _plot_started_queue_cumulative(
            axes[row_idx, 2],
            cluster_events,
            cluster_steps,
            queue_names,
            title=f"{cluster} jobs started",
        )
        _plot_queue_wait_pressure(
            axes[row_idx, 3],
            cluster_steps,
            cluster_metrics,
            queue_names,
            title=f"{cluster} wait pressure",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Queue evolution saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def plot_job_event_timeline(
    df,
    schedulers,
    output_file: str = "job_event_timeline.html",
):
    """
    Create an interactive horizontal HTML timeline with job lifecycle events.
    """
    import html
    import json
    import math
    import os
    import pandas as pd

    events, links, lanes = _job_event_timeline_data(schedulers)
    if not events:
        print(f"No job events available to write {os.path.basename(output_file)}")
        return

    event_times = [event["timestamp"] for event in events]
    budget_df = df.copy()
    if not isinstance(budget_df.index, pd.DatetimeIndex):
        budget_df.index = pd.to_datetime(budget_df.index)
    budget_df = budget_df.sort_index()

    min_time = min([budget_df.index.min()] + event_times)
    max_time = max([budget_df.index.max()] + event_times)
    total_seconds = max(1.0, (max_time - min_time).total_seconds())
    total_hours = total_seconds / 3600.0

    left_margin = 230
    right_margin = 120
    timeline_width = max(2600, int(total_hours * 42))
    svg_width = left_margin + timeline_width + right_margin
    budget_top = 54
    budget_height = 238
    axis_y = budget_top + budget_height + 34
    lanes_top = axis_y + 46
    lane_height = 76
    svg_height = lanes_top + len(lanes) * lane_height + 72

    def x_for_time(timestamp) -> float:
        return left_margin + ((timestamp - min_time).total_seconds() / total_seconds) * timeline_width

    lane_y = {
        lane["key"]: lanes_top + idx * lane_height
        for idx, lane in enumerate(lanes)
    }
    event_offsets = {
        "submitted": 18,
        "started": 38,
        "finished": 58,
    }

    for idx, event in enumerate(events):
        event["dom_id"] = f"job-event-{idx}"
        event["x"] = round(x_for_time(event["timestamp"]), 2)
        event["y"] = round(lane_y[event["lane_key"]] + event_offsets[event["type"]], 2)
        event["time"] = event["timestamp"].isoformat(sep=" ")
        event.pop("timestamp")

    for link in links:
        source = events[link["source_index"]]
        target = events[link["target_index"]]
        link["x1"] = source["x"]
        link["y1"] = source["y"]
        link["x2"] = target["x"]
        link["y2"] = target["y"]

    time_ticks_svg = _job_event_time_ticks_svg(
        min_time=min_time,
        max_time=max_time,
        left_margin=left_margin,
        timeline_width=timeline_width,
        axis_y=axis_y,
        lanes_top=lanes_top,
        svg_height=svg_height,
    )
    budget_svg, budget_samples = _job_event_budget_svg(
        budget_df=budget_df,
        min_time=min_time,
        total_seconds=total_seconds,
        left_margin=left_margin,
        timeline_width=timeline_width,
        budget_top=budget_top,
        budget_height=budget_height,
        lanes_top=lanes_top,
        svg_height=svg_height,
    )
    lanes_svg = _job_event_lanes_svg(
        lanes=lanes,
        lane_y=lane_y,
        lane_height=lane_height,
        left_margin=left_margin,
        svg_width=svg_width,
        html_escape=html.escape,
    )
    links_svg = "\n".join(
        (
            f'<line class="job-link link-{html.escape(link["phase"])}" '
            f'data-job-key="{html.escape(link["job_key"])}" '
            f'x1="{link["x1"]}" y1="{link["y1"]}" x2="{link["x2"]}" y2="{link["y2"]}" />'
        )
        for link in links
    )
    events_svg = "\n".join(
        _job_event_circle_svg(event, html.escape)
        for event in events
    )

    data = {
        "events": events,
        "eventById": {event["id"]: event for event in events},
        "budgetSamples": budget_samples,
        "summary": {
            "jobs": len({event["job_key"] for event in events}),
            "events": len(events),
            "start": min_time.isoformat(sep=" "),
            "end": max_time.isoformat(sep=" "),
        },
    }
    json_data = json.dumps(data, ensure_ascii=True).replace("</", "<\\/")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Event Timeline</title>
<style>
{_job_event_timeline_css()}
</style>
</head>
<body>
<header>
  <div>
    <h1>Job Event Timeline</h1>
    <p>{data["summary"]["jobs"]} jobs, {data["summary"]["events"]} events, from {html.escape(data["summary"]["start"])} to {html.escape(data["summary"]["end"])}</p>
  </div>
  <div class="legend">
    <span><i class="submitted"></i>submitted</span>
    <span><i class="started"></i>started</span>
    <span><i class="finished"></i>finished</span>
    <span><i class="wait-link"></i>waiting interval</span>
    <span><i class="run-link"></i>running interval</span>
    <span><i class="renewable"></i>renewable budget</span>
    <span><i class="battery"></i>battery budget</span>
    <span><i class="grid"></i>grid budget</span>
    <span><i class="compute-power"></i>compute power</span>
  </div>
</header>
<main>
  <aside id="details">
    <h2>No event selected</h2>
    <p>Click a circle to highlight its job. Use the previous/next buttons to move through submitted, started, and finished events for that job.</p>
    <div class="search">
      <input id="jobSearch" type="text" placeholder="cluster:job_id or job_id">
      <button id="searchBtn">Find job</button>
    </div>
    <div id="eventDetails"></div>
    <div class="buttons">
      <button id="prevBtn" disabled>Previous event</button>
      <button id="nextBtn" disabled>Next event</button>
      <button id="clearBtn">Clear</button>
    </div>
  </aside>
  <section class="timeline-shell">
    <div id="timelineScroll" class="timeline-scroll">
      <svg id="timelineSvg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">
        <rect class="canvas-bg" x="0" y="0" width="{svg_width}" height="{svg_height}" />
        {budget_svg}
        {time_ticks_svg}
        {lanes_svg}
        <g id="linksLayer">{links_svg}</g>
        <g id="eventsLayer">{events_svg}</g>
        <g id="cursorLayer">
          <line id="budgetCursor" class="budget-cursor" x1="0" y1="{budget_top}" x2="0" y2="{svg_height - 34}" visibility="hidden" />
        <g id="budgetTooltip" class="budget-tooltip" visibility="hidden">
            <rect id="budgetTooltipBox" x="0" y="0" width="210" height="116" rx="6" />
            <text id="budgetTooltipTime" x="10" y="18"></text>
            <text id="budgetTooltipRenewable" x="10" y="39"></text>
            <text id="budgetTooltipBattery" x="10" y="58"></text>
            <text id="budgetTooltipGrid" x="10" y="77"></text>
            <text id="budgetTooltipPower" x="10" y="96"></text>
          </g>
        </g>
      </svg>
    </div>
  </section>
</main>
<script id="timeline-data" type="application/json">{json_data}</script>
<script>
{_job_event_timeline_js()}
</script>
</body>
</html>
"""

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Job event timeline saved to: {os.path.basename(output_file)}")


def _job_event_timeline_data(schedulers):
    import pandas as pd

    queue_names = set()
    cluster_names = []
    jobs_by_lane = {}
    raw_jobs = []

    for scheduler in schedulers:
        cluster = scheduler.cluster_type
        cluster_names.append(cluster)
        for job in scheduler.job_queue.jobs:
            queue = _job_queue_label(scheduler, job)
            queue_names.add(queue)
            lane_key = f"{cluster}|{queue}"
            jobs_by_lane[lane_key] = jobs_by_lane.get(lane_key, 0) + 1
            raw_jobs.append((scheduler, job, cluster, queue, lane_key))

    ordered_queues = _ordered_queue_names(queue_names)
    lanes = []
    for cluster in cluster_names:
        for queue in ordered_queues:
            lane_key = f"{cluster}|{queue}"
            if lane_key not in jobs_by_lane:
                continue
            lanes.append(
                {
                    "key": lane_key,
                    "cluster": cluster,
                    "queue": queue,
                    "jobs": jobs_by_lane[lane_key],
                }
            )

    events = []
    for _, job, cluster, queue, lane_key in raw_jobs:
        job_key = f"{cluster}:{job.job_id}"
        arrival_ts = getattr(job, "arrival_time", None)
        start_ts = getattr(job, "start_time", None)
        completion_ts = getattr(job, "completion_time", None)
        wait_h = _hours_between(arrival_ts, start_ts)
        run_h = _hours_between(start_ts, completion_ts)
        lifecycle = [
            ("submitted", arrival_ts),
            ("started", start_ts),
            ("finished", completion_ts),
        ]
        previous_event_id = None
        previous_event_index = None

        for event_type, timestamp in lifecycle:
            if timestamp is None or pd.isna(timestamp):
                continue

            event_id = f"{job_key}:{event_type}"
            event = {
                "id": event_id,
                "job_key": job_key,
                "job_id": str(job.job_id),
                "cluster": cluster,
                "queue": queue,
                "lane_key": lane_key,
                "type": event_type,
                "timestamp": pd.Timestamp(timestamp),
                "prev_id": previous_event_id,
                "next_id": None,
                "priority": getattr(job, "priority", ""),
                "nodes": getattr(job, "num_nodes", ""),
                "status": getattr(job, "status", ""),
                "requested_walltime_h": round(getattr(job, "requested_walltime_s", 0.0) / 3600.0, 3),
                "wait_h": wait_h,
                "run_h": run_h,
            }
            events.append(event)
            current_event_index = len(events) - 1

            if previous_event_index is not None:
                events[previous_event_index]["next_id"] = event_id

            previous_event_id = event_id
            previous_event_index = current_event_index

    events.sort(key=lambda event: (event["timestamp"], event["cluster"], event["job_id"], event["type"]))
    links = _rebuild_job_event_links(events)
    return events, links, lanes


def _hours_between(start_time, end_time):
    import pandas as pd

    if start_time is None or end_time is None or pd.isna(start_time) or pd.isna(end_time):
        return ""
    return round((pd.Timestamp(end_time) - pd.Timestamp(start_time)).total_seconds() / 3600.0, 3)


def _rebuild_job_event_links(events):
    event_index_by_id = {
        event["id"]: idx
        for idx, event in enumerate(events)
    }
    links = []
    for event in events:
        if not event["next_id"]:
            continue
        target_event = events[event_index_by_id[event["next_id"]]]
        phase = (
            "wait"
            if event["type"] == "submitted" and target_event["type"] == "started"
            else "run"
        )
        links.append(
            {
                "job_key": event["job_key"],
                "phase": phase,
                "source_index": event_index_by_id[event["id"]],
                "target_index": event_index_by_id[event["next_id"]],
            }
        )
    return links


def _job_event_time_ticks_svg(
    min_time,
    max_time,
    left_margin,
    timeline_width,
    axis_y,
    lanes_top,
    svg_height,
):
    import html
    import pandas as pd

    total_seconds = max(1.0, (max_time - min_time).total_seconds())
    tick_count = 12
    parts = [
        f'<line class="axis-line" x1="{left_margin}" y1="{axis_y}" x2="{left_margin + timeline_width}" y2="{axis_y}" />'
    ]
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        timestamp = min_time + pd.Timedelta(seconds=total_seconds * ratio)
        x = left_margin + timeline_width * ratio
        label = timestamp.strftime("%m-%d %H:%M")
        parts.append(
            f'<line class="tick-line" x1="{x:.2f}" y1="{axis_y - 8}" x2="{x:.2f}" y2="{svg_height - 36}" />'
        )
        parts.append(
            f'<text class="tick-label" x="{x:.2f}" y="{axis_y + 22}" text-anchor="middle">{html.escape(label)}</text>'
        )
    return "\n".join(parts)


def _job_event_budget_svg(
    budget_df,
    min_time,
    total_seconds,
    left_margin,
    timeline_width,
    budget_top,
    budget_height,
    lanes_top,
    svg_height,
):
    import math
    import pandas as pd

    budget_columns = [
        ("budget_renewable_w", "renewable_w", "budget-renewable", "Renewable budget"),
        ("budget_battery_w", "battery_w", "budget-battery", "Battery budget"),
        ("budget_grid_w", "grid_w", "budget-grid", "Grid budget"),
    ]
    cluster_power_columns = [
        column
        for column in budget_df.columns
        if column.endswith("_power") and column not in {"battery_power_w"}
    ]
    has_budget = any(column in budget_df.columns for column, _, _, _ in budget_columns)
    has_power = bool(cluster_power_columns)
    if budget_df.empty or not (has_budget or has_power):
        return "", []

    step = max(1, math.ceil(len(budget_df) / 900))
    sampled = budget_df.iloc[::step].copy()
    if sampled.index[-1] != budget_df.index[-1]:
        sampled = pd.concat([sampled, budget_df.iloc[[-1]]])
    if cluster_power_columns:
        sampled["_compute_power_w"] = (
            sampled[cluster_power_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        )
    else:
        sampled["_compute_power_w"] = 0.0

    max_budget = 1.0
    for column, _, _, _ in budget_columns:
        if column in sampled.columns:
            max_budget = max(max_budget, float(sampled[column].fillna(0).max()))
    max_budget = max(max_budget, float(sampled["_compute_power_w"].max()))

    track_gap = 12
    title_height = 26
    tracks = [
        *budget_columns,
        (None, "compute_power_w", "compute-power", "Total compute power"),
    ]
    track_height = (budget_height - title_height - track_gap * (len(tracks) - 1)) / len(tracks)
    max_budget_label = _format_watts(max_budget)
    parts = [
        f'<text class="budget-title" x="20" y="{budget_top + 18}">Budget and power</text>',
        f'<text class="budget-label" x="{left_margin}" y="{budget_top + 18}">Shared scale, max {max_budget_label}. Compare budget sources with actual compute power.</text>',
        f'<rect class="budget-frame" x="{left_margin}" y="{budget_top + title_height}" width="{timeline_width}" height="{budget_height - title_height}" />',
    ]

    budget_samples = []
    for timestamp, row in sampled.iterrows():
        x = left_margin + ((timestamp - min_time).total_seconds() / total_seconds) * timeline_width
        renewable_w = max(0.0, float(row.get("budget_renewable_w", 0.0)))
        battery_w = max(0.0, float(row.get("budget_battery_w", 0.0)))
        grid_w = max(0.0, float(row.get("budget_grid_w", 0.0)))
        compute_power_w = max(0.0, float(row.get("_compute_power_w", 0.0)))
        total_budget_w = renewable_w + battery_w + grid_w
        budget_samples.append(
            {
                "x": round(x, 2),
                "time": timestamp.isoformat(sep=" "),
                "renewable_w": round(renewable_w, 2),
                "battery_w": round(battery_w, 2),
                "grid_w": round(grid_w, 2),
                "total_w": round(total_budget_w, 2),
                "compute_power_w": round(compute_power_w, 2),
                "headroom_w": round(total_budget_w - compute_power_w, 2),
            }
        )

    sample_keys = {
        "budget_renewable_w": "renewable_w",
        "budget_battery_w": "battery_w",
        "budget_grid_w": "grid_w",
        None: "compute_power_w",
    }
    for track_idx, (column, _, css_class, label) in enumerate(tracks):
        track_top = budget_top + title_height + track_idx * (track_height + track_gap)
        track_bottom = track_top + track_height
        parts.append(
            f'<text class="budget-track-label" x="20" y="{track_top + track_height * 0.62:.2f}">{label}</text>'
        )
        parts.append(
            f'<line class="budget-zero-line" x1="{left_margin}" y1="{track_bottom:.2f}" x2="{left_margin + timeline_width}" y2="{track_bottom:.2f}" />'
        )
        parts.append(
            f'<line class="budget-max-line" x1="{left_margin}" y1="{track_top:.2f}" x2="{left_margin + timeline_width}" y2="{track_top:.2f}" />'
        )

        points = []
        for sample in budget_samples:
            value = sample.get(sample_keys[column], 0.0)
            y = track_bottom - (value / max_budget) * track_height
            points.append((sample["x"], y))

        if not points:
            continue

        area_points = [f"M {points[0][0]:.2f} {track_bottom:.2f}"]
        area_points.extend(f"L {x:.2f} {y:.2f}" for x, y in points)
        area_points.append(f"L {points[-1][0]:.2f} {track_bottom:.2f}")
        area_points.append("Z")
        line_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

        parts.append(f'<path class="{css_class} budget-area" d="{" ".join(area_points)}" />')
        parts.append(f'<polyline class="{css_class} budget-line" points="{line_points}" />')

    return "\n".join(parts), budget_samples


def _format_watts(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} MW"
    if value >= 1_000:
        return f"{value / 1_000:.1f} kW"
    return f"{value:.0f} W"


def _job_event_lanes_svg(
    lanes,
    lane_y,
    lane_height,
    left_margin,
    svg_width,
    html_escape,
):
    parts = []
    for idx, lane in enumerate(lanes):
        y = lane_y[lane["key"]]
        row_class = "lane-even" if idx % 2 == 0 else "lane-odd"
        label = f'{lane["cluster"]} / {lane["queue"]}'
        parts.append(
            f'<rect class="{row_class}" x="0" y="{y}" width="{svg_width}" height="{lane_height}" />'
        )
        parts.append(
            f'<text class="lane-label" x="18" y="{y + 27}">{html_escape(label)}</text>'
        )
        parts.append(
            f'<text class="lane-count" x="18" y="{y + 49}">{lane["jobs"]} jobs</text>'
        )
        parts.append(
            f'<line class="lane-separator" x1="{left_margin}" y1="{y + lane_height}" x2="{svg_width - 40}" y2="{y + lane_height}" />'
        )
    return "\n".join(parts)


def _job_event_circle_svg(event, html_escape):
    title = (
        f'{event["job_key"]} {event["type"]}\\n'
        f'{event["time"]}\\n'
        f'queue={event["queue"]}; priority={event["priority"]}; nodes={event["nodes"]}'
    )
    return (
        f'<circle id="{event["dom_id"]}" class="event-dot event-{event["type"]}" '
        f'data-event-id="{html_escape(event["id"])}" data-job-key="{html_escape(event["job_key"])}" '
        f'cx="{event["x"]}" cy="{event["y"]}" r="3.6">'
        f'<title>{html_escape(title)}</title></circle>'
    )


def _job_event_timeline_css():
    return """
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --ink: #20242c;
  --muted: #667085;
  --line: #d8dde7;
  --submitted: #2563eb;
  --started: #7c3aed;
  --finished: #111827;
  --renewable: #2ecc71;
  --battery: #f1c40f;
  --grid: #e74c3c;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: center;
  padding: 16px 20px;
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid var(--line);
}
h1 { margin: 0 0 4px; font-size: 20px; }
p { margin: 0; color: var(--muted); font-size: 13px; }
main {
  display: grid;
  grid-template-columns: 320px 1fr;
  min-height: calc(100vh - 70px);
}
#details {
  position: sticky;
  top: 70px;
  align-self: start;
  height: calc(100vh - 70px);
  overflow: auto;
  padding: 16px;
  background: var(--panel);
  border-right: 1px solid var(--line);
}
#details h2 { margin: 0 0 8px; font-size: 18px; }
#eventDetails {
  margin-top: 16px;
  display: grid;
  gap: 8px;
  font-size: 13px;
}
.detail-row {
  display: grid;
  grid-template-columns: 92px 1fr;
  gap: 8px;
  padding-bottom: 6px;
  border-bottom: 1px solid #eef1f6;
}
.detail-row b { color: var(--muted); font-weight: 600; }
.buttons, .search {
  display: flex;
  gap: 8px;
  margin-top: 14px;
  flex-wrap: wrap;
}
input {
  min-width: 0;
  flex: 1;
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0 10px;
}
button {
  height: 34px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
  border-radius: 6px;
  padding: 0 10px;
  cursor: pointer;
}
button:disabled { cursor: not-allowed; opacity: 0.45; }
.legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px 14px;
  font-size: 12px;
  color: var(--muted);
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i {
  width: 11px;
  height: 11px;
  border-radius: 999px;
  display: inline-block;
}
.legend .submitted { background: var(--submitted); }
.legend .started { background: var(--started); }
.legend .finished { background: var(--finished); }
.legend .wait-link {
  width: 20px;
  height: 0;
  border-top: 2px dashed var(--submitted);
  border-radius: 0;
  background: transparent;
}
.legend .run-link {
  width: 20px;
  height: 0;
  border-top: 2px solid #667085;
  border-radius: 0;
  background: transparent;
}
.legend .renewable { background: var(--renewable); }
.legend .battery { background: var(--battery); }
.legend .grid { background: var(--grid); }
.legend .compute-power {
  width: 20px;
  height: 0;
  border-top: 2px solid #111827;
  border-radius: 0;
  background: transparent;
}
.timeline-shell { min-width: 0; }
.timeline-scroll {
  width: 100%;
  height: calc(100vh - 70px);
  overflow: auto;
  background: #fff;
}
.canvas-bg { fill: #fff; }
.budget-frame {
  fill: rgba(255, 255, 255, 0.55);
  stroke: var(--line);
}
.budget-title {
  font-size: 13px;
  fill: var(--ink);
  font-weight: 700;
}
.budget-label {
  font-size: 11px;
  fill: var(--muted);
}
.budget-track-label {
  font-size: 12px;
  fill: var(--ink);
  font-weight: 650;
}
.budget-zero-line {
  stroke: #d0d5dd;
  stroke-width: 1;
}
.budget-max-line {
  stroke: #eaecf0;
  stroke-width: 1;
}
.budget-area {
  stroke: none;
  opacity: 0.14;
  pointer-events: none;
}
.budget-line {
  fill: none !important;
  stroke-width: 1.8;
  opacity: 0.95;
  pointer-events: none;
}
.budget-area.budget-renewable { fill: var(--renewable); }
.budget-line.budget-renewable { stroke: var(--renewable); }
.budget-area.budget-battery { fill: var(--battery); }
.budget-line.budget-battery { stroke: #d6a600; }
.budget-area.budget-grid { fill: var(--grid); }
.budget-line.budget-grid { stroke: var(--grid); }
.budget-area.compute-power { fill: #111827; opacity: 0.08; }
.budget-line.compute-power { stroke: #111827; }
.budget-cursor {
  stroke: #111827;
  stroke-width: 1.1;
  stroke-dasharray: 4 4;
  opacity: 0.72;
  pointer-events: none;
}
.budget-tooltip {
  pointer-events: none;
}
.budget-tooltip rect {
  fill: rgba(255, 255, 255, 0.96);
  stroke: #98a2b3;
  filter: drop-shadow(0 4px 10px rgba(15, 23, 42, 0.18));
}
.budget-tooltip text {
  font-size: 12px;
  fill: var(--ink);
}
.axis-line { stroke: #98a2b3; stroke-width: 1; }
.tick-line {
  stroke: #d0d5dd;
  stroke-width: 1;
  opacity: 0.55;
}
.tick-label {
  font-size: 11px;
  fill: var(--muted);
}
.lane-even { fill: rgba(248, 250, 252, 0.88); }
.lane-odd { fill: rgba(241, 245, 249, 0.88); }
.lane-label {
  font-size: 13px;
  fill: var(--ink);
  font-weight: 700;
}
.lane-count {
  font-size: 11px;
  fill: var(--muted);
}
.lane-separator {
  stroke: #e5e7eb;
  stroke-width: 1;
}
.job-link {
  stroke: #667085;
  stroke-width: 1.1;
  opacity: 0.16;
  pointer-events: none;
}
.link-wait {
  stroke: var(--submitted);
  stroke-dasharray: 4 4;
  opacity: 0.24;
}
.link-run {
  stroke: #667085;
  opacity: 0.12;
}
.event-dot {
  stroke: #fff;
  stroke-width: 0.9;
  cursor: pointer;
  opacity: 0.82;
  transition: opacity 120ms ease, stroke-width 120ms ease;
}
.event-dot:hover {
  opacity: 1;
  stroke: #111827;
  stroke-width: 1.4;
}
.event-submitted { fill: var(--submitted); }
.event-started { fill: var(--started); }
.event-finished { fill: var(--finished); }
body.has-selection .event-dot { opacity: 0.08; }
body.has-selection .job-link { opacity: 0.025; }
body.has-selection .event-dot.selected-job {
  opacity: 1;
  stroke: #111827;
  stroke-width: 1.8;
}
body.has-selection .job-link.selected-job {
  opacity: 0.75;
  stroke: #111827;
  stroke-width: 2;
}
body.has-selection .event-dot.selected-event {
  stroke: #f97316;
  stroke-width: 2.4;
}
"""


def _job_event_timeline_js():
    return """
const timelineData = JSON.parse(document.getElementById("timeline-data").textContent);
const eventById = timelineData.eventById;
const budgetSamples = timelineData.budgetSamples || [];
const detailsTitle = document.querySelector("#details h2");
const detailsBody = document.getElementById("eventDetails");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const clearBtn = document.getElementById("clearBtn");
const searchBtn = document.getElementById("searchBtn");
const searchInput = document.getElementById("jobSearch");
const scrollBox = document.getElementById("timelineScroll");
const svg = document.getElementById("timelineSvg");
const budgetCursor = document.getElementById("budgetCursor");
const budgetTooltip = document.getElementById("budgetTooltip");
const budgetTooltipTime = document.getElementById("budgetTooltipTime");
const budgetTooltipRenewable = document.getElementById("budgetTooltipRenewable");
const budgetTooltipBattery = document.getElementById("budgetTooltipBattery");
const budgetTooltipGrid = document.getElementById("budgetTooltipGrid");
const budgetTooltipPower = document.getElementById("budgetTooltipPower");
let selectedEventId = null;

document.querySelectorAll(".event-dot").forEach((node) => {
  node.addEventListener("click", (event) => {
    event.stopPropagation();
    selectEvent(node.dataset.eventId, true);
  });
});

clearBtn.addEventListener("click", clearSelection);
prevBtn.addEventListener("click", () => {
  const event = eventById[selectedEventId];
  if (event && event.prev_id) selectEvent(event.prev_id, true);
});
nextBtn.addEventListener("click", () => {
  const event = eventById[selectedEventId];
  if (event && event.next_id) selectEvent(event.next_id, true);
});
searchBtn.addEventListener("click", findJob);
searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") findJob();
});
svg.addEventListener("mousemove", updateBudgetCursor);
svg.addEventListener("mouseleave", hideBudgetCursor);

function selectEvent(eventId, center) {
  const event = eventById[eventId];
  if (!event) return;
  selectedEventId = eventId;
  document.body.classList.add("has-selection");

  document.querySelectorAll("[data-job-key]").forEach((node) => {
    node.classList.toggle("selected-job", node.dataset.jobKey === event.job_key);
  });
  document.querySelectorAll(".event-dot").forEach((node) => {
    const selected = node.dataset.eventId === eventId;
    node.classList.toggle("selected-event", selected);
    node.setAttribute("r", selected ? "7.2" : "3.6");
  });

  detailsTitle.textContent = `${event.job_key} ${event.type}`;
  detailsBody.innerHTML = [
    detailRow("time", event.time),
    detailRow("cluster", event.cluster),
    detailRow("queue", event.queue),
    detailRow("status", event.status),
    detailRow("priority", event.priority),
    detailRow("nodes", event.nodes),
    detailRow("requested h", event.requested_walltime_h),
    detailRow("wait h", event.wait_h),
    detailRow("run h", event.run_h),
    detailRow("budget near event", formatBudgetSummary(nearestBudget(Number(event.x))))
  ].join("");
  prevBtn.disabled = !event.prev_id;
  nextBtn.disabled = !event.next_id;

  if (center) centerEvent(event);
}

function detailRow(label, value) {
  return `<div class="detail-row"><b>${escapeHtml(label)}</b><span>${escapeHtml(String(value ?? ""))}</span></div>`;
}

function centerEvent(event) {
  const x = Number(event.x);
  const y = Number(event.y);
  scrollBox.scrollTo({
    left: Math.max(0, x - scrollBox.clientWidth * 0.48),
    top: Math.max(0, y - scrollBox.clientHeight * 0.45),
    behavior: "smooth"
  });
}

function updateBudgetCursor(mouseEvent) {
  if (!budgetSamples.length) return;
  const point = svgPoint(mouseEvent);
  const sample = nearestBudget(point.x);
  if (!sample) return;

  budgetCursor.setAttribute("x1", sample.x);
  budgetCursor.setAttribute("x2", sample.x);
  budgetCursor.setAttribute("visibility", "visible");

  const tooltipX = Math.min(
    Number(svg.getAttribute("width")) - 205,
    Math.max(12, point.x + 14)
  );
  const tooltipY = Math.max(12, point.y - 128);
  budgetTooltip.setAttribute("transform", `translate(${tooltipX}, ${tooltipY})`);
  budgetTooltip.setAttribute("visibility", "visible");
  budgetTooltipTime.textContent = sample.time;
  budgetTooltipRenewable.textContent = `renewable: ${formatW(sample.renewable_w)}`;
  budgetTooltipBattery.textContent = `battery: ${formatW(sample.battery_w)}`;
  budgetTooltipGrid.textContent = `grid: ${formatW(sample.grid_w)} | total: ${formatW(sample.total_w)}`;
  budgetTooltipPower.textContent = `power: ${formatW(sample.compute_power_w)} | headroom: ${formatW(sample.headroom_w)}`;
}

function hideBudgetCursor() {
  budgetCursor.setAttribute("visibility", "hidden");
  budgetTooltip.setAttribute("visibility", "hidden");
}

function svgPoint(mouseEvent) {
  const point = svg.createSVGPoint();
  point.x = mouseEvent.clientX;
  point.y = mouseEvent.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function nearestBudget(x) {
  if (!budgetSamples.length || !Number.isFinite(x)) return null;
  let best = budgetSamples[0];
  let bestDistance = Math.abs(Number(best.x) - x);
  for (let i = 1; i < budgetSamples.length; i += 1) {
    const distance = Math.abs(Number(budgetSamples[i].x) - x);
    if (distance < bestDistance) {
      best = budgetSamples[i];
      bestDistance = distance;
    }
  }
  return best;
}

function formatBudgetSummary(sample) {
  if (!sample) return "";
  return `budget ${formatW(sample.total_w)}, power ${formatW(sample.compute_power_w)}, headroom ${formatW(sample.headroom_w)}`;
}

function formatW(value) {
  const watts = Number(value || 0);
  if (watts >= 1000000) return `${(watts / 1000000).toFixed(2)} MW`;
  if (watts >= 1000) return `${(watts / 1000).toFixed(1)} kW`;
  return `${watts.toFixed(0)} W`;
}

function clearSelection() {
  selectedEventId = null;
  document.body.classList.remove("has-selection");
  document.querySelectorAll(".selected-job, .selected-event").forEach((node) => {
    node.classList.remove("selected-job", "selected-event");
  });
  document.querySelectorAll(".event-dot").forEach((node) => node.setAttribute("r", "3.6"));
  detailsTitle.textContent = "No event selected";
  detailsBody.innerHTML = "";
  prevBtn.disabled = true;
  nextBtn.disabled = true;
}

function findJob() {
  const query = searchInput.value.trim().toLowerCase();
  if (!query) return;
  const event = timelineData.events.find((candidate) => {
    return candidate.job_key.toLowerCase() === query
      || candidate.job_key.toLowerCase().endsWith(`:${query}`)
      || candidate.job_id.toLowerCase() === query;
  });
  if (event) selectEvent(event.id, true);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
"""


def _load_scheduler_debug_events(results_dir: str):
    import glob
    import os
    import pandas as pd

    wanted_columns = {
        "Time",
        "Cluster",
        "Event",
        "Queue_Pending_Counts",
        "Queue_Oldest_Wait_s",
        "Queue_Min_Deadline_s",
        "Queue_Missed_Deadlines",
        "Queue_Name",
    }
    frames = []
    for csv_file in sorted(glob.glob(os.path.join(results_dir, "scheduler_debug_events_*.csv"))):
        available_columns = pd.read_csv(csv_file, nrows=0).columns
        usecols = [column for column in available_columns if column in wanted_columns]
        df = pd.read_csv(csv_file, usecols=usecols, low_memory=False)
        if df.empty:
            continue
        if "Cluster" not in df.columns:
            df["Cluster"] = os.path.basename(csv_file)[len("scheduler_debug_events_"):-4]
        df["Cluster"] = df["Cluster"].astype(str)
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_jobs_metrics(results_dir: str):
    import glob
    import os
    import pandas as pd

    wanted_columns = {
        "Queue_Name",
        "Queue_Max_Wait_s",
        "Queue_Wait_Time_s",
    }
    frames = []
    for csv_file in sorted(glob.glob(os.path.join(results_dir, "jobs_metrics_*.csv"))):
        available_columns = pd.read_csv(csv_file, nrows=0).columns
        usecols = [column for column in available_columns if column in wanted_columns]
        df = pd.read_csv(csv_file, usecols=usecols, low_memory=False)
        if df.empty:
            continue
        df["Cluster"] = os.path.basename(csv_file)[len("jobs_metrics_"):-4]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_scheduler_arrivals(results_dir: str, schedulers=None):
    import os
    import pickle
    import pandas as pd

    if schedulers is None:
        schedulers_path = os.path.join(results_dir, "schedulers.pkl")
        if os.path.exists(schedulers_path):
            with open(schedulers_path, "rb") as f:
                schedulers = pickle.load(f)

    if not schedulers:
        return pd.DataFrame(columns=["Time", "Cluster", "Queue"])

    rows = []
    for scheduler in schedulers:
        cluster = scheduler.cluster_type
        for job in scheduler.job_queue.jobs:
            rows.append(
                {
                    "Time": job.arrival_time,
                    "Cluster": cluster,
                    "Queue": _job_queue_label(scheduler, job),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Time", "Cluster", "Queue"])

    df = pd.DataFrame(rows)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df.dropna(subset=["Time"])


def _parse_queue_map(raw_value) -> dict[str, float]:
    import pandas as pd

    if raw_value is None or pd.isna(raw_value):
        return {}

    parsed = {}
    for part in str(raw_value).split(";"):
        if "=" not in part:
            continue
        queue_name, value = part.split("=", 1)
        queue_name = queue_name.strip()
        value = value.strip()
        if not queue_name or not value:
            continue
        try:
            parsed[queue_name] = float(value)
        except ValueError:
            continue
    return parsed


def _queue_names_from_debug_and_metrics(debug_df, metrics_df, arrivals_df=None) -> list[str]:
    names = set()

    for column in (
        "Queue_Pending_Counts",
        "Queue_Oldest_Wait_s",
        "Queue_Min_Deadline_s",
        "Queue_Missed_Deadlines",
    ):
        if column not in debug_df.columns:
            continue
        for value in debug_df[column].dropna():
            names.update(_parse_queue_map(value))

    if "Queue_Name" in debug_df.columns:
        names.update(
            str(value)
            for value in debug_df["Queue_Name"].dropna()
            if str(value).strip()
        )
    if not metrics_df.empty and "Queue_Name" in metrics_df.columns:
        names.update(
            str(value)
            for value in metrics_df["Queue_Name"].dropna()
            if str(value).strip()
        )
    if arrivals_df is not None and not arrivals_df.empty and "Queue" in arrivals_df.columns:
        names.update(
            str(value)
            for value in arrivals_df["Queue"].dropna()
            if str(value).strip()
        )

    return _ordered_queue_names(names)


def _ordered_queue_names(queue_names) -> list[str]:
    preferred = ["short", "medium", "long", "unclassified"]
    present = {str(name) for name in queue_names if str(name)}
    ordered = [name for name in preferred if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _queue_color(queue_name: str) -> str:
    colors = {
        "short": "#2ecc71",
        "medium": "#f1c40f",
        "long": "#e74c3c",
        "unclassified": "#95a5a6",
    }
    fallback = ["#3498db", "#9b59b6", "#1abc9c", "#f39c12", "#34495e"]
    if queue_name in colors:
        return colors[queue_name]
    stable_index = sum(ord(char) for char in queue_name) % len(fallback)
    return fallback[stable_index]


def _expand_queue_map_rows(df, source_column: str, value_column: str):
    import pandas as pd

    if df.empty or source_column not in df.columns:
        return pd.DataFrame(columns=["Time", "Queue", value_column])

    rows = []
    for _, row in df.iterrows():
        for queue_name, value in _parse_queue_map(row.get(source_column)).items():
            rows.append(
                {
                    "Time": row["Time"],
                    "Queue": queue_name,
                    value_column: value,
                }
            )
    return pd.DataFrame(rows, columns=["Time", "Queue", value_column])


def _queue_time_pivot(rows_df, times, queue_names, value_column: str):
    import pandas as pd

    index = pd.DatetimeIndex(times).drop_duplicates().sort_values()
    if rows_df.empty:
        return pd.DataFrame(0.0, index=index, columns=queue_names)

    pivot = rows_df.pivot_table(
        index="Time",
        columns="Queue",
        values=value_column,
        aggfunc="last",
    )
    pivot = pivot.reindex(index=index, columns=queue_names).fillna(0.0)
    return pivot.sort_index()


def _format_queue_time_axis(ax) -> None:
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.grid(True, alpha=0.25)


def _plot_arrival_queue_cumulative(ax, cluster_arrivals, cluster_steps, queue_names, title: str) -> None:
    import pandas as pd

    if cluster_arrivals.empty:
        ax.text(
            0.5,
            0.5,
            "No arrival data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title(title, fontweight="bold")
        _format_queue_time_axis(ax)
        return

    if cluster_steps.empty:
        times = cluster_arrivals["Time"]
    else:
        times = pd.concat([cluster_steps["Time"], cluster_arrivals["Time"]])
    times = times.drop_duplicates().sort_values()

    grouped = (
        cluster_arrivals.groupby(["Time", "Queue"])
        .size()
        .reset_index(name="arrivals")
    )
    pivot = _queue_time_pivot(grouped, times, queue_names, "arrivals").cumsum()

    plotted = False
    for queue in queue_names:
        if pivot[queue].max() <= 0:
            continue
        ax.plot(
            pivot.index,
            pivot[queue],
            label=queue,
            color=_queue_color(queue),
            linewidth=1.8,
        )
        plotted = True

    if plotted:
        ax.legend(fontsize=7, loc="upper left")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Cumulative submissions")
    _format_queue_time_axis(ax)


def _plot_pending_queue_stack(ax, cluster_steps, queue_names, title: str) -> None:
    import numpy as np

    if cluster_steps.empty:
        ax.text(0.5, 0.5, "No step data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        return

    counts = _expand_queue_map_rows(cluster_steps, "Queue_Pending_Counts", "pending")
    pivot = _queue_time_pivot(counts, cluster_steps["Time"], queue_names, "pending")
    visible_queues = [queue for queue in queue_names if pivot[queue].sum() > 0]

    if not visible_queues:
        ax.plot(pivot.index, np.zeros(len(pivot.index)), color="#95a5a6", linewidth=1.5)
        ax.set_ylim(bottom=0)
    else:
        ax.stackplot(
            pivot.index,
            [pivot[queue].to_numpy() for queue in visible_queues],
            labels=visible_queues,
            colors=[_queue_color(queue) for queue in visible_queues],
            alpha=0.85,
            linewidth=0.4,
            edgecolor="white",
        )
        ax.legend(fontsize=7, loc="upper left")

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Pending jobs")
    _format_queue_time_axis(ax)


def _plot_started_queue_cumulative(ax, cluster_events, cluster_steps, queue_names, title: str) -> None:
    import pandas as pd

    if cluster_events.empty:
        ax.text(0.5, 0.5, "No event data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        return

    if cluster_steps.empty:
        times = cluster_events["Time"]
    else:
        times = cluster_steps["Time"]

    if "Queue_Name" not in cluster_events.columns:
        starts = cluster_events.iloc[0:0].copy()
    else:
        starts = cluster_events[
            (cluster_events["Event"] == "job_started")
            & cluster_events["Queue_Name"].notna()
            & (cluster_events["Queue_Name"].astype(str).str.strip() != "")
        ]

    if starts.empty:
        ax.text(0.5, 0.5, "No job_started events", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        _format_queue_time_axis(ax)
        return

    grouped = (
        starts.groupby(["Time", "Queue_Name"])
        .size()
        .reset_index(name="started")
        .rename(columns={"Queue_Name": "Queue"})
    )
    times = pd.concat([times, starts["Time"]]).drop_duplicates().sort_values()
    pivot = _queue_time_pivot(grouped, times, queue_names, "started").cumsum()

    plotted = False
    for queue in queue_names:
        if pivot[queue].max() <= 0:
            continue
        ax.plot(
            pivot.index,
            pivot[queue],
            label=queue,
            color=_queue_color(queue),
            linewidth=1.8,
        )
        plotted = True

    if plotted:
        ax.legend(fontsize=7, loc="upper left")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Cumulative starts")
    _format_queue_time_axis(ax)


def _plot_queue_wait_pressure(ax, cluster_steps, cluster_metrics, queue_names, title: str) -> None:
    oldest_wait = _expand_queue_map_rows(
        cluster_steps,
        "Queue_Oldest_Wait_s",
        "oldest_wait_s",
    )

    if not oldest_wait.empty:
        pivot = _queue_time_pivot(
            oldest_wait,
            cluster_steps["Time"],
            queue_names,
            "oldest_wait_s",
        ) / 3600.0

        plotted = False
        for queue in queue_names:
            if pivot[queue].max() <= 0:
                continue
            ax.plot(
                pivot.index,
                pivot[queue],
                label=f"{queue} oldest wait",
                color=_queue_color(queue),
                linewidth=1.6,
            )
            plotted = True

        _draw_queue_sla_lines(ax, cluster_metrics, queue_names)
        if plotted:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Oldest pending wait (h)")
        _format_queue_time_axis(ax)
        return

    _plot_queue_wait_boxplot(ax, cluster_metrics, queue_names, title)


def _draw_queue_sla_lines(ax, cluster_metrics, queue_names) -> None:
    if cluster_metrics.empty or "Queue_Max_Wait_s" not in cluster_metrics.columns:
        return

    for queue in queue_names:
        queue_rows = cluster_metrics[cluster_metrics["Queue_Name"] == queue]
        if queue_rows.empty:
            continue
        max_wait_s = queue_rows["Queue_Max_Wait_s"].fillna(0).max()
        if max_wait_s <= 0:
            continue
        ax.axhline(
            max_wait_s / 3600.0,
            color=_queue_color(queue),
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
        )


def _plot_queue_wait_boxplot(ax, cluster_metrics, queue_names, title: str) -> None:
    if (
        cluster_metrics.empty
        or "Queue_Name" not in cluster_metrics.columns
        or "Queue_Wait_Time_s" not in cluster_metrics.columns
    ):
        ax.text(
            0.5,
            0.5,
            "Run a new simulation to capture wait pressure",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title(title, fontweight="bold")
        return

    data = []
    labels = []
    colors = []
    for queue in queue_names:
        waits = cluster_metrics.loc[
            cluster_metrics["Queue_Name"] == queue,
            "Queue_Wait_Time_s",
        ].dropna()
        if waits.empty:
            continue
        data.append(waits / 3600.0)
        labels.append(queue)
        colors.append(_queue_color(queue))

    if not data:
        ax.text(0.5, 0.5, "No queue wait metrics", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontweight="bold")
        return

    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"{title} final distribution", fontweight="bold")
    ax.set_ylabel("Wait time (h)")
    ax.grid(True, axis="y", alpha=0.25)


def plot_result_df(df, output_file: str = "simulation_results.png", min_soc: float = 0.3):
    """
    Original 4-panel energy plot.
    Shows: Actor Power, Delta Power, Battery SoC, Grid Power.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # Panel 1: Actor Power (aggregated by cluster type)
    ax1 = axes[0]
    
    # Find power_meters columns and aggregate by cluster prefix
    power_cols = [col for col in df.columns if 'power_meters.' in col]
    
    # Pre-aggregated per-cluster power columns (from custom monitor fns)
    cluster_power_cols = [col for col in df.columns if col.endswith('_power')]
    
    if power_cols:
        # Extract unique cluster prefixes (e.g., "arm_a64fx", "x86_epyc", "gpu_a100")
        import re
        prefixes = {}
        for col in power_cols:
            # Match pattern like "power_meters.arm_a64fx-123"
            match = re.search(r'power_meters\.([a-zA-Z0-9_]+)-\d+', col)
            if match:
                prefix = match.group(1)
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(col)
        
        cluster_palette = {
            'arm_a64fx': '#3498db',
            'x86_epyc': '#e74c3c',
            'gpu_a100': '#2ecc71',
        }
        fallback_colors = ['#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
        
        total_power = None
        for i, (prefix, cols) in enumerate(sorted(prefixes.items())):
            cluster_sum = df[cols].fillna(0).sum(axis=1).abs()
            color = cluster_palette.get(prefix, fallback_colors[i % len(fallback_colors)])
            ax1.plot(df.index, cluster_sum / 1000, label=f'{prefix} ({len(cols)} nodes)',
                     color=color, linewidth=1.5, alpha=0.85)
            if total_power is None:
                total_power = cluster_sum.copy()
            else:
                total_power += cluster_sum
        
        if total_power is not None:
            ax1.plot(df.index, total_power / 1000, label='Total HPC',
                     color='black', linewidth=2, linestyle='--', alpha=0.6)
        
        # Add renewable generation if available.
        renewable_cols = [col for col in df.columns if 'Generator' in col and '.p' in col]
        if renewable_cols:
            renewable_power = df[renewable_cols].fillna(0).abs().sum(axis=1) / 1000
            ax1.fill_between(df.index, renewable_power, alpha=0.15, color='#f1c40f', label='Renewable Generation')
            ax1.plot(df.index, renewable_power, color='#f1c40f', linewidth=1, alpha=0.6)
        
        ax1.set_ylabel('Power (kW)')
    elif cluster_power_cols:
        # Use pre-aggregated per-cluster power columns (detailed_state=False)
        cluster_palette = {
            'arm_a64fx': '#3498db',
            'x86_epyc': '#e74c3c',
            'gpu_a100': '#2ecc71',
        }
        fallback_colors = ['#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
        
        total_power = None
        for i, col in enumerate(sorted(cluster_power_cols)):
            prefix = col.replace('_power', '')
            power_kw = df[col].ffill().fillna(0) / 1000
            color = cluster_palette.get(prefix, fallback_colors[i % len(fallback_colors)])
            ax1.plot(df.index, power_kw, label=prefix,
                     color=color, linewidth=1.5, alpha=0.85)
            if total_power is None:
                total_power = power_kw.copy()
            else:
                total_power += power_kw
        
        if total_power is not None:
            ax1.plot(df.index, total_power, label='Total HPC',
                     color='black', linewidth=2, linestyle='--', alpha=0.6)
        
        renewable_cols = [col for col in df.columns if 'Generator' in col and '.p' in col]
        if renewable_cols:
            renewable_power = df[renewable_cols].fillna(0).abs().sum(axis=1) / 1000
            ax1.fill_between(df.index, renewable_power, alpha=0.15, color='#f1c40f', label='Renewable Generation')
            ax1.plot(df.index, renewable_power, color='#f1c40f', linewidth=1, alpha=0.6)
        
        ax1.set_ylabel('Power (kW)')
    elif 'p_delta' in df.columns:
        ax1.plot(df.index, -df['p_delta'].fillna(0) / 1000, label='Total Consumption', color='#4169E1', linewidth=1.5)
        ax1.set_ylabel('Power (kW)')
    
    if 'battery_power_w' in df.columns:
        batt_power = -df['battery_power_w'].fillna(0) / 1000
        ax1.plot(df.index, batt_power, color='#e67e22', linewidth=1.5, label='Battery Output')
        
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_title('Actor Power (Aggregated by Cluster)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Delta Power
    ax2 = axes[1]
    if 'p_delta' in df.columns:
        p_delta = df['p_delta'].fillna(0) / 1000
        ax2.fill_between(df.index, p_delta, 0, where=(p_delta >= 0), color='#90EE90', alpha=0.7, label='Surplus')
        ax2.fill_between(df.index, p_delta, 0, where=(p_delta < 0), color='#FF6B6B', alpha=0.7, label='Deficit')
        ax2.plot(df.index, p_delta, color='black', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Power (kW)')
    ax2.set_title('Delta Power', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Battery SoC
    ax3 = axes[2]
    soc_col = None
    for col in df.columns:
        if 'soc' in col.lower() or 'state_of_charge' in col.lower():
            soc_col = col
            break
    if soc_col:
        soc = df[soc_col].ffill() * 100
        ax3.plot(df.index, soc, color='#2E8B57', linewidth=2, label='SoC')
        ax3.axhline(y=min_soc * 100, color='gray', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 105)
    ax3.set_ylabel('SoC (%)')
    ax3.set_title('Battery State of Charge', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Grid Power
    ax4 = axes[3]
    if 'p_delta' in df.columns:
        grid_power = -df['p_delta'].fillna(0) / 1000
        ax4.fill_between(df.index, grid_power, 0, where=(grid_power >= 0), color='#FF6B6B', alpha=0.7, label='Import (from grid)')
        ax4.fill_between(df.index, grid_power, 0, where=(grid_power < 0), color='#90EE90', alpha=0.7, label='Export (to grid)')
        ax4.plot(df.index, grid_power, color='black', linewidth=0.5, alpha=0.5)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Power (kW)')
    ax4.set_title('Grid Power', fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Energy results saved to: {os.path.basename(output_file)}")
    plt.close(fig)
    return fig


def plot_frequency_timeline(
    schedulers,
    output_file: str = "frequency_timeline.png",
):
    """
    Plot DVFS frequency over time for the single busiest node of each cluster.

    For every cluster scheduler the node with the most completed jobs is
    selected.  Its ``frequency_history`` (recorded in ``ComputeNode``) is
    drawn as a step plot so abrupt frequency changes are clearly visible.

    Args:
        schedulers: list of ``JobQueueScheduler`` (one per cluster).
        output_file: path for the saved PNG.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    n_clusters = len(schedulers)
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 5 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    fig.suptitle(
        'DVFS Frequency Timeline — Busiest Node per Cluster',
        fontsize=16, fontweight='bold', y=0.98,
    )

    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]
        ct = scheduler.cluster_type
        color = palette[idx % len(palette)]

        # Pick the single node with the most completed jobs
        busiest = max(
            scheduler.rm.nodes.values(),
            key=lambda n: n.total_jobs_executed,
            default=None,
        )

        if busiest is None or busiest.total_jobs_executed == 0 or not busiest.frequency_history:
            ax.text(
                0.5, 0.5, f'No jobs completed on {ct}',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='gray',
            )
            ax.set_title(f'{ct}', fontweight='bold')
            continue

        times = [t for t, _ in busiest.frequency_history]
        freqs = [f if f is not None else 0.0 for _, f in busiest.frequency_history]

        label = f'{busiest.name} ({busiest.total_jobs_executed} jobs)'
        ax.step(times, freqs, where='post', linewidth=1.5,
                color=color, label=label, alpha=0.85)

        # Reference line for base frequency
        base_freq = scheduler.dvfs_policy.base_frequency_ghz
        ax.axhline(y=base_freq, color='gray', linestyle=':', alpha=0.5,
                    label=f'Base freq ({base_freq} GHz)')

        ax.set_ylabel('Frequency (GHz)')
        ax.set_title(f'{ct}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')

    # Time-axis formatting
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for a in axes:
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Frequency timeline saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def _budget_source_it_series(df):
    """Return planned source budgets converted to IT-side watts.

    Source budgets are facility-side values.  Cluster budgets and measured
    cluster power are IT-side values, so every source must cross the same PUE
    boundary before the series are compared or stacked.
    """
    import pandas as pd

    if "budget_pue" in df.columns:
        pue = pd.to_numeric(df["budget_pue"], errors="coerce")
        pue = pue.where(pue > 0).ffill().bfill().fillna(1.0)
    else:
        pue = pd.Series(1.0, index=df.index)

    def source(column: str):
        if column not in df.columns:
            return pd.Series(0.0, index=df.index)
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        return values.clip(lower=0.0) / pue

    return {
        "renewable": source("budget_renewable_w"),
        "battery": source("budget_battery_w"),
        "grid": source("budget_grid_w"),
    }


def plot_budget_distribution(df, schedulers, output_file: str = "budget_distribution.png"):
    """
    3-panel chart showing energy budget distribution over time:
      1. Stacked area — budget per cluster (kW)
      2. Budget vs actual power per cluster (line overlay)
      3. PUE-normalized planned sources vs actual IT power
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    cluster_types = [s.cluster_type for s in schedulers]
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    cluster_colors = {ct: palette[i % len(palette)] for i, ct in enumerate(cluster_types)}

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('Energy Budget Distribution', fontsize=16, fontweight='bold', y=0.98)

    # ====== Panel 1: Stacked area — budget per cluster ======
    ax = axes[0]
    budget_series = {}
    for ct in cluster_types:
        col = f'{ct}_budget'
        if col in df.columns:
            budget_series[ct] = df[col].ffill().fillna(0) / 1000  # W → kW

    if budget_series:
        labels = list(budget_series.keys())
        data = np.array([budget_series[ct].values for ct in labels])
        colors = [cluster_colors[ct] for ct in labels]
        ax.stackplot(df.index, data, labels=labels, colors=colors, alpha=0.7)
        # Total budget line
        total = sum(budget_series.values())
        ax.plot(df.index, total, color='black', linewidth=2, linestyle='--', alpha=0.7, label='Total budget')

    ax.set_ylabel('Budget (kW)')
    ax.set_title('Budget Allocation per Cluster (stacked)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ====== Panel 2: Budget vs actual power (per cluster, overlaid) ======
    ax = axes[1]
    for ct in cluster_types:
        color = cluster_colors[ct]
        budget_col = f'{ct}_budget'
        power_col = f'{ct}_power'

        if budget_col in df.columns:
            budget_kw = df[budget_col].ffill().fillna(0) / 1000
            ax.plot(df.index, budget_kw, color=color, linewidth=1.5, linestyle='--',
                    alpha=0.7, label=f'{ct} budget')

        if power_col in df.columns:
            power_kw = df[power_col].ffill().fillna(0) / 1000
            ax.fill_between(df.index, power_kw, alpha=0.2, color=color)
            ax.plot(df.index, power_kw, color=color, linewidth=1, alpha=0.9,
                    label=f'{ct} power')

    ax.set_ylabel('Power (kW)')
    ax.set_title('Budget Ceiling vs Actual Power per Cluster', fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ====== Panel 3: PUE-normalized planned sources vs actual IT power ======
    ax = axes[2]
    total_budget_kw = None
    if budget_series:
        total_budget_kw = sum(budget_series.values())

    sources = _budget_source_it_series(df)
    renewable_kw = sources["renewable"] / 1000.0
    battery_kw = sources["battery"] / 1000.0
    grid_kw = sources["grid"] / 1000.0
    source_total_kw = renewable_kw + battery_kw + grid_kw

    if any(column in df.columns for column in (
        "budget_renewable_w", "budget_battery_w", "budget_grid_w",
    )):
        ax.fill_between(
            df.index, 0, renewable_kw, alpha=0.55,
            color="#f1c40f", label="Renewable budget (IT equivalent)",
        )
        ax.fill_between(
            df.index, renewable_kw, renewable_kw + battery_kw, alpha=0.55,
            color="#2ecc71", label="Battery budget (IT equivalent)",
        )
        ax.fill_between(
            df.index, renewable_kw + battery_kw, source_total_kw, alpha=0.55,
            color="#e74c3c", label="Grid budget (IT equivalent)",
        )
        ax.plot(
            df.index, source_total_kw, color="black", linewidth=1.6,
            linestyle="--", alpha=0.8, label="Planned IT budget",
        )

        actual_power_cols = [
            f"{ct}_power" for ct in cluster_types
            if f"{ct}_power" in df.columns
        ]
        if actual_power_cols:
            actual_it_kw = (
                df[actual_power_cols].ffill().fillna(0.0).sum(axis=1) / 1000.0
            )
            ax.plot(
                df.index, actual_it_kw, color="#0072B2", linewidth=1.4,
                label="Actual IT power",
            )
    elif total_budget_kw is not None:
        ax.plot(
            df.index, total_budget_kw, color="black", linewidth=2,
            label="Total IT budget",
        )
    else:
        ax.text(0.5, 0.5, 'No budget data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.set_ylabel('Power (kW)')
    ax.set_title(
        'Planned IT Budget Sources vs Actual IT Power', fontweight='bold',
    )
    ax.set_xlabel('Time')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Time axis formatting
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for a in axes:
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Budget distribution saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def plot_carbon_analysis(
    df,
    schedulers,
    plot_output: str = "carbon_analysis.png",
    csv_output: str = "carbon_metrics.csv",
    monitor_step_size_s: int = 60,
):
    """
    Per-cluster energy and CO₂ analysis.

    Computes:
    - Cumulative energy consumed per cluster (kWh) — tracked by each scheduler.
    - CO₂ emitted per cluster (g → kg cumulative) — proportionally attributed
      from the system-wide CO₂ based on each cluster's share of total power.
    - Carbon intensity per cluster (gCO₂/kWh).

    Saves a ``csv_output`` with per-step and cumulative values, and a 3-panel
    ``plot_output`` PNG.

    Args:
        df: Monitor CSV loaded as a DataFrame (output of ``monitor.to_csv()``).
        schedulers: List of ``JobQueueScheduler`` instances.
        plot_output: Path for the PNG plot.
        csv_output: Path for the per-cluster metrics CSV.
        monitor_step_size_s: Monitor step size in seconds (default 60).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    cluster_types = [s.cluster_type for s in schedulers]
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    cluster_colors = {ct: palette[i % len(palette)] for i, ct in enumerate(cluster_types)}

    # ── Parse total system CO₂ from monitor (stored as tuple string "(val,)") ──
    co2_series = pd.Series(0.0, index=df.index)
    if 'co2' in df.columns:
        co2_series = pd.to_numeric(
            df['co2'].fillna(0).astype(str).str.strip('(),').str.split(',').str[0],
            errors='coerce',
        ).fillna(0)

    # ── Per-cluster power at each monitor step ──────────────────────────────
    power = {}
    for ct in cluster_types:
        col = f'{ct}_power'
        if col in df.columns:
            power[ct] = df[col].ffill().fillna(0)
        else:
            power[ct] = pd.Series(0.0, index=df.index)

    total_power = sum(power.values())

    # ── Per-step energy per cluster (kWh) ──────────────────────────────────
    step_dt_h = monitor_step_size_s / 3600
    step_energy = {ct: power[ct] / 1000 * step_dt_h for ct in cluster_types}  # kWh per step

    # ── Cumulative energy per cluster — prefer scheduler's own accumulator ──
    # The monitor records scheduler.total_energy_wh / 1000 at each 60 s mark.
    # Use it directly if available, otherwise fall back to step integral.
    cumulative_energy = {}
    for ct in cluster_types:
        energy_col = f'{ct}_energy_kwh'
        if energy_col in df.columns:
            cumulative_energy[ct] = df[energy_col].ffill().fillna(0)
        else:
            cumulative_energy[ct] = step_energy[ct].cumsum()

    # ── Energy source split per cluster ────────────────────────────────────
    # Source budgets are global. Attribute each step's consumed cluster energy
    # by the same global grid/renewable mix used by the budget-source plots.
    def _source_series(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[column], errors='coerce').ffill().fillna(0).clip(lower=0)

    grid_source_w = _source_series('budget_grid_w')
    renewable_source_w = _source_series('budget_renewable_w')
    if 'budget_battery_w' in df.columns:
        # The two-way provenance view groups scheduled battery discharge with
        # renewable energy, matching the renewable-reserve policies.
        renewable_source_w = renewable_source_w + _source_series('budget_battery_w')

    if (grid_source_w + renewable_source_w).sum() <= 0:
        renewable_cols = [c for c in df.columns if 'Generator' in c and '.p' in c]
        if renewable_cols:
            renewable_source_w = df[renewable_cols].apply(
                pd.to_numeric,
                errors='coerce',
            ).fillna(0).abs().sum(axis=1)
            grid_source_w = (total_power - renewable_source_w).clip(lower=0)

    source_total_w = grid_source_w + renewable_source_w
    source_total_safe = source_total_w.where(source_total_w > 0, np.nan)
    grid_fraction = (grid_source_w / source_total_safe).fillna(0)
    renewable_fraction = (renewable_source_w / source_total_safe).fillna(0)

    grid_step_energy = {}
    renewable_step_energy = {}
    cumulative_grid_energy = {}
    cumulative_renewable_energy = {}
    for ct in cluster_types:
        source_step_energy = cumulative_energy[ct].diff().fillna(cumulative_energy[ct]).clip(lower=0)
        grid_step_energy[ct] = source_step_energy * grid_fraction
        renewable_step_energy[ct] = source_step_energy * renewable_fraction
        cumulative_grid_energy[ct] = grid_step_energy[ct].cumsum()
        cumulative_renewable_energy[ct] = renewable_step_energy[ct].cumsum()

    # ── CO₂ attribution per cluster (proportional to power share) ──────────
    step_co2_g = {}  # gCO₂ per step, per cluster
    for ct in cluster_types:
        fraction = power[ct].where(total_power > 0, 0) / total_power.where(total_power > 0, 1)
        step_co2_g[ct] = co2_series * fraction

    cumulative_co2_kg = {ct: step_co2_g[ct].cumsum() / 1000 for ct in cluster_types}

    # ── Carbon intensity per cluster (gCO₂/kWh) — rolling ──────────────────
    # Avoid div-by-zero: only compute where cumulative energy > 0
    co2_intensity = {}
    for ct in cluster_types:
        cum_e = cumulative_energy[ct].replace(0, np.nan)
        co2_intensity[ct] = (cumulative_co2_kg[ct] * 1000) / cum_e  # gCO₂/kWh

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out = pd.DataFrame(index=df.index)
    out['co2_total_g'] = co2_series
    out['co2_total_kg_cumulative'] = co2_series.cumsum() / 1000
    out['total_energy_kwh'] = sum(step_energy.values()).cumsum()
    for ct in cluster_types:
        out[f'{ct}_power_w'] = power[ct]
        out[f'{ct}_step_energy_kwh'] = step_energy[ct]
        out[f'{ct}_energy_kwh_cumulative'] = cumulative_energy[ct]
        out[f'{ct}_grid_step_energy_kwh'] = grid_step_energy[ct]
        out[f'{ct}_grid_energy_kwh_cumulative'] = cumulative_grid_energy[ct]
        out[f'{ct}_renewable_step_energy_kwh'] = renewable_step_energy[ct]
        out[f'{ct}_renewable_energy_kwh_cumulative'] = cumulative_renewable_energy[ct]
        out[f'{ct}_step_co2_g'] = step_co2_g[ct]
        out[f'{ct}_co2_kg_cumulative'] = cumulative_co2_kg[ct]
        out[f'{ct}_co2_intensity_gco2_per_kwh'] = co2_intensity[ct]
    out.to_csv(csv_output)
    print(f"✅ Carbon metrics saved to: {os.path.basename(csv_output)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    fig.suptitle('Cluster Energy & Carbon Footprint Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Cumulative energy per cluster
    ax = axes[0]
    source_colors = {
        'grid': '#e74c3c',
        'renewable': '#f1c40f',
    }
    source_fill_handles = []
    total_renewable_e = None
    total_grid_e = None
    total_e = None
    for ct in cluster_types:
        color = cluster_colors[ct]
        total_renewable_e = (
            cumulative_renewable_energy[ct]
            if total_renewable_e is None
            else total_renewable_e + cumulative_renewable_energy[ct]
        )
        total_grid_e = (
            cumulative_grid_energy[ct]
            if total_grid_e is None
            else total_grid_e + cumulative_grid_energy[ct]
        )
        ax.plot(df.index, cumulative_energy[ct], label=f'{ct}', color=color, linewidth=2)
        total_e = cumulative_energy[ct] if total_e is None else total_e + cumulative_energy[ct]

    if total_renewable_e is not None and total_grid_e is not None:
        ax.fill_between(
            df.index,
            0,
            total_renewable_e,
            color=source_colors['renewable'],
            alpha=0.14,
        )
        ax.fill_between(
            df.index,
            total_renewable_e,
            total_renewable_e + total_grid_e,
            color=source_colors['grid'],
            alpha=0.14,
        )

    if total_e is not None:
        ax.plot(df.index, total_e, label='Total', color='black', linewidth=2,
                linestyle='--', alpha=0.6)

    if cluster_types:
        import matplotlib.patches as mpatches
        source_fill_handles = [
            mpatches.Patch(
                color=source_colors['renewable'],
                alpha=0.30,
                label='Total renewable fill',
            ),
            mpatches.Patch(
                color=source_colors['grid'],
                alpha=0.30,
                label='Total grid fill',
            ),
        ]

    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Cumulative Energy Consumed per Cluster', fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + source_fill_handles, labels + [h.get_label() for h in source_fill_handles],
              loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cumulative CO₂ per cluster
    ax = axes[1]
    total_co2 = None
    for ct in cluster_types:
        color = cluster_colors[ct]
        ax.plot(df.index, cumulative_co2_kg[ct], label=f'{ct}', color=color,
                linewidth=1.7, zorder=3)
        total_co2 = cumulative_co2_kg[ct] if total_co2 is None else total_co2 + cumulative_co2_kg[ct]

    if total_co2 is not None:
        ax.fill_between(df.index, 0, total_co2, alpha=0.28,
                        color='#d9c7f2', zorder=1)
        ax.plot(df.index, total_co2, label='Total CO₂', color='#8e44ad',
                linewidth=2.6, zorder=4)
        ax.set_ylim(bottom=0)

    ax.set_ylabel('CO₂ (kg)')
    ax.set_title('Cumulative CO₂ Emissions per Cluster\n(attributed proportionally to power draw, grid-only)', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Carbon intensity per cluster (gCO₂/kWh)
    ax = axes[2]
    for ct in cluster_types:
        color = cluster_colors[ct]
        ax.plot(df.index, co2_intensity[ct].fillna(0), label=f'{ct}', color=color,
                linewidth=1.5, alpha=0.85)

    ax.set_ylabel('Carbon Intensity (gCO₂/kWh)')
    ax.set_title('Effective Carbon Intensity per Cluster', fontweight='bold')
    ax.set_xlabel('Time')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Time axis
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    for a in axes:
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)

    # Final summary annotation
    summary_lines = ["Final totals:"]
    for ct in cluster_types:
        e_final = cumulative_energy[ct].iloc[-1] if len(cumulative_energy[ct]) > 0 else 0
        co2_final = cumulative_co2_kg[ct].iloc[-1] if len(cumulative_co2_kg[ct]) > 0 else 0
        summary_lines.append(f"  {ct}: {e_final:,.0f} kWh | {co2_final:.1f} kg CO₂")
    axes[1].text(
        0.01, 0.97, "\n".join(summary_lines),
        transform=axes[1].transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(plot_output, dpi=150, bbox_inches='tight')
    print(f"✅ Carbon analysis plot saved to: {os.path.basename(plot_output)}")
    plt.close(fig)

def plot_job_metrics(schedulers, output_file: str = "job_metrics.png"):
    """
    Plots the Job SLA and Contention Metrics from the exported CSVs.
    Shows the Stretch Factor distribution by Priority, its relationship with
    predicted execution time, and actual model runtime vs observed runtime.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os

    dfs = []
    results_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    for scheduler in schedulers:
        ct = scheduler.cluster_type
        csv_name = f"jobs_metrics_{ct}.csv"
        candidate_dirs = [getattr(scheduler, 'out_dir', ''), results_dir]
        for candidate_dir in dict.fromkeys(candidate_dirs):
            if not candidate_dir:
                continue
            csv_file = os.path.join(candidate_dir, csv_name)
            if not os.path.exists(csv_file):
                continue
            df = pd.read_csv(csv_file)
            df['Cluster'] = ct
            dfs.append(df)
            break
            
    if not dfs:
        print(f"⚠️ No jobs_metrics CSVs found to plot {os.path.basename(output_file)}")
        return
        
    all_data = pd.concat(dfs, ignore_index=True)
    if 'Actual_Model_Exec_Time_s' not in all_data.columns and 'Predicted_Exec_Time_s' in all_data.columns:
        all_data['Actual_Model_Exec_Time_s'] = all_data['Predicted_Exec_Time_s']

    numeric_cols = [
        'Priority',
        'Initial_Frequency_GHz',
        'Final_Frequency_GHz',
        'Frequency_Changes',
        'Stretch_Factor',
        'Predicted_Exec_Time_s',
        'Actual_Model_Exec_Time_s',
        'Actual_Compute_Cycles',
        'Actual_Runtime_Overhead_s',
        'Real_Exec_Time_s',
        'Prediction_Error_s',
        'Prediction_Error_Pct',
        'Runtime_Error_Factor',
    ]
    for col in numeric_cols:
        if col in all_data.columns:
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
    all_data = all_data.replace([np.inf, -np.inf], np.nan)

    clusters = all_data['Cluster'].dropna().unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(clusters))))
    cluster_colors = {ct: colors[idx] for idx, ct in enumerate(clusters)}
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('Job SLA and Contention Metrics', fontsize=16, fontweight='bold')
    
    # ====== Panel 1: Stretch Factor by Priority ======
    ax = axes[0]
    priority_data = all_data[['Priority', 'Stretch_Factor']].dropna()
    if priority_data.empty:
        ax.text(0.5, 0.5, 'No priority/stretch data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    else:
        priorities = sorted(priority_data['Priority'].unique())
        box_data = [
            priority_data[priority_data['Priority'] == p]['Stretch_Factor']
            for p in priorities
        ]
        bp = ax.boxplot(box_data, patch_artist=True)
        priority_colors = {1: '#2ecc71', 2: '#f1c40f', 3: '#e74c3c'}
        for patch, priority in zip(bp['boxes'], priorities):
            patch.set_facecolor(priority_colors.get(priority, '#95a5a6'))
            patch.set_alpha(0.7)

        ax.set_xticklabels([f"P{int(p)}" for p in priorities])
        ax.set_ylabel('Stretch Factor (Real+Wait / Ideal)')
        ax.set_xlabel('Job Priority')
        ax.set_title('Stretch Factor Distribution by Priority', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect (1.0)')
        ax.legend(loc='upper right')

    # ====== Panel 2: Stretch Factor vs Predicted Execution Time ======
    ax = axes[1]
    runtime_data = all_data[['Cluster', 'Predicted_Exec_Time_s', 'Stretch_Factor']].dropna()
    runtime_data = runtime_data[runtime_data['Predicted_Exec_Time_s'] > 0]

    if runtime_data.empty:
        ax.text(0.5, 0.5, 'No runtime/stretch data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    else:
        for ct in clusters:
            subset = runtime_data[runtime_data['Cluster'] == ct]
            if subset.empty:
                continue
            ax.scatter(
                subset['Predicted_Exec_Time_s'],
                subset['Stretch_Factor'],
                alpha=0.35,
                label=ct,
                color=cluster_colors[ct],
                s=16,
            )

        x = runtime_data['Predicted_Exec_Time_s'].to_numpy(dtype=float)
        y = runtime_data['Stretch_Factor'].to_numpy(dtype=float)
        use_log_x = x.min() > 0 and (x.max() / x.min()) >= 20

        if len(x) >= 2 and not np.allclose(x, x[0]):
            if use_log_x:
                fit_x = np.log10(x)
                line_x = np.geomspace(x.min(), x.max(), 200)
                slope, intercept = np.polyfit(fit_x, y, 1)
                line_y = slope * np.log10(line_x) + intercept
                corr = np.corrcoef(fit_x, y)[0, 1]
                ax.set_xscale('log')
                trend_label = f'Linear trend on log-runtime (r={corr:.2f})'
            else:
                line_x = np.linspace(x.min(), x.max(), 200)
                slope, intercept = np.polyfit(x, y, 1)
                line_y = slope * line_x + intercept
                corr = np.corrcoef(x, y)[0, 1]
                trend_label = f'Linear trend (r={corr:.2f})'

            ax.plot(line_x, line_y, color='black', linestyle='--', linewidth=2, label=trend_label)

        if runtime_data['Predicted_Exec_Time_s'].nunique() >= 4:
            bin_edges = (
                np.geomspace(x.min(), x.max(), 7)
                if use_log_x
                else np.linspace(x.min(), x.max(), 7)
            )
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) >= 3:
                binned = runtime_data.copy()
                binned['Runtime_Bin'] = pd.cut(
                    binned['Predicted_Exec_Time_s'],
                    bins=bin_edges,
                    include_lowest=True,
                )
                median_trend = (
                    binned.groupby('Runtime_Bin', observed=False)[
                        ['Predicted_Exec_Time_s', 'Stretch_Factor']
                    ]
                    .median()
                    .dropna()
                )
                if not median_trend.empty:
                    ax.plot(
                        median_trend['Predicted_Exec_Time_s'],
                        median_trend['Stretch_Factor'],
                        color='#8e44ad',
                        linewidth=2.5,
                        marker='o',
                        label='Median by runtime bin',
                    )

        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.4, label='Perfect (1.0)')
        ax.set_xlabel('Predicted Execution Time (s)')
        ax.set_ylabel('Stretch Factor')
        ax.set_title('Stretch Factor vs Predicted Execution Time', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # ====== Panel 3: Actual Model vs Real Execution Time ======
    ax = axes[2]
    overhead_data = all_data[['Cluster', 'Actual_Model_Exec_Time_s', 'Real_Exec_Time_s']].dropna()
    if overhead_data.empty:
        ax.text(0.5, 0.5, 'No actual/real runtime data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    else:
        for ct in clusters:
            subset = overhead_data[overhead_data['Cluster'] == ct]
            if subset.empty:
                continue
            ax.scatter(
                subset['Actual_Model_Exec_Time_s'],
                subset['Real_Exec_Time_s'],
                alpha=0.5,
                label=ct,
                color=cluster_colors[ct],
                s=15,
            )

        max_val = max(
            overhead_data['Actual_Model_Exec_Time_s'].max(),
            overhead_data['Real_Exec_Time_s'].max(),
        )
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No Overhead (y=x)')

        ax.set_xlabel('Actual Model Execution Time (s)')
        ax.set_ylabel('Real Execution Time (s)')
        ax.set_title('Actual Model vs Real Execution Time (Contention)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Job metrics saved to: {os.path.basename(output_file)}")
    plt.close(fig)

def plot_cluster_job_timeline(schedulers, output_file: str = "cluster_job_timeline.png"):
    """
    Cluster-level Gantt-style chart showing job execution order.
    Jobs are packed into abstract "lanes" using broken_barh for clean rendering.
    Colored by priority: 1=low, 2=normal, 3=high.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates
    from datetime import timedelta

    color_map = {1: '#2ecc71', 2: '#f1c40f', 3: '#e74c3c'}

    # ── 1. Pre-process: collect jobs and pack into lanes ──────────────────
    cluster_data = []

    for scheduler in schedulers:
        ct = scheduler.cluster_type
        jobs_to_plot = []
        for job in scheduler.job_queue.jobs:
            if job.status in ['running', 'completed'] and job.start_time is not None:
                start = job.start_time
                if hasattr(job, 'completion_time') and job.completion_time is not None:
                    end = job.completion_time
                else:
                    duration = job.actual_t_exec or job.remaining_time
                    end = start + timedelta(seconds=duration)
                jobs_to_plot.append((start, end, job))

        jobs_to_plot.sort(key=lambda x: x[0])

        # Greedy interval packing
        lanes = []       # end-time of last job in each lane
        job_lanes = []   # (lane_idx, start, end, job)
        for start, end, job in jobs_to_plot:
            placed = False
            for i, lane_end in enumerate(lanes):
                if start >= lane_end:
                    lanes[i] = end
                    job_lanes.append((i, start, end, job))
                    placed = True
                    break
            if not placed:
                lanes.append(end)
                job_lanes.append((len(lanes) - 1, start, end, job))

        cluster_data.append({
            'ct': ct,
            'n_jobs': len(jobs_to_plot),
            'job_lanes': job_lanes,
            'num_lanes': len(lanes),
        })

    # ── 2. Figure layout ─────────────────────────────────────────────────
    n_clusters = len(schedulers)
    panel_h = 6  # fixed comfortable height per cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=(18, panel_h * n_clusters))
    if n_clusters == 1:
        axes = [axes]

    fig.suptitle('Cluster-Level Job Execution Timeline',
                 fontsize=16, fontweight='bold', y=0.99)

    # ── 3. Draw each cluster ─────────────────────────────────────────────
    for idx, data in enumerate(cluster_data):
        ax = axes[idx]
        ct = data['ct']
        job_lanes = data['job_lanes']
        num_lanes = data['num_lanes']
        n_jobs = data['n_jobs']

        if not job_lanes:
            ax.text(0.5, 0.5, f'No jobs executed on {ct}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_title(f'{ct}', fontweight='bold')
            continue

        # Group bars by priority so we batch-draw with broken_barh per lane
        # This is much faster and cleaner than thousands of individual barh calls
        bar_height = 0.8
        for lane_idx, start, end, job in job_lanes:
            priority = getattr(job, 'priority', 2)
            color = color_map.get(priority, '#95a5a6')

            start_num = mdates.date2num(start)
            width = mdates.date2num(end) - start_num

            ax.broken_barh(
                [(start_num, width)],
                (lane_idx - bar_height / 2, bar_height),
                facecolors=color,
                edgecolors='none',
                alpha=0.85,
            )

        ax.set_ylim(-0.5, num_lanes - 0.5)
        ax.set_yticks([])
        ax.set_ylabel(f'Lanes ({num_lanes})')
        ax.set_title(f'{ct}  —  {n_jobs} jobs in {num_lanes} lanes', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_xlabel('Time')

    # ── 4. Legend & formatting ───────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=color_map[1], label='Priority 1 (Low)'),
        mpatches.Patch(facecolor=color_map[2], label='Priority 2 (Normal)'),
        mpatches.Patch(facecolor=color_map[3], label='Priority 3 (High)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=10)

    for a in axes:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
        formatter = mdates.ConciseDateFormatter(locator)
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Cluster job timeline saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def plot_budget_vs_forecast(
    results_dir: str,
    df=None,
    output_file: str = "budget_vs_forecast.png",
    figsize: tuple[float, float] = (16, 6),
    dpi: int = 150,
) -> None:
    """
    Aggregate actual IT power, active-plan forecast, and runtime IT budget.

    Plots the cluster-aggregate power draw measured in the simulation ("used").
    New runs use ``puzzle_active_forecast.csv``, a compact unsampled export of
    every plan while it was in force.  Detailed plan exports may remain sampled
    without corrupting this timeline.  All compared values are IT-side watts.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd

    slots_path = os.path.join(results_dir, "puzzle_forecast_slots.csv")
    active_path = os.path.join(results_dir, "puzzle_active_forecast.csv")
    if not os.path.exists(slots_path):
        raise FileNotFoundError(
            "puzzle_forecast_slots.csv not found. Run a simulation with "
            "--budget-policy puzzle first."
        )
    slots_df = pd.read_csv(slots_path, parse_dates=["start_time", "end_time"])

    if df is None:
        df = pd.read_csv(
            os.path.join(results_dir, "energy_simulation_result.csv"),
            index_col=0,
            parse_dates=True,
        )
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # -- Forecast in force -------------------------------------------------
    if os.path.exists(active_path):
        active = pd.read_csv(active_path, parse_dates=["start_time", "end_time"])
        forecast = active.sort_values("start_time")[[
            "start_time", "planned_it_power_w",
        ]].rename(columns={"planned_it_power_w": "forecast_w"})
    else:
        # Legacy fallback.  Never bridge across plans omitted by the sampled
        # detailed export: missing intervals are deliberately left blank.
        totals = slots_df[slots_df["cluster_type"] == "__total__"].copy()
        if totals.empty:
            raise ValueError("No __total__ rows in puzzle_forecast_slots.csv.")
        # Legacy files predate facility-side puzzle packing: their rectangle
        # allocations were stored as IT watts, so do not apply PUE retroactively.
        totals["forecast_w"] = (
            totals["allocated_w"] + totals["grid_allocated_w"]
        )

        summary_path = os.path.join(results_dir, "puzzle_plan_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path, parse_dates=["plan_start"])
            all_starts = summary.set_index("plan_id")["plan_start"].sort_index()
        else:
            all_starts = totals.groupby("plan_id")["start_time"].min().sort_index()

        pieces = []
        for pid, piece in totals.groupby("plan_id"):
            later = all_starts[all_starts.index > pid]
            if not later.empty:
                active_end = later.iloc[0]
                piece = piece[piece["start_time"] < active_end]
            pieces.append(piece[["start_time", "forecast_w"]])
            if not later.empty:
                pieces.append(pd.DataFrame({
                    "start_time": [active_end], "forecast_w": [float("nan")],
                }))
        forecast = pd.concat(pieces).sort_values("start_time")
    # The last plan's slots (and any extrapolated tail) can run past the end
    # of the simulation; clip to the measured range so the axis stays tight.
    forecast = forecast[forecast["start_time"] <= df.index.max()]

    # -- Aggregate power actually used --------------------------------------
    clusters = [
        str(c) for c in slots_df["cluster_type"].dropna().unique()
        if str(c) != "__total__"
    ]
    power_cols = [f"{ct}_power" for ct in clusters if f"{ct}_power" in df.columns]
    if not power_cols:
        raise ValueError("No <cluster>_power columns in the results dataframe.")
    used_kw = df[power_cols].ffill().fillna(0).sum(axis=1) / 1000.0

    # -- Plot ---------------------------------------------------------------
    # Okabe-Ito blue/vermillion: colorblind-safe pair, plus line-style and
    # fill as secondary encoding so identity never rides on color alone.
    used_color = "#0072B2"
    forecast_color = "#D55E00"

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(used_kw.index, used_kw, color=used_color, alpha=0.15)
    ax.plot(
        used_kw.index, used_kw,
        color=used_color, linewidth=1.8, label="Used (measured aggregate)",
    )
    ax.step(
        forecast["start_time"], forecast["forecast_w"] / 1000.0,
        where="post", color=forecast_color, linewidth=2.0, linestyle="--",
        label="Forecast (active plan)",
    )
    if "budget_scheduler_w" in df.columns:
        runtime_budget_kw = (
            pd.to_numeric(df["budget_scheduler_w"], errors="coerce")
            .ffill().fillna(0.0) / 1000.0
        )
        ax.step(
            runtime_budget_kw.index, runtime_budget_kw,
            where="post", color="#009E73", linewidth=1.5, linestyle=":",
            label="Runtime IT budget",
        )

    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Time")
    ax.set_title(
        "Actual IT power vs active-plan forecast and runtime budget",
        fontweight="bold",
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"✅ Budget vs forecast saved to: {os.path.basename(output_file)}")
    plt.close(fig)


def _stitch_puzzle_green_board(slots_df):
    """Return the per-cluster green board that was active at each slot.

    Forecast exports contain overlapping rolling plans.  For a given timestamp,
    the plan with the greatest id is the most recently built plan that could
    have governed that timestamp.  The aggregate ``__total__`` rows are omitted
    because comparison plots are rendered independently per cluster.
    """
    import pandas as pd

    required = {
        "plan_id",
        "cluster_type",
        "start_time",
        "end_time",
        "green_available_w",
    }
    missing = required.difference(slots_df.columns)
    if missing:
        raise ValueError(
            "Puzzle forecast slots are missing columns: "
            + ", ".join(sorted(missing))
        )

    stitched = slots_df.copy()
    stitched["start_time"] = pd.to_datetime(
        stitched["start_time"], format="mixed", errors="raise"
    )
    stitched["end_time"] = pd.to_datetime(
        stitched["end_time"], format="mixed", errors="raise"
    )
    stitched = stitched[stitched["cluster_type"] != "__total__"]
    stitched = (
        stitched.sort_values(
            ["cluster_type", "start_time", "plan_id"]
        )
        .drop_duplicates(
            subset=["cluster_type", "start_time"], keep="last"
        )
        .sort_values(["cluster_type", "start_time"])
        .reset_index(drop=True)
    )
    return stitched[
        [
            "plan_id",
            "cluster_type",
            "start_time",
            "end_time",
            "green_available_w",
        ]
    ]


def _job_average_power_w(job, node_cfg: dict, start, end) -> float:
    """Average modelled full-node power over a clipped execution interval."""
    import pandas as pd

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        return 0.0

    idle_w = float(node_cfg.get("idle_power_w", 0.0))
    table = {
        float(frequency): float(power)
        for frequency, power in node_cfg.get("frequencies", {}).items()
    }

    def power_at(frequency) -> float:
        if table:
            if frequency is None:
                frequency = node_cfg.get("base_frequency_ghz")
            selected = min(
                table,
                key=lambda value: abs(value - float(frequency)),
            )
            dynamic_w = table[selected]
        else:
            dynamic_w = float(node_cfg.get("max_power_w", 0.0))
        return idle_w + dynamic_w

    # Multiple DVFS changes can share the launch timestamp.  The last entry is
    # the frequency that actually remained in force after that event.
    history_by_time = {}
    for timestamp, frequency in getattr(job, "frequency_history", []) or []:
        history_by_time[pd.Timestamp(timestamp)] = frequency
    history = sorted(history_by_time.items())

    default_frequency = getattr(job, "assigned_frequency", None)
    current_frequency = default_frequency
    for timestamp, frequency in history:
        if timestamp <= start:
            current_frequency = frequency
        else:
            break

    boundaries = [start]
    boundaries.extend(
        timestamp for timestamp, _ in history if start < timestamp < end
    )
    boundaries.append(end)

    energy_ws = 0.0
    for seg_start, seg_end in zip(boundaries, boundaries[1:]):
        for timestamp, frequency in history:
            if timestamp <= seg_start:
                current_frequency = frequency
            else:
                break
        duration_s = (seg_end - seg_start).total_seconds()
        energy_ws += power_at(current_frequency) * duration_s

    nodes = max(0, int(getattr(job, "num_nodes", 0)))
    duration_s = (end - start).total_seconds()
    return energy_ws / duration_s * nodes


def _pack_real_jobs_to_power(scheduler, node_cfg: dict, start, end):
    """Bottom-pack real jobs into fungible power bands for visualisation."""
    import pandas as pd

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    candidates = []
    for job in scheduler.job_queue.jobs:
        job_start = getattr(job, "start_time", None)
        job_end = getattr(job, "completion_time", None)
        if job_start is None or job_end is None:
            continue
        clipped_start = max(pd.Timestamp(job_start), start)
        clipped_end = min(pd.Timestamp(job_end), end)
        if clipped_end <= clipped_start:
            continue
        height_kw = _job_average_power_w(
            job, node_cfg, clipped_start, clipped_end
        ) / 1000.0
        if height_kw <= 0.0:
            continue
        candidates.append((clipped_start, clipped_end, job, height_kw))

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[3],
            -(item[1] - item[0]).total_seconds(),
            int(item[2].job_id),
        )
    )

    placed = []
    placements = []
    for job_start, job_end, job, height_kw in candidates:
        active = sorted(
            (
                (y_start, y_end)
                for active_end, y_start, y_end in placed
                if active_end > job_start
            ),
            key=lambda band: band[0],
        )
        y_start = 0.0
        for active_start, active_end in active:
            if y_start + height_kw <= active_start:
                break
            if y_start < active_end:
                y_start = active_end
        placed.append((job_end, y_start, y_start + height_kw))
        placements.append(
            {
                "start": job_start,
                "end": job_end,
                "job": job,
                "y_kw": y_start,
                "height_kw": height_kw,
            }
        )
    return placements


def plot_scheduler_board_comparison(
    reference_results_dir: str,
    scheduler_results,
    output_dir: str,
    *,
    window_start=None,
    window_hours: float = 12.0,
    clusters=None,
    dpi: int = 160,
    show: bool = False,
) -> list[str]:
    """Compare real scheduler executions against one real Puzzle green board.

    ``reference_results_dir`` must contain ``puzzle_forecast_slots.csv``.  The
    stitched, per-cluster ``green_available_w`` curve from that run is reused
    unchanged in every scheduler row.  For non-Puzzle schedulers it is therefore
    a counterfactual reference, not the budget that constrained their run.

    ``scheduler_results`` accepts either a mapping ``label -> results_dir`` or
    an ordered iterable of ``(label, results_dir)`` pairs.  One PNG is written
    per cluster so every policy uses the same time and power scales.
    """
    import json
    import pickle
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if window_hours <= 0.0:
        raise ValueError("window_hours must be positive")

    reference_root = Path(reference_results_dir).expanduser().resolve()
    slots_path = reference_root / "puzzle_forecast_slots.csv"
    if not slots_path.is_file():
        raise FileNotFoundError(
            f"Puzzle forecast slots not found: {slots_path}"
        )
    board = _stitch_puzzle_green_board(pd.read_csv(slots_path))
    if board.empty:
        raise ValueError(f"Puzzle green board is empty: {slots_path}")

    if hasattr(scheduler_results, "items"):
        result_items = list(scheduler_results.items())
    else:
        result_items = list(scheduler_results)
    if not result_items:
        raise ValueError("At least one scheduler result is required")

    workspace_root = reference_root.parent.parent
    node_config_path = workspace_root / "datasets" / "node_config.json"
    with node_config_path.open(encoding="utf-8") as handle:
        node_config = json.load(handle)

    runs = []
    for label, results_dir in result_items:
        root = Path(results_dir).expanduser().resolve()
        with (root / "schedulers.pkl").open("rb") as handle:
            schedulers = pickle.load(handle)
        energy = pd.read_csv(
            root / "energy_simulation_result.csv",
            index_col=0,
            parse_dates=True,
        )
        runs.append(
            {
                "label": str(label),
                "root": root,
                "schedulers": {
                    scheduler.cluster_type: scheduler
                    for scheduler in schedulers
                },
                "energy": energy,
            }
        )

    if window_start is None:
        window_start = board["start_time"].min()
    window_start = pd.Timestamp(window_start)
    window_end = window_start + pd.Timedelta(hours=float(window_hours))

    available_clusters = sorted(
        set(board["cluster_type"]).intersection(
            *(set(run["schedulers"]) for run in runs)
        )
    )
    selected_clusters = (
        available_clusters
        if clusters is None
        else [str(cluster) for cluster in clusters]
    )
    unknown = set(selected_clusters).difference(available_clusters)
    if unknown:
        raise ValueError(
            "Clusters unavailable in every run/reference: "
            + ", ".join(sorted(unknown))
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = []

    green_a = "#20b77a"
    green_b = "#11875d"
    grid_orange = "#cf651c"
    board_edge = "#124f3d"

    for cluster in selected_clusters:
        cluster_board = board[
            (board["cluster_type"] == cluster)
            & (board["end_time"] > window_start)
            & (board["start_time"] < window_end)
        ].copy()
        if cluster_board.empty:
            continue
        cluster_board = cluster_board.sort_values("start_time")
        board_series = pd.Series(
            cluster_board["green_available_w"].to_numpy(dtype=float) / 1000.0,
            index=pd.DatetimeIndex(cluster_board["start_time"]),
        )
        board_times = list(cluster_board["start_time"])
        board_values = list(board_series.to_numpy())
        board_times.append(cluster_board["end_time"].iloc[-1])
        board_values.append(board_values[-1])

        run_data = []
        max_y_kw = max(board_values, default=0.0)
        for run in runs:
            scheduler = run["schedulers"][cluster]
            placements = _pack_real_jobs_to_power(
                scheduler,
                node_config.get(cluster, {}),
                window_start,
                window_end,
            )
            if placements:
                max_y_kw = max(
                    max_y_kw,
                    max(
                        placement["y_kw"] + placement["height_kw"]
                        for placement in placements
                    ),
                )

            energy = run["energy"]
            energy_window = energy[
                (energy.index >= window_start)
                & (energy.index <= window_end)
            ]
            power_col = f"{cluster}_power"
            actual_kw = (
                pd.to_numeric(
                    energy_window.get(power_col, 0.0), errors="coerce"
                ).fillna(0.0) / 1000.0
            )
            if len(actual_kw):
                max_y_kw = max(max_y_kw, float(actual_kw.max()))
                active_board = board_series.reindex(
                    actual_kw.index, method="ffill"
                ).fillna(0.0)
                actual_energy = float(actual_kw.sum())
                aligned_energy = float(
                    np.minimum(actual_kw.to_numpy(), active_board.to_numpy()).sum()
                )
                aligned_fraction = (
                    aligned_energy / actual_energy
                    if actual_energy > 0.0 else 0.0
                )
            else:
                active_board = pd.Series(dtype=float)
                aligned_fraction = 0.0

            run_data.append(
                {
                    "run": run,
                    "placements": placements,
                    "actual_kw": actual_kw,
                    "active_board": active_board,
                    "aligned_fraction": aligned_fraction,
                }
            )

        fig, axes = plt.subplots(
            len(run_data),
            1,
            figsize=(22, max(4.0, 3.4 * len(run_data))),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        for row, data in enumerate(run_data):
            ax = axes[row, 0]
            ax.set_facecolor("#fbfcfa")
            ax.fill_between(
                board_times,
                board_values,
                step="post",
                color="#dceae5",
                alpha=0.72,
                linewidth=0.0,
                zorder=0,
            )

            actual_kw = data["actual_kw"]
            active_board = data["active_board"]
            if len(actual_kw):
                ax.fill_between(
                    actual_kw.index,
                    active_board.to_numpy(),
                    actual_kw.to_numpy(),
                    where=(actual_kw.to_numpy() > active_board.to_numpy()),
                    step="post",
                    color=grid_orange,
                    alpha=0.18,
                    hatch="///",
                    linewidth=0.0,
                    zorder=1,
                )

            for placement in data["placements"]:
                job = placement["job"]
                job_start = placement["start"]
                job_end = placement["end"]
                y_kw = placement["y_kw"]
                height_kw = placement["height_kw"]
                job_tone = (
                    green_a if int(job.job_id) % 2 == 0 else green_b
                )

                split_times = [job_start]
                split_times.extend(
                    timestamp
                    for timestamp in board_series.index
                    if job_start < timestamp < job_end
                )
                split_times.append(job_end)
                for seg_start, seg_end in zip(
                    split_times, split_times[1:]
                ):
                    midpoint = seg_start + (seg_end - seg_start) / 2
                    board_idx = board_series.index.searchsorted(
                        midpoint, side="right"
                    ) - 1
                    green_ceiling_kw = (
                        float(board_series.iloc[board_idx])
                        if board_idx >= 0 else 0.0
                    )
                    green_height_kw = max(
                        0.0,
                        min(
                            height_kw,
                            green_ceiling_kw - y_kw,
                        ),
                    )
                    x = mdates.date2num(seg_start)
                    width = (
                        mdates.date2num(seg_end)
                        - mdates.date2num(seg_start)
                    )
                    if green_height_kw > 0.0:
                        ax.add_patch(mpatches.Rectangle(
                            (x, y_kw), width, green_height_kw,
                            facecolor=job_tone,
                            edgecolor="none",
                            alpha=0.92,
                            zorder=2,
                        ))
                    grid_height_kw = height_kw - green_height_kw
                    if grid_height_kw > 0.0:
                        ax.add_patch(mpatches.Rectangle(
                            (x, y_kw + green_height_kw),
                            width,
                            grid_height_kw,
                            facecolor=grid_orange,
                            edgecolor="none",
                            alpha=0.90,
                            zorder=2,
                        ))

                x = mdates.date2num(job_start)
                width = mdates.date2num(job_end) - x
                ax.add_patch(mpatches.Rectangle(
                    (x, y_kw),
                    width,
                    height_kw,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=3,
                ))
                duration_h = (job_end - job_start).total_seconds() / 3600.0
                if duration_h >= 1.5 and height_kw >= 1.5:
                    ax.text(
                        x + width / 2,
                        y_kw + height_kw / 2,
                        f"J{job.job_id}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                        fontweight="bold",
                        zorder=4,
                    )

            ax.step(
                board_times,
                board_values,
                where="post",
                color=board_edge,
                linewidth=2.0,
                zorder=5,
            )
            if len(actual_kw):
                ax.step(
                    actual_kw.index,
                    actual_kw.to_numpy(),
                    where="post",
                    color="#30343b",
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=5,
                )

            label = data["run"]["label"]
            n_jobs = len(data["placements"])
            aligned_pct = 100.0 * data["aligned_fraction"]
            # Keep the row label inside its own axes.  External axes titles can
            # be covered by the neighbouring axes patch in tall shared-axis
            # figures (most visible in the compact GPU panels).
            ax.text(
                0.004,
                0.976,
                f"{label}  ·  {n_jobs} jobs ativos  ·  "
                f"{aligned_pct:.1f}% da energia IT sob o green board",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="#202328",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.84,
                    "pad": 1.5,
                },
                zorder=10,
            )
            ax.set_ylabel("Potência (kW)")
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.7)
            ax.set_ylim(0.0, max(1.0, max_y_kw * 1.06))

        axes[-1, 0].set_xlabel("Tempo")
        axes[-1, 0].set_xlim(window_start, window_end)
        axes[-1, 0].xaxis.set_major_locator(mdates.HourLocator(interval=2))
        axes[-1, 0].xaxis.set_major_formatter(
            mdates.DateFormatter("%b %d\n%H:%M")
        )

        legend_handles = [
            mpatches.Patch(color=green_a, label="Job sob green board"),
            mpatches.Patch(
                color=grid_orange,
                label="Job acima do board (grid/recourse necessário)",
            ),
            mpatches.Patch(
                facecolor="#dceae5",
                edgecolor=board_edge,
                label="Green board real do Puzzle",
            ),
            plt.Line2D(
                [0], [0], color="#30343b", linewidth=1.2,
                label="Potência IT medida",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.972),
        )
        fig.suptitle(
            f"Packing real contra o mesmo green board · {cluster}\n"
            f"{window_start:%Y-%m-%d %H:%M} → "
            f"{window_end:%Y-%m-%d %H:%M}",
            fontsize=15,
            fontweight="bold",
            y=0.998,
        )
        fig.text(
            0.5,
            0.006,
            "Nos schedulers não-Puzzle, o board é uma referência "
            "contrafactual; a atribuição verde/grid é fungível e visual, "
            "não uma atribuição física por job.",
            ha="center",
            fontsize=8.5,
            color="#555555",
        )
        fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.94), h_pad=1.2)

        output_path = output_root / (
            f"scheduler_board_comparison_{cluster}.png"
        )
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        output_paths.append(str(output_path))

    return output_paths


def plot_combined_puzzle(
    results_dir: str,
    plan_id: str | int = "first",
    show: bool = False,
    figsize: tuple[float, float] = (24, 10),
    dpi: int = 150,
    real_only: bool = False,
) -> None:
    """
    Combined Puzzle Visualization: Forecast on the left, Real Gantt-Puzzle on the right.
    Generates a single puzzle_combined.png file for all clusters in results_dir.

    When ``real_only`` is set, or when no puzzle forecast CSVs exist (i.e. the
    run used a non-puzzle budget policy), only the real-execution "gantt-puzzle"
    panels are drawn — one per cluster over the whole simulation — with the
    renewable envelope overlaid as a green board. This makes any policy
    comparable to the puzzle on the same canvas (saved as ``real_puzzle.png``).
    """
    import os
    import pickle
    import json
    from datetime import timedelta
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates
    from matplotlib.colors import hsv_to_rgb

    # -- 1. Helper functions ------------------------------------------------
    def load_forecast(res_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        slots_path = os.path.join(res_dir, "puzzle_forecast_slots.csv")
        jobs_path = os.path.join(res_dir, "puzzle_forecast_jobs.csv")
        if not os.path.exists(slots_path) or not os.path.exists(jobs_path):
            raise FileNotFoundError(
                "Puzzle forecast CSVs not found. Run a simulation with "
                "--budget-policy puzzle first."
            )
        slots_df = pd.read_csv(slots_path, parse_dates=["start_time", "end_time"])
        jobs_df = pd.read_csv(jobs_path, parse_dates=["plan_start", "plan_end", "arrival_time", "start_time", "end_time"])
        return slots_df, jobs_df

    def available_plan_ids(slots_df: pd.DataFrame) -> list[int]:
        plan_ids = sorted(int(x) for x in slots_df["plan_id"].dropna().unique())
        if not plan_ids:
            raise ValueError("No puzzle forecast plans found.")
        return plan_ids

    def worst_spill_plan_ids(res_dir: str, plan_ids: list[int]) -> list[int]:
        """Rank plans by grid spills (descending) so the carbon-worst days
        come first; falls back to the natural order if the summary is absent."""
        summary_path = os.path.join(res_dir, "puzzle_plan_summary.csv")
        if not os.path.exists(summary_path):
            return plan_ids
        summary = pd.read_csv(summary_path)
        if "num_grid_spilled" not in summary.columns:
            return plan_ids
        ranked = (
            summary[summary["plan_id"].isin(plan_ids)]
            .sort_values("num_grid_spilled", ascending=False)["plan_id"]
            .astype(int)
            .tolist()
        )
        # Keep any plan missing from the summary, appended in natural order.
        return ranked + [p for p in plan_ids if p not in ranked]

    def resolve_plan_ids(slots_df: pd.DataFrame, pid, res_dir: str) -> list[int]:
        """Resolve a plan selector to a concrete, ordered list of plan ids.

        Accepts an int, a comma-separated string, a list, or the keywords
        ``first`` / ``latest`` / ``all`` / ``worst`` (worst grid-spill plan).
        """
        plan_ids = available_plan_ids(slots_df)
        valid = set(plan_ids)

        def _check(selected: int) -> int:
            if selected not in valid:
                raise ValueError(
                    f"Plan {selected} not found. Available plans: {plan_ids}"
                )
            return selected

        if isinstance(pid, (list, tuple)):
            return [_check(int(p)) for p in pid]
        if isinstance(pid, str):
            key = pid.strip().lower()
            if key == "first":
                return [plan_ids[0]]
            if key == "latest":
                return [plan_ids[-1]]
            if key == "all":
                return plan_ids
            if key == "worst":
                return [worst_spill_plan_ids(res_dir, plan_ids)[0]]
            if "," in key:
                return [_check(int(p)) for p in key.split(",") if p.strip()]
        return [_check(int(pid))]

    def forecast_clusters(slots_df: pd.DataFrame) -> list[str]:
        clusters_found = [
            str(cluster)
            for cluster in slots_df["cluster_type"].dropna().unique()
            if str(cluster) != "__total__"
        ]
        return sorted(clusters_found)

    def load_schedulers(res_dir: str):
        path = os.path.join(res_dir, "schedulers.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_results_df(res_dir: str):
        path = os.path.join(res_dir, "energy_simulation_result.csv")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def load_node_config(res_dir: str) -> dict:
        workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(res_dir)))
        node_config_path = os.path.join(workspace_root, "datasets", "node_config.json")
        if os.path.exists(node_config_path):
            with open(node_config_path) as f:
                return json.load(f)
        config_path = os.path.join(res_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            return cfg.get("node_config", cfg)
        return {}

    def node_power_at_frequency(cfg: dict, frequency) -> float:
        idle_power = float(cfg.get("idle_power_w", 0.0))
        freq_table = {
            float(f): float(p)
            for f, p in cfg.get("frequencies", {}).items()
        }
        if freq_table:
            if frequency is None:
                frequency = cfg.get("base_frequency_ghz")
            if frequency is not None:
                frequency = float(frequency)
                selected = min(freq_table, key=lambda f: abs(f - frequency))
                dynamic_power = freq_table[selected]
            else:
                dynamic_power = max(freq_table.values())
        else:
            dynamic_power = float(cfg.get("max_power_w", 0.0))
        return idle_power + dynamic_power

    def job_power_w(job, cfg: dict) -> float:
        frequency = None
        if getattr(job, "frequency_history", None):
            frequency = job.frequency_history[0][1]
        if frequency is None:
            frequency = getattr(job, "assigned_frequency", None)
        return node_power_at_frequency(cfg, frequency) * job.num_nodes

    def pack_jobs_to_power(jobs_to_plot_list, cfg: dict):
        sorted_jobs = sorted(
            jobs_to_plot_list,
            key=lambda x: (
                x[0],
                -job_power_w(x[2], cfg),
                -(x[1] - x[0]).total_seconds()
            )
        )
        placed_list = []
        job_placements = []
        for start, end, job in sorted_jobs:
            h = job_power_w(job, cfg) / 1000.0
            active = [p for p in placed_list if p[0] > start]
            y_start = 0.0
            while True:
                overlap = False
                for _, ay_start, ay_end in active:
                    if not (y_start + h <= ay_start or y_start >= ay_end):
                        overlap = True
                        y_start = ay_end
                        break
                if not overlap:
                    break
            placed_list.append((end, y_start, y_start + h))
            job_placements.append((y_start, start, end, job, h))
        return job_placements

    def plot_cluster_forecast_on_ax(ax_obj, slots_df, jobs_df, cluster_name, resolved_pid, color_map=None):
        slot_df = slots_df[
            (slots_df["plan_id"] == resolved_pid)
            & (slots_df["cluster_type"] == cluster_name)
        ].sort_values("slot_idx")
        if slot_df.empty:
            ax_obj.text(0.5, 0.5, f"No forecast slots found for {cluster_name}",
                        ha="center", va="center", fontsize=11, color="gray")
            ax_obj.set_title(f"Forecast - {cluster_name} (No data)", fontsize=11, color="#f0f6fc", fontweight="bold")
            return

        job_df = jobs_df[
            (jobs_df["plan_id"] == resolved_pid)
            & (jobs_df["cluster_type"] == cluster_name)
            & (jobs_df["placed"].astype(str).str.lower().isin(["true", "1"]))
        ].copy()
        job_df = job_df.dropna(subset=["start_time", "end_time"])

        times = list(slot_df["start_time"]) + [slot_df["end_time"].iloc[-1]]
        envelope_kw = list(slot_df["total_available_w"] / 1000.0) + [0.0]
        allocated_kw = list(slot_df["allocated_w"] / 1000.0) + [0.0]

        # The green board is the only ceiling the packer is allowed to fill;
        # anything above it is the grid-overflow penalty lane. Draw it
        # explicitly so grid-spilled pieces visibly stick out above it.
        has_green = "green_available_w" in slot_df.columns
        if has_green:
            green_kw = list(slot_df["green_available_w"] / 1000.0) + [0.0]
            # Green ceiling per slot index, for inferring grid spills when the
            # jobs CSV predates the used_grid column.
            green_by_idx = dict(
                zip(slot_df["slot_idx"], slot_df["green_available_w"])
            )
        else:
            green_kw = envelope_kw
            green_by_idx = {}

        ax_obj.fill_between(
            times,
            envelope_kw,
            step="post",
            alpha=0.20,
            color="#1a3a5c",
            label="Free budget (unused)",
        )
        # Idle-power floor: power to keep nodes switched on, drawn as a
        # distinct band at the bottom so it isn't mistaken for free budget.
        # Jobs (dynamic power) are stacked on top of this floor.
        if "idle_reserved_w" in slot_df.columns:
            idle_kw = list(slot_df["idle_reserved_w"] / 1000.0) + [0.0]
            ax_obj.fill_between(
                times,
                idle_kw,
                step="post",
                alpha=0.45,
                color="#6b4f2a",
                label="Idle power (nodes on)",
            )
        ax_obj.step(
            times,
            envelope_kw,
            where="post",
            color="#4a9eff",
            linewidth=1.8,
        )
        if has_green:
            ax_obj.step(
                times,
                green_kw,
                where="post",
                color="#3fb950",
                linewidth=1.6,
                alpha=0.95,
                label="Green board ceiling",
            )
        ax_obj.step(
            times,
            allocated_kw,
            where="post",
            color="#f2cc60",
            linewidth=1.2,
            alpha=0.9,
            label="Planned allocated (idle+compute)",
        )

        has_used_grid = "used_grid" in job_df.columns
        grid_spill_seen = False
        for row in job_df.itertuples(index=False):
            start = row.start_time
            end = row.end_time
            if pd.isna(start) or pd.isna(end):
                continue

            x_num = mdates.date2num(start)
            w = mdates.date2num(end) - x_num
            y = float(row.stack_base_w) / 1000.0
            h = float(row.power_w) / 1000.0

            # A piece is on the grid penalty lane if it was flagged as such, or
            # (for older CSVs) if its top rises above the green ceiling.
            if has_used_grid:
                is_grid = str(row.used_grid).strip().lower() in ("true", "1")
            else:
                ceil_w = green_by_idx.get(getattr(row, "start_slot_idx", None))
                is_grid = (
                    ceil_w is not None
                    and (float(row.stack_base_w) + float(row.power_w)) > ceil_w + 1.0
                )

            jid = int(row.job_id)
            color = color_map.get(jid) if color_map else None
            if color is None:
                color = hsv_to_rgb(((jid * 0.618033988749895) % 1.0, 0.65, 0.82))

            if is_grid:
                # Grid spill = carbon penalty: red, hatched, so it reads as a
                # cost rather than a normal green placement.
                grid_spill_seen = True
                rect = mpatches.Rectangle(
                    (x_num, y),
                    w,
                    h,
                    linewidth=0.8,
                    edgecolor=(0.90, 0.25, 0.21, 0.95),
                    facecolor=(0.80, 0.20, 0.18, 0.45),
                    hatch="///",
                )
            else:
                rect = mpatches.Rectangle(
                    (x_num, y),
                    w,
                    h,
                    linewidth=0.5,
                    edgecolor=(*color, 0.85),
                    facecolor=(*color, 0.65),
                )
            ax_obj.add_patch(rect)

            width_hours = (end - start).total_seconds() / 3600.0
            if h > 0.2 and width_hours > 0.25:
                ax_obj.text(
                    x_num + w / 2,
                    y + h / 2,
                    f"J{int(row.job_id)}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    fontweight="bold",
                    alpha=0.9,
                )

        p_start = slot_df["start_time"].iloc[0]
        p_end = slot_df["end_time"].iloc[-1]
        ax_obj.set_title(
            f"Forecast Plan #{resolved_pid} - {cluster_name}\n"
            f"{p_start:%Y-%m-%d %H:%M} to {p_end:%Y-%m-%d %H:%M} | {len(job_df)} planned jobs",
            fontsize=11, color="#f0f6fc", fontweight="bold"
        )
        ax_obj.set_xlabel("Time", fontsize=10, color="#c9d1d9")
        ax_obj.set_ylabel("Power (kW)", fontsize=10, color="#c9d1d9")
        ax_obj.set_xlim(mdates.date2num(p_start), mdates.date2num(p_end))
        max_y = max(envelope_kw) if envelope_kw else 0.0
        ax_obj.set_ylim(0, max_y * 1.08 if max_y > 0 else 1)

        ax_obj.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax_obj.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax_obj.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax_obj.grid(axis="y", alpha=0.18, linestyle="--", color="#484f58")
        ax_obj.grid(axis="x", alpha=0.08, linestyle=":", color="#484f58")
        ax_obj.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_obj.spines.values():
            spine.set_color("#30363d")
            spine.set_linewidth(0.5)
        handles, _ = ax_obj.get_legend_handles_labels()
        handles.append(
            mpatches.Patch(
                facecolor=hsv_to_rgb((0.55, 0.62, 0.84)),
                alpha=0.65,
                label="Compute (dynamic job power)",
            )
        )
        if grid_spill_seen:
            handles.append(
                mpatches.Patch(
                    facecolor=(0.80, 0.20, 0.18, 0.45),
                    edgecolor=(0.90, 0.25, 0.21, 0.95),
                    hatch="///",
                    label="Grid spill (carbon penalty)",
                )
            )
        ax_obj.legend(
            handles=handles,
            loc="upper right", fontsize=8,
            facecolor="#161b22", edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )

    def plot_cluster_all_forecasts_on_ax(ax_obj, slots_df, jobs_df, cluster_name, highlight_plan_id=None, color_map=None, real_launch=None):
        """Rolling forecast across ALL plans — latest plan wins per time slot.

        ``real_launch`` maps job_id -> actual dispatch datetime. A job that has
        already launched is frozen at the placement from the plan that was in
        force at its launch; later replans cannot move it. Jobs that never
        launched fall back to their last planned placement.
        """
        slot_df = slots_df[slots_df["cluster_type"] == cluster_name].copy()
        if slot_df.empty:
            ax_obj.text(0.5, 0.5, f"No forecast slots found for {cluster_name}",
                        ha="center", va="center", fontsize=11, color="gray")
            ax_obj.set_title(f"Forecast - {cluster_name} (No data)", fontsize=11, color="#f0f6fc", fontweight="bold")
            return

        # For each unique start_time keep the slot from the most recent plan
        slot_df = (
            slot_df.sort_values(["start_time", "plan_id"])
            .groupby("start_time", as_index=False)
            .last()
            .sort_values("start_time")
            .reset_index(drop=True)
        )

        times = list(slot_df["start_time"]) + [slot_df["end_time"].iloc[-1]]
        envelope_kw = list(slot_df["total_available_w"] / 1000.0) + [0.0]
        allocated_kw = list(slot_df["allocated_w"] / 1000.0) + [0.0]

        has_green = "green_available_w" in slot_df.columns
        if has_green:
            green_kw = list(slot_df["green_available_w"] / 1000.0) + [0.0]
            green_by_time = dict(zip(slot_df["start_time"], slot_df["green_available_w"]))
        else:
            green_kw = envelope_kw
            green_by_time = {}

        ax_obj.fill_between(times, envelope_kw, step="post", alpha=0.20, color="#1a3a5c", label="Free budget (unused)")
        if "idle_reserved_w" in slot_df.columns:
            idle_kw = list(slot_df["idle_reserved_w"] / 1000.0) + [0.0]
            ax_obj.fill_between(times, idle_kw, step="post", alpha=0.45, color="#6b4f2a", label="Idle power (nodes on)")
        ax_obj.step(times, envelope_kw, where="post", color="#4a9eff", linewidth=1.8)
        if has_green:
            ax_obj.step(times, green_kw, where="post", color="#3fb950", linewidth=1.6, alpha=0.95, label="Green board ceiling")
        ax_obj.step(times, allocated_kw, where="post", color="#f2cc60", linewidth=1.2, alpha=0.9, label="Planned allocated (idle+compute)")

        # Job rectangles: keep each job's LAST planned placement across all plans
        job_df = jobs_df[
            (jobs_df["cluster_type"] == cluster_name)
            & (jobs_df["placed"].astype(str).str.lower().isin(["true", "1"]))
        ].copy()
        job_df = job_df.dropna(subset=["start_time", "end_time"])
        if not job_df.empty:
            # Plan creation instant = the first slot start of each plan.
            plan_created = slots_df.groupby("plan_id")["start_time"].min()
            real_launch = real_launch or {}
            picked = []
            for jid, g in job_df.groupby("job_id"):
                g = g.copy()
                g["_pc"] = g["plan_id"].map(plan_created)
                launch = real_launch.get(int(jid))
                if launch is not None:
                    # Freeze at the plan in force when the job actually started:
                    # the latest plan created at/before the launch instant.
                    elig = g[g["_pc"] <= launch]
                    row = (elig if not elig.empty else g).sort_values("_pc").iloc[-1]
                else:
                    # Never launched -> show its final planned intent.
                    row = g.sort_values("plan_id").iloc[-1]
                picked.append(row)
            job_df = pd.DataFrame(picked).drop(columns=["_pc"], errors="ignore")

        has_used_grid = "used_grid" in job_df.columns
        grid_spill_seen = False
        for row in job_df.itertuples(index=False):
            start = row.start_time
            end = row.end_time
            if pd.isna(start) or pd.isna(end):
                continue
            x_num = mdates.date2num(start)
            w = mdates.date2num(end) - x_num
            y = float(row.stack_base_w) / 1000.0
            h = float(row.power_w) / 1000.0

            if has_used_grid:
                is_grid = str(row.used_grid).strip().lower() in ("true", "1")
            else:
                ceil_w = green_by_time.get(start)
                is_grid = ceil_w is not None and (float(row.stack_base_w) + float(row.power_w)) > ceil_w + 1.0

            jid = int(row.job_id)
            color = color_map.get(jid) if color_map else None
            if color is None:
                color = hsv_to_rgb(((jid * 0.618033988749895) % 1.0, 0.65, 0.82))

            if is_grid:
                grid_spill_seen = True
                rect = mpatches.Rectangle((x_num, y), w, h, linewidth=0.8,
                                          edgecolor=(0.90, 0.25, 0.21, 0.95), facecolor=(0.80, 0.20, 0.18, 0.45), hatch="///")
            else:
                rect = mpatches.Rectangle((x_num, y), w, h, linewidth=0.5,
                                          edgecolor=(*color, 0.85), facecolor=(*color, 0.65))
            ax_obj.add_patch(rect)

            width_hours = (end - start).total_seconds() / 3600.0
            if h > 0.2 and width_hours > 0.25:
                ax_obj.text(x_num + w / 2, y + h / 2, f"J{jid}", ha="center", va="center",
                            fontsize=5, color="white", fontweight="bold", alpha=0.9)

        if highlight_plan_id is not None:
            hp = slots_df[slots_df["plan_id"] == highlight_plan_id]
            if not hp.empty:
                ax_obj.axvspan(hp["start_time"].min(), hp["end_time"].max(),
                               color="#4a9eff", alpha=0.12, zorder=0, label=f"Plan #{highlight_plan_id} window")

        p_start = slot_df["start_time"].iloc[0]
        p_end = slot_df["end_time"].iloc[-1]
        n_plans = slots_df[slots_df["cluster_type"] == cluster_name]["plan_id"].nunique()
        ax_obj.set_title(
            f"Rolling forecast - {cluster_name}  ({n_plans} plans · {len(job_df)} placed jobs)",
            fontsize=11, color="#f0f6fc", fontweight="bold",
        )
        ax_obj.set_xlabel("Time", fontsize=10, color="#c9d1d9")
        ax_obj.set_ylabel("Power (kW)", fontsize=10, color="#c9d1d9")
        ax_obj.set_xlim(mdates.date2num(p_start), mdates.date2num(p_end))
        # Scale to the budget the puzzle actually fills — the green board ceiling
        # and the placed jobs — NOT total_available, whose grid-overflow lane
        # spikes far above and would squash everything into the panel floor.
        budget_series = green_kw if has_green else envelope_kw
        max_y = max(budget_series) if budget_series else 0.0
        if not job_df.empty:
            max_y = max(max_y, float((job_df["stack_base_w"] + job_df["power_w"]).max()) / 1000.0)
        ax_obj.set_ylim(0, max_y * 1.05 if max_y > 0 else 1)

        span_h = (p_end - p_start).total_seconds() / 3600.0
        if span_h > 72:
            ax_obj.xaxis.set_major_locator(mdates.DayLocator())
            ax_obj.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax_obj.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        elif span_h > 24:
            ax_obj.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax_obj.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax_obj.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        else:
            ax_obj.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
            ax_obj.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax_obj.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

        ax_obj.grid(axis="y", alpha=0.18, linestyle="--", color="#484f58")
        ax_obj.grid(axis="x", alpha=0.08, linestyle=":", color="#484f58")
        ax_obj.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_obj.spines.values():
            spine.set_color("#30363d")
            spine.set_linewidth(0.5)

        handles, _ = ax_obj.get_legend_handles_labels()
        handles.append(mpatches.Patch(facecolor=hsv_to_rgb((0.55, 0.62, 0.84)), alpha=0.65, label="Compute (dynamic job power)"))
        if grid_spill_seen:
            handles.append(mpatches.Patch(facecolor=(0.80, 0.20, 0.18, 0.45), edgecolor=(0.90, 0.25, 0.21, 0.95), hatch="///", label="Grid spill (carbon penalty)"))
        ax_obj.legend(handles=handles, loc="upper right", fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    def plot_cluster_real_on_ax(
        ax_obj,
        sched_obj,
        results_df,
        node_cfg,
        cluster_name,
        window_start=None,
        window_end=None,
        color_map=None,
        green_series=None,
    ) -> bool:
        plot_df = results_df
        if window_start is not None and window_end is not None:
            plot_df = results_df[
                (results_df.index >= window_start)
                & (results_df.index <= window_end)
            ]

        jobs_to_plot_list = []
        for job in sched_obj.job_queue.jobs:
            if job.status in ["running", "completed"] and job.start_time is not None:
                start = job.start_time
                if getattr(job, "completion_time", None) is not None:
                    end = job.completion_time
                else:
                    duration = job.actual_t_exec or job.remaining_time
                    end = start + timedelta(seconds=duration)

                if window_start is not None and window_end is not None:
                    if end < window_start or start > window_end:
                        continue
                    start = max(start, window_start)
                    end = min(end, window_end)
                    if end <= start:
                        continue

                jobs_to_plot_list.append((start, end, job))

        if not jobs_to_plot_list:
            ax_obj.text(0.5, 0.5, f"No executed jobs found for {cluster_name}",
                        ha="center", va="center", fontsize=11, color="gray")
            ax_obj.set_title(
                f"Real Gantt-Puzzle - {cluster_name} (No data)",
                fontsize=11,
                color="#e6edf3",
                fontweight="bold",
            )
            return True

        cfg = node_cfg.get(cluster_name, {})
        total_nodes = len(sched_obj.timeline)
        placements = pack_jobs_to_power(jobs_to_plot_list, cfg)

        budget_col = f"{cluster_name}_budget"
        if budget_col in plot_df.columns and not plot_df.empty:
            times = plot_df.index
            budget_kw = plot_df[budget_col].values / 1000.0
            ax_obj.step(
                times,
                budget_kw,
                where="post",
                color="#4a9eff",
                linewidth=1.8,
                alpha=0.9,
                label="Real budget",
            )
            ax_obj.fill_between(
                times,
                budget_kw,
                step="post",
                alpha=0.15,
                color="#1a3a5c",
                linewidth=0,
            )

        # Optional green "renewable board" reference so non-puzzle runs are
        # comparable to the puzzle's green ceiling: how much of the executed
        # load sat under renewable vs was pushed above it (grid).
        if green_series is not None:
            gs = green_series
            if window_start is not None and window_end is not None:
                gs = gs[(gs.index >= window_start) & (gs.index <= window_end)]
            if len(gs) and not plot_df.empty:
                ax_obj.step(
                    gs.index, gs.values / 1000.0, where="post",
                    color="#3fb950", linewidth=1.6, alpha=0.95,
                    label="Renewable board (est. cluster share)",
                )

        power_col = f"{cluster_name}_power"
        if power_col in plot_df.columns and not plot_df.empty:
            ax_obj.step(
                plot_df.index,
                plot_df[power_col].values / 1000.0,
                where="post",
                color="#f2cc60",
                linewidth=1.1,
                alpha=0.85,
                label="Measured power",
            )

        idle_per_node_kw = float(cfg.get("idle_power_w", 0.0)) / 1000.0
        real_grid_spill_seen = False
        for (y_start, start, end, job, height_kw) in placements:
            start_num = mdates.date2num(start)
            w = mdates.date2num(end) - start_num

            jid = int(job.job_id)
            rgb = color_map.get(jid) if color_map else None
            if rgb is None:
                rgb = hsv_to_rgb(((jid * 0.618033988749895) % 1.0, 0.65, 0.82))

            # Jobs the plan powered from the grid penalty lane are drawn red and
            # hatched so the carbon cost is visible in execution too.  Matches
            # both "grid" (dispatched at the planned slot) and "front_run_grid"
            # (grid-planned job anticipated by the dispatcher).
            is_grid = str(
                getattr(job, "planned_energy_source", "")
            ).lower().endswith("grid")
            if is_grid:
                real_grid_spill_seen = True

            # Each bar is a running node's full power: split it into the
            # idle baseline (brown, bottom) and dynamic compute (job hue,
            # top), matching the forecast panel's idle/compute layering.
            idle_h = min(height_kw, idle_per_node_kw * getattr(job, "num_nodes", 0))
            if idle_h > 0:
                ax_obj.add_patch(mpatches.Rectangle(
                    (start_num, y_start), w, idle_h,
                    linewidth=0.0, facecolor=(0.42, 0.31, 0.16, 0.75),
                ))
            if is_grid:
                ax_obj.add_patch(mpatches.Rectangle(
                    (start_num, y_start + idle_h),
                    w,
                    max(0.0, height_kw - idle_h),
                    linewidth=0.8,
                    edgecolor=(0.90, 0.25, 0.21, 0.95),
                    facecolor=(0.80, 0.20, 0.18, 0.45),
                    hatch="///",
                ))
            else:
                ax_obj.add_patch(mpatches.FancyBboxPatch(
                    (start_num, y_start + idle_h),
                    w,
                    max(0.0, height_kw - idle_h),
                    boxstyle="round,pad=0,rounding_size=0.001",
                    linewidth=0.5,
                    edgecolor=(*rgb, 0.7),
                    facecolor=(*rgb, 0.65),
                ))

            width_hours = (end - start).total_seconds() / 3600.0
            if height_kw > 0.2 and width_hours > 0.5:
                cx = start_num + w / 2
                cy = y_start + height_kw / 2
                ax_obj.text(
                    cx, cy, f"J{job.job_id}",
                    ha="center", va="center",
                    fontsize=5, color="white", fontweight="bold",
                    alpha=0.9
                )

        ax_obj.set_xlabel("Time", fontsize=10, color="#c9d1d9")
        ax_obj.set_ylabel("Power (kW)", fontsize=10, color="#c9d1d9")
        
        max_y = 1.0
        if budget_col in plot_df.columns and not plot_df.empty:
            max_y = max(max_y, float(plot_df[budget_col].max()) / 1000.0)
        if power_col in plot_df.columns and not plot_df.empty:
            max_y = max(max_y, float(plot_df[power_col].max()) / 1000.0)
        if placements:
            max_job_top = max(
                y_start + height_kw
                for y_start, _, _, _, height_kw in placements
            )
            max_y = max(max_y, max_job_top)
            
        ax_obj.set_ylim(0, max_y * 1.05)
        x_start = window_start if window_start is not None else plot_df.index[0]
        x_end = window_end if window_end is not None else plot_df.index[-1]
        ax_obj.set_xlim(mdates.date2num(x_start), mdates.date2num(x_end))

        # Adaptive tick density: a fixed 1h minor locator overflows matplotlib's
        # 1000-tick cap on long (multi-week) windows, so scale the interval to
        # the span shown.
        span_h = (mdates.date2num(x_end) - mdates.date2num(x_start)) * 24.0
        if span_h <= 36:
            major_h, minor_h = 6, 1
        elif span_h <= 24 * 14:
            major_h, minor_h = 24, 6
        else:
            major_h, minor_h = 24 * 7, 24
        ax_obj.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
        ax_obj.xaxis.set_major_locator(mdates.HourLocator(interval=major_h))
        ax_obj.xaxis.set_minor_locator(mdates.HourLocator(interval=minor_h))

        ax_obj.grid(axis="y", alpha=0.15, linestyle="--", color="#484f58")
        ax_obj.grid(axis="x", alpha=0.08, linestyle=":", color="#484f58")
        ax_obj.tick_params(colors="#8b949e", labelsize=8)

        for spine in ax_obj.spines.values():
            spine.set_color("#30363d")
            spine.set_linewidth(0.5)

        ax_obj.set_title(
            f"Real Gantt-Puzzle - {cluster_name}\n"
            f"{len(jobs_to_plot_list)} jobs in forecast window | "
            f"Capacity: {total_nodes} nodes",
            fontsize=11, color="#e6edf3", fontweight="bold"
        )

        handles, _ = ax_obj.get_legend_handles_labels()
        handles.append(mpatches.Patch(
            facecolor=(0.42, 0.31, 0.16, 0.75), label="Idle power (nodes on)",
        ))
        handles.append(mpatches.Patch(
            facecolor=hsv_to_rgb((0.55, 0.65, 0.80)), alpha=0.65,
            label="Compute (dynamic job power)",
        ))
        if real_grid_spill_seen:
            handles.append(mpatches.Patch(
                facecolor=(0.80, 0.20, 0.18, 0.45),
                edgecolor=(0.90, 0.25, 0.21, 0.95),
                hatch="///",
                label="Grid spill (carbon penalty)",
            ))
        ax_obj.legend(
            handles=handles,
            loc="upper right", fontsize=8,
            facecolor="#161b22", edgecolor="#30363d",
            labelcolor="#c9d1d9",
        )
        return True

    # -- 2. Main Logic -----------------------------------------------------
    # Load the heavy inputs ONCE (schedulers.pkl can be hundreds of MB and the
    # energy CSV tens of thousands of rows), then render every requested plan
    # from the in-memory data. This keeps multi-plan / "all" runs cheap: the
    # cost scales with the number of figures drawn, not with re-reading disk.
    try:
        slots, jobs = load_forecast(results_dir)
        forecast_available = True
    except FileNotFoundError:
        slots, jobs = None, None
        forecast_available = False

    schedulers = load_schedulers(results_dir)
    energy_df = load_results_df(results_dir)
    node_config = load_node_config(results_dir)

    # Build one shared color map from ALL job IDs that appear in either panel
    # so the same job always gets exactly the same colour regardless of which
    # panel renders it or what order it is encountered in.
    _all_forecast_ids = (
        set(jobs["job_id"].dropna().astype(int).tolist())
        if forecast_available else set()
    )
    _all_real_ids = set(
        int(job.job_id)
        for sched in schedulers
        for job in sched.job_queue.jobs
        if getattr(job, "job_id", None) is not None
    )
    _all_job_ids = sorted(_all_forecast_ids | _all_real_ids)
    job_color_map: dict[int, tuple] = {
        jid: hsv_to_rgb(((jid * 0.618033988749895) % 1.0, 0.65, 0.82))
        for jid in _all_job_ids
    }

    plt.style.use("dark_background")

    sched_map = {s.cluster_type: s for s in schedulers}
    clusters = forecast_clusters(slots) if forecast_available else sorted(sched_map)
    valid_clusters = [c for c in clusters if c in sched_map]
    if not valid_clusters:
        print("No valid clusters found to plot combined puzzle.")
        return

    def cluster_green_series(cluster: str):
        """Estimated renewable board (W) for a cluster over the whole sim.

        System green = renewable + battery portion of the scheduler budget,
        split across clusters by their share of the total per-cluster budget.
        Approximate (the per-cluster renewable split is not persisted), but a
        fair reference line for comparing how each policy tracks renewables.
        """
        if "budget_renewable_w" not in energy_df.columns:
            return None
        sys_green = energy_df["budget_renewable_w"].fillna(0.0)
        if "budget_battery_w" in energy_df.columns:
            sys_green = sys_green + energy_df["budget_battery_w"].fillna(0.0)
        budget_cols = [
            f"{c}_budget" for c in valid_clusters
            if f"{c}_budget" in energy_df.columns
        ]
        ccol = f"{cluster}_budget"
        if not budget_cols or ccol not in energy_df.columns:
            return sys_green / max(1, len(valid_clusters))
        total = energy_df[budget_cols].sum(axis=1).replace(0.0, np.nan)
        share = (energy_df[ccol].fillna(0.0) / total).fillna(0.0)
        return sys_green * share

    # Real-only mode: no forecast (non-puzzle run) or explicitly requested.
    if real_only or not forecast_available:
        win_start, win_end = energy_df.index.min(), energy_df.index.max()
        M = len(valid_clusters)
        # Wide aspect: these panels are multi-day (sometimes multi-week) time
        # series, so favour horizontal room for readability.
        real_width = max(figsize[0] * 1.6, 34.0)
        fig, axes = plt.subplots(
            M, 1, figsize=(real_width, 4.8 * M),
            facecolor="#0d1117", squeeze=False,
        )
        for idx, cluster in enumerate(valid_clusters):
            ax = axes[idx, 0]
            ax.set_facecolor("#0d1117")
            plot_cluster_real_on_ax(
                ax, sched_map[cluster], energy_df, node_config, cluster,
                window_start=win_start, window_end=win_end,
                color_map=job_color_map, green_series=cluster_green_series(cluster),
            )
        fig.suptitle(
            "Real Gantt-Puzzle (execution) — green line = renewable board",
            color="#e6edf3", fontsize=16, fontweight="bold", y=0.99,
        )
        fig.tight_layout(pad=3.0)
        out = os.path.join(results_dir, "real_puzzle.png")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        if show:
            plt.show()
        else:
            plt.close(fig)
        print("✅ Real puzzle plot saved to: real_puzzle.png")
        return

    target_plan_ids = resolve_plan_ids(slots, plan_id, results_dir)

    # A single targeted plan keeps the canonical filename for back-compat with
    # callers (plot_energy.py) that expect puzzle_combined.png; multiple plans
    # get per-plan suffixed filenames so they don't overwrite each other.
    single = len(target_plan_ids) == 1

    def render_plan(selected_plan_id: int) -> str:
        selected_slots = slots[
            slots["plan_id"] == selected_plan_id
        ].sort_values("slot_idx")
        if selected_slots.empty:
            raise ValueError(f"Plan {selected_plan_id} has no forecast slots.")
        plan_start = selected_slots["start_time"].min()
        plan_end = selected_slots["end_time"].max()

        # The execution (right) column spans the WHOLE simulation, not just this
        # plan's 24h window, so the full run is visible; the plan's window is
        # shaded on it to keep the plan↔execution correspondence. The forecast
        # (left) column stays the 24h lookahead — that is all a single plan is.
        sim_start = energy_df.index.min()
        sim_end = energy_df.index.max()
        sim_days = max(1.0, (sim_end - sim_start).total_seconds() / 86400.0)

        M = len(valid_clusters)
        # Two rows per cluster: forecast (24h plan) on top, execution below.
        # Both rows get the SAME height so plan and real are the same size.
        row_h = 5.5  # inches per panel — tall enough to actually read the puzzle
        h_ratios = [1.0, 1.0] * M
        fig_w = max(28.0, 20.0 + min(sim_days, 10.0) * 1.4)
        fig, axes = plt.subplots(
            2 * M, 1,
            figsize=(fig_w, row_h * 2 * M),
            facecolor="#0d1117",
            squeeze=False,
            gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
        )

        for idx, cluster in enumerate(valid_clusters):
            scheduler = sched_map[cluster]
            ax1 = axes[2 * idx, 0]      # forecast on top
            ax2 = axes[2 * idx + 1, 0]  # execution below

            ax1.set_facecolor("#0d1117")
            ax2.set_facecolor("#0d1117")

            # Real dispatch instants, so already-launched jobs are frozen in the
            # rolling forecast at the placement they actually committed to.
            real_launch = {}
            for job in getattr(scheduler.job_queue, "jobs", []):
                if getattr(job, "start_time", None) is not None and job.status in ("running", "completed"):
                    real_launch[int(job.job_id)] = job.start_time

            plot_cluster_all_forecasts_on_ax(
                ax1, slots, jobs, cluster,
                highlight_plan_id=selected_plan_id,
                color_map=job_color_map,
                real_launch=real_launch,
            )
            plot_cluster_real_on_ax(
                ax2,
                scheduler,
                energy_df,
                node_config,
                cluster,
                window_start=sim_start,
                window_end=sim_end,
                color_map=job_color_map,
                green_series=cluster_green_series(cluster),
            )
            # Shade this plan's 24h window on the full-run execution panel.
            ax2.axvspan(
                plan_start, plan_end,
                color="#4a9eff", alpha=0.10, zorder=0,
                label=f"Plan #{selected_plan_id} window",
            )
            # Same vertical (power) scale for THIS cluster's plan and execution,
            # so the planned puzzle height and the real one line up. Each cluster
            # keeps its own range — arm dwarfs gpu, so a shared scale is useless.
            y_top = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(0, y_top)
            ax2.set_ylim(0, y_top)
            # Match horizontal (time) span: both panels cover the full sim,
            # so a job sits at the same x in plan and execution.
            ax1.set_xlim(sim_start, sim_end)
            ax2.set_xlim(sim_start, sim_end)

        fig.suptitle(
            f"Puzzle Scheduling: 24h Plan #{selected_plan_id} "
            f"({plan_start:%b %d %H:%M} → {plan_end:%b %d %H:%M})  •  "
            f"full {sim_days:.1f}-day execution",
            color="#e6edf3",
            fontsize=16,
            fontweight="bold",
            y=1.01,
        )

        fig.tight_layout(pad=2.5)

        fname = (
            "puzzle_combined.png" if single
            else f"puzzle_combined_plan{selected_plan_id:02d}.png"
        )
        fig.savefig(
            os.path.join(results_dir, fname),
            dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(),
        )
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fname

    print(
        f"Generating combined Puzzle plot(s) for {len(target_plan_ids)} "
        f"plan(s): {target_plan_ids}..."
    )
    for selected_plan_id in target_plan_ids:
        fname = render_plan(selected_plan_id)
        print(f"✅ Combined Puzzle plot saved to: {fname}")


def plot_combined_gantt_chart(
    schedulers: list['JobQueueScheduler'],
    output_file: str = "gantt_combined.png",
    max_nodes: int = 20
):
    """
    Create a combined Gantt chart showing job execution timeline for all clusters.
    Auto-limits each cluster to max_nodes (default 20) for readability.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import os

    M = len(schedulers)
    if M == 0:
        print("No schedulers to plot combined Gantt chart.")
        return

    fig, axes = plt.subplots(M, 1, figsize=(16, max(6 * M, 3 * M)), sharex=True)
    if M == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    all_earliest_times = []
    for scheduler in schedulers:
        nodes_with_work = [
            name for name in sorted(scheduler.timeline.keys())
            if len(scheduler.timeline[name]) > 0
        ]
        nodes_to_plot = nodes_with_work[:max_nodes]
        for node_name in nodes_to_plot:
            for start_time, end_time, job_id in scheduler.timeline[node_name]:
                all_earliest_times.append(start_time)
                if end_time:
                    all_earliest_times.append(end_time)

    if not all_earliest_times:
        print("No timeline data to plot for any cluster.")
        plt.close(fig)
        return

    ref_time = min(all_earliest_times)

    for idx, scheduler in enumerate(schedulers):
        ax = axes[idx]
        nodes_with_work = [
            name for name in sorted(scheduler.timeline.keys())
            if len(scheduler.timeline[name]) > 0
        ]
        nodes_to_plot = nodes_with_work[:max_nodes]

        for node_idx, node_name in enumerate(nodes_to_plot):
            for start_time, end_time, job_id in scheduler.timeline[node_name]:
                if end_time is None:
                    continue
                start_hours = (start_time - ref_time).total_seconds() / 3600
                end_hours = (end_time - ref_time).total_seconds() / 3600
                duration_hours = end_hours - start_hours

                if job_id is None:
                    ax.barh(node_idx, duration_hours, left=start_hours, height=0.8,
                           color='lightgray', edgecolor='black', linewidth=0.3)
                else:
                    color = colors[job_id % len(colors)]
                    ax.barh(node_idx, duration_hours, left=start_hours, height=0.8,
                           color=color, edgecolor='black', linewidth=0.3)

        ax.set_yticks(range(len(nodes_to_plot)))
        ax.set_yticklabels(nodes_to_plot, fontsize=8)
        ax.set_ylabel('Nodes', fontsize=10)

        total_nodes = len(scheduler.timeline)
        shown = len(nodes_to_plot)
        title = f'{scheduler.cluster_type} — Gantt Chart ({shown}/{total_nodes} nodes shown)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        idle_patch = mpatches.Patch(color='lightgray', label='Idle')
        job_patch = mpatches.Patch(color=colors[0], label='Job Execution')
        ax.legend(handles=[idle_patch, job_patch], loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (hours)', fontsize=12)

    fig.suptitle('Supercomputer Gantt Charts', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Combined Gantt chart saved to: {os.path.basename(output_file)}")
    plt.close(fig)
