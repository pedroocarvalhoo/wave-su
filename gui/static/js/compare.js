/* ═══════════════════════════════════════════════════════════════════════════
   WAVE GUI — Compare Tab
   Multi-run comparison with metrics table + side-by-side chart gallery
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Load Compare Selector ───────────────────────────────────────────────────

async function loadCompareSelector() {
    try {
        const data = await API.get('/api/results');
        AppState.results = data.results;
        renderSupercomputerSidebars();
        renderCompareSelector();
    } catch (err) {
        document.getElementById('compare-selector').innerHTML =
            `<div class="empty-state"><span class="title">Failed to load results</span></div>`;
    }
}

function renderCompareSelector() {
    const selector = document.getElementById('compare-selector');
    const runs = runsForSelectedSupercomputer();
    const visibleIds = new Set(runs.map(run => run.id));
    AppState.selectedCompareRuns = AppState.selectedCompareRuns.filter(id => visibleIds.has(id));
    const label = AppState.activeSystem ? supercomputerLabel(AppState.activeSystem) : 'Selected supercomputer';
    document.getElementById('compare-system-context').textContent = `${label} / ${runs.length} runs available`;

    if (!runs.length) {
        selector.innerHTML = `<div class="empty-state">
            <span class="title">No runs for ${escapeHtml(label)}</span>
            <span class="subtitle">Run a simulation for this supercomputer to start comparing results.</span>
        </div>`;
    } else {
        selector.innerHTML = runs.map(run => {
            const index = AppState.selectedCompareRuns.indexOf(run.id);
            const metrics = run.summary_metrics || {};
            const policies = [run.scheduling_policy, run.dvfs_policy, run.budget_policy].filter(Boolean);
            return `<button type="button" class="compare-checkbox-card ${index >= 0 ? 'selected' : ''}"
                         data-run-id="${escapeHtml(run.id)}" aria-pressed="${index >= 0}">
                ${index >= 0 ? `<span class="compare-badge">${index + 1}</span>` : '<span class="compare-badge-empty"></span>'}
                <span class="card-content">
                    <span class="card-system">${escapeHtml(run.label || run.id)}</span>
                    <span class="card-details">${escapeHtml(policies.join(' / '))}</span>
                    <span class="card-details">${metrics.total_co2_kg != null ? `${formatNumber(metrics.total_co2_kg)} kg CO₂` : ''}
                        ${metrics.total_jobs_completed != null ? ` / ${metrics.total_jobs_completed} jobs` : ''}</span>
                </span>
            </button>`;
        }).join('');
        selector.querySelectorAll('[data-run-id]').forEach(button => {
            button.addEventListener('click', () => toggleCompareRun(button.dataset.runId));
        });
    }
    updateCompareButton();
}

function toggleCompareRun(runId) {
    const index = AppState.selectedCompareRuns.indexOf(runId);
    if (index >= 0) AppState.selectedCompareRuns.splice(index, 1);
    else AppState.selectedCompareRuns.push(runId);
    document.getElementById('compare-results').classList.add('hidden');
    renderCompareSelector();
}


function updateCompareButton() {
    const btn = document.getElementById('btn-compare');
    const count = AppState.selectedCompareRuns.length;
    btn.disabled = count < 2;
    btn.textContent = count < 2
        ? `Select at least 2 runs (${count} selected)`
        : `Compare ${count} Runs`;
}


// ── Run Comparison ──────────────────────────────────────────────────────────

async function runComparison() {
    const runIds = [...AppState.selectedCompareRuns];
    const system = AppState.activeSystem;

    if (runIds.length < 2) {
        showToast('Select at least 2 runs', 'warning');
        return;
    }

    try {
        const data = await API.post('/api/results/compare', { run_ids: runIds });
        if (system !== AppState.activeSystem || runIds.join() !== AppState.selectedCompareRuns.join()) return;
        renderComparison(data.comparisons.filter(run => run.supercomputer === system));

        const resultsEl = document.getElementById('compare-results');
        resultsEl.classList.remove('hidden');

        // Auto-scroll to comparison area smoothly, accounting for sticky header offset
        setTimeout(() => {
            const headerOffset = document.querySelector('.header').offsetHeight
                + document.getElementById('main-nav').offsetHeight + 16;
            const elementPosition = resultsEl.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }, 150);
    } catch (err) {
        showToast('Comparison failed: ' + err.message, 'error');
    }
}


// ── Render Comparison ───────────────────────────────────────────────────────

let compareData = [];
let compareSortMetric = null;
let compareSortAsc = true;

function renderComparison(comparisons) {
    compareData = comparisons;
    renderCompareMetrics(comparisons);
    renderCompareImages(comparisons);
}


// ── Metrics Table ───────────────────────────────────────────────────────────

const COMPARE_METRICS = [
    { key: 'total_co2_kg',           label: 'Total CO₂ (kg)',           lower_better: true,  format: v => formatMetricNumber(v, 2) },
    { key: 'total_energy_kwh',       label: 'Total Energy (kWh)',       lower_better: true,  format: v => formatMetricNumber(v, 2) },
    { key: 'total_jobs_completed',   label: 'Jobs Completed',           lower_better: false, format: v => v?.toLocaleString() || '—' },
    { key: 'p95_wait_time_s', label: 'P95 Wait Time', lower_better: true, format: v => formatDuration(v) },
    { key: 'deadline_misses', label: 'Waiting-bound misses', lower_better: true, format: v => v ?? '—' },
    { key: 'physical_grid_import_kwh', label: 'Grid import (kWh)', lower_better: true, format: v => formatMetricNumber(v, 2) },
    { key: 'avg_wait_time_s',        label: 'Avg Wait Time (s)',        lower_better: true,  format: v => formatDuration(v) },
    { key: 'avg_stretch_factor',     label: 'Avg Stretch Factor',       lower_better: true,  format: v => v?.toFixed(3) || '—' },
    { key: 'co2_per_job',            label: 'CO₂ per Job (g)',          lower_better: true,  format: v => v?.toFixed(2) || '—', derived: true },
    { key: 'energy_per_job',         label: 'Energy per Job (kWh)',     lower_better: true,  format: v => v?.toFixed(3) || '—', derived: true },
];


function renderCompareMetrics(comparisons) {
    const container = document.getElementById('compare-table-container');

    // Enrich with derived metrics
    comparisons.forEach(c => {
        const m = c.summary_metrics || {};
        const jobs = m.total_jobs_completed;
        m.co2_per_job = m.total_co2_kg != null && jobs > 0 ? (m.total_co2_kg * 1000) / jobs : null;
        m.energy_per_job = jobs > 0 && m.total_energy_kwh != null ? m.total_energy_kwh / jobs : null;
    });

    // Sort comparisons if a metric is selected
    let ordered = [...comparisons];
    if (compareSortMetric) {
        const metricDef = COMPARE_METRICS.find(m => m.key === compareSortMetric);
        ordered.sort((a, b) => {
            const va = (a.summary_metrics || {})[compareSortMetric] ?? Infinity;
            const vb = (b.summary_metrics || {})[compareSortMetric] ?? Infinity;
            return compareSortAsc ? va - vb : vb - va;
        });
    }

    // Build table
    let html = '<table class="compare-table"><thead><tr>';
    html += '<th class="metric-name">Metric</th>';
    ordered.forEach(c => {
        const shortName = c.label || c.config.run_tag || c.id;
        const policies = `${c.scheduling_policy} / ${c.budget_policy}`;
        html += `<th onclick="sortCompareByColumn('${c.id}')" title="${escapeHtml(policies)}">
            ${escapeHtml(shortName)}<br><span style="font-size:0.7rem;color:var(--text-muted)">${c.timestamp ? formatTimestamp(c.timestamp) : ''}</span>
        </th>`;
    });
    html += '</tr></thead><tbody>';

    // Each metric row
    COMPARE_METRICS.forEach(metric => {
        html += '<tr>';
        const sortIndicator = compareSortMetric === metric.key
            ? (compareSortAsc ? ' ▲' : ' ▼')
            : '';
        html += `<td class="metric-name" onclick="sortCompareByMetric('${metric.key}')" style="cursor:pointer">
            ${escapeHtml(metric.label)}${sortIndicator}
        </td>`;

        // Get all values for this metric
        const values = ordered.map(c => (c.summary_metrics || {})[metric.key]);
        const numValues = values.filter(v => v != null && !isNaN(v));
        const minVal = numValues.length ? Math.min(...numValues) : null;
        const maxVal = numValues.length ? Math.max(...numValues) : null;

        ordered.forEach((c, i) => {
            const val = values[i];
            const formatted = val != null ? metric.format(val) : '—';

            let cls = '';
            if (val != null && numValues.length > 1 && minVal !== maxVal) {
                if (metric.lower_better) {
                    if (val === minVal) cls = 'best';
                    if (val === maxVal) cls = 'worst';
                } else {
                    if (val === maxVal) cls = 'best';
                    if (val === minVal) cls = 'worst';
                }
            }

            // Inline bar
            let barWidth = 0;
            if (val != null && maxVal > 0) {
                barWidth = Math.round((val / maxVal) * 80);
            }

            html += `<td class="${cls}">
                <span class="compare-bar" style="width:${barWidth}px"></span>
                ${formatted}
            </td>`;
        });

        html += '</tr>';
    });

    for (const [key, label, badge] of [
        ['scheduling_policy', 'Scheduling Policy', 'blue'],
        ['dvfs_policy', 'DVFS Policy', 'purple'],
        ['budget_policy', 'Budget Policy', 'amber'],
        ['alloc_strategy', 'Allocation Strategy', 'cyan'],
    ]) {
        if (!ordered.some(run => run[key])) continue;
        html += `<tr><td class="metric-name">${label}</td>`;
        ordered.forEach(run => {
            html += `<td>${run[key] ? `<span class="badge badge-${badge}">${escapeHtml(run[key])}</span>` : '—'}</td>`;
        });
        html += '</tr>';
    }

    html += '</tbody></table>';
    container.innerHTML = html;
}


function sortCompareByMetric(metricKey) {
    if (compareSortMetric === metricKey) {
        compareSortAsc = !compareSortAsc;
    } else {
        compareSortMetric = metricKey;
        compareSortAsc = true;
    }
    renderCompareMetrics(compareData);
}


// ── Chart Comparison (Side-by-Side Images) ──────────────────────────────────

function renderCompareImages(comparisons) {
    const grid = document.getElementById('compare-images-grid');

    // Collect all unique image names across runs
    const allImages = new Set();
    comparisons.forEach(c => (c.images || []).forEach(img => allImages.add(img)));

    if (allImages.size === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <span class="icon">📈</span>
                <span class="title">No charts available</span>
            </div>
        `;
        return;
    }

    // Group by image name, show side-by-side
    let html = '';
    allImages.forEach(imgName => {
        html += `
            <div class="compare-image-row">
                <div class="compare-image-row-title">${escapeHtml(imgName)}</div>
                <div class="compare-image-row-images">
                    ${comparisons.map(c => {
                        const hasImage = (c.images || []).includes(imgName);
                        if (hasImage) {
                            const shortLabel = c.label || c.config.run_tag || c.id;
                            return `
                                <div class="compare-image-cell">
                                    <img src="/api/results/${c.id}/images/${imgName}"
                                         alt="${imgName}"
                                         loading="lazy"
                                         onclick="openImageModal(this.src)">
                                    <div class="cell-label">${escapeHtml(shortLabel)}<br>${c.timestamp ? formatTimestamp(c.timestamp) : ''}</div>
                                </div>
                            `;
                        } else {
                            return `
                                <div class="compare-image-cell">
                                    <div style="display:flex;align-items:center;justify-content:center;height:200px;
                                                background:var(--bg-surface);border-radius:var(--radius-sm);
                                                color:var(--text-muted);font-size:0.85rem;">
                                        Not available
                                    </div>
                                    <div class="cell-label">${escapeHtml(c.label || c.config.run_tag || c.id)}</div>
                                </div>
                            `;
                        }
                    }).join('')}
                </div>
            </div>
        `;
    });

    grid.innerHTML = html;
}


// ── View Toggle ─────────────────────────────────────────────────────────────

function showCompareView(view) {
    const metricsView = document.getElementById('compare-metrics-view');
    const imagesView = document.getElementById('compare-images-view');
    const metricsTab = document.getElementById('compare-tab-metrics');
    const imagesTab = document.getElementById('compare-tab-images');

    if (view === 'metrics') {
        metricsView.classList.remove('hidden');
        imagesView.classList.add('hidden');
        metricsTab.classList.add('active');
        imagesTab.classList.remove('active');
    } else {
        metricsView.classList.add('hidden');
        imagesView.classList.remove('hidden');
        metricsTab.classList.remove('active');
        imagesTab.classList.add('active');
    }
}
