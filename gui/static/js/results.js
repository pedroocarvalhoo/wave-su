/* ═══════════════════════════════════════════════════════════════════════════
   WAVE GUI — Results Tab
   Browse results grouped by supercomputer, sortable by any metric,
   view plots, paginated CSV tables
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Results sort state ──────────────────────────────────────────────────────
let resultsSortMetric = '';  // '' = date sort
let resultsSortAsc = false;  // false = descending (high→low or newest first)

// ── Load Results List ───────────────────────────────────────────────────────

async function loadResults() {
    const container = document.getElementById('results-grouped-container');
    container.innerHTML = `
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
        <div class="skeleton skeleton-card"></div>
    `;

    try {
        const data = await API.get('/api/results');
        AppState.results = data.results;
        renderSupercomputerSidebars();
        renderResultsForSystem(data.results);
    } catch (err) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-circle-exclamation empty-icon"></i>
                <span class="title">Failed to load results</span>
                <span class="subtitle">${escapeHtml(err.message)}</span>
            </div>
        `;
    }
}


// ── Sort controls ───────────────────────────────────────────────────────────

function initResultsSort() {
    const select = document.getElementById('results-sort-metric');
    if (select) {
        select.addEventListener('change', () => {
            resultsSortMetric = select.value;
            if (AppState.results) renderResultsForSystem(AppState.results);
        });
    }
}

function toggleResultsSortDir() {
    resultsSortAsc = !resultsSortAsc;
    const btn = document.getElementById('results-sort-dir');
    if (btn) {
        const arrow = resultsSortAsc
            ? '<i class="fa-solid fa-arrow-up-wide-short btn-icon-svg"></i>'
            : '<i class="fa-solid fa-arrow-down-wide-short btn-icon-svg"></i>';
        btn.innerHTML = `${arrow} ${resultsSortAsc ? 'Low → High' : 'High → Low'}`;
    }
    if (AppState.results) renderResultsForSystem(AppState.results);
}

// Init sort controls on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initResultsSort();
});


// ── Sort results ────────────────────────────────────────────────────────────

function sortResults(results) {
    const sorted = [...results];

    if (!resultsSortMetric) {
        // Sort by timestamp
        sorted.sort((a, b) => {
            const ta = a.timestamp || '';
            const tb = b.timestamp || '';
            return resultsSortAsc ? ta.localeCompare(tb) : tb.localeCompare(ta);
        });
    } else {
        sorted.sort((a, b) => {
            const va = (a.summary_metrics || {})[resultsSortMetric];
            const vb = (b.summary_metrics || {})[resultsSortMetric];
            const na = va != null ? parseFloat(va) : (resultsSortAsc ? Infinity : -Infinity);
            const nb = vb != null ? parseFloat(vb) : (resultsSortAsc ? Infinity : -Infinity);
            return resultsSortAsc ? na - nb : nb - na;
        });
    }

    return sorted;
}


// ── Results for the selected supercomputer ──────────────────────────────────

function renderResultsForSystem(results) {
    const container = document.getElementById('results-grouped-container');
    const system = ensureSupercomputerSelection();
    const runs = results.filter(run => run.supercomputer === system);
    const label = system ? supercomputerLabel(system) : 'Selected supercomputer';
    document.getElementById('results-system-context').textContent = `${label} / ${runs.length} runs`;

    if (!runs.length) {
        container.innerHTML = `<div class="empty-state">
            <i class="fa-solid fa-chart-line empty-icon"></i>
            <span class="title">No results for ${escapeHtml(label)}</span>
            <span class="subtitle">Start a simulation for this supercomputer to see its results here.</span>
        </div>`;
        return;
    }
    container.innerHTML = `<div class="results-grid">
        ${sortResults(runs).map(run => renderResultCard(run)).join('')}
    </div>`;
}


function renderResultCard(r) {
    const m = r.summary_metrics || {};
    return `
        <div class="result-card" onclick="showResultDetail('${r.id}')">
            <div class="result-card-header">
                <div class="result-card-system">${escapeHtml(r.label || r.supercomputer)}</div>
                ${r.timestamp ? `<div class="result-card-date">${formatTimestamp(r.timestamp)}</div>` : ''}
            </div>
            <div class="result-card-policies">
                ${r.scheduling_policy ? `<span class="badge badge-blue">${escapeHtml(r.scheduling_policy)}</span>` : ''}
                ${r.dvfs_policy ? `<span class="badge badge-purple">${escapeHtml(r.dvfs_policy)}</span>` : ''}
                ${r.budget_policy ? `<span class="badge badge-amber">${escapeHtml(r.budget_policy)}</span>` : ''}
                ${r.alloc_strategy ? `<span class="badge badge-cyan">${escapeHtml(r.alloc_strategy)}</span>` : ''}
            </div>
            <div class="result-card-metrics">
                <div class="metric-item">
                    <span class="metric-label">CO₂</span>
                    <span class="metric-value">${formatNumber(m.total_co2_kg)} kg</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Energy</span>
                    <span class="metric-value">${formatNumber(m.total_energy_kwh)} kWh</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Jobs</span>
                    <span class="metric-value">${m.total_jobs_completed || '—'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">P95 Wait</span>
                    <span class="metric-value">${formatDuration(m.p95_wait_time_s)}</span>
                </div>
            </div>
        </div>
    `;
}


// ── Result Detail View ──────────────────────────────────────────────────────

async function showResultDetail(runId) {
    const knownRun = AppState.results.find(run => run.id === runId);
    if (knownRun && knownRun.supercomputer !== AppState.activeSystem) {
        selectSupercomputer(knownRun.supercomputer);
    }
    AppState.currentResultId = runId;

    document.getElementById('results-list-view').classList.add('hidden');
    document.getElementById('result-detail-view').classList.remove('hidden');

    try {
        const result = await API.get(`/api/results/${runId}`);
        if (AppState.currentResultId === runId) renderResultDetail(result);
    } catch (err) {
        showToast('Failed to load result: ' + err.message, 'error');
    }
}


function showResultsList() {
    document.getElementById('result-detail-view').classList.add('hidden');
    document.getElementById('results-list-view').classList.remove('hidden');
    AppState.currentResultId = null;
}


function renderResultDetail(result) {
    // Title
    document.getElementById('result-detail-title').innerHTML = `
        <i class="fa-solid fa-server card-icon"></i>
        ${escapeHtml(result.label || result.supercomputer)} ${result.timestamp ? formatTimestamp(result.timestamp) : ''}
    `;

    // Config grid
    const cfg = result.config;
    const configGrid = document.getElementById('result-config-grid');
    const configItems = [
        ['Scheduling Policy', cfg.scheduling_policy],
        ['DVFS Policy', cfg.dvfs_policy],
        ['Budget Policy', cfg.budget_policy],
        ['Allocation Strategy', cfg.alloc_strategy],
        ['Sim Start', cfg.sim_start],
        ['Adaptive Step', cfg.scheduler_adaptive_step == null ? '—' : cfg.scheduler_adaptive_step ? 'Yes' : 'No'],
        ['Nodes Off Start', cfg.start_nodes_off == null ? '—' : cfg.start_nodes_off ? 'Yes' : 'No'],
        ['Prediction Error', cfg.runtime_prediction_error_rate],
        ['Microgrid Step', `${cfg.microgrid_step_size_s == null ? '—' : cfg.microgrid_step_size_s + 's'}`],
        ['Energy Step', `${cfg.energy_scheduler_step_size_s == null ? '—' : cfg.energy_scheduler_step_size_s + 's'}`],
    ];

    // Add budget policy details
    const bpd = cfg.budget_policy_details || {};
    if (bpd.grid_power_limit != null) configItems.push(['Grid Power Limit', `${formatNumber(bpd.grid_power_limit)}W`]);
    if (bpd.carbon_intensity_threshold_g_per_kwh != null) configItems.push(['Carbon Threshold', `${bpd.carbon_intensity_threshold_g_per_kwh} gCO₂/kWh`]);
    const renewableChargeFraction = bpd.renewable_charge_fraction ?? bpd.pv_charge_fraction;
    if (renewableChargeFraction != null) configItems.push(['Renewable Charge Fraction', renewableChargeFraction]);
    if (cfg.renewable_sources) configItems.push(['Renewable Sources', cfg.renewable_sources.join(', ')]);

    // Clusters
    if (cfg.clusters) {
        cfg.clusters.forEach(c => configItems.push([`Cluster: ${c.type}`, `${c.num_nodes} nodes`]));
    }

    configGrid.innerHTML = configItems.filter(([key, val]) => val != null && val !== '' && val !== '—').map(([key, val]) => `
        <div class="config-item">
            <span class="config-key">${escapeHtml(key)}</span>
            <span class="config-value">${escapeHtml(String(val ?? '—'))}</span>
        </div>
    `).join('');

    // Image gallery
    const gallery = document.getElementById('result-gallery');
    if (result.images && result.images.length > 0) {
        gallery.innerHTML = result.images.map(img => `
            <div class="gallery-item" onclick="openImageModal('/api/results/${result.id}/images/${img}')">
                <img src="/api/results/${result.id}/images/${img}" alt="${img}" loading="lazy">
                <div class="gallery-item-label">${img}</div>
            </div>
        `).join('');
    } else {
        gallery.innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-chart-area empty-icon"></i>
                <span class="title">No plots available</span>
                <span class="subtitle">Run the simulation without --no-plots to generate charts</span>
            </div>
        `;
    }

    // CSV tabs
    renderCsvTabs(result);
}


// ── CSV Tables ──────────────────────────────────────────────────────────────

let currentCsvSort = { col: null, asc: true };
let currentCsvFile = null;

function renderCsvTabs(result) {
    const tabsContainer = document.getElementById('csv-tabs');
    const csvs = (result.csvs || []).filter(f => f.endsWith('.csv'));

    if (csvs.length === 0) {
        tabsContainer.innerHTML = '';
        document.getElementById('csv-table-area').innerHTML = `
            <div class="empty-state">
                <i class="fa-solid fa-file-csv empty-icon"></i>
                <span class="title">No CSV data</span>
            </div>
        `;
        return;
    }

    tabsContainer.innerHTML = csvs.map((csv, i) => `
        <button class="csv-tab ${i === 0 ? 'active' : ''}"
                onclick="loadCsvData('${result.id}', '${csv}', this)"
                data-csv="${csv}">
            ${csv}
        </button>
    `).join('');

    // Load first CSV
    loadCsvData(result.id, csvs[0]);
}


async function loadCsvData(runId, filename, tabEl = null, offset = 0) {
    // Update tab active state
    if (tabEl) {
        document.querySelectorAll('.csv-tab').forEach(t => t.classList.remove('active'));
        tabEl.classList.add('active');
    }

    currentCsvFile = filename;
    currentCsvSort = { col: null, asc: true };

    const area = document.getElementById('csv-table-area');
    area.innerHTML = '<div class="skeleton skeleton-card" style="height:300px"></div>';

    try {
        const data = await API.get(
            `/api/results/${runId}/csv/${filename}?offset=${offset}&limit=200` +
            (currentCsvSort.col ? `&sort_col=${currentCsvSort.col}&sort_asc=${currentCsvSort.asc}` : '')
        );
        renderCsvTable(runId, filename, data);
    } catch (err) {
        area.innerHTML = `<div class="empty-state"><span class="title">Error: ${escapeHtml(err.message)}</span></div>`;
    }
}


function renderCsvTable(runId, filename, data) {
    const area = document.getElementById('csv-table-area');
    const { headers, rows, total_rows, offset, limit } = data;

    let html = '<div class="table-container"><table class="data-table"><thead><tr>';

    // Headers with sort
    headers.forEach(h => {
        const isSorted = currentCsvSort.col === h;
        const sortIcon = isSorted ? (currentCsvSort.asc ? '▲' : '▼') : '⇅';
        html += `
            <th onclick="sortCsvColumn('${runId}', '${filename}', '${h}')" class="${isSorted ? 'sorted' : ''}">
                ${escapeHtml(h)}
                <span class="sort-indicator">${sortIcon}</span>
            </th>
        `;
    });

    html += '</tr></thead><tbody>';

    // Rows
    rows.forEach(row => {
        html += '<tr>';
        row.forEach(cell => {
            // Format numbers nicely
            const num = Number(cell);
            let display = cell;
            if (Number.isFinite(num) && cell.trim() !== '' && Math.abs(num) > 0.001
                    && !(Number.isInteger(num) && !Number.isSafeInteger(num))) {
                display = num.toLocaleString('en-US', { maximumFractionDigits: 4 });
            }
            html += `<td>${escapeHtml(display || '')}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table></div>';

    // Pagination
    const totalPages = Math.ceil(total_rows / limit);
    const currentPage = Math.floor(offset / limit) + 1;

    html += `
        <div class="table-pagination">
            <div class="table-pagination-info">
                Showing ${offset + 1}–${Math.min(offset + rows.length, total_rows)} of ${total_rows.toLocaleString()} rows
            </div>
            <div class="table-pagination-controls">
                <button class="btn btn-ghost btn-sm" onclick="loadCsvPage('${runId}', '${filename}', 0)"
                        ${offset === 0 ? 'disabled' : ''}>⏮ First</button>
                <button class="btn btn-ghost btn-sm" onclick="loadCsvPage('${runId}', '${filename}', ${Math.max(0, offset - limit)})"
                        ${offset === 0 ? 'disabled' : ''}>◀ Prev</button>
                <span class="text-secondary" style="padding: 0 8px;">Page ${currentPage} / ${totalPages}</span>
                <button class="btn btn-ghost btn-sm" onclick="loadCsvPage('${runId}', '${filename}', ${offset + limit})"
                        ${offset + limit >= total_rows ? 'disabled' : ''}>Next ▶</button>
                <button class="btn btn-ghost btn-sm" onclick="loadCsvPage('${runId}', '${filename}', ${(totalPages - 1) * limit})"
                        ${offset + limit >= total_rows ? 'disabled' : ''}>Last ⏭</button>
            </div>
        </div>
    `;

    area.innerHTML = html;
}


function loadCsvPage(runId, filename, offset) {
    const area = document.getElementById('csv-table-area');

    // Build sort params
    let url = `/api/results/${runId}/csv/${filename}?offset=${offset}&limit=200`;
    if (currentCsvSort.col) {
        url += `&sort_col=${currentCsvSort.col}&sort_asc=${currentCsvSort.asc}`;
    }

    area.innerHTML = '<div class="skeleton skeleton-card" style="height:300px"></div>';

    API.get(url).then(data => {
        renderCsvTable(runId, filename, data);
    }).catch(err => {
        area.innerHTML = `<div class="empty-state"><span class="title">Error loading page</span></div>`;
    });
}


function sortCsvColumn(runId, filename, colName) {
    if (currentCsvSort.col === colName) {
        currentCsvSort.asc = !currentCsvSort.asc;
    } else {
        currentCsvSort.col = colName;
        currentCsvSort.asc = true;
    }

    let url = `/api/results/${runId}/csv/${filename}?offset=0&limit=200&sort_col=${colName}&sort_asc=${currentCsvSort.asc}`;

    API.get(url).then(data => {
        renderCsvTable(runId, filename, data);
    });
}
