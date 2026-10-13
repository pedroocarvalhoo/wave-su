/* ═══════════════════════════════════════════════════════════════════════════
   WAVE GUI — Datasets Tab
   View existing datasets, generate new ones
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Load Datasets ───────────────────────────────────────────────────────────

let DatasetSupercomputers = {};

async function loadDatasets() {
    const container = document.getElementById('datasets-container');

    container.innerHTML = `
        <div class="skeleton skeleton-card" style="height:200px"></div>
        <div class="skeleton skeleton-card" style="height:200px;margin-top:16px"></div>
    `;

    try {
        const [datasetsData, scData] = await Promise.all([
            API.get('/api/datasets'),
            AppState.supercomputers ? Promise.resolve({ supercomputers: AppState.supercomputers }) : API.get('/api/supercomputers'),
        ]);

        const datasets = datasetsData.datasets;
        const sc = scData.supercomputers || scData;
        DatasetSupercomputers = sc;

        AppState.datasets = datasets;
        renderSupercomputerSidebars();
        renderSelectedDatasets();
    } catch (err) {
        container.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                <span class="title">Failed to load datasets</span>
                <span class="subtitle">${escapeHtml(err.message)}</span>
            </div>
        `;
    }
}


function renderSelectedDatasets() {
    const system = ensureSupercomputerSelection();
    const sc = AppState.supercomputers || DatasetSupercomputers;
    const genSystem = document.getElementById('gen-system');
    document.getElementById('datasets-system-context').textContent = system
        ? supercomputerLabel(system) : 'Select a supercomputer';
    if (!system || !sc[system]) {
        document.getElementById('gen-form').classList.add('hidden');
        document.getElementById('datasets-container').innerHTML = `<div class="empty-state">
            <span class="title">This machine has no registered workload configuration</span>
        </div>`;
        return;
    }
    document.getElementById('gen-form').classList.remove('hidden');
    const previousSystem = genSystem.value;
    genSystem.innerHTML = `<option value="${escapeHtml(system)}">${escapeHtml(supercomputerLabel(system))}</option>`;
    if (previousSystem !== system || !document.querySelector('.dataset-weight-input')) {
        renderGenerationWeights(sc, system);
    }
    document.getElementById('datasets-system-context').textContent = `${supercomputerLabel(system)} / ${Object.keys(getSystemClusters(sc, system)).length} clusters`;
    const clusters = AppState.datasets[system];
    if (clusters) renderDatasets({[system]: clusters}, sc);
    else document.getElementById('datasets-container').innerHTML = `<div class="empty-state">
        <span class="title">No workload data loaded</span>
        <span class="subtitle">Refresh this view to load the machine's queue files.</span>
    </div>`;
}


function getSystemClusters(sc, systemName) {
    const systemConfig = sc[systemName] || {};
    return systemConfig.clusters || systemConfig;
}


function renderGenerationWeights(sc, systemName) {
    const container = document.getElementById('gen-cluster-weights');
    if (!container || !systemName) return;

    const clusters = getSystemClusters(sc, systemName);
    container.innerHTML = Object.entries(clusters).map(([ctype, nodes]) => `
        <label class="dataset-weight-row">
            <span>
                <span class="dataset-weight-name">${escapeHtml(ctype)}</span>
                <span class="dataset-weight-nodes">${Number(nodes).toLocaleString()} nodes</span>
            </span>
            <input
                type="number"
                class="form-input dataset-weight-input"
                data-cluster="${escapeHtml(ctype)}"
                value="1"
                min="0.1"
                max="10"
                step="0.1"
            >
        </label>
    `).join('');
}


function renderDatasets(datasets, sc) {
    const container = document.getElementById('datasets-container');
    let html = '';

    for (const [systemName, clusters] of Object.entries(datasets)) {
        html += `
            <div class="dataset-system">
                <div class="dataset-system-title">
                    <i class="fa-solid fa-server dataset-system-icon" style="margin-right: var(--space-xs)"></i>
                    ${escapeHtml(supercomputerLabel(systemName))}
                </div>
                <div class="dataset-clusters">
        `;

        for (const [ctype, info] of Object.entries(clusters)) {
            const systemClusters = getSystemClusters(sc, systemName);
            const scCluster = systemClusters ? systemClusters[ctype] : null;
            const nodeCount = scCluster || '—';
            const queueEntries = Object.entries(info.queues || {});
            const readyQueues = queueEntries.filter(([_, queueInfo]) => queueInfo.exists).length;

            html += `
                <div class="dataset-cluster-card">
                    <div class="cluster-name">${escapeHtml(ctype)}</div>
                    <div class="cluster-info">
                        Nodes: ${nodeCount}
                    </div>
                    <div class="cluster-info">
                        Queue datasets: ${readyQueues}/${queueEntries.length}
                    </div>
                    <div class="cluster-status ${info.exists ? 'exists' : 'missing'}">
                        ${info.exists
                            ? `<i class="fa-solid fa-circle-check status-icon"></i> ${info.rows.toLocaleString()} jobs total (${info.size_kb} KB)`
                            : '<i class="fa-solid fa-circle-xmark status-icon"></i> Missing queue datasets'}
                    </div>
                    <div class="mg-dataset-status">
                        ${queueEntries.map(([queueName, queueInfo]) => `
                            <div class="mg-dataset-row">
                                <span class="mg-dataset-dot ${queueInfo.exists ? 'ready' : 'missing'}"></span>
                                <span style="color: var(--text-secondary)">${escapeHtml(queueName)}</span>
                                <span class="queue-job-count">${queueInfo.exists ? `${queueInfo.rows.toLocaleString()} jobs` : 'missing'}</span>
                            </div>
                            <div class="dataset-file-name">${escapeHtml(queueInfo.filename)}</div>
                        `).join('')}
                    </div>
                </div>
            `;
        }

        html += '</div></div>';
    }

    container.innerHTML = html;
}


// ── Generate Datasets ───────────────────────────────────────────────────────

async function generateDatasets() {
    const system = document.getElementById('gen-system').value;
    const numJobs = parseInt(document.getElementById('gen-num-jobs').value) || 500;
    const minJobsRaw = document.getElementById('gen-min-jobs').value;
    const minJobs = minJobsRaw ? parseInt(minJobsRaw) : null;
    const loadStrategy = document.getElementById('gen-load-strategy').value;
    const clusterLoadWeights = {};

    document.querySelectorAll('.dataset-weight-input').forEach(input => {
        const value = parseFloat(input.value);
        if (Number.isFinite(value) && value > 0) {
            clusterLoadWeights[input.dataset.cluster] = value;
        }
    });

    if (!system) {
        showToast('Select a supercomputer', 'warning');
        return;
    }

    if (numJobs < 1) {
        showToast('Number of jobs must be at least 1', 'warning');
        return;
    }
    if (minJobs !== null && (minJobs < 1 || minJobs > numJobs)) {
        showToast('Minimum jobs must be between 1 and the reference job count', 'warning');
        return;
    }

    const btn = document.getElementById('btn-gen-datasets');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-arrows-rotate btn-icon-svg spin"></i> Generating...';

    try {
        const result = await API.post('/api/datasets/generate', {
            system,
            num_jobs: numJobs,
            min_jobs: minJobs,
            load_strategy: loadStrategy,
            cluster_load_weights: clusterLoadWeights,
        });

        const counts = result.summary && result.summary.job_counts
            ? Object.entries(result.summary.job_counts).map(([name, count]) => `${name}: ${count}`).join(', ')
            : '';
        showToast(`Datasets generated for ${system}${counts ? ` (${counts})` : ''}!`, 'success');
        loadDatasets(); // Refresh
    } catch (err) {
        showToast('Generation failed: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles btn-icon-svg"></i> Generate Datasets';
    }
}
