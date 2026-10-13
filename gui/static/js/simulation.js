/* ═══════════════════════════════════════════════════════════════════════════
   WAVE GUI — Simulation Tab (Microgrid Diagram)
   Interactive node-based configuration with expand/collapse animation
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Node → Parameter mapping ────────────────────────────────────────────────

const NODE_PARAMS = {
    hpc: ['system'],
    solar: ['renewable_sources', 'renewable_source_scales'],
    grid: ['microgrid_mode', 'grid_power_limit', 'allow_grid_charging', 'pue'],
    battery: [
        'battery_soc_min', 'battery_soc_max', 'battery_capacity_kwh',
        'battery_p_charge_max', 'battery_p_discharge_max',
    ],
    policies: [
        'scheduling_policy', 'dvfs_policy', 'eft_dvfs_alpha', 'eft_dvfs_beta',
    ],
    energy_scheduler: [
        'budget_policy', 'allocation_strategy', 'energy_scheduler_step_size',
        'renewable_charge_fraction',
        'carbon_grid_threshold', 'low_carbon_grid_fraction', 'high_carbon_grid_fraction',
        'day_grid_fraction', 'night_grid_fraction',
    ],
    datasets: ['jobs_dir']
};

const BUDGET_POLICY_PARAMS = {
    grid: [],
    'daytime-renewable': ['renewable_charge_fraction', 'day_grid_fraction', 'night_grid_fraction'],
    'carbon-minimizing': [
        'renewable_charge_fraction',
        'carbon_grid_threshold',
        'low_carbon_grid_fraction',
        'high_carbon_grid_fraction',
    ],
    puzzle: ['renewable_charge_fraction', 'carbon_grid_threshold'],
};

const BUDGET_POLICY_SPECIFIC_PARAMS = new Set(Object.values(BUDGET_POLICY_PARAMS).flat());

const ADVANCED_PARAMS = [
    'microgrid_step_size',
    'runtime_prediction_error_rate', 'runtime_prediction_seed',
    'adaptive_step', 'start_nodes_off', 'keep_idle_nodes_on', 'node_power_on_gate',
    'sla_policy', 'sla_grid_override', 'run_tag', 'no_plots', 'no_snapshot',
    'plot_workers', 'scheduler_debug_log', 'verbose_scheduler',
];

// Parameters that should use a slider (0-1 range or percentage-like)
const SLIDER_PARAMS = new Set([
    'low_carbon_grid_fraction', 'high_carbon_grid_fraction',
    'day_grid_fraction', 'night_grid_fraction', 'runtime_prediction_error_rate',
    'battery_soc_min', 'battery_soc_max',
]);

// Currently expanded node (null = none)
let _expandedNode = null;
let CustomNodeTypes = {};

// ── Init Microgrid ──────────────────────────────────────────────────────────

// Helper to get an element and throw a descriptive error if it is missing
function getRequiredElement(id) {
    const el = document.getElementById(id);
    if (!el) {
        throw new Error(`Required DOM element with ID "${id}" was not found.`);
    }
    return el;
}

// ── Init Microgrid ──────────────────────────────────────────────────────────

function initSimulationForm() {
    const params = AppState.parameterMetadata;
    const sc = AppState.supercomputers;
    if (!params || !sc) return;

    // ── HPC node: system selector ──
    buildHpcNode(params, sc);

    // ── Renewable sources node ──
    buildNodeBody('solar', NODE_PARAMS.solar, params);
    updateSolarSummary();

    // ── Power Grid node ──
    buildNodeBody('grid', NODE_PARAMS.grid, params);
    updateGridSummary();

    // ── Battery node ──
    buildNodeBody('battery', NODE_PARAMS.battery, params);
    updateBatterySummary();

    // ── Policies node ──
    buildPoliciesNode(params);
    updatePoliciesSummary();

    // ── Energy Scheduler node ──
    buildNodeBody('energy_scheduler', NODE_PARAMS.energy_scheduler, params);
    updateEnergySchedulerSummary();
    updatePolicySpecificVisibility();

    // ── Datasets node ──
    loadDatasetsStatus();

    // ── Advanced params node ──
    buildNodeBody('advanced', ADVANCED_PARAMS, params);
    updateAdvancedSummary();

    // ── Form submission ──
    const simForm = getRequiredElement('simulation-form');
    simForm.addEventListener('submit', (e) => {
        e.preventDefault();
        startSimulation();
    });

    // Listen for system change to update HPC node and datasets
    observeSystemChange();
}


// ── Build HPC Node ──────────────────────────────────────────────────────────

function buildHpcNode(params, sc) {
    const systems = Object.keys(sc);
    const defaultSystem = systems[0] || '';

    const body = getRequiredElement('mg-hpc-body');
    const systemMeta = params.system;

    body.innerHTML = `
        <div class="form-grid">
            <div class="form-group">
                <label class="form-label">
                    ${systemMeta.label}
                    ${tooltipHtml(systemMeta.description)}
                </label>
                <select class="form-select" id="param-system" name="system">
                    ${systems.map(s => {
                        const label = sc[s]?.label || s;
                        return `<option value="${s}">${escapeHtml(label)}</option>`;
                    }).join('')}
                </select>
            </div>
        </div>
        <div class="custom-system-actions">
            <button type="button" class="btn btn-ghost btn-sm" onclick="openCustomSystemBuilder()">
                <i class="fa-solid fa-plus btn-icon-svg"></i>
                Create my own
            </button>
        </div>
        <div class="custom-system-builder hidden" id="custom-system-builder">
            <div class="custom-builder-header">
                <div>
                    <div class="custom-builder-title">Custom Supercomputer</div>
                    <div class="custom-builder-subtitle">Define clusters from existing or new node types.</div>
                </div>
                <button type="button" class="btn btn-ghost btn-sm" onclick="closeCustomSystemBuilder()">Close</button>
            </div>
            <div class="form-grid">
                <div class="form-group">
                    <label class="form-label">System ID</label>
                    <input type="text" class="form-input" id="custom-system-id" placeholder="my_supercomputer">
                </div>
                <div class="form-group">
                    <label class="form-label">Display Name</label>
                    <input type="text" class="form-input" id="custom-system-label" placeholder="My Supercomputer">
                </div>
            </div>

            <div class="custom-builder-section">
                <div class="custom-builder-section-header">
                    <span>Clusters</span>
                    <button type="button" class="btn btn-ghost btn-sm" onclick="addCustomClusterRow()">
                        <i class="fa-solid fa-plus btn-icon-svg"></i>
                        Add Cluster
                    </button>
                </div>
                <div class="custom-cluster-list" id="custom-cluster-list"></div>
            </div>

            <div class="custom-builder-section">
                <div class="custom-builder-section-header">
                    <span>Add Processor / Node Type</span>
                    <button type="button" class="btn btn-ghost btn-sm" onclick="addCustomFrequencyRow()">
                        <i class="fa-solid fa-plus btn-icon-svg"></i>
                        Add Frequency
                    </button>
                </div>
                <div class="form-grid">
                    <div class="form-group">
                        <label class="form-label">Node Type ID</label>
                        <input type="text" class="form-input" id="custom-node-id" placeholder="x86_custom_cpu">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Base Frequency (GHz)</label>
                        <input type="number" class="form-input" id="custom-node-base-frequency" min="0.1" step="0.01" value="2.0">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Idle Power (W)</label>
                        <input type="number" class="form-input" id="custom-node-idle-power" min="0" step="1" value="100">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Max Power (W)</label>
                        <input type="number" class="form-input" id="custom-node-max-power" min="1" step="1" value="500">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Memory Bandwidth (GB/s)</label>
                        <input type="number" class="form-input" id="custom-node-memory-bandwidth" min="1" step="0.1" value="100">
                    </div>
                </div>
                <div class="custom-frequency-list" id="custom-frequency-list"></div>
                <button type="button" class="btn btn-ghost btn-sm" onclick="addCustomNodeType()">
                    <i class="fa-solid fa-microchip btn-icon-svg"></i>
                    Add Processor Type
                </button>
                <div class="custom-node-type-list" id="custom-node-type-list"></div>
            </div>

            <div class="custom-builder-footer">
                <button type="button" class="btn btn-success" id="btn-save-custom-system" onclick="saveCustomSupercomputer()">
                    <i class="fa-solid fa-floppy-disk btn-icon-svg"></i>
                    Save Supercomputer
                </button>
            </div>
        </div>
    `;

    updateHpcSummary(defaultSystem, sc);
}


function updateHpcSummary(system, sc) {
    const title = getRequiredElement('mg-hpc-title');
    const summary = getRequiredElement('mg-hpc-summary');
    const badges = getRequiredElement('mg-hpc-clusters');

    if (!sc) sc = AppState.supercomputers;
    if (!system) system = document.getElementById('param-system')?.value;

    const systemConfig = sc?.[system];
    title.textContent = systemConfig?.label || system || 'HPC Supercomputer';
    const clusters = systemConfig?.clusters || systemConfig || {};
    const clusterEntries = Object.entries(clusters);

    if (clusterEntries.length > 0) {
        summary.textContent = `${clusterEntries.length} clusters`;
        badges.innerHTML = clusterEntries.map(([name, nodes]) =>
            `<span class="mg-cluster-badge">${name}: ${nodes} nodes</span>`
        ).join('');
    } else {
        summary.textContent = 'Select a system';
        badges.innerHTML = '';
    }
}


function observeSystemChange() {
    const select = document.getElementById('param-system');
    if (select) {
        select.addEventListener('change', () => {
            updateHpcSummary(select.value);
            loadDatasetsStatus();
        });
    }
}


function sanitizeClientId(value) {
    return String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9_]+/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_+|_+$/g, '');
}


function customNodeTypeOptions(selected = '') {
    const nodeConfig = { ...(AppState.nodeConfig || {}), ...CustomNodeTypes };
    return Object.keys(nodeConfig).sort().map(type => `
        <option value="${escapeHtml(type)}" ${type === selected ? 'selected' : ''}>
            ${escapeHtml(type)}
        </option>
    `).join('');
}


function refreshCustomClusterSelects() {
    document.querySelectorAll('.custom-cluster-type').forEach(select => {
        const selected = select.value;
        select.innerHTML = customNodeTypeOptions(selected);
        if (selected) select.value = selected;
    });
}


function openCustomSystemBuilder() {
    const builder = document.getElementById('custom-system-builder');
    if (!builder) return;
    builder.classList.remove('hidden');
    CustomNodeTypes = CustomNodeTypes || {};

    if (!document.querySelector('.custom-cluster-row')) {
        addCustomClusterRow();
    }
    if (!document.querySelector('.custom-frequency-row')) {
        addCustomFrequencyRow('1.5', '300');
        addCustomFrequencyRow('2.0', '500');
    }
}


function closeCustomSystemBuilder() {
    document.getElementById('custom-system-builder')?.classList.add('hidden');
}


function addCustomClusterRow(nodeType = '', nodes = 1) {
    const container = document.getElementById('custom-cluster-list');
    if (!container) return;
    const row = document.createElement('div');
    row.className = 'custom-cluster-row';
    row.innerHTML = `
        <select class="form-select custom-cluster-type">
            ${customNodeTypeOptions(nodeType)}
        </select>
        <input type="number" class="form-input custom-cluster-nodes" min="1" step="1" value="${nodes}">
        <button type="button" class="btn btn-ghost btn-sm" onclick="removeCustomRow(this)" title="Remove cluster">
            <i class="fa-solid fa-trash"></i>
        </button>
    `;
    container.appendChild(row);
}


function addCustomFrequencyRow(freq = '', power = '') {
    const container = document.getElementById('custom-frequency-list');
    if (!container) return;
    const row = document.createElement('div');
    row.className = 'custom-frequency-row';
    row.innerHTML = `
        <input type="number" class="form-input custom-frequency-ghz" min="0.1" step="0.001" placeholder="GHz" value="${freq}">
        <input type="number" class="form-input custom-frequency-power" min="1" step="1" placeholder="Power W" value="${power}">
        <button type="button" class="btn btn-ghost btn-sm" onclick="removeCustomRow(this)" title="Remove frequency">
            <i class="fa-solid fa-trash"></i>
        </button>
    `;
    container.appendChild(row);
}


function removeCustomRow(button) {
    button.closest('.custom-cluster-row, .custom-frequency-row')?.remove();
}


function renderCustomNodeTypeList() {
    const list = document.getElementById('custom-node-type-list');
    if (!list) return;
    const entries = Object.entries(CustomNodeTypes);
    list.innerHTML = entries.map(([id, config]) => `
        <span class="custom-node-chip">
            ${escapeHtml(id)}
            <span>${Number(config.max_power_w).toLocaleString()}W</span>
        </span>
    `).join('');
}


function addCustomNodeType() {
    const rawId = document.getElementById('custom-node-id')?.value;
    const id = sanitizeClientId(rawId);
    const baseFrequency = parseFloat(document.getElementById('custom-node-base-frequency')?.value);
    const idlePower = parseFloat(document.getElementById('custom-node-idle-power')?.value);
    const maxPower = parseFloat(document.getElementById('custom-node-max-power')?.value);
    const memoryBandwidth = parseFloat(document.getElementById('custom-node-memory-bandwidth')?.value);

    if (!id) {
        showToast('Node type ID is required', 'warning');
        return;
    }
    if ((AppState.nodeConfig && AppState.nodeConfig[id]) || CustomNodeTypes[id]) {
        showToast(`Node type ${id} already exists`, 'warning');
        return;
    }
    if (![baseFrequency, idlePower, maxPower, memoryBandwidth].every(Number.isFinite)) {
        showToast('Fill all processor numeric fields', 'warning');
        return;
    }
    if (maxPower < idlePower) {
        showToast('Max power must be greater than idle power', 'warning');
        return;
    }

    const frequencies = {};
    document.querySelectorAll('.custom-frequency-row').forEach(row => {
        const freq = parseFloat(row.querySelector('.custom-frequency-ghz')?.value);
        const power = parseFloat(row.querySelector('.custom-frequency-power')?.value);
        if (Number.isFinite(freq) && Number.isFinite(power) && freq > 0 && power > 0) {
            frequencies[String(freq)] = power;
        }
    });
    frequencies[String(baseFrequency)] = maxPower;

    CustomNodeTypes[id] = {
        base_frequency_ghz: baseFrequency,
        idle_power_w: idlePower,
        max_power_w: maxPower,
        memory_bandwidth_gb_s: memoryBandwidth,
        frequencies,
    };

    document.getElementById('custom-node-id').value = '';
    renderCustomNodeTypeList();
    refreshCustomClusterSelects();
    showToast(`Processor type ${id} added to this supercomputer draft`, 'success');
}


async function saveCustomSupercomputer() {
    const id = sanitizeClientId(document.getElementById('custom-system-id')?.value);
    const label = document.getElementById('custom-system-label')?.value?.trim() || id;
    if (!id) {
        showToast('System ID is required', 'warning');
        return;
    }

    const clusters = [];
    document.querySelectorAll('.custom-cluster-row').forEach(row => {
        const nodeType = row.querySelector('.custom-cluster-type')?.value;
        const numNodes = parseInt(row.querySelector('.custom-cluster-nodes')?.value, 10);
        if (nodeType && Number.isInteger(numNodes) && numNodes > 0) {
            clusters.push({ node_type: nodeType, num_nodes: numNodes });
        }
    });
    if (clusters.length === 0) {
        showToast('Add at least one cluster', 'warning');
        return;
    }

    const nodeTypes = Object.entries(CustomNodeTypes).map(([nodeId, config]) => ({
        id: nodeId,
        ...config,
    }));

    const btn = document.getElementById('btn-save-custom-system');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-arrows-rotate btn-icon-svg spin"></i> Saving...';
    try {
        const result = await API.post('/api/supercomputers', {
            id,
            label,
            clusters,
            node_types: nodeTypes,
        });
        AppState.supercomputers = result.supercomputers;
        AppState.nodeConfig = result.node_config;
        CustomNodeTypes = {};

        buildHpcNode(AppState.parameterMetadata, AppState.supercomputers);
        observeSystemChange();
        const select = document.getElementById('param-system');
        if (select) {
            select.value = result.system;
            updateHpcSummary(result.system);
        }
        loadDatasetsStatus();
        if (typeof loadDatasets === 'function') {
            loadDatasets();
        }
        showToast(`Supercomputer ${result.system} created`, 'success');
    } catch (err) {
        showToast('Failed to create supercomputer: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-floppy-disk btn-icon-svg"></i> Save Supercomputer';
    }
}


// ── Build Policies Node (with alpha-beta slider) ────────────────────────────

function buildPoliciesNode(params) {
    const body = getRequiredElement('mg-policies-body');
    const regularParams = ['scheduling_policy', 'dvfs_policy'];
    const fields = regularParams.map(key => buildFormField(key, params[key])).join('');
    const alphaBeta = buildAlphaBetaSlider(params);

    const puzzleFields = Object.keys(params).filter(key => key.startsWith('puzzle_'))
        .map(key => buildFormField(key, params[key])).join('');
    body.innerHTML = `<div class="form-grid">${fields}${alphaBeta}${puzzleFields}</div>`;
}


// ── Build Generic Node Body ─────────────────────────────────────────────────

function buildNodeBody(nodeId, paramKeys, params) {
    const body = document.getElementById(`mg-${nodeId}-body`);
    if (!body) return;

    const fields = paramKeys.map(key => buildFormField(key, params[key])).join('');
    body.innerHTML = `<div class="form-grid">${fields}</div>`;
}


// ── Load Datasets Status ────────────────────────────────────────────────────

async function loadDatasetsStatus() {
    const system = document.getElementById('param-system')?.value;
    const summary = getRequiredElement('mg-datasets-summary');
    const body = getRequiredElement('mg-datasets-body');
    const workload = document.getElementById('param-jobs_dir')?.value || '';
    if (!system) return;

    try {
        const data = await API.get('/api/datasets' + (workload ? `?system=${encodeURIComponent(system)}&jobs_dir=${encodeURIComponent(workload)}` : ''));
        const datasets = data.datasets[system] || {};
        const entries = Object.entries(datasets);
        const queueEntries = entries.flatMap(([cluster, d]) =>
            Object.entries(d.queues || {}).map(([queue, info]) => ({ cluster, queue, info }))
        );
        const readyQueues = queueEntries.filter(item => item.info.exists).length;
        const allReady = queueEntries.length > 0 && readyQueues === queueEntries.length;
        const noneReady = readyQueues === 0;

        if (noneReady) {
            summary.textContent = 'No datasets generated';
        } else if (allReady) {
            summary.textContent = `${queueEntries.length} queue datasets ready`;
        } else {
            summary.textContent = `${readyQueues}/${queueEntries.length} queue datasets ready`;
        }

        body.innerHTML = `
            <div class="form-grid">${buildFormField('jobs_dir', AppState.parameterMetadata.jobs_dir)}</div>
            <div class="mg-dataset-status">
                ${entries.map(([cluster, d]) => `
                    <div class="mg-dataset-row">
                        <span class="mg-dataset-dot ${d.exists ? 'ready' : 'missing'}"></span>
                        <span style="color: var(--text-secondary)">${cluster}</span>
                        <span style="color: var(--text-muted); margin-left: auto;">
                            ${d.exists ? `${d.rows} jobs` : `${Object.values(d.queues || {}).filter(q => q.exists).length}/${Object.keys(d.queues || {}).length} queues`}
                        </span>
                    </div>
                    ${Object.entries(d.queues || {}).map(([queue, q]) => `
                        <div class="mg-dataset-row mg-dataset-row--queue">
                            <span class="mg-dataset-dot ${q.exists ? 'ready' : 'missing'}"></span>
                            <span style="color: var(--text-muted)">${queue}</span>
                            <span style="color: var(--text-muted); margin-left: auto;">
                                ${q.exists ? `${q.rows} jobs` : 'missing'}
                            </span>
                        </div>
                    `).join('')}
                `).join('')}
            </div>
            <div style="margin-top: var(--space-md); text-align: center;">
                <button type="button" class="btn btn-ghost btn-sm" onclick="event.stopPropagation(); switchTab('datasets')">
                    <i class="fa-solid fa-arrow-right btn-icon-svg"></i>
                    Go to Datasets tab
                </button>
            </div>
        `;
        const input = document.getElementById('param-jobs_dir');
        input.value = workload;
        input.addEventListener('change', loadDatasetsStatus);
    } catch (err) {
        summary.textContent = 'Error loading';
    }
}


// ── Node Expand / Collapse ──────────────────────────────────────────────────

function expandNode(nodeId) {
    const diagram = document.getElementById('mg-diagram');
    const node = document.getElementById(`mg-${nodeId}`);
    if (!node || node.classList.contains('expanded')) return;

    // Collapse any currently expanded node first
    if (_expandedNode) {
        collapseNodeImmediate();
    }

    _expandedNode = nodeId;
    node.classList.add('expanded');
    node.querySelector('.mg-node__header').setAttribute('aria-expanded', 'true');
    diagram.classList.add('node-expanded');
}


function collapseNode() {
    if (!_expandedNode) return;

    const diagram = document.getElementById('mg-diagram');
    const node = document.getElementById(`mg-${_expandedNode}`);

    if (node) {
        node.classList.remove('expanded');
        node.querySelector('.mg-node__header').setAttribute('aria-expanded', 'false');
    }
    diagram.classList.remove('node-expanded');
    _expandedNode = null;

    // Update all summaries after editing
    updateAllSummaries();
}


function collapseNodeImmediate() {
    if (!_expandedNode) return;
    const node = document.getElementById(`mg-${_expandedNode}`);
    if (node) {
        node.classList.remove('expanded');
        node.querySelector('.mg-node__header').setAttribute('aria-expanded', 'false');
    }
    document.getElementById('mg-diagram').classList.remove('node-expanded');
    _expandedNode = null;
}


// ── Summary Updaters ────────────────────────────────────────────────────────

function updateAllSummaries() {
    updateSolarSummary();
    updateGridSummary();
    updatePoliciesSummary();
    updateEnergySchedulerSummary();
    updateHpcSummary();
    updateBatterySummary();
    updateAdvancedSummary();
}


function updateSolarSummary() {
    const summary = document.getElementById('mg-solar-summary');
    if (summary) {
        const selected = Array.from(
            document.querySelectorAll('input[name="renewable_sources"]:checked')
        ).map(input => input.dataset.label || input.value);
        const sourceText = selected.length ? selected.join(' + ') : 'No sources';
        summary.textContent = sourceText;
    }
}


function updateBatterySummary() {
    const capacityEl = document.getElementById('param-battery_capacity_kwh');
    const minSocEl = document.getElementById('param-battery_soc_min');
    const maxSocEl = document.getElementById('param-battery_soc_max');
    const summary = document.getElementById('mg-battery-summary');
    if (summary) {
        const capacity = capacityEl ? parseFloat(capacityEl.value) : 2000;
        const minSoc = minSocEl ? Math.round(parseFloat(minSocEl.value) * 100) : 20;
        const maxSoc = maxSocEl ? Math.round(parseFloat(maxSocEl.value) * 100) : 80;
        summary.textContent = `${formatNumber(capacity)}kWh · ${minSoc}%-${maxSoc}% SOC`;
    }
}


function updateGridSummary() {
    const modeEl = document.getElementById('param-microgrid_mode');
    const limit = document.getElementById('param-grid_power_limit');
    const chargingEl = document.getElementById('param-allow_grid_charging');
    const pueEl = document.getElementById('param-pue');
    const summary = document.getElementById('mg-grid-summary');
    if (summary) {
        const modeValue = modeEl?.value || 'grid-connected';
        if (limit) {
            limit.disabled = modeValue === 'islanded';
        }
        if (chargingEl) {
            chargingEl.disabled = modeValue === 'islanded';
            if (modeValue === 'islanded') chargingEl.checked = false;
            const label = chargingEl.nextElementSibling;
            if (label) label.textContent = chargingEl.checked ? 'Enabled' : 'Disabled';
        }
        const modeLabel = modeValue === 'islanded' ? 'Islanded' : 'Grid connected';
        const l = limit ? parseFloat(limit.value) : 200000;
        const effectiveLimit = modeValue === 'islanded' ? 0 : l;
        const limitLabel = modeValue === 'islanded'
            ? `${formatNumber(effectiveLimit)}W effective`
            : `${formatNumber(effectiveLimit)}W`;
        const chargingLabel = chargingEl?.checked ? 'grid charging' : 'surplus charging';
        const pue = pueEl ? parseFloat(pueEl.value) : 1.3;
        const pueLabel = Number.isFinite(pue) ? `PUE ${pue.toFixed(2)}` : 'PUE';
        summary.textContent = `${modeLabel} · ${limitLabel} · ${chargingLabel} · ${pueLabel}`;
    }
}


function updatePoliciesSummary() {
    const sched = document.getElementById('param-scheduling_policy');
    const dvfs = document.getElementById('param-dvfs_policy');
    const summary = document.getElementById('mg-policies-summary');
    if (summary) {
        const parts = [];
        if (sched) {
            const label = document.querySelector('#csw-scheduling_policy .custom-select-value');
            parts.push(label ? label.textContent : sched.value);
        }
        if (dvfs) {
            const label = document.querySelector('#csw-dvfs_policy .custom-select-value');
            parts.push(label ? label.textContent : dvfs.value);
        }
        summary.textContent = parts.join(' / ') || 'Configure Job Scheduler';
    }
}

function updateEnergySchedulerSummary() {
    const budget = document.getElementById('param-budget_policy');
    const allocation = document.getElementById('param-allocation_strategy');
    const stepEl = document.getElementById('param-energy_scheduler_step_size');
    const summary = document.getElementById('mg-energy_scheduler-summary');
    if (summary) {
        const parts = [];
        if (budget) {
            const label = document.querySelector('#csw-budget_policy .custom-select-value');
            parts.push(label ? label.textContent : budget.value);
        }
        if (allocation) {
            const label = document.querySelector('#csw-allocation_strategy .custom-select-value');
            parts.push(label ? label.textContent : allocation.value);
        }
        if (budget?.value === 'carbon-minimizing') {
            const threshold = parseFloat(document.getElementById('param-carbon_grid_threshold')?.value);
            if (!isNaN(threshold)) parts.push(`≤ ${threshold} gCO₂/kWh`);
        } else if (budget?.value === 'daytime-renewable') {
            const day = parseFloat(document.getElementById('param-day_grid_fraction')?.value);
            const night = parseFloat(document.getElementById('param-night_grid_fraction')?.value);
            if (!isNaN(day) && !isNaN(night)) {
                parts.push(`day ${Math.round(day * 100)}% / night ${Math.round(night * 100)}%`);
            }
        }
        if (stepEl) {
            const seconds = parseFloat(stepEl.value);
            if (!isNaN(seconds)) {
                if (seconds >= 60 && seconds % 60 === 0) {
                    parts.push(`step: ${seconds / 60}m`);
                } else {
                    parts.push(`step: ${seconds}s`);
                }
            }
        }
        summary.textContent = parts.join(' / ') || 'Configure Energy Scheduler';
    }
}


function updateAdvancedSummary() {
    const microgridStep = parseFloat(document.getElementById('param-microgrid_step_size')?.value);
    const plotWorkers = parseFloat(document.getElementById('param-plot_workers')?.value);
    const debugEnabled = document.getElementById('param-scheduler_debug_log')?.checked;
    const summary = document.getElementById('mg-advanced-summary');
    if (!summary) return;

    const parts = [];
    if (!isNaN(microgridStep)) parts.push(`microgrid ${microgridStep}s`);
    if (!isNaN(plotWorkers)) parts.push(`${plotWorkers} plot workers`);
    parts.push(debugEnabled ? 'debug on' : 'debug off');
    summary.textContent = parts.join(' · ');
}


function updatePolicySpecificVisibility() {
    syncPuzzlePolicySelection();

    const budgetPolicy = document.getElementById('param-budget_policy')?.value || 'carbon-minimizing';
    const visibleBudgetParams = new Set(BUDGET_POLICY_PARAMS[budgetPolicy] || []);

    BUDGET_POLICY_SPECIFIC_PARAMS.forEach(key => {
        const group = document.querySelector(`[data-param-key="${key}"]`);
        if (group) {
            group.classList.toggle('hidden', !visibleBudgetParams.has(key));
        }
    });

    const schedulingPolicy = document.getElementById('param-scheduling_policy')?.value || 'eft-dvfs';
    document.querySelectorAll('[data-param-key^="puzzle_"]').forEach(group => group.classList.toggle('hidden', schedulingPolicy !== 'puzzle'));
    const alphaBeta = document.querySelector('[data-param-key="eft_dvfs_balance"]');
    if (alphaBeta) {
        alphaBeta.classList.toggle('hidden', schedulingPolicy !== 'eft-dvfs');
    }

    updateGridSummary();
    updateSolarSummary();
    updatePoliciesSummary();
    updateEnergySchedulerSummary();
}


// ── Form Field Builders (preserved from original) ───────────────────────────

function buildFormField(key, meta) {
    if (!meta) return '';

    const tooltip = tooltipHtml(meta.description);

    if (meta.type === 'select' && meta.options) {
        const options = meta.options.map(opt => {
            const optTooltip = opt.description
                ? `<span class="option-tooltip-trigger tooltip-trigger">?
                     <span class="option-tooltip-text tooltip-text">${escapeHtml(opt.description)}</span>
                   </span>`
                : '';
            return `
                <div class="select-option-row" data-value="${opt.value}" tabindex="0"
                     onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();this.click();}">
                    <span class="select-option-label">${opt.label}</span>
                    ${optTooltip}
                </div>
            `;
        });

        const defaultOpt = meta.options.find(o => o.value === meta.default) || meta.options[0];

        return `
            <div class="form-group" data-param-key="${key}">
                <label class="form-label">${meta.label} ${tooltip}</label>
                <div class="custom-select-wrapper" id="csw-${key}">
                    <button type="button" class="custom-select-trigger" onclick="toggleCustomSelect('${key}')">
                        <span class="custom-select-value">${defaultOpt.label}</span>
                        <i class="fa-solid fa-chevron-down custom-select-arrow"></i>
                    </button>
                    <div class="custom-select-dropdown">
                        ${options.join('')}
                    </div>
                    <input type="hidden" id="param-${key}" name="${key}" value="${meta.default}">
                </div>
            </div>
        `;
    }

    if (meta.type === 'boolean') {
        const checked = meta.default ? 'checked' : '';
        return `
            <div class="form-group" data-param-key="${key}">
                <label class="form-label">${meta.label} ${tooltip}</label>
                <label class="form-checkbox-wrap">
                    <input type="checkbox" class="form-checkbox" id="param-${key}" name="${key}" ${checked}>
                    <span class="form-checkbox-label">${meta.default ? 'Enabled' : 'Disabled'}</span>
                </label>
            </div>
        `;
    }

    if (meta.type === 'multiselect') {
        const selected = new Set(meta.default || []);
        const options = meta.options || [];
        return `
            <div class="form-group form-group-wide" data-param-key="${key}">
                <label class="form-label">${meta.label} ${tooltip}</label>
                <div class="source-option-list">
                    ${options.map(opt => `
                        <label class="form-checkbox-wrap source-option">
                            <input type="checkbox"
                                class="form-checkbox renewable-source-checkbox"
                                name="${key}"
                                value="${opt.value}"
                                data-label="${escapeHtml(opt.label)}"
                                ${selected.has(opt.value) ? 'checked' : ''}
                                onchange="updateSolarSummary()">
                            <span class="form-checkbox-label">${escapeHtml(opt.label)}</span>
                        </label>
                    `).join('')}
                </div>
            </div>
        `;
    }

    if (meta.type === 'source_scales') {
        const options = meta.options || [];
        const defaults = meta.default || {};
        return `
            <div class="form-group form-group-wide" data-param-key="${key}">
                <label class="form-label">${meta.label} ${tooltip}</label>
                <div class="source-scale-grid">
                    ${options.map(opt => `
                        <label class="source-scale-row">
                            <span>${escapeHtml(opt.label)}</span>
                            <input type="number"
                                class="form-input"
                                id="param-${key}-${opt.value}"
                                data-source="${opt.value}"
                                data-source-scale="true"
                                value="${defaults[opt.value] ?? 0}"
                                ${meta.min !== undefined ? `min="${meta.min}"` : ''}
                                step="1000"
                                onchange="updateSolarSummary()">
                        </label>
                    `).join('')}
                </div>
            </div>
        `;
    }

    if (meta.type === 'number') {
        if (SLIDER_PARAMS.has(key)) {
            return buildSliderField(key, meta, tooltip);
        }

        const step = meta.step || (Number.isInteger(meta.default) ? 1 : 0.01);
        return `
            <div class="form-group" data-param-key="${key}">
                <label class="form-label">${meta.label} ${tooltip}</label>
                <input type="number" class="form-input" id="param-${key}" name="${key}"
                    value="${meta.default ?? ''}"
                    ${meta.min !== undefined ? `min="${meta.min}"` : ''}
                    ${meta.max !== undefined ? `max="${meta.max}"` : ''}
                    step="${step}">
            </div>
        `;
    }

    if (meta.type === 'text') {
        return `<div class="form-group" data-param-key="${key}">
            <label class="form-label">${meta.label} ${tooltip}</label>
            <input type="text" class="form-input" id="param-${key}" name="${key}" value="${escapeHtml(meta.default || '')}">
        </div>`;
    }
    return '';
}


function buildSliderField(key, meta, tooltip) {
    const min = meta.min !== undefined ? meta.min : 0;
    const max = meta.max !== undefined ? meta.max : 1;
    const step = meta.step || 0.05;
    const value = meta.default;
    const pct = Math.round(value * 100);

    return `
        <div class="form-group" data-param-key="${key}">
            <label class="form-label">${meta.label} ${tooltip}</label>
            <div class="slider-wrapper">
                <input type="range" class="form-slider" id="param-${key}" name="${key}"
                    min="${min}" max="${max}" step="${step}" value="${value}"
                    oninput="updateSliderDisplay(this, '${key}')">
                <span class="slider-value" id="slider-val-${key}">${pct}%</span>
            </div>
        </div>
    `;
}


function updateSliderDisplay(slider, key) {
    const val = parseFloat(slider.value);
    const display = document.getElementById(`slider-val-${key}`);
    if (display) {
        display.textContent = `${Math.round(val * 100)}%`;
    }
    updateAllSummaries();
}


function buildAlphaBetaSlider(params) {
    const alphaMeta = params.eft_dvfs_alpha;
    const betaMeta = params.eft_dvfs_beta;
    if (!alphaMeta || !betaMeta) return '';

    const alpha = alphaMeta.default;
    const beta = betaMeta.default;

    return `
        <div class="form-group form-group-wide" data-param-key="eft_dvfs_balance">
            <label class="form-label">
                EFT-DVFS Balance: Runtime ↔ Energy
                ${tooltipHtml('Controls the trade-off between runtime performance (Alpha) and energy savings (Beta). Alpha + Beta = 1.0. Move right for more energy savings, left for faster job completion.')}
            </label>
            <div class="alpha-beta-slider-container">
                <div class="alpha-beta-labels">
                    <span class="ab-label ab-label-alpha">
                        <i class="fa-solid fa-gauge-high ab-icon"></i>
                        Runtime (α)
                    </span>
                    <span class="ab-label ab-label-beta">
                        Energy (β)
                        <i class="fa-solid fa-leaf ab-icon"></i>
                    </span>
                </div>
                <input type="range" class="form-slider alpha-beta-slider" id="param-ab-slider"
                    min="0" max="1" step="0.05" value="${beta}"
                    oninput="updateAlphaBeta(this)">
                <div class="alpha-beta-values">
                    <span class="ab-value" id="ab-alpha-val">α = ${alpha.toFixed(2)}</span>
                    <span class="ab-value" id="ab-beta-val">β = ${beta.toFixed(2)}</span>
                </div>
                <input type="hidden" id="param-eft_dvfs_alpha" name="eft_dvfs_alpha" value="${alpha}">
                <input type="hidden" id="param-eft_dvfs_beta" name="eft_dvfs_beta" value="${beta}">
            </div>
        </div>
    `;
}


function updateAlphaBeta(slider) {
    const beta = parseFloat(slider.value);
    const alpha = Math.round((1 - beta) * 100) / 100;
    const betaFixed = Math.round(beta * 100) / 100;

    document.getElementById('ab-alpha-val').textContent = `α = ${alpha.toFixed(2)}`;
    document.getElementById('ab-beta-val').textContent = `β = ${betaFixed.toFixed(2)}`;
    document.getElementById('param-eft_dvfs_alpha').value = alpha;
    document.getElementById('param-eft_dvfs_beta').value = betaFixed;
    updateAllSummaries();
}


// ── Custom Select (with option tooltips) ────────────────────────────────────

function setCustomSelectValue(key, value, label = null) {
    const wrapper = document.getElementById(`csw-${key}`);
    const input = document.getElementById(`param-${key}`);
    if (!wrapper || !input) return;

    const row = wrapper.querySelector(`.select-option-row[data-value="${value}"]`);
    const optionLabel = label
        || row?.querySelector('.select-option-label')?.textContent
        || value;

    wrapper.querySelector('.custom-select-value').textContent = optionLabel;
    input.value = value;
}

let previousSchedulingPolicy = null;

function syncPuzzlePolicySelection() {
    const schedulingPolicy = document.getElementById('param-scheduling_policy');
    const budgetPolicy = document.getElementById('param-budget_policy');
    if (!schedulingPolicy || !budgetPolicy) return;

    const changed = schedulingPolicy.value !== previousSchedulingPolicy;
    previousSchedulingPolicy = schedulingPolicy.value;
    if (changed && schedulingPolicy.value === 'puzzle' && budgetPolicy.value !== 'puzzle') {
        setCustomSelectValue('budget_policy', 'puzzle');
    }
    document.querySelectorAll('#csw-budget_policy .select-option-row').forEach(row => {
        const disabled = schedulingPolicy.value === 'puzzle' && row.dataset.value !== 'puzzle';
        row.classList.toggle('disabled', disabled);
        row.setAttribute('aria-disabled', String(disabled));
        row.title = disabled ? 'Puzzle scheduling requires the Puzzle energy budget' : '';
    });
}

function toggleCustomSelect(key) {
    const wrapper = document.getElementById(`csw-${key}`);
    const isOpen = wrapper.classList.contains('open');

    // Close all open custom selects
    document.querySelectorAll('.custom-select-wrapper.open').forEach(w => {
        w.classList.remove('open');
        const card = w.closest('.card') || w.closest('.mg-node');
        if (card) card.classList.remove('has-open-select');
    });

    if (!isOpen) {
        wrapper.classList.add('open');
        const card = wrapper.closest('.card') || wrapper.closest('.mg-node');
        if (card) card.classList.add('has-open-select');

        // Add click handlers to options
        wrapper.querySelectorAll('.select-option-row').forEach(row => {
            row.onclick = (e) => {
                if (row.classList.contains('disabled')) return;
                // Don't close if clicking the tooltip trigger
                if (e.target.closest('.option-tooltip-trigger')) return;

                const value = row.dataset.value;
                const label = row.querySelector('.select-option-label').textContent;
                setCustomSelectValue(key, value, label);
                wrapper.classList.remove('open');
                if (card) card.classList.remove('has-open-select');
                syncPuzzlePolicySelection();
                updatePolicySpecificVisibility();
            };
        });
    }
}

// Close custom selects when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.custom-select-wrapper')) {
        document.querySelectorAll('.custom-select-wrapper.open').forEach(w => {
            w.classList.remove('open');
            const card = w.closest('.card') || w.closest('.mg-node');
            if (card) card.classList.remove('has-open-select');
        });
    }
});


// ── Collect Form Values ─────────────────────────────────────────────────────

function collectFormValues() {
    const params = AppState.parameterMetadata;
    const values = {};

    // System
    values.system = document.getElementById('param-system').value;

    // All params from all nodes + advanced
    const allKeys = Object.keys(params).filter(key => key !== 'system');

    for (const key of allKeys) {
        const meta = params[key];

        if (meta.type === 'multiselect') {
            values[key] = Array.from(
                document.querySelectorAll(`input[name="${key}"]:checked`)
            ).map(input => input.value);
            continue;
        }

        if (meta.type === 'source_scales') {
            values[key] = {};
            document.querySelectorAll('[data-source-scale="true"]').forEach(input => {
                values[key][input.dataset.source] = parseFloat(input.value);
            });
            continue;
        }

        const el = document.getElementById(`param-${key}`);
        if (!el) continue;

        if (key === 'sla_grid_override' || key === 'puzzle_fallback_dispatch') {
            values[key] = el.value === '' ? null : el.value === 'true';
        } else if (meta.type === 'boolean') {
            values[key] = el.checked;
        } else if (meta.type === 'number') {
            values[key] = el.value === '' ? null : parseFloat(el.value);
        } else {
            values[key] = el.value || null;
        }
    }

    return values;
}


// ── Start Simulation ────────────────────────────────────────────────────────

async function startSimulation() {
    if (AppState.simRunning) {
        showToast('A simulation is already running', 'warning');
        return;
    }

    // Collapse any expanded node before starting
    collapseNode();

    const values = collectFormValues();
    if (!values.renewable_sources || values.renewable_sources.length === 0) {
        showToast('Select at least one renewable source', 'warning');
        return;
    }

    try {
        const result = await API.post('/api/simulation/start', values);

        AppState.simRunning = true;
        setStatus('Simulation running...', 'running');

        // Show terminal and cancel button
        document.getElementById('sim-terminal-card').classList.remove('hidden');
        document.getElementById('btn-cancel-sim').classList.remove('hidden');
        document.getElementById('btn-start-sim').disabled = true;
        document.getElementById('sim-status-text').textContent = `PID: ${result.pid}`;

        clearTerminal();
        appendTerminalLine(`$ ${result.command}`, 'info');
        appendTerminalLine('');

        showToast('Simulation started', 'success');
        connectSSE();

    } catch (err) {
        showToast('Failed to start: ' + err.message, 'error');
    }
}


// ── Connect SSE ─────────────────────────────────────────────────────────────

function connectSSE() {
    if (AppState.sseSource) {
        AppState.sseSource.close();
    }

    AppState.sseSource = API.sse('/api/simulation/status', (data) => {
        if (data.type === 'log') {
            let cls = '';
            if (data.line.includes('ERROR') || data.line.includes('Traceback')) cls = 'error';
            else if (data.line.includes('✅') || data.line.includes('Complete')) cls = 'success';
            else if (data.line.startsWith('[')) cls = 'info';
            appendTerminalLine(data.line, cls);
        }

        if (data.type === 'complete') {
            AppState.simRunning = false;
            const cancelled = data.cancelled === true;
            const success = data.returncode === 0 && !cancelled;
            setStatus(cancelled ? 'Simulation cancelled' : success ? 'Simulation complete' : 'Simulation failed', success || cancelled ? 'idle' : 'error');
            appendTerminalLine('', '');
            appendTerminalLine(
                cancelled ? 'Simulation cancelled. Partial results saved.' : success ? '✅ Simulation completed successfully!' : `❌ Simulation exited with code ${data.returncode}`,
                success ? 'success' : cancelled ? 'info' : 'error'
            );

            document.getElementById('btn-cancel-sim').classList.add('hidden');
            document.getElementById('btn-start-sim').disabled = false;
            document.getElementById('sim-status-text').textContent = '';

            if (AppState.sseSource) {
                AppState.sseSource.close();
                AppState.sseSource = null;
            }

            if (!cancelled) handleSimulationComplete(success);
        }
    }, (err) => {
        console.error('SSE error:', err);
    });
}

async function handleSimulationComplete(success) {
    if (success) {
        try {
            // Brief sleep for filesystem synchronization
            await new Promise(r => setTimeout(r, 1000));
            const data = await API.get('/api/results');
            AppState.results = data.results;

            if (AppState.results && AppState.results.length > 0) {
                const latestRun = AppState.results.find(run => run.source === 'simulation');
                if (latestRun) showResultsToast(latestRun.id);
            } else {
                showToast('Simulation completed!', 'success');
            }
        } catch (e) {
            showToast('Simulation completed!', 'success');
        }
    } else {
        showToast('Simulation failed', 'error');
    }
}


// ── Cancel Simulation ───────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.mg-node__header').forEach(header => {
        const node = header.closest('.mg-node');
        const nodeId = node.id.replace('mg-', '');
        header.tabIndex = 0;
        header.setAttribute('role', 'button');
        header.setAttribute('aria-expanded', 'false');
        header.setAttribute('aria-controls', `mg-${nodeId}-body`);
        header.addEventListener('keydown', event => {
            if (event.target !== header || !['Enter', ' '].includes(event.key)) return;
            event.preventDefault();
            if (_expandedNode === nodeId) collapseNode();
            else expandNode(nodeId);
        });
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && _expandedNode) {
            const header = document.querySelector(`#mg-${_expandedNode} .mg-node__header`);
            collapseNode();
            header.focus();
        }
    });
    const cancelBtn = document.getElementById('btn-cancel-sim');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', async () => {
            try {
                await API.post('/api/simulation/cancel');
                setStatus('Saving partial results...', 'running');
                showToast('Cancelling simulation', 'warning');
            } catch (err) {
                showToast('Failed to cancel: ' + err.message, 'error');
            }
        });
    }

    // Collapse expanded node when clicking on the diagram backdrop
    const diagram = document.getElementById('mg-diagram');
    if (diagram) {
        diagram.addEventListener('click', (e) => {
            if (_expandedNode && !e.target.closest('.mg-node.expanded')) {
                collapseNode();
            }
        });
    }

    // Update checkbox labels on change
    document.addEventListener('change', (e) => {
        if (
            e.target.classList.contains('form-checkbox')
            && !e.target.classList.contains('renewable-source-checkbox')
        ) {
            const label = e.target.nextElementSibling;
            if (label) label.textContent = e.target.checked ? 'Enabled' : 'Disabled';
        }
        if (e.target.id === 'param-keep_idle_nodes_on' && e.target.checked) {
            const startOff = document.getElementById('param-start_nodes_off');
            startOff.checked = false;
            startOff.nextElementSibling.textContent = 'Disabled';
        }
        if (e.target.id === 'param-start_nodes_off' && e.target.checked) {
            const keepIdle = document.getElementById('param-keep_idle_nodes_on');
            keepIdle.checked = false;
            keepIdle.nextElementSibling.textContent = 'Disabled';
        }
        if (e.target.id?.startsWith('param-')) {
            updateAllSummaries();
        }
    });
});
