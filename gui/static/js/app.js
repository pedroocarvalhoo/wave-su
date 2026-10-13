/* ═══════════════════════════════════════════════════════════════════════════
   WAVE GUI — App Shell (Routing, API Client, Utils)
   ═══════════════════════════════════════════════════════════════════════════ */

// ── API Client ──────────────────────────────────────────────────────────────

const API = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`GET ${url}: ${res.status}`);
        return res.json();
    },

    async post(url, body = {}) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const error = await res.json().catch(() => ({}));
            const detail = error.detail || error.error;
            const message = Array.isArray(detail)
                ? detail.map(item => item.msg.replace(/^Value error, /, '')).join('; ')
                : detail;
            throw new Error(message || `Request failed (${res.status})`);
        }
        return res.json();
    },

    sse(url, onMessage, onError) {
        const source = new EventSource(url);
        source.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                onMessage(data);
            } catch (err) {
                console.error('SSE parse error:', err);
            }
        };
        source.onerror = (e) => {
            if (onError) onError(e);
            source.close();
        };
        return source;
    },
};


// ── State ───────────────────────────────────────────────────────────────────

const AppState = {
    supercomputers: null,
    nodeConfig: null,
    parameterMetadata: null,
    results: [],
    selectedCompareRuns: [],
    currentResultId: null,
    simRunning: false,
    sseSource: null,
    activeSystem: null,
    datasets: {},
};


// ── Tab Routing ─────────────────────────────────────────────────────────────

function initRouting() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            switchTab(tabId);
        });
    });

    // Hash-based routing
    const hash = location.hash.replace('#', '') || 'simulation';
    switchTab(hash);
}

function switchTab(tabId) {
    // Deactivate all tabs and panels
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

    // Activate selected
    const navTab = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    const panel = document.getElementById(`tab-${tabId}`);
    if (navTab) navTab.classList.add('active');
    if (panel) panel.classList.add('active');

    location.hash = tabId;
    window.scrollTo({ top: 0, behavior: 'instant' });

    // Trigger load on tab switch
    if (tabId === 'results') loadResults();
    if (tabId === 'compare') loadCompareSelector();
    if (tabId === 'datasets') loadDatasets();
}


// ── Toast Notifications ─────────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> <span>${message}</span>`;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function showResultsToast(runId) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast success clickable-toast';
    toast.style.position = 'relative';
    toast.style.cursor = 'pointer';
    toast.style.borderLeft = '3px solid var(--accent-emerald)';
    toast.style.display = 'flex';
    toast.style.flexDirection = 'column';
    toast.style.alignItems = 'flex-start';
    toast.style.gap = 'var(--space-xs)';

    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: var(--space-sm); font-weight: 700; color: var(--text-primary); padding-right: var(--space-lg);">
            <span>✅</span>
            <span>Simulation Completed!</span>
        </div>
        <div style="font-size: 0.82rem; color: var(--accent-blue); text-decoration: underline; margin-left: 24px;">
            Click here to view results
        </div>
    `;

    // Create close button dynamically to ensure robust click handler and stopPropagation
    const closeBtn = document.createElement('button');
    closeBtn.className = 'toast-close-btn';
    closeBtn.innerHTML = '&times;';
    closeBtn.title = 'Dismiss';
    closeBtn.onclick = (e) => {
        e.stopPropagation();
        toast.remove();
    };
    toast.appendChild(closeBtn);

    toast.onclick = () => {
        switchTab('results');
        showResultDetail(runId);
        toast.remove();
    };

    container.appendChild(toast);
}


// ── Utility Functions ───────────────────────────────────────────────────────

function formatNumber(n, decimals = 1) {
    if (n === null || n === undefined || isNaN(n)) return '—';
    const num = parseFloat(n);
    if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(decimals) + 'M';
    if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(decimals) + 'k';
    return num.toFixed(decimals);
}

function formatMetricNumber(value, decimals = 2) {
    if (value == null || !Number.isFinite(Number(value))) return '—';
    return Number(value).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
}

function formatDuration(seconds) {
    if (seconds == null || isNaN(seconds)) return '—';
    const s = parseFloat(seconds);
    if (s < 60) return `${s.toFixed(0)}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    return `${h}h ${m}m`;
}

function formatTimestamp(ts) {
    if (!ts) return '—';
    // Format: 20260530_112755 -> 2026-05-30 11:27:55
    const m = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (m) return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}`;
    return ts;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}


// ── Section Toggle ──────────────────────────────────────────────────────────

function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) section.classList.toggle('collapsed');
}


// ── Image Modal ─────────────────────────────────────────────────────────────

function openImageModal(src) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('image-modal-img');
    img.src = src;
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    modal.classList.remove('active');
    document.body.style.overflow = '';
}

// Close on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeImageModal();
});


// ── Status Indicator ────────────────────────────────────────────────────────

function setStatus(text, type = 'idle') {
    document.body.classList.toggle('simulation-running', type === 'running');
    const dot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    dot.className = 'status-dot';
    if (type === 'running') dot.classList.add('running');
    if (type === 'error') dot.classList.add('error');
    statusText.textContent = text;
}


// ── Terminal ────────────────────────────────────────────────────────────────

function clearTerminal() {
    const terminal = document.getElementById('sim-terminal');
    terminal.innerHTML = '';
}

function appendTerminalLine(text, cls = '') {
    const terminal = document.getElementById('sim-terminal');
    const line = document.createElement('div');
    line.className = `terminal-line ${cls}`;
    line.textContent = text;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}


// ── Create Tooltip HTML ─────────────────────────────────────────────────────

function tooltipHtml(text) {
    return `<span class="tooltip-trigger">?<span class="tooltip-text">${escapeHtml(text)}</span></span>`;
}


// ── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    initRouting();

    try {
        // Load supercomputers and parameters in parallel
        const [scData, paramData] = await Promise.all([
            API.get('/api/supercomputers'),
            API.get('/api/parameters'),
        ]);

        AppState.supercomputers = scData.supercomputers;
        AppState.nodeConfig = scData.node_config;
        AppState.parameterMetadata = paramData.parameters;
        renderSupercomputerSidebars();

        initSimulationForm();
        setStatus('Ready');
    } catch (err) {
        console.error('Init error:', err);
        showToast('Failed to load configuration: ' + err.message, 'error');
        setStatus('Error', 'error');
    }

    // Check if simulation is running
    try {
        const simStatus = await API.get('/api/simulation/running');
        if (simStatus.running) {
            AppState.simRunning = true;
            setStatus('Simulation running...', 'running');
            connectSSE();
        }
    } catch (e) {}
});
