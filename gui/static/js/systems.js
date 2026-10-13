/* Shared supercomputer selection for Results, Compare and Datasets. */

function availableSupercomputers() {
    const registered = Object.keys(AppState.supercomputers || {});
    const recorded = AppState.results.map(run => run.supercomputer);
    return [...new Set([...registered, ...recorded, ...Object.keys(AppState.datasets)])]
        .filter(Boolean)
        .sort((a, b) => supercomputerLabel(a).localeCompare(supercomputerLabel(b)));
}

function supercomputerLabel(system) {
    return AppState.supercomputers?.[system]?.label || system.replaceAll('_', ' ');
}

function ensureSupercomputerSelection() {
    const systems = availableSupercomputers();
    if (!systems.includes(AppState.activeSystem)) {
        AppState.activeSystem = AppState.results[0]?.supercomputer || systems[0] || null;
    }
    return AppState.activeSystem;
}

function runsForSelectedSupercomputer() {
    const system = ensureSupercomputerSelection();
    return AppState.results.filter(run => run.supercomputer === system);
}

function renderSupercomputerSidebars() {
    const selected = ensureSupercomputerSelection();
    const systems = availableSupercomputers();
    for (const tab of ['results', 'compare', 'datasets']) {
        const navigation = document.getElementById(`${tab}-systems`);
        if (!navigation) continue;
        navigation.innerHTML = systems.map(system => {
            const runs = AppState.results.filter(run => run.supercomputer === system).length;
            const clusters = AppState.datasets[system] || {};
            const jobs = Object.values(clusters).reduce((sum, cluster) => sum + Math.max(0, cluster.rows || 0), 0);
            const count = tab === 'datasets'
                ? (Object.keys(clusters).length ? `${jobs.toLocaleString()} jobs` : 'Workloads')
                : `${runs} ${runs === 1 ? 'run' : 'runs'}`;
            return `<button type="button" class="system-choice ${selected === system ? 'active' : ''}"
                         data-system="${escapeHtml(system)}" aria-pressed="${selected === system}">
                <span class="system-choice-name">${escapeHtml(supercomputerLabel(system))}</span>
                <span class="system-choice-count">${count}</span>
            </button>`;
        }).join('');
        navigation.querySelectorAll('.system-choice').forEach(button => {
            button.addEventListener('click', () => selectSupercomputer(button.dataset.system));
        });
    }
}

function selectSupercomputer(system) {
    if (system === AppState.activeSystem) return;
    const focusedNavigation = document.activeElement?.closest('.system-list')?.id;
    AppState.activeSystem = system;
    AppState.selectedCompareRuns = [];
    compareData = [];
    document.getElementById('compare-results').classList.add('hidden');
    showResultsList();
    renderSupercomputerSidebars();
    renderResultsForSystem(AppState.results);
    renderCompareSelector();
    renderSelectedDatasets();
    if (focusedNavigation) {
        document.getElementById(focusedNavigation).querySelector('.system-choice.active')?.focus({preventScroll: true});
    }
    window.scrollTo({ top: 0, behavior: 'instant' });
}
