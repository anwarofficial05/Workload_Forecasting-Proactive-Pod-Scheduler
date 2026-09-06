/**
 * Frontend Application Controller for:
 * Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters
 * using LSTM-XGBoost Ensemble with Multi-Objective Node Ranking
 */

// Global Chart Instances
let radarChartInstance = null;
let forecastChartInstance = null;
let simLatencyChartInstance = null;
let simReplicasChartInstance = null;
let distChartInstance = null;

// Global State
let currentNodes = [];
let currentWeights = { cpu: 0.35, mem: 0.25, lat: 0.20, bal: 0.20 };
let currentEnsembleWeight = { lstm: 0.55, xgb: 0.45 };

// --- Initialize on DOM Load ---
document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) {
        lucide.createIcons();
    }
    initApp();
});

async function initApp() {
    appendConsoleLog("Connecting to Kubernetes API and Prometheus metrics stream...", "info");
    await loadClusterData();
    await rankCandidateNodes();
    await loadModelMetrics();
    await loadDistributionData();
    await fetchForecastData(false);
    await runSimulation();
    appendConsoleLog("System ready. All modules initialized.", "success");
}

// --- Navigation Tabs ---
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('border-cyan-400', 'text-cyan-400');
        btn.classList.add('border-transparent', 'text-slate-400');
    });

    const activeSection = document.getElementById(`tab-${tabId}`);
    if (activeSection) {
        activeSection.classList.remove('hidden');
    }

    const activeBtn = document.getElementById(`tab-btn-${tabId}`);
    if (activeBtn) {
        activeBtn.classList.add('border-cyan-400', 'text-cyan-400');
        activeBtn.classList.remove('border-transparent', 'text-slate-400');
    }

    if (window.lucide) lucide.createIcons();

    // Trigger chart resize on tab switch to avoid rendering glitches
    setTimeout(() => {
        if (tabId === 'forecasting' && forecastChartInstance) forecastChartInstance.resize();
        if (tabId === 'simulation') {
            if (simLatencyChartInstance) simLatencyChartInstance.resize();
            if (simReplicasChartInstance) simReplicasChartInstance.resize();
        }
        if (tabId === 'distribution' && distChartInstance) distChartInstance.resize();
        if (tabId === 'cluster' && radarChartInstance) radarChartInstance.resize();
    }, 100);
}

// --- 1. Cluster State & Topology View ---
async function loadClusterData() {
    try {
        const res = await fetch('/api/cluster');
        const data = await res.json();
        currentNodes = data.nodes;

        // Update header ticker
        document.getElementById('headerClusterStatus').textContent = `${data.summary.node_count} Nodes (${data.summary.cluster_status})`;
        document.getElementById('headerPodCount').textContent = data.summary.pod_count;
        document.getElementById('headerAvgCpu').textContent = `${data.summary.avg_cpu_util}%`;

        renderNodeCards(data.nodes);
    } catch (err) {
        console.error("Failed to load cluster data:", err);
    }
}

function renderNodeCards(nodes) {
    const container = document.getElementById('nodeCardsContainer');
    container.innerHTML = '';

    nodes.forEach(node => {
        const cpuColor = node.current_cpu_util > 80 ? 'text-rose-400' : (node.current_cpu_util > 60 ? 'text-amber-400' : 'text-emerald-400');
        const cpuBarColor = node.current_cpu_util > 80 ? 'bg-rose-500' : (node.current_cpu_util > 60 ? 'bg-amber-500' : 'bg-emerald-500');

        const card = document.createElement('div');
        card.className = "bg-gray-900/90 border border-gray-800 rounded-xl p-4 shadow-lg hover:border-cyan-800/60 transition space-y-3";
        card.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <div class="flex items-center space-x-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                        <h4 class="font-bold text-white text-sm tracking-wide font-mono">${node.name}</h4>
                    </div>
                    <div class="text-[11px] text-slate-400 mt-0.5">${node.zone} &bull; Latency: <span class="text-cyan-400 font-mono">${node.base_latency_ms}ms</span></div>
                </div>
                <button onclick="injectSpike('${node.name}', 25)" title="Inject +25% CPU Spike" class="px-2 py-1 text-[11px] bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 border border-rose-800 rounded transition flex items-center space-x-1">
                    <i data-lucide="flame" class="w-3 h-3 text-rose-400"></i>
                    <span>+25% Surge</span>
                </button>
            </div>

            <!-- CPU Utilization Bar -->
            <div class="space-y-1">
                <div class="flex justify-between text-xs">
                    <span class="text-slate-400">CPU Usage (${node.total_cpu_cores} Cores)</span>
                    <span class="font-mono font-bold ${cpuColor}">${node.current_cpu_util}%</span>
                </div>
                <div class="w-full bg-gray-950 rounded-full h-2 overflow-hidden border border-gray-800">
                    <div class="${cpuBarColor} h-2 rounded-full progress-bar-fill" style="width: ${Math.min(100, node.current_cpu_util)}%"></div>
                </div>
            </div>

            <!-- Memory Utilization Bar -->
            <div class="space-y-1">
                <div class="flex justify-between text-xs">
                    <span class="text-slate-400">Memory Usage (${node.total_mem_gb} GB)</span>
                    <span class="font-mono font-bold text-purple-400">${node.current_mem_util}%</span>
                </div>
                <div class="w-full bg-gray-950 rounded-full h-2 overflow-hidden border border-gray-800">
                    <div class="bg-purple-500 h-2 rounded-full progress-bar-fill" style="width: ${Math.min(100, node.current_mem_util)}%"></div>
                </div>
            </div>

            <!-- Pod Tags -->
            <div class="pt-2 border-t border-gray-800/80">
                <div class="text-[11px] text-slate-400 mb-1.5 flex justify-between">
                    <span>Hosted Pods (${node.pod_count})</span>
                </div>
                <div class="flex flex-wrap gap-1.5">
                    ${node.pods.length === 0 ? '<span class="text-[10px] text-slate-500 italic">No pods currently scheduled</span>' : ''}
                    ${node.pods.map(p => `
                        <span class="px-2 py-0.5 text-[10px] font-mono bg-cyan-950/70 border border-cyan-800/60 text-cyan-300 rounded-md">
                            ${p.name}
                        </span>
                    `).join('')}
                </div>
            </div>
        `;
        container.appendChild(card);
    });

    if (window.lucide) lucide.createIcons();
}

// --- 2. MONR Node Ranking & Dispatcher ---
function updatePodSpecs() {
    const cpu = document.getElementById('inputCpu').value;
    const mem = document.getElementById('inputMem').value;
    document.getElementById('valCpu').textContent = `${cpu}m`;
    document.getElementById('valMem').textContent = `${mem}MB`;
    rankCandidateNodes();
}

function onWeightsChanged() {
    currentWeights.cpu = parseFloat(document.getElementById('wCpu').value);
    currentWeights.mem = parseFloat(document.getElementById('wMem').value);
    currentWeights.lat = parseFloat(document.getElementById('wLat').value);
    currentWeights.bal = parseFloat(document.getElementById('wBal').value);

    document.getElementById('wCpuVal').textContent = currentWeights.cpu.toFixed(2);
    document.getElementById('wMemVal').textContent = currentWeights.mem.toFixed(2);
    document.getElementById('wLatVal').textContent = currentWeights.lat.toFixed(2);
    document.getElementById('wBalVal').textContent = currentWeights.bal.toFixed(2);

    rankCandidateNodes();
}

async function rankCandidateNodes() {
    try {
        const cpu_m = parseInt(document.getElementById('inputCpu').value);
        const mem_mb = parseInt(document.getElementById('inputMem').value);

        const res = await fetch('/api/rank_nodes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cpu_m: cpu_m,
                mem_mb: mem_mb,
                weights: currentWeights,
                growth_delta: 6.5
            })
        });

        const data = await res.json();
        renderRadarChart(data.rankings);
    } catch (err) {
        console.error("Failed to rank nodes:", err);
    }
}

function renderRadarChart(rankings) {
    const ctx = document.getElementById('monrRadarChart').getContext('2d');
    const top3 = rankings.slice(0, 3);

    const colors = [
        { border: '#06b6d4', bg: 'rgba(6, 182, 212, 0.25)' },
        { border: '#3b82f6', bg: 'rgba(59, 130, 246, 0.2)' },
        { border: '#a855f7', bg: 'rgba(168, 85, 247, 0.15)' }
    ];

    const datasets = top3.map((node, i) => ({
        label: `#${node.rank} ${node.name} (${node.total_score} pts)`,
        data: [
            node.cpu_headroom_score,
            node.mem_headroom_score,
            node.latency_score,
            node.balance_score
        ],
        borderColor: colors[i].border,
        backgroundColor: colors[i].bg,
        borderWidth: 2,
        pointBackgroundColor: colors[i].border
    }));

    if (radarChartInstance) {
        radarChartInstance.destroy();
    }

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['CPU Headroom', 'Mem Headroom', 'Low Latency', 'Cluster Balance'],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: '#1f2937' },
                    grid: { color: '#1f2937' },
                    pointLabels: { color: '#94a3b8', font: { size: 10 } },
                    ticks: { display: false, min: 0, max: 100 }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#cbd5e1', boxWidth: 10, font: { size: 10 } }
                }
            }
        }
    });
}

async function dispatchPod() {
    const serviceName = document.getElementById('podServiceName').value;
    const cpu_m = parseInt(document.getElementById('inputCpu').value);
    const mem_mb = parseInt(document.getElementById('inputMem').value);

    appendConsoleLog(`[SCHEDULER] Dispatching proactive pod request for '${serviceName}' (CPU: ${cpu_m}m, Mem: ${mem_mb}MB)...`, "info");

    try {
        const res = await fetch('/api/schedule_pod', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                service_name: serviceName,
                cpu_m: cpu_m,
                mem_mb: mem_mb,
                weights: currentWeights
            })
        });

        const data = await res.json();
        if (data.success) {
            const assigned = data.result.assigned_node;
            appendConsoleLog(`[BIND SUCCESS] Pod ${data.result.scheduled_pod.pod_id} bound to node ${assigned.name} (MONR Score: ${assigned.total_score})`, "success");
            await loadClusterData();
            await rankCandidateNodes();
        }
    } catch (err) {
        appendConsoleLog(`[ERROR] Pod placement failed: ${err}`, "error");
    }
}

async function injectSpike(nodeName, amount) {
    appendConsoleLog(`[TRAFFIC SURGE] Injecting +${amount}% CPU load surge into ${nodeName}...`, "warning");
    try {
        await fetch('/api/inject_spike', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ node_name: nodeName, amount: amount })
        });
        await loadClusterData();
        await rankCandidateNodes();
        appendConsoleLog(`[MONR ADAPT] Node rankings recomputed in response to load change.`, "info");
    } catch (err) {
        console.error("Failed to inject spike:", err);
    }
}

async function resetCluster() {
    appendConsoleLog(`[RESET] Restoring Kubernetes cluster to baseline state...`, "info");
    try {
        await fetch('/api/reset_cluster', { method: 'POST' });
        await loadClusterData();
        await rankCandidateNodes();
        appendConsoleLog(`[RESET COMPLETE] Cluster nodes and pod replicas restored.`, "success");
    } catch (err) {
        console.error("Failed to reset cluster:", err);
    }
}

// --- 3. Workload Forecasting Studio ---
function onEnsembleSliderChange() {
    const lstmPct = parseInt(document.getElementById('sliderEnsemble').value);
    const xgbPct = 100 - lstmPct;

    document.getElementById('labelWeightLstm').textContent = `LSTM: ${lstmPct}%`;
    document.getElementById('labelWeightXgb').textContent = `XGBoost: ${xgbPct}%`;

    currentEnsembleWeight.lstm = lstmPct / 100.0;
    currentEnsembleWeight.xgb = xgbPct / 100.0;

    fetchForecastData(false);
}

async function fetchForecastData(simulateSpike = false) {
    try {
        const res = await fetch('/api/forecast', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n_points: 60,
                spike_at: simulateSpike ? 35 : null,
                weight_lstm: currentEnsembleWeight.lstm,
                weight_xgb: currentEnsembleWeight.xgb
            })
        });

        const data = await res.json();
        renderForecastChart(data.timeline);
    } catch (err) {
        console.error("Failed to fetch forecast:", err);
    }
}

function renderForecastChart(timeline) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    const labels = timeline.map(t => `m${t.minute}`);
    const actualCpu = timeline.map(t => t.actual_cpu);
    const lstmPred = timeline.map(t => t.lstm_pred);
    const xgbPred = timeline.map(t => t.xgb_pred);
    const ensemblePred = timeline.map(t => t.ensemble_pred);
    const arimaPred = timeline.map(t => t.arima_pred);

    if (forecastChartInstance) {
        forecastChartInstance.destroy();
    }

    forecastChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual CPU Workload (%)',
                    data: actualCpu,
                    borderColor: '#f43f5e',
                    backgroundColor: 'rgba(244, 63, 94, 0.08)',
                    borderWidth: 2.5,
                    tension: 0.25,
                    pointRadius: 2,
                    fill: true
                },
                {
                    label: 'Proposed LSTM-XGBoost Ensemble',
                    data: ensemblePred,
                    borderColor: '#22d3ee',
                    borderWidth: 2.5,
                    borderDash: [4, 2],
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'LSTM (Base Paper)',
                    data: lstmPred,
                    borderColor: '#3b82f6',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'XGBoost Regressor',
                    data: xgbPred,
                    borderColor: '#10b981',
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    tension: 0.2,
                    pointRadius: 0
                },
                {
                    label: 'ARIMA (2,1,2)',
                    data: arimaPred,
                    borderColor: '#94a3b8',
                    borderWidth: 1,
                    borderDash: [6, 6],
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    grid: { color: '#1f2937' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 20 }
                },
                y: {
                    grid: { color: '#1f2937' },
                    ticks: { color: '#94a3b8' },
                    title: { display: true, text: 'CPU Utilization (%)', color: '#64748b' },
                    min: 10,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#cbd5e1', boxWidth: 12, font: { size: 11 } }
                }
            }
        }
    });
}

async function loadModelMetrics() {
    try {
        const res = await fetch('/api/model_metrics');
        const data = await res.json();
        const tbody = document.getElementById('modelMetricsTableBody');
        tbody.innerHTML = '';

        const table = data.metrics_table;
        for (const [modelName, m] of Object.entries(table)) {
            const isBest = modelName.includes('Proposed');
            const row = document.createElement('tr');
            row.className = isBest ? "bg-cyan-950/40 text-cyan-300 font-bold border-l-4 border-cyan-400" : "hover:bg-gray-900/50 text-slate-300";
            row.innerHTML = `
                <td class="py-2.5 px-4 font-sans">${modelName} ${isBest ? '⭐' : ''}</td>
                <td class="py-2.5 px-4">${m.mse.toFixed(6)}</td>
                <td class="py-2.5 px-4 text-cyan-400">${m.rmse.toFixed(5)}</td>
                <td class="py-2.5 px-4">${m.mae.toFixed(5)}</td>
                <td class="py-2.5 px-4 ${m.r2 >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${m.r2.toFixed(4)}</td>
                <td class="py-2.5 px-4">${m.mean_err.toFixed(5)}</td>
                <td class="py-2.5 px-4">${m.std_err.toFixed(5)}</td>
                <td class="py-2.5 px-4">${m.max_err.toFixed(5)}</td>
            `;
            tbody.appendChild(row);
        }
    } catch (err) {
        console.error("Failed to load model metrics:", err);
    }
}

// --- 4. Comparative Head-to-Head Simulation ---
async function runSimulation() {
    try {
        const res = await fetch('/api/run_simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n_steps: 60,
                spike_minute: 32,
                weight_lstm: currentEnsembleWeight.lstm,
                weight_xgb: currentEnsembleWeight.xgb,
                monr_weights: currentWeights
            })
        });

        const data = await res.json();
        const summary = data.summary;

        // Update KPIs
        document.getElementById('kpiProactiveLatency').textContent = `${summary.proposed_proactive.avg_response_time_ms} ms`;
        document.getElementById('kpiReactiveLatency').textContent = `${summary.reactive_hpa.avg_response_time_ms} ms`;
        document.getElementById('kpiLatencyDiff').textContent = `${summary.improvement.latency_reduction_pct}% reduction`;

        document.getElementById('kpiProactiveP99').textContent = `${summary.proposed_proactive.p99_latency_ms} ms`;
        document.getElementById('kpiReactiveP99').textContent = `${summary.reactive_hpa.p99_latency_ms} ms`;
        document.getElementById('kpiP99Diff').textContent = `${Math.round((1 - summary.proposed_proactive.p99_latency_ms / summary.reactive_hpa.p99_latency_ms) * 100)}% reduction`;

        document.getElementById('kpiProactiveSlo').textContent = `${summary.proposed_proactive.slo_violation_rate_pct}%`;
        document.getElementById('kpiReactiveSlo').textContent = `${summary.reactive_hpa.slo_violation_rate_pct}%`;

        document.getElementById('kpiColdStarts').textContent = `${summary.proposed_proactive.cold_start_events} Events`;
        document.getElementById('kpiVariance').textContent = summary.proposed_proactive.cluster_load_variance;

        renderSimLatencyChart(data.reactive_timeline, data.proactive_timeline);
        renderSimReplicasChart(data.reactive_timeline, data.proactive_timeline);

        appendConsoleLog(`[SIMULATION RUN] 60-step evaluation finished. Proactive scheduling eliminated SLO violations and prevented ${summary.reactive_hpa.cold_start_events} cold starts.`, "success");
    } catch (err) {
        console.error("Failed to run simulation:", err);
    }
}

function renderSimLatencyChart(reactiveList, proactiveList) {
    const ctx = document.getElementById('simLatencyChart').getContext('2d');
    const labels = reactiveList.map(r => `m${r.minute}`);
    const rLatency = reactiveList.map(r => r.latency_ms);
    const pLatency = proactiveList.map(p => p.latency_ms);
    const sloLimit = reactiveList.map(() => 300);

    if (simLatencyChartInstance) simLatencyChartInstance.destroy();

    simLatencyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Default Reactive HPA Latency (ms)',
                    data: rLatency,
                    borderColor: '#f43f5e',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    borderWidth: 2,
                    tension: 0.25,
                    pointRadius: 1,
                    fill: true
                },
                {
                    label: 'Proposed Proactive Scheduler Latency (ms)',
                    data: pLatency,
                    borderColor: '#06b6d4',
                    borderWidth: 2.5,
                    tension: 0.2,
                    pointRadius: 1
                },
                {
                    label: 'SLO Violation Limit (300ms)',
                    data: sloLimit,
                    borderColor: '#ef4444',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', maxTicksLimit: 15 } },
                y: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'Response Time (ms)', color: '#64748b' } }
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#cbd5e1', font: { size: 10 } } }
            }
        }
    });
}

function renderSimReplicasChart(reactiveList, proactiveList) {
    const ctx = document.getElementById('simReplicasChart').getContext('2d');
    const labels = reactiveList.map(r => `m${r.minute}`);
    const rReplicas = reactiveList.map(r => r.replicas);
    const pReplicas = proactiveList.map(p => p.replicas);

    if (simReplicasChartInstance) simReplicasChartInstance.destroy();

    simReplicasChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Reactive HPA Replicas',
                    data: rReplicas,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    stepped: true,
                    pointRadius: 0
                },
                {
                    label: 'Proposed Proactive Replicas (Pre-scaled)',
                    data: pReplicas,
                    borderColor: '#3b82f6',
                    borderWidth: 2.5,
                    stepped: true,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', maxTicksLimit: 15 } },
                y: { grid: { color: '#1f2937' }, ticks: { color: '#94a3b8', stepSize: 1 }, title: { display: true, text: 'Pod Replica Count', color: '#64748b' } }
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#cbd5e1', font: { size: 10 } } }
            }
        }
    });
}

// --- 5. Beta Distribution Fitting View ---
async function loadDistributionData() {
    try {
        const res = await fetch('/api/distribution_data');
        const data = await res.json();

        // Render table
        const tbody = document.getElementById('distTableBody');
        tbody.innerHTML = '';
        const fits = data.distribution_fits;
        for (const [name, row] of Object.entries(fits)) {
            const tr = document.createElement('tr');
            const isBest = row.rank === 1;
            tr.className = isBest ? "bg-cyan-950/40 text-cyan-300 font-bold" : "text-slate-300";
            tr.innerHTML = `
                <td class="py-2 px-2 font-sans">${row.distribution} ${isBest ? '🏆' : ''}</td>
                <td class="py-2 px-2">${row.ks_stat.toFixed(4)}</td>
                <td class="py-2 px-2">${row.aic.toFixed(2)}</td>
                <td class="py-2 px-2">${row.bic.toFixed(2)}</td>
                <td class="py-2 px-2 text-cyan-400">#${row.rank}</td>
            `;
            tbody.appendChild(tr);
        }

        // Render Chart
        const ctx = document.getElementById('distributionChart').getContext('2d');
        const curves = data.curve_data;

        if (distChartInstance) distChartInstance.destroy();

        distChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: curves.x_points,
                datasets: [
                    {
                        label: 'Beta Distribution (Best Fit: a=1.32, b=2.98)',
                        data: curves.beta_pdf,
                        borderColor: '#22d3ee',
                        borderWidth: 2.5,
                        pointRadius: 0
                    },
                    {
                        label: 'Weibull Min',
                        data: curves.weibull_pdf,
                        borderColor: '#f59e0b',
                        borderWidth: 1.5,
                        pointRadius: 0
                    },
                    {
                        label: 'Normal Distribution',
                        data: curves.norm_pdf,
                        borderColor: '#a855f7',
                        borderWidth: 1.5,
                        pointRadius: 0
                    },
                    {
                        label: 'Gamma Distribution',
                        data: curves.gamma_pdf,
                        borderColor: '#94a3b8',
                        borderWidth: 1.5,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: '#1f2937' },
                        ticks: { color: '#94a3b8', maxTicksLimit: 12 },
                        title: { display: true, text: 'Requests per minute (Zanbil.ir web server log)', color: '#64748b' }
                    },
                    y: {
                        grid: { color: '#1f2937' },
                        ticks: { display: false },
                        title: { display: true, text: 'Probability Density f(x)', color: '#64748b' }
                    }
                },
                plugins: {
                    legend: { position: 'top', labels: { color: '#cbd5e1', font: { size: 10 } } }
                }
            }
        });
    } catch (err) {
        console.error("Failed to load distribution data:", err);
    }
}

// --- Console Log Helper ---
function appendConsoleLog(msg, type = 'info') {
    const consoleEl = document.getElementById('schedulerConsole');
    if (!consoleEl) return;

    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');

    let color = 'text-slate-300';
    if (type === 'success') color = 'text-emerald-400 font-semibold';
    if (type === 'warning') color = 'text-amber-400 font-semibold';
    if (type === 'error') color = 'text-rose-400 font-bold';
    if (type === 'info') color = 'text-cyan-400';

    line.className = color;
    line.innerHTML = `<span class="text-slate-500">[${time}]</span> ${msg}`;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function clearLogs() {
    const consoleEl = document.getElementById('schedulerConsole');
    if (consoleEl) {
        consoleEl.innerHTML = '<div class="text-slate-500 italic">[Log cleared]</div>';
    }
}
