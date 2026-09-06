/**
 * Main Frontend Application Controller
 * Connects UI tabs, backend REST APIs, Simulation loops, and Chart Visualizations.
 */

// Global State
const State = {
  currentTab: 'overview',
  simMode: 'proposed',
  simPlaying: false,
  simTimer: null,
  simStepIndex: 0,
  currentWorkload: null,
  currentForecast: null,
  currentRankings: null,
  clusterNodes: [],
  activePods: [],
  rankingWeights: {
    w_cpu: 0.35,
    w_mem: 0.30,
    w_lat: 0.20,
    w_bal: 0.15
  }
};

// DOM Content Loaded Initializer
document.addEventListener('DOMContentLoaded', async () => {
  if (window.lucide) lucide.createIcons();
  AppCharts.initDefaults();
  AppCharts.initLiveLatencyChart();

  await loadInitialData();
  await refreshNodeRankings();
  await loadInitialClusterState();
});

// Tab Switching
function switchTab(tabId) {
  State.currentTab = tabId;

  // Toggle active button styles
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.getElementById(`tab-${tabId}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Toggle visible pane
  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.add('hidden');
  });
  const targetPane = document.getElementById(`pane-${tabId}`);
  if (targetPane) {
    targetPane.classList.remove('hidden');
  }

  // Refresh charts if rendering on hidden tab
  if (tabId === 'data-engine' && State.currentWorkload) {
    AppCharts.renderSyntheticWorkload(State.currentWorkload);
  } else if (tabId === 'models' && State.currentForecast) {
    AppCharts.renderForecastingStudio(State.currentForecast);
  } else if (tabId === 'node-ranking' && State.currentRankings) {
    AppCharts.renderNodeRadar(State.currentRankings.rankings);
  }

  if (window.lucide) lucide.createIcons();
}

// Modal Toggle
function toggleModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;
  modal.classList.toggle('hidden');
}

// Load Initial Data from Backend
async function loadInitialData() {
  try {
    // 1. Fetch Distribution Fitting Data (Base Paper Table I)
    const fitRes = await fetch('/api/data/fit-distributions');
    const fitData = await fitRes.json();
    populateDistributionTable(fitData.comparison_table);
    AppCharts.renderDistributionFit(fitData.curve_data);

    // 2. Fetch Initial Synthetic Workload & Decomposition
    await regenerateSyntheticWorkload();

    // 3. Fetch Initial Forecasting Studio Data
    const fcRes = await fetch('/api/forecast/predict');
    State.currentForecast = await fcRes.json();
    AppCharts.renderForecastingStudio(State.currentForecast);
    populateMetricsTable(State.currentForecast.metrics);
    AppCharts.renderFeatureImportance(State.currentForecast.feature_importance);

  } catch (err) {
    console.error('Error loading initial data:', err);
  }
}

// Populate Distribution Table (Table I)
function populateDistributionTable(tableData) {
  const tbody = document.getElementById('distributionTableBody');
  if (!tbody) return;

  tbody.innerHTML = tableData.map(row => `
    <tr class="${row.best ? 'bg-cyan-950/20 text-cyan-300 font-bold' : 'text-slate-300'}">
      <td class="p-3">#${row.rank}</td>
      <td class="p-3">${row.distribution}</td>
      <td class="p-3">${row.ks_stat}</td>
      <td class="p-3">${row.p_value}</td>
      <td class="p-3 ${row.best ? 'text-cyan-400' : ''}">${row.aic.toLocaleString()}</td>
      <td class="p-3 ${row.best ? 'text-cyan-400' : ''}">${row.bic.toLocaleString()}</td>
      <td class="p-3">
        ${row.best ? '<span class="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">Best Fit (Rank 1)</span>' : '<span class="text-slate-500">Rejected</span>'}
      </td>
    </tr>
  `).join('');
}

// Re-fetch Distribution Data
async function refreshDistributionData() {
  const res = await fetch('/api/data/fit-distributions');
  const fitData = await res.json();
  populateDistributionTable(fitData.comparison_table);
  AppCharts.renderDistributionFit(fitData.curve_data);
}

// Regenerate Synthetic Workload
async function regenerateSyntheticWorkload() {
  const scenario = document.getElementById('workloadScenarioSelect')?.value || 'beta_k6';
  const duration = parseInt(document.getElementById('workloadDurationSelect')?.value || '60');

  const res = await fetch('/api/data/generate-workload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario, duration })
  });
  State.currentWorkload = await res.json();
  AppCharts.renderSyntheticWorkload(State.currentWorkload);

  // Decomposition
  const decompRes = await fetch('/api/data/decomposition');
  const decompData = await decompRes.json();
  AppCharts.renderDecomposition(decompData.decomposition);

  // Refresh forecasts
  const fcRes = await fetch('/api/forecast/predict');
  State.currentForecast = await fcRes.json();
  AppCharts.renderForecastingStudio(State.currentForecast);
}

// Populate Metrics Comparison Table (Table III & IV)
function populateMetricsTable(metrics) {
  const tbody = document.getElementById('metricsTableBody');
  if (!tbody) return;

  const rows = [
    { name: 'ARIMA (2,1,2) [Benchmark]', key: 'ARIMA', class: 'text-slate-400' },
    { name: 'GRU (Hidden=50) [Benchmark]', key: 'GRU', class: 'text-slate-400' },
    { name: 'LSTM (Base Paper Champion)', key: 'LSTM', class: 'text-blue-400 font-semibold' },
    { name: 'XGBoost (Burst Adapter)', key: 'XGBoost', class: 'text-amber-400 font-semibold' },
    { name: 'Proposed LSTM-XGBoost Ensemble', key: 'Proposed_Ensemble', class: 'bg-cyan-950/30 text-cyan-300 font-bold border-l-2 border-cyan-400' },
    { name: 'Vanilla Reactive HPA (Lagged)', key: 'Reactive_HPA', class: 'text-rose-400' }
  ];

  tbody.innerHTML = rows.map(r => {
    const m = metrics[r.key];
    if (!m) return '';
    return `
      <tr class="${r.class}">
        <td class="p-2.5">${r.name}</td>
        <td class="p-2.5">${m.mse}</td>
        <td class="p-2.5">${m.rmse}</td>
        <td class="p-2.5">${m.mae}</td>
        <td class="p-2.5">${m.r2}</td>
        <td class="p-2.5">${m.std_error}</td>
        <td class="p-2.5">${m.max_error}</td>
      </tr>
    `;
  }).join('');
}

// -------------------------------------------------------------
// Multi-Objective Node Ranking Handlers
// -------------------------------------------------------------
function onWeightSliderChange() {
  const w_cpu = parseFloat(document.getElementById('slider-w-cpu').value);
  const w_mem = parseFloat(document.getElementById('slider-w-mem').value);
  const w_lat = parseFloat(document.getElementById('slider-w-lat').value);
  const w_bal = parseFloat(document.getElementById('slider-w-bal').value);

  // Normalize
  const total = (w_cpu + w_mem + w_lat + w_bal) || 1.0;
  State.rankingWeights = {
    w_cpu: +(w_cpu / total).toFixed(2),
    w_mem: +(w_mem / total).toFixed(2),
    w_lat: +(w_lat / total).toFixed(2),
    w_bal: +(w_bal / total).toFixed(2)
  };

  document.getElementById('label-w-cpu').innerText = State.rankingWeights.w_cpu;
  document.getElementById('label-w-mem').innerText = State.rankingWeights.w_mem;
  document.getElementById('label-w-lat').innerText = State.rankingWeights.w_lat;
  document.getElementById('label-w-bal').innerText = State.rankingWeights.w_bal;

  refreshNodeRankings();
}

function setWeightPreset(preset) {
  document.querySelectorAll('.btn-preset').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`preset-${preset}`)?.classList.add('active');

  if (preset === 'balanced') {
    setSliderValues(0.35, 0.30, 0.20, 0.15);
  } else if (preset === 'latency') {
    setSliderValues(0.20, 0.20, 0.50, 0.10);
  } else if (preset === 'packing') {
    setSliderValues(0.50, 0.40, 0.05, 0.05);
  } else if (preset === 'balance') {
    setSliderValues(0.20, 0.20, 0.10, 0.50);
  }
}

function setSliderValues(cpu, mem, lat, bal) {
  document.getElementById('slider-w-cpu').value = cpu;
  document.getElementById('slider-w-mem').value = mem;
  document.getElementById('slider-w-lat').value = lat;
  document.getElementById('slider-w-bal').value = bal;
  onWeightSliderChange();
}

async function refreshNodeRankings() {
  try {
    const res = await fetch('/api/ranking/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        weights: State.rankingWeights,
        pod_demand: { cpu_m: 450, mem_mb: 512 }
      })
    });
    State.currentRankings = await res.json();
    ClusterView.renderRankedNodeCards(State.currentRankings.rankings);
    AppCharts.renderNodeRadar(State.currentRankings.rankings);

    const rationaleElem = document.getElementById('rankingDecisionRationale');
    if (rationaleElem) {
      rationaleElem.innerHTML = `<strong>Scheduler Extender Decision:</strong> ${State.currentRankings.rationale}`;
    }
  } catch (err) {
    console.error('Error refreshing rankings:', err);
  }
}

// -------------------------------------------------------------
// Cluster Simulator & Proactive Scheduling Execution
// -------------------------------------------------------------
async function loadInitialClusterState() {
  const res = await fetch('/api/cluster/reset', { method: 'POST' });
  const data = await res.json();
  State.clusterNodes = data.nodes;
  State.activePods = data.active_pods;
  ClusterView.renderNodes(State.clusterNodes, null, State.activePods);
}

function setSimMode(mode) {
  State.simMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`mode-btn-${mode}`)?.classList.add('active');
}

function toggleSimPlay() {
  State.simPlaying = !State.simPlaying;
  const playBtnText = document.getElementById('playText');
  const playIcon = document.getElementById('playIcon');

  if (State.simPlaying) {
    playBtnText.innerText = 'Pause Simulation';
    playIcon.setAttribute('data-lucide', 'pause');
    State.simTimer = setInterval(stepSimOnce, 1200);
  } else {
    playBtnText.innerText = 'Start Live Simulation';
    playIcon.setAttribute('data-lucide', 'play');
    clearInterval(State.simTimer);
  }
  if (window.lucide) lucide.createIcons();
}

async function stepSimOnce() {
  try {
    const res = await fetch('/api/cluster/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: State.simMode,
        step_index: State.simStepIndex
      })
    });
    const step = await res.json();
    State.simStepIndex += 1;

    // Update UI Elements
    document.getElementById('simTimeLabel').innerText = step.tick;
    document.getElementById('simTrafficLabel').innerText = `${Math.round(step.traffic_actual)} req/m`;
    document.getElementById('simReplicasLabel').innerText = `${step.running_replicas} Pods`;

    if (step.scaling_action) {
      document.getElementById('simSchedulerActionBanner').innerText = step.scaling_action;
    }

    // Render nodes & event stream
    ClusterView.renderNodes(step.nodes, step.target_node, step.active_pods);
    ClusterView.renderEventStream(step.recent_events);

    // Update live latency chart
    AppCharts.updateLiveLatency(step.tick, step.latency_ms);

  } catch (err) {
    console.error('Error during simulation step:', err);
    toggleSimPlay();
  }
}

async function resetSimCluster() {
  if (State.simPlaying) toggleSimPlay();
  State.simStepIndex = 0;
  await loadInitialClusterState();
  AppCharts.initLiveLatencyChart();
  document.getElementById('simTimeLabel').innerText = 'T+00m';
  document.getElementById('simSchedulerActionBanner').innerText = 'Cluster reset to baseline.';
}

// -------------------------------------------------------------
// Comparative Benchmark Execution
// -------------------------------------------------------------
async function runFullBenchmark() {
  const btn = document.getElementById('btnRunBenchmark');
  if (btn) btn.innerHTML = '<span class="animate-spin mr-1">⏳</span> Running Benchmark...';

  try {
    const res = await fetch('/api/benchmark/run', { method: 'POST' });
    const data = await res.json();
    const comp = data.comparison;

    // Update Reactive Card
    document.getElementById('bm-reac-latency').innerText = `${comp.reactive.avg_latency_ms} ms`;
    document.getElementById('bm-reac-slo').innerText = `${comp.reactive.slo_violation_rate_pct}%`;
    document.getElementById('bm-reac-peak').innerText = `${comp.reactive.max_latency_ms} ms`;
    document.getElementById('bm-reac-imb').innerText = `σ = ${comp.reactive.avg_load_imbalance_std}%`;

    // Update LSTM Card
    document.getElementById('bm-lstm-latency').innerText = `${comp.lstm_only.avg_latency_ms} ms`;
    document.getElementById('bm-lstm-slo').innerText = `${comp.lstm_only.slo_violation_rate_pct}%`;
    document.getElementById('bm-lstm-peak').innerText = `${comp.lstm_only.max_latency_ms} ms`;
    document.getElementById('bm-lstm-imb').innerText = `σ = ${comp.lstm_only.avg_load_imbalance_std}%`;

    // Update Proposed Card
    document.getElementById('bm-prop-latency').innerText = `${comp.proposed.avg_latency_ms} ms`;
    document.getElementById('bm-prop-slo').innerText = `${comp.proposed.slo_violation_rate_pct}%`;
    document.getElementById('bm-prop-peak').innerText = `${comp.proposed.max_latency_ms} ms`;
    document.getElementById('bm-prop-imb').innerText = `σ = ${comp.proposed.avg_load_imbalance_std}%`;

    // Render comparative charts
    AppCharts.renderBenchmarkCharts(data);

    if (btn) btn.innerHTML = '<i data-lucide="check" class="w-4 h-4 mr-1"></i> Benchmark Completed';
    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error('Error running benchmark:', err);
    if (btn) btn.innerText = 'Run Head-to-Head Benchmark';
  }
}

// -------------------------------------------------------------
// K8s Webhook Interactive Testing
// -------------------------------------------------------------
async function testFilterWebhook() {
  const samplePayload = {
    Pod: {
      metadata: { name: "checkout-service-pod-902" },
      spec: {
        containers: [{
          name: "checkout",
          resources: { requests: { cpu: "500m", memory: "512Mi" } }
        }]
      }
    }
  };

  const res = await fetch('/k8s/v1/filter', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(samplePayload)
  });
  const data = await res.json();
  document.getElementById('codeFilterOutput').innerText = JSON.stringify(data, null, 2);
}

async function testPrioritizeWebhook() {
  const samplePayload = {
    Pod: {
      metadata: { name: "checkout-service-pod-902" },
      spec: {
        containers: [{
          name: "checkout",
          resources: { requests: { cpu: "500m", memory: "512Mi" } }
        }]
      }
    }
  };

  const res = await fetch('/k8s/v1/prioritize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(samplePayload)
  });
  const data = await res.json();
  document.getElementById('codePrioritizeOutput').innerText = JSON.stringify(data, null, 2);
}

// Print Viva Summary
function printReviewReport() {
  window.print();
}
