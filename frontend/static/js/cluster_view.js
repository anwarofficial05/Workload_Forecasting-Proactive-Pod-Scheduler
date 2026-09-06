/**
 * Kubernetes Cluster Topology & Pod Visualizer
 * Renders heterogeneous nodes, CPU/Memory gauges, and animated Pod containers.
 */

const ClusterView = {
  renderNodes(nodes, lastPlacedNodeId = null, activePods = []) {
    const container = document.getElementById('clusterNodesContainer');
    if (!container) return;

    // Group pods by node
    const podsByNode = {};
    nodes.forEach(n => podsByNode[n.id] = []);
    activePods.forEach(p => {
      if (podsByNode[p.node_id]) {
        podsByNode[p.node_id].push(p);
      }
    });

    container.innerHTML = nodes.map(node => {
      const isWinner = (node.id === lastPlacedNodeId);
      const cpuPct = node.cpu_pct || Math.round((node.cpu_used_m / node.cpu_capacity_m) * 100);
      const memPct = node.mem_pct || Math.round((node.mem_used_mb / node.mem_capacity_mb) * 100);
      const podsOnThisNode = podsByNode[node.id] || [];

      // CPU status color
      let cpuColor = 'bg-cyan-500';
      if (cpuPct > 85) cpuColor = 'bg-rose-500';
      else if (cpuPct > 70) cpuColor = 'bg-amber-500';

      return `
        <div class="worker-node-card ${isWinner ? 'highlight-winner ring-2 ring-cyan-400' : ''} flex flex-col justify-between" id="card-${node.id}">
          <div>
            <!-- Node Header -->
            <div class="flex items-center justify-between border-b border-surface-border/60 pb-2 mb-3">
              <div class="flex items-center gap-2">
                <i data-lucide="server" class="w-4 h-4 ${isWinner ? 'text-cyan-400 animate-bounce' : 'text-slate-400'}"></i>
                <h4 class="text-xs font-bold font-mono text-white">${node.name}</h4>
              </div>
              <span class="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-elevated text-slate-400 border border-surface-border">
                ${node.zone}
              </span>
            </div>

            <!-- Resource Utilization Gauges -->
            <div class="space-y-2 text-[11px] font-mono mb-4">
              <!-- CPU Bar -->
              <div>
                <div class="flex justify-between mb-0.5">
                  <span class="text-slate-400">CPU: ${node.cpu_used_m}/${node.cpu_capacity_m}m</span>
                  <span class="${cpuPct > 80 ? 'text-rose-400 font-bold' : 'text-slate-300'}">${cpuPct}%</span>
                </div>
                <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div class="${cpuColor} h-full transition-all duration-300" style="width: ${Math.min(100, cpuPct)}%"></div>
                </div>
              </div>

              <!-- Memory Bar -->
              <div>
                <div class="flex justify-between mb-0.5">
                  <span class="text-slate-400">RAM: ${node.mem_used_mb}/${node.mem_capacity_mb}MB</span>
                  <span class="text-slate-300">${memPct}%</span>
                </div>
                <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div class="bg-blue-500 h-full transition-all duration-300" style="width: ${Math.min(100, memPct)}%"></div>
                </div>
              </div>

              <div class="flex justify-between text-[10px] text-slate-500 pt-1">
                <span>Net Latency: <strong class="text-slate-400">${node.latency_ms}ms</strong></span>
                <span>Pods: <strong class="text-cyan-400">${podsOnThisNode.length}</strong></span>
              </div>
            </div>
          </div>

          <!-- Pods Allocated on this Node -->
          <div class="pt-2 border-t border-surface-border/50">
            <span class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold block mb-1.5">
              Scheduled Pods
            </span>
            <div class="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
              ${podsOnThisNode.length === 0 ? '<span class="text-[10px] text-slate-600 italic">No pods currently</span>' : ''}
              ${podsOnThisNode.map(pod => `
                <span class="pod-pill ${pod.status === 'Running' ? 'running' : 'pending'}" title="${pod.id} (${pod.status}) - ${pod.type}">
                  <span class="w-1.5 h-1.5 rounded-full ${pod.status === 'Running' ? 'bg-emerald-400' : 'bg-amber-400 animate-ping'}"></span>
                  ${pod.name}
                </span>
              `).join('')}
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Re-initialize Lucide icons on newly inserted DOM elements
    if (window.lucide) {
      lucide.createIcons();
    }
  },

  renderRankedNodeCards(rankings) {
    const container = document.getElementById('rankedNodeCardsContainer');
    if (!container) return;

    container.innerHTML = rankings.map((r, i) => {
      const isTop = (i === 0);
      return `
        <div class="p-4 rounded-xl ${isTop ? 'bg-gradient-to-r from-cyan-950/20 via-surface-elevated to-surface-elevated border border-cyan-500/50 shadow-lg shadow-cyan-500/10' : 'bg-surface-elevated border border-surface-border'} transition-all">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div class="flex items-center gap-3">
              <div class="w-7 h-7 rounded-lg ${isTop ? 'bg-cyan-500 text-black' : 'bg-slate-800 text-slate-400'} font-mono font-bold flex items-center justify-center text-xs">
                #${r.rank}
              </div>
              <div>
                <div class="flex items-center gap-2">
                  <h4 class="text-sm font-bold text-white font-mono">${r.name}</h4>
                  ${isTop ? '<span class="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">Placement Winner</span>' : ''}
                </div>
                <span class="text-xs text-slate-400">Zone: ${r.zone} • Ingress Latency: ${r.latency_ms}ms</span>
              </div>
            </div>

            <div class="text-right">
              <span class="text-xs text-slate-400 block">Composite Score</span>
              <span class="text-base font-bold font-mono ${isTop ? 'text-cyan-300' : 'text-slate-200'}">
                ${r.composite_score.toFixed(4)}
              </span>
            </div>
          </div>

          <!-- Multi-Objective Breakdown Scores -->
          <div class="grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-surface-border/50 text-[11px] font-mono">
            <div class="p-1.5 rounded bg-surface-bg text-center">
              <span class="text-slate-500 block text-[9px]">CPU HEADROOM</span>
              <span class="text-cyan-400 font-bold">${(r.score_breakdown.cpu_headroom * 100).toFixed(1)}%</span>
            </div>
            <div class="p-1.5 rounded bg-surface-bg text-center">
              <span class="text-slate-500 block text-[9px]">RAM HEADROOM</span>
              <span class="text-blue-400 font-bold">${(r.score_breakdown.mem_headroom * 100).toFixed(1)}%</span>
            </div>
            <div class="p-1.5 rounded bg-surface-bg text-center">
              <span class="text-slate-500 block text-[9px]">NET LOCALITY</span>
              <span class="text-amber-400 font-bold">${(r.score_breakdown.latency * 100).toFixed(1)}%</span>
            </div>
            <div class="p-1.5 rounded bg-surface-bg text-center">
              <span class="text-slate-500 block text-[9px]">LOAD BALANCE</span>
              <span class="text-emerald-400 font-bold">${(r.score_breakdown.load_balance * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      `;
    }).join('');
  },

  renderEventStream(events) {
    const stream = document.getElementById('simEventLogStream');
    if (!stream) return;

    if (!events || events.length === 0) {
      stream.innerHTML = '<div class="text-slate-500">Awaiting events...</div>';
      return;
    }

    stream.innerHTML = events.map(ev => {
      let badgeClass = 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30';
      if (ev.mode === 'reactive') badgeClass = 'text-rose-400 bg-rose-500/10 border-rose-500/30';
      else if (ev.mode === 'lstm_only') badgeClass = 'text-blue-400 bg-blue-500/10 border-blue-500/30';

      return `
        <div class="p-2 rounded bg-surface-bg border border-surface-border/50 flex items-start gap-2">
          <span class="text-[10px] px-1.5 py-0.5 rounded border font-mono shrink-0 ${badgeClass}">${ev.tick}</span>
          <span class="text-slate-300 leading-snug">${ev.message}</span>
        </div>
      `;
    }).join('');
  }
};
