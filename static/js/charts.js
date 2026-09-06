/**
 * Chart.js Visualization Managers
 * Implements interactive charts for:
 * 1. Distribution Fitting (Beta vs Normal vs Gamma)
 * 2. Synthetic Traffic & Time-Series Decomposition
 * 3. AI Forecasting Studio (LSTM, XGBoost, Ensemble, Reactive HPA)
 * 4. Multi-Objective Node Radar Analysis
 * 5. Real-time Cluster Latency & SLO Tracking
 * 6. Head-to-Head Comparative Benchmark
 */

const AppCharts = {
  distributionChart: null,
  workloadChart: null,
  decompositionChart: null,
  forecastingChart: null,
  featureImpChart: null,
  radarChart: null,
  liveLatencyChart: null,
  benchmarkLatencyChart: null,
  benchmarkReplicasChart: null,

  // Global Chart.js defaults for dark theme
  initDefaults() {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = '#1f293d';
    Chart.defaults.font.family = 'JetBrains Mono, Inter, monospace';
    Chart.defaults.font.size = 11;
  },

  // 1. Distribution Fit Chart
  renderDistributionFit(curveData) {
    const ctx = document.getElementById('chartDistributionFit')?.getContext('2d');
    if (!ctx) return;
    if (this.distributionChart) this.distributionChart.destroy();

    this.distributionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: curveData.x,
        datasets: [
          {
            label: 'Beta Fit (α=1.32, β=2.98) - Best AIC',
            data: curveData.beta_pdf,
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.15)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.3,
            pointRadius: 0
          },
          {
            label: 'Normal Distribution Fit',
            data: curveData.norm_pdf,
            borderColor: '#f59e0b',
            borderWidth: 1.5,
            borderDash: [4, 4],
            fill: false,
            tension: 0.3,
            pointRadius: 0
          },
          {
            label: 'Gamma Distribution Fit',
            data: curveData.gamma_pdf,
            borderColor: '#a855f7',
            borderWidth: 1.5,
            borderDash: [2, 2],
            fill: false,
            tension: 0.3,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12 } },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          x: { title: { display: true, text: 'Requests / Minute (Scaled Demand)' } },
          y: { title: { display: true, text: 'Probability Density' } }
        }
      }
    });
  },

  // 2. Synthetic Workload Chart
  renderSyntheticWorkload(workloadData) {
    const ctx = document.getElementById('chartSyntheticWorkload')?.getContext('2d');
    if (!ctx) return;
    if (this.workloadChart) this.workloadChart.destroy();

    this.workloadChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: workloadData.time_labels,
        datasets: [
          {
            label: 'Requests / Min (Beta Variate)',
            data: workloadData.requests_per_minute,
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.08)',
            borderWidth: 2,
            fill: true,
            tension: 0.2,
            pointRadius: 1,
            yAxisID: 'y'
          },
          {
            label: 'Pod CPU Utilization (%)',
            data: workloadData.cpu_usage_pct,
            borderColor: '#10b981',
            borderWidth: 2,
            fill: false,
            tension: 0.2,
            pointRadius: 1,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12 } }
        },
        scales: {
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            title: { display: true, text: 'Req / Min' }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            grid: { drawOnChartArea: false },
            title: { display: true, text: 'CPU %' },
            min: 0,
            max: 100
          }
        }
      }
    });
  },

  // 3. Time Series Decomposition
  renderDecomposition(decomp) {
    const ctx = document.getElementById('chartDecomposition')?.getContext('2d');
    if (!ctx) return;
    if (this.decompositionChart) this.decompositionChart.destroy();

    const labels = decomp.original.map((_, i) => `T+${i}m`);
    this.decompositionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Original CPU %',
            data: decomp.original,
            borderColor: '#94a3b8',
            borderWidth: 1.5,
            pointRadius: 0
          },
          {
            label: 'Trend Component',
            data: decomp.trend,
            borderColor: '#38bdf8',
            borderWidth: 2,
            pointRadius: 0
          },
          {
            label: 'Seasonal Cycle (12m)',
            data: decomp.seasonal,
            borderColor: '#a855f7',
            borderWidth: 1.5,
            borderDash: [3, 3],
            pointRadius: 0
          },
          {
            label: 'Residual',
            data: decomp.residual,
            borderColor: '#f43f5e',
            borderWidth: 1,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 10 } },
          tooltip: { mode: 'index', intersect: false }
        }
      }
    });
  },

  // 4. Forecasting Studio Master Comparison Chart
  renderForecastingStudio(forecastData) {
    const ctx = document.getElementById('chartForecastingStudio')?.getContext('2d');
    if (!ctx) return;
    if (this.forecastingChart) this.forecastingChart.destroy();

    const labels = forecastData.indices.map(idx => `T+${idx}m`);

    this.forecastingChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Actual Workload Demand',
            data: forecastData.actual,
            borderColor: '#ffffff',
            borderWidth: 2.5,
            pointRadius: 1,
            tension: 0.2
          },
          {
            label: 'Proposed Ensemble (LSTM+XGBoost)',
            data: forecastData.ensemble,
            borderColor: '#38bdf8',
            borderWidth: 2.5,
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            fill: true,
            pointRadius: 2,
            tension: 0.2
          },
          {
            label: 'LSTM (Base Paper)',
            data: forecastData.lstm,
            borderColor: '#3b82f6',
            borderWidth: 1.8,
            borderDash: [3, 3],
            pointRadius: 0,
            tension: 0.2
          },
          {
            label: 'XGBoost Forecaster',
            data: forecastData.xgboost,
            borderColor: '#fbbf24',
            borderWidth: 1.5,
            borderDash: [2, 2],
            pointRadius: 0,
            tension: 0.2
          },
          {
            label: 'Vanilla Reactive HPA (Lagged)',
            data: forecastData.reactive_hpa,
            borderColor: '#f43f5e',
            borderWidth: 1.8,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12 } },
          tooltip: {
            callbacks: {
              label: (context) => `${context.dataset.label}: ${context.parsed.y} req/m`
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Time Horizon (Minutes)' } },
          y: { title: { display: true, text: 'Workload Demand (Req/Min)' } }
        }
      }
    });
  },

  // 5. XGBoost Feature Importance Bar Chart
  renderFeatureImportance(featureData) {
    const ctx = document.getElementById('chartFeatureImportance')?.getContext('2d');
    if (!ctx) return;
    if (this.featureImpChart) this.featureImpChart.destroy();

    const labels = featureData.map(f => f.feature.replace(/_/g, ' '));
    const values = featureData.map(f => f.importance);

    this.featureImpChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Importance %',
          data: values,
          backgroundColor: '#38bdf8',
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { max: 40, title: { display: true, text: 'Relative Weight (%)' } }
        }
      }
    });
  },

  // 6. Multi-Objective Node Radar Chart
  renderNodeRadar(rankings) {
    const ctx = document.getElementById('chartRadarObjectives')?.getContext('2d');
    if (!ctx) return;
    if (this.radarChart) this.radarChart.destroy();

    const colors = ['#38bdf8', '#10b981', '#f59e0b', '#a855f7'];
    const datasets = rankings.map((r, i) => ({
      label: r.name,
      data: [
        r.score_breakdown.cpu_headroom * 100,
        r.score_breakdown.mem_headroom * 100,
        r.score_breakdown.latency * 100,
        r.score_breakdown.load_balance * 100
      ],
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length] + '20',
      borderWidth: 2,
      pointRadius: 3
    }));

    this.radarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['CPU Headroom', 'RAM Headroom', 'Network Locality', 'Load Balance'],
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: { display: false },
            grid: { color: '#1f293d' },
            pointLabels: { font: { size: 10, weight: '600' }, color: '#cbd5e1' }
          }
        },
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 8, font: { size: 10 } } }
        }
      }
    });
  },

  // 7. Real-Time Cluster Latency Chart
  initLiveLatencyChart() {
    const ctx = document.getElementById('chartLiveLatency')?.getContext('2d');
    if (!ctx) return;
    if (this.liveLatencyChart) this.liveLatencyChart.destroy();

    this.liveLatencyChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Cluster Response Time (ms)',
            data: [],
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.2,
            pointRadius: 2
          },
          {
            label: 'SLO Violation Threshold (200ms)',
            data: [],
            borderColor: '#f43f5e',
            borderWidth: 1.5,
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          y: { min: 0, max: 400, title: { display: true, text: 'Latency (ms)' } }
        }
      }
    });
  },

  updateLiveLatency(tick, latency) {
    if (!this.liveLatencyChart) return;
    const chart = this.liveLatencyChart;
    if (chart.data.labels.length > 25) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
      chart.data.datasets[1].data.shift();
    }
    chart.data.labels.push(tick);
    chart.data.datasets[0].data.push(latency);
    chart.data.datasets[1].data.push(200); // SLO line
    chart.update('none');
  },

  // 8. Comparative Benchmark Charts
  renderBenchmarkCharts(benchmarkData) {
    // Latency comparison
    const ctx1 = document.getElementById('chartBenchmarkLatency')?.getContext('2d');
    if (ctx1) {
      if (this.benchmarkLatencyChart) this.benchmarkLatencyChart.destroy();
      const comp = benchmarkData.comparison;
      const n = comp.proposed.time_series_latency.length;
      const labels = Array.from({ length: n }, (_, i) => `T+${i + 1}m`);

      this.benchmarkLatencyChart = new Chart(ctx1, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Reactive Vanilla K8s',
              data: comp.reactive.time_series_latency,
              borderColor: '#f43f5e',
              borderWidth: 2,
              tension: 0.2,
              pointRadius: 0
            },
            {
              label: 'LSTM-Only Proactive',
              data: comp.lstm_only.time_series_latency,
              borderColor: '#3b82f6',
              borderWidth: 1.8,
              tension: 0.2,
              pointRadius: 0
            },
            {
              label: 'Proposed Ensemble + MO Ranking',
              data: comp.proposed.time_series_latency,
              borderColor: '#10b981',
              borderWidth: 2.2,
              tension: 0.2,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top', labels: { boxWidth: 10 } } },
          scales: {
            y: { title: { display: true, text: 'Response Time (ms)' } }
          }
        }
      });
    }

    // Replicas comparison
    const ctx2 = document.getElementById('chartBenchmarkReplicas')?.getContext('2d');
    if (ctx2) {
      if (this.benchmarkReplicasChart) this.benchmarkReplicasChart.destroy();
      const comp = benchmarkData.comparison;
      const n = comp.proposed.time_series_replicas.length;
      const labels = Array.from({ length: n }, (_, i) => `T+${i + 1}m`);

      this.benchmarkReplicasChart = new Chart(ctx2, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Reactive Pod Count (Delayed)',
              data: comp.reactive.time_series_replicas,
              borderColor: '#f43f5e',
              borderWidth: 2,
              stepped: true
            },
            {
              label: 'Proposed Proactive Pod Count (Ahead of Surge)',
              data: comp.proposed.time_series_replicas,
              borderColor: '#10b981',
              borderWidth: 2,
              stepped: true
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: 'top', labels: { boxWidth: 10 } } },
          scales: {
            y: { min: 1, max: 12, title: { display: true, text: 'Running Pod Replicas' } }
          }
        }
      });
    }
  }
};
