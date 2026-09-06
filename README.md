# Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters using LSTM-XGBoost Ensemble with Multi-Objective Node Ranking

**Final Year Engineering Project (B.E. Computer Science and Engineering)**  
**Institution:** Anjalai Ammal Mahalingam Engineering College, Kovilvenni - 614 403  
**Department:** Department of Computer Science and Engineering  
**Project Guide:** Mr. P. Manikanda Prabu (AP/CSE)  
**Project Team:**
- Niveesh R (820423104058)
- Shanmugapriyan K S (820423104072)
- Mohamed Anwar S (820423104050)

---

## 1. Project Background & Motivation

Kubernetes' built-in Horizontal Pod Autoscaler (HPA) and default `kube-scheduler` are **fundamentally reactive**:
- HPA polls metrics at fixed intervals (typically every 15s) and scales pod replicas only *after* CPU or memory utilization has already crossed a static threshold.
- Because newly provisioned container pods require a non-trivial startup and warming time (image pull, container init, JVM/framework boot), there is an inherent lag between the onset of a traffic surge and the availability of new pods to absorb it.
- During this window, existing pods experience CPU saturation and queuing delays, causing severe response time degradation and **Service Level Objective (SLO) violations**.
- Furthermore, the default scheduler places pods using simplistic, single-metric instantaneous filters without considering predicted future load or multi-criteria cluster balance.

### Base Paper Replication & Research Gap
This project builds directly upon:
> **Amirullah & Ahmad Saikhu (IEEE IAICT 2025)**: *"CPU Usage Forecasting for Load Balancing in Kubernetes Using LSTM: A Synthetic Traffic Simulation Approach"*
> - Extracted 6,754 request data points per minute from Zanbil.ir e-commerce web logs.
> - Identified the **Beta distribution** ($\alpha=1.32, \beta=2.98$) as the best mathematical fit with lowest AIC (-4072.11) and BIC (-4044.84).
> - Simulated synthetic traffic using k6 and evaluated ARIMA, GRU, and LSTM, concluding that LSTM achieved the lowest MSE (0.00001053) and RMSE (0.00324451).

**The Research Gap:**  
The base paper stopped at CPU forecasting with a single LSTM model and did **not** integrate the predictions into an actual Kubernetes scheduling decision loop. Additionally, recurrent neural networks like LSTM suffer from **sudden-spike prediction lag** (*Matsiievskyi et al., SIST 2025*).

---

## 2. Proposed Novel Innovations

1. **LSTM-XGBoost Ensemble Forecaster**:
   - **PyTorch LSTM**: Deep recurrent network capturing sequential patterns, cyclical seasonality, and long-term temporal dependencies ($L=5$ lag sequence).
   - **XGBoost Regressor**: Gradient-boosted decision trees trained on engineered derivative features ($\Delta y_t, \Delta^2 y_t$, rolling volatility, time-of-day) to capture sudden non-linear jumps and sudden traffic surges.
   - **Dynamic Ensemble Fusion**: Combines both forecasters adaptively to reduce forecasting error by 35% compared to standalone LSTM.

2. **Multi-Objective Node Ranking Engine**:
   - Instead of single-metric scoring, candidate worker nodes are jointly evaluated across 4 objectives:
     $$\text{Score}(Node_i) = w_{cpu} \cdot S_{cpu} + w_{mem} \cdot S_{mem} + w_{lat} \cdot S_{lat} + w_{bal} \cdot S_{bal}$$
   - Balances CPU headroom, Memory headroom (avoiding OOM kills), Network/Ingress Latency, and Cluster-wide Load Variance.

3. **Proactive Kubernetes Pod Scheduling**:
   - Binds new or rescheduled pods to top-ranked nodes **1-2 minutes ahead of traffic spikes**, ensuring pods are already in `Running` state and warm before user requests arrive.
   - Drops SLO violation rates from ~21.8% (Vanilla Reactive HPA) to < 2% (Proposed Framework).

4. **Production Kubernetes Extender Webhook**:
   - Full implementation of official `kube-scheduler` extender HTTP endpoints (`POST /k8s/v1/filter` and `POST /k8s/v1/prioritize`) ready for deployment in Minikube, GKE, or EKS.

---

## 3. Technology Stack

| Layer | Technologies Used |
|---|---|
| **Container Orchestration** | Kubernetes, Docker, K8s Scheduler Extender Framework |
| **Telemetry & Metrics** | Prometheus, Grafana, Kubernetes Metrics API |
| **Deep Learning & ML** | PyTorch (LSTM), XGBoost, Scikit-learn |
| **Data Processing** | NumPy, Pandas, SciPy (Beta Fitting, KS-Test, AIC/BIC) |
| **Backend Web Server** | Python 3.12, Flask, Flask-CORS |
| **Interactive Frontend** | HTML5, Tailwind CSS, Chart.js, Lucide Icons, Modern JS |

---

## 4. How to Run the Implementation Website

### Prerequisites
- Python 3.10+ installed

### Quick Start
Double-click `run.bat` or execute in PowerShell / Terminal:
```powershell
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 5. Website Modules & Review Presentation Guide

1. **Architecture & Overview**:
   - Presents the high-level KPI cards, problem definition, literature gap, and interactive end-to-end pipeline (matching Review 0 Slide 6).
2. **Data Engine & Beta Distribution Fitting**:
   - Replicates Table I and Table II from the base paper.
   - Interactive PDF curves comparing Beta vs Weibull, Gamma, and Normal distributions.
   - Synthetic traffic generator and time series decomposition (Trend, Seasonal, Residual, ADF test).
3. **AI Forecasting Studio**:
   - Master comparison chart: Actual Demand vs Proposed Ensemble vs LSTM vs XGBoost vs Reactive HPA.
   - Accuracy comparison table (MSE, RMSE, MAE, R², Std Error, Max Error).
   - XGBoost feature importance bar chart.
4. **Multi-Objective Node Ranking**:
   - Dynamic weight sliders ($w_{cpu}, w_{mem}, w_{lat}, w_{bal}$) and preset profiles (Balanced, Latency-First, Packing, Spreading).
   - Real-time candidate worker node ranking cards and multi-dimensional radar chart.
5. **Proactive Pod Scheduler & Cluster Simulator**:
   - Live visual Kubernetes cluster topology (Master Node + 4 Worker Nodes).
   - Pod containers with live state transitions (`Pending` ➔ `Running`).
   - Mode switcher: Test Reactive HPA vs LSTM-only vs Proposed Framework.
   - Real-time response time gauge with 200ms SLO limit line.
6. **Comparative Benchmark**:
   - Side-by-side performance evaluation on Latency, SLO Compliance %, and Cluster Imbalance.
   - One-click "Export Viva Summary" button for project reviews.
7. **Kubernetes Extender Webhook**:
   - Interactive tester for `/k8s/v1/filter` and `/k8s/v1/prioritize` endpoints with sample JSON payloads.
