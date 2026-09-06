"""
Workload-Forecasting Driven Proactive Pod Scheduler
Main Web Application & REST API Server
Team Members: Niveesh R, Shanmugapriyan K S, Mohamed Anwar S
Guide: Mr. P. Manikanda Prabu (AP/CSE)
Institution: Anjalai Ammal Mahalingam Engineering College
"""

import os
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from backend.data_engine import DataEngine
from backend.models import WorkloadForecaster
from backend.node_ranker import MultiObjectiveNodeRanker
from backend.scheduler import ClusterSimulator
from backend.k8s_extender_api import create_k8s_extender_blueprint

app = Flask(
    __name__,
    static_folder="frontend/static",
    template_folder="frontend/templates"
)
CORS(app)

# Initialize core system modules
data_engine = DataEngine(random_seed=42)
forecaster = WorkloadForecaster(seq_len=5, hidden_units=50)
node_ranker = MultiObjectiveNodeRanker()
cluster_sim = ClusterSimulator(node_ranker=node_ranker)

# Register Kubernetes Scheduler Extender Blueprint
app.register_blueprint(create_k8s_extender_blueprint(node_ranker, cluster_sim))

# Pre-cache baseline traffic and fit
cached_workload = data_engine.generate_synthetic_workload(scenario="beta_k6", duration_minutes=60)
cached_forecast = forecaster.predict_all(cached_workload["requests_per_minute"])

# -------------------------------------------------------------
# Web UI Entry Point
# -------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------------------------------------
# System Status & Metadata
# -------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": "healthy",
        "project": "Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters using LSTM-XGBoost Ensemble with Multi-Objective Node Ranking",
        "institution": "Anjalai Ammal Mahalingam Engineering College",
        "department": "Department of Computer Science and Engineering",
        "guide": "Mr. P. Manikanda Prabu (AP/CSE)",
        "team": [
            {"name": "Niveesh R", "reg_no": "820423104058"},
            {"name": "Shanmugapriyan K S", "reg_no": "820423104072"},
            {"name": "Mohamed Anwar S", "reg_no": "820423104050"}
        ],
        "base_paper": {
            "title": "CPU Usage Forecasting for Load Balancing in Kubernetes Using LSTM: A Synthetic Traffic Simulation Approach",
            "authors": "Amirullah & Ahmad Saikhu",
            "venue": "IEEE IAICT 2025",
            "doi": "10.1109/IAICT65714.2025.11100781"
        },
        "modules_active": [
            "Data Collection & Synthetic Generator",
            "Beta Distribution Fitting Engine",
            "PyTorch LSTM Neural Network",
            "XGBoost Gradient Boosted Forecaster",
            "Dynamic Ensemble Fusion Layer",
            "Multi-Objective Node Ranking Engine",
            "Proactive Kubernetes Pod Scheduler",
            "SLA/SLO Telemetry & Benchmark Monitor"
        ]
    })

# -------------------------------------------------------------
# Module 1: Data Analytics & Distribution Fitting
# -------------------------------------------------------------
@app.route("/api/data/zanbil-stats", methods=["GET"])
def get_zanbil_stats():
    return jsonify({
        "dataset_name": "Zanbil.ir E-Commerce Server Logs",
        "source": "Harvard Dataverse (Zaker, 2019)",
        "total_records": 6754,
        "timeframe": "2019-01-22 03:56 to 2019-01-26 20:29",
        "mean_requests": 1505.65,
        "std_dev": 857.62,
        "min_requests": 84.0,
        "max_requests": 4590.0,
        "median_requests": 1594.0
    })

@app.route("/api/data/fit-distributions", methods=["GET"])
def get_fit_distributions():
    return jsonify(data_engine.fit_distributions(sample_size=1200))

@app.route("/api/data/generate-workload", methods=["POST"])
def generate_workload():
    global cached_workload, cached_forecast
    params = request.get_json(silent=True) or {}
    scenario = params.get("scenario", "beta_k6")
    duration = int(params.get("duration", 60))
    spike_at = int(params.get("spike_at", 35))
    spike_intensity = float(params.get("spike_intensity", 2.2))

    cached_workload = data_engine.generate_synthetic_workload(
        scenario=scenario,
        duration_minutes=duration,
        spike_at=spike_at,
        spike_intensity=spike_intensity
    )
    # Refresh predictions
    cached_forecast = forecaster.predict_all(cached_workload["requests_per_minute"])

    return jsonify(cached_workload)

@app.route("/api/data/decomposition", methods=["GET"])
def get_decomposition():
    series = cached_workload["cpu_usage_pct"]
    decomp = data_engine.decompose_time_series(series)
    acf_pacf = data_engine.compute_acf_pacf(series, max_lags=15)
    return jsonify({
        "decomposition": decomp,
        "acf_pacf": acf_pacf
    })

# -------------------------------------------------------------
# Module 2 & 3: Forecasting Studio (LSTM + XGBoost Ensemble)
# -------------------------------------------------------------
@app.route("/api/forecast/predict", methods=["GET", "POST"])
def get_forecast():
    global cached_forecast
    params = request.get_json(silent=True) or {}
    series = params.get("series")
    if series:
        cached_forecast = forecaster.predict_all(series)
    return jsonify(cached_forecast)

# -------------------------------------------------------------
# Module 4: Multi-Objective Node Ranking
# -------------------------------------------------------------
@app.route("/api/ranking/score", methods=["POST"])
def rank_nodes():
    params = request.get_json(silent=True) or {}
    weights = params.get("weights")
    if weights:
        node_ranker.update_weights(
            w_cpu=float(weights.get("w_cpu", 0.35)),
            w_mem=float(weights.get("w_mem", 0.30)),
            w_lat=float(weights.get("w_lat", 0.20)),
            w_bal=float(weights.get("w_bal", 0.15))
        )
    pod_demand = params.get("pod_demand", {"cpu_m": 450, "mem_mb": 512})
    result = node_ranker.rank_nodes(cluster_sim.nodes, pod_demand)
    return jsonify(result)

# -------------------------------------------------------------
# Module 5 & 6: Proactive Pod Scheduler & Cluster Simulator
# -------------------------------------------------------------
@app.route("/api/cluster/step", methods=["POST"])
def run_cluster_step():
    params = request.get_json(silent=True) or {}
    mode = params.get("mode", "proposed")
    step_num = int(params.get("step_index", cluster_sim.step_idx))

    traffic_series = cached_workload["requests_per_minute"]
    idx = step_num % len(traffic_series)
    cur_traffic = traffic_series[idx]

    # Forecast values for current step
    fc = {
        "lstm": cached_forecast["lstm"][min(idx, len(cached_forecast["lstm"])-1)],
        "ensemble": cached_forecast["ensemble"][min(idx, len(cached_forecast["ensemble"])-1)]
    }

    step_result = cluster_sim.run_step(cur_traffic, fc, mode=mode)
    return jsonify(step_result)

@app.route("/api/cluster/reset", methods=["POST"])
def reset_cluster():
    cluster_sim.reset_cluster()
    return jsonify({
        "status": "reset",
        "nodes": cluster_sim.nodes,
        "active_pods": cluster_sim.active_pods
    })

# -------------------------------------------------------------
# Module 7: Comprehensive Benchmark & Evaluation
# -------------------------------------------------------------
@app.route("/api/benchmark/run", methods=["POST"])
def run_benchmark():
    traffic_series = cached_workload["requests_per_minute"]
    results = cluster_sim.run_comparative_benchmark(traffic_series, cached_forecast)
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"================================================================")
    print(f"Workload-Forecasting Proactive Pod Scheduler Server Starting...")
    print(f"Final Year Project - Anjalai Ammal Mahalingam Engineering College")
    print(f"Listening on http://127.0.0.1:{port}")
    print(f"================================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
