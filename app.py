"""
Flask Web Application for Final Year Project:
"Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters
using LSTM-XGBoost Ensemble with Multi-Objective Node Ranking"
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.stats as stats
import json

from ml_engine import (
    forecaster,
    generate_base_paper_workload,
    compute_distribution_fits
)
from scheduler_engine import (
    cluster_state,
    calculate_monr_scores,
    schedule_pod_proactively,
    run_comparative_simulation
)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status': 'online',
        'project_title': 'Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters',
        'ensemble_models': ['PyTorch-LSTM (50 units)', 'XGBoost-Regressor (Tree-Boosted)', 'Hybrid-Weighted-Fusion'],
        'baseline_models': ['ARIMA (2,1,2)', 'GRU (50 units)'],
        'decision_engine': 'Multi-Objective Node Ranking (MONR)',
        'base_paper': 'Amirullah & Saikhu (IEEE IAICT 2025)'
    })

@app.route('/api/cluster', methods=['GET'])
def get_cluster():
    nodes_data = [node.to_dict() for node in cluster_state.nodes]
    avg_cpu = float(np.mean([n['current_cpu_util'] for n in nodes_data]))
    avg_mem = float(np.mean([n['current_mem_util'] for n in nodes_data]))
    total_pods = len(cluster_state.active_deployments)
    
    return jsonify({
        'nodes': nodes_data,
        'active_deployments': cluster_state.active_deployments,
        'summary': {
            'node_count': len(nodes_data),
            'pod_count': total_pods,
            'avg_cpu_util': round(avg_cpu, 1),
            'avg_mem_util': round(avg_mem, 1),
            'cluster_status': 'Healthy' if avg_cpu < 80 else 'Warning'
        }
    })

@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    data = request.get_json() or {}
    n_points = int(data.get('n_points', 60))
    spike_at = data.get('spike_at')
    if spike_at is not None:
        spike_at = int(spike_at)
        
    w_lstm = float(data.get('weight_lstm', 0.55))
    w_xgb = float(data.get('weight_xgb', 0.45))
    
    # Generate workload series based on Beta distribution
    df = generate_base_paper_workload(n_points=n_points, spike_at=spike_at)
    cpu_series = list(df['cpu_usage'].values)
    
    # Run predictions across the timeline
    seq_len = 5
    timeline = []
    
    for i in range(seq_len, len(cpu_series)):
        window = cpu_series[:i]
        pred = forecaster.predict_next(window, weights=(w_lstm, w_xgb))
        timeline.append({
            'minute': int(df['minute'].iloc[i]),
            'actual_requests': float(df['requests'].iloc[i]),
            'actual_cpu': float(df['cpu_usage'].iloc[i]),
            'actual_mem': float(df['mem_usage'].iloc[i]),
            'lstm_pred': float(pred['lstm']),
            'xgb_pred': float(pred['xgboost']),
            'ensemble_pred': float(pred['ensemble']),
            'gru_pred': float(pred['gru']),
            'arima_pred': float(pred['arima'])
        })
        
    # Evaluate next-step prediction
    latest_pred = forecaster.predict_next(cpu_series, weights=(w_lstm, w_xgb))
    
    return jsonify({
        'timeline': timeline,
        'latest_prediction': {
            'lstm': float(latest_pred['lstm']),
            'xgboost': float(latest_pred['xgboost']),
            'ensemble': float(latest_pred['ensemble']),
            'gru': float(latest_pred['gru']),
            'arima': float(latest_pred['arima']),
            'weights': latest_pred['weights']
        }
    })

@app.route('/api/rank_nodes', methods=['POST'])
def rank_nodes():
    data = request.get_json() or {}
    pod_req = {
        'cpu_m': int(data.get('cpu_m', 600)),
        'mem_mb': int(data.get('mem_mb', 512))
    }
    growth_delta = float(data.get('growth_delta', 6.5))
    weights = data.get('weights', {
        'cpu': 0.35,
        'mem': 0.25,
        'lat': 0.20,
        'bal': 0.20
    })
    
    rankings = calculate_monr_scores(pod_req, forecast_growth_delta=growth_delta, weights=weights)
    return jsonify({
        'pod_request': pod_req,
        'weights_used': weights,
        'rankings': rankings,
        'top_node': rankings[0] if rankings else None
    })

@app.route('/api/schedule_pod', methods=['POST'])
def schedule_pod():
    data = request.get_json() or {}
    service_name = data.get('service_name', 'microservice-task')
    cpu_m = int(data.get('cpu_m', 500))
    mem_mb = int(data.get('mem_mb', 512))
    weights = data.get('weights')
    
    result = schedule_pod_proactively(service_name, cpu_m=cpu_m, mem_mb=mem_mb, weights=weights)
    return jsonify({
        'success': True,
        'message': f"Pod {result['scheduled_pod']['pod_id']} proactively scheduled on node {result['assigned_node']['name']} (Rank #1)",
        'result': result
    })

@app.route('/api/run_simulation', methods=['POST'])
def run_simulation():
    data = request.get_json() or {}
    n_steps = int(data.get('n_steps', 60))
    spike_min = int(data.get('spike_minute', 32))
    w_lstm = float(data.get('weight_lstm', 0.55))
    w_xgb = float(data.get('weight_xgb', 0.45))
    monr_weights = data.get('monr_weights')
    
    sim_result = run_comparative_simulation(
        n_steps=n_steps,
        spike_minute=spike_min,
        ensemble_weights=(w_lstm, w_xgb),
        monr_weights=monr_weights
    )
    return jsonify(sim_result)

@app.route('/api/distribution_data', methods=['GET'])
def get_distribution_data():
    fits = compute_distribution_fits()
    # Generate sample histogram and theoretical curve points
    x_vals = np.linspace(84, 4590, 80).tolist()
    # Beta curve (normalized to density)
    a, b = 1.32, 2.98
    norm_x = (np.array(x_vals) - 84) / (4590 - 84)
    beta_pdf = (stats.beta.pdf(norm_x, a, b) / (4590 - 84)).tolist()
    norm_pdf = stats.norm.pdf(x_vals, 1505.65, 857.62).tolist()
    weibull_pdf = stats.weibull_min.pdf(x_vals, 1.84, 84, 1700.2).tolist()
    gamma_pdf = stats.gamma.pdf(x_vals, 3.41, 0, 441.5).tolist()
    
    return jsonify({
        'distribution_fits': fits,
        'curve_data': {
            'x_points': [round(x, 1) for x in x_vals],
            'beta_pdf': beta_pdf,
            'norm_pdf': norm_pdf,
            'weibull_pdf': weibull_pdf,
            'gamma_pdf': gamma_pdf
        },
        'base_paper_metadata': {
            'title': 'CPU Usage Forecasting for Load Balancing in Kubernetes Using LSTM: A Synthetic Traffic Simulation Approach',
            'authors': 'Amirullah & Ahmad Saikhu (Sepuluh Nopember Institute of Technology, 2025 IEEE IAICT)',
            'dataset': 'Zanbil.ir E-Commerce Web Server Log (6,754 min entries)',
            'best_fit': 'Beta Distribution (AIC: -4072.11, BIC: -4044.84, a=1.32, b=2.98)'
        }
    })

@app.route('/api/model_metrics', methods=['GET'])
def get_model_metrics():
    metrics = forecaster.evaluate_models_on_dataset()
    return jsonify({
        'metrics_table': metrics,
        'base_paper_comparison': {
            'note': 'Extends base paper Table III & IV by integrating XGBoost and Multi-Objective Placement.',
            'base_paper_lstm_mse': 0.00001053,
            'base_paper_lstm_rmse': 0.00324451,
            'base_paper_gru_mse': 0.00001225,
            'base_paper_arima_mse': 0.00001771
        }
    })

@app.route('/api/reset_cluster', methods=['POST'])
def reset_cluster():
    cluster_state.reset()
    return jsonify({'success': True, 'message': 'Kubernetes cluster state restored to baseline.'})

@app.route('/api/inject_spike', methods=['POST'])
def inject_spike():
    data = request.get_json() or {}
    node_name = data.get('node_name', 'worker-us-east-a')
    amount = float(data.get('amount', 25.0))
    
    for node in cluster_state.nodes:
        if node.name == node_name:
            node.current_cpu_util = min(98.0, node.current_cpu_util + amount)
            node.current_mem_util = min(95.0, node.current_mem_util + (amount * 0.6))
            break
            
    return jsonify({
        'success': True,
        'message': f"Injected +{amount}% load spike on {node_name}"
    })

if __name__ == '__main__':
    print("Starting Final Year Project Implementation Web Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)
