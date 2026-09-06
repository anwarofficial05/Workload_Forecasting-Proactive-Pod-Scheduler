"""
Verification script for Flask App & Endpoints
"""
from app import app
import json

client = app.test_client()

def test_endpoints():
    print("Testing GET / ...")
    r = client.get('/')
    assert r.status_code == 200, f"GET / failed with {r.status_code}"
    print("[OK] GET / (HTML delivered)")

    print("Testing GET /api/status ...")
    r = client.get('/api/status')
    assert r.status_code == 200
    data = json.loads(r.data)
    print("[OK] GET /api/status:", data['project_title'])

    print("Testing GET /api/cluster ...")
    r = client.get('/api/cluster')
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] GET /api/cluster: {len(data['nodes'])} nodes, {len(data['active_deployments'])} pods")

    print("Testing POST /api/forecast ...")
    r = client.post('/api/forecast', json={'n_points': 60, 'spike_at': 35, 'weight_lstm': 0.6, 'weight_xgb': 0.4})
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] POST /api/forecast: {len(data['timeline'])} timeline points returned")

    print("Testing POST /api/rank_nodes ...")
    r = client.post('/api/rank_nodes', json={'cpu_m': 500, 'mem_mb': 512})
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] POST /api/rank_nodes: Top node is {data['top_node']['name']} (score: {data['top_node']['total_score']})")

    print("Testing POST /api/schedule_pod ...")
    r = client.post('/api/schedule_pod', json={'service_name': 'test-service', 'cpu_m': 400, 'mem_mb': 256})
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] POST /api/schedule_pod: {data['message']}")

    print("Testing POST /api/run_simulation ...")
    r = client.post('/api/run_simulation', json={'n_steps': 60, 'spike_minute': 32})
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] POST /api/run_simulation: Avg Proactive Latency={data['summary']['proposed_proactive']['avg_response_time_ms']}ms vs Reactive={data['summary']['reactive_hpa']['avg_response_time_ms']}ms")

    print("Testing GET /api/distribution_data ...")
    r = client.get('/api/distribution_data')
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] GET /api/distribution_data: Best fit = {data['base_paper_metadata']['best_fit']}")

    print("Testing GET /api/model_metrics ...")
    r = client.get('/api/model_metrics')
    assert r.status_code == 200
    data = json.loads(r.data)
    print(f"[OK] GET /api/model_metrics: Models evaluated = {list(data['metrics_table'].keys())}")

    print("Testing POST /api/reset_cluster ...")
    r = client.post('/api/reset_cluster')
    assert r.status_code == 200
    print("[OK] POST /api/reset_cluster")

    print("\nALL ENDPOINTS & ARCHITECTURAL MODULES PASSED SUCCESSFULLY! [PASS]")

if __name__ == '__main__':
    test_endpoints()
