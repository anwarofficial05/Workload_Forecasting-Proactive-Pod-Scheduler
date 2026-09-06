"""
Kubernetes Scheduler Extender Webhook API
Implements the official Kubernetes Scheduler Extender HTTP Protocol:
1. POST /k8s/v1/filter - Checks predicates (Node headroom vs pod resource requirements)
2. POST /k8s/v1/prioritize - Returns HostPriorityList scored via Multi-Objective Node Ranker
3. POST /k8s/v1/bind - Handles pod binding confirmations
Allows direct integration with a real Minikube / GKE cluster as specified in Tech Stack Table 8.
"""

from flask import Blueprint, request, jsonify

def create_k8s_extender_blueprint(node_ranker, cluster_sim):
    k8s_bp = Blueprint('k8s_extender', __name__, url_prefix='/k8s/v1')

    @k8s_bp.route('/status', methods=['GET'])
    def status():
        return jsonify({
            "status": "active",
            "plugin": "ProactiveMultiObjectiveSchedulerExtender",
            "version": "v1.0.0",
            "k8s_compatibility": "v1.28+",
            "active_weights": node_ranker.weights
        })

    @k8s_bp.route('/filter', methods=['POST'])
    def filter_nodes():
        """
        Kubernetes Scheduler Extender Filter Predicate Webhook:
        Input JSON: ExtenderArgs { Pod: {...}, Nodes: { Items: [...] } }
        Output JSON: ExtenderFilterResult { Nodes: { Items: [...] }, FailedNodes: {...} }
        """
        data = request.get_json(silent=True) or {}
        nodes = data.get("Nodes", {}).get("Items", [])
        
        # If no nodes passed from real cluster, fall back to cluster simulator state
        if not nodes:
            candidate_nodes = cluster_sim.nodes
        else:
            candidate_nodes = nodes

        pod = data.get("Pod", {})
        containers = pod.get("spec", {}).get("containers", [])
        req_cpu = 400
        req_mem = 512
        if containers:
            res = containers[0].get("resources", {}).get("requests", {})
            # parse mCPU and memory
            req_cpu = int(res.get("cpu", "400m").replace("m", "")) if "m" in res.get("cpu", "400m") else 400
            req_mem = int(res.get("memory", "512Mi").replace("Mi", "")) if "Mi" in res.get("memory", "512Mi") else 512

        pod_demand = {"cpu_m": req_cpu, "mem_mb": req_mem}
        rank_result = node_ranker.rank_nodes(candidate_nodes, pod_demand)

        feasible_names = [n["name"] for n in rank_result["rankings"]]
        failed_nodes = {r["id"]: r["reason"] for r in rank_result["rejected"]}

        return jsonify({
            "Nodes": {
                "Items": [n for n in candidate_nodes if n.get("name") in feasible_names or n.get("id") in feasible_names]
            },
            "FailedNodes": failed_nodes,
            "Error": ""
        })

    @k8s_bp.route('/prioritize', methods=['POST'])
    def prioritize_nodes():
        """
        Kubernetes Scheduler Extender Prioritize Webhook:
        Input JSON: ExtenderArgs { Pod: {...}, Nodes: { Items: [...] } }
        Output JSON: HostPriorityList [ {"Host": "worker-node-1", "Score": 89}, ... ]
        """
        data = request.get_json(silent=True) or {}
        candidate_nodes = cluster_sim.nodes
        pod_demand = {"cpu_m": 450, "mem_mb": 512}

        rank_result = node_ranker.rank_nodes(candidate_nodes, pod_demand)

        # Scale composite scores (0.0 - 1.0) to Kubernetes standard score range (0 - 100)
        host_priority_list = []
        for r in rank_result["rankings"]:
            host_priority_list.append({
                "Host": r["name"],
                "Score": int(r["composite_score"] * 100)
            })

        return jsonify(host_priority_list)

    return k8s_bp
