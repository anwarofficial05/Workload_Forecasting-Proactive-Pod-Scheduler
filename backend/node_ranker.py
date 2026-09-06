"""
Multi-Objective Node Ranking Engine
Implements:
1. Multi-criteria worker node evaluation:
   - Predicted CPU Headroom Score
   - Predicted Memory Headroom Score
   - Network & Ingress Latency Impact Score
   - Cluster-Wide Load Balance / Variance Penalty
2. Dynamic weighting profiles (Balanced, Latency Critical, High Packing, Spread)
3. Feasibility Filtering (Kubernetes Predicates) + Scoring (Kubernetes Priorities)
"""

import numpy as np

class MultiObjectiveNodeRanker:
    def __init__(self, weights=None):
        # Default balanced weights as formulated in project report
        self.default_weights = {
            "w_cpu": 0.35,
            "w_mem": 0.30,
            "w_lat": 0.20,
            "w_bal": 0.15
        }
        self.weights = weights or self.default_weights

    def update_weights(self, w_cpu, w_mem, w_lat, w_bal):
        """Allows dynamic adjustment of objective weights via UI sliders."""
        total = w_cpu + w_mem + w_lat + w_bal
        if total <= 0:
            total = 1.0
        self.weights = {
            "w_cpu": round(w_cpu / total, 3),
            "w_mem": round(w_mem / total, 3),
            "w_lat": round(w_lat / total, 3),
            "w_bal": round(w_bal / total, 3)
        }

    def rank_nodes(self, nodes_state, pod_demand, custom_weights=None):
        """
        Ranks candidate nodes for placing an incoming pod with predicted resource demand.
        nodes_state: list of dicts, e.g.:
        [
            {
                "id": "node-1", "name": "worker-node-1",
                "cpu_capacity_m": 4000, "cpu_used_m": 2400,
                "mem_capacity_mb": 8192, "mem_used_mb": 4500,
                "latency_ms": 12.4, "zone": "us-east-1a",
                "pods_count": 6, "status": "Ready"
            }, ...
        ]
        pod_demand: dict, e.g. {"cpu_m": 500, "mem_mb": 512}
        """
        active_weights = custom_weights or self.weights
        w_cpu = active_weights.get("w_cpu", 0.35)
        w_mem = active_weights.get("w_mem", 0.30)
        w_lat = active_weights.get("w_lat", 0.20)
        w_bal = active_weights.get("w_bal", 0.15)

        demand_cpu = float(pod_demand.get("cpu_m", 400))
        demand_mem = float(pod_demand.get("mem_mb", 512))

        # 1. Filter Phase (K8s Predicates: check if node can accommodate pod)
        feasible_nodes = []
        rejected_nodes = []

        for node in nodes_state:
            if node.get("status") != "Ready":
                rejected_nodes.append({"id": node["id"], "reason": "Node NotReady"})
                continue

            future_cpu = node["cpu_used_m"] + demand_cpu
            future_mem = node["mem_used_mb"] + demand_mem

            if future_cpu > node["cpu_capacity_m"]:
                rejected_nodes.append({"id": node["id"], "reason": f"Insufficient CPU ({future_cpu}/{node['cpu_capacity_m']}m)"})
                continue
            if future_mem > node["mem_capacity_mb"]:
                rejected_nodes.append({"id": node["id"], "reason": f"Insufficient Memory ({future_mem}/{node['mem_capacity_mb']}MB)"})
                continue

            feasible_nodes.append(node)

        if not feasible_nodes:
            return {
                "top_node": None,
                "rankings": [],
                "rejected": rejected_nodes,
                "status": "All nodes filtered out (Resource Exhaustion)"
            }

        # 2. Score Phase (K8s Priorities: Multi-Objective Ranking)
        max_lat = max([n["latency_ms"] for n in feasible_nodes]) + 5.0
        min_lat = max(0.1, min([n["latency_ms"] for n in feasible_nodes]))

        # Calculate current cluster mean utilization for balance baseline
        curr_cpu_pcts = [n["cpu_used_m"] / n["cpu_capacity_m"] for n in feasible_nodes]

        rankings = []
        for node in feasible_nodes:
            # Future utilization ratios
            future_cpu_ratio = (node["cpu_used_m"] + demand_cpu) / node["cpu_capacity_m"]
            future_mem_ratio = (node["mem_used_mb"] + demand_mem) / node["mem_capacity_mb"]

            # Headroom scores (higher is better)
            score_cpu_headroom = max(0.0, 1.0 - future_cpu_ratio)
            score_mem_headroom = max(0.0, 1.0 - future_mem_ratio)

            # Latency score (lower latency = higher score)
            norm_latency = (node["latency_ms"] - min_lat) / (max_lat - min_lat + 1e-6)
            score_latency = max(0.0, 1.0 - norm_latency)

            # Cluster load balance score:
            # Compute simulated variance across all nodes if this node is chosen
            hypothetical_cpu_pcts = []
            for other in feasible_nodes:
                if other["id"] == node["id"]:
                    hypothetical_cpu_pcts.append(future_cpu_ratio)
                else:
                    hypothetical_cpu_pcts.append(other["cpu_used_m"] / other["cpu_capacity_m"])
            
            hypothetical_var = np.var(hypothetical_cpu_pcts)
            # Normalize balance score: lower variance is better balance
            score_balance = max(0.0, 1.0 - np.sqrt(hypothetical_var) * 2.0)

            # Weighted Composite Multi-Objective Score
            composite_score = (
                w_cpu * score_cpu_headroom +
                w_mem * score_mem_headroom +
                w_lat * score_latency +
                w_bal * score_balance
            )

            rankings.append({
                "id": node["id"],
                "name": node["name"],
                "zone": node.get("zone", "zone-default"),
                "composite_score": round(float(composite_score), 4),
                "score_breakdown": {
                    "cpu_headroom": round(float(score_cpu_headroom), 4),
                    "mem_headroom": round(float(score_mem_headroom), 4),
                    "latency": round(float(score_latency), 4),
                    "load_balance": round(float(score_balance), 4)
                },
                "future_cpu_pct": round(float(future_cpu_ratio * 100), 1),
                "future_mem_pct": round(float(future_mem_ratio * 100), 1),
                "latency_ms": round(float(node["latency_ms"]), 1),
                "current_pods": node.get("pods_count", 0)
            })

        # Sort descending by composite score
        rankings.sort(key=lambda x: x["composite_score"], reverse=True)

        for idx, item in enumerate(rankings):
            item["rank"] = idx + 1

        top_node = rankings[0]
        rationale = (
            f"Node '{top_node['name']}' selected as Rank #1 with composite score {top_node['composite_score']:.3f}. "
            f"Headrooms: CPU {top_node['score_breakdown']['cpu_headroom']*100:.1f}%, "
            f"RAM {top_node['score_breakdown']['mem_headroom']*100:.1f}%. "
            f"Latency: {top_node['latency_ms']}ms."
        )

        return {
            "top_node": top_node,
            "rankings": rankings,
            "rejected": rejected_nodes,
            "rationale": rationale,
            "active_weights": {
                "w_cpu": w_cpu,
                "w_mem": w_mem,
                "w_lat": w_lat,
                "w_bal": w_bal
            }
        }
