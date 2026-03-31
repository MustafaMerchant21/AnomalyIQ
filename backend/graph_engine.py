"""
graph_engine.py — AnomalyIQ NetworkX Graph Intelligence Engine

Builds an entity graph from test samples and computes graph-based fraud signals:
- PageRank hub detection
- Fraud ring clustering
- Shared device fingerprint analysis
- Connected anomaly count

Returns: graph_score (0–1), signals[], pagerank_score, connected_anomalies,
         total_nodes, total_edges, hub_count
"""

import math
from typing import Any


def build_entity_graph(samples: list[dict]) -> Any:
    """
    Build a bipartite entity graph from test samples.

    Nodes:
      - transactions: "tx_{id}"
      - entity values: "ent_{feature}_{bin}" (quantized values for matching)

    Edges:
      - transaction ↔ entity (transaction shares entity value with others)
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    G = nx.Graph()

    for i, sample in enumerate(samples):
        tx_node = f"tx_{i}"
        label = sample.get("_true_label", 0)
        score = sample.get("anomaly_score", 0.0)

        # Create entity nodes for shared features (binned to enable matching)
        raw = sample.get("_raw_values", {})
        raw_categ = sample.get("_raw_categorical_values", {})
        
        G.add_node(tx_node, type="transaction", label=label, score=score, index=i, raw=raw, raw_categ=raw_categ)
        
        # Store basic account info directly on the node for easy blacklist/UI parsing
        cc_num = raw.get("cc_num", raw_categ.get("cc_num", "Unknown Account"))
        first = raw_categ.get("first", raw.get("first", ""))
        last = raw_categ.get("last", raw.get("last", ""))
        name = f"{first} {last}".strip() or "Unknown User"
        nx.set_node_attributes(G, {tx_node: {"cc_num": cc_num, "name": name}})

        # 1. Numeric features (binned)
        features = {k: v for k, v in raw.items() if k not in ("_true_label",)}
        for feat, val in features.items():
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            bucket = _get_entity_key(feat, val_f)
            ent_node = f"ent_{feat}_{bucket}"
            if not G.has_node(ent_node):
                G.add_node(ent_node, type="entity", feature=feat, bucket=bucket)
            G.add_edge(tx_node, ent_node, feature=feat, value=val_f)

        # 2. Categorical features (exact string match)
        for feat, val in raw_categ.items():
            if val is None or str(val).lower() in ("nan", "none", "null", ""):
                continue
            bucket = str(val).strip()
            ent_node = f"ent_{feat}_{bucket}"
            if not G.has_node(ent_node):
                G.add_node(ent_node, type="entity", feature=feat, bucket=bucket)
            G.add_edge(tx_node, ent_node, feature=feat, value=val)

    return G


def _get_entity_key(feature_name: str, value: Any) -> str:
    """Quantize/format a value into discrete buckets to detect shared entities."""
    if isinstance(value, str):
        return value.strip()

    try:
        val_f = float(value)
    except (TypeError, ValueError):
        return str(value)

    fname = feature_name.lower()

    # Binary features: exact value
    if any(k in fname for k in ["foreign", "device", "high_risk", "flag", "binary"]):
        return str(int(round(val_f)))

    # Hour/time: exact integer
    if any(k in fname for k in ["hour", "time", "hr"]):
        return str(int(val_f) % 24)

    # Velocity: integer bins
    if any(k in fname for k in ["velocity", "burst", "count", "rate"]):
        return str(int(val_f))

    # Amount/distance: logarithmic bins
    if val_f <= 0:
        return "0"
    log_bin = int(math.log10(max(val_f, 1)))
    return f"log{log_bin}"


def compute_graph_score(G: Any, fraud_label: int = 1) -> dict:
    """
    Compute graph-based fraud signals and a composite graph score.

    Returns dict with graph_score, signals, pagerank_score, connected_anomalies,
    total_nodes, total_edges, hub_count.
    """
    try:
        import networkx as nx
    except ImportError:
        return _empty_graph_score()

    if G is None or G.number_of_nodes() == 0:
        return _empty_graph_score()

    tx_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "transaction"]
    ent_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "entity"]

    if not tx_nodes:
        return _empty_graph_score()

    signals = []
    score_components = []

    # ── 1. PageRank hub detection ────────────────────────────────────────────
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
        tx_pr = {n: pr[n] for n in tx_nodes if n in pr}

        if tx_pr:
            max_pr = max(tx_pr.values())
            avg_pr = sum(tx_pr.values()) / len(tx_pr)

            # Transactions with PageRank > 3x average are hubs
            hub_threshold = avg_pr * 3
            hub_nodes = [n for n, v in tx_pr.items() if v > hub_threshold]
            hub_count = len(hub_nodes)

            if hub_count > 0:
                signals.append(
                    f"{hub_count} transaction(s) are high-connectivity hubs suggesting coordinated fraud ring activity"
                )
                score_components.append(min(0.4, hub_count / max(len(tx_nodes), 1) * 10))

            # Check if hub nodes are fraud
            fraud_hubs = sum(
                1 for n in hub_nodes
                if G.nodes[n].get("label", 0) == fraud_label
            )
            if fraud_hubs > 0:
                signals.append(
                    f"{fraud_hubs} hub node(s) are labeled as fraud — indicative of fraud ring leadership"
                )
                score_components.append(0.3 * fraud_hubs / max(hub_count, 1))

            pagerank_score = min(1.0, max_pr / max(avg_pr, 1e-10) / 10)
        else:
            hub_count = 0
            pagerank_score = 0.0

    except Exception:
        hub_count = 0
        pagerank_score = 0.0
        tx_pr = {}

    # ── 2. Fraud ring clustering ─────────────────────────────────────────────
    try:
        # Find connected components in transaction subgraph (via shared entities)
        tx_subgraph = nx.Graph()
        tx_subgraph.add_nodes_from(tx_nodes)

        for ent in ent_nodes:
            neighbors = [n for n in G.neighbors(ent) if n in set(tx_nodes)]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    tx_subgraph.add_edge(neighbors[i], neighbors[j])

        components = list(nx.connected_components(tx_subgraph))
        large_components = [c for c in components if len(c) >= 3]

        if large_components:
            # Check for fraud clusters
            fraud_clusters = 0
            for comp in large_components:
                fraud_in_comp = sum(
                    1 for n in comp
                    if G.nodes[n].get("label", 0) == fraud_label
                )
                if fraud_in_comp / len(comp) > 0.5:
                    fraud_clusters += 1

            if fraud_clusters > 0:
                signals.append(
                    f"{fraud_clusters} fraud cluster(s) detected with ≥3 interconnected suspicious transactions"
                )
                score_components.append(min(0.35, fraud_clusters * 0.15))

    except Exception:
        pass

    # ── 3. Shared entity fingerprints ───────────────────────────────────────
    try:
        high_degree_entities = [
            (n, G.degree(n)) for n in ent_nodes
            if G.degree(n) >= 5
        ]

        if high_degree_entities:
            shared_count = len(high_degree_entities)
            signals.append(
                f"{shared_count} shared entity fingerprint(s) link multiple transactions — possible account takeover"
            )
            score_components.append(min(0.25, shared_count * 0.05))

    except Exception:
        pass

    # ── 4. Connected anomaly count ───────────────────────────────────────────
    fraud_tx_nodes = [
        n for n in tx_nodes
        if G.nodes[n].get("label", 0) == fraud_label
    ]
    connected_anomalies = len(fraud_tx_nodes)

    if connected_anomalies > 0:
        ratio = connected_anomalies / max(len(tx_nodes), 1)
        if ratio > 0.15:
            signals.append(
                f"Elevated fraud concentration: {connected_anomalies} of {len(tx_nodes)} "
                f"transactions ({ratio*100:.1f}%) are anomalous"
            )
            score_components.append(min(0.3, ratio * 2))

    # ── 5. Isolation signal ──────────────────────────────────────────────────
    isolated_fraud = [
        n for n in fraud_tx_nodes
        if G.degree(n) <= 1
    ]
    if isolated_fraud and len(isolated_fraud) > connected_anomalies * 0.4:
        signals.append(
            f"{len(isolated_fraud)} anomalous transaction(s) are isolated with no shared entities — "
            f"suggesting advanced evasion techniques"
        )
        score_components.append(0.1)

    # ── Composite graph score ────────────────────────────────────────────────
    graph_score = min(1.0, sum(score_components)) if score_components else 0.0

    # Cap signals at 5
    signals = signals[:5]

    return {
        "graph_score": round(graph_score, 4),
        "signals": signals,
        "pagerank_score": round(pagerank_score, 4),
        "connected_anomalies": connected_anomalies,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "hub_count": hub_count
    }


def _empty_graph_score() -> dict:
    """Return default zero-value graph score dict."""
    return {
        "graph_score": 0.0,
        "signals": ["Insufficient data for graph analysis"],
        "pagerank_score": 0.0,
        "connected_anomalies": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "hub_count": 0,
        "fraud_rings": []
    }


def run_graph_analysis(samples: list[dict]) -> dict:
    """
    Full graph analysis pipeline:
    1. Build entity graph from samples
    2. Compute graph score and signals

    Args:
        samples: list of scored transaction dicts with _raw_values and _true_label
    Returns:
        Full graph score result dict
    """
    if not samples:
        return _empty_graph_score()

    try:
        G = build_entity_graph(samples)
        result = compute_graph_score(G)
        result["fraud_rings"] = extract_fraud_rings(G)
        return result
    except Exception as e:
        result = _empty_graph_score()
        result["signals"] = [f"Graph analysis encountered an error — defaulting to baseline"]
        return result


def extract_fraud_rings(G: Any, fraud_label: int = 1) -> list[dict]:
    """
    Extract highly connected communities (fraud rings) using Louvain Community Detection.
    Also detects Hub Accounts using PageRank.
    """
    if G is None or G.number_of_nodes() == 0:
        return []

    try:
        import networkx as nx
        from networkx.algorithms import community
    except ImportError:
        return []

    tx_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "transaction"]
    ent_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "entity"]

    if not tx_nodes:
        return []

    # 1. Compute PageRank for Hub Detection
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
        tx_pr = {n: pr[n] for n in tx_nodes if n in pr}
        if tx_pr:
            avg_pr = sum(tx_pr.values()) / len(tx_pr)
            hub_threshold = avg_pr * 2  # Slightly lower threshold for visual clusters
            nx.set_node_attributes(G, {n: {"is_hub": bool(pr[n] > hub_threshold)} for n in tx_nodes})
        else:
             nx.set_node_attributes(G, {n: {"is_hub": False} for n in tx_nodes})
    except Exception:
        nx.set_node_attributes(G, {n: {"is_hub": False} for n in tx_nodes})

    # 2. Extract Communities
    try:
        # Louvain Community Detection
        communities = community.louvain_communities(G)
    except Exception:
        # Fallback to connected components
        communities = list(nx.connected_components(G))

    rings = []
    
    for idx, comp in enumerate(communities):
        comp_tx = [n for n in comp if n in set(tx_nodes)]
        comp_ent = [n for n in comp if n in set(ent_nodes)]
        
        # Only care about multi-transaction communities
        if len(comp_tx) < 2:
            continue
            
        fraud_tx = [n for n in comp_tx if G.nodes[n].get("label", 0) == fraud_label]
        
        # Must have at least one fraud to be a 'fraud ring'
        if not fraud_tx:
            continue

        # Build node payloads
        nodes_data = []
        for n in comp_tx:
            node_data = G.nodes[n]
            
            # Combine raw features for frontend display
            raw_feats = node_data.get("raw", {})
            cat_feats = node_data.get("raw_categ", {})
            combined_features = {
                **{k: v for k, v in raw_feats.items() if not k.startswith("_")},
                **{k: v for k, v in cat_feats.items() if not k.startswith("_")}
            }
            
            nodes_data.append({
                "id": n,
                "label": node_data.get("label", 0),
                "score": round(node_data.get("score", 0.0), 3),
                "is_hub": node_data.get("is_hub", False),
                "cc_num": str(node_data.get("cc_num", "")),
                "name": str(node_data.get("name", "")),
                "index": node_data.get("index", -1),
                "features": combined_features
            })
            
        # Build shared entities payloads
        shared_entities = []
        for n in comp_ent:
            # Only include entities with degree >= 2 in this cluster, to avoid noise
            if G.degree(n) >= 2:
                node_data = G.nodes[n]
                shared_entities.append({
                    "id": n,
                    "feature": str(node_data.get("feature", "")),
                    "bucket": str(node_data.get("bucket", "")),
                    "degree": G.degree(n)
                })
            
        rings.append({
            "cluster_id": f"cluster-{idx+1:03d}",
            "transaction_count": len(comp_tx),
            "fraud_count": len(fraud_tx),
            "hub_count": sum(1 for n in nodes_data if n["is_hub"]),
            "transactions": sorted(nodes_data, key=lambda x: (-x["is_hub"], -x["score"])),
            "shared_entities": sorted(shared_entities, key=lambda x: -x["degree"]),
            "summary": "Pending AI analysis..."  # To be filled by the explainer
        })
        
    # Sort rings by severity: highest fraud count, then transaction count
    rings.sort(key=lambda x: (-x["fraud_count"], -x["transaction_count"]))
    return rings
