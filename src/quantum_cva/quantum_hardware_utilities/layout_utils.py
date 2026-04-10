import math
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism
import numpy as np


def build_undirected_coupling_graph(coupling_map):
    """
    Convierte el coupling map en un grafo no dirigido.
    """
    G = nx.Graph()
    G.add_edges_from(tuple(edge) for edge in coupling_map)
    return G


def _extract_error_from_gate_entry(gate_entry):
    params = gate_entry.get("parameters", [])
    for p in params:
        name = str(p.get("name", "")).lower()
        if "error" in name:
            return float(p["value"])
    if params:
        return float(params[0]["value"])
    return np.nan


def _safe_log_pos(x, floor=1e-16):
    return math.log(max(float(x), floor))


def _qubit_param_map(qubit_entry):
    return {str(p["name"]): float(p["value"]) for p in qubit_entry}


def build_backend_quality_maps(
    backend,
    *,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
):
    """
    Construye automáticamente:
      - preferred_scores: score por qubit
      - avoided_qubits: qubits a evitar
      - edge_scores: score por arista física
      - diagnostics: métricas crudas para inspección

    El score favorece:
      - T1 alto
      - T2 alto
      - readout error bajo
      - error local medio de 2 qubits bajo
    """
    props = backend.properties().to_dict()
    coupling_map = backend.configuration().coupling_map
    G = build_undirected_coupling_graph(coupling_map)

    qubit_metrics = {}
    for q, entry in enumerate(props["qubits"]):
        m = _qubit_param_map(entry)
        ro = m.get("readout_error", np.nan)

        if not np.isfinite(ro):
            p01 = m.get("prob_meas0_prep1", np.nan)
            p10 = m.get("prob_meas1_prep0", np.nan)
            if np.isfinite(p01) and np.isfinite(p10):
                ro = 0.5 * (p01 + p10)

        qubit_metrics[q] = {
            "T1": m.get("T1", np.nan),
            "T2": m.get("T2", np.nan),
            "readout_error": ro,
        }

    edge_error_map = {}
    for gate in props["gates"]:
        qubits = gate.get("qubits", [])
        if len(qubits) != 2:
            continue
        key = tuple(sorted((int(qubits[0]), int(qubits[1]))))
        err = _extract_error_from_gate_entry(gate)
        if key not in edge_error_map or (
            np.isfinite(err) and err < edge_error_map[key]
        ):
            edge_error_map[key] = err

    local_mean_2q = {}
    for q in G.nodes:
        errs = []
        for nb in G.neighbors(q):
            e = edge_error_map.get(tuple(sorted((q, nb))), np.nan)
            if np.isfinite(e):
                errs.append(e)
        local_mean_2q[q] = float(np.mean(errs)) if errs else np.nan

    preferred_scores = {}
    for q in G.nodes:
        t1 = qubit_metrics[q]["T1"]
        t2 = qubit_metrics[q]["T2"]
        ro = qubit_metrics[q]["readout_error"]
        e2 = local_mean_2q[q]

        score = 0.0
        if np.isfinite(t1):
            score += 0.35 * _safe_log_pos(t1)
        if np.isfinite(t2):
            score += 0.35 * _safe_log_pos(t2)
        if np.isfinite(ro):
            score += 0.20 * (-_safe_log_pos(ro))
        if np.isfinite(e2):
            score += 0.10 * (-_safe_log_pos(e2))

        preferred_scores[q] = float(score)

    edge_scores = {}
    for edge, err in edge_error_map.items():
        if np.isfinite(err):
            edge_scores[edge] = float(-_safe_log_pos(err))
        else:
            edge_scores[edge] = -1e6

    ro_vals = np.array(
        [
            qubit_metrics[q]["readout_error"]
            for q in G.nodes
            if np.isfinite(qubit_metrics[q]["readout_error"])
        ],
        dtype=float,
    )
    e2_vals = np.array(
        [local_mean_2q[q] for q in G.nodes if np.isfinite(local_mean_2q[q])],
        dtype=float,
    )

    ro_thr = np.quantile(ro_vals, readout_quantile) if ro_vals.size else np.inf
    e2_thr = (
        np.quantile(e2_vals, local_2q_quantile) if e2_vals.size else np.inf
    )

    avoided_qubits = {
        q
        for q in G.nodes
        if (
            (
                np.isfinite(qubit_metrics[q]["readout_error"])
                and qubit_metrics[q]["readout_error"] >= ro_thr
            )
            or (
                np.isfinite(local_mean_2q[q])
                and local_mean_2q[q] >= e2_thr
            )
        )
    }

    diagnostics = {
        "qubit_metrics": qubit_metrics,
        "local_mean_2q": local_mean_2q,
        "edge_error_map": edge_error_map,
        "readout_threshold": float(ro_thr),
        "local_2q_threshold": float(e2_thr),
    }
    return preferred_scores, avoided_qubits, edge_scores, diagnostics


def _path_score(
    path,
    preferred_scores,
    edge_scores=None,
    *,
    default_node_score=0.15,
    default_edge_score=0.0,
    close_cycle=False,
    node_weight=1.0,
    edge_weight=0.35,
):
    node_term = sum(preferred_scores.get(q, default_node_score) for q in path)

    edges = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
    if close_cycle and len(path) >= 3:
        edges.append(tuple(sorted((path[-1], path[0]))))

    edge_term = 0.0
    if edge_scores is not None:
        edge_term = sum(edge_scores.get(e, default_edge_score) for e in edges)

    return float(node_weight * node_term + edge_weight * edge_term)


def find_best_chain(
    G,
    preferred_scores,
    avoided_qubits,
    length,
    default_score=0.15,
    max_starts=40,
    edge_scores=None,
):
    avoided_qubits = set(avoided_qubits)

    def node_score(q):
        if q in avoided_qubits:
            return -10.0
        return preferred_scores.get(q, default_score)

    start_nodes = sorted(G.nodes, key=node_score, reverse=True)[:max_starts]

    best_path = None
    best_score = -np.inf

    for start in start_nodes:
        if start in avoided_qubits:
            continue

        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()

            if len(path) == length:
                score = _path_score(
                    path,
                    preferred_scores,
                    edge_scores=edge_scores,
                    default_node_score=default_score,
                    close_cycle=False,
                )
                if score > best_score:
                    best_score = score
                    best_path = path.copy()
                continue

            neighbors = sorted(
                [
                    nb
                    for nb in G.neighbors(current)
                    if nb not in path and nb not in avoided_qubits
                ],
                key=node_score,
                reverse=True,
            )

            for nb in neighbors:
                stack.append((nb, path + [nb]))

    if best_path is None:
        raise RuntimeError("No se ha encontrado una cadena conectada con las restricciones dadas.")

    return best_path, float(best_score)


def find_best_cycle(
    G,
    preferred_scores,
    avoided_qubits,
    length,
    default_score=0.15,
    max_starts=40,
    edge_scores=None,
):
    avoided_qubits = set(avoided_qubits)

    def node_score(q):
        if q in avoided_qubits:
            return -10.0
        return preferred_scores.get(q, default_score)

    start_nodes = sorted(G.nodes, key=node_score, reverse=True)[:max_starts]

    best_cycle = None
    best_score = -np.inf

    for start in start_nodes:
        if start in avoided_qubits:
            continue

        stack = [(start, [start])]
        while stack:
            current, path = stack.pop()

            if len(path) == length:
                if G.has_edge(path[-1], path[0]):
                    score = _path_score(
                        path,
                        preferred_scores,
                        edge_scores=edge_scores,
                        default_node_score=default_score,
                        close_cycle=True,
                    )
                    if score > best_score:
                        best_score = score
                        best_cycle = path.copy()
                continue

            neighbors = sorted(
                [
                    nb
                    for nb in G.neighbors(current)
                    if nb not in path and nb not in avoided_qubits
                ],
                key=node_score,
                reverse=True,
            )

            for nb in neighbors:
                stack.append((nb, path + [nb]))

    if best_cycle is None:
        raise RuntimeError("No se ha encontrado un ciclo conectado con las restricciones dadas.")

    return best_cycle, float(best_score)


def find_best_crca2(G, preferred_scores, avoided_qubits, default_score=0.15, edge_scores=None):
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()
    T.add_edges_from([(0, 2), (1, 2)])
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[0], inv_map[1], inv_map[2]]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'crca2'.")
    return best_layout, float(best_score)


def find_best_tree_bus(G, preferred_scores, avoided_qubits, default_score=0.15, edge_scores=None):
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()
    T.add_edges_from([(0, 1), (1, 2), (2, 3), (5, 4), (4, 0), (7, 6), (6, 1), (9, 8), (8, 3)])
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)
    best_layout, best_score = None, -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)
        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(10)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'tree_bus'.")
    return best_layout, best_score


def find_best_snowflake(G, preferred_scores, avoided_qubits, default_score=0.15, edge_scores=None):
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()
    T.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (4, 5), (2, 6), (6, 7), (3, 8), (8, 9)])
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)
    best_layout, best_score = None, -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)
        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(10)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'snowflake'.")
    return best_layout, float(best_score)


def find_best_qcbm_heavyhex8(G, preferred_scores, avoided_qubits, default_score=0.15, edge_scores=None):
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()
    T.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (4, 7)])
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)
    best_layout, best_score = None, -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)
        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(8)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'qcbm_heavyhex8'.")
    return best_layout, float(best_score)


def find_best_time_tree4(G, preferred_scores, avoided_qubits, default_score=0.15, edge_scores=None):
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()
    T.add_edges_from([(0, 4), (1, 4), (4, 6), (6, 5), (5, 2), (5, 3)])
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)
    best_layout, best_score = None, -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)
        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(7)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'time_tree4'.")
    return best_layout, float(best_score)

def find_best_qcbm_heavyhex6(G, preferred_scores, avoided_qubits, default_score=0.15, edge_scores=None):
    """
    Busca una topología óptima para 6 qubits en Heavy Hex.
    Estructura ramificada: un nodo central (0) conectado a 3 ramas (1, 2, 3),
    y dos de esas ramas se extienden un nivel más (a 4 y 5).
    """
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()
    # Grafo estrella ramificado ideal para minimizar distancias en 6 qubits
    T.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 5)])
    
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)
    best_layout, best_score = None, -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)
        if score > best_score:
            best_score = score
            # Guardamos el layout manteniendo el orden de los nodos lógicos 0 a 5
            best_layout = [inv_map[i] for i in range(6)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'qcbm_heavyhex6'.")
    return best_layout, float(best_score)


def find_best_heavy_hex_star(
    G, preferred_scores, avoided_qubits, n_price, default_score=0.15, edge_scores=None
):
    avoided_qubits = set(avoided_qubits)
    T = nx.Graph()

    a = n_price + 2
    T.add_edges_from([
        (a, 0),
        (a, 1),
        (a, 2)
    ])
    for i in range(2, n_price + 1):
        T.add_edge(i, i + 1)

    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)
    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout, best_score = None, -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)
        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(n_price + 3)]

    if best_layout is None:
        raise RuntimeError("No se encontró la topología 'heavy_hex_star'.")

    return best_layout, float(best_score)


def select_best_layout(
    backend,
    *,
    topology,
    length,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
    default_score=0.15,
    max_starts=40,
    relax_if_needed=True,
):
    valid_topologies = {
        "linear",
        "circular",
        "tree_bus",
        "snowflake",
        "qcbm_heavyhex6",
        "qcbm_heavyhex8",
        "time_tree4",
        "crca2",
        "heavy_hex_star",
    }

    if topology not in valid_topologies:
        raise ValueError(f"topology must be one of {valid_topologies}. Got: '{topology}'")

    coupling_map = backend.configuration().coupling_map
    G = build_undirected_coupling_graph(coupling_map)

    preferred_scores, avoided_qubits, edge_scores, diagnostics = (
        build_backend_quality_maps(
            backend,
            readout_quantile=readout_quantile,
            local_2q_quantile=local_2q_quantile,
        )
    )

    tried = []

    def _try_cycle(current_avoided):
        return find_best_cycle(G, preferred_scores, current_avoided, length, default_score=default_score, max_starts=max_starts, edge_scores=edge_scores)

    def _try_chain(current_avoided):
        return find_best_chain(G, preferred_scores, current_avoided, length, default_score=default_score, max_starts=max_starts, edge_scores=edge_scores)
    
    def _try_crca2(current_avoided):
        return find_best_crca2(G, preferred_scores, current_avoided, default_score=default_score, edge_scores=edge_scores)

    def _try_tree_bus(current_avoided):
        return find_best_tree_bus(G, preferred_scores, current_avoided, default_score=default_score, edge_scores=edge_scores)

    def _try_snowflake(current_avoided):
        return find_best_snowflake(G, preferred_scores, current_avoided, default_score=default_score, edge_scores=edge_scores)

    def _try_qcbm_heavyhex8(current_avoided):
        return find_best_qcbm_heavyhex8(G, preferred_scores, current_avoided, default_score=default_score, edge_scores=edge_scores)

    def _try_qcbm_heavyhex6(current_avoided):
        return find_best_qcbm_heavyhex6(G, preferred_scores, current_avoided, default_score=default_score, edge_scores=edge_scores)

    def _try_heavy_hex_star(current_avoided):
        n_price = length - 3
        if n_price < 1:
            raise ValueError("length debe ser al menos 4 para heavy_hex_star (2 time + >=1 price + 1 ancilla)")
        return find_best_heavy_hex_star(G, preferred_scores, current_avoided, n_price, default_score=default_score, edge_scores=edge_scores)

    if topology == "linear":
        layout, score = _try_chain(avoided_qubits)
        metadata = {
            "graph": G, "preferred_scores": preferred_scores,
            "avoided_qubits": avoided_qubits, "edge_scores": edge_scores,
            "diagnostics": diagnostics, "selected_topology": "linear",
            "fallback_used": False, "tried": tried,
        }
        return layout, score, metadata
    
    if topology == "heavy_hex_star":
        try:
            layout, score = _try_heavy_hex_star(avoided_qubits)
            metadata = {
                "graph": G, "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits, "edge_scores": edge_scores,
                "diagnostics": diagnostics, "selected_topology": "heavy_hex_star",
                "fallback_used": False, "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict heavy_hex_star failed")

        if relax_if_needed:
            avoided_sorted = sorted(avoided_qubits, key=lambda q: preferred_scores.get(q, -1e9))
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_heavy_hex_star(relaxed_avoided)
                metadata = {
                    "graph": G, "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided, "edge_scores": edge_scores,
                    "diagnostics": diagnostics, "selected_topology": "heavy_hex_star",
                    "fallback_used": True, "tried": tried + ["relaxed heavy_hex_star succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed heavy_hex_star failed")
                
    for t_name, t_func in [
        ("crca2", _try_crca2), ("tree_bus", _try_tree_bus), ("snowflake", _try_snowflake),
        ("qcbm_heavyhex8", _try_qcbm_heavyhex8), ("qcbm_heavyhex6", _try_qcbm_heavyhex6)
    ]:
        if topology == t_name:
            try:
                layout, score = t_func(avoided_qubits)
                metadata = {
                    "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": avoided_qubits,
                    "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": t_name,
                    "fallback_used": False, "tried": tried,
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append(f"strict {t_name} failed")

            if relax_if_needed:
                avoided_sorted = sorted(avoided_qubits, key=lambda q: preferred_scores.get(q, -1e9))
                relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
                try:
                    layout, score = t_func(relaxed_avoided)
                    metadata = {
                        "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": relaxed_avoided,
                        "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": t_name,
                        "fallback_used": True, "tried": tried + [f"relaxed {t_name} succeeded"],
                    }
                    return layout, score, metadata
                except RuntimeError:
                    tried.append(f"relaxed {t_name} failed")

    if topology == "time_tree4":
        try:
            layout, score = find_best_time_tree4(G, preferred_scores, avoided_qubits, default_score=default_score, edge_scores=edge_scores)
            metadata = {
                "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": "time_tree4",
                "fallback_used": False, "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict time_tree4 failed")

        if relax_if_needed:
            avoided_sorted = sorted(avoided_qubits, key=lambda q: preferred_scores.get(q, -1e9))
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = find_best_time_tree4(G, preferred_scores, relaxed_avoided, default_score=default_score, edge_scores=edge_scores)
                metadata = {
                    "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": "time_tree4",
                    "fallback_used": True, "tried": tried + ["relaxed time_tree4 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed time_tree4 failed")

    if topology == "circular":
        try:
            layout, score = _try_cycle(avoided_qubits)
            metadata = {
                "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": "circular",
                "fallback_used": False, "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict circular failed")

        if relax_if_needed:
            avoided_sorted = sorted(avoided_qubits, key=lambda q: preferred_scores.get(q, -1e9))
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_cycle(relaxed_avoided)
                metadata = {
                    "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": "circular",
                    "fallback_used": True, "tried": tried + ["relaxed circular succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed circular failed")

            layout, score = _try_chain(relaxed_avoided)
            metadata = {
                "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": relaxed_avoided,
                "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": "linear",
                "fallback_used": True, "tried": tried + ["fallback to linear"],
            }
            return layout, score, metadata

    if relax_if_needed and topology not in ["linear"]:
        avoided_sorted = sorted(avoided_qubits, key=lambda q: preferred_scores.get(q, -1e9))
        relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
        try:
            layout, score = _try_chain(relaxed_avoided)
            metadata = {
                "graph": G, "preferred_scores": preferred_scores, "avoided_qubits": relaxed_avoided,
                "edge_scores": edge_scores, "diagnostics": diagnostics, "selected_topology": "linear",
                "fallback_used": True, "tried": tried + ["fallback to linear"],
            }
            return layout, score, metadata
        except:
            pass

    raise RuntimeError("No se ha encontrado ninguna topología válida con las restricciones dadas.")


def draw_local_subgraph(G, layout, topology="linear", figsize=(8, 5)):
    """
    Dibuja el layout elegido y sus vecinos inmediatos.
    """
    nodes = set(layout)
    for q in layout:
        nodes.update(G.neighbors(q))

    sub = G.subgraph(nodes).copy()
    pos = nx.spring_layout(sub, seed=7)

    node_colors = []
    node_sizes = []
    for node in sub.nodes:
        if node in layout:
            node_colors.append("gold")
            node_sizes.append(700)
        else:
            node_colors.append("lightgray")
            node_sizes.append(300)

    layout_edges = set()

    if topology == "linear":
        layout_edges = {tuple(sorted((layout[i], layout[i + 1]))) for i in range(len(layout) - 1)}

    elif topology == "circular":
        layout_edges = {tuple(sorted((layout[i], layout[i + 1]))) for i in range(len(layout) - 1)}
        if len(layout) >= 3:
            layout_edges.add(tuple(sorted((layout[-1], layout[0]))))

    elif topology == "crca2":
        layout_edges = {tuple(sorted((layout[0], layout[2]))), tuple(sorted((layout[1], layout[2])))}

    elif topology == "tree_bus":
        layout_edges = {
            tuple(sorted((layout[0], layout[1]))), tuple(sorted((layout[1], layout[2]))), tuple(sorted((layout[2], layout[3]))),
            tuple(sorted((layout[5], layout[4]))), tuple(sorted((layout[4], layout[0]))), tuple(sorted((layout[7], layout[6]))),
            tuple(sorted((layout[6], layout[1]))), tuple(sorted((layout[9], layout[8]))), tuple(sorted((layout[8], layout[3]))),
        }

    elif topology == "snowflake":
        layout_edges = {
            tuple(sorted((layout[0], layout[1]))), tuple(sorted((layout[0], layout[2]))), tuple(sorted((layout[0], layout[3]))),
            tuple(sorted((layout[1], layout[4]))), tuple(sorted((layout[4], layout[5]))), tuple(sorted((layout[2], layout[6]))),
            tuple(sorted((layout[6], layout[7]))), tuple(sorted((layout[3], layout[8]))), tuple(sorted((layout[8], layout[9]))),
        }

    elif topology == "qcbm_heavyhex6":
        layout_edges = {
            tuple(sorted((layout[0], layout[1]))), tuple(sorted((layout[0], layout[2]))),
            tuple(sorted((layout[0], layout[3]))), tuple(sorted((layout[1], layout[4]))),
            tuple(sorted((layout[2], layout[5])))
        }
        
    elif topology == "qcbm_heavyhex8":
        layout_edges = {
            tuple(sorted((layout[0], layout[1]))), tuple(sorted((layout[1], layout[2]))), tuple(sorted((layout[2], layout[3]))),
            tuple(sorted((layout[3], layout[4]))), tuple(sorted((layout[4], layout[5]))), tuple(sorted((layout[2], layout[6]))),
            tuple(sorted((layout[4], layout[7]))),
        }

    elif topology == "time_tree4":
        layout_edges = {
            tuple(sorted((layout[0], layout[4]))), tuple(sorted((layout[1], layout[4]))), tuple(sorted((layout[4], layout[6]))),
            tuple(sorted((layout[6], layout[5]))), tuple(sorted((layout[5], layout[2]))), tuple(sorted((layout[5], layout[3]))),
        }

    elif topology == "heavy_hex_star":
        a = len(layout) - 1
        layout_edges = {
            tuple(sorted((layout[a], layout[0]))),
            tuple(sorted((layout[a], layout[1]))),
            tuple(sorted((layout[a], layout[2]))),
        }
        for i in range(2, a - 1):
            layout_edges.add(tuple(sorted((layout[i], layout[i + 1]))))

    else:
        raise ValueError(f"Topology {topology} not recognized in drawing function.")

    edge_colors = []
    edge_widths = []

    for u, v in sub.edges:
        if tuple(sorted((u, v))) in layout_edges:
            edge_colors.append("crimson")
            edge_widths.append(3.0)
        else:
            edge_colors.append("gray")
            edge_widths.append(1.0)

    plt.figure(figsize=figsize)
    nx.draw_networkx(
        sub,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=edge_colors,
        width=edge_widths,
        font_size=10,
    )
    plt.title(f"Subgrafo local alrededor del layout elegido ({topology})")
    plt.axis("off")
    plt.show()


def circuit_metrics(qc):
    """
    Devuelve métricas simples pero útiles del circuito.
    """
    ops = qc.count_ops()
    two_qubit_gates = sum(1 for inst in qc.data if len(inst.qubits) == 2)

    return {
        "depth": qc.depth(),
        "size": qc.size(),
        "width": qc.width(),
        "num_parameters": len(qc.parameters),
        "two_qubit_gates": two_qubit_gates,
        "swap_count": int(ops.get("swap", 0)),
        "measure_count": int(ops.get("measure", 0)),
        "ops": dict(ops),
    }


def summarize_circuit(qc, label="circuito"):
    m = circuit_metrics(qc)
    print(f"Resumen de {label}:")
    print(f"  depth            = {m['depth']}")
    print(f"  size             = {m['size']}")
    print(f"  width            = {m['width']}")
    print(f"  num_parameters   = {m['num_parameters']}")
    print(f"  two_qubit_gates  = {m['two_qubit_gates']}")
    print(f"  swap_count       = {m['swap_count']}")
    print(f"  measure_count    = {m['measure_count']}")
    print("  count_ops        =")
    pprint(m["ops"])