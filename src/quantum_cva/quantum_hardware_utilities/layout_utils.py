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
    """
    Busca una cadena simple de 'length' qubits físicos.
    """
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
        raise RuntimeError(
            "No se ha encontrado una cadena conectada con las restricciones dadas."
        )

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
    """
    Busca un ciclo simple de 'length' qubits físicos.
    Útil para topologías lógicas circulares.
    """
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
        raise RuntimeError(
            "No se ha encontrado un ciclo conectado con las restricciones dadas."
        )

    return best_cycle, float(best_score)

def find_best_crca2(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
    Busca la topología mínima de CRCA para 2 controles:

        c0 - a - c1

    Layout devuelto:
        [c0, c1, a]
    """
    avoided_qubits = set(avoided_qubits)

    T = nx.Graph()
    # logical indices:
    # 0=c0, 1=c1, 2=a
    T.add_edges_from([
        (0, 2),
        (1, 2),
    ])

    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}

        node_term = sum(
            preferred_scores.get(inv_map[q], default_score)
            for q in T.nodes
        )

        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [
                tuple(sorted((inv_map[u], inv_map[v])))
                for u, v in T.edges
            ]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[0], inv_map[1], inv_map[2]]

    if best_layout is None:
        raise RuntimeError(
            "No se ha encontrado la topología 'crca2' con las restricciones dadas."
        )

    return best_layout, float(best_score)

def find_best_tree_bus(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
        Search for a specific topology for a 10-qubit QCBM:
        - A central time bus (4 qubits: B0-B1-B2-B3)
        - 3 branches (underlyings S1, S2, S3 with 2 qubits each)
            connected to B0, B1, and B3.
    """
    avoided_qubits = set(avoided_qubits)

        # 1. Define the theoretical graph (template) of the ansatz
    T = nx.Graph()
        # Time bus (logical nodes 0, 1, 2, 3)
    T.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
        # Underlying branches
        # Branch 1 (connected to bus node 0): logical nodes 4, 5
    T.add_edges_from([(5, 4), (4, 0)])
        # Branch 2 (connected to bus node 1): logical nodes 6, 7
    T.add_edges_from([(7, 6), (6, 1)])
        # Branch 3 (connected to bus node 3): logical nodes 8, 9
    T.add_edges_from([(9, 8), (8, 3)])

        # Filter the physical graph G by removing avoided nodes to speed up search
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)
    
    best_layout = None
    best_score = -np.inf

    # Iterate over all isomorphisms found (all ways the comb fits on the chip)
    for mapping in GM.subgraph_isomorphisms_iter():
        # mapping is {physical_node: logical_node}
        # Invert to iterate over the logical structure easily: {logical_node: physical_node}
        inv_map = {v: k for k, v in mapping.items()}
        
        # Compute node score
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        
        # Compute edge score
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        # Use the same weights as in _path_score (node_weight=1.0, edge_weight=0.35)
        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            # Returned layout is an ordered list of physical nodes:
            # [Bus0-3, Branch1_0-1, Branch2_0-1, Branch3_0-1]
            best_layout = [inv_map[i] for i in range(10)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'tree_bus' con las restricciones dadas.")

    return best_layout, best_score

def find_best_snowflake(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
    Busca la topología 'snowflake' (Copo de Nieve) para un QCBM de 10 qubits:
    - Hub central de tiempo (qubit 0).
    - 3 Puntas de tiempo conectadas al Hub (qubits 1, 2, 3).
    - 3 Ramas de subyacentes colgando de las puntas (4-5 cuelga del 1, etc.).
    """
    avoided_qubits = set(avoided_qubits)

    # Definimos el grafo lógico del Snowflake
    T = nx.Graph()
    # Hub y Puntas de Tiempo
    T.add_edges_from([(0, 1), (0, 2), (0, 3)])
    # Rama Subyacente 1 (conecta a la punta 1)
    T.add_edges_from([(1, 4), (4, 5)])
    # Rama Subyacente 2 (conecta a la punta 2)
    T.add_edges_from([(2, 6), (6, 7)])
    # Rama Subyacente 3 (conecta a la punta 3)
    T.add_edges_from([(3, 8), (8, 9)])

    # Filtrar nodos malos para acelerar isomorfismo
    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)
    
    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        # Invertir mapeo para tener {nodo_logico: nodo_fisico}
        inv_map = {v: k for k, v in mapping.items()}
        
        # Calcular Score (Nodos)
        node_term = sum(preferred_scores.get(inv_map[q], default_score) for q in T.nodes)
        
        # Calcular Score (Aristas)
        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [tuple(sorted((inv_map[u], inv_map[v]))) for u, v in T.edges]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(10)]

    if best_layout is None:
        raise RuntimeError("No se ha encontrado la topología 'snowflake' con las restricciones dadas.")

    return best_layout, float(best_score)


def find_best_qcbm_heavyhex8(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
    Busca la topología lógica heavy-hex-friendly para un QCBM de 8 qubits.

    Grafo lógico:
      - ciclo hexagonal: 0-1-2-3-4-5-0
      - hojas: 6 unida a 1, 7 unida a 4

    Layout devuelto:
      [q0, q1, q2, q3, q4, q5, q6, q7]
    """
    avoided_qubits = set(avoided_qubits)

    T = nx.Graph()
    T.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (2, 6),
        (4, 7),
    ])

    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}

        node_term = sum(
            preferred_scores.get(inv_map[q], default_score)
            for q in T.nodes
        )

        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [
                tuple(sorted((inv_map[u], inv_map[v])))
                for u, v in T.edges
            ]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(8)]

    if best_layout is None:
        raise RuntimeError(
            "No se ha encontrado la topología 'qcbm_heavyhex8' con las restricciones dadas."
        )

    return best_layout, float(best_score)

def find_best_time_tree4(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
    Busca la topología para CRCA native con 4 controles temporales:
        t0,t1 -> h0
        t2,t3 -> h1
        h0 - a - h1
    Layout devuelto:
        [t0, t1, t2, t3, h0, h1, a]
    """
    avoided_qubits = set(avoided_qubits)

    T = nx.Graph()
    # logical indices:
    # 0=t0, 1=t1, 2=t2, 3=t3, 4=h0, 5=h1, 6=a
    T.add_edges_from([
        (0, 4),
        (1, 4),
        (4, 6),
        (6, 5),
        (5, 2),
        (5, 3),
    ])

    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}

        node_term = sum(
            preferred_scores.get(inv_map[q], default_score)
            for q in T.nodes
        )

        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [
                tuple(sorted((inv_map[u], inv_map[v])))
                for u, v in T.edges
            ]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(7)]

    if best_layout is None:
        raise RuntimeError(
            "No se ha encontrado la topología 'time_tree4' con las restricciones dadas."
        )

    return best_layout, float(best_score)


def find_best_crca_heavyhex8(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
    Busca la topología lógica heavy-hex-friendly del CRCA native para 8 controles.

    Layout devuelto:
      [c0, c1, c2, c3, c4, c5, c6, c7,
       h0, h1, h2, h3,
       b0, b1, b2, b3,
       g0, g1, a]
    """
    avoided_qubits = set(avoided_qubits)

    T = nx.Graph()

    # logical labels:
    # 0..7   = controles
    # 8..11  = h0..h3
    # 12..15 = b0..b3
    # 16..17 = g0..g1
    # 18     = a
    T.add_edges_from([
        (0, 8), (1, 8),
        (2, 9), (3, 9),
        (4, 10), (5, 10),
        (6, 11), (7, 11),

        (8, 12),
        (9, 13),
        (10, 14),
        (11, 15),

        (12, 16), (13, 16),
        (14, 17), (15, 17),

        (16, 18), (17, 18),
    ])

    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}

        node_term = sum(
            preferred_scores.get(inv_map[q], default_score)
            for q in T.nodes
        )

        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [
                tuple(sorted((inv_map[u], inv_map[v])))
                for u, v in T.edges
            ]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(19)]

    if best_layout is None:
        raise RuntimeError(
            "No se ha encontrado la topología 'crca_heavyhex8' con las restricciones dadas."
        )

    return best_layout, float(best_score)

def find_best_crca_heavyhex10(
    G,
    preferred_scores,
    avoided_qubits,
    default_score=0.15,
    edge_scores=None,
):
    """
    Busca la topología lógica heavy-hex-friendly del CRCA native para 10 controles.

    Layout devuelto:
      [t0, t1, t2, t3, s0, s1, s2, s3, s4, s5,
       h0, h1, h2, h3, h4,
       b0, b1, b2, b3, b4,
       g0, g1, b5, b6, a]
    """
    avoided_qubits = set(avoided_qubits)

    T = nx.Graph()

    # logical labels:
    # 0=t0, 1=t1, 2=t2, 3=t3,
    # 4=s0, 5=s1, 6=s2, 7=s3, 8=s4, 9=s5,
    # 10=h0, 11=h1, 12=h2, 13=h3, 14=h4,
    # 15=b0, 16=b1, 17=b2, 18=b3, 19=b4,
    # 20=g0, 21=g1, 22=b5, 23=b6, 24=a

    T.add_edges_from([
        (0, 10), (1, 10),
        (2, 11), (3, 11),
        (4, 12), (5, 12),
        (6, 13), (7, 13),
        (8, 14), (9, 14),

        (10, 15),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),

        (15, 20), (16, 20),
        (17, 21), (18, 21),

        (20, 22),
        (21, 23),

        (22, 24), (23, 24), (19, 24),
    ])

    valid_nodes = [n for n in G.nodes if n not in avoided_qubits]
    G_sub = G.subgraph(valid_nodes)

    GM = isomorphism.GraphMatcher(G_sub, T)

    best_layout = None
    best_score = -np.inf

    for mapping in GM.subgraph_isomorphisms_iter():
        inv_map = {v: k for k, v in mapping.items()}

        node_term = sum(
            preferred_scores.get(inv_map[q], default_score)
            for q in T.nodes
        )

        edge_term = 0.0
        if edge_scores is not None:
            edges_physical = [
                tuple(sorted((inv_map[u], inv_map[v])))
                for u, v in T.edges
            ]
            edge_term = sum(edge_scores.get(e, 0.0) for e in edges_physical)

        score = float(1.0 * node_term + 0.35 * edge_term)

        if score > best_score:
            best_score = score
            best_layout = [inv_map[i] for i in range(25)]

    if best_layout is None:
        raise RuntimeError(
            "No se ha encontrado la topología 'crca_heavyhex10' con las restricciones dadas."
        )

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
        "qcbm_heavyhex8",
        "time_tree4",
        "crca2",
        "crca_tree10",
        "crca_heavyhex8",
        "crca_heavyhex10",
    }

    if topology not in valid_topologies:
        raise ValueError(
            f"topology must be one of {valid_topologies}. Got: '{topology}'"
        )

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
        return find_best_cycle(
            G,
            preferred_scores,
            current_avoided,
            length,
            default_score=default_score,
            max_starts=max_starts,
            edge_scores=edge_scores,
        )

    def _try_chain(current_avoided):
        return find_best_chain(
            G,
            preferred_scores,
            current_avoided,
            length,
            default_score=default_score,
            max_starts=max_starts,
            edge_scores=edge_scores,
        )
    
    def _try_crca2(current_avoided):
        return find_best_crca2(
            G,
            preferred_scores,
            current_avoided,
            default_score=default_score,
            edge_scores=edge_scores,
        )

    def _try_tree_bus(current_avoided):
        return find_best_tree_bus(
            G,
            preferred_scores,
            current_avoided,
            default_score=default_score,
            edge_scores=edge_scores,
        )

    def _try_snowflake(current_avoided):
        return find_best_snowflake(
            G,
            preferred_scores,
            current_avoided,
            default_score=default_score,
            edge_scores=edge_scores,
        )

    def _try_qcbm_heavyhex8(current_avoided):
        return find_best_qcbm_heavyhex8(
            G,
            preferred_scores,
            current_avoided,
            default_score=default_score,
            edge_scores=edge_scores,
        )

    def _try_crca_heavyhex8(current_avoided):
        return find_best_crca_heavyhex8(
            G,
            preferred_scores,
            current_avoided,
            default_score=default_score,
            edge_scores=edge_scores,
        )

    def _try_crca_heavyhex10(current_avoided):
        return find_best_crca_heavyhex10(
            G,
            preferred_scores,
            current_avoided,
            default_score=default_score,
            edge_scores=edge_scores,
        )

    if topology == "linear":
        layout, score = _try_chain(avoided_qubits)
        metadata = {
            "graph": G,
            "preferred_scores": preferred_scores,
            "avoided_qubits": avoided_qubits,
            "edge_scores": edge_scores,
            "diagnostics": diagnostics,
            "selected_topology": "linear",
            "fallback_used": False,
            "tried": tried,
        }
        return layout, score, metadata
    
    if topology == "crca2":
        try:
            layout, score = _try_crca2(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "crca2",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict crca2 failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_crca2(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "crca2",
                    "fallback_used": True,
                    "tried": tried + ["relaxed crca2 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed crca2 failed")

    if topology == "tree_bus":
        try:
            layout, score = _try_tree_bus(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "tree_bus",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict tree_bus failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_tree_bus(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "tree_bus",
                    "fallback_used": True,
                    "tried": tried + ["relaxed tree_bus succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed tree_bus failed")

            layout, score = _try_chain(relaxed_avoided)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": relaxed_avoided,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "linear",
                "fallback_used": True,
                "tried": tried + ["fallback to linear"],
            }
            return layout, score, metadata

    if topology == "snowflake":
        try:
            layout, score = _try_snowflake(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "snowflake",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict snowflake failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_snowflake(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "snowflake",
                    "fallback_used": True,
                    "tried": tried + ["relaxed snowflake succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed snowflake failed")

            layout, score = _try_chain(relaxed_avoided)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": relaxed_avoided,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "linear",
                "fallback_used": True,
                "tried": tried + ["fallback to linear"],
            }
            return layout, score, metadata

    if topology == "qcbm_heavyhex8":
        try:
            layout, score = _try_qcbm_heavyhex8(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "qcbm_heavyhex8",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict qcbm_heavyhex8 failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_qcbm_heavyhex8(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "qcbm_heavyhex8",
                    "fallback_used": True,
                    "tried": tried + ["relaxed qcbm_heavyhex8 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed qcbm_heavyhex8 failed")

            layout, score = _try_chain(relaxed_avoided)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": relaxed_avoided,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "linear",
                "fallback_used": True,
                "tried": tried + ["fallback to linear"],
            }
            return layout, score, metadata

    if topology == "time_tree4":
        try:
            layout, score = find_best_time_tree4(
                G,
                preferred_scores,
                avoided_qubits,
                default_score=default_score,
                edge_scores=edge_scores,
            )
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "time_tree4",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict time_tree4 failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = find_best_time_tree4(
                    G,
                    preferred_scores,
                    relaxed_avoided,
                    default_score=default_score,
                    edge_scores=edge_scores,
                )
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "time_tree4",
                    "fallback_used": True,
                    "tried": tried + ["relaxed time_tree4 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed time_tree4 failed")

    if topology == "crca_tree10":
        try:
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "crca_tree10",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict crca_tree10 failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "crca_tree10",
                    "fallback_used": True,
                    "tried": tried + ["relaxed crca_tree10 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed crca_tree10 failed")

    if topology == "crca_heavyhex8":
        try:
            layout, score = _try_crca_heavyhex8(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "crca_heavyhex8",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict crca_heavyhex8 failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_crca_heavyhex8(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "crca_heavyhex8",
                    "fallback_used": True,
                    "tried": tried + ["relaxed crca_heavyhex8 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed crca_heavyhex8 failed")

    if topology == "crca_heavyhex10":
        try:
            layout, score = _try_crca_heavyhex10(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "crca_heavyhex10",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict crca_heavyhex10 failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_crca_heavyhex10(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "crca_heavyhex10",
                    "fallback_used": True,
                    "tried": tried + ["relaxed crca_heavyhex10 succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed crca_heavyhex10 failed")

    if topology == "circular":
        try:
            layout, score = _try_cycle(avoided_qubits)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": avoided_qubits,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "circular",
                "fallback_used": False,
                "tried": tried,
            }
            return layout, score, metadata
        except RuntimeError:
            tried.append("strict circular failed")

        if relax_if_needed:
            avoided_sorted = sorted(
                avoided_qubits,
                key=lambda q: preferred_scores.get(q, -1e9),
            )
            relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
            try:
                layout, score = _try_cycle(relaxed_avoided)
                metadata = {
                    "graph": G,
                    "preferred_scores": preferred_scores,
                    "avoided_qubits": relaxed_avoided,
                    "edge_scores": edge_scores,
                    "diagnostics": diagnostics,
                    "selected_topology": "circular",
                    "fallback_used": True,
                    "tried": tried + ["relaxed circular succeeded"],
                }
                return layout, score, metadata
            except RuntimeError:
                tried.append("relaxed circular failed")

            layout, score = _try_chain(relaxed_avoided)
            metadata = {
                "graph": G,
                "preferred_scores": preferred_scores,
                "avoided_qubits": relaxed_avoided,
                "edge_scores": edge_scores,
                "diagnostics": diagnostics,
                "selected_topology": "linear",
                "fallback_used": True,
                "tried": tried + ["fallback to linear"],
            }
            return layout, score, metadata

    raise RuntimeError(
        "No se ha encontrado ninguna topología válida con las restricciones dadas."
    )
def draw_local_subgraph(G, layout, topology="linear", figsize=(8, 5)):
    """
    Dibuja el layout elegido y sus vecinos inmediatos.

    Topologías soportadas:
      - linear
      - circular
      - tree_bus
      - snowflake
      - qcbm_heavyhex8
      - time_tree4
      - crca_tree10
      - crca_heavyhex8
      - crca_heavyhex10
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
        layout_edges = {
            tuple(sorted((layout[i], layout[i + 1])))
            for i in range(len(layout) - 1)
        }

    elif topology == "circular":
        layout_edges = {
            tuple(sorted((layout[i], layout[i + 1])))
            for i in range(len(layout) - 1)
        }
        if len(layout) >= 3:
            layout_edges.add(tuple(sorted((layout[-1], layout[0]))))

    elif topology == "crca2":
        if len(layout) != 3:
            raise ValueError(
                "crca2 expects layout of length 3: [c0, c1, a]"
            )

        layout_edges = {
            tuple(sorted((layout[0], layout[2]))),
            tuple(sorted((layout[1], layout[2]))),
        }

    elif topology == "tree_bus":
        layout_edges = {
            tuple(sorted((layout[0], layout[1]))),
            tuple(sorted((layout[1], layout[2]))),
            tuple(sorted((layout[2], layout[3]))),
            tuple(sorted((layout[5], layout[4]))),
            tuple(sorted((layout[4], layout[0]))),
            tuple(sorted((layout[7], layout[6]))),
            tuple(sorted((layout[6], layout[1]))),
            tuple(sorted((layout[9], layout[8]))),
            tuple(sorted((layout[8], layout[3]))),
        }

    elif topology == "snowflake":
        layout_edges = {
            tuple(sorted((layout[0], layout[1]))),
            tuple(sorted((layout[0], layout[2]))),
            tuple(sorted((layout[0], layout[3]))),
            tuple(sorted((layout[1], layout[4]))),
            tuple(sorted((layout[4], layout[5]))),
            tuple(sorted((layout[2], layout[6]))),
            tuple(sorted((layout[6], layout[7]))),
            tuple(sorted((layout[3], layout[8]))),
            tuple(sorted((layout[8], layout[9]))),
        }

    elif topology == "qcbm_heavyhex8":
        if len(layout) != 8:
            raise ValueError(
                "qcbm_heavyhex8 expects layout of length 8: "
                "[q0, q1, q2, q3, q4, q5, q6, q7]"
            )

        layout_edges = {
            tuple(sorted((layout[0], layout[1]))),
            tuple(sorted((layout[1], layout[2]))),
            tuple(sorted((layout[2], layout[3]))),
            tuple(sorted((layout[3], layout[4]))),
            tuple(sorted((layout[4], layout[5]))),
            tuple(sorted((layout[2], layout[6]))),
            tuple(sorted((layout[4], layout[7]))),
        }

    elif topology == "time_tree4":
        if len(layout) != 7:
            raise ValueError(
                "time_tree4 expects layout of length 7: "
                "[t0, t1, t2, t3, h0, h1, a]"
            )

        layout_edges = {
            tuple(sorted((layout[0], layout[4]))),
            tuple(sorted((layout[1], layout[4]))),
            tuple(sorted((layout[4], layout[6]))),
            tuple(sorted((layout[6], layout[5]))),
            tuple(sorted((layout[5], layout[2]))),
            tuple(sorted((layout[5], layout[3]))),
        }

    elif topology == "crca_tree10":
        if len(layout) != 18:
            raise ValueError(
                "crca_tree10 expects layout of length 18: "
                "[t0, t1, t2, t3, s0, s1, s2, s3, s4, s5, "
                "h0, h1, h2, h3, h4, g0, g1, a]"
            )

        layout_edges = {
            tuple(sorted((layout[0], layout[10]))),
            tuple(sorted((layout[1], layout[10]))),
            tuple(sorted((layout[2], layout[11]))),
            tuple(sorted((layout[3], layout[11]))),
            tuple(sorted((layout[4], layout[12]))),
            tuple(sorted((layout[5], layout[12]))),
            tuple(sorted((layout[6], layout[13]))),
            tuple(sorted((layout[7], layout[13]))),
            tuple(sorted((layout[8], layout[14]))),
            tuple(sorted((layout[9], layout[14]))),
            tuple(sorted((layout[10], layout[15]))),
            tuple(sorted((layout[11], layout[15]))),
            tuple(sorted((layout[12], layout[16]))),
            tuple(sorted((layout[13], layout[16]))),
            tuple(sorted((layout[15], layout[17]))),
            tuple(sorted((layout[16], layout[17]))),
            tuple(sorted((layout[14], layout[17]))),
        }

    elif topology == "crca_heavyhex8":
        if len(layout) != 19:
            raise ValueError(
                "crca_heavyhex8 expects layout of length 19: "
                "[c0, c1, c2, c3, c4, c5, c6, c7, "
                "h0, h1, h2, h3, "
                "b0, b1, b2, b3, "
                "g0, g1, a]"
            )

        layout_edges = {
            tuple(sorted((layout[0], layout[8]))),
            tuple(sorted((layout[1], layout[8]))),
            tuple(sorted((layout[2], layout[9]))),
            tuple(sorted((layout[3], layout[9]))),
            tuple(sorted((layout[4], layout[10]))),
            tuple(sorted((layout[5], layout[10]))),
            tuple(sorted((layout[6], layout[11]))),
            tuple(sorted((layout[7], layout[11]))),

            tuple(sorted((layout[8], layout[12]))),
            tuple(sorted((layout[9], layout[13]))),
            tuple(sorted((layout[10], layout[14]))),
            tuple(sorted((layout[11], layout[15]))),

            tuple(sorted((layout[12], layout[16]))),
            tuple(sorted((layout[13], layout[16]))),
            tuple(sorted((layout[14], layout[17]))),
            tuple(sorted((layout[15], layout[17]))),

            tuple(sorted((layout[16], layout[18]))),
            tuple(sorted((layout[17], layout[18]))),
        }

    elif topology == "crca_heavyhex10":
        if len(layout) != 25:
            raise ValueError(
                "crca_heavyhex10 expects layout of length 25: "
                "[t0, t1, t2, t3, s0, s1, s2, s3, s4, s5, "
                "h0, h1, h2, h3, h4, "
                "b0, b1, b2, b3, b4, "
                "g0, g1, b5, b6, a]"
            )

        layout_edges = {
            tuple(sorted((layout[0], layout[10]))),
            tuple(sorted((layout[1], layout[10]))),
            tuple(sorted((layout[2], layout[11]))),
            tuple(sorted((layout[3], layout[11]))),
            tuple(sorted((layout[4], layout[12]))),
            tuple(sorted((layout[5], layout[12]))),
            tuple(sorted((layout[6], layout[13]))),
            tuple(sorted((layout[7], layout[13]))),
            tuple(sorted((layout[8], layout[14]))),
            tuple(sorted((layout[9], layout[14]))),

            tuple(sorted((layout[10], layout[15]))),
            tuple(sorted((layout[11], layout[16]))),
            tuple(sorted((layout[12], layout[17]))),
            tuple(sorted((layout[13], layout[18]))),
            tuple(sorted((layout[14], layout[19]))),

            tuple(sorted((layout[15], layout[20]))),
            tuple(sorted((layout[16], layout[20]))),
            tuple(sorted((layout[17], layout[21]))),
            tuple(sorted((layout[18], layout[21]))),

            tuple(sorted((layout[20], layout[22]))),
            tuple(sorted((layout[21], layout[23]))),

            tuple(sorted((layout[22], layout[24]))),
            tuple(sorted((layout[23], layout[24]))),
            tuple(sorted((layout[19], layout[24]))),
        }

    else:
        raise ValueError(
            "topology must be 'linear', 'circular', 'tree_bus', "
            "'snowflake', 'qcbm_heavyhex8', 'time_tree4', 'crca_tree10', "
            "'crca_heavyhex8' or 'crca_heavyhex10'."
        )

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
    plt.title("Subgrafo local alrededor del layout elegido")
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