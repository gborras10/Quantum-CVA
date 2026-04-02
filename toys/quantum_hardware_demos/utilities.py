import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

# =========================
# Funciones auxiliares para conectividad y métricas
# =========================
def build_undirected_coupling_graph(coupling_map):
    """
    Convierte el coupling map en un grafo no dirigido.
    """
    G = nx.Graph()
    G.add_edges_from(tuple(edge) for edge in coupling_map)
    return G


def find_best_chain(G, preferred_scores, avoided_qubits, length, default_score=0.15, max_starts=40):
    """
    Busca una cadena simple de 'length' qubits físicos.
    - favorece qubits con score alto,
    - evita qubits problemáticos,
    - permite usar qubits intermedios no rankeados si hace falta.
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
                score = sum(node_score(q) for q in path)
                if score > best_score:
                    best_score = score
                    best_path = path.copy()
                continue

            neighbors = sorted(
                [nb for nb in G.neighbors(current) if nb not in path and nb not in avoided_qubits],
                key=node_score,
                reverse=True,
            )

            for nb in neighbors:
                stack.append((nb, path + [nb]))

    if best_path is None:
        raise RuntimeError(
            "No se ha encontrado una cadena conectada con las restricciones dadas."
        )

    return best_path, best_score


def draw_local_subgraph(G, chain, figsize=(8, 5)):
    """
    Dibuja la cadena elegida y sus vecinos inmediatos para visualizar
    el layout físico local.
    """
    nodes = set(chain)
    for q in chain:
        nodes.update(G.neighbors(q))

    sub = G.subgraph(nodes).copy()
    pos = nx.spring_layout(sub, seed=7)

    node_colors = []
    node_sizes = []
    for node in sub.nodes:
        if node in chain:
            node_colors.append("gold")
            node_sizes.append(700)
        else:
            node_colors.append("lightgray")
            node_sizes.append(300)

    edge_colors = []
    edge_widths = []
    chain_edges = set(tuple(sorted((chain[i], chain[i + 1]))) for i in range(len(chain) - 1))

    for u, v in sub.edges:
        if tuple(sorted((u, v))) in chain_edges:
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
    plt.title("Subgrafo local alrededor de la cadena elegida")
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