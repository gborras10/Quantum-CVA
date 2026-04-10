from collections.abc import Sequence
from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.quantum_hardware_utilities.layout_utils import select_best_layout, summarize_circuit


def _as_1d_float(array_like: np.ndarray) -> np.ndarray:
    return np.asarray(array_like, dtype=float).ravel()


def _npz_int(npz_data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in npz_data:
        return int(default)
    return int(np.asarray(npz_data[key]).item())


def _npz_str(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz_data:
        return str(default)
    return str(np.asarray(npz_data[key]).item())


def _metadata_dict(npz_data: np.lib.npyio.NpzFile) -> dict:
    if "metadata" not in npz_data:
        return {}
    maybe_dict = npz_data["metadata"]
    if hasattr(maybe_dict, "item"):
        maybe_dict = maybe_dict.item()
    return maybe_dict if isinstance(maybe_dict, dict) else {}


def _assert_param_size(label: str, theta: np.ndarray, expected_size: int) -> None:
    actual_size = int(np.asarray(theta).size)
    if actual_size != int(expected_size):
        raise ValueError(
            f"Parameter-size mismatch for {label}: expected {expected_size}, got {actual_size}."
        )


def _assert_vector_size(label: str, values: np.ndarray, expected_size: int) -> None:
    actual_size = int(np.asarray(values).size)
    if actual_size != int(expected_size):
        raise ValueError(
            f"Vector-size mismatch for {label}: expected {expected_size}, got {actual_size}."
        )


def _bind_qcbm_ansatz(qcbm: MLQcbmCircuit, theta: np.ndarray) -> Any:
    bind_map = {qcbm.theta[i]: float(theta[i]) for i in range(qcbm.n_params)}
    return qcbm.qc.assign_parameters(bind_map, inplace=False)


def _bind_crca_eval(crca: CrcaCircuit, theta: np.ndarray) -> Any:
    bind_map = {crca.theta[i]: float(theta[i]) for i in range(crca.n_params)}
    return crca.qc_eval.assign_parameters(bind_map, inplace=False)


def _build_backend_graph(backend: Any) -> dict[int, set[int]]:
    coupling_map = backend.configuration().coupling_map
    adjacency: dict[int, set[int]] = {}
    for a, b in coupling_map:
        qa = int(a)
        qb = int(b)
        adjacency.setdefault(qa, set()).add(qb)
        adjacency.setdefault(qb, set()).add(qa)
    return adjacency


def _shortest_path_len(adjacency: dict[int, set[int]], src: int, dst: int) -> int | None:
    if src == dst:
        return 0

    seen = {src}
    queue: list[tuple[int, int]] = [(src, 0)]
    head = 0

    while head < len(queue):
        node, dist = queue[head]
        head += 1
        for neighbor in adjacency.get(node, ()):
            if neighbor == dst:
                return dist + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))

    return None


def _pick_best_free_qubit_near_targets(
    *,
    preferred: int | None,
    used: set[int],
    adjacency: dict[int, set[int]],
    target_nodes: Sequence[int],
    backend_num_qubits: int,
    preferred_bonus: float = 0.25,
) -> int:
    """
    Pick a free physical qubit close to target_nodes.

    Score priority:
    1) smaller max distance to targets
    2) smaller sum of distances to targets
    3) if tied, slight preference for preferred
    4) lower physical index
    """
    targets = [int(q) for q in target_nodes]
    if not targets:
        raise ValueError("target_nodes must be non-empty.")

    best_q: int | None = None
    best_score: tuple[float, float, float, int] | None = None

    for q in range(int(backend_num_qubits)):
        if q in used:
            continue

        dists = [_shortest_path_len(adjacency, q, t) for t in targets]
        if any(d is None for d in dists):
            continue

        max_dist = float(max(dists))
        sum_dist = float(sum(dists))
        pref_penalty = 0.0 if (preferred is not None and q == int(preferred)) else preferred_bonus
        score = (max_dist, sum_dist, pref_penalty, q)

        if best_score is None or score < best_score:
            best_score = score
            best_q = q

    if best_q is None:
        raise RuntimeError("No feasible free qubit found near the target nodes.")

    return int(best_q)


def _select_layout_for_training(
    backend: Any,
    *,
    topology: str,
    length: int,
    readout_quantile: float,
    local_2q_quantile: float,
    relax_if_needed: bool,
) -> tuple[list[int], float, dict[str, Any]]:
    return select_best_layout(
        backend,
        topology=topology,
        length=length,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )


def _transpile_with_layout(
    circuit: Any,
    *,
    backend: Any,
    initial_layout: Sequence[int],
    optimization_level: int,
    seed_transpiler: int,
    approximation_degree: float,
) -> Any:
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        initial_layout=[int(q) for q in initial_layout],
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )
    return pm.run(circuit)


def _transpile_with_sabre_layout(
    circuit: Any,
    *,
    backend: Any,
    optimization_level: int,
    seed_transpiler: int,
    approximation_degree: float,
) -> Any:
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )
    return pm.run(circuit)


def _two_qubit_gate_count(circuit: Any) -> int:
    return int(sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2))


def _transpile_with_sabre_multiseed(
    circuit: Any,
    *,
    backend: Any,
    optimization_level: int,
    seeds: Sequence[int],
    approximation_degree: float,
) -> tuple[Any, int, dict[str, int], list[dict[str, int]]]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("At least one seed is required for Sabre multi-seed transpilation.")

    best_circuit: Any | None = None
    best_seed: int | None = None
    best_metrics: dict[str, int] | None = None
    best_key: tuple[int, int, int, int] | None = None
    all_metrics: list[dict[str, int]] = []

    for seed in seed_list:
        transpiled = _transpile_with_sabre_layout(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed,
            approximation_degree=approximation_degree,
        )
        metrics = {
            "seed": seed,
            "depth": int(transpiled.depth()),
            "two_qubit_gates": _two_qubit_gate_count(transpiled),
            "size": int(transpiled.size()),
        }
        all_metrics.append(metrics)

        key = (
            metrics["depth"],
            metrics["two_qubit_gates"],
            metrics["size"],
            seed,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_circuit = transpiled
            best_seed = seed
            best_metrics = metrics

    if best_circuit is None or best_seed is None or best_metrics is None:
        raise RuntimeError("Sabre multi-seed transpilation failed to produce a circuit.")

    return best_circuit, best_seed, best_metrics, all_metrics


def _choose_positive_exposure_physical_params(
    trained_params: np.ndarray,
    physical_n_params: int,
    *,
    fallback_seed: int,
    fallback_scale: float,
) -> np.ndarray:
    if trained_params.size == physical_n_params:
        return trained_params

    rng = np.random.default_rng(fallback_seed)
    return fallback_scale * rng.standard_normal(physical_n_params).astype(float)


def _build_improved_cva_initial_layout(
    *,
    real_backend: Any,
    qcbm_layout: Sequence[int],
    positive_exposure_layout: Sequence[int],
    default_layout: Sequence[int],
    discount_layout: Sequence[int],
) -> tuple[list[int], list[int], list[int], int, int, int]:
    cva_state_layout = [int(qubit) for qubit in qcbm_layout]
    used_physical = set(cva_state_layout)

    ancilla_exposure_pref = int(positive_exposure_layout[-1])
    ancilla_default_pref = int(default_layout[-1])
    ancilla_discount_pref = int(discount_layout[-1])

    backend_num_qubits = int(real_backend.configuration().num_qubits)
    backend_graph = _build_backend_graph(real_backend)

    ancilla_exposure_phys = _pick_best_free_qubit_near_targets(
        preferred=ancilla_exposure_pref,
        used=used_physical,
        adjacency=backend_graph,
        target_nodes=cva_state_layout,
        backend_num_qubits=backend_num_qubits,
    )
    used_physical.add(ancilla_exposure_phys)

    time_phys = [int(qubit) for qubit in qcbm_layout[:2]]

    ancilla_default_phys = _pick_best_free_qubit_near_targets(
        preferred=ancilla_default_pref,
        used=used_physical,
        adjacency=backend_graph,
        target_nodes=time_phys,
        backend_num_qubits=backend_num_qubits,
    )
    used_physical.add(ancilla_default_phys)

    ancilla_discount_phys = _pick_best_free_qubit_near_targets(
        preferred=ancilla_discount_pref,
        used=used_physical,
        adjacency=backend_graph,
        target_nodes=time_phys,
        backend_num_qubits=backend_num_qubits,
    )
    used_physical.add(ancilla_discount_phys)

    cva_initial_layout = cva_state_layout + [
        ancilla_exposure_phys,
        ancilla_default_phys,
        ancilla_discount_phys,
    ]

    return (
        cva_initial_layout,
        cva_state_layout,
        time_phys,
        ancilla_exposure_phys,
        ancilla_default_phys,
        ancilla_discount_phys,
    )


def _print_layout_summary(
    *,
    qcbm_requested_topology: str,
    qcbm_layout_meta: dict[str, Any],
    qcbm_layout: Sequence[int],
    physical_positive_exposure_topology: str,
    positive_exposure_layout_meta: dict[str, Any],
    positive_exposure_layout: Sequence[int],
    default_requested_topology: str,
    default_layout_meta: dict[str, Any],
    default_layout: Sequence[int],
    discount_requested_topology: str,
    discount_layout_meta: dict[str, Any],
    discount_layout: Sequence[int],
    cva_initial_layout: Sequence[int],
) -> None:
    print("\n=== Layout Summary ===")
    print(
        "QCBM topology requested/effective: "
        f"{qcbm_requested_topology} / {qcbm_layout_meta['selected_topology']}"
    )
    print(f"QCBM layout: {[int(q) for q in qcbm_layout]}")
    print(
        "Positive exposure topology requested/effective: "
        f"{physical_positive_exposure_topology} / {positive_exposure_layout_meta['selected_topology']}"
    )
    print(f"Positive exposure layout: {[int(q) for q in positive_exposure_layout]}")
    print(
        "Default probabilities topology requested/effective: "
        f"{default_requested_topology} / {default_layout_meta['selected_topology']}"
    )
    print(f"Default probabilities layout: {[int(q) for q in default_layout]}")
    print(
        "Discount factors topology requested/effective: "
        f"{discount_requested_topology} / {discount_layout_meta['selected_topology']}"
    )
    print(f"Discount factors layout: {[int(q) for q in discount_layout]}")
    print(f"CVA initial layout (state + ancillas): {[int(q) for q in cva_initial_layout]}")


def _print_circuit_summary(
    *,
    qc_qcbm_isa: Any,
    qc_positive_exposure_isa: Any,
    qc_default_isa: Any,
    qc_discount_isa: Any,
    qc_cva_logical: Any,
    qc_cva_isa: Any,
) -> None:
    print("\n=== Circuit Summary ===")
    summarize_circuit(qc_qcbm_isa, label="QCBM transpiled")
    summarize_circuit(qc_positive_exposure_isa, label="Positive exposure transpiled")
    summarize_circuit(qc_default_isa, label="Default probabilities transpiled")
    summarize_circuit(qc_discount_isa, label="Discount factors transpiled")
    summarize_circuit(qc_cva_logical, label="CVA logical (agregado)")
    summarize_circuit(qc_cva_isa, label="CVA transpiled (agregado)")


def _bit_label(bit: Any) -> str:
    register = getattr(bit, "_register", None)
    index = getattr(bit, "_index", None)
    if register is None or index is None:
        return str(bit)
    return f"{register.name}[{int(index)}]"


def _format_gate_params(params: Sequence[Any]) -> str:
    if not params:
        return "-"

    formatted: list[str] = []
    for param in params:
        if isinstance(param, (int, float, np.integer, np.floating)):
            formatted.append(f"{float(param):.8g}")
        else:
            formatted.append(str(param))
    return ", ".join(formatted)


def _operation_text(
    *,
    circuit: Any,
    instruction_index: str,
    operation: Any,
    qubits: Sequence[Any],
) -> str:
    physical_qubits = [int(circuit.find_bit(qubit).index) for qubit in qubits]
    qubit_text = ", ".join(f"q{physical}" for physical in physical_qubits) if physical_qubits else "-"
    params_text = _format_gate_params(operation.params)

    connection_text = ""
    if len(physical_qubits) == 2:
        connection_text = f" | conexion=q{physical_qubits[0]}<->q{physical_qubits[1]}"
    elif len(physical_qubits) > 2:
        connected = "<->".join(f"q{physical}" for physical in physical_qubits)
        connection_text = f" | conexion={connected}"

    return (
        f"{instruction_index}: gate={operation.name}"
        f" | qubits_fisicos=[{qubit_text}]"
        f" | params={params_text}"
        f"{connection_text}"
    )


def _write_transpiled_cva_circuit_text_snapshot(
    *,
    circuit: Any,
    output_path: Path,
    backend_name: str,
    transpilation_method: str,
    selected_seed: int,
    selected_metrics: dict[str, int],
) -> None:
    active_physical_qubits = sorted(
        {
            int(circuit.find_bit(qubit).index)
            for instruction in circuit.data
            for qubit in instruction.qubits
        }
    )

    count_ops = circuit.count_ops()
    two_qubit_edges: dict[tuple[int, int], int] = {}
    qubit_timelines: dict[int, list[str]] = {physical: [] for physical in active_physical_qubits}

    for instruction_index, instruction in enumerate(circuit.data):
        line = _operation_text(
            circuit=circuit,
            instruction_index=f"{instruction_index:04d}",
            operation=instruction.operation,
            qubits=instruction.qubits,
        )
        physical_qubits = [int(circuit.find_bit(qubit).index) for qubit in instruction.qubits]

        for physical in physical_qubits:
            qubit_timelines.setdefault(physical, []).append(line)

        if len(physical_qubits) == 2:
            edge = tuple(sorted((physical_qubits[0], physical_qubits[1])))
            two_qubit_edges[edge] = two_qubit_edges.get(edge, 0) + 1

    layout = getattr(circuit, "layout", None)
    initial_layout_entries: list[tuple[str, int]] = []
    final_layout_entries: list[tuple[str, int]] = []

    if layout is not None:
        initial_virtual_layout = layout.initial_virtual_layout()
        final_virtual_layout = layout.final_virtual_layout()

        if initial_virtual_layout is not None:
            initial_layout_entries = sorted(
                (
                    (_bit_label(bit), int(physical))
                    for bit, physical in initial_virtual_layout.get_virtual_bits().items()
                    if int(physical) in active_physical_qubits
                ),
                key=lambda item: (item[0], item[1]),
            )

        if final_virtual_layout is not None:
            final_layout_entries = sorted(
                (
                    (_bit_label(bit), int(physical))
                    for bit, physical in final_virtual_layout.get_virtual_bits().items()
                    if int(physical) in active_physical_qubits
                ),
                key=lambda item: (item[0], item[1]),
            )

    dag = circuit_to_dag(circuit)
    layer_blocks: list[list[str]] = []
    for layer in dag.layers():
        op_nodes = list(layer["graph"].op_nodes())
        if not op_nodes:
            continue

        ordered_nodes = sorted(
            op_nodes,
            key=lambda node: (
                min(int(circuit.find_bit(qubit).index) for qubit in node.qargs) if node.qargs else -1,
                node.name,
            ),
        )
        layer_lines: list[str] = []
        for node_index, node in enumerate(ordered_nodes):
            layer_lines.append(
                _operation_text(
                    circuit=circuit,
                    instruction_index=f"L{len(layer_blocks):03d}.{node_index:02d}",
                    operation=node.op,
                    qubits=node.qargs,
                )
            )
        layer_blocks.append(layer_lines)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The encoding .* has a limited charset.*",
                category=RuntimeWarning,
            )
            circuit_text_image = str(
                circuit.draw(
                    output="text",
                    fold=-1,
                    idle_wires=False,
                    with_layout=True,
                    cregbundle=False,
                )
            )
    except Exception as exc:  # pragma: no cover - best effort export only
        circuit_text_image = f"No se pudo generar el drawer de texto de Qiskit: {exc}"

    lines: list[str] = [
        "CVA transpiled circuit textual snapshot",
        "======================================",
        "",
        "Este fichero describe el circuito CVA transpilado final seleccionado (`qc_cva_isa`).",
        "",
        "Resumen general",
        "---------------",
        f"backend_name: {backend_name}",
        f"transpilation_method: {transpilation_method}",
        f"selected_seed: {selected_seed}",
        f"depth: {selected_metrics['depth']}",
        f"two_qubit_gates: {selected_metrics['two_qubit_gates']}",
        f"size: {selected_metrics['size']}",
        f"num_qubits_transpiled_circuit: {circuit.num_qubits}",
        f"num_clbits_transpiled_circuit: {circuit.num_clbits}",
        "",
        "Qubits fisicos usados en el circuito final",
        "------------------------------------------",
        f"active_physical_qubits: {active_physical_qubits}",
        f"num_active_physical_qubits: {len(active_physical_qubits)}",
        "",
        "Mapeo logico -> fisico al inicio de la transpilacion",
        "----------------------------------------------------",
    ]

    if initial_layout_entries:
        lines.extend(f"{logical} -> q{physical}" for logical, physical in initial_layout_entries)
    else:
        lines.append("No disponible.")

    lines.extend(
        [
            "",
            "Mapeo logico -> fisico al final de la transpilacion",
            "---------------------------------------------------",
        ]
    )
    if final_layout_entries:
        lines.extend(f"{logical} -> q{physical}" for logical, physical in final_layout_entries)
    else:
        lines.append("No disponible.")

    lines.extend(
        [
            "",
            "Conteo de puertas",
            "-----------------",
        ]
    )
    lines.extend(
        f"{gate_name}: {int(count)}"
        for gate_name, count in sorted(count_ops.items(), key=lambda item: (item[0], item[1]))
    )

    lines.extend(
        [
            "",
            "Conexiones fisicas usadas por puertas de 2 qubits",
            "-------------------------------------------------",
        ]
    )
    if two_qubit_edges:
        lines.extend(
            f"q{edge[0]} <-> q{edge[1]}: {count} puertas de 2 qubits"
            for edge, count in sorted(two_qubit_edges.items(), key=lambda item: (item[0][0], item[0][1]))
        )
    else:
        lines.append("No hay puertas de 2 qubits en el circuito.")

    lines.extend(
        [
            "",
            "Secuencia completa de instrucciones sobre qubits fisicos",
            "-------------------------------------------------------",
        ]
    )
    if circuit.data:
        lines.extend(
            _operation_text(
                circuit=circuit,
                instruction_index=f"{instruction_index:04d}",
                operation=instruction.operation,
                qubits=instruction.qubits,
            )
            for instruction_index, instruction in enumerate(circuit.data)
        )
    else:
        lines.append("El circuito no contiene instrucciones.")

    lines.extend(
        [
            "",
            "Descripcion por capas (layer-by-layer)",
            "--------------------------------------",
        ]
    )
    if layer_blocks:
        for layer_index, layer_lines in enumerate(layer_blocks):
            lines.append(f"Layer {layer_index:03d}:")
            lines.extend(f"  {line}" for line in layer_lines)
    else:
        lines.append("No se han encontrado capas con operaciones.")

    lines.extend(
        [
            "",
            "Uso de cada qubit fisico en orden temporal",
            "------------------------------------------",
        ]
    )
    if qubit_timelines:
        for physical in active_physical_qubits:
            lines.append(f"q{physical}:")
            timeline = qubit_timelines.get(physical, [])
            if timeline:
                lines.extend(f"  {entry}" for entry in timeline)
            else:
                lines.append("  Sin operaciones.")
    else:
        lines.append("No se detectaron qubits fisicos activos.")

    lines.extend(
        [
            "",
            'Drawer de Qiskit en texto ("imagen" del circuito)',
            "--------------------------------------------------",
            circuit_text_image,
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _compute_statevector_submodel_diagnostics(
    *,
    qcbm: MLQcbmCircuit,
    qcbm_parameters: np.ndarray,
    qcbm_target_distribution: np.ndarray,
    crca_default_probabilities: CrcaCircuit,
    default_probabilities_parameters: np.ndarray,
    default_probabilities_target: np.ndarray,
    crca_discount_factors: CrcaCircuit,
    discount_factors_parameters: np.ndarray,
    discount_factors_target: np.ndarray,
    crca_positive_exposure: CrcaCircuit,
    positive_exposure_parameters: np.ndarray,
    positive_exposure_target: np.ndarray,
) -> tuple[float, float, float, float]:
    qcbm_probabilities_statevector = qcbm.probabilities(
        qcbm_parameters,
        shots=None,
        seed=None,
    )
    qcbm_kl_statevector = qcbm.metrics(
        qcbm_target_distribution,
        qcbm_probabilities_statevector,
        eps=1e-12,
    )["kl"]

    default_probabilities_statevector = crca_default_probabilities.function_values(
        default_probabilities_parameters,
        shots=None,
        seed=None,
    )
    discount_factors_statevector = crca_discount_factors.function_values(
        discount_factors_parameters,
        shots=None,
        seed=None,
    )
    positive_exposure_statevector = crca_positive_exposure.function_values(
        positive_exposure_parameters,
        shots=None,
        seed=None,
    )

    default_probabilities_l2 = float(
        np.linalg.norm(default_probabilities_statevector - default_probabilities_target, ord=2)
    )
    discount_factors_l2 = float(np.linalg.norm(discount_factors_statevector - discount_factors_target, ord=2))
    positive_exposure_l2 = float(np.linalg.norm(positive_exposure_statevector - positive_exposure_target, ord=2))

    return qcbm_kl_statevector, default_probabilities_l2, discount_factors_l2, positive_exposure_l2


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "text.usetex": False,
        }
    )


def _percent_relative_error(estimate: float, reference: float) -> float:
    return float(np.abs(estimate - reference) / reference * 100.0)
