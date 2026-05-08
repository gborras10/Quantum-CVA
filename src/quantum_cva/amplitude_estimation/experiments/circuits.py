from __future__ import annotations

import time
import types
from collections.abc import Iterable, Sequence
from typing import Any

from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit_algorithms import EstimationProblem


def _safe_decompose(circuit: QuantumCircuit, reps: int = 10) -> QuantumCircuit:
    try:
        return circuit.decompose(reps=int(reps))
    except TypeError:
        out = circuit
        for _ in range(max(0, int(reps))):
            out = out.decompose()
        return out


def set_circuit_metadata(
    circuit: QuantumCircuit,
    *,
    k: int,
    source: str,
    extra: dict[str, Any] | None = None,
) -> QuantumCircuit:
    metadata = dict(getattr(circuit, "metadata", None) or {})
    metadata.update(
        {
            "source": str(source),
            "grover_power": int(k),
            "amplification_factor": int(2 * int(k) + 1),
            "K_value": int(2 * int(k) + 1),
        }
    )
    if extra:
        metadata.update(extra)
    circuit.metadata = metadata
    return circuit


def build_unmeasured_query_circuit(
    problem: EstimationProblem,
    k: int,
    *,
    source: str = "ae_query",
    decompose_reps: int = 10,
) -> QuantumCircuit:
    """Construct ``Q^k A |0>`` without measurements."""
    k_int = int(k)
    if k_int < 0:
        raise ValueError("Grover power k must be non-negative.")

    num_qubits = max(
        int(problem.state_preparation.num_qubits),
        int(problem.grover_operator.num_qubits),
    )
    circuit = QuantumCircuit(num_qubits, name=f"AE_k_{k_int}")
    circuit.compose(problem.state_preparation, inplace=True)
    if k_int > 0:
        grover_power = problem.grover_operator.power(k_int)
        if hasattr(grover_power, "decompose"):
            grover_power = _safe_decompose(grover_power, reps=decompose_reps)
        circuit.compose(grover_power, inplace=True)
    return set_circuit_metadata(circuit, k=k_int, source=source)


def construct_measured_circuit(
    problem: EstimationProblem,
    k: int,
    *,
    source: str = "ae_query",
    classical_register_name: str = "c0",
    barrier: bool = True,
    decompose_reps: int = 10,
) -> QuantumCircuit:
    """Construct ``Q^k A |0>`` and measure all objective qubits."""
    circuit = build_unmeasured_query_circuit(
        problem,
        k,
        source=source,
        decompose_reps=decompose_reps,
    )
    creg = ClassicalRegister(len(problem.objective_qubits), classical_register_name)
    circuit.add_register(creg)
    if barrier:
        circuit.barrier()
    circuit.measure(problem.objective_qubits, creg[:])
    return circuit


def construct_metadata_query_circuit(
    problem: EstimationProblem,
    k: int,
    *,
    source: str = "ae_query_metadata_only",
    measurement: bool = False,
    classical_register_name: str = "c0",
) -> QuantumCircuit:
    """Construct a minimal metadata-only query circuit for fast ideal runs.

    This intentionally does not compose ``A`` or ``Q^k``. It is only valid with
    samplers that use ``circuit.metadata['grover_power']`` instead of simulating
    circuit state evolution.
    """
    k_int = int(k)
    if k_int < 0:
        raise ValueError("Grover power k must be non-negative.")
    objective_qubits = [int(qubit) for qubit in problem.objective_qubits]
    num_qubits = max(
        int(problem.state_preparation.num_qubits),
        max(objective_qubits, default=-1) + 1,
    )
    circuit = QuantumCircuit(num_qubits, name=f"AE_metadata_k_{k_int}")
    if measurement:
        creg = ClassicalRegister(len(objective_qubits), classical_register_name)
        circuit.add_register(creg)
        if objective_qubits:
            circuit.measure(objective_qubits, creg[:])
    return set_circuit_metadata(
        circuit,
        k=k_int,
        source=source,
        extra={"construction_mode": "metadata_only"},
    )


def build_reference_circuits(
    problem: EstimationProblem,
    reference_ks: Iterable[int] = (0, 1, 2, 4, 8),
) -> list[QuantumCircuit]:
    normalized = tuple(sorted({int(k) for k in reference_ks}))
    if not normalized:
        raise ValueError("reference_ks must contain at least one value.")
    return [construct_measured_circuit(problem, k, source="reference") for k in normalized]


def circuit_k(circuit: QuantumCircuit) -> int | None:
    metadata = dict(getattr(circuit, "metadata", None) or {})
    for key in ("grover_power", "bae_control"):
        if key in metadata:
            return int(metadata[key])
    return None


def two_qubit_count(circuit: QuantumCircuit) -> int:
    return int(
        sum(
            1
            for instruction in circuit.data
            if int(instruction.operation.num_qubits) == 2
        )
    )


def count_by_qubit_arity(circuit: QuantumCircuit) -> tuple[int, int, int]:
    one_qubit = 0
    two_qubit = 0
    multi_qubit = 0
    for instruction in circuit.data:
        arity = int(instruction.operation.num_qubits)
        if arity == 1:
            one_qubit += 1
        elif arity == 2:
            two_qubit += 1
        elif arity > 2:
            multi_qubit += 1
    return one_qubit, two_qubit, multi_qubit


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    one_qubit, two_qubit, multi_qubit = count_by_qubit_arity(circuit)
    ops = {str(name): int(count) for name, count in circuit.count_ops().items()}
    return {
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "width": int(circuit.width()),
        "one_qubit_gates": int(one_qubit),
        "two_qubit_gates": int(two_qubit),
        "multi_qubit_gates": int(multi_qubit),
        "swap_count": int(ops.get("swap", 0)),
        "ops": ops,
    }


def active_qubits(circuit: QuantumCircuit) -> list[int]:
    return sorted(
        {
            int(circuit.find_bit(qubit).index)
            for instruction in circuit.data
            for qubit in instruction.qubits
        }
    )


def stable_circuit_key(circuit: QuantumCircuit) -> str:
    try:
        return qasm3.dumps(circuit)
    except Exception:
        ops = tuple(sorted((str(k), int(v)) for k, v in circuit.count_ops().items()))
        return str((circuit.num_qubits, circuit.num_clbits, circuit.size(), circuit.depth(), ops))


def patch_construct_circuit(
    solver: Any,
    *,
    source: str,
    decompose_reps: int = 10,
    cache_by_k: bool = True,
    circuit_cache: dict[tuple[Any, ...], QuantumCircuit] | None = None,
    construction_mode: str = "full",
) -> None:
    """Patch solvers that expose Qiskit-style ``construct_circuit`` methods.

    The patched method caches by problem identity, ``k`` and measurement mode.
    Adaptive AE algorithms often revisit the same Grover power, and for CVA
    circuits rebuilding ``A Q^k`` dominates wall-clock time.
    """

    cache: dict[tuple[Any, ...], QuantumCircuit] = (
        circuit_cache if circuit_cache is not None else {}
    )
    mode = str(construction_mode).strip().lower()
    if mode not in {"full", "metadata_only"}:
        raise ValueError("construction_mode must be 'full' or 'metadata_only'.")
    metrics: dict[str, float | int] = {
        "construct_circuit_wall_seconds": 0.0,
        "construct_circuit_cache_hits": 0,
        "construct_circuit_cache_misses": 0,
    }

    def _construct(
        self: Any,
        estimation_problem: EstimationProblem,
        k: int = 0,
        measurement: bool = False,
    ) -> QuantumCircuit:
        k_int = int(k)
        measured = bool(measurement)
        grover_identity = (
            "metadata_only"
            if mode == "metadata_only"
            else id(estimation_problem.grover_operator)
        )
        cache_key = (
            id(estimation_problem.state_preparation),
            grover_identity,
            k_int,
            measured,
            mode,
        )
        start = time.perf_counter()
        if cache_by_k and cache_key in cache:
            metrics["construct_circuit_cache_hits"] = int(
                metrics["construct_circuit_cache_hits"]
            ) + 1
            metrics["construct_circuit_wall_seconds"] = float(
                metrics["construct_circuit_wall_seconds"]
            ) + (time.perf_counter() - start)
            return cache[cache_key]

        if mode == "metadata_only":
            circuit = construct_metadata_query_circuit(
                estimation_problem,
                k_int,
                source=source,
                measurement=measured,
            )
        elif measurement:
            circuit = construct_measured_circuit(
                estimation_problem,
                k_int,
                source=source,
                decompose_reps=decompose_reps,
            )
        else:
            circuit = build_unmeasured_query_circuit(
                estimation_problem,
                k_int,
                source=source,
                decompose_reps=decompose_reps,
            )
        if cache_by_k:
            cache[cache_key] = circuit
        metrics["construct_circuit_cache_misses"] = int(
            metrics["construct_circuit_cache_misses"]
        ) + 1
        metrics["construct_circuit_wall_seconds"] = float(
            metrics["construct_circuit_wall_seconds"]
        ) + (time.perf_counter() - start)
        return circuit

    solver._construct_circuit_cache = cache
    solver._construct_circuit_metrics = metrics
    solver.construct_circuit = types.MethodType(_construct, solver)


def extract_initial_layout(
    transpiled_circuit: QuantumCircuit,
    *,
    logical_qubit_count: int,
) -> list[int] | None:
    layout = getattr(transpiled_circuit, "layout", None)
    if layout is None or not hasattr(layout, "initial_index_layout"):
        return None
    try:
        indices = list(layout.initial_index_layout())
    except Exception:
        return None
    if len(indices) < int(logical_qubit_count):
        return None
    return [int(idx) for idx in indices[: int(logical_qubit_count)]]
