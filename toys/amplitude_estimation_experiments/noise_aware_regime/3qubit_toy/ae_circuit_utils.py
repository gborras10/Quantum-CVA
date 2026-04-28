from __future__ import annotations

from math import pi
from typing import Any, Iterable, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem
from quantum_cva.quantum_hardware_utilities.layout_utils import select_best_layout
from quantum_cva.quantum_hardware_utilities.transpile_utils import (
    DEFAULT_TRANSPILER_SEEDS,
    FixedTranspilationPlan,
    LayoutCandidate,
    select_best_fixed_transpilation_plan,
)


INITIAL_LAYOUT = [98, 111, 110]
PHYSICAL_BACKEND_NAME = "ibm_basquecountry"
CANONICAL_OBJECTIVE_RY_OFFSET = -0.10
OBJECTIVE_QUBITS = [2]
REFLECTION_QUBITS = [0, 1, 2]
TRANSPILER_OPTIMIZATION_LEVEL = 3
TRANSPILER_SEED = 1234
DEFAULT_AE_REFERENCE_KS = (0, 1, 2, 4, 8)
DEFAULT_ROUTING_METHOD = "sabre"
CVA_STRESS_LAYERS = 3


def canonical_objective_ry_offset() -> float:
    return float(CANONICAL_OBJECTIVE_RY_OFFSET)


def ry_as_rz_rx_rz(qc: QuantumCircuit, theta: float, qubit: int) -> None:
    qc.rz(-pi / 2, qubit)
    qc.rx(theta, qubit)
    qc.rz(pi / 2, qubit)


def append_cva_stress_layers(qc: QuantumCircuit, layers: int) -> None:
    """Add structured market-credit-exposure layers to the CVA toy ansatz."""
    for layer in range(int(layers)):
        scale = float(layer + 1)

        # q0: market factor, q1: credit/time stress, q2: loss/exposure flag.
        ry_as_rz_rx_rz(qc, 0.10 + 0.02 * scale, 0)
        qc.rz(0.05 * scale, 0)
        ry_as_rz_rx_rz(qc, -0.08 + 0.015 * scale, 1)
        qc.rz(-0.04 * scale, 1)

        qc.rzz(0.18 + 0.03 * scale, 0, 1)
        qc.cry(0.16 + 0.04 * scale, 0, 1)

        qc.rzz(-0.20 - 0.025 * scale, 1, 2)
        qc.cry(0.24 + 0.035 * scale, 1, 2)
        qc.rz(0.03 * scale, 2)


def build_state_preparation(
    objective_ry_offset: float = CANONICAL_OBJECTIVE_RY_OFFSET,
) -> QuantumCircuit:
    objective_ry_offset = float(objective_ry_offset)
    state_preparation = QuantumCircuit(3, name="A")
    state_preparation.metadata = {
        "canonical_topology": True,
        "objective_ry_offset": float(objective_ry_offset),
    }

    # This deliberately compact toy keeps a single local-rotation layer, one
    # non-trivial entangling/mixing layer, and a final objective shift.
    state_preparation.ry(0.72, 0)
    state_preparation.rz(0.23, 0)
    state_preparation.ry(0.51, 1)
    state_preparation.rz(-0.17, 1)
    state_preparation.ry(0.65 + objective_ry_offset, 2)
    state_preparation.rz(0.19, 2)

    state_preparation.cry(0.60, 0, 2)
    state_preparation.cry(-0.45, 1, 2)
    state_preparation.cry(0.35, 0, 1)
    state_preparation.cry(0.25, 2, 0)
    state_preparation.rzz(0.42, 0, 1)
    state_preparation.rzz(-0.28, 1, 2)

    state_preparation.ry(objective_ry_offset, 2)

    return state_preparation


def build_oracle() -> QuantumCircuit:
    oracle = QuantumCircuit(3, name="oracle")
    oracle.rz(pi, 2)
    return oracle


def build_estimation_problem(
    objective_ry_offset: float = CANONICAL_OBJECTIVE_RY_OFFSET,
) -> EstimationProblem:
    objective_ry_offset = float(objective_ry_offset)
    state_preparation = build_state_preparation(objective_ry_offset)
    grover_operator = GroverOperator(
        oracle=build_oracle(),
        state_preparation=state_preparation,
        reflection_qubits=REFLECTION_QUBITS,
        insert_barriers=False,
    )

    return EstimationProblem(
        state_preparation=state_preparation,
        grover_operator=grover_operator,
        objective_qubits=OBJECTIVE_QUBITS,
    )


def true_amplitude(problem: EstimationProblem) -> float:
    state = Statevector.from_instruction(problem.state_preparation)
    probabilities = state.probabilities_dict(qargs=list(problem.objective_qubits))
    good_key = "1" * len(problem.objective_qubits)
    return float(probabilities.get(good_key, 0.0))


def build_problem_with_true_amplitude(
    objective_ry_offset: float = CANONICAL_OBJECTIVE_RY_OFFSET,
) -> tuple[EstimationProblem, float]:
    problem = build_estimation_problem(float(objective_ry_offset))
    return problem, true_amplitude(problem)


def construct_measured_circuit(
    problem: EstimationProblem,
    k: int,
) -> QuantumCircuit:
    num_qubits = max(
        problem.state_preparation.num_qubits,
        problem.grover_operator.num_qubits,
    )
    circuit = QuantumCircuit(num_qubits, name=f"AE_k_{k}")
    state_prep_metadata = getattr(problem.state_preparation, "metadata", None) or {}
    circuit.metadata = {
        "canonical_topology": True,
        "objective_ry_offset": float(
            state_prep_metadata.get("objective_ry_offset", canonical_objective_ry_offset())
        ),
        "grover_power": int(k),
        "K_value": int(2 * int(k) + 1),
    }
    circuit.compose(problem.state_preparation, inplace=True)
    if k > 0:
        grover_power = problem.grover_operator.power(k)
        if hasattr(grover_power, "decompose"):
            grover_power = grover_power.decompose(reps=10)
        circuit.compose(grover_power, inplace=True)

    creg = ClassicalRegister(len(problem.objective_qubits), "c0")
    circuit.add_register(creg)
    circuit.measure(problem.objective_qubits, creg[:])
    return circuit


def transpile_for_execution(
    circuit: QuantumCircuit,
    backend: object,
    *,
    initial_layout: list[int] | None = None,
    basis_gates: list[str] | None = None,
) -> QuantumCircuit:
    transpile_kwargs = {
        "backend": backend,
        "optimization_level": TRANSPILER_OPTIMIZATION_LEVEL,
        "seed_transpiler": TRANSPILER_SEED,
    }
    if initial_layout is not None:
        transpile_kwargs["initial_layout"] = initial_layout
    if basis_gates is not None:
        transpile_kwargs["basis_gates"] = basis_gates
    return transpile(circuit, **transpile_kwargs)


def build_reference_circuits(
    problem: EstimationProblem,
    reference_ks: Iterable[int] = DEFAULT_AE_REFERENCE_KS,
) -> list[QuantumCircuit]:
    normalized_reference_ks = tuple(sorted({int(k) for k in reference_ks}))
    if not normalized_reference_ks:
        raise ValueError("reference_ks must contain at least one value.")
    return [construct_measured_circuit(problem, k) for k in normalized_reference_ks]


def quality_layout_candidates(
    backend: Any,
    logical_qubit_count: int,
) -> list[LayoutCandidate]:
    candidates: dict[tuple[int, ...], LayoutCandidate] = {}

    for topology in ("linear", "crca2"):
        try:
            layout, score, _ = select_best_layout(
                backend,
                topology=topology,
                length=int(logical_qubit_count),
            )
        except Exception:
            continue

        initial_layout = tuple(int(q) for q in layout)
        candidates.setdefault(
            initial_layout,
            LayoutCandidate(
                initial_layout=initial_layout,
                source=f"quality_{topology}(score={float(score):.4f})",
            ),
        )

    return list(candidates.values())


def choose_transpilation_plan(
    backend: Any,
    problem: EstimationProblem,
    *,
    optimization_level: int = TRANSPILER_OPTIMIZATION_LEVEL,
    reference_ks: Iterable[int] = DEFAULT_AE_REFERENCE_KS,
    routing_method: str | None = DEFAULT_ROUTING_METHOD,
    discovery_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    evaluation_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
) -> FixedTranspilationPlan:
    reference_circuits = build_reference_circuits(problem, reference_ks)
    logical_qubit_count = max(
        problem.state_preparation.num_qubits,
        problem.grover_operator.num_qubits,
    )
    candidate_layouts = quality_layout_candidates(
        backend,
        logical_qubit_count=logical_qubit_count,
    )

    return select_best_fixed_transpilation_plan(
        backend,
        reference_circuits,
        candidate_layouts=candidate_layouts,
        optimization_level=int(optimization_level),
        routing_method=routing_method,
        discovery_seeds=tuple(int(seed) for seed in discovery_seeds),
        evaluation_seeds=tuple(int(seed) for seed in evaluation_seeds),
        include_sabre_candidates=True,
    )
