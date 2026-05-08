from __future__ import annotations

from typing import Any, Literal

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem


GroverCandidate = Literal[
    "qiskit_standard",
    "qiskit_standard_clean",
    "custom_full",
    "custom_full_flat",
]


def build_grover_candidate(
    state_preparation: QuantumCircuit,
    objective_qubits: list[int],
    good_bitstring: str = "111",
    *,
    candidate: GroverCandidate = "custom_full",
    reflection_qubits: list[int] | None = None,
    mcx_mode: str = "auto",
) -> QuantumCircuit:
    """Build one of the supported Grover candidates.

    ``custom_full_flat`` returns the same mathematical operator as
    ``custom_full``. Its flattening policy is applied by benchmark/query
    builders when constructing ``A Q^k``.
    """
    _validate_inputs(state_preparation, objective_qubits, good_bitstring)
    if reflection_qubits is not None:
        raise ValueError(
            "Only the full zero reflection is supported in the public API; "
            "reflection_qubits must be None."
        )

    if candidate == "qiskit_standard":
        _require_all_ones_good_state(good_bitstring, objective_qubits)
        return _build_qiskit_standard_grover(
            state_preparation,
            objective_qubits,
        )

    if candidate == "qiskit_standard_clean":
        _require_all_ones_good_state(good_bitstring, objective_qubits)
        return _build_qiskit_standard_grover(
            _clean_state_preparation(state_preparation),
            objective_qubits,
        )

    if candidate in {"custom_full", "custom_full_flat"}:
        return _build_custom_full_grover(
            state_preparation,
            objective_qubits,
            good_bitstring,
            mcx_mode=mcx_mode,
        )

    raise ValueError(f"Unsupported Grover candidate: {candidate!r}.")


def good_probability_from_statevector(
    circuit: QuantumCircuit,
    objective_qubits: list[int],
    good_bitstring: str = "111",
) -> float:
    """Return the marginal probability of ``good_bitstring``.

    The convention is exactly Qiskit's
    ``Statevector.probabilities_dict(qargs=objective_qubits)`` convention:
    the rightmost character in a returned key corresponds to
    ``objective_qubits[0]``. The same convention is used when constructing
    phase oracles for non-symmetric states such as ``"101"``.
    """
    _validate_good_bitstring(good_bitstring, len(objective_qubits))
    state = Statevector.from_instruction(_remove_measurements_and_barriers(circuit))
    probabilities = state.probabilities_dict(qargs=list(objective_qubits))
    total = 0.0
    width = len(objective_qubits)
    for bitstring, probability in probabilities.items():
        if _normalize_bitstring(bitstring, width) == str(good_bitstring):
            total += float(probability)
    return float(np.clip(total, 0.0, 1.0))


def validate_grover_amplification(
    state_preparation: QuantumCircuit,
    grover_operator: QuantumCircuit,
    objective_qubits: list[int],
    a_true: float,
    k_values: list[int],
    good_bitstring: str = "111",
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Validate ``Q^k A|0>`` against the ideal amplification law."""
    _validate_inputs(state_preparation, objective_qubits, good_bitstring)
    amplitude = float(np.clip(float(a_true), 0.0, 1.0))
    theta = float(np.arcsin(np.sqrt(amplitude)))
    rows: list[dict[str, Any]] = []
    passed = True

    for k in [int(value) for value in k_values]:
        if k < 0:
            raise ValueError("Grover powers must be non-negative.")
        circuit = _build_power_query(
            state_preparation,
            grover_operator,
            k,
            flatten_grover=False,
        )
        observed = good_probability_from_statevector(
            circuit,
            objective_qubits,
            good_bitstring,
        )
        expected = float(np.sin((2 * k + 1) * theta) ** 2)
        abs_error = abs(observed - expected)
        ok = bool(abs_error <= float(atol))
        passed = passed and ok
        rows.append(
            {
                "k": k,
                "observed": float(observed),
                "expected": float(expected),
                "abs_error": float(abs_error),
                "passed": ok,
            }
        )

    return {
        "passed": bool(passed),
        "atol": float(atol),
        "a_true": amplitude,
        "rows": rows,
    }


def compare_statevectors_up_to_global_phase(
    circuit_a: QuantumCircuit,
    circuit_b: QuantumCircuit,
    atol: float = 1e-8,
) -> bool:
    """Return whether two circuits prepare the same state up to global phase."""
    if int(circuit_a.num_qubits) != int(circuit_b.num_qubits):
        return False
    state_a = Statevector.from_instruction(
        _remove_measurements_and_barriers(circuit_a)
    ).data
    state_b = Statevector.from_instruction(
        _remove_measurements_and_barriers(circuit_b)
    ).data
    if state_a.shape != state_b.shape:
        return False
    index = int(np.argmax(np.abs(state_b)))
    if abs(state_b[index]) <= float(atol):
        return bool(np.allclose(state_a, state_b, atol=atol))
    phase = state_a[index] / state_b[index]
    return bool(np.allclose(state_a, phase * state_b, atol=atol))


def circuit_two_qubit_count(circuit: QuantumCircuit) -> int:
    """Count instructions acting on exactly two qubits."""
    return int(
        sum(
            1
            for instruction in circuit.data
            if int(instruction.operation.num_qubits) == 2
        )
    )


def circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Return logical circuit metrics used by tests and benchmarks."""
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

    ops = {str(name): int(count) for name, count in circuit.count_ops().items()}
    return {
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "width": int(circuit.width()),
        "one_qubit_gates": int(one_qubit),
        "two_qubit_gates": int(two_qubit),
        "multi_qubit_gates": int(multi_qubit),
        "ops": ops,
    }


def _build_qiskit_standard_grover(
    state_preparation: QuantumCircuit,
    objective_qubits: list[int],
) -> QuantumCircuit:
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=list(objective_qubits),
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "")
        == "1" * len(objective_qubits),
    )
    return problem.grover_operator


def _clean_state_preparation(circuit: QuantumCircuit) -> QuantumCircuit:
    """Remove non-unitary/no-op optimization blockers without changing ``A``."""
    stripped = _remove_measurements_and_barriers(circuit)
    candidate = stripped
    for reps in (1, 2):
        try:
            decomposed = stripped.decompose(reps=reps)
            if compare_statevectors_up_to_global_phase(
                stripped,
                decomposed,
                atol=1e-10,
            ):
                candidate = decomposed
        except Exception:
            break
    candidate.name = f"{circuit.name}_clean"
    return candidate


def _build_phase_oracle(
    num_qubits: int,
    objective_qubits: list[int],
    good_bitstring: str = "111",
    *,
    mcx_mode: str = "auto",
) -> QuantumCircuit:
    """Build ``S_f`` marking exactly ``good_bitstring`` with phase ``-1``."""
    _validate_good_bitstring(good_bitstring, len(objective_qubits))
    _validate_mcx_mode(mcx_mode)
    circuit = QuantumCircuit(int(num_qubits), name="S_f")
    objectives = [int(qubit) for qubit in objective_qubits]

    for local_index, qubit in enumerate(objectives):
        if good_bitstring[-1 - local_index] == "0":
            circuit.x(qubit)

    target = objectives[-1]
    if len(objectives) == 1:
        circuit.z(target)
    else:
        circuit.h(target)
        circuit.append(MCXGate(len(objectives) - 1), [*objectives[:-1], target])
        circuit.h(target)

    for local_index, qubit in reversed(list(enumerate(objectives))):
        if good_bitstring[-1 - local_index] == "0":
            circuit.x(qubit)

    return circuit


def _build_zero_reflection(
    num_qubits: int,
    reflection_qubits: list[int] | None = None,
    *,
    mcx_mode: str = "auto",
) -> QuantumCircuit:
    """Build full ``S_0`` reflection over ``reflection_qubits``."""
    _validate_mcx_mode(mcx_mode)
    circuit = QuantumCircuit(int(num_qubits), name="S_0")
    qubits = (
        list(range(int(num_qubits)))
        if reflection_qubits is None
        else [int(qubit) for qubit in reflection_qubits]
    )
    if not qubits:
        raise ValueError("Zero reflection needs at least one qubit.")

    circuit.x(qubits)
    if len(qubits) == 1:
        circuit.z(qubits[0])
    else:
        target = qubits[-1]
        circuit.h(target)
        circuit.append(MCXGate(len(qubits) - 1), [*qubits[:-1], target])
        circuit.h(target)
    circuit.x(qubits)
    return circuit


def _build_custom_full_grover(
    state_preparation: QuantumCircuit,
    objective_qubits: list[int],
    good_bitstring: str = "111",
    *,
    mcx_mode: str = "auto",
) -> QuantumCircuit:
    """Build ``Q = - A S_0 A† S_f`` with full ``S_0``."""
    num_qubits = int(state_preparation.num_qubits)
    circuit = QuantumCircuit(num_qubits, name="Q_custom_full")
    circuit.compose(
        _build_phase_oracle(
            num_qubits,
            objective_qubits,
            good_bitstring,
            mcx_mode=mcx_mode,
        ),
        inplace=True,
    )
    circuit.compose(state_preparation.inverse(), inplace=True)
    circuit.compose(
        _build_zero_reflection(num_qubits, mcx_mode=mcx_mode),
        inplace=True,
    )
    circuit.compose(state_preparation, inplace=True)
    circuit.global_phase += np.pi
    return circuit


def _build_power_query(
    state_preparation: QuantumCircuit,
    grover_operator: QuantumCircuit,
    k: int,
    *,
    flatten_grover: bool,
) -> QuantumCircuit:
    num_qubits = max(
        int(state_preparation.num_qubits),
        int(grover_operator.num_qubits),
    )
    circuit = QuantumCircuit(num_qubits, name=f"A_Q_{int(k)}")
    circuit.compose(state_preparation, inplace=True)
    for _ in range(int(k)):
        if flatten_grover:
            circuit.compose(grover_operator, inplace=True)
        else:
            circuit.compose(grover_operator.power(1), inplace=True)
    return circuit


def _remove_measurements_and_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    clean = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    clean.global_phase = circuit.global_phase
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"measure", "barrier"}:
            continue
        if instruction.clbits:
            raise ValueError(
                f"Cannot clean non-unitary instruction with classical bits: "
                f"{operation.name}."
            )
        q_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        clean.append(operation, q_indices)
    return clean


def _validate_inputs(
    state_preparation: QuantumCircuit,
    objective_qubits: list[int],
    good_bitstring: str,
) -> None:
    if not objective_qubits:
        raise ValueError("objective_qubits must not be empty.")
    if any(
        int(qubit) < 0 or int(qubit) >= state_preparation.num_qubits
        for qubit in objective_qubits
    ):
        raise ValueError("objective_qubits must be valid qubit indices of A.")
    if len(set(int(qubit) for qubit in objective_qubits)) != len(objective_qubits):
        raise ValueError("objective_qubits must not contain duplicates.")
    _validate_good_bitstring(good_bitstring, len(objective_qubits))


def _validate_good_bitstring(good_bitstring: str, width: int) -> None:
    text = str(good_bitstring)
    if len(text) != int(width):
        raise ValueError(
            "good_bitstring length must match objective_qubits length: "
            f"{len(text)} != {int(width)}."
        )
    if any(bit not in {"0", "1"} for bit in text):
        raise ValueError("good_bitstring must contain only '0' and '1'.")


def _validate_mcx_mode(mcx_mode: str) -> None:
    if str(mcx_mode) not in {"auto", "noancilla"}:
        raise ValueError(
            "Only mcx_mode='auto' or 'noancilla' are supported; both use "
            "Qiskit's generic MCXGate so synthesis is left to the transpiler."
        )


def _require_all_ones_good_state(
    good_bitstring: str,
    objective_qubits: list[int],
) -> None:
    expected = "1" * len(objective_qubits)
    if str(good_bitstring) != expected:
        raise ValueError(
            "qiskit_standard candidates use Qiskit's EstimationProblem default "
            f"oracle and only support good_bitstring={expected!r}."
        )


def _normalize_bitstring(value: object, width: int) -> str:
    text = str(value).replace(" ", "").strip()
    if text.startswith(("0x", "0X")):
        return format(int(text, 16), f"0{int(width)}b")
    return text.zfill(int(width))
