from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem

from quantum_cva.amplitude_estimation.experiments.circuits import (
    construct_measured_circuit,
)
from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.problems import (
    bundle_from_problem,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (
    FastIdealAmplificationSampler,
    count_good_from_counts,
    extract_result_counts,
)
from quantum_cva.amplitude_estimation.grover import (
    build_grover_candidate,
    circuit_metrics,
    compare_statevectors_up_to_global_phase,
    good_probability_from_statevector,
    validate_grover_amplification,
)
from quantum_cva.amplitude_estimation.grover import candidates as grover_candidates


K_VALUES = [0, 1, 2, 3, 4]


def _synthetic_state_preparation() -> QuantumCircuit:
    circuit = QuantumCircuit(3, name="synthetic_A")
    circuit.ry(0.72, 0)
    circuit.ry(0.51, 1)
    circuit.ry(0.65, 2)
    circuit.cx(0, 2)
    circuit.ry(-0.21, 2)
    circuit.cx(0, 2)
    return circuit


def _synthetic_bundle(good_bitstring: str = "111"):
    state_preparation = _synthetic_state_preparation()
    objective_qubits = [0, 1, 2]
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=objective_qubits,
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "")
        == str(good_bitstring),
    )
    if good_bitstring == "111":
        problem.grover_operator = problem.grover_operator
    return bundle_from_problem(
        problem,
        target_name="synthetic",
        good_bitstring=good_bitstring,
    )


def _load_real_cva_bundle_or_skip():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "cva_pricing_pipeline"
        / "multi_asset"
        / "6q_instance"
        / "full_cva_pipeline.py"
    )
    if not config_path.exists():
        pytest.skip("6q CVA config is not available.")
    spec = importlib.util.spec_from_file_location("full_cva_pipeline_6q", config_path)
    if spec is None or spec.loader is None:
        pytest.skip("6q CVA config cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return build_6q_cva_problem_bundle(module.CONFIG, repo_root=repo_root)
    except FileNotFoundError as exc:
        pytest.skip(f"6q CVA artifacts are not available: {exc}")
    except Exception as exc:
        pytest.skip(f"6q CVA bundle cannot be built: {exc}")


def _query_circuit(
    state_preparation: QuantumCircuit,
    grover_operator: QuantumCircuit,
    k: int,
) -> QuantumCircuit:
    circuit = QuantumCircuit(state_preparation.num_qubits)
    circuit.compose(state_preparation, inplace=True)
    if int(k) > 0:
        circuit.compose(grover_operator.power(int(k)), inplace=True)
    return circuit


def test_clean_state_preparation_preserves_statevector() -> None:
    original = _synthetic_state_preparation()
    measured = original.copy()
    measured.barrier()
    creg = ClassicalRegister(1, "c")
    measured.add_register(creg)
    measured.measure(0, creg[0])

    clean = grover_candidates._clean_state_preparation(measured)

    assert compare_statevectors_up_to_global_phase(original, clean)
    assert "measure" not in clean.count_ops()
    assert "barrier" not in clean.count_ops()


def test_real_cva_initial_good_probability_matches_bundle_true_amplitude() -> None:
    bundle = _load_real_cva_bundle_or_skip()

    probability = good_probability_from_statevector(
        bundle.problem.state_preparation,
        list(bundle.problem.objective_qubits),
        "111",
    )

    assert np.isclose(probability, bundle.true_amplitude, atol=1e-10)


@pytest.mark.parametrize(
    "candidate",
    ["qiskit_standard", "qiskit_standard_clean", "custom_full"],
)
def test_grover_amplification_formula(candidate: str) -> None:
    bundle = _synthetic_bundle()
    state_preparation = bundle.problem.state_preparation
    if candidate == "qiskit_standard_clean":
        state_for_validation = grover_candidates._clean_state_preparation(
            state_preparation
        )
    else:
        state_for_validation = state_preparation

    grover = build_grover_candidate(
        state_preparation,
        list(bundle.problem.objective_qubits),
        "111",
        candidate=candidate,
    )

    result = validate_grover_amplification(
        state_for_validation,
        grover,
        list(bundle.problem.objective_qubits),
        bundle.true_amplitude,
        K_VALUES,
        "111",
        atol=1e-8,
    )

    assert result["passed"], result["rows"]


def test_custom_full_matches_qiskit_standard() -> None:
    bundle = _synthetic_bundle()
    state_preparation = bundle.problem.state_preparation
    objective_qubits = list(bundle.problem.objective_qubits)
    standard = build_grover_candidate(
        state_preparation,
        objective_qubits,
        "111",
        candidate="qiskit_standard",
    )
    custom = build_grover_candidate(
        state_preparation,
        objective_qubits,
        "111",
        candidate="custom_full",
    )

    for k in K_VALUES:
        standard_query = _query_circuit(state_preparation, standard, k)
        custom_query = _query_circuit(state_preparation, custom, k)
        assert compare_statevectors_up_to_global_phase(
            custom_query,
            standard_query,
            atol=1e-8,
        )


def test_good_bitstring_101_oracle_endianness() -> None:
    objective_qubits = [0, 1, 2]
    oracle = grover_candidates._build_phase_oracle(
        3,
        objective_qubits,
        "101",
    )

    for basis_index in range(8):
        basis = QuantumCircuit(3)
        for qubit in range(3):
            if (basis_index >> qubit) & 1:
                basis.x(qubit)

        prepared = Statevector.from_instruction(basis)
        key = next(iter(prepared.probabilities_dict(qargs=objective_qubits)))
        basis.compose(oracle, inplace=True)
        state = Statevector.from_instruction(basis).data
        phase = state[basis_index]

        if str(key) == "101":
            assert np.isclose(phase, -1.0)
        else:
            assert np.isclose(phase, 1.0)


def test_candidate_metrics_are_reported() -> None:
    bundle = _synthetic_bundle()
    candidates = [
        "qiskit_standard",
        "qiskit_standard_clean",
        "custom_full",
        "custom_full_flat",
    ]
    for candidate in candidates:
        grover = build_grover_candidate(
            bundle.problem.state_preparation,
            list(bundle.problem.objective_qubits),
            "111",
            candidate=candidate,
        )
        metrics = circuit_metrics(grover)
        assert metrics["depth"] >= 0
        assert metrics["size"] > 0
        assert isinstance(metrics["ops"], dict)
        assert "two_qubit_gates" in metrics
        assert "multi_qubit_gates" in metrics


def test_custom_full_estimation_problem_compatibility() -> None:
    bundle = _synthetic_bundle()
    problem = EstimationProblem(
        state_preparation=bundle.problem.state_preparation,
        objective_qubits=list(bundle.problem.objective_qubits),
        is_good_state=bundle.problem.is_good_state,
        post_processing=lambda amplitude: 10.0 * float(amplitude),
    )
    custom = build_grover_candidate(
        problem.state_preparation,
        list(problem.objective_qubits),
        "111",
        candidate="custom_full",
    )
    problem.grover_operator = custom
    custom_bundle = bundle_from_problem(
        problem,
        target_name="synthetic",
        good_bitstring="111",
    )

    measured = construct_measured_circuit(problem, 2)
    assert problem.objective_qubits == [0, 1, 2]
    assert problem.is_good_state("111")
    assert measured.num_clbits == 3
    assert problem.grover_operator.power(2).num_qubits == custom.num_qubits
    assert np.isclose(custom_bundle.process(0.2), 2.0)

    sampler = FastIdealAmplificationSampler(custom_bundle, T=None, seed=123)
    counts = extract_result_counts(sampler.run([measured], shots=17).result(), 0)
    assert sum(counts.values()) == 17
    assert count_good_from_counts(counts, custom_bundle) >= 0
