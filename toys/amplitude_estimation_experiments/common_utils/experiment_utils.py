from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem

from quantum_cva.algorithms.proposed_algorithms.cabiae_known_t import CABIQAE
from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE       
from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
    StandaloneBAEHardware as StandaloneBAE,
)


AMPLITUDE_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
CANONICAL_TOY_DIR = AMPLITUDE_EXPERIMENTS_DIR / "noise_aware_regime" / "3qubit_toy"
for path in (CANONICAL_TOY_DIR, AMPLITUDE_EXPERIMENTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from ae_circuit_utils import (  # noqa: E402
    CANONICAL_OBJECTIVE_RY_OFFSET,
    OBJECTIVE_QUBITS,
    build_problem_with_true_amplitude,
)


ALGORITHM_LABELS = {
    "biqae": "BIQAE",
    "iqae": "IQAE",
    "cabiqae": "CABIQAE",
    "cabiqaev2": "CABIQAEv2",
    "cabiaelatent": "CABIQAE-LTheta",
    "cabiqae_latentt": "CABIQAE_latentt",
    "bae": "BAE",
}

def build_problem(
    objective_ry_offset: float = CANONICAL_OBJECTIVE_RY_OFFSET,
) -> EstimationProblem:
    problem, _ = build_problem_with_true_amplitude(float(objective_ry_offset))
    return problem


def true_amplitude_for_offset(
    objective_ry_offset: float = CANONICAL_OBJECTIVE_RY_OFFSET,
) -> float:
    _, a_true = build_problem_with_true_amplitude(float(objective_ry_offset))
    return float(a_true)


def canonical_true_amplitude() -> float:
    a_true = true_amplitude_for_offset(CANONICAL_OBJECTIVE_RY_OFFSET)
    return float(a_true)


def infer_grover_power(circuit: QuantumCircuit) -> int:
    metadata = circuit.metadata or {}
    if "grover_power" in metadata:
        return int(metadata["grover_power"])

    count_ops = circuit.count_ops()
    q_count = int(count_ops.get("Q", 0))
    if q_count > 0:
        return q_count

    ry_count = int(count_ops.get("ry", 0))
    return max(0, ry_count - 1)


def contrast_decay(k: int, T: float | None) -> float:
    if T is None or np.isinf(T):
        return 1.0
    return float(np.exp(-(2.0 * float(k) + 1.0) / float(T)))


def apply_contrast_decay(p_ideal: float, k: int, T: float | None) -> float:
    contrast = contrast_decay(k, T)
    return float(np.clip(0.5 + contrast * (p_ideal - 0.5), 0.0, 1.0))


def _remove_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    clean = QuantumCircuit(circuit.num_qubits)
    for instruction, qargs, cargs in circuit.data:
        if instruction.name in {"measure", "barrier"}:
            continue
        clean.append(instruction, qargs, cargs)
    return clean


def ideal_good_probability(circuit: QuantumCircuit) -> float:
    clean = _remove_measurements(circuit)
    state = Statevector.from_instruction(clean)
    qargs = list(OBJECTIVE_QUBITS)
    if max(qargs) >= clean.num_qubits:
        qargs = [clean.num_qubits - 1]
    probs = state.probabilities_dict(qargs=qargs)
    return float(probs.get("1", 0.0))


class _FakeCounts:
    def __init__(self, counts: dict[str, int]):
        self._counts = dict(counts)

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeData:
    def __init__(self, counts: dict[str, int]):
        self.c0 = _FakeCounts(counts)


class _FakePubResult:
    def __init__(self, counts: dict[str, int]):
        self.data = _FakeData(counts)


class _FakeSamplerJob:
    def __init__(self, pub_results: list[_FakePubResult]):
        self._pub_results = pub_results

    def result(self) -> list[_FakePubResult]:
        return self._pub_results


class ContrastDecaySampler:
    """Minimal SamplerV2-compatible wrapper for one-case noise experiments."""

    def __init__(self, T: float | None = None, seed: int | None = None):
        self._T = T
        self._rng = np.random.default_rng(seed)

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _FakeSamplerJob:
        pub_results: list[_FakePubResult] = []

        for circuit in circuits:
            k = infer_grover_power(circuit)
            p_ideal = ideal_good_probability(circuit)
            p_obs = apply_contrast_decay(p_ideal, k, self._T)
            one_counts = int(self._rng.binomial(shots, p_obs))
            counts = {"0": int(shots - one_counts), "1": one_counts}
            pub_results.append(_FakePubResult(counts))

        return _FakeSamplerJob(pub_results)


def _build_solver(
    algorithm: str,
    epsilon: float,
    alpha: float,
    sampler: Any,
    T: float | None,
    cap_kappa: float,
    cabiqae_scheduler_mode: str,
    max_shots_same_k: int | None,
) -> tuple[Any, bool]:
    if algorithm == "biqae":
        return (
            BIQAE(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=sampler,
                min_ratio=2,
                confint_method="beta",
            ),
            True,
        )

    if algorithm == "iqae":
        return (
            BIQAE(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=sampler,
                min_ratio=2,
                confint_method="beta",
            ),
            False,
        )

    if algorithm == "cabiqae":
        return (
            CABIQAE(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=sampler,
                min_ratio=2,
                confint_method="beta",
                noise_model="ideal" if T is None or np.isinf(T) else "exponential_contrast",
                T_known=None if T is None or np.isinf(T) else float(T),
                cap_kappa=cap_kappa,
                use_noise_cap=True,
                max_shots_same_k=max_shots_same_k,
            ),
            True,
        )

    if algorithm in ("cabiaelatent", "cabiqae_latentt"):
        return (
            CABIQAELatentTheta(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=sampler,
                min_ratio=2,
                confint_method="beta",
                noise_model="ideal" if T is None or np.isinf(T) else "exponential_contrast",
                T_known=None if T is None or np.isinf(T) else float(T),
                cap_kappa=cap_kappa,
                use_noise_cap=True,
                max_shots_same_k=max_shots_same_k,
            ),
            True,
        )

    if algorithm == "bae":
        return (
            StandaloneBAE(
                epsilon_target=epsilon,
                alpha=alpha,
                sampler=sampler,
                noise_model="ideal" if T is None or np.isinf(T) else "exponential_contrast",
                T_known=None if T is None or np.isinf(T) else float(T),
                cap_kappa=cap_kappa,
                max_shots_same_k=max_shots_same_k,
            ),
            True,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_single_case(
    problem: EstimationProblem,
    algorithm: str,
    epsilon: float,
    alpha: float,
    sampler: Any,
    T: float | None,
    cap_kappa: float = 2.2,
    n_shots: int = 10,
    cabiqae_scheduler_mode: str = "legacy",
    max_shots_same_k: int | None = None,
) -> dict[str, float | int]:
    solver, bayes = _build_solver(
        algorithm,
        epsilon,
        alpha,
        sampler,
        T,
        cap_kappa,
        cabiqae_scheduler_mode,
        max_shots_same_k,
    )

    t0 = time.perf_counter()
    result = solver.estimate(problem, bayes=bayes, show_details=False, n_shots=n_shots)
    elapsed = time.perf_counter() - t0

    cost = getattr(result, "num_state_prep_calls", None)
    if cost is None:
        cost = getattr(result, "num_oracle_queries")

    ci_low, ci_high = (float(x) for x in result.confidence_interval)
    powers = [int(k) for k in (getattr(result, "powers", None) or [])]
    circuit_depths = [int(d) for d in (getattr(result, "circuit_depths", None) or [])]

    return {
        "estimation": float(result.estimation),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "cost": int(cost),
        "iterations": int(len(powers)),
        "k_max": int(max(powers) if powers else 0),
        "max_depth": int(max(circuit_depths) if circuit_depths else 0),
        "elapsed_sec": float(elapsed),
    }


def format_T_value(T: float | None) -> str:
    return "inf" if T is None or np.isinf(T) else f"{float(T):g}"
