from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem

from quantum_cva.algorithms.third_party.standalone_bae_hardware import StandaloneBAEHardware


class _FakeCountsBin:
    def __init__(self, counts: dict[str, int]):
        self._counts = dict(counts)

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeData:
    def __init__(self, counts: dict[str, int]):
        self.c0 = _FakeCountsBin(counts)


class _FakePubResult:
    def __init__(self, counts: dict[str, int]):
        self.data = _FakeData(counts)


class _FakeSamplerJob:
    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def result(self) -> list[_FakePubResult]:
        return [_FakePubResult(self._counts)]


class TrackingCountsSampler:
    """Sampler stub returning measured counts and recording control usage."""

    def __init__(self, base_probability: float = 0.18):
        self.base_probability = float(base_probability)
        self.controls: list[int] = []
        self.calls: int = 0

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _FakeSamplerJob:
        circuit = circuits[0]
        meta = circuit.metadata or {}
        control = int(meta.get("grover_power", 0))

        # Deterministic, control-dependent counts for reproducible smoke testing.
        p1 = min(0.95, self.base_probability + 0.02 * (control % 7))
        ones = int(round(shots * p1))
        counts = {"0": int(shots - ones), "1": int(ones)}

        self.controls.append(control)
        self.calls += 1
        return _FakeSamplerJob(counts)


class BernoulliA(QuantumCircuit):
    def __init__(self, probability: float):
        super().__init__(1)
        self.ry(2.0 * np.arcsin(np.sqrt(probability)), 0)


class BernoulliQ(QuantumCircuit):
    def __init__(self, probability: float):
        super().__init__(1)
        self._theta_p = float(2.0 * np.arcsin(np.sqrt(probability)))

    def power(self, k: int) -> QuantumCircuit:
        q_k = QuantumCircuit(1, name=f"Q^{k}")
        q_k.metadata = {"grover_power": int(k)}
        for _ in range(max(0, int(k))):
            q_k.ry(2.0 * self._theta_p, 0)
        return q_k


def build_problem(a_true: float) -> EstimationProblem:
    return EstimationProblem(
        state_preparation=BernoulliA(a_true),
        grover_operator=BernoulliQ(a_true),
        objective_qubits=[0],
    )


def run_smoke_test() -> None:
    # Deliberately high true amplitude; the sampler counts are low and should dominate.
    problem = build_problem(a_true=0.93)
    sampler = TrackingCountsSampler(base_probability=0.15)

    solver = StandaloneBAEHardware(
        epsilon_target=1e-3,
        alpha=0.05,
        sampler=sampler,
        noise_model="ideal",
        T_known=None,
        wNs=20,
        Ns=5,
        TNs=0,
        Npart=400,
        Nevals=25,
    )

    # Guardrail: fail fast if some hidden statevector shortcut is attempted.
    original_from_instruction = Statevector.from_instruction

    def _forbidden_statevector(*args, **kwargs):
        raise RuntimeError("Statevector usage is forbidden in hardware-ready BAE smoke test.")

    try:
        Statevector.from_instruction = classmethod(_forbidden_statevector)
        result = solver.estimate(problem, n_shots=5, max_queries=250)
    finally:
        Statevector.from_instruction = original_from_instruction

    history = result.history
    queries = history.get("queries", [])
    estimations = history.get("estimations", [])

    assert sampler.calls > 0, "The external sampler was never called."
    assert len(queries) > 0, "BAE hardware wrapper returned an empty query trajectory."
    assert len(estimations) > 0, "BAE hardware wrapper returned an empty estimation trajectory."
    assert len(queries) == len(estimations), "queries/estimations history length mismatch."
    assert len(history.get("controls", [])) == len(queries), "Missing control history."
    assert len(history.get("K_sequence", [])) == len(queries), "Missing K-sequence history."

    # If hidden a_true shortcuts were used, estimate would drift close to 0.93.
    assert float(result.estimation) < 0.8, "Estimate appears decoupled from measured counts."

    if result.K_sequence:
        assert result.K_max == max(result.K_sequence), "K_max is inconsistent with K_sequence."

    print("StandaloneBAEHardware smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
