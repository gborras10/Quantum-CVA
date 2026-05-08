from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import EstimationProblem

from quantum_cva.amplitude_estimation.experiments.problems import bundle_from_problem


def build_demo_problem_bundle() -> object:
    """Small three-objective-qubit problem used by runner defaults and tests."""
    state_preparation = QuantumCircuit(3, name="demo_A")
    state_preparation.ry(0.72, 0)
    state_preparation.ry(0.51, 1)
    state_preparation.ry(0.65, 2)
    state_preparation.cx(0, 2)
    state_preparation.ry(-0.21, 2)
    state_preparation.cx(0, 2)
    state_preparation.cx(1, 2)
    state_preparation.ry(0.18, 2)
    state_preparation.cx(1, 2)

    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=[0, 1, 2],
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "") == "111",
        post_processing=lambda amplitude: float(amplitude),
    )
    problem.grover_operator = problem.grover_operator
    return bundle_from_problem(
        problem,
        target_name="amplitude",
        good_bitstring="111",
        metadata={"builder": "demo_3_objective_qubits"},
    )
