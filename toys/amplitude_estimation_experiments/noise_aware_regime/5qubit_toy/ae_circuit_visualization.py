from __future__ import annotations

from math import pi

import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem
from qiskit_ibm_runtime import QiskitRuntimeService


INITIAL_LAYOUT = [90, 91, 98, 111, 110]


def ry_as_rz_rx_rz(qc, theta, qubit):
    qc.rz(-pi / 2, qubit)
    qc.rx(theta, qubit)
    qc.rz(pi / 2, qubit)


def build_state_preparation(offset):
    A = QuantumCircuit(5, name="A")

    ry_as_rz_rx_rz(A, 0.72, 2)
    A.rz(0.23, 2)
    ry_as_rz_rx_rz(A, 0.51, 3)
    A.rz(-0.17, 3)
    ry_as_rz_rx_rz(A, 0.36 + offset, 4)
    A.rz(0.19, 4)

    A.rzz(0.42, 2, 3)
    A.cz(3, 4)

    A.rx(-0.31, 2)
    A.rz(0.12, 2)
    A.rx(0.27, 3)
    A.rz(-0.21, 3)
    A.rx(0.44, 4)
    A.rz(0.08, 4)

    A.rzz(-0.28, 3, 4)

    ry_as_rz_rx_rz(A, offset, 4)

    return A


def build_estimation_problem(offset):
    A = build_state_preparation(offset)

    oracle = QuantumCircuit(5, name="oracle")
    oracle.rz(pi, 4)

    grover_operator = GroverOperator(
        oracle=oracle,
        state_preparation=A,
        reflection_qubits=[2, 3, 4],
        insert_barriers=False,
    )

    return EstimationProblem(
        state_preparation=A,
        grover_operator=grover_operator,
        objective_qubits=[4],
    )


def construct_measured_circuit(problem, k):
    num_qubits = max(
        problem.state_preparation.num_qubits,
        problem.grover_operator.num_qubits,
    )
    circuit = QuantumCircuit(num_qubits, name=f"AE_k_{k}")
    circuit.compose(problem.state_preparation, inplace=True)
    if k > 0:
        circuit.compose(problem.grover_operator.power(k), inplace=True)

    creg = ClassicalRegister(len(problem.objective_qubits), "c0")
    circuit.add_register(creg)
    circuit.measure(problem.objective_qubits, creg[:])
    return circuit


def transpile_and_report(backend, k, offset):
    problem = build_estimation_problem(offset)
    logical = construct_measured_circuit(problem, k)

    state = Statevector.from_instruction(problem.state_preparation)
    probabilities = state.probabilities_dict(qargs=[4])
    a_true = float(probabilities.get("1", 0.0))

    transpiled = transpile(
        logical,
        backend=backend,
        initial_layout=INITIAL_LAYOUT,
        optimization_level=3,
        seed_transpiler=1234,
    )

    print(f"a_true: {a_true}")
    print(f"logical depth: {logical.depth()}")
    print(f"logical gate counts: {dict(logical.count_ops())}")
    print(f"transpiled depth: {transpiled.depth()}")
    print(f"transpiled gate counts: {dict(transpiled.count_ops())}")
    print(
        "number of 2-qubit gates: "
        f"{sum(1 for instruction in transpiled.data if len(instruction.qubits) == 2)}"
    )
    print(f"initial layout usado: {INITIAL_LAYOUT}")

    logical.draw("mpl", fold=-1)
    transpiled.draw("mpl", idle_wires=False, fold=-1)
    plt.show()


if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.backend("ibm_basquecountry")
    transpile_and_report(backend, k=1, offset=0.0)
