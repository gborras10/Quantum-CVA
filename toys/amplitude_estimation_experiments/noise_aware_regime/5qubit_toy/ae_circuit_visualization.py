from __future__ import annotations

import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService

from ae_pipeline_utils import (
    PHYSICAL_BACKEND_NAME,
    build_estimation_problem,
    choose_transpilation_plan,
    construct_measured_circuit,
    true_amplitude,
)


def transpile_and_report(backend, k, offset):
    problem = build_estimation_problem(offset)
    logical = construct_measured_circuit(problem, k)

    a_true = true_amplitude(problem)

    plan = choose_transpilation_plan(backend, problem, reference_ks=(0, 1, 2, 4, 8))
    transpiled = plan.build_pass_manager(backend).run(logical)

    print(f"a_true: {a_true}")
    print(f"logical depth: {logical.depth()}")
    print(f"logical gate counts: {dict(logical.count_ops())}")
    print(f"transpiled depth: {transpiled.depth()}")
    print(f"transpiled gate counts: {dict(transpiled.count_ops())}")
    print(
        "number of 2-qubit gates: "
        f"{sum(1 for instruction in transpiled.data if len(instruction.qubits) == 2)}"
    )
    print(f"initial layout usado: {list(plan.initial_layout)}")
    print(f"layout source: {plan.candidate_source}")
    print(f"seed transpiler usado: {plan.seed_transpiler}")

    logical.draw("mpl", fold=-1)
    transpiled.draw("mpl", idle_wires=False, fold=-1)
    plt.show()


if __name__ == "__main__":
    service = QiskitRuntimeService()
    backend = service.backend(PHYSICAL_BACKEND_NAME)
    transpile_and_report(backend, k=1, offset=0.0)
