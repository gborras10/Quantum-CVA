import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)

# ===================== Global Configuration =====================

BACKEND_NAME = "ibm_basquecountry"
SEARCH_TOPOLOGY = "crca_heavyhex10"
TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 4
N_PRICE = 6
N_ANCILLAS = 15
N_CONTROLS = M_TIME + N_PRICE
N_TOTAL = N_CONTROLS + N_ANCILLAS
N_LAYERS = 1

THETA_SEED = 42


MAXITER = 30
LEARNING_RATE = 0.05
PERTURBATION = 0.02


def main():

    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    v_t = benchmark["v_joint_t"]
    c_v = float(benchmark["C_v"])
    f_target = v_t / c_v

    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_heavyhex10",
        native_1q_order=("rx", "rz"),
        name="crca_positive_exposure_native_heavyhex10",
    )

    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=SEARCH_TOPOLOGY,
        length=crca.qc.num_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
    )

    tqc_ansatz = transpile(
        crca.qc,
        backend=real_backend,
        initial_layout=chosen_layout,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=SEED_TRANSPILER,
    )

    tqc_eval = transpile(
        crca.qc_eval,
        backend=real_backend,
        initial_layout=chosen_layout,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=SEED_TRANSPILER,
    )

    summarize_circuit(tqc_ansatz)
    summarize_circuit(tqc_eval)

    rng = np.random.default_rng(THETA_SEED)

    # IMPORTANTE: escala mayor rompe barren plateau inicial

    theta = 2.0 * rng.standard_normal(crca.n_params)

    f0_statevector = crca.function_values(theta, shots=None)

    cost_fn = crca.cost_fn(f_target, shots=None)

    cost_history = []

    eval_counter = 0

    def eval_cost(x):

        nonlocal eval_counter

        fx = float(cost_fn(x))

        eval_counter += 1

        cost_history.append(fx)

        print(f"[eval {eval_counter:04d}] cost = {fx:.8e}")

        return fx

    # ===================== SPSA =====================

    t0 = time.perf_counter()

    for k in range(MAXITER):

        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        theta_plus = theta + PERTURBATION * delta
        theta_minus = theta - PERTURBATION * delta

        loss_plus = eval_cost(theta_plus)
        loss_minus = eval_cost(theta_minus)

        grad = (loss_plus - loss_minus) / (2 * PERTURBATION) * delta

        theta = theta - LEARNING_RATE * grad

        print(f"[iter {k+1:03d}] approx grad step done")

    elapsed = time.perf_counter() - t0

    theta_best = theta.copy()

    f_star_statevector = crca.function_values(theta_best, shots=None)

    cost_history_arr = np.array(cost_history)

    best_so_far = np.minimum.accumulate(cost_history_arr)

    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1]]
    )

    time_labels = [format(i, f"0{M_TIME}b") for i in range(2**M_TIME)]

    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0_statevector,
        after=f_star_statevector,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        labels=time_labels,
        xlabel="Time register |t>",
        ylabel="f(t)",
        cost_ylabel="L2 loss (SPSA statevector)",
        title_before="Before training",
        title_after="After training",
        cost_log_x=False,
        cost_log_y=True,
    )

    plt.show()

    print("Training finished")
    print("Elapsed time:", elapsed)
    print("Total evaluations:", eval_counter)


if __name__ == "__main__":
    main()