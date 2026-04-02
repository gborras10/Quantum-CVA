import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize

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
SEARCH_TOPOLOGY = "time_tree4"
TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 4
N_PRICE = 0
N_LAYERS = 1

THETA_SEED = 42
N_ITERS = 150
RHOBEG = 0.40


def main() -> None:
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )
    q_t: np.ndarray = benchmark["q_t"]
    c_q: float = float(benchmark["C_q"])
    f_target: np.ndarray = q_t / c_q

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "quantum"
        / "training"
        / "crca"
        / "default_probabilities_time_tree4_statevector.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ===================== CRCA Configuration =====================
    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_probabilities_time_tree4",
    )

    # ===================== Backend & Layout =====================
    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    n_logical_qubits = crca.qc.num_qubits
    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=SEARCH_TOPOLOGY,
        length=n_logical_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
    )

    print("backend_name      =", BACKEND_NAME)
    print("search_topology   =", SEARCH_TOPOLOGY)
    print("chosen_layout     =", chosen_layout)
    print("layout_score      =", layout_score)
    print("fallback_used     =", layout_meta["fallback_used"])
    print("tried             =", layout_meta["tried"])

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

    summarize_circuit(tqc_ansatz, label="CRCA ansatz transpiled for hardware")
    summarize_circuit(tqc_eval, label="CRCA eval transpiled for hardware")

    # ===================== Training =====================
    rng = np.random.default_rng(THETA_SEED)
    x0: np.ndarray = 0.5 * rng.standard_normal(crca.n_params).astype(float)
    f0_statevector: np.ndarray = crca.function_values(x0, shots=None, seed=None)

    cost_statevector = crca.cost_fn(
        f_target,
        shots=None,
        seed=None,
    )

    cost_history: list[float] = []
    theta_history: list[np.ndarray] = []

    t0: float = time.perf_counter()
    res = minimize(
        fun=cost_statevector,
        x0=x0,
        method="COBYLA",
        options={
            "maxiter": int(N_ITERS),
            "rhobeg": float(RHOBEG),
            "disp": False,
        },
        callback=lambda x: (
            cost_history.append(float(cost_statevector(x))),
            theta_history.append(np.asarray(x, dtype=float).copy()),
        ),
    )
    elapsed_time: float = time.perf_counter() - t0

    cost_history_arr = np.asarray(cost_history, dtype=float)
    if cost_history_arr.size == 0:
        cost_history_arr = np.asarray([float(res.fun)], dtype=float)
        theta_best = np.asarray(res.x, dtype=float).copy()
        best_fx = float(res.fun)
    else:
        best_pos = int(np.argmin(cost_history_arr))
        theta_best = theta_history[best_pos].copy()
        best_fx = float(cost_history_arr[best_pos])

    print(f"Training complete in {elapsed_time:.1f} s")
    print(f"Best L2 cost observed: {best_fx:.8f}")

    theta_last: np.ndarray = np.asarray(res.x, dtype=float)
    f_star_statevector: np.ndarray = crca.function_values(theta_best, shots=None, seed=None)

    # ===================== Plots =====================
    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
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
        cost_ylabel=f"L2 loss  (COBYLA, rhobeg={RHOBEG:.4g}, statevector)",
        title_before="Before training  (CRCA default probability, COBYLA statevector)",
        title_after="After training  (best-iter, CRCA default probability, COBYLA statevector)",
        cost_log_x=False,
        cost_log_y=True,
    )
    plt.show()

    # ===================== Results Saving =====================
    metadata = {
        "model": "CRCA",
        "task": "default_probability",
        "ancilla_observable": "P(a=1 | control=i)",
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_controls": crca.n_controls,
        "n_layers": N_LAYERS,
        "n_parameters": crca.n_params,
        "optimizer": "COBYLA",
        "optimizer_library": "scipy",
        "rhobeg": RHOBEG,
        "maxiter": N_ITERS,
        "cost_function": "L2",
        "shots": None,
        "stochastic_cost": False,
        "shot_seed": None,
        "best_iter_cost_observed": best_fx,
        "stopping_criterion": "maxiter",
        "backend_name": BACKEND_NAME,
        "requested_topology": SEARCH_TOPOLOGY,
        "layout_score": float(layout_score),
        "fallback_used": bool(layout_meta["fallback_used"]),
        "transpile_optimization_level": TRANSPILATION_OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "note": (
            "CRCA default-probability training via statevector-only COBYLA. "
            "theta_star = best-iteration parameters."
        ),
    }

    np.savez(
        out_path,
        theta_star=theta_best,
        theta_last=theta_last,
        theta_init=x0,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        f_target=f_target,
        f_init_statevector=f0_statevector,
        f_star_statevector=f_star_statevector,
        elapsed_time=np.float64(elapsed_time),
        best_cost=np.float64(best_fx),
        C_q=np.float64(c_q),
        n_iters=np.int64(N_ITERS),
        rhobeg=np.float64(RHOBEG),
        theta_seed=np.int64(THETA_SEED),
        backend_name=np.array(BACKEND_NAME),
        requested_topology=np.array(SEARCH_TOPOLOGY),
        chosen_layout=np.array(chosen_layout, dtype=int),
        layout_score=np.float64(layout_score),
        fallback_used=np.bool_(layout_meta["fallback_used"]),
        tried_layout_search=np.array(layout_meta["tried"], dtype=object),
        transpiled_ansatz_depth=np.int64(tqc_ansatz.depth()),
        transpiled_ansatz_size=np.int64(tqc_ansatz.size()),
        transpiled_ansatz_ops=np.array(dict(tqc_ansatz.count_ops()), dtype=object),
        transpiled_eval_depth=np.int64(tqc_eval.depth()),
        transpiled_eval_size=np.int64(tqc_eval.size()),
        transpiled_eval_ops=np.array(dict(tqc_eval.count_ops()), dtype=object),
        transpile_optimization_level=np.int64(TRANSPILATION_OPT_LEVEL),
        seed_transpiler=np.int64(SEED_TRANSPILER),
        metadata=np.array(metadata, dtype=object),
    )


if __name__ == "__main__":
    main()
