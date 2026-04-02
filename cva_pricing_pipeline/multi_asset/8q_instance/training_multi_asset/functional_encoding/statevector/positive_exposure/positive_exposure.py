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
SEARCH_TOPOLOGY = "crca_heavyhex8"
TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 2
N_PRICE = 6
N_ANCILLAS = 11
N_CONTROLS = M_TIME + N_PRICE
N_TOTAL = N_CONTROLS + N_ANCILLAS
N_LAYERS = 1

THETA_SEED = 42
INIT_SCALE = 0.2

# Support-aware loss
TARGET_THRESHOLD = 1e-10
RELATIVE_EPS = 1e-4
LAMBDA_POS = 50.0
LAMBDA_ZERO = 1.0

# COBYLA
COBYLA_MAXITER = 300
COBYLA_TOL = 1e-6
COBYLA_RHOBEG = 0.5


def main() -> None:
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "8q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    v_t = benchmark["v_joint_t"]
    c_v = float(benchmark["C_v"])

    # 2D target only for reference if needed
    f_target_2d = np.asarray(v_t / c_v, dtype=float)

    # 1D target exactly matching crca.function_values(...)
    f_target = f_target_2d.reshape(-1)

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "8q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "positive_exposure"
        / "positive_exposure_native_heavyhex8_statevector.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_heavyhex8",
        native_1q_order=("rx", "rz"),
        name="crca_positive_exposure_native_heavyhex8",
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
    theta = INIT_SCALE * rng.standard_normal(crca.n_params)
    theta_init = theta.copy()

    f0_statevector = np.asarray(crca.function_values(theta, shots=None), dtype=float).reshape(-1)

    pos_mask = f_target > TARGET_THRESHOLD
    zero_mask = ~pos_mask

    n_pos = int(np.count_nonzero(pos_mask))
    n_zero = int(np.count_nonzero(zero_mask))

    print(f"Positive-support bins = {n_pos}")
    print(f"Zero-support bins     = {n_zero}")
    print(
        "Using support-aware loss with "
        f"lambda_pos={LAMBDA_POS:.1f}, lambda_zero={LAMBDA_ZERO:.1f}, "
        f"relative_eps={RELATIVE_EPS:.1e}"
    )

    def support_aware_cost(x: np.ndarray) -> float:
        fx = np.asarray(
            crca.function_values(np.asarray(x, dtype=float), shots=None),
            dtype=float,
        ).reshape(-1)

        pos_term = 0.0
        zero_term = 0.0

        if n_pos > 0:
            rel_diff = (fx[pos_mask] - f_target[pos_mask]) / (f_target[pos_mask] + RELATIVE_EPS)
            pos_term = float(np.mean(rel_diff * rel_diff))

        if n_zero > 0:
            zero_term = float(np.mean(fx[zero_mask] * fx[zero_mask]))

        return LAMBDA_POS * pos_term + LAMBDA_ZERO * zero_term

    eval_cost_history: list[float] = []
    iter_cost_history: list[float] = []
    eval_counter = 0

    best_loss = float("inf")
    best_theta = theta.copy()

    def eval_cost(x: np.ndarray) -> float:
        nonlocal eval_counter, best_loss, best_theta

        x = np.asarray(x, dtype=float)
        fx = float(support_aware_cost(x))

        eval_counter += 1
        eval_cost_history.append(fx)

        if fx < best_loss:
            best_loss = fx
            best_theta = x.copy()

        print(f"[eval {eval_counter:04d}][COBYLA] cost = {fx:.8e}")
        return fx

    def record_iter(xk: np.ndarray) -> None:
        xk = np.asarray(xk, dtype=float)
        fx = float(support_aware_cost(xk))
        iter_cost_history.append(fx)
        print(f"[iter {len(iter_cost_history)-1:04d}][COBYLA] iter_cost = {fx:.8e}")

    initial_loss = float(support_aware_cost(theta))
    iter_cost_history.append(initial_loss)
    best_loss = initial_loss
    best_theta = theta.copy()

    print(f"Initial support-aware cost = {initial_loss:.8e}")
    print("Starting optimization: COBYLA only")

    t0 = time.perf_counter()

    res_cobyla = minimize(
        eval_cost,
        x0=theta,
        method="COBYLA",
        callback=record_iter,
        options={
            "maxiter": COBYLA_MAXITER,
            "tol": COBYLA_TOL,
            "rhobeg": COBYLA_RHOBEG,
            "disp": False,
        },
    )

    elapsed = time.perf_counter() - t0

    theta_last = np.asarray(res_cobyla.x, dtype=float)
    final_loss = float(support_aware_cost(theta_last))
    iter_cost_history.append(final_loss)

    if final_loss < best_loss:
        best_loss = final_loss
        best_theta = theta_last.copy()

    theta_best = best_theta.copy()
    f_star_statevector = np.asarray(crca.function_values(theta_best, shots=None), dtype=float).reshape(-1)

    cost_history_arr = np.array(iter_cost_history, dtype=float)
    eval_cost_history_arr = np.array(eval_cost_history, dtype=float)

    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(np.r_[True, best_so_far[1:] < best_so_far[:-1]])

    print(f"COBYLA finished | success = {res_cobyla.success} | message = {res_cobyla.message}")
    print(f"COBYLA final support-aware cost = {final_loss:.8e}")
    print(f"Best support-aware cost observed = {best_loss:.8e}")

    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0_statevector,
        after=f_star_statevector,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel="Support-aware loss",
        title_before="Before training",
        title_after="After training",
        cost_log_x=False,
        cost_log_y=True,
    )

    plt.show()

    metadata = {
        "model": "CRCA",
        "task": "positive_exposure",
        "ancilla_observable": "P(a=1 | control=i)",
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_ancillas": N_ANCILLAS,
        "n_controls": crca.n_controls,
        "n_layers": N_LAYERS,
        "n_parameters": crca.n_params,
        "optimizer": "COBYLA",
        "optimizer_library": "scipy",
        "init_scale": INIT_SCALE,
        "cobyla_maxiter": COBYLA_MAXITER,
        "cobyla_tol": COBYLA_TOL,
        "cobyla_rhobeg": COBYLA_RHOBEG,
        "loss_name": "support_aware_relative_plus_zero_penalty",
        "target_threshold": TARGET_THRESHOLD,
        "relative_eps": RELATIVE_EPS,
        "lambda_pos": LAMBDA_POS,
        "lambda_zero": LAMBDA_ZERO,
        "n_positive_support_bins": n_pos,
        "n_zero_support_bins": n_zero,
        "shots": None,
        "stochastic_cost": False,
        "best_eval_cost_observed": float(np.min(eval_cost_history_arr)) if eval_cost_history_arr.size else float(initial_loss),
        "best_iter_cost_observed": float(best_loss),
        "backend_name": BACKEND_NAME,
        "requested_topology": SEARCH_TOPOLOGY,
        "layout_score": float(layout_score),
        "fallback_used": bool(layout_meta["fallback_used"]),
        "transpile_optimization_level": TRANSPILATION_OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "theta_seed": THETA_SEED,
        "result_success": bool(res_cobyla.success),
        "result_message": str(res_cobyla.message),
        "result_nfev": int(res_cobyla.nfev),
        "note": (
            "CRCA positive exposure training via COBYLA on a support-aware loss. "
            "theta_star = best parameters observed during optimization."
        ),
    }

    np.savez(
        out_path,
        theta_star=theta_best,
        theta_last=theta_last,
        theta_init=theta_init,
        cost_history=cost_history_arr,
        eval_cost_history=eval_cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        f_target=f_target,
        f_target_2d=f_target_2d,
        f_init_statevector=f0_statevector,
        f_star_statevector=f_star_statevector,
        elapsed_time=np.float64(elapsed),
        best_cost=np.float64(best_loss),
        final_cost=np.float64(final_loss),
        C_v=np.float64(c_v),
        n_iters=np.int64(len(iter_cost_history) - 1),
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
        result_success=np.bool_(res_cobyla.success),
        result_nfev=np.int64(res_cobyla.nfev),
        metadata=np.array(metadata, dtype=object),
    )

    print("Training finished")
    print("Elapsed time:", elapsed)
    print("Total evaluations:", eval_counter)
    print("COBYLA nfev:", res_cobyla.nfev)
    print("Best iterate cost:", best_loss)


if __name__ == "__main__":
    main()
