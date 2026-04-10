import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import select_best_layout, summarize_circuit

from exposure_utils import build_support_aware_cost

# ===================== Global Configuration =====================
BACKEND_NAME = "ibm_basquecountry"
LOGICAL_TOPOLOGY = "crca_comb"
TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 2
N_PRICE = 6
N_QUBITS = M_TIME + N_PRICE
N_LAYERS = 4
THETA_SEED = 42
INIT_SCALE = 1.0

LOSS_MODE = "support_aware"  
TARGET_THRESHOLD = 1e-10
RELATIVE_EPS = 1e-4  
LAMBDA_POS = 15.0
LAMBDA_ZERO = 100.0

# Stage 1: COBYLA
STAGE1_MAXITER = 3000
STAGE1_TOL = 1e-6
STAGE1_RHOBEG = 0.30

# Stage 2: L-BFGS-B
STAGE2_MAXITER = 10000
STAGE2_MAXFUN = 500000
STAGE2_FTOL = 1e-12
STAGE2_GTOL = 1e-10 

def main() -> None:
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data = np.load(
        repo_root / "data" / "multi_asset" / "8q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    saving_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "8q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "positive_exposure"
        / "positive_exposure_native_comb_logical_statevector.npz"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)

    v_t = data["v_joint_t"]
    c_v = float(data["C_v"])
    f_target_2d = np.asarray(v_t / c_v, dtype=float)
    f_target = f_target_2d.reshape(-1)

    # ===================== CRCA Configuration =====================
    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_comb",
        name="crca_positive_exposure_native_tv_logical",
    )

    # =================== Backend & Layout ====================
    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=LOGICAL_TOPOLOGY,
        length=N_QUBITS,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
    )

    effective_topology = layout_meta["selected_topology"]

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
    theta = INIT_SCALE * rng.standard_normal(crca.n_params)
    theta_init = theta.copy()

    f0_statevector = np.asarray(crca.function_values(theta, shots=None), dtype=float).reshape(-1)

    if LOSS_MODE == "l2":
        objective = crca.cost_fn(f_target, shots=None)
        cost_label = "L2 loss"
        metadata_loss = {
            "loss_name": "l2",
        }
        print("Using original L2 loss from CrcaCircuit.cost_fn")
    elif LOSS_MODE == "support_aware":
        objective, pos_mask, zero_mask = build_support_aware_cost(
            TARGET_THRESHOLD, 
            RELATIVE_EPS, 
            LAMBDA_POS, 
            LAMBDA_ZERO, 
            crca, 
            f_target)
        cost_label = "Support-aware loss"
        metadata_loss = {
            "loss_name": "support_aware_relative_plus_zero_penalty",
            "target_threshold": TARGET_THRESHOLD,
            "relative_eps": RELATIVE_EPS,
            "lambda_pos": LAMBDA_POS,
            "lambda_zero": LAMBDA_ZERO,
            "n_positive_support_bins": int(np.count_nonzero(pos_mask)),
            "n_zero_support_bins": int(np.count_nonzero(zero_mask)),
        }
        print(
            "Using support-aware loss with "
            f"lambda_pos={LAMBDA_POS:.1f}, lambda_zero={LAMBDA_ZERO:.1f}, "
            f"relative_eps={RELATIVE_EPS:.1e}"
        )
    else:
        raise ValueError("LOSS_MODE must be 'l2' or 'support_aware'.")

    eval_cost_history: list[float] = []
    eval_counter = 0

    best_loss = float("inf")
    best_theta = theta.copy()

    def eval_cost(x: np.ndarray, stage: str) -> float:
        nonlocal eval_counter, best_loss, best_theta

        x = np.asarray(x, dtype=float)
        fx = float(objective(x))

        eval_counter += 1
        eval_cost_history.append(fx)

        if fx < best_loss:
            best_loss = fx
            best_theta = x.copy()

        print(f"[eval {eval_counter:04d}][{stage}] cost = {fx:.8e}")
        return fx

    def record_iter(xk: np.ndarray, history: list[float], stage: str, offset: int = 0) -> None:
        xk = np.asarray(xk, dtype=float)
        fx = float(objective(xk))
        history.append(fx)
        print(f"[iter {offset + len(history)-1:04d}][{stage}] iter_cost = {fx:.8e}")

    initial_loss = float(objective(theta))
    stage1_iter_cost_history: list[float] = [initial_loss]
    stage2_iter_cost_history: list[float] = []
    best_loss = initial_loss
    best_theta = theta.copy()

    print(f"Initial {cost_label.lower()} = {initial_loss:.8e}")
    print("Starting Stage 1 (Global optimization with COBYLA)")

    t0 = time.perf_counter()

    res_stage1 = minimize(
        lambda x: eval_cost(x, "COBYLA"),
        x0=theta,
        method="COBYLA",
        callback=lambda xk: record_iter(xk, stage1_iter_cost_history, "COBYLA"),
        options={
            "maxiter": STAGE1_MAXITER,
            "tol": STAGE1_TOL,
            "rhobeg": STAGE1_RHOBEG,
            "disp": False,
        },
    )

    theta_stage1_last = np.asarray(res_stage1.x, dtype=float)
    stage1_final_loss = float(objective(theta_stage1_last))
    stage1_iter_cost_history.append(stage1_final_loss)

    if stage1_final_loss < best_loss:
        best_loss = stage1_final_loss
        best_theta = theta_stage1_last.copy()

    print(f"Stage 1 finished | success = {res_stage1.success} | message = {res_stage1.message}")
    print(f"Stage 1 final {cost_label.lower()} = {stage1_final_loss:.8e}")

    theta_stage2_init = best_theta.copy()
    stage2_init_loss = float(objective(theta_stage2_init))
    stage2_iter_cost_history.append(stage2_init_loss)

    print("Starting Stage 2 (Refinement with L-BFGS-B)")

    stage1_effective_len = len(stage1_iter_cost_history)
    res_stage2 = minimize(
        lambda x: eval_cost(x, "L-BFGS-B"),
        x0=theta_stage2_init,
        method="L-BFGS-B",
        callback=lambda xk: record_iter(
            xk,
            stage2_iter_cost_history,
            "L-BFGS-B",
            offset=stage1_effective_len - 1,
        ),
        options={
            "maxiter": STAGE2_MAXITER,
            "maxfun": STAGE2_MAXFUN,
            "ftol": STAGE2_FTOL,
            "gtol": STAGE2_GTOL,
            "disp": False,
        },
    )

    elapsed = time.perf_counter() - t0

    theta_last = np.asarray(res_stage2.x, dtype=float)
    final_loss = float(objective(theta_last))
    stage2_iter_cost_history.append(final_loss)

    if final_loss < best_loss:
        best_loss = final_loss
        best_theta = theta_last.copy()

    theta_best = best_theta.copy()
    f_star_statevector = np.asarray(crca.function_values(theta_best, shots=None), dtype=float).reshape(-1)

    stage1_cost_history_arr = np.array(stage1_iter_cost_history, dtype=float)
    stage2_cost_history_arr = np.array(stage2_iter_cost_history, dtype=float)

    # Avoid duplicate point if stage2 starts exactly at the stage1 endpoint.
    if np.isclose(stage1_cost_history_arr[-1], stage2_cost_history_arr[0], rtol=1e-12, atol=1e-15):
        cost_history_arr = np.r_[stage1_cost_history_arr, stage2_cost_history_arr[1:]]
    else:
        cost_history_arr = np.r_[stage1_cost_history_arr, stage2_cost_history_arr]

    eval_cost_history_arr = np.array(eval_cost_history, dtype=float)

    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(np.r_[True, best_so_far[1:] < best_so_far[:-1]])

    print(f"Stage 2 finished | success = {res_stage2.success} | message = {res_stage2.message}")
    print(f"Stage 2 final {cost_label.lower()} = {final_loss:.8e}")
    print(f"Best {cost_label.lower()} observed = {best_loss:.8e}")

    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0_statevector,
        after=f_star_statevector,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel=cost_label,
        title_before="Before training",
        title_after="After training",
        cost_log_x=False,
        cost_log_y=True,
    )

    plt.show()

    metadata = {
        "model": "CRCA",
        "task": "positive_exposure",
        "ansatz_type": "native_tv",
        "training_mode": "logical_untranspiled_statevector",
        "ancilla_observable": "P(a=1 | control=i)",
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_layers": N_LAYERS,
        "n_controls": crca.n_controls,
        "n_parameters": crca.n_params,
        "optimizer": "2-stage (COBYLA -> L-BFGS-B)",
        "optimizer_library": "scipy",
        "init_scale": INIT_SCALE,
        "stage1_optimizer": "COBYLA",
        "stage1_maxiter": STAGE1_MAXITER,
        "stage1_tol": STAGE1_TOL,
        "stage1_rhobeg": STAGE1_RHOBEG,
        "stage2_optimizer": "L-BFGS-B",
        "stage2_maxiter": STAGE2_MAXITER,
        "stage2_maxfun": STAGE2_MAXFUN,
        "stage2_ftol": STAGE2_FTOL,
        "stage2_gtol": STAGE2_GTOL,
        "shots": None,
        "stochastic_cost": False,
        "theta_seed": THETA_SEED,
        "stage1_success": bool(res_stage1.success),
        "stage1_message": str(res_stage1.message),
        "stage1_nfev": int(res_stage1.nfev),
        "stage2_success": bool(res_stage2.success),
        "stage2_message": str(res_stage2.message),
        "stage2_nfev": int(res_stage2.nfev),
        "result_success": bool(res_stage2.success),
        "result_message": str(res_stage2.message),
        "result_nfev": int(res_stage2.nfev),
        "best_eval_cost_observed": float(np.min(eval_cost_history_arr)) if eval_cost_history_arr.size else float(initial_loss),
        "best_iter_cost_observed": float(best_loss),
        "note": (
            "CRCA positive exposure training with native_tv logical ansatz, "
            "no transpilation, exact statevector. "
            "2-stage optimization: COBYLA (global search) + L-BFGS-B (refinement)."
        ),
    }
    metadata.update(metadata_loss)

    np.savez(
        saving_path,
        theta_star=theta_best,
        theta_last=theta_last,
        theta_init=theta_init,
        cost_history=cost_history_arr,
        stage1_cost_history=stage1_cost_history_arr,
        stage2_cost_history=stage2_cost_history_arr,
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
        n_iters=np.int64(len(cost_history_arr) - 1),
        theta_seed=np.int64(THETA_SEED),
        stage1_success=np.bool_(res_stage1.success),
        stage1_nfev=np.int64(res_stage1.nfev),
        stage2_success=np.bool_(res_stage2.success),
        stage2_nfev=np.int64(res_stage2.nfev),
        metadata=np.array(metadata, dtype=object),
    )

    print("Training finished")
    print("Elapsed time:", elapsed)
    print("Total evaluations:", eval_counter)
    print("Stage 1 (COBYLA) nfev:", res_stage1.nfev)
    print("Stage 2 (L-BFGS-B) nfev:", res_stage2.nfev)
    print("Best iterate cost:", best_loss)


if __name__ == "__main__":
    main()
