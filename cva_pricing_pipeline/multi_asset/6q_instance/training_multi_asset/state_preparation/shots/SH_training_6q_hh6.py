import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)


# ===================== Global Configuration =====================
BACKEND_NAME = "ibm_basquecountry"
LOGICAL_TOPOLOGY = "qcbm_heavyhex6"

N_LAYERS = 6
EPS_COST = 1e-9
INIT_SCALE = 0.01
SEED = 42
SEED_TRANSPILER = 1234

SHOTS = 5000
N_ITERS = 7000
RESAMPLINGS = 3
DIRICHLET_ALPHA = 1.0

CHECKPOINT_TOL = 1e-15


def main() -> None:
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data = np.load(
        repo_root / "data" / "multi_asset" / "6q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    saving_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "qcbm"
        / "training_qcbm_heavyhex6_shots.npz"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)

    ptg = np.asarray(data["p_target"], dtype=float).ravel()
    ptg /= ptg.sum()

    dim = ptg.size
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("La distribución target no tiene dimensión potencia de 2.")

    target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))

    # ===================== Backend & Layout =====================
    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=LOGICAL_TOPOLOGY,
        length=n_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
    )

    effective_topology = layout_meta["selected_topology"]

    print(f"backend_name       = {BACKEND_NAME}")
    print(f"requested_topology = {LOGICAL_TOPOLOGY}")
    print(f"effective_topology = {effective_topology}")
    print(f"chosen_layout      = {chosen_layout}")
    print(f"layout_score       = {layout_score}")
    print(f"fallback_used      = {layout_meta['fallback_used']}")
    print(f"tried              = {layout_meta['tried']}")

    # ===================== QCBM Configuration =====================
    qcbm = MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=N_LAYERS,
        name="G_p_shots_transpiled",
        entangler="rzz",
        topology=effective_topology,
        backend=AerSimulator(method="automatic"),
        transpile_backend=real_backend,
        noise_model=None,
        simulation_method="automatic",
        optimization_level=3,
        initial_layout=chosen_layout,
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=SEED_TRANSPILER,
    )

    summarize_circuit(qcbm._tqc, label="Transpiled QCBM Circuit")

    # ===================== Cost function =====================
    cost_shots = qcbm.cost_fn(
        ptg,
        eps=EPS_COST,
        shots=SHOTS,
        seed=None,
        rescaled=True,
        smoothing="dirichlet",
        alpha=DIRICHLET_ALPHA,
    )

    # ===================== Initialization =====================
    rng = np.random.default_rng(SEED)
    theta0 = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)

    cost_history: list[float] = []
    theta_history: list[np.ndarray] = [theta0.copy()]
    best = {"fx": float("inf"), "x": theta0.copy()}

    def callback(nfev, x, fx, stepsize, accepted):
        del nfev, stepsize, accepted
        fx = float(fx)
        x_arr = np.asarray(x, dtype=float).copy()
        cost_history.append(fx)
        theta_history.append(x_arr)
        if fx < best["fx"]:
            best["fx"] = fx
            best["x"] = x_arr.copy()

    print("\nCalibrating SPSA hyperparameters...")
    LEARNING_RATE, PERTURBATION = SPSA.calibrate(cost_shots, theta0)

    opt = SPSA(
        maxiter=int(N_ITERS),
        learning_rate=LEARNING_RATE,
        perturbation=PERTURBATION,
        resamplings=RESAMPLINGS,
        blocking=False,
        callback=callback,
        trust_region=True,
        regularization=0.1,
    )

    # ===================== Training =====================
    print("\nStarting shots-based SPSA training...")
    t0 = time.perf_counter()
    res = opt.minimize(fun=cost_shots, x0=theta0)
    elapsed_time = time.perf_counter() - t0

    theta_last = np.asarray(res.x, dtype=float)
    theta_star = best["x"].copy()

    print(f"Training complete. Best cost observed: {best['fx']:.6e}")
    print(f"Elapsed time (s): {elapsed_time:.2f}")
    print("success:", getattr(res, "success", None))
    print("message:", getattr(res, "message", None))

    # ===================== Probabilities & Metrics =====================
    p0 = qcbm.probabilities(theta0, shots=SHOTS, seed=None)
    p_last = qcbm.probabilities(theta_last, shots=SHOTS, seed=None)
    p_star = qcbm.probabilities(theta_star, shots=SHOTS, seed=None)

    cost_history_arr = np.asarray(cost_history, dtype=float)
    theta_history_arr = np.asarray(theta_history, dtype=float)
    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - CHECKPOINT_TOL]
    )
    kl_history = np.maximum(cost_history_arr, 1e-15)

    metrics_best = qcbm.metrics(ptg, p_star, eps=EPS_COST)
    metrics_last = qcbm.metrics(ptg, p_last, eps=EPS_COST)

    print("\nMetrics summary (best iterate):")
    print(metrics_best)

    # ===================== Plots =====================
    plot_training_diagnostics_multi_asset(
        target=ptg,
        before=p0,
        after=p_star,
        cost_history=kl_history,
        best_so_far=best_so_far,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel=(
            f"Rescaled CE"
        ),
        title_before="Before training (shots)",
        title_after="After training (best iterate, shots)",
        cost_log_x=False,
        cost_log_y=True,
    )
    plt.show()

    # ===================== Save circuit =====================
    circuit_save_path = saving_path.with_name("trained_qcbm_circuit_shots.qpy")
    with open(circuit_save_path, "wb") as f:
        qpy.dump(qcbm._tqc, f)
    print(f"\n[OK] Transpiled circuit saved to: {circuit_save_path}")

    # ===================== Save results =====================
    np.savez(
        saving_path,
        theta_star=theta_star,
        theta_last=theta_last,
        theta_init=theta0,
        cost_history=cost_history_arr,
        kl_history=kl_history,
        best_so_far=best_so_far,
        best_idx=best_idx,
        theta_history=theta_history_arr,
        p_target=ptg,
        p_init=p0,
        p_last=p_last,
        p_star=p_star,
        elapsed_time=np.float64(elapsed_time),
        best_cost=np.float64(best["fx"]),
        target_entropy=np.float64(target_entropy),
        n_iters=np.int64(N_ITERS),
        shots=np.int64(SHOTS),
        epsilon=np.float64(EPS_COST),
        theta_seed=np.int64(SEED),
        init_scale=np.float64(INIT_SCALE),
        resamplings=np.int64(RESAMPLINGS),
        dirichlet_alpha=np.float64(DIRICHLET_ALPHA),
        n_qubits=np.int64(n_qubits),
        n_layers=np.int64(N_LAYERS),
        backend_name=np.array(BACKEND_NAME),
        requested_topology=np.array(LOGICAL_TOPOLOGY),
        effective_topology=np.array(effective_topology),
        chosen_layout=np.array(chosen_layout, dtype=int),
        layout_score=np.float64(layout_score),
        fallback_used=np.bool_(layout_meta["fallback_used"]),
        tried_layout_search=np.array(layout_meta["tried"], dtype=object),
        transpiled_depth=np.int64(qcbm._tqc.depth()),
        transpiled_size=np.int64(qcbm._tqc.size()),
        transpiled_ops=np.array(dict(qcbm._tqc.count_ops()), dtype=object),
        metrics_best=np.array(metrics_best, dtype=object),
        metrics_last=np.array(metrics_last, dtype=object),
    )
    print(f"[OK] Results saved to: {saving_path}")


if __name__ == "__main__":
    main()
