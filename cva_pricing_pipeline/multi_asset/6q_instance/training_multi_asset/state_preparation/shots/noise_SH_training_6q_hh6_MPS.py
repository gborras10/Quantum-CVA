from __future__ import annotations

import pathlib
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)

from quantum_cva.multi_asset.quantum.training.state_prep_mps.mps import (
MLMpsCircuit,
)


try:
    from quantum_cva.multi_asset.quantum.training.utilities.mps_training_utils import (
        run_mps_fit,
        save_experiment,
        select_checkpoint_indices,
        serializable_result_dict,
    )
except ModuleNotFoundError:
    from quantum_cva.multi_asset.quantum.training.utilities.mps_training_utils import (
        run_mps_fit,
        save_experiment, 
        select_checkpoint_indices,
        serializable_result_dict,
    )


# ===================== Global Configuration =====================
BACKEND_NAME = "ibm_basquecountry"

# MPS circuits are naturally local-window / nearest-neighbour circuits.
LOGICAL_TOPOLOGY = "linear"

BOND_DIM = 2
MPS_CUTOFF = 0.0
EPS_COST = 1e-9
SEED = 42
SEED_TRANSPILER = 1234
SIMULATOR_SEED = 20260407

# Gradient-MPS training configuration.
# Use init="tt_svd" for stable refinement; use init="random" for a more purely
# variational/generative training run closer in spirit to Alcazar's MPS fitting.
MPS_METHOD = "gradient"
MPS_INIT = "tt_svd"
MPS_OPTIMIZER = "adam"
MPS_MAXITER = 50
MPS_LR = 5e-2
MPS_TOL = 1e-10
MPS_MINITER = 50
MPS_INIT_SCALE = 1e-3

SHOTS = 60000
DIRICHLET_ALPHA = 1.0  # saved for comparability with QCBM shots runs

KL_EVAL_RUNS = 10
KL_EVAL_SHOTS = 100000
KL_EVAL_SEED_BASE = 42

OPTIMIZATION_LEVEL = 3
LAYOUT_METHOD = "trivial"
ROUTING_METHOD = "sabre"  # safer for local multi-qubit MPS unitaries

CHECKPOINT_EVERY = 25
CHECKPOINT_TOP_K = 20
CHECKPOINT_N_RECENT = 10
CHECKPOINT_TOL = 1e-15

# =====================================================================
# Fixed backend-noise snapshot
# ---------------------------------------------------------------------
NOISE_SNAPSHOT_ISO_UTC = "2026-04-07T12:10:00+00:00"


def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
    dt = datetime.fromisoformat(snapshot_iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_property_value(props, qubit: int, name: str):
    try:
        out = props.qubit_property(qubit)
        value = out.get(name, (None, None))[0]
        return None if value is None else float(value)
    except Exception:
        return None


def _cross_entropy(ptg: np.ndarray, p: np.ndarray, *, eps: float) -> float:
    ptg = np.asarray(ptg, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    ptg = np.clip(ptg, 0.0, None)
    p = np.clip(p, 0.0, None)
    ptg /= float(ptg.sum())
    p /= float(p.sum())
    return float(-np.sum(ptg * np.log(np.clip(p, eps, 1.0))))


def _dirichlet_smooth(p: np.ndarray, *, shots: int, alpha: float) -> np.ndarray:
    """Apply the same symmetric Dirichlet/Laplace smoothing idea as the QCBM shots objective."""
    p = np.asarray(p, dtype=float).ravel()
    if alpha <= 0.0:
        out = np.maximum(p, 0.0)
        out /= float(out.sum())
        return out

    dim = p.size
    out = (float(shots) * p + float(alpha)) / (float(shots) + float(alpha) * dim)
    out = np.maximum(out, 0.0)
    out /= float(out.sum())
    return out


def plot_p_train_vs_target(
    *,
    target: np.ndarray,
    p_train: np.ndarray,
    title: str,
    train_label: str = "trained",
):
    """Plot p_train and target with the same bar style as the QCBM diagnostics."""
    target = np.asarray(target, dtype=float).ravel()
    p_train = np.asarray(p_train, dtype=float).ravel()
    if target.shape != p_train.shape:
        raise ValueError(f"Shape mismatch: target{target.shape} vs p_train{p_train.shape}.")

    x = np.arange(target.size)

    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.bar(x, p_train, width=0.82, label=train_label, alpha=0.90)
    ax.bar(x, target, width=0.68, label="target", alpha=0.55)

    ax.set_xlabel("Control basis state |i>")
    ax.set_ylabel("f(i)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")

    if target.size <= 80:
        tick_step = max(1, target.size // 8)
        ticks = list(range(0, target.size, tick_step))
        if ticks[-1] != target.size - 1:
            ticks.append(target.size - 1)
        ax.set_xticks(ticks)

    fig.tight_layout()
    return fig


def main() -> None:
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
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
        / "mps"
        / "shots"
        / (
            f"D{BOND_DIM}_training_mps_{MPS_METHOD}_{MPS_INIT}_"
            "linear_shots_backend_noise_snapshot.npz"
        )
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = saving_path.with_name(saving_path.stem + "_checkpoints.npz")

    ptg = np.asarray(data["p_target"], dtype=float).ravel()
    ptg = np.clip(ptg, 0.0, None)
    ptg /= float(ptg.sum())

    dim = ptg.size
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("Target distribution dimension must be a power of 2.")

    target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))

    # ===================== Backend, historical properties & Layout =====================
    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    backend_props = real_backend.properties(datetime=snapshot_dt_utc)
    if backend_props is None:
        raise RuntimeError(
            "Could not retrieve historical backend properties for timestamp "
            f"{NOISE_SNAPSHOT_ISO_UTC}."
        )

    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=LOGICAL_TOPOLOGY,
        length=n_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
    )

    effective_topology = layout_meta["selected_topology"]

    print(f"backend_name              = {BACKEND_NAME}")
    print(f"requested_topology        = {LOGICAL_TOPOLOGY}")
    print(f"effective_topology        = {effective_topology}")
    print(f"chosen_layout             = {chosen_layout}")
    print(f"layout_score              = {layout_score}")
    print(f"fallback_used             = {layout_meta['fallback_used']}")
    print(f"tried                     = {layout_meta['tried']}")
    print(f"noise_snapshot_utc        = {snapshot_dt_utc.isoformat()}")
    print(f"backend_props_last_update = {getattr(backend_props, 'last_update_date', None)}")

    # ===================== Fixed noise model from backend snapshot =====================
    used_noise_fallback = False
    try:
        noise_model = NoiseModel.from_backend_properties(
            backend_props,
            thermal_relaxation=False,
        )
    except AttributeError:
        used_noise_fallback = True
        print(
            "Noise model could not be created from backend properties. "
            "Falling back to noise model from backend (non-snapshot)."
        )
        noise_model = NoiseModel.from_backend(real_backend)

    coupling_map = getattr(real_backend, "coupling_map", None)
    if coupling_map is None:
        try:
            coupling_map = real_backend.configuration().coupling_map
        except Exception:
            coupling_map = None

    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=coupling_map,
        seed_simulator=SIMULATOR_SEED,
    )

    # ===================== MPS Configuration =====================
    mps = MLMpsCircuit(
        n_qubits=n_qubits,
        bond_dim=BOND_DIM,
        name=f"G_p_mps_D{BOND_DIM}_{MPS_METHOD}_{MPS_INIT}_shots_backend_noise_snapshot",
        backend=noisy_backend,
        transpile_backend=real_backend,
        noise_model=noise_model,
        basis_gates=list(noise_model.basis_gates),
        simulation_method="density_matrix",
        optimization_level=OPTIMIZATION_LEVEL,
        initial_layout=chosen_layout,
        layout_method=LAYOUT_METHOD,
        routing_method=ROUTING_METHOD,
        seed_transpiler=SEED_TRANSPILER,
        build_circuit_on_fit=False,
    )

    # ===================== MPS Training =====================
    # New implementation: the MPS tensors are trained by gradient descent on
    # cross-entropy/KL. TT-SVD is used only as optional initialization and as
    # final canonicalization for the isometry-to-circuit construction.
    print(
        "\nStarting gradient-based MPS training...\n"
        f"method={MPS_METHOD} | init={MPS_INIT} | optimizer={MPS_OPTIMIZER} | "
        f"maxiter={MPS_MAXITER} | lr={MPS_LR}"
    )
    train_t0 = time.perf_counter()
    mps_run = run_mps_fit(
        ptg,
        mps=mps,
        target_entropy=target_entropy,
        eps=EPS_COST,
        cutoff=MPS_CUTOFF,
        rebuild_circuit=True,
        method=MPS_METHOD,
        init=MPS_INIT,
        optimizer=MPS_OPTIMIZER,
        maxiter=MPS_MAXITER,
        lr=MPS_LR,
        tol=MPS_TOL,
        miniter=MPS_MINITER,
        seed=SEED,
        init_scale=MPS_INIT_SCALE,
    )
    fit_elapsed = time.perf_counter() - train_t0

    summarize_circuit(mps._tqc, label="Transpiled Gradient-MPS Circuit")
    print("\nMPS circuit summary:")
    print(mps.circuit_summary())

    result = mps_run["result"]
    print(
        f"bond_dim={BOND_DIM} | "
        f"D_eff={mps_run['effective_bond_dim']} | "
        f"D_circ={mps_run['circuit_bond_dim']} | "
        f"ideal_CE={mps_run['ce_final']:.6e} | "
        f"ideal_KL={mps_run['kl_final']:.6e} | "
        f"trunc_err={mps_run['truncation_error']:.6e} | "
        f"n_iters={getattr(result, 'n_iters', -1)} | "
        f"converged={getattr(result, 'converged', None)} | "
        f"fit_t={fit_elapsed:.2f}s"
    )

    # ===================== KL under frozen backend noise =====================
    print("\nEvaluating KL under noise with fitted gradient-MPS circuit...")
    kl_eval_values: list[float] = []
    p_eval_values: list[np.ndarray] = []
    for run_idx in range(KL_EVAL_RUNS):
        run_seed = KL_EVAL_SEED_BASE + run_idx
        p_mps_noise = mps.probabilities(
            shots=KL_EVAL_SHOTS,
            seed=run_seed,
            use_backend=True,
        )
        p_mps_noise_smooth = _dirichlet_smooth(
            p_mps_noise,
            shots=KL_EVAL_SHOTS,
            alpha=DIRICHLET_ALPHA,
        )
        kl_val = float(mps.metrics(ptg, p_mps_noise_smooth, eps=EPS_COST)["kl"])
        kl_eval_values.append(kl_val)
        p_eval_values.append(p_mps_noise)
        print(
            f"[KL eval {run_idx + 1:02d}/{KL_EVAL_RUNS}] "
            f"seed={run_seed} | KL={kl_val:.6e}"
        )

    kl_eval_values_arr = np.asarray(kl_eval_values, dtype=float)
    kl_eval_mean = float(np.mean(kl_eval_values_arr))
    kl_eval_std = (
        float(np.std(kl_eval_values_arr, ddof=1))
        if kl_eval_values_arr.size > 1
        else 0.0
    )
    print(f"KL mean (gradient-MPS under noisy model): {kl_eval_mean:.6e}")
    print(f"KL std  (gradient-MPS under noisy model): {kl_eval_std:.6e}")

    # ===================== Probabilities & Metrics =====================
    p_train_before = np.asarray(mps_run["p_init"], dtype=float).ravel()
    p_init = p_train_before

    p_ideal_after = mps.probabilities(use_backend=False)
    p_star_raw = mps.probabilities(shots=SHOTS, seed=None, use_backend=True)
    p_star = _dirichlet_smooth(p_star_raw, shots=SHOTS, alpha=DIRICHLET_ALPHA)
    p_last = p_star.copy()
    p_train_after = p_star

    ce_init = _cross_entropy(ptg, p_init, eps=EPS_COST)
    kl_init = float(ce_init - target_entropy)
    ce_after_noise = _cross_entropy(ptg, p_star, eps=EPS_COST)
    kl_after_noise = float(ce_after_noise - target_entropy)
    ce_ideal_after = _cross_entropy(ptg, p_ideal_after, eps=EPS_COST)
    kl_ideal_after = float(ce_ideal_after - target_entropy)

    cost_history_arr = np.asarray(mps_run["cost_history"], dtype=float).ravel()
    kl_history = np.asarray(mps_run["kl_history"], dtype=float).ravel()
    kl_history = np.maximum(kl_history, 1e-15)
    best_so_far = np.minimum.accumulate(kl_history)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - CHECKPOINT_TOL]
    )

    theta_init = np.asarray(mps_run["theta_init"], dtype=float)
    theta_star = np.asarray(mps_run["theta_star"], dtype=float)
    theta_last = theta_star.copy()
    theta_history_arr = np.asarray(mps_run["theta_history"], dtype=float)

    metrics_best = mps.metrics(ptg, p_star, eps=EPS_COST)
    metrics_last = mps.metrics(ptg, p_last, eps=EPS_COST)
    metrics_ideal = mps.metrics(ptg, p_ideal_after, eps=EPS_COST)

    print("\nMetrics summary (fitted gradient-MPS under noise):")
    print(metrics_best)
    print("\nMetrics summary (ideal fitted gradient-MPS, no backend noise):")
    print(metrics_ideal)

    # ===================== Checkpoints =====================
    checkpoint_idx = select_checkpoint_indices(
        cost_history_arr,
        every=CHECKPOINT_EVERY,
        top_k=CHECKPOINT_TOP_K,
        n_recent=CHECKPOINT_N_RECENT,
    )
    checkpoint_theta = theta_history_arr[checkpoint_idx]
    checkpoint_ce = cost_history_arr[checkpoint_idx]
    checkpoint_kl = checkpoint_ce - target_entropy
    checkpoint_stage = checkpoint_idx.astype(int)
    checkpoint_iter_in_stage = checkpoint_idx.astype(int)

    elapsed_time = float(fit_elapsed)

    # ===================== Plots =====================
    plot_training_diagnostics_multi_asset(
        target=ptg,
        before=p_train_before,
        after=p_train_after,
        cost_history=kl_history,
        best_so_far=best_so_far,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel="Ideal KL during gradient-MPS training",
        title_before=f"Before MPS training ({MPS_INIT} baseline)",
        title_after=f"After gradient-MPS training (D={BOND_DIM}, shots + backend noise)",
        cost_log_x=False,
        cost_log_y=True,
    )

    fig_before = plot_p_train_vs_target(
        target=ptg,
        p_train=p_train_before,
        title=f"Before MPS training ({MPS_INIT} baseline)",
        train_label="trained",
    )
    p_train_before_plot_path = saving_path.with_name(
        f"p_train_before_mps_D{BOND_DIM}_{MPS_METHOD}_{MPS_INIT}_noise.png"
    )
    fig_before.savefig(p_train_before_plot_path, dpi=180, bbox_inches="tight")
    print(f"\n[OK] p_train before plot saved to: {p_train_before_plot_path}")

    fig_after = plot_p_train_vs_target(
        target=ptg,
        p_train=p_train_after,
        title=f"After gradient-MPS training (D={BOND_DIM}, shots + noise)",
        train_label="trained",
    )
    p_train_after_plot_path = saving_path.with_name(
        f"p_train_after_mps_D{BOND_DIM}_{MPS_METHOD}_{MPS_INIT}_noise.png"
    )
    fig_after.savefig(p_train_after_plot_path, dpi=180, bbox_inches="tight")
    print(f"[OK] p_train after plot saved to: {p_train_after_plot_path}")

    plt.show()

    # ===================== Save circuit =====================
    circuit_save_path = saving_path.with_name(
        f"trained_mps_circuit_D{BOND_DIM}_{MPS_METHOD}_{MPS_INIT}_shots_backend_noise_snapshot.qpy"
    )
    with open(circuit_save_path, "wb") as f:
        qpy.dump(mps._tqc, f)
    print(f"\n[OK] Transpiled circuit saved to: {circuit_save_path}")

    # ===================== Save results =====================
    t1_values = [_safe_property_value(backend_props, q, "T1") for q in chosen_layout]
    t2_values = [_safe_property_value(backend_props, q, "T2") for q in chosen_layout]
    readout_values = [
        _safe_property_value(backend_props, q, "readout_error") for q in chosen_layout
    ]

    result_dict = serializable_result_dict(mps_run["result"])
    circuit_summary = mps.circuit_summary()
    resource_estimate = mps.alcazar_resource_estimate()

    main_save_data = {
        "theta_star": theta_star,
        "theta_last": theta_last,
        "theta_init": theta_init,
        "theta_history": theta_history_arr,
        "cost_history": cost_history_arr,
        "kl_history": kl_history,
        "best_so_far": best_so_far,
        "best_idx": best_idx,
        "checkpoint_idx": checkpoint_idx,
        "checkpoint_theta": checkpoint_theta,
        "checkpoint_ce": checkpoint_ce,
        "checkpoint_kl": checkpoint_kl,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_iter_in_stage": checkpoint_iter_in_stage,
        "checkpoint_every": np.int64(CHECKPOINT_EVERY),
        "checkpoint_top_k": np.int64(CHECKPOINT_TOP_K),
        "checkpoint_n_recent": np.int64(CHECKPOINT_N_RECENT),
        "p_target": ptg,
        "p_init": p_init,
        "p_last": p_last,
        "p_star": p_star,
        "p_star_raw": p_star_raw,
        "p_ideal_after": p_ideal_after,
        "p_train_before": p_train_before,
        "p_train_after": p_train_after,
        "p_train_before_plot_path": np.array(str(p_train_before_plot_path)),
        "p_train_after_plot_path": np.array(str(p_train_after_plot_path)),
        "elapsed_time": np.float64(elapsed_time),
        "fit_elapsed": np.float64(fit_elapsed),
        "best_cost": np.float64(kl_after_noise),
        "target_entropy": np.float64(target_entropy),
        "ce_init": np.float64(ce_init),
        "kl_init": np.float64(kl_init),
        "ce_ideal_after": np.float64(ce_ideal_after),
        "kl_ideal_after": np.float64(kl_ideal_after),
        "ce_after_noise": np.float64(ce_after_noise),
        "kl_after_noise": np.float64(kl_after_noise),
        "ideal_ce_final": np.float64(mps_run["ce_final"]),
        "ideal_kl_final": np.float64(mps_run["kl_final"]),
        "kl_eval_values_mps_noise": kl_eval_values_arr,
        "kl_eval_mean_mps_noise": np.float64(kl_eval_mean),
        "kl_eval_std_mps_noise": np.float64(kl_eval_std),
        "kl_eval_runs": np.int64(KL_EVAL_RUNS),
        "kl_eval_shots": np.int64(KL_EVAL_SHOTS),
        "kl_eval_seed_base": np.int64(KL_EVAL_SEED_BASE),
        "n_iters": np.int64(len(cost_history_arr)),
        "shots": np.int64(SHOTS),
        "epsilon": np.float64(EPS_COST),
        "theta_seed": np.int64(SEED),
        "dirichlet_alpha": np.float64(DIRICHLET_ALPHA),
        "n_qubits": np.int64(n_qubits),
        "bond_dim": np.int64(BOND_DIM),
        "effective_bond_dim": np.int64(mps.effective_bond_dim),
        "circuit_bond_dim": np.int64(mps.circuit_bond_dim),
        "d_qubits": np.int64(mps.d_qubits),
        "mps_method": np.array(MPS_METHOD),
        "mps_init": np.array(MPS_INIT),
        "mps_optimizer": np.array(MPS_OPTIMIZER),
        "mps_maxiter": np.int64(MPS_MAXITER),
        "mps_lr": np.float64(MPS_LR),
        "mps_tol": np.float64(MPS_TOL),
        "mps_miniter": np.int64(MPS_MINITER),
        "mps_init_scale": np.float64(MPS_INIT_SCALE),
        "mps_cutoff": np.float64(MPS_CUTOFF),
        "truncation_error": np.float64(mps_run["truncation_error"]),
        "backend_name": np.array(BACKEND_NAME),
        "requested_topology": np.array(LOGICAL_TOPOLOGY),
        "effective_topology": np.array(effective_topology),
        "chosen_layout": np.array(chosen_layout, dtype=int),
        "layout_score": np.float64(layout_score),
        "fallback_used": np.bool_(layout_meta["fallback_used"]),
        "tried_layout_search": np.array(layout_meta["tried"], dtype=object),
        "transpiled_depth": np.int64(mps._tqc.depth()),
        "transpiled_size": np.int64(mps._tqc.size()),
        "transpiled_ops": np.array(dict(mps._tqc.count_ops()), dtype=object),
        "optimization_level": np.int64(OPTIMIZATION_LEVEL),
        "layout_method": np.array(LAYOUT_METHOD),
        "routing_method": np.array(ROUTING_METHOD),
        "seed_transpiler": np.int64(SEED_TRANSPILER),
        "metrics_best": np.array(metrics_best, dtype=object),
        "metrics_last": np.array(metrics_last, dtype=object),
        "metrics_ideal": np.array(metrics_ideal, dtype=object),
        "mps_training_result": np.array(result_dict, dtype=object),
        "mps_circuit_summary": np.array(circuit_summary, dtype=object),
        "mps_resource_estimate": np.array(resource_estimate, dtype=object),
        "noise_snapshot_iso_utc": np.array(snapshot_dt_utc.isoformat()),
        "backend_props_last_update": np.array(str(getattr(backend_props, "last_update_date", None))),
        "simulator_seed": np.int64(SIMULATOR_SEED),
        "simulator_method": np.array("density_matrix"),
        "noise_basis_gates": np.array(list(noise_model.basis_gates), dtype=object),
        "used_noise_fallback": np.bool_(used_noise_fallback),
        "snapshot_t1_chosen_layout": np.array(t1_values, dtype=object),
        "snapshot_t2_chosen_layout": np.array(t2_values, dtype=object),
        "snapshot_readout_error_chosen_layout": np.array(readout_values, dtype=object),
    }

    checkpoint_save_data = {
        "checkpoint_idx": checkpoint_idx,
        "checkpoint_theta": checkpoint_theta,
        "checkpoint_ce": checkpoint_ce,
        "checkpoint_kl": checkpoint_kl,
        "checkpoint_stage": checkpoint_stage,
        "checkpoint_iter_in_stage": checkpoint_iter_in_stage,
        "theta_star": theta_star,
        "p_target": ptg,
        "p_train_before": p_train_before,
        "p_train_after": p_train_after,
        "target_entropy": np.float64(target_entropy),
    }

    save_experiment(saving_path, checkpoint_path, main_save_data, checkpoint_save_data)
    print(f"[OK] Results saved to: {saving_path}")


if __name__ == "__main__":
    main()
