# python imports
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

# quantum_cva imports
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)


# Optional fallback if you keep all training state-prep code under multi_asset.
from quantum_cva.multi_asset.quantum.training.state_prep_mps.mps import MLMpsCircuit

from quantum_cva.multi_asset.quantum.training.utilities.mps_training_utils import (
    run_mps_fit,
    save_experiment,
    select_checkpoint_indices,
    serializable_result_dict,
)

# ===================== Global Configuration =====================
BACKEND_NAME = "ibm_basquecountry"

# For an MPS circuit the logical pattern is a nearest-neighbour chain of
# overlapping local windows. A linear layout is usually the correct pre-layout.
LOGICAL_TOPOLOGY = "linear"

BOND_DIM = 2
MPS_CUTOFF = 0.0
EPS_COST = 1e-9
SEED = 42

MPS_METHOD = "gradient"
MPS_INIT = "tt_svd"      # use "random" for a stricter generative-training baseline
MPS_OPTIMIZER = "adam"
MPS_MAXITER = 1000
MPS_LR = 5e-2
MPS_TOL = 1e-10
MPS_MINITER = 25
MPS_INIT_SCALE = 1e-3

OPTIMIZATION_LEVEL = 3
LAYOUT_METHOD = "trivial"
ROUTING_METHOD = "sabre"  # safer than "none" for D>2 because MPS has 3q local unitaries
SEED_TRANSPILER = 1234

CHECKPOINT_EVERY = 25
CHECKPOINT_TOP_K = 20
CHECKPOINT_N_RECENT = 10


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


def main():
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
        / "statevector"
        / f"training_mps_gradient_D{BOND_DIM}_hh6.npz"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = saving_path.with_name(saving_path.stem + "_checkpoints.npz")

    ptg = np.asarray(data["p_target"], dtype=float).ravel()
    ptg = np.clip(ptg, 0.0, None)
    ptg /= float(ptg.sum())

    dim = ptg.size
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("p_target length must be a power of two.")

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

    # ===================== MPS Configuration =====================
    mps = MLMpsCircuit(
        n_qubits=n_qubits,
        bond_dim=BOND_DIM,
        name=f"G_p_mps_D{BOND_DIM}_statevector_transpiled",
        backend=AerSimulator(method="statevector"),
        transpile_backend=real_backend,
        noise_model=None,
        simulation_method="statevector",
        optimization_level=OPTIMIZATION_LEVEL,
        initial_layout=chosen_layout,
        layout_method=LAYOUT_METHOD,
        routing_method=ROUTING_METHOD,
        seed_transpiler=SEED_TRANSPILER,
        build_circuit_on_fit=False,
    )

    target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))

    # ===================== MPS Training =====================
    # The MPS fit is classical and deterministic. It replaces the QCBM's
    # COBYLA + L-BFGS-B stages by one TT-SVD compression of sqrt(p_target).
    print("\nStarting MPS training (classical TT-SVD fit)...")
    mps_run = run_mps_fit(
        ptg,
        mps=mps,
        target_entropy=target_entropy,
        eps=EPS_COST,
        cutoff=MPS_CUTOFF,
        rebuild_circuit=True,
    )

    print(
        f"bond_dim={BOND_DIM} | "
        f"D_eff={mps_run['effective_bond_dim']} | "
        f"D_circ={mps_run['circuit_bond_dim']} | "
        f"CE={mps_run['ce_final']:.6e} | "
        f"KL={mps_run['kl_final']:.6e} | "
        f"trunc_err={mps_run['truncation_error']:.6e} | "
        f"t={mps_run['elapsed_time']:.2f}s"
    )

    summarize_circuit(mps._tqc, label="Transpiled MPS Circuit")
    print("\nMPS circuit summary:")
    print(mps.circuit_summary())

    # ===================== Processing final results =====================
    theta_star = mps_run["theta_star"]
    theta_init = mps_run["theta_init"]
    p0 = mps_run["p_init"]
    p_star = mps_run["p_star"]

    # Explicit naming for MPS training diagnostics.
    # p_train_before is the distribution before fitting the MPS. Since the
    # deterministic MPS fit has no random variational initialization, we use
    # the same uniform baseline as in the QCBM-style diagnostic.
    p_train_before = p0
    p_train_after = p_star
    cost_history = mps_run["cost_history"]
    theta_history = mps_run["theta_history"]
    elapsed_time = mps_run["elapsed_time"]

    print("-" * 30)
    print("success:", True)
    print("message:", f"MPS fitted by {MPS_METHOD} / {MPS_OPTIMIZER}.")
    print("final CE:", mps_run["ce_final"])
    print("final KL:", mps_run["kl_final"])
    print(f"Total elapsed: {elapsed_time:.2f}s")

    # ===================== Metrics & Checkpoints =====================
    checkpoint_idx = select_checkpoint_indices(
        cost_history,
        every=CHECKPOINT_EVERY,
        top_k=CHECKPOINT_TOP_K,
        n_recent=CHECKPOINT_N_RECENT,
    )

    checkpoint_theta = theta_history[checkpoint_idx]
    checkpoint_ce = cost_history[checkpoint_idx]
    checkpoint_kl = checkpoint_ce - target_entropy

    # Two pseudo-stages: 0 = uniform baseline, 1 = fitted MPS.
    checkpoint_stage = checkpoint_idx.astype(int)
    checkpoint_iter_in_stage = checkpoint_idx.astype(int)

    ms = mps.metrics(ptg, p_star, eps=EPS_COST)
    print("\nMetrics summary:")
    print(ms)

    kl_history = np.maximum(cost_history - target_entropy, 1e-15)
    best_so_far = np.minimum.accumulate(kl_history)

    # ===================== Plots =====================
    plot_training_diagnostics_multi_asset(
        target=ptg,
        before=p_train_before,
        after=p_train_after,
        cost_history=kl_history,
        best_so_far=best_so_far,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel="KL evolution",
        title_before="Before MPS fitting (uniform)",
        title_after=f"After MPS fitting (D={BOND_DIM})",
        cost_log_x=True,
        cost_log_y=True,
    )

    fig_before = plot_p_train_vs_target(
        target=ptg,
        p_train=p_train_before,
        title="Before MPS fitting",
        train_label="trained",
    )
    p_train_before_plot_path = saving_path.with_name(f"p_train_before_mps_D{BOND_DIM}.png")
    fig_before.savefig(p_train_before_plot_path, dpi=180, bbox_inches="tight")
    print(f"\n[OK] p_train before plot saved to: {p_train_before_plot_path}")

    fig_after = plot_p_train_vs_target(
        target=ptg,
        p_train=p_train_after,
        title=f"After MPS fitting (D={BOND_DIM})",
        train_label="trained",
    )
    p_train_after_plot_path = saving_path.with_name(f"p_train_after_mps_D{BOND_DIM}.png")
    fig_after.savefig(p_train_after_plot_path, dpi=180, bbox_inches="tight")
    print(f"[OK] p_train after plot saved to: {p_train_after_plot_path}")

    # Backward-compatible aggregate path name. It points to the after-fit plot.
    p_train_plot_path = p_train_after_plot_path
    plt.show()

    # ===================== Save Physical Circuit =====================
    circuit_save_path = saving_path.with_name(f"trained_mps_circuit_D{BOND_DIM}.qpy")
    with open(circuit_save_path, "wb") as f:
        qpy.dump(mps._tqc, f)
    print(f"\n[OK] Transpiled circuit saved to: {circuit_save_path}")

    result_dict = serializable_result_dict(mps_run["result"])
    circuit_summary = mps.circuit_summary()
    resource_estimate = mps.alcazar_resource_estimate()

    # ===================== Results saving =====================
    main_save_data = {
        "theta_star": theta_star,
        "theta_init": theta_init,
        "best_so_far": best_so_far,
        "cost_history": cost_history,
        "theta_history": theta_history,
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
        "p_init": p0,
        "p_star": p_star,
        "p_train_before": p_train_before,
        "p_train_after": p_train_after,
        "p_train_plot_path": np.array(str(p_train_plot_path)),
        "p_train_before_plot_path": np.array(str(p_train_before_plot_path)),
        "p_train_after_plot_path": np.array(str(p_train_after_plot_path)),
        "elapsed_time": np.float64(elapsed_time),
        "n_iters": np.int64(len(cost_history)),
        "theta_seed": np.int64(SEED),
        "n_qubits": np.int64(n_qubits),
        "bond_dim": np.int64(BOND_DIM),
        "effective_bond_dim": np.int64(mps.effective_bond_dim),
        "circuit_bond_dim": np.int64(mps.circuit_bond_dim),
        "d_qubits": np.int64(mps.d_qubits),
        "mps_cutoff": np.float64(MPS_CUTOFF),
        "truncation_error": np.float64(mps_run["truncation_error"]),
        "eps_cost": np.float64(EPS_COST),
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
        "seed": np.int64(SEED),
        "fit_elapsed": np.float64(mps_run["elapsed_time"]),
        "ce_init": np.float64(mps_run["ce_init"]),
        "kl_init": np.float64(mps_run["kl_init"]),
        "ce_final": np.float64(mps_run["ce_final"]),
        "kl_final": np.float64(mps_run["kl_final"]),
        "metrics": np.array(ms, dtype=object),
        "mps_training_result": np.array(result_dict, dtype=object),
        "mps_circuit_summary": np.array(circuit_summary, dtype=object),
        "mps_resource_estimate": np.array(resource_estimate, dtype=object),
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


if __name__ == "__main__":
    main()
