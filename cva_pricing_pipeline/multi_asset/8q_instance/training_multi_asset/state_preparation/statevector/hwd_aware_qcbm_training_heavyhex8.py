# python imports
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

# quantum_cva imports
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import plot_training_diagnostics_multi_asset
from quantum_cva.quantum_hardware_utilities.layout_utils import select_best_layout, summarize_circuit
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from qcbm_training_utils import (
    run_stage1,
    run_stage2,
    select_checkpoint_indices,
    plot_training_dynamics,
    save_experiment
)

# ===================== Configuraciones Globales =====================
BACKEND_NAME = "ibm_basquecountry"
LOGICAL_TOPOLOGY = "qcbm_heavyhex8"  

N_LAYERS = 8
EPS_COST = 1e-12
INIT_SCALE = 1.0
SEED = 42

STAGE1_MAXITER = 1000
STAGE1_RHOBEG = 0.25
STAGE2_MAXITER = 10000
STAGE2_MAXFUN = 100000
    
CHECKPOINT_EVERY = 25
CHECKPOINT_TOP_K = 20
CHECKPOINT_N_RECENT = 10

def main():
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data = np.load(
        repo_root / "data" / "multi_asset"/ "8q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    saving_path = (
        repo_root 
        / "data" 
        / "multi_asset" 
        / "8q_instance"
        / "quantum" 
        / "training" 
        / "qcbm" 
        / "best_8q_10lay_heavyhex8"
        / "8q_sv_BEST_heavyhex8_rzz.npz"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = saving_path.with_name(saving_path.stem + "_checkpoints.npz")

    ptg = np.asarray(data["p_target"], dtype=float).ravel()
    ptg /= ptg.sum()

    dim = ptg.size
    n_qubits = int(np.log2(dim))

    # ===================== Backend & Layout ====================
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
        name="G_p_statevector_transpiled",
        entangler="rzz",
        topology=effective_topology,
        backend=AerSimulator(method="statevector"),
        transpile_backend=real_backend,
        noise_model=None,
        simulation_method="statevector",
        optimization_level=3,
        initial_layout=chosen_layout,
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=1234,
    )

    summarize_circuit(qcbm._tqc, label="Transpiled QCBM Circuit")

    cost_statevector = qcbm.cost_fn(ptg, eps=EPS_COST)
    target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))

    # =====================  Stage 1 Training =====================
    rng = np.random.default_rng(SEED)
    x0 = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)

    print("\nStarting Stage 1 (Global Optimization with COBYLA)...")
    best_run = run_stage1(
        x0, maxiter=STAGE1_MAXITER, rhobeg=STAGE1_RHOBEG,
        cost_fn=cost_statevector, qcbm=qcbm, target_entropy=target_entropy,
    )
    best_run.update({"theta_init": x0, "seed": SEED})

    print(f"seed={SEED} | CE={best_run['ce_final']:.6e} | KL={best_run['kl_final']:.6e} | t={best_run['elapsed_time']:.2f}s")

    # =====================  Stage 2 Training =====================
    print("\nStarting Stage 2 (Refinement with L-BFGS-B)...")
    refine_run = run_stage2(
        best_run["theta_star"], maxiter=STAGE2_MAXITER, cost_fn=cost_statevector,
        qcbm=qcbm, target_entropy=target_entropy, maxfun=STAGE2_MAXFUN,
    )

    # Procesing final results
    theta_star = refine_run["theta_star"]
    p_star = refine_run["p_star"]
    p0 = qcbm.probabilities(best_run["theta_init"])

    # Avoid duplicating the last point of stage 1 if it's the same as the first point of stage 2
    cost_history = np.r_[best_run["cost_history"], refine_run["cost_history"][1:]]
    theta_history = np.vstack([best_run["theta_history"], refine_run["theta_history"][1:]])
    elapsed_time = best_run["elapsed_time"] + refine_run["elapsed_time"]

    print("-" * 30)
    print("success:", refine_run["result"].success)
    print("message:", refine_run["result"].message)
    print("final CE:", refine_run["ce_final"])
    print("final KL:", refine_run["kl_final"])
    print(f"Total elapsed: {elapsed_time:.2f}s")

    # ===================== Metrics & Checkpoints =====================
    checkpoint_idx = select_checkpoint_indices(
        cost_history, every=CHECKPOINT_EVERY, top_k=CHECKPOINT_TOP_K, n_recent=CHECKPOINT_N_RECENT,
    )

    checkpoint_theta = theta_history[checkpoint_idx]
    checkpoint_ce = cost_history[checkpoint_idx]
    checkpoint_kl = checkpoint_ce - target_entropy

    n_stage1 = len(best_run["cost_history"])
    checkpoint_stage = np.where(checkpoint_idx < n_stage1, 1, 2).astype(int)
    checkpoint_iter_in_stage = np.where(checkpoint_stage == 1, checkpoint_idx, checkpoint_idx - n_stage1 + 1).astype(int)

    ms = qcbm.metrics(ptg, p_star, eps=EPS_COST)
    print("\nMetrics summary:")
    print(ms)

    kl_history = np.maximum(cost_history - target_entropy, 1e-15)
    best_so_far = np.minimum.accumulate(kl_history)

    # ===================== PLots =====================
    plot_training_diagnostics_multi_asset(
        target=ptg,
        before=p0,
        after=p_star,
        cost_history=kl_history,
        best_so_far=best_so_far,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel="KL evolution",
        title_before="Before training",
        title_after="After training",
        cost_log_x=False,
        cost_log_y=True,
    )
    plt.show()

    # ===================== Results saving =====================
    main_save_data = {
        "theta_star": theta_star, "theta_init": best_run["theta_init"],
        "best_so_far": np.minimum.accumulate(np.maximum(cost_history - target_entropy, 1e-15)),
        "cost_history": cost_history, "theta_history": theta_history,
        "checkpoint_idx": checkpoint_idx, "checkpoint_theta": checkpoint_theta,
        "checkpoint_ce": checkpoint_ce, "checkpoint_kl": checkpoint_kl,
        "checkpoint_stage": checkpoint_stage, "checkpoint_iter_in_stage": checkpoint_iter_in_stage,
        "checkpoint_every": np.int64(CHECKPOINT_EVERY), "checkpoint_top_k": np.int64(CHECKPOINT_TOP_K),
        "checkpoint_n_recent": np.int64(CHECKPOINT_N_RECENT),
        "p_target": ptg, "p_init": p0, "p_star": p_star,
        "elapsed_time": np.float64(elapsed_time), "n_iters": np.int64(len(cost_history)), "theta_seed": np.int64(SEED),
        "n_qubits": np.int64(n_qubits), "n_layers": np.int64(N_LAYERS),
        "init_scale": np.float64(INIT_SCALE), "eps_cost": np.float64(EPS_COST),
        "backend_name": np.array(BACKEND_NAME), "requested_topology": np.array(LOGICAL_TOPOLOGY),
        "effective_topology": np.array(effective_topology), "chosen_layout": np.array(chosen_layout, dtype=int),
        "layout_score": np.float64(layout_score), "fallback_used": np.bool_(layout_meta["fallback_used"]),
        "tried_layout_search": np.array(layout_meta["tried"], dtype=object),
        "transpiled_depth": np.int64(qcbm._tqc.depth()), "transpiled_size": np.int64(qcbm._tqc.size()),
        "transpiled_ops": np.array(dict(qcbm._tqc.count_ops()), dtype=object),
        "stage1_maxiter": np.int64(STAGE1_MAXITER), 
        "stage1_rhobeg": np.float64(STAGE1_RHOBEG),
        "stage2_maxiter": np.int64(STAGE2_MAXITER), 
        "stage2_maxfun": np.int64(STAGE2_MAXFUN),
        "rhobeg": np.float64(STAGE1_RHOBEG),
        "n_layers": np.int64(N_LAYERS),
        "eps_cost": np.float64(EPS_COST),
        "init_scale": np.float64(INIT_SCALE),
        "seed": np.int64(SEED),
        "stage1_nit": np.int64(getattr(best_run["result"], "nit", -1)),
        "stage2_nit": np.int64(getattr(refine_run["result"], "nit", -1)),
        "stage1_elapsed": np.float64(best_run["elapsed_time"]), "stage2_elapsed": np.float64(refine_run["elapsed_time"]),
        "stage1_ce_final": np.float64(best_run["ce_final"]), "stage1_kl_final": np.float64(best_run["kl_final"]),
        "stage2_ce_final": np.float64(refine_run["ce_final"]), "stage2_kl_final": np.float64(refine_run["kl_final"]),
        "metrics": np.array(ms, dtype=object),
    }

    checkpoint_save_data = {
        "checkpoint_idx": checkpoint_idx, "checkpoint_theta": checkpoint_theta,
        "checkpoint_ce": checkpoint_ce, "checkpoint_kl": checkpoint_kl,
        "checkpoint_stage": checkpoint_stage, "checkpoint_iter_in_stage": checkpoint_iter_in_stage,
        "theta_star": theta_star, "p_target": ptg, "target_entropy": np.float64(target_entropy),
    }

    save_experiment(saving_path, checkpoint_path, main_save_data, checkpoint_save_data)


if __name__ == "__main__":
    main()