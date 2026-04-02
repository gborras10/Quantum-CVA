# hwd_aware_qcbm_training_heavyhex8_shots.py

from __future__ import annotations

import pathlib
import numpy as np
import matplotlib.pyplot as plt

from qiskit_algorithms.optimizers import SPSA


# ============================================================
# CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)

MAXITER_STAGE1 = 200
MAXITER_STAGE2 = 200

CHECKPOINT_EVERY = 10
EVAL_KL_EVERY = 25

OUTPUT_DIR = pathlib.Path("training_runs_heavyhex8")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# NUMERICAL UTILS
# ============================================================

def laplace_smooth(p, alpha=0.5):
    p = np.asarray(p)
    return (p + alpha) / (p.sum() + alpha * len(p))


def kl_divergence(p, q):
    eps = 1e-12
    return np.sum(p * np.log((p + eps) / (q + eps)))


def cross_entropy(p, q):
    eps = 1e-12
    return -np.sum(p * np.log(q + eps))


# ============================================================
# SHOTS SCHEDULER
# ============================================================

def shots_schedule(iteration):

    if iteration < 200:
        return 2000

    elif iteration < 350:
        return 2000

    return 2000


# ============================================================
# COST FUNCTION FACTORY
# ============================================================

def make_cost_function(qcbm, target):

    def cost(theta, iteration=None):

        shots = shots_schedule(iteration or 0)

        probs = qcbm.probabilities(
            theta,
            shots=shots,
        )

        probs = laplace_smooth(probs)

        return cross_entropy(target, probs)

    return cost


# ============================================================
# EXACT KL EVALUATION
# ============================================================

def exact_kl(qcbm, theta, target):

    probs = qcbm.probabilities(theta, shots=None)

    return kl_divergence(target, probs)


# ============================================================
# TRAINING STAGE
# ============================================================

def train_stage(
    stage_name,
    qcbm,
    theta,
    target,
    maxiter,
    checkpoints,
    second_order=False,
):

    print(f"\n=== {stage_name} ===")

    cost_fn = make_cost_function(qcbm, target)

    print("Calibrating SPSA...")

    a_gen, c_gen = SPSA.calibrate(
        lambda x: cost_fn(x, iteration=0),
        theta,
        target_magnitude=0.1,
    )

    cost_history = []
    kl_history = []

    iteration_counter = 0

    def callback(nfev, x, fx, stepsize, accepted):

        nonlocal iteration_counter

        iteration_counter += 1

        cost_val = float(fx)
        cost_history.append(cost_val)

        if iteration_counter % CHECKPOINT_EVERY == 0:
            checkpoints.append(x.copy())

        if iteration_counter % EVAL_KL_EVERY == 0:

            kl_val = exact_kl(qcbm, x, target)
            kl_history.append(kl_val)

            print(
                f"iter={iteration_counter} "
                f"cost={cost_val:.6f} "
                f"KL_exact={kl_val:.6f}"
            )

    optimizer = SPSA(
        maxiter=maxiter,
        learning_rate=a_gen,
        perturbation=c_gen,
        blocking=True,
        trust_region=True,
        resamplings=4,
        second_order=second_order,
        callback=callback,
    )

    result = optimizer.minimize(
        fun=lambda x: cost_fn(x, iteration_counter),
        x0=theta,
    )

    theta = result.x

    return theta, cost_history, kl_history


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_qcbm(qcbm, target, theta0):

    checkpoints = []
    theta = theta0.copy()

    # --------------------------------------------------------
    # STAGE 1: SPSA estándar
    # --------------------------------------------------------

    theta, cost1, kl1 = train_stage(
        "STAGE 1 (SPSA)",
        qcbm,
        theta,
        target,
        maxiter=MAXITER_STAGE1,
        checkpoints=checkpoints,
        second_order=False,
    )

    # --------------------------------------------------------
    # STAGE 2: 2-SPSA refinamiento final
    # --------------------------------------------------------

    theta, cost2, kl2 = train_stage(
        "STAGE 2 (2-SPSA refinement)",
        qcbm,
        theta,
        target,
        maxiter=MAXITER_STAGE2,
        checkpoints=checkpoints,
        second_order=True,
    )

    return theta, checkpoints, cost1 + cost2, kl1 + kl2


# ============================================================
# CHECKPOINT SELECTION
# ============================================================

def select_best_checkpoint(qcbm, checkpoints, target):

    best_theta = None
    best_kl = np.inf

    for theta in checkpoints:

        kl = exact_kl(qcbm, theta, target)

        if kl < best_kl:
            best_kl = kl
            best_theta = theta

    print(f"\nBest checkpoint KL={best_kl:.6f}")

    return best_theta


# ============================================================
# DIAGNOSTICS
# ============================================================

def plot_training(cost_history):

    cost_history = np.asarray(cost_history, dtype=float).ravel()

    if cost_history.size == 0:
        print("No cost history to plot.")
        return

    best_so_far = np.minimum.accumulate(cost_history)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
    )

    steps = np.arange(cost_history.size)

    plt.figure(figsize=(8, 4))

    plt.plot(
        best_idx,
        best_so_far[best_idx],
        linewidth=2.2,
        marker="o",
        markersize=3.2,
        label="best-so-far",
        color="steelblue",
        zorder=2,
    )

    plt.plot(
        steps,
        cost_history,
        linestyle="none",
        marker=".",
        markersize=2.2,
        alpha=0.55,
        label="cost",
        color="#b94d95",
        zorder=3,
    )

    if cost_history.size > 1:
        plt.xscale("linear")
    if np.all(cost_history > 0):
        plt.yscale("log")

    plt.title("Training Cost Evolution")
    plt.xlabel("Optimization Step")
    plt.ylabel("Cross-entropy")
    plt.grid(True, which="both")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN TRAINING EXECUTION
# ============================================================

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)


def main():

    repo_root = next(
        parent for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data = np.load(
        repo_root
        / "data"
        / "multi_asset"
        / "8q_instance"
        / "benchmark"
        / "three_asset_instance.npz",
        allow_pickle=True,
    )

    target = np.asarray(data["p_target"], dtype=float).ravel()
    target /= target.sum()

    n_qubits = int(np.log2(len(target)))

    print(f"\nn_qubits = {n_qubits}")

    # =====================================================
    # BACKEND + LAYOUT
    # =====================================================

    service = QiskitRuntimeService(channel="ibm_cloud")

    backend = service.backend(
        "ibm_basquecountry",
        use_fractional_gates=True,
    )

    layout, score, meta = select_best_layout(
        backend,
        topology="qcbm_heavyhex8",
        length=n_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
    )

    print("\nLayout selected:", layout)
    print("Layout score:", score)

    # =====================================================
    # BUILD QCBM
    # =====================================================

    qcbm = MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=6,
        entangler="rzz",
        topology=meta["selected_topology"],
        backend=AerSimulator(method="density_matrix"),
        transpile_backend=backend,
        simulation_method="density_matrix",
        optimization_level=3,
        initial_layout=layout,
        routing_method="none",
    )

    summarize_circuit(qcbm._tqc)

    # =====================================================
    # INITIAL PARAMETERS
    # =====================================================

    rng = np.random.default_rng(SEED)

    theta0 = 0.1 * rng.standard_normal(qcbm.n_params)

    print("\nStarting SPSA training...\n")

    # =====================================================
    # TRAIN
    # =====================================================

    theta_final, checkpoints, cost_history, kl_history = train_qcbm(
        qcbm,
        target,
        theta0,
    )

    print("\nSelecting best checkpoint...")

    theta_best = select_best_checkpoint(
        qcbm,
        checkpoints,
        target,
    )

    # =====================================================
    # FINAL METRICS
    # =====================================================

    final_probs = qcbm.probabilities(theta_best, shots=None)

    final_kl = kl_divergence(target, final_probs)

    print("\nFinal KL:", final_kl)

    # =====================================================
    # SAVE
    # =====================================================

    save_file = OUTPUT_DIR / "best_qcbm_heavyhex8_shots.npz"

    np.savez(
        save_file,
        theta_best=theta_best,
        theta_final=theta_final,
        checkpoints=np.array(checkpoints),
        cost_history=np.array(cost_history),
        kl_history=np.array(kl_history),
        final_kl=final_kl,
    )

    print("\nSaved to:", save_file)

    # =====================================================
    # PLOT
    # =====================================================

    plot_training(cost_history)


if __name__ == "__main__":
    main()