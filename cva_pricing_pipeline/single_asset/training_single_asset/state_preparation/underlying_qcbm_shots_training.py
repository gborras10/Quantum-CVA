# python utils
import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib
from qiskit_algorithms.optimizers import SPSA

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)

# ------------------ Loading target probability distribution ------------------
ptg: np.ndarray = np.load(
    "data/single_asset/benchmark/run_classical_cva_single_asset.npz"
)["p_target"]

# ------------------ Output path setup ------------------
out_path = pathlib.Path(
    "data/single_asset/qcbm/qcbm_training_results_shots.npz"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------

# Build the joint (flattened) probability distribution
#target_probability_distribution: JointQcbmTarget = build_joint_target_from_P_bin(
#    conditional_probability_distribution
#)

#ptg: np.ndarray = target_probability_distribution.p_tg  

# =============================================================================
#                        QCBM ansatz definition
# =============================================================================
num_qubits_price: int = 2
num_qubits_time: int = 2
num_qubits: int = num_qubits_price + num_qubits_time

qcbm = MLQcbmCircuit(n_qubits=num_qubits, n_layers=2, name="G_p")
qc, theta = qcbm.qc, qcbm.theta

# ----------------------- Shots-based SPSA training ---------------------------
# Hyperparameters
theta_seed: int = 42
probability_seed: int = 105 # for reproducibility of probability estimation 
n_iters: int = 200
shots: int = 10000
epsilon: float = 1e-9

rng = np.random.default_rng(theta_seed)
x0 = np.zeros(qcbm.n_params)
p0_shots: np.ndarray = 0.1 * qcbm.probabilities(x0, shots=shots, seed=probability_seed)

cost_shots = qcbm.cost_fn(
    ptg,
    shots=shots,
    seed=None,
    smoothing="dirichlet",
    alpha=1.0,
    metric="l2",
)

cost_history: list[float] = []
best = {"fx": float("inf"), "x": np.empty(qcbm.n_params)}

def cb(nfev, x, fx, dx, accept):
    fx = float(fx)
    cost_history.append(fx)
    if fx < best["fx"]:
        best["fx"] = fx
        best["x"] = np.asarray(x, dtype=float).copy()

lr, pert = SPSA.calibrate(
    cost_shots,
    x0,
)

shots_optimizer = SPSA(
    maxiter=int(n_iters),
    learning_rate=lr,
    perturbation=pert,
    resamplings={0: 1, 200: 3, 600: 5},
    last_avg=20,
    blocking=False,
    trust_region=False,
    second_order=False,
    perturbation_dims=None,
    callback=cb,
)

# Training in the shots-based framework
t0: float = time.perf_counter()
res = shots_optimizer.minimize(fun=cost_shots, x0=x0)
t1: float = time.perf_counter()
elapsed_time: float = t1 - t0

print(f"Training complete in {elapsed_time:.1f} s")

theta_last: np.ndarray = np.asarray(res.x, dtype=float)
theta_best: np.ndarray = best["x"].copy()
p_star_best: np.ndarray = qcbm.probabilities(theta_best, shots=shots, seed=probability_seed)

# Metrics to quantify training quality
shots_metrics = qcbm.metrics(ptg, p_star_best)
print(f"KL divergence: {shots_metrics['kl']}")

# -------------------- Plot (similar to Alcazar) --------------------
cost_history_arr = np.asarray(cost_history, dtype=float)
best_so_far = np.minimum.accumulate(cost_history_arr)
best_idx = np.flatnonzero(
    np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
)

labels = [format(i, f"0{qcbm.n_qubits}b") for i in range(qcbm.dim)]
fig_dist, fig_cost = plot_training_diagnostics_multi_asset(
    target=ptg,
    before=p0_shots,
    after=p_star_best,
    cost_history=cost_history_arr,
    best_so_far=best_so_far,
    best_idx=best_idx,
    labels=labels,
    xlabel="Computational basis state |x⟩",
    ylabel="Probability",
    cost_ylabel=(
        f"Rescaled CE"
    ),
    title_before="Before training  (QCBM, SPSA shots)",
    title_after="After training  (best-iter, QCBM, SPSA shots)",
    cost_log_x=False,
    cost_log_y=True,
)
plt.show()

# ------------------------ Save training results ----------------------
np.savez(
    out_path,
    # Parameters 
    theta_star=theta_best,
    theta_last=theta_last,
    theta_init=x0,
    # Training dynamics
    cost_history=cost_history_arr,
    best_so_far=best_so_far,
    best_idx=best_idx,
    # Probability distributions
    p_target=ptg,
    p_init=p0_shots,
    p_star=p_star_best,
    # Scalar metadata
    elapsed_time=np.float64(elapsed_time),
    best_cost=np.float64(best["fx"]),
    n_iters=np.int64(n_iters),
    shots=np.int64(shots),
    epsilon=np.float64(epsilon),
    theta_seed=np.int64(theta_seed),
    # Metrics dict (allow_pickle=True required when loading)
    metrics=np.array(shots_metrics, dtype=object),
)