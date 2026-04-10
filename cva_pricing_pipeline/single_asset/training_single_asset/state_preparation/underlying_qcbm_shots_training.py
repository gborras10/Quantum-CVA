# python utils
import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib
from qiskit_algorithms.optimizers import SPSA

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset as plot_training_diagnostics,
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

qcbm = MLQcbmCircuit(n_qubits=num_qubits, n_layers=2, entangler='rxx', name="G_p")
qc, theta = qcbm.qc, qcbm.theta

# ----------------------- Shots-based SPSA training ---------------------------
# Hyperparameters
theta_seed: int = 355
probability_seed: int = 105

N_ITERS: int = 2500
SHOTS: int = 5000

# Tuned hyperparams for SPSA (minimized for KL divergence)
LR: float = 0.007
PERT: float = 0.07
eps: float = float(1e-9)

rng = np.random.default_rng(theta_seed)
x0 = rng.standard_normal(len(theta)).astype(float)

cost_shots = qcbm.cost_fn(
    ptg,
    eps=eps,
    shots=SHOTS,
    seed=None,
    rescaled=True,
    smoothing="dirichlet",
    alpha=1.0,
)

# Plain SPSA run (store best theta)
theta0 = np.asarray(x0, dtype=float).copy()

cost_history: list[float] = []
best = {"fx": float("inf"), "x": theta0.copy()}

def cb(nfev, x, fx, stepsize, accepted):
    del nfev, stepsize, accepted
    fx = float(fx)
    cost_history.append(fx)
    if fx < best["fx"]:
        best["fx"] = fx
        best["x"] = np.asarray(x, dtype=float).copy()

opt = SPSA(
    maxiter=int(N_ITERS),
    learning_rate=LR,
    perturbation=PERT,
    resamplings=3,
    blocking=False,
    callback=cb,
    trust_region=True,
    regularization=0.1,
)

# Training in the shots-based framework
_t0: float = time.perf_counter()
res = opt.minimize(fun=cost_shots, x0=theta0)
_spsa_time: float = time.perf_counter() - _t0

print(f"Training complete. Best cost observed: {best['fx']:.6f}")
print(f"SPSA optimization time (s): {_spsa_time:.2f}")

theta_last: np.ndarray = np.asarray(res.x, dtype=float)
theta_best: np.ndarray = best["x"].copy()

# Diagnostics plot
cost_history = np.asarray(cost_history, dtype=float)
best_so_far = np.minimum.accumulate(cost_history)
tol = 1e-15
best_idx = np.flatnonzero(
    np.r_[True, best_so_far[1:] < best_so_far[:-1] - tol]
)

p0_shots = qcbm.probabilities(theta0, shots=SHOTS, seed=None)
p_star_best = qcbm.probabilities(theta_best, shots=SHOTS, seed=None)

# Metrics to quantify training quality
shots_metrics = qcbm.metrics(ptg, p_star_best)
print(f"KL divergence: {shots_metrics['kl']}")

labels = [format(i, f"0{qcbm.n_qubits}b") for i in range(qcbm.dim)]
plot_training_diagnostics(
    target=ptg,
    before=p0_shots,
    after=p_star_best,
    cost_history=cost_history,
    best_so_far=best_so_far,
    best_idx=best_idx,
    labels=labels,
    xlabel="Computational basis state |x⟩",
    ylabel="Probability",
    cost_ylabel=f"Rescaled CE (SPSA, LR={LR:.6f}, PERT={PERT:.6f}, shots={SHOTS})",
    title_before="Before training (QCBM, SPSA shot-based)",
    title_after="After training (best-iter, QCBM, SPSA shot-based)",
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
    theta_init=theta0,
    # Training dynamics
    cost_history=cost_history,
    best_so_far=best_so_far,
    best_idx=best_idx,
    # Probability distributions
    p_target=ptg,
    p_init=p0_shots,
    p_star=p_star_best,
    # Scalar metadata
    elapsed_time=np.float64(_spsa_time),
    best_cost=np.float64(best["fx"]),
    n_iters=np.int64(N_ITERS),
    shots=np.int64(SHOTS),
    epsilon=np.float64(eps),
    theta_seed=np.int64(theta_seed),
    learning_rate=np.float64(LR),
    perturbation=np.float64(PERT),
    # Metrics dict (allow_pickle=True required when loading)
    metrics=np.array(shots_metrics, dtype=object),
)