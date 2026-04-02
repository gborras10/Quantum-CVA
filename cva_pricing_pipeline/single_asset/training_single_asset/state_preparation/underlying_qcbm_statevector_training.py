# python utils
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    minimize_with_cost_history,
    plot_training_diagnostics_multi_asset,
)

# from quantum_cva.cross_validation_utils import SPSAHyperparamCV

# ------------------ Output path setup ------------------
out_path = pathlib.Path(
    "data/single_asset/qcbm/qcbm_training_results_statevector.npz"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------

# ------------------ Loading target probability distribution ------------------
target_probability_distribution_flatten: np.ndarray = np.load(
    "data/single_asset/benchmark/run_classical_cva_single_asset.npz"
)["p_target"]

# Build the joint (flatten) probability distribution
'''
target_probability_distribution: JointQcbmTarget = (
    build_joint_target_from_P_bin(conditional_probability_distribution)
)

target_probability_distribution_flatten: np.ndarray = (
    target_probability_distribution.p_tg
)
'''

# =============================================================================
#                        QCBM ansatz definition & training
# =============================================================================
num_qubits_price = 2
num_qubits_time = 2
num_qubits = num_qubits_price + num_qubits_time

# Ansatz definition
qcbm: MLQcbmCircuit = MLQcbmCircuit(
    n_qubits=num_qubits, 
    n_layers=2, 
    name="G_p_statevector",
    entangler="rxx",
    topology="all-to-all",
    simulation_method="statevector",
)
qc, theta = qcbm.qc, qcbm.theta

# -------------------------- Statevector training ----------------------------
# Hyperparameters
EPS_COST: int = 1e-12
theta_seed = 355
N_ITERS = 7000
RHOBEG = 0.5
METHOD = "COBYLA"

rng = np.random.default_rng(theta_seed)
x0 = rng.standard_normal(len(theta)).astype(float)

# Define cost function
cost = qcbm.cost_fn(target_probability_distribution_flatten, eps=EPS_COST)

# Run the optimization
res, cost_history = minimize_with_cost_history(
    cost,
    x0=x0,
    minimize_fn=minimize,
    method=METHOD,
    options={"maxiter": int(N_ITERS), 
             "rhobeg": RHOBEG,
             "disp": True},
)

theta_star = np.asarray(res.x, dtype=float)

# ------ Probabilities: p0 -> before training, p_star -> after training ------ 
p0 = qcbm.probabilities(x0)
p_star = qcbm.probabilities(theta_star)
# ----------------------------------------------------------------------------

print("success:", res.success)
print("message:", res.message)
print("nfev:", getattr(res, "nfev", None), "nit:", getattr(res, "nit", None))
print("final cost:", float(res.fun))

after_train_cost_statevector = -np.sum(
    target_probability_distribution_flatten
    * np.log(
        np.maximum(
            target_probability_distribution_flatten,
            EPS_COST,
        )
    )
)
statevector_result_reescaled_plot = np.maximum(
    cost_history - after_train_cost_statevector, 1e-12
)
best_so_far = np.minimum.accumulate(statevector_result_reescaled_plot)
best_idx = np.flatnonzero(
    np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
)

fig_dist, fig_cost = plot_training_diagnostics_multi_asset(
    target=target_probability_distribution_flatten,
    before=p0,
    after=p_star,
    cost_history=statevector_result_reescaled_plot,
    best_so_far=best_so_far,
    best_idx=best_idx,
    labels=[format(i, f"0{qcbm.n_qubits}b") for i in range(qcbm.dim)],
)
plt.show()

statevector_metrics = qcbm.metrics(
    target_probability_distribution_flatten,
    p_star,
)

# ------------------------ Save training results ----------------------
np.savez(
    out_path,
    # Parameters — theta_star is what run_cva_statevector.py expects
    theta_star=theta_star,
    theta_init=x0,
    # Training dynamics
    best_so_far=best_so_far,
    best_idx=best_idx,
    # Probability distributions
    p_target=target_probability_distribution_flatten,
    p_init=p0,
    p_star=p_star,
    # Scalar metadata
    n_iters=np.int64(N_ITERS),
    theta_seed=np.int64(theta_seed),
    # Metrics dict (allow_pickle=True required when loading)
    metrics=np.array(statevector_metrics, dtype=object),
)