# python utils
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

# quantum_cva utils
from quantum_cva.single_asset.state_prep.qcbm.target_distribution import (
    build_joint_target_from_P_bin,
    JointQcbmTarget,
)
from quantum_cva.single_asset.state_prep.qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    minimize_with_cost_history,
    plot_training_diagnostics,
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
    name="G_p_statevector"
)
qc, theta = qcbm.qc, qcbm.theta

# -------------------------- Statevector training ----------------------------
# Hyperparameters
epsilon_cost: int = 1e-12
theta_seed: int = 355
n_iters: int = 7000
rhobeg: float = 0.5
method: str = "COBYLA"

rng = np.random.default_rng(theta_seed)
x0: np.ndarray = rng.standard_normal(qcbm.n_params).astype(float)

cost_statevector: callable = qcbm.cost_fn(
    target_probability_distribution_flatten,
    eps=epsilon_cost,
)

# Perform the QCBM training in the statevector backend
t0: float = time.perf_counter()
statevector_result, cost_history = minimize_with_cost_history(
    cost_statevector,
    x0=x0,
    minimize_fn=minimize,
    method=method,
    options={"maxiter": int(n_iters), "rhobeg": rhobeg, "disp": True},
)
t1: float = time.perf_counter()
elapsed_time: float = t1 - t0

theta_best_sv: np.ndarray = np.asarray(
    statevector_result.x,
    dtype=float,
)

initial_probabilities: np.ndarray = qcbm.probabilities(
    rng.standard_normal(len(theta)).astype(float)
)
trained_probabilities_statevector: np.ndarray = qcbm.probabilities(
    theta_best_sv
)  # sample(seed)

after_train_cost_statevector = -np.sum(
    target_probability_distribution_flatten
    * np.log(
        np.maximum(
            target_probability_distribution_flatten,
            epsilon_cost,
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

fig_dist, fig_cost = plot_training_diagnostics(
    target=target_probability_distribution_flatten,
    before=initial_probabilities,
    after=trained_probabilities_statevector,
    cost_history=statevector_result_reescaled_plot,
    best_so_far=best_so_far,
    best_idx=best_idx,
    labels=[format(i, f"0{qcbm.n_qubits}b") for i in range(qcbm.dim)],
)
plt.show()

statevector_metrics = qcbm.metrics(
    target_probability_distribution_flatten,
    trained_probabilities_statevector,
)

# ------------------------ Save training results ----------------------
np.savez(
    out_path,
    # Parameters — theta_star is what run_cva_statevector.py expects
    theta_star=theta_best_sv,
    theta_init=x0,
    # Training dynamics
    best_so_far=best_so_far,
    best_idx=best_idx,
    # Probability distributions
    p_target=target_probability_distribution_flatten,
    p_init=initial_probabilities,
    p_star=trained_probabilities_statevector,
    # Scalar metadata
    elapsed_time=np.float64(elapsed_time),
    n_iters=np.int64(n_iters),
    theta_seed=np.int64(theta_seed),
    # Metrics dict (allow_pickle=True required when loading)
    metrics=np.array(statevector_metrics, dtype=object),
)