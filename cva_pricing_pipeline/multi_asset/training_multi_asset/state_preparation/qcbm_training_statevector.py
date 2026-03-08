# python utils
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    minimize_with_cost_history,
    plot_training_diagnostics,
)

# =============================================================================
#                           Load benchmark data
# =============================================================================
path = (
    pathlib.Path(__file__).resolve().parents[3]
    / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz"
)
data = np.load(path, allow_pickle=True)
p_target = data["p_target"]

# Flatten and validate
ptg = np.asarray(p_target, dtype=float).ravel()
dim = ptg.size
n_qubits = int(np.log2(dim))
if 2**n_qubits != dim:
    raise ValueError(f"p_target length must be a power of two; got {dim}.")

# =============================================================================
#                        QCBM ansatz definition
# =============================================================================
N_LAYERS = 8
EPS_COST = 1e-12

qcbm = MLQcbmCircuit(n_qubits=n_qubits, n_layers=N_LAYERS, name="G_p_statevector")
qc, theta = qcbm.qc, qcbm.theta

print(qc.draw(output="text", fold=120))
print("n_layers:", qcbm.n_layers)
print("n_params:", qcbm.n_params)

# =============================================================================
#                   Statevector training (COBYLA)
# =============================================================================
# ---- Hyperparameters ----
THETA_SEED = 355
N_ITERS    = 5000
RHOBEG     = 0.5
METHOD     = "COBYLA"

rng = np.random.default_rng(THETA_SEED)
x0  = rng.standard_normal(len(theta)).astype(float)

# ---- Cost function (ideal / statevector) ----
cost = qcbm.cost_fn(ptg, eps=EPS_COST)

# ---- Optimize ----
t0 = time.perf_counter()
res, cost_history = minimize_with_cost_history(
    cost,
    x0=x0,
    minimize_fn=minimize,
    method=METHOD,
    options={"maxiter": int(N_ITERS), "rhobeg": RHOBEG, "disp": True},
)
t1 = time.perf_counter()
elapsed_time = t1 - t0

theta_star = np.asarray(res.x, dtype=float)

# Probabilities before / after training
p0     = qcbm.probabilities(x0)
p_star = qcbm.probabilities(theta_star)

print("\nsuccess:", res.success)
print("message:", res.message)
print("nfev:", getattr(res, "nfev", None), "  nit:", getattr(res, "nit", None))
print("final cost:", float(res.fun))
print(f"elapsed time: {elapsed_time:.2f} s")

# =============================================================================
#                   Diagnostics plots
# =============================================================================
C_star        = -np.sum(ptg * np.log(np.maximum(ptg, EPS_COST)))
rescaled_plot = np.maximum(cost_history - C_star, 1e-12)
best_so_far   = np.minimum.accumulate(rescaled_plot)
best_idx      = np.flatnonzero(
    np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
)

fig_dist, fig_cost = plot_training_diagnostics(
    target=ptg,
    before=p0,
    after=p_star,
    cost_history=rescaled_plot,
    best_so_far=best_so_far,
    best_idx=best_idx,
    labels=[format(i, f"0{qcbm.n_qubits}b") for i in range(qcbm.dim)],
)
plt.show()

# =============================================================================
#                   Metrics
# =============================================================================
ms = qcbm.metrics(ptg, p_star, eps=EPS_COST)

print("\n=== FIT METRICS ===")
print("KL(ptg || p*)  =", float(ms["kl"]))
print("L1             =", float(ms["l1"]))
print("TV = 0.5*L1    =", float(ms["tv"]))
print("Linf           =", float(ms["linf"]))

print("\n=== DISTRIBUTIONS ===")
print("ptg:\n",    ptg)
print("\np_star:\n", p_star)

# =============================================================================
#                   Save results
# =============================================================================
out_dir = (
    pathlib.Path(__file__).resolve().parents[3]
    / "data" / "multi_asset" / "training" / "qcbm"
)
out_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    out_dir / "qcbm_statevector_training_results.npz",
    theta_star=theta_star,
    p_star=p_star,
    ptg=ptg,
    cost_history=cost_history,
)
print(f"\nResults saved to {out_dir / 'qcbm_statevector_training_results.npz'}")