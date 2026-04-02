# python utils
import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib
from qiskit_algorithms.optimizers import SPSA

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import plot_training_diagnostics_multi_asset

# ------------------ Loading target function values ------------------
benchmark = np.load(
    "data/single_asset/benchmark/run_classical_cva_single_asset.npz"
)
p_t: np.ndarray = benchmark["p_t"]   
C_p: float = float(benchmark["C_p"])

f_target: np.ndarray = p_t / C_p   

# ------------------ Output path setup ------------------
out_path = pathlib.Path(
    "data/single_asset/crca/discount_factors/training_results_shots.npz"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
# -------------------------------------------------------

# =============================================================================
#                        CRCA ansatz definition
# =============================================================================
m_time: int = 2   
n_price: int = 0  
n_layers: int = 1

crca = CrcaCircuit(m_time=m_time, n_price=n_price, n_layers=n_layers,
                   name="crca_discount_factor")

# ----------------------- Shots-based SPSA training ---------------------------
# Hyperparameters
theta_seed: int = 42
n_iters: int = 1000
shots: int = 5000
learning_rate: float = 0.25
perturbation: float = 0.05

rng = np.random.default_rng(theta_seed)
x0: np.ndarray = 0.5 * rng.standard_normal(crca.n_params).astype(float)
f0_shots: np.ndarray = crca.function_values(x0, shots=shots, seed=None)

cost_shots = crca.cost_fn(
    f_target,
    shots=shots,
    seed=None,
)

cost_history: list[float] = []
best = {"fx": float("inf"), "x": np.empty(crca.n_params)}


def cb(nfev, x, fx, dx, accept):
    fx = float(fx)
    cost_history.append(fx)
    if fx < best["fx"]:
        best["fx"] = fx
        best["x"] = np.asarray(x, dtype=float).copy()


shots_optimizer = SPSA(
    maxiter=int(n_iters),
    learning_rate=learning_rate,
    perturbation=perturbation,
    resamplings=3,
    blocking=False,
    callback=cb,
    trust_region=True,
)

# Training in the shots-based framework
t0: float = time.perf_counter()
res = shots_optimizer.minimize(fun=cost_shots, x0=x0)
t1: float = time.perf_counter()
elapsed_time: float = t1 - t0

print(f"Training complete in {elapsed_time:.1f} s")
print(f"Best L2 cost observed: {best['fx']:.8f}")

theta_last: np.ndarray = np.asarray(res.x, dtype=float)
theta_best: np.ndarray = best["x"].copy()
f_star_shots: np.ndarray = crca.function_values(theta_best, shots=shots, seed=None)

# -------------------- Plot (similar to Alcazar) --------------------
cost_history_arr = np.asarray(cost_history, dtype=float)
best_so_far = np.minimum.accumulate(cost_history_arr)
best_idx = np.flatnonzero(
    np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
)

time_labels = [format(i, f"0{m_time}b") for i in range(2**m_time)]
fig_dist, fig_cost = plot_training_diagnostics_multi_asset(
    target=f_target,
    before=f0_shots,
    after=f_star_shots,
    cost_history=cost_history_arr,
    best_so_far=best_so_far,
    best_idx=best_idx,
    labels=time_labels,
    xlabel="Time register |t⟩",
    ylabel="f(t)",
    cost_ylabel=(
        f"L2 loss  (SPSA, LR={learning_rate:.4g},"
        f" PERT={perturbation:.4g}, shots={shots})"
    ),
    title_before="Before training  (CRCA discount factor, SPSA shots)",
    title_after="After training  (best-iter, CRCA discount factor, SPSA shots)",
    cost_log_x=False,
    cost_log_y=True,
)
plt.show()

# ------------------------ Save training results ----------------------
metadata = {
    "model": "CRCA",
    "task": "discount_factor",
    "ancilla_observable": "P(a=1 | control=i)",
    "m_time": m_time,
    "n_price": n_price,
    "n_controls": crca.n_controls,
    "n_layers": n_layers,
    "n_parameters": crca.n_params,
    "optimizer": "SPSA",
    "optimizer_library": "qiskit-algorithms",
    "learning_rate": learning_rate,
    "perturbation": perturbation,
    "maxiter": n_iters,
    "resamplings": 3,
    "blocking": False,
    "trust_region": True,
    "cost_function": "L2",
    "shots": shots,
    "stochastic_cost": True,
    "shot_seed": None,
    "best_iter_cost_observed": float(best["fx"]),
    "stopping_criterion": "maxiter",
    "note": (
        "CRCA discount-factor training via shots-only SPSA. "
        "theta_star = best-iteration parameters."
    ),
}

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
    # Function values
    f_target=f_target,
    f_init=f0_shots,
    f_star_shots=f_star_shots,
    # Scalar metadata
    elapsed_time=np.float64(elapsed_time),
    best_cost=np.float64(best["fx"]),
    C_p=np.float64(C_p),
    n_iters=np.int64(n_iters),
    shots=np.int64(shots),
    learning_rate=np.float64(learning_rate),
    perturbation=np.float64(perturbation),
    theta_seed=np.int64(theta_seed),
    # Metadata dict (allow_pickle=True required when loading)
    metadata=np.array(metadata, dtype=object),
)