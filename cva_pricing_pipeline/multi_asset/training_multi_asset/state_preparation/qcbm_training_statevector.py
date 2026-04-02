# python utils
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)

# =============================================================================
#                           Load benchmark data
# =============================================================================
repo_root = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

path = repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz"
data = np.load(path, allow_pickle=True)
p_target = data["p_target"]

# Flatten and validate
ptg = np.asarray(p_target, dtype=float).ravel()
ptg = ptg / ptg.sum()   # por seguridad
dim = ptg.size
n_qubits = int(np.log2(dim))
if 2**n_qubits != dim:
    raise ValueError(f"p_target length must be a power of two; got {dim}.")

# =============================================================================
#                        QCBM ansatz definition
# =============================================================================
N_LAYERS = 10          # probar 8 o 10 antes que 12
EPS_COST = 1e-12
ALPHA = 0.2            # peso MMD; el resto CE/KL

qcbm = MLQcbmCircuit(n_qubits=n_qubits, n_layers=N_LAYERS, name="G_p_statevector")
qc, theta = qcbm.qc, qcbm.theta

print(qc.draw(output="text", fold=120))
print("n_layers:", qcbm.n_layers)
print("n_params:", qcbm.n_params)

# =============================================================================
#   MMD kernel (Liu & Wang, 2018 — "Differentiable Learning of QCBM")
# =============================================================================
def _hamming_distance_matrix(n: int) -> np.ndarray:
    idx = np.arange(2**n, dtype=np.int64)
    xor = idx[:, None] ^ idx[None, :]
    d = np.zeros((2**n, 2**n), dtype=float)
    for b in range(n):
        d += (xor >> b) & 1
    return d

BANDWIDTHS = [0.5, 1.0, 2.0, 4.0]
D_HAM = _hamming_distance_matrix(n_qubits)
K_MMD = sum(np.exp(-D_HAM**2 / (2.0 * s**2)) for s in BANDWIDTHS)

# Pre-compute constant term p_tg^T K p_tg
K_ptg = K_MMD @ ptg
C_TGT = float(ptg @ K_ptg)

def mmd_cost_from_p(p: np.ndarray) -> float:
    return float(C_TGT - 2.0 * (ptg @ (K_MMD @ p)) + p @ (K_MMD @ p))

def ce_cost_from_p(p: np.ndarray) -> float:
    # CE(ptg, p) = KL(ptg || p) + const
    p_safe = np.clip(p, EPS_COST, 1.0)
    return float(-np.sum(ptg * np.log(p_safe)))

def hybrid_cost(x: np.ndarray) -> float:
    p = qcbm.probabilities(x)
    mmd = mmd_cost_from_p(p)
    ce  = ce_cost_from_p(p)
    return float(ALPHA * mmd + (1.0 - ALPHA) * ce)

# =============================================================================
#   Statevector training — L-BFGS-B + hybrid loss
# =============================================================================
THETA_SEED = 355

rng = np.random.default_rng(THETA_SEED)
# Mejor inicialización: pequeña amplitud
x0 = 0.01 * rng.standard_normal(len(theta)).astype(float)

t0 = time.perf_counter()
_cost_history: list[float] = []
_kl_history: list[float] = []
_nfev_history: list[float] = []
_nfev_at_iter: list[int] = []
_iter_counter: list[int] = [0]
_best_cost: list[float] = [float("inf")]
_best_kl: list[float] = [float("inf")]
_nfev_counter: list[int] = [0]

# caché simple para evitar recomputación redundante en callback
_last_x: list[np.ndarray | None] = [None]
_last_p: list[np.ndarray | None] = [None]
_last_cost: list[float | None] = [None]

def _kl_value(p: np.ndarray) -> float:
    p_safe = np.clip(p, EPS_COST, 1.0)
    ptg_safe = np.clip(ptg, EPS_COST, 1.0)
    return float(np.sum(ptg_safe * (np.log(ptg_safe) - np.log(p_safe))))

def _get_p_cached(x: np.ndarray) -> np.ndarray:
    if _last_x[0] is not None and np.array_equal(x, _last_x[0]):
        return _last_p[0]
    p = qcbm.probabilities(x)
    _last_x[0] = np.array(x, copy=True)
    _last_p[0] = p
    return p

def _tracked_hybrid_cost(x: np.ndarray) -> float:
    p = _get_p_cached(x)
    mmd = mmd_cost_from_p(p)
    ce = ce_cost_from_p(p)
    c = float(ALPHA * mmd + (1.0 - ALPHA) * ce)

    _last_cost[0] = c
    _nfev_counter[0] += 1
    _nfev_history.append(c)
    return c

def _callback(xk: np.ndarray) -> None:
    _iter_counter[0] += 1

    # usa caché si scipy acaba de evaluar ese mismo punto
    p = _get_p_cached(xk)
    c = _last_cost[0]
    if c is None:
        c = _tracked_hybrid_cost(xk)

    kl = _kl_value(p)

    _cost_history.append(c)
    _kl_history.append(kl)
    _nfev_at_iter.append(_nfev_counter[0])

    if c < _best_cost[0]:
        _best_cost[0] = c
    if kl < _best_kl[0]:
        _best_kl[0] = kl

    if _iter_counter[0] % 10 == 0 or _iter_counter[0] == 1:
        elapsed = time.perf_counter() - t0
        print(
            f"  iter {_iter_counter[0]:>4} | nfev {_nfev_counter[0]:>6} | "
            f"loss = {c:.6e} | KL = {kl:.6e} | "
            f"best loss = {_best_cost[0]:.6e} | best KL = {_best_kl[0]:.6e} | "
            f"{elapsed:6.1f}s",
            flush=True,
        )

print(f"n_params = {qcbm.n_params}")
print("Iniciando entrenamiento L-BFGS-B...")
res = minimize(
    _tracked_hybrid_cost,
    x0=x0,
    method="L-BFGS-B",
    callback=_callback,
    options=dict(
        maxiter=300,
        maxcor=50,
        maxfun=100_000,
        ftol=1e-11,
        gtol=1e-6,
        eps=5e-5,
        disp=True,
    ),
)

cost_history = np.asarray(_cost_history, dtype=float)
kl_history = np.asarray(_kl_history, dtype=float)
nfev_history = np.asarray(_nfev_history, dtype=float)
nfev_at_iter = np.asarray(_nfev_at_iter, dtype=int)

t1 = time.perf_counter()
elapsed_time = t1 - t0

theta_star = res.x

# Probabilities before / after training
p0 = qcbm.probabilities(x0)
p_star = qcbm.probabilities(theta_star)

print("\nsuccess:", res.success)
print("message:", res.message)
print("nfev:", getattr(res, "nfev", None), "  nit:", getattr(res, "nit", None))
print("final loss:", float(res.fun))
print("final KL(ptg || p*):", _kl_value(p_star))
print(f"elapsed time: {elapsed_time:.2f} s")

# =============================================================================
#                   Diagnostics plots
# =============================================================================
rescaled_plot = np.maximum(cost_history, 1e-15)
best_so_far = np.minimum.accumulate(rescaled_plot)
best_idx = np.flatnonzero(
    np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
)

# pérdida vs nfev
rescaled_nfev = np.maximum(nfev_history, 1e-15)
best_nfev = np.minimum.accumulate(rescaled_nfev)
nfev_x = np.arange(1, len(nfev_history) + 1, dtype=float)

fig_nfev, ax_nfev = plt.subplots(figsize=(8, 4))
ax_nfev.plot(nfev_x, rescaled_nfev, lw=0.5, alpha=0.4, color="steelblue", label="loss (cada eval)")
ax_nfev.plot(nfev_x, best_nfev, lw=1.5, color="darkorange", label="mejor hasta ahora")
if len(nfev_at_iter) > 0:
    ax_nfev.scatter(
        nfev_at_iter, np.maximum(cost_history, 1e-15),
        s=20, zorder=5, color="crimson", label="fin de iteración",
    )
ax_nfev.set_xscale("linear")
ax_nfev.set_yscale("log")
ax_nfev.set_xlabel("Número de evaluaciones (nfev)")
ax_nfev.set_ylabel("Loss híbrida")
ax_nfev.set_title("Convergencia vs evaluaciones de función")
ax_nfev.legend()
ax_nfev.grid(True, which="both", alpha=0.3, linestyle="--")
fig_nfev.tight_layout()

# KL vs iteración
fig_kl, ax_kl = plt.subplots(figsize=(8, 4))
ax_kl.plot(np.maximum(kl_history, 1e-15), lw=1.5)
ax_kl.set_yscale("log")
ax_kl.set_xlabel("Iteración")
ax_kl.set_ylabel("KL(ptg || p)")
ax_kl.grid(True, which="both", alpha=0.3, linestyle="--")
fig_kl.tight_layout()

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
print("ptg:\n", ptg)
print("\np_star:\n", p_star)

# =============================================================================
#                   Save results
# =============================================================================
out_dir = repo_root / "data" / "multi_asset" / "quantum" / "training" / "qcbm"
out_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    out_dir / "qcbm_statevector_training_results.npz",
    theta_star=theta_star,
    p_star=p_star,
    ptg=ptg,
    cost_history=cost_history,
    kl_history=kl_history,
    metrics=ms,
    training_time=elapsed_time,
    alpha=ALPHA,
    bandwidths=np.asarray(BANDWIDTHS, dtype=float),
)
print(f"\nResults saved to {out_dir / 'qcbm_statevector_training_results.npz'}")