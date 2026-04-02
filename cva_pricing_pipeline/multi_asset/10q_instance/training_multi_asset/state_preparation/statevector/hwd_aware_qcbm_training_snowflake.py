import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator

from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)

def minimize_with_cost_history(
    cost_fn,
    *,
    x0,
    minimize_fn,
    method,
    options,
):
    x0 = np.asarray(x0, dtype=float)

    f0 = float(cost_fn(x0))

    # Históricos alineados: coste[i] corresponde a theta[i]
    cost_history: list[float] = [f0]
    theta_history: list[np.ndarray] = [x0.copy()]

    def wrapped(x):
        x = np.asarray(x, dtype=float)
        return float(cost_fn(x))

    def callback(xk):
        # Iterado aceptado por el optimizador
        xk = np.asarray(xk, dtype=float)
        fk = float(cost_fn(xk))

        cost_history.append(fk)
        theta_history.append(xk.copy())

    res = minimize_fn(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    x_final = np.asarray(res.x, dtype=float)
    f_final = float(res.fun)

    # Asegura que el último punto guardado coincide con el resultado final
    same_theta = np.allclose(theta_history[-1], x_final, rtol=0.0, atol=1e-15)
    same_cost = abs(cost_history[-1] - f_final) <= 1e-15

    if not (same_theta and same_cost):
        cost_history.append(f_final)
        theta_history.append(x_final.copy())

    return (
        res,
        np.asarray(cost_history, dtype=float),
        np.vstack(theta_history),
    )


def run_stage1(
    x0: np.ndarray,
    *,
    maxiter: int,
    rhobeg: float,
    cost_fn,
    qcbm,
    target_entropy: float,
) -> dict[str, object]:
    t0 = time.perf_counter()

    result, cost_history, theta_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        minimize_fn=minimize,
        method="COBYLA",
        options={
            "maxiter": int(maxiter),
            "rhobeg": float(rhobeg),
            "disp": True,
        },
    )

    elapsed = time.perf_counter() - t0

    theta_star = np.asarray(result.x, dtype=float)
    p_star = qcbm.probabilities(theta_star)

    ce_final = float(result.fun)
    kl_final = float(ce_final - target_entropy)

    return {
        "result": result,
        "theta_star": theta_star,
        "p_star": p_star,
        "cost_history": np.asarray(cost_history, dtype=float),
        "theta_history": np.asarray(theta_history, dtype=float),
        "elapsed_time": elapsed,
        "ce_final": ce_final,
        "kl_final": kl_final,
    }


def run_stage2(
    x0: np.ndarray,
    *,
    maxiter: int,
    cost_fn,
    qcbm,
    target_entropy: float,
    maxfun: int = 300000,
) -> dict[str, object]:
    t0 = time.perf_counter()

    result, cost_history, theta_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        minimize_fn=minimize,
        method="L-BFGS-B",
    options={
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "ftol": 1e-12,   
                "gtol": 1e-7,      
                "eps": 1e-4,        
                "maxls": 50,       
                "maxcor": 20,       
                "disp": True
            },
    )

    elapsed = time.perf_counter() - t0

    theta_star = np.asarray(result.x, dtype=float)
    p_star = qcbm.probabilities(theta_star)

    ce_final = float(result.fun)
    kl_final = float(ce_final - target_entropy)

    return {
        "result": result,
        "theta_star": theta_star,
        "p_star": p_star,
        "cost_history": np.asarray(cost_history, dtype=float),
        "theta_history": np.asarray(theta_history, dtype=float),
        "elapsed_time": elapsed,
        "ce_final": ce_final,
        "kl_final": kl_final,
    }


def select_checkpoint_indices(
    cost_history: np.ndarray,
    *,
    every: int = 25,
    top_k: int = 20,
    n_recent: int = 10,
) -> np.ndarray:
    """
    Selección híbrida de checkpoints:
    - puntos periódicos a lo largo de toda la trayectoria,
    - mejores puntos por coste,
    - últimos puntos del refinamiento.

    Así no te quedas solo con los "mejores KL", porque a veces en hardware/noise
    gana un punto algo peor en training pero más robusto al resto del pipeline.
    """
    cost_history = np.asarray(cost_history, dtype=float).ravel()
    n = cost_history.size
    if n == 0:
        return np.empty(0, dtype=int)

    idx = {0, n - 1}

    every = max(1, int(every))
    idx.update(range(0, n, every))

    top_k = min(int(top_k), n)
    idx.update(np.argsort(cost_history)[:top_k].tolist())

    n_recent = min(int(n_recent), n)
    idx.update(range(n - n_recent, n))

    return np.array(sorted(idx), dtype=int)

# ===================== Load target =====================
repo_root = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

data = np.load(
    repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz",
    allow_pickle=True,
)

saving_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "quantum"
    / "training"
    / "qcbm"
    / "10q_sv_BEST_snowflake.npz"
)

saving_path.parent.mkdir(parents=True, exist_ok=True)
checkpoint_path = saving_path.with_name(saving_path.stem + "_checkpoints.npz")

ptg = np.asarray(data["p_target"], dtype=float).ravel()
ptg /= ptg.sum()

dim = ptg.size
n_qubits = int(np.log2(dim))
if 2**n_qubits != dim:
    raise ValueError(f"p_target length must be a power of two; got {dim}.")

# ===================== Backend / layout ====================
BACKEND_NAME = "ibm_basquecountry"
SEARCH_TOPOLOGY = "snowflake"
EFFECTIVE_TOPOLOGY = "optimized_snowflake"

service = QiskitRuntimeService(channel="ibm_cloud")
real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

base_layout, layout_score, layout_meta = select_best_layout(
    real_backend,
    topology=SEARCH_TOPOLOGY,
    length=n_qubits,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
)

# Aplicamos el remapeo guiado por Información Mutua para poner q5 en el Hub físico
chosen_layout = [
    base_layout[4],  # q0 (Activo 1)
    base_layout[1],  # q1 (Activo 1)
    base_layout[6],  # q2 (Activo 2)
    base_layout[2],  # q3 (Activo 2)
    base_layout[3],  # q4 (Activo 3)
    base_layout[0],  # q5 (Hub Físico es el Activo 3)
    base_layout[7],  # q6 (Tiempo)
    base_layout[9],  # q7 (Tiempo)
    base_layout[8],  # q8 (Tiempo)
    base_layout[5]   # q9 (Tiempo)
]

print("backend_name      =", BACKEND_NAME)
print("search_topology   =", SEARCH_TOPOLOGY)
print("effective_topology=", EFFECTIVE_TOPOLOGY)
print("base_layout       =", base_layout)
print("chosen_layout     =", chosen_layout)
print("layout_score      =", layout_score)
print("fallback_used     =", layout_meta["fallback_used"])
print("tried             =", layout_meta["tried"])

# ===================== QCBM Setup =====================
N_LAYERS = 6
EPS_COST = 1e-9

qcbm = MLQcbmCircuit(
    n_qubits=n_qubits,
    n_layers=N_LAYERS,
    name="G_p_statevector_transpiled",
    entangler="rzz",
    topology=EFFECTIVE_TOPOLOGY,
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

summarize_circuit(qcbm._tqc, label="QCBM traspilado para entrenamiento")

cost_statevector = qcbm.cost_fn(ptg, eps=EPS_COST)
target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))

# ===================== Training hyperparameters =====================
INIT_SCALE = 2 * np.pi 
SEED = 42

STAGE1_MAXITER = 1500
STAGE1_RHOBEG = 0.5
STAGE2_MAXITER = 10000
STAGE2_MAXFUN = 500000

# checkpoints candidatos para AE/CVA
CHECKPOINT_EVERY = 25
CHECKPOINT_TOP_K = 20
CHECKPOINT_N_RECENT = 10

# ===================== Training Stage 1 =====================
best_run = None

rng = np.random.default_rng(SEED)
x0 = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)

print("Iniciando Stage 1 (Optimización Global)...")
run = run_stage1(
    x0,
    maxiter=STAGE1_MAXITER,
    rhobeg=STAGE1_RHOBEG,
    cost_fn=cost_statevector,
    qcbm=qcbm,
    target_entropy=target_entropy,
)

print(
    f"seed={SEED} | CE={run['ce_final']:.6e} | "
    f"KL={run['kl_final']:.6e} | t={run['elapsed_time']:.2f}s"
)

if best_run is None or run["ce_final"] < best_run["ce_final"]:
    best_run = {**run, "theta_init": x0, "seed": SEED}

# ===================== Training Stage 2 =====================
print("\nIniciando Stage 2 (Refinamiento)...")
refine_run = run_stage2(
    best_run["theta_star"],
    maxiter=STAGE2_MAXITER,
    cost_fn=cost_statevector,
    qcbm=qcbm,
    target_entropy=target_entropy,
    maxfun=STAGE2_MAXFUN,
)

theta_star = refine_run["theta_star"]
p_star = refine_run["p_star"]
p0 = qcbm.probabilities(best_run["theta_init"])

# Evita duplicar el primer punto de Stage 2, que coincide con el final de Stage 1
cost_history = np.r_[
    best_run["cost_history"],
    refine_run["cost_history"][1:],
]

theta_history = np.vstack([
    best_run["theta_history"],
    refine_run["theta_history"][1:],
])

elapsed_time = best_run["elapsed_time"] + refine_run["elapsed_time"]

print("-" * 30)
print("success:", refine_run["result"].success)
print("message:", refine_run["result"].message)
print("final CE:", refine_run["ce_final"])
print("final KL:", refine_run["kl_final"])
print(f"Total elapsed: {elapsed_time:.2f}s")

print("\nComprobación histórico:")
print("stage1 nit:", getattr(best_run["result"], "nit", None))
print("stage1 len(cost_history):", len(best_run["cost_history"]))
print("stage1 len(theta_history):", len(best_run["theta_history"]))
print("stage2 nit:", getattr(refine_run["result"], "nit", None))
print("stage2 len(cost_history):", len(refine_run["cost_history"]))
print("stage2 len(theta_history):", len(refine_run["theta_history"]))
print("total len(cost_history):", len(cost_history))
print("total len(theta_history):", len(theta_history))

# ===================== Candidate checkpoints =====================
checkpoint_idx = select_checkpoint_indices(
    cost_history,
    every=CHECKPOINT_EVERY,
    top_k=CHECKPOINT_TOP_K,
    n_recent=CHECKPOINT_N_RECENT,
)

checkpoint_theta = theta_history[checkpoint_idx]
checkpoint_ce = cost_history[checkpoint_idx]
checkpoint_kl = checkpoint_ce - target_entropy

n_stage1 = len(best_run["cost_history"])
checkpoint_stage = np.where(checkpoint_idx < n_stage1, 1, 2).astype(int)
checkpoint_iter_in_stage = np.where(
    checkpoint_stage == 1,
    checkpoint_idx,
    checkpoint_idx - n_stage1 + 1,
).astype(int)

print("\nResumen checkpoints:")
print("n_checkpoints =", len(checkpoint_idx))
print("checkpoint_idx =", checkpoint_idx.tolist())

# ===================== Metrics =====================
ms = qcbm.metrics(ptg, p_star, eps=EPS_COST)
print("\nMetrics summary:")
print(ms)

# ===================== Plots =====================
rescaled_plot = np.maximum(cost_history - target_entropy, 1e-15)
best_so_far = np.minimum.accumulate(rescaled_plot)
iters = np.arange(1, len(rescaled_plot) + 1)

n_stage1_iters = len(best_run["cost_history"])

fig, ax = plt.subplots(1, 1, figsize=(9, 6))

ax.plot(
    iters,
    rescaled_plot,
    lw=1.0,
    color="gray",
    alpha=0.5,
    zorder=1
)

ax.scatter(
    iters[:n_stage1_iters],
    rescaled_plot[:n_stage1_iters],
    color="blue",
    s=15,
    alpha=0.7,
    label="Stage 1 (COBYLA)",
    zorder=2
)

ax.scatter(
    iters[n_stage1_iters:],
    rescaled_plot[n_stage1_iters:],
    color="red",
    s=15,
    alpha=0.7,
    label="Stage 2 (L-BFGS-B)",
    zorder=2
)

ax.axvline(
    x=n_stage1_iters,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Transition Stage 1 -> 2"
)

ax.plot(
    iters,
    best_so_far,
    lw=2.0,
    color="green",
    linestyle=":",
    label="Best KL so far",
    zorder=3
)

ax.set_yscale("log")
ax.set_xlabel("Evaluation")
ax.set_ylabel(r"$KL(p_{\text{target}} \parallel p_{\theta})$")
ax.set_title("KL evolution during training (Stage 1 vs Stage 2)")
ax.grid(True, which="both", alpha=0.3, linestyle="--")
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()

# ===================== Save training results =====================
np.savez(
    saving_path,
    # Parameters
    theta_star=theta_star,
    theta_init=best_run["theta_init"],

    # Full training dynamics
    best_so_far=best_so_far,
    cost_history=cost_history,
    theta_history=theta_history,

    # Candidate checkpoints for AE/CVA selection
    checkpoint_idx=checkpoint_idx,
    checkpoint_theta=checkpoint_theta,
    checkpoint_ce=checkpoint_ce,
    checkpoint_kl=checkpoint_kl,
    checkpoint_stage=checkpoint_stage,
    checkpoint_iter_in_stage=checkpoint_iter_in_stage,
    checkpoint_every=np.int64(CHECKPOINT_EVERY),
    checkpoint_top_k=np.int64(CHECKPOINT_TOP_K),
    checkpoint_n_recent=np.int64(CHECKPOINT_N_RECENT),

    # Probability distributions
    p_target=ptg,
    p_init=p0,
    p_star=p_star,

    # Timing / iterations
    elapsed_time=np.float64(elapsed_time),
    n_iters=np.int64(len(cost_history)),
    theta_seed=np.int64(SEED),

    # Model metadata
    n_qubits=np.int64(n_qubits),
    n_layers=np.int64(N_LAYERS),
    init_scale=np.float64(INIT_SCALE),
    eps_cost=np.float64(EPS_COST),

    # Backend / layout metadata
    backend_name=np.array(BACKEND_NAME),
    requested_topology=np.array(SEARCH_TOPOLOGY),
    effective_topology=np.array(EFFECTIVE_TOPOLOGY),
    chosen_layout=np.array(chosen_layout, dtype=int),
    layout_score=np.float64(layout_score),
    fallback_used=np.bool_(layout_meta["fallback_used"]),
    tried_layout_search=np.array(layout_meta["tried"], dtype=object),
    transpiled_depth=np.int64(qcbm._tqc.depth()),
    transpiled_size=np.int64(qcbm._tqc.size()),
    transpiled_ops=np.array(dict(qcbm._tqc.count_ops()), dtype=object),

    # Optimizer summaries
    stage1_nit=np.int64(getattr(best_run["result"], "nit", -1)),
    stage2_nit=np.int64(getattr(refine_run["result"], "nit", -1)),
    stage1_elapsed=np.float64(best_run["elapsed_time"]),
    stage2_elapsed=np.float64(refine_run["elapsed_time"]),
    stage1_ce_final=np.float64(best_run["ce_final"]),
    stage1_kl_final=np.float64(best_run["kl_final"]),
    stage2_ce_final=np.float64(refine_run["ce_final"]),
    stage2_kl_final=np.float64(refine_run["kl_final"]),

    # Metrics dict
    metrics=np.array(ms, dtype=object),
)

# opcional: fichero ligero solo con candidatos
np.savez(
    checkpoint_path,
    checkpoint_idx=checkpoint_idx,
    checkpoint_theta=checkpoint_theta,
    checkpoint_ce=checkpoint_ce,
    checkpoint_kl=checkpoint_kl,
    checkpoint_stage=checkpoint_stage,
    checkpoint_iter_in_stage=checkpoint_iter_in_stage,
    theta_star=theta_star,
    p_target=ptg,
    target_entropy=np.float64(target_entropy),
)

print(f"\nResultados guardados en: {saving_path}")
print(f"Checkpoints guardados en: {checkpoint_path}")