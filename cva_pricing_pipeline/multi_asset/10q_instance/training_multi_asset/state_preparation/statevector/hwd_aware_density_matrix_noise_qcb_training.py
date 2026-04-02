# training_density_matrix_refine_from_statevector.py
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

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

    cost_history: list[float] = [float(cost_fn(x0))]

    def wrapped(x):
        x = np.asarray(x, dtype=float)
        return float(cost_fn(x))

    def callback(xk):
        xk = np.asarray(xk, dtype=float)
        cost_history.append(float(cost_fn(xk)))

    res = minimize_fn(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    f_final = float(res.fun)
    if abs(cost_history[-1] - f_final) > 1e-15:
        cost_history.append(f_final)

    return res, np.asarray(cost_history, dtype=float)


def run_cobyla_refinement(
    x0: np.ndarray,
    *,
    maxiter: int,
    rhobeg: float,
    cost_fn,
    qcbm,
    target_entropy: float,
) -> dict[str, object]:
    t0 = time.perf_counter()

    result, cost_history = minimize_with_cost_history(
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
        "elapsed_time": elapsed,
        "ce_final": ce_final,
        "kl_final": kl_final,
    }


def run_lbfgsb_refinement(
    x0: np.ndarray,
    *,
    maxiter: int,
    maxfun: int,
    cost_fn,
    qcbm,
    target_entropy: float,
) -> dict[str, object]:
    t0 = time.perf_counter()

    result, cost_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        minimize_fn=minimize,
        method="L-BFGS-B",
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "ftol": 1e-10,
            "gtol": 1e-6,
            "eps": 1e-3,
            "maxls": 20,
            "maxcor": 10,
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
        "elapsed_time": elapsed,
        "ce_final": ce_final,
        "kl_final": kl_final,
    }


def ema_log(y: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    z = np.log(np.clip(y, 1e-15, None))

    out = np.empty_like(z)
    out[0] = z[0]
    for i in range(1, len(z)):
        out[i] = alpha * z[i] + (1.0 - alpha) * out[i - 1]

    return np.exp(out)


# ===================== Paths =====================
repo_root = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

target_path = (
    repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz"
)

statevector_results_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "quantum"
    / "training"
    / "qcbm"
    / "10qbits_sv_BEST.npz"
)

saving_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "quantum"
    / "training"
    / "qcbm"
    / "10q_dm_real_noise_refined_from_sv.npz"
)
saving_path.parent.mkdir(parents=True, exist_ok=True)


# ===================== Load target =====================
data = np.load(target_path, allow_pickle=True)
ptg = np.asarray(data["p_target"], dtype=float).ravel()
ptg /= ptg.sum()

dim = ptg.size
n_qubits = int(np.log2(dim))
if 2**n_qubits != dim:
    raise ValueError(f"p_target length must be a power of two; got {dim}.")


# ===================== Load statevector solution =====================
sv_data = np.load(statevector_results_path, allow_pickle=True)

theta_sv = np.asarray(sv_data["theta_star"], dtype=float)
p_sv = np.asarray(sv_data["p_star"], dtype=float)

if "n_layers" not in sv_data:
    raise KeyError("El fichero de statevector no contiene 'n_layers'.")
if "backend_name" in sv_data:
    print("backend_name en SV =", sv_data["backend_name"])
if "effective_topology" in sv_data:
    print("effective_topology en SV =", sv_data["effective_topology"])

N_LAYERS = int(sv_data["n_layers"])


# ===================== Backend / layout =====================
BACKEND_NAME = "ibm_basquecountry"
LOGICAL_TOPOLOGY = "circular"

service = QiskitRuntimeService(channel="ibm_cloud")
real_backend = service.backend(BACKEND_NAME)
real_noise_model = NoiseModel.from_backend(real_backend)

chosen_layout, layout_score, layout_meta = select_best_layout(
    real_backend,
    topology=LOGICAL_TOPOLOGY,
    length=n_qubits,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
)

effective_topology = layout_meta["selected_topology"]

print("backend_name      =", BACKEND_NAME)
print("requested_topology=", LOGICAL_TOPOLOGY)
print("effective_topology=", effective_topology)
print("chosen_layout     =", chosen_layout)
print("layout_score      =", layout_score)
print("fallback_used     =", layout_meta["fallback_used"])
print("tried             =", layout_meta["tried"])


# ===================== QCBM Setup =====================
EPS_COST = 1e-9

qcbm = MLQcbmCircuit(
    n_qubits=n_qubits,
    n_layers=N_LAYERS,
    name="G_p_density_matrix_transpiled_real_noise_refined_from_sv",
    entangler="cz",
    topology=effective_topology,
    backend=AerSimulator(
        method="density_matrix",
        noise_model=real_noise_model,
    ),
    transpile_backend=real_backend,
    noise_model=real_noise_model,
    simulation_method="density_matrix",
    optimization_level=3,
    initial_layout=chosen_layout,
    layout_method="trivial",
    routing_method="none",
    seed_transpiler=1234,
)

summarize_circuit(
    qcbm._tqc,
    label="QCBM traspilado (density matrix + ruido real, refine from sv)",
)

if theta_sv.size != qcbm.n_params:
    raise ValueError(
        f"theta_star de statevector tiene tamaño {theta_sv.size}, "
        f"pero el QCBM actual espera {qcbm.n_params} parámetros."
    )

cost_density_matrix_noisy = qcbm.cost_fn(ptg, eps=EPS_COST)
target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))


# ===================== Quick timing probe =====================
print("\nMidiendo tiempo de una evaluación ruidosa...")
t_probe_0 = time.perf_counter()
c_probe = float(cost_density_matrix_noisy(theta_sv))
t_probe = time.perf_counter() - t_probe_0
print(f"cost(theta_sv) = {c_probe:.6e}")
print(f"single noisy cost eval = {t_probe:.3f} s")


# ===================== Hyperparameters =====================
# Refinamiento corto y realista partiendo de la solución en statevector.
RUN_LBFGSB_STAGE = False

COBYLA_MAXITER = 120
COBYLA_RHOBEG = 0.12

LBFGSB_MAXITER = 25
LBFGSB_MAXFUN = 1500


# ===================== Baseline at statevector point =====================
p_dm_init = qcbm.probabilities(theta_sv)
ms_init = qcbm.metrics(ptg, p_dm_init, eps=EPS_COST)
ce_init = float(cost_density_matrix_noisy(theta_sv))
kl_init = float(ce_init - target_entropy)

print("\nEstado inicial (theta_star de statevector evaluado con ruido):")
print("CE init:", ce_init)
print("KL init:", kl_init)
print("Metrics init:", ms_init)


# ===================== Stage 1: COBYLA refine =====================
print("\nIniciando refinamiento ruidoso con COBYLA...")
stage1 = run_cobyla_refinement(
    theta_sv,
    maxiter=COBYLA_MAXITER,
    rhobeg=COBYLA_RHOBEG,
    cost_fn=cost_density_matrix_noisy,
    qcbm=qcbm,
    target_entropy=target_entropy,
)

theta_after_stage1 = stage1["theta_star"]
p_after_stage1 = stage1["p_star"]


# ===================== Optional Stage 2: short L-BFGS-B refine =====================
if RUN_LBFGSB_STAGE:
    print("\nIniciando refinamiento adicional con L-BFGS-B...")
    stage2 = run_lbfgsb_refinement(
        theta_after_stage1,
        maxiter=LBFGSB_MAXITER,
        maxfun=LBFGSB_MAXFUN,
        cost_fn=cost_density_matrix_noisy,
        qcbm=qcbm,
        target_entropy=target_entropy,
    )

    theta_star = stage2["theta_star"]
    p_star = stage2["p_star"]

    cost_history = np.r_[
        stage1["cost_history"],
        stage2["cost_history"][1:],
    ]

    elapsed_time = stage1["elapsed_time"] + stage2["elapsed_time"]
else:
    stage2 = None
    theta_star = theta_after_stage1
    p_star = p_after_stage1
    cost_history = np.asarray(stage1["cost_history"], dtype=float)
    elapsed_time = float(stage1["elapsed_time"])


print("\nResumen final:")
print("-" * 40)
print("elapsed_time total:", elapsed_time)
print("stage1 success:", stage1["result"].success)
print("stage1 message:", stage1["result"].message)
print("stage1 CE final:", stage1["ce_final"])
print("stage1 KL final:", stage1["kl_final"])

if stage2 is not None:
    print("stage2 success:", stage2["result"].success)
    print("stage2 message:", stage2["result"].message)
    print("stage2 CE final:", stage2["ce_final"])
    print("stage2 KL final:", stage2["kl_final"])

print("len(cost_history):", len(cost_history))


# ===================== Final metrics =====================
ms_final = qcbm.metrics(ptg, p_star, eps=EPS_COST)
print("\nMetrics final:")
print(ms_final)


# ===================== Curves for plotting =====================
rescaled_cost_history = np.maximum(cost_history - target_entropy, 1e-15)
best_so_far = np.minimum.accumulate(rescaled_cost_history)
display_curve = ema_log(rescaled_cost_history, alpha=0.12)

iters = np.arange(1, len(rescaled_cost_history) + 1)
n_stage1 = len(stage1["cost_history"])


# ===================== Plot =====================
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

ax.plot(
    iters,
    rescaled_cost_history,
    lw=0.8,
    alpha=0.12,
    label="_nolegend_",
)

ax.plot(
    iters,
    display_curve,
    lw=1.8,
    alpha=0.95,
    label="CE - H(ptg)",
)

ax.plot(
    iters,
    best_so_far,
    lw=2.0,
    color="red",
    label="best so far",
)

if stage2 is not None:
    ax.axvline(
        n_stage1,
        color="gray",
        linestyle="--",
        lw=1.2,
        alpha=0.8,
        label="fin stage 1",
    )

ax.set_yscale("log")
ax.set_xlabel("Iteración")
ax.set_ylabel("CE - H(ptg)")
ax.set_title("Refinamiento desde statevector en density matrix con ruido real")
ax.grid(True, which="both", alpha=0.3, linestyle="--")
ax.legend()
fig.tight_layout()
plt.show()


# ===================== Save training results =====================
np.savez_compressed(
    saving_path,
    # parameters
    theta_sv=theta_sv,
    theta_star=theta_star,
    # distributions
    p_target=ptg,
    p_sv=p_sv,
    p_dm_init=p_dm_init,
    p_star=p_star,
    # costs / metrics at start
    ce_init=np.float64(ce_init),
    kl_init=np.float64(kl_init),
    metrics_init=np.array(ms_init, dtype=object),
    # training histories
    cost_history=cost_history,
    cost_history_stage1=stage1["cost_history"],
    cost_history_stage2=(
        np.asarray(stage2["cost_history"], dtype=float)
        if stage2 is not None
        else np.array([], dtype=float)
    ),
    rescaled_cost_history=rescaled_cost_history,
    best_so_far=best_so_far,
    display_curve=display_curve,
    target_entropy=np.float64(target_entropy),
    # optimizer summaries
    used_lbfgsb_stage=np.bool_(stage2 is not None),
    stage1_nit=np.int64(getattr(stage1["result"], "nit", -1)),
    stage1_elapsed=np.float64(stage1["elapsed_time"]),
    stage1_ce_final=np.float64(stage1["ce_final"]),
    stage1_kl_final=np.float64(stage1["kl_final"]),
    stage2_nit=np.int64(
        getattr(stage2["result"], "nit", -1) if stage2 is not None else -1
    ),
    stage2_elapsed=np.float64(stage2["elapsed_time"] if stage2 is not None else 0.0),
    stage2_ce_final=np.float64(stage2["ce_final"] if stage2 is not None else np.nan),
    stage2_kl_final=np.float64(stage2["kl_final"] if stage2 is not None else np.nan),
    # timing
    probe_eval_time_s=np.float64(t_probe),
    elapsed_time=np.float64(elapsed_time),
    n_iters=np.int64(len(cost_history)),
    # model metadata
    n_qubits=np.int64(n_qubits),
    n_layers=np.int64(N_LAYERS),
    eps_cost=np.float64(EPS_COST),
    simulation_method=np.array("density_matrix"),
    entangler=np.array("cz"),
    # backend / layout metadata
    backend_name=np.array(BACKEND_NAME),
    requested_topology=np.array(LOGICAL_TOPOLOGY),
    effective_topology=np.array(effective_topology),
    chosen_layout=np.array(chosen_layout, dtype=int),
    layout_score=np.float64(layout_score),
    fallback_used=np.bool_(layout_meta["fallback_used"]),
    tried_layout_search=np.array(layout_meta["tried"], dtype=object),
    transpiled_depth=np.int64(qcbm._tqc.depth()),
    transpiled_size=np.int64(qcbm._tqc.size()),
    transpiled_ops=np.array(dict(qcbm._tqc.count_ops()), dtype=object),
    noise_model_basis_gates=np.array(
        list(getattr(real_noise_model, "basis_gates", [])),
        dtype=object,
    ),
    # final metrics
    metrics_final=np.array(ms_final, dtype=object),
    # provenance
    source_statevector_file=np.array(str(statevector_results_path)),
)

print(f"\nResultados guardados en: {saving_path}")
print("Claves para plotear luego:")
print(
    [
        "cost_history",
        "cost_history_stage1",
        "cost_history_stage2",
        "rescaled_cost_history",
        "best_so_far",
        "display_curve",
    ]
)