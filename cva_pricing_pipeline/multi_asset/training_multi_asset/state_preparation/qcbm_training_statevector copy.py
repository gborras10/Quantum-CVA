# python utils
import pathlib
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit


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

ptg_raw = np.asarray(p_target, dtype=float).ravel()
ptg_raw = np.maximum(ptg_raw, 1e-15)
ptg_raw /= ptg_raw.sum()

dim = ptg_raw.size
n_qubits = int(np.log2(dim))
if 2**n_qubits != dim:
    raise ValueError(f"p_target length must be a power of two; got {dim}.")


# =============================================================================
#                          Reorder target by Gray code
# =============================================================================
def gray_code(i: int) -> int:
    return i ^ (i >> 1)

gray_order = np.argsort(np.fromiter((gray_code(i) for i in range(dim)), dtype=np.int64))
inv_gray_order = np.empty_like(gray_order)
inv_gray_order[gray_order] = np.arange(dim)

ptg = ptg_raw[gray_order]


# =============================================================================
#                              Hyperparameters
# =============================================================================
EPS_COST = 1e-12
THETA_SEED = 355

# stage 1 -> smaller model
N_LAYERS_STAGE1 = 6

# stage 2 -> larger model
N_LAYERS_STAGE2 = 12

# temperature curriculum
TAU_SCHEDULE = [0.30, 0.60, 0.90, 1.00]

# SPSA schedule for each tau stage
SPSA_SCHEDULE = [
    (250, 0.10, 0.10, 50.0),
    (300, 0.06, 0.06, 50.0),
    (400, 0.03, 0.03, 60.0),
]

# extra final refinement with smaller steps
FINAL_SPSA_SCHEDULE = [
    (400, 0.015, 0.020, 80.0),
    (500, 0.008, 0.010, 100.0),
]


# =============================================================================
#                              Utilities
# =============================================================================
def project_angles(x: np.ndarray) -> np.ndarray:
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def tempered_target(p: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(p, dtype=float).ravel() ** tau
    x = np.maximum(x, 1e-15)
    return x / x.sum()


def ce_cost_from_p(target: np.ndarray, p: np.ndarray) -> float:
    p_safe = np.clip(p, EPS_COST, 1.0)
    return float(-np.sum(target * np.log(p_safe)))


def kl_value(target: np.ndarray, p: np.ndarray) -> float:
    p_safe = np.clip(p, EPS_COST, 1.0)
    t_safe = np.clip(target, EPS_COST, 1.0)
    return float(np.sum(t_safe * (np.log(t_safe) - np.log(p_safe))))


def best_kl_so_far(hist: list[float]) -> np.ndarray:
    arr = np.asarray(hist, dtype=float)
    return np.minimum.accumulate(arr)


def expand_theta(theta_small: np.ndarray, n_params_big: int, rng: np.random.Generator) -> np.ndarray:
    theta_big = np.zeros(n_params_big, dtype=float)
    n_copy = min(len(theta_small), n_params_big)
    theta_big[:n_copy] = theta_small[:n_copy]
    if n_params_big > n_copy:
        theta_big[n_copy:] = rng.uniform(-0.01, 0.01, size=n_params_big - n_copy)
    return project_angles(theta_big)


# =============================================================================
#                              SPSA optimizer
# =============================================================================
def spsa_optimize(
    qcbm: MLQcbmCircuit,
    target: np.ndarray,
    x0: np.ndarray,
    rng: np.random.Generator,
    schedule: list[tuple[int, float, float, float]],
    alpha: float = 0.602,
    gamma: float = 0.101,
    log_prefix: str = "",
) -> tuple[np.ndarray, list[float], list[float], list[float]]:
    x = project_angles(np.array(x0, dtype=float, copy=True))

    # histories
    ce_history: list[float] = []
    kl_history: list[float] = []
    step_history: list[float] = []

    def loss_fn(theta: np.ndarray) -> float:
        p = qcbm.probabilities(theta)
        return ce_cost_from_p(target, p)

    # initial point
    p0 = qcbm.probabilities(x)
    ce0 = ce_cost_from_p(target, p0)
    kl0 = kl_value(ptg, p0)

    ce_history.append(ce0)
    kl_history.append(kl0)
    step_history.append(0.0)

    best_x = x.copy()
    best_kl = kl0

    global_step = 0
    print(f"{log_prefix} step {global_step:>5} | CE = {ce0:.6e} | KL(full) = {kl0:.6e}", flush=True)

    for phase_idx, (n_steps, a, c, A) in enumerate(schedule, start=1):
        print(
            f"{log_prefix} phase {phase_idx}/{len(schedule)} "
            f"| steps={n_steps} a={a:.3g} c={c:.3g} A={A:.1f}",
            flush=True,
        )

        for k in range(1, n_steps + 1):
            global_step += 1

            ak = a / ((k + A) ** alpha)
            ck = c / (k ** gamma)

            delta = rng.choice(np.array([-1.0, 1.0], dtype=float), size=x.shape)

            x_plus = project_angles(x + ck * delta)
            x_minus = project_angles(x - ck * delta)

            f_plus = loss_fn(x_plus)
            f_minus = loss_fn(x_minus)

            ghat = (f_plus - f_minus) / (2.0 * ck) * (1.0 / delta)
            x = project_angles(x - ak * ghat)

            if global_step % 10 == 0 or global_step == 1:
                p_now = qcbm.probabilities(x)
                ce_now = ce_cost_from_p(target, p_now)
                kl_now = kl_value(ptg, p_now)

                ce_history.append(ce_now)
                kl_history.append(kl_now)
                step_history.append(global_step)

                if kl_now < best_kl:
                    best_kl = kl_now
                    best_x = x.copy()

                if global_step % 50 == 0 or global_step == 1:
                    print(
                        f"{log_prefix} step {global_step:>5} | CE = {ce_now:.6e} | "
                        f"KL(full) = {kl_now:.6e} | best KL = {best_kl:.6e}",
                        flush=True,
                    )

    # final evaluation
    p_end = qcbm.probabilities(x)
    kl_end = kl_value(ptg, p_end)
    if kl_end < best_kl:
        best_x = x.copy()

    return best_x, ce_history, kl_history, step_history


# =============================================================================
#                        Stage 1: 6-layer QCBM
# =============================================================================
rng = np.random.default_rng(THETA_SEED)

qcbm_stage1 = MLQcbmCircuit(
    n_qubits=n_qubits,
    n_layers=N_LAYERS_STAGE1,
    name="G_p_statevector_stage1",
)

print("Stage 1 circuit")
print("n_qubits:", qcbm_stage1.n_qubits)
print("n_layers:", qcbm_stage1.n_layers)
print("n_params:", qcbm_stage1.n_params)

theta_1 = rng.uniform(-0.15, 0.15, size=qcbm_stage1.n_params).astype(float)

global_ce_hist: list[float] = []
global_kl_hist: list[float] = []
global_step_hist: list[float] = []
stage_boundaries: list[tuple[float, str]] = []

t0 = time.perf_counter()

for tau in TAU_SCHEDULE:
    stage_name = f"stage1_tau_{tau:.2f}"
    stage_target = tempered_target(ptg, tau)
    stage_boundaries.append((global_step_hist[-1] if global_step_hist else 0.0, stage_name))

    print("\n" + "=" * 90)
    print(f"Running {stage_name}")
    print("=" * 90)

    theta_1, ce_hist, kl_hist, step_hist = spsa_optimize(
        qcbm=qcbm_stage1,
        target=stage_target,
        x0=theta_1,
        rng=rng,
        schedule=SPSA_SCHEDULE,
        log_prefix="[stage1]",
    )

    step_offset = global_step_hist[-1] if global_step_hist else 0.0
    if global_step_hist:
        step_hist = [step_offset + s for s in step_hist[1:]]
        ce_hist = ce_hist[1:]
        kl_hist = kl_hist[1:]
    global_ce_hist.extend(ce_hist)
    global_kl_hist.extend(kl_hist)
    global_step_hist.extend(step_hist)


# =============================================================================
#                        Stage 2: 12-layer QCBM
# =============================================================================
qcbm_stage2 = MLQcbmCircuit(
    n_qubits=n_qubits,
    n_layers=N_LAYERS_STAGE2,
    name="G_p_statevector_stage2",
)

print("\nStage 2 circuit")
print("n_qubits:", qcbm_stage2.n_qubits)
print("n_layers:", qcbm_stage2.n_layers)
print("n_params:", qcbm_stage2.n_params)

theta_2 = expand_theta(theta_1, qcbm_stage2.n_params, rng)

for tau in TAU_SCHEDULE:
    stage_name = f"stage2_tau_{tau:.2f}"
    stage_target = tempered_target(ptg, tau)
    stage_boundaries.append((global_step_hist[-1] if global_step_hist else 0.0, stage_name))

    print("\n" + "=" * 90)
    print(f"Running {stage_name}")
    print("=" * 90)

    theta_2, ce_hist, kl_hist, step_hist = spsa_optimize(
        qcbm=qcbm_stage2,
        target=stage_target,
        x0=theta_2,
        rng=rng,
        schedule=SPSA_SCHEDULE,
        log_prefix="[stage2]",
    )

    step_offset = global_step_hist[-1] if global_step_hist else 0.0
    step_hist = [step_offset + s for s in step_hist[1:]]
    ce_hist = ce_hist[1:]
    kl_hist = kl_hist[1:]
    global_ce_hist.extend(ce_hist)
    global_kl_hist.extend(kl_hist)
    global_step_hist.extend(step_hist)


# =============================================================================
#                      Final full-target refinement
# =============================================================================
stage_boundaries.append((global_step_hist[-1] if global_step_hist else 0.0, "final_full_refine"))

print("\n" + "=" * 90)
print("Running final full-target refinement")
print("=" * 90)

theta_star, ce_hist, kl_hist, step_hist = spsa_optimize(
    qcbm=qcbm_stage2,
    target=ptg,
    x0=theta_2,
    rng=rng,
    schedule=FINAL_SPSA_SCHEDULE,
    log_prefix="[final ]",
)

step_offset = global_step_hist[-1] if global_step_hist else 0.0
step_hist = [step_offset + s for s in step_hist[1:]]
ce_hist = ce_hist[1:]
kl_hist = kl_hist[1:]
global_ce_hist.extend(ce_hist)
global_kl_hist.extend(kl_hist)
global_step_hist.extend(step_hist)

elapsed_time = time.perf_counter() - t0


# =============================================================================
#                          Final probabilities / metrics
# =============================================================================
p_star_gray = qcbm_stage2.probabilities(theta_star)
p_star = np.empty_like(p_star_gray)
p_star[gray_order] = p_star_gray[inv_gray_order]  # placeholder to preserve shape
# correct inverse mapping
p_star = np.empty_like(p_star_gray)
p_star[gray_order] = p_star_gray
p_star_original = p_star

final_kl = kl_value(ptg_raw, p_star_original)
final_ce = ce_cost_from_p(ptg_raw, p_star_original)
ms = qcbm_stage2.metrics(ptg_raw, p_star_original, eps=EPS_COST)

print("\n=== FINAL SUMMARY ===")
print("final CE:", final_ce)
print("final KL(ptg || p*):", final_kl)
print(f"elapsed time: {elapsed_time:.2f} s")

print("\n=== FIT METRICS ===")
print("KL(ptg || p*)  =", float(ms["kl"]))
print("L1             =", float(ms["l1"]))
print("TV = 0.5*L1    =", float(ms["tv"]))
print("Linf           =", float(ms["linf"]))


# =============================================================================
#                               Diagnostics
# =============================================================================
global_step_arr = np.asarray(global_step_hist, dtype=float)
global_kl_arr = np.asarray(global_kl_hist, dtype=float)
global_ce_arr = np.asarray(global_ce_hist, dtype=float)

fig_kl, ax_kl = plt.subplots(figsize=(9, 4.5))
ax_kl.plot(global_step_arr, np.maximum(global_kl_arr, 1e-15), lw=1.5, label="KL(full)")
ax_kl.plot(global_step_arr, np.maximum(best_kl_so_far(global_kl_hist), 1e-15), lw=1.2, label="best KL so far")
ax_kl.set_yscale("log")
ax_kl.set_xlabel("SPSA step")
ax_kl.set_ylabel("KL(ptg || p)")
ax_kl.set_title("KL convergence with 2-stage QCBM + temperature curriculum")
for xpos, name in stage_boundaries[1:]:
    ax_kl.axvline(x=xpos, color="gray", linestyle="--", alpha=0.35)
ax_kl.legend()
ax_kl.grid(True, which="both", alpha=0.3, linestyle="--")
fig_kl.tight_layout()

top_k = min(30, dim)
top_idx = np.argsort(ptg_raw)[-top_k:][::-1]
labels = [format(i, f"0{n_qubits}b") for i in top_idx]

fig_dist, ax_dist = plt.subplots(figsize=(12, 5))
x = np.arange(top_k)
width = 0.4
ax_dist.bar(x - width / 2, ptg_raw[top_idx], width=width, label="ptg", alpha=0.8)
ax_dist.bar(x + width / 2, p_star_original[top_idx], width=width, label="p_star", alpha=0.8)
ax_dist.set_xticks(x)
ax_dist.set_xticklabels(labels, rotation=90)
ax_dist.set_ylabel("Probability")
ax_dist.set_title(f"Top {top_k} most probable states")
ax_dist.legend()
ax_dist.grid(True, axis="y", alpha=0.3, linestyle="--")
fig_dist.tight_layout()

fig_scatter, ax_scatter = plt.subplots(figsize=(6, 6))
ax_scatter.scatter(ptg_raw, p_star_original, s=8, alpha=0.6)
xy_max = float(max(ptg_raw.max(), p_star_original.max()))
ax_scatter.plot([0.0, xy_max], [0.0, xy_max], linestyle="--", linewidth=1.0)
ax_scatter.set_xlabel("ptg")
ax_scatter.set_ylabel("p_star")
ax_scatter.set_title("Target vs learned distribution")
ax_scatter.grid(True, alpha=0.3, linestyle="--")
fig_scatter.tight_layout()

plt.show()


# =============================================================================
#                              Save results
# =============================================================================
out_dir = repo_root / "data" / "multi_asset" / "quantum" / "training" / "qcbm"
out_dir.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    out_dir / "qcbm_statevector_training_results_ambitious_spsa.npz",
    theta_star=theta_star,
    p_star=p_star_original,
    ptg=ptg_raw,
    gray_order=gray_order,
    global_step_history=global_step_arr,
    global_ce_history=global_ce_arr,
    global_kl_history=global_kl_arr,
    metrics=ms,
    training_time=elapsed_time,
    n_layers_stage1=N_LAYERS_STAGE1,
    n_layers_stage2=N_LAYERS_STAGE2,
    tau_schedule=np.asarray(TAU_SCHEDULE, dtype=float),
    theta_seed=THETA_SEED,
    final_ce=float(final_ce),
    final_kl=float(final_kl),
)

print(f"\nResults saved to {out_dir / 'qcbm_statevector_training_results_ambitious_spsa.npz'}")