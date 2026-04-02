from dataclasses import dataclass
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit_algorithms.optimizers import QNSPSA
from qiskit_algorithms.utils import algorithm_globals

try:
    from qiskit.primitives import StatevectorSampler
except ImportError:
    StatevectorSampler = None

try:
    from qiskit_aer.primitives import SamplerV2 as AerSampler
except ImportError:
    AerSampler = None

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)

# python utils
import time
import numpy as np
from scipy.optimize import minimize

# quantum_cva utils
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    minimize_with_cost_history,
)

@dataclass(frozen=True)
class ShotStageConfig:
    name: str
    maxiter: int
    n_avg: int
    lambda_reg: float
    learning_rate: float
    perturbation: float
    resamplings: int
    regularization: float
    hessian_delay: int


@dataclass(frozen=True)
class TrainingConfig:
    n_layers: int = 12
    eps_cost: float = 1e-9

    init_scale: float = 0.01
    seed: int = 355
    probability_seed: int = 105

    stage1_maxiter: int = 1000
    stage1_rhobeg: float = 0.15
    stage2_maxiter: int = 50

    shots_train: int = 8000
    warm_start_sigma: float = 1e-3
    monitor_every: int = 5

    shot_stages: tuple[ShotStageConfig, ...] = (
        ShotStageConfig(
            name="Local shots stage 1",
            maxiter=120,
            n_avg=2,
            lambda_reg=1e-2,
            learning_rate=2.5e-3,
            perturbation=3.0e-2,
            resamplings=4,
            regularization=5e-2,
            hessian_delay=40,
        ),
        ShotStageConfig(
            name="Local shots stage 2",
            maxiter=120,
            n_avg=4,
            lambda_reg=5e-3,
            learning_rate=1.2e-3,
            perturbation=2.0e-2,
            resamplings=6,
            regularization=5e-2,
            hessian_delay=20,
        ),
    )


@dataclass(frozen=True)
class Paths:
    repo_root: pathlib.Path
    target_path: pathlib.Path
    out_path: pathlib.Path


@dataclass
class ExactKlTracker:
    qcbm: MLQcbmCircuit
    target: np.ndarray
    eps_cost: float
    monitor_every: int

    current_stage_name: str
    surrogate_loss_history: list[float]
    kl_history_exact: list[float]
    kl_eval_iters: list[int]
    stage_boundaries: list[int]
    stage_labels: list[str]

    best_exact_value: float
    best_exact_x: np.ndarray

    @classmethod
    def create(
        cls,
        *,
        qcbm: MLQcbmCircuit,
        target: np.ndarray,
        eps_cost: float,
        monitor_every: int,
        x0: np.ndarray,
    ) -> "ExactKlTracker":
        x0 = np.asarray(x0, dtype=float).copy()
        p0_exact = qcbm.probabilities(x0)
        metrics0 = qcbm.metrics(target, p0_exact, eps=eps_cost)
        best_value = float(metrics0["kl"])

        return cls(
            qcbm=qcbm,
            target=target,
            eps_cost=eps_cost,
            monitor_every=monitor_every,
            current_stage_name="",
            surrogate_loss_history=[],
            kl_history_exact=[best_value],
            kl_eval_iters=[0],
            stage_boundaries=[],
            stage_labels=[],
            best_exact_value=best_value,
            best_exact_x=x0,
        )

    def evaluate_total_kl_exact(self, theta: np.ndarray) -> float:
        p_exact = self.qcbm.probabilities(theta)
        metrics_exact = self.qcbm.metrics(self.target, p_exact, eps=self.eps_cost)
        return float(metrics_exact["kl"])

    def set_stage(self, name: str) -> None:
        self.current_stage_name = name

    def record(self, iter_idx: int, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)

        if self.kl_eval_iters and self.kl_eval_iters[-1] == iter_idx:
            return

        kl_val = self.evaluate_total_kl_exact(theta)

        self.kl_eval_iters.append(iter_idx)
        self.kl_history_exact.append(kl_val)

        if kl_val < self.best_exact_value:
            self.best_exact_value = kl_val
            self.best_exact_x = theta.copy()

        print(
            f"{self.current_stage_name} | "
            f"iter {iter_idx:4d} | "
            f"surrogate={self.surrogate_loss_history[-1]:.6e} | "
            f"KL_total_exact(curr)={kl_val:.6e} | "
            f"KL_total_exact(best)={self.best_exact_value:.6e}"
        )

    def callback(self, nfev, x, fx, stepsize, accepted) -> None:
        _ = (nfev, stepsize, accepted)

        x = np.asarray(x, dtype=float)
        fx = float(fx)

        self.surrogate_loss_history.append(fx)
        iter_idx = len(self.surrogate_loss_history)

        if iter_idx % self.monitor_every == 0:
            self.record(iter_idx, x)
        elif iter_idx % 25 == 0:
            print(
                f"{self.current_stage_name} | "
                f"iter {iter_idx:4d} | surrogate={fx:.6e}"
            )

    def close_stage(self, theta: np.ndarray) -> None:
        self.record(len(self.surrogate_loss_history), theta)
        self.stage_boundaries.append(len(self.surrogate_loss_history))
        self.stage_labels.append(self.current_stage_name)


def build_paths(current_file: str) -> Paths:
    current_path = pathlib.Path(current_file).resolve()

    repo_root = next(
        parent
        for parent in current_path.parents
        if (parent / "pyproject.toml").exists()
    )

    target_path = (
        repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz"
    )

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "quantum"
        / "training"
        / "qcbm"
        / "training_hybrid_statevector_qnspsa_local.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo_root=repo_root,
        target_path=target_path,
        out_path=out_path,
    )


def load_target(target_path: pathlib.Path) -> tuple[np.ndarray, int]:
    data = np.load(target_path, allow_pickle=True)

    p_target = np.asarray(data["p_target"], dtype=float).ravel()
    p_target /= p_target.sum()

    dim = p_target.size
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError(f"p_target length must be a power of two. Got {dim}.")

    return p_target, n_qubits


def build_qcbm(n_qubits: int, n_layers: int) -> MLQcbmCircuit:
    return MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        name="G_p_hybrid_statevector_qnspsa",
        entangler="rxx",
        topology="circular",
    )


def make_regularized_averaged_cost(
    *,
    qcbm: MLQcbmCircuit,
    target: np.ndarray,
    theta_ref: np.ndarray,
    shots: int,
    eps: float,
    n_avg: int,
    lambda_reg: float,
):
    base_cost = qcbm.cost_fn(
        target,
        eps=eps,
        shots=shots,
        seed=None,
        rescaled=True,
        smoothing="dirichlet",
        alpha=1.0,
    )

    theta_ref = np.asarray(theta_ref, dtype=float).copy()

    def cost(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        vals = [float(base_cost(theta)) for _ in range(int(n_avg))]
        ce_avg = float(np.mean(vals))
        reg = float(lambda_reg) * float(np.sum((theta - theta_ref) ** 2))
        return ce_avg + reg

    return cost


def build_fidelity_sampler(shots: int, seed: int):
    if StatevectorSampler is not None:
        return StatevectorSampler()

    if AerSampler is not None:
        try:
            return AerSampler(options={"default_shots": shots, "seed": seed})
        except TypeError:
            try:
                return AerSampler(default_shots=shots, seed=seed)
            except TypeError:
                return AerSampler()

    raise ImportError(
        "Neither qiskit.primitives.StatevectorSampler nor "
        "qiskit_aer.primitives.SamplerV2 is available."
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
    result, cost_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        minimize_fn=minimize,
        method="COBYLA",
        options={"maxiter": int(maxiter), "rhobeg": rhobeg, "disp": True},
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

def run_stage2(
    x0: np.ndarray,
    *,
    maxiter: int,
    cost_fn,
    qcbm,
    target_entropy: float,
    maxfun: int = 50000,
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
            "ftol": 1e-15,
            "gtol": 1e-12,
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

def run_statevector_pretraining(
    *,
    qcbm: MLQcbmCircuit,
    p_target: np.ndarray,
    cfg: TrainingConfig,
) -> dict:
    cost_statevector = qcbm.cost_fn(p_target, eps=cfg.eps_cost)
    target_entropy = -np.sum(
        p_target * np.log(np.clip(p_target, cfg.eps_cost, 1.0))
    )

    rng_sv = np.random.default_rng(cfg.seed)
    theta0_sv = cfg.init_scale * rng_sv.standard_normal(qcbm.n_params).astype(float)

    print("\n=== PHASE A: EXACT STATEVECTOR PRETRAINING ===")

    run1 = run_stage1(
        theta0_sv,
        maxiter=cfg.stage1_maxiter,
        rhobeg=cfg.stage1_rhobeg,
        cost_fn=cost_statevector,
        qcbm=qcbm,
        target_entropy=target_entropy,
    )

    print(
        f"Stage 1 | CE={run1['ce_final']:.6e} | "
        f"KL={run1['kl_final']:.6e} | "
        f"t={run1['elapsed_time']:.2f}s"
    )

    run2 = run_stage2(
        run1["theta_star"],
        maxiter=cfg.stage2_maxiter,
        cost_fn=cost_statevector,
        qcbm=qcbm,
        target_entropy=target_entropy,
        maxfun=30000,
    )

    theta_sv = np.asarray(run2["theta_star"], dtype=float)
    p_sv_exact = qcbm.probabilities(theta_sv)

    sv_cost_history = np.r_[run1["cost_history"], run2["cost_history"]]
    sv_kl_history = np.maximum(sv_cost_history - target_entropy, 1e-16)
    sv_elapsed_time = run1["elapsed_time"] + run2["elapsed_time"]

    metrics_sv_exact = qcbm.metrics(p_target, p_sv_exact, eps=cfg.eps_cost)
    kl_sv_final = float(metrics_sv_exact["kl"])

    print("Statevector success:", run2["result"].success)
    print("Statevector message:", run2["result"].message)
    print("Statevector final exact KL:", kl_sv_final)
    print(f"Statevector elapsed: {sv_elapsed_time:.2f}s")

    return {
        "theta0_sv": theta0_sv,
        "theta_sv": theta_sv,
        "p_sv_exact": p_sv_exact,
        "sv_cost_history": sv_cost_history,
        "sv_kl_history": sv_kl_history,
        "sv_elapsed_time": sv_elapsed_time,
        "metrics_sv_exact": metrics_sv_exact,
        "kl_sv_final": kl_sv_final,
    }


def evaluate_sampled_reference(
    *,
    qcbm: MLQcbmCircuit,
    p_target: np.ndarray,
    theta_sv: np.ndarray,
    cfg: TrainingConfig,
) -> dict:
    p_sv_sampled = qcbm.probabilities(
        theta_sv,
        shots=cfg.shots_train,
        seed=cfg.probability_seed,
    )
    metrics_sv_sampled = qcbm.metrics(
        p_target,
        p_sv_sampled,
        eps=cfg.eps_cost,
    )

    print(
        f"Same theta_sv sampled with {cfg.shots_train} shots | "
        f"sampled KL = {metrics_sv_sampled['kl']:.6e}"
    )

    return {
        "p_sv_sampled": p_sv_sampled,
        "metrics_sv_sampled": metrics_sv_sampled,
    }


def build_shots_warm_start(
    *,
    qcbm: MLQcbmCircuit,
    p_target: np.ndarray,
    theta_sv: np.ndarray,
    cfg: TrainingConfig,
) -> dict:
    algorithm_globals.random_seed = cfg.seed
    rng_shots = np.random.default_rng(cfg.seed)

    x0_shots = theta_sv + cfg.warm_start_sigma * rng_shots.standard_normal(qcbm.n_params)

    p0_exact = qcbm.probabilities(x0_shots)
    p0_sampled = qcbm.probabilities(
        x0_shots,
        shots=cfg.shots_train,
        seed=cfg.probability_seed,
    )

    metrics_p0_exact = qcbm.metrics(p_target, p0_exact, eps=cfg.eps_cost)

    print(
        f"Warm-start exact KL before shots refinement: "
        f"{metrics_p0_exact['kl']:.6e}"
    )

    return {
        "x0_shots": np.asarray(x0_shots, dtype=float),
        "p0_exact": p0_exact,
        "p0_sampled": p0_sampled,
        "metrics_p0_exact": metrics_p0_exact,
    }


def run_local_shots_refinement(
    *,
    qcbm: MLQcbmCircuit,
    p_target: np.ndarray,
    theta_sv: np.ndarray,
    x0_shots: np.ndarray,
    cfg: TrainingConfig,
) -> dict:
    fidelity_sampler = build_fidelity_sampler(
        shots=cfg.shots_train,
        seed=cfg.probability_seed,
    )
    fidelity = QNSPSA.get_fidelity(qcbm.qc, fidelity_sampler)

    tracker = ExactKlTracker.create(
        qcbm=qcbm,
        target=p_target,
        eps_cost=cfg.eps_cost,
        monitor_every=cfg.monitor_every,
        x0=x0_shots,
    )

    theta_current = np.asarray(x0_shots, dtype=float).copy()

    print("\n=== PHASE B: LOCAL SHOTS REFINEMENT AROUND theta_sv ===")

    t0_shots = time.perf_counter()

    for stage_cfg in cfg.shot_stages:
        tracker.set_stage(stage_cfg.name)
        print(f"\n--- {stage_cfg.name} ---")

        stage_cost = make_regularized_averaged_cost(
            qcbm=qcbm,
            target=p_target,
            theta_ref=theta_sv,
            shots=cfg.shots_train,
            eps=cfg.eps_cost,
            n_avg=stage_cfg.n_avg,
            lambda_reg=stage_cfg.lambda_reg,
        )

        optimizer = QNSPSA(
            fidelity=fidelity,
            maxiter=stage_cfg.maxiter,
            learning_rate=stage_cfg.learning_rate,
            perturbation=stage_cfg.perturbation,
            blocking=False,
            allowed_increase=None,
            resamplings=stage_cfg.resamplings,
            regularization=stage_cfg.regularization,
            hessian_delay=stage_cfg.hessian_delay,
            callback=tracker.callback,
        )

        res = optimizer.minimize(
            fun=stage_cost,
            x0=theta_current,
        )

        theta_current = np.asarray(res.x, dtype=float)
        tracker.close_stage(theta_current)

    shots_elapsed_time = time.perf_counter() - t0_shots

    return {
        "theta_last": theta_current.copy(),
        "theta_star": tracker.best_exact_x.copy(),
        "best_exact_kl": tracker.best_exact_value,
        "surrogate_loss_history": np.asarray(tracker.surrogate_loss_history, dtype=float),
        "kl_history_exact": np.asarray(tracker.kl_history_exact, dtype=float),
        "kl_eval_iters": np.asarray(tracker.kl_eval_iters, dtype=int),
        "stage_boundaries": tracker.stage_boundaries,
        "stage_labels": tracker.stage_labels,
        "shots_elapsed_time": shots_elapsed_time,
    }


def compute_final_metrics(
    *,
    qcbm: MLQcbmCircuit,
    p_target: np.ndarray,
    theta_star: np.ndarray,
    cfg: TrainingConfig,
) -> dict:
    p_star_exact = qcbm.probabilities(theta_star)
    p_star_sampled = qcbm.probabilities(
        theta_star,
        shots=cfg.shots_train,
        seed=cfg.probability_seed,
    )

    metrics_final_exact = qcbm.metrics(p_target, p_star_exact, eps=cfg.eps_cost)
    metrics_final_sampled = qcbm.metrics(p_target, p_star_sampled, eps=cfg.eps_cost)

    return {
        "p_star_exact": p_star_exact,
        "p_star_sampled": p_star_sampled,
        "metrics_final_exact": metrics_final_exact,
        "metrics_final_sampled": metrics_final_sampled,
    }


def plot_distributions(
    *,
    target: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    title_before: str,
    title_after: str,
) -> None:
    x = np.arange(target.size)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(x, target, lw=1.6, label="target")
    axes[0].plot(x, before, lw=1.2, label="before")
    axes[0].set_ylabel("Probability")
    axes[0].set_title(title_before)
    axes[0].grid(True, alpha=0.25, linestyle="--")
    axes[0].legend()

    axes[1].plot(x, target, lw=1.6, label="target")
    axes[1].plot(x, after, lw=1.2, label="after")
    axes[1].set_xlabel("Computational basis index")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(title_after)
    axes[1].grid(True, alpha=0.25, linestyle="--")
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def plot_exact_kl_history(
    *,
    kl_eval_iters: np.ndarray,
    kl_history: np.ndarray,
    kl_best_so_far: np.ndarray,
    kl_best_idx: np.ndarray,
    stage_boundaries: list[int],
    stage_labels: list[str],
    kl_statevector_reference: float,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.plot(
        kl_eval_iters,
        kl_history,
        lw=1.2,
        marker="o",
        markersize=3,
        label="Exact total KL(current)",
    )
    ax.plot(
        kl_eval_iters,
        kl_best_so_far,
        lw=2.0,
        label="Exact total KL(best so far)",
    )
    ax.scatter(
        kl_eval_iters[kl_best_idx],
        kl_best_so_far[kl_best_idx],
        s=20,
        zorder=5,
    )

    ax.set_yscale("log")

    ymax = float(np.max(kl_history))
    y_text = ymax if ymax > 0 else 1.0

    for boundary, label in zip(stage_boundaries, stage_labels):
        ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
        ax.text(
            boundary,
            y_text,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            alpha=0.8,
        )

    ax.axhline(
        kl_statevector_reference,
        linestyle=":",
        linewidth=1.8,
        label=f"Statevector reference KL = {kl_statevector_reference:.3e}",
    )

    ax.set_xlabel("Shot-optimizer iteration")
    ax.set_ylabel("Exact total KL(target || p_theta)")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    plt.show()


def save_results(
    *,
    out_path: pathlib.Path,
    cfg: TrainingConfig,
    p_target: np.ndarray,
    sv: dict,
    sampled_ref: dict,
    warm: dict,
    shots: dict,
    final: dict,
) -> None:
    kl_best_so_far = np.minimum.accumulate(shots["kl_history_exact"])
    kl_best_idx = np.flatnonzero(
        np.r_[True, kl_best_so_far[1:] < kl_best_so_far[:-1] - 1e-15]
    )

    np.savez(
        out_path,
        theta_init_statevector=sv["theta0_sv"],
        theta_star_statevector=sv["theta_sv"],
        sv_kl_history=sv["sv_kl_history"],
        sv_elapsed_time=np.float64(sv["sv_elapsed_time"]),
        p_statevector_exact=sv["p_sv_exact"],
        p_statevector_sampled=sampled_ref["p_sv_sampled"],
        metrics_statevector_exact=np.array(sv["metrics_sv_exact"], dtype=object),
        metrics_statevector_sampled=np.array(sampled_ref["metrics_sv_sampled"], dtype=object),
        theta_init_shots=warm["x0_shots"],
        theta_last_shots=shots["theta_last"],
        theta_star=shots["theta_star"],
        surrogate_loss_history=shots["surrogate_loss_history"],
        kl_history_exact=shots["kl_history_exact"],
        kl_eval_iters=shots["kl_eval_iters"],
        kl_best_so_far=kl_best_so_far,
        stage_boundaries=np.asarray(shots["stage_boundaries"], dtype=int),
        p_target=p_target,
        p_init_exact=warm["p0_exact"],
        p_init_sampled=warm["p0_sampled"],
        p_star_exact=final["p_star_exact"],
        p_star_sampled=final["p_star_sampled"],
        shots_elapsed_time=np.float64(shots["shots_elapsed_time"]),
        best_exact_kl=np.float64(shots["best_exact_kl"]),
        shots=np.int64(cfg.shots_train),
        monitor_every=np.int64(cfg.monitor_every),
        warm_start_sigma=np.float64(cfg.warm_start_sigma),
        theta_seed=np.int64(cfg.seed),
        probability_seed=np.int64(cfg.probability_seed),
        optimizer_method=np.array("QNSPSA local refinement around theta_sv"),
        metrics_final_exact=np.array(final["metrics_final_exact"], dtype=object),
        metrics_final_sampled=np.array(final["metrics_final_sampled"], dtype=object),
        kl_best_idx=kl_best_idx,
    )

    print(f"\nSaved training results to: {out_path}")