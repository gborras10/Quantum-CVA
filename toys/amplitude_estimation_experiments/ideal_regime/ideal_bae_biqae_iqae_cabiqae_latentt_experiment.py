import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add source directory to path.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from quantum_cva.algorithms.proposed_algorithms.cabiae_known_t_latent_theta import CABIQAELatentTheta
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE
from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
    StandaloneBAEHardware as StandaloneBAE,
)
from toys.amplitude_estimation_experiments.common_utils.experiment_utils import (
    ContrastDecaySampler,
    build_problem,
)


ALGORITHMS = ("bae", "biqae", "iqae", "cabiqae_latentt")
ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "iqae": "IQAE",
    "cabiqae_latentt": "CABIQAE_latentt",
}
ALGORITHM_STYLES = {
    "bae": {"color": "#1D3557", "marker": "o"},
    "biqae": {"color": "#E76F51", "marker": "s"},
    "iqae": {"color": "#457B9D", "marker": "D"},
    "cabiqae_latentt": {"color": "#2A9D8F", "marker": "^"},
}

# Keep the same experiment shape as the noisy script, but force ideal dynamics.
# Tuned to push all methods deeper into the 1e6-query regime.
ALGORITHM_CONFIG = {
    "bae": {
        "epsilon_target": 5e-5,
        "n_shots": 5,
        "max_queries": 1_000_000,
        "estimate_T": False,
    },
    "biqae": {
        "epsilon_target": 5e-5,
        "n_shots": 5,
        "max_queries": None,
    },
    "iqae": {
        "epsilon_target": 5e-5,
        "n_shots": 5,
        "max_queries": None,
    },
    "cabiqae_latentt": {
        "epsilon_target": 5e-5,
        "n_shots": 100,
        "max_queries": None,
    },
}


def _safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    valid_counts = np.sum(~np.isnan(values), axis=axis)
    summed = np.nansum(values, axis=axis)
    means = np.full(summed.shape, np.nan, dtype=float)
    np.divide(summed, valid_counts, out=means, where=valid_counts > 0)
    return means


def _build_solver(
    algorithm: str,
    epsilon_target: float,
    alpha: float,
    n_shots: int,
    seed: int,
    estimate_T: bool = False,
) -> tuple[Any, bool]:
    ideal_sampler = ContrastDecaySampler(T=None, seed=seed)

    if algorithm == "bae":
        return (
            StandaloneBAE(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=ideal_sampler,
                noise_model="ideal",
                T_known=None,
                cap_kappa=2.0,
                estimate_T=estimate_T,
                T_range=None,
                TNs=0,
                wNs=100,
                Ns=n_shots,
            ),
            True,
        )

    if algorithm == "biqae":
        return (
            BIQAE(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=ideal_sampler,
                min_ratio=2,
                confint_method="beta",
            ),
            True,
        )

    if algorithm == "iqae":
        return (
            BIQAE(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=ideal_sampler,
                min_ratio=2,
                confint_method="beta",
            ),
            False,
        )

    if algorithm == "cabiqae_latentt":
        return (
            CABIQAELatentTheta(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=ideal_sampler,
                min_ratio=2,
                confint_method="beta",
                noise_model="ideal",
                T_known=None,
                cap_kappa=2.0,
                use_noise_cap=True,
            ),
            True,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _extract_trace(
    algorithm: str,
    result: Any,
    n_shots: int,
) -> tuple[np.ndarray, np.ndarray]:
    if algorithm == "bae":
        history = getattr(result, "history", {}) or {}
        queries = np.asarray(history.get("queries", []), dtype=float)
        estimations = np.asarray(history.get("estimations", []), dtype=float)
        usable = min(len(queries), len(estimations))
        return queries[:usable], estimations[:usable]

    powers = np.asarray(getattr(result, "powers", []) or [], dtype=float)
    estimate_intervals = getattr(result, "estimate_intervals", []) or []
    usable = min(len(powers), max(0, len(estimate_intervals) - 1))
    if usable <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    # estimate_intervals includes an initial [0, 1] interval before the first shot.
    interval_array = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
    if interval_array.ndim != 2 or interval_array.shape[1] != 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    estimations = np.mean(interval_array, axis=1)
    queries = np.cumsum(n_shots * (2.0 * powers[:usable] + 1.0))
    return queries.astype(float), estimations.astype(float)


def _interpolate_nsqe(
    queries: np.ndarray,
    normalized_sq_errors: np.ndarray,
    query_grid: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(queries, query_grid, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(queries) - 1)

    curve = np.full(len(query_grid), np.nan, dtype=float)
    curve[valid] = normalized_sq_errors[idx[valid]]
    return curve


def run_experiment() -> None:
    """
    Compare BAE, BIQAE, IQAE and CABIQAE_latentt in an ideal regime with
    random amplitudes, matching the same experiment flow used in the noisy script.
    """
    n_rep = 20
    a_range = (0.05, 0.95)

    alpha = 0.05
    max_queries = 10**6
    query_grid = np.logspace(2, np.log10(max_queries), num=120)

    interpolated_nsqe = {
        algorithm: np.full((n_rep, len(query_grid)), np.nan, dtype=float)
        for algorithm in ALGORITHMS
    }
    rng = np.random.default_rng(1234)

    print(
        "Running ideal amplitude-estimation benchmark "
        f"({', '.join(ALGORITHM_LABELS[a] for a in ALGORITHMS)}) "
        f"with {n_rep} repetitions..."
    )
    print(
        "Algorithm budgets: "
        + ", ".join(
            (
                f"{ALGORITHM_LABELS[a]}(eps={ALGORITHM_CONFIG[a]['epsilon_target']}, "
                f"shots={ALGORITHM_CONFIG[a]['n_shots']}, "
                f"max_q={ALGORITHM_CONFIG[a]['max_queries']})"
            )
            for a in ALGORITHMS
        )
    )

    for rep in range(n_rep):
        target_a = float(rng.uniform(*a_range))
        problem = build_problem(target_a)

        for alg_idx, algorithm in enumerate(ALGORITHMS):
            alg_cfg = ALGORITHM_CONFIG[algorithm]
            seed = int(1_000_000 + rep * 100 + alg_idx)
            solver, bayes = _build_solver(
                algorithm=algorithm,
                epsilon_target=float(alg_cfg["epsilon_target"]),
                alpha=alpha,
                n_shots=int(alg_cfg["n_shots"]),
                seed=seed,
                estimate_T=bool(alg_cfg.get("estimate_T", False)),
            )

            try:
                if algorithm == "bae":
                    # Legacy BAE internals rely on NumPy's global RNG.
                    np.random.seed(seed)
                    result = solver.estimate(
                        problem,
                        n_shots=int(alg_cfg["n_shots"]),
                        max_queries=int(alg_cfg["max_queries"] or max_queries),
                    )
                else:
                    result = solver.estimate(
                        problem,
                        bayes=bayes,
                        show_details=False,
                        n_shots=int(alg_cfg["n_shots"]),
                    )
            except Exception as exc:
                print(
                    f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                    f"failed ({exc})"
                )
                continue

            queries, estimations = _extract_trace(algorithm, result, int(alg_cfg["n_shots"]))
            if len(queries) == 0:
                print(
                    f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                    "no trajectory returned"
                )
                continue

            normalized_sq_errors = np.square((estimations / target_a) - 1.0)
            rep_curve = _interpolate_nsqe(queries, normalized_sq_errors, query_grid)
            interpolated_nsqe[algorithm][rep, :] = rep_curve

            print(
                f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                f"a={target_a:.3f}, "
                f"final nRMSE={np.sqrt(normalized_sq_errors[-1]):.3e}, "
                f"queries={int(queries[-1])}"
            )

    sqrt_limit = 1.0 / np.sqrt(query_grid)
    heisenberg_like = 3.0 / query_grid

    plt.figure(figsize=(10, 6))
    for algorithm in ALGORITHMS:
        mean_nsqe = _safe_nanmean(interpolated_nsqe[algorithm], axis=0)
        mean_rmse = np.sqrt(mean_nsqe)
        style = ALGORITHM_STYLES[algorithm]
        plt.loglog(
            query_grid,
            mean_rmse,
            color=style["color"],
            marker=style["marker"],
            markersize=3,
            markevery=8,
            linewidth=2.0,
            label=f"{ALGORITHM_LABELS[algorithm]} (mean)",
        )

    plt.loglog(query_grid, sqrt_limit, "--", color="#6D6875", label=r"$\mathcal{O}(1/\sqrt{N_q})$")
    plt.loglog(query_grid, heisenberg_like, "-.", color="#6D6875", label=r"$\mathcal{O}(1/N_q)$")

    plt.grid(True, which="both", ls="--", alpha=0.35)
    plt.xlabel("Cumulative query count")
    plt.ylabel("Normalized RMSE")
    plt.title("Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE_latentt")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(current_dir, "bae_biqae_iqae_cabiqae_latentt_ideal_rmse.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    run_experiment()
