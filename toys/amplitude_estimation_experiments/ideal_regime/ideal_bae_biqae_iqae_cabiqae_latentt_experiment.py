import os
import sys
from typing import Any

import numpy as np

# Add source directory to path.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE
from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
    StandaloneBAEHardware as StandaloneBAE,
)
from toys.amplitude_estimation_experiments.common_utils.experiment_utils import (
    ContrastDecaySampler,
    build_problem,
)
from toys.amplitude_estimation_experiments.common_utils.plotting_utils import (
    plot_query_benchmark_with_confidence_bands,
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

ALGORITHM_CONFIG = {
    "bae": {
        "epsilon_target": 5e-5,
        "n_shots": None,
        "max_queries": 3e5,
        "estimate_T": False,
    },
    "biqae": {
        "epsilon_target": 5e-5,
        "n_shots": None,
        "max_queries": None,
    },
    "iqae": {
        "epsilon_target": 5e-5,
        "n_shots": None,
        "max_queries": None,
    },
    "cabiqae_latentt": {
        "epsilon_target": 5e-5,
        "n_shots": None,
        "max_queries": None,
    },
}


def _effective_n_shots(algorithm: str, configured_n_shots: int | None) -> int:
    if configured_n_shots is not None:
        return int(configured_n_shots)
    if algorithm == "bae":
        return 20
    return 10


def _build_solver(
    algorithm: str,
    epsilon_target: float,
    alpha: float,
    n_shots: int | None,
    seed: int,
    estimate_T: bool = False,
) -> tuple[Any, bool]:
    ideal_sampler = ContrastDecaySampler(T=None, seed=seed)

    if algorithm == "bae":
        bae_kwargs: dict[str, Any] = {
            "epsilon_target": epsilon_target,
            "alpha": alpha,
            "sampler": ideal_sampler,
            "noise_model": "ideal",
            "T_known": None,
            "cap_kappa": 2.0,
            "estimate_T": estimate_T,
            "T_range": None,
            "TNs": 0,
            "wNs": 100,
        }
        if n_shots is not None:
            bae_kwargs["Ns"] = int(n_shots)
        return (
            StandaloneBAE(**bae_kwargs),
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
    n_shots: int | None,
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
    effective_n_shots = _effective_n_shots(algorithm, n_shots)
    queries = np.cumsum(effective_n_shots * (2.0 * powers[:usable] + 1.0))
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
    random amplitudes.
    """
    n_rep = 500    
    a_range = (0.1, 0.4)

    alpha = 0.05
    max_queries = 1e6
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
            configured_n_shots = (
                None if alg_cfg["n_shots"] is None else int(alg_cfg["n_shots"])
            )
            seed = int(1_000_000 + rep * 100 + alg_idx)
            solver, bayes = _build_solver(
                algorithm=algorithm,
                epsilon_target=float(alg_cfg["epsilon_target"]),
                alpha=alpha,
                n_shots=configured_n_shots,
                seed=seed,
                estimate_T=bool(alg_cfg.get("estimate_T", False)),
            )

            try:
                if algorithm == "bae":
                    np.random.seed(seed)
                    estimate_kwargs: dict[str, Any] = {
                        "max_queries": int(alg_cfg["max_queries"] or max_queries),
                    }
                    if configured_n_shots is not None:
                        estimate_kwargs["n_shots"] = configured_n_shots
                    result = solver.estimate(problem, **estimate_kwargs)
                else:
                    estimate_kwargs = {
                        "bayes": bayes,
                        "show_details": False,
                    }
                    if configured_n_shots is not None:
                        estimate_kwargs["n_shots"] = configured_n_shots
                    result = solver.estimate(problem, **estimate_kwargs)
            except Exception as exc:
                print(
                    f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                    f"failed ({exc})"
                )
                continue

            queries, estimations = _extract_trace(algorithm, result, configured_n_shots)
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

    out_path = os.path.join(current_dir, "bae_biqae_iqae_cabiqae_latentt_ideal_rmse.png")
    rmse_curves = {
        algorithm: np.sqrt(interpolated_nsqe[algorithm])
        for algorithm in ALGORITHMS
    }
    plot_query_benchmark_with_confidence_bands(
        query_grid=query_grid,
        curves_by_algorithm=rmse_curves,
        algorithms=ALGORITHMS,
        algorithm_labels=ALGORITHM_LABELS,
        algorithm_styles=ALGORITHM_STYLES,
        output_path=out_path,
        ylabel="Median normalized RMSE",
        title="Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE_latentt",
        confidence_level=0.95,
        statistic="median",
        bootstrap_samples=1000,
        show=True,
    )
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    run_experiment()
