from __future__ import annotations

import csv
import os
import sys
from typing import Any
from unittest import result

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")

for path in [src_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
try:
    from quantum_cva.algorithms.proposed_algorithms.cabiae_known_t_latent_theta import (
        CABIQAELatentTheta,
    )
    from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE
    from quantum_cva.algorithms.third_party.standalone_bae import (
        StandaloneBAE as StandaloneBAELegacy,
    )
    from toys.amplitude_estimation_experiments.common_utils.experiment_utils import (
        ContrastDecaySampler,
        build_problem,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Fast-but-informative configuration
# --------------------------------------------------------------------------------------
MAX_QUERIES = 3e4
EPSILON_TARGET = 3e-3
NUM_SHOTS = 40

def _stage_cap_for_t(t_noise: float | None) -> int | None:
    if t_noise is None or np.isinf(t_noise):
        return None          
    if float(t_noise) <= 20.0:
        return 400           
    if float(t_noise) <= 200.0:
        return 1200          
    return 2000

# Keep one hard-noise regime, one medium-noise regime, and one clean regime
T_SWEEP = (20.0, 200.0, np.inf)

# Fixed amplitudes: small / medium / large
A_VALUES = (0.05, 0.20, 0.50)

# Fewer reps, still enough to see coverage/stability patterns
N_REP = 4
ALPHA = 0.05
QUERY_GRID_POINTS = 80

ALGORITHMS = ("bae", "biqae", "cabiqae_latentt")

ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE_latentt",
}

ALGORITHM_STYLES = {
    "bae": {"color": "#1D3557", "marker": "o"},
    "biqae": {"color": "#E76F51", "marker": "s"},
    "cabiqae_latentt": {"color": "#2A9D8F", "marker": "^"},
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    valid_counts = np.sum(~np.isnan(values), axis=axis)
    summed = np.nansum(values, axis=axis)
    means = np.full(summed.shape, np.nan, dtype=float)
    np.divide(summed, valid_counts, out=means, where=valid_counts > 0)
    return means


def _format_t_label(t_noise: float) -> str:
    return "inf" if np.isinf(t_noise) else f"{int(t_noise)}"


def _finite_t_value(t_noise: float) -> float | None:
    return None if np.isinf(t_noise) else float(t_noise)


def _build_solver(
    algorithm: str,
    epsilon_target: float,
    alpha: float,
    n_shots: int,
    seed: int,
    t_noise: float | None,
    max_shots_same_k: int | None,
) -> tuple[Any, bool]:
    sampler = ContrastDecaySampler(T=t_noise, seed=seed)

    if algorithm == "bae":
        solver = StandaloneBAELegacy(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=sampler,
            noise_model="ideal" if t_noise is None or np.isinf(t_noise) else "exponential_contrast",
            T_known=None if t_noise is None or np.isinf(t_noise) else float(t_noise),
            cap_kappa=2.0,
            estimate_T=False,
            T_range=None if t_noise is None or np.isinf(t_noise) else (0.5 * float(t_noise), 1.5 * float(t_noise)),
            TNs=0,
            wNs=60,
            Ns=n_shots,
        )
        return solver, True

    if algorithm == "biqae":
        solver = BIQAE(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=sampler,
            min_ratio=2,
            confint_method="beta",
            max_shots_same_k=max_shots_same_k,
        )
        return solver, True

    if algorithm == "cabiqae_latentt":
        solver = CABIQAELatentTheta(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=sampler,
            min_ratio=2,
            confint_method="beta",
            noise_model="ideal" if t_noise is None or np.isinf(t_noise) else "exponential_contrast",
            T_known=None if t_noise is None or np.isinf(t_noise) else float(t_noise),
            cap_kappa=1000.0,
            use_noise_cap=True,
            max_shots_same_k=max_shots_same_k,
        )
        return solver, True

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _extract_bae_k_sequence(
    result: Any,
    history: dict[str, Any],
    queries: np.ndarray,
    n_shots: int,
) -> np.ndarray:
    k_seq_candidates = []

    obj_seq = getattr(result, "circuit_depths", None)
    if obj_seq is not None:
        arr = np.asarray(obj_seq, dtype=float).ravel()
        if arr.size > 0:
            k_seq_candidates.append(arr)

    hist_seq = history.get("circuit_depths", None)
    if hist_seq is not None:
        arr = np.asarray(hist_seq, dtype=float).ravel()
        if arr.size > 0:
            k_seq_candidates.append(arr)

    if len(k_seq_candidates) > 0:
        return np.asarray(np.rint(k_seq_candidates[0]), dtype=int)

    if len(queries) == 0:
        return np.asarray([], dtype=int)

    dq = np.diff(np.r_[0.0, queries])
    inferred = dq / float(n_shots)
    return np.asarray(np.rint(inferred), dtype=int)


def _extract_trace(
    algorithm: str,
    result: Any,
    n_shots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if algorithm == "bae":
        history = getattr(result, "history", {}) or {}
        queries = np.asarray(history.get("queries", []), dtype=float)
        estimations = np.asarray(history.get("estimations", []), dtype=float)
        K_sequence = _extract_bae_k_sequence(result, history, queries, n_shots)

        usable = min(len(queries), len(estimations), len(K_sequence))
        if usable <= 0:
            return (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                np.asarray([], dtype=int),
            )

        return (
            queries[:usable].astype(float),
            estimations[:usable].astype(float),
            K_sequence[:usable].astype(int),
        )

    powers = np.asarray(getattr(result, "powers", []) or [], dtype=float)
    estimate_intervals = getattr(result, "estimate_intervals", []) or []

    usable = min(len(powers), max(0, len(estimate_intervals) - 1))
    if usable <= 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=int),
        )

    interval_array = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
    estimations = np.mean(interval_array, axis=1)
    K_sequence = (2.0 * powers[:usable] + 1.0).astype(int)
    queries = np.cumsum(n_shots * K_sequence)

    return queries.astype(float), estimations.astype(float), K_sequence.astype(int)


def _interpolate_nsqe(
    queries: np.ndarray,
    errors: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if len(queries) == 0:
        return np.full(len(grid), np.nan, dtype=float)

    idx = np.searchsorted(queries, grid, side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(queries) - 1)

    curve = np.full(len(grid), np.nan, dtype=float)
    curve[valid] = errors[idx[valid]]
    return curve


def _save_metrics_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    fieldnames = [
        "rep",
        "T_noise",
        "algorithm",
        "a_true",
        "final_queries",
        "final_nRMSE",
        "K_max",
        "coverage",
        "radius",
        "max_depth",
        "terminated_early",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_by_t_and_algorithm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    t_values = sorted({float(r["T_noise"]) for r in rows})

    for t_noise in t_values:
        for alg_key in ALGORITHMS:
            alg_name = ALGORITHM_LABELS[alg_key]
            subset = [
                r
                for r in rows
                if float(r["T_noise"]) == t_noise and r["algorithm"] == alg_name
            ]
            if not subset:
                continue

            final_nrmse = np.asarray([float(r["final_nRMSE"]) for r in subset], dtype=float)
            coverage = np.asarray([float(r["coverage"]) for r in subset], dtype=float)
            radius = np.asarray([float(r["radius"]) for r in subset], dtype=float)
            queries = np.asarray([float(r["final_queries"]) for r in subset], dtype=float)
            k_max = np.asarray([float(r["K_max"]) for r in subset], dtype=float)

            summary_rows.append(
                {
                    "T_noise": t_noise,
                    "algorithm": alg_name,
                    "n_runs": int(len(subset)),
                    "nRMSE_median": float(np.nanmedian(final_nrmse)),
                    "coverage_mean": float(np.nanmean(coverage)),
                    "radius_median": float(np.nanmedian(radius)),
                    "queries_median": float(np.nanmedian(queries)),
                    "kmax_median": float(np.nanmedian(k_max)),
                }
            )

    return summary_rows


def _print_t_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Fast summary by noise regime ===")
    t_values = sorted({float(r["T_noise"]) for r in rows})
    for t_noise in t_values:
        t_subset = [r for r in rows if float(r["T_noise"]) == t_noise]
        t_subset.sort(key=lambda x: float(x["nRMSE_median"]))
        print(f"T={_format_t_label(t_noise)}")
        for row in t_subset:
            print(
                "  "
                + f"{row['algorithm']:16s} "
                + f"nRMSE_med={row['nRMSE_median']:.3e} | "
                + f"coverage={row['coverage_mean']:.2f} | "
                + f"Q_med={int(row['queries_median']):6d} | "
                + f"K_med={row['kmax_median']:.0f}"
            )


def _plot_panels(
    curves_by_t: dict[float, dict[str, list[np.ndarray]]],
    query_grid: np.ndarray,
    output_path: str,
) -> None:
    t_values = sorted(curves_by_t.keys())
    fig, axes = plt.subplots(1, len(t_values), figsize=(5.4 * len(t_values), 4.0), squeeze=False)

    for j, t_noise in enumerate(t_values):
        ax = axes[0, j]
        for alg in ALGORITHMS:
            if len(curves_by_t[t_noise][alg]) == 0:
                continue
            stacked = np.vstack(curves_by_t[t_noise][alg])
            mean_rmse = np.sqrt(_safe_nanmean(stacked, axis=0))
            style = ALGORITHM_STYLES[alg]
            ax.loglog(
                query_grid,
                mean_rmse,
                color=style["color"],
                marker=style["marker"],
                markersize=3,
                markevery=8,
                linewidth=1.8,
                label=ALGORITHM_LABELS[alg],
            )

        ax.loglog(
            query_grid,
            1.0 / np.sqrt(query_grid),
            "--",
            color="gray",
            alpha=0.6,
            label=r"$\mathcal{O}(1/\sqrt{N_q})$",
        )
        ax.loglog(
            query_grid,
            3.0 / query_grid,
            "-.",
            color="black",
            alpha=0.5,
            label=r"$\mathcal{O}(1/N_q)$",
        )
        ax.set_title(f"T={_format_t_label(t_noise)}")
        ax.set_xlabel("Total Queries")
        ax.set_ylabel("Normalized RMSE")
        ax.grid(True, which="both", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=250)


def _plot_summary(summary_rows: list[dict[str, Any]], output_path: str) -> None:
    finite_t = sorted({float(r["T_noise"]) for r in summary_rows if not np.isinf(float(r["T_noise"]))})
    x_vals = np.asarray(finite_t, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for alg in ALGORITHMS:
        alg_name = ALGORITHM_LABELS[alg]
        style = ALGORITHM_STYLES[alg]
        y_nrmse = []
        y_cov = []
        y_queries = []

        for t_noise in x_vals:
            row = next(
                (
                    r
                    for r in summary_rows
                    if r["algorithm"] == alg_name and float(r["T_noise"]) == float(t_noise)
                ),
                None,
            )
            if row is None:
                y_nrmse.append(np.nan)
                y_cov.append(np.nan)
                y_queries.append(np.nan)
            else:
                y_nrmse.append(float(row["nRMSE_median"]))
                y_cov.append(float(row["coverage_mean"]))
                y_queries.append(float(row["queries_median"]))

        axes[0].plot(x_vals, y_nrmse, color=style["color"], marker=style["marker"], label=alg_name)
        axes[1].plot(x_vals, y_cov, color=style["color"], marker=style["marker"], label=alg_name)
        axes[2].plot(x_vals, y_queries, color=style["color"], marker=style["marker"], label=alg_name)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Median final nRMSE vs T")
    axes[0].set_xlabel("T")
    axes[0].set_ylabel("Median final nRMSE")
    axes[0].grid(True, which="both", alpha=0.2)

    axes[1].set_xscale("log")
    axes[1].set_title("Coverage vs T")
    axes[1].set_xlabel("T")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].axhline(1.0 - ALPHA, linestyle="--", color="gray", alpha=0.6)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, which="both", alpha=0.2)

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Median queries vs T")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel("Median final queries")
    axes[2].grid(True, which="both", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=250)


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    query_grid = np.logspace(2, np.log10(MAX_QUERIES), num=QUERY_GRID_POINTS)

    curves_by_t: dict[float, dict[str, list[np.ndarray]]] = {
        float(t_noise): {alg: [] for alg in ALGORITHMS}
        for t_noise in T_SWEEP
    }
    metric_rows: list[dict[str, Any]] = []

    print("Running fast informative noisy benchmark...")
    print(f"T sweep: {[ _format_t_label(float(t)) for t in T_SWEEP ]}")
    print(f"A values: {list(A_VALUES)}")
    print(f"Repetitions per scenario: {N_REP}")
    print(f"Shots per iteration: {NUM_SHOTS}")

    for t_noise in T_SWEEP:
        t_label = _format_t_label(float(t_noise))

        for a_true in A_VALUES:
            problem = build_problem(float(a_true))

            for rep in range(N_REP):
                for alg_idx, alg_key in enumerate(ALGORITHMS):
                    seed = int(
                        1_000_000
                        + (9999 if np.isinf(t_noise) else int(t_noise)) * 10_000
                        + int(1000 * a_true) * 100
                        + rep * 10
                        + alg_idx
                    )

                    try:
                        stage_cap = _stage_cap_for_t(_finite_t_value(float(t_noise)))
                        solver, is_bayes = _build_solver(
                            alg_key,
                            EPSILON_TARGET,
                            ALPHA,
                            NUM_SHOTS,
                            seed,
                            _finite_t_value(float(t_noise)),
                            stage_cap,
                        )

                        if alg_key == "bae":
                            np.random.seed(seed)
                            result = solver.estimate(
                                problem,
                                n_shots=NUM_SHOTS,
                                max_queries=int(MAX_QUERIES),
                            )
                        else:
                            result = solver.estimate(
                                problem,
                                bayes=is_bayes,
                                n_shots=NUM_SHOTS,
                                show_details=False,
                            )

                        queries, estimations, K_sequence = _extract_trace(
                            alg_key,
                            result,
                            NUM_SHOTS,
                        )

                        if len(queries) == 0:
                            print(
                                f"T={t_label:>4s} | a={a_true:.2f} | rep={rep + 1:02d} | "
                                f"{alg_key:15s} | no trajectory"
                            )
                            continue

                        nsqe = np.square((estimations / float(a_true)) - 1.0)
                        interpolated = _interpolate_nsqe(queries, nsqe, query_grid)
                        curves_by_t[float(t_noise)][alg_key].append(interpolated)

                        final_nrmse = float(np.sqrt(nsqe[-1]))
                        final_queries = int(getattr(result, "num_state_prep_calls", queries[-1]))                       
                        K_max = int(np.max(K_sequence)) if len(K_sequence) > 0 else 0

                        confidence_interval = getattr(result, "confidence_interval", None)
                        if confidence_interval is None:
                            radius = np.nan
                            coverage = np.nan
                        else:
                            ci_low = float(confidence_interval[0])
                            ci_high = float(confidence_interval[1])
                            radius = 0.5 * (ci_high - ci_low)
                            coverage = float(ci_low <= float(a_true) <= ci_high)

                        max_depth = getattr(result, "circuit_depths", None)
                        max_depth_scalar = np.nan if not max_depth else float(np.max(max_depth))

                        terminated_early = bool(getattr(result, "terminated_early", False))

                        metric_rows.append(
                            {
                                "rep": rep + 1,
                                "T_noise": float(t_noise),
                                "algorithm": ALGORITHM_LABELS[alg_key],
                                "a_true": float(a_true),
                                "final_queries": final_queries,
                                "final_nRMSE": final_nrmse,
                                "K_max": K_max,
                                "coverage": coverage,
                                "radius": radius,
                                "max_depth": max_depth_scalar,
                                "terminated_early": terminated_early,
                            }
                        )

                        print(
                            f"T={t_label:>4s} | a={a_true:.2f} | rep={rep + 1:02d} | "
                            f"{alg_key:15s} | Q={final_queries:6d} | "
                            f"RMSE={final_nrmse:.3e} | cov={coverage:.0f} | Kmax={K_max:3d}"
                        )

                    except Exception as e:
                        print(
                            f"Error in T={t_label}, a={a_true:.2f}, {alg_key}, rep {rep + 1}: {e}"
                        )

    csv_output = os.path.join(current_dir, "metrics_noisy_fast.csv")
    _save_metrics_csv(metric_rows, csv_output)
    print(f"\nSaved metrics table to: {csv_output}")

    summary_rows = _aggregate_by_t_and_algorithm(metric_rows)
    _print_t_summary(summary_rows)

    panels_output = os.path.join(current_dir, "ae_comparison_noisy_fast_panels.png")
    _plot_panels(curves_by_t, query_grid, panels_output)
    print(f"Saved panel plot to: {panels_output}")

    summary_output = os.path.join(current_dir, "ae_summary_noisy_fast.png")
    _plot_summary(summary_rows, summary_output)
    print(f"Saved summary plot to: {summary_output}")

    plt.show()


if __name__ == "__main__":
    run_experiment()