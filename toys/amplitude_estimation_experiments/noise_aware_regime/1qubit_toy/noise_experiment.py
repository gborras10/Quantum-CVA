from __future__ import annotations

import csv
import os
import sys
from typing import Any

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
    from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
        StandaloneBAEHardware,
    )
    from toys.amplitude_estimation_experiments.common_utils.experiment_utils import (
        ContrastDecaySampler,
        build_problem,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Global configuration
# --------------------------------------------------------------------------------------
MAX_QUERIES = 8e4
EPSILON_TARGET = 3e-3
NUM_SHOTS = 50

# Informative benchmark regime: multiple noise severities and target amplitudes.
T_SWEEP = (20.0, 50.0, 200.0, 1_000.0, np.inf)
A_SWEEP_SIZE = 6
BINOMIAL_N = 100
BINOMIAL_P = 0.50
A_MIN = 0.02
A_MAX = 0.98
N_REP = 12
ALPHA = 0.05
QUERY_GRID_POINTS = 180

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

ALGORITHM_CONFIG = {
    "bae": {
        "epsilon_target": EPSILON_TARGET,
        "n_shots": NUM_SHOTS,
        "max_queries": MAX_QUERIES,
        "estimate_T": False,
    },
    "biqae": {
        "epsilon_target": EPSILON_TARGET,
        "n_shots": NUM_SHOTS,
        "max_queries": None,
    },
    "cabiqae_latentt": {
        "epsilon_target": EPSILON_TARGET,
        "n_shots": NUM_SHOTS,
        "max_queries": None,
    },
}

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """Compute mean ignoring NaNs safely."""
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
    t_noise: float | None,
    estimate_T: bool = False,
) -> tuple[Any, bool]:
    """
    Build the solver for each algorithm under the same noisy regime.
    """
    sampler = ContrastDecaySampler(T=t_noise, seed=seed)

    if algorithm == "bae":
        solver = StandaloneBAELegacy(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=sampler,
            noise_model="ideal" if t_noise is None or np.isinf(t_noise) else "exponential_contrast",
            T_known=None if t_noise is None or np.isinf(t_noise) else float(t_noise),
            cap_kappa=2.0,
            estimate_T=estimate_T,
            T_range=None
            if t_noise is None or np.isinf(t_noise)
            else (0.5 * float(t_noise), 1.5 * float(t_noise)),
            TNs=0,
            wNs=100,
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
            cap_kappa=2.0,
            use_noise_cap=True,
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
    """
    Return:
      - cumulative queries
      - point estimates
      - K_sequence = 2k+1 used at each iteration/stage
    """
    if algorithm in {"bae"}:
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
    if interval_array.ndim != 2 or interval_array.shape[1] != 2:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=int),
        )

    estimations = np.mean(interval_array, axis=1)
    K_sequence = (2.0 * powers[:usable] + 1.0).astype(int)
    queries = np.cumsum(n_shots * K_sequence)

    return queries.astype(float), estimations.astype(float), K_sequence.astype(int)


def _interpolate_nsqe(
    queries: np.ndarray,
    errors: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Map errors onto a common query grid for averaging across repetitions.
    """
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
        "K_sequence",
        "coverage",
        "radius",
        "max_depth",
        "auc_log_rmse",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_scalar_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for alg_key in ALGORITHMS:
        alg_name = ALGORITHM_LABELS[alg_key]
        alg_rows = [r for r in rows if r["algorithm"] == alg_name]
        if not alg_rows:
            continue

        coverage = np.array([float(r["coverage"]) for r in alg_rows], dtype=float)
        radius = np.array([float(r["radius"]) for r in alg_rows], dtype=float)
        final_nrmse = np.array([float(r["final_nRMSE"]) for r in alg_rows], dtype=float)
        kmax = np.array([float(r["K_max"]) for r in alg_rows], dtype=float)
        max_depth = np.array([float(r["max_depth"]) for r in alg_rows], dtype=float)
        queries = np.array([float(r["final_queries"]) for r in alg_rows], dtype=float)

        summary[alg_name] = {
            "coverage_mean": float(np.mean(coverage)),
            "radius_mean": float(np.mean(radius)),
            "radius_median": float(np.median(radius)),
            "final_nrmse_mean": float(np.mean(final_nrmse)),
            "kmax_mean": float(np.mean(kmax)),
            "kmax_global": float(np.max(kmax)),
            "max_depth_mean": float(np.mean(max_depth)),
            "queries_mean": float(np.mean(queries)),
        }
    return summary


def _finite_t_value(t_noise: float) -> float | None:
    return None if np.isinf(t_noise) else float(t_noise)


def _format_t_label(t_noise: float) -> str:
    return "inf" if np.isinf(t_noise) else f"{int(t_noise)}"


def _sample_random_a_sweep(rng: np.random.Generator) -> np.ndarray:
    """
    Draw a random amplitude sweep from a binomial law and keep values away from 0 and 1
    to avoid unstable normalized-error denominators.
    """
    draws = rng.binomial(BINOMIAL_N, BINOMIAL_P, size=4 * A_SWEEP_SIZE).astype(float)
    amps = draws / float(BINOMIAL_N)
    amps = amps[(amps >= A_MIN) & (amps <= A_MAX)]
    amps = np.unique(amps)

    if amps.size >= A_SWEEP_SIZE:
        rng.shuffle(amps)
        return np.sort(amps[:A_SWEEP_SIZE])

    # Fallback in the unlikely case of too many repeated/support-edge binomial draws.
    needed = A_SWEEP_SIZE - amps.size
    extra = rng.uniform(A_MIN, A_MAX, size=needed)
    merged = np.concatenate([amps, extra])
    return np.sort(merged[:A_SWEEP_SIZE])


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
            auc = np.asarray([float(r["auc_log_rmse"]) for r in subset], dtype=float)

            summary_rows.append(
                {
                    "T_noise": t_noise,
                    "algorithm": alg_name,
                    "n_runs": int(len(subset)),
                    "nRMSE_median": float(np.nanmedian(final_nrmse)),
                    "nRMSE_p90": float(np.nanpercentile(final_nrmse, 90)),
                    "coverage_mean": float(np.nanmean(coverage)),
                    "radius_median": float(np.nanmedian(radius)),
                    "queries_median": float(np.nanmedian(queries)),
                    "kmax_median": float(np.nanmedian(k_max)),
                    "auc_log_rmse_median": float(np.nanmedian(auc)),
                }
            )

    return summary_rows


def _save_t_summary_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    if not rows:
        return

    fieldnames = [
        "T_noise",
        "algorithm",
        "n_runs",
        "nRMSE_median",
        "nRMSE_p90",
        "coverage_mean",
        "radius_median",
        "queries_median",
        "kmax_median",
        "auc_log_rmse_median",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_t_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No summary rows available.")
        return

    print("\n=== Robust summary by noise regime (lower nRMSE/AUC is better) ===")
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
                + f"nRMSE_p90={row['nRMSE_p90']:.3e} | "
                + f"AUC_med={row['auc_log_rmse_median']:.3f} | "
                + f"coverage={row['coverage_mean']:.2f} | "
                + f"Q_med={int(row['queries_median']):6d}"
            )


def _plot_rmse_panels_by_t(
    curves_by_t: dict[float, dict[str, np.ndarray]],
    query_grid: np.ndarray,
    output_path: str,
) -> None:
    t_values = sorted(curves_by_t.keys())
    n_panels = len(t_values)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for idx, t_noise in enumerate(t_values):
        ax = axes[idx // n_cols][idx % n_cols]
        for alg in ALGORITHMS:
            mean_nsqe = curves_by_t[t_noise].get(alg, np.full_like(query_grid, np.nan, dtype=float))
            mean_rmse = np.sqrt(mean_nsqe)
            style = ALGORITHM_STYLES[alg]
            ax.loglog(
                query_grid,
                mean_rmse,
                color=style["color"],
                marker=style["marker"],
                markersize=3,
                markevery=10,
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

        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(f"T={_format_t_label(t_noise)}")
        ax.set_xlabel("Total Queries")
        ax.set_ylabel("Normalized RMSE")

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.suptitle("Noise-aware AE comparison by decoherence regime")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=300)


def _plot_summary_vs_t(summary_rows: list[dict[str, Any]], output_path: str) -> None:
    if not summary_rows:
        return

    finite_t = sorted({float(r["T_noise"]) for r in summary_rows if not np.isinf(float(r["T_noise"]))})
    x_vals = np.asarray(finite_t, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

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
    axes[0].set_xlabel("T (higher = less noise)")
    axes[0].set_ylabel("Median final nRMSE")
    axes[0].grid(True, which="both", alpha=0.2)

    axes[1].set_xscale("log")
    axes[1].set_title("Coverage vs T")
    axes[1].set_xlabel("T (higher = less noise)")
    axes[1].set_ylabel("Empirical coverage")
    axes[1].axhline(1.0 - ALPHA, linestyle="--", color="gray", alpha=0.6)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, which="both", alpha=0.2)

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Median query cost vs T")
    axes[2].set_xlabel("T (higher = less noise)")
    axes[2].set_ylabel("Median final queries")
    axes[2].grid(True, which="both", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=300)


def _plot_coverage_vs_radius(
    summary: dict[str, dict[str, float]],
    alpha: float,
) -> None:
    """
    Useful coverage plot: empirical coverage vs mean final interval radius.
    Coverage alone is not informative; this shows the calibration-width trade-off.
    """
    plt.figure(figsize=(9, 6))

    for alg_key in ALGORITHMS:
        alg_name = ALGORITHM_LABELS[alg_key]
        if alg_name not in summary:
            continue

        x = summary[alg_name]["radius_mean"]
        y = summary[alg_name]["coverage_mean"]
        k_mean = summary[alg_name]["kmax_mean"]
        style = ALGORITHM_STYLES[alg_key]

        plt.scatter(
            x,
            y,
            s=120,
            color=style["color"],
            marker=style["marker"],
            label=alg_name,
        )
        plt.annotate(
            f"{alg_name}\nmean K={k_mean:.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    plt.axhline(
        1.0 - alpha,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label=f"Nominal coverage = {1.0 - alpha:.2f}",
    )
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Mean final CI radius")
    plt.ylabel("Empirical final coverage")
    plt.title("Coverage vs interval radius (aggregated)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    n_rep = N_REP
    alpha = ALPHA
    max_total_queries = MAX_QUERIES
    query_grid = np.logspace(2, np.log10(max_total_queries), num=QUERY_GRID_POINTS)

    rng = np.random.default_rng()
    a_sweep = _sample_random_a_sweep(rng)

    scenario_count = n_rep * len(a_sweep) * len(T_SWEEP)
    results_store = {
        (float(t_noise), float(a_true), alg): np.full((n_rep, len(query_grid)), np.nan, dtype=float)
        for t_noise in T_SWEEP
        for a_true in a_sweep
        for alg in ALGORITHMS
    }
    curves_by_t = {
        float(t_noise): {alg: np.full((scenario_count, len(query_grid)), np.nan, dtype=float) for alg in ALGORITHMS}
        for t_noise in T_SWEEP
    }
    # Store K summaries + scalar metrics
    metric_rows: list[dict[str, Any]] = []
    global_k_max = {alg: 0 for alg in ALGORITHMS}

    print("Running informative noisy benchmark across multiple regimes...")
    print(f"T sweep: {[ _format_t_label(float(t)) for t in T_SWEEP ]}")
    print(
        "Amplitude sweep (random binomial): "
        + str([round(float(a), 4) for a in a_sweep.tolist()])
    )
    print(f"Repetitions per scenario: {n_rep}")
    print(
        "Configs: "
        + ", ".join(
            (
                f"{ALGORITHM_LABELS[a]}(eps={ALGORITHM_CONFIG[a]['epsilon_target']}, "
                f"shots={ALGORITHM_CONFIG[a]['n_shots']}, "
                f"max_q={ALGORITHM_CONFIG[a]['max_queries']})"
            )
            for a in ALGORITHMS
        )
    )

    t_curves_cursor = {float(t_noise): {alg: 0 for alg in ALGORITHMS} for t_noise in T_SWEEP}

    for t_noise in T_SWEEP:
        t_label = _format_t_label(float(t_noise))
        for a_true in a_sweep:
            problem = build_problem(float(a_true))

            for rep in range(n_rep):
                for alg_idx, alg_key in enumerate(ALGORITHMS):
                    cfg = ALGORITHM_CONFIG[alg_key]
                    seed = int(1_000_000 + int(float(t_noise if not np.isinf(t_noise) else 9_999)) * 10_000 + int(a_true * 1000) * 10 + rep * 100 + alg_idx)

                    try:
                        solver, is_bayes = _build_solver(
                            alg_key,
                            cfg["epsilon_target"],
                            alpha,
                            cfg["n_shots"],
                            seed,
                            t_noise=_finite_t_value(float(t_noise)),
                            estimate_T=cfg.get("estimate_T", False),
                        )

                        if alg_key in {"bae"}:
                            np.random.seed(seed)
                            result = solver.estimate(
                                problem,
                                n_shots=int(cfg["n_shots"]),
                                max_queries=int(cfg["max_queries"] or max_total_queries),
                            )
                        else:
                            result = solver.estimate(
                                problem,
                                bayes=is_bayes,
                                n_shots=int(cfg["n_shots"]),
                                show_details=False,
                            )

                        queries, estimations, K_sequence = _extract_trace(
                            alg_key,
                            result,
                            int(cfg["n_shots"]),
                        )

                        if len(queries) == 0:
                            print(
                                f"T={t_label:>4s} | a={a_true:.2f} | rep={rep + 1:02d} | "
                                f"{alg_key:15s} | no trajectory returned"
                            )
                            continue

                        K_max = int(np.max(K_sequence)) if len(K_sequence) > 0 else -1
                        global_k_max[alg_key] = max(global_k_max[alg_key], K_max)

                        nsqe = np.square((estimations / float(a_true)) - 1.0)
                        interpolated = _interpolate_nsqe(
                            queries,
                            nsqe,
                            query_grid,
                        )
                        results_store[(float(t_noise), float(a_true), alg_key)][rep, :] = interpolated

                        cursor = t_curves_cursor[float(t_noise)][alg_key]
                        if cursor < curves_by_t[float(t_noise)][alg_key].shape[0]:
                            curves_by_t[float(t_noise)][alg_key][cursor, :] = interpolated
                            t_curves_cursor[float(t_noise)][alg_key] += 1

                        final_nrmse = float(np.sqrt(nsqe[-1]))
                        final_queries = int(queries[-1])

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
                        if max_depth is None or len(max_depth) == 0:
                            max_depth_scalar = np.nan
                        else:
                            max_depth_scalar = float(np.max(max_depth))

                        valid_curve = interpolated[np.isfinite(interpolated)]
                        if valid_curve.size == 0:
                            auc_log_rmse = np.nan
                        else:
                            rmse_curve = np.sqrt(np.clip(interpolated, 1e-20, None))
                            auc_log_rmse = float(
                                np.trapezoid(
                                    np.log10(np.clip(rmse_curve, 1e-20, None)),
                                    x=np.log10(query_grid),
                                )
                            )

                        metric_rows.append(
                            {
                                "rep": rep + 1,
                                "T_noise": float(t_noise),
                                "algorithm": ALGORITHM_LABELS[alg_key],
                                "a_true": float(a_true),
                                "final_queries": final_queries,
                                "final_nRMSE": final_nrmse,
                                "K_max": K_max,
                                "K_sequence": " ".join(str(int(x)) for x in K_sequence.tolist()),
                                "coverage": coverage,
                                "radius": radius,
                                "max_depth": max_depth_scalar,
                                "auc_log_rmse": auc_log_rmse,
                            }
                        )

                        print(
                            f"T={t_label:>4s} | a={a_true:.2f} | rep={rep + 1:02d} | {alg_key:15s} | "
                            f"Q={final_queries:7d} | RMSE={final_nrmse:.3e} | "
                            f"coverage={coverage:.0f} | radius={radius:.3e}"
                        )

                    except Exception as e:
                        print(
                            f"Error in T={t_label}, a={a_true:.2f}, {alg_key}, rep {rep + 1}: {e}"
                        )

    csv_output = os.path.join(current_dir, "metrics_noisy_informative_regime.csv")
    _save_metrics_csv(metric_rows, csv_output)
    print(f"\nSaved metrics table to: {csv_output}")

    t_summary_rows = _aggregate_by_t_and_algorithm(metric_rows)
    t_summary_output = os.path.join(current_dir, "summary_noisy_informative_regime.csv")
    _save_t_summary_csv(t_summary_rows, t_summary_output)
    _print_t_summary(t_summary_rows)
    print(f"Saved summary table to: {t_summary_output}")

    summary = _summarize_scalar_metrics(metric_rows)
    print("\nSummary metrics by algorithm:")
    for alg_name, vals in summary.items():
        print(alg_name, vals)

    # ----------------------------------------------------------------------------------
    # Plot 1: RMSE vs queries (aggregated over all scenarios)
    # ----------------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    for alg in ALGORITHMS:
        stacked = np.vstack(
            [
                results_store[(float(t_noise), float(a_true), alg)]
                for t_noise in T_SWEEP
                for a_true in a_sweep
            ]
        )
        mean_rmse = np.sqrt(_safe_nanmean(stacked, axis=0))
        plt.loglog(
            query_grid,
            mean_rmse,
            label=f"{ALGORITHM_LABELS[alg]} (max K={global_k_max[alg]})",
            **ALGORITHM_STYLES[alg],
            markersize=4,
            markevery=10,
        )

    plt.loglog(
        query_grid,
        1.0 / np.sqrt(query_grid),
        "--",
        color="gray",
        alpha=0.5,
        label=r"SQL $\mathcal{O}(1/\sqrt{N_q})$",
    )
    plt.loglog(
        query_grid,
        3.0 / query_grid,
        "-.",
        color="black",
        alpha=0.5,
        label=r"Heisenberg $\mathcal{O}(1/N_q)$",
    )

    plt.title("AE Algorithm Comparison (informative noisy regime)")
    plt.xlabel("Total Queries")
    plt.ylabel("Normalized RMSE")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()

    output = os.path.join(current_dir, "ae_comparison_noisy_informative_regime.png")
    plt.savefig(output, dpi=300)
    print(f"\nPlot saved to: {output}")

    # ----------------------------------------------------------------------------------
    # Plot 1b: RMSE panels by T
    # ----------------------------------------------------------------------------------
    curves_by_t_mean = {
        t_noise: {alg: _safe_nanmean(curves_by_t[t_noise][alg], axis=0) for alg in ALGORITHMS}
        for t_noise in curves_by_t
    }
    output_panels = os.path.join(current_dir, "ae_comparison_noisy_by_T_panels.png")
    _plot_rmse_panels_by_t(curves_by_t_mean, query_grid, output_panels)
    print(f"Panel plot saved to: {output_panels}")

    # ----------------------------------------------------------------------------------
    # Plot 1c: scalar trends vs T
    # ----------------------------------------------------------------------------------
    output_trends = os.path.join(current_dir, "ae_summary_vs_T.png")
    _plot_summary_vs_t(t_summary_rows, output_trends)
    print(f"Trend plot saved to: {output_trends}")

    # ----------------------------------------------------------------------------------
    # Plot 2: Coverage vs something useful -> radius
    # ----------------------------------------------------------------------------------
    _plot_coverage_vs_radius(summary, alpha=alpha)

    plt.show()


if __name__ == "__main__":
    run_experiment()