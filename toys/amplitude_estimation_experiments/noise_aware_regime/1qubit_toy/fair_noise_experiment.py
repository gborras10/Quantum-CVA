from __future__ import annotations

import csv
import os
import sys
import time
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
    from toys.amplitude_estimation_experiments.common_utils.experiment_utils import (
        ContrastDecaySampler,
        build_problem,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Fair budget-aligned experiment configuration
# --------------------------------------------------------------------------------------
# This epsilon is NOT the quantity being benchmarked.
# It is only used to force BIQAE/CABIQAE to produce sufficiently long trajectories.
RUN_EPSILON = 5e-5

# Common budget checkpoints for comparison.
BUDGETS = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600 ], dtype=int)
MAX_ANALYSIS_BUDGET = int(BUDGETS[-1])

NUM_SHOTS = 40
ALPHA = 0.05

# Scenarios
T_SWEEP = (20.0, 200.0, 500.0, np.inf)
A_VALUES = (0.05, 0.20, 0.80)
N_REP = 20

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
) -> tuple[Any, bool]:
    sampler = ContrastDecaySampler(T=t_noise, seed=seed)

    if algorithm == "bae":
        solver = StandaloneBAELegacy(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=sampler,
            noise_model="ideal" if t_noise is None or np.isinf(t_noise) else "exponential_contrast",
            T_known=None if t_noise is None or np.isinf(t_noise) else float(t_noise),
            cap_kappa=False,
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
            max_shots_same_k=None,
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
            max_shots_same_k=None,
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


def _estimate_at_budget(
    queries: np.ndarray,
    estimations: np.ndarray,
    budget: int,
) -> float | None:
    if len(queries) == 0:
        return None
    idx = np.searchsorted(queries, budget, side="right") - 1
    if idx < 0:
        return None
    return float(estimations[idx])


def _save_budget_rows(rows: list[dict[str, Any]], out_path: str) -> None:
    fieldnames = [
        "rep",
        "T_noise",
        "a_true",
        "budget",
        "algorithm",
        "estimate",
        "abs_error",
        "nrmse",
        "common_budget_ceiling",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_final_rows(rows: list[dict[str, Any]], out_path: str) -> None:
    fieldnames = [
        "rep",
        "T_noise",
        "a_true",
        "algorithm",
        "final_queries",
        "final_estimate",
        "final_abs_error",
        "final_nrmse",
        "final_coverage",
        "runtime_seconds",
        "k_max",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_budget_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    t_values = sorted({float(r["T_noise"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})

    for t_noise in t_values:
        for budget in budgets:
            for alg in ALGORITHMS:
                alg_name = ALGORITHM_LABELS[alg]
                subset = [
                    r
                    for r in rows
                    if float(r["T_noise"]) == t_noise
                    and int(r["budget"]) == budget
                    and r["algorithm"] == alg_name
                ]
                if not subset:
                    continue

                abs_errors = np.asarray([float(r["abs_error"]) for r in subset], dtype=float)
                nrmse = np.asarray([float(r["nrmse"]) for r in subset], dtype=float)

                summary_rows.append(
                    {
                        "T_noise": float(t_noise),
                        "budget": int(budget),
                        "algorithm": alg_name,
                        "n_points": int(len(subset)),
                        "abs_error_median": float(np.nanmedian(abs_errors)),
                        "abs_error_mean": float(np.nanmean(abs_errors)),
                        "nrmse_median": float(np.nanmedian(nrmse)),
                        "nrmse_mean": float(np.nanmean(nrmse)),
                    }
                )
    return summary_rows


def _print_budget_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\n=== Budget-aligned summary (fair comparison) ===")
    t_values = sorted({float(r["T_noise"]) for r in summary_rows})

    for t_noise in t_values:
        print(f"T={_format_t_label(t_noise)}")
        t_rows = [r for r in summary_rows if float(r["T_noise"]) == t_noise]
        budgets = sorted({int(r["budget"]) for r in t_rows})

        for budget in budgets:
            b_rows = [r for r in t_rows if int(r["budget"]) == budget]
            b_rows.sort(key=lambda x: float(x["nrmse_median"]))
            print(f"  Budget={budget}")
            for row in b_rows:
                print(
                    "    "
                    + f"{row['algorithm']:16s} "
                    + f"nRMSE_med={row['nrmse_median']:.3e} | "
                    + f"AbsErr_med={row['abs_error_median']:.3e} | "
                    + f"n={row['n_points']}"
                )


def _plot_budget_panels(
    summary_rows: list[dict[str, Any]],
    output_path: str,
) -> None:
    t_values = sorted({float(r["T_noise"]) for r in summary_rows})
    fig, axes = plt.subplots(1, len(t_values), figsize=(5.5 * len(t_values), 4.2), squeeze=False)

    for j, t_noise in enumerate(t_values):
        ax = axes[0, j]
        t_rows = [r for r in summary_rows if float(r["T_noise"]) == t_noise]

        for alg in ALGORITHMS:
            alg_name = ALGORITHM_LABELS[alg]
            a_rows = [r for r in t_rows if r["algorithm"] == alg_name]
            if not a_rows:
                continue

            budgets = np.asarray([int(r["budget"]) for r in a_rows], dtype=float)
            nrmse = np.asarray([float(r["nrmse_median"]) for r in a_rows], dtype=float)

            order = np.argsort(budgets)
            budgets = budgets[order]
            nrmse = nrmse[order]

            style = ALGORITHM_STYLES[alg]
            ax.loglog(
                budgets,
                nrmse,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4,
                label=alg_name,
            )

        ax.loglog(
            BUDGETS,
            1.0 / np.sqrt(BUDGETS),
            "--",
            color="gray",
            alpha=0.6,
            label=r"$\mathcal{O}(1/\sqrt{N_q})$",
        )
        ax.loglog(
            BUDGETS,
            3.0 / BUDGETS,
            "-.",
            color="black",
            alpha=0.5,
            label=r"$\mathcal{O}(1/N_q)$",
        )

        ax.set_title(f"T={_format_t_label(t_noise)}")
        ax.set_xlabel("Common query budget")
        ax.set_ylabel("Median normalized RMSE")
        ax.grid(True, which="both", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=250)


def _print_final_summary(final_rows: list[dict[str, Any]]) -> None:
    print("\n=== Final-stop auxiliary summary (not the primary fair metric) ===")
    t_values = sorted({float(r["T_noise"]) for r in final_rows})

    for t_noise in t_values:
        print(f"T={_format_t_label(t_noise)}")
        for alg in ALGORITHMS:
            alg_name = ALGORITHM_LABELS[alg]
            subset = [
                r for r in final_rows
                if float(r["T_noise"]) == t_noise and r["algorithm"] == alg_name
            ]
            if not subset:
                continue

            q = np.asarray([float(r["final_queries"]) for r in subset], dtype=float)
            nr = np.asarray([float(r["final_nrmse"]) for r in subset], dtype=float)
            cov = np.asarray([float(r["final_coverage"]) for r in subset], dtype=float)
            runtime = np.asarray([float(r["runtime_seconds"]) for r in subset], dtype=float)
            kmax = np.asarray([float(r["k_max"]) for r in subset], dtype=float)

            print(
                "  "
                + f"{alg_name:16s} "
                + f"Q_med={int(np.nanmedian(q)):6d} | "
                + f"nRMSE_med={np.nanmedian(nr):.3e} | "
                + f"coverage={np.nanmean(cov):.2f} | "
                + f"time_med={np.nanmedian(runtime):.3f}s | "
                + f"K_med={np.nanmedian(kmax):.0f}"
            )


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    budget_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    print("Running FAIR budget-aligned benchmark...")
    print(f"T sweep: {[ _format_t_label(float(t)) for t in T_SWEEP ]}")
    print(f"A values: {list(A_VALUES)}")
    print(f"Repetitions per scenario: {N_REP}")
    print(f"Shots per iteration: {NUM_SHOTS}")
    print(f"Run epsilon for BIQAE/CABIQAE trajectories: {RUN_EPSILON}")
    print(f"Common budgets: {list(BUDGETS)}")

    for t_noise in T_SWEEP:
        t_label = _format_t_label(float(t_noise))

        for a_true in A_VALUES:
            problem = build_problem(float(a_true))

            for rep in range(N_REP):
                run_results: dict[str, dict[str, Any]] = {}

                # --------------------------------------------------------------
                # Run each algorithm once for this scenario/repetition
                # --------------------------------------------------------------
                for alg_idx, alg_key in enumerate(ALGORITHMS):
                    seed = int(
                        1_000_000
                        + (9999 if np.isinf(t_noise) else int(t_noise)) * 10_000
                        + int(1000 * a_true) * 100
                        + rep * 10
                        + alg_idx
                    )

                    try:
                        solver, is_bayes = _build_solver(
                            alg_key,
                            RUN_EPSILON,
                            ALPHA,
                            NUM_SHOTS,
                            seed,
                            _finite_t_value(float(t_noise)),
                        )

                        if alg_key == "bae":
                            np.random.seed(seed)
                            t0 = time.perf_counter()
                            result = solver.estimate(
                                problem,
                                n_shots=NUM_SHOTS,
                                max_queries=MAX_ANALYSIS_BUDGET,
                            )
                        else:
                            t0 = time.perf_counter()
                            result = solver.estimate(
                                problem,
                                bayes=is_bayes,
                                n_shots=NUM_SHOTS,
                                show_details=False,
                            )
                        runtime_seconds = float(time.perf_counter() - t0)

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

                        final_est = float(estimations[-1])
                        final_abs_error = abs(final_est - float(a_true))
                        final_nrmse = final_abs_error / float(a_true)

                        confidence_interval = getattr(result, "confidence_interval", None)
                        if confidence_interval is None:
                            final_coverage = np.nan
                        else:
                            ci_low = float(confidence_interval[0])
                            ci_high = float(confidence_interval[1])
                            final_coverage = float(ci_low <= float(a_true) <= ci_high)

                        final_queries = int(getattr(result, "num_state_prep_calls", queries[-1]))
                        k_max = int(np.max(K_sequence)) if len(K_sequence) > 0 else 0

                        final_rows.append(
                            {
                                "rep": rep + 1,
                                "T_noise": float(t_noise),
                                "a_true": float(a_true),
                                "algorithm": ALGORITHM_LABELS[alg_key],
                                "final_queries": final_queries,
                                "final_estimate": final_est,
                                "final_abs_error": final_abs_error,
                                "final_nrmse": final_nrmse,
                                "final_coverage": final_coverage,
                                "runtime_seconds": runtime_seconds,
                                "k_max": k_max,
                            }
                        )

                        run_results[alg_key] = {
                            "queries": queries,
                            "estimations": estimations,
                            "k_sequence": K_sequence,
                            "result": result,
                        }

                        print(
                            f"T={t_label:>4s} | a={a_true:.2f} | rep={rep + 1:02d} | "
                            f"{alg_key:15s} | "
                            f"Qmax={int(queries[-1]):6d} | "
                            f"final nRMSE={final_nrmse:.3e} | "
                            f"time={runtime_seconds:.3f}s | "
                            f"Kmax={k_max:3d}"
                        )

                    except Exception as e:
                        print(
                            f"Error in T={t_label}, a={a_true:.2f}, {alg_key}, rep {rep + 1}: {e}"
                        )

                # --------------------------------------------------------------
                # Fair budget-aligned comparison:
                # only compare up to the common max budget reached by ALL three
                # --------------------------------------------------------------
                if not all(alg in run_results for alg in ALGORITHMS):
                    continue

                common_budget_ceiling = min(
                    int(run_results[alg]["queries"][-1])
                    for alg in ALGORITHMS
                )
                common_budget_ceiling = min(common_budget_ceiling, MAX_ANALYSIS_BUDGET)

                for budget in BUDGETS:
                    if int(budget) > common_budget_ceiling:
                        continue

                    for alg_key in ALGORITHMS:
                        est = _estimate_at_budget(
                            run_results[alg_key]["queries"],
                            run_results[alg_key]["estimations"],
                            int(budget),
                        )
                        if est is None:
                            continue

                        abs_error = abs(est - float(a_true))
                        nrmse = abs_error / float(a_true)

                        budget_rows.append(
                            {
                                "rep": rep + 1,
                                "T_noise": float(t_noise),
                                "a_true": float(a_true),
                                "budget": int(budget),
                                "algorithm": ALGORITHM_LABELS[alg_key],
                                "estimate": float(est),
                                "abs_error": float(abs_error),
                                "nrmse": float(nrmse),
                                "common_budget_ceiling": int(common_budget_ceiling),
                            }
                        )

    # --------------------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------------------
    budget_csv = os.path.join(current_dir, "budget_aligned_rows.csv")
    _save_budget_rows(budget_rows, budget_csv)
    print(f"\nSaved fair budget-aligned rows to: {budget_csv}")

    final_csv = os.path.join(current_dir, "final_stop_aux_rows.csv")
    _save_final_rows(final_rows, final_csv)
    print(f"Saved final-stop auxiliary rows to: {final_csv}")

    # --------------------------------------------------------------------------
    # Summaries
    # --------------------------------------------------------------------------
    budget_summary = _aggregate_budget_rows(budget_rows)
    _print_budget_summary(budget_summary)
    _print_final_summary(final_rows)

    # --------------------------------------------------------------------------
    # Plots
    # --------------------------------------------------------------------------
    plot_path = os.path.join(current_dir, "fair_budget_aligned_panels.png")
    _plot_budget_panels(budget_summary, plot_path)
    print(f"Saved fair budget-aligned plot to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    run_experiment()