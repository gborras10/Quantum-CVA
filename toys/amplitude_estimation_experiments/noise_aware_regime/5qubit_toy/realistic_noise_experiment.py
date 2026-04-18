from __future__ import annotations

import itertools
import math
import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")
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
    from realistic_utils import (
        BAE_KIND,
        AerCountSampler,
        build_large_problem,
        build_noise_model,
        build_solver,
        calibrate_effective_T,
        estimate_at_budget,
        extract_trace,
        plot_budget_panels,
        save_csv,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
RUN_EPSILON = 8e-4
ALPHA = 0.05
NUM_SHOTS = 64

# Budget-aligned comparison.
BUDGETS = np.array([400, 800, 1600, 3200, 6400, 12800, 19200], dtype=int)
MAX_ANALYSIS_BUDGET = int(BUDGETS[-1])

# Three realistic noise severities on the same larger circuit.
NOISE_PROFILES = {
    "mild": {"scale": 0.75},
    "moderate": {"scale": 1.00},
    "harsh": {"scale": 1.50},
}

# Sweep several amplitude scenarios by perturbing the objective-qubit rotation.
OBJECTIVE_RY_OFFSETS = (-1.05, -0.70, -0.35, 0.0, 0.35, 0.70, 1.05)
MIN_VALID_A_TRUE = 0.04
MAX_VALID_A_TRUE = 0.96

# Repetitions per (profile, amplitude scenario).
N_REP = 8

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

# Probe depths used to calibrate an effective exponential contrast time.
PROBE_KS = [0, 1, 2, 3, 4, 6, 8]
PROBE_SHOTS = 12000
CALIBRATION_REPS = 2

# Inference configuration for robust statistical conclusions.
BOOTSTRAP_RESAMPLES = 2000
PAIRWISE_ALPHA = 0.05


# --------------------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    resamples: int,
    alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    center = float(np.mean(arr))
    if arr.size <= 1 or resamples <= 1:
        return center, center, center

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(resamples, arr.size))
    means = np.mean(arr[idx], axis=1)

    q_low = float(np.quantile(means, alpha / 2.0))
    q_high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return center, q_low, q_high


def exact_two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n <= 0:
        return np.nan

    k = int(min(wins, losses))
    tail = 0.0
    for i in range(k + 1):
        tail += math.comb(n, i)

    p_value = min(1.0, 2.0 * tail / float(2**n))
    return float(p_value)


def wilson_interval(wins: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan

    z = 1.959963984540054
    p_hat = float(wins) / float(total)
    total_f = float(total)

    denom = 1.0 + (z * z) / total_f
    center = (p_hat + (z * z) / (2.0 * total_f)) / denom
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total_f)) / total_f) / denom

    return max(0.0, center - radius), min(1.0, center + radius)


# --------------------------------------------------------------------------------------
# Aggregation and reporting helpers
# --------------------------------------------------------------------------------------
def build_amplitude_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for scenario_id, offset in enumerate(OBJECTIVE_RY_OFFSETS, start=1):
        problem, a_true = build_large_problem(objective_ry_offset=float(offset))
        if not (MIN_VALID_A_TRUE <= a_true <= MAX_VALID_A_TRUE):
            print(
                f"Skipping scenario s{scenario_id:02d}: "
                f"offset={offset:+.3f} gives a_true={a_true:.4f}"
            )
            continue

        scenarios.append(
            {
                "scenario_id": int(scenario_id),
                "objective_ry_offset": float(offset),
                "problem": problem,
                "a_true": float(a_true),
            }
        )

    if not scenarios:
        raise RuntimeError("No valid amplitude scenarios were generated for robust benchmarking.")

    return scenarios


def aggregate_budget_rows_robust(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []

    profile_names = sorted({str(r["profile"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})

    for profile in profile_names:
        for budget in budgets:
            for alg in ALGORITHMS:
                alg_name = ALGORITHM_LABELS[alg]
                subset = [
                    r
                    for r in rows
                    if str(r["profile"]) == profile
                    and int(r["budget"]) == budget
                    and r["algorithm"] == alg_name
                ]
                if not subset:
                    continue

                abs_err = np.asarray([float(r["abs_error"]) for r in subset], dtype=float)
                nrmse = np.asarray([float(r["nrmse"]) for r in subset], dtype=float)
                n_points = int(nrmse.size)

                seed = int(10_000 + budget * 17 + sum(ord(ch) for ch in f"{profile}-{alg_name}"))
                nrmse_mean, nrmse_ci_low, nrmse_ci_high = bootstrap_mean_ci(
                    nrmse,
                    resamples=BOOTSTRAP_RESAMPLES,
                    alpha=ALPHA,
                    seed=seed,
                )

                summary_rows.append(
                    {
                        "profile": profile,
                        "budget": int(budget),
                        "algorithm": alg_name,
                        "n_points": n_points,
                        "abs_error_median": float(np.nanmedian(abs_err)),
                        "abs_error_mean": float(np.nanmean(abs_err)),
                        "nrmse_median": float(np.nanmedian(nrmse)),
                        "nrmse_mean": float(nrmse_mean),
                        "nrmse_q25": float(np.nanquantile(nrmse, 0.25)),
                        "nrmse_q75": float(np.nanquantile(nrmse, 0.75)),
                        "nrmse_mean_ci_low": float(nrmse_ci_low),
                        "nrmse_mean_ci_high": float(nrmse_ci_high),
                    }
                )

    return summary_rows


def build_pairwise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairwise_rows: list[dict[str, Any]] = []
    profile_names = sorted({str(r["profile"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})
    algorithm_names = [ALGORITHM_LABELS[alg] for alg in ALGORITHMS]

    for profile in profile_names:
        for budget in budgets:
            subset = [
                r
                for r in rows
                if str(r["profile"]) == profile and int(r["budget"]) == budget
            ]
            if not subset:
                continue

            paired_map: dict[tuple[int, int], dict[str, float]] = {}
            for r in subset:
                pair_id = (int(r["scenario_id"]), int(r["rep"]))
                paired_map.setdefault(pair_id, {})[str(r["algorithm"])] = float(r["nrmse"])

            for alg_a, alg_b in itertools.combinations(algorithm_names, 2):
                deltas: list[float] = []
                for value_map in paired_map.values():
                    if alg_a in value_map and alg_b in value_map:
                        deltas.append(float(value_map[alg_a] - value_map[alg_b]))

                if not deltas:
                    continue

                delta_arr = np.asarray(deltas, dtype=float)
                wins_a = int(np.sum(delta_arr < -1.0e-12))
                wins_b = int(np.sum(delta_arr > 1.0e-12))
                ties = int(delta_arr.size - wins_a - wins_b)
                non_ties = int(wins_a + wins_b)

                if non_ties > 0:
                    win_rate_a = float(wins_a / non_ties)
                    win_ci_low, win_ci_high = wilson_interval(wins_a, non_ties)
                    sign_pvalue = exact_two_sided_sign_test_pvalue(wins_a, wins_b)
                else:
                    win_rate_a = np.nan
                    win_ci_low = np.nan
                    win_ci_high = np.nan
                    sign_pvalue = np.nan

                seed = int(50_000 + budget * 19 + sum(ord(ch) for ch in f"{profile}-{alg_a}-{alg_b}"))
                mean_delta, delta_ci_low, delta_ci_high = bootstrap_mean_ci(
                    delta_arr,
                    resamples=BOOTSTRAP_RESAMPLES,
                    alpha=PAIRWISE_ALPHA,
                    seed=seed,
                )

                pairwise_rows.append(
                    {
                        "profile": profile,
                        "budget": int(budget),
                        "algorithm_a": alg_a,
                        "algorithm_b": alg_b,
                        "n_pairs": int(delta_arr.size),
                        "wins_a": int(wins_a),
                        "wins_b": int(wins_b),
                        "ties": int(ties),
                        "win_rate_a": float(win_rate_a),
                        "win_rate_ci_low": float(win_ci_low),
                        "win_rate_ci_high": float(win_ci_high),
                        "mean_delta_nrmse": float(mean_delta),
                        "median_delta_nrmse": float(np.nanmedian(delta_arr)),
                        "delta_ci_low": float(delta_ci_low),
                        "delta_ci_high": float(delta_ci_high),
                        "sign_test_pvalue": float(sign_pvalue),
                    }
                )

    return pairwise_rows


def print_budget_summary_robust(
    summary_rows: list[dict[str, Any]],
    profile_names: list[str],
) -> None:
    print("\n=== Robust budget-aligned summary ===")
    for profile_name in profile_names:
        print(f"profile={profile_name}")
        profile_rows = [r for r in summary_rows if str(r["profile"]) == profile_name]
        budgets = sorted({int(r["budget"]) for r in profile_rows})

        for budget in budgets:
            budget_rows = [r for r in profile_rows if int(r["budget"]) == budget]
            budget_rows.sort(key=lambda x: float(x["nrmse_median"]))
            print(f"  budget={budget}")
            for row in budget_rows:
                print(
                    "    "
                    + f"{row['algorithm']:16s} "
                    + f"nRMSE_med={row['nrmse_median']:.3e} | "
                    + f"nRMSE_mean={row['nrmse_mean']:.3e} "
                    + f"[{row['nrmse_mean_ci_low']:.3e}, {row['nrmse_mean_ci_high']:.3e}] | "
                    + f"IQR=[{row['nrmse_q25']:.3e}, {row['nrmse_q75']:.3e}] | "
                    + f"n={row['n_points']}"
                )


def print_pairwise_summary(
    pairwise_rows: list[dict[str, Any]],
    profile_names: list[str],
) -> None:
    if not pairwise_rows:
        print("\nNo pairwise rows available.")
        return

    max_budget = max(int(r["budget"]) for r in pairwise_rows)
    print("\n=== Pairwise statistical comparison at max budget ===")
    print(f"budget={max_budget}")

    for profile_name in profile_names:
        print(f"profile={profile_name}")
        subset = [
            r
            for r in pairwise_rows
            if str(r["profile"]) == profile_name and int(r["budget"]) == max_budget
        ]
        if not subset:
            print("  no pairwise results")
            continue

        subset.sort(key=lambda x: float(np.nan_to_num(x["sign_test_pvalue"], nan=1.0)))
        for row in subset:
            p_value = float(row["sign_test_pvalue"])
            ci_low = float(row["delta_ci_low"])
            ci_high = float(row["delta_ci_high"])

            if np.isfinite(p_value) and p_value < PAIRWISE_ALPHA and ci_high < 0.0:
                verdict = f"{row['algorithm_a']} better"
            elif np.isfinite(p_value) and p_value < PAIRWISE_ALPHA and ci_low > 0.0:
                verdict = f"{row['algorithm_b']} better"
            else:
                verdict = "inconclusive"

            print(
                "  "
                + f"{row['algorithm_a']:16s} vs {row['algorithm_b']:16s} | "
                + f"mean_delta={row['mean_delta_nrmse']:.3e} "
                + f"[{row['delta_ci_low']:.3e}, {row['delta_ci_high']:.3e}] | "
                + f"win_rate={row['win_rate_a']:.2f} "
                + f"[{row['win_rate_ci_low']:.2f}, {row['win_rate_ci_high']:.2f}] | "
                + f"p={row['sign_test_pvalue']:.3e} | "
                + verdict
            )


def print_final_summary_robust(
    final_rows: list[dict[str, Any]],
    profile_names: list[str],
) -> None:
    print("\n=== Final-stop auxiliary summary ===")
    for profile_name in profile_names:
        print(f"profile={profile_name}")
        for alg in ALGORITHMS:
            alg_name = ALGORITHM_LABELS[alg]
            subset = [
                r
                for r in final_rows
                if str(r["profile"]) == profile_name and r["algorithm"] == alg_name
            ]
            if not subset:
                continue

            q = np.asarray([float(r["final_queries"]) for r in subset], dtype=float)
            nr = np.asarray([float(r["final_nrmse"]) for r in subset], dtype=float)
            cov = np.asarray([float(r["coverage"]) for r in subset], dtype=float)
            runtime = np.asarray([float(r["runtime_seconds"]) for r in subset], dtype=float)
            kmax = np.asarray([float(r["k_max"]) for r in subset], dtype=float)

            seed = int(90_000 + sum(ord(ch) for ch in f"{profile_name}-{alg_name}"))
            nr_mean, nr_ci_low, nr_ci_high = bootstrap_mean_ci(
                nr,
                resamples=BOOTSTRAP_RESAMPLES,
                alpha=ALPHA,
                seed=seed,
            )

            print(
                "  "
                + f"{alg_name:16s} "
                + f"Q_med={int(np.nanmedian(q)):6d} | "
                + f"nRMSE_med={np.nanmedian(nr):.3e} | "
                + f"nRMSE_mean={nr_mean:.3e} [{nr_ci_low:.3e}, {nr_ci_high:.3e}] | "
                + f"coverage={np.nanmean(cov):.2f} | "
                + f"time_med={np.nanmedian(runtime):.3f}s | "
                + f"K_med={np.nanmedian(kmax):.0f}"
            )


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    scenarios = build_amplitude_scenarios()

    print("Large-circuit robust benchmark")
    print(f"BAE backend mode: {BAE_KIND}")
    print(f"Amplitude scenarios kept = {len(scenarios)}")
    for scenario in scenarios:
        print(
            f"  s{scenario['scenario_id']:02d}: "
            f"offset={scenario['objective_ry_offset']:+.3f}, "
            f"a_true={scenario['a_true']:.4f}"
        )
    print(f"Budgets = {list(BUDGETS)}")
    print(f"Repetitions per scenario = {N_REP}")
    print(f"Probe ks = {PROBE_KS}")
    print(f"Probe shots = {PROBE_SHOTS}")

    calibration_rows: list[dict[str, Any]] = []
    calibration_summary_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # Robust T_eff calibration per (noise profile, amplitude scenario)
    # --------------------------------------------------------------------------
    calibrated_t_eff: dict[tuple[str, int], float | None] = {}

    for i, (profile_name, cfg) in enumerate(NOISE_PROFILES.items()):
        scale = float(cfg["scale"])
        noise_model = build_noise_model(scale)
        for scenario in scenarios:
            scenario_id = int(scenario["scenario_id"])
            objective_ry_offset = float(scenario["objective_ry_offset"])
            problem = scenario["problem"]
            a_true = float(scenario["a_true"])

            t_eff_samples: list[float] = []
            used_points_samples: list[int] = []
            for cal_rep in range(CALIBRATION_REPS):
                calib_sampler = AerCountSampler(
                    noise_model=noise_model,
                    seed=10_000 + i * 10_000 + scenario_id * 100 + cal_rep,
                )
                cal = calibrate_effective_T(
                    profile_name=profile_name,
                    sampler=calib_sampler,
                    problem=problem,
                    probe_ks=PROBE_KS,
                    probe_shots=PROBE_SHOTS,
                )

                used_points_samples.append(int(cal.used_points))
                if cal.t_eff is not None and np.isfinite(cal.t_eff) and float(cal.t_eff) > 0.0:
                    t_eff_samples.append(float(cal.t_eff))

                for row in cal.rows:
                    calibration_rows.append(
                        {
                            "profile": profile_name,
                            "scale": scale,
                            "scenario_id": scenario_id,
                            "objective_ry_offset": objective_ry_offset,
                            "a_true": a_true,
                            "calibration_rep": cal_rep + 1,
                            "t_eff_rep": np.nan if cal.t_eff is None else float(cal.t_eff),
                            "used_points": int(cal.used_points),
                            **row,
                        }
                    )

            if t_eff_samples:
                t_eff = float(np.median(np.asarray(t_eff_samples, dtype=float)))
            else:
                t_eff = None
            calibrated_t_eff[(profile_name, scenario_id)] = t_eff

            calibration_summary_rows.append(
                {
                    "profile": profile_name,
                    "scale": scale,
                    "scenario_id": scenario_id,
                    "objective_ry_offset": objective_ry_offset,
                    "a_true": a_true,
                    "t_eff": np.nan if t_eff is None else float(t_eff),
                    "n_valid_t_eff": int(len(t_eff_samples)),
                    "calibration_reps": int(CALIBRATION_REPS),
                    "used_points_median": float(np.median(np.asarray(used_points_samples, dtype=float))),
                }
            )

            print(
                f"[Calibration] profile={profile_name:9s} | s{scenario_id:02d} | "
                f"a_true={a_true:.4f} | "
                f"T_eff={'None' if t_eff is None else f'{t_eff:.3f}'} | "
                f"valid_reps={len(t_eff_samples)}/{CALIBRATION_REPS}"
            )

    # --------------------------------------------------------------------------
    # Main benchmark
    # --------------------------------------------------------------------------
    total_runs = len(NOISE_PROFILES) * len(scenarios) * N_REP * len(ALGORITHMS)
    run_counter = 0

    for profile_idx, (profile_name, cfg) in enumerate(NOISE_PROFILES.items()):
        scale = float(cfg["scale"])
        noise_model = build_noise_model(scale)

        for scenario in scenarios:
            scenario_id = int(scenario["scenario_id"])
            objective_ry_offset = float(scenario["objective_ry_offset"])
            problem = scenario["problem"]
            a_true = float(scenario["a_true"])
            t_eff = calibrated_t_eff[(profile_name, scenario_id)]

            print(
                f"\nRunning profile={profile_name}, s{scenario_id:02d}, "
                f"offset={objective_ry_offset:+.3f}, a_true={a_true:.4f}, scale={scale:.2f}, "
                f"T_eff={'None' if t_eff is None else f'{t_eff:.3f}'}"
            )

            for rep in range(N_REP):
                run_results: dict[str, dict[str, Any]] = {}

                for alg_idx, alg_key in enumerate(ALGORITHMS):
                    run_counter += 1
                    seed = int(
                        1_000_000
                        + profile_idx * 100_000
                        + scenario_id * 1_000
                        + rep * 10
                        + alg_idx
                    )

                    noisy_sampler = AerCountSampler(
                        noise_model=noise_model,
                        seed=seed + 2718,
                    )

                    try:
                        solver, is_bayes = build_solver(
                            alg_key,
                            RUN_EPSILON,
                            ALPHA,
                            NUM_SHOTS,
                            seed,
                            noisy_sampler,
                            t_eff,
                        )

                        t0 = time.perf_counter()

                        if alg_key == "bae":
                            np.random.seed(seed)
                            result = solver.estimate(
                                problem,
                                n_shots=NUM_SHOTS,
                                max_queries=MAX_ANALYSIS_BUDGET,
                            )
                        else:
                            result = solver.estimate(
                                problem,
                                bayes=is_bayes,
                                n_shots=NUM_SHOTS,
                                show_details=False,
                            )

                        runtime_seconds = float(time.perf_counter() - t0)

                        queries, estimations, k_sequence = extract_trace(
                            alg_key,
                            result,
                            NUM_SHOTS,
                        )

                        if len(queries) == 0:
                            print(
                                f"[{run_counter:04d}/{total_runs}] "
                                f"profile={profile_name:9s} | s{scenario_id:02d} | rep={rep + 1:02d} | "
                                f"{alg_key:15s} | no trajectory"
                            )
                            continue

                        final_est = float(estimations[-1])
                        final_abs_error = abs(final_est - a_true)
                        final_nrmse = final_abs_error / a_true
                        final_queries = int(getattr(result, "num_state_prep_calls", queries[-1]))
                        k_max = int(np.max(k_sequence)) if len(k_sequence) > 0 else 0

                        ci = getattr(result, "confidence_interval", None)
                        if ci is None:
                            coverage = np.nan
                        else:
                            coverage = float(float(ci[0]) <= a_true <= float(ci[1]))

                        final_rows.append(
                            {
                                "profile": profile_name,
                                "scale": scale,
                                "scenario_id": scenario_id,
                                "objective_ry_offset": objective_ry_offset,
                                "a_true": a_true,
                                "rep": rep + 1,
                                "algorithm": ALGORITHM_LABELS[alg_key],
                                "final_queries": final_queries,
                                "final_estimate": final_est,
                                "final_abs_error": final_abs_error,
                                "final_nrmse": final_nrmse,
                                "coverage": coverage,
                                "runtime_seconds": runtime_seconds,
                                "k_max": k_max,
                            }
                        )

                        run_results[alg_key] = {
                            "queries": queries,
                            "estimations": estimations,
                            "k_sequence": k_sequence,
                        }

                        print(
                            f"[{run_counter:04d}/{total_runs}] "
                            f"profile={profile_name:9s} | s{scenario_id:02d} | rep={rep + 1:02d} | "
                            f"{alg_key:15s} | Qmax={int(queries[-1]):6d} | "
                            f"final nRMSE={final_nrmse:.3e} | time={runtime_seconds:.3f}s | Kmax={k_max:3d}"
                        )

                    except Exception as e:
                        print(
                            f"[{run_counter:04d}/{total_runs}] "
                            f"Error in profile={profile_name}, s{scenario_id:02d}, "
                            f"rep={rep + 1}, alg={alg_key}: {e}"
                        )

                # --------------------------------------------------------------
                # Budget-aligned comparison
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
                        est = estimate_at_budget(
                            run_results[alg_key]["queries"],
                            run_results[alg_key]["estimations"],
                            int(budget),
                        )
                        if est is None:
                            continue

                        abs_error = abs(est - a_true)
                        nrmse = abs_error / a_true

                        budget_rows.append(
                            {
                                "profile": profile_name,
                                "scale": scale,
                                "scenario_id": scenario_id,
                                "objective_ry_offset": objective_ry_offset,
                                "a_true": a_true,
                                "rep": rep + 1,
                                "budget": int(budget),
                                "algorithm": ALGORITHM_LABELS[alg_key],
                                "estimate": float(est),
                                "abs_error": float(abs_error),
                                "nrmse": float(nrmse),
                                "common_budget_ceiling": int(common_budget_ceiling),
                            }
                        )

    # --------------------------------------------------------------------------
    # Save everything
    # --------------------------------------------------------------------------
    calibration_csv = os.path.join(current_dir, "robust_large_realistic_calibration_probe_rows.csv")
    save_csv(calibration_rows, calibration_csv)

    calibration_summary_csv = os.path.join(
        current_dir,
        "robust_large_realistic_calibration_summary.csv",
    )
    save_csv(calibration_summary_rows, calibration_summary_csv)

    budget_csv = os.path.join(current_dir, "robust_large_realistic_budget_rows.csv")
    save_csv(budget_rows, budget_csv)

    final_csv = os.path.join(current_dir, "robust_large_realistic_final_rows.csv")
    save_csv(final_rows, final_csv)

    budget_summary_rows = aggregate_budget_rows_robust(budget_rows)
    summary_csv = os.path.join(current_dir, "robust_large_realistic_budget_summary.csv")
    save_csv(budget_summary_rows, summary_csv)

    pairwise_rows = build_pairwise_rows(budget_rows)
    pairwise_csv = os.path.join(current_dir, "robust_large_realistic_pairwise.csv")
    save_csv(pairwise_rows, pairwise_csv)

    print(f"\nSaved calibration rows to: {calibration_csv}")
    print(f"Saved calibration summary to: {calibration_summary_csv}")
    print(f"Saved budget-aligned rows to: {budget_csv}")
    print(f"Saved final auxiliary rows to: {final_csv}")
    print(f"Saved budget summary rows to: {summary_csv}")
    print(f"Saved pairwise comparison rows to: {pairwise_csv}")

    # --------------------------------------------------------------------------
    # Summary and inference
    # --------------------------------------------------------------------------
    profile_names = list(NOISE_PROFILES.keys())
    print_budget_summary_robust(budget_summary_rows, profile_names)
    print_pairwise_summary(pairwise_rows, profile_names)
    print_final_summary_robust(final_rows, profile_names)

    # --------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------
    plot_path = os.path.join(current_dir, "robust_large_realistic_budget_aligned_panels.png")
    plot_budget_panels(
        budget_summary_rows,
        plot_path,
        NOISE_PROFILES.keys(),
        ALGORITHMS,
        ALGORITHM_LABELS,
        ALGORITHM_STYLES,
        BUDGETS,
    )
    print(f"Saved plot to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    run_experiment()