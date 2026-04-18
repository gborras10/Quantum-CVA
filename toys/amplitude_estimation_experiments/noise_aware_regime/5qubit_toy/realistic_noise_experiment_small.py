from __future__ import annotations

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
        aggregate_budget_rows,
        build_large_problem,
        build_noise_model,
        build_solver,
        calibrate_effective_T,
        estimate_at_budget,
        extract_trace,
        plot_budget_panels,
        print_budget_summary,
        print_final_summary,
        save_csv,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
RUN_EPSILON = 1e-3
ALPHA = 0.05
NUM_SHOTS = 48

# Budget-aligned comparison.
BUDGETS = np.array([300, 600, 1200, 2400, 4800, 9600, 19200], dtype=int)
MAX_ANALYSIS_BUDGET = int(BUDGETS[-1])

# Three realistic noise severities on the same larger circuit.
NOISE_PROFILES = {
    "mild": {"scale": 0.75},
    "moderate": {"scale": 1.00},
    "harsh": {"scale": 1.50},
}

N_REP = 1

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
PROBE_SHOTS = 20000


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    problem, a_true = build_large_problem()
    print("Large-circuit benchmark")
    print(f"BAE backend mode: {BAE_KIND}")
    print(f"True amplitude a_true = {a_true:.6f}")
    print(f"Budgets = {list(BUDGETS)}")
    print(f"Probe ks = {PROBE_KS}")

    calibration_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # One T_eff calibration per noise profile
    # --------------------------------------------------------------------------
    calibrated_t_eff: dict[str, float | None] = {}

    for i, (profile_name, cfg) in enumerate(NOISE_PROFILES.items()):
        scale = float(cfg["scale"])
        calib_sampler = AerCountSampler(
            noise_model=build_noise_model(scale),
            seed=10_000 + i,
        )
        cal = calibrate_effective_T(
            profile_name=profile_name,
            sampler=calib_sampler,
            problem=problem,
            probe_ks=PROBE_KS,
            probe_shots=PROBE_SHOTS,
        )
        calibrated_t_eff[profile_name] = cal.t_eff

        print(
            f"[Calibration] {profile_name:9s} | "
            f"T_eff={cal.t_eff if cal.t_eff is not None else 'None'} | "
            f"used_points={cal.used_points}"
        )

        for row in cal.rows:
            calibration_rows.append(
                {
                    "profile": profile_name,
                    "scale": scale,
                    "t_eff": np.nan if cal.t_eff is None else float(cal.t_eff),
                    **row,
                }
            )

    # --------------------------------------------------------------------------
    # Main benchmark
    # --------------------------------------------------------------------------
    for profile_idx, (profile_name, cfg) in enumerate(NOISE_PROFILES.items()):
        scale = float(cfg["scale"])
        t_eff = calibrated_t_eff[profile_name]

        print(
            f"\nRunning profile={profile_name}, scale={scale:.2f}, "
            f"T_eff={'None' if t_eff is None else f'{t_eff:.3f}'}"
        )

        for rep in range(N_REP):
            run_results: dict[str, dict[str, Any]] = {}
            noisy_sampler = AerCountSampler(
                noise_model=build_noise_model(scale),
                seed=100_000 + profile_idx * 1000 + rep,
            )

            for alg_idx, alg_key in enumerate(ALGORITHMS):
                seed = int(1_000_000 + profile_idx * 10_000 + rep * 100 + alg_idx)
 
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
                            f"profile={profile_name:9s} | rep={rep + 1:02d} | "
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
                        f"profile={profile_name:9s} | rep={rep + 1:02d} | "
                        f"{alg_key:15s} | "
                        f"Qmax={int(queries[-1]):6d} | "
                        f"final nRMSE={final_nrmse:.3e} | "
                        f"time={runtime_seconds:.3f}s | "
                        f"Kmax={k_max:3d}"
                    )

                except Exception as e:
                    print(
                        f"Error in profile={profile_name}, rep={rep + 1}, alg={alg_key}: {e}"
                    )

            # ------------------------------------------------------------------
            # Budget-aligned comparison
            # ------------------------------------------------------------------
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
    calibration_csv = os.path.join(current_dir, "large_realistic_calibration.csv")
    save_csv(calibration_rows, calibration_csv)

    budget_csv = os.path.join(current_dir, "large_realistic_budget_rows.csv")
    save_csv(budget_rows, budget_csv)

    final_csv = os.path.join(current_dir, "large_realistic_final_rows.csv")
    save_csv(final_rows, final_csv)

    print(f"\nSaved calibration rows to: {calibration_csv}")
    print(f"Saved budget-aligned rows to: {budget_csv}")
    print(f"Saved final auxiliary rows to: {final_csv}")

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    summary_rows = aggregate_budget_rows(
        budget_rows,
        ALGORITHMS,
        ALGORITHM_LABELS,
    )

    print_budget_summary(summary_rows, NOISE_PROFILES.keys())
    print_final_summary(final_rows, NOISE_PROFILES.keys(), ALGORITHMS, ALGORITHM_LABELS)

    # --------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------
    plot_path = os.path.join(current_dir, "large_realistic_budget_aligned_panels.png")
    plot_budget_panels(
        summary_rows,
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
