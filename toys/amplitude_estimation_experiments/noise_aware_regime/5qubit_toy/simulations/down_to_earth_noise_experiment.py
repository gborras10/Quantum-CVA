from __future__ import annotations

import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

import warnings

warnings.filterwarnings("ignore")
# --------------------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
toy_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")

for path in [src_dir, toy_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
try:
    from ae_pipeline_utils import (
        BAE_KIND,
        PHYSICAL_BACKEND_NAME,
        TRANSPILER_OPTIMIZATION_LEVEL,
        AerCountSampler,
        aggregate_budget_rows,
        build_problem_with_true_amplitude,
        build_noise_model,
        build_solver,
        calibrate_effective_T,
        choose_transpilation_plan,
        estimate_at_budget,
        extract_trace,
        kmax_at_budget,
        plot_budget_panels,
        print_budget_summary,
        print_final_summary,
        save_csv,
        time_to_budget,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
RUN_EPSILON_BY_ALGORITHM = {
    "bae": 5e-3,
    "biqae": 5e-4,
    "cabiqae_latentt": 3e-3,
}
ALPHA = 0.05
NUM_SHOTS = 64

BUDGETS = np.unique(
    np.round(np.logspace(np.log10(300), np.log10(15000), 8)).astype(int)  
)

MAX_ANALYSIS_BUDGET = int(BUDGETS[-1])

NOISE_PROFILE_NAME = "realistic"

N_REP = 10
OBJECTIVE_RY_OFFSETS = np.sort(np.random.default_rng(12345).uniform(-0.8, 0.8, 8))
MIN_VALID_A_TRUE = 0.04
MAX_VALID_A_TRUE = 0.96

PROBE_KS = [0, 1, 2, 4, 8]          
PROBE_SHOTS = 6_000                 

ALGORITHMS = ("bae", "biqae", "cabiqae_latentt")

ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE_latentt",
}

ALGORITHM_STYLES = {
    "bae": {"color": "#1D3557", "marker": "o"},
    "biqae": {"color": "#E63946", "marker": "s"},
    "cabiqae_latentt": {"color": "#2A9D8F", "marker": "^"},
}


def load_physical_transpile_backend():
    service = QiskitRuntimeService()
    return service.backend(PHYSICAL_BACKEND_NAME)


def select_benchmark_transpilation_plan(physical_backend, scenarios):
    if not scenarios:
        raise RuntimeError("Cannot select a transpilation plan without scenarios.")
    return choose_transpilation_plan(
        physical_backend,
        scenarios[0]["problem"],
        optimization_level=TRANSPILER_OPTIMIZATION_LEVEL,
        reference_ks=PROBE_KS,
    )


def build_amplitude_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for scenario_id, offset in enumerate(OBJECTIVE_RY_OFFSETS, start=1):
        problem, a_true = build_problem_with_true_amplitude(
            objective_ry_offset=float(offset)
        )
        if not (MIN_VALID_A_TRUE <= a_true <= MAX_VALID_A_TRUE):
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
        raise RuntimeError("No valid amplitude scenarios were generated.")
    return scenarios


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    scenarios = build_amplitude_scenarios()
    physical_backend = load_physical_transpile_backend()
    transpilation_plan = select_benchmark_transpilation_plan(
        physical_backend,
        scenarios,
    )

    print("Large-circuit benchmark")
    print(f"BAE backend mode: {BAE_KIND}")
    print(f"Physical transpilation backend = {PHYSICAL_BACKEND_NAME}")
    print(f"Initial layout = {list(transpilation_plan.initial_layout)}")
    print(f"Layout source = {transpilation_plan.candidate_source}")
    print(
        "Layout search metrics = "
        f"swaps:{transpilation_plan.aggregate_swap_count}, "
        f"2q:{transpilation_plan.aggregate_two_qubit_gates}, "
        f"depth:{transpilation_plan.aggregate_depth}, "
        f"size:{transpilation_plan.aggregate_size}, "
        f"plans:{transpilation_plan.evaluated_plans}"
    )
    print(f"Amplitude scenarios = {len(scenarios)}")
    print(f"Budgets = {list(BUDGETS)}")
    print(f"Repetitions per scenario = {N_REP}")
    print(f"Probe ks = {PROBE_KS}, probe shots = {PROBE_SHOTS}")
    print(
        "Execution circuits are transpiled with "
        f"optimization_level={TRANSPILER_OPTIMIZATION_LEVEL}, "
        f"seed_transpiler={transpilation_plan.seed_transpiler}, "
        f"routing_method={transpilation_plan.routing_method}"
    )

    calibration_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # T_eff calibration (una vez por scenario)
    # --------------------------------------------------------------------------
    calibrated_t_eff: dict[int, float | None] = {}

    for scenario in scenarios:
        scenario_id = int(scenario["scenario_id"])
        offset = float(scenario["objective_ry_offset"])
        problem = scenario["problem"]
        a_true = float(scenario["a_true"])

        calib_sampler = AerCountSampler(
            noise_model=build_noise_model(1.0),
            seed=10_000 + scenario_id,
            transpile_backend=physical_backend,
            transpilation_plan=transpilation_plan,
        )
        cal = calibrate_effective_T(
            profile_name=NOISE_PROFILE_NAME,
            sampler=calib_sampler,
            problem=problem,
            probe_ks=PROBE_KS,
            probe_shots=PROBE_SHOTS,
        )
        calibrated_t_eff[scenario_id] = cal.t_eff

        print(
            f"[Calibration] s{scenario_id:02d} | "
            f"offset={offset:+.3f} | a_true={a_true:.6f} | "
            f"T_eff={cal.t_eff if cal.t_eff is not None else 'None'} | "
            f"used_points={cal.used_points}"
        )

        for row in cal.rows:
            calibration_rows.append(
                {
                    "profile": NOISE_PROFILE_NAME,
                    "scenario_id": scenario_id,
                    "objective_ry_offset": offset,
                    "a_true": a_true,
                    "t_eff": np.nan if cal.t_eff is None else float(cal.t_eff),
                    **row,
                }
            )

    # --------------------------------------------------------------------------
    # Main benchmark
    # --------------------------------------------------------------------------
    for scenario in scenarios:
        scenario_id = int(scenario["scenario_id"])
        offset = float(scenario["objective_ry_offset"])
        problem = scenario["problem"]
        a_true = float(scenario["a_true"])
        t_eff = calibrated_t_eff[scenario_id]

        print(
            f"\nRunning s{scenario_id:02d} | "
            f"offset={offset:+.3f} | a_true={a_true:.6f} | "
            f"T_eff={'None' if t_eff is None else f'{t_eff:.3f}'}"
        )

        for rep in range(N_REP):
            run_results: dict[str, dict[str, Any]] = {}
            noisy_sampler = AerCountSampler(
                noise_model=build_noise_model(1.0),
                seed=100_000 + scenario_id * 1000 + rep,
                transpile_backend=physical_backend,
                transpilation_plan=transpilation_plan,
            )

            for alg_idx, alg_key in enumerate(ALGORITHMS):
                seed = int(1_000_000 + scenario_id * 10_000 + rep * 100 + alg_idx)

                try:
                    run_epsilon = RUN_EPSILON_BY_ALGORITHM[alg_key]
                    solver, is_bayes = build_solver(
                        alg_key,
                        run_epsilon,
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
                            f"  s{scenario_id:02d} | rep={rep + 1:02d} | "
                            f"{alg_key:15s} | no trajectory"
                        )
                        continue

                    final_est = float(estimations[-1])
                    final_abs_error = abs(final_est - a_true)
                    final_nrmse = final_abs_error / a_true
                    final_queries = int(getattr(result, "num_state_prep_calls", queries[-1]))
                    k_max = int(np.max(k_sequence)) if len(k_sequence) > 0 else 0

                    ci = getattr(result, "confidence_interval", None)
                    coverage = (
                        np.nan
                        if ci is None
                        else float(float(ci[0]) <= a_true <= float(ci[1]))
                    )

                    final_rows.append(
                        {
                            "profile": NOISE_PROFILE_NAME,
                            "scenario_id": scenario_id,
                            "objective_ry_offset": offset,
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
                        "runtime_seconds": runtime_seconds,
                    }

                    print(
                        f"  s{scenario_id:02d} | rep={rep + 1:02d} | "
                        f"{alg_key:15s} | "
                        f"Qmax={int(queries[-1]):6d} | "
                        f"final nRMSE={final_nrmse:.3e} | "
                        f"time={runtime_seconds:.3f}s | "
                        f"Kmax={k_max:3d}"
                    )

                except Exception as e:
                    print(
                        f"  Error in s{scenario_id:02d}, rep={rep + 1}, alg={alg_key}: {e}"
                    )

            # ------------------------------------------------------------------
            # Budget-aligned comparison — LÓGICA CORREGIDA
            #
            # Problema original: se usaba
            #
            #   common_budget_ceiling = min(trayectoria_alg_1[-1],
            #                              trayectoria_alg_2[-1],
            #                              trayectoria_alg_3[-1])
            #
            # y se descartaban los budgets > ceiling para el run entero.
            # Esto hacía que a budgets altos solo sobrevivieran los runs donde
            # todos los algoritmos tardaron mucho (casos difíciles), sesgando
            # la muestra de forma distinta en cada punto del eje x.
            #
            # Solución: filtrar POR BUDGET, no por run.
            # Para cada budget B, solo se incluye este run si los tres algoritmos
            # tienen trayectoria que llega genuinamente a B queries. Así la
            # muestra de runs que compone cada punto del eje x es la misma para
            # los tres algoritmos, comparación limpia.
            # ------------------------------------------------------------------
            if not all(alg in run_results for alg in ALGORITHMS):
                # Algún algoritmo petó completamente: run no comparable.
                continue

            for budget in BUDGETS:
                if int(budget) > MAX_ANALYSIS_BUDGET:
                    continue

                # Todos los algoritmos deben haber alcanzado este budget
                # con su propia trayectoria (sin plateau ni extrapolación).
                all_reached = all(
                    len(run_results[alg]["queries"]) > 0
                    and int(run_results[alg]["queries"][-1]) >= int(budget)
                    for alg in ALGORITHMS
                )
                if not all_reached:
                    # En este budget concreto, al menos un algoritmo no
                    # llegó: descartamos solo este punto, no el run entero.
                    continue

                for alg_key in ALGORITHMS:
                    budget_int = int(budget)
                    est = estimate_at_budget(
                        run_results[alg_key]["queries"],
                        run_results[alg_key]["estimations"],
                        budget_int,
                    )
                    if est is None:
                        continue

                    kmax_budget = kmax_at_budget(
                        run_results[alg_key]["queries"],
                        run_results[alg_key]["k_sequence"],
                        budget_int,
                    )
                    time_budget = time_to_budget(
                        run_results[alg_key]["queries"],
                        float(run_results[alg_key]["runtime_seconds"]),
                        budget_int,
                    )
                    if kmax_budget is None or time_budget is None:
                        continue

                    abs_error = abs(est - a_true)
                    nrmse = abs_error / a_true

                    budget_rows.append(
                        {
                            "profile": NOISE_PROFILE_NAME,
                            "scenario_id": scenario_id,
                            "objective_ry_offset": offset,
                            "a_true": a_true,
                            "rep": rep + 1,
                            "budget": budget_int,
                            "algorithm": ALGORITHM_LABELS[alg_key],
                            "estimate": float(est),
                            "abs_error": float(abs_error),
                            "nrmse": float(nrmse),
                            "k_max_budget": int(kmax_budget),
                            "time_to_budget_seconds": float(time_budget),
                        }
                    )

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    calibration_csv = os.path.join(current_dir, "large_realistic_calibration.csv")
    save_csv(calibration_rows, calibration_csv)

    budget_csv = os.path.join(current_dir, "large_realistic_budget_rows.csv")
    save_csv(budget_rows, budget_csv)

    final_csv = os.path.join(current_dir, "large_realistic_final_rows.csv")
    save_csv(final_rows, final_csv)

    print(f"\nSaved calibration rows  → {calibration_csv}")
    print(f"Saved budget-aligned rows → {budget_csv}")
    print(f"Saved final rows         → {final_csv}")

    # --------------------------------------------------------------------------
    # Summary + plot
    # --------------------------------------------------------------------------
    summary_rows = aggregate_budget_rows(budget_rows, ALGORITHMS, ALGORITHM_LABELS)

    # Muestra cuántos runs contribuyen a cada budget (útil para detectar
    # si algún algoritmo desaparece a budgets altos por no llegar).
    _print_coverage_table(budget_rows)

    print_budget_summary(summary_rows, [NOISE_PROFILE_NAME])
    print_final_summary(final_rows, [NOISE_PROFILE_NAME], ALGORITHMS, ALGORITHM_LABELS)

    plot_path = os.path.join(current_dir, "large_realistic_budget_aligned_panels.png")
    plot_budget_panels(
        summary_rows,
        plot_path,
        [NOISE_PROFILE_NAME],
        ALGORITHMS,
        ALGORITHM_LABELS,
        ALGORITHM_STYLES,
        BUDGETS,
    )
    print(f"Saved plot → {plot_path}")
    _plot_error_vs_runtime(summary_rows)
    plt.show()


def _print_coverage_table(budget_rows: list[dict[str, Any]]) -> None:
    """
    Imprime cuántos runs (scenario, rep) contribuyen a cada (budget, algoritmo).
    Si un algoritmo desaparece a budgets altos es señal de que sus trayectorias
    son más cortas que el budget: el filtro por-budget lo está excluyendo.
    """
    print("\n=== Runs que contribuyen a cada budget (per algoritmo) ===")
    budgets = sorted({int(r["budget"]) for r in budget_rows})
    alg_names = list(ALGORITHM_LABELS.values())

    header = f"{'budget':>8}  " + "  ".join(f"{a:>18}" for a in alg_names)
    print(header)
    for budget in budgets:
        counts = {
            alg: sum(
                1 for r in budget_rows
                if int(r["budget"]) == budget and r["algorithm"] == alg
            )
            for alg in alg_names
        }
        row = f"{budget:>8}  " + "  ".join(f"{counts[a]:>18}" for a in alg_names)
        print(row)


def _plot_error_vs_runtime(summary_rows: list[dict[str, Any]]) -> None:
    """
    Muestra una figura adicional con error frente a runtime.
    Cada punto representa un budget agregado y se anota con su valor.
    """
    if not summary_rows:
        return

    profiles = sorted({str(row["profile"]) for row in summary_rows})
    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(5.8 * len(profiles), 4.8),
        squeeze=False,
    )

    for j, profile in enumerate(profiles):
        ax = axes[0, j]
        prof_rows = [row for row in summary_rows if str(row["profile"]) == profile]

        for alg_key in ALGORITHMS:
            alg_name = ALGORITHM_LABELS[alg_key]
            subset = [row for row in prof_rows if row["algorithm"] == alg_name]
            if not subset:
                continue

            runtime = np.asarray(
                [float(row.get("time_to_budget_seconds_median", np.nan)) for row in subset],
                dtype=float,
            )
            error = np.asarray(
                [float(row.get("nrmse_median", np.nan)) for row in subset],
                dtype=float,
            )
            budgets = np.asarray([int(row["budget"]) for row in subset], dtype=int)

            valid = np.isfinite(runtime) & np.isfinite(error) & (runtime > 0.0) & (error > 0.0)
            if not np.any(valid):
                continue

            runtime = runtime[valid]
            error = error[valid]
            budgets = budgets[valid]

            order = np.argsort(runtime)
            runtime = runtime[order]
            error = error[order]
            budgets = budgets[order]

            style = ALGORITHM_STYLES[alg_key]
            ax.loglog(
                runtime,
                error,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5,
                label=alg_name,
            )

            for x_val, y_val, budget in zip(runtime, error, budgets):
                ax.annotate(
                    f"B={budget}",
                    xy=(x_val, y_val),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=style["color"],
                    alpha=0.9,
                )

        ax.set_xlabel("Median runtime to budget [s]")
        ax.set_ylabel("Median normalized RMSE")
        ax.grid(True, which="both", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(labels), 4),
            frameon=False,
            columnspacing=1.6,
            handlelength=2.2,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

if __name__ == "__main__":
    run_experiment()
