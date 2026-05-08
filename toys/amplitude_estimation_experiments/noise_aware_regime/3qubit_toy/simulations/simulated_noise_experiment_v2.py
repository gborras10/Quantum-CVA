from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

# --------------------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
TOY_DIR = CURRENT_DIR.parent
ROOT_DIR = CURRENT_DIR.parents[4]
SRC_DIR = ROOT_DIR / "src"

for path in (SRC_DIR, TOY_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# --------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------
from ae_pipeline_utils import (  # noqa: E402
    BAE_KIND,
    PHYSICAL_BACKEND_NAME,
    TRANSPILER_OPTIMIZATION_LEVEL,
    AerCountSampler,
    build_noise_model,
    build_problem_with_true_amplitude,
    build_solver,
    calibrate_effective_T,
    choose_transpilation_plan,
    extract_trace,
    save_csv,
)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
NOISE_PROFILE_NAME = "realistic"

# Iterative algorithms run with epsilon sweep.
ALGORITHMS_ITERATIVE = ("cabiqae_latentt",)
# BAE is run in phase 2 with max_queries matched to iterative medians.
ALGORITHM_BAE =  "bae"

ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE_latentt",
}

# Selective algorithm execution: set to None to run all, or tuple of specific algorithms.
# If BAE is not included, phase 2 will be skipped automatically.
ALGORITHMS_TO_RUN = ("cabiqae_latentt",)  # Run all by default

ALPHA = 0.05
NUM_SHOTS = 128

PROBE_KS = [0, 1, 2, 3, 4, 6, 8, 10, 12]
PROBE_SHOTS = 12_000

N_REP = 15

OBJECTIVE_RY_OFFSETS = np.array([0.1, 0.60, 0.85], dtype=float)
MIN_VALID_A_TRUE = 0.04
MAX_VALID_A_TRUE = 0.96

# Epsilon targets used for iterative algorithms and as matching anchors for BAE budgets.
EPSILON_TARGETS = np.array([5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3], dtype=float)

# BAE epsilon is kept fixed; stopping is controlled by max_queries target in phase 2.
BAE_INTERNAL_EPSILON = 1e-6
BAE_FALLBACK_MAX_QUERIES = 15_000

ALGORITHM_CONFIG = {
    "bae": {"n_shots": None},
    "biqae": {"n_shots": None},
    "cabiqae_latentt": {"n_shots": None},
}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def load_physical_transpile_backend() -> Any:
    service = QiskitRuntimeService()
    return service.backend(PHYSICAL_BACKEND_NAME)


def build_amplitude_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for scenario_id, offset in enumerate(OBJECTIVE_RY_OFFSETS, start=1):
        problem, a_true = build_problem_with_true_amplitude(objective_ry_offset=float(offset))
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


def calibrate_scenarios(
    scenarios: list[dict[str, Any]],
    physical_backend: Any,
    transpilation_plan: Any,
) -> tuple[dict[int, float | None], list[dict[str, Any]]]:
    calibrated_t_eff: dict[int, float | None] = {}
    calibration_rows: list[dict[str, Any]] = []

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

    return calibrated_t_eff, calibration_rows


def _configured_n_shots_for(algorithm_key: str) -> int | None:
    raw = ALGORITHM_CONFIG[algorithm_key]["n_shots"]
    return None if raw is None else int(raw)


def _final_row_from_result(
    *,
    phase: str,
    repetition: int,
    scenario_id: int,
    algorithm_key: str,
    epsilon_target: float,
    objective_ry_offset: float,
    a_true: float,
    result: Any,
    num_queries: int,
    elapsed_runtime_seconds: float,
    max_queries_bae_target: float | None = None,
) -> dict[str, Any]:
    final_estimate = float(result.estimation)
    abs_error = abs(final_estimate - float(a_true))
    normalized_abs_error = float(abs_error / float(a_true))
    confidence_interval = getattr(result, "confidence_interval", None)

    return {
        "profile": NOISE_PROFILE_NAME,
        "phase": phase,
        "repetition": int(repetition),
        "scenario_id": int(scenario_id),
        "algorithm": ALGORITHM_LABELS[algorithm_key],
        "algorithm_key": algorithm_key,
        "epsilon_target": float(epsilon_target),
        "objective_ry_offset": float(objective_ry_offset),
        "a_true": float(a_true),
        "final_estimate": final_estimate,
        "abs_error": float(abs_error),
        "normalized_abs_error": normalized_abs_error,
        "num_queries": int(num_queries),
        "elapsed_runtime_seconds": float(elapsed_runtime_seconds),
        "confidence_interval": confidence_interval,
        "max_queries_bae_target": (
            float(max_queries_bae_target) if max_queries_bae_target is not None else np.nan
        ),
        "n_shots": _configured_n_shots_for(algorithm_key),
        "bae_kind": BAE_KIND,
        "noise_profile": NOISE_PROFILE_NAME,
    }


def phase_1_iterative_algorithms(
    *,
    scenarios: list[dict[str, Any]],
    physical_backend: Any,
    transpilation_plan: Any,
    calibrated_t_eff: dict[int, float | None],
    n_rep: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    final_estimations: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    queries_per_eps_alg: dict[tuple[str, str], list[int]] = {}

    print("\n" + "=" * 90)
    print("PHASE 1: Iterative algorithms with epsilon sweep under simulated noise")
    print("=" * 90)

    for scenario in scenarios:
        scenario_id = int(scenario["scenario_id"])
        objective_ry_offset = float(scenario["objective_ry_offset"])
        problem = scenario["problem"]
        target_a = float(scenario["a_true"])
        t_eff = calibrated_t_eff[scenario_id]

        for rep in range(n_rep):
            noisy_sampler = AerCountSampler(
                noise_model=build_noise_model(1.0),
                seed=100_000 + scenario_id * 10_000 + rep,
                transpile_backend=physical_backend,
                transpilation_plan=transpilation_plan,
            )

            for eps_idx, epsilon_target in enumerate(EPSILON_TARGETS):
                for alg_idx, algorithm_key in enumerate(ALGORITHMS_ITERATIVE):
                    seed = int(1_000_000 + scenario_id * 100_000 + rep * 1_000 + eps_idx * 100 + alg_idx)
                    configured_n_shots = _configured_n_shots_for(algorithm_key)
                    run_n_shots = int(configured_n_shots) if configured_n_shots is not None else int(NUM_SHOTS)

                    try:
                        solver, is_bayes = build_solver(
                            algorithm_key,
                            float(epsilon_target),
                            float(ALPHA),
                            run_n_shots,
                            seed,
                            noisy_sampler,
                            t_eff,
                        )

                        t0 = time.perf_counter()
                        result = solver.estimate(
                            problem,
                            bayes=is_bayes,
                            n_shots=run_n_shots,
                            show_details=False,
                        )
                        elapsed_runtime_seconds = float(time.perf_counter() - t0)

                        queries, _, _ = extract_trace(algorithm_key, result, run_n_shots)
                        if len(queries) == 0:
                            raise RuntimeError("empty trajectory")

                        num_queries = int(queries[-1])
                        final_row = _final_row_from_result(
                            phase="iterative",
                            repetition=rep,
                            scenario_id=scenario_id,
                            algorithm_key=algorithm_key,
                            epsilon_target=float(epsilon_target),
                            objective_ry_offset=objective_ry_offset,
                            a_true=target_a,
                            result=result,
                            num_queries=num_queries,
                            elapsed_runtime_seconds=elapsed_runtime_seconds,
                        )
                        final_estimations.append(final_row)

                        key = (str(float(epsilon_target)), algorithm_key)
                        queries_per_eps_alg.setdefault(key, []).append(num_queries)

                        print(
                            f"s{scenario_id:02d} rep={rep + 1:02d} eps={epsilon_target:.1e} "
                            f"{ALGORITHM_LABELS[algorithm_key]:15s} Q={num_queries:6d} "
                            f"nabs_err={final_row['normalized_abs_error']:.3e} t={elapsed_runtime_seconds:.3f}s"
                        )

                    except Exception as exc:  # noqa: BLE001
                        error_rows.append(
                            {
                                "phase": "iterative",
                                "scenario_id": scenario_id,
                                "repetition": int(rep),
                                "epsilon_target": float(epsilon_target),
                                "algorithm": ALGORITHM_LABELS[algorithm_key],
                                "algorithm_key": algorithm_key,
                                "objective_ry_offset": objective_ry_offset,
                                "a_true": target_a,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )

    median_queries: dict[str, dict[str, float]] = {}
    for epsilon_target in EPSILON_TARGETS:
        epsilon_key = str(float(epsilon_target))
        median_queries[epsilon_key] = {}
        for algorithm_key in ALGORITHMS_ITERATIVE:
            values = queries_per_eps_alg.get((epsilon_key, algorithm_key), [])
            if values:
                median_queries[epsilon_key][algorithm_key] = float(np.median(np.asarray(values, dtype=float)))

    return final_estimations, error_rows, median_queries


def phase_2_bae_budget_matched(
    *,
    scenarios: list[dict[str, Any]],
    physical_backend: Any,
    transpilation_plan: Any,
    calibrated_t_eff: dict[int, float | None],
    median_queries: dict[str, dict[str, float]],
    n_rep: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final_estimations: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    print("\n" + "=" * 90)
    print("PHASE 2: BAE with max_queries matched to iterative median query targets")
    print("=" * 90)

    for scenario in scenarios:
        scenario_id = int(scenario["scenario_id"])
        objective_ry_offset = float(scenario["objective_ry_offset"])
        problem = scenario["problem"]
        target_a = float(scenario["a_true"])
        t_eff = calibrated_t_eff[scenario_id]

        for rep in range(n_rep):
            noisy_sampler = AerCountSampler(
                noise_model=build_noise_model(1.0),
                seed=200_000 + scenario_id * 10_000 + rep,
                transpile_backend=physical_backend,
                transpilation_plan=transpilation_plan,
            )

            for eps_idx, epsilon_target in enumerate(EPSILON_TARGETS):
                epsilon_key = str(float(epsilon_target))
                per_alg_targets = [
                    median_queries.get(epsilon_key, {}).get(alg) for alg in ALGORITHMS_ITERATIVE
                ]
                valid_targets = [float(value) for value in per_alg_targets if value is not None]
                max_queries_target = (
                    float(np.median(np.asarray(valid_targets, dtype=float)))
                    if valid_targets
                    else float(BAE_FALLBACK_MAX_QUERIES)
                )

                algorithm_key = ALGORITHM_BAE
                seed = int(3_000_000 + scenario_id * 100_000 + rep * 1_000 + eps_idx)
                configured_n_shots = _configured_n_shots_for(algorithm_key)
                run_n_shots = int(configured_n_shots) if configured_n_shots is not None else int(NUM_SHOTS)

                try:
                    solver, _ = build_solver(
                        algorithm_key,
                        float(BAE_INTERNAL_EPSILON),
                        float(ALPHA),
                        run_n_shots,
                        seed,
                        noisy_sampler,
                        t_eff,
                    )

                    t0 = time.perf_counter()
                    np.random.seed(seed)
                    result = solver.estimate(
                        problem,
                        n_shots=run_n_shots,
                        max_queries=int(max_queries_target),
                    )
                    elapsed_runtime_seconds = float(time.perf_counter() - t0)

                    queries, _, _ = extract_trace(algorithm_key, result, run_n_shots)
                    if len(queries) == 0:
                        raise RuntimeError("empty trajectory")

                    num_queries = int(queries[-1])
                    final_row = _final_row_from_result(
                        phase="bae_budget_match",
                        repetition=rep,
                        scenario_id=scenario_id,
                        algorithm_key=algorithm_key,
                        epsilon_target=float(epsilon_target),
                        objective_ry_offset=objective_ry_offset,
                        a_true=target_a,
                        result=result,
                        num_queries=num_queries,
                        elapsed_runtime_seconds=elapsed_runtime_seconds,
                        max_queries_bae_target=max_queries_target,
                    )
                    final_estimations.append(final_row)

                    print(
                        f"s{scenario_id:02d} rep={rep + 1:02d} eps={epsilon_target:.1e} "
                        f"{ALGORITHM_LABELS[algorithm_key]:15s} Qtarget={int(max_queries_target):6d} "
                        f"Qactual={num_queries:6d} nabs_err={final_row['normalized_abs_error']:.3e} "
                        f"t={elapsed_runtime_seconds:.3f}s"
                    )

                except Exception as exc:  # noqa: BLE001
                    error_rows.append(
                        {
                            "phase": "bae_budget_match",
                            "scenario_id": scenario_id,
                            "repetition": int(rep),
                            "epsilon_target": float(epsilon_target),
                            "algorithm": ALGORITHM_LABELS[algorithm_key],
                            "algorithm_key": algorithm_key,
                            "objective_ry_offset": objective_ry_offset,
                            "a_true": target_a,
                            "max_queries_bae_target": max_queries_target,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    return final_estimations, error_rows


def _median_query_rows(median_queries: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epsilon_key, values in sorted(median_queries.items(), key=lambda item: float(item[0])):
        per_alg = [values.get(alg) for alg in ALGORITHMS_ITERATIVE if values.get(alg) is not None]
        match_target = float(np.median(np.asarray(per_alg, dtype=float))) if per_alg else float(BAE_FALLBACK_MAX_QUERIES)
        row = {
            "epsilon_target": float(epsilon_key),
            "bae_query_target": float(match_target),
        }
        for alg in ALGORITHMS_ITERATIVE:
            key = f"{alg}_median_queries"
            row[key] = float(values[alg]) if alg in values else np.nan
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    scenarios = build_amplitude_scenarios()
    physical_backend = load_physical_transpile_backend()
    transpilation_plan = choose_transpilation_plan(
        physical_backend,
        scenarios[0]["problem"],
        optimization_level=TRANSPILER_OPTIMIZATION_LEVEL,
        reference_ks=PROBE_KS,
    )

    print("Simulated-noise v2 benchmark")
    print(f"Noise profile               = {NOISE_PROFILE_NAME}")
    print(f"BAE backend mode            = {BAE_KIND}")
    print(f"Physical transpile backend  = {PHYSICAL_BACKEND_NAME}")
    print(f"Scenarios                   = {len(scenarios)}")
    print(f"Repetitions per scenario    = {N_REP}")
    print(f"Iterative eps targets       = {EPSILON_TARGETS.tolist()}")
    print(
        "Algorithms                  = "
        + ", ".join(ALGORITHM_LABELS[a] for a in (*ALGORITHMS_ITERATIVE, ALGORITHM_BAE))
    )

    calibrated_t_eff, calibration_rows = calibrate_scenarios(
        scenarios,
        physical_backend,
        transpilation_plan,
    )

    final_iterative, errors_iterative, median_queries = phase_1_iterative_algorithms(
        scenarios=scenarios,
        physical_backend=physical_backend,
        transpilation_plan=transpilation_plan,
        calibrated_t_eff=calibrated_t_eff,
        n_rep=N_REP,
    )

    # Phase 2 (BAE) is only executed if BAE is in ALGORITHMS_TO_RUN (or if all algorithms are run).
    run_phase_2_bae = ALGORITHMS_TO_RUN is None or "bae" in ALGORITHMS_TO_RUN
    
    if run_phase_2_bae:
        final_bae, errors_bae = phase_2_bae_budget_matched(
            scenarios=scenarios,
            physical_backend=physical_backend,
            transpilation_plan=transpilation_plan,
            calibrated_t_eff=calibrated_t_eff,
            median_queries=median_queries,
            n_rep=N_REP,
        )
    else:
        print("\nPHASE 2: BAE skipped (not in ALGORITHMS_TO_RUN)\n")
        final_bae, errors_bae = [], []

    final_estimations = final_iterative + final_bae
    error_rows = errors_iterative + errors_bae
    query_target_rows = _median_query_rows(median_queries)

    output_dir = CURRENT_DIR / "experiment_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = "large_realistic_v2"
    calibration_csv = output_dir / f"{output_prefix}_calibration.csv"
    final_estimations_csv = output_dir / f"{output_prefix}_final_estimations.csv"
    errors_csv = output_dir / f"{output_prefix}_errors.csv"
    query_targets_csv = output_dir / f"{output_prefix}_bae_query_targets.csv"

    save_csv(calibration_rows, str(calibration_csv))
    save_csv(final_estimations, str(final_estimations_csv))
    save_csv(error_rows, str(errors_csv))
    save_csv(query_target_rows, str(query_targets_csv))

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPLETED")
    print("=" * 90)
    print(f"Saved calibration rows     -> {calibration_csv}")
    print(f"Saved final estimations    -> {final_estimations_csv}")
    print(f"Saved error rows           -> {errors_csv}")
    print(f"Saved BAE query targets    -> {query_targets_csv}")
    print(f"Final rows count           -> {len(final_estimations)}")
    print(f"Error rows count           -> {len(error_rows)}")


if __name__ == "__main__":
    run_experiment()
