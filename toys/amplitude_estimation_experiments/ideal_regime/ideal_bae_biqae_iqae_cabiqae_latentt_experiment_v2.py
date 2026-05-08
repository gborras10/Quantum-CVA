import os
import sys
import time
from typing import Any
import random
import numpy as np

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
    true_amplitude_for_offset,
)
from toys.amplitude_estimation_experiments.ideal_regime.ideal_utils import (
    extract_trace,
    save_csv,
)


ALGORITHMS_ITERATIVE = ("cabiqae_latentt",)
ALGORITHM_LABELS = {
    "cabiqae_latentt": "CABIQAE",
}

OBJECTIVE_RY_OFFSETS = np.array([0.30, 0.44, 0.48, 0.52, 0.56, 0.60, 0.63, 0.67], dtype=float)

# Epsilon targets to sweep over (for iterative algorithms)
EPSILON_TARGETS = np.array([5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2], dtype=float)

# Base algorithm configuration
ALGORITHM_CONFIG = {
    "cabiqae_latentt": {
        "n_shots": None,
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


def phase_1_iterative_algorithms(n_rep: int = 50) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, float]]]:
    """
    Phase 1: Run iterative algorithms (BIQAE, IQAE, CABIQAE) with epsilon sweep.
    Calculate and return median queries for each epsilon and algorithm.
    
    Returns:
        - final_estimations: list of result dicts
        - error_rows: list of error dicts
        - median_queries: dict[epsilon_str][algorithm] = median_queries
    """
    alpha = 0.05
    
    final_estimations: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    queries_per_eps_alg: dict[tuple[str, str], list[int]] = {}  # (epsilon_str, alg) -> [queries...]
    
    rng = np.random.default_rng(1234)
    
    print("=" * 80)
    print("PHASE 1: Running iterative algorithms (BIQAE, IQAE, CABIQAE) with epsilon sweep")
    print("=" * 80)
    print(f"Epsilon targets: {EPSILON_TARGETS}")
    print(f"Repetitions: {n_rep}\n")

    for rep in range(n_rep):
        objective_ry_offset = float(rng.choice(OBJECTIVE_RY_OFFSETS))
        problem = build_problem(objective_ry_offset)
        target_a = true_amplitude_for_offset(objective_ry_offset)

        for eps_idx, epsilon_target in enumerate(EPSILON_TARGETS):
            for alg_idx, algorithm in enumerate(ALGORITHMS_ITERATIVE):
                alg_cfg = ALGORITHM_CONFIG[algorithm]
                configured_n_shots = (
                    None if alg_cfg["n_shots"] is None else int(alg_cfg["n_shots"])
                )
                seed = int(1_000_000 + rep * 1000 + eps_idx * 100 + alg_idx)
                solver, bayes = _build_solver(
                    algorithm=algorithm,
                    epsilon_target=float(epsilon_target),
                    alpha=alpha,
                    n_shots=configured_n_shots,
                    seed=seed,
                    estimate_T=bool(alg_cfg.get("estimate_T", False)),
                )

                try:
                    start_time = time.perf_counter()
                    estimate_kwargs = {
                        "bayes": bayes,
                        "show_details": False,
                    }
                    if configured_n_shots is not None:
                        estimate_kwargs["n_shots"] = configured_n_shots
                    result = solver.estimate(problem, **estimate_kwargs)
                    elapsed_runtime_seconds = time.perf_counter() - start_time
                except Exception as exc:
                    error_rows.append(
                        {
                            "phase": "1_iterative",
                            "repetition": rep,
                            "epsilon_target": epsilon_target,
                            "algorithm": ALGORITHM_LABELS[algorithm],
                            "algorithm_key": algorithm,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "objective_ry_offset": objective_ry_offset,
                            "a_true": target_a,
                        }
                    )
                    print(
                        f"Rep {rep + 1}/{n_rep} | Eps {epsilon_target:.1e} | {ALGORITHM_LABELS[algorithm]}: "
                        f"FAILED ({exc})"
                    )
                    continue

                # Extract final estimation and queries
                try:
                    final_estimate = result.estimation
                    confidence_interval = result.confidence_interval if hasattr(result, 'confidence_interval') else None
                    
                    # Extract the query trajectory to get total queries
                    queries, estimations, _ = extract_trace(
                        algorithm,
                        result,
                        configured_n_shots,
                        _effective_n_shots,
                    )
                    num_queries = int(queries[-1]) if len(queries) > 0 else None
                except Exception as exc:
                    error_rows.append(
                        {
                            "phase": "1_iterative",
                            "repetition": rep,
                            "epsilon_target": epsilon_target,
                            "algorithm": ALGORITHM_LABELS[algorithm],
                            "algorithm_key": algorithm,
                            "error_type": "ExtractionError",
                            "error": f"Could not extract final estimate: {str(exc)}",
                            "objective_ry_offset": objective_ry_offset,
                            "a_true": target_a,
                        }
                    )
                    print(
                        f"Rep {rep + 1}/{n_rep} | Eps {epsilon_target:.1e} | {ALGORITHM_LABELS[algorithm]}: "
                        f"could not extract final estimate"
                    )
                    continue

                abs_error = abs(final_estimate - target_a)
                rel_error = abs_error / target_a if target_a != 0 else abs_error

                final_estimations.append(
                    {
                        "phase": "1_iterative",
                        "repetition": rep,
                        "epsilon_target": epsilon_target,
                        "algorithm": ALGORITHM_LABELS[algorithm],
                        "algorithm_key": algorithm,
                        "objective_ry_offset": objective_ry_offset,
                        "a_true": target_a,
                        "final_estimate": final_estimate,
                        "abs_error": abs_error,
                        "rel_error": rel_error,
                        "num_queries": num_queries,
                        "confidence_interval": confidence_interval,
                        "elapsed_runtime_seconds": elapsed_runtime_seconds,
                        "n_shots": configured_n_shots,
                    }
                )

                # Track queries for median calculation
                key = (str(epsilon_target), algorithm)
                if key not in queries_per_eps_alg:
                    queries_per_eps_alg[key] = []
                if num_queries is not None:
                    queries_per_eps_alg[key].append(num_queries)

                print(
                    f"Rep {rep + 1}/{n_rep} | Eps {epsilon_target:.1e} | {ALGORITHM_LABELS[algorithm]}: "
                    f"a={target_a:.3f}, final_est={final_estimate:.3f}, "
                    f"abs_err={abs_error:.3e}, queries={num_queries}, "
                    f"runtime={elapsed_runtime_seconds:.3f}s"
                )

    # Calculate median queries for each epsilon and algorithm
    median_queries: dict[str, dict[str, float]] = {}
    print("\n" + "=" * 80)
    print("MEDIAN QUERIES CALCULATION")
    print("=" * 80)
    for eps_idx, epsilon_target in enumerate(EPSILON_TARGETS):
        eps_str = str(epsilon_target)
        if eps_str not in median_queries:
            median_queries[eps_str] = {}
        
        for algorithm in ALGORITHMS_ITERATIVE:
            key = (eps_str, algorithm)
            if key in queries_per_eps_alg and len(queries_per_eps_alg[key]) > 0:
                median_q = np.median(queries_per_eps_alg[key])
                median_queries[eps_str][algorithm] = median_q
                print(
                    f"Eps {epsilon_target:.1e} | {ALGORITHM_LABELS[algorithm]}: "
                    f"median queries = {median_q:.0f}"
                )
            else:
                print(
                    f"Eps {epsilon_target:.1e} | {ALGORITHM_LABELS[algorithm]}: "
                    f"NO DATA"
                )
    
    return final_estimations, error_rows, median_queries


def phase_2_bae_fixed_queries(median_queries: dict[str, dict[str, float]], n_rep: int = 50) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Phase 2: Run BAE with fixed max_queries (equal to median queries from iterative algorithms).
    
    Args:
        median_queries: dict[epsilon_str][algorithm] = median_queries
        n_rep: number of repetitions
    
    Returns:
        - final_estimations: list of result dicts
        - error_rows: list of error dicts
    """
    alpha = 0.05
    max_queries_default = 1e6
    
    final_estimations: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    
    rng = np.random.default_rng(5678)  # Different seed for phase 2
    algorithm = "bae"
    

    print("\n" + "=" * 80)
    print("PHASE 2: Running BAE with fixed max_queries (median from iterative algorithms)")
    print("=" * 80)
    print(f"Repetitions: {n_rep}\n")

    for rep in range(n_rep):
        objective_ry_offset = float(rng.choice(OBJECTIVE_RY_OFFSETS))
        problem = build_problem(objective_ry_offset)
        target_a = true_amplitude_for_offset(objective_ry_offset)

        for eps_idx, epsilon_target in enumerate(EPSILON_TARGETS):
            eps_str = str(epsilon_target)
            
            # Use median queries from iterative algorithms for this epsilon
            # Average over the three iterative algorithms
            if eps_str in median_queries:
                median_qs = [median_queries[eps_str].get(alg, None) for alg in ALGORITHMS_ITERATIVE]
                valid_qs = [q for q in median_qs if q is not None]
                if len(valid_qs) > 0:
                    max_queries_bae = np.median(valid_qs)
                else:
                    max_queries_bae = max_queries_default
            else:
                max_queries_bae = max_queries_default
            
            alg_cfg = ALGORITHM_CONFIG[algorithm]
            configured_n_shots = (
                None if alg_cfg["n_shots"] is None else int(alg_cfg["n_shots"])
            )
            seed = int(2_000_000 + rep * 1000 + eps_idx)
            solver, bayes = _build_solver(
                algorithm=algorithm,
                epsilon_target=1e-6,  # BAE uses this internally, but max_queries controls the stopping
                alpha=alpha,
                n_shots=configured_n_shots,
                seed=seed,
                estimate_T=bool(alg_cfg.get("estimate_T", False)),
            )

            try:
                start_time = time.perf_counter()
                np.random.seed(seed)
                estimate_kwargs: dict[str, Any] = {
                    "max_queries": int(max_queries_bae),
                }
                if configured_n_shots is not None:
                    estimate_kwargs["n_shots"] = configured_n_shots
                result = solver.estimate(problem, **estimate_kwargs)
                elapsed_runtime_seconds = time.perf_counter() - start_time
            except Exception as exc:
                error_rows.append(
                    {
                        "phase": "2_bae",
                        "repetition": rep,
                        "epsilon_target": epsilon_target,
                        "max_queries_bae": max_queries_bae,
                        "algorithm": ALGORITHM_LABELS[algorithm],
                        "algorithm_key": algorithm,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "objective_ry_offset": objective_ry_offset,
                        "a_true": target_a,
                    }
                )
                print(
                    f"Rep {rep + 1}/{n_rep} | Eps {epsilon_target:.1e} | MaxQ {max_queries_bae:.0f} | {ALGORITHM_LABELS[algorithm]}: "
                    f"FAILED ({exc})"
                )
                continue

            # Extract final estimation and queries
            try:
                final_estimate = result.estimation
                confidence_interval = result.confidence_interval if hasattr(result, 'confidence_interval') else None
                
                # Extract the query trajectory to get total queries
                queries, estimations, _ = extract_trace(
                    algorithm,
                    result,
                    configured_n_shots,
                    _effective_n_shots,
                )
                num_queries = int(queries[-1]) if len(queries) > 0 else None
            except Exception as exc:
                error_rows.append(
                    {
                        "phase": "2_bae",
                        "repetition": rep,
                        "epsilon_target": epsilon_target,
                        "max_queries_bae": max_queries_bae,
                        "algorithm": ALGORITHM_LABELS[algorithm],
                        "algorithm_key": algorithm,
                        "error_type": "ExtractionError",
                        "error": f"Could not extract final estimate: {str(exc)}",
                        "objective_ry_offset": objective_ry_offset,
                        "a_true": target_a,
                    }
                )
                print(
                    f"Rep {rep + 1}/{n_rep} | Eps {epsilon_target:.1e} | MaxQ {max_queries_bae:.0f} | {ALGORITHM_LABELS[algorithm]}: "
                    f"could not extract final estimate"
                )
                continue

            abs_error = abs(final_estimate - target_a)
            rel_error = abs_error / target_a if target_a != 0 else abs_error

            final_estimations.append(
                {
                    "phase": "2_bae",
                    "repetition": rep,
                    "epsilon_target": epsilon_target,
                    "max_queries_bae": max_queries_bae,
                    "algorithm": ALGORITHM_LABELS[algorithm],
                    "algorithm_key": algorithm,
                    "objective_ry_offset": objective_ry_offset,
                    "a_true": target_a,
                    "final_estimate": final_estimate,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "num_queries": num_queries,
                    "confidence_interval": confidence_interval,
                    "elapsed_runtime_seconds": elapsed_runtime_seconds,
                    "n_shots": configured_n_shots,
                }
            )

            print(
                f"Rep {rep + 1}/{n_rep} | Eps {epsilon_target:.1e} | MaxQ {max_queries_bae:.0f} | {ALGORITHM_LABELS[algorithm]}: "
                f"a={target_a:.3f}, final_est={final_estimate:.3f}, "
                f"abs_err={abs_error:.3e}, actual_queries={num_queries}, "
                f"runtime={elapsed_runtime_seconds:.3f}s"
            )

    return final_estimations, error_rows


def run_experiment() -> None:
    """
    Two-phase experiment:
    Phase 1: Run iterative algorithms with epsilon sweep, calculate median queries
    Phase 2: Run BAE with fixed max_queries (median from Phase 1)
    """
    n_rep = 100

    # Phase 1
    final_estimations_p1, error_rows_p1, median_queries = phase_1_iterative_algorithms(n_rep=n_rep)
    
    # Phase 2 (only if BAE in algorithms)
    if "bae" in ALGORITHMS_ITERATIVE:
        final_estimations_p2, error_rows_p2 = phase_2_bae_fixed_queries(median_queries, n_rep=n_rep)
    else:
        final_estimations_p2, error_rows_p2 = [], []

    # Combine results
    final_estimations = final_estimations_p1 + final_estimations_p2
    error_rows = error_rows_p1 + error_rows_p2

    output_dir = os.path.join(current_dir, "experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = "bae_biqae_iqae_cabiqae_latentt_ideal_v2"
    
    save_csv(final_estimations, os.path.join(output_dir, f"{output_prefix}_final_estimations.csv"))
    save_csv(error_rows, os.path.join(output_dir, f"{output_prefix}_errors.csv"))

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"Saved {len(final_estimations)} final estimations to {os.path.join(output_dir, f'{output_prefix}_final_estimations.csv')}")
    if error_rows:
        print(f"Saved {len(error_rows)} errors to {os.path.join(output_dir, f'{output_prefix}_errors.csv')}")


if __name__ == "__main__":
    run_experiment()
