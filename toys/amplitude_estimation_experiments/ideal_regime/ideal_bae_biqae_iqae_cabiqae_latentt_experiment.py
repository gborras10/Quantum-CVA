import os
import sys
import time
from typing import Any

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
from toys.amplitude_estimation_experiments.ideal_regime.ideal_benchmark_outputs import (
    aggregate_budget_summary,
    extract_trace,
    plot_budget_summary,
    save_csv,
    trace_rows_from_result,
)


ALGORITHMS = ("bae", "biqae", "iqae", "cabiqae_latentt")
ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "iqae": "IQAE",
    "cabiqae_latentt": "CABIQAE",
}
ALGORITHM_STYLES = {
    "bae": {"color": "#E07A5F", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "iqae": {"color": "#4C78A8", "marker": "D"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
}

OBJECTIVE_RY_OFFSETS = np.array([0.30, 0.44, 0.48, 0.52, 0.56, 0.60, 0.63, 0.67], dtype=float)

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
    queries, estimations, _ = extract_trace(
        algorithm,
        result,
        n_shots,
        _effective_n_shots,
    )
    return queries, estimations


def run_experiment() -> None:
    """
    Compare BAE, BIQAE, IQAE and CABIQAE_latentt in an ideal regime with
    the canonical 3-qubit AE topology used by the hardware experiment.
    """
    n_rep = 25    

    alpha = 0.05
    max_queries = 1e6

    trace_rows_all: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1234)
    print(
        "Running ideal amplitude-estimation benchmark "
        f"({', '.join(ALGORITHM_LABELS[a] for a in ALGORITHMS)}) "
        f"with {n_rep} repetitions on the canonical 3-qubit topology..."
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
        objective_ry_offset = float(rng.choice(OBJECTIVE_RY_OFFSETS))
        problem = build_problem(objective_ry_offset)
        target_a = true_amplitude_for_offset(objective_ry_offset)

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
                start_time = time.perf_counter()
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
                elapsed_runtime_seconds = time.perf_counter() - start_time
            except Exception as exc:
                error_rows.append(
                    {
                        "run_kind": "ideal_simulation",
                        "repetition": rep,
                        "algorithm": ALGORITHM_LABELS[algorithm],
                        "algorithm_key": algorithm,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "objective_ry_offset": objective_ry_offset,
                        "a_true": target_a,
                    }
                )
                print(
                    f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                    f"failed ({exc})"
                )
                continue

            queries, estimations = _extract_trace(algorithm, result, configured_n_shots)
            if len(queries) == 0:
                error_rows.append(
                    {
                        "run_kind": "ideal_simulation",
                        "repetition": rep,
                        "algorithm": ALGORITHM_LABELS[algorithm],
                        "algorithm_key": algorithm,
                        "error_type": "EmptyTrace",
                        "error": "no trajectory returned",
                        "objective_ry_offset": objective_ry_offset,
                        "a_true": target_a,
                    }
                )
                print(
                    f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                    "no trajectory returned"
                )
                continue

            trace_rows, final_row = trace_rows_from_result(
                result,
                algorithm=algorithm,
                algorithm_labels=ALGORITHM_LABELS,
                repetition=rep,
                a_true=target_a,
                objective_ry_offset=objective_ry_offset,
                n_shots=configured_n_shots,
                elapsed_wall_seconds=elapsed_runtime_seconds,
                effective_n_shots=_effective_n_shots,
            )
            trace_rows_all.extend(trace_rows)
            final_rows.append(final_row)

            print(
                f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                f"offset={objective_ry_offset:+.3f}, a={target_a:.3f}, "
                f"final nAE={final_row['final_normalized_abs_error']:.3e}, "
                f"queries={int(queries[-1])}, "
                f"runtime={elapsed_runtime_seconds:.3f}s"
            )

    output_prefix = "bae_biqae_iqae_cabiqae_latentt_ideal"
    budget_summary = aggregate_budget_summary(
        trace_rows_all,
        total_repetitions=n_rep,
    )
    save_csv(trace_rows_all, os.path.join(current_dir, f"{output_prefix}_trace_rows.csv"))
    save_csv(final_rows, os.path.join(current_dir, f"{output_prefix}_final_rows.csv"))
    save_csv(trace_rows_all, os.path.join(current_dir, f"{output_prefix}_budget_rows.csv"))
    save_csv(budget_summary, os.path.join(current_dir, f"{output_prefix}_budget_summary.csv"))
    save_csv(error_rows, os.path.join(current_dir, f"{output_prefix}_errors.csv"))

    out_path = os.path.join(current_dir, "bae_biqae_iqae_cabiqae_latentt_ideal_rmse.png")
    plot_budget_summary(
        budget_summary,
        algorithms=ALGORITHMS,
        algorithm_labels=ALGORITHM_LABELS,
        algorithm_styles=ALGORITHM_STYLES,
        output_path=out_path,
        title="Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE_latentt",
    )
    print(f"Saved plot to {out_path}")
    print(f"Saved CSV outputs with prefix {os.path.join(current_dir, output_prefix)}")


if __name__ == "__main__":
    run_experiment()
