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
from quantum_cva.algorithms.proposed_algorithms.elf_qae import ELFQAE
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
    aggregate_budget_summary,
    extract_trace,
    plot_final_runtime_scatter_from_budget_rows,
    plot_budget_summary,
    save_csv,
    trace_rows_from_result,
)


ALGORITHMS = ("bae", "biqae", "iqae", "cabiqae_latentt", "elf_qae")
ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "iqae": "IQAE",
    "cabiqae_latentt": "CABIQAE",
    "elf_qae": "ELF-QAE",
}
ALGORITHM_STYLES = {
    "bae": {"color": "#E07A5F", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "iqae": {"color": "#4C78A8", "marker": "D"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
    "elf_qae": {"color": "#6D597A", "marker": "v"},
}

OBJECTIVE_RY_OFFSETS = np.array([0.30, 0.44, 0.48, 0.52, 0.56, 0.60, 0.63, 0.67], dtype=float)

ALGORITHM_CONFIG = {
    "bae": {
        "epsilon_target": 5e-3,
        "n_shots": None,
        "max_queries": 7e3,
        "estimate_T": False,
    },
    "biqae": {
        "epsilon_target": 2e-3,
        "n_shots": None,
        "max_queries": None,
    },
    "iqae": {
        "epsilon_target": 2e-3,
        "n_shots": None,
        "max_queries": None,
    },
    "cabiqae_latentt": {
        "epsilon_target": 2e-3,
        "n_shots": None,
        "max_queries": None,
    },
    "elf_qae": {
        "epsilon_target": 2e-3,
        "n_shots": 1,
        "max_queries": 7e3,
        "layers": 1,
        "layer_selection": "fixed",
        "optimizer_restarts": 0,
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
    max_queries: int | None = None,
    estimate_T: bool = False,
    layers: int | None = None,
    layer_selection: str = "fixed",
    optimizer_restarts: int = 0,
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

    if algorithm == "elf_qae":
        elf_layers = int(layers if layers is not None else 8)
        return (
            ELFQAE(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=ideal_sampler,
                layers=elf_layers,
                max_layers=elf_layers,
                layer_selection=layer_selection,
                circuit_fidelity=1.0,
                max_state_prep_calls=max_queries,
                max_rounds=(
                    max(1, int(float(max_queries) // (2 * elf_layers + 1)))
                    if max_queries is not None
                    else 10_000
                ),
                optimizer_restarts=optimizer_restarts,
                random_seed=seed,
            ),
            False,
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
    Compare BAE, BIQAE, IQAE, CABIQAE_latentt and ELF-QAE in an ideal regime
    with the canonical 3-qubit AE topology used by the hardware experiment.
    """
    n_rep = 100    

    alpha = 0.05
    max_queries = 1e5

    trace_rows_all: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    final_runtimes: dict[str, list[float]] = {algorithm: [] for algorithm in ALGORITHMS}
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
                max_queries=(
                    None
                    if alg_cfg["max_queries"] is None
                    else int(alg_cfg["max_queries"])
                ),
                estimate_T=bool(alg_cfg.get("estimate_T", False)),
                layers=(
                    None
                    if alg_cfg.get("layers") is None
                    else int(alg_cfg["layers"])
                ),
                layer_selection=str(alg_cfg.get("layer_selection", "fixed")),
                optimizer_restarts=int(alg_cfg.get("optimizer_restarts", 0)),
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
                elif algorithm == "elf_qae":
                    estimate_kwargs = {"show_details": False}
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
            final_runtimes[algorithm].append(float(elapsed_runtime_seconds))

            print(
                f"Rep {rep + 1}/{n_rep} | {ALGORITHM_LABELS[algorithm]}: "
                f"offset={objective_ry_offset:+.3f}, a={target_a:.3f}, "
                f"final nAE={final_row['final_normalized_abs_error']:.3e}, "
                f"queries={int(queries[-1])}, "
                f"runtime={elapsed_runtime_seconds:.3f}s"
            )

    print("Runtime summary by algorithm:")
    for algorithm in ALGORITHMS:
        runtimes = np.asarray(final_runtimes[algorithm], dtype=float)
        runtimes = runtimes[np.isfinite(runtimes)]
        if runtimes.size == 0:
            print(f"  {ALGORITHM_LABELS[algorithm]}: no successful runtime samples")
            continue
        print(
            f"  {ALGORITHM_LABELS[algorithm]}: "
            f"median={np.median(runtimes):.3f}s, "
            f"mean={np.mean(runtimes):.3f}s, "
            f"n={runtimes.size}"
        )

    output_dir = os.path.join(current_dir, "elf_experiment_results")
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal"
    budget_summary = aggregate_budget_summary(
        trace_rows_all,
        total_repetitions=n_rep,
    )
    save_csv(trace_rows_all, os.path.join(output_dir, f"{output_prefix}_trace_rows.csv"))
    save_csv(final_rows, os.path.join(output_dir, f"{output_prefix}_final_rows.csv"))
    save_csv(trace_rows_all, os.path.join(output_dir, f"{output_prefix}_budget_rows.csv"))
    save_csv(budget_summary, os.path.join(output_dir, f"{output_prefix}_budget_summary.csv"))
    save_csv(error_rows, os.path.join(output_dir, f"{output_prefix}_errors.csv"))

    out_path = os.path.join(
        output_dir,
        "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal_rmse.png",
    )
    plot_budget_summary(
        budget_summary,
        algorithms=ALGORITHMS,
        algorithm_labels=ALGORITHM_LABELS,
        algorithm_styles=ALGORITHM_STYLES,
        output_path=out_path,
        title="Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE_latentt vs ELF-QAE",
    )
    runtime_scatter_path = os.path.join(
        output_dir,
        "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal_final_error_runtime_scatter.png",
    )
    plot_final_runtime_scatter_from_budget_rows(
        trace_rows_all,
        algorithms=ALGORITHMS,
        algorithm_labels=ALGORITHM_LABELS,
        algorithm_styles=ALGORITHM_STYLES,
        output_path=runtime_scatter_path,
        title="Final error versus runtime: ideal ELF-QAE comparison",
        summary_path=os.path.join(
            output_dir,
            "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal_final_error_runtime_scatter_summary.csv",
        ),
    )
    print(f"Saved plot to {out_path}")
    print(f"Saved runtime scatter plot to {runtime_scatter_path}")
    print(f"Saved CSV outputs with prefix {os.path.join(output_dir, output_prefix)}")


if __name__ == "__main__":
    run_experiment()
