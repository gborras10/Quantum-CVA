from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from pipeline_common import (
    DEFAULT_RUN_DIR,
    REPO_ROOT,
    add_cva_alias_columns,
    add_cva_aliases,
    load_config,
    parse_int_list,
    parse_name_list,
    preferred_field_order,
)
from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.io import (
    RunPaths,
    save_csv,
    save_json,
    write_trace_bundle,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (
    ContrastDecaySampler,
    FastIdealAmplificationSampler,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    normalize_algorithm_key,
    run_algorithm_once,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (
    aggregate_budget_summary,
)
from quantum_cva.amplitude_estimation.experiments.traces import rows_at_budgets


DEFAULT_ALGORITHMS = "cabiqae_latentt,biqae,iqae,bae"
DEFAULT_BUDGETS = "64,128,256,512,1024,2048,4096,8192,16384,32768,65536"
USE_SAMPLER = False
FAST_IDEAL_CIRCUITS: bool | None = None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, yes/no, 1/0."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run noiseless amplitude-estimation experiments on the real 6q CVA "
            "EstimationProblem."
        )
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline config object as 'module:CONFIG'. Defaults to full_cva_pipeline:CONFIG.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Python file containing a PipelineConfig object.",
    )
    parser.add_argument(
        "--config-attr",
        default="CONFIG",
        help="Attribute used with --config-path.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--epsilon-target", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shots", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=300_000)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--sampler",
        type=_parse_bool,
        default=USE_SAMPLER,
        metavar="{true,false}",
        help=(
            "true uses the statevector circuit sampler for each A Q^k circuit; "
            "false uses the fast closed-form ideal amplification law. "
            f"Default: {USE_SAMPLER}."
        ),
    )
    parser.add_argument(
        "--use-sampler",
        dest="sampler",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-sampler",
        dest="sampler",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fast-circuits",
        type=_parse_bool,
        default=FAST_IDEAL_CIRCUITS,
        metavar="{true,false}",
        help=(
            "true skips construction of A Q^k and returns metadata-only "
            "circuits. Auto default: true when --sampler false, false when "
            "--sampler true."
        ),
    )
    parser.add_argument(
        "--use-fast-circuits",
        dest="fast_circuits",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-fast-circuits",
        dest="fast_circuits",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cap-kappa", type=float, default=1000.0)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for budget-summary CIs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue running remaining algorithms after a solver failure.",
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate plots after writing CSV outputs.",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Forward verbose details to the underlying AE solvers.",
    )
    return parser


def _validate_bundle(bundle: Any) -> None:
    if str(bundle.good_bitstring) != "111":
        raise ValueError(
            "The 6q CVA AE pipeline expects good_bitstring='111', got "
            f"{bundle.good_bitstring!r}."
        )
    objective_qubits = list(bundle.problem.objective_qubits)
    if len(objective_qubits) != 3:
        raise ValueError(
            "The 6q CVA AE pipeline expects exactly three objective qubits, got "
            f"{objective_qubits!r}."
        )


def _save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    save_csv(rows, path, fieldnames=preferred_field_order(rows))


def _sampler_mode(use_sampler: bool) -> str:
    return "statevector_circuit_sampler" if use_sampler else "fast_ideal_formula"


def _resolve_fast_circuits(args: argparse.Namespace) -> bool:
    if args.fast_circuits is None:
        return not bool(args.sampler)
    fast_circuits = bool(args.fast_circuits)
    if fast_circuits and bool(args.sampler):
        raise ValueError("--fast-circuits true requires --sampler false.")
    return fast_circuits


def _build_noiseless_sampler(bundle: Any, *, use_sampler: bool, seed: int) -> Any:
    if use_sampler:
        return ContrastDecaySampler(bundle, T=None, seed=seed)
    return FastIdealAmplificationSampler(bundle, T=None, seed=seed)


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _format_float(value: Any, digits: int = 6) -> str:
    number = _as_float(value)
    if not math.isfinite(number):
        return "nan"
    if digits <= 0:
        return f"{number:.0f}"
    return f"{number:.{digits}g}"


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def _min_metric(
    rows: list[dict[str, Any]],
    final: dict[str, Any],
    row_key: str,
    final_key: str | None = None,
) -> float:
    values = [_as_float(row.get(row_key)) for row in rows]
    if final_key is not None:
        values.append(_as_float(final.get(final_key)))
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _min_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else math.nan


def _max_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _query_metric(rows: list[dict[str, Any]], final: dict[str, Any]) -> float:
    final_queries = _as_float(final.get("final_queries"))
    if math.isfinite(final_queries):
        return final_queries
    values = [_as_float(row.get("query_budget_actual", row.get("query_budget"))) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _annotate_terminal_metrics(
    rows: list[dict[str, Any]],
    final: dict[str, Any],
    *,
    max_queries: int,
) -> dict[str, Any]:
    final_queries = _query_metric(rows, final)
    k_values = [_as_float(row.get("grover_power")) for row in rows]
    k_values.append(_as_float(final.get("k_max")))
    finite_k = [value for value in k_values if math.isfinite(value)]
    k_max = int(max(finite_k)) if finite_k else 0

    final["k_max"] = int(k_max)
    final["amplification_factor_max"] = int(2 * k_max + 1)
    final["final_queries"] = float(final_queries)
    final["max_queries_requested"] = int(max_queries)
    final["max_queries_reached"] = bool(
        math.isfinite(final_queries) and final_queries >= float(max_queries)
    )
    final["min_abs_error"] = _min_metric(rows, final, "abs_error", "final_abs_error")
    final["min_normalized_abs_error"] = _min_metric(
        rows,
        final,
        "normalized_abs_error",
        "final_normalized_abs_error",
    )
    final["min_processed_abs_error"] = _min_metric(
        rows,
        final,
        "processed_abs_error",
        "processed_abs_error",
    )
    final["min_processed_relative_error"] = _min_metric(
        rows,
        final,
        "processed_relative_error",
        "processed_relative_error",
    )
    return final


def _print_run_metrics(
    *,
    repetition: int,
    algorithm: str,
    final: dict[str, Any],
) -> None:
    print(
        f"rep={repetition}, "
        f"algorithm={algorithm}, "
        f"K_max={final.get('k_max')}, "
        f"queries={_format_float(final.get('final_queries'), 0)}/"
        f"{final.get('max_queries_requested')}, "
        f"min_amp_abs_error={_format_float(final.get('min_abs_error'))}, "
        f"min_amp_rel_error={_format_float(final.get('min_normalized_abs_error'))}, "
        f"min_cva_rel_error={_format_float(final.get('min_processed_relative_error'))}, "
        f"runtime_ae={_format_float(final.get('runtime_wall_seconds'))}s"
    )


def _print_algorithm_summary(final_rows: list[dict[str, Any]]) -> None:
    if not final_rows:
        return
    print("")
    print("Resumen por algoritmo:")
    algorithms = sorted({str(row.get("algorithm_key", row.get("algorithm", ""))) for row in final_rows})
    for algorithm in algorithms:
        group = [
            row
            for row in final_rows
            if str(row.get("algorithm_key", row.get("algorithm", ""))) == algorithm
        ]
        if not group:
            continue
        runtimes = [_as_float(row.get("runtime_wall_seconds")) for row in group]
        queries = [_as_float(row.get("final_queries")) for row in group]
        k_values = [_as_float(row.get("k_max")) for row in group]
        min_amp_errors = [_as_float(row.get("min_abs_error")) for row in group]
        min_cva_errors = [
            _as_float(row.get("min_processed_relative_error")) for row in group
        ]
        reached_count = sum(1 for row in group if bool(row.get("max_queries_reached")))
        label = ALGORITHM_LABELS.get(algorithm, algorithm)
        print(
            "  "
            f"{label}: runs={len(group)}, "
            f"K_max_max={_format_float(_max_finite(k_values), 0)}, "
            f"queries_max={_format_float(_max_finite(queries), 0)}, "
            f"max_queries_reached={reached_count}/{len(group)}, "
            f"min_amp_abs_error={_format_float(_min_finite(min_amp_errors))}, "
            f"min_cva_rel_error={_format_float(_min_finite(min_cva_errors))}, "
            f"runtime_ae_median={_format_float(_median(runtimes))}s, "
            f"runtime_ae_total={_format_float(sum(value for value in runtimes if math.isfinite(value)))}s"
        )


def run_pipeline(args: argparse.Namespace) -> RunPaths:
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    bundle = build_6q_cva_problem_bundle(config, repo_root=args.repo_root)
    _validate_bundle(bundle)

    paths = RunPaths(Path(args.run_dir))
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    algorithms = tuple(
        normalize_algorithm_key(name) for name in parse_name_list(args.algorithms)
    )
    budgets = parse_int_list(args.budgets)
    sampler_mode = _sampler_mode(bool(args.sampler))
    fast_circuits = _resolve_fast_circuits(args)
    construct_circuit_mode = "metadata_only" if fast_circuits else "full"
    if fast_circuits and "elf_qae" in algorithms:
        raise ValueError(
            "--fast-circuits true is implemented for BAE/IQAE/BIQAE/CABIQAE. "
            "Use --fast-circuits false for ELF-QAE."
        )

    metadata = {
        "mode": "ideal_noiseless",
        "pipeline": "6q_cva_ae_noseless_simulation",
        "instance_name": "6q_instance",
        "target_name": bundle.target_name,
        "good_bitstring": bundle.good_bitstring,
        "objective_qubits": list(bundle.problem.objective_qubits),
        "a_true": float(bundle.true_amplitude),
        "processed_true_value": float(bundle.processed_true_value),
        "cva_true": float(bundle.processed_true_value),
        "algorithms": list(algorithms),
        "algorithm_labels": {
            key: ALGORITHM_LABELS.get(key, key) for key in algorithms
        },
        "repetitions": int(args.repetitions),
        "epsilon_target": float(args.epsilon_target),
        "alpha": float(args.alpha),
        "n_shots": int(args.n_shots),
        "max_queries": int(args.max_queries),
        "budgets": budgets,
        "sampler": bool(args.sampler),
        "sampler_mode": sampler_mode,
        "fast_circuits": bool(fast_circuits),
        "construct_circuit_mode": construct_circuit_mode,
        "cap_kappa": float(args.cap_kappa),
        "seed": int(args.seed),
        "created_at_epoch": time.time(),
        "repo_root": str(Path(args.repo_root).resolve()),
        "problem_metadata": bundle.metadata,
    }
    save_json(metadata, paths.config)

    trace_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    construct_circuit_cache: dict[tuple[Any, ...], Any] = {}

    trace_extra = {
        "simulation_regime": "ideal_noiseless",
        "instance_name": "6q_instance",
        "sampler": bool(args.sampler),
        "sampler_mode": sampler_mode,
        "fast_circuits": bool(fast_circuits),
        "construct_circuit_mode": construct_circuit_mode,
    }
    print(
        "Loaded 6q CVA AE problem: "
        f"a_true={bundle.true_amplitude:.12g}, "
        f"CVA={bundle.processed_true_value:.12g}, "
        f"good={bundle.good_bitstring}, "
        f"sampler={bool(args.sampler)} ({sampler_mode}), "
        f"fast_circuits={fast_circuits} ({construct_circuit_mode})."
    )

    for rep in range(int(args.repetitions)):
        for alg_index, algorithm in enumerate(algorithms):
            run_seed = int(args.seed) + 1009 * rep + 37 * alg_index
            print(
                f"[rep {rep + 1}/{args.repetitions}] {algorithm} "
                f"(seed={run_seed})"
            )
            sampler = _build_noiseless_sampler(
                bundle,
                use_sampler=bool(args.sampler),
                seed=run_seed,
            )
            try:
                rows, final = run_algorithm_once(
                    algorithm,
                    sampler,
                    bundle,
                    run_kind="ideal_noiseless",
                    repetition=rep,
                    epsilon_target=float(args.epsilon_target),
                    alpha=float(args.alpha),
                    n_shots=int(args.n_shots),
                    max_queries=int(args.max_queries),
                    seed=run_seed,
                    algorithm_labels=ALGORITHM_LABELS,
                    cap_kappa=float(args.cap_kappa),
                    construct_circuit_cache=construct_circuit_cache,
                    construct_circuit_mode=construct_circuit_mode,
                    trace_extra=trace_extra,
                    show_details=bool(args.show_details),
                )
            except Exception as exc:
                error_rows.append(
                    {
                        "run_kind": "ideal_noiseless",
                        "simulation_regime": "ideal_noiseless",
                        "instance_name": "6q_instance",
                        "repetition": rep,
                        "algorithm": ALGORITHM_LABELS.get(algorithm, algorithm),
                        "algorithm_key": algorithm,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"  failed: {type(exc).__name__}: {exc}")
                if not bool(args.continue_on_error):
                    raise
                continue

            rows = add_cva_aliases(rows)
            final = add_cva_alias_columns(final)
            final = _annotate_terminal_metrics(
                rows,
                final,
                max_queries=int(args.max_queries),
            )
            trace_rows.extend(rows)
            final_rows.append(final)
            selected_budget_rows = rows_at_budgets(
                rows,
                budgets,
                run_kind="ideal_noiseless",
            )
            for budget_row in selected_budget_rows:
                budget_row.update(trace_extra)
            budget_rows.extend(
                add_cva_aliases(selected_budget_rows)
            )
            print(
                "  final: "
                f"Nq={final['final_queries']:.0f}, "
                f"a_hat={final['final_estimate']:.12g}, "
                f"CVA_hat={final['cva_estimate']:.12g}, "
                f"rel_CVA_err={final['cva_relative_error']:.4g}"
            )
            _print_run_metrics(
                repetition=rep,
                algorithm=algorithm,
                final=final,
            )

    _print_algorithm_summary(final_rows)

    summary_rows = add_cva_aliases(
        aggregate_budget_summary(
            budget_rows,
            total_repetitions=int(args.repetitions),
            group_by_budget=True,
            bootstrap_samples=int(args.bootstrap_samples),
        )
    )

    _save_csv(trace_rows, paths.direct_trace)
    _save_csv(final_rows, paths.direct_final)
    _save_csv(budget_rows, paths.replay_budget)
    _save_csv(summary_rows, paths.budget_summary)
    _save_csv(error_rows, paths.errors)
    write_trace_bundle(
        paths.trace_bundle,
        trace_rows=trace_rows,
        budget_rows=budget_rows,
    )
    paths.write_manifest()

    if bool(args.make_plots):
        try:
            from plot_noiseless_cva_ae import make_plots

            make_plots(paths.run_dir, algorithms=algorithms)
        except Exception as exc:
            error_rows.append(
                {
                    "run_kind": "ideal_noiseless",
                    "simulation_regime": "ideal_noiseless",
                    "instance_name": "6q_instance",
                    "error_type": type(exc).__name__,
                    "error": f"Plot generation failed: {exc}",
                }
            )
            _save_csv(error_rows, paths.errors)
            print(f"Plot generation failed: {type(exc).__name__}: {exc}")
            if not bool(args.continue_on_error):
                raise

    return paths


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = run_pipeline(args)
    print(f"Wrote noiseless CVA AE outputs to {paths.run_dir}")


if __name__ == "__main__":
    main()
