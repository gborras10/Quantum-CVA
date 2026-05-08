from __future__ import annotations

import argparse
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_aer import AerSimulator

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
from quantum_cva.amplitude_estimation.experiments.hardware import (
    ExperimentState,
    analyze_amplification,
    backend_snapshot,
    build_pass_manager_for_backend,
    effective_t_for_algorithms,
    run_amplification_scan,
    run_preflight,
    run_readout_calibration,
)
from quantum_cva.amplitude_estimation.experiments.io import (
    RunPaths,
    load_json,
    save_csv,
    save_json,
    write_trace_bundle,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (
    AerCountSampler,
    LoggedAerSampler,
    build_noise_model,
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


DEFAULT_ALGORITHMS = "cabiqae_latentt,biqae,bae"
DEFAULT_BUDGETS = "128,256,512,1024,2048,4096,8192,16384"
DEFAULT_REFERENCE_KS = "0,1,2,3,4"
RUN_KIND = "simulated_noise"


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


def _parse_nonnegative_int_list(raw: str | list[int | str]) -> list[int]:
    if isinstance(raw, str):
        tokens = raw.replace(";", ",").replace(" ", ",").split(",")
    else:
        tokens = list(raw)
    values = [int(token) for token in tokens if str(token).strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    if any(value < 0 for value in values):
        raise ValueError("Grover powers must be non-negative integers.")
    return values


def _parse_noise_floor(value: Any) -> str | float:
    text = str(value).strip().lower()
    if text in {"fit", "fitted", "estimate", "estimated"}:
        return "fit"
    if text in {"uniform", "uniform_objective", "objective_uniform"}:
        return "uniform_objective"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Expected a probability in [0, 1] or 'uniform_objective'."
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(
            "Expected a finite probability in [0, 1]."
        )
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulated-noise amplitude-estimation experiments on the real "
            "6q CVA EstimationProblem."
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
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--epsilon-target", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shots", type=int, default=128)
    parser.add_argument("--max-queries", type=int, default=16_384)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise-profile", default="realistic")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--noise-floor",
        type=_parse_noise_floor,
        default=0.5,
        help=(
            "Asymptotic good-state probability under full contrast loss. "
            "Use 'uniform_objective' for 1 / 2**objective_width, or 'fit' "
            "to estimate it from the amplification scan. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--aer-method",
        default="density_matrix",
        help="AerSimulator method used for noisy circuit execution.",
    )
    parser.add_argument(
        "--transpile-backend-name",
        default=None,
        help=(
            "Optional IBM backend name used only for transpilation layout and "
            "basis. If omitted, a local AerSimulator pass manager is used."
        ),
    )
    parser.add_argument(
        "--use-fractional-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Request IBM backend targets with use_fractional_gates=True for "
            "hardware-aware transpilation. Only applies when "
            "--transpile-backend-name is an IBM backend. Default: true."
        ),
    )
    parser.add_argument("--transpiler-optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=1234)
    parser.add_argument("--routing-method", default="sabre")
    parser.add_argument(
        "--layout-search-strategy",
        choices=("fast", "exhaustive", "preset"),
        default="fast",
        help=(
            "Transpilation layout strategy. 'fast' uses one Qiskit SABRE preset pass "
            "on a representative reference circuit and freezes that layout. "
            "'exhaustive' keeps the older multi-seed layout search. "
            "'preset' uses Qiskit's preset pass manager without freezing a layout."
        ),
    )
    parser.add_argument("--reference-ks", default=DEFAULT_REFERENCE_KS)
    parser.add_argument("--max-grover-power", type=int, default=4)
    parser.add_argument(
        "--execution-max-grover-power",
        type=int,
        default=None,
        help=(
            "Optional hard k cap during AE execution. If omitted, "
            "--max-grover-power only controls preflight/calibration and does "
            "not refuse larger k values selected by adaptive algorithms."
        ),
    )
    parser.add_argument("--max-isa-depth", type=int, default=10**9)
    parser.add_argument("--max-isa-2q", type=int, default=10**9)
    parser.add_argument(
        "--calibrate",
        type=_parse_bool,
        default=True,
        metavar="{true,false}",
        help=(
            "Run readout and amplification scans to estimate T_eff for "
            "noise-aware solvers. Default: true."
        ),
    )
    parser.add_argument(
        "--t-eff",
        type=float,
        default=None,
        help="Override T_eff. If set, it is used instead of fitted calibration T_eff.",
    )
    parser.add_argument("--readout-shots", type=int, default=512)
    parser.add_argument("--scan-repeats", type=int, default=1)
    parser.add_argument("--scan-shots", type=int, default=512)
    parser.add_argument(
        "--min-ideal-offset",
        type=float,
        default=0.15,
        help="Minimum |p_ideal - noise_floor| used in contrast fitting.",
    )
    parser.add_argument(
        "--min-fit-contrast-z",
        type=float,
        default=2.0,
        help="Minimum contrast/se ratio for a calibration point to enter the T_eff fit.",
    )
    parser.add_argument(
        "--min-visible-contrast-z",
        type=float,
        default=3.0,
        help="Minimum contrast/se ratio for robust k_visible selection.",
    )
    parser.add_argument(
        "--min-baseline-fit-points",
        type=int,
        default=4,
        help="Minimum valid contrast points required when --noise-floor fit is used.",
    )
    parser.add_argument("--cap-kappa", type=float, default=1000.0)
    parser.add_argument(
        "--cabiqae-hard-k-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable CABIQAE's hard noise cap k <= floor((cap_kappa*T_eff - 1)/2). "
            "By default this runner disables only that hard cap, while keeping "
            "CABIQAE's noise-aware Fisher scheduler active."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for budget-summary CIs.",
    )
    parser.add_argument("--actual-query-max-bins", type=int, default=12)
    parser.add_argument("--actual-query-min-points-per-bin", type=int, default=5)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue running remaining algorithms after a solver failure.",
    )
    parser.add_argument(
        "--make-plots",
        nargs="?",
        const=True,
        type=_parse_bool,
        default=True,
        metavar="{true,false}",
        help="Generate plots after writing CSV outputs.",
    )
    parser.add_argument(
        "--no-make-plots",
        dest="make_plots",
        action="store_false",
        help=argparse.SUPPRESS,
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


def _query_metric(rows: list[dict[str, Any]], final: dict[str, Any]) -> float:
    final_queries = _as_float(final.get("final_queries"))
    if math.isfinite(final_queries):
        return final_queries
    values = [
        _as_float(row.get("query_budget_actual", row.get("query_budget")))
        for row in rows
    ]
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


def _annotate_calibration_k_excess(
    rows: list[dict[str, Any]],
    final: dict[str, Any],
    *,
    calibration_max_k: int,
) -> None:
    max_k = int(calibration_max_k)
    for row in rows:
        row_k = _as_float(row.get("grover_power"))
        row["grover_power_exceeds_calibration"] = bool(
            math.isfinite(row_k) and int(row_k) > max_k
        )
    final_k = _as_float(final.get("k_max"))
    final["k_max_exceeds_calibration"] = bool(
        math.isfinite(final_k) and int(final_k) > max_k
    )


def _load_transpile_backend(
    name: str | None,
    *,
    use_fractional_gates: bool = True,
) -> tuple[Any, str, str]:
    if name is None or str(name).strip().lower() in {"", "aer", "local", "local_aer"}:
        return AerSimulator(), "local_aer", "aer_simulator"

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    backend = service.backend(
        str(name),
        use_fractional_gates=bool(use_fractional_gates),
    )
    return backend, str(name), "ibm_runtime_backend"


def _resolve_noise_floor(raw: str | float, bundle: Any) -> float:
    if str(raw) == "fit":
        raise ValueError("--noise-floor fit must be resolved from calibration.")
    if str(raw) == "uniform_objective":
        return float(1.0 / (2.0 ** int(bundle.objective_width)))
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"noise_floor must be in [0, 1], got {raw}.")
    return value


def _build_aer_sampler(
    *,
    noise_model: Any,
    seed: int,
    aer_method: str,
    transpile_backend: Any,
    pass_manager: Any,
) -> AerCountSampler:
    return AerCountSampler(
        noise_model=noise_model,
        seed=int(seed),
        method=str(aer_method),
        transpile_backend=transpile_backend,
        pass_manager=pass_manager,
    )


def _run_calibration(
    *,
    state: ExperimentState,
    bundle: Any,
    noise_model: Any,
    transpile_backend: Any,
    pass_manager: Any,
    args: argparse.Namespace,
    contrast_baseline: float | str,
) -> float | None:
    sampler = LoggedAerSampler(
        _build_aer_sampler(
            noise_model=noise_model,
            seed=int(args.seed) + 17,
            aer_method=str(args.aer_method),
            transpile_backend=transpile_backend,
            pass_manager=pass_manager,
        ),
        state.job_rows,
        max_grover_power=int(args.max_grover_power),
    )

    readout_rows, readout_params = run_readout_calibration(
        sampler,
        bundle,
        shots=int(args.readout_shots),
    )
    state.readout_rows = readout_rows
    state.calibration_summary["readout"] = readout_params

    state.amplification_count_rows = run_amplification_scan(
        sampler,
        bundle,
        grover_powers=list(range(int(args.max_grover_power) + 1)),
        repeats=int(args.scan_repeats),
        shots=int(args.scan_shots),
        seed=int(args.seed) + 101,
        verbose=True,
    )
    points, calibration, _ = analyze_amplification(
        state.amplification_count_rows,
        bundle,
        readout_params,
        min_ideal_offset=float(args.min_ideal_offset),
        contrast_baseline=contrast_baseline,
        min_fit_contrast_z=float(args.min_fit_contrast_z),
        min_visible_contrast_z=float(args.min_visible_contrast_z),
        min_baseline_fit_points=int(args.min_baseline_fit_points),
    )
    state.amplification_point_rows = points
    state.calibration_summary.update(calibration)
    return effective_t_for_algorithms(state.calibration_summary)


def _print_run_metrics(final: dict[str, Any]) -> None:
    print(
        "  metrics: "
        f"K_max={final.get('k_max')}, "
        f"queries={_format_float(final.get('final_queries'), 0)}/"
        f"{final.get('max_queries_requested')}, "
        f"min_amp_rel_error={_format_float(final.get('min_normalized_abs_error'))}, "
        f"min_cva_rel_error={_format_float(final.get('min_processed_relative_error'))}, "
        f"runtime_ae={_format_float(final.get('runtime_wall_seconds'))}s"
    )


def _load_reusable_calibration(paths: RunPaths) -> dict[str, Any]:
    calibration = load_json(paths.calibration_summary, default={}) or {}
    if not isinstance(calibration, dict):
        raise ValueError(
            "Existing calibration_summary.json is not a JSON object: "
            f"{paths.calibration_summary}"
        )
    return dict(calibration)


def run_pipeline(args: argparse.Namespace) -> RunPaths:
    execution_max_grover_power = (
        None
        if args.execution_max_grover_power is None
        else int(args.execution_max_grover_power)
    )
    print("Loading 6q CVA pipeline config...", flush=True)
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    print("Building 6q CVA AE problem bundle...", flush=True)
    bundle = build_6q_cva_problem_bundle(config, repo_root=args.repo_root)
    _validate_bundle(bundle)
    print(
        "Built 6q CVA AE problem: "
        f"a_true={bundle.true_amplitude:.12g}, "
        f"CVA={bundle.processed_true_value:.12g}, "
        f"good={bundle.good_bitstring}.",
        flush=True,
    )

    paths = RunPaths(Path(args.run_dir))
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    algorithms = tuple(
        normalize_algorithm_key(name) for name in parse_name_list(args.algorithms)
    )
    budgets = parse_int_list(args.budgets)
    reference_ks = _parse_nonnegative_int_list(args.reference_ks)
    max_budget = max(max(budgets), int(args.max_queries))
    fit_noise_floor = str(args.noise_floor) == "fit"
    noise_floor = (
        math.nan
        if fit_noise_floor
        else _resolve_noise_floor(args.noise_floor, bundle)
    )

    print(
        "Loading transpilation backend "
        f"{args.transpile_backend_name or 'local_aer'} "
        f"(fractional_gates={bool(args.use_fractional_gates)})...",
        flush=True,
    )
    backend, backend_name, backend_mode = _load_transpile_backend(
        args.transpile_backend_name,
        use_fractional_gates=bool(args.use_fractional_gates),
    )
    print(
        f"Using transpilation backend '{backend_name}' in mode '{backend_mode}'.",
        flush=True,
    )
    print("Writing backend snapshot...", flush=True)
    snapshot = backend_snapshot(backend, mode=backend_mode, channel="local_noise_sim")
    snapshot["use_fractional_gates_requested"] = bool(args.use_fractional_gates)
    snapshot["use_fractional_gates_applied"] = bool(
        backend_mode == "ibm_runtime_backend" and args.use_fractional_gates
    )
    save_json(
        snapshot,
        paths.backend_snapshot,
    )
    print("Building transpilation pass manager...", flush=True)
    pass_manager, transpilation_metadata = build_pass_manager_for_backend(
        backend,
        bundle,
        mode="dry-run" if backend_mode == "aer_simulator" else "hardware",
        optimization_level=int(args.transpiler_optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=reference_ks,
        routing_method=str(args.routing_method)
        if str(args.routing_method).lower() != "none"
        else None,
        layout_search_strategy=str(args.layout_search_strategy),
        verbose=True,
    )
    print(
        "Built pass manager: "
        f"strategy={transpilation_metadata.get('strategy')}, "
        f"fallback={transpilation_metadata.get('fallback_used')}.",
        flush=True,
    )
    print(
        f"Building local Aer noise model: {args.noise_profile}@scale={args.noise_scale}...",
        flush=True,
    )
    noise_model = build_noise_model(
        float(args.noise_scale),
        profile=str(args.noise_profile),
    )
    print("Noise model ready.", flush=True)

    metadata = {
        "run_id": paths.run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "mode": "simulated_noise",
        "pipeline": "6q_cva_ae_noisy_simulation",
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
        "budget_policy": "last_trace_point_with_Nq_less_or_equal_budget",
        "actual_query_summary_policy": "log_query_bins_with_median_query_and_median_error",
        "noise_profile": str(args.noise_profile),
        "noise_scale": float(args.noise_scale),
        "noise_floor": None if fit_noise_floor else float(noise_floor),
        "noise_floor_arg": str(args.noise_floor),
        "noise_floor_fit_requested": bool(fit_noise_floor),
        "aer_method": str(args.aer_method),
        "seed": int(args.seed),
        "calibrate": bool(args.calibrate),
        "t_eff_override": None if args.t_eff is None else float(args.t_eff),
        "readout_shots": int(args.readout_shots),
        "scan_repeats": int(args.scan_repeats),
        "scan_shots": int(args.scan_shots),
        "min_fit_contrast_z": float(args.min_fit_contrast_z),
        "min_visible_contrast_z": float(args.min_visible_contrast_z),
        "min_baseline_fit_points": int(args.min_baseline_fit_points),
        "max_grover_power": int(args.max_grover_power),
        "calibration_max_grover_power": int(args.max_grover_power),
        "execution_max_grover_power": execution_max_grover_power,
        "execution_k_cap_enforced": bool(execution_max_grover_power is not None),
        "max_isa_depth": int(args.max_isa_depth),
        "max_isa_2q": int(args.max_isa_2q),
        "reference_ks": reference_ks,
        "transpile_backend_name": backend_name,
        "transpile_backend_mode": backend_mode,
        "use_fractional_gates_requested": bool(args.use_fractional_gates),
        "use_fractional_gates_applied": bool(
            backend_mode == "ibm_runtime_backend" and args.use_fractional_gates
        ),
        "layout_search_strategy": str(args.layout_search_strategy),
        "transpilation": transpilation_metadata,
        "cap_kappa": float(args.cap_kappa),
        "cabiqae_hard_k_cap": bool(args.cabiqae_hard_k_cap),
        "cabiqae_scheduler": (
            "noise_aware_fisher_with_hard_cap"
            if bool(args.cabiqae_hard_k_cap)
            else "noise_aware_fisher_no_hard_cap"
        ),
        "created_at_epoch": time.time(),
        "repo_root": str(Path(args.repo_root).resolve()),
        "problem_metadata": bundle.metadata,
    }
    reusable_calibration: dict[str, Any] = {}
    if not bool(args.calibrate):
        reusable_calibration = _load_reusable_calibration(paths)
        if reusable_calibration:
            print(
                "Loaded existing calibration summary: "
                f"status={reusable_calibration.get('calibration_status')}, "
                f"T_eff={_format_float(effective_t_for_algorithms(reusable_calibration))}, "
                f"noise_floor={_format_float(reusable_calibration.get('contrast_baseline'))}.",
                flush=True,
            )
        else:
            print(
                "No existing calibration_summary.json found; running without fitted T_eff.",
                flush=True,
            )

    state = ExperimentState(paths=paths, config=metadata)
    state.calibration_summary.update(reusable_calibration)
    save_json(metadata, paths.config)

    print(
        "Running ISA preflight/transpilation report "
        f"for k=0..{int(args.max_grover_power)}...",
        flush=True,
    )
    preflight_rows, allowed_max = run_preflight(
        bundle,
        pass_manager,
        state,
        max_grover_power=int(args.max_grover_power),
        max_isa_depth=int(args.max_isa_depth),
        max_isa_2q=int(args.max_isa_2q),
        verbose=True,
    )
    metadata["max_grover_power_after_preflight"] = int(allowed_max)
    print(
        "Loaded 6q CVA AE problem under simulated noise: "
        f"a_true={bundle.true_amplitude:.12g}, "
        f"CVA={bundle.processed_true_value:.12g}, "
        f"good={bundle.good_bitstring}, "
        f"noise={args.noise_profile}@scale={args.noise_scale}, "
        f"noise_floor={'fit' if fit_noise_floor else f'{noise_floor:.6g}'}, "
        f"backend={backend_name}, "
        f"k_allowed={allowed_max}/{args.max_grover_power}, "
        f"execution_k_cap={'none' if execution_max_grover_power is None else execution_max_grover_power}."
    )

    t_eff = float(args.t_eff) if args.t_eff is not None else None
    if bool(args.calibrate) and args.t_eff is None:
        t_eff = _run_calibration(
            state=state,
            bundle=bundle,
            noise_model=noise_model,
            transpile_backend=backend,
            pass_manager=pass_manager,
            args=args,
            contrast_baseline="fit" if fit_noise_floor else float(noise_floor),
        )
    elif not bool(args.calibrate):
        if t_eff is None:
            t_eff = effective_t_for_algorithms(state.calibration_summary)
        if not state.calibration_summary:
            state.calibration_summary["calibration_status"] = "skipped"
        else:
            state.calibration_summary.setdefault(
                "calibration_status",
                "reused_existing_calibration",
            )
    if args.t_eff is not None:
        state.calibration_summary["calibration_status"] = "manual_t_eff_override"
        state.calibration_summary["t_eff_manual"] = float(args.t_eff)
    if fit_noise_floor:
        if "contrast_baseline" not in state.calibration_summary:
            raise ValueError(
                "--noise-floor fit requires a calibration_summary.json with "
                "'contrast_baseline' when --calibrate false is used."
            )
        noise_floor = float(state.calibration_summary["contrast_baseline"])
    state.calibration_summary.setdefault("contrast_baseline", float(noise_floor))
    metadata["noise_floor"] = float(noise_floor)
    metadata["contrast_baseline_mode"] = str(
        state.calibration_summary.get("contrast_baseline_mode", "fixed")
    )
    metadata["t_eff"] = None if t_eff is None else float(t_eff)
    metadata["calibration_summary"] = state.calibration_summary
    save_json(metadata, paths.config)
    save_json(state.calibration_summary, paths.calibration_summary)
    _save_csv(state.job_rows, paths.runtime_jobs)
    _save_csv(state.readout_rows, paths.readout_calibration)
    _save_csv(state.amplification_count_rows, paths.amplification_counts)
    _save_csv(state.amplification_point_rows, paths.amplification_points)

    trace_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    construct_circuit_cache: dict[tuple[Any, ...], Any] = {}

    trace_extra = {
        "simulation_regime": "simulated_noise",
        "instance_name": "6q_instance",
        "noise_profile": str(args.noise_profile),
        "noise_scale": float(args.noise_scale),
        "noise_floor": float(noise_floor),
        "aer_method": str(args.aer_method),
        "t_eff": np.nan if t_eff is None else float(t_eff),
        "calibration_status": str(
            state.calibration_summary.get("calibration_status", "")
        ),
        "max_grover_power": int(args.max_grover_power),
        "calibration_max_grover_power": int(args.max_grover_power),
        "execution_max_grover_power": (
            np.nan if execution_max_grover_power is None else int(execution_max_grover_power)
        ),
        "execution_k_cap_enforced": bool(execution_max_grover_power is not None),
        "cabiqae_hard_k_cap": bool(args.cabiqae_hard_k_cap),
        "cabiqae_scheduler": (
            "noise_aware_fisher_with_hard_cap"
            if bool(args.cabiqae_hard_k_cap)
            else "noise_aware_fisher_no_hard_cap"
        ),
        "transpile_backend_name": backend_name,
        "use_fractional_gates_applied": bool(
            backend_mode == "ibm_runtime_backend" and args.use_fractional_gates
        ),
        "transpilation_strategy": str(transpilation_metadata.get("strategy", "")),
        "n_shots": int(args.n_shots),
    }

    for rep in range(int(args.repetitions)):
        aer = _build_aer_sampler(
            noise_model=noise_model,
            seed=int(args.seed) + 7919 * rep,
            aer_method=str(args.aer_method),
            transpile_backend=backend,
            pass_manager=pass_manager,
        )
        sampler = LoggedAerSampler(
            aer,
            state.job_rows,
            max_grover_power=execution_max_grover_power,
        )
        for alg_index, algorithm in enumerate(algorithms):
            run_seed = int(args.seed) + 1009 * rep + 37 * alg_index
            print(
                f"[rep {rep + 1}/{args.repetitions}] {algorithm} "
                f"(seed={run_seed}, T_eff={_format_float(t_eff)})"
            )
            try:
                rows, final = run_algorithm_once(
                    algorithm,
                    sampler,
                    bundle,
                    run_kind=RUN_KIND,
                    repetition=rep,
                    epsilon_target=float(args.epsilon_target),
                    alpha=float(args.alpha),
                    n_shots=int(args.n_shots),
                    max_queries=int(max_budget),
                    t_eff=t_eff,
                    seed=run_seed,
                    algorithm_labels=ALGORITHM_LABELS,
                    cap_kappa=float(args.cap_kappa),
                    disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
                    construct_circuit_cache=construct_circuit_cache,
                    construct_circuit_mode="full",
                    solver_kwargs={"noise_floor": float(noise_floor)},
                    trace_extra=trace_extra,
                    show_details=bool(args.show_details),
                )
            except Exception as exc:
                error_rows.append(
                    {
                        "run_kind": RUN_KIND,
                        "simulation_regime": "simulated_noise",
                        "noise_profile": str(args.noise_profile),
                        "noise_scale": float(args.noise_scale), 
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
                max_queries=int(max_budget),
            )
            _annotate_calibration_k_excess(
                rows,
                final,
                calibration_max_k=int(args.max_grover_power),
            )
            trace_rows.extend(rows)
            final_rows.append(final)

            selected_budget_rows = rows_at_budgets(
                rows,
                budgets,
                run_kind=RUN_KIND,
            )
            for budget_row in selected_budget_rows:
                budget_row.update(trace_extra)
                budget_row["grover_power_exceeds_calibration"] = bool(
                    int(budget_row.get("grover_power", 0)) > int(args.max_grover_power)
                )
            budget_rows.extend(add_cva_aliases(selected_budget_rows))
            print(
                "  final: "
                f"Nq={final['final_queries']:.0f}, "
                f"a_hat={final['final_estimate']:.12g}, "
                f"CVA_hat={final['cva_estimate']:.12g}, "
                f"rel_CVA_err={final['cva_relative_error']:.4g}"
            )
            _print_run_metrics(final)

    fixed_budget_summary_rows = add_cva_aliases(
        aggregate_budget_summary(
            budget_rows,
            total_repetitions=int(args.repetitions),
            group_by_budget=True,
            bootstrap_samples=int(args.bootstrap_samples),
        )
    )
    actual_query_summary_rows = add_cva_aliases(
        aggregate_budget_summary(
            trace_rows,
            total_repetitions=int(args.repetitions),
            max_bins=int(args.actual_query_max_bins),
            min_points_per_bin=int(args.actual_query_min_points_per_bin),
            bootstrap_samples=int(args.bootstrap_samples),
        )
    )

    budget_rows_path = paths.run_dir / "budget_rows.csv"
    actual_query_summary_path = paths.run_dir / "actual_query_summary.csv"
    _save_csv(trace_rows, paths.direct_trace)
    _save_csv(final_rows, paths.direct_final)
    _save_csv(budget_rows, budget_rows_path)
    _save_csv(budget_rows, paths.replay_budget)
    _save_csv(fixed_budget_summary_rows, paths.budget_summary)
    _save_csv(actual_query_summary_rows, actual_query_summary_path)
    _save_csv(state.job_rows, paths.runtime_jobs)
    _save_csv(error_rows, paths.errors)
    write_trace_bundle(
        paths.trace_bundle,
        trace_rows=trace_rows,
        budget_rows=budget_rows,
        amplification_rows=state.amplification_point_rows,
    )
    paths.write_manifest()

    if bool(args.make_plots):
        try:
            from plot_noisy_cva_ae import make_plots

            make_plots(paths.run_dir, algorithms=algorithms)
        except Exception as exc:
            error_rows.append(
                {
                    "run_kind": RUN_KIND,
                    "simulation_regime": "simulated_noise",
                    "instance_name": "6q_instance",
                    "error_type": type(exc).__name__,
                    "error": f"Plot generation failed: {exc}",
                }
            )
            _save_csv(error_rows, paths.errors)
            print(f"Plot generation failed: {type(exc).__name__}: {exc}")
            if not bool(args.continue_on_error):
                raise

    print("")
    print("Experiment policy:")
    print("  Fixed-budget rows: choose last trace point with N_q <= Budget.")
    print("  Actual-query summary: log bins over observed N_q, plot medians per bin.")
    print("  --max-grover-power controls preflight/calibration, not AE execution.")
    print("  BAE is capped by max(budgets, --max-queries), not by v2 median matching.")

    return paths


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.execution_max_grover_power is not None and int(args.execution_max_grover_power) < 0:
        parser.error("--execution-max-grover-power must be non-negative.")
    if str(args.noise_floor) == "fit":
        if bool(args.calibrate) and args.t_eff is not None:
            parser.error("--noise-floor fit requires calibration, so do not pass --t-eff.")
    paths = run_pipeline(args)
    print(f"Wrote noisy CVA AE outputs to {paths.run_dir}")


if __name__ == "__main__":
    main()
