from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import (
    DEFAULT_RUN_DIR,
    REPO_ROOT,
    add_cva_alias_columns,
    load_config,
    parse_int_list,
    parse_name_list,
    preferred_field_order,
)
from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.io import load_csv, save_csv, save_json
from quantum_cva.amplitude_estimation.experiments.statistics import (
    aggregate_budget_summary,
)


DEFAULT_BUDGETS = "128,256,512,1024,2048,4096,8192,16384"
DEFAULT_ROBUSTNESS_DIR = (
    REPO_ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "cva_robustness_test"
)
RUN_KIND = "classical_monte_carlo"
ALGORITHM_KEY = "classical_mc"
ALGORITHM_LABEL = "Classical MC"
DEFAULT_SAMPLE_MODEL = "noisy_contrast"


@dataclass(frozen=True)
class MonteCarloCase:
    case_id: str
    call_strike: float
    put_strike: float
    benchmark_path: Path
    exposure_path: Path
    config: Any
    bundle: Any


def _load_existing_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _optional_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _optional_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finite_positive_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f) or value_f <= 0.0:
        return None
    return value_f


def _finite_probability_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f) or not 0.0 <= value_f <= 1.0:
        return None
    return value_f


def _parse_bool_or_none(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, 1/0, or omitted.")


def _resolve_existing_robustness_case_dir(
    robustness_dir: Path,
    case_id: str,
) -> Path:
    candidates = [
        robustness_dir / "data" / case_id,
        robustness_dir / case_id,
    ]
    for candidate in candidates:
        benchmark = candidate / "benchmark" / "benchmark.npz"
        exposure = (
            candidate
            / "quantum"
            / "training"
            / "crca"
            / "positive_exposure"
            / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
        )
        if benchmark.exists() and exposure.exists():
            return candidate
    details = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find complete robustness artifacts for case {case_id!r}. "
        f"Checked:\n{details}"
    )


def _with_case_paths(config: Any, *, case_dir: Path) -> Any:
    paths = replace(
        config.paths,
        benchmark_relative_path=str(case_dir / "benchmark" / "benchmark.npz"),
        crca_exposure_training_relative_path=str(
            case_dir
            / "quantum"
            / "training"
            / "crca"
            / "positive_exposure"
            / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
        ),
        results_dir_relative_path=str(case_dir / "pipeline_run"),
    )
    return replace(config, paths=paths)


def _with_case_strikes(
    config: Any,
    *,
    call_strike: float,
    put_strike: float,
) -> Any:
    if len(config.instruments) < 2:
        raise ValueError("Expected at least two instruments in the CVA config.")
    instruments = list(config.instruments)
    instruments[0] = replace(instruments[0], strike=float(call_strike))
    instruments[1] = replace(instruments[1], strike=float(put_strike))
    return replace(config, instruments=tuple(instruments))


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


def _load_robustness_cases(
    *,
    base_config: Any,
    robustness_dir: Path,
    selected_case_ids: tuple[str, ...] | None,
    repo_root: str | Path,
) -> list[MonteCarloCase]:
    config_path = robustness_dir / "robustness_sweep_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Robustness sweep config does not exist: {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_cases = list(payload.get("cases", []))
    if not raw_cases:
        raise ValueError(f"No cases found in {config_path}.")

    selected = None if selected_case_ids is None else set(selected_case_ids)
    cases: list[MonteCarloCase] = []
    for raw in raw_cases:
        case_id = str(raw["case_id"])
        if selected is not None and case_id not in selected:
            continue
        case_dir = _resolve_existing_robustness_case_dir(robustness_dir, case_id)
        config = _with_case_strikes(
            base_config,
            call_strike=float(raw["call_strike"]),
            put_strike=float(raw["put_strike"]),
        )
        config = _with_case_paths(config, case_dir=case_dir)
        bundle = build_6q_cva_problem_bundle(config, repo_root=repo_root)
        _validate_bundle(bundle)
        cases.append(
            MonteCarloCase(
                case_id=case_id,
                call_strike=float(raw["call_strike"]),
                put_strike=float(raw["put_strike"]),
                benchmark_path=case_dir / "benchmark" / "benchmark.npz",
                exposure_path=(
                    case_dir
                    / "quantum"
                    / "training"
                    / "crca"
                    / "positive_exposure"
                    / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
                ),
                config=config,
                bundle=bundle,
            )
        )

    if not cases:
        requested = sorted(selected or [])
        raise ValueError(f"No robustness cases selected. Requested={requested!r}.")
    if selected is not None:
        found = {case.case_id for case in cases}
        missing = sorted(selected.difference(found))
        if missing:
            raise ValueError(f"Unknown robustness case ids: {missing!r}.")
    return cases


def _load_single_case(*, config: Any, repo_root: str | Path) -> MonteCarloCase:
    bundle = build_6q_cva_problem_bundle(config, repo_root=repo_root)
    _validate_bundle(bundle)
    return MonteCarloCase(
        case_id="base",
        call_strike=float(config.instruments[0].strike),
        put_strike=float(config.instruments[1].strike),
        benchmark_path=Path(config.paths.benchmark_relative_path),
        exposure_path=Path(config.paths.crca_exposure_training_relative_path),
        config=config,
        bundle=bundle,
    )


def _processed_fields(bundle: Any, estimate: float) -> dict[str, float | str]:
    processed = float(bundle.process(float(estimate)))
    true_value = float(bundle.processed_true_value)
    processed_abs_error = abs(processed - true_value)
    return {
        "target_name": str(bundle.target_name),
        "processed_true_value": true_value,
        "processed_estimate": processed,
        "processed_abs_error": processed_abs_error,
        "processed_relative_error": processed_abs_error / max(abs(true_value), 1e-12),
    }


def _load_calibration_summary(run_dir: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "calibration_summary.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    payload = run_config.get("calibration_summary", {})
    return dict(payload) if isinstance(payload, dict) else {}


def _hardware_sampling_parameters(
    *,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    grover_power = int(args.sampling_grover_power)
    source = str(args.hardware_probability_source).strip().lower()
    if source == "mitigated_points":
        for row in load_csv(run_dir / "amplification_points.csv"):
            if int(float(row.get("grover_power", -1))) != grover_power:
                continue
            probability = _finite_probability_or_none(row.get("p_hw_mitigated"))
            if probability is None:
                break
            return {
                "sample_model": "bernoulli_hardware_good_state",
                "sampling_probability_source": f"p_hw_mitigated(k={grover_power})",
                "sampling_grover_power": grover_power,
                "sampling_probability": float(probability),
                "sampling_probability_se": max(
                    0.0, _optional_float(row.get("p_hw_mitigated_se"), 0.0)
                ),
                "sampling_probability_shots": int(float(row.get("shots", 0))),
                "contrast_model": "empirical_hardware",
                "contrast_baseline": np.nan,
                "contrast_prefactor": np.nan,
                "t_eff": np.nan,
                "noise_amplification_factor": 1.0,
                "noise_contrast": np.nan,
            }
        raise ValueError(f"No valid amplification_points.csv row found for k={grover_power}.")
    if source != "raw_counts":
        raise ValueError(
            "--hardware-probability-source must be mitigated_points or raw_counts."
        )
    successes = 0
    shots = 0
    for row in load_csv(run_dir / "amplification_counts.csv"):
        if int(float(row.get("grover_power", -1))) != grover_power:
            continue
        successes += int(float(row.get("good_counts", row.get("one_counts", 0))))
        shots += int(float(row.get("shots", 0)))
    if shots <= 0:
        raise ValueError(f"No amplification_counts.csv rows found for k={grover_power}.")
    probability = float(successes / shots)
    return {
        "sample_model": "bernoulli_hardware_good_state",
        "sampling_probability_source": f"raw_counts(k={grover_power})",
        "sampling_grover_power": grover_power,
        "sampling_probability": probability,
        "sampling_probability_se": math.sqrt(max(probability * (1.0 - probability), 0.0) / shots),
        "sampling_probability_shots": shots,
        "contrast_model": "empirical_hardware",
        "contrast_baseline": np.nan,
        "contrast_prefactor": np.nan,
        "t_eff": np.nan,
        "noise_amplification_factor": 1.0,
        "noise_contrast": np.nan,
    }


def _resolve_noisy_sampling_parameters(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    run_config: dict[str, Any],
    calibration_summary: dict[str, Any],
) -> dict[str, Any]:
    sample_model = str(args.sample_model).strip().lower()
    if sample_model == "hardware_counts":
        return _hardware_sampling_parameters(args=args, run_dir=run_dir)
    if sample_model == "ideal":
        return {
            "sample_model": "bernoulli_good_state",
            "contrast_model": "ideal",
            "contrast_baseline": np.nan,
            "contrast_prefactor": 1.0,
            "t_eff": np.nan,
            "noise_amplification_factor": 1.0,
            "noise_contrast": 1.0,
        }
    if sample_model != "noisy_contrast":
        raise ValueError(f"Unknown sample model: {args.sample_model!r}.")

    baseline = _finite_probability_or_none(args.contrast_baseline)
    if baseline is None:
        baseline = _finite_probability_or_none(
            calibration_summary.get("contrast_baseline")
        )
    if baseline is None:
        baseline = _finite_probability_or_none(run_config.get("noise_floor"))
    if baseline is None:
        raise ValueError(
            "No valid noise baseline found. Provide --contrast-baseline or use a "
            "run directory with calibration_summary.json/config.json."
        )

    contrast_model = str(args.contrast_model).strip().lower()
    if contrast_model not in {"auto", "free_intercept", "zero_intercept"}:
        raise ValueError(
            "--contrast-model must be one of auto, free_intercept, zero_intercept."
        )
    if contrast_model == "auto":
        contrast_model = (
            "free_intercept"
            if calibration_summary.get("contrast_prefactor") is not None
            and calibration_summary.get("t_eff_free_intercept") is not None
            else "zero_intercept"
        )

    prefactor = _finite_positive_or_none(args.contrast_prefactor)
    t_eff = _finite_positive_or_none(args.t_eff)
    if contrast_model == "free_intercept":
        if prefactor is None:
            prefactor = _finite_positive_or_none(
                calibration_summary.get("contrast_prefactor")
            )
        if t_eff is None:
            t_eff = _finite_positive_or_none(
                calibration_summary.get("t_eff_free_intercept")
            )
    else:
        prefactor = 1.0 if prefactor is None else float(prefactor)
        if t_eff is None:
            t_eff = _finite_positive_or_none(run_config.get("t_eff"))
        if t_eff is None:
            t_eff = _finite_positive_or_none(
                calibration_summary.get("t_eff_zero_intercept")
            )

    if t_eff is None:
        raise ValueError(
            "No valid T_eff found for noisy Monte Carlo. Provide --t-eff or use a "
            "run directory with calibration_summary.json/config.json."
        )
    if prefactor is None:
        prefactor = 1.0

    amplification_factor = float(args.noise_amplification_factor)
    if not math.isfinite(amplification_factor) or amplification_factor < 0.0:
        raise ValueError("--noise-amplification-factor must be finite and non-negative.")
    contrast = float(
        np.clip(
            float(prefactor) * math.exp(-amplification_factor / float(t_eff)),
            0.0,
            1.0,
        )
    )
    return {
        "sample_model": "bernoulli_noisy_good_state",
        "contrast_model": contrast_model,
        "contrast_baseline": float(baseline),
        "contrast_prefactor": float(prefactor),
        "t_eff": float(t_eff),
        "noise_amplification_factor": float(amplification_factor),
        "noise_contrast": float(contrast),
    }


def _sampling_probability(true_amplitude: float, noise_params: dict[str, Any]) -> float:
    if str(noise_params["sample_model"]) == "bernoulli_hardware_good_state":
        return float(noise_params["sampling_probability"])
    if str(noise_params["sample_model"]) == "bernoulli_good_state":
        return float(np.clip(true_amplitude, 0.0, 1.0))
    baseline = float(noise_params["contrast_baseline"])
    contrast = float(noise_params["noise_contrast"])
    return float(
        np.clip(baseline + contrast * (float(true_amplitude) - baseline), 0.0, 1.0)
    )


def _monte_carlo_budget_rows(
    *,
    case: MonteCarloCase,
    budgets: list[int],
    repetition: int,
    seed: int,
    noise_params: dict[str, Any],
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    if not budgets:
        return []
    ordered_budgets = sorted(set(int(budget) for budget in budgets))
    if ordered_budgets[0] <= 0:
        raise ValueError("Budgets must be positive integers.")

    rng = np.random.default_rng(int(seed))
    true_amplitude = float(case.bundle.true_amplitude)
    sampling_amplitude = _sampling_probability(true_amplitude, noise_params)
    probability_mode = str(noise_params.get("probability_mode", "fixed"))
    probability_se = float(noise_params.get("sampling_probability_se", 0.0))
    probability_se_scale = float(noise_params.get("probability_se_scale", 1.0))
    if probability_mode == "normal":
        repetition_sampling_amplitude = float(
            np.clip(rng.normal(sampling_amplitude, probability_se_scale * probability_se), 0.0, 1.0)
        )
    elif probability_mode == "fixed":
        repetition_sampling_amplitude = sampling_amplitude
    else:
        raise ValueError("--probability-mode must be fixed or normal.")
    total_successes = 0
    previous_budget = 0
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for step_index, budget in enumerate(ordered_budgets):
        increment = int(budget - previous_budget)
        if increment > 0:
            total_successes += int(rng.binomial(increment, repetition_sampling_amplitude))
            previous_budget = int(budget)

        estimate = float(total_successes / float(budget))
        abs_error = abs(estimate - true_amplitude)
        elapsed = time.perf_counter() - start
        row = {
            "run_kind": RUN_KIND,
            "simulation_regime": "classical_monte_carlo",
            "repetition": int(repetition),
            "algorithm": ALGORITHM_LABEL,
            "algorithm_key": ALGORITHM_KEY,
            "step_index": int(step_index),
            "budget": int(budget),
            "query_budget": float(budget),
            "query_budget_actual": float(budget),
            "estimate": estimate,
            "abs_error": abs_error,
            "normalized_abs_error": abs_error / max(true_amplitude, 1e-12),
            "normalized_sq_error": (
                (estimate - true_amplitude) / max(true_amplitude, 1e-12)
            )
            ** 2,
            "grover_power": 0,
            "grover_power_exceeds_calibration": False,
            "k_max_budget": 0,
            "amplification_factor": 1,
            "a_true": true_amplitude,
            "a_sampling": sampling_amplitude,
            "a_sampling_repetition": repetition_sampling_amplitude,
            "noise_bias": sampling_amplitude - true_amplitude,
            "contrast_baseline": noise_params["contrast_baseline"],
            "contrast_baseline_mode": noise_params["contrast_model"],
            "contrast_prefactor": noise_params["contrast_prefactor"],
            "noise_amplification_factor": noise_params["noise_amplification_factor"],
            "noise_contrast": noise_params["noise_contrast"],
            "t_eff": noise_params["t_eff"],
            "runtime_wall_seconds": float(elapsed),
            "time_to_budget_seconds": float(elapsed),
            "coverage": np.nan,
            "classical_samples": int(budget),
            "successes": int(total_successes),
            "sample_model": noise_params["sample_model"],
            **_processed_fields(case.bundle, estimate),
            **extra,
        }
        rows.append(add_cva_alias_columns(row))

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a classical Monte Carlo baseline for the same 6q CVA amplitude "
            "target used by the noisy AE experiment."
        )
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--output-name",
        default="montecarlo_budget_rows.csv",
        help="CSV filename written inside --run-dir.",
    )
    parser.add_argument(
        "--summary-output-name",
        default="montecarlo_budget_summary.csv",
        help="Optional budget-summary CSV filename written inside --run-dir.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--config-attr", default="CONFIG")
    parser.add_argument(
        "--robustness-suite",
        type=_parse_bool_or_none,
        default=None,
        nargs="?",
        const=True,
        help=(
            "Use robustness cases. If omitted, the value is inferred from "
            "--run-dir/config.json when available."
        ),
    )
    parser.add_argument("--robustness-dir", default=None)
    parser.add_argument("--robustness-cases", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--noise-profile", default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--noise-floor", type=float, default=None)
    parser.add_argument(
        "--sample-model",
        choices=("hardware_counts", "noisy_contrast", "ideal"),
        default=DEFAULT_SAMPLE_MODEL,
        help=(
            "Distribution used by the classical Monte Carlo. 'hardware_counts' "
            "samples an empirical hardware probability; 'noisy_contrast' "
            "samples a biased noisy amplitude from the calibration contrast "
            "model; 'ideal' samples a_true directly."
        ),
    )
    parser.add_argument(
        "--hardware-probability-source",
        choices=("mitigated_points", "raw_counts"),
        default="mitigated_points",
    )
    parser.add_argument("--sampling-grover-power", type=int, default=0)
    parser.add_argument("--probability-mode", choices=("fixed", "normal"), default="fixed")
    parser.add_argument("--probability-se-scale", type=float, default=1.0)
    parser.add_argument(
        "--contrast-model",
        choices=("auto", "free_intercept", "zero_intercept"),
        default="auto",
        help=(
            "Contrast-decay model for --sample-model noisy_contrast. 'auto' "
            "uses the fitted free-intercept calibration when available."
        ),
    )
    parser.add_argument(
        "--contrast-baseline",
        type=float,
        default=None,
        help=(
            "Noisy asymptotic good-state probability. Defaults to "
            "calibration_summary.contrast_baseline, then config noise_floor."
        ),
    )
    parser.add_argument(
        "--contrast-prefactor",
        type=float,
        default=None,
        help="Optional fitted contrast prefactor. Defaults to calibration if available.",
    )
    parser.add_argument(
        "--t-eff",
        type=float,
        default=None,
        help="Optional effective contrast-decay scale. Defaults to calibration/config.",
    )
    parser.add_argument(
        "--noise-amplification-factor",
        type=float,
        default=1.0,
        help=(
            "Effective noisy circuit amplification factor used for the classical "
            "sample probability. Use 1 for direct A sampling."
        ),
    )
    parser.add_argument("--n-shots", type=int, default=None)
    parser.add_argument("--max-grover-power", type=int, default=None)
    parser.add_argument("--calibration-case-id", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    run_config = _load_existing_run_config(run_dir)
    calibration_summary = _load_calibration_summary(run_dir, run_config)
    noise_params = _resolve_noisy_sampling_parameters(
        args=args,
        run_dir=run_dir,
        run_config=run_config,
        calibration_summary=calibration_summary,
    )
    noise_params["probability_mode"] = str(args.probability_mode)
    noise_params["probability_se_scale"] = float(args.probability_se_scale)

    base_config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    robustness_suite = (
        bool(run_config.get("robustness_suite", False))
        if args.robustness_suite is None
        else bool(args.robustness_suite)
    )
    robustness_dir = Path(
        args.robustness_dir
        or run_config.get("robustness_dir")
        or str(DEFAULT_ROBUSTNESS_DIR)
    )
    selected_case_ids = (
        parse_name_list(args.robustness_cases)
        if args.robustness_cases is not None
        else None
    )
    if selected_case_ids is None and robustness_suite and run_config.get("cases"):
        selected_case_ids = tuple(str(case["case_id"]) for case in run_config["cases"])

    cases = (
        _load_robustness_cases(
            base_config=base_config,
            robustness_dir=robustness_dir,
            selected_case_ids=selected_case_ids,
            repo_root=args.repo_root,
        )
        if robustness_suite
        else [_load_single_case(config=base_config, repo_root=args.repo_root)]
    )

    budgets = (
        parse_int_list(args.budgets)
        if args.budgets is not None
        else [int(value) for value in run_config.get("budgets", [])]
    )
    if not budgets:
        budgets = parse_int_list(DEFAULT_BUDGETS)

    repetitions = _optional_int(
        args.repetitions,
        int(run_config.get("replay_repetitions", run_config.get("repetitions", 20))),
    )
    seed = _optional_int(args.seed, int(run_config.get("seed", 12345)))
    noise_profile = str(args.noise_profile or run_config.get("noise_profile", "none"))
    noise_scale = _optional_float(
        args.noise_scale,
        _optional_float(run_config.get("noise_scale"), np.nan),
    )
    noise_floor = _optional_float(
        args.noise_floor,
        _optional_float(run_config.get("noise_floor"), np.nan),
    )
    n_shots = _optional_int(args.n_shots, int(run_config.get("n_shots", 0)))
    max_grover_power = _optional_int(
        args.max_grover_power,
        int(run_config.get("max_grover_power", 0)),
    )
    calibration_case_id = str(
        args.calibration_case_id
        or run_config.get("calibration_case_id")
        or cases[0].case_id
    )

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        case_extra = {
            "noise_profile": noise_profile,
            "noise_scale": noise_scale,
            "noise_floor": noise_floor,
            "instance_name": "6q_instance",
            "case_id": case.case_id,
            "call_strike": float(case.call_strike),
            "put_strike": float(case.put_strike),
            "benchmark_path": str(case.benchmark_path),
            "exposure_training_path": str(case.exposure_path),
            "cva_true": float(case.bundle.processed_true_value),
            "calibration_case_id": calibration_case_id,
            "n_shots": n_shots,
            "max_grover_power": max_grover_power,
        }
        for case_repetition in range(repetitions):
            suite_run_index = case_index * repetitions + case_repetition
            rep_extra = {
                **case_extra,
                "case_repetition": int(case_repetition),
                "suite_run_index": int(suite_run_index),
            }
            rep_seed = seed + 104729 * case_index + 7919 * case_repetition
            rows.extend(
                _monte_carlo_budget_rows(
                    case=case,
                    budgets=budgets,
                    repetition=suite_run_index,
                    seed=rep_seed,
                    noise_params=noise_params,
                    extra=rep_extra,
                )
            )

    output_path = run_dir / str(args.output_name)
    save_csv(rows, output_path, fieldnames=preferred_field_order(rows))

    summary_rows = aggregate_budget_summary(
        rows,
        total_repetitions=len(cases) * repetitions,
        group_by_budget=True,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    summary_path = run_dir / str(args.summary_output_name)
    summary_rows = [add_cva_alias_columns(row) for row in summary_rows]
    save_csv(
        summary_rows,
        summary_path,
        fieldnames=preferred_field_order(summary_rows),
    )

    metadata_path = run_dir / "montecarlo_config.json"
    save_json(
        {
            "run_kind": RUN_KIND,
            "algorithm_key": ALGORITHM_KEY,
            "sample_model": noise_params["sample_model"],
            "contrast_model": noise_params["contrast_model"],
            "contrast_baseline": noise_params["contrast_baseline"],
            "contrast_prefactor": noise_params["contrast_prefactor"],
            "t_eff": noise_params["t_eff"],
            "noise_amplification_factor": noise_params["noise_amplification_factor"],
            "noise_contrast": noise_params["noise_contrast"],
            "sampling_probability_source": noise_params.get("sampling_probability_source"),
            "sampling_grover_power": noise_params.get("sampling_grover_power"),
            "sampling_probability_se": noise_params.get("sampling_probability_se", 0.0),
            "sampling_probability_shots": noise_params.get("sampling_probability_shots"),
            "probability_mode": noise_params["probability_mode"],
            "probability_se_scale": noise_params["probability_se_scale"],
            "a_true": float(cases[0].bundle.true_amplitude),
            "a_sampling": _sampling_probability(
                float(cases[0].bundle.true_amplitude),
                noise_params,
            ),
            "run_dir": str(run_dir),
            "output_csv": str(output_path),
            "summary_csv": str(summary_path),
            "source_run_config": str(run_dir / "config.json")
            if (run_dir / "config.json").exists()
            else None,
            "robustness_suite": robustness_suite,
            "robustness_dir": str(robustness_dir),
            "cases": [
                {
                    "case_id": case.case_id,
                    "call_strike": float(case.call_strike),
                    "put_strike": float(case.put_strike),
                    "a_true": float(case.bundle.true_amplitude),
                    "a_sampling": _sampling_probability(
                        float(case.bundle.true_amplitude),
                        noise_params,
                    ),
                    "cva_true": float(case.bundle.processed_true_value),
                }
                for case in cases
            ],
            "repetitions": repetitions,
            "total_case_repetitions": len(cases) * repetitions,
            "budgets": budgets,
            "seed": seed,
        },
        metadata_path,
    )

    print(f"Wrote {len(rows)} Monte Carlo budget rows to {output_path}")
    print(f"Wrote Monte Carlo budget summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
