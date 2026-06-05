from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BETA_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in BETA_DIR.parents if (parent / "pyproject.toml").exists())
for path in (BETA_DIR, REPO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from hardware_bae_biqae_cabiqae_core import (  # noqa: E402
    empirical_contrast_model,
    load_csv,
    parse_int_list,
    save_csv,
    save_json,
)


DEFAULT_RUN_DIR = BETA_DIR / "experiment_results" / "csv_results"
ALGORITHM_KEY = "classical_mc"
ALGORITHM_LABEL = "Classical MC"


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _probability_from_counts(rows: list[Mapping[str, Any]], grover_power: int) -> tuple[float, float, int]:
    one = 0
    shots = 0
    for row in rows:
        if int(float(row.get("grover_power", -1))) != int(grover_power):
            continue
        one += int(float(row.get("one_counts", 0)))
        shots += int(float(row.get("shots", 0)))
    if shots <= 0:
        raise ValueError(f"No amplification_counts rows found for k={int(grover_power)}.")
    p = float(one / shots)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / shots)
    return p, float(se), int(shots)


def _probability_from_points(rows: list[Mapping[str, Any]], grover_power: int) -> tuple[float, float, int]:
    for row in rows:
        if int(float(row.get("grover_power", -1))) != int(grover_power):
            continue
        p = _as_float(row.get("p_hw_mitigated"))
        se = _as_float(row.get("p_hw_mitigated_se"), 0.0)
        shots = int(float(row.get("shots", 0)))
        if np.isfinite(p):
            return float(np.clip(p, 0.0, 1.0)), float(max(se, 0.0)), shots
    raise ValueError(f"No amplification_points rows found for k={int(grover_power)}.")


def _noisy_contrast_probability(
    *,
    a_true: float,
    calibration_summary: Mapping[str, Any],
    amplification_factor: float,
) -> tuple[float, dict[str, Any]]:
    model_name, prefactor, t_eff = empirical_contrast_model(calibration_summary)
    contrast = float(
        np.clip(float(prefactor) * np.exp(-float(amplification_factor) / float(t_eff)), 0.0, 1.0)
    )
    p = float(np.clip(0.5 + contrast * (float(a_true) - 0.5), 0.0, 1.0))
    return p, {
        "contrast_model": model_name,
        "contrast_prefactor": float(prefactor),
        "t_eff": float(t_eff),
        "noise_amplification_factor": float(amplification_factor),
        "noise_contrast": float(contrast),
    }


def resolve_sampling_probability(
    *,
    run_dir: Path,
    config: Mapping[str, Any],
    calibration_summary: Mapping[str, Any],
    sample_model: str,
    hardware_probability_source: str,
    grover_power: int,
    amplification_factor: float,
) -> tuple[float, float, dict[str, Any]]:
    a_true = float(config["a_true"])
    model = str(sample_model).strip().lower()
    if model == "ideal":
        return a_true, 0.0, {
            "sample_model": "bernoulli_good_state",
            "sampling_probability_source": "ideal_a_true",
        }
    if model == "noisy_contrast":
        p, metadata = _noisy_contrast_probability(
            a_true=a_true,
            calibration_summary=calibration_summary,
            amplification_factor=float(amplification_factor),
        )
        metadata.update(
            {
                "sample_model": "bernoulli_noisy_good_state",
                "sampling_probability_source": "calibrated_contrast_model",
            }
        )
        return p, 0.0, metadata
    if model != "hardware_counts":
        raise ValueError("--sample-model must be hardware_counts, noisy_contrast, or ideal.")

    source = str(hardware_probability_source).strip().lower()
    if source == "mitigated_points":
        p, se, shots = _probability_from_points(
            load_csv(run_dir / "amplification_points.csv"),
            int(grover_power),
        )
    elif source == "raw_counts":
        p, se, shots = _probability_from_counts(
            load_csv(run_dir / "amplification_counts.csv"),
            int(grover_power),
        )
    else:
        raise ValueError("--hardware-probability-source must be mitigated_points or raw_counts.")
    return p, se, {
        "sample_model": "bernoulli_hardware_good_state",
        "sampling_probability_source": source,
        "sampling_grover_power": int(grover_power),
        "sampling_probability_shots": int(shots),
        "sampling_probability_se": float(se),
    }


def generate_monte_carlo_rows(
    *,
    budgets: list[int],
    repetitions: int,
    seed: int,
    a_true: float,
    a_sampling: float,
    objective_ry_offset: float,
    probability_se: float,
    probability_mode: str,
    probability_se_scale: float,
    metadata: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_budgets = sorted({int(budget) for budget in budgets if int(budget) > 0})
    if not ordered_budgets:
        raise ValueError("At least one positive budget is required.")
    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for rep in range(int(repetitions)):
        rng = np.random.default_rng(int(seed) + 7919 * rep)
        if str(probability_mode) == "normal":
            rep_probability = float(
                np.clip(
                    rng.normal(float(a_sampling), float(probability_se_scale) * float(probability_se)),
                    0.0,
                    1.0,
                )
            )
        elif str(probability_mode) == "fixed":
            rep_probability = float(a_sampling)
        else:
            raise ValueError("--probability-mode must be fixed or normal.")

        successes = 0
        previous_budget = 0
        start = time.perf_counter()
        rep_rows: list[dict[str, Any]] = []
        for step_index, budget in enumerate(ordered_budgets):
            increment = int(budget - previous_budget)
            if increment > 0:
                successes += int(rng.binomial(increment, rep_probability))
                previous_budget = int(budget)
            estimate = float(successes / float(budget))
            abs_error = abs(estimate - float(a_true))
            normalized_abs_error = abs_error / max(float(a_true), 1e-12)
            elapsed = time.perf_counter() - start
            row = {
                "run_kind": "hardware_classical_monte_carlo",
                "repetition": int(rep),
                "algorithm": ALGORITHM_LABEL,
                "algorithm_key": ALGORITHM_KEY,
                "step_index": int(step_index),
                "budget": int(budget),
                "query_budget": float(budget),
                "query_budget_actual": float(budget),
                "estimate": estimate,
                "abs_error": abs_error,
                "normalized_abs_error": normalized_abs_error,
                "normalized_sq_error": normalized_abs_error**2,
                "grover_power": 0,
                "k_max_budget": 0,
                "amplification_factor": 1,
                "a_true": float(a_true),
                "a_sampling": float(a_sampling),
                "a_sampling_repetition": float(rep_probability),
                "noise_bias": float(a_sampling) - float(a_true),
                "objective_ry_offset": float(objective_ry_offset),
                "runtime_wall_seconds": float(elapsed),
                "time_to_budget_seconds": float(elapsed),
                "classical_samples": int(budget),
                "successes": int(successes),
                **dict(metadata),
            }
            rows.append(row)
            rep_rows.append(row)
        final = dict(rep_rows[-1])
        final_rows.append(
            {
                "run_kind": final["run_kind"],
                "repetition": int(rep),
                "algorithm": ALGORITHM_LABEL,
                "algorithm_key": ALGORITHM_KEY,
                "a_true": float(a_true),
                "a_sampling": float(a_sampling),
                "a_sampling_repetition": float(rep_probability),
                "objective_ry_offset": float(objective_ry_offset),
                "final_queries": float(final["query_budget_actual"]),
                "final_estimate": float(final["estimate"]),
                "final_abs_error": float(final["abs_error"]),
                "final_normalized_abs_error": float(final["normalized_abs_error"]),
                "runtime_wall_seconds": float(final["runtime_wall_seconds"]),
                **dict(metadata),
            }
        )
    return rows, final_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a classical Monte Carlo baseline for the beta hardware replay "
            "experiment. It can sample the empirical hardware k=0 distribution, the "
            "calibrated contrast model, or the ideal amplitude."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-budget-rows", default="montecarlo_budget_rows.csv")
    parser.add_argument("--output-final-rows", default="montecarlo_final_rows.csv")
    parser.add_argument("--output-config", default="montecarlo_config.json")
    parser.add_argument(
        "--sample-model",
        choices=("hardware_counts", "noisy_contrast", "ideal"),
        default="hardware_counts",
    )
    parser.add_argument(
        "--hardware-probability-source",
        choices=("mitigated_points", "raw_counts"),
        default="mitigated_points",
    )
    parser.add_argument("--sampling-grover-power", type=int, default=0)
    parser.add_argument("--noise-amplification-factor", type=float, default=1.0)
    parser.add_argument("--probability-mode", choices=("fixed", "normal"), default="fixed")
    parser.add_argument("--probability-se-scale", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = _load_json(run_dir / "config.json")
    calibration_summary = _load_json(run_dir / "calibration_summary.json")
    if "a_true" not in config:
        raise ValueError(f"{run_dir / 'config.json'} is missing a_true.")

    budgets = parse_int_list(args.budgets) if args.budgets else parse_int_list(config.get("budgets", []))
    repetitions = int(args.repetitions if args.repetitions is not None else config.get("replay_repetitions", 1))
    seed = int(args.seed if args.seed is not None else config.get("seed", 12345))
    a_sampling, probability_se, metadata = resolve_sampling_probability(
        run_dir=run_dir,
        config=config,
        calibration_summary=calibration_summary,
        sample_model=str(args.sample_model),
        hardware_probability_source=str(args.hardware_probability_source),
        grover_power=int(args.sampling_grover_power),
        amplification_factor=float(args.noise_amplification_factor),
    )
    rows, final_rows = generate_monte_carlo_rows(
        budgets=budgets,
        repetitions=repetitions,
        seed=seed,
        a_true=float(config["a_true"]),
        a_sampling=float(a_sampling),
        objective_ry_offset=float(config.get("objective_ry_offset", np.nan)),
        probability_se=float(probability_se),
        probability_mode=str(args.probability_mode),
        probability_se_scale=float(args.probability_se_scale),
        metadata=metadata,
    )

    budget_path = run_dir / str(args.output_budget_rows)
    final_path = run_dir / str(args.output_final_rows)
    config_path = run_dir / str(args.output_config)
    save_csv(rows, budget_path)
    save_csv(final_rows, final_path)
    save_json(
        {
            "run_kind": "hardware_classical_monte_carlo",
            "algorithm_key": ALGORITHM_KEY,
            "run_dir": str(run_dir),
            "budget_rows_csv": str(budget_path),
            "final_rows_csv": str(final_path),
            "repetitions": repetitions,
            "budgets": budgets,
            "seed": seed,
            "a_true": float(config["a_true"]),
            "a_sampling": float(a_sampling),
            "probability_mode": str(args.probability_mode),
            "probability_se_scale": float(args.probability_se_scale),
            **metadata,
        },
        config_path,
    )
    print(f"Wrote {len(rows)} Monte Carlo budget rows to {budget_path}")
    print(f"Wrote {len(final_rows)} Monte Carlo final rows to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
