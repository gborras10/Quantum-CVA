from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
ALGORITHM_KEY = "classical_mc"
ALGORITHM_LABEL = "Classical MC"
DEFAULT_SEED = 12345

DEFAULT_LEGACY_DATASETS = (
    (
        CURRENT_DIR
        / "experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_ideal_budget_rows.csv",
        CURRENT_DIR / "experiment_results" / "classical_mc_ideal_budget_rows.csv",
    ),
    (
        CURRENT_DIR
        / "elf_experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal_budget_rows.csv",
        CURRENT_DIR / "elf_experiment_results" / "classical_mc_ideal_budget_rows.csv",
    ),
)
DEFAULT_V2_DATASET = (
    CURRENT_DIR
    / "experiment_results"
    / "bae_biqae_iqae_cabiqae_latentt_ideal_v2_final_estimations.csv",
    CURRENT_DIR / "experiment_results" / "classical_mc_ideal_v2_budget_rows.csv",
)


def _as_float(value: Any, default: float = np.nan) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _legacy_repetition_cases(
    source_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, float | int]]:
    cases_by_rep: dict[int, dict[str, float | int]] = {}
    for row in source_rows:
        repetition = _as_float(row.get("repetition"))
        a_true = _as_float(row.get("a_true"))
        objective_ry_offset = _as_float(row.get("objective_ry_offset"))
        if not np.isfinite(repetition) or not np.isfinite(a_true):
            continue
        rep = int(repetition)
        if rep not in cases_by_rep:
            cases_by_rep[rep] = {
                "repetition": rep,
                "a_true": float(a_true),
                "objective_ry_offset": float(objective_ry_offset)
                if np.isfinite(objective_ry_offset)
                else np.nan,
            }
    return [cases_by_rep[key] for key in sorted(cases_by_rep)]


def _legacy_budgets(source_rows: Sequence[Mapping[str, Any]]) -> list[int]:
    budgets: set[int] = set()
    for row in source_rows:
        budget = _as_float(row.get("budget"), _as_float(row.get("query_budget_actual")))
        if np.isfinite(budget) and budget > 0.0:
            budgets.add(int(round(float(budget))))
    if not budgets:
        raise ValueError("No positive budgets found in source CSV.")
    return sorted(budgets)


def _monte_carlo_rows_for_case(
    *,
    repetition: int,
    a_true: float,
    objective_ry_offset: float,
    budgets: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    total_successes = 0
    previous_budget = 0
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for step_index, budget in enumerate(sorted(set(int(value) for value in budgets))):
        if budget <= 0:
            continue
        increment = int(budget - previous_budget)
        if increment > 0:
            total_successes += int(rng.binomial(increment, float(a_true)))
            previous_budget = int(budget)
        estimate = float(total_successes / float(budget))
        abs_error = abs(estimate - float(a_true))
        normalized_abs_error = abs_error / max(float(a_true), 1e-12)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "run_kind": "ideal_classical_monte_carlo",
                "repetition": int(repetition),
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
                "a_sampling": float(a_true),
                "objective_ry_offset": float(objective_ry_offset),
                "runtime_wall_seconds": float(elapsed),
                "time_to_budget_seconds": float(elapsed),
                "classical_samples": int(budget),
                "successes": int(total_successes),
                "sample_model": "bernoulli_good_state",
            }
        )
    return rows


def generate_legacy_monte_carlo(
    *,
    source_csv: Path,
    output_csv: Path,
    seed: int,
) -> list[dict[str, Any]]:
    source_rows = _read_csv(source_csv)
    cases = _legacy_repetition_cases(source_rows)
    budgets = _legacy_budgets(source_rows)
    rows: list[dict[str, Any]] = []
    for case in cases:
        rep = int(case["repetition"])
        rows.extend(
            _monte_carlo_rows_for_case(
                repetition=rep,
                a_true=float(case["a_true"]),
                objective_ry_offset=float(case["objective_ry_offset"]),
                budgets=budgets,
                seed=int(seed) + 7919 * rep,
            )
        )
    _save_csv(rows, output_csv)
    return rows


def _v2_repetition_uid(row: Mapping[str, Any], idx: int) -> int:
    rep_value = _as_float(row.get("repetition"), -1.0)
    epsilon_target = _as_float(row.get("epsilon_target"), 0.0)
    phase = str(row.get("phase", "")).strip().lower()
    phase_tag = 1 if "iterative" in phase else 2
    eps_rank = (
        int(round(float(epsilon_target) * 1_000_000_000.0))
        if np.isfinite(epsilon_target)
        else 0
    )
    return int(int(rep_value) * 100000 + phase_tag * 10000 + eps_rank + idx)


def generate_v2_monte_carlo(
    *,
    source_csv: Path,
    output_csv: Path,
    seed: int,
) -> list[dict[str, Any]]:
    source_rows = _read_csv(source_csv)
    rows: list[dict[str, Any]] = []
    for idx, source in enumerate(source_rows):
        budget = _as_float(source.get("num_queries"))
        a_true = _as_float(source.get("a_true"))
        objective_ry_offset = _as_float(source.get("objective_ry_offset"))
        if not np.isfinite(budget) or budget <= 0.0 or not np.isfinite(a_true):
            continue
        repetition_uid = _v2_repetition_uid(source, idx)
        rows.extend(
            _monte_carlo_rows_for_case(
                repetition=repetition_uid,
                a_true=float(a_true),
                objective_ry_offset=float(objective_ry_offset)
                if np.isfinite(objective_ry_offset)
                else np.nan,
                budgets=[int(round(float(budget)))],
                seed=int(seed) + 7919 * repetition_uid,
            )
        )
        rows[-1]["epsilon_target"] = _as_float(source.get("epsilon_target"))
        rows[-1]["phase"] = str(source.get("phase", ""))
    _save_csv(rows, output_csv)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ideal classical Monte Carlo budget rows for the ideal AE toy "
            "experiments. No noise model is applied."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Generate only legacy budget-row Monte Carlo CSVs.",
    )
    parser.add_argument(
        "--v2-only",
        action="store_true",
        help="Generate only the v2 Monte Carlo proxy budget-row CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.legacy_only and args.v2_only:
        raise ValueError("--legacy-only and --v2-only are mutually exclusive.")

    if not args.v2_only:
        for source_csv, output_csv in DEFAULT_LEGACY_DATASETS:
            rows = generate_legacy_monte_carlo(
                source_csv=source_csv,
                output_csv=output_csv,
                seed=int(args.seed),
            )
            print(f"Wrote {len(rows)} ideal Monte Carlo rows to {output_csv}")

    if not args.legacy_only:
        source_csv, output_csv = DEFAULT_V2_DATASET
        rows = generate_v2_monte_carlo(
            source_csv=source_csv,
            output_csv=output_csv,
            seed=int(args.seed),
        )
        print(f"Wrote {len(rows)} ideal v2 Monte Carlo rows to {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
