from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from toys.amplitude_estimation_experiments.ideal_regime.ideal_utils import (  # noqa: E402
    aggregate_budget_summary,
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
    save_csv,
)


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------

# False -> legacy large_realistic budget/final rows
# True  -> v2 final_estimations flow
USE_V2_RESULTS = False

GENERATE_ERROR_VS_QUERIES = True
GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER = True
GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER = True

CONNECT_ERROR_PLOT_POINTS = True

ALGORITHMS_TO_PLOT = (
    "bae",
    "biqae",
    "cabiqae_latentt",
)

QUERY_MAX_BINS = 14
QUERY_MIN_POINTS_PER_BIN = 45
QUERY_BOOTSTRAP_SAMPLES = 2000
QUERY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
QUERY_BOOTSTRAP_SEED = 12345
QUERY_MAX_PLOT_POINTS_PER_ALGORITHM = 20
DROP_BBINNED_POINT_INDICES: dict[str, tuple[int, ...]] = {
    "BAE": (0,2),
    "BIQAE": (1,),
}

# Coarser v2 bins because there is one final estimation per run.
V2_QUERY_MAX_BINS = 8
V2_QUERY_MIN_POINTS_PER_BIN = 5

SCATTER_MAX_POINTS_PER_ALGORITHM: int | None = None
SCATTER_POINT_SAMPLE_SEED = 12345


ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE_latentt",
}
ALGORITHM_STYLES = {
    "bae": {"color": "#E07A5F", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
}

DATASETS: dict[str, dict[str, Any]] = {
    "large_realistic": {
        "results_dir": CURRENT_DIR / "experiment_results" / "plots",
        "budget_rows_csv": CURRENT_DIR / "experiment_results" / "large_realistic_budget_rows.csv",
        "output_prefix": "large_realistic",
        "algorithms": ("bae", "biqae", "cabiqae_latentt"),
        "title": "Simulated noise comparison: BAE vs BIQAE vs CABIQAE_latentt",
    }
}

V2_DATASETS: dict[str, dict[str, Any]] = {
    "large_realistic_v2": {
        "results_dir": CURRENT_DIR / "lambda1.0_experiment_results" / "plots_v2",
        "final_rows_csv": CURRENT_DIR / "lambda1.0_experiment_results" / "large_realistic_v2_final_estimations.csv",
        "output_prefix": "large_realistic_v2",
        "algorithms": ("bae", "biqae", "cabiqae_latentt"),
        "title": "Simulated noise v2: matched-query final estimates",
    }
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_repetitions(rows: list[dict[str, Any]]) -> int:
    repetitions: set[int] = set()
    for row in rows:
        raw = row.get("repetition", row.get("rep"))
        if raw in (None, ""):
            continue
        try:
            repetitions.add(int(float(raw)))
        except ValueError:
            continue
    return len(repetitions)


def algorithms_to_plot_for(available_algorithms: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(algorithm for algorithm in available_algorithms if algorithm in ALGORITHMS_TO_PLOT)
    if not selected:
        raise ValueError("ALGORITHMS_TO_PLOT does not overlap with dataset algorithms")
    return selected


def algorithm_labels_for(algorithms: tuple[str, ...]) -> dict[str, str]:
    return {algorithm: ALGORITHM_LABELS[algorithm] for algorithm in algorithms}


def algorithm_styles_for(algorithms: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {algorithm: ALGORITHM_STYLES[algorithm] for algorithm in algorithms}


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _legacy_rows_to_budget_rows(
    rows: list[dict[str, Any]],
    *,
    algorithms: tuple[str, ...],
) -> list[dict[str, Any]]:
    budget_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        algorithm_label = str(row.get("algorithm", "")).strip()
        algorithm_key = next((key for key, label in ALGORITHM_LABELS.items() if label == algorithm_label), "")
        if algorithm_key not in algorithms:
            continue

        query_budget = _as_float(row.get("query_budget_actual", row.get("query_budget", row.get("budget"))))
        rel_error = _as_float(row.get("normalized_abs_error", row.get("nrmse")))
        abs_error = _as_float(row.get("abs_error"))
        estimate = _as_float(row.get("estimate"))
        a_true = _as_float(row.get("a_true"))
        runtime_seconds = _as_float(row.get("runtime_seconds", row.get("time_to_budget_seconds")))

        if not np.isfinite(query_budget) or query_budget <= 0.0:
            continue
        if not np.isfinite(rel_error) or rel_error <= 0.0:
            continue

        rep_raw = row.get("rep", row.get("repetition"))
        try:
            repetition = int(float(rep_raw)) if rep_raw not in (None, "") else idx
        except (TypeError, ValueError):
            repetition = idx

        budget_rows.append(
            {
                "run_kind": "simulated_noise",
                "repetition": repetition,
                "algorithm": ALGORITHM_LABELS[algorithm_key],
                "algorithm_key": algorithm_key,
                "budget": int(round(query_budget)),
                "query_budget_actual": float(query_budget),
                "estimate": float(estimate) if np.isfinite(estimate) else np.nan,
                "abs_error": float(abs_error) if np.isfinite(abs_error) else np.nan,
                "normalized_abs_error": float(rel_error),
                "normalized_sq_error": float(rel_error**2),
                "a_true": float(a_true) if np.isfinite(a_true) else np.nan,
                "runtime_wall_seconds": float(runtime_seconds) if np.isfinite(runtime_seconds) else np.nan,
            }
        )
    return budget_rows


def _v2_rows_to_budget_rows(
    rows: list[dict[str, Any]],
    *,
    algorithms: tuple[str, ...],
) -> list[dict[str, Any]]:
    budget_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        algorithm_key = str(row.get("algorithm_key", "")).strip().lower()
        if algorithm_key not in algorithms:
            continue

        query_budget = _as_float(row.get("num_queries"))
        abs_error = _as_float(row.get("abs_error"))
        estimate = _as_float(row.get("final_estimate"))
        a_true = _as_float(row.get("a_true"))
        normalized_abs_error = _as_float(row.get("normalized_abs_error"))
        if not np.isfinite(normalized_abs_error):
            try:
                normalized_abs_error = float(abs_error / a_true)
            except ZeroDivisionError:
                normalized_abs_error = float("nan")
        runtime_seconds = _as_float(row.get("elapsed_runtime_seconds"))
        epsilon_target = _as_float(row.get("epsilon_target"))

        if not np.isfinite(query_budget) or query_budget <= 0.0:
            continue
        if not np.isfinite(normalized_abs_error) or normalized_abs_error <= 0.0:
            continue

        rep_raw = row.get("repetition")
        scenario_raw = row.get("scenario_id")
        try:
            rep_value = int(float(rep_raw)) if rep_raw not in (None, "") else -1
        except (TypeError, ValueError):
            rep_value = -1
        try:
            scenario_value = int(float(scenario_raw)) if scenario_raw not in (None, "") else 0
        except (TypeError, ValueError):
            scenario_value = 0

        phase = str(row.get("phase", "")).strip().lower()
        phase_tag = 1 if "iterative" in phase else 2
        eps_rank = int(round(epsilon_target * 1_000_000_000.0)) if np.isfinite(epsilon_target) else 0
        repetition_uid = int(scenario_value * 1_000_000 + rep_value * 10000 + phase_tag * 1000 + eps_rank + idx)

        budget_rows.append(
            {
                "run_kind": "simulated_noise_v2",
                "repetition": repetition_uid,
                "algorithm": ALGORITHM_LABELS.get(algorithm_key, algorithm_key.upper()),
                "algorithm_key": algorithm_key,
                "budget": int(round(query_budget)),
                "query_budget_actual": float(query_budget),
                "estimate": float(estimate) if np.isfinite(estimate) else np.nan,
                "abs_error": float(abs_error) if np.isfinite(abs_error) else np.nan,
                "normalized_abs_error": float(normalized_abs_error),
                "normalized_sq_error": float(normalized_abs_error**2),
                "a_true": float(a_true) if np.isfinite(a_true) else np.nan,
                "runtime_wall_seconds": float(runtime_seconds) if np.isfinite(runtime_seconds) else np.nan,
                "epsilon_target": float(epsilon_target) if np.isfinite(epsilon_target) else np.nan,
                "phase": phase,
                "step_index": eps_rank,
            }
        )
    return budget_rows


def regenerate_dataset(dataset_key: str) -> None:
    config = DATASETS[dataset_key]
    rows = load_rows(Path(config["budget_rows_csv"]))
    algorithms = algorithms_to_plot_for(tuple(config["algorithms"]))
    labels = algorithm_labels_for(algorithms)
    styles = algorithm_styles_for(algorithms)
    results_dir = Path(config["results_dir"])
    output_prefix = str(config["output_prefix"])

    budget_rows = _legacy_rows_to_budget_rows(rows, algorithms=algorithms)
    total_repetitions = count_repetitions(budget_rows)

    if GENERATE_ERROR_VS_QUERIES:
        budget_summary = aggregate_budget_summary(
            budget_rows,
            total_repetitions=total_repetitions,
            max_bins=QUERY_MAX_BINS,
            min_points_per_bin=QUERY_MIN_POINTS_PER_BIN,
            bootstrap_samples=QUERY_BOOTSTRAP_SAMPLES,
            confidence_level=QUERY_BOOTSTRAP_CONFIDENCE_LEVEL,
            bootstrap_seed=QUERY_BOOTSTRAP_SEED,
        )
        summary_path = results_dir / f"{output_prefix}_budget_summary.csv"
        output_path = results_dir / f"{output_prefix}_rmse.png"
        save_csv(budget_summary, summary_path)
        plot_budget_summary(
            budget_summary,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=str(config["title"]),
            max_points_per_algorithm=QUERY_MAX_PLOT_POINTS_PER_ALGORITHM,
            connect_points=CONNECT_ERROR_PLOT_POINTS,
            drop_binned_point_indices=DROP_BBINNED_POINT_INDICES,
        )
        print(f"[{dataset_key}] saved error-vs-queries plot: {output_path}")
        print(f"[{dataset_key}] saved error-vs-queries summary: {summary_path}")

    if GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_runtime_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_runtime_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=f"Final error versus runtime: {dataset_key}",
            summary_path=summary_path,
            max_points_per_algorithm=SCATTER_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=SCATTER_POINT_SAMPLE_SEED,
            x_kind="runtime",
        )
        print(f"[{dataset_key}] saved runtime scatter: {output_path}")

    if GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_queries_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_queries_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=f"Final error versus query count: {dataset_key}",
            summary_path=summary_path,
            max_points_per_algorithm=SCATTER_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=SCATTER_POINT_SAMPLE_SEED,
            x_kind="queries",
        )
        print(f"[{dataset_key}] saved query scatter: {output_path}")


def regenerate_v2_dataset(dataset_key: str) -> None:
    config = V2_DATASETS[dataset_key]
    rows = load_rows(Path(config["final_rows_csv"]))
    algorithms = algorithms_to_plot_for(tuple(config["algorithms"]))
    labels = algorithm_labels_for(algorithms)
    styles = algorithm_styles_for(algorithms)
    results_dir = Path(config["results_dir"])
    output_prefix = str(config["output_prefix"])

    budget_rows = _v2_rows_to_budget_rows(rows, algorithms=algorithms)
    total_repetitions = count_repetitions(budget_rows)

    proxy_rows_path = results_dir / f"{output_prefix}_budget_rows_proxy.csv"
    save_csv(budget_rows, proxy_rows_path)
    print(f"[{dataset_key}] saved v2 proxy budget rows: {proxy_rows_path}")

    if GENERATE_ERROR_VS_QUERIES:
        budget_summary = aggregate_budget_summary(
            budget_rows,
            total_repetitions=total_repetitions,
            max_bins=V2_QUERY_MAX_BINS,
            min_points_per_bin=V2_QUERY_MIN_POINTS_PER_BIN,
            bootstrap_samples=QUERY_BOOTSTRAP_SAMPLES,
            confidence_level=QUERY_BOOTSTRAP_CONFIDENCE_LEVEL,
            bootstrap_seed=QUERY_BOOTSTRAP_SEED,
        )
        summary_path = results_dir / f"{output_prefix}_budget_summary.csv"
        output_path = results_dir / f"{output_prefix}_rmse.png"
        save_csv(budget_summary, summary_path)
        if budget_summary:
            plot_budget_summary(
                budget_summary,
                algorithms=algorithms,
                algorithm_labels=labels,
                algorithm_styles=styles,
                output_path=output_path,
                title=str(config["title"]),
                max_points_per_algorithm=QUERY_MAX_PLOT_POINTS_PER_ALGORITHM,
                connect_points=CONNECT_ERROR_PLOT_POINTS,
                drop_binned_point_indices=DROP_BBINNED_POINT_INDICES,
            )
            print(f"[{dataset_key}] saved v2 rmse plot: {output_path}")
        else:
            print(f"[{dataset_key}] skipped v2 rmse plot: no valid v2 aggregated bins")
        print(f"[{dataset_key}] saved v2 rmse summary: {summary_path}")

    if GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_runtime_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_runtime_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=f"Final error versus runtime: {dataset_key}",
            summary_path=summary_path,
            max_points_per_algorithm=SCATTER_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=SCATTER_POINT_SAMPLE_SEED,
            x_kind="runtime",
        )
        print(f"[{dataset_key}] saved v2 runtime scatter: {output_path}")

    if GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_queries_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_queries_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=f"Final error versus query count: {dataset_key}",
            summary_path=summary_path,
            max_points_per_algorithm=SCATTER_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=SCATTER_POINT_SAMPLE_SEED,
            x_kind="queries",
        )
        print(f"[{dataset_key}] saved v2 query scatter: {output_path}")


def main() -> None:
    if USE_V2_RESULTS:
        for dataset_key in V2_DATASETS:
            regenerate_v2_dataset(dataset_key)
        return

    for dataset_key in DATASETS:
        regenerate_dataset(dataset_key)


if __name__ == "__main__":
    main()