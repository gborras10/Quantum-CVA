from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[2]
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

# Toggle plotting mode.
# False -> legacy budget-row plots.
# True  -> plots from v2 final_estimations CSV.
USE_V2_RESULTS = False

# Choose which datasets to regenerate. Options: "ideal", "elf".
GENERATE_DATASET_KEYS = ("ideal",)

# Choose which v2 datasets to regenerate. Options: "ideal_v2".
GENERATE_V2_DATASET_KEYS = ("ideal_v2",)

# Choose which plots to regenerate for the selected datasets.
GENERATE_ERROR_VS_QUERIES = True
GENERATE_RUNTIME_GAUSSIAN_SCATTER = True
GENERATE_QUERY_GAUSSIAN_SCATTER = True
INCLUDE_MONTE_CARLO = True #! False to generate runtime plots!!!!

# Whether the error-vs-queries plots should connect the plotted medians.
CONNECT_ERROR_PLOT_POINTS = True

# Only algorithms in this list will be plotted.
# Keep the order if you want the legend order to match this tuple.
ALGORITHMS_TO_PLOT = (
    "cabiqae_latentt",
    "biqae",
    "bae",
    "classical_mc",
)

# Error-vs-queries binned benchmark plot settings.
QUERY_MAX_BINS = 20
QUERY_MIN_POINTS_PER_BIN = 30
QUERY_BOOTSTRAP_SAMPLES = 10000
QUERY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.90
QUERY_BOOTSTRAP_SEED = 12345
QUERY_MAX_PLOT_POINTS_PER_ALGORITHM = 12

# Scatter with Gaussian contours settings.
# Use None to plot all final points per algorithm. Use an integer to subsample.
GAUSSIAN_MAX_POINTS_PER_ALGORITHM: int | None = None
GAUSSIAN_POINT_SAMPLE_SEED = 12345

# v2 plot settings
GENERATE_V2_ERROR_VS_EPSILON = True
GENERATE_V2_RUNTIME_VS_EPSILON = True
GENERATE_V2_ERROR_VS_QUERIES_SCATTER = True
GENERATE_V2_ERROR_VS_RUNTIME_SCATTER = True
# Coarser binning for v2 because the CSV contains one final estimate per run.
V2_QUERY_MAX_BINS = 10
V2_QUERY_MIN_POINTS_PER_BIN = 10


ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "iqae": "IQAE",
    "cabiqae_latentt": "CABIQAE",
    "elf_qae": "ELF-QAE",
    "classical_mc": "DCS",
}
ALGORITHM_STYLES = {
    "bae": {"color": "#E07A5F", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "iqae": {"color": "#4C78A8", "marker": "D"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
    "elf_qae": {"color": "#6D597A", "marker": "v"},
    "classical_mc": {"color": "#2A9D8F", "marker": "X"},
}

DATASETS: dict[str, dict[str, Any]] = {
    "ideal": {
        "results_dir": CURRENT_DIR / "experiment_results" / "plots",
        "budget_rows_csv": CURRENT_DIR
        / "experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_ideal_budget_rows.csv",
        "monte_carlo_budget_rows_csv": CURRENT_DIR
        / "experiment_results"
        / "classical_mc_ideal_budget_rows.csv",
        "output_prefix": "bae_biqae_iqae_cabiqae_latentt_ideal",
        "algorithms": ("bae", "biqae", "iqae", "cabiqae_latentt", "classical_mc"),
        "title": "Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE",
    },
    "elf": {
        "results_dir": CURRENT_DIR / "elf_experiment_results" / "plots",
        "budget_rows_csv": CURRENT_DIR
        / "elf_experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal_budget_rows.csv",
        "monte_carlo_budget_rows_csv": CURRENT_DIR
        / "elf_experiment_results"
        / "classical_mc_ideal_budget_rows.csv",
        "output_prefix": "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal",
        "algorithms": ("bae", "biqae", "iqae", "cabiqae_latentt", "elf_qae", "classical_mc"),
        "title": "Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE vs ELF-QAE",
    },
}

V2_DATASETS: dict[str, dict[str, Any]] = {
    "ideal_v2": {
        "results_dir": CURRENT_DIR / "experiment_results" / "plots_v2",
        "final_rows_csv": CURRENT_DIR
        / "experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_ideal_v2_final_estimations.csv",
        "monte_carlo_budget_rows_csv": CURRENT_DIR
        / "experiment_results"
        / "classical_mc_ideal_v2_budget_rows.csv",
        "output_prefix": "bae_biqae_iqae_cabiqae_latentt_ideal_v2",
        "algorithms": ("bae", "biqae", "iqae", "cabiqae_latentt", "classical_mc"),
        "title": "Ideal regime v2 comparison: matched-query final estimates",
    }
}


def load_budget_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_final_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_repetitions(rows: list[dict[str, Any]]) -> int:
    repetitions: set[int] = set()
    for row in rows:
        raw = row.get("repetition")
        if raw in (None, ""):
            continue
        try:
            repetitions.add(int(float(raw)))
        except ValueError:
            continue
    return len(repetitions)


def algorithm_labels_for(algorithms: tuple[str, ...]) -> dict[str, str]:
    return {algorithm: ALGORITHM_LABELS[algorithm] for algorithm in algorithms}


def algorithm_styles_for(algorithms: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {algorithm: ALGORITHM_STYLES[algorithm] for algorithm in algorithms}


def maybe_append_monte_carlo_rows(
    rows: list[dict[str, Any]],
    monte_carlo_path: Path | None,
) -> list[dict[str, Any]]:
    if not INCLUDE_MONTE_CARLO:
        return rows
    if monte_carlo_path is None:
        return rows
    if not monte_carlo_path.exists():
        raise FileNotFoundError(
            f"Monte Carlo rows requested but not found: {monte_carlo_path}. "
            "Run toys/amplitude_estimation_experiments/ideal_regime/montecarlo_path.py first."
        )
    return [*rows, *load_budget_rows(monte_carlo_path)]


def algorithms_to_plot_for(available_algorithms: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(algorithm for algorithm in available_algorithms if algorithm in ALGORITHMS_TO_PLOT)
    if not selected:
        raise ValueError(
            "ALGORITHMS_TO_PLOT does not overlap with the algorithms available in this dataset"
        )
    return selected


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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
        rel_error = _as_float(row.get("rel_error"))
        abs_error = _as_float(row.get("abs_error"))
        estimate = _as_float(row.get("final_estimate"))
        a_true = _as_float(row.get("a_true"))
        runtime_seconds = _as_float(row.get("elapsed_runtime_seconds"))
        objective_ry_offset = _as_float(row.get("objective_ry_offset"))
        epsilon_target = _as_float(row.get("epsilon_target"))

        if not np.isfinite(query_budget) or query_budget <= 0.0:
            continue
        if not np.isfinite(rel_error) or rel_error <= 0.0:
            continue

        rep_raw = row.get("repetition")
        try:
            rep_value = int(float(rep_raw)) if rep_raw not in (None, "") else -1
        except (TypeError, ValueError):
            rep_value = -1

        phase = str(row.get("phase", "")).strip().lower()
        phase_tag = 1 if "iterative" in phase else 2
        eps_rank = int(round(epsilon_target * 1_000_000_000.0)) if np.isfinite(epsilon_target) else 0
        repetition_uid = int(rep_value * 100000 + phase_tag * 10000 + eps_rank + idx)

        budget_rows.append(
            {
                "run_kind": "ideal_simulation_v2",
                "repetition": repetition_uid,
                "algorithm": ALGORITHM_LABELS.get(algorithm_key, algorithm_key.upper()),
                "algorithm_key": algorithm_key,
                "budget": int(round(query_budget)),
                "query_budget_actual": float(query_budget),
                "estimate": float(estimate) if np.isfinite(estimate) else np.nan,
                "abs_error": float(abs_error) if np.isfinite(abs_error) else np.nan,
                "normalized_abs_error": float(rel_error),
                "normalized_sq_error": float(rel_error**2),
                "a_true": float(a_true) if np.isfinite(a_true) else np.nan,
                "objective_ry_offset": float(objective_ry_offset)
                if np.isfinite(objective_ry_offset)
                else np.nan,
                "runtime_wall_seconds": float(runtime_seconds)
                if np.isfinite(runtime_seconds)
                else np.nan,
                "epsilon_target": float(epsilon_target) if np.isfinite(epsilon_target) else np.nan,
                "phase": phase,
                "step_index": eps_rank,
            }
        )
    return budget_rows


def regenerate_dataset(dataset_key: str) -> None:
    config = DATASETS[dataset_key]
    rows = load_budget_rows(config["budget_rows_csv"])
    rows = maybe_append_monte_carlo_rows(
        rows,
        Path(config["monte_carlo_budget_rows_csv"])
        if config.get("monte_carlo_budget_rows_csv")
        else None,
    )
    total_repetitions = count_repetitions(rows)
    algorithms = algorithms_to_plot_for(tuple(config["algorithms"]))
    labels = algorithm_labels_for(algorithms)
    styles = algorithm_styles_for(algorithms)
    results_dir = Path(config["results_dir"])
    output_prefix = str(config["output_prefix"])

    if GENERATE_ERROR_VS_QUERIES:
        budget_summary = aggregate_budget_summary(
            rows,
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
        )
        print(f"[{dataset_key}] saved error-vs-queries plot: {output_path}")
        print(f"[{dataset_key}] saved error-vs-queries summary: {summary_path}")

    if GENERATE_RUNTIME_GAUSSIAN_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_runtime_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_runtime_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title="",
            summary_path=summary_path,
            max_points_per_algorithm=GAUSSIAN_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=GAUSSIAN_POINT_SAMPLE_SEED,
            x_kind="runtime",
        )
        print(f"[{dataset_key}] saved runtime scatter: {output_path}")
        print(f"[{dataset_key}] saved runtime scatter summary: {summary_path}")

    if GENERATE_QUERY_GAUSSIAN_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_queries_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_queries_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title="",
            summary_path=summary_path,
            max_points_per_algorithm=GAUSSIAN_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=GAUSSIAN_POINT_SAMPLE_SEED,
            x_kind="queries",
        )
        print(f"[{dataset_key}] saved query scatter: {output_path}")
        print(f"[{dataset_key}] saved query scatter summary: {summary_path}")


def regenerate_v2_dataset(dataset_key: str) -> None:
    config = V2_DATASETS[dataset_key]
    rows = load_final_rows(Path(config["final_rows_csv"]))
    algorithms = algorithms_to_plot_for(tuple(config["algorithms"]))
    labels = algorithm_labels_for(algorithms)
    styles = algorithm_styles_for(algorithms)
    results_dir = Path(config["results_dir"])
    output_prefix = str(config["output_prefix"])
    title = str(config["title"])
    budget_rows = _v2_rows_to_budget_rows(rows, algorithms=algorithms)
    budget_rows = maybe_append_monte_carlo_rows(
        budget_rows,
        Path(config["monte_carlo_budget_rows_csv"])
        if config.get("monte_carlo_budget_rows_csv")
        else None,
    )
    total_repetitions = count_repetitions(budget_rows)

    proxy_rows_path = results_dir / f"{output_prefix}_budget_rows_proxy.csv"
    save_csv(budget_rows, proxy_rows_path)
    print(f"[{dataset_key}] saved v2 proxy budget rows: {proxy_rows_path}")

    if GENERATE_V2_ERROR_VS_EPSILON:
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
                title=title,
                max_points_per_algorithm=QUERY_MAX_PLOT_POINTS_PER_ALGORITHM,
                connect_points=CONNECT_ERROR_PLOT_POINTS,
            )
            print(f"[{dataset_key}] saved v2 rmse plot: {output_path}")
        else:
            print(f"[{dataset_key}] skipped v2 rmse plot: no valid v2 aggregated bins")
        print(f"[{dataset_key}] saved v2 rmse summary: {summary_path}")
        print(f"[{dataset_key}] v2 rmse source csv: {config['final_rows_csv']}")

    if GENERATE_V2_RUNTIME_VS_EPSILON:
        output_path = results_dir / f"{output_prefix}_final_error_runtime_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_runtime_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=f"Final error versus runtime: {dataset_key} comparison",
            summary_path=summary_path,
            max_points_per_algorithm=GAUSSIAN_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=GAUSSIAN_POINT_SAMPLE_SEED,
            x_kind="runtime",
        )
        print(f"[{dataset_key}] saved v2 runtime scatter: {output_path}")
        print(f"[{dataset_key}] saved v2 runtime scatter summary: {summary_path}")

    if GENERATE_V2_ERROR_VS_QUERIES_SCATTER:
        output_path = results_dir / f"{output_prefix}_final_error_queries_scatter.png"
        summary_path = results_dir / f"{output_prefix}_final_error_queries_scatter_summary.csv"
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            algorithms=algorithms,
            algorithm_labels=labels,
            algorithm_styles=styles,
            output_path=output_path,
            title=f"Final error versus query count: {dataset_key} comparison",
            summary_path=summary_path,
            max_points_per_algorithm=GAUSSIAN_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=GAUSSIAN_POINT_SAMPLE_SEED,
            x_kind="queries",
        )
        print(f"[{dataset_key}] saved v2 query scatter: {output_path}")
        print(f"[{dataset_key}] saved v2 query scatter summary: {summary_path}")

    if GENERATE_V2_ERROR_VS_RUNTIME_SCATTER:
        # Alias toggle kept for convenience: same output as runtime scatter above.
        if not GENERATE_V2_RUNTIME_VS_EPSILON:
            output_path = results_dir / f"{output_prefix}_final_error_runtime_scatter.png"
            summary_path = results_dir / f"{output_prefix}_final_error_runtime_scatter_summary.csv"
            plot_final_runtime_scatter_from_budget_rows(
                budget_rows,
                algorithms=algorithms,
                algorithm_labels=labels,
                algorithm_styles=styles,
                output_path=output_path,
                title=f"Final error versus runtime: {dataset_key} comparison",
                summary_path=summary_path,
                max_points_per_algorithm=GAUSSIAN_MAX_POINTS_PER_ALGORITHM,
                point_sample_seed=GAUSSIAN_POINT_SAMPLE_SEED,
                x_kind="runtime",
            )
            print(f"[{dataset_key}] saved v2 runtime scatter: {output_path}")
            print(f"[{dataset_key}] saved v2 runtime scatter summary: {summary_path}")


def main() -> None:
    if USE_V2_RESULTS:
        for dataset_key in GENERATE_V2_DATASET_KEYS:
            if dataset_key not in V2_DATASETS:
                raise KeyError(f"Unknown v2 dataset key: {dataset_key}")
            regenerate_v2_dataset(dataset_key)
        return

    for dataset_key in GENERATE_DATASET_KEYS:
        if dataset_key not in DATASETS:
            raise KeyError(f"Unknown dataset key: {dataset_key}")
        regenerate_dataset(dataset_key)


if __name__ == "__main__":
    main()
