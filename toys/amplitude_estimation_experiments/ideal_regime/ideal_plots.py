from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any


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

# Choose which datasets to regenerate. Options: "ideal", "elf".
GENERATE_DATASET_KEYS = ("ideal", "elf")

# Choose which plots to regenerate for the selected datasets.
GENERATE_ERROR_VS_QUERIES = True
GENERATE_RUNTIME_GAUSSIAN_SCATTER = True
GENERATE_QUERY_GAUSSIAN_SCATTER = True

# Error-vs-queries binned benchmark plot settings.
QUERY_MAX_BINS = 12
QUERY_MIN_POINTS_PER_BIN = 100
QUERY_BOOTSTRAP_SAMPLES = 2000
QUERY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
QUERY_BOOTSTRAP_SEED = 12345
QUERY_MAX_PLOT_POINTS_PER_ALGORITHM = 14

# Scatter with Gaussian contours settings.
# Use None to plot all final points per algorithm. Use an integer to subsample.
GAUSSIAN_MAX_POINTS_PER_ALGORITHM: int | None = None
GAUSSIAN_POINT_SAMPLE_SEED = 12345


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

DATASETS: dict[str, dict[str, Any]] = {
    "ideal": {
        "results_dir": CURRENT_DIR / "experiment_results" / "plots",
        "budget_rows_csv": CURRENT_DIR
        / "experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_ideal_budget_rows.csv",
        "output_prefix": "bae_biqae_iqae_cabiqae_latentt_ideal",
        "algorithms": ("bae", "biqae", "iqae", "cabiqae_latentt"),
        "title": "Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE",
    },
    "elf": {
        "results_dir": CURRENT_DIR / "elf_experiment_results" / "plots",
        "budget_rows_csv": CURRENT_DIR
        / "elf_experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal_budget_rows.csv",
        "output_prefix": "bae_biqae_iqae_cabiqae_latentt_elf_qae_ideal",
        "algorithms": ("bae", "biqae", "iqae", "cabiqae_latentt", "elf_qae"),
        "title": "Ideal regime comparison: BAE vs BIQAE vs IQAE vs CABIQAE vs ELF-QAE",
    },
}


def load_budget_rows(path: Path) -> list[dict[str, Any]]:
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


def regenerate_dataset(dataset_key: str) -> None:
    config = DATASETS[dataset_key]
    rows = load_budget_rows(config["budget_rows_csv"])
    total_repetitions = count_repetitions(rows)
    algorithms = tuple(config["algorithms"])
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
            title=f"Final error versus runtime: {dataset_key} comparison",
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
            title=f"Final error versus query count: {dataset_key} comparison",
            summary_path=summary_path,
            max_points_per_algorithm=GAUSSIAN_MAX_POINTS_PER_ALGORITHM,
            point_sample_seed=GAUSSIAN_POINT_SAMPLE_SEED,
            x_kind="queries",
        )
        print(f"[{dataset_key}] saved query scatter: {output_path}")
        print(f"[{dataset_key}] saved query scatter summary: {summary_path}")


def main() -> None:
    for dataset_key in GENERATE_DATASET_KEYS:
        if dataset_key not in DATASETS:
            raise KeyError(f"Unknown dataset key: {dataset_key}")
        regenerate_dataset(dataset_key)


if __name__ == "__main__":
    main()
