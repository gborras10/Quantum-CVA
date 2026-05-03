from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
TOY_DIR = CURRENT_DIR.parent
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from ae_actual_query_plots import plot_actual_query_error  # noqa: E402
from ae_final_error_plots import plot_final_error_scatter  # noqa: E402


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------

GENERATE_ERROR_VS_QUERIES = True
GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER = True
GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER = True

# Directorio de entrada (inputs)
INPUT_DIR = CURRENT_DIR / "experiment_results"

# Directorio de salida (outputs)
RESULTS_DIR = INPUT_DIR / "plots"

BUDGET_ROWS_CSV = INPUT_DIR / "large_realistic_budget_rows.csv"
FINAL_ROWS_CSV = INPUT_DIR / "large_realistic_final_rows.csv"

# Binned median normalized absolute error vs actual queries.
QUERY_MAX_BINS = 12
QUERY_MIN_POINTS_PER_BIN = 50
QUERY_BOOTSTRAP_SAMPLES = 2000
QUERY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
QUERY_BOOTSTRAP_SEED = 12345

# Drop binned points after sorting by actual query cost.
# BAE index 0 removes the ~15-query warm-up point.
# BIQAE index 1 removes the second plotted point visible as an outlier.
QUERY_DROP_BINNED_POINT_INDICES = {
    "BAE": (0, 2),
    "BIQAE": (1,),
}

# Final scatter plots with Gaussian contours.
# Use None to plot all 8 * 25 final points per algorithm. Use an integer to subsample.
SCATTER_MAX_POINTS_PER_ALGORITHM: int | None = None
SCATTER_POINT_SAMPLE_SEED = 12345

# Output names.
ERROR_VS_QUERIES_PNG = RESULTS_DIR / "large_realistic_actual_queries.png"
ERROR_VS_QUERIES_SUMMARY_CSV = RESULTS_DIR / "large_realistic_budget_summary.csv"
FINAL_SCATTER_PREFIX = "large_realistic_final_error"


def main() -> None:
    if GENERATE_ERROR_VS_QUERIES:
        budget_rows = pd.read_csv(BUDGET_ROWS_CSV)
        plot_actual_query_error(
            budget_rows,
            ERROR_VS_QUERIES_PNG,
            summary_path=ERROR_VS_QUERIES_SUMMARY_CSV,
            pdf_path=ERROR_VS_QUERIES_PNG.with_suffix(".pdf"),
            max_bins=QUERY_MAX_BINS,
            min_points_per_bin=QUERY_MIN_POINTS_PER_BIN,
            bootstrap_samples=QUERY_BOOTSTRAP_SAMPLES,
            confidence_level=QUERY_BOOTSTRAP_CONFIDENCE_LEVEL,
            bootstrap_seed=QUERY_BOOTSTRAP_SEED,
            drop_binned_point_indices=QUERY_DROP_BINNED_POINT_INDICES,
        )
        print(f"Saved error-vs-queries plot: {ERROR_VS_QUERIES_PNG}")
        print(f"Saved error-vs-queries summary: {ERROR_VS_QUERIES_SUMMARY_CSV}")

    if GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER or GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER:
        final_rows = pd.read_csv(FINAL_ROWS_CSV)
        if GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER:
            query_plot = RESULTS_DIR / f"{FINAL_SCATTER_PREFIX}_queries.png"
            query_summary = RESULTS_DIR / f"{FINAL_SCATTER_PREFIX}_queries_summary.csv"
            plot_final_error_scatter(
                final_rows,
                query_plot,
                x_kind="queries",
                title="Final error versus query count under large realistic noise",
                summary_path=query_summary,
                pdf_path=query_plot.with_suffix(".pdf"),
                max_points_per_algorithm=SCATTER_MAX_POINTS_PER_ALGORITHM,
                point_sample_seed=SCATTER_POINT_SAMPLE_SEED,
                draw_query_median_lines=False,
                draw_query_scaling_guides=False,
            )
            print(f"Saved final query scatter: {query_plot}")
            print(f"Saved final query scatter summary: {query_summary}")
        if GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER:
            runtime_plot = RESULTS_DIR / f"{FINAL_SCATTER_PREFIX}_runtime.png"
            runtime_summary = RESULTS_DIR / f"{FINAL_SCATTER_PREFIX}_runtime_summary.csv"
            plot_final_error_scatter(
                final_rows,
                runtime_plot,
                x_kind="runtime",
                title="Final error versus runtime under large realistic noise",
                summary_path=runtime_summary,
                pdf_path=runtime_plot.with_suffix(".pdf"),
                max_points_per_algorithm=SCATTER_MAX_POINTS_PER_ALGORITHM,
                point_sample_seed=SCATTER_POINT_SAMPLE_SEED,
                draw_query_median_lines=False,
                draw_query_scaling_guides=False,
            )
            print(f"Saved final runtime scatter: {runtime_plot}")
            print(f"Saved final runtime scatter summary: {runtime_summary}")


if __name__ == "__main__":
    main()
