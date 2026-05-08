from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
TOY_DIR = CURRENT_DIR.parents[1]
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from ae_final_error_plots import plot_final_error_scatter  # noqa: E402
from hardware_replay_query_plot import (  # noqa: E402
    DEFAULT_BUDGETS,
    budget_rows_from_trace_rows,
    plot_hardware_replay_actual_queries,
)


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------

GENERATE_ERROR_VS_QUERIES = True
GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER = True
GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER = True

# Source data directory.
INPUT_DIR = CURRENT_DIR / "experiment_results" / "csv_results"
TRACE_ROWS_CSV = INPUT_DIR / "replay_trace_rows.csv"
REPLAY_BUDGET_ROWS_CSV = INPUT_DIR / "replay_budget_rows.csv"
FINAL_ROWS_CSV = INPUT_DIR / "replay_final_rows.csv"

# Output directory.
OUTPUT_DIR = CURRENT_DIR / "experiment_results" / "plots"

# Binned median normalized absolute error vs actual queries.
# These defaults intentionally match ideal_regime.ideal_utils.aggregate_budget_summary.
QUERY_MAX_BINS = 12
QUERY_MIN_POINTS_PER_BIN = 100
QUERY_BOOTSTRAP_SAMPLES = 2000
QUERY_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
QUERY_BOOTSTRAP_SEED = 12345
QUERY_MAX_QUERIES: float | None = None

# Final scatter plots with Gaussian contours.
# Use None to plot all final points per algorithm. Use an integer to subsample.
SCATTER_MAX_POINTS_PER_ALGORITHM: int | None = None
SCATTER_POINT_SAMPLE_SEED = 12345

# Output names.
ERROR_VS_QUERIES_PNG = OUTPUT_DIR / "hardware_replay_actual_queries.png"
ERROR_VS_QUERIES_SUMMARY_CSV = OUTPUT_DIR / "hardware_replay_actual_queries_summary.csv"
FINAL_SCATTER_PREFIX = "hardware_replay_final_error"


def _require_columns(df: pd.DataFrame, names: tuple[str, ...], *, path: Path) -> None:
    missing = [name for name in names if name not in df]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def final_rows_from_trace(trace_path: Path) -> pd.DataFrame:
    trace = pd.read_csv(trace_path)
    _require_columns(
        trace,
        (
            "algorithm",
            "repetition",
            "query_budget",
            "normalized_abs_error",
            "step_index",
        ),
        path=trace_path,
    )

    df = trace.copy()
    for name in ("query_budget", "normalized_abs_error", "step_index"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    if "runtime_wall_seconds" in df:
        df["runtime_wall_seconds"] = pd.to_numeric(df["runtime_wall_seconds"], errors="coerce")
    df = df[
        np.isfinite(df["query_budget"])
        & np.isfinite(df["normalized_abs_error"])
        & np.isfinite(df["step_index"])
    ]
    if df.empty:
        raise ValueError(f"{trace_path} has no finite replay trace rows")

    final = (
        df.sort_values(["algorithm", "repetition", "query_budget", "step_index"])
        .groupby(["algorithm", "repetition"], as_index=False)
        .tail(1)
        .copy()
    )
    final["final_queries"] = final["query_budget"]
    final["final_normalized_abs_error"] = final["normalized_abs_error"]
    return final


def load_final_rows() -> pd.DataFrame:
    if FINAL_ROWS_CSV.exists() and FINAL_ROWS_CSV.stat().st_size > 0:
        final = pd.read_csv(FINAL_ROWS_CSV)
        if "final_queries" in final and "final_normalized_abs_error" in final:
            return final
    return final_rows_from_trace(TRACE_ROWS_CSV)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build hardware replay plots.")
    parser.add_argument(
        "--max-queries",
        type=float,
        default=QUERY_MAX_QUERIES,
        help="Only plot error-vs-queries rows with query cost at or below this value.",
    )
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if GENERATE_ERROR_VS_QUERIES:
        if REPLAY_BUDGET_ROWS_CSV.exists() and REPLAY_BUDGET_ROWS_CSV.stat().st_size > 0:
            plot_rows = pd.read_csv(REPLAY_BUDGET_ROWS_CSV)
        else:
            plot_rows = budget_rows_from_trace_rows(pd.read_csv(TRACE_ROWS_CSV), DEFAULT_BUDGETS)
        plot_hardware_replay_actual_queries(
            plot_rows,
            ERROR_VS_QUERIES_PNG,
            summary_path=ERROR_VS_QUERIES_SUMMARY_CSV,
            max_queries=args.max_queries,
            max_bins=QUERY_MAX_BINS,
            min_points_per_bin=QUERY_MIN_POINTS_PER_BIN,
            bootstrap_samples=QUERY_BOOTSTRAP_SAMPLES,
            confidence_level=QUERY_BOOTSTRAP_CONFIDENCE_LEVEL,
            bootstrap_seed=QUERY_BOOTSTRAP_SEED,
        )
        print(f"Saved error-vs-queries plot: {ERROR_VS_QUERIES_PNG}")
        print(f"Saved error-vs-queries summary: {ERROR_VS_QUERIES_SUMMARY_CSV}")

    if GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER or GENERATE_FINAL_ERROR_VS_RUNTIME_SCATTER:
        final_rows = load_final_rows()
        if GENERATE_FINAL_ERROR_VS_QUERIES_SCATTER:
            query_plot = OUTPUT_DIR / f"{FINAL_SCATTER_PREFIX}_queries.png"
            query_summary = OUTPUT_DIR / f"{FINAL_SCATTER_PREFIX}_queries_summary.csv"
            plot_final_error_scatter(
                final_rows,
                query_plot,
                x_kind="queries",
                title="Final error versus query count under hardware replay",
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
            if "runtime_wall_seconds" not in final_rows and "runtime_seconds" not in final_rows:
                print("Skipped runtime scatter: final rows have no runtime column.")
            else:
                runtime_plot = OUTPUT_DIR / f"{FINAL_SCATTER_PREFIX}_runtime.png"
                runtime_summary = OUTPUT_DIR / f"{FINAL_SCATTER_PREFIX}_runtime_summary.csv"
                plot_final_error_scatter(
                    final_rows,
                    runtime_plot,
                    x_kind="runtime",
                    title="Final error versus runtime under hardware replay",
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
