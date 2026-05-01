from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
TOY_DIR = CURRENT_DIR.parent
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from ae_actual_query_plots import plot_actual_query_error


def default_input_path(output_dir: Path) -> Path:
    candidates = (
        output_dir / "trace_rows.csv",
        output_dir / "large_realistic_trace_rows.csv",
        output_dir.parent / "large_realistic_trace_rows.csv",
        output_dir / "large_realistic_budget_rows.csv",
        output_dir.parent / "large_realistic_budget_rows.csv",
        output_dir / "budget_rows.csv",
    )
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return output_dir / "budget_rows.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the query-error plot with the actual-query visual policy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Trace rows preferred; budget rows are accepted as a fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "experiment_results",
        help="Directory where rmse_vs_queries.png will be written.",
    )
    parser.add_argument("--max-bins", type=int, default=12)
    parser.add_argument("--min-points-per-bin", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    input_path = (
        args.input.expanduser().resolve()
        if args.input is not None
        else default_input_path(output_dir).expanduser().resolve()
    )
    rows = pd.read_csv(input_path)
    if args.input is None and not {"query_budget", "query_budget_actual", "final_queries"}.intersection(rows.columns):
        raise ValueError(
            "The default input only has nominal budget columns. "
            "Rerun simulated_noise_experiment.py to create trace rows, "
            "or pass an explicit --input file with actual query costs."
        )
    plot_actual_query_error(
        rows,
        output_dir / "rmse_vs_queries.png",
        summary_path=output_dir / "rmse_vs_queries_summary.csv",
        pdf_path=output_dir / "rmse_vs_queries.pdf",
        max_bins=int(args.max_bins),
        min_points_per_bin=int(args.min_points_per_bin),
    )
    print(f"Saved query-error plot: {output_dir / 'rmse_vs_queries.png'}")
    print(f"Source data: {input_path}")


if __name__ == "__main__":
    main()
