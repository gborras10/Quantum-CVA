from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOY_DIR = Path(__file__).resolve().parents[2]
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from ae_final_error_plots import plot_final_error_scatter
from plot_hardware import append_optional_monte_carlo, plot_replay_actual_queries


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
            "runtime_wall_seconds",
            "normalized_abs_error",
            "step_index",
        ),
        path=trace_path,
    )

    df = trace.copy()
    for name in ("query_budget", "runtime_wall_seconds", "normalized_abs_error", "step_index"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df[
        np.isfinite(df["query_budget"])
        & np.isfinite(df["runtime_wall_seconds"])
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


def rebuild_plots(
    run_dir: Path,
    *,
    include_monte_carlo: bool = False,
    monte_carlo_budget_rows: Path | None = None,
    monte_carlo_final_rows: Path | None = None,
) -> None:
    run_dir = run_dir.expanduser().resolve()
    trace_path = run_dir / "replay_trace_rows.csv"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_replay_actual_queries(
        run_dir,
        out_dir,
        output_stem="hardware_replay_budget",
        include_monte_carlo=include_monte_carlo,
        monte_carlo_budget_rows=monte_carlo_budget_rows,
    )

    final_rows = final_rows_from_trace(trace_path)
    final_rows = append_optional_monte_carlo(
        final_rows,
        monte_carlo_final_rows or (run_dir / "montecarlo_final_rows.csv"),
        include_monte_carlo=include_monte_carlo,
    )
    plot_final_error_scatter(
        final_rows,
        out_dir / "triple_gaussian_error_queries.png",
        x_kind="queries",
        title="",
        summary_path=out_dir / "triple_gaussian_error_queries_summary.csv",
        pdf_path=out_dir / "triple_gaussian_error_queries.pdf",
        draw_query_median_lines=False,
        draw_query_scaling_guides=False,
    )
    plot_final_error_scatter(
        final_rows,
        out_dir / "triple_gaussian_error_runtime.png",
        x_kind="runtime",
        title="",
        summary_path=out_dir / "triple_gaussian_error_runtime_summary.csv",
        pdf_path=out_dir / "triple_gaussian_error_runtime.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild replay budget and final-error plots from replay_trace_rows.csv."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing replay_trace_rows.csv.")
    parser.add_argument(
        "--include-monte-carlo",
        action="store_true",
        help="Append montecarlo_budget_rows.csv and montecarlo_final_rows.csv to replay plots.",
    )
    parser.add_argument("--monte-carlo-budget-rows", type=Path, default=None)
    parser.add_argument("--monte-carlo-final-rows", type=Path, default=None)
    args = parser.parse_args()

    rebuild_plots(
        Path(args.run_dir),
        include_monte_carlo=args.include_monte_carlo,
        monte_carlo_budget_rows=args.monte_carlo_budget_rows,
        monte_carlo_final_rows=args.monte_carlo_final_rows,
    )
    print(f"Rebuilt replay trace plots in: {Path(args.run_dir).expanduser().resolve() / 'plots'}")


if __name__ == "__main__":
    main()
