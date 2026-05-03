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
from plot_hardware import plot_replay_actual_queries


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


def rebuild_plots(run_dir: Path) -> None:
    run_dir = run_dir.expanduser().resolve()
    trace_path = run_dir / "replay_trace_rows.csv"
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_replay_actual_queries(run_dir, out_dir, output_stem="hardware_replay_budget")

    final_rows = final_rows_from_trace(trace_path)
    plot_final_error_scatter(
        final_rows,
        out_dir / "triple_gaussian_error_queries.png",
        x_kind="queries",
        title="Final error versus query cost under hardware replay",
        summary_path=out_dir / "triple_gaussian_error_queries_summary.csv",
        pdf_path=out_dir / "triple_gaussian_error_queries.pdf",
    )
    plot_final_error_scatter(
        final_rows,
        out_dir / "triple_gaussian_error_runtime.png",
        x_kind="runtime",
        title="Final error versus runtime under hardware replay",
        summary_path=out_dir / "triple_gaussian_error_runtime_summary.csv",
        pdf_path=out_dir / "triple_gaussian_error_runtime.pdf",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild replay budget and final-error plots from replay_trace_rows.csv."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing replay_trace_rows.csv.")
    args = parser.parse_args()

    rebuild_plots(Path(args.run_dir))
    print(f"Rebuilt replay trace plots in: {Path(args.run_dir).expanduser().resolve() / 'plots'}")


if __name__ == "__main__":
    main()
