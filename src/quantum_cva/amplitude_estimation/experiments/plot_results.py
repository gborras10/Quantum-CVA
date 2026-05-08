from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.io import RunPaths, load_csv
from quantum_cva.amplitude_estimation.experiments.plotting import (
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate reusable AE plots.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prefix", default="ae")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = RunPaths(Path(args.run_dir))
    summary_rows = load_csv(paths.budget_summary) if paths.budget_summary.exists() else []
    budget_rows = load_csv(paths.replay_budget) if paths.replay_budget.exists() else []
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        plot_budget_summary(
            summary_rows,
            output_path=paths.plots_dir / f"{args.prefix}_budget_summary.png",
        )
    if budget_rows:
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            output_path=paths.plots_dir / f"{args.prefix}_final_error_runtime.png",
            summary_path=paths.plots_dir / f"{args.prefix}_final_error_runtime_summary.csv",
            x_kind="runtime",
        )
        plot_final_runtime_scatter_from_budget_rows(
            budget_rows,
            output_path=paths.plots_dir / f"{args.prefix}_final_error_queries.png",
            summary_path=paths.plots_dir / f"{args.prefix}_final_error_queries_summary.csv",
            x_kind="queries",
        )


if __name__ == "__main__":
    main()
