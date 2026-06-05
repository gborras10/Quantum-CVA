"""Command-line coordinator for hardware CVA plots.

`plot_hwd_CVA.py` delegates here.  This module does not draw figures itself; it
routes the requested plot kind to the standard diagnostics module or the
publication-style calibration module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_calibration_plots import (
    plot_calibration_paper,
)
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_plots import (
    plot_hardware_run,
)


def build_arg_parser() -> argparse.ArgumentParser:
    # One CLI replaces the old split between `plot_hardware_cva_ae.py` and
    # `plot_calibration_paper.py`.
    parser = argparse.ArgumentParser(
        description="Plot 6q CVA hardware AE artifacts from an existing run directory."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--kind",
        choices=("standard", "calibration-paper", "all"),
        default="all",
    )
    parser.add_argument(
        "--max-queries",
        type=float,
        default=None,
        help="Only include replay rows at or below this actual query cost.",
    )
    parser.add_argument("--include-monte-carlo", action="store_true")
    parser.add_argument("--monte-carlo-budget-rows", type=Path, default=None)
    parser.add_argument(
        "--paper-output-dir",
        default=None,
        help="Directory for publication-style calibration figures.",
    )
    parser.add_argument(
        "--require-counts",
        action="store_true",
        help="Raise an error if amplification_counts.csv is missing.",
    )
    return parser


class HardwareCvaPlotCli:
    """Small coordinator for standard and publication-style hardware plots."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "HardwareCvaPlotCli":
        args: argparse.Namespace = build_arg_parser().parse_args(argv)
        return cls(args)

    def run(self) -> None:
        args: argparse.Namespace = self.args
        # Standard plots are broad diagnostics for a completed run.
        if args.kind in {"standard", "all"}:
            plots_dir: Path = plot_hardware_run(
                args.run_dir,
                max_queries=args.max_queries,
                include_monte_carlo=bool(args.include_monte_carlo),
                monte_carlo_budget_rows=args.monte_carlo_budget_rows,
            )
            print(f"Plots saved in: {plots_dir}")

        # Calibration-paper plots are narrower and format the same artifacts for
        # publication figures.
        if args.kind in {"calibration-paper", "all"}:
            paper_dir: Path = plot_calibration_paper(
                args.run_dir,
                output_dir=args.paper_output_dir,
                require_counts=bool(args.require_counts),
            )
            print(f"Paper calibration plots saved in: {paper_dir}")


def run_cli(argv: list[str] | None = None) -> None:
    cli: HardwareCvaPlotCli = HardwareCvaPlotCli.from_cli(argv)
    cli.run()


def main(argv: list[str] | None = None) -> None:
    run_cli(argv)


if __name__ == "__main__":
    main()
