from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.configs import PlotConfig
from quantum_cva.amplitude_estimation.experiments.plot_runner import AePlotRunner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate reusable AE plots.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prefix", default="ae")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = PlotConfig(run_dir=Path(args.run_dir), prefix=str(args.prefix))
    AePlotRunner(config).run()


if __name__ == "__main__":
    main()
