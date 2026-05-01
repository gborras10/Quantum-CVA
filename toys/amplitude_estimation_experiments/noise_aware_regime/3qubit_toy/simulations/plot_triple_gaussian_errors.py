from __future__ import annotations

import argparse
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
TOY_DIR = CURRENT_DIR.parent
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from ae_final_error_plots import plot_final_error_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final-error scatter plots versus queries and runtime."
    )
    parser.add_argument(
        "--final-rows",
        type=Path,
        default=CURRENT_DIR / "experiment_results" / "final_rows.csv",
        help="CSV with one final row per algorithm repetition.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "experiment_results",
        help="Directory where plots and summaries will be written.",
    )
    parser.add_argument(
        "--title-suffix",
        default="under realistic noise",
        help="Text appended to the plot titles.",
    )
    args = parser.parse_args()

    plot_final_error_figures(
        args.final_rows.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        title_suffix=args.title_suffix,
    )
    print(f"Saved final-error plots in: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
