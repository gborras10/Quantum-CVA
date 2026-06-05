from __future__ import annotations

import pathlib
import sys


BASE_DIR = pathlib.Path(__file__).resolve().parent
CRCA_DIR = BASE_DIR.parent
sys.path.insert(0, str(CRCA_DIR))

from crca_paper_plots import CrcaPaperPlotConfig, generate_crca_paper_plots


def main() -> None:
    generate_crca_paper_plots(
        CrcaPaperPlotConfig(
            output_dir=BASE_DIR,
            data_path=BASE_DIR / "training_crca2.npz",
            objective_label=r"$\mathcal{L}_q$",
            trained_label=r"$F_{\phi_q}(x)$",
            target_label=r"$q(x)$",
            combined_stem="default_probabilities_training_and_fit",
        )
    )


if __name__ == "__main__":
    main()
