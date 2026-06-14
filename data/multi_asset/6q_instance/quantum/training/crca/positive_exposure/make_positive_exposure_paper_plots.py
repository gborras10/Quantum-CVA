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
            data_path=BASE_DIR / "training_heavy_hex_star.npz",
            objective_label=r"$\mathcal{L}_v$",
            trained_label=r"$F_{\phi_v^*}(x)$",
            target_label=r"$\tilde{v}(x)$",
            cost_stem="cost_evol_statevector",
            histogram_stem="histogram_statevector",
            combined_stem="positive_exposure_training_and_fit",
        )
    )


if __name__ == "__main__":
    main()
