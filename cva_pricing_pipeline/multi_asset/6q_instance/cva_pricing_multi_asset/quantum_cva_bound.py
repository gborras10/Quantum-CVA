from __future__ import annotations

from pathlib import Path

import pandas as pd

from bound_utils import RegimeConfig, compute_bound, load_inputs, print_files_used


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

NOISELESS_RESULTS_DIR = (
    SCRIPT_DIR / "quantum" / "ae_cva" / "noiseless_simulation"
)
NOISE_RESULTS_DIR = (
    SCRIPT_DIR
    / "quantum"
    / "ae_cva"
    / "hardware"
    / "results"
    / "hardware_cva_ae_20260512_083742"
)

BENCHMARK_FILE = (
    REPO_ROOT
    / "data"
    / "multi_asset"
    / "6q_instance"
    / "benchmark"
    / "three_asset_instance.npz"
)

QCBM_STATEVECTOR_FILE = (
    REPO_ROOT
    / "data"
    / "multi_asset"
    / "6q_instance"
    / "quantum"
    / "training"
    / "qcbm"
    / "statevector"
    / "training_qcbm_heavyhex6_6lay.npz"
)
QCBM_NOISE_FILE = (
    REPO_ROOT
    / "data"
    / "multi_asset"
    / "6q_instance"
    / "quantum"
    / "training"
    / "qcbm"
    / "shots"
    / "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz"
)

CRCA_DIR = (
    REPO_ROOT
    / "data"
    / "multi_asset"
    / "6q_instance"
    / "quantum"
    / "training"
    / "crca"
)

CRCA_STATEVECTOR_FILES = {
    "v": CRCA_DIR / "positive_exposure" / "training_heavy_hex_star.npz",
    "p": CRCA_DIR / "discount_factors" / "training_crca2.npz",
    "q": CRCA_DIR / "default_probabilities" / "training_crca2.npz",
}
CRCA_NOISE_FILES = {
    "v": CRCA_DIR
    / "positive_exposure"
    / "training_heavy_hex_star_shots_backend_noise_snapshot.npz",
    "p": CRCA_DIR
    / "discount_factors"
    / "training_crca2_shots_backend_noise_snapshot.npz",
    "q": CRCA_DIR
    / "default_probabilities"
    / "training_crca2_shots_backend_noise_snapshot.npz",
}

SUMMARY_CSV = SCRIPT_DIR / "quantum_cva_bound_summary.csv"

REQUIRED_COLUMNS = [
    "regime",
    "selected_AE_algorithm",
    "CVA_MC_hat",
    "CVA_Delta",
    "CVA_SV_hat",
    "CVA_AE_hat",
    "Error_observed",
    "C_scale",
    "K_QCBM",
    "sqrt_2_K_QCBM",
    "delta_v",
    "delta_p",
    "delta_q",
    "Delta_AE",
    "B_MC_Delta",
    "B_QCBM",
    "B_rot",
    "B_AE",
    "Bound_total",
    "bound_check",
    "absolute_continuity_check",
    "num_support_violations",
    "target_mass_on_violations",
    "min_P_hat_on_target_support",
]

def main() -> None:
    configs = [
        RegimeConfig(
            regime="noiseless",
            results_dir=NOISELESS_RESULTS_DIR,
            qcbm_file=QCBM_STATEVECTOR_FILE,
            crca_files=CRCA_STATEVECTOR_FILES,
            qcbm_hat_key="p_star",
            crca_hat_keys={
                "v": ["f_star_statevector", "f_star"],
                "p": ["f_star_statevector", "f_star"],
                "q": ["f_star_statevector", "f_star"],
            },
            ae_algorithm_filter=None,
        ),
        RegimeConfig(
            regime="noise_hardware",
            results_dir=NOISE_RESULTS_DIR,
            qcbm_file=QCBM_NOISE_FILE,
            crca_files=CRCA_NOISE_FILES,
            qcbm_hat_key="p_star",
            crca_hat_keys={
                "v": ["f_star", "f_star_shots", "f_star_statevector"],
                "p": ["f_star_shots", "f_star", "f_star_statevector"],
                "q": ["f_star_shots", "f_star", "f_star_statevector"],
            },
            ae_algorithm_filter=("biqae", "cabiqae"),
        ),
    ]

    inputs_list = [load_inputs(config) for config in configs]
    rows = [compute_bound(inputs) for inputs in inputs_list]
    print_files_used(inputs_list, rows)

    visible = [{key: row[key] for key in REQUIRED_COLUMNS} for row in rows]
    summary = pd.DataFrame(visible, columns=REQUIRED_COLUMNS)
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\nSummary:")
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.float_format",
        "{:.12g}".format,
    ):
        print(summary.to_string(index=False))
    print(f"\nSaved CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
