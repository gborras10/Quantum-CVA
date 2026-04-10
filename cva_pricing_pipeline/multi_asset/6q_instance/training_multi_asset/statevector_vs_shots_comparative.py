from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from comparative_utils import (
    BACKEND_NAME,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_REPETITIONS,
    DEFAULT_SHOTS_GRID,
    collect_qcbm_family_results,
    discover_qcbm_checkpoints,
    make_summary_tables,
    plot_depth_and_2q,
    plot_excess_over_ideal,
    plot_kl_vs_layers,
    plot_kl_vs_shots_by_layer,
    plot_noise_floor_ratio,
    plot_scatter_ideal_vs_noisy,
    repo_root_from_script,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch post-training comparison for a family of QCBM checkpoints (CSV + figures only).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/multi_asset/6q_instance/quantum/training/qcbm/statevector",
        help="Directory containing training_qcbm_*.npz checkpoints.",
    )
    parser.add_argument(
        "--backend-name",
        type=str,
        default=BACKEND_NAME,
        help="IBM backend name used to build the Aer noise model.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Number of repetitions per shot count.",
    )
    parser.add_argument(
        "--shots-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHOTS_GRID),
        help="List of shot counts to evaluate.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="Bootstrap resamples for mean confidence intervals.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs/qcbm_family_comparison",
        help="Output directory relative to the repository root.",
    )
    args = parser.parse_args()

    script_path = pathlib.Path(__file__)
    repo_root = repo_root_from_script(script_path)
    checkpoint_dir = repo_root / args.checkpoint_dir
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_qcbm_checkpoints(checkpoint_dir)
    raw_df, inventory_df, backend_info = collect_qcbm_family_results(
        checkpoints,
        shots_grid=list(args.shots_grid),
        repetitions=int(args.repetitions),
        backend_name=str(args.backend_name),
    )
    summary_df, fits_df, paired_df, max_summary_df = make_summary_tables(
        raw_df,
        shots_grid=list(args.shots_grid),
        bootstrap_reps=int(args.bootstrap_reps),
    )

    backend_info_df = pd.DataFrame([backend_info])
    statevector_table = (
        raw_df[raw_df["scenario"] == "statevector"][
            ["checkpoint_name", "checkpoint_file", "n_layers", "kl"]
        ]
        .rename(columns={"kl": "exact_statevector_kl"})
        .sort_values(["n_layers", "checkpoint_name"])
    )

    raw_df.to_csv(output_dir / "raw_qcbm_family_results.csv", index=False)
    summary_df.to_csv(output_dir / "summary_by_layer_shots.csv", index=False)
    max_summary_df.to_csv(output_dir / "summary_at_max_shots.csv", index=False)
    fits_df.to_csv(output_dir / "fit_summary.csv", index=False)
    paired_df.to_csv(output_dir / "paired_comparison_max_shots.csv", index=False)
    inventory_df.to_csv(output_dir / "checkpoint_inventory.csv", index=False)
    statevector_table.to_csv(output_dir / "statevector_reference.csv", index=False)
    backend_info_df.to_csv(output_dir / "backend_info.csv", index=False)

    plot_kl_vs_layers(max_summary_df, output_dir / "kl_vs_layers.png")
    plot_excess_over_ideal(max_summary_df, output_dir / "excess_over_ideal_vs_layers.png")
    plot_noise_floor_ratio(fits_df, output_dir / "noise_floor_ratio_vs_layers.png")
    plot_scatter_ideal_vs_noisy(max_summary_df, output_dir / "ideal_vs_noisy_scatter.png")
    plot_kl_vs_shots_by_layer(summary_df, fits_df, output_dir / "kl_vs_shots_by_layer.png")
    plot_depth_and_2q(inventory_df, output_dir / "depth_and_2q_vs_layers.png")

    print("\n=== QCBM family comparison generated successfully ===")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {output_dir}")
    print("Generated CSV files:")
    print("  - raw_qcbm_family_results.csv")
    print("  - summary_by_layer_shots.csv")
    print("  - summary_at_max_shots.csv")
    print("  - fit_summary.csv")
    print("  - paired_comparison_max_shots.csv")
    print("  - checkpoint_inventory.csv")
    print("  - statevector_reference.csv")
    print("  - backend_info.csv")
    print("Generated figures:")
    print("  - kl_vs_layers.png")
    print("  - excess_over_ideal_vs_layers.png")
    print("  - noise_floor_ratio_vs_layers.png")
    print("  - ideal_vs_noisy_scatter.png")
    print("  - kl_vs_shots_by_layer.png")
    print("  - depth_and_2q_vs_layers.png")


if __name__ == "__main__":
    main()
