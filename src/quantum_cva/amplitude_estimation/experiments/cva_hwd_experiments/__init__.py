"""Hardware CVA amplitude-estimation experiment package.

This subpackage contains the reusable implementation behind the thin launchers
in `cva_pricing_pipeline/.../quantum/ae_cva/hardware`.

The intended split is:
- `cva_hardware_runner`: orchestrates preflight, dry-run, live hardware,
  top-up, recovery, direct execution, and replay.
- `cva_hardware_common`: shared path bootstrapping, config loading, parsing,
  and CSV field ordering helpers.
- `cva_hardware_plots`: standard diagnostic plots for completed runs.
- `cva_hardware_calibration_plots`: publication-style calibration figures.
- `cva_hardware_plot_cli`: small CLI coordinator for plot modes.
"""
