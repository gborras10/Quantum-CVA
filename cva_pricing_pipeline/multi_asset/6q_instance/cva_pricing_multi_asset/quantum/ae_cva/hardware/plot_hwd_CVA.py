from __future__ import annotations

import sys
from pathlib import Path


# ======================================================================
#                       Project import paths
# ======================================================================
# Plot generation is separated from the experiment runner.  All plotting
# implementation is reusable code under src/quantum_cva.
current_file: Path = Path(__file__).resolve()
repo_root: Path = next(
    parent for parent in current_file.parents if (parent / "pyproject.toml").exists()
)
src_path: Path = repo_root / "src"
instance_path: Path = (
    repo_root / "cva_pricing_pipeline" / "multi_asset" / "6q_instance"
)

for import_path in (src_path, instance_path):
    import_path_text: str = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)


# ======================================================================
#                       Plot existing hardware run
# ======================================================================
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_plot_cli import (
    HardwareCvaPlotCli,
)


if __name__ == "__main__":
    plot_cli: HardwareCvaPlotCli = HardwareCvaPlotCli.from_cli()
    plot_cli.run()
