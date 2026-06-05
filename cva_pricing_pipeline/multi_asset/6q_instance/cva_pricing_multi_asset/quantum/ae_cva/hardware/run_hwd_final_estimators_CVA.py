from __future__ import annotations

import sys
from pathlib import Path


# ======================================================================
#                       Project import paths
# ======================================================================
# This file is intentionally only an experiment launcher.  The reusable
# implementation lives in quantum_cva.amplitude_estimation.experiments.
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
#                       Run final CABIQAE estimators
# ======================================================================
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_final_estimators import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main()
