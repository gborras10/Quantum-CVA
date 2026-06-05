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
#                       Run hardware CVA AE experiment
# ======================================================================
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (  # noqa: E402
    HardwareCvaExperimentRunner,
)


DEFAULT_RUN_ARGUMENTS: list[str] = [
    "--mode",
    "hardware",
    "--hardware-executor",
    "qctrl",
    "--instance-name",
    "premium_new_usa",
    "--backend-name",
    "ibm_pittsburgh",
    "--max-grover-power",
    "4",
    "--scan-grover-powers",
    "0,1,2,3,4",
    "--scan-repeats",
    "2",
    "--scan-shots",
    "4096",
    "--readout-shots",
    "8192",
    "--direct-shots",
    "128",
    "--max-direct-calls",
    "4",
    "--skip-direct",
    "--algorithms",
    "cabiqae,biqae",
    "--budgets",
    "128,256,512,768,1024,1536,2048,3072,4096,6144,8192,10000,15000,20000,30000,50000,75000,100000,150000,200000",
    "--epsilon-target",
    "0.08",
    "--cabiqae-epsilon-target",
    "0.05",
    "--biqae-epsilon-target",
    "0.002",
    "--replay-repetitions",
    "100",
    "--replay-max-calls",
    "4096",
    "--replay-probability-mode",
    "normal",
    "--replay-probability-se-scale",
    "1.0",
    "--extrapolate",
    "true",
    "--noise-floor",
    "fit",
    "--cap-kappa",
    "3.0",
    "--cabiqae-hard-k-cap",
    "--session-max-time",
    "24h",
    "--soft-wallclock-limit",
    "315360000",
    "--optimization-level",
    "3",
    "--seed-transpiler",
    "1234",
    "--layout-search-strategy",
    "exhaustive",
    "--reference-ks",
    "0,1,2,3,4",
    "--seed",
    "12345",
    "--no-use-fractional-gates",
    "--verbose",
]


if __name__ == "__main__":
    experiment_runner: HardwareCvaExperimentRunner = (
        HardwareCvaExperimentRunner.from_cli(DEFAULT_RUN_ARGUMENTS + sys.argv[1:])
    )
    experiment_runner.run()
