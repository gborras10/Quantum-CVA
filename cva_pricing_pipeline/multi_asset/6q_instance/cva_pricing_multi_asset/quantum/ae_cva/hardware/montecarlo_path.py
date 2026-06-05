from __future__ import annotations

import runpy
import sys
from pathlib import Path


current_file: Path = Path(__file__).resolve()
script_dir: Path = current_file.parent
repo_root: Path = next(
    parent for parent in current_file.parents if (parent / "pyproject.toml").exists()
)
src_path: Path = repo_root / "src"
instance_path: Path = (
    repo_root / "cva_pricing_pipeline" / "multi_asset" / "6q_instance"
)

for import_path in (src_path, instance_path, script_dir):
    import_path_text: str = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

if __name__ == "__main__":
    noisy_monte_carlo_script: Path = (
        script_dir.parent / "noisy_simulation" / "montecarlo_path.py"
    )
    noisy_script_dir_text: str = str(noisy_monte_carlo_script.parent)
    if noisy_script_dir_text not in sys.path:
        sys.path.insert(0, noisy_script_dir_text)

    has_run_dir_arg: bool = any(
        arg == "--run-dir" or arg.startswith("--run-dir=") for arg in sys.argv[1:]
    )
    if not has_run_dir_arg:
        sys.argv.extend(["--run-dir", str(script_dir / "experiment_results")])

    has_sample_model_arg: bool = any(
        arg == "--sample-model" or arg.startswith("--sample-model=") for arg in sys.argv[1:]
    )
    if not has_sample_model_arg:
        sys.argv.extend(["--sample-model", "hardware_counts"])

    has_probability_mode_arg: bool = any(
        arg == "--probability-mode" or arg.startswith("--probability-mode=")
        for arg in sys.argv[1:]
    )
    if not has_probability_mode_arg:
        sys.argv.extend(["--probability-mode", "normal"])

    runpy.run_path(str(noisy_monte_carlo_script), run_name="__main__")
