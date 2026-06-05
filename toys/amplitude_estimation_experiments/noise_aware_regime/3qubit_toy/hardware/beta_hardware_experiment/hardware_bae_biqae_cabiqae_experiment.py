"""Thin compatibility launcher for the old hardware AE experiment."""

from quantum_cva.amplitude_estimation.experiments.legacy_launchers import run_hardware


if __name__ == "__main__":
    run_hardware()

