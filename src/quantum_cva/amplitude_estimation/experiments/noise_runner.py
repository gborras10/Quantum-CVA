from __future__ import annotations

from dataclasses import dataclass

from quantum_cva.amplitude_estimation.experiments.configs import NoiseSimulationConfig
from quantum_cva.amplitude_estimation.experiments.hardware import run_dry_run_experiment
from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle


@dataclass(slots=True)
class NoiseSimulationRunner:
    config: NoiseSimulationConfig
    bundle: AEProblemBundle

    def run(self) -> None:
        cfg = self.config
        run_dry_run_experiment(
            self.bundle,
            run_dir=cfg.run_dir,
            algorithms=cfg.algorithm.algorithms,
            budgets=cfg.budgets,
            max_grover_power=int(cfg.max_grover_power),
            scan_repeats=int(cfg.scan_repeats),
            scan_shots=int(cfg.scan_shots),
            readout_shots=int(cfg.readout_shots),
            direct_shots=int(cfg.direct_shots),
            replay_repetitions=int(cfg.replay_repetitions),
            epsilon_target=float(cfg.algorithm.epsilon_target),
            alpha=float(cfg.algorithm.alpha),
            seed=int(cfg.algorithm.seed),
            noise_scale=float(cfg.noise_scale),
            noise_profile=str(cfg.noise_profile),
            contrast_baseline=float(cfg.contrast_baseline),
        )

