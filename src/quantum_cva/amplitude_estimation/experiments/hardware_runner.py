from __future__ import annotations

from dataclasses import dataclass

from quantum_cva.amplitude_estimation.experiments.configs import HardwareReplayConfig
from quantum_cva.amplitude_estimation.experiments.hardware import (
    effective_contrast_prefactor_for_algorithms,
    effective_t_for_algorithms,
    load_existing_state,
    run_dry_run_experiment,
    run_replay,
)
from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle


@dataclass(slots=True)
class HardwareReplayRunner:
    config: HardwareReplayConfig
    bundle: AEProblemBundle

    def run(self) -> None:
        cfg = self.config
        if cfg.mode == "dry-run":
            run_dry_run_experiment(
                self.bundle,
                run_dir=cfg.run_dir,
                algorithms=cfg.algorithm.algorithms,
                budgets=cfg.budgets,
                max_grover_power=int(cfg.max_grover_power),
                direct_shots=int(cfg.direct_shots),
                replay_repetitions=int(cfg.replay_repetitions),
                epsilon_target=float(cfg.algorithm.epsilon_target),
                alpha=float(cfg.algorithm.alpha),
                seed=int(cfg.algorithm.seed),
                contrast_baseline=float(cfg.contrast_baseline),
            )
            return

        state = load_existing_state(cfg.run_dir)
        p_by_k = {
            int(row["grover_power"]): float(row["p_hw_mitigated"])
            for row in state.amplification_point_rows
        }
        p_se_by_k = {
            int(row["grover_power"]): float(row["p_hw_mitigated_se"])
            for row in state.amplification_point_rows
        }
        run_replay(
            state,
            self.bundle,
            algorithms=cfg.algorithm.algorithms,
            p_by_k=p_by_k,
            p_se_by_k=p_se_by_k,
            replay_probability_mode="normal",
            replay_probability_se_scale=1.0,
            budgets=cfg.budgets,
            repetitions=int(cfg.replay_repetitions),
            n_shots=int(cfg.direct_shots),
            epsilon_target=float(cfg.algorithm.epsilon_target),
            alpha=float(cfg.algorithm.alpha),
            t_eff=effective_t_for_algorithms(state.calibration_summary),
            seed=int(cfg.algorithm.seed),
            contrast_prefactor=effective_contrast_prefactor_for_algorithms(
                state.calibration_summary
            ),
        )

