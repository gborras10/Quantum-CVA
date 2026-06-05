from __future__ import annotations

import time
from dataclasses import dataclass

from quantum_cva.amplitude_estimation.experiments.configs import IdealExperimentConfig
from quantum_cva.amplitude_estimation.experiments.io import RunPaths, save_csv, save_json
from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle
from quantum_cva.amplitude_estimation.experiments.samplers import ContrastDecaySampler
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    run_algorithm_once,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (
    aggregate_budget_summary,
)
from quantum_cva.amplitude_estimation.experiments.traces import rows_at_budgets


@dataclass(slots=True)
class IdealExperimentRunner:
    config: IdealExperimentConfig
    bundle: AEProblemBundle

    def run(self) -> RunPaths:
        cfg = self.config
        paths = RunPaths(cfg.run_dir)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_config(paths)

        trace_rows: list[dict] = []
        final_rows: list[dict] = []
        error_rows: list[dict] = []
        budget_rows: list[dict] = []

        for rep in range(int(cfg.repetitions)):
            for alg_index, algorithm in enumerate(cfg.algorithm.algorithms):
                sampler = ContrastDecaySampler(
                    self.bundle,
                    T=None,
                    seed=int(cfg.algorithm.seed) + 1009 * rep + alg_index,
                )
                try:
                    rows, final = run_algorithm_once(
                        algorithm,
                        sampler,
                        self.bundle,
                        run_kind="ideal_simulation",
                        repetition=rep,
                        epsilon_target=float(cfg.algorithm.epsilon_target),
                        alpha=float(cfg.algorithm.alpha),
                        n_shots=int(cfg.n_shots),
                        max_queries=int(cfg.max_queries),
                        seed=int(cfg.algorithm.seed) + rep + alg_index,
                        algorithm_labels=ALGORITHM_LABELS,
                    )
                except Exception as exc:
                    error_rows.append(
                        {
                            "repetition": rep,
                            "algorithm": algorithm,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    continue
                trace_rows.extend(rows)
                final_rows.append(final)
                budget_rows.extend(
                    rows_at_budgets(rows, cfg.budgets, run_kind="ideal_simulation")
                )

        summary_rows = aggregate_budget_summary(
            budget_rows,
            total_repetitions=int(cfg.repetitions),
            group_by_budget=True,
        )
        save_csv(trace_rows, paths.direct_trace)
        save_csv(final_rows, paths.direct_final)
        save_csv(error_rows, paths.errors)
        save_csv(budget_rows, paths.replay_budget)
        save_csv(summary_rows, paths.budget_summary)
        paths.write_manifest()
        return paths

    def _write_config(self, paths: RunPaths) -> None:
        cfg = self.config
        save_json(
            {
                "mode": "ideal",
                "target_name": self.bundle.target_name,
                "a_true": float(self.bundle.true_amplitude),
                "processed_true_value": float(self.bundle.processed_true_value),
                "algorithms": list(cfg.algorithm.algorithms),
                "repetitions": int(cfg.repetitions),
                "epsilon_target": float(cfg.algorithm.epsilon_target),
                "alpha": float(cfg.algorithm.alpha),
                "n_shots": int(cfg.n_shots),
                "max_queries": int(cfg.max_queries),
                "budgets": list(cfg.budgets),
                "created_at_epoch": time.time(),
                "problem_metadata": self.bundle.metadata,
            },
            paths.config,
        )

