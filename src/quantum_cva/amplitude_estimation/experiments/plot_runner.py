from __future__ import annotations

from dataclasses import dataclass

from quantum_cva.amplitude_estimation.experiments.configs import PlotConfig
from quantum_cva.amplitude_estimation.experiments.io import RunPaths, load_csv
from quantum_cva.amplitude_estimation.experiments.plotting import (
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
)


@dataclass(slots=True)
class AePlotRunner:
    config: PlotConfig

    def run(self) -> None:
        paths = RunPaths(self.config.run_dir)
        budget_rows = load_csv(paths.replay_budget) if paths.replay_budget.exists() else []
        summary_rows = (
            load_csv(paths.budget_summary) if paths.budget_summary.exists() else []
        )
        paths.plots_dir.mkdir(parents=True, exist_ok=True)
        if summary_rows:
            plot_budget_summary(
                summary_rows,
                output_path=paths.plots_dir / f"{self.config.prefix}_budget_summary.png",
            )
        if budget_rows and self.config.include_final_scatter:
            plot_final_runtime_scatter_from_budget_rows(
                budget_rows,
                output_path=paths.plots_dir / f"{self.config.prefix}_final_error_runtime.png",
                summary_path=paths.plots_dir
                / f"{self.config.prefix}_final_error_runtime_summary.csv",
                x_kind="runtime",
            )
            plot_final_runtime_scatter_from_budget_rows(
                budget_rows,
                output_path=paths.plots_dir / f"{self.config.prefix}_final_error_queries.png",
                summary_path=paths.plots_dir
                / f"{self.config.prefix}_final_error_queries_summary.csv",
                x_kind="queries",
            )
