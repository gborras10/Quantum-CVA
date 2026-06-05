"""Standard diagnostic plots for hardware CVA amplitude-estimation runs.

The input is a completed run directory with the CSV/JSON files written by
`cva_hardware_runner`.  The figures produced here are operational diagnostics:
amplification scan, direct live traces, replay budget curves, and final
query/runtime scatter plots.

Publication-style calibration plots live in `cva_hardware_calibration_plots`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from quantum_cva.amplitude_estimation.experiments.io import RunPaths, load_csv
from quantum_cva.amplitude_estimation.experiments.plotting import (
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
    save_figure_png_and_pdf,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    ALGORITHM_STYLES,
)

PAPER_FINAL_ALGORITHMS = ("cabiqae", "biqae", "classical_mc")
PLOT_ALGORITHM_LABELS = {
    **ALGORITHM_LABELS,
    "classical_mc": "DCS",
}
PLOT_ALGORITHM_STYLES = {
    **ALGORITHM_STYLES,
    "classical_mc": {"color": "#2A9D8F", "marker": "X"},
}
PAPER_STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    # Missing CSVs are valid for partial workflows, e.g. preflight-only runs.
    if not path.exists() or path.stat().st_size == 0:
        return []
    return load_csv(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _append_optional_rows(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    include: bool,
) -> list[dict[str, Any]]:
    if not include:
        return rows
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{path} does not exist. Run montecarlo_path.py first.")
    return [*rows, *_load_rows(path)]


def _summary_rows_with_optional_monte_carlo(
    paths: RunPaths,
    *,
    include_monte_carlo: bool,
) -> list[dict[str, Any]]:
    return _append_optional_rows(
        _load_rows(paths.budget_summary),
        paths.run_dir / "montecarlo_budget_summary.csv",
        include=include_monte_carlo,
    )


def _attach_classical_runtime_profile(
    rows: list[dict[str, Any]],
    profile_path: Path,
) -> list[dict[str, Any]]:
    if not profile_path.exists() or profile_path.stat().st_size == 0:
        return rows
    profile_rows = _load_rows(profile_path)
    profile_runtime = {
        (
            int(_as_float(row.get("repetition"), -1.0)),
            int(_as_float(row.get("budget"), -1.0)),
        ): _as_float(row.get("runtime_wall_seconds"))
        for row in profile_rows
        if _algorithm_key(row) == "classical_mc"
    }
    enriched: list[dict[str, Any]] = []
    for row in rows:
        if (
            _algorithm_key(row) == "classical_mc"
            and not np.isfinite(_as_float(row.get("runtime_wall_seconds")))
        ):
            key = (
                int(_as_float(row.get("repetition"), -1.0)),
                int(_as_float(row.get("budget"), -1.0)),
            )
            runtime = profile_runtime.get(key, np.nan)
            if np.isfinite(runtime) and runtime > 0.0:
                row = {
                    **row,
                    "runtime_wall_seconds": runtime,
                    "time_to_budget_seconds": runtime,
                }
        enriched.append(row)
    return enriched


def _algorithm_key(row: Mapping[str, Any]) -> str:
    return str(row.get("algorithm_key", row.get("algorithm", "")))


def _algorithm_label(row: Mapping[str, Any]) -> str:
    key = _algorithm_key(row)
    return ALGORITHM_LABELS.get(key, str(row.get("algorithm", key)))


def _style(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return ALGORITHM_STYLES.get(_algorithm_key(row), {"color": "#333333", "marker": "o"})


def plot_amplification(paths: RunPaths) -> None:
    # Shows whether Grover amplification is still visible after hardware noise
    # and readout mitigation.  This is the first diagnostic to inspect.
    rows = _load_rows(paths.amplification_points)
    if not rows:
        return
    rows = sorted(rows, key=lambda row: _as_float(row.get("grover_power")))
    x = np.asarray([_as_float(row.get("grover_power")) for row in rows], dtype=float)
    p_ideal = np.asarray([_as_float(row.get("p_ideal")) for row in rows], dtype=float)
    p_mitigated = np.asarray(
        [_as_float(row.get("p_hw_mitigated")) for row in rows],
        dtype=float,
    )
    p_mitigated_se = np.asarray(
        [_as_float(row.get("p_hw_mitigated_se"), 0.0) for row in rows],
        dtype=float,
    )
    p_raw = np.asarray(
        [_as_float(row.get("p_raw", row.get("p_hw_raw"))) for row in rows],
        dtype=float,
    )

    calibration = _load_json(paths.calibration_summary)
    baseline = _as_float(calibration.get("contrast_baseline"), 0.125)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(x, p_ideal, color="black", marker="o", linewidth=1.4, label="Ideal")
    ax.errorbar(
        x,
        p_mitigated,
        yerr=np.maximum(p_mitigated_se, 0.0),
        color="#1F6F8B",
        marker="s",
        linewidth=1.4,
        capsize=3,
        label="Mitigated",
    )
    if np.any(np.isfinite(p_raw)):
        ax.scatter(x, p_raw, color="#D55E00", marker=".", label="Raw")
    ax.axhline(
        baseline,
        color="#6E6E6E",
        linestyle="--",
        linewidth=1.0,
        label=f"Contrast floor={baseline:.3g}",
    )
    ax.set_xlabel(r"Grover power $k$")
    ax.set_ylabel(r"$P(\mathrm{objective}=111)$")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#BFBFBF", linewidth=0.55, alpha=0.32)
    ax.grid(True, which="minor", color="#D7D7D7", linewidth=0.40, alpha=0.16)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure_png_and_pdf(fig, paths.plots_dir / "amplification_scan.png")
    save_figure_png_and_pdf(fig, paths.plots_dir / "amplification_scan_paper.png")
    plt.close(fig)


def plot_direct_trace(paths: RunPaths) -> None:
    # Direct live traces come from algorithms executed against the sampler,
    # before the replay expansion step.
    rows = _load_rows(paths.direct_trace)
    if not rows:
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_algorithm_key(row)].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    a_true = _as_float(rows[0].get("a_true"))
    for key, group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: _as_float(row.get("query_budget")))
        x = np.asarray([_as_float(row.get("query_budget")) for row in group], dtype=float)
        estimates = np.asarray([_as_float(row.get("estimate")) for row in group], dtype=float)
        errors = np.asarray(
            [_as_float(row.get("normalized_abs_error")) for row in group],
            dtype=float,
        )
        style = _style(group[0])
        label = ALGORITHM_LABELS.get(key, key)
        axes[0].plot(
            x,
            estimates,
            marker=style.get("marker", "o"),
            color=style.get("color"),
            label=label,
        )
        valid = np.isfinite(x) & np.isfinite(errors) & (x > 0.0) & (errors > 0.0)
        if np.any(valid):
            axes[1].plot(
                x[valid],
                errors[valid],
                marker=style.get("marker", "o"),
                color=style.get("color"),
                label=label,
            )
    if np.isfinite(a_true):
        axes[0].axhline(a_true, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("State-preparation calls")
    axes[0].set_ylabel("Amplitude estimate")
    axes[0].set_title("Direct live estimate")
    axes[1].set_xlabel("State-preparation calls")
    axes[1].set_ylabel("Normalized amplitude error")
    axes[1].set_title("Direct live error")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure_png_and_pdf(fig, paths.plots_dir / "direct_live_trace.png")
    plt.close(fig)


def plot_replay_outputs(
    paths: RunPaths,
    *,
    max_queries: float | None,
    include_monte_carlo: bool,
    monte_carlo_budget_rows: Path | None,
) -> None:
    # Replay rows are the main comparison surface for AE algorithms because they
    # reuse the same measured hardware probabilities across many repetitions.
    budget_rows = _load_rows(paths.replay_budget)
    if not budget_rows:
        return
    budget_rows = _append_optional_rows(
        budget_rows,
        monte_carlo_budget_rows or (paths.run_dir / "montecarlo_budget_rows.csv"),
        include=include_monte_carlo,
    )
    if max_queries is not None:
        budget_rows = [
            row
            for row in budget_rows
            if _as_float(row.get("query_budget_actual", row.get("budget"))) <= float(max_queries)
        ]
    if not budget_rows:
        return

    summary_rows = _summary_rows_with_optional_monte_carlo(
        paths,
        include_monte_carlo=include_monte_carlo,
    )
    if max_queries is not None:
        summary_rows = [
            row
            for row in summary_rows
            if _as_float(row.get("query_budget_actual_mean", row.get("budget")))
            <= float(max_queries)
        ]

    plot_budget_summary(
        summary_rows,
        output_path=paths.plots_dir / "hardware_replay_budget.png",
        metric_key="normalized_abs_error_median",
        ylabel="Median normalized amplitude error",
        max_points_per_algorithm=None,
        algorithm_labels=PLOT_ALGORITHM_LABELS,
        algorithm_styles=PLOT_ALGORITHM_STYLES,
    )
    plot_budget_summary(
        summary_rows,
        output_path=paths.plots_dir / "hardware_replay_cva_budget.png",
        metric_key="processed_relative_error_median",
        ylabel="Median relative CVA error",
        max_points_per_algorithm=None,
        algorithm_labels=PLOT_ALGORITHM_LABELS,
        algorithm_styles=PLOT_ALGORITHM_STYLES,
    )
    paper_final_rows = budget_rows
    if not include_monte_carlo:
        monte_carlo_rows_path = monte_carlo_budget_rows or (paths.run_dir / "montecarlo_budget_rows.csv")
        if monte_carlo_rows_path.exists() and monte_carlo_rows_path.stat().st_size > 0:
            paper_final_rows = _append_optional_rows(
                paper_final_rows,
                monte_carlo_rows_path,
                include=True,
            )
            if max_queries is not None:
                paper_final_rows = [
                    row
                    for row in paper_final_rows
                    if _as_float(row.get("query_budget_actual", row.get("budget")))
                    <= float(max_queries)
                ]
    paper_final_rows = _attach_classical_runtime_profile(
        paper_final_rows,
        paths.run_dir / "classical_runtime_profile" / "montecarlo_budget_rows.csv",
    )
    with mpl.rc_context(PAPER_STYLE):
        plot_final_runtime_scatter_from_budget_rows(
            paper_final_rows,
            output_path=paths.plots_dir / "hardware_replay_final_error_queries.png",
            algorithms=PAPER_FINAL_ALGORITHMS,
            algorithm_labels=PLOT_ALGORITHM_LABELS,
            algorithm_styles=PLOT_ALGORITHM_STYLES,
            summary_path=paths.plots_dir / "hardware_replay_final_error_queries_summary.csv",
            x_kind="queries",
            title="",
            gaussian_contours=True,
            paper_style=True,
        )
        plot_final_runtime_scatter_from_budget_rows(
            paper_final_rows,
            output_path=paths.plots_dir / "hardware_replay_final_error_runtime.png",
            algorithms=PAPER_FINAL_ALGORITHMS,
            algorithm_labels=PLOT_ALGORITHM_LABELS,
            algorithm_styles=PLOT_ALGORITHM_STYLES,
            summary_path=paths.plots_dir / "hardware_replay_final_error_runtime_summary.csv",
            x_kind="runtime",
            title="",
            gaussian_contours=True,
            paper_style=True,
        )


def plot_budget_summaries(paths: RunPaths, *, include_monte_carlo: bool) -> None:
    rows = _summary_rows_with_optional_monte_carlo(
        paths,
        include_monte_carlo=include_monte_carlo,
    )
    if not rows:
        return
    plot_budget_summary(
        rows,
        output_path=paths.plots_dir / "budget_summary_amplitude.png",
        ylabel="Median normalized amplitude error",
        algorithm_labels=PLOT_ALGORITHM_LABELS,
        algorithm_styles=PLOT_ALGORITHM_STYLES,
    )
    plot_budget_summary(
        rows,
        output_path=paths.plots_dir / "budget_summary_cva.png",
        metric_key="processed_relative_error_median",
        ylabel="Median relative CVA error",
        algorithm_labels=PLOT_ALGORITHM_LABELS,
        algorithm_styles=PLOT_ALGORITHM_STYLES,
    )


class HardwareCvaPlotter:
    """Builds the standard diagnostic plots for a 6q CVA hardware AE run."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        max_queries: float | None = None,
        include_monte_carlo: bool = False,
        monte_carlo_budget_rows: str | Path | None = None,
    ) -> None:
        self.paths: RunPaths = RunPaths(Path(run_dir).expanduser().resolve())
        self.max_queries: float | None = max_queries
        self.include_monte_carlo: bool = bool(include_monte_carlo)
        self.monte_carlo_budget_rows: Path | None = (
            None
            if monte_carlo_budget_rows is None
            else Path(monte_carlo_budget_rows).expanduser().resolve()
        )

    def run(self) -> Path:
        # Each plotting function is idempotent and skips missing inputs, so the
        # same plot command can be used for dry-run, hardware, and replay-only
        # directories.
        self.paths.plots_dir.mkdir(parents=True, exist_ok=True)
        with mpl.rc_context(PAPER_STYLE):
            plot_amplification(self.paths)
            plot_direct_trace(self.paths)
            plot_replay_outputs(
                self.paths,
                max_queries=self.max_queries,
                include_monte_carlo=self.include_monte_carlo,
                monte_carlo_budget_rows=self.monte_carlo_budget_rows,
            )
            plot_budget_summaries(
                self.paths,
                include_monte_carlo=self.include_monte_carlo,
            )
        return self.paths.plots_dir


def plot_hardware_run(
    run_dir: str | Path,
    *,
    max_queries: float | None = None,
    include_monte_carlo: bool = False,
    monte_carlo_budget_rows: str | Path | None = None,
) -> Path:
    plotter: HardwareCvaPlotter = HardwareCvaPlotter(
        run_dir,
        max_queries=max_queries,
        include_monte_carlo=include_monte_carlo,
        monte_carlo_budget_rows=monte_carlo_budget_rows,
    )
    return plotter.run()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot 6q CVA hardware AE artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--max-queries",
        type=float,
        default=None,
        help="Only include replay rows at or below this actual query cost.",
    )
    parser.add_argument("--include-monte-carlo", action="store_true")
    parser.add_argument("--monte-carlo-budget-rows", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    plots_dir: Path = plot_hardware_run(
        args.run_dir,
        max_queries=args.max_queries,
        include_monte_carlo=bool(args.include_monte_carlo),
        monte_carlo_budget_rows=args.monte_carlo_budget_rows,
    )
    print(f"Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
