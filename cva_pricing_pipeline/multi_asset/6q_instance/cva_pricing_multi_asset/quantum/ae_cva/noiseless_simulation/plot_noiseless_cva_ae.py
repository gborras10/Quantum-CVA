from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pipeline_common import DEFAULT_RUN_DIR, add_cva_aliases, parse_name_list
from quantum_cva.amplitude_estimation.experiments.io import (
    RunPaths,
    load_csv,
    save_csv,
)
from quantum_cva.amplitude_estimation.experiments.plotting import (
    _log_limits,
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    ALGORITHM_STYLES,
    normalize_algorithm_key,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (
    as_float,
    finite_positive_pair_mask,
    query_budget,
)


PAPER_RUNTIME_ALGORITHMS = ("cabiqae_latentt", "biqae", "classical_mc")
PLOT_ALGORITHM_LABELS = {
    **ALGORITHM_LABELS,
    "classical_mc": "Classical MC",
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot noiseless 6q CVA amplitude-estimation outputs."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated algorithm keys. Defaults to algorithms in the CSVs.",
    )
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return add_cva_aliases(load_csv(path))


def _normalise_algorithms(
    algorithms: Sequence[str] | str | None,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...] | None:
    if algorithms is not None:
        return tuple(normalize_algorithm_key(name) for name in parse_name_list(algorithms))
    keys = sorted({str(row.get("algorithm_key", "")) for row in rows if row.get("algorithm_key")})
    return tuple(keys) if keys else None


def _summary_projected_to_metric(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    ci_low_key = f"{metric_key}_ci_low"
    ci_high_key = f"{metric_key}_ci_high"
    se_key = (
        f"{metric_key.removesuffix('_median')}_se"
        if metric_key.endswith("_median")
        else f"{metric_key}_se"
    )
    for row in rows:
        value = as_float(row.get(metric_key))
        if not np.isfinite(value) or value <= 0.0:
            continue
        out = dict(row)
        out["normalized_abs_error_median"] = value
        out["normalized_abs_error_median_ci_low"] = as_float(row.get(ci_low_key))
        out["normalized_abs_error_median_ci_high"] = as_float(row.get(ci_high_key))
        out["normalized_abs_error_se"] = as_float(row.get(se_key), 0.0)
        projected.append(out)
    return projected


def _budget_rows_projected_to_metric(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        value = as_float(row.get(metric_key))
        if not np.isfinite(value) or value <= 0.0:
            continue
        out = dict(row)
        out["normalized_abs_error"] = value
        projected.append(out)
    return projected


def _budget_like_rows_from_final_rows(
    final_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for final in final_rows:
        queries = as_float(final.get("final_queries"))
        if not np.isfinite(queries) or queries <= 0.0:
            continue
        rows.append(
            {
                "run_kind": final.get("run_kind", "ideal_noiseless"),
                "repetition": final.get("repetition", 0),
                "algorithm": final.get("algorithm", ""),
                "algorithm_key": final.get("algorithm_key", ""),
                "budget": int(round(queries)),
                "query_budget_actual": queries,
                "runtime_wall_seconds": final.get("runtime_wall_seconds", np.nan),
                "normalized_abs_error": final.get(
                    "final_normalized_abs_error",
                    final.get("normalized_abs_error", np.nan),
                ),
                "processed_relative_error": final.get(
                    "processed_relative_error",
                    final.get("cva_relative_error", np.nan),
                ),
            }
        )
    return rows


def _terminal_rows(
    rows: Sequence[Mapping[str, Any]],
    algorithms: Sequence[str],
) -> list[Mapping[str, Any]]:
    final_by_run: dict[tuple[str, int], Mapping[str, Any]] = {}
    allowed = set(algorithms)
    for row in rows:
        algorithm = str(row.get("algorithm_key", row.get("algorithm", "")))
        if algorithm not in allowed:
            continue
        key = (algorithm, int(as_float(row.get("repetition"), 0)))
        current = final_by_run.get(key)
        if current is None or query_budget(row) > query_budget(current):
            final_by_run[key] = row
    return list(final_by_run.values())


def _add_log_gaussian_contours(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    color: str,
) -> None:
    points = np.column_stack([np.log10(x_values), np.log10(y_values)])
    if points.shape[0] < 3:
        return

    lower, upper = np.quantile(points, [0.025, 0.975], axis=0)
    central = points[np.all((points >= lower) & (points <= upper), axis=1)]
    if central.shape[0] < 3:
        central = points
    covariance = np.cov(central, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1.0e-10)
    center = np.median(points, axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    for radius, alpha in ((np.sqrt(2.30), 0.24), (np.sqrt(5.99), 0.14)):
        outline = center + radius * (circle @ transform.T)
        ax.plot(
            10.0 ** outline[:, 0],
            10.0 ** outline[:, 1],
            color=color,
            linewidth=0.85,
            alpha=alpha,
            zorder=1,
        )


def _plot_paper_final_amplitude_runtime(
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    summary_path: Path,
    x_kind: str = "runtime",
) -> None:
    if x_kind not in {"runtime", "queries"}:
        raise ValueError("x_kind must be 'runtime' or 'queries'.")
    terminal_rows = _terminal_rows(budget_rows, PAPER_RUNTIME_ALGORITHMS)
    summary_rows: list[dict[str, Any]] = []
    all_x: list[float] = []
    all_y: list[float] = []

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 4.25))
        legend_handles: list[Line2D] = []
        for algorithm in PAPER_RUNTIME_ALGORITHMS:
            group = [
                row for row in terminal_rows if str(row.get("algorithm_key", "")) == algorithm
            ]
            x_values = np.asarray(
                [
                    as_float(row.get("runtime_wall_seconds"))
                    if x_kind == "runtime"
                    else query_budget(row)
                    for row in group
                ],
                dtype=float,
            )
            y_values = np.asarray(
                [as_float(row.get("normalized_abs_error")) for row in group],
                dtype=float,
            )
            valid = finite_positive_pair_mask(x_values, y_values)
            x_values = x_values[valid]
            y_values = y_values[valid]
            if x_values.size == 0:
                continue

            label = PLOT_ALGORITHM_LABELS.get(algorithm, algorithm)
            style = PLOT_ALGORITHM_STYLES.get(
                algorithm, {"color": "#333333", "marker": "o"}
            )
            color = str(style.get("color", "#333333"))
            marker = str(style.get("marker", "o"))
            all_x.extend(x_values.tolist())
            all_y.extend(y_values.tolist())
            _add_log_gaussian_contours(ax, x_values, y_values, color=color)
            ax.scatter(
                x_values,
                y_values,
                s=16,
                marker=marker,
                color=color,
                alpha=0.30,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            ax.scatter(
                [float(np.median(x_values))],
                [float(np.median(y_values))],
                s=64,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.85,
                zorder=4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    linestyle="None",
                    markersize=6.5,
                    label=label,
                )
            )
            summary_rows.append(
                {
                    "algorithm": label,
                    "algorithm_key": algorithm,
                    "mean_final_normalized_abs_error": float(np.mean(y_values)),
                    "median_final_normalized_abs_error": float(np.median(y_values)),
                    "n": int(x_values.size),
                }
            )
            if x_kind == "runtime":
                summary_rows[-1]["mean_runtime_seconds"] = float(np.mean(x_values))
                summary_rows[-1]["median_runtime_seconds"] = float(np.median(x_values))
            else:
                summary_rows[-1]["mean_final_queries"] = float(np.mean(x_values))
                summary_rows[-1]["median_final_queries"] = float(np.median(x_values))

        x_limits = _log_limits(np.asarray(all_x, dtype=float), pad_fraction=0.08)
        y_limits = _log_limits(np.asarray(all_y, dtype=float), pad_fraction=0.10)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(
            r"Classical runtime (seconds)"
            if x_kind == "runtime"
            else r"Final query count $N_q$"
        )
        ax.set_ylabel("Final normalized absolute error")
        ax.grid(True, which="major", color="#BFBFBF", linewidth=0.55, alpha=0.32)
        ax.grid(True, which="minor", color="#D7D7D7", linewidth=0.40, alpha=0.16)
        ax.legend(handles=legend_handles, loc="best")
        fig.tight_layout(pad=0.35)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

    save_csv(summary_rows, summary_path)


def make_plots(
    run_dir: str | Path,
    *,
    algorithms: Sequence[str] | str | None = None,
) -> list[Path]:
    paths = RunPaths(Path(run_dir))
    summary_rows = _load_rows(paths.budget_summary)
    budget_rows = _load_rows(paths.replay_budget)
    final_rows = _load_rows(paths.direct_final)
    if not budget_rows:
        budget_rows = _budget_like_rows_from_final_rows(final_rows)

    algorithm_keys = _normalise_algorithms(algorithms, budget_rows or summary_rows)
    plots_dir = paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    amplitude_budget_plot = plots_dir / "noseless_cva_ae_amplitude_error_vs_queries.png"
    plot_budget_summary(
        summary_rows,
        output_path=amplitude_budget_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        title="6q CVA AE, noiseless simulation",
        metric_key="normalized_abs_error_median",
        ylabel="Median relative amplitude error",
    )
    written.append(amplitude_budget_plot)

    cva_summary_rows = _summary_projected_to_metric(
        summary_rows,
        "processed_relative_error_median",
    )
    cva_budget_plot = plots_dir / "noseless_cva_ae_cva_relative_error_vs_queries.png"
    plot_budget_summary(
        cva_summary_rows,
        output_path=cva_budget_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        title="6q CVA AE, noiseless simulation",
        metric_key="normalized_abs_error_median",
        ylabel="Median relative CVA error",
    )
    written.append(cva_budget_plot)

    amp_runtime_plot = plots_dir / "noseless_cva_ae_final_amplitude_error_vs_runtime.png"
    amp_runtime_summary = paths.run_dir / "final_amplitude_error_vs_runtime_summary.csv"
    monte_carlo_rows = _load_rows(paths.run_dir / "montecarlo_budget_rows.csv")
    if not monte_carlo_rows:
        raise FileNotFoundError(
            "The paper runtime panel requires montecarlo_budget_rows.csv in "
            f"{paths.run_dir}."
        )
    amp_runtime_rows = [*budget_rows, *monte_carlo_rows]
    _plot_paper_final_amplitude_runtime(
        amp_runtime_rows,
        output_path=amp_runtime_plot,
        summary_path=amp_runtime_summary,
        x_kind="runtime",
    )
    written.append(amp_runtime_plot)

    amp_query_plot = plots_dir / "noseless_cva_ae_final_amplitude_error_vs_queries.png"
    amp_query_summary = paths.run_dir / "final_amplitude_error_vs_queries_summary.csv"
    _plot_paper_final_amplitude_runtime(
        amp_runtime_rows,
        output_path=amp_query_plot,
        summary_path=amp_query_summary,
        x_kind="queries",
    )
    written.append(amp_query_plot)

    cva_budget_rows = _budget_rows_projected_to_metric(
        budget_rows,
        "processed_relative_error",
    )
    cva_budget_rows = [
        row for row in cva_budget_rows if str(row.get("algorithm_key", "")).lower() != "bae"
    ]
    cva_runtime_plot = plots_dir / "noseless_cva_ae_final_cva_error_vs_runtime.png"
    cva_runtime_summary = paths.run_dir / "final_cva_error_vs_runtime_summary.csv"
    plot_final_runtime_scatter_from_budget_rows(
        cva_budget_rows,
        output_path=cva_runtime_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        summary_path=cva_runtime_summary,
        x_kind="runtime",
        title="6q CVA AE final CVA error",
    )
    written.append(cva_runtime_plot)

    cva_query_plot = plots_dir / "noseless_cva_ae_final_cva_error_vs_queries.png"
    cva_query_summary = paths.run_dir / "final_cva_error_vs_queries_summary.csv"
    plot_final_runtime_scatter_from_budget_rows(
        cva_budget_rows,
        output_path=cva_query_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        summary_path=cva_query_summary,
        x_kind="queries",
        title="6q CVA AE final CVA error",
    )
    written.append(cva_query_plot)

    plot_index = [
        {
            "plot_png": str(path),
            "plot_pdf": str(path.with_suffix(".pdf")),
        }
        for path in written
    ]
    save_csv(plot_index, paths.run_dir / "plot_index.csv")
    return written


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    algorithms = args.algorithms if args.algorithms else None
    written = make_plots(args.run_dir, algorithms=algorithms)
    print(f"Wrote {len(written)} plot families to {RunPaths(Path(args.run_dir)).plots_dir}")


if __name__ == "__main__":
    main()
