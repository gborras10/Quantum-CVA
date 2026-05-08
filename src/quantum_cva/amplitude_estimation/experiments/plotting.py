from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from quantum_cva.amplitude_estimation.experiments.io import save_csv
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    ALGORITHM_STYLES,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (
    as_float,
    finite_positive_pair_mask,
    query_budget,
    select_log_spaced_indices,
)


ERROR_BAR_CAP_FRACTION = 0.95
CONNECTING_LINE_ALPHA = 0.56


def cap_log_errorbar(
    center_values: np.ndarray,
    yerr_values: np.ndarray,
    *,
    cap_fraction: float = ERROR_BAR_CAP_FRACTION,
) -> np.ndarray:
    center_values = np.asarray(center_values, dtype=float)
    yerr_values = np.asarray(yerr_values, dtype=float)
    yerr = np.where(np.isfinite(yerr_values), yerr_values, 0.0)
    yerr = np.maximum(yerr, 0.0)
    cap = np.maximum(0.0, float(cap_fraction) * center_values)
    return np.minimum(yerr, cap)


def bootstrap_ci_errorbar(
    center_values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    *,
    fallback_yerr_values: np.ndarray | None = None,
    cap_fraction: float = ERROR_BAR_CAP_FRACTION,
) -> np.ndarray:
    center_values = np.asarray(center_values, dtype=float)
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)

    lower = np.where(np.isfinite(ci_low), center_values - ci_low, np.nan)
    upper = np.where(np.isfinite(ci_high), ci_high - center_values, np.nan)
    ci_valid = np.isfinite(lower) & np.isfinite(upper) & (lower >= 0.0) & (upper >= 0.0)

    if fallback_yerr_values is None:
        fallback = np.zeros_like(center_values, dtype=float)
    else:
        fallback = cap_log_errorbar(
            center_values,
            np.asarray(fallback_yerr_values, dtype=float),
            cap_fraction=cap_fraction,
        )

    lower = np.where(ci_valid, lower, fallback)
    upper = np.where(ci_valid, upper, fallback)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    lower = np.minimum(lower, np.maximum(0.0, float(cap_fraction) * center_values))
    return np.vstack([lower, upper])


def metric_ci_keys(metric_key: str) -> tuple[str | None, str | None]:
    if metric_key.endswith("_median"):
        return f"{metric_key}_ci_low", f"{metric_key}_ci_high"
    return None, None


def plot_median_ci_errorbar(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    ci_low_values: np.ndarray,
    ci_high_values: np.ndarray,
    *,
    style: Mapping[str, Any],
    label: str,
    fallback_yerr_values: np.ndarray | None = None,
    linewidth: float = 2.0,
    markersize: float = 5.6,
    line_alpha: float = CONNECTING_LINE_ALPHA,
    markevery: int | None = None,
) -> None:
    color = style.get("color")
    if linewidth > 0.0 and x_values.size >= 2:
        ax.plot(
            x_values,
            y_values,
            color=color,
            linestyle="-",
            linewidth=linewidth,
            alpha=line_alpha,
            markevery=markevery,
            zorder=2,
        )
    ax.errorbar(
        x_values,
        y_values,
        yerr=bootstrap_ci_errorbar(
            y_values,
            ci_low_values,
            ci_high_values,
            fallback_yerr_values=fallback_yerr_values,
        ),
        color=color,
        marker=style.get("marker", "o"),
        linestyle="None",
        linewidth=0.0,
        markersize=markersize,
        elinewidth=1.0,
        capsize=2.8,
        label=label,
        markevery=markevery,
        zorder=3,
    )


def plot_median_se_errorbar(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    yerr_values: np.ndarray,
    *,
    style: Mapping[str, Any],
    label: str,
    linewidth: float = 2.0,
    markersize: float = 5.6,
    markevery: int | None = None,
) -> None:
    plot_median_ci_errorbar(
        ax,
        x_values,
        y_values,
        np.full_like(np.asarray(y_values, dtype=float), np.nan),
        np.full_like(np.asarray(y_values, dtype=float), np.nan),
        style=style,
        label=label,
        fallback_yerr_values=yerr_values,
        linewidth=linewidth,
        markersize=markersize,
        markevery=markevery,
    )


def add_query_scaling_guides(
    ax: plt.Axes,
    guide_points: list[tuple[float, float]] | tuple[np.ndarray, np.ndarray],
    *,
    include_linear: bool = True,
    include_sqrt: bool = True,
) -> None:
    if isinstance(guide_points, tuple):
        x_values = np.asarray(guide_points[0], dtype=float)
        y_values = np.asarray(guide_points[1], dtype=float)
    else:
        x_values = np.asarray([x for x, _ in guide_points], dtype=float)
        y_values = np.asarray([y for _, y in guide_points], dtype=float)
    valid = finite_positive_pair_mask(x_values, y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size == 0:
        return

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    x0 = float(x_values[0])
    x_max = float(x_values[-1])
    if not np.isfinite(x0) or not np.isfinite(x_max) or x0 <= 0.0 or x_max <= x0:
        return
    y0 = power_fit_anchor_y0(x_values, y_values, x0)
    if not np.isfinite(y0) or y0 <= 0.0:
        return

    guide_x = np.geomspace(x0, x_max, num=200)
    if include_linear:
        ax.loglog(
            guide_x,
            y0 * (x0 / guide_x),
            color="black",
            linestyle="--",
            linewidth=1.15,
            alpha=0.82,
            label=r"$O(1/N)$",
            zorder=1,
        )
    if include_sqrt:
        ax.loglog(
            guide_x,
            y0 * np.sqrt(x0 / guide_x),
            color="black",
            linestyle=":",
            linewidth=1.35,
            alpha=0.82,
            label=r"$O(1/\sqrt{N})$",
            zorder=1,
        )


def power_fit_anchor_y0(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x0: float | None = None,
) -> float:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = finite_positive_pair_mask(x_values, y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size == 0:
        return np.nan
    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    x0_value = float(x_values[0] if x0 is None else x0)
    if x_values.size >= 2 and float(np.nanmax(x_values)) > float(np.nanmin(x_values)):
        fit_slope, fit_log_intercept = np.polyfit(
            np.log(x_values),
            np.log(y_values),
            deg=1,
        )
        return float(np.exp(fit_log_intercept) * x0_value**fit_slope)
    return float(y_values[0])


def save_figure_png_and_pdf(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    except PermissionError:
        pass


def plot_budget_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    algorithms: Sequence[str] | None = None,
    algorithm_labels: Mapping[str, str] | None = None,
    algorithm_styles: Mapping[str, Mapping[str, Any]] | None = None,
    title: str = "",
    metric_key: str = "normalized_abs_error_median",
    ylabel: str = "Median normalized absolute error",
    max_points_per_algorithm: int | None = 14,
) -> None:
    if not summary_rows:
        return
    algorithm_labels = dict(ALGORITHM_LABELS if algorithm_labels is None else algorithm_labels)
    algorithm_styles = dict(ALGORITHM_STYLES if algorithm_styles is None else algorithm_styles)
    if algorithms is None:
        algorithms = sorted(
            {str(row.get("algorithm_key", row.get("algorithm", ""))) for row in summary_rows}
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    guide_points: list[tuple[float, float]] = []
    for algorithm in algorithms:
        label = algorithm_labels.get(algorithm, algorithm)
        group = [
            row
            for row in summary_rows
            if str(row.get("algorithm_key", "")) == algorithm
            or str(row.get("algorithm", "")) == label
        ]
        if not group:
            continue
        group = sorted(group, key=query_budget)
        x_values = np.asarray([query_budget(row) for row in group], dtype=float)
        y_values = np.asarray([as_float(row.get(metric_key)) for row in group], dtype=float)
        fallback_yerr_values = np.asarray(
            [as_float(row.get("normalized_abs_error_se"), 0.0) for row in group],
            dtype=float,
        )
        ci_low_key, ci_high_key = metric_ci_keys(metric_key)
        ci_low_values = np.asarray(
            [as_float(row.get(ci_low_key)) if ci_low_key else np.nan for row in group],
            dtype=float,
        )
        ci_high_values = np.asarray(
            [as_float(row.get(ci_high_key)) if ci_high_key else np.nan for row in group],
            dtype=float,
        )
        valid = finite_positive_pair_mask(x_values, y_values)
        if not np.any(valid):
            continue
        selected = (
            np.flatnonzero(valid)
            if max_points_per_algorithm is None
            else select_log_spaced_indices(x_values, valid, int(max_points_per_algorithm))
        )
        guide_points.extend(
            (float(x), float(y)) for x, y in zip(x_values[selected], y_values[selected])
        )
        plot_median_ci_errorbar(
            ax,
            x_values[selected],
            y_values[selected],
            ci_low_values[selected],
            ci_high_values[selected],
            style=algorithm_styles.get(algorithm, {"color": "#333333", "marker": "o"}),
            label=label,
            fallback_yerr_values=fallback_yerr_values[selected],
        )
    add_query_scaling_guides(ax, guide_points)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Actual query cost $N_q$")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure_png_and_pdf(fig, output_path)
    plt.close(fig)


def _log_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return None
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if np.isclose(low, high):
        return low / 1.6, high * 1.6
    log_low = np.log10(low)
    log_high = np.log10(high)
    span = log_high - log_low
    return 10.0 ** (log_low - pad_fraction * span), 10.0 ** (
        log_high + pad_fraction * span
    )


def plot_final_runtime_scatter_from_budget_rows(
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    algorithms: Sequence[str] | None = None,
    algorithm_labels: Mapping[str, str] | None = None,
    algorithm_styles: Mapping[str, Mapping[str, Any]] | None = None,
    summary_path: str | Path | None = None,
    x_kind: str = "runtime",
    title: str = "",
) -> list[dict[str, Any]]:
    if x_kind not in {"runtime", "queries"}:
        raise ValueError("x_kind must be 'runtime' or 'queries'.")
    if not budget_rows:
        return []
    algorithm_labels = dict(ALGORITHM_LABELS if algorithm_labels is None else algorithm_labels)
    algorithm_styles = dict(ALGORITHM_STYLES if algorithm_styles is None else algorithm_styles)
    if algorithms is None:
        algorithms = sorted(
            {str(row.get("algorithm_key", row.get("algorithm", ""))) for row in budget_rows}
        )

    by_run: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in budget_rows:
        key = (str(row.get("algorithm_key", row.get("algorithm", ""))), int(as_float(row.get("repetition"), 0)))
        current = by_run.get(key)
        if current is None or query_budget(row) > query_budget(current):
            by_run[key] = row
    final_rows = list(by_run.values())

    summary_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    legend_handles: list[Line2D] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for algorithm in algorithms:
        label = algorithm_labels.get(algorithm, algorithm)
        group = [
            row
            for row in final_rows
            if str(row.get("algorithm_key", "")) == algorithm
            or str(row.get("algorithm", "")) == label
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
        style = algorithm_styles.get(algorithm, {"color": "#333333", "marker": "o"})
        all_x.extend(x_values.tolist())
        all_y.extend(y_values.tolist())
        ax.scatter(
            x_values,
            y_values,
            s=30,
            marker=style.get("marker", "o"),
            color=style.get("color"),
            alpha=0.45,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.scatter(
            [float(np.nanmedian(x_values))],
            [float(np.nanmedian(y_values))],
            s=135,
            marker=style.get("marker", "o"),
            facecolor=style.get("color"),
            edgecolor="black",
            linewidth=1.1,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle="None",
                markersize=8,
                label=f"{label} (n={x_values.size})",
            )
        )
        summary_row = {
            "algorithm": label,
            "algorithm_key": algorithm,
            "n": int(x_values.size),
            "median_final_normalized_abs_error": float(np.nanmedian(y_values)),
            "mean_final_normalized_abs_error": float(np.nanmean(y_values)),
        }
        if x_kind == "runtime":
            summary_row["median_runtime_seconds"] = float(np.nanmedian(x_values))
            summary_row["mean_runtime_seconds"] = float(np.nanmean(x_values))
        else:
            summary_row["median_final_queries"] = float(np.nanmedian(x_values))
            summary_row["mean_final_queries"] = float(np.nanmean(x_values))
        summary_rows.append(summary_row)

    if summary_path is not None:
        save_csv(summary_rows, summary_path)
    x_limits = _log_limits(np.asarray(all_x, dtype=float))
    y_limits = _log_limits(np.asarray(all_y, dtype=float), pad_fraction=0.12)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime [s]" if x_kind == "runtime" else r"Final query count $N_q$")
    ax.set_ylabel("Final normalized absolute error")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, loc="best")
    fig.tight_layout()
    save_figure_png_and_pdf(fig, output_path)
    plt.close(fig)
    return summary_rows


def plot_replay_actual_queries(
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path,
    algorithms: Sequence[str] | None = None,
    algorithm_labels: Mapping[str, str] | None = None,
    algorithm_styles: Mapping[str, Mapping[str, Any]] | None = None,
    title: str = "",
) -> None:
    plot_budget_summary(
        budget_rows,
        output_path=output_path,
        algorithms=algorithms,
        algorithm_labels=algorithm_labels,
        algorithm_styles=algorithm_styles,
        title=title,
        metric_key="normalized_abs_error",
        ylabel="Normalized absolute error at replay budget",
        max_points_per_algorithm=None,
    )
