from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common_utils.plotting_utils import (  # noqa: E402
    add_query_scaling_guides,
    log_binned_median_se,
    plot_median_se_errorbar,
)


FINAL_ERROR_STYLE = {
    "BAE": {"color": "#E07A5F", "marker": "^"},
    "BIQAE": {"color": "#A23B72", "marker": "s"},
    "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
}

ALGORITHM_ORDER = ("BAE", "BIQAE", "CABIQAE")


def _algorithm_label(value: object) -> str:
    raw = str(value)
    key = raw.strip().lower()
    if key in {"cabiqae", "cabiqae_latentt", "cabiqae-latentt"}:
        return "CABIQAE"
    if key == "biqae":
        return "BIQAE"
    if key == "bae":
        return "BAE"
    return raw


def _first_existing(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df:
            return name
    raise KeyError(f"None of these columns are present: {', '.join(names)}")


def _log_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None

    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= 0.0:
        return None
    if np.isclose(low, high):
        return low / 1.6, high * 1.6

    log_low = np.log10(low)
    log_high = np.log10(high)
    span = log_high - log_low
    return 10.0 ** (log_low - pad_fraction * span), 10.0 ** (log_high + pad_fraction * span)


def _plot_log_gaussian_contours(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    color: str,
    levels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> None:
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 3:
        return

    log_points = np.vstack([np.log10(x_values), np.log10(y_values)])
    covariance = np.cov(log_points)
    if not np.all(np.isfinite(covariance)):
        return

    covariance = covariance + np.eye(2) * 1e-12
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if np.any(eigvals <= 0.0) or not np.all(np.isfinite(eigvals)):
        return

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    center = np.mean(log_points, axis=1)
    angles = np.linspace(0.0, 2.0 * np.pi, 360)
    unit_circle = np.vstack([np.cos(angles), np.sin(angles)])

    for level in levels:
        ellipse = center[:, None] + eigvecs @ (
            np.sqrt(eigvals)[:, None] * float(level) * unit_circle
        )
        ax.plot(
            10.0 ** ellipse[0],
            10.0 ** ellipse[1],
            color=color,
            linewidth=1.25,
            alpha=0.92,
            zorder=2,
        )


def _summary_text(summary: pd.DataFrame, *, x_kind: str) -> str:
    lines: list[str] = []
    for algorithm in ALGORITHM_ORDER:
        row = summary[summary["algorithm"] == algorithm]
        if row.empty:
            continue
        median_error = float(row["median_final_mnae"].iloc[0])
        if x_kind == "queries":
            q50 = float(row["median_final_queries"].iloc[0])
            lines.append(f"{algorithm}: q50={q50:.0f}, MNAE50={median_error:.2e}")
        else:
            t50 = float(row["median_runtime_seconds"].iloc[0])
            lines.append(f"{algorithm}: t50={t50:.2f}s, MNAE50={median_error:.2e}")
    return "\n".join(lines)


def plot_final_error_scatter(
    final_rows: pd.DataFrame,
    output_path: Path,
    *,
    x_kind: str,
    title: str,
    summary_path: Path | None = None,
    pdf_path: Path | None = None,
    max_points_per_algorithm: int | None = None,
    point_sample_seed: int = 12345,
    draw_query_median_lines: bool = True,
    draw_query_scaling_guides: bool = True,
) -> pd.DataFrame:
    if final_rows.empty:
        raise ValueError("final_rows is empty")
    if x_kind not in {"queries", "runtime"}:
        raise ValueError("x_kind must be 'queries' or 'runtime'")

    df = final_rows.copy()
    df["algorithm"] = df["algorithm"].map(_algorithm_label)
    error_col = _first_existing(df, ("final_normalized_abs_error", "final_nrmse"))
    x_col = (
        _first_existing(df, ("final_queries",))
        if x_kind == "queries"
        else _first_existing(df, ("runtime_wall_seconds", "runtime_seconds"))
    )

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[error_col] = pd.to_numeric(df[error_col], errors="coerce")
    df = df[
        np.isfinite(df[x_col])
        & np.isfinite(df[error_col])
        & (df[x_col] > 0.0)
        & (df[error_col] > 0.0)
    ]
    if df.empty:
        raise ValueError("No positive finite rows are available for plotting")

    summary_rows: list[dict[str, float | int | str]] = []
    for algorithm in ALGORITHM_ORDER:
        group = df[df["algorithm"] == algorithm]
        if group.empty:
            continue
        row: dict[str, float | int | str] = {
            "algorithm": algorithm,
            "n": int(len(group)),
            "median_final_mnae": float(np.nanmedian(group[error_col].to_numpy(dtype=float))),
        }
        if x_kind == "queries":
            row["median_final_queries"] = float(np.nanmedian(group[x_col].to_numpy(dtype=float)))
        else:
            row["median_runtime_seconds"] = float(np.nanmedian(group[x_col].to_numpy(dtype=float)))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    legend_handles: list[Line2D] = []
    all_x: list[float] = []
    all_y: list[float] = []
    query_guide_points: list[tuple[float, float]] = []
    query_reference_handles: list[Line2D] = []
    rng = np.random.default_rng(int(point_sample_seed))

    for algorithm in ALGORITHM_ORDER:
        group = df[df["algorithm"] == algorithm]
        if group.empty:
            continue
        style = FINAL_ERROR_STYLE[algorithm]
        x_values = group[x_col].to_numpy(dtype=float)
        y_values = group[error_col].to_numpy(dtype=float)
        plot_x_values = x_values
        plot_y_values = y_values
        if max_points_per_algorithm is not None and x_values.size > int(max_points_per_algorithm):
            selected = rng.choice(
                x_values.size,
                size=max(1, int(max_points_per_algorithm)),
                replace=False,
            )
            plot_x_values = x_values[selected]
            plot_y_values = y_values[selected]
        all_x.extend(plot_x_values.tolist())
        all_y.extend(plot_y_values.tolist())

        ax.scatter(
            plot_x_values,
            plot_y_values,
            s=28,
            marker=style["marker"],
            color=style["color"],
            alpha=0.42,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        _plot_log_gaussian_contours(ax, plot_x_values, plot_y_values, color=style["color"])

        median_x = float(np.nanmedian(x_values))
        median_y = float(np.nanmedian(y_values))
        ax.scatter(
            [median_x],
            [median_y],
            s=145,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="black",
            linewidth=1.1,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="None",
                markersize=8,
                label=f"{algorithm} (n={plot_x_values.size}/{len(group)})"
                if plot_x_values.size != len(group)
                else f"{algorithm} (n={len(group)})",
            )
        )

        if x_kind == "queries" and draw_query_median_lines:
            x_bins, y_bins, yerr_bins, _ = log_binned_median_se(
                x_values,
                y_values,
                max_bins=8,
                min_points_per_bin=3,
            )
            if x_bins.size:
                order = np.argsort(x_bins)
                x_bins = x_bins[order]
                y_bins = y_bins[order]
                yerr_bins = yerr_bins[order]
                query_guide_points.extend(
                    (float(x), float(y))
                    for x, y in zip(x_bins, y_bins)
                    if np.isfinite(x) and np.isfinite(y) and x > 0.0 and y > 0.0
                )
                plot_median_se_errorbar(
                    ax,
                    x_bins,
                    y_bins,
                    yerr_bins,
                    style=style,
                    label="_nolegend_",
                    linewidth=2.1,
                    markersize=5.8,
                    zorder=6,
                )

    if x_kind == "queries" and draw_query_scaling_guides:
        query_reference_handles = add_query_scaling_guides(ax, query_guide_points)

    median_handle = Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        markerfacecolor="white",
        linestyle="None",
        markersize=8,
        markeredgewidth=1.3,
        label="median",
    )
    contour_handle = Line2D(
        [0],
        [0],
        color="0.25",
        linewidth=1.7,
        label="Gaussian contours",
    )
    legend_handles.extend([median_handle, contour_handle])
    legend_handles.extend(query_reference_handles)

    ax.set_xscale("log")
    ax.set_yscale("log")
    x_limits = _log_limits(np.asarray(all_x, dtype=float), pad_fraction=0.07)
    y_limits = _log_limits(np.asarray(all_y, dtype=float), pad_fraction=0.12)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel("Final query count" if x_kind == "queries" else "Runtime [s]")
    ax.set_ylabel("Final normalized absolute error")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    text = _summary_text(summary, x_kind=x_kind)
    if text:
        ax.text(
            0.03,
            0.04,
            text,
            transform=ax.transAxes,
            fontsize=9.5,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.92,
            },
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(pdf_path)
        except PermissionError as exc:
            print(f"Skipped locked PDF output {pdf_path}: {exc}")
    plt.close(fig)
    return summary


def plot_final_error_figures(
    final_rows_path: Path,
    output_dir: Path,
    *,
    title_suffix: str,
    output_prefix: str = "triple_gaussian_error",
    max_points_per_algorithm: int | None = None,
    point_sample_seed: int = 12345,
    draw_query_median_lines: bool = True,
    draw_query_scaling_guides: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows = pd.read_csv(final_rows_path)
    queries_summary = plot_final_error_scatter(
        final_rows,
        output_dir / f"{output_prefix}_queries.png",
        x_kind="queries",
        title=f"Final error versus query cost {title_suffix}",
        summary_path=output_dir / f"{output_prefix}_queries_summary.csv",
        pdf_path=output_dir / f"{output_prefix}_queries.pdf",
        max_points_per_algorithm=max_points_per_algorithm,
        point_sample_seed=point_sample_seed,
        draw_query_median_lines=draw_query_median_lines,
        draw_query_scaling_guides=draw_query_scaling_guides,
    )
    runtime_summary = pd.DataFrame()
    if "runtime_wall_seconds" in final_rows or "runtime_seconds" in final_rows:
        runtime_summary = plot_final_error_scatter(
            final_rows,
            output_dir / f"{output_prefix}_runtime.png",
            x_kind="runtime",
            title=f"Final error versus runtime {title_suffix}",
            summary_path=output_dir / f"{output_prefix}_runtime_summary.csv",
            pdf_path=output_dir / f"{output_prefix}_runtime.pdf",
            max_points_per_algorithm=max_points_per_algorithm,
            point_sample_seed=point_sample_seed,
            draw_query_median_lines=draw_query_median_lines,
            draw_query_scaling_guides=draw_query_scaling_guides,
        )
    return queries_summary, runtime_summary
