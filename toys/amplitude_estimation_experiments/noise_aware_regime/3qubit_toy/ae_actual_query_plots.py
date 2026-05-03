from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common_utils.plotting_utils import (  # noqa: E402
    log_query_bin_indices,
)


ACTUAL_QUERY_STYLE = {
    "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
    "BIQAE": {"color": "#A23B72", "marker": "s"},
    "BAE": {"color": "#E07A5F", "marker": "^"},
}

ALGORITHM_ORDER = ("BAE", "BIQAE", "CABIQAE")


def normalize_algorithm_name(value: object) -> str:
    raw = str(value)
    key = raw.strip().lower()
    if key in {"cabiqae", "cabiqae_latentt", "cabiqae-latentt"}:
        return "CABIQAE"
    if key == "biqae":
        return "BIQAE"
    if key == "bae":
        return "BAE"
    return raw


def first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in df:
            return name
    raise KeyError(f"None of these columns are present: {', '.join(names)}")


def run_count_label(algorithm: str, group: pd.DataFrame) -> str:
    if {"scenario_id", "rep"}.issubset(group.columns):
        n_runs = int(group[["scenario_id", "rep"]].drop_duplicates().shape[0])
        return f"{algorithm} ({n_runs} reps)"
    if {"objective_ry_offset", "rep"}.issubset(group.columns):
        n_runs = int(group[["objective_ry_offset", "rep"]].drop_duplicates().shape[0])
        return f"{algorithm} ({n_runs} reps)"
    if "repetition" in group:
        n_runs = int(group["repetition"].nunique())
        return f"{algorithm} ({n_runs} reps)"
    if "rep" in group:
        n_runs = int(group["rep"].nunique())
        return f"{algorithm} ({n_runs} reps)"
    return str(algorithm)


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan

    center = float(np.nanmedian(finite))
    if finite.size == 1 or int(bootstrap_samples) <= 0:
        return center, center, center

    indices = rng.integers(0, finite.size, size=(int(bootstrap_samples), finite.size))
    medians = np.nanmedian(finite[indices], axis=1)
    alpha = 1.0 - float(confidence_level)
    low, high = np.nanquantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(low), float(high)


def bootstrap_ci_errorbar(
    center_values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> np.ndarray:
    center_values = np.asarray(center_values, dtype=float)
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)
    lower = np.where(np.isfinite(ci_low), center_values - ci_low, 0.0)
    upper = np.where(np.isfinite(ci_high), ci_high - center_values, 0.0)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    lower = np.minimum(lower, np.maximum(0.0, 0.95 * center_values))
    return np.vstack([lower, upper])


def add_power_fit_query_scaling_guides(
    ax: plt.Axes,
    guide_points: list[tuple[float, float]],
    *,
    include_linear: bool = True,
    include_sqrt: bool = True,
    alpha: float = 0.82,
) -> None:
    x_values = np.asarray([x for x, _ in guide_points], dtype=float)
    y_values = np.asarray([y for _, y in guide_points], dtype=float)
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )
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

    if x_values.size >= 2 and float(np.nanmax(x_values)) > float(np.nanmin(x_values)):
        fit_slope, fit_log_intercept = np.polyfit(np.log(x_values), np.log(y_values), deg=1)
        y0 = float(np.exp(fit_log_intercept) * x0**fit_slope)
    else:
        y0 = float(y_values[0])
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
            alpha=alpha,
            label=r"$O(1/N)$",
        )
    if include_sqrt:
        ax.loglog(
            guide_x,
            y0 * np.sqrt(x0 / guide_x),
            color="black",
            linestyle=":",
            linewidth=1.35,
            alpha=alpha,
            label=r"$O(1/\sqrt{N})$",
        )


def plot_actual_query_error(
    rows: pd.DataFrame,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    pdf_path: Path | None = None,
    max_bins: int = 12,
    min_points_per_bin: int = 15,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 12345,
    drop_binned_point_indices: dict[str, tuple[int, ...]] | None = None,
) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("rows is empty")

    df = rows.copy()
    df["algorithm"] = df["algorithm"].map(normalize_algorithm_name)
    query_col = first_existing_column(
        df,
        ("query_budget", "query_budget_actual", "budget", "final_queries"),
    )
    error_col = first_existing_column(
        df,
        ("normalized_abs_error", "nrmse", "final_normalized_abs_error", "final_nrmse"),
    )
    df[query_col] = pd.to_numeric(df[query_col], errors="coerce")
    df[error_col] = pd.to_numeric(df[error_col], errors="coerce")
    df = df[
        np.isfinite(df[query_col])
        & np.isfinite(df[error_col])
        & (df[query_col] > 0.0)
        & (df[error_col] > 0.0)
    ]
    if df.empty:
        raise ValueError("No positive finite rows are available for plotting")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    guide_points: list[tuple[float, float]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(int(bootstrap_seed))
    drop_binned_point_indices = drop_binned_point_indices or {}

    for algorithm in ALGORITHM_ORDER:
        group = df[df["algorithm"] == algorithm]
        if group.empty:
            continue

        style = ACTUAL_QUERY_STYLE[algorithm]
        query_budget = group[query_col].to_numpy(dtype=float)
        error = group[error_col].to_numpy(dtype=float)
        bin_indices = log_query_bin_indices(
            query_budget,
            error,
            max_bins=max_bins,
            min_points_per_bin=min_points_per_bin,
        )
        if not bin_indices:
            continue

        x_values: list[float] = []
        x_median_values: list[float] = []
        y_values: list[float] = []
        ci_low_values: list[float] = []
        ci_high_values: list[float] = []
        n_values: list[int] = []
        for indices in bin_indices:
            q_bin = query_budget[indices]
            e_bin = error[indices]
            valid_bin = (
                np.isfinite(q_bin)
                & np.isfinite(e_bin)
                & (q_bin > 0.0)
                & (e_bin > 0.0)
            )
            q_bin = q_bin[valid_bin]
            e_bin = e_bin[valid_bin]
            if e_bin.size == 0:
                continue
            center, ci_low, ci_high = bootstrap_median_ci(
                e_bin,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            x_values.append(float(np.nanmean(q_bin)))
            x_median_values.append(float(np.nanmedian(q_bin)))
            y_values.append(center)
            ci_low_values.append(ci_low)
            ci_high_values.append(ci_high)
            n_values.append(int(e_bin.size))

        x_values_arr = np.asarray(x_values, dtype=float)
        x_median_arr = np.asarray(x_median_values, dtype=float)
        y_values_arr = np.asarray(y_values, dtype=float)
        ci_low_arr = np.asarray(ci_low_values, dtype=float)
        ci_high_arr = np.asarray(ci_high_values, dtype=float)
        n_values_arr = np.asarray(n_values, dtype=int)
        if x_values_arr.size == 0:
            continue

        order = np.argsort(x_values_arr)
        x_values_arr = x_values_arr[order]
        x_median_arr = x_median_arr[order]
        y_values_arr = y_values_arr[order]
        ci_low_arr = ci_low_arr[order]
        ci_high_arr = ci_high_arr[order]
        n_values_arr = n_values_arr[order]

        drop_indices = {
            int(idx)
            for idx in drop_binned_point_indices.get(algorithm, ())
            if -x_values_arr.size <= int(idx) < x_values_arr.size
        }
        if drop_indices:
            keep = np.ones(x_values_arr.size, dtype=bool)
            for idx in drop_indices:
                keep[idx % x_values_arr.size] = False
            x_values_arr = x_values_arr[keep]
            x_median_arr = x_median_arr[keep]
            y_values_arr = y_values_arr[keep]
            ci_low_arr = ci_low_arr[keep]
            ci_high_arr = ci_high_arr[keep]
            n_values_arr = n_values_arr[keep]
        if x_values_arr.size == 0:
            continue

        guide_points.extend(
            (float(x), float(y))
            for x, y in zip(x_values_arr, y_values_arr)
            if np.isfinite(x) and np.isfinite(y) and x > 0.0 and y > 0.0
        )
        summary_rows.extend(
            {
                "algorithm": algorithm,
                "query_cost_mean": float(x),
                "query_cost_median": float(x_median),
                "normalized_abs_error_median": float(y),
                "normalized_abs_error_median_ci_low": float(ci_low),
                "normalized_abs_error_median_ci_high": float(ci_high),
                "n_points": int(n),
            }
            for x, x_median, y, ci_low, ci_high, n in zip(
                x_values_arr,
                x_median_arr,
                y_values_arr,
                ci_low_arr,
                ci_high_arr,
                n_values_arr,
            )
        )
        ax.errorbar(
            x_values_arr,
            y_values_arr,
            yerr=bootstrap_ci_errorbar(y_values_arr, ci_low_arr, ci_high_arr),
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linewidth=2.0,
            markersize=5.6,
            elinewidth=1.0,
            capsize=2.8,
            label=run_count_label(algorithm, group),
            zorder=3,
        )

    add_power_fit_query_scaling_guides(ax, guide_points)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Actual query cost $N_q$")
    ax.set_ylabel("Median normalized absolute error")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False, loc="lower left")
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

    summary = pd.DataFrame(summary_rows)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return summary
