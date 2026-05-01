from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def log_binned_query_summary(
    query_budget: np.ndarray,
    error: np.ndarray,
    *,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(query_budget)
        & np.isfinite(error)
        & (query_budget > 0.0)
        & (error > 0.0)
    )
    query_budget = query_budget[valid]
    error = error[valid]
    if query_budget.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

    if query_budget.size <= max_bins:
        order = np.argsort(query_budget)
        return (
            query_budget[order],
            error[order],
            np.zeros(query_budget.size, dtype=float),
            np.ones(query_budget.size, dtype=int),
        )

    q_min = float(np.nanmin(query_budget))
    q_max = float(np.nanmax(query_budget))
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min <= 0.0 or q_max <= q_min:
        return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])

    edges = np.geomspace(q_min, q_max, num=max_bins + 1)
    bin_index = np.digitize(query_budget, edges, right=False) - 1
    bin_index = np.clip(bin_index, 0, max_bins - 1)

    x_values: list[float] = []
    y_values: list[float] = []
    yerr_values: list[float] = []
    n_values: list[int] = []
    for idx in range(max_bins):
        mask = bin_index == idx
        n_points = int(np.sum(mask))
        if n_points < min_points_per_bin:
            continue
        q_bin = query_budget[mask]
        e_bin = error[mask]
        x_values.append(float(np.nanmedian(q_bin)))
        y_values.append(float(np.nanmedian(e_bin)))
        err_std = float(np.nanstd(e_bin, ddof=1)) if n_points > 1 else 0.0
        yerr_values.append(err_std / np.sqrt(n_points))
        n_values.append(n_points)

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(yerr_values, dtype=float),
        np.asarray(n_values, dtype=int),
    )


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


def plot_actual_query_error(
    rows: pd.DataFrame,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    pdf_path: Path | None = None,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
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

    for algorithm in ALGORITHM_ORDER:
        group = df[df["algorithm"] == algorithm]
        if group.empty:
            continue

        style = ACTUAL_QUERY_STYLE[algorithm]
        query_budget = group[query_col].to_numpy(dtype=float)
        error = group[error_col].to_numpy(dtype=float)
        x_values, y_values, yerr, n_values = log_binned_query_summary(
            query_budget,
            error,
            max_bins=max_bins,
            min_points_per_bin=min_points_per_bin,
        )
        if x_values.size == 0:
            continue

        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        yerr = yerr[order]
        n_values = n_values[order]
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        yerr = np.minimum(yerr, np.maximum(0.0, 0.95 * y_values))

        guide_points.extend(
            (float(x), float(y))
            for x, y in zip(x_values, y_values)
            if np.isfinite(x) and np.isfinite(y) and x > 0.0 and y > 0.0
        )
        summary_rows.extend(
            {
                "algorithm": algorithm,
                "query_cost_median": float(x),
                "normalized_abs_error_median": float(y),
                "normalized_abs_error_se": float(err),
                "n_points": int(n),
            }
            for x, y, err, n in zip(x_values, y_values, yerr, n_values)
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=yerr,
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            markersize=5.6,
            elinewidth=1.0,
            capsize=2.8,
            label=run_count_label(algorithm, group),
        )

    if guide_points:
        x_values = np.asarray([x for x, _ in guide_points], dtype=float)
        y_values = np.asarray([y for _, y in guide_points], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.size:
            x0 = float(np.nanmin(x_values))
            y0_values = y_values[np.isclose(x_values, x0)]
            y0 = float(np.nanmedian(y0_values)) if y0_values.size else float(np.nanmedian(y_values))
            guide_x = np.geomspace(x0, float(np.nanmax(x_values)), num=200)
            ax.loglog(
                guide_x,
                y0 * (x0 / guide_x),
                color="black",
                linestyle="--",
                linewidth=1.15,
                alpha=0.82,
                label=r"$O(1/N)$",
            )
            ax.loglog(
                guide_x,
                y0 * np.sqrt(x0 / guide_x),
                color="black",
                linestyle=":",
                linewidth=1.35,
                alpha=0.82,
                label=r"$O(1/\sqrt{N})$",
            )

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
        fig.savefig(pdf_path)
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return summary
