from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiment_utils import ALGORITHM_LABELS, format_T_value


ERROR_BAR_CAP_FRACTION = 0.95

STYLE = {
    "biqae": {"color": "#A23B72", "marker": "o"},
    "cabiqae": {"color": "#1F6F8B", "marker": "s"},
    "cabiqae_no_cap": {"color": "#1F6F8B", "marker": "s"},
    "cabiqae_with_cap": {"color": "#E07A5F", "marker": "^"},
}


def _style_for(algorithm: str) -> dict[str, str]:
    return STYLE.get(algorithm, {"color": "#333333", "marker": "o"})


def finite_positive_pair_mask(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x_values and y_values must have the same shape.")
    return (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1) / np.sqrt(float(values.size)))


def log_query_bin_indices(
    query_budget: np.ndarray,
    error: np.ndarray,
    *,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
) -> list[np.ndarray]:
    query_budget = np.asarray(query_budget, dtype=float)
    error = np.asarray(error, dtype=float)
    valid = finite_positive_pair_mask(query_budget, error)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return []

    q_valid = query_budget[valid_indices]
    if q_valid.size <= int(max_bins):
        order = np.argsort(q_valid)
        return [np.asarray([int(valid_indices[idx])], dtype=int) for idx in order]

    q_min = float(np.nanmin(q_valid))
    q_max = float(np.nanmax(q_valid))
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min <= 0.0 or q_max <= q_min:
        order = np.argsort(q_valid)
        return [np.asarray([int(valid_indices[idx])], dtype=int) for idx in order]

    edges = np.geomspace(q_min, q_max, num=int(max_bins) + 1)
    local_bins = np.digitize(q_valid, edges, right=False) - 1
    local_bins = np.clip(local_bins, 0, int(max_bins) - 1)

    bin_indices: list[np.ndarray] = []
    for bin_idx in range(int(max_bins)):
        indices = valid_indices[np.flatnonzero(local_bins == bin_idx)]
        if indices.size < int(min_points_per_bin):
            continue
        bin_indices.append(indices.astype(int))
    return bin_indices


def log_binned_median_se(
    query_budget: np.ndarray,
    error: np.ndarray,
    *,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_budget = np.asarray(query_budget, dtype=float)
    error = np.asarray(error, dtype=float)

    x_values: list[float] = []
    y_values: list[float] = []
    yerr_values: list[float] = []
    n_values: list[int] = []
    for indices in log_query_bin_indices(
        query_budget,
        error,
        max_bins=max_bins,
        min_points_per_bin=min_points_per_bin,
    ):
        q_bin = query_budget[indices]
        e_bin = error[indices]
        finite = np.isfinite(q_bin) & np.isfinite(e_bin) & (q_bin > 0.0) & (e_bin > 0.0)
        q_bin = q_bin[finite]
        e_bin = e_bin[finite]
        if e_bin.size == 0:
            continue
        x_values.append(float(np.nanmedian(q_bin)))
        y_values.append(float(np.nanmedian(e_bin)))
        yerr_values.append(standard_error(e_bin))
        n_values.append(int(e_bin.size))

    return (
        np.asarray(x_values, dtype=float),
        np.asarray(y_values, dtype=float),
        np.asarray(yerr_values, dtype=float),
        np.asarray(n_values, dtype=int),
    )


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


def select_log_spaced_indices(
    x_values: np.ndarray,
    valid: np.ndarray,
    max_points: int,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) <= int(max_points):
        return valid_indices

    valid_x = x_values[valid_indices]
    if (
        np.all(np.isfinite(valid_x))
        and np.nanmin(valid_x) > 0.0
        and np.nanmax(valid_x) > np.nanmin(valid_x)
    ):
        targets = np.geomspace(float(np.nanmin(valid_x)), float(np.nanmax(valid_x)), num=int(max_points))
        selected: set[int] = {int(valid_indices[0]), int(valid_indices[-1])}
        for target in targets[1:-1]:
            pos = int(np.searchsorted(valid_x, target, side="left"))
            candidates = []
            if 0 <= pos < len(valid_indices):
                candidates.append(pos)
            if 0 <= pos - 1 < len(valid_indices):
                candidates.append(pos - 1)
            if not candidates:
                continue
            nearest = min(candidates, key=lambda idx: abs(float(valid_x[idx]) - float(target)))
            selected.add(int(valid_indices[nearest]))
        return np.asarray(sorted(selected), dtype=int)

    positions = np.linspace(0, len(valid_indices) - 1, num=int(max_points))
    return valid_indices[np.unique(np.rint(positions).astype(int))]


def plot_median_se_errorbar(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    yerr_values: np.ndarray,
    *,
    style: dict,
    label: str,
    cap_fraction: float = ERROR_BAR_CAP_FRACTION,
    linewidth: float = 2.0,
    markersize: float = 5.6,
    elinewidth: float = 1.0,
    capsize: float = 2.8,
    zorder: int = 3,
    markevery: int | None = None,
) -> np.ndarray:
    yerr_capped = cap_log_errorbar(y_values, yerr_values, cap_fraction=cap_fraction)
    ax.errorbar(
        x_values,
        y_values,
        yerr=yerr_capped,
        color=style.get("color"),
        marker=style.get("marker", "o"),
        linewidth=linewidth,
        markersize=markersize,
        elinewidth=elinewidth,
        capsize=capsize,
        label=label,
        zorder=zorder,
        markevery=markevery,
    )
    return yerr_capped


def add_query_scaling_guides(
    ax: plt.Axes,
    guide_points: list[tuple[float, float]] | tuple[np.ndarray, np.ndarray],
    *,
    include_linear: bool = True,
    include_sqrt: bool = True,
    alpha: float = 0.82,
) -> list:
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
        return []

    x0 = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    if not np.isfinite(x0) or not np.isfinite(x_max) or x0 <= 0.0 or x_max <= x0:
        return []
    y0_values = y_values[np.isclose(x_values, x0)]
    y0 = float(np.nanmedian(y0_values)) if y0_values.size else float(np.nanmedian(y_values))
    guide_x = np.geomspace(x0, x_max, num=200)

    handles = []
    if include_linear:
        (line,) = ax.loglog(
            guide_x,
            y0 * (x0 / guide_x),
            color="black",
            linestyle="--",
            linewidth=1.15,
            alpha=alpha,
            label=r"$O(1/N)$",
        )
        handles.append(line)
    if include_sqrt:
        (line,) = ax.loglog(
            guide_x,
            y0 * np.sqrt(x0 / guide_x),
            color="black",
            linestyle=":",
            linewidth=1.35,
            alpha=alpha,
            label=r"$O(1/\sqrt{N})$",
        )
        handles.append(line)
    return handles


def plot_metric_vs_epsilon(
    summary_rows: list[dict],
    a_true: float,
    T: float | None,
    algorithms: tuple[str, ...],
    key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    logy: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for algorithm in algorithms:
        rows = [
            row
            for row in summary_rows
            if row["a_true"] == a_true and row["T"] == T and row["algorithm"] == algorithm
        ]
        rows.sort(key=lambda row: row["epsilon"], reverse=True)
        if not rows:
            continue

        style = _style_for(algorithm)
        ax.plot(
            np.array([row["epsilon"] for row in rows], dtype=float),
            np.array([row[key] for row in rows], dtype=float),
            marker=style["marker"],
            linewidth=2.0,
            markersize=6,
            color=style["color"],
            label=ALGORITHM_LABELS[algorithm],
        )

    ax.invert_xaxis()
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"Target accuracy $\epsilon$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_noise_aware_metric_vs_epsilon(
    summary_rows: list[dict],
    a_true: float,
    T: float | None,
    key: str,
    ylabel: str,
    title: str,
    logy: bool,
    algorithms: tuple[str, ...] = ("biqae", "cabiqae"),
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for algorithm in algorithms:
        rows = [
            row
            for row in summary_rows
            if row["a_true"] == a_true and row["T"] == T and row["algorithm"] == algorithm
        ]
        rows.sort(key=lambda row: row["epsilon"], reverse=True)
        if not rows:
            continue

        style = _style_for(algorithm)
        ax.plot(
            np.array([row["epsilon"] for row in rows], dtype=float),
            np.array([row[key] for row in rows], dtype=float),
            marker=style["marker"],
            linewidth=2.0,
            markersize=6,
            color=style["color"],
            label=ALGORITHM_LABELS[algorithm],
        )

    ax.invert_xaxis()
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"Target accuracy $\epsilon$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()


def plot_noise_aware_metric_vs_T(
    T_values: list[float | None],
    summary_rows: list[dict],
    a_true: float,
    epsilon: float,
    key: str,
    ylabel: str,
    title: str,
    logy: bool,
    algorithms: tuple[str, ...] = ("biqae", "cabiqae"),
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    tick_positions = np.arange(len(T_values), dtype=float)
    tick_labels = [format_T_value(T) for T in T_values]

    for algorithm in algorithms:
        rows_by_T = {
            row["T"]: row
            for row in summary_rows
            if row["a_true"] == a_true and row["epsilon"] == epsilon and row["algorithm"] == algorithm
        }

        x_coords = np.array([idx for idx, T in enumerate(T_values) if T in rows_by_T], dtype=float)
        if x_coords.size == 0:
            continue

        y_values = np.array([rows_by_T[T_values[int(x)]][key] for x in x_coords], dtype=float)
        style = _style_for(algorithm)
        ax.plot(
            x_coords,
            y_values,
            marker=style["marker"],
            linewidth=2.0,
            markersize=6,
            color=style["color"],
            label=ALGORITHM_LABELS[algorithm],
        )

    if logy:
        ax.set_yscale("log")
    ax.set_xticks(tick_positions, tick_labels, rotation=30, ha="right")
    ax.set_xlabel("Noise scale T")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)
    fig.tight_layout()


def show_noise_aware_plots(
    summary_rows: list[dict],
    a_values: list[float],
    epsilons: list[float],
    T_values: list[float | None],
) -> None:
    for a_true in a_values:
        for T in T_values:
            T_label = format_T_value(T)
            for key, ylabel, logy in (
                ("rmse", "RMSE", True),
                ("mean_cost", "Oracle complexity (A-accesses)", True),
                ("coverage", "Coverage", False),
                ("mean_k_max", r"Mean $k_{\max}$", True),
                ("mean_max_depth", "Mean max circuit depth", True),
            ):
                plot_noise_aware_metric_vs_epsilon(
                    summary_rows=summary_rows,
                    a_true=a_true,
                    T=T,
                    key=key,
                    ylabel=ylabel,
                    title=fr"{ylabel} vs $\epsilon$   ($a_{{true}}={a_true:.1f}, T={T_label}$)",
                    logy=logy,
                )

        for epsilon in epsilons:
            for key, ylabel in (
                ("mean_cost", "Oracle complexity (A-accesses)"),
                ("rmse", "RMSE"),
                ("mean_max_depth", "Mean max circuit depth"),
            ):
                plot_noise_aware_metric_vs_T(
                    T_values=T_values,
                    summary_rows=summary_rows,
                    a_true=a_true,
                    epsilon=epsilon,
                    key=key,
                    ylabel=ylabel,
                    title=fr"{ylabel} vs $T$   ($a_{{true}}={a_true:.1f}, \epsilon={epsilon:.1e}$)",
                    logy=True,
                )


def _bootstrap_curve_statistic(
    samples: np.ndarray,
    confidence_level: float,
    statistic: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    finite = np.asarray(samples[np.isfinite(samples)], dtype=float)
    if finite.size == 0:
        return np.nan, np.nan, np.nan

    if statistic == "median":
        reducer = np.median
    elif statistic == "mean":
        reducer = np.mean
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    center = float(reducer(finite))
    if finite.size == 1:
        return center, center, center

    bootstrap_idx = rng.integers(0, finite.size, size=(bootstrap_samples, finite.size))
    bootstrap_values = finite[bootstrap_idx]
    bootstrap_stats = reducer(bootstrap_values, axis=1)
    alpha = 1.0 - float(confidence_level)
    ci_low, ci_high = np.quantile(bootstrap_stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(ci_low), float(ci_high)


def summarize_curve_matrix_with_ci(
    curves: np.ndarray,
    confidence_level: float = 0.95,
    statistic: str = "median",
    bootstrap_samples: int = 1000,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if curves.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_runs, n_points).")

    centers = np.full(curves.shape[1], np.nan, dtype=float)
    ci_lows = np.full(curves.shape[1], np.nan, dtype=float)
    ci_highs = np.full(curves.shape[1], np.nan, dtype=float)
    rng = np.random.default_rng(seed)

    for point_idx in range(curves.shape[1]):
        center, ci_low, ci_high = _bootstrap_curve_statistic(
            curves[:, point_idx],
            confidence_level=confidence_level,
            statistic=statistic,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        centers[point_idx] = center
        ci_lows[point_idx] = ci_low
        ci_highs[point_idx] = ci_high

    return centers, ci_lows, ci_highs


def summarize_curve_matrix_with_median_se(
    curves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if curves.ndim != 2:
        raise ValueError("Expected a 2D array of shape (n_runs, n_points).")

    centers = np.full(curves.shape[1], np.nan, dtype=float)
    se_values = np.full(curves.shape[1], np.nan, dtype=float)
    n_values = np.zeros(curves.shape[1], dtype=int)
    for point_idx in range(curves.shape[1]):
        values = np.asarray(curves[:, point_idx], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        centers[point_idx] = float(np.nanmedian(values))
        se_values[point_idx] = standard_error(values)
        n_values[point_idx] = int(values.size)
    return centers, se_values, n_values


def plot_query_benchmark_with_confidence_bands(
    query_grid: np.ndarray,
    curves_by_algorithm: dict[str, np.ndarray],
    algorithms: tuple[str, ...] | list[str],
    algorithm_labels: dict[str, str],
    algorithm_styles: dict[str, dict[str, str]],
    output_path: Path | str,
    ylabel: str,
    title: str,
    confidence_level: float = 0.95,
    statistic: str = "median",
    bootstrap_samples: int = 1000,
    seed: int = 1234,
    show: bool = False,
) -> None:
    del confidence_level, statistic, bootstrap_samples, seed
    query_grid = np.asarray(query_grid, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in algorithms:
        if algorithm not in curves_by_algorithm:
            continue

        centers, se_values, _ = summarize_curve_matrix_with_median_se(
            np.asarray(curves_by_algorithm[algorithm], dtype=float)
        )

        style = algorithm_styles[algorithm]
        valid_center = np.isfinite(centers) & (centers > 0.0)
        if np.any(valid_center):
            plot_median_se_errorbar(
                ax,
                query_grid[valid_center],
                centers[valid_center],
                se_values[valid_center],
                style=style,
                markevery=8,
                label=algorithm_labels.get(algorithm, algorithm),
                linewidth=1.8,
                markersize=4,
            )

    reference_queries = query_grid[np.isfinite(query_grid) & (query_grid > 0.0)]
    if reference_queries.size == 0:
        raise ValueError("query_grid must contain at least one positive finite value.")
    reference_anchor = float(reference_queries[0])
    sqrt_reference = 1.0 / np.sqrt(reference_queries)
    # Anchor the 1/N guide to the same leftmost point as the 1/sqrt(N) guide.
    linear_reference = np.sqrt(reference_anchor) / reference_queries

    ax.loglog(
        reference_queries,
        sqrt_reference,
        "--",
        color="gray",
        alpha=0.6,
        label=r"$\mathcal{O}(1/\sqrt{N_q})$",
    )
    ax.loglog(
        reference_queries,
        linear_reference,
        "-.",
        color="black",
        alpha=0.5,
        label=r"$\mathcal{O}(1/N_q)$",
    )

    ax.set_xlabel("Common query budget")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    legend_ncol = min(len(labels), 4) if labels else 1
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=legend_ncol,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.2)
    if show:
        plt.show()
    plt.close(fig)
