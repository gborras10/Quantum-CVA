from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experiment_utils import ALGORITHM_LABELS, format_T_value


STYLE = {
    "biqae": {"color": "#A23B72", "marker": "o"},
    "cabiqae": {"color": "#1F6F8B", "marker": "s"},
    "cabiqae_no_cap": {"color": "#1F6F8B", "marker": "s"},
    "cabiqae_with_cap": {"color": "#E07A5F", "marker": "^"},
}


def _style_for(algorithm: str) -> dict[str, str]:
    return STYLE.get(algorithm, {"color": "#333333", "marker": "o"})


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
    query_grid = np.asarray(query_grid, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))

    for alg_idx, algorithm in enumerate(algorithms):
        if algorithm not in curves_by_algorithm:
            continue

        centers, ci_lows, ci_highs = summarize_curve_matrix_with_ci(
            np.asarray(curves_by_algorithm[algorithm], dtype=float),
            confidence_level=confidence_level,
            statistic=statistic,
            bootstrap_samples=bootstrap_samples,
            seed=seed + alg_idx,
        )

        style = algorithm_styles[algorithm]
        valid_center = np.isfinite(centers) & (centers > 0.0)
        valid_band = (
            valid_center
            & np.isfinite(ci_lows)
            & np.isfinite(ci_highs)
            & (ci_lows > 0.0)
            & (ci_highs > 0.0)
        )

        if np.any(valid_band):
            ax.fill_between(
                query_grid[valid_band],
                ci_lows[valid_band],
                ci_highs[valid_band],
                color=style["color"],
                alpha=0.14,
                linewidth=0.0,
                zorder=1,
            )

        if np.any(valid_center):
            ax.loglog(
                query_grid[valid_center],
                centers[valid_center],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4,
                markevery=8,
                label=algorithm_labels.get(algorithm, algorithm),
            )

    ax.loglog(
        query_grid,
        1.0 / np.sqrt(query_grid),
        "--",
        color="gray",
        alpha=0.6,
        label=r"$\mathcal{O}(1/\sqrt{N_q})$",
    )
    ax.loglog(
        query_grid,
        3.0 / query_grid,
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
