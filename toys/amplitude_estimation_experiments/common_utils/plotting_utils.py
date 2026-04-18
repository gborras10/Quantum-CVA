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
    ax.set_title(title)
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
    ax.set_title(title)
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
    ax.set_title(title)
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