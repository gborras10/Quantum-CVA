from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogFormatterMathtext, LogLocator


BLUE = "#4c78a8"
PINK = "#d65f9e"
ORANGE = "#ff8c00"
TARGET_BLUE = "#4169e1"
BAR_WIDTH = 0.68


PAPER_RC = {
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "Latin Modern Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
    "axes.labelsize": 16,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.15,
    "lines.solid_capstyle": "round",
}


@dataclass(frozen=True)
class CrcaPaperPlotConfig:
    output_dir: Path
    data_path: Path
    objective_label: str
    trained_label: str
    target_label: str
    cost_stem: str = "cost_evol"
    histogram_stem: str = "histogram"
    combined_stem: str | None = None
    histogram_y_label: str = "Probability (measured)"
    trained_key: str = "f_star_statevector"


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=350)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", dpi=350)
    plt.close(fig)


def _paper_axes(
    ax: plt.Axes,
    *,
    log_y: bool = False,
    grid_x: bool = True,
) -> None:
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    if log_y:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.minorticks_on()
    grid_axis = "both" if grid_x else "y"
    ax.grid(
        True,
        which="major",
        axis=grid_axis,
        color="#c7c7c7",
        linewidth=0.75,
        alpha=0.72,
    )
    ax.grid(
        True,
        which="minor",
        axis=grid_axis,
        color="#e4e4e4",
        linewidth=0.42,
        alpha=0.88,
    )
    if not grid_x:
        ax.grid(False, which="both", axis="x")
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5.5,
        width=1.0,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=3.0,
        width=0.8,
        top=False,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(1.15)


def _basis_ticks(size: int) -> np.ndarray:
    if size <= 16:
        return np.arange(size)
    return np.unique(np.r_[np.arange(0, size, 8), size - 1])


def _plot_cost_on_axis(
    ax: plt.Axes,
    cost: np.ndarray,
    best: np.ndarray,
    objective_label: str,
) -> None:
    x = np.arange(cost.size)
    ax.semilogy(
        x,
        best,
        color=BLUE,
        linewidth=2.5,
        marker="o",
        markersize=3.4,
        markevery=max(1, cost.size // 24),
        label="best-so-far",
    )
    ax.semilogy(
        x,
        cost,
        linestyle="None",
        marker=".",
        markersize=3.0,
        color=PINK,
        alpha=0.9,
        label="cost",
    )
    ax.set_xlabel("iterations")
    ax.set_ylabel(objective_label)
    _paper_axes(ax, log_y=True)
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)


def _plot_histogram_on_axis(
    ax: plt.Axes,
    target: np.ndarray,
    trained: np.ndarray,
    config: CrcaPaperPlotConfig,
) -> None:
    x = np.arange(target.size)
    ax.bar(
        x,
        trained,
        width=BAR_WIDTH,
        color=ORANGE,
        edgecolor="#9a4f00",
        linewidth=0.25,
        alpha=0.95,
        zorder=2,
        label=config.trained_label,
    )
    ax.bar(
        x,
        target,
        width=BAR_WIDTH,
        color=TARGET_BLUE,
        edgecolor="#1f3f99",
        linewidth=0.25,
        alpha=0.52,
        zorder=3,
        label=config.target_label,
    )
    ax.set_xlabel(r"Computational basis state $x$")
    ax.set_ylabel(config.histogram_y_label)
    ax.set_xticks(_basis_ticks(target.size))
    _paper_axes(ax, grid_x=False)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)


def _arrays(data: np.lib.npyio.NpzFile, trained_key: str) -> tuple[np.ndarray, ...]:
    cost = np.asarray(data["cost_history"], dtype=float)
    best = np.asarray(data["best_so_far"], dtype=float)
    target = np.asarray(data["f_target"], dtype=float)
    trained = np.asarray(data[trained_key], dtype=float)
    return cost, best, target, trained


def plot_cost_evolution(data: np.lib.npyio.NpzFile, config: CrcaPaperPlotConfig) -> None:
    cost, best, _, _ = _arrays(data, config.trained_key)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        _plot_cost_on_axis(ax, cost, best, config.objective_label)
        fig.tight_layout(pad=0.45)
        _save(fig, config.output_dir, config.cost_stem)


def plot_histogram(data: np.lib.npyio.NpzFile, config: CrcaPaperPlotConfig) -> None:
    _, _, target, trained = _arrays(data, config.trained_key)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        _plot_histogram_on_axis(ax, target, trained, config)
        fig.tight_layout(pad=0.45)
        _save(fig, config.output_dir, config.histogram_stem)


def plot_combined(data: np.lib.npyio.NpzFile, config: CrcaPaperPlotConfig) -> None:
    if config.combined_stem is None:
        return

    cost, best, target, trained = _arrays(data, config.trained_key)
    combined_rc = {
        **PAPER_RC,
        "axes.labelsize": 13,
        "legend.fontsize": 8.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
    }

    with plt.rc_context(combined_rc):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.6, 4.9),
            gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.22},
        )
        _plot_cost_on_axis(axes[0], cost, best, config.objective_label)
        _plot_histogram_on_axis(axes[1], target, trained, config)
        fig.tight_layout(pad=0.55)
        _save(fig, config.output_dir, config.combined_stem)


def generate_crca_paper_plots(config: CrcaPaperPlotConfig) -> None:
    data = np.load(config.data_path, allow_pickle=True)
    plot_cost_evolution(data, config)
    plot_histogram(data, config)
    plot_combined(data, config)
