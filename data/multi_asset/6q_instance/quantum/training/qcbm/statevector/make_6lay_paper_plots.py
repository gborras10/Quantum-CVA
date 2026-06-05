from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogFormatterMathtext, LogLocator


BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_qcbm_heavyhex6_6lay.npz"

BLUE = "#4c78a8"
PINK = "#d65f9e"
ORANGE = "#ff8c00"
TARGET_BLUE = "#4169e1"


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


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(BASE_DIR / f"{stem}.png", bbox_inches="tight", dpi=350)
    fig.savefig(BASE_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=350)
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


def _training_kl(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(data["p_target"], dtype=float)
    target_entropy = -float(np.sum(target[target > 0.0] * np.log(target[target > 0.0])))
    kl_history = np.maximum(np.asarray(data["cost_history"], dtype=float) - target_entropy, 1e-15)
    best_kl_history = np.minimum.accumulate(kl_history)
    x = np.arange(kl_history.size)
    return x, kl_history, best_kl_history


def plot_kl_trajectory(data: np.lib.npyio.NpzFile) -> None:
    x, kl_history, best_kl_history = _training_kl(data)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.semilogy(
            x,
            best_kl_history,
            color=BLUE,
            linewidth=2.5,
            marker="o",
            markersize=3.4,
            markevery=45,
            label="best-so-far",
        )
        ax.semilogy(
            x,
            kl_history,
            linestyle="None",
            marker=".",
            markersize=3.0,
            color=PINK,
            alpha=0.9,
            label="cost",
        )
        ax.set_xlabel("iterations")
        ax.set_ylabel(r"$\mathcal{L}_{\mathrm{QCBM}}$")
        _paper_axes(ax, log_y=True)
        ax.legend(frameon=False, loc="upper right", handlelength=2.4)
        fig.tight_layout(pad=0.45)
        _save(fig, "cost_evol_6lay")


def plot_distribution_histogram(data: np.lib.npyio.NpzFile) -> None:
    target = np.asarray(data["p_target"], dtype=float)
    trained = np.asarray(data["p_star"], dtype=float)
    x = np.arange(target.size)
    width = 0.68
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.bar(
            x,
            trained,
            width=width,
            color=ORANGE,
            edgecolor="#9a4f00",
            linewidth=0.25,
            alpha=0.95,
            zorder=2,
            label=r"$P_{\theta}$",
        )
        ax.bar(
            x,
            target,
            width=width,
            color=TARGET_BLUE,
            edgecolor="#1f3f99",
            linewidth=0.25,
            alpha=0.52,
            zorder=3,
            label=r"$P_{\mathrm{target}}$",
        )
        ax.set_xlabel(r"Computational basis state $x$")
        ax.set_ylabel(r"$P_{\mathrm{target}}(x)$")
        ticks = np.r_[np.arange(0, target.size, 8), target.size - 1]
        ax.set_xticks(np.unique(ticks))
        _paper_axes(ax, grid_x=False)
        ax.legend(frameon=False, loc="upper right", handlelength=1.8)
        fig.tight_layout(pad=0.45)
        _save(fig, "histogram_6lay")


def plot_combined(data: np.lib.npyio.NpzFile) -> None:
    x, kl_history, best_kl_history = _training_kl(data)
    target = np.asarray(data["p_target"], dtype=float)
    trained = np.asarray(data["p_star"], dtype=float)
    states = np.arange(target.size)
    width = 0.68

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

        axes[0].semilogy(
            x,
            best_kl_history,
            color=BLUE,
            linewidth=2.5,
            marker="o",
            markersize=3.4,
            markevery=45,
            label="best-so-far",
        )
        axes[0].semilogy(
            x,
            kl_history,
            linestyle="None",
            marker=".",
            markersize=3.0,
            color=PINK,
            alpha=0.9,
            label="cost",
        )
        axes[0].set_xlabel("iterations")
        axes[0].set_ylabel(r"$\mathcal{L}_{\mathrm{QCBM}}(\theta)$")
        _paper_axes(axes[0], log_y=True)
        axes[0].legend(frameon=False, loc="upper right", handlelength=2.4)

        axes[1].bar(
            states,
            trained,
            width=width,
            color=ORANGE,
            edgecolor="#9a4f00",
            linewidth=0.25,
            alpha=0.95,
            zorder=2,
            label=r"$P_{\theta}$",
        )
        axes[1].bar(
            states,
            target,
            width=width,
            color=TARGET_BLUE,
            edgecolor="#1f3f99",
            linewidth=0.25,
            alpha=0.52,
            zorder=3,
            label=r"$P_{\mathrm{target}}$",
        )
        axes[1].set_xlabel(r"Computational basis state $x$")
        axes[1].set_ylabel(r"$\mathrm{Probability~(measured)}$", fontweight="normal")
        ticks = np.r_[np.arange(0, target.size, 8), target.size - 1]
        axes[1].set_xticks(np.unique(ticks))
        _paper_axes(axes[1], grid_x=False)
        axes[1].legend(frameon=False, loc="upper right", handlelength=1.8)

        fig.tight_layout(pad=0.55)
        _save(fig, "qcbm_6lay_training_and_distribution")


def main() -> None:
    data = np.load(DATA_PATH, allow_pickle=True)
    plot_kl_trajectory(data)
    plot_distribution_histogram(data)
    plot_combined(data)


if __name__ == "__main__":
    main()
