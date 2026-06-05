"""Publication-style calibration plots for hardware CVA AE runs.

This module reads the calibration artifacts produced by the hardware runner:
`amplification_points.csv`, `amplification_counts.csv`, and
`calibration_summary.json`.

It creates three paper-oriented views:
- amplified good-state probability versus Grover amplification factor;
- fitted contrast decay and visibility diagnostics;
- optional measured bitstring-distribution heatmap.

The plotting code is intentionally separate from the runner so figures can be
rebuilt from saved artifacts without rerunning hardware jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, PercentFormatter


BITSTRINGS_3Q = [format(i, "03b") for i in range(8)]

# Okabe--Ito inspired palette, readable in print and color-blind friendly.
COLORS = {
    "black": "#111111",
    "grey": "#7A7A7A",
    "light_grey": "#D9D9D9",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "savefig.transparent": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.7,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.55,
            "grid.color": "#9A9A9A",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.fancybox": False,
            "legend.edgecolor": "#D0D0D0",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _sorted_points(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: _f(row.get("amplification_factor", row.get("grover_power"))))


def _get_arrays(points: list[dict[str, str]]) -> dict[str, np.ndarray]:
    # Normalize the CSV string values into numeric arrays once, then keep all
    # plotting code expressed in terms of those arrays.
    rows = _sorted_points(points)
    return {
        "k": np.asarray([_f(r.get("grover_power")) for r in rows], dtype=float),
        "K": np.asarray([_f(r.get("amplification_factor")) for r in rows], dtype=float),
        "p_ideal": np.asarray([_f(r.get("p_ideal")) for r in rows], dtype=float),
        "p_raw": np.asarray([_f(r.get("p_raw")) for r in rows], dtype=float),
        "p_raw_se": np.asarray([_f(r.get("p_raw_se"), 0.0) for r in rows], dtype=float),
        "p_hw": np.asarray([_f(r.get("p_hw_mitigated")) for r in rows], dtype=float),
        "p_hw_se": np.asarray([_f(r.get("p_hw_mitigated_se"), 0.0) for r in rows], dtype=float),
        "contrast": np.asarray([_f(r.get("contrast_mitigated")) for r in rows], dtype=float),
        "contrast_se": np.asarray([_f(r.get("contrast_mitigated_se")) for r in rows], dtype=float),
        "z": np.asarray([_f(r.get("contrast_signal_z")) for r in rows], dtype=float),
        "used": np.asarray([_b(r.get("used_in_fit")) for r in rows], dtype=bool),
        "visible": np.asarray([_b(r.get("visible_by_contrast")) for r in rows], dtype=bool),
    }


def _theta_from_points(K: np.ndarray, p_ideal: np.ndarray) -> float:
    finite = np.isfinite(K) & np.isfinite(p_ideal) & (K > 0.0)
    if not np.any(finite):
        return float("nan")
    # Prefer K = 1, because p_ideal(K=1) = a = sin²(theta).
    idx_one = np.where(finite & np.isclose(K, 1.0))[0]
    idx = int(idx_one[0]) if idx_one.size else int(np.where(finite)[0][0])
    return float(np.arcsin(np.sqrt(np.clip(p_ideal[idx], 0.0, 1.0))) / K[idx])


def _ideal_probability(K: np.ndarray, theta: float) -> np.ndarray:
    return np.sin(K * theta) ** 2


def _model_probability(K: np.ndarray, theta: float, summary: dict[str, Any]) -> np.ndarray:
    # Contrast model used only for visualization.  The fitted parameters are
    # produced by the runner and saved in calibration_summary.json.
    baseline = _f(summary.get("contrast_baseline"), 0.125)
    prefactor = _f(summary.get("contrast_prefactor"))
    slope = _f(summary.get("free_intercept_slope"))
    t_zero = _f(summary.get("t_eff_zero_intercept"))

    q = _ideal_probability(K, theta)
    if np.isfinite(prefactor) and prefactor > 0.0 and np.isfinite(slope):
        contrast = prefactor * np.exp(slope * K)
    elif np.isfinite(t_zero) and t_zero > 0.0:
        contrast = np.exp(-K / t_zero)
    else:
        return np.full_like(K, np.nan, dtype=float)

    return baseline + contrast * (q - baseline)


def _finite_ylim(*arrays: np.ndarray, pad: float = 0.06, lo: float = -0.02, hi: float = 1.02) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return lo, hi
    ymin, ymax = float(values.min()), float(values.max())
    span = max(ymax - ymin, 0.08)
    return max(lo, ymin - pad * span), min(hi, ymax + pad * span)


def _line_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str,
    zorder: int = 3,
    alpha: float = 0.16,
    ci: float = 1.96,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    x, y = x[finite], y[finite]
    se = np.nan_to_num(se[finite], nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(x)
    x, y, se = x[order], y[order], se[order]
    lower = np.clip(y - ci * se, 0.0, 1.0)
    upper = np.clip(y + ci * se, 0.0, 1.0)
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0.0, zorder=zorder - 2)
    ax.plot(x, y, color=color, linewidth=1.65, marker=marker, markersize=4.2, label=label, zorder=zorder)


def _annotate_panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.982,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def plot_probability_panel(
    points: list[dict[str, str]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    # Main calibration figure: ideal amplification, raw hardware, mitigated
    # hardware, fitted noisy model, and residuals.
    data = _get_arrays(points)
    k = data["k"]
    K = data["K"]
    p_ideal = data["p_ideal"]
    p_raw = data["p_raw"]
    p_raw_se = data["p_raw_se"]
    p_hw = data["p_hw"]
    p_hw_se = data["p_hw_se"]
    used = data["used"]
    visible = data["visible"]

    baseline = _f(summary.get("contrast_baseline"), 0.125)
    theta = _theta_from_points(K, p_ideal)
    k_max = int(np.nanmax(k)) if np.any(np.isfinite(k)) else 0
    K_grid = np.linspace(1.0, max(float(np.nanmax(K)), 1.0), 800)
    k_ticks = np.arange(0, k_max + 1, dtype=int)
    K_ticks = 2 * k_ticks + 1

    ideal_curve = _ideal_probability(K_grid, theta) if np.isfinite(theta) else np.full_like(K_grid, np.nan)
    model_curve = _model_probability(K_grid, theta, summary) if np.isfinite(theta) else np.full_like(K_grid, np.nan)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.15, 5.85),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.075},
        constrained_layout=False,
    )

    ax = axes[0]
    ax.plot(
        K_grid,
        ideal_curve,
        color=COLORS["black"],
        linewidth=2.05,
        label=r"Ideal amplification, $\sin^2(K\theta)$",
        zorder=2,
    )
    if np.any(np.isfinite(model_curve)):
        ax.plot(
            K_grid,
            model_curve,
            color=COLORS["orange"],
            linewidth=1.85,
            linestyle="-.",
            label="Fitted contrast model",
            zorder=2,
        )
    _line_band(
        ax,
        K,
        p_hw,
        p_hw_se,
        color=COLORS["blue"],
        label="Readout-mitigated hardware",
        marker="o",
        zorder=5,
        alpha=0.18,
    )
    _line_band(
        ax,
        K,
        p_raw,
        p_raw_se,
        color=COLORS["red"],
        label="Raw hardware",
        marker="D",
        zorder=4,
        alpha=0.10,
    )

    ax.axhline(
        baseline,
        color=COLORS["grey"],
        linestyle=(0, (4.0, 2.0)),
        linewidth=1.15,
        label=rf"Baseline $b={baseline:.4f}$",
        zorder=1,
    )
    if np.any(visible):
        ax.scatter(
            K[visible],
            p_hw[visible],
            s=98,
            facecolors="none",
            edgecolors=COLORS["green"],
            linewidths=1.45,
            label="Visible by contrast",
            zorder=7,
        )
    if np.any(used):
        ax.scatter(
            K[used],
            p_hw[used],
            s=33,
            facecolors=COLORS["green"],
            edgecolors="white",
            linewidths=0.65,
            label="Used in fit",
            zorder=8,
        )

    ax.set_ylabel(r"Success probability, $P(111)$")
    ax.set_title("Hardware calibration of the amplified CVA oracle", pad=7)
    ax.set_ylim(_finite_ylim(ideal_curve, model_curve, p_hw + 1.96 * p_hw_se, p_raw + 1.96 * p_raw_se))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper right", ncol=1, borderpad=0.35, handlelength=2.4)
    _annotate_panel(ax, "a")

    residual = p_hw - p_ideal
    axes[1].axhline(0.0, color=COLORS["black"], linewidth=0.9, zorder=1)
    axes[1].fill_between(
        K,
        residual - 1.96 * np.maximum(p_hw_se, 0.0),
        residual + 1.96 * np.maximum(p_hw_se, 0.0),
        color=COLORS["purple"],
        alpha=0.16,
        linewidth=0.0,
        zorder=2,
    )
    axes[1].plot(
        K,
        residual,
        color=COLORS["purple"],
        marker="o",
        markersize=4.0,
        linewidth=1.45,
        zorder=3,
    )
    axes[1].set_xlabel(r"Amplification factor $K=2k+1$")
    axes[1].set_ylabel(r"$p_{\rm hw}-p_{\rm ideal}$")
    axes[1].set_xticks(K_ticks)
    axes[1].set_xticklabels([f"{int(x)}\n$k={int(kk)}$" for x, kk in zip(K_ticks, k_ticks)])
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].yaxis.set_minor_locator(AutoMinorLocator(2))
    _annotate_panel(axes[1], "b")

    _save(fig, output_dir / "paper_calibration_probability")


def plot_contrast_fit(
    points: list[dict[str, str]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    # This panel explains why the runner selected the effective T_eff and the
    # visible k cap used by replay and direct live execution.
    data = _get_arrays(points)
    K = data["K"]
    contrast = data["contrast"]
    contrast_se = data["contrast_se"]
    z = data["z"]
    used = data["used"]

    finite = np.isfinite(K) & np.isfinite(contrast) & (contrast > 0.0) & np.isfinite(contrast_se)
    excluded = finite & ~used
    used_valid = finite & used

    K_grid = np.linspace(max(0.0, float(np.nanmin(K)) - 0.25), float(np.nanmax(K)) + 0.35, 600)
    slope_free = _f(summary.get("free_intercept_slope"))
    prefactor = _f(summary.get("contrast_prefactor"))
    t_zero = _f(summary.get("t_eff_zero_intercept"))

    fit_free = np.full_like(K_grid, np.nan, dtype=float)
    fit_zero = np.full_like(K_grid, np.nan, dtype=float)
    if np.isfinite(prefactor) and prefactor > 0.0 and np.isfinite(slope_free):
        fit_free = prefactor * np.exp(slope_free * K_grid)
    if np.isfinite(t_zero) and t_zero > 0.0:
        fit_zero = np.exp(-K_grid / t_zero)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.65),
        gridspec_kw={"width_ratios": [2.15, 1.0], "wspace": 0.24},
    )

    ax = axes[0]
    if np.any(excluded):
        ax.errorbar(
            K[excluded],
            contrast[excluded],
            yerr=1.96 * np.maximum(contrast_se[excluded], 0.0),
            color=COLORS["grey"],
            marker="o",
            linestyle="None",
            markersize=4.3,
            capsize=2.5,
            elinewidth=0.9,
            label="Valid, excluded",
            zorder=3,
        )
    if np.any(used_valid):
        ax.errorbar(
            K[used_valid],
            contrast[used_valid],
            yerr=1.96 * np.maximum(contrast_se[used_valid], 0.0),
            color=COLORS["blue"],
            marker="s",
            linestyle="None",
            markersize=4.7,
            capsize=2.8,
            elinewidth=1.0,
            label="Used in fit",
            zorder=4,
        )

    if np.any(np.isfinite(fit_free)):
        ax.plot(K_grid, fit_free, color=COLORS["orange"], linewidth=1.85, label="Free-intercept fit")
    if np.any(np.isfinite(fit_zero)):
        ax.plot(K_grid, fit_zero, color=COLORS["green"], linewidth=1.55, linestyle="--", label="Zero-intercept fit")

    ax.set_yscale("log")
    ax.set_xlabel(r"Amplification factor $K=2k+1$")
    ax.set_ylabel("Estimated contrast")
    ax.set_title("Depth-dependent contrast decay", pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.legend(loc="upper right", borderpad=0.35, handlelength=2.25)
    _annotate_panel(ax, "a")

    # Reliability box: this is intentionally visible, because two-point fits should be flagged.
    fit_points = int(_f(summary.get("fit_points"), 0))
    text = (
        rf"$b={_f(summary.get('contrast_baseline')):.5f}$" "\n"
        rf"$T_{{\rm eff}}^{{(0)}}={_f(summary.get('t_eff_zero_intercept')):.3g}$" "\n"
        rf"$T_{{\rm eff}}^{{\rm free}}={_f(summary.get('t_eff_free_intercept')):.3g}$" "\n"
        rf"fit points $={fit_points}$" "\n"
        rf"$k_{{\rm visible}}={int(_f(summary.get('k_visible'), 0))}$"
    )
    ax.text(
        0.035,
        0.045,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#CFCFCF", "alpha": 0.96},
    )

    ax2 = axes[1]
    order = np.argsort(K)
    K_ord = K[order]
    z_ord = z[order]
    used_ord = used[order]
    bars = ax2.bar(
        K_ord,
        z_ord,
        color=[COLORS["blue"] if flag else COLORS["light_grey"] for flag in used_ord],
        edgecolor="#FFFFFF",
        linewidth=0.7,
        width=0.72,
        zorder=3,
    )
    for bar, val in zip(bars, z_ord):
        if np.isfinite(val):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                max(0.0, bar.get_height()) + 0.35,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.4,
                rotation=0,
            )
    ax2.axhline(
        _f(summary.get("min_fit_contrast_z"), 1.0),
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.05,
        label="Fit threshold",
        zorder=2,
    )
    ax2.axhline(
        _f(summary.get("min_visible_contrast_z"), 2.0),
        color=COLORS["green"],
        linestyle=":",
        linewidth=1.25,
        label="Visibility threshold",
        zorder=2,
    )
    ax2.set_xlabel(r"$K$")
    ax2.set_ylabel("Contrast z-score")
    ax2.set_title("Visibility", pad=6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax2.legend(loc="upper right", borderpad=0.3, handlelength=1.8, fontsize=7.7)
    _annotate_panel(ax2, "b")

    _save(fig, output_dir / "paper_contrast_fit")


def _aggregate_bitstrings(count_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    by_k: dict[int, defaultdict[str, int]] = {}
    shots_by_k: defaultdict[int, int] = defaultdict(int)
    for row in count_rows:
        k = int(_f(row.get("grover_power"), 0))
        by_k.setdefault(k, defaultdict(int))
        try:
            counts = json.loads(row.get("counts_json", "{}"))
        except json.JSONDecodeError:
            counts = {}
        for bitstring, count in counts.items():
            cleaned = str(bitstring).replace(" ", "")
            by_k[k][cleaned] += int(count)
            shots_by_k[k] += int(count)
    ks = np.asarray(sorted(by_k), dtype=int)
    probs = np.zeros((len(ks), len(BITSTRINGS_3Q)), dtype=float)
    for i, k in enumerate(ks):
        shots = max(int(shots_by_k[k]), 1)
        for j, bitstring in enumerate(BITSTRINGS_3Q):
            probs[i, j] = by_k[k].get(bitstring, 0) / shots
    return ks, probs


def plot_bitstring_heatmap(
    count_rows: list[dict[str, str]],
    points: list[dict[str, str]],
    output_dir: Path,
) -> None:
    # Optional count-level diagnostic.  It verifies that the good-state mass is
    # not hiding a broader objective-register distribution issue.
    ks, probs = _aggregate_bitstrings(count_rows)
    if ks.size == 0:
        return

    point_by_k = {int(_f(r.get("grover_power"), 0)): r for r in points}
    good_prob = np.asarray([_f(point_by_k.get(int(k), {}).get("p_raw"), np.nan) for k in ks], dtype=float)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.65),
        gridspec_kw={"width_ratios": [2.25, 1.0], "wspace": 0.23},
    )

    vmax = max(0.18, float(np.nanpercentile(probs, 98)))
    im = axes[0].imshow(probs, aspect="auto", cmap="magma_r", vmin=0.0, vmax=vmax)
    axes[0].set_xticks(np.arange(len(BITSTRINGS_3Q)))
    axes[0].set_xticklabels(BITSTRINGS_3Q, rotation=0)
    axes[0].set_yticks(np.arange(len(ks)))
    axes[0].set_yticklabels([rf"$k={int(k)}$" for k in ks])
    axes[0].set_xlabel("Measured objective bitstring")
    axes[0].set_ylabel("Grover power")
    axes[0].set_title("Three-ancilla output distribution", pad=6)
    axes[0].grid(False)
    _annotate_panel(axes[0], "a")

    cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.025)
    cbar.set_label("Empirical probability")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    good_idx = BITSTRINGS_3Q.index("111")
    y = np.arange(len(ks))
    axes[1].barh(y, probs[:, good_idx], color=COLORS["blue"], edgecolor="white", linewidth=0.6)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([rf"$k={int(k)}$" for k in ks])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"Raw $P(111)$")
    axes[1].set_title("Good-state mass", pad=6)
    axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    xmax = min(1.0, max(0.18, float(np.nanmax(probs[:, good_idx]) + 0.04)))
    axes[1].set_xlim(0.0, xmax)
    for i, val in enumerate(good_prob):
        if np.isfinite(val):
            axes[1].text(min(val + 0.006, xmax * 0.98), i, f"{val:.3f}", va="center", fontsize=7.7)
    _annotate_panel(axes[1], "b")

    _save(fig, output_dir / "paper_bitstring_distribution")


class HardwareCvaCalibrationPlotter:
    """Builds publication-style calibration plots from hardware run artifacts."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        output_dir: str | Path | None = None,
        require_counts: bool = False,
    ) -> None:
        self.run_dir: Path = Path(run_dir).expanduser().resolve()
        self.output_dir: Path = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else self.run_dir / "plots" / "paper"
        )
        self.require_counts: bool = bool(require_counts)

    def run(self) -> Path:
        # All inputs are read from disk, so these plots are reproducible from a
        # saved run directory without any backend access.
        points: list[dict[str, str]] = _read_csv(
            self.run_dir / "amplification_points.csv"
        )
        counts: list[dict[str, str]] = _read_csv(
            self.run_dir / "amplification_counts.csv"
        )
        summary: dict[str, Any] = _read_json(
            self.run_dir / "calibration_summary.json"
        )

        if not points:
            raise FileNotFoundError(
                f"No amplification_points.csv found in {self.run_dir}"
            )
        if self.require_counts and not counts:
            raise FileNotFoundError(
                f"No amplification_counts.csv found in {self.run_dir}"
            )

        _style()
        plot_probability_panel(points, summary, self.output_dir)
        plot_contrast_fit(points, summary, self.output_dir)
        if counts:
            plot_bitstring_heatmap(counts, points, self.output_dir)
        else:
            print(
                "Skipping bitstring heatmap: amplification_counts.csv not found or empty."
            )
        return self.output_dir


def plot_calibration_paper(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    require_counts: bool = False,
) -> Path:
    plotter: HardwareCvaCalibrationPlotter = HardwareCvaCalibrationPlotter(
        run_dir,
        output_dir=output_dir,
        require_counts=require_counts,
    )
    return plotter.run()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality calibration plots for a hardware CVA AE run."
    )
    parser.add_argument("--run-dir", required=True, help="Directory containing amplification_points.csv and calibration_summary.json.")
    parser.add_argument("--output-dir", default=None, help="Directory where PDF/PNG figures will be written.")
    parser.add_argument(
        "--require-counts",
        action="store_true",
        help="Raise an error if amplification_counts.csv is missing. By default the bitstring figure is skipped.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_dir: Path = plot_calibration_paper(
        args.run_dir,
        output_dir=args.output_dir,
        require_counts=bool(args.require_counts),
    )
    print(f"Wrote paper calibration plots to {output_dir}")


if __name__ == "__main__":
    main()
