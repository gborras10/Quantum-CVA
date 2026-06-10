from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "plots"
TOY_RUN_DIR = (
    ROOT
    / "toys"
    / "amplitude_estimation_experiments"
    / "noise_aware_regime"
    / "3qubit_toy"
    / "hardware"
    / "beta_hardware_experiment"
    / "experiment_results"
    / "csv_results"
)
CVA_RUN_DIR = (
    ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "cva_pricing_multi_asset"
    / "quantum"
    / "ae_cva"
    / "hardware"
    / "results"
    / "q-ctrl_hardware_basquecountry"
)

BLUE = "#1F77B4"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D7D7D7"
CONTRAST_Y_MIN = -0.08
CONTRAST_Y_MAX = 1.14


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.9,
            "axes.labelsize": 12,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 12,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def read_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    points = pd.read_csv(run_dir / "amplification_points.csv").sort_values("grover_power")
    summary = json.loads((run_dir / "calibration_summary.json").read_text(encoding="utf-8"))
    return points, summary


def grover_probability(k: np.ndarray, a0: float) -> np.ndarray:
    theta = np.arcsin(np.sqrt(np.clip(float(a0), 0.0, 1.0)))
    return np.sin((2.0 * k + 1.0) * theta) ** 2


def grover_power(amplification_factor: np.ndarray | pd.Series | float) -> np.ndarray:
    return (np.asarray(amplification_factor, dtype=float) - 1.0) / 2.0


def add_grid(ax: plt.Axes) -> None:
    ax.minorticks_on()
    ax.grid(True, which="major", color=LIGHT_GRAY, linewidth=0.48, alpha=0.62)
    ax.grid(True, which="minor", color=LIGHT_GRAY, linewidth=0.25, alpha=0.30)


def draw_amplification_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    *,
    baseline: float,
    x_padding: float,
) -> tuple[object, object]:
    k_min = float(points["grover_power"].min())
    k_max = float(points["grover_power"].max())
    k_grid = np.linspace(k_min, k_max, 2200)
    a0 = float(points.loc[points["grover_power"].idxmin(), "p_ideal"])

    ideal = ax.plot(
        k_grid,
        grover_probability(k_grid, a0),
        color="black",
        linewidth=1.15,
        label="Ideal Grover oscillation",
        zorder=1,
    )[0]
    hardware = ax.errorbar(
        points["grover_power"],
        points["p_hw_mitigated"],
        yerr=points["p_hw_mitigated_se"],
        fmt="o",
        ms=3.7,
        mfc="white",
        mec=BLUE,
        mew=1.0,
        ecolor=BLUE,
        color=BLUE,
        elinewidth=0.8,
        capsize=2.3,
        capthick=0.8,
        label="Hardware mitigated",
        zorder=3,
    )
    ax.axhline(
        baseline,
        color=GRAY,
        linestyle=(0, (2.5, 2.5)),
        linewidth=0.8,
        alpha=0.85,
        zorder=0,
    )
    ax.set_xlim(k_min - x_padding, k_max + x_padding)
    ax.set_ylim(-0.025, 1.025)
    add_grid(ax)
    return ideal, hardware


def fit_free_intercept(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    sigma2 = float(np.sum(residuals**2) / max(len(x) - 2, 1))
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return float(beta[0]), float(beta[1]), cov


def weighted_covariance(
    x: np.ndarray,
    y: np.ndarray,
    contrast: np.ndarray,
    contrast_se: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    log_se = np.maximum(contrast_se / np.maximum(contrast, 1e-12), 1e-6)
    weights = np.minimum(1.0 / (log_se * log_se), 1e6)
    design = np.column_stack([np.ones_like(x), x])
    residuals = y - (intercept + slope * x)
    sigma2 = float(np.sum(weights * residuals**2) / max(len(x) - 2, 1))
    return sigma2 * np.linalg.pinv(design.T @ (weights[:, None] * design))


def confidence_band(
    x_grid: np.ndarray,
    intercept: float,
    slope: float,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones_like(x_grid), x_grid])
    mean = intercept + slope * x_grid
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design, cov, design), 0.0))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def contrast_fit(
    points: pd.DataFrame,
    summary: dict[str, object],
    *,
    use_summary_fit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    included = points[points["used_in_fit"].astype(bool)].copy()
    included = included[
        np.isfinite(included["contrast_mitigated"]) & (included["contrast_mitigated"] > 0.0)
    ]
    x = included["amplification_factor"].to_numpy(dtype=float)
    contrast = included["contrast_mitigated"].to_numpy(dtype=float)
    contrast_se = included["contrast_mitigated_se"].to_numpy(dtype=float)
    y = np.log(contrast)

    if use_summary_fit:
        intercept = float(np.log(float(summary["contrast_prefactor"])))
        slope = float(summary["free_intercept_slope"])
        cov = weighted_covariance(x, y, contrast, contrast_se, intercept, slope)
    else:
        intercept, slope, cov = fit_free_intercept(x, y)

    x_grid = np.linspace(1.0, float(points["amplification_factor"].max()), 1000)
    y_free, y_low, y_high = confidence_band(x_grid, intercept, slope, cov)
    zero_slope = -1.0 / float(summary["t_eff_zero_intercept"])
    return x_grid, np.exp(y_free), np.exp(y_low), np.exp(y_high), np.exp(zero_slope * x_grid)


def scatter_excluded(
    ax: plt.Axes,
    excluded: pd.DataFrame,
    *,
    annotate_extreme: bool,
) -> object:
    in_range = excluded[
        excluded["contrast_mitigated"].between(CONTRAST_Y_MIN, CONTRAST_Y_MAX, inclusive="both")
    ]
    handle = ax.scatter(
        grover_power(in_range["amplification_factor"]),
        in_range["contrast_mitigated"],
        s=20,
        facecolors="none",
        edgecolors="#999999",
        linewidths=0.85,
        alpha=0.85,
        label="Excluded",
        zorder=2,
    )
    below = excluded[excluded["contrast_mitigated"] < CONTRAST_Y_MIN]
    above = excluded[excluded["contrast_mitigated"] > CONTRAST_Y_MAX]
    ax.scatter(
        grover_power(below["amplification_factor"]),
        np.full(len(below), CONTRAST_Y_MIN + 0.012),
        marker="v",
        s=25,
        facecolors="none",
        edgecolors="#999999",
        linewidths=0.85,
        alpha=0.85,
        zorder=2,
    )
    ax.scatter(
        grover_power(above["amplification_factor"]),
        np.full(len(above), CONTRAST_Y_MAX - 0.012),
        marker="^",
        s=25,
        facecolors="none",
        edgecolors="#999999",
        linewidths=0.85,
        alpha=0.85,
        zorder=2,
    )
    if annotate_extreme and not below.empty:
        extreme = below.loc[below["contrast_mitigated"].idxmin()]
        factor = float(extreme["amplification_factor"])
        x = float(grover_power(factor))
        y = CONTRAST_Y_MIN + 0.012
        ax.annotate(
            rf"$C_{{{int(factor)}}}={float(extreme['contrast_mitigated']):.2f}$",
            xy=(x, y),
            xytext=(8, 18),
            textcoords="offset points",
            fontsize=8.5,
            color=GRAY,
            arrowprops={"arrowstyle": "-|>", "color": GRAY, "linewidth": 0.75},
            zorder=5,
        )
    return handle


def draw_contrast_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    summary: dict[str, object],
    *,
    x_padding: float,
    use_summary_fit: bool,
    annotate_extreme: bool,
) -> tuple[object, object, object, object, object]:
    included = points[points["used_in_fit"].astype(bool)]
    excluded = points[~points["used_in_fit"].astype(bool)]
    x_grid, free, free_low, free_high, zero = contrast_fit(
        points,
        summary,
        use_summary_fit=use_summary_fit,
    )

    included_handle = ax.errorbar(
        grover_power(included["amplification_factor"]),
        included["contrast_mitigated"],
        yerr=included["contrast_mitigated_se"],
        fmt="o",
        ms=3.5,
        mfc="white",
        mec=BLUE,
        mew=1.0,
        ecolor=BLUE,
        color=BLUE,
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        label="Included",
        zorder=4,
    )
    excluded_handle = scatter_excluded(ax, excluded, annotate_extreme=annotate_extreme)
    k_grid = grover_power(x_grid)
    free_handle = ax.plot(k_grid, free, color=ORANGE, linewidth=1.35, label="Free-intercept")[0]
    zero_handle = ax.plot(
        k_grid,
        zero,
        color=GREEN,
        linestyle=(0, (4, 2)),
        linewidth=1.2,
        label="Zero-intercept",
    )[0]
    ci_handle = ax.fill_between(
        k_grid,
        free_low,
        free_high,
        color=ORANGE,
        alpha=0.14,
        linewidth=0,
        label="95% mean CI",
    )
    ax.set_xlim(0.0, float(points["grover_power"].max()) + x_padding)
    ax.set_ylim(CONTRAST_Y_MIN, CONTRAST_Y_MAX)
    add_grid(ax)
    return included_handle, excluded_handle, free_handle, zero_handle, ci_handle


def make_figure() -> None:
    toy_points, toy_summary = read_run(TOY_RUN_DIR)
    cva_points, cva_summary = read_run(CVA_RUN_DIR)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.2), sharey="col")
    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.095, top=0.895, wspace=0.15, hspace=0.13)

    amplification_handles = draw_amplification_panel(
        axes[0, 0],
        toy_points,
        baseline=0.5,
        x_padding=0.5,
    )
    draw_amplification_panel(
        axes[1, 0],
        cva_points,
        baseline=float(cva_summary["contrast_baseline"]),
        x_padding=0.1,
    )
    contrast_handles = draw_contrast_panel(
        axes[0, 1],
        toy_points,
        toy_summary,
        x_padding=0.5,
        use_summary_fit=False,
        annotate_extreme=False,
    )
    draw_contrast_panel(
        axes[1, 1],
        cva_points,
        cva_summary,
        x_padding=0.1,
        use_summary_fit=True,
        annotate_extreme=True,
    )

    fig.legend(
        amplification_handles,
        ["Ideal Grover oscillation", "Hardware mitigated"],
        loc="upper center",
        bbox_to_anchor=(0.285, 0.985),
        ncol=2,
        handlelength=2.7,
        columnspacing=1.5,
    )
    fig.legend(
        contrast_handles,
        ["Included", "Excluded", "Free-intercept", "Zero-intercept", "95% mean CI"],
        loc="upper center",
        bbox_to_anchor=(0.77, 0.995),
        ncol=3,
        handlelength=2.4,
        columnspacing=1.0,
        handletextpad=0.65,
    )

    fig.text(0.285, 0.055, r"Grover power $k$", ha="center", va="center", fontsize=13)
    fig.text(
        0.775,
        0.055,
        r"Grover power $k$",
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.text(
        0.028,
        0.495,
        "Good-state probability",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=13,
    )
    fig.text(
        0.515,
        0.495,
        r"Empirical contrast $C_K$",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=13,
    )

    output = OUTPUT_DIR / "hardware_amplification_contrast_grid"
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_figure()


if __name__ == "__main__":
    main()
