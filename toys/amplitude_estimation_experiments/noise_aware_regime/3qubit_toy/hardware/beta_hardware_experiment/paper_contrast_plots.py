from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "experiment_results" / "csv_results"
PLOTS_DIR = ROOT / "experiment_results" / "plots"


BLUE = "#1F77B4"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D7D7D7"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "lines.linewidth": 1.6,
            "patch.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def read_inputs() -> tuple[pd.DataFrame, dict[str, object]]:
    points = pd.read_csv(RUN_DIR / "amplification_points.csv").sort_values("grover_power")
    summary = json.loads((RUN_DIR / "calibration_summary.json").read_text(encoding="utf-8"))
    return points, summary


def grover_probability(k: np.ndarray, a0: float) -> np.ndarray:
    theta = np.arcsin(np.sqrt(np.clip(float(a0), 0.0, 1.0)))
    return np.sin((2.0 * k + 1.0) * theta) ** 2


def fit_free_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = max(len(x) - 2, 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    cov = sigma2 * np.linalg.inv(design.T @ design)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0.0 else np.nan
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "cov": cov,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(r2),
    }


def fit_zero_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope = float(np.sum(x * y) / np.sum(x * x))
    residuals = y - slope * x
    dof = max(len(x) - 1, 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    slope_se = float(np.sqrt(sigma2 / np.sum(x * x)))
    ss_tot_zero = float(np.sum(y**2))
    r2_zero = 1.0 - float(np.sum(residuals**2)) / ss_tot_zero if ss_tot_zero > 0.0 else np.nan
    return {
        "slope": slope,
        "slope_se": slope_se,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2_zero": float(r2_zero),
    }


def confidence_band_free(
    x_new: np.ndarray,
    intercept: float,
    slope: float,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design_new = np.column_stack([np.ones_like(x_new), x_new])
    mean = intercept + slope * x_new
    se = np.sqrt(np.einsum("ij,jk,ik->i", design_new, cov, design_new))
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi


def draw_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        clip_on=False,
    )


def plot_amplification(points: pd.DataFrame) -> None:
    a0 = float(points.loc[points["grover_power"].idxmin(), "p_ideal"])
    k_min = float(points["grover_power"].min())
    k_max = float(points["grover_power"].max())
    k_grid = np.linspace(k_min, k_max, 1800)
    p_grid = grover_probability(k_grid, a0)

    fig, ax = plt.subplots(figsize=(6.65, 3.0), constrained_layout=True)
    ax.plot(
        k_grid,
        p_grid,
        color="black",
        linewidth=1.1,
        label=r"Ideal Grover oscillation",
        zorder=1,
    )
    ax.errorbar(
        points["grover_power"],
        points["p_hw_mitigated"],
        yerr=points["p_hw_mitigated_se"],
        fmt="o",
        ms=3.3,
        mfc="white",
        mec=BLUE,
        mew=0.9,
        ecolor=BLUE,
        color=BLUE,
        elinewidth=0.8,
        capsize=2.4,
        capthick=0.8,
        label=r"Hardware mitigated",
        zorder=3,
    )
    ax.axhline(0.5, color=GRAY, linestyle=(0, (2.5, 2.5)), linewidth=0.8, alpha=0.85)
    ax.set_xlabel(r"Grover power $k$")
    ax.set_ylabel(r"Good-state probability")
    ax.set_xlim(k_min - 0.5, k_max + 0.5)
    ax.set_ylim(-0.025, 1.025)
    ax.minorticks_on()
    ax.grid(True, which="major", color=LIGHT_GRAY, linewidth=0.45, alpha=0.55)
    ax.grid(True, which="minor", color=LIGHT_GRAY, linewidth=0.25, alpha=0.28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.145),
        ncol=2,
        handlelength=2.6,
        columnspacing=1.4,
    )

    fig.savefig(PLOTS_DIR / "amplification_scan.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PLOTS_DIR / "amplification_scan_paper.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PLOTS_DIR / "amplification_scan_paper.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_contrast_calibration(points: pd.DataFrame, summary: dict[str, object]) -> pd.DataFrame:
    fit_points = points[points["used_in_fit"].astype(bool)].copy()
    fit_points = fit_points[np.isfinite(fit_points["contrast_mitigated"])]
    fit_points = fit_points[fit_points["contrast_mitigated"] > 0.0]

    x = fit_points["amplification_factor"].to_numpy(dtype=float)
    y = np.log(fit_points["contrast_mitigated"].to_numpy(dtype=float))
    free = fit_free_intercept(x, y)
    zero = fit_zero_intercept(x, y)

    x_grid = np.linspace(1.0, float(points["amplification_factor"].max()), 800)
    y_free, y_free_lo, y_free_hi = confidence_band_free(
        x_grid,
        float(free["intercept"]),
        float(free["slope"]),
        np.asarray(free["cov"], dtype=float),
    )
    y_zero = float(zero["slope"]) * x_grid

    fig, ax0 = plt.subplots(figsize=(4.15, 3.0), constrained_layout=True)
    included = points[points["used_in_fit"].astype(bool)]
    excluded = points[~points["used_in_fit"].astype(bool)]

    ax0.errorbar(
        included["amplification_factor"],
        included["contrast_mitigated"],
        yerr=included["contrast_mitigated_se"],
        fmt="o",
        ms=3.2,
        mfc="white",
        mec=BLUE,
        mew=0.9,
        ecolor=BLUE,
        color=BLUE,
        elinewidth=0.75,
        capsize=2.2,
        label="Included",
        zorder=4,
    )
    ax0.scatter(
        excluded["amplification_factor"],
        excluded["contrast_mitigated"],
        s=18,
        facecolors="none",
        edgecolors=GRAY,
        linewidths=0.75,
        alpha=0.72,
        label="Excluded",
        zorder=2,
    )
    free_line = ax0.plot(x_grid, np.exp(y_free), color=ORANGE, linewidth=1.25, label="Free-intercept")[0]
    ci_patch = ax0.fill_between(
        x_grid,
        np.exp(y_free_lo),
        np.exp(y_free_hi),
        color=ORANGE,
        alpha=0.14,
        linewidth=0,
        label="95% mean CI",
    )
    zero_line = ax0.plot(
        x_grid,
        np.exp(y_zero),
        color=GREEN,
        linestyle=(0, (4, 2)),
        linewidth=1.1,
        label="Zero-intercept",
    )[0]
    ax0.set_xlabel(r"Amplification factor $K=2k+1$")
    ax0.set_ylabel(r"Empirical contrast $C_K$")
    ax0.set_xlim(0.0, float(points["amplification_factor"].max()) + 2.0)
    ax0.set_ylim(-0.08, 1.14)
    ax0.minorticks_on()
    ax0.grid(True, which="major", color=LIGHT_GRAY, linewidth=0.45, alpha=0.55)
    ax0.grid(True, which="minor", color=LIGHT_GRAY, linewidth=0.25, alpha=0.25)

    handles, labels = ax0.get_legend_handles_labels()
    ordered = []
    for name in ("Included", "Excluded"):
        idx = labels.index(name)
        ordered.append((handles[idx], labels[idx]))
    ordered.extend([(free_line, "Free-intercept"), (zero_line, "Zero-intercept"), (ci_patch, "95% mean CI")])
    fig.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        handlelength=2.2,
        columnspacing=0.95,
    )

    fig.savefig(PLOTS_DIR / "contrast_calibration_paper.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PLOTS_DIR / "contrast_calibration_paper.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    prefactor = float(np.exp(float(free["intercept"])))
    t_free = float(-1.0 / float(free["slope"]))
    t_zero = float(-1.0 / float(zero["slope"]))
    fit_summary = pd.DataFrame(
        [
            {
                "model": "free_intercept",
                "contrast_prefactor": prefactor,
                "slope_log_contrast": float(free["slope"]),
                "t_eff": t_free,
                "rmse_log_contrast": float(free["rmse"]),
                "r2_log_contrast": float(free["r2"]),
                "fit_points": int(len(fit_points)),
            },
            {
                "model": "zero_intercept",
                "contrast_prefactor": 1.0,
                "slope_log_contrast": float(zero["slope"]),
                "t_eff": t_zero,
                "rmse_log_contrast": float(zero["rmse"]),
                "r2_log_contrast": float(zero["r2_zero"]),
                "fit_points": int(len(fit_points)),
            },
        ]
    )
    fit_summary["calibration_status"] = str(summary.get("calibration_status", ""))
    fit_summary.to_csv(RUN_DIR / "contrast_calibration_fit_summary.csv", index=False)
    return fit_summary


def write_point_usage_summary(points: pd.DataFrame) -> None:
    rows = []
    rows.append({"category": "included_in_fit", "count": int(points["used_in_fit"].astype(bool).sum())})
    for reason, group in points[~points["used_in_fit"].astype(bool)].groupby("fit_exclusion_reason"):
        rows.append({"category": str(reason), "count": int(len(group))})
    pd.DataFrame(rows).to_csv(RUN_DIR / "contrast_calibration_point_usage.csv", index=False)


def main() -> None:
    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    points, summary = read_inputs()
    plot_amplification(points)
    plot_contrast_calibration(points, summary)
    write_point_usage_summary(points)


if __name__ == "__main__":
    main()
