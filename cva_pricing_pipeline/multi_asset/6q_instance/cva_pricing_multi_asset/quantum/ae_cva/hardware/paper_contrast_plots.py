from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#1F77B4"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D7D7D7"
QUERY_STYLES = {
    "BIQAE": {"color": "#A23B72", "marker": "s"},
    "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
    "CABIQAE (contrast model)": {"color": "#1F6F8B", "marker": "o"},
    "DCS": {"color": "#2A9D8F", "marker": "X"},
}

DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "q_ctrl_hardware"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference-style paper plots for a hardware CVA AE run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Results directory containing amplification_points.csv and calibration_summary.json.",
    )
    monte_carlo_group = parser.add_mutually_exclusive_group()
    monte_carlo_group.add_argument(
        "--include-monte-carlo",
        dest="include_monte_carlo",
        action="store_true",
        help="Append montecarlo_budget_summary.csv as a DCS baseline (default).",
    )
    monte_carlo_group.add_argument(
        "--no-monte-carlo",
        dest="include_monte_carlo",
        action="store_false",
        help="Generate the replay plot without the DCS baseline.",
    )
    parser.set_defaults(include_monte_carlo=True)
    parser.add_argument(
        "--monte-carlo-summary",
        type=Path,
        default=None,
        help="Optional DCS summary CSV; defaults to <run-dir>/montecarlo_budget_summary.csv.",
    )
    return parser.parse_args()


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


def read_inputs(
    run_dir: Path,
    *,
    include_monte_carlo: bool = False,
    monte_carlo_summary: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    points = pd.read_csv(run_dir / "amplification_points.csv").sort_values("grover_power")
    summary = json.loads((run_dir / "calibration_summary.json").read_text(encoding="utf-8"))
    budget_summary = pd.read_csv(run_dir / "budget_summary.csv")
    config_path = run_dir / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    if bool(config.get("cabiqae_replay_contrast_model_all_k", False)):
        budget_summary.loc[
            budget_summary["algorithm_key"].isin(
                {"cabiqae", "cabiqae_known_t", "cabiqae_latentt"}
            ),
            "algorithm",
        ] = "CABIQAE (contrast model)"
    if include_monte_carlo:
        mc_path = monte_carlo_summary or (run_dir / "montecarlo_budget_summary.csv")
        if not mc_path.exists():
            raise FileNotFoundError(
                f"{mc_path} does not exist. Generate the DCS baseline first."
            )
        budget_summary = pd.concat(
            [budget_summary, pd.read_csv(mc_path)],
            ignore_index=True,
            sort=False,
        )
        budget_summary.loc[budget_summary["algorithm"] == "Classical MC", "algorithm"] = "DCS"
    return points, summary, budget_summary


def grover_probability(k: np.ndarray, a0: float) -> np.ndarray:
    theta = np.arcsin(np.sqrt(np.clip(float(a0), 0.0, 1.0)))
    return np.sin((2.0 * k + 1.0) * theta) ** 2


def fit_free_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = max(len(x) - 2, 1)
    sigma2 = float(np.sum(residuals**2) / dof)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "cov": cov}


def fit_zero_intercept(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(x * y) / np.sum(x * x))


def confidence_band_free(
    x_new: np.ndarray,
    intercept: float,
    slope: float,
    cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design_new = np.column_stack([np.ones_like(x_new), x_new])
    mean = intercept + slope * x_new
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", design_new, cov, design_new), 0.0))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def weighted_covariance_free_intercept(
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
    dof = max(len(x) - 2, 1)
    sigma2 = float(np.sum(weights * residuals**2) / dof)
    return sigma2 * np.linalg.pinv(design.T @ (weights[:, None] * design))


def plot_amplification(
    points: pd.DataFrame,
    summary: dict[str, object],
    plots_dir: Path,
) -> None:
    a0 = float(points.loc[points["grover_power"].idxmin(), "p_ideal"])
    baseline = float(summary.get("contrast_baseline", 0.125))
    k_min = float(points["grover_power"].min())
    k_max = float(points["grover_power"].max())
    k_grid = np.linspace(k_min, k_max, 1800)

    fig, ax = plt.subplots(figsize=(6.65, 3.0), constrained_layout=True)
    ax.plot(
        k_grid,
        grover_probability(k_grid, a0),
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
    ax.axhline(
        baseline,
        color=GRAY,
        linestyle=(0, (2.5, 2.5)),
        linewidth=0.8,
        alpha=0.85,
    )
    ax.set_xlabel(r"Grover power $k$")
    ax.set_ylabel(r"Good-state probability")
    ax.set_xlim(k_min - 0.1, k_max + 0.1)
    ax.set_ylim(-0.025, 1.025)
    ax.set_xticks(points["grover_power"])
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

    fig.savefig(plots_dir / "amplification_scan.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(plots_dir / "amplification_scan.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(plots_dir / "amplification_scan_paper.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(plots_dir / "amplification_scan_paper.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_contrast_calibration(
    points: pd.DataFrame,
    summary: dict[str, object],
    plots_dir: Path,
) -> None:
    included = points[points["used_in_fit"].astype(bool)].copy()
    included = included[
        np.isfinite(included["contrast_mitigated"]) & (included["contrast_mitigated"] > 0.0)
    ]
    excluded = points[~points.index.isin(included.index)]
    valid_fit = str(summary.get("calibration_status", "")) == "ok"
    if valid_fit and len(included) < 2:
        raise ValueError("Contrast calibration plot requires at least two positive fit points.")

    x_grid = np.linspace(1.0, float(points["amplification_factor"].max()), 800)

    fig, ax0 = plt.subplots(figsize=(4.15, 3.0), constrained_layout=True)
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
        label="Included" if valid_fit else "Candidate",
        zorder=4,
    )
    y_min, y_max = -0.08, 1.14
    excluded_in_range = excluded[
        excluded["contrast_mitigated"].between(y_min, y_max, inclusive="both")
    ]
    excluded_off_scale = excluded[~excluded.index.isin(excluded_in_range.index)]
    ax0.scatter(
        excluded_in_range["amplification_factor"],
        excluded_in_range["contrast_mitigated"],
        s=18,
        facecolors="none",
        edgecolors=GRAY,
        linewidths=0.75,
        alpha=0.72,
        label="Excluded",
        zorder=2,
    )
    for point in excluded_off_scale.itertuples():
        clipped_y = y_max - 0.035 if point.contrast_mitigated > y_max else y_min + 0.035
        direction = 1.0 if point.contrast_mitigated > y_max else -1.0
        ax0.scatter(
            [point.amplification_factor],
            [clipped_y],
            s=18,
            facecolors="none",
            edgecolors=GRAY,
            linewidths=0.75,
            alpha=0.72,
            zorder=2,
        )
        ax0.annotate(
            rf"$C_{{{int(point.amplification_factor)}}}={point.contrast_mitigated:.2f}$",
            xy=(point.amplification_factor, clipped_y),
            xytext=(6, -14 if direction > 0.0 else 10),
            textcoords="offset points",
            fontsize=7.5,
            color=GRAY,
            ha="left",
            va="top" if direction > 0.0 else "bottom",
            arrowprops={"arrowstyle": "-|>", "color": GRAY, "linewidth": 0.65},
            zorder=5,
    )
    if valid_fit:
        x = included["amplification_factor"].to_numpy(dtype=float)
        contrast = included["contrast_mitigated"].to_numpy(dtype=float)
        contrast_se = included["contrast_mitigated_se"].to_numpy(dtype=float)
        y = np.log(contrast)
        prefactor = float(summary["contrast_prefactor"])
        free_intercept = float(np.log(prefactor))
        free_slope = float(summary["free_intercept_slope"])
        t_zero = float(summary["t_eff_zero_intercept"])
        zero_slope = -1.0 / t_zero
        cov = weighted_covariance_free_intercept(
            x,
            y,
            contrast,
            contrast_se,
            free_intercept,
            free_slope,
        )
        y_free, y_free_lo, y_free_hi = confidence_band_free(
            x_grid,
            free_intercept,
            free_slope,
            cov,
        )
        free_line = ax0.plot(
            x_grid, np.exp(y_free), color=ORANGE, linewidth=1.25, label="Free-intercept"
        )[0]
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
            np.exp(zero_slope * x_grid),
            color=GREEN,
            linestyle=(0, (4, 2)),
            linewidth=1.1,
            label="Zero-intercept",
        )[0]
    else:
        ax0.text(
            0.98,
            0.94,
            "No valid decaying fit",
            transform=ax0.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            color=GRAY,
        )
    ax0.set_xlabel(r"Amplification factor $K=2k+1$")
    ax0.set_ylabel(r"Empirical contrast $C_K$")
    ax0.set_xlim(0.0, float(points["amplification_factor"].max()) + 0.5)
    ax0.set_ylim(y_min, y_max)
    ax0.set_xticks(points["amplification_factor"])
    ax0.minorticks_on()
    ax0.grid(True, which="major", color=LIGHT_GRAY, linewidth=0.45, alpha=0.55)
    ax0.grid(True, which="minor", color=LIGHT_GRAY, linewidth=0.25, alpha=0.25)

    handles, labels = ax0.get_legend_handles_labels()
    marker_label = "Included" if valid_fit else "Candidate"
    ordered = [(handles[labels.index(name)], name) for name in (marker_label, "Excluded")]
    if valid_fit:
        ordered.extend(
            [
                (free_line, "Free-intercept"),
                (zero_line, "Zero-intercept"),
                (ci_patch, "95% mean CI"),
            ]
        )
    fig.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3 if valid_fit else 2,
        handlelength=2.2,
        columnspacing=0.95,
    )

    fig.savefig(plots_dir / "contrast_calibration_paper.png", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(plots_dir / "contrast_calibration_paper.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _query_plot_style() -> dict[str, object]:
    return {
        "font.serif": ["Computer Modern Roman", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.25,
        "lines.solid_capstyle": "round",
    }


def _ci_errorbar(
    center: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> np.ndarray:
    lower = np.maximum(center - ci_low, 0.0)
    upper = np.maximum(ci_high - center, 0.0)
    return np.vstack([lower, upper])


def plot_replay_query_error(budget_summary: pd.DataFrame, plots_dir: Path) -> None:
    x_col = "query_budget_actual_mean"
    y_col = "processed_relative_error_median"
    low_col = "processed_relative_error_median_ci_low"
    high_col = "processed_relative_error_median_ci_high"
    required = {"algorithm", x_col, y_col, low_col, high_col}
    missing = sorted(required.difference(budget_summary.columns))
    if missing:
        raise ValueError(f"Query-error plot inputs are missing columns: {', '.join(missing)}")

    rows = budget_summary.copy()
    for col in (x_col, y_col, low_col, high_col, "budget"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows[
        rows["algorithm"].isin(QUERY_STYLES)
        & np.isfinite(rows[x_col])
        & np.isfinite(rows[y_col])
        & (rows[x_col] > 0.0)
        & (rows[y_col] > 0.0)
    ].copy()
    if rows.empty:
        raise ValueError("Query-error plot has no positive supported algorithm summary points.")

    # Do not draw an identical stop twice when increasing the nominal budget
    # produced the same realized query cost and median estimator output.
    rows = (
        rows.sort_values(["algorithm", "budget"])
        .drop_duplicates(subset=["algorithm", x_col, y_col], keep="last")
        .sort_values(["algorithm", x_col])
    )
    rows.to_csv(plots_dir / "hardware_replay_actual_queries_paper_summary.csv", index=False)

    with mpl.rc_context(_query_plot_style()):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        x_min = float(rows[x_col].min())
        x_max = float(rows[x_col].max())
        anchor_y = float(rows[np.isclose(rows[x_col], x_min)][y_col].min())
        guide_x = np.geomspace(x_min, x_max, 200)
        ax.loglog(
            guide_x,
            anchor_y * np.sqrt(x_min / guide_x),
            color="#303030",
            linestyle=":",
            linewidth=1.35,
            label=r"$O(1/\sqrt{N})$",
            zorder=1,
        )

        for algorithm in QUERY_STYLES:
            group = rows[rows["algorithm"] == algorithm].sort_values(x_col)
            if group.empty:
                continue
            style = QUERY_STYLES[algorithm]
            x = group[x_col].to_numpy(dtype=float)
            y = group[y_col].to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=_ci_errorbar(
                    y,
                    group[low_col].to_numpy(dtype=float),
                    group[high_col].to_numpy(dtype=float),
                ),
                fmt=style["marker"],
                color=style["color"],
                linestyle="-",
                linewidth=2.0,
                markersize=5.6,
                elinewidth=1.0,
                capsize=2.8,
                label=algorithm,
                zorder=3,
            )

        ax.set_xlabel(r"$N_q$")
        ax.set_ylabel("Median relative CVA error")
        ax.set_xlim(x_min / 1.18, x_max * 1.18)
        ax.grid(True, which="major", color="#c7c7c7", linewidth=0.8, alpha=0.72)
        ax.grid(True, which="minor", color="#e7e7e7", linewidth=0.45, alpha=0.85)
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=6.0,
            width=1.05,
            top=True,
            right=True,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            length=3.2,
            width=0.85,
            top=True,
            right=True,
        )
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(1.25)
        ax.legend(frameon=False, loc="lower left", handlelength=2.7)

        output = plots_dir / "hardware_replay_cva_budget"
        fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    points, summary, budget_summary = read_inputs(
        run_dir,
        include_monte_carlo=bool(args.include_monte_carlo),
        monte_carlo_summary=args.monte_carlo_summary,
    )
    plot_amplification(points, summary, plots_dir)
    plot_contrast_calibration(points, summary, plots_dir)
    plot_replay_query_error(budget_summary, plots_dir)


if __name__ == "__main__":
    main()
