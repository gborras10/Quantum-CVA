from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE_RUN_DIR = (
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
DEFAULT_DCS_RUN_DIR = DEFAULT_HARDWARE_RUN_DIR.with_name("q_ctrl_hardware")
DEFAULT_NOISELESS_RUN_DIR = (
    ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "cva_pricing_multi_asset"
    / "quantum"
    / "ae_cva"
    / "noiseless_simulation"
    / "experiment_results"
)
DEFAULT_OUTPUT = ROOT / "plots" / "ae_cva_qctrl_basquecountry_noiseless_final_error_grid"

ALGORITHM_ORDER = ("CABIQAE", "BIQAE", "DCS")
STYLES = {
    "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
    "BIQAE": {"color": "#A23B72", "marker": "s"},
    "DCS": {"color": "#2A9D8F", "marker": "X"},
}
KEY_TO_LABEL = {
    "cabiqae": "CABIQAE",
    "cabiqae_latentt": "CABIQAE",
    "biqae": "BIQAE",
    "classical_mc": "DCS",
}
VISUAL_JITTER_DEX = 0.0035


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the 6q hardware/noiseless CVA final-error density grid."
    )
    parser.add_argument("--hardware-run-dir", type=Path, default=DEFAULT_HARDWARE_RUN_DIR)
    parser.add_argument("--dcs-run-dir", type=Path, default=DEFAULT_DCS_RUN_DIR)
    parser.add_argument("--noiseless-run-dir", type=Path, default=DEFAULT_NOISELESS_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def numeric_fallback(rows: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    primary_values = (
        pd.to_numeric(rows[primary], errors="coerce")
        if primary in rows
        else pd.Series(np.nan, index=rows.index)
    )
    fallback_values = (
        pd.to_numeric(rows[fallback], errors="coerce")
        if fallback in rows
        else pd.Series(np.nan, index=rows.index)
    )
    return primary_values.fillna(fallback_values)


def read_budget_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows["algorithm"] = rows["algorithm_key"].map(KEY_TO_LABEL)
    rows["final_queries"] = numeric_fallback(rows, "query_budget_actual", "budget")
    rows["runtime_wall_seconds"] = numeric_fallback(
        rows,
        "runtime_wall_seconds",
        "time_to_budget_seconds",
    )
    rows["final_relative_error"] = numeric_fallback(
        rows,
        "processed_relative_error",
        "normalized_abs_error",
    )
    rows["repetition"] = pd.to_numeric(rows["repetition"], errors="coerce")
    rows = rows[
        rows["algorithm"].isin(ALGORITHM_ORDER)
        & np.isfinite(rows["repetition"])
        & np.isfinite(rows["final_queries"])
        & np.isfinite(rows["runtime_wall_seconds"])
        & np.isfinite(rows["final_relative_error"])
        & (rows["final_queries"] > 0.0)
        & (rows["runtime_wall_seconds"] > 0.0)
        & (rows["final_relative_error"] > 0.0)
    ].copy()
    return (
        rows.sort_values(["algorithm", "repetition", "final_queries"])
        .groupby(["algorithm", "repetition"], as_index=False)
        .tail(1)
    )


def read_hardware_rows(hardware_run_dir: Path, dcs_run_dir: Path) -> pd.DataFrame:
    quantum = read_budget_rows(hardware_run_dir / "replay_budget_rows.csv")
    quantum = quantum[quantum["algorithm"].isin(("CABIQAE", "BIQAE"))]
    classical = read_budget_rows(dcs_run_dir / "montecarlo_budget_rows.csv")
    classical = classical[classical["algorithm"] == "DCS"]
    return pd.concat([quantum, classical], ignore_index=True)


def read_noiseless_rows(noiseless_run_dir: Path) -> pd.DataFrame:
    quantum = read_budget_rows(noiseless_run_dir / "replay_budget_rows.csv")
    quantum = quantum[quantum["algorithm"].isin(("CABIQAE", "BIQAE"))]
    classical = read_budget_rows(noiseless_run_dir / "montecarlo_budget_rows.csv")
    classical = classical[classical["algorithm"] == "DCS"]
    return pd.concat([quantum, classical], ignore_index=True)


def density_coordinates(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinates = np.column_stack([np.log10(x_values), np.log10(y_values)])
    rendered = coordinates + rng.normal(0.0, VISUAL_JITTER_DEX, size=coordinates.shape)
    if len(rendered) < 3:
        return rendered, np.ones(len(rendered), dtype=float)
    try:
        estimator = gaussian_kde(rendered.T, bw_method="scott")
        density = estimator(rendered.T)
    except np.linalg.LinAlgError:
        density = np.ones(len(rendered), dtype=float)
    return rendered, density


def hpd_density_levels(grid_density: np.ndarray) -> tuple[float, float] | None:
    finite = grid_density[np.isfinite(grid_density) & (grid_density > 0.0)]
    if finite.size < 3:
        return None
    ordered = np.sort(finite)[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    threshold_50 = float(ordered[min(np.searchsorted(cumulative, 0.50), len(ordered) - 1)])
    threshold_90 = float(ordered[min(np.searchsorted(cumulative, 0.90), len(ordered) - 1)])
    if threshold_90 >= threshold_50:
        return None
    return threshold_90, threshold_50


def draw_kde_regions(ax: plt.Axes, rendered: np.ndarray, *, color: str) -> None:
    if len(rendered) < 5:
        return
    try:
        estimator = gaussian_kde(rendered.T, bw_method="scott")
    except np.linalg.LinAlgError:
        return
    lower = np.quantile(rendered, 0.01, axis=0)
    upper = np.quantile(rendered, 0.99, axis=0)
    padding = np.maximum((upper - lower) * 0.20, 0.015)
    x_log = np.linspace(lower[0] - padding[0], upper[0] + padding[0], 95)
    y_log = np.linspace(lower[1] - padding[1], upper[1] + padding[1], 95)
    grid_x, grid_y = np.meshgrid(x_log, y_log)
    density = estimator(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    levels = hpd_density_levels(density)
    if levels is None:
        return
    level_90, level_50 = levels
    ax.contourf(
        10.0**grid_x,
        10.0**grid_y,
        density,
        levels=[level_90, level_50, float(np.nanmax(density))],
        colors=[color, color],
        alpha=0.055,
        antialiased=True,
        zorder=1,
    )
    ax.contour(
        10.0**grid_x,
        10.0**grid_y,
        density,
        levels=[level_90, level_50],
        colors=[color, color],
        linewidths=[0.8, 1.05],
        alpha=0.48,
        zorder=2,
    )


def density_adaptive_style(density: np.ndarray, color: str) -> tuple[np.ndarray, np.ndarray]:
    if len(density) == 0 or np.allclose(density, density[0]):
        normalized = np.zeros(len(density), dtype=float)
    else:
        low, high = np.quantile(density, [0.05, 0.95])
        normalized = np.clip((density - low) / max(high - low, 1e-12), 0.0, 1.0)
    sizes = 15.0 - 7.0 * normalized
    alpha = 0.58 - 0.30 * normalized
    return sizes, np.asarray([to_rgba(color, float(value)) for value in alpha])


def draw_algorithm(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    algorithm: str,
    x_column: str,
    seed: int,
) -> None:
    style = STYLES[algorithm]
    group = rows[rows["algorithm"] == algorithm]
    x_values = group[x_column].to_numpy(dtype=float)
    y_values = group["final_relative_error"].to_numpy(dtype=float)
    rendered, density = density_coordinates(x_values, y_values, seed=seed)
    draw_kde_regions(ax, rendered, color=style["color"])
    sizes, colors = density_adaptive_style(density, style["color"])
    order = np.argsort(density)[::-1]
    ax.scatter(
        10.0 ** rendered[order, 0],
        10.0 ** rendered[order, 1],
        s=sizes[order],
        marker=style["marker"],
        facecolors=colors[order],
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
        zorder=3,
    )
    ax.scatter(
        [float(np.median(x_values))],
        [float(np.median(y_values))],
        s=70,
        marker=style["marker"],
        facecolor=style["color"],
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )


def logarithmic_limits(values: np.ndarray, *, padding: float) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0.0)]
    lower = float(np.min(np.log10(finite)))
    upper = float(np.max(np.log10(finite)))
    span = max(upper - lower, 0.1)
    return 10.0 ** (lower - padding * span), 10.0 ** (upper + padding * span)


def draw_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    *,
    x_column: str,
    panel_seed: int,
) -> None:
    for index, algorithm in enumerate(ALGORITHM_ORDER):
        draw_algorithm(
            ax,
            rows,
            algorithm=algorithm,
            x_column=x_column,
            seed=panel_seed + 101 * index,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*logarithmic_limits(rows[x_column].to_numpy(dtype=float), padding=0.08))
    ax.grid(True, which="major", color="#BFBFBF", linewidth=0.55, alpha=0.38)
    ax.grid(True, which="minor", color="#D7D7D7", linewidth=0.40, alpha=0.18)


def summary_rows(dataset: str, rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for algorithm in ALGORITHM_ORDER:
        group = rows[rows["algorithm"] == algorithm]
        records.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "n": int(len(group)),
                "median_final_queries": float(group["final_queries"].median()),
                "median_runtime_seconds": float(group["runtime_wall_seconds"].median()),
                "median_final_relative_error": float(group["final_relative_error"].median()),
            }
        )
    return pd.DataFrame(records)


def make_figure(
    hardware_run_dir: Path,
    dcs_run_dir: Path,
    noiseless_run_dir: Path,
    output: Path,
) -> None:
    hardware = read_hardware_rows(hardware_run_dir, dcs_run_dir)
    noiseless = read_noiseless_rows(noiseless_run_dir)
    all_rows = pd.concat([hardware, noiseless], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.25), sharey=True)
    fig.subplots_adjust(
        left=0.075,
        right=0.955,
        bottom=0.095,
        top=0.885,
        wspace=0.12,
        hspace=0.13,
    )
    draw_panel(axes[0, 0], hardware, x_column="final_queries", panel_seed=1200)
    draw_panel(axes[0, 1], hardware, x_column="runtime_wall_seconds", panel_seed=2200)
    draw_panel(axes[1, 0], noiseless, x_column="final_queries", panel_seed=3200)
    draw_panel(axes[1, 1], noiseless, x_column="runtime_wall_seconds", panel_seed=4200)

    y_limits = logarithmic_limits(all_rows["final_relative_error"].to_numpy(dtype=float), padding=0.08)
    for ax in axes.flat:
        ax.set_ylim(*y_limits)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STYLES[algorithm]["color"],
            marker=STYLES[algorithm]["marker"],
            linestyle="None",
            markersize=7.0,
            label=algorithm,
        )
        for algorithm in ALGORITHM_ORDER
    ]
    legend_handles.append(
        Line2D([0], [0], color="#666666", linewidth=1.0, label="50% / 90% KDE regions")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=4,
        handlelength=2.1,
        columnspacing=1.3,
        handletextpad=0.55,
    )

    fig.text(0.305, 0.045, r"Final query count $N_q$", ha="center", va="center", fontsize=13)
    fig.text(0.765, 0.045, "Runtime [s]", ha="center", va="center", fontsize=13)
    fig.text(0.025, 0.495, "Final relative error", ha="center", va="center", rotation="vertical", fontsize=13)
    fig.text(0.977, 0.695, "Hardware replay", ha="center", va="center", rotation=-90, fontsize=11)
    fig.text(0.977, 0.290, "Noiseless replay", ha="center", va="center", rotation=-90, fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    pd.concat(
        [
            summary_rows("hardware_replay", hardware),
            summary_rows("noiseless_replay", noiseless),
        ],
        ignore_index=True,
    ).to_csv(output.with_name(f"{output.name}_summary.csv"), index=False)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    make_figure(
        args.hardware_run_dir.resolve(),
        args.dcs_run_dir.resolve(),
        args.noiseless_run_dir.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
