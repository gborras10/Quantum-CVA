from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "entangler_comparison"
TABLES_DIR = RESULTS_DIR / "tables"
OUTPUT_DIR = BASE_DIR / "paper_figures"

ENTANGLER_ORDER = ["rzz", "rxx", "cz"]
ENTANGLER_LABELS = {"rzz": "RZZ", "rxx": "RXX", "cz": "CZ"}
ENTANGLER_COLORS = {
    "rzz": "#0072B2",
    "rxx": "#D55E00",
    "cz": "#009E73",
}
ENTANGLER_MARKERS = {"rzz": "o", "rxx": "s", "cz": "D"}
LAYER_COLORS = {
    2: "#1f77b4",
    4: "#ff7f0e",
    6: "#2ca02c",
    8: "#d62728",
    10: "#9467bd",
    12: "#8c564b",
    14: "#e377c2",
    16: "#7f7f7f",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.6,
            "legend.frameon": False,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def light_grid(ax: plt.Axes, *, y_only: bool = False) -> None:
    axis = "y" if y_only else "both"
    ax.grid(True, which="major", axis=axis, color="#D9D9D9", linewidth=0.55, alpha=0.75)
    ax.grid(True, which="minor", axis=axis, color="#ECECEC", linewidth=0.35, alpha=0.55)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_df = pd.read_csv(TABLES_DIR / "ansatz_entangler_per_run_results.csv")
    summary_df = pd.read_csv(TABLES_DIR / "ansatz_entangler_summary.csv")
    summary_df["entangler_label"] = pd.Categorical(
        summary_df["entangler_label"],
        categories=ENTANGLER_ORDER,
        ordered=True,
    )
    summary_df = summary_df.sort_values(["entangler_label", "n_layers"])
    return run_df, summary_df


def read_run_config() -> dict:
    config_path = RESULTS_DIR / "run_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def plot_summary_figure(summary_df: pd.DataFrame, config: dict) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.35, 2.35),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.12},
    )

    convergence_threshold = float(summary_df["convergence_threshold"].iloc[0])

    ax = axes[0]
    for entangler in ENTANGLER_ORDER:
        group = summary_df[summary_df["entangler_label"].astype(str) == entangler]
        if group.empty:
            continue
        ax.plot(
            group["n_layers"],
            group["kl_final_median"],
            color=ENTANGLER_COLORS[entangler],
            marker=ENTANGLER_MARKERS[entangler],
            label=ENTANGLER_LABELS[entangler],
        )
        if (group["kl_final_q75"] > group["kl_final_q25"]).any():
            ax.fill_between(
                group["n_layers"].to_numpy(dtype=float),
                group["kl_final_q25"].to_numpy(dtype=float),
                group["kl_final_q75"].to_numpy(dtype=float),
                color=ENTANGLER_COLORS[entangler],
                alpha=0.16,
                linewidth=0,
            )
    ax.axhline(
        convergence_threshold,
        color="#555555",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
    )
    ax.text(
        0.03,
        0.08,
        f"target KL = {convergence_threshold:g}",
        transform=ax.transAxes,
        color="#555555",
        fontsize=7.3,
    )
    ax.set_yscale("log")
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel(r"Median final KL$(p_{\rm tgt}\Vert p_\theta)$")
    ax.set_xticks(sorted(summary_df["n_layers"].unique()))
    light_grid(ax)
    add_panel_label(ax, "A")

    ax = axes[1]
    for entangler in ENTANGLER_ORDER:
        group = summary_df[summary_df["entangler_label"].astype(str) == entangler]
        if group.empty:
            continue
        ax.plot(
            group["ansatz_two_qubit_depth"],
            group["kl_final_median"],
            color=ENTANGLER_COLORS[entangler],
            marker=ENTANGLER_MARKERS[entangler],
            linestyle="-",
            alpha=0.95,
        )
        for _, row in group.iterrows():
            is_pareto = bool(row["pareto_efficient_kl_vs_two_qubit_depth"])
            if is_pareto:
                ax.scatter(
                    [row["ansatz_two_qubit_depth"]],
                    [row["kl_final_median"]],
                    s=84,
                    facecolors="none",
                    edgecolors=ENTANGLER_COLORS[entangler],
                    linewidths=1.1,
                    zorder=5,
                )
    ax.axhline(
        convergence_threshold,
        color="#555555",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
    )
    ax.set_yscale("log")
    ax.set_xlabel("Backend two-qubit depth")
    ax.set_ylabel("Median final KL")
    depth_layer_ticks = (
        summary_df[["ansatz_two_qubit_depth", "n_layers"]]
        .drop_duplicates()
        .sort_values("ansatz_two_qubit_depth")
    )
    ax.set_xticks(depth_layer_ticks["ansatz_two_qubit_depth"])
    ax.set_xticklabels(
        [
            f"{int(row.ansatz_two_qubit_depth)}\nL={int(row.n_layers)}"
            for row in depth_layer_ticks.itertuples(index=False)
        ]
    )
    light_grid(ax)
    add_panel_label(ax, "B")

    ax = axes[2]
    for entangler in ENTANGLER_ORDER:
        group = summary_df[summary_df["entangler_label"].astype(str) == entangler]
        if group.empty:
            continue
        ax.plot(
            group["n_layers"],
            group["neg_log10_kl_per_two_qubit_depth_median"],
            color=ENTANGLER_COLORS[entangler],
            marker=ENTANGLER_MARKERS[entangler],
            label=ENTANGLER_LABELS[entangler],
        )
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel(r"$-\log_{10}({\rm KL})$ per 2q depth")
    ax.set_xticks(sorted(summary_df["n_layers"].unique()))
    light_grid(ax, y_only=True)
    add_panel_label(ax, "C")

    handles = [
        Line2D(
            [0],
            [0],
            color=ENTANGLER_COLORS[e],
            marker=ENTANGLER_MARKERS[e],
            label=ENTANGLER_LABELS[e],
        )
        for e in ENTANGLER_ORDER
        if e in set(summary_df["entangler_label"].astype(str))
    ]
    pareto_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="#333333",
        markerfacecolor="none",
        linewidth=0,
        label="Pareto point",
    )
    fig.legend(
        handles=handles + [pareto_handle],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.12),
        columnspacing=1.6,
        handlelength=1.8,
    )

    backend = config.get("backend_name", "backend")
    fractional = config.get("use_fractional_gates", None)
    fractional_text = (
        "fractional gates enabled"
        if fractional is True
        else "fractional gates disabled"
        if fractional is False
        else "fractional gate setting unavailable"
    )
    fig.text(
        0.995,
        -0.025,
        f"Depth metrics from backend-transpiled ansatz templates on {backend}; {fractional_text}.",
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
    )

    save_figure(fig, "fig_ansatz_entangler_summary_paper")
    plt.close(fig)


def load_histories(run_df: pd.DataFrame) -> dict[tuple[str, int], list[np.ndarray]]:
    histories: dict[tuple[str, int], list[np.ndarray]] = {}
    for _, row in run_df.iterrows():
        entangler = str(row["entangler_label"])
        n_layers = int(row["n_layers"])
        theta_seed = int(row["theta_seed"])
        npz_path = (
            RESULTS_DIR
            / entangler
            / f"L{n_layers:02d}"
            / f"seed_{theta_seed:06d}"
            / f"qcbm_{entangler}_L{n_layers:02d}_seed{theta_seed:06d}.npz"
        )
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=False)
        histories.setdefault((entangler, n_layers), []).append(
            np.asarray(data["best_kl_history"], dtype=float)
        )
    return histories


def median_history(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(curve) for curve in curves)
    padded = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        padded[i, : len(curve)] = np.maximum(curve, 1e-15)
    x = np.arange(max_len)
    median = np.nanmedian(padded, axis=0)
    q25 = np.nanpercentile(padded, 25, axis=0)
    q75 = np.nanpercentile(padded, 75, axis=0)
    return x, median, np.vstack([q25, q75])


def plot_training_trajectories(run_df: pd.DataFrame, config: dict) -> None:
    histories = load_histories(run_df)
    if not histories:
        return

    layers = sorted({layer for _, layer in histories})
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.35, 2.55),
        sharey=True,
        constrained_layout=True,
    )

    threshold = float(run_df["target_entropy"].notna().any())
    threshold = float(run_df["kl_final"].min() * 0 + 1e-3) if threshold else 1e-3

    for ax, entangler in zip(axes, ENTANGLER_ORDER):
        for layer in layers:
            curves = histories.get((entangler, layer), [])
            if not curves:
                continue
            color = LAYER_COLORS.get(layer, "#444444")
            x, median, band = median_history(curves)
            if len(curves) > 1:
                ax.fill_between(
                    x,
                    band[0],
                    band[1],
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )
            for curve in curves:
                ax.plot(
                    np.arange(len(curve)),
                    np.maximum(curve, 1e-15),
                    color=color,
                    alpha=0.16 if len(curves) > 1 else 0.0,
                    linewidth=0.75,
                )
            ax.plot(x, median, color=color, label=f"L={layer}")
            ax.scatter(
                [x[-1]],
                [median[-1]],
                s=17,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
            )
        ax.axhline(
            threshold,
            color="#555555",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
        )
        ax.set_title(ENTANGLER_LABELS[entangler])
        ax.set_yscale("log")
        light_grid(ax)
    axes[0].set_ylabel("Best-so-far KL")
    layer_handles = [
        Line2D(
            [0],
            [0],
            color=LAYER_COLORS.get(layer, "#444444"),
            label=f"L={layer}",
        )
        for layer in layers
    ]
    fig.legend(
        handles=layer_handles,
        loc="upper center",
        ncol=len(layer_handles),
        bbox_to_anchor=(0.5, 1.11),
        title="Layers",
        title_fontsize=7.6,
        columnspacing=1.5,
        handlelength=1.9,
    )
    axes[1].set_xlabel("Recorded optimizer checkpoint")
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    add_panel_label(axes[2], "C")

    backend = config.get("backend_name", "backend")
    fig.text(
        0.995,
        -0.025,
        f"Training is ideal statevector; resource annotations use {backend} transpiled templates.",
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    save_figure(fig, "fig_ansatz_training_trajectories_paper")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    run_df, summary_df = load_tables()
    config = read_run_config()
    plot_summary_figure(summary_df, config)
    plot_training_trajectories(run_df, config)
    print(f"Saved paper figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
