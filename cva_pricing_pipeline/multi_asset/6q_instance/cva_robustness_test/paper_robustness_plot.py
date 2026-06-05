from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
RESULTS_CSV = ROOT / "cva_robustness_case_results.csv"


CASE_ORDER = [
    "base",
    "call_down_10pct",
    "call_up_10pct",
    "put_down_10pct",
    "put_up_10pct",
    "both_down_10pct",
    "both_up_10pct",
]

CASE_LABELS = {
    "base": "Base",
    "call_down_10pct": "Call\n-10%",
    "call_up_10pct": "Call\n+10%",
    "put_down_10pct": "Put\n-10%",
    "put_up_10pct": "Put\n+10%",
    "both_down_10pct": "Both\n-10%",
    "both_up_10pct": "Both\n+10%",
}


BLUE = "#1F77B4"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
LIGHT_GRAY = "#D8D8D8"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 15.5,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 12.0,
            "lines.linewidth": 1.25,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.8,
            "ytick.major.size": 4.8,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["status"].eq("ok")].copy()
    numeric_cols = [
        "call_strike_scale",
        "put_strike_scale",
        "cva_statevector",
        "cva_classical_reference",
        "absolute_error",
        "signed_error",
        "absolute_relative_error_pct",
        "signed_relative_error_pct",
        "classical_cva_std_err_mc_continuous",
        "exposure_l2_statevector",
        "shots",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    order = {case: i for i, case in enumerate(CASE_ORDER)}
    df["case_order"] = df["case_id"].map(order).fillna(999).astype(int)
    df["case_label"] = df["case_id"].map(CASE_LABELS).fillna(df["case_id"])
    return df.sort_values("case_order").reset_index(drop=True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        clip_on=False,
    )


def plot_scenario_map(ax: plt.Axes, df: pd.DataFrame) -> None:
    rel = df["signed_relative_error_pct"].to_numpy(float)
    vmax = float(np.nanmax(np.abs(rel)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    scatter = ax.scatter(
        df["call_strike_scale"],
        df["put_strike_scale"],
        c=rel,
        s=170,
        cmap="RdBu_r",
        norm=norm,
        edgecolor="black",
        linewidth=0.55,
        zorder=3,
    )
    for _, row in df.iterrows():
        ax.text(
            float(row["call_strike_scale"]),
            float(row["put_strike_scale"]),
            f"{float(row['signed_relative_error_pct']):+.1f}",
            ha="center",
            va="center",
            fontsize=8.7,
            color="black",
            zorder=4,
        )
    ax.set_xlabel("Call-strike scale")
    ax.set_ylabel("Put-strike scale")
    ax.set_xlim(0.86, 1.14)
    ax.set_ylim(0.86, 1.14)
    ax.set_xticks([0.9, 1.0, 1.1])
    ax.set_yticks([0.9, 1.0, 1.1])
    ax.grid(True, color=LIGHT_GRAY, linewidth=0.45, alpha=0.65, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Signed relative error (%)")
    cbar.ax.tick_params(labelsize=5.0)
    panel_label(ax, "")


def plot_cva_values(ax: plt.Axes, df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    classical = df["cva_classical_reference"].to_numpy(float)
    quantum = df["cva_statevector"].to_numpy(float)
    classical_se = df["classical_cva_std_err_mc_continuous"].to_numpy(float)
    finite_se = np.where(np.isfinite(classical_se), classical_se, 0.0)

    ax.errorbar(
        x - 0.11,
        classical,
        yerr=finite_se,
        fmt="o",
        color=BLUE,
        mfc="white",
        mec=BLUE,
        mew=0.9,
        ms=4.0,
        capsize=2.3,
        elinewidth=0.8,
        label="Classical reference",
        zorder=3,
    )
    ax.plot(
        x + 0.11,
        quantum,
        "s",
        color=ORANGE,
        mfc="white",
        mec=ORANGE,
        mew=0.9,
        ms=4.0,
        label="Statevector CVA",
        zorder=4,
    )
    for xi, cva_c, cva_q in zip(x, classical, quantum):
        ax.plot([xi - 0.11, xi + 0.11], [cva_c, cva_q], color="#BDBDBD", linewidth=0.7, zorder=1)
    ax.set_ylabel("CVA")
    ax.set_xticks(x)
    ax.set_xticklabels(df["case_label"], rotation=0)
    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.set_ylim(0.18, 1.02)
    ax.grid(True, axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.65)
    ax.legend(loc="best", handlelength=1.6)
    panel_label(ax, "")


def plot_relative_errors(ax: plt.Axes, df: pd.DataFrame) -> None:
    x = np.arange(len(df))
    rel = df["signed_relative_error_pct"].to_numpy(float)
    abs_rel = np.abs(rel)
    colors = np.where(rel >= 0.0, ORANGE, GREEN)

    ax.bar(x, rel, width=0.58, color=colors, alpha=0.88, edgecolor="black", linewidth=0.45)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for xi, yi, ai in zip(x, rel, abs_rel):
        va = "bottom" if yi >= 0.0 else "top"
        offset = 2.1 if yi >= 0.0 else -2.1
        ax.text(xi, yi + offset, f"{ai:.1f}%", ha="center", va=va, fontsize=9.5)
    ax.set_ylabel("Relative error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["case_label"], rotation=0)
    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.set_ylim(-36.0, 116.0)
    ax.grid(True, axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.65)
    ax.text(
        0.015,
        0.955,
        rf"mean $|\epsilon_r|={abs_rel.mean():.1f}\%$, RMSE $={np.sqrt(np.mean(rel**2)):.1f}\%$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.0,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
    )


def write_latex_table(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Scenario": CASE_LABELS.get(str(row["case_id"]), str(row["case_id"]))
                .replace("\n", " ")
                .replace("%", "\\%"),
                "Call scale": f"{float(row['call_strike_scale']):.1f}",
                "Put scale": f"{float(row['put_strike_scale']):.1f}",
                "Classical CVA": f"{float(row['cva_classical_reference']):.4f}",
                "Statevector CVA": f"{float(row['cva_statevector']):.4f}",
                "Relative error (\\%)": f"{float(row['signed_relative_error_pct']):+.1f}",
                "Exposure L2": f"{float(row['exposure_l2_statevector']):.3f}",
            }
        )
    table = pd.DataFrame(rows)
    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lcccccc",
        caption=(
            "Robustness-test results for the multi-asset CVA pipeline under "
            "10\\% call- and put-strike perturbations."
        ),
        label="tab:cva_robustness_results",
    )
    (PLOTS_DIR / "paper_robustness_results_table.tex").write_text(latex, encoding="utf-8")
    table.to_csv(PLOTS_DIR / "paper_robustness_results_table.csv", index=False)


def main() -> None:
    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    fig, ax_err = plt.subplots(figsize=(6.85, 3.55), constrained_layout=True)
    plot_relative_errors(ax_err, df)
    fig.savefig(PLOTS_DIR / "paper_robustness_summary.png", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(PLOTS_DIR / "paper_robustness_summary.pdf", bbox_inches="tight", pad_inches=0.035)
    write_latex_table(df)
    plt.close(fig)


if __name__ == "__main__":
    main()
