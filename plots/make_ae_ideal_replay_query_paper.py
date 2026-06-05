from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "toys"
    / "amplitude_estimation_experiments"
    / "ideal_regime"
    / "experiment_results"
)
DEFAULT_AE_TRACE = DEFAULT_RUN_DIR / "bae_biqae_iqae_cabiqae_latentt_ideal_trace_rows.csv"
DEFAULT_DCS_TRACE = DEFAULT_RUN_DIR / "classical_mc_ideal_budget_rows.csv"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "plots_v2" / "ideal_replay_median_relative_error_paper"

ALGORITHM_ORDER = ("bae", "biqae", "cabiqae_latentt", "classical_mc")
LEGEND_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE",
    "classical_mc": "DCS",
}
STYLES = {
    "bae": {"color": "#E76F51", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
    "classical_mc": {"color": "#2A9D8F", "marker": "X"},
}
DEFAULT_TARGET_QUERIES = (
    300,
    600,
    1_000,
    1_500,
    2_500,
    4_000,
    8_000,
    12_000,
    20_000,
    40_000,
    60_000,
)
PAPER_STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.linewidth": 1.25,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6.0,
    "ytick.major.size": 6.0,
    "xtick.minor.size": 3.2,
    "ytick.minor.size": 3.2,
    "xtick.major.width": 1.05,
    "ytick.major.width": 1.05,
    "xtick.minor.width": 0.85,
    "ytick.minor.width": 0.85,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the ideal AE replay median relative-error paper plot."
    )
    parser.add_argument("--ae-trace", type=Path, default=DEFAULT_AE_TRACE)
    parser.add_argument("--dcs-trace", type=Path, default=DEFAULT_DCS_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--targets",
        default=",".join(str(value) for value in DEFAULT_TARGET_QUERIES),
        help="Comma-separated query budgets sampled from each replay trajectory.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for median confidence intervals.",
    )
    return parser.parse_args()


def parse_targets(value: str) -> np.ndarray:
    targets = np.asarray([float(part.strip()) for part in value.split(",") if part.strip()])
    targets = targets[np.isfinite(targets) & (targets > 0.0)]
    if targets.size == 0:
        raise ValueError("--targets must contain at least one positive finite query budget.")
    return np.asarray(sorted(set(float(target) for target in targets)), dtype=float)


def read_trace(path: Path, algorithms: set[str]) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = {"algorithm_key", "repetition", "query_budget_actual", "normalized_abs_error"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows = rows[rows["algorithm_key"].isin(algorithms)].copy()
    for column in ("repetition", "query_budget_actual", "normalized_abs_error"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows[
        np.isfinite(rows["repetition"])
        & np.isfinite(rows["query_budget_actual"])
        & np.isfinite(rows["normalized_abs_error"])
        & (rows["query_budget_actual"] > 0.0)
        & (rows["normalized_abs_error"] > 0.0)
    ].copy()
    return rows.sort_values(["algorithm_key", "repetition", "query_budget_actual"])


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(finite))
    if finite.size == 1 or int(samples) <= 0:
        return median, median, median
    boot = np.empty(int(samples), dtype=float)
    for idx in range(int(samples)):
        boot[idx] = float(np.median(rng.choice(finite, size=finite.size, replace=True)))
    return median, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def sample_trajectory_rows(
    rows: pd.DataFrame,
    *,
    algorithm_key: str,
    targets: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    algorithm_rows = rows[rows["algorithm_key"] == algorithm_key]
    sampled: list[dict[str, float | int | str]] = []
    by_repetition = [
        group.sort_values("query_budget_actual")
        for _, group in algorithm_rows.groupby("repetition", sort=False)
    ]
    for target in targets:
        values: list[float] = []
        actual_queries: list[float] = []
        for group in by_repetition:
            prefix = group[group["query_budget_actual"] <= float(target)]
            if prefix.empty:
                continue
            row = prefix.iloc[-1]
            values.append(float(row["normalized_abs_error"]))
            actual_queries.append(float(row["query_budget_actual"]))
        median, low, high = bootstrap_median_ci(
            np.asarray(values, dtype=float),
            samples=bootstrap_samples,
            rng=rng,
        )
        if not np.isfinite(median):
            continue
        sampled.append(
            {
                "algorithm_key": algorithm_key,
                "algorithm": LEGEND_LABELS[algorithm_key],
                "target_query_budget": float(target),
                "query_budget_actual_median": float(np.median(actual_queries)),
                "normalized_abs_error_median": median,
                "normalized_abs_error_median_ci_low": low,
                "normalized_abs_error_median_ci_high": high,
                "n_runs": int(len(values)),
            }
        )
    return sampled


def build_summary(args: argparse.Namespace) -> pd.DataFrame:
    targets = parse_targets(str(args.targets))
    ae_rows = read_trace(
        args.ae_trace,
        {"bae", "biqae", "cabiqae_latentt"},
    )
    dcs_rows = read_trace(args.dcs_trace, {"classical_mc"})
    rng = np.random.default_rng(12345)
    rows: list[dict[str, float | int | str]] = []
    combined = pd.concat([ae_rows, dcs_rows], ignore_index=True)
    for algorithm_key in ALGORITHM_ORDER:
        rows.extend(
            sample_trajectory_rows(
                combined,
                algorithm_key=algorithm_key,
                targets=targets,
                bootstrap_samples=max(0, int(args.bootstrap_samples)),
                rng=rng,
            )
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No sampled trajectory rows were available for the requested targets.")
    return summary


def _ci_errorbar(center: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    lower = np.where(np.isfinite(low), center - low, 0.0)
    upper = np.where(np.isfinite(high), high - center, 0.0)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    lower = np.minimum(lower, 0.95 * center)
    upper = np.minimum(upper, 4.0 * center)
    return np.vstack([lower, upper])


def add_query_guides(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x_min = float(summary["query_budget_actual_median"].min())
    x_max = float(summary["query_budget_actual_median"].max())
    first_target = float(summary["target_query_budget"].min())
    first_rows = summary[summary["target_query_budget"] == first_target]
    anchor_y = float(first_rows["normalized_abs_error_median"].median())
    guide_x = np.geomspace(x_min, x_max, 300)
    ax.loglog(
        guide_x,
        anchor_y * (x_min / guide_x),
        color="#303030",
        linestyle="--",
        linewidth=1.25,
        label=r"$O(1/N)$",
        zorder=1,
    )
    ax.loglog(
        guide_x,
        anchor_y * np.sqrt(x_min / guide_x),
        color="#303030",
        linestyle=":",
        linewidth=1.55,
        label=r"$O(1/\sqrt{N})$",
        zorder=1,
    )


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output.with_name(f"{output.name}_summary.csv"), index=False)

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.55, 3.85), constrained_layout=True)
        add_query_guides(ax, summary)
        for algorithm_key in ALGORITHM_ORDER:
            group = summary[summary["algorithm_key"] == algorithm_key].sort_values(
                "query_budget_actual_median"
            )
            if group.empty:
                continue
            style = STYLES[algorithm_key]
            x_values = group["query_budget_actual_median"].to_numpy(dtype=float)
            y_values = group["normalized_abs_error_median"].to_numpy(dtype=float)
            low = group["normalized_abs_error_median_ci_low"].to_numpy(dtype=float)
            high = group["normalized_abs_error_median_ci_high"].to_numpy(dtype=float)
            ax.errorbar(
                x_values,
                y_values,
                yerr=_ci_errorbar(y_values, low, high),
                fmt=style["marker"],
                color=style["color"],
                linestyle="-",
                linewidth=1.75,
                markersize=5.0,
                markeredgewidth=0.9,
                elinewidth=0.95,
                capsize=3.0,
                capthick=0.95,
                label=LEGEND_LABELS[algorithm_key],
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_q$")
        ax.set_ylabel("Median relative error")
        ax.set_xlim(240.0, 70_000.0)
        ax.set_ylim(5.0e-5, 6.0e-2)
        ax.grid(True, which="major", color="#c7c7c7", linewidth=0.78, alpha=0.72)
        ax.grid(True, which="minor", color="#e7e7e7", linewidth=0.43, alpha=0.82)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(1.25)
        ax.legend(frameon=False, loc="lower left", handlelength=2.7)
        fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    make_plot(summary, args.output)
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_name(f'{args.output.name}_summary.csv')}")


if __name__ == "__main__":
    main()
