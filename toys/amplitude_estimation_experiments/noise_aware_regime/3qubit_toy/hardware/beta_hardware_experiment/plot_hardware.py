from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parents[4]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
TOY_DIR = Path(__file__).resolve().parents[2]
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from common_utils.plotting_utils import (  # noqa: E402
    add_query_scaling_guides,
    log_binned_median_se,
    plot_median_se_errorbar,
)
from ae_final_error_plots import plot_final_error_figures, plot_final_error_scatter
from hardware_replay_query_plot import (
    DEFAULT_BUDGETS,
    append_monte_carlo_rows,
    budget_rows_from_trace_rows,
    plot_hardware_replay_actual_queries,
)


STYLE = {
    "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
    "BIQAE": {"color": "#A23B72", "marker": "s"},
    "BAE": {"color": "#E07A5F", "marker": "^"},
    "Classical MC": {"color": "#2A9D8F", "marker": "X"},
}


def maybe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def append_optional_monte_carlo(
    rows: pd.DataFrame,
    monte_carlo_path: Path,
    *,
    include_monte_carlo: bool,
) -> pd.DataFrame:
    if not include_monte_carlo:
        return rows
    if not monte_carlo_path.exists() or monte_carlo_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{monte_carlo_path} does not exist. Run montecarlo_path.py first."
        )
    monte_carlo = pd.read_csv(monte_carlo_path)
    if monte_carlo.empty:
        return rows
    return pd.concat([rows, monte_carlo], ignore_index=True, sort=False)


def column(df: pd.DataFrame, name: str, fallback: str) -> pd.Series:
    if name in df:
        return df[name]
    return df[fallback]


def replay_success_counts(run_dir: Path) -> tuple[dict[str, int], int | None]:
    final = maybe_read(run_dir / "replay_final_rows.csv")
    counts: dict[str, int] = {}
    if not final.empty and "algorithm" in final:
        counts = final.groupby("algorithm").size().astype(int).to_dict()
    total: int | None = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            total = int(config["replay_repetitions"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            total = None
    return counts, total


def budget_label(algorithm: str, group: pd.DataFrame, success_counts: dict[str, int], total: int | None) -> str:
    label = str(algorithm)
    parts: list[str] = []
    if total is not None and total > 0:
        successes = int(success_counts.get(str(algorithm), 0))
        parts.append(f"ok={100.0 * successes / int(total):.0f}%")
    if parts:
        label = f"{label} ({', '.join(parts)})"
    return label


def amplification_factors(group: pd.DataFrame) -> np.ndarray:
    if "amplification_factor_median" in group:
        return group["amplification_factor_median"].to_numpy(dtype=float)
    if "grover_power_max_median" in group:
        return 2.0 * group["grover_power_max_median"].to_numpy(dtype=float) + 1.0
    return np.full(len(group), np.nan, dtype=float)


def actual_trace_label(algorithm: str, group: pd.DataFrame) -> str:
    if "repetition" not in group:
        return str(algorithm)
    n_repetitions = int(group["repetition"].nunique())
    return f"{algorithm} ({n_repetitions} reps)"


def plot_amplification(run_dir: Path, out_dir: Path) -> None:
    points = maybe_read(run_dir / "amplification_points.csv")
    if points.empty:
        return
    points = points.sort_values("grover_power")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        points["grover_power"],
        points["p_ideal"],
        color="black",
        marker="o",
        linewidth=1.5,
        label="Ideal",
    )
    ax.errorbar(
        points["grover_power"],
        points["p_hw_mitigated"],
        yerr=points.get("p_hw_mitigated_se"),
        color="#1F6F8B",
        marker="s",
        linewidth=1.5,
        capsize=3,
        label="Hardware mitigated",
    )
    ax.scatter(
        points["grover_power"],
        points["p_hw_raw"],
        color="#E07A5F",
        marker=".",
        label="Hardware raw",
    )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Grover power k")
    ax.set_ylabel("P(objective=1)")
    ax.set_title("Hardware amplification scan")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "amplification_scan.png", dpi=220)
    plt.close(fig)


def plot_direct_trace(run_dir: Path, out_dir: Path) -> None:
    trace = maybe_read(run_dir / "direct_trace_rows.csv")
    if trace.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    a_true = float(trace["a_true"].dropna().iloc[0])
    error_guide_points: list[tuple[float, float]] = []
    for algorithm, group in trace.groupby("algorithm"):
        group = group.sort_values("query_budget")
        style = STYLE.get(str(algorithm), {"color": None, "marker": "o"})
        axes[0].plot(
            group["query_budget"],
            group["estimate"],
            marker=style["marker"],
            color=style["color"],
            label=algorithm,
        )
        error = column(group, "normalized_abs_error", "nrmse").to_numpy(dtype=float)
        query_budget = group["query_budget"].to_numpy(dtype=float)
        x_bins, y_bins, yerr_bins, _ = log_binned_median_se(
            query_budget,
            error,
            max_bins=14,
            min_points_per_bin=1,
        )
        if x_bins.size:
            order = np.argsort(x_bins)
            x_bins = x_bins[order]
            y_bins = y_bins[order]
            yerr_bins = yerr_bins[order]
            error_guide_points.extend(
                (float(x), float(y))
                for x, y in zip(x_bins, y_bins)
                if np.isfinite(x) and np.isfinite(y) and x > 0.0 and y > 0.0
            )
            plot_median_se_errorbar(
                axes[1],
                x_bins,
                y_bins,
                yerr_bins,
                style=style,
                label=algorithm,
            )
    axes[0].axhline(a_true, color="black", linestyle="--", linewidth=1.0)
    add_query_scaling_guides(axes[1], error_guide_points)
    axes[0].set_xlabel("State-preparation calls")
    axes[0].set_ylabel("Estimate")
    axes[0].set_title("Direct live estimate")
    axes[1].set_xlabel("State-preparation calls")
    axes[1].set_ylabel("Normalized absolute error")
    axes[1].set_title("Direct live error")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "direct_live_trace.png", dpi=220)
    plt.close(fig)


def plot_replay_budget(
    run_dir: Path,
    out_dir: Path,
    *,
    max_queries: float | None = None,
    include_monte_carlo: bool = False,
    monte_carlo_budget_rows: Path | None = None,
    x_query_stat: str = "mean",
) -> None:
    plot_replay_actual_queries(
        run_dir,
        out_dir,
        output_stem="hardware_replay_budget",
        max_queries=max_queries,
        include_monte_carlo=include_monte_carlo,
        monte_carlo_budget_rows=monte_carlo_budget_rows,
        x_query_stat=x_query_stat,
    )


def plot_replay_actual_queries(
    run_dir: Path,
    out_dir: Path,
    *,
    output_stem: str = "hardware_replay_actual_queries",
    max_bins: int = 12,
    min_points_per_bin: int = 100,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 12345,
    max_queries: float | None = None,
    drop_binned_point_indices: dict[str, tuple[int, ...]] | None = None,
    include_monte_carlo: bool = False,
    monte_carlo_budget_rows: Path | None = None,
    x_query_stat: str = "mean",
) -> None:
    budget_rows = maybe_read(run_dir / "replay_budget_rows.csv")
    if budget_rows.empty:
        trace = maybe_read(run_dir / "replay_trace_rows.csv")
        if trace.empty or "algorithm" not in trace or "query_budget" not in trace:
            return
        budget_rows = budget_rows_from_trace_rows(trace, DEFAULT_BUDGETS)
    if budget_rows.empty or "algorithm" not in budget_rows:
        return
    if "normalized_abs_error" not in budget_rows and "nrmse" not in budget_rows:
        return
    budget_rows = append_monte_carlo_rows(
        budget_rows,
        monte_carlo_budget_rows or (run_dir / "montecarlo_budget_rows.csv"),
        include_monte_carlo=include_monte_carlo,
    )

    plot_hardware_replay_actual_queries(
        budget_rows,
        out_dir / f"{output_stem}.png",
        summary_path=out_dir / f"{output_stem}_summary.csv",
        max_queries=max_queries,
        max_bins=max_bins,
        min_points_per_bin=min_points_per_bin,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        drop_binned_point_indices=drop_binned_point_indices,
        x_query_stat=x_query_stat,
    )


def plot_final_comparison(
    run_dir: Path,
    out_dir: Path,
    *,
    include_monte_carlo: bool = False,
    monte_carlo_final_rows: Path | None = None,
) -> None:
    frames = []
    for name in ("direct_final_rows.csv", "replay_final_rows.csv"):
        df = maybe_read(run_dir / name)
        if not df.empty:
            frames.append(df)
    if include_monte_carlo:
        monte_carlo = append_optional_monte_carlo(
            pd.DataFrame(),
            monte_carlo_final_rows or (run_dir / "montecarlo_final_rows.csv"),
            include_monte_carlo=True,
        )
        if not monte_carlo.empty:
            frames.append(monte_carlo)
    if not frames:
        return
    final = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = []
    values = []
    colors = []
    for (run_kind, algorithm), group in final.groupby(["run_kind", "algorithm"]):
        labels.append(f"{run_kind}\n{algorithm}")
        values.append(
            float(
                np.nanmedian(
                    column(group, "final_normalized_abs_error", "final_nrmse").to_numpy(dtype=float)
                )
            )
        )
        colors.append(STYLE.get(str(algorithm), {}).get("color"))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Median final normalized absolute error")
    ax.set_title("Final estimates")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "final_comparison.png", dpi=220)
    plt.close(fig)


def plot_replay_final_error_figures(
    run_dir: Path,
    out_dir: Path,
    *,
    include_monte_carlo: bool = False,
    monte_carlo_final_rows: Path | None = None,
) -> None:
    final_path = run_dir / "replay_final_rows.csv"
    if not final_path.exists() or final_path.stat().st_size == 0:
        return
    if include_monte_carlo:
        final_rows = append_optional_monte_carlo(
            pd.read_csv(final_path),
            monte_carlo_final_rows or (run_dir / "montecarlo_final_rows.csv"),
            include_monte_carlo=True,
        )
        plot_final_error_scatter(
            final_rows,
            out_dir / "triple_gaussian_error_queries.png",
            x_kind="queries",
            title="Final error versus query cost under hardware replay",
            summary_path=out_dir / "triple_gaussian_error_queries_summary.csv",
            pdf_path=out_dir / "triple_gaussian_error_queries.pdf",
        )
        if "runtime_wall_seconds" in final_rows or "runtime_seconds" in final_rows:
            plot_final_error_scatter(
                final_rows,
                out_dir / "triple_gaussian_error_runtime.png",
                x_kind="runtime",
                title="Final error versus runtime under hardware replay",
                summary_path=out_dir / "triple_gaussian_error_runtime_summary.csv",
                pdf_path=out_dir / "triple_gaussian_error_runtime.pdf",
            )
        return
    plot_final_error_figures(
        final_path,
        out_dir,
        title_suffix="under hardware replay",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot beta hardware experiment artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--max-queries",
        type=float,
        default=None,
        help="Only plot replay error-vs-queries rows with query cost at or below this value.",
    )
    parser.add_argument(
        "--include-monte-carlo",
        action="store_true",
        help="Append montecarlo_budget_rows.csv and montecarlo_final_rows.csv to replay plots.",
    )
    parser.add_argument("--monte-carlo-budget-rows", type=Path, default=None)
    parser.add_argument("--monte-carlo-final-rows", type=Path, default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_amplification(run_dir, out_dir)
    plot_direct_trace(run_dir, out_dir)
    plot_replay_budget(
        run_dir,
        out_dir,
        max_queries=args.max_queries,
        include_monte_carlo=args.include_monte_carlo,
        monte_carlo_budget_rows=args.monte_carlo_budget_rows,
    )
    plot_replay_actual_queries(
        run_dir,
        out_dir,
        max_queries=args.max_queries,
        include_monte_carlo=args.include_monte_carlo,
        monte_carlo_budget_rows=args.monte_carlo_budget_rows,
        drop_binned_point_indices={
            "BAE": (-2, -1),
            "CABIQAE": (-2, -1),
        },
    )
    plot_final_comparison(
        run_dir,
        out_dir,
        include_monte_carlo=args.include_monte_carlo,
        monte_carlo_final_rows=args.monte_carlo_final_rows,
    )
    plot_replay_final_error_figures(
        run_dir,
        out_dir,
        include_monte_carlo=args.include_monte_carlo,
        monte_carlo_final_rows=args.monte_carlo_final_rows,
    )
    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
