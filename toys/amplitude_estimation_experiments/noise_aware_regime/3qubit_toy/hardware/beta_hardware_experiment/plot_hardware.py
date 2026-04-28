from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = {
    "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
    "BIQAE": {"color": "#A23B72", "marker": "s"},
    "BAE": {"color": "#E07A5F", "marker": "^"},
}


def maybe_read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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
    if "n_runs" in group:
        n_runs = group["n_runs"].dropna().astype(int)
        if not n_runs.empty:
            n_min = int(n_runs.min())
            n_max = int(n_runs.max())
            parts.append(f"n={n_min}" if n_min == n_max else f"n={n_min}-{n_max}")
    if parts:
        label = f"{label} ({', '.join(parts)})"
    return label


def amplification_factors(group: pd.DataFrame) -> np.ndarray:
    if "amplification_factor_median" in group:
        return group["amplification_factor_median"].to_numpy(dtype=float)
    if "grover_power_max_median" in group:
        return 2.0 * group["grover_power_max_median"].to_numpy(dtype=float) + 1.0
    return np.full(len(group), np.nan, dtype=float)


def bootstrap_median_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    median = float(np.median(values))
    if len(values) == 1:
        return median, median, median
    rng = np.random.default_rng(12345)
    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_medians[i] = np.median(rng.choice(values, size=len(values), replace=True))
    low = float(np.quantile(boot_medians, alpha / 2))
    high = float(np.quantile(boot_medians, 1 - alpha / 2))
    return median, low, high


def replay_trace_error_band(run_dir: Path, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    trace = maybe_read(run_dir / "replay_trace_rows.csv")
    if trace.empty or "algorithm" not in trace or "repetition" not in trace:
        return None
    if "normalized_abs_error" not in trace and "nrmse" not in trace:
        return None
    error_col = "normalized_abs_error" if "normalized_abs_error" in trace else "nrmse"
    algorithm = str(group["algorithm"].iloc[0])
    alg_trace = trace[trace["algorithm"].astype(str) == algorithm].copy()
    if alg_trace.empty:
        return None
    alg_trace["query_budget"] = alg_trace["query_budget"].astype(float)
    alg_trace[error_col] = alg_trace[error_col].astype(float)

    lows: list[float] = []
    highs: list[float] = []
    for budget in group["budget"].to_numpy(dtype=float):
        values: list[float] = []
        for _, rep_rows in alg_trace.groupby("repetition"):
            ordered = rep_rows.sort_values("query_budget")
            if float(ordered["query_budget"].iloc[-1]) < float(budget):
                continue
            candidates = ordered[ordered["query_budget"] <= float(budget)]
            if candidates.empty:
                continue
            chosen = candidates.iloc[-1]
            values.append(float(chosen[error_col]))
        _, low, high = bootstrap_median_ci(np.asarray(values, dtype=float))
        lows.append(low)
        highs.append(high)
    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def median_ci_band(run_dir: Path, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "normalized_abs_error_median_ci_low" in group and "normalized_abs_error_median_ci_high" in group:
        return (
            group["normalized_abs_error_median_ci_low"].to_numpy(dtype=float),
            group["normalized_abs_error_median_ci_high"].to_numpy(dtype=float),
        )
    trace_band = replay_trace_error_band(run_dir, group)
    if trace_band is not None:
        return trace_band
    return (
        column(group, "normalized_abs_error_q25", "nrmse_q25").to_numpy(dtype=float),
        column(group, "normalized_abs_error_q75", "nrmse_q75").to_numpy(dtype=float),
    )


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
        axes[1].semilogy(
            group["query_budget"],
            column(group, "normalized_abs_error", "nrmse"),
            marker=style["marker"],
            color=style["color"],
            label=algorithm,
        )
    axes[0].axhline(a_true, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("State-preparation calls")
    axes[0].set_ylabel("Estimate")
    axes[0].set_title("Direct live estimate")
    axes[1].set_xlabel("State-preparation calls")
    axes[1].set_ylabel("Normalized absolute error")
    axes[1].set_title("Direct live error")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "direct_live_trace.png", dpi=220)
    plt.close(fig)


def plot_replay_budget(run_dir: Path, out_dir: Path) -> None:
    summary = maybe_read(run_dir / "budget_summary.csv")
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    guide_points: list[tuple[float, float]] = []
    success_counts, total_repetitions = replay_success_counts(run_dir)
    for algorithm, group in summary.groupby("algorithm"):
        group = group.sort_values("budget")
        style = STYLE.get(str(algorithm), {"color": None, "marker": "o"})
        y_median = column(group, "normalized_abs_error_median", "nrmse_median")
        if "normalized_abs_error_se" in group:
            yerr = group["normalized_abs_error_se"].to_numpy(dtype=float)
        else:
            yerr = np.zeros(len(group), dtype=float)
        y_values = y_median.to_numpy(dtype=float)
        yerr = np.asarray(yerr, dtype=float)
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        yerr = np.minimum(yerr, np.maximum(0.0, 0.95 * y_values))
        guide_points.extend(
            (float(x), float(y))
            for x, y in zip(group["budget"], y_median)
            if np.isfinite(float(x)) and np.isfinite(float(y)) and float(x) > 0 and float(y) > 0
        )
        ax.errorbar(
            group["budget"],
            y_values,
            yerr=yerr,
            marker=style["marker"],
            color=style["color"],
            linewidth=1.6,
            elinewidth=1.0,
            capsize=3,
            label=budget_label(str(algorithm), group, success_counts, total_repetitions),
        )
        for x_val, y_val, k_val in zip(group["budget"].to_numpy(dtype=float), y_values, amplification_factors(group)):
            if np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(k_val):
                ax.annotate(
                    f"K={int(np.rint(k_val))}",
                    xy=(x_val, y_val),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=style["color"],
                    alpha=0.9,
                )
    if guide_points:
        budgets = np.asarray(sorted(summary["budget"].dropna().unique()), dtype=float)
        budgets = budgets[budgets > 0]
        if budgets.size:
            x0 = float(budgets[0])
            y0_values = [y for x, y in guide_points if np.isclose(x, x0)]
            y0 = float(np.nanmedian(y0_values)) if y0_values else float(np.nanmedian([y for _, y in guide_points]))
            ax.loglog(
                budgets,
                y0 * (x0 / budgets),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="O(1/N)",
            )
            ax.loglog(
                budgets,
                y0 * np.sqrt(x0 / budgets),
                color="black",
                linestyle=":",
                linewidth=1.4,
                label=r"O(1/\sqrt{N})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Query budget")
    ax.set_ylabel("Median normalized absolute error (SE bars)")
    ax.set_title("Hardware replay budget comparison")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "hardware_replay_budget.png", dpi=220)
    plt.close(fig)


def plot_final_comparison(run_dir: Path, out_dir: Path) -> None:
    frames = []
    for name in ("direct_final_rows.csv", "replay_final_rows.csv"):
        df = maybe_read(run_dir / name)
        if not df.empty:
            frames.append(df)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot beta hardware experiment artifacts.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    plot_amplification(run_dir, out_dir)
    plot_direct_trace(run_dir, out_dir)
    plot_replay_budget(run_dir, out_dir)
    plot_final_comparison(run_dir, out_dir)
    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
