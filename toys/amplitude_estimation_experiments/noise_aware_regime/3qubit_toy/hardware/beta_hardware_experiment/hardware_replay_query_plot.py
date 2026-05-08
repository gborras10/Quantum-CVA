from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[5]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from toys.amplitude_estimation_experiments.ideal_regime.ideal_utils import (  # noqa: E402
    aggregate_budget_summary,
    plot_budget_summary,
    save_csv,
)


HARDWARE_REPLAY_ALGORITHMS = ("bae", "biqae", "cabiqae_latentt")
HARDWARE_REPLAY_ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE",
}
HARDWARE_REPLAY_ALGORITHM_STYLES = {
    "bae": {"color": "#E07A5F", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
}
DEFAULT_INPUT_DIR = CURRENT_DIR / "experiment_results" / "csv_results"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "experiment_results" / "plots"
DEFAULT_BUDGETS = (
    128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096, 6144,
    8192, 12288, 16384, 24576, 32768,
    49152, 65536, 98304, 131072
)


# Plot parameters
MAX_BINS = 12
MIN_POINTS_PER_BIN = 5
BOOTSTRAP_SAMPLES = 2000
MAX_QUERIES = 50000


def parse_budget_list(raw: str | Sequence[int | float]) -> list[int]:
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    budgets = sorted({int(float(value)) for value in values})
    if not budgets:
        raise ValueError("At least one budget is required.")
    return budgets


def _as_float(value: object, default: float = np.nan) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _algorithm_aliases(algorithm: str) -> set[str]:
    normalized = str(algorithm).strip()
    aliases = {normalized, normalized.lower()}
    for key, label in HARDWARE_REPLAY_ALGORITHM_LABELS.items():
        if normalized == key or normalized.lower() == key.lower():
            aliases.update({key, key.lower(), label, label.lower()})
        if normalized == label or normalized.lower() == label.lower():
            aliases.update({key, key.lower(), label, label.lower()})
    return aliases


def parse_drop_point_specs(raw: str | Sequence[str] | None) -> set[tuple[str, int]]:
    """Parse point filters such as 'cabiqae_latentt:65536,BIQAE:131072'."""
    if raw is None:
        return set()
    parts = [raw] if isinstance(raw, str) else list(raw)
    specs: set[tuple[str, int]] = set()
    for item in parts:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            normalized = token.replace("@", ":")
            if ":" not in normalized:
                raise ValueError(
                    "Drop point specs must use 'algorithm:budget', "
                    f"received {token!r}."
                )
            algorithm, budget_raw = normalized.rsplit(":", 1)
            algorithm = algorithm.strip()
            if not algorithm:
                raise ValueError(f"Drop point spec has an empty algorithm: {token!r}.")
            specs.add((algorithm, int(float(budget_raw.strip()))))
    return specs


def drop_summary_points(
    summary_rows: Sequence[Mapping[str, object]],
    drop_points: set[tuple[str, int]],
) -> list[dict[str, object]]:
    if not drop_points:
        return [dict(row) for row in summary_rows]

    expanded = {
        (alias, int(budget))
        for algorithm, budget in drop_points
        for alias in _algorithm_aliases(algorithm)
    }
    kept: list[dict[str, object]] = []
    dropped: list[tuple[str, int]] = []
    for row in summary_rows:
        algorithm_key = str(row.get("algorithm_key", ""))
        algorithm_label = str(row.get("algorithm", ""))
        budget = int(round(_as_float(row.get("budget"))))
        should_drop = (
            (algorithm_key, budget) in expanded
            or (algorithm_key.lower(), budget) in expanded
            or (algorithm_label, budget) in expanded
            or (algorithm_label.lower(), budget) in expanded
        )
        if should_drop:
            dropped.append((algorithm_key or algorithm_label, budget))
        else:
            kept.append(dict(row))

    if dropped:
        formatted = ", ".join(f"{algorithm}:{budget}" for algorithm, budget in dropped)
        print(f"Dropped plot points: {formatted}")
    else:
        requested = ", ".join(f"{algorithm}:{budget}" for algorithm, budget in sorted(drop_points))
        print(f"No plot points matched --drop-points: {requested}")
    return kept


def _bootstrap_median_ci(
    values: Sequence[object],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    finite = np.asarray([_as_float(value) for value in values], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.nanmedian(finite))
    if finite.size == 1 or int(bootstrap_samples) <= 0:
        return center, center, center
    sample_indices = rng.integers(0, finite.size, size=(int(bootstrap_samples), finite.size))
    medians = np.nanmedian(finite[sample_indices], axis=1)
    alpha = 1.0 - float(confidence_level)
    low, high = np.nanquantile(medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(low), float(high)

def _total_repetitions(rows: pd.DataFrame) -> int:
    if "repetition" in rows:
        repetitions = pd.to_numeric(rows["repetition"], errors="coerce").dropna()
        if not repetitions.empty:
            return int(repetitions.nunique())
    if "rep" in rows:
        repetitions = pd.to_numeric(rows["rep"], errors="coerce").dropna()
        if not repetitions.empty:
            return int(repetitions.nunique())
    return int(max(len(rows), 1))


def budget_rows_from_trace_rows(
    trace_rows: pd.DataFrame | Sequence[Mapping[str, object]],
    budgets: Sequence[int | float],
) -> pd.DataFrame:
    frame = trace_rows.copy() if isinstance(trace_rows, pd.DataFrame) else pd.DataFrame(list(trace_rows))
    required = {"algorithm", "repetition", "query_budget"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Replay trace rows are missing required columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["query_budget"] = pd.to_numeric(frame["query_budget"], errors="coerce")
    frame = frame[frame["query_budget"].gt(0.0)]
    rows: list[dict[str, object]] = []
    for (_, _), group in frame.groupby(["algorithm", "repetition"], sort=False):
        ordered = group.sort_values("query_budget")
        if ordered.empty:
            continue
        for budget in parse_budget_list(budgets):
            candidates = ordered[ordered["query_budget"].le(float(budget))]
            if candidates.empty:
                continue
            chosen = candidates.iloc[-1].to_dict()
            chosen["run_kind"] = "hardware_replay"
            chosen["budget"] = int(budget)
            chosen["query_budget_actual"] = float(chosen["query_budget"])
            rows.append(chosen)
    return pd.DataFrame(rows)


def aggregate_fixed_budget_summary(
    rows: pd.DataFrame,
    *,
    total_repetitions: int,
    bootstrap_samples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    if rows.empty:
        return []
    frame = rows.copy()
    frame["budget"] = pd.to_numeric(frame["budget"], errors="coerce")
    query_col = "query_budget_actual" if "query_budget_actual" in frame else "query_budget"
    frame[query_col] = pd.to_numeric(frame[query_col], errors="coerce")
    frame["normalized_abs_error"] = pd.to_numeric(frame["normalized_abs_error"], errors="coerce")
    frame = frame[
        frame["budget"].notna()
        & frame[query_col].gt(0.0)
        & frame["normalized_abs_error"].gt(0.0)
    ]
    if frame.empty:
        return []

    rng = np.random.default_rng(int(bootstrap_seed))
    summary: list[dict[str, object]] = []
    for (algorithm, budget), group in frame.groupby(["algorithm", "budget"], sort=True):
        fallback = pd.Series(np.nan, index=group.index)
        n_points = int(len(group))
        repetitions = pd.to_numeric(group.get("repetition"), errors="coerce").dropna()
        n_runs = int(repetitions.nunique()) if not repetitions.empty else n_points
        query_values = group[query_col].to_numpy(dtype=float)
        error_values = group["normalized_abs_error"].to_numpy(dtype=float)
        estimates = pd.to_numeric(group.get("estimate", fallback), errors="coerce").to_numpy(dtype=float)
        abs_error = pd.to_numeric(group.get("abs_error", fallback), errors="coerce").to_numpy(dtype=float)
        k_vals = pd.to_numeric(
            group["k_max_budget"] if "k_max_budget" in group else group.get("grover_power", fallback),
            errors="coerce",
        ).to_numpy(dtype=float)
        amp_vals = pd.to_numeric(group.get("amplification_factor", fallback), errors="coerce").to_numpy(dtype=float)
        runtime = pd.to_numeric(
            group["time_to_budget_seconds"] if "time_to_budget_seconds" in group else group.get("runtime_wall_seconds", fallback),
            errors="coerce",
        ).to_numpy(dtype=float)
        median, ci_low, ci_high = _bootstrap_median_ci(
            error_values,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            rng=rng,
        )
        std = float(np.nanstd(error_values, ddof=1)) if n_points > 1 else 0.0
        summary.append(
            {
                "run_kind": "hardware_replay",
                "budget": int(round(float(budget))),
                "algorithm": str(algorithm),
                "algorithm_key": str(group.get("algorithm_key", pd.Series([algorithm])).iloc[0]),
                "n_points": n_points,
                "n_runs": n_runs,
                "total_repetitions": int(total_repetitions),
                "success_rate": float(n_runs / max(int(total_repetitions), 1)),
                "estimate_mean": float(np.nanmean(estimates)),
                "estimate_median": float(np.nanmedian(estimates)),
                "query_budget_actual": float(np.nanmean(query_values)),
                "query_budget_actual_mean": float(np.nanmean(query_values)),
                "query_budget_actual_median": float(np.nanmedian(query_values)),
                "query_budget_actual_q25": float(np.nanquantile(query_values, 0.25)),
                "query_budget_actual_q75": float(np.nanquantile(query_values, 0.75)),
                "abs_error_mean": float(np.nanmean(abs_error)),
                "abs_error_median": float(np.nanmedian(abs_error)),
                "normalized_abs_error_mean": float(np.nanmean(error_values)),
                "normalized_abs_error_median": median,
                "normalized_abs_error_median_ci_low": ci_low,
                "normalized_abs_error_median_ci_high": ci_high,
                "normalized_abs_error_std": std,
                "normalized_abs_error_se": float(std / np.sqrt(n_points)) if n_points > 0 else np.nan,
                "normalized_abs_error_q25": float(np.nanquantile(error_values, 0.25)),
                "normalized_abs_error_q75": float(np.nanquantile(error_values, 0.75)),
                "grover_power_max_median": float(np.nanmedian(k_vals)),
                "amplification_factor_median": float(np.nanmedian(amp_vals)),
                "runtime_wall_seconds_mean": float(np.nanmean(runtime)),
                "runtime_wall_seconds_median": float(np.nanmedian(runtime)),
            }
        )
    return summary


def plot_hardware_replay_actual_queries(
    rows: pd.DataFrame | Sequence[Mapping[str, object]],
    output_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    max_queries: float | None = None,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
    bootstrap_samples: int = 10000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 12345,
    drop_points: set[tuple[str, int]] | None = None,
    title: str = "Hardware replay comparison: BAE vs BIQAE vs CABIQAE",
) -> list[dict[str, object]]:
    """Build the hardware replay query plot through the ideal-regime pipeline."""
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        raise ValueError("No replay trace rows are available for plotting.")
    if max_queries is not None:
        if float(max_queries) <= 0.0:
            raise ValueError("max_queries must be positive.")
        query_col = "query_budget" if "query_budget" in frame else "query_budget_actual"
        if query_col not in frame:
            raise ValueError("Replay trace rows must contain query_budget or query_budget_actual.")
        frame[query_col] = pd.to_numeric(frame[query_col], errors="coerce")
        frame = frame[frame[query_col].le(float(max_queries))]
        if frame.empty:
            raise ValueError(f"No replay trace rows remain with {query_col} <= {float(max_queries):g}.")

    if "budget" in frame:
        summary_rows = aggregate_fixed_budget_summary(
            frame,
            total_repetitions=_total_repetitions(frame),
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
    else:
        summary_rows = aggregate_budget_summary(
            frame.to_dict("records"),
            total_repetitions=_total_repetitions(frame),
            max_bins=max_bins,
            min_points_per_bin=min_points_per_bin,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
    for row in summary_rows:
        row["run_kind"] = "hardware_replay"
    summary_rows = drop_summary_points(summary_rows, drop_points or set())

    if summary_path is not None:
        save_csv(summary_rows, summary_path)

    plot_budget_summary(
        summary_rows,
        algorithms=HARDWARE_REPLAY_ALGORITHMS,
        algorithm_labels=HARDWARE_REPLAY_ALGORITHM_LABELS,
        algorithm_styles=HARDWARE_REPLAY_ALGORITHM_STYLES,
        output_path=output_path,
        title=title,
        connect_points=True,
    )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the hardware replay actual-query plot with the ideal-regime plotting pipeline."
    )
    parser.add_argument("--budget-rows", type=Path, default=DEFAULT_INPUT_DIR / "replay_budget_rows.csv")
    parser.add_argument(
        "--budgets",
        default=",".join(str(budget) for budget in DEFAULT_BUDGETS),
        help="Comma-separated budgets used to reconstruct robust snapshots when --budget-rows is absent.",
    )
    parser.add_argument("--trace-rows", type=Path, default=DEFAULT_INPUT_DIR / "replay_trace_rows.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "hardware_replay_actual_queries.png")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "hardware_replay_actual_queries_summary.csv",
    )
    parser.add_argument(
        "--max-queries",
        type=float,
        default=None,
        help="Only plot replay trace rows with query cost at or below this value.",
    )
    parser.add_argument(
        "--drop-points",
        action="append",
        default=None,
        help=(
            "Comma-separated points to remove from the generated plot and summary, "
            "using 'algorithm:budget' or 'algorithm@budget'. Example: "
            "cabiqae_latentt:65536,cabiqae_latentt:131072"
        ),
    )
    args = parser.parse_args()

    if args.budget_rows.exists() and args.budget_rows.stat().st_size > 0:
        plot_rows = pd.read_csv(args.budget_rows)
    else:
        plot_rows = budget_rows_from_trace_rows(pd.read_csv(args.trace_rows), parse_budget_list(args.budgets))
    plot_hardware_replay_actual_queries(
        plot_rows,
        args.output,
        summary_path=args.summary,
        max_queries=args.max_queries if args.max_queries is not None else MAX_QUERIES ,
        max_bins=MAX_BINS,
        min_points_per_bin=MIN_POINTS_PER_BIN,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        drop_points=parse_drop_point_specs(args.drop_points),
    )
    print(f"Saved error-vs-queries plot: {args.output}")
    print(f"Saved error-vs-queries summary: {args.summary}")


if __name__ == "__main__":
    main()
