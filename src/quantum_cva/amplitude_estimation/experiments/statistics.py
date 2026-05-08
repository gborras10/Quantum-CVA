from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def as_float(value: Any, default: float = np.nan) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def finite_positive_pair_mask(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x_values and y_values must have the same shape.")
    return (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )


def standard_error(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.nanstd(values, ddof=1) / math.sqrt(float(values.size)))


def log_query_bin_indices(
    query_budget: np.ndarray,
    error: np.ndarray,
    *,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
) -> list[np.ndarray]:
    query_budget = np.asarray(query_budget, dtype=float)
    error = np.asarray(error, dtype=float)
    valid_indices = np.flatnonzero(finite_positive_pair_mask(query_budget, error))
    if valid_indices.size == 0:
        return []

    q_valid = query_budget[valid_indices]
    if q_valid.size <= int(max_bins):
        order = np.argsort(q_valid)
        return [np.asarray([int(valid_indices[idx])], dtype=int) for idx in order]

    q_min = float(np.nanmin(q_valid))
    q_max = float(np.nanmax(q_valid))
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_min <= 0.0 or q_max <= q_min:
        order = np.argsort(q_valid)
        return [np.asarray([int(valid_indices[idx])], dtype=int) for idx in order]

    edges = np.geomspace(q_min, q_max, num=int(max_bins) + 1)
    local_bins = np.digitize(q_valid, edges, right=False) - 1
    local_bins = np.clip(local_bins, 0, int(max_bins) - 1)

    bins: list[np.ndarray] = []
    for bin_idx in range(int(max_bins)):
        indices = valid_indices[np.flatnonzero(local_bins == bin_idx)]
        if indices.size >= int(min_points_per_bin):
            bins.append(indices.astype(int))
    return bins


def select_log_spaced_indices(
    x_values: np.ndarray,
    valid: np.ndarray,
    max_points: int,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) <= int(max_points):
        return valid_indices

    valid_x = x_values[valid_indices]
    if (
        np.all(np.isfinite(valid_x))
        and np.nanmin(valid_x) > 0.0
        and np.nanmax(valid_x) > np.nanmin(valid_x)
    ):
        targets = np.geomspace(
            float(np.nanmin(valid_x)),
            float(np.nanmax(valid_x)),
            num=int(max_points),
        )
        selected: set[int] = {int(valid_indices[0]), int(valid_indices[-1])}
        for target in targets[1:-1]:
            pos = int(np.searchsorted(valid_x, target, side="left"))
            candidates = []
            if 0 <= pos < len(valid_indices):
                candidates.append(pos)
            if 0 <= pos - 1 < len(valid_indices):
                candidates.append(pos - 1)
            if candidates:
                nearest = min(
                    candidates,
                    key=lambda idx: abs(float(valid_x[idx]) - float(target)),
                )
                selected.add(int(valid_indices[nearest]))
        return np.asarray(sorted(selected), dtype=int)

    positions = np.linspace(0, len(valid_indices) - 1, num=int(max_points))
    return valid_indices[np.unique(np.rint(positions).astype(int))]


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.mean(finite))
    if finite.size == 1 or int(n_boot) <= 0:
        return center, center, center
    rng = np.random.default_rng(12345) if rng is None else rng
    boot = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        boot[idx] = float(np.mean(rng.choice(finite, size=finite.size, replace=True)))
    low = float(np.quantile(boot, float(alpha) / 2.0))
    high = float(np.quantile(boot, 1.0 - float(alpha) / 2.0))
    return center, low, high


def bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan
    center = float(np.median(finite))
    if finite.size == 1 or int(n_boot) <= 0:
        return center, center, center
    rng = np.random.default_rng(12345) if rng is None else rng
    boot = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        boot[idx] = float(np.median(rng.choice(finite, size=finite.size, replace=True)))
    low = float(np.quantile(boot, float(alpha) / 2.0))
    high = float(np.quantile(boot, 1.0 - float(alpha) / 2.0))
    return center, low, high


def query_budget(row: Mapping[str, Any]) -> float:
    for key in (
        "query_budget",
        "query_budget_actual",
        "query_budget_actual_mean",
        "budget",
        "final_queries",
    ):
        value = as_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def aggregate_budget_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_repetitions: int | None = None,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
    group_by_budget: bool = False,
    bootstrap_samples: int = 2000,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not rows:
        return summary

    algorithms = sorted({str(row["algorithm"]) for row in rows})
    rng = np.random.default_rng(12345)
    for algorithm in algorithms:
        alg_rows = [row for row in rows if str(row["algorithm"]) == algorithm]
        query_values = np.asarray([query_budget(row) for row in alg_rows], dtype=float)
        error_values = np.asarray(
            [as_float(row.get("normalized_abs_error")) for row in alg_rows],
            dtype=float,
        )
        valid = finite_positive_pair_mask(query_values, error_values)
        if not np.any(valid):
            continue

        if group_by_budget:
            budget_values = np.asarray([as_float(row.get("budget")) for row in alg_rows])
            bin_indices = [
                np.flatnonzero(valid & (budget_values == value)).astype(int)
                for value in sorted({float(x) for x in budget_values[valid] if np.isfinite(x)})
            ]
        else:
            bin_indices = log_query_bin_indices(
                query_values,
                error_values,
                max_bins=max_bins,
                min_points_per_bin=min_points_per_bin,
            )

        for indices in bin_indices:
            if indices.size == 0:
                continue
            subset = [alg_rows[int(idx)] for idx in indices]
            n_subset = int(len(subset))
            repetitions = {
                int(as_float(row.get("repetition")))
                for row in subset
                if np.isfinite(as_float(row.get("repetition")))
            }
            n_runs = int(len(repetitions)) if repetitions else n_subset

            q_actual = np.asarray([query_budget(row) for row in subset], dtype=float)
            estimates = np.asarray([as_float(row.get("estimate")) for row in subset])
            abs_error = np.asarray([as_float(row.get("abs_error")) for row in subset])
            norm_abs = np.asarray([as_float(row.get("normalized_abs_error")) for row in subset])
            processed_abs = np.asarray([as_float(row.get("processed_abs_error")) for row in subset])
            processed_rel = np.asarray([as_float(row.get("processed_relative_error")) for row in subset])
            runtime = np.asarray(
                [
                    as_float(
                        row.get(
                            "time_to_budget_seconds",
                            row.get("runtime_wall_seconds", np.nan),
                        )
                    )
                    for row in subset
                ],
                dtype=float,
            )
            k_vals = np.asarray(
                [as_float(row.get("k_max_budget", row.get("grover_power"))) for row in subset],
                dtype=float,
            )
            amp_factors = np.asarray(
                [as_float(row.get("amplification_factor")) for row in subset],
                dtype=float,
            )

            _, abs_low, abs_high = bootstrap_mean_ci(
                abs_error,
                n_boot=bootstrap_samples,
                rng=rng,
            )
            norm_mean, norm_low, norm_high = bootstrap_mean_ci(
                norm_abs,
                n_boot=bootstrap_samples,
                rng=rng,
            )
            norm_median, norm_median_low, norm_median_high = bootstrap_median_ci(
                norm_abs,
                n_boot=bootstrap_samples,
                rng=rng,
            )
            processed_abs_median, processed_abs_low, processed_abs_high = bootstrap_median_ci(
                processed_abs,
                n_boot=bootstrap_samples,
                rng=rng,
            )
            processed_rel_median, processed_rel_low, processed_rel_high = bootstrap_median_ci(
                processed_rel,
                n_boot=bootstrap_samples,
                rng=rng,
            )
            runtime_mean, runtime_low, runtime_high = bootstrap_mean_ci(
                runtime,
                n_boot=bootstrap_samples,
                rng=rng,
            )
            runtime_median, runtime_med_low, runtime_med_high = bootstrap_median_ci(
                runtime,
                n_boot=bootstrap_samples,
                rng=rng,
            )

            summary.append(
                {
                    "budget": int(round(float(as_float(subset[0].get("budget"), np.nanmedian(q_actual)))))
                    if group_by_budget
                    else int(round(float(np.nanmedian(q_actual)))),
                    "algorithm": algorithm,
                    "algorithm_key": str(subset[0].get("algorithm_key", algorithm)),
                    "target_name": str(subset[0].get("target_name", "amplitude")),
                    "n_points": n_subset,
                    "n_runs": n_runs,
                    "total_repetitions": np.nan
                    if total_repetitions is None
                    else int(total_repetitions),
                    "success_rate": np.nan
                    if total_repetitions is None or int(total_repetitions) <= 0
                    else float(n_runs / int(total_repetitions)),
                    "estimate_mean": float(np.nanmean(estimates)),
                    "query_budget_actual_mean": float(np.nanmean(q_actual)),
                    "query_budget_actual_median": float(np.nanmedian(q_actual)),
                    "abs_error_mean": float(np.nanmean(abs_error)),
                    "abs_error_median": float(np.nanmedian(abs_error)),
                    "mae_ci_low": abs_low,
                    "mae_ci_high": abs_high,
                    "normalized_abs_error_mean": norm_mean,
                    "normalized_abs_error_ci_low": norm_low,
                    "normalized_abs_error_ci_high": norm_high,
                    "normalized_abs_error_median": norm_median,
                    "normalized_abs_error_median_ci_low": norm_median_low,
                    "normalized_abs_error_median_ci_high": norm_median_high,
                    "normalized_abs_error_std": float(np.nanstd(norm_abs, ddof=1))
                    if n_subset > 1
                    else 0.0,
                    "normalized_abs_error_se": standard_error(norm_abs),
                    "processed_abs_error_median": processed_abs_median,
                    "processed_abs_error_median_ci_low": processed_abs_low,
                    "processed_abs_error_median_ci_high": processed_abs_high,
                    "processed_abs_error_se": standard_error(processed_abs),
                    "processed_relative_error_median": processed_rel_median,
                    "processed_relative_error_median_ci_low": processed_rel_low,
                    "processed_relative_error_median_ci_high": processed_rel_high,
                    "processed_relative_error_se": standard_error(processed_rel),
                    "grover_power_max_median": float(np.nanmedian(k_vals)),
                    "amplification_factor_median": float(np.nanmedian(amp_factors)),
                    "runtime_wall_seconds_mean": runtime_mean,
                    "runtime_wall_seconds_ci_low": runtime_low,
                    "runtime_wall_seconds_ci_high": runtime_high,
                    "runtime_wall_seconds_median": runtime_median,
                    "runtime_wall_seconds_median_ci_low": runtime_med_low,
                    "runtime_wall_seconds_median_ci_high": runtime_med_high,
                    "runtime_wall_seconds_se": standard_error(runtime),
                }
            )
    return summary
