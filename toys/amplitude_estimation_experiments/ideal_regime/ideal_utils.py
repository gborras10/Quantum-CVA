from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from toys.amplitude_estimation_experiments.common_utils.plotting_utils import (
    log_query_bin_indices,
    select_log_spaced_indices,
    standard_error,
)


def save_csv(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def extract_trace(
    algorithm: str,
    result: Any,
    n_shots: int | None,
    effective_n_shots: Callable[[str, int | None], int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if algorithm == "bae":
        history = getattr(result, "history", {}) or {}
        queries = np.asarray(history.get("queries", []), dtype=float)
        estimations = np.asarray(history.get("estimations", []), dtype=float)
        controls = np.asarray(history.get("controls", getattr(result, "powers", [])) or [], dtype=float)
        usable = min(len(queries), len(estimations))
        if usable <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        if len(controls) >= usable:
            amplification_factors = 2.0 * controls[:usable] + 1.0
        else:
            increments = np.diff(np.concatenate([[0.0], queries[:usable]]))
            shots = max(1, effective_n_shots(algorithm, n_shots))
            amplification_factors = np.maximum(1.0, np.round(increments / shots))
        return queries[:usable], estimations[:usable], amplification_factors.astype(float)

    if algorithm == "elf_qae":
        elf_layers = np.asarray(getattr(result, "elf_layers", []) or [], dtype=float)
        estimate_intervals = getattr(result, "estimate_intervals", []) or []
        usable = min(len(elf_layers), max(0, len(estimate_intervals) - 1))
        if usable <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        interval_array = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
        if interval_array.ndim != 2 or interval_array.shape[1] != 2:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
        amplification_factors = 2.0 * elf_layers[:usable] + 1.0
        queries = np.cumsum(amplification_factors)
        estimations = np.mean(interval_array, axis=1)
        return queries.astype(float), estimations.astype(float), amplification_factors.astype(float)

    powers = np.asarray(getattr(result, "powers", []) or [], dtype=float)
    estimate_intervals = getattr(result, "estimate_intervals", []) or []
    usable = min(len(powers), max(0, len(estimate_intervals) - 1))
    if usable <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    interval_array = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
    if interval_array.ndim != 2 or interval_array.shape[1] != 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    estimations = np.mean(interval_array, axis=1)
    amplification_factors = 2.0 * powers[:usable] + 1.0
    queries = np.cumsum(effective_n_shots(algorithm, n_shots) * amplification_factors)
    return queries.astype(float), estimations.astype(float), amplification_factors.astype(float)


def trace_rows_from_result(
    result: Any,
    *,
    algorithm: str,
    algorithm_labels: Mapping[str, str],
    repetition: int,
    a_true: float,
    objective_ry_offset: float,
    n_shots: int | None,
    elapsed_wall_seconds: float,
    effective_n_shots: Callable[[str, int | None], int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries, estimates, amplification_factors = extract_trace(
        algorithm,
        result,
        n_shots,
        effective_n_shots,
    )
    final_queries_for_timing = float(
        getattr(result, "num_state_prep_calls", queries[-1] if len(queries) else np.nan)
    )
    rows: list[dict[str, Any]] = []
    for idx, (query, estimate, amplification_factor) in enumerate(
        zip(queries, estimates, amplification_factors)
    ):
        amp_int = int(round(float(amplification_factor)))
        control_k = max(0, (amp_int - 1) // 2)
        prefix_amplification = np.asarray(amplification_factors[: idx + 1], dtype=float)
        k_max_budget = max(
            0,
            int((int(round(float(np.nanmax(prefix_amplification)))) - 1) // 2),
        )
        if not np.isfinite(final_queries_for_timing) or final_queries_for_timing <= 0.0:
            runtime_wall_seconds = np.nan
        else:
            runtime_wall_seconds = float(elapsed_wall_seconds) * min(
                max(float(query) / final_queries_for_timing, 0.0),
                1.0,
            )
        abs_error = abs(float(estimate) - float(a_true))
        rows.append(
            {
                "run_kind": "ideal_simulation",
                "repetition": int(repetition),
                "algorithm": algorithm_labels.get(algorithm, algorithm),
                "algorithm_key": algorithm,
                "step_index": int(idx),
                "budget": int(round(float(query))),
                "query_budget": float(query),
                "query_budget_actual": float(query),
                "estimate": float(estimate),
                "abs_error": float(abs_error),
                "normalized_abs_error": float(abs_error / max(float(a_true), 1e-12)),
                "normalized_sq_error": float((float(estimate) / max(float(a_true), 1e-12) - 1.0) ** 2),
                "grover_power": int(control_k),
                "k_max_budget": int(k_max_budget),
                "amplification_factor": int(amp_int),
                "a_true": float(a_true),
                "objective_ry_offset": float(objective_ry_offset),
                "runtime_wall_seconds": float(runtime_wall_seconds),
                "time_to_budget_seconds": float(runtime_wall_seconds),
            }
        )

    final_estimate = float(getattr(result, "estimation", rows[-1]["estimate"] if rows else np.nan))
    ci = getattr(result, "confidence_interval", None)
    ci_low = np.nan
    ci_high = np.nan
    coverage = np.nan
    if ci is not None:
        ci_low = float(ci[0])
        ci_high = float(ci[1])
        coverage = float(ci_low <= float(a_true) <= ci_high)
    final_queries = float(getattr(result, "num_state_prep_calls", queries[-1] if len(queries) else np.nan))
    k_max = int(max((int(row["grover_power"]) for row in rows), default=0))
    final_abs_error = abs(final_estimate - float(a_true)) if np.isfinite(final_estimate) else np.nan
    final_row = {
        "run_kind": "ideal_simulation",
        "repetition": int(repetition),
        "algorithm": algorithm_labels.get(algorithm, algorithm),
        "algorithm_key": algorithm,
        "a_true": float(a_true),
        "objective_ry_offset": float(objective_ry_offset),
        "final_queries": final_queries,
        "final_estimate": final_estimate,
        "final_abs_error": float(final_abs_error),
        "final_normalized_abs_error": float(final_abs_error / max(float(a_true), 1e-12))
        if np.isfinite(final_abs_error)
        else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "coverage": coverage,
        "k_max": k_max,
        "amplification_factor_max": int(2 * k_max + 1),
        "runtime_wall_seconds": float(elapsed_wall_seconds),
    }
    return rows, final_row


def rows_at_budgets(
    trace_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[float],
) -> list[dict[str, Any]]:
    if not trace_rows:
        return []
    ordered = sorted(trace_rows, key=lambda row: float(row["query_budget"]))
    if not ordered:
        return []
    rows: list[dict[str, Any]] = []
    final_query = float(ordered[-1]["query_budget"])
    for budget in budgets:
        budget_float = float(budget)
        if final_query < budget_float:
            continue
        candidates = [row for row in ordered if float(row["query_budget"]) <= budget_float]
        if not candidates:
            continue
        chosen = candidates[-1]
        rows.append(
            {
                "run_kind": "ideal_simulation",
                "repetition": int(chosen["repetition"]),
                "algorithm": chosen["algorithm"],
                "algorithm_key": chosen["algorithm_key"],
                "budget": int(round(budget_float)),
                "query_budget_actual": float(chosen["query_budget"]),
                "estimate": float(chosen["estimate"]),
                "abs_error": float(chosen["abs_error"]),
                "normalized_abs_error": float(chosen["normalized_abs_error"]),
                "normalized_sq_error": float(chosen["normalized_sq_error"]),
                "grover_power": int(chosen["grover_power"]),
                "amplification_factor": int(chosen["amplification_factor"]),
                "a_true": float(chosen["a_true"]),
                "objective_ry_offset": float(chosen["objective_ry_offset"]),
                "runtime_wall_seconds": float(chosen.get("runtime_wall_seconds", np.nan)),
            }
        )
    return rows


def aggregate_budget_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    total_repetitions: int,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 12345,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not rows:
        return summary
    algorithms = sorted({str(row["algorithm"]) for row in rows})
    rng = np.random.default_rng(int(bootstrap_seed))

    for algorithm in algorithms:
        alg_rows = [row for row in rows if str(row["algorithm"]) == algorithm]
        query_budget = np.asarray([_query_budget(row) for row in alg_rows], dtype=float)
        normalized_abs_error = np.asarray(
            [_as_float(row.get("normalized_abs_error")) for row in alg_rows],
            dtype=float,
        )
        valid = (
            np.isfinite(query_budget)
            & np.isfinite(normalized_abs_error)
            & (query_budget > 0.0)
            & (normalized_abs_error > 0.0)
        )
        if not np.any(valid):
            continue

        bin_indices = log_query_bin_indices(
            query_budget,
            normalized_abs_error,
            max_bins=max_bins,
            min_points_per_bin=min_points_per_bin,
        )

        for indices in bin_indices:
            if indices.size == 0:
                continue
            subset = [alg_rows[int(idx)] for idx in indices]
            n_subset = len(subset)
            if n_subset <= 0:
                continue
            repetitions = {
                int(_as_float(row.get("repetition")))
                for row in subset
                if np.isfinite(_as_float(row.get("repetition")))
            }
            n_runs = len(repetitions) if repetitions else n_subset
            query_budget_actual = np.asarray([_query_budget(row) for row in subset], dtype=float)
            if not subset:
                continue
            estimates = np.asarray([_as_float(row.get("estimate")) for row in subset], dtype=float)
            abs_error = np.asarray([_as_float(row.get("abs_error")) for row in subset], dtype=float)
            normalized_abs_error = np.asarray(
                [_as_float(row.get("normalized_abs_error")) for row in subset],
                dtype=float,
            )
            normalized_sq_error = np.asarray(
                [_normalized_sq_error(row) for row in subset],
                dtype=float,
            )
            a_true = np.asarray([_as_float(row.get("a_true")) for row in subset], dtype=float)
            k_vals = np.asarray(
                [
                    _as_float(row.get("k_max_budget"), _as_float(row.get("grover_power")))
                    for row in subset
                ],
                dtype=float,
            )
            amplification_factors = np.asarray(
                [_as_float(row.get("amplification_factor")) for row in subset],
                dtype=float,
            )
            runtime_wall_seconds = np.asarray(
                [
                    _as_float(
                        row.get(
                            "time_to_budget_seconds",
                            row.get("runtime_wall_seconds", np.nan),
                        )
                    )
                    for row in subset
                ],
                dtype=float,
            )
            normalized_abs_error_std = (
                float(np.nanstd(normalized_abs_error, ddof=1)) if n_subset > 1 else 0.0
            )
            (
                normalized_abs_error_median,
                normalized_abs_error_median_ci_low,
                normalized_abs_error_median_ci_high,
            ) = _bootstrap_median_ci(
                normalized_abs_error,
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            )
            normalized_rmse = float(np.sqrt(np.nanmean(normalized_sq_error)))
            normalized_standard_deviation = _normalized_standard_deviation(estimates, a_true)
            runtime_std = float(np.nanstd(runtime_wall_seconds, ddof=1)) if n_subset > 1 else 0.0
            query_budget_actual_mean = float(np.nanmean(query_budget_actual))
            summary.append(
                {
                    "run_kind": "ideal_simulation",
                    "budget": int(round(query_budget_actual_mean)),
                    "algorithm": algorithm,
                    "algorithm_key": str(subset[0].get("algorithm_key", algorithm)),
                    "n_points": int(n_subset),
                    "n_runs": int(n_runs),
                    "total_repetitions": int(total_repetitions),
                    "success_rate": float(n_runs / max(int(total_repetitions), 1)),
                    "estimate_mean": float(np.nanmean(estimates)),
                    "estimate_median": float(np.nanmedian(estimates)),
                    "query_budget_actual": query_budget_actual_mean,
                    "query_budget_actual_mean": query_budget_actual_mean,
                    "query_budget_actual_median": float(np.nanmedian(query_budget_actual)),
                    "query_budget_actual_q25": float(np.nanquantile(query_budget_actual, 0.25)),
                    "query_budget_actual_q75": float(np.nanquantile(query_budget_actual, 0.75)),
                    "abs_error_mean": float(np.nanmean(abs_error)),
                    "abs_error_median": float(np.nanmedian(abs_error)),
                    "normalized_abs_error_mean": float(np.nanmean(normalized_abs_error)),
                    "normalized_abs_error_median": normalized_abs_error_median,
                    "normalized_abs_error_median_ci_low": normalized_abs_error_median_ci_low,
                    "normalized_abs_error_median_ci_high": normalized_abs_error_median_ci_high,
                    "normalized_abs_error_std": normalized_abs_error_std,
                    "normalized_abs_error_se": standard_error(normalized_abs_error),
                    "normalized_sq_error_mean": float(np.nanmean(normalized_sq_error)),
                    "normalized_rmse": normalized_rmse,
                    "normalized_rmse_median": normalized_rmse,
                    "normalized_standard_deviation": normalized_standard_deviation,
                    "normalized_abs_error_q25": float(np.nanquantile(normalized_abs_error, 0.25)),
                    "normalized_abs_error_q75": float(np.nanquantile(normalized_abs_error, 0.75)),
                    "grover_power_max_median": float(np.nanmedian(k_vals)),
                    "amplification_factor_median": float(np.nanmedian(amplification_factors)),
                    "runtime_wall_seconds_mean": float(np.nanmean(runtime_wall_seconds)),
                    "runtime_wall_seconds_median": float(np.nanmedian(runtime_wall_seconds)),
                    "runtime_wall_seconds_std": runtime_std,
                    "runtime_wall_seconds_se": float(runtime_std / math.sqrt(n_subset))
                    if n_subset > 0
                    else np.nan,
                }
            )
    return summary


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalized_sq_error(row: Mapping[str, Any]) -> float:
    value = _as_float(row.get("normalized_sq_error"))
    if np.isfinite(value):
        return value

    estimate = _as_float(row.get("estimate"))
    a_true = _as_float(row.get("a_true"))
    if not np.isfinite(estimate) or not np.isfinite(a_true) or a_true == 0.0:
        return np.nan
    return float(((estimate - a_true) / a_true) ** 2)


def _normalized_standard_deviation(estimates: np.ndarray, a_true: np.ndarray) -> float:
    estimates = np.asarray(estimates, dtype=float)
    a_true = np.asarray(a_true, dtype=float)
    valid = np.isfinite(estimates) & np.isfinite(a_true) & (a_true != 0.0)
    if not np.any(valid):
        return np.nan

    estimates = estimates[valid]
    a_true = a_true[valid]
    normalized_centered_sq: list[np.ndarray] = []
    for amplitude in np.unique(a_true):
        group = estimates[a_true == amplitude]
        if group.size == 0:
            continue
        centered = (group - float(np.nanmean(group))) / float(amplitude)
        normalized_centered_sq.append(centered**2)
    if not normalized_centered_sq:
        return np.nan
    return float(np.sqrt(np.nanmean(np.concatenate(normalized_centered_sq))))


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    confidence_level: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, np.nan

    center = float(np.nanmedian(finite))
    if finite.size == 1 or int(bootstrap_samples) <= 0:
        return center, center, center

    sample_indices = rng.integers(
        0,
        finite.size,
        size=(int(bootstrap_samples), finite.size),
    )
    bootstrap_medians = np.nanmedian(finite[sample_indices], axis=1)
    alpha = 1.0 - float(confidence_level)
    low, high = np.nanquantile(bootstrap_medians, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(low), float(high)


def _query_budget(row: Mapping[str, Any]) -> float:
    for key in (
        "query_budget",
        "query_budget_actual",
        "query_budget_actual_mean",
        "budget",
        "final_queries",
    ):
        value = _as_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def plot_budget_summary(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    algorithms: Sequence[str],
    algorithm_labels: Mapping[str, str],
    algorithm_styles: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    title: str,
    max_points_per_algorithm: int = 14,
) -> None:
    if not summary_rows:
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    guide_points: list[tuple[float, float]] = []

    for algorithm in algorithms:
        label = algorithm_labels.get(algorithm, algorithm)
        group = [
            row
            for row in summary_rows
            if str(row.get("algorithm_key", "")) == algorithm or str(row.get("algorithm", "")) == label
        ]
        if not group:
            continue
        group = sorted(group, key=lambda row: _query_budget(row))
        x_values = np.asarray([_query_budget(row) for row in group], dtype=float)
        y_values = np.asarray(
            [_as_float(row.get("normalized_abs_error_median")) for row in group],
            dtype=float,
        )
        ci_low = np.asarray(
            [_as_float(row.get("normalized_abs_error_median_ci_low")) for row in group],
            dtype=float,
        )
        ci_high = np.asarray(
            [_as_float(row.get("normalized_abs_error_median_ci_high")) for row in group],
            dtype=float,
        )
        style = algorithm_styles.get(algorithm, {})

        valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
        if not np.any(valid):
            continue
        selected = select_log_spaced_indices(
            x_values,
            valid,
            max(2, int(max_points_per_algorithm)),
        )

        valid_points = [(float(x), float(y)) for x, y in zip(x_values[selected], y_values[selected])]
        guide_points.extend(valid_points)
        n_runs = np.asarray(
            [int(_as_float(group[idx].get("n_runs"), 0.0)) for idx in selected],
            dtype=int,
        )
        if n_runs.size and int(np.nanmin(n_runs)) != int(np.nanmax(n_runs)):
            plot_label = f"{label} (n={int(np.nanmin(n_runs))}-{int(np.nanmax(n_runs))})"
        elif n_runs.size:
            plot_label = f"{label} (n={int(n_runs[0])})"
        else:
            plot_label = label

        yerr_selected = _bootstrap_ci_errorbar(
            y_values[selected],
            ci_low[selected],
            ci_high[selected],
        )
        ax.errorbar(
            x_values[selected],
            y_values[selected],
            yerr=yerr_selected,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linewidth=2.0,
            markersize=5.6,
            elinewidth=1.0,
            capsize=2.8,
            label=plot_label,
            zorder=3,
        )
    _add_power_fit_query_scaling_guides(ax, guide_points)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Actual query cost $N_q$")
    ax.set_ylabel("Median normalized absolute error")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    path = Path(output_path)
    _save_figure_png_and_pdf(fig, path)
    plt.close(fig)


def plot_final_runtime_scatter_from_budget_rows(
    budget_rows: Sequence[Mapping[str, Any]],
    *,
    algorithms: Sequence[str],
    algorithm_labels: Mapping[str, str],
    algorithm_styles: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    title: str = "",
    summary_path: str | Path | None = None,
    max_points_per_algorithm: int | None = None,
    point_sample_seed: int = 12345,
    x_kind: str = "runtime",
) -> list[dict[str, Any]]:
    if x_kind not in {"runtime", "queries"}:
        raise ValueError("x_kind must be 'runtime' or 'queries'")

    final_rows = _final_rows_from_budget_rows(budget_rows)
    if not final_rows:
        return []

    summary_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    legend_handles: list[Line2D] = []
    all_x: list[float] = []
    all_y: list[float] = []
    rng = np.random.default_rng(int(point_sample_seed))

    for algorithm in algorithms:
        label = algorithm_labels.get(algorithm, algorithm)
        group = [
            row
            for row in final_rows
            if str(row.get("algorithm_key", "")) == algorithm
            or str(row.get("algorithm", "")) == label
        ]
        x_values = np.asarray(
            [
                _runtime_seconds(row) if x_kind == "runtime" else _query_budget(row)
                for row in group
            ],
            dtype=float,
        )
        y_values = np.asarray(
            [_as_float(row.get("normalized_abs_error")) for row in group],
            dtype=float,
        )
        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & (x_values > 0.0)
            & (y_values > 0.0)
        )
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.size == 0:
            continue

        style = algorithm_styles.get(algorithm, {})
        color = style.get("color")
        marker = style.get("marker", "o")
        plot_x_values = x_values
        plot_y_values = y_values
        if max_points_per_algorithm is not None and x_values.size > int(max_points_per_algorithm):
            selected = rng.choice(
                x_values.size,
                size=max(1, int(max_points_per_algorithm)),
                replace=False,
            )
            plot_x_values = x_values[selected]
            plot_y_values = y_values[selected]
        all_x.extend(plot_x_values.tolist())
        all_y.extend(plot_y_values.tolist())

        ax.scatter(
            plot_x_values,
            plot_y_values,
            s=28,
            marker=marker,
            color=color,
            alpha=0.42,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        _plot_log_gaussian_contours(ax, plot_x_values, plot_y_values, color=str(color))

        median_runtime = float(np.nanmedian(x_values))
        median_error = float(np.nanmedian(y_values))
        ax.scatter(
            [median_runtime],
            [median_error],
            s=145,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=5,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=8,
                label=f"{label} (n={plot_x_values.size}/{x_values.size})"
                if plot_x_values.size != x_values.size
                else f"{label} (n={x_values.size})",
            )
        )
        summary_rows.append(
            {
                "algorithm": label,
                "algorithm_key": algorithm,
                "n": int(x_values.size),
                "median_final_normalized_abs_error": median_error,
                "mean_final_normalized_abs_error": float(np.nanmean(y_values)),
            }
        )
        if x_kind == "runtime":
            summary_rows[-1]["median_runtime_seconds"] = median_runtime
            summary_rows[-1]["mean_runtime_seconds"] = float(np.nanmean(x_values))
        else:
            summary_rows[-1]["median_final_queries"] = median_runtime
            summary_rows[-1]["mean_final_queries"] = float(np.nanmean(x_values))

    if summary_path is not None:
        save_csv(summary_rows, summary_path)

    median_handle = Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        markerfacecolor="white",
        linestyle="None",
        markersize=8,
        markeredgewidth=1.3,
        label="median",
    )
    contour_handle = Line2D(
        [0],
        [0],
        color="0.25",
        linewidth=1.7,
        label="Gaussian contours",
    )
    legend_handles.extend([median_handle, contour_handle])

    x_limits = _log_limits(np.asarray(all_x, dtype=float), pad_fraction=0.07)
    y_limits = _log_limits(np.asarray(all_y, dtype=float), pad_fraction=0.12)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime [s]" if x_kind == "runtime" else r"Final query count $N_q$")
    ax.set_ylabel("Final normalized absolute error")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_png_and_pdf(fig, path)
    plt.close(fig)
    return summary_rows


def _save_figure_png_and_pdf(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    try:
        fig.savefig(output_path.with_suffix(".pdf"))
    except PermissionError as exc:
        print(f"Skipped locked PDF output {output_path.with_suffix('.pdf')}: {exc}")


def _final_rows_from_budget_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_run: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        algorithm_key = str(row.get("algorithm_key", row.get("algorithm", "")))
        repetition_value = _as_float(row.get("repetition"))
        if not algorithm_key or not np.isfinite(repetition_value):
            continue
        key = (algorithm_key, int(repetition_value))
        current = by_run.get(key)
        if current is None or _final_row_order_key(row) > _final_row_order_key(current):
            by_run[key] = row
    return list(by_run.values())


def _final_row_order_key(row: Mapping[str, Any]) -> tuple[float, float]:
    step_index = _as_float(row.get("step_index"), -1.0)
    query_budget = _query_budget(row)
    return (
        step_index if np.isfinite(step_index) else -1.0,
        query_budget if np.isfinite(query_budget) else -1.0,
    )


def _runtime_seconds(row: Mapping[str, Any]) -> float:
    for key in ("runtime_wall_seconds", "time_to_budget_seconds", "runtime_seconds"):
        value = _as_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def _log_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None

    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or high <= 0.0:
        return None
    if np.isclose(low, high):
        return low / 1.6, high * 1.6

    log_low = np.log10(low)
    log_high = np.log10(high)
    span = log_high - log_low
    return 10.0 ** (log_low - pad_fraction * span), 10.0 ** (log_high + pad_fraction * span)


def _plot_log_gaussian_contours(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    color: str,
    levels: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> None:
    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 3:
        return

    log_points = np.vstack([np.log10(x_values), np.log10(y_values)])
    covariance = np.cov(log_points)
    if not np.all(np.isfinite(covariance)):
        return

    covariance = covariance + np.eye(2) * 1e-12
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if np.any(eigvals <= 0.0) or not np.all(np.isfinite(eigvals)):
        return

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    center = np.mean(log_points, axis=1)
    angles = np.linspace(0.0, 2.0 * np.pi, 360)
    unit_circle = np.vstack([np.cos(angles), np.sin(angles)])

    for level in levels:
        ellipse = center[:, None] + eigvecs @ (
            np.sqrt(eigvals)[:, None] * float(level) * unit_circle
        )
        ax.plot(
            10.0 ** ellipse[0],
            10.0 ** ellipse[1],
            color=color,
            linewidth=1.25,
            alpha=0.92,
            zorder=2,
        )


def _bootstrap_ci_errorbar(
    center_values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
) -> np.ndarray:
    center_values = np.asarray(center_values, dtype=float)
    ci_low = np.asarray(ci_low, dtype=float)
    ci_high = np.asarray(ci_high, dtype=float)

    lower = np.where(np.isfinite(ci_low), center_values - ci_low, 0.0)
    upper = np.where(np.isfinite(ci_high), ci_high - center_values, 0.0)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    lower = np.minimum(lower, np.maximum(0.0, 0.95 * center_values))
    return np.vstack([lower, upper])


def _add_power_fit_query_scaling_guides(
    ax: plt.Axes,
    guide_points: list[tuple[float, float]] | tuple[np.ndarray, np.ndarray],
    *,
    include_linear: bool = True,
    include_sqrt: bool = True,
    alpha: float = 0.82,
) -> list:
    if isinstance(guide_points, tuple):
        x_values = np.asarray(guide_points[0], dtype=float)
        y_values = np.asarray(guide_points[1], dtype=float)
    else:
        x_values = np.asarray([x for x, _ in guide_points], dtype=float)
        y_values = np.asarray([y for _, y in guide_points], dtype=float)

    valid = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size == 0:
        return []

    order = np.argsort(x_values)
    x_values = x_values[order]
    y_values = y_values[order]
    x0 = float(x_values[0])
    x_max = float(x_values[-1])
    if not np.isfinite(x0) or not np.isfinite(x_max) or x0 <= 0.0 or x_max <= x0:
        return []

    if x_values.size >= 2 and float(np.nanmax(x_values)) > float(np.nanmin(x_values)):
        fit_slope, fit_log_intercept = np.polyfit(np.log(x_values), np.log(y_values), deg=1)
        y0 = float(np.exp(fit_log_intercept) * x0**fit_slope)
    else:
        y0 = float(y_values[0])
    if not np.isfinite(y0) or y0 <= 0.0:
        return []

    guide_x = np.geomspace(x0, x_max, num=200)

    handles = []
    if include_linear:
        (line,) = ax.loglog(
            guide_x,
            y0 * (x0 / guide_x),
            color="black",
            linestyle="--",
            linewidth=1.15,
            alpha=alpha,
            label=r"$O(1/N)$",
        )
        handles.append(line)
    if include_sqrt:
        (line,) = ax.loglog(
            guide_x,
            y0 * np.sqrt(x0 / guide_x),
            color="black",
            linestyle=":",
            linewidth=1.35,
            alpha=alpha,
            label=r"$O(1/\sqrt{N})$",
        )
        handles.append(line)
    return handles
