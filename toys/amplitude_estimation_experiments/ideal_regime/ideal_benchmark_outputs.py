from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


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
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not rows:
        return summary
    algorithms = sorted({str(row["algorithm"]) for row in rows})

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

        valid_indices = np.flatnonzero(valid)
        q_valid = query_budget[valid]
        q_min = float(np.nanmin(q_valid))
        q_max = float(np.nanmax(q_valid))
        if q_valid.size <= max_bins or not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
            bin_indices = [np.asarray([idx], dtype=int) for idx in valid_indices[np.argsort(q_valid)]]
        else:
            edges = np.geomspace(q_min, q_max, num=int(max_bins) + 1)
            local_bins = np.digitize(q_valid, edges, right=False) - 1
            local_bins = np.clip(local_bins, 0, int(max_bins) - 1)
            bin_indices = [
                valid_indices[np.flatnonzero(local_bins == bin_idx)]
                for bin_idx in range(int(max_bins))
            ]

        for indices in bin_indices:
            if indices.size == 0:
                continue
            if indices.size < int(min_points_per_bin) and q_valid.size > max_bins:
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
                [_as_float(row.get("normalized_sq_error")) for row in subset],
                dtype=float,
            )
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
            runtime_std = float(np.nanstd(runtime_wall_seconds, ddof=1)) if n_subset > 1 else 0.0
            summary.append(
                {
                    "run_kind": "ideal_simulation",
                    "budget": int(round(float(np.nanmedian(query_budget_actual)))),
                    "algorithm": algorithm,
                    "algorithm_key": str(subset[0].get("algorithm_key", algorithm)),
                    "n_points": int(n_subset),
                    "n_runs": int(n_runs),
                    "total_repetitions": int(total_repetitions),
                    "success_rate": float(n_runs / max(int(total_repetitions), 1)),
                    "estimate_mean": float(np.nanmean(estimates)),
                    "estimate_median": float(np.nanmedian(estimates)),
                    "query_budget_actual_mean": float(np.nanmean(query_budget_actual)),
                    "query_budget_actual_median": float(np.nanmedian(query_budget_actual)),
                    "query_budget_actual_q25": float(np.nanquantile(query_budget_actual, 0.25)),
                    "query_budget_actual_q75": float(np.nanquantile(query_budget_actual, 0.75)),
                    "abs_error_mean": float(np.nanmean(abs_error)),
                    "abs_error_median": float(np.nanmedian(abs_error)),
                    "normalized_abs_error_mean": float(np.nanmean(normalized_abs_error)),
                    "normalized_abs_error_median": float(np.nanmedian(normalized_abs_error)),
                    "normalized_abs_error_std": normalized_abs_error_std,
                    "normalized_abs_error_se": float(normalized_abs_error_std / math.sqrt(n_subset))
                    if n_subset > 0
                    else np.nan,
                    "normalized_sq_error_mean": float(np.nanmean(normalized_sq_error)),
                    "normalized_rmse": float(np.sqrt(np.nanmean(normalized_sq_error))),
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


def _query_budget(row: Mapping[str, Any]) -> float:
    for key in ("query_budget", "query_budget_actual", "budget", "final_queries"):
        value = _as_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def _select_plot_indices(
    x_values: np.ndarray,
    valid: np.ndarray,
    max_points: int,
) -> np.ndarray:
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) <= max_points:
        return valid_indices

    valid_x = x_values[valid_indices]
    if np.all(np.isfinite(valid_x)) and np.nanmin(valid_x) > 0.0 and np.nanmax(valid_x) > np.nanmin(valid_x):
        targets = np.geomspace(float(np.nanmin(valid_x)), float(np.nanmax(valid_x)), num=max_points)
        selected: set[int] = {int(valid_indices[0]), int(valid_indices[-1])}
        for target in targets[1:-1]:
            pos = int(np.searchsorted(valid_x, target, side="left"))
            candidates = []
            if 0 <= pos < len(valid_indices):
                candidates.append(pos)
            if 0 <= pos - 1 < len(valid_indices):
                candidates.append(pos - 1)
            if not candidates:
                continue
            nearest = min(candidates, key=lambda idx: abs(float(valid_x[idx]) - float(target)))
            selected.add(int(valid_indices[nearest]))
        return np.asarray(sorted(selected), dtype=int)

    positions = np.linspace(0, len(valid_indices) - 1, num=max_points)
    return valid_indices[np.unique(np.rint(positions).astype(int))]


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
        y_values = np.asarray([_as_float(row.get("normalized_abs_error_median")) for row in group], dtype=float)
        yerr = np.asarray([_as_float(row.get("normalized_abs_error_se"), 0.0) for row in group], dtype=float)
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)
        yerr = np.minimum(yerr, np.maximum(0.0, 0.95 * y_values))
        style = algorithm_styles.get(algorithm, {})
        color = style.get("color")
        marker = style.get("marker", "o")

        valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
        if not np.any(valid):
            continue
        selected = _select_plot_indices(
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

        ax.errorbar(
            x_values[selected],
            y_values[selected],
            yerr=yerr[selected],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.6,
            elinewidth=1.0,
            capsize=2.8,
            label=plot_label,
        )
    if guide_points:
        x_values = np.asarray([x for x, _ in guide_points], dtype=float)
        y_values = np.asarray([y for _, y in guide_points], dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.size:
            x0 = float(np.nanmin(x_values))
            y0_values = y_values[np.isclose(x_values, x0)]
            y0 = float(np.nanmedian(y0_values)) if y0_values.size else float(np.nanmedian(y_values))
            budgets = np.geomspace(x0, float(np.nanmax(x_values)), num=200)
            ax.loglog(
                budgets,
                y0 * (x0 / budgets),
                color="black",
                linestyle="--",
                linewidth=1.15,
                alpha=0.82,
                label=r"$O(1/N)$",
            )
            ax.loglog(
                budgets,
                y0 * np.sqrt(x0 / budgets),
                color="black",
                linestyle=":",
                linewidth=1.35,
                alpha=0.82,
                label=r"$O(1/\sqrt{N})$",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Actual query cost $N_q$")
    ax.set_ylabel("Median normalized absolute error")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    path = Path(output_path)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
