from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle
from quantum_cva.amplitude_estimation.experiments.statistics import as_float


def default_effective_n_shots(
    algorithm: str,
    configured_n_shots: int | None,
) -> int:
    if configured_n_shots is not None:
        return int(configured_n_shots)
    return 20 if str(algorithm) == "bae" else 10


def _extract_bae_amplification_factors(
    result: Any,
    history: Mapping[str, Any],
    queries: np.ndarray,
    n_shots: int | None,
    effective_n_shots: Callable[[str, int | None], int],
) -> np.ndarray:
    controls = np.asarray(
        history.get("controls", getattr(result, "powers", [])) or [],
        dtype=float,
    )
    if controls.size:
        return (2.0 * controls + 1.0).astype(float)

    circuit_depths = np.asarray(
        history.get("circuit_depths", getattr(result, "circuit_depths", [])) or [],
        dtype=float,
    )
    if circuit_depths.size:
        return (2.0 * circuit_depths + 1.0).astype(float)

    if queries.size == 0:
        return np.asarray([], dtype=float)
    increments = np.diff(np.concatenate([[0.0], queries]))
    shots = max(1, int(effective_n_shots("bae", n_shots)))
    return np.maximum(1.0, np.round(increments / float(shots))).astype(float)


def extract_trace(
    algorithm: str,
    result: Any,
    n_shots: int | None,
    effective_n_shots: Callable[[str, int | None], int] = default_effective_n_shots,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract query budgets, estimates, and amplification factors."""
    algorithm = str(algorithm)
    if algorithm == "bae":
        history = getattr(result, "history", {}) or {}
        queries = np.asarray(history.get("queries", []), dtype=float)
        estimations = np.asarray(history.get("estimations", []), dtype=float)
        amp_factors = _extract_bae_amplification_factors(
            result,
            history,
            queries,
            n_shots,
            effective_n_shots,
        )
        usable = min(len(queries), len(estimations), len(amp_factors))
        if usable <= 0:
            return np.asarray([]), np.asarray([]), np.asarray([])
        return (
            queries[:usable].astype(float),
            estimations[:usable].astype(float),
            amp_factors[:usable].astype(float),
        )

    if algorithm == "elf_qae":
        elf_layers = np.asarray(getattr(result, "elf_layers", []) or [], dtype=float)
        estimate_intervals = getattr(result, "estimate_intervals", []) or []
        usable = min(len(elf_layers), max(0, len(estimate_intervals) - 1))
        if usable <= 0:
            return np.asarray([]), np.asarray([]), np.asarray([])
        intervals = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            return np.asarray([]), np.asarray([]), np.asarray([])
        amp_factors = 2.0 * elf_layers[:usable] + 1.0
        queries = np.cumsum(amp_factors)
        estimations = np.mean(intervals, axis=1)
        return queries.astype(float), estimations.astype(float), amp_factors.astype(float)

    powers = np.asarray(getattr(result, "powers", []) or [], dtype=float)
    estimate_intervals = getattr(result, "estimate_intervals", []) or []
    usable = min(len(powers), max(0, len(estimate_intervals) - 1))
    if usable <= 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    intervals = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        return np.asarray([]), np.asarray([]), np.asarray([])
    amp_factors = 2.0 * powers[:usable] + 1.0
    queries = np.cumsum(effective_n_shots(algorithm, n_shots) * amp_factors)
    estimations = np.mean(intervals, axis=1)
    return queries.astype(float), estimations.astype(float), amp_factors.astype(float)


def _processed_fields(bundle: AEProblemBundle, estimate: float) -> dict[str, float | str]:
    processed = bundle.process(float(estimate))
    processed_abs_error = abs(float(processed) - float(bundle.processed_true_value))
    denom = max(abs(float(bundle.processed_true_value)), 1e-12)
    return {
        "target_name": bundle.target_name,
        "processed_true_value": float(bundle.processed_true_value),
        "processed_estimate": float(processed),
        "processed_abs_error": float(processed_abs_error),
        "processed_relative_error": float(processed_abs_error / denom),
    }


def trace_rows_from_result(
    result: Any,
    *,
    bundle: AEProblemBundle,
    algorithm: str,
    algorithm_labels: Mapping[str, str],
    repetition: int,
    n_shots: int | None,
    elapsed_wall_seconds: float | None,
    run_kind: str,
    effective_n_shots: Callable[[str, int | None], int] = default_effective_n_shots,
    extra: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries, estimates, amp_factors = extract_trace(
        algorithm,
        result,
        n_shots,
        effective_n_shots,
    )
    final_queries_for_timing = float(
        getattr(
            result,
            "num_state_prep_calls",
            queries[-1] if len(queries) else np.nan,
        )
    )
    rows: list[dict[str, Any]] = []
    for idx, (query, estimate, amplification_factor) in enumerate(
        zip(queries, estimates, amp_factors)
    ):
        amp_int = int(round(float(amplification_factor)))
        k = max(0, (amp_int - 1) // 2)
        prefix = np.asarray(amp_factors[: idx + 1], dtype=float)
        k_max_budget = max(0, int((int(round(float(np.nanmax(prefix)))) - 1) // 2))
        if (
            elapsed_wall_seconds is None
            or not np.isfinite(final_queries_for_timing)
            or final_queries_for_timing <= 0.0
        ):
            runtime_wall_seconds = np.nan
        else:
            runtime_wall_seconds = float(elapsed_wall_seconds) * min(
                max(float(query) / final_queries_for_timing, 0.0),
                1.0,
            )
        abs_error = abs(float(estimate) - float(bundle.true_amplitude))
        row = {
            "run_kind": run_kind,
            "repetition": int(repetition),
            "algorithm": algorithm_labels.get(algorithm, algorithm),
            "algorithm_key": algorithm,
            "step_index": int(idx),
            "budget": int(round(float(query))),
            "query_budget": float(query),
            "query_budget_actual": float(query),
            "estimate": float(estimate),
            "abs_error": float(abs_error),
            "normalized_abs_error": float(abs_error / max(float(bundle.true_amplitude), 1e-12)),
            "normalized_sq_error": float(
                ((float(estimate) - float(bundle.true_amplitude)) / max(float(bundle.true_amplitude), 1e-12))
                ** 2
            ),
            "grover_power": int(k),
            "k_max_budget": int(k_max_budget),
            "amplification_factor": int(amp_int),
            "a_true": float(bundle.true_amplitude),
            "runtime_wall_seconds": float(runtime_wall_seconds),
            "time_to_budget_seconds": float(runtime_wall_seconds),
            **_processed_fields(bundle, float(estimate)),
        }
        if extra:
            row.update(dict(extra))
        rows.append(row)

    final_estimate = float(
        getattr(result, "estimation", rows[-1]["estimate"] if rows else np.nan)
    )
    ci = getattr(result, "confidence_interval", None)
    ci_low = np.nan
    ci_high = np.nan
    processed_ci_low = np.nan
    processed_ci_high = np.nan
    coverage = np.nan
    if ci is not None:
        ci_low = float(ci[0])
        ci_high = float(ci[1])
        processed_ci_low = bundle.process(ci_low)
        processed_ci_high = bundle.process(ci_high)
        coverage = float(ci_low <= float(bundle.true_amplitude) <= ci_high)

    final_queries = float(
        getattr(result, "num_state_prep_calls", queries[-1] if len(queries) else np.nan)
    )
    k_max = int(max((int(row["grover_power"]) for row in rows), default=0))
    final_abs_error = (
        abs(final_estimate - float(bundle.true_amplitude))
        if np.isfinite(final_estimate)
        else np.nan
    )
    final_row = {
        "run_kind": run_kind,
        "repetition": int(repetition),
        "algorithm": algorithm_labels.get(algorithm, algorithm),
        "algorithm_key": algorithm,
        "a_true": float(bundle.true_amplitude),
        "final_queries": final_queries,
        "final_estimate": final_estimate,
        "final_abs_error": float(final_abs_error),
        "final_normalized_abs_error": float(
            final_abs_error / max(float(bundle.true_amplitude), 1e-12)
        )
        if np.isfinite(final_abs_error)
        else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "processed_ci_low": processed_ci_low,
        "processed_ci_high": processed_ci_high,
        "coverage": coverage,
        "k_max": k_max,
        "amplification_factor_max": int(2 * k_max + 1),
        "runtime_wall_seconds": float(elapsed_wall_seconds)
        if elapsed_wall_seconds is not None and np.isfinite(elapsed_wall_seconds)
        else np.nan,
        **_processed_fields(bundle, final_estimate),
    }
    if extra:
        final_row.update(dict(extra))
    return rows, final_row


def rows_at_budgets(
    trace_rows: Sequence[Mapping[str, Any]],
    budgets: Sequence[int | float],
    *,
    run_kind: str | None = None,
) -> list[dict[str, Any]]:
    if not trace_rows:
        return []
    ordered = sorted(trace_rows, key=lambda row: float(row["query_budget"]))
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        budget_float = float(budget)
        if float(ordered[-1]["query_budget"]) < budget_float:
            continue
        candidates = [
            row for row in ordered if float(row["query_budget"]) <= budget_float
        ]
        if not candidates:
            continue
        chosen = candidates[-1]
        out = {
            "run_kind": run_kind or str(chosen.get("run_kind", "")),
            "repetition": int(as_float(chosen.get("repetition"))),
            "algorithm": chosen["algorithm"],
            "algorithm_key": chosen["algorithm_key"],
            "budget": int(round(budget_float)),
            "query_budget_actual": float(chosen["query_budget"]),
            "estimate": float(chosen["estimate"]),
            "abs_error": float(chosen["abs_error"]),
            "normalized_abs_error": float(chosen["normalized_abs_error"]),
            "normalized_sq_error": float(chosen.get("normalized_sq_error", np.nan)),
            "grover_power": int(chosen["grover_power"]),
            "k_max_budget": int(chosen.get("k_max_budget", chosen["grover_power"])),
            "amplification_factor": int(chosen["amplification_factor"]),
            "a_true": float(chosen["a_true"]),
            "runtime_wall_seconds": float(
                chosen.get("time_to_budget_seconds", chosen.get("runtime_wall_seconds", np.nan))
            ),
            "time_to_budget_seconds": float(
                chosen.get("time_to_budget_seconds", chosen.get("runtime_wall_seconds", np.nan))
            ),
            "target_name": str(chosen.get("target_name", "amplitude")),
            "processed_true_value": float(chosen.get("processed_true_value", np.nan)),
            "processed_estimate": float(chosen.get("processed_estimate", np.nan)),
            "processed_abs_error": float(chosen.get("processed_abs_error", np.nan)),
            "processed_relative_error": float(chosen.get("processed_relative_error", np.nan)),
        }
        for key in ("profile", "replay_probability_source", "replay_probability_extrapolated"):
            if key in chosen:
                out[key] = chosen[key]
        rows.append(out)
    return rows
