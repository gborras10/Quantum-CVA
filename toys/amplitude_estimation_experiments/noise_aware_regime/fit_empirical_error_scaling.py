from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TOY_DIR = ROOT_DIR / "toys" / "amplitude_estimation_experiments"
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

HARDWARE_DIR = CURRENT_DIR / "3qubit_toy" / "hardware" / "beta_hardware_experiment"
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

from toys.amplitude_estimation_experiments.common_utils.plotting_utils import (  # noqa: E402
    select_log_spaced_indices,
)
from toys.amplitude_estimation_experiments.ideal_regime.ideal_utils import (  # noqa: E402
    aggregate_budget_summary,
)


def _load_module_from_path(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hardware_module = _load_module_from_path(
    "hardware_replay_query_plot",
    HARDWARE_DIR / "hardware_replay_query_plot.py",
)
DEFAULT_BUDGETS = _hardware_module.DEFAULT_BUDGETS
aggregate_fixed_budget_summary = _hardware_module.aggregate_fixed_budget_summary
budget_rows_from_trace_rows = _hardware_module.budget_rows_from_trace_rows


ALGORITHM_LABELS = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae_latentt": "CABIQAE",
    "classical_mc": "Classical MC",
    "iqae": "IQAE",
}


@dataclass(frozen=True)
class DatasetSpec:
    regime: str
    title: str
    summary_csv: Path
    budget_rows_csv: Path
    trace_rows_csv: Path | None
    monte_carlo_budget_rows_csv: Path | None
    plot_max_points_per_algorithm: int


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        regime="ideal",
        title="Ideal regime",
        summary_csv=ROOT_DIR
        / "toys"
        / "amplitude_estimation_experiments"
        / "ideal_regime"
        / "experiment_results"
        / "plots"
        / "bae_biqae_iqae_cabiqae_latentt_ideal_budget_summary.csv",
        budget_rows_csv=ROOT_DIR
        / "toys"
        / "amplitude_estimation_experiments"
        / "ideal_regime"
        / "experiment_results"
        / "bae_biqae_iqae_cabiqae_latentt_ideal_budget_rows.csv",
        trace_rows_csv=None,
        monte_carlo_budget_rows_csv=ROOT_DIR
        / "toys"
        / "amplitude_estimation_experiments"
        / "ideal_regime"
        / "experiment_results"
        / "classical_mc_ideal_budget_rows.csv",
        plot_max_points_per_algorithm=12,
    ),
    DatasetSpec(
        regime="hardware",
        title="Noise-aware hardware replay",
        summary_csv=HARDWARE_DIR
        / "experiment_results"
        / "csv_results"
        / "plots"
        / "hardware_replay_actual_queries_summary.csv",
        budget_rows_csv=HARDWARE_DIR
        / "experiment_results"
        / "csv_results"
        / "replay_budget_rows.csv",
        trace_rows_csv=HARDWARE_DIR
        / "experiment_results"
        / "csv_results"
        / "replay_trace_rows.csv",
        monte_carlo_budget_rows_csv=HARDWARE_DIR
        / "experiment_results"
        / "csv_results"
        / "montecarlo_budget_rows.csv",
        plot_max_points_per_algorithm=14,
    ),
)


def _read_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _append_extra_rows(base_rows: pd.DataFrame, extra_path: Path | None) -> pd.DataFrame:
    if extra_path is None or not extra_path.exists() or extra_path.stat().st_size == 0:
        return base_rows
    extra = pd.read_csv(extra_path)
    if extra.empty:
        return base_rows
    if "algorithm" in base_rows:
        has_classical_mc = base_rows["algorithm"].astype(str).str.lower().eq("classical mc").any()
        has_classical_mc_key = False
        if "algorithm_key" in base_rows:
            has_classical_mc_key = base_rows["algorithm_key"].astype(str).str.lower().eq("classical_mc").any()
        if has_classical_mc or has_classical_mc_key:
            return base_rows
    return pd.concat([base_rows, extra], ignore_index=True, sort=False)


def _count_repetitions(rows: pd.DataFrame) -> int:
    if "repetition" not in rows:
        return 0
    repetitions = pd.to_numeric(rows["repetition"], errors="coerce").dropna()
    return int(repetitions.nunique()) if not repetitions.empty else 0


def _normalize_algorithm_key(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"cabiqae", "cabiqae_latentt", "cabiqae-latentt"}:
        return "cabiqae_latentt"
    return raw


def _algorithm_label(algorithm_key: str) -> str:
    return ALGORITHM_LABELS.get(algorithm_key, algorithm_key.upper())


def _query_budget_column(df: pd.DataFrame) -> str:
    for column_name in (
        "query_budget_actual_mean",
        "query_budget_actual_median",
        "query_budget_actual",
        "query_budget",
        "budget",
        "final_queries",
    ):
        if column_name in df:
            return column_name
    raise ValueError("No query-budget column found in the input data.")


def _load_or_build_summary(spec: DatasetSpec) -> pd.DataFrame:
    summary = _read_csv_if_available(spec.summary_csv)
    if not summary.empty:
        return summary

    if spec.regime == "ideal":
        rows = _read_csv_if_available(spec.budget_rows_csv)
        if rows.empty:
            raise FileNotFoundError(f"No data found at {spec.budget_rows_csv}")
        rows = _append_extra_rows(rows, spec.monte_carlo_budget_rows_csv)
        summary_rows = aggregate_budget_summary(
            rows.to_dict("records"),
            total_repetitions=_count_repetitions(rows),
            max_bins=12,
            min_points_per_bin=30,
            bootstrap_samples=10_000,
            confidence_level=0.90,
            bootstrap_seed=12345,
        )
        return pd.DataFrame(summary_rows)

    rows = _read_csv_if_available(spec.budget_rows_csv)
    if rows.empty:
        trace_rows = _read_csv_if_available(spec.trace_rows_csv) if spec.trace_rows_csv else pd.DataFrame()
        if trace_rows.empty:
            raise FileNotFoundError(f"No replay rows found at {spec.budget_rows_csv} or {spec.trace_rows_csv}")
        rows = budget_rows_from_trace_rows(trace_rows, DEFAULT_BUDGETS)
    rows = _append_extra_rows(rows, spec.monte_carlo_budget_rows_csv)
    summary_rows = aggregate_fixed_budget_summary(
        rows,
        total_repetitions=_count_repetitions(rows),
        bootstrap_samples=2000,
        confidence_level=0.95,
        bootstrap_seed=12345,
    )
    return pd.DataFrame(summary_rows)


def _selected_plot_rows(summary: pd.DataFrame, *, max_points_per_algorithm: int) -> pd.DataFrame:
    if summary.empty:
        return summary

    x_col = _query_budget_column(summary)
    key_column = "algorithm_key" if "algorithm_key" in summary else "algorithm"
    selected_frames: list[pd.DataFrame] = []

    for algorithm_key, group in summary.groupby(summary[key_column].map(_normalize_algorithm_key), sort=False):
        ordered = group.sort_values(x_col).reset_index(drop=True)
        x_values = pd.to_numeric(ordered[x_col], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(ordered["normalized_abs_error_median"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
        selected = select_log_spaced_indices(x_values, valid, max(2, int(max_points_per_algorithm)))
        if selected.size == 0:
            continue
        selected_frames.append(ordered.iloc[selected].copy())

    if not selected_frames:
        return summary.iloc[0:0].copy()
    return pd.concat(selected_frames, ignore_index=True, sort=False)


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return {
            "slope": np.nan,
            "slope_se": np.nan,
            "intercept_log": np.nan,
            "intercept_log_se": np.nan,
            "prefactor": np.nan,
            "rmse_log": np.nan,
            "mae_log": np.nan,
            "r2_log": np.nan,
            "adj_r2_log": np.nan,
            "residual_std_log": np.nan,
        }

    log_x = np.log(x)
    log_y = np.log(y)
    if x.size >= 3:
        coeffs, cov = np.polyfit(log_x, log_y, deg=1, cov=True)
        slope = float(coeffs[0])
        intercept_log = float(coeffs[1])
        slope_se = float(np.sqrt(cov[0, 0]))
        intercept_log_se = float(np.sqrt(cov[1, 1]))
    else:
        slope, intercept_log = np.polyfit(log_x, log_y, deg=1)
        slope = float(slope)
        intercept_log = float(intercept_log)
        slope_se = np.nan
        intercept_log_se = np.nan

    fitted = intercept_log + slope * log_x
    residuals = log_y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    dof = max(int(x.size) - 2, 1)
    residual_std = float(np.sqrt(ss_res / dof))
    adj_r2 = np.nan
    if np.isfinite(r2) and x.size > 2:
        adj_r2 = float(1.0 - (1.0 - r2) * (x.size - 1) / (x.size - 2))

    return {
        "slope": slope,
        "slope_se": slope_se,
        "intercept_log": intercept_log,
        "intercept_log_se": intercept_log_se,
        "prefactor": float(np.exp(intercept_log)),
        "rmse_log": rmse,
        "mae_log": mae,
        "r2_log": float(r2),
        "adj_r2_log": adj_r2,
        "residual_std_log": residual_std,
    }


def _fit_dataset(spec: DatasetSpec) -> list[dict[str, Any]]:
    summary = _load_or_build_summary(spec)
    if summary.empty:
        return []

    selected = _selected_plot_rows(summary, max_points_per_algorithm=spec.plot_max_points_per_algorithm)
    if selected.empty:
        return []

    x_col = _query_budget_column(selected)
    key_column = "algorithm_key" if "algorithm_key" in selected else "algorithm"
    rows: list[dict[str, Any]] = []

    grouped = selected.groupby(selected[key_column].map(_normalize_algorithm_key), sort=False)
    for algorithm_key, group in grouped:
        ordered = group.sort_values(x_col)
        x_values = pd.to_numeric(ordered[x_col], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(ordered["normalized_abs_error_median"], errors="coerce").to_numpy(dtype=float)
        metrics = _fit_power_law(x_values, y_values)
        valid = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0.0) & (y_values > 0.0)
        x_valid = x_values[valid]
        y_valid = y_values[valid]
        if x_valid.size == 0:
            continue
        rows.append(
            {
                "regime": spec.regime,
                "regime_label": spec.title,
                "algorithm": _algorithm_label(algorithm_key),
                "algorithm_key": algorithm_key,
                "fit_scope": "plotted_points",
                "x_column": x_col,
                "y_column": "normalized_abs_error_median",
                "n_points": int(x_valid.size),
                "x_min": float(np.nanmin(x_valid)),
                "x_max": float(np.nanmax(x_valid)),
                "y_min": float(np.nanmin(y_valid)),
                "y_max": float(np.nanmax(y_valid)),
                **metrics,
                "model": "log(y) = log(c) + m log(x)",
                "source_summary_csv": str(spec.summary_csv),
            }
        )
    return rows


def build_table(regimes: list[str]) -> pd.DataFrame:
    selected_specs = [spec for spec in DATASETS if spec.regime in regimes]
    if not selected_specs:
        raise ValueError(f"Unknown regimes requested: {', '.join(regimes)}")

    rows: list[dict[str, Any]] = []
    for spec in selected_specs:
        rows.extend(_fit_dataset(spec))

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    order = [
        "regime",
        "regime_label",
        "algorithm",
        "algorithm_key",
        "fit_scope",
        "x_column",
        "y_column",
        "n_points",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "slope",
        "slope_se",
        "intercept_log",
        "intercept_log_se",
        "prefactor",
        "rmse_log",
        "mae_log",
        "r2_log",
        "adj_r2_log",
        "residual_std_log",
        "model",
        "source_summary_csv",
    ]
    return table[order].sort_values(["regime", "algorithm"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit empirical error scaling versus query budget for the ideal and hardware regimes."
    )
    parser.add_argument(
        "--regimes",
        default="ideal,hardware",
        help="Comma-separated list of regimes to process: ideal, hardware.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CURRENT_DIR / "empirical_error_scaling_table.csv",
        help="CSV file where the fit table will be written.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip printing the table to stdout.",
    )
    args = parser.parse_args()

    regimes = [part.strip().lower() for part in args.regimes.split(",") if part.strip()]
    table = build_table(regimes)
    if table.empty:
        raise RuntimeError("No fit rows were produced. Check that the source CSVs exist and contain data.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    if not args.no_print:
        display_cols = [
            "regime",
            "algorithm",
            "n_points",
            "slope",
            "prefactor",
            "r2_log",
            "rmse_log",
        ]
        print(table[display_cols].to_string(index=False))
        print(f"\nSaved fit table to: {args.output}")


if __name__ == "__main__":
    main()