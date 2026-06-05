"""Reusable calibrated CABIQAE epsilon sweep for the 6q CVA instance.

This launcher complements ``run_hwd_CVA.py``.  It keeps the same CVA problem,
but runs only CABIQAE and records one final estimate per ``epsilon_target`` and
repetition.  By default it reuses a persisted hardware calibration and executes
local calibrated replays.  The primary output is the median normalized absolute
amplitude error as a function of the actual query cost.

Live hardware sweeps require ``--confirm-hardware-sweep`` because the number of
adaptive sampler jobs can be much larger than a calibration-only run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# ======================================================================
#                       Project import paths
# ======================================================================
current_file: Path = Path(__file__).resolve()
repo_root: Path = next(
    parent for parent in current_file.parents if (parent / "pyproject.toml").exists()
)
src_path: Path = repo_root / "src"
instance_path: Path = (
    repo_root / "cva_pricing_pipeline" / "multi_asset" / "6q_instance"
)

for import_path in (src_path, instance_path):
    import_path_text: str = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)


# ======================================================================
#                       Reusable hardware experiment APIs
# ======================================================================
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_common import (  # noqa: E402
    CURRENT_DIR,
    add_cva_aliases,
    load_config,
    parse_int_list,
    parse_name_list,
    parse_nonnegative_int_list,
    preferred_field_order,
)
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (  # noqa: E402
    _base_config,
    _build_problem_bundle,
    _execute_non_replay_phases,
    _load_backend_for_mode,
    _qctrl_submission_metadata,
    _replay_inputs_from_state,
    _resolve_backend_defaults,
    _resolve_noise_floor,
    _state_extra,
    build_arg_parser as build_hardware_arg_parser,
)
from quantum_cva.amplitude_estimation.experiments.hardware import (  # noqa: E402
    ExperimentState,
    backend_snapshot,
    build_pass_manager_for_backend,
    effective_contrast_prefactor_for_algorithms,
    effective_t_for_algorithms,
    load_existing_state,
    make_replay_probability_extrapolator,
    run_preflight,
    sample_replay_probabilities,
)
from quantum_cva.amplitude_estimation.experiments.io import (  # noqa: E402
    RunPaths,
    load_json,
    save_csv,
    save_json,
)
from quantum_cva.amplitude_estimation.experiments.plotting import (  # noqa: E402
    add_query_scaling_guides,
    save_figure_png_and_pdf,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (  # noqa: E402
    AerCountSampler,
    LoggedAerSampler,
    QctrlPerformanceManagementSampler,
    ReplayCountSampler,
    RuntimeCountSampler,
    build_noise_model,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (  # noqa: E402
    ALGORITHM_LABELS,
    ALGORITHM_STYLES,
    run_algorithm_once,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (  # noqa: E402
    bootstrap_median_ci,
    standard_error,
)


ALGORITHM_KEY = "cabiqae"
DEFAULT_EPSILON_TARGETS = "0.1,0.05,0.02,0.01,0.005,0.002,0.001"
DEFAULT_CALIBRATION_RUN_DIR = CURRENT_DIR / "results" / "q_ctrl_hardware"
FINAL_ESTIMATIONS_FILENAME = "final_cabiqae_estimations.csv"
FINAL_SUMMARY_FILENAME = "final_cabiqae_error_vs_queries_summary.csv"
FINAL_PLOT_FILENAME = "final_cabiqae_normalized_error_vs_queries.png"
PAPER_STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
}

# These defaults intentionally follow run_hwd_CVA.py.  Command-line arguments
# are appended afterwards, so an explicit user option overrides each default.
DEFAULT_RUN_ARGUMENTS: list[str] = [
    "--mode",
    "replay-only",
    "--hardware-executor",
    "qctrl",
    "--instance-name",
    "premium_new_usa",
    "--backend-name",
    "ibm_pittsburgh",
    "--algorithms",
    "cabiqae",
    "--max-grover-power",
    "4",
    "--scan-grover-powers",
    "0,1,2,3,4",
    "--scan-repeats",
    "2",
    "--scan-shots",
    "4096",
    "--readout-shots",
    "8192",
    "--direct-shots",
    "128",
    "--max-direct-calls",
    "64",
    "--replay-max-calls",
    "4096",
    "--replay-probability-mode",
    "normal",
    "--replay-probability-se-scale",
    "1.0",
    "--extrapolate",
    "true",
    "--noise-floor",
    "fit",
    "--cap-kappa",
    "3.0",
    "--cabiqae-hard-k-cap",
    "--session-max-time",
    "24h",
    "--soft-wallclock-limit",
    "315360000",
    "--optimization-level",
    "3",
    "--seed-transpiler",
    "1234",
    "--layout-search-strategy",
    "exhaustive",
    "--reference-ks",
    "0,1,2,3,4",
    "--seed",
    "12345",
    "--no-use-fractional-gates",
    "--verbose",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_hardware_arg_parser()
    parser.description = (
        "Run final direct CABIQAE estimates for a grid of epsilon_target values "
        "using the real 6q CVA problem."
    )
    for action in parser._actions:
        if action.dest == "mode":
            action.choices = ("replay-only", "preflight", "dry-run", "hardware")
    parser.add_argument(
        "--calibration-run-dir",
        default=str(DEFAULT_CALIBRATION_RUN_DIR),
        help=(
            "Existing hardware run whose persisted calibration is reused in "
            f"--mode replay-only. Default: {DEFAULT_CALIBRATION_RUN_DIR}."
        ),
    )
    parser.add_argument(
        "--epsilon-targets",
        default=DEFAULT_EPSILON_TARGETS,
        help=(
            "Comma-separated CABIQAE epsilon_target grid. "
            f"Default: {DEFAULT_EPSILON_TARGETS}."
        ),
    )
    parser.add_argument(
        "--final-repetitions",
        type=int,
        default=5,
        help="Independent final CABIQAE estimates per epsilon_target. Default: 5.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap samples for median confidence intervals. Default: 2000.",
    )
    parser.add_argument(
        "--confirm-hardware-sweep",
        action="store_true",
        help="Required in --mode hardware before any live QPU jobs are submitted.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the resolved sweep plan without loading a backend or submitting jobs.",
    )
    parser.add_argument(
        "--overwrite-run-dir",
        action="store_true",
        help="Allow writing into a non-empty explicit --run-dir.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the sweep after the first failed final estimator.",
    )
    return parser


def _parse_epsilon_targets(raw: str | Sequence[str]) -> tuple[float, ...]:
    values = tuple(float(token) for token in parse_name_list(raw))
    if not values:
        raise ValueError("At least one epsilon_target is required.")
    invalid = [value for value in values if not np.isfinite(value) or not 0.0 < value <= 0.5]
    if invalid:
        raise ValueError(f"epsilon_target values must be finite and in (0, 0.5], got {invalid}.")
    if len(set(values)) != len(values):
        raise ValueError("epsilon_target values must be unique.")
    return values


def _validate_args(args: argparse.Namespace, epsilon_targets: Sequence[float]) -> None:
    if str(args.mode) not in {"replay-only", "preflight", "dry-run", "hardware"}:
        raise ValueError(
            "This final-estimator launcher supports only --mode replay-only, "
            "preflight, dry-run, or hardware."
        )
    if int(args.final_repetitions) <= 0:
        raise ValueError("--final-repetitions must be positive.")
    if int(args.bootstrap_samples) < 0:
        raise ValueError("--bootstrap-samples must be non-negative.")
    if int(args.max_direct_calls) <= 0:
        raise ValueError("--max-direct-calls must be positive.")
    if not epsilon_targets:
        raise ValueError("At least one epsilon_target is required.")
    requested_algorithms = tuple(
        str(value).strip().lower().replace("-", "_")
        for value in parse_name_list(args.algorithms)
    )
    if requested_algorithms != (ALGORITHM_KEY,):
        raise ValueError(
            "This final-estimator launcher executes only --algorithms cabiqae."
        )


def _force_cabiqae_only(args: argparse.Namespace, epsilon_targets: Sequence[float]) -> None:
    # The base runner exposes multi-algorithm and replay flags.  This launcher is
    # deliberately narrower: calibrate once, run direct CABIQAE finals, no replay.
    args.algorithms = ALGORITHM_KEY
    args.skip_direct = True
    args.skip_replay = True
    args.skip_plots = True
    args.epsilon_target = float(epsilon_targets[0])
    args.cabiqae_epsilon_target = float(epsilon_targets[0])


def _print_plan(args: argparse.Namespace, epsilon_targets: Sequence[float]) -> None:
    final_runs = len(epsilon_targets) * int(args.final_repetitions)
    max_calls_per_estimator = (
        int(args.replay_max_calls)
        if str(args.mode) == "replay-only"
        else int(args.max_direct_calls)
    )
    max_adaptive_calls = final_runs * max_calls_per_estimator
    print("=" * 80)
    print("CABIQAE FINAL-ESTIMATOR SWEEP")
    print("=" * 80)
    print(f"mode                    : {args.mode}")
    print(f"hardware_executor       : {args.hardware_executor}")
    print(f"backend_name            : {args.backend_name}")
    if str(args.mode) == "replay-only":
        print(f"calibration_run_dir     : {Path(args.calibration_run_dir).expanduser().resolve()}")
        print("qpu_submissions         : none (local calibrated replay)")
        print(f"replay_probability_mode : {args.replay_probability_mode}")
        print(f"replay_extrapolate      : {bool(args.extrapolate)}")
    print(f"epsilon_targets         : {list(epsilon_targets)}")
    print(f"final_repetitions       : {int(args.final_repetitions)}")
    print(f"final_estimators        : {final_runs}")
    print(f"direct_shots            : {int(args.direct_shots)}")
    print(f"max_calls_per_estimator : {max_calls_per_estimator}")
    print(f"max_adaptive_calls      : {max_adaptive_calls}")
    if str(args.mode) != "replay-only":
        print(f"scan_grover_powers      : {args.scan_grover_powers}")
        print(f"scan_repeats            : {int(args.scan_repeats)}")
        print(f"scan_shots              : {int(args.scan_shots)}")
    print(
        "query_metric            : state-preparation calls "
        "(sum of shots * (2k + 1))"
    )


def _create_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = CURRENT_DIR / "runs" / f"hardware_cva_cabiqae_final_{timestamp}"
    if run_dir.exists() and any(run_dir.iterdir()) and not bool(args.overwrite_run_dir):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. "
            "Use --overwrite-run-dir only when replacing its artifacts is intentional."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _metric_summary(
    values: Sequence[Any],
    *,
    prefix: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    finite = np.asarray([_as_float(value) for value in values], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_median_ci_low": np.nan,
            f"{prefix}_median_ci_high": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_se": np.nan,
        }
    median, ci_low, ci_high = bootstrap_median_ci(
        finite,
        n_boot=int(bootstrap_samples),
        rng=rng,
    )
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(median),
        f"{prefix}_median_ci_low": float(ci_low),
        f"{prefix}_median_ci_high": float(ci_high),
        f"{prefix}_std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        f"{prefix}_se": float(standard_error(finite)),
    }


def _epsilon_matches(row: Mapping[str, Any], epsilon_target: float) -> bool:
    return bool(np.isclose(_as_float(row.get("epsilon_target")), float(epsilon_target)))


def _summary_rows(
    final_rows: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
    *,
    epsilon_targets: Sequence[float],
    repetitions: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for epsilon_target in epsilon_targets:
        group = [row for row in final_rows if _epsilon_matches(row, epsilon_target)]
        failures = [
            row
            for row in error_rows
            if str(row.get("phase")) in {"direct_final", "calibration_replay_final"}
            and _epsilon_matches(row, epsilon_target)
        ]
        row: dict[str, Any] = {
            "algorithm": ALGORITHM_LABELS[ALGORITHM_KEY],
            "algorithm_key": ALGORITHM_KEY,
            "epsilon_target": float(epsilon_target),
            "n_requested": int(repetitions),
            "n_success": len(group),
            "n_failed": len(failures),
            "success_rate": float(len(group) / max(int(repetitions), 1)),
            **_metric_summary(
                [item.get("final_queries") for item in group],
                prefix="final_queries",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            **_metric_summary(
                [item.get("final_normalized_abs_error") for item in group],
                prefix="normalized_abs_error",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            **_metric_summary(
                [item.get("processed_relative_error") for item in group],
                prefix="processed_relative_error",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
            **_metric_summary(
                [item.get("runtime_wall_seconds") for item in group],
                prefix="runtime_wall_seconds",
                bootstrap_samples=bootstrap_samples,
                rng=rng,
            ),
        }
        row["median_final_queries"] = row["final_queries_median"]
        row["mean_final_queries"] = row["final_queries_mean"]
        row["cva_relative_error_median"] = row["processed_relative_error_median"]
        row["cva_relative_error_median_ci_low"] = row[
            "processed_relative_error_median_ci_low"
        ]
        row["cva_relative_error_median_ci_high"] = row[
            "processed_relative_error_median_ci_high"
        ]
        summary.append(row)
    return summary


def _final_rows_path(state: ExperimentState) -> Path:
    return state.paths.run_dir / FINAL_ESTIMATIONS_FILENAME


def _summary_path(state: ExperimentState) -> Path:
    return state.paths.run_dir / FINAL_SUMMARY_FILENAME


def _final_estimator_rows(state: ExperimentState) -> list[dict[str, Any]]:
    if str(state.config.get("final_estimators_execution_mode")) == "calibration_replay":
        return state.replay_final_rows
    return state.direct_final_rows


def _save_final_outputs(
    state: ExperimentState,
    args: argparse.Namespace,
    epsilon_targets: Sequence[float],
) -> list[dict[str, Any]]:
    final_rows = _final_estimator_rows(state)
    summary_rows = _summary_rows(
        final_rows,
        state.error_rows,
        epsilon_targets=epsilon_targets,
        repetitions=int(args.final_repetitions),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    state.config["final_estimators_artifacts"] = {
        "final_estimations_csv": str(_final_rows_path(state)),
        "summary_csv": str(_summary_path(state)),
        "plot_png": str(state.paths.plots_dir / FINAL_PLOT_FILENAME),
    }
    state.persist()
    save_csv(
        final_rows,
        _final_rows_path(state),
        fieldnames=preferred_field_order(final_rows),
    )
    save_csv(summary_rows, _summary_path(state))
    return summary_rows


def _plot_final_error_vs_queries(
    state: ExperimentState,
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    final_rows = _final_estimator_rows(state)
    if not final_rows or not summary_rows:
        return

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 4.25))
        style = ALGORITHM_STYLES[ALGORITHM_KEY]
        color = style["color"]
        marker = style["marker"]

        individual_x = np.asarray(
            [_as_float(row.get("final_queries")) for row in final_rows],
            dtype=float,
        )
        individual_y = np.asarray(
            [_as_float(row.get("final_normalized_abs_error")) for row in final_rows],
            dtype=float,
        )
        individual_valid = (
            np.isfinite(individual_x)
            & np.isfinite(individual_y)
            & (individual_x > 0.0)
            & (individual_y > 0.0)
        )
        if np.any(individual_valid):
            ax.scatter(
                individual_x[individual_valid],
                individual_y[individual_valid],
                color=color,
                marker=marker,
                alpha=0.30,
                s=16,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )

        ordered = sorted(
            summary_rows,
            key=lambda row: _as_float(row.get("median_final_queries")),
        )
        x_values = np.asarray(
            [_as_float(row.get("median_final_queries")) for row in ordered],
            dtype=float,
        )
        y_values = np.asarray(
            [_as_float(row.get("normalized_abs_error_median")) for row in ordered],
            dtype=float,
        )
        ci_low = np.asarray(
            [
                _as_float(row.get("normalized_abs_error_median_ci_low"))
                for row in ordered
            ],
            dtype=float,
        )
        ci_high = np.asarray(
            [
                _as_float(row.get("normalized_abs_error_median_ci_high"))
                for row in ordered
            ],
            dtype=float,
        )
        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & np.isfinite(ci_low)
            & np.isfinite(ci_high)
            & (x_values > 0.0)
            & (y_values > 0.0)
        )
        if np.any(valid):
            lower = np.maximum(0.0, y_values[valid] - ci_low[valid])
            upper = np.maximum(0.0, ci_high[valid] - y_values[valid])
            lower = np.minimum(lower, 0.95 * y_values[valid])
            ax.errorbar(
                x_values[valid],
                y_values[valid],
                yerr=np.vstack([lower, upper]),
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=2.0,
                markersize=6.5,
                elinewidth=1.0,
                capsize=2.8,
                label="CABIQAE",
                zorder=3,
            )
            add_query_scaling_guides(
                ax,
                list(zip(x_values[valid].tolist(), y_values[valid].tolist())),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Final query count $N_q$")
        ax.set_ylabel("Median relative CVA error")
        ax.grid(True, which="major", color="#BFBFBF", linewidth=0.55, alpha=0.32)
        ax.grid(True, which="minor", color="#D7D7D7", linewidth=0.40, alpha=0.16)
        ax.legend(loc="best")
        fig.tight_layout(pad=0.35)
        save_figure_png_and_pdf(fig, state.paths.plots_dir / FINAL_PLOT_FILENAME)
        plt.close(fig)


def _execution_run_kind(
    epsilon_index: int,
    repetition: int,
    *,
    phase: str = "direct_final",
) -> str:
    return f"{phase}_eps_{int(epsilon_index):02d}_rep_{int(repetition):03d}"


def _set_sampler_call_cap(
    sampler: Any,
    *,
    execution_run_kind: str,
    max_calls: int,
) -> None:
    if not hasattr(sampler, "max_calls_by_context"):
        return
    sampler.max_calls_by_context[f"{execution_run_kind}_{ALGORITHM_KEY}"] = int(max_calls)


def _run_cabiqae_final_sweep(
    state: ExperimentState,
    sampler: Any,
    bundle: Any,
    args: argparse.Namespace,
    *,
    epsilon_targets: Sequence[float],
) -> None:
    if args.t_eff is not None:
        t_eff = float(args.t_eff)
        contrast_prefactor = 1.0
    else:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        contrast_prefactor = effective_contrast_prefactor_for_algorithms(
            state.calibration_summary
        )
    contrast_baseline = float(state.calibration_summary["contrast_baseline"])
    state.config["final_estimators_solver"] = {
        "algorithm": ALGORITHM_KEY,
        "epsilon_targets": [float(value) for value in epsilon_targets],
        "final_repetitions": int(args.final_repetitions),
        "direct_shots": int(args.direct_shots),
        "max_sampler_calls_per_estimator": int(args.max_direct_calls),
        "t_eff": None if t_eff is None else float(t_eff),
        "contrast_prefactor": float(contrast_prefactor),
        "contrast_baseline": float(contrast_baseline),
        "query_metric": "state_preparation_calls",
    }
    state.persist()

    for epsilon_index, epsilon_target in enumerate(epsilon_targets):
        for repetition in range(int(args.final_repetitions)):
            execution_run_kind = _execution_run_kind(epsilon_index, repetition)
            _set_sampler_call_cap(
                sampler,
                execution_run_kind=execution_run_kind,
                max_calls=int(args.max_direct_calls),
            )
            try:
                trace_rows, final_row = run_algorithm_once(
                    ALGORITHM_KEY,
                    sampler,
                    bundle,
                    run_kind=execution_run_kind,
                    repetition=int(repetition),
                    epsilon_target=float(epsilon_target),
                    alpha=float(args.alpha),
                    n_shots=int(args.direct_shots),
                    max_queries=sys.maxsize,
                    t_eff=t_eff,
                    seed=int(args.seed) + 1009 * epsilon_index + repetition,
                    algorithm_labels=ALGORITHM_LABELS,
                    cap_kappa=float(args.cap_kappa),
                    disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
                    solver_kwargs={
                        "noise_floor": float(contrast_baseline),
                        "contrast_prefactor": float(contrast_prefactor),
                    },
                    trace_extra={
                        **_state_extra(state),
                        "epsilon_target": float(epsilon_target),
                        "execution_context": execution_run_kind,
                        "n_shots": int(args.direct_shots),
                    },
                    show_details=bool(args.show_details),
                )
                for row in trace_rows:
                    row["run_kind"] = "direct_final"
                final_row.update(
                    {
                        "run_kind": "direct_final",
                        "epsilon_target": float(epsilon_target),
                        "execution_context": execution_run_kind,
                        "n_shots": int(args.direct_shots),
                        "n_trace_steps": len(trace_rows),
                        "query_budget_actual": final_row["final_queries"],
                        "estimate": final_row["final_estimate"],
                        "abs_error": final_row["final_abs_error"],
                        "normalized_abs_error": final_row["final_normalized_abs_error"],
                    }
                )
                state.direct_trace_rows.extend(add_cva_aliases(trace_rows))
                state.direct_final_rows.extend(add_cva_aliases([final_row]))
                print(
                    "[direct_final] "
                    f"eps={float(epsilon_target):.3g} "
                    f"rep={repetition + 1}/{int(args.final_repetitions)} "
                    f"estimate={_as_float(final_row['final_estimate']):.8g} "
                    f"normalized_abs_error="
                    f"{_as_float(final_row['final_normalized_abs_error']):.4g} "
                    f"queries={_as_float(final_row['final_queries']):.0f}",
                    flush=True,
                )
            except Exception as exc:
                state.error_rows.append(
                    {
                        "phase": "direct_final",
                        "algorithm": ALGORITHM_KEY,
                        "algorithm_key": ALGORITHM_KEY,
                        "epsilon_target": float(epsilon_target),
                        "repetition": int(repetition),
                        "execution_context": execution_run_kind,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp_epoch": time.time(),
                    }
                )
                print(
                    "[direct_final] "
                    f"eps={float(epsilon_target):.3g} "
                    f"rep={repetition + 1}/{int(args.final_repetitions)} "
                    f"FAILED: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if bool(args.fail_fast):
                    _save_final_outputs(state, args, epsilon_targets)
                    raise
            _save_final_outputs(state, args, epsilon_targets)

    summary_rows = _save_final_outputs(state, args, epsilon_targets)
    try:
        _plot_final_error_vs_queries(state, summary_rows)
    except Exception as exc:
        state.error_rows.append(
            {
                "phase": "final_plot",
                "algorithm": ALGORITHM_KEY,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_epoch": time.time(),
            }
        )
        state.persist()
        print(f"[final_plot] FAILED: {type(exc).__name__}: {exc}", flush=True)
    if not state.direct_final_rows:
        raise RuntimeError("All final CABIQAE estimators failed. Inspect errors.csv.")


def _validate_calibration_compatibility(
    source_state: ExperimentState,
    bundle: Any,
    args: argparse.Namespace,
) -> None:
    source_config = source_state.config
    mismatches: list[str] = []
    if str(source_config.get("good_bitstring")) != str(bundle.good_bitstring):
        mismatches.append("good_bitstring")
    if [int(value) for value in source_config.get("objective_qubits", [])] != [
        int(value) for value in bundle.problem.objective_qubits
    ]:
        mismatches.append("objective_qubits")
    if str(source_config.get("target_name")) != str(bundle.target_name):
        mismatches.append("target_name")

    for key, current_value in (
        ("a_true", bundle.true_amplitude),
        ("processed_true_value", bundle.processed_true_value),
    ):
        source_value = _as_float(source_config.get(key))
        if not np.isfinite(source_value) or not np.isclose(
            source_value,
            float(current_value),
            rtol=0.0,
            atol=1e-12,
        ):
            mismatches.append(key)

    source_metadata = dict(source_config.get("bundle_metadata", {}) or {})
    current_metadata = dict(bundle.metadata)
    for key in (
        "builder",
        "qcbm_topology",
        "qcbm_n_layers",
        "exposure_ansatz",
        "exposure_n_layers",
        "artifact_paths",
    ):
        if source_metadata.get(key) != current_metadata.get(key):
            mismatches.append(f"bundle_metadata.{key}")

    if mismatches:
        raise ValueError(
            "The persisted calibration is incompatible with the current CVA "
            f"problem bundle: {sorted(set(mismatches))}."
        )
    if not source_state.amplification_point_rows:
        raise ValueError("Calibration source has no amplification_points.csv rows.")
    if "contrast_baseline" not in source_state.calibration_summary:
        raise ValueError("Calibration source has no fitted contrast_baseline.")
    if bool(args.extrapolate):
        make_replay_probability_extrapolator(bundle, source_state.calibration_summary)


def _initialize_reused_calibration_experiment(
    args: argparse.Namespace,
    *,
    epsilon_targets: Sequence[float],
) -> tuple[ExperimentState, Any]:
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    source_dir = Path(args.calibration_run_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Calibration run directory does not exist: {source_dir}")
    if args.run_dir and Path(args.run_dir).expanduser().resolve() == source_dir:
        raise ValueError("--run-dir must differ from --calibration-run-dir.")

    source_state = load_existing_state(source_dir)
    _validate_calibration_compatibility(source_state, bundle, args)

    run_dir = _create_run_dir(args)
    paths = RunPaths(run_dir)
    contrast_baseline = float(source_state.calibration_summary["contrast_baseline"])
    state = ExperimentState(
        paths=paths,
        config=_base_config(
            args=args,
            bundle=bundle,
            run_dir=run_dir,
            mode="replay-only",
            algorithms=(ALGORITHM_KEY,),
            budgets=parse_int_list(args.budgets),
            contrast_baseline=contrast_baseline,
        ),
    )
    state.config.update(
        {
            "pipeline": "6q_cva_cabiqae_final_estimators",
            "final_estimators_execution_mode": "calibration_replay",
            "epsilon_target_grid": [float(value) for value in epsilon_targets],
            "final_repetitions": int(args.final_repetitions),
            "bootstrap_samples": int(args.bootstrap_samples),
            "calibration_reuse": {
                "source_run_dir": str(source_dir),
                "source_run_id": str(source_state.config.get("run_id", source_dir.name)),
                "source_calibration_summary_json": str(
                    source_state.paths.calibration_summary
                ),
                "source_amplification_points_csv": str(
                    source_state.paths.amplification_points
                ),
                "source_readout_calibration_csv": str(
                    source_state.paths.readout_calibration
                ),
                "new_qpu_submissions": False,
            },
        }
    )
    state.readout_rows = [dict(row) for row in source_state.readout_rows]
    state.amplification_count_rows = [
        dict(row) for row in source_state.amplification_count_rows
    ]
    state.amplification_point_rows = [
        dict(row) for row in source_state.amplification_point_rows
    ]
    state.calibration_summary = dict(source_state.calibration_summary)
    state.session_details = {
        "mode": "replay-only",
        "created_at_epoch": time.time(),
        "calibration_source_run_dir": str(source_dir),
        "new_qpu_submissions": False,
    }
    source_snapshot = load_json(source_state.paths.backend_snapshot, default={}) or {}
    save_json(
        {
            **dict(source_snapshot),
            "calibration_reused_from": str(source_dir),
            "new_qpu_submissions": False,
        },
        paths.backend_snapshot,
    )
    if source_state.paths.transpilation_report.exists():
        shutil.copy2(
            source_state.paths.transpilation_report,
            paths.transpilation_report,
        )
    state.persist()
    return state, bundle


def _run_cabiqae_calibration_replay_sweep(
    state: ExperimentState,
    bundle: Any,
    args: argparse.Namespace,
    *,
    epsilon_targets: Sequence[float],
) -> None:
    p_by_k, p_se_by_k = _replay_inputs_from_state(state)
    if str(args.replay_probability_mode) == "normal" and p_se_by_k is None:
        raise ValueError(
            "--replay-probability-mode normal requires standard errors in "
            "amplification_points.csv."
        )
    if args.t_eff is not None:
        t_eff = float(args.t_eff)
        contrast_prefactor = 1.0
    else:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        contrast_prefactor = effective_contrast_prefactor_for_algorithms(
            state.calibration_summary
        )
    contrast_baseline = float(state.calibration_summary["contrast_baseline"])
    extrapolate_probability = None
    if bool(args.extrapolate):
        extrapolate_probability, extrapolation_metadata = (
            make_replay_probability_extrapolator(bundle, state.calibration_summary)
        )
        state.config["replay_extrapolation_model"] = extrapolation_metadata
    extrapolated_cache: dict[int, float] = {}
    source_dir = Path(args.calibration_run_dir).expanduser().resolve()
    state.config["final_estimators_solver"] = {
        "algorithm": ALGORITHM_KEY,
        "execution_mode": "calibration_replay",
        "calibration_source_run_dir": str(source_dir),
        "epsilon_targets": [float(value) for value in epsilon_targets],
        "final_repetitions": int(args.final_repetitions),
        "direct_shots": int(args.direct_shots),
        "replay_max_calls_per_estimator": int(args.replay_max_calls),
        "replay_probability_mode": str(args.replay_probability_mode),
        "replay_probability_se_scale": float(args.replay_probability_se_scale),
        "replay_extrapolate": bool(args.extrapolate),
        "t_eff": None if t_eff is None else float(t_eff),
        "contrast_prefactor": float(contrast_prefactor),
        "contrast_baseline": float(contrast_baseline),
        "query_metric": "state_preparation_calls",
        "construct_circuit_mode": "metadata_only",
    }
    state.persist()

    for epsilon_index, epsilon_target in enumerate(epsilon_targets):
        for repetition in range(int(args.final_repetitions)):
            experiment_index = epsilon_index * int(args.final_repetitions) + repetition
            execution_context = _execution_run_kind(
                epsilon_index,
                repetition,
                phase="calibration_replay_final",
            )
            sampled_p_by_k = sample_replay_probabilities(
                p_by_k,
                p_se_by_k,
                mode=str(args.replay_probability_mode),
                rng=np.random.default_rng(int(args.seed) + 7919 * experiment_index),
                se_scale=float(args.replay_probability_se_scale),
            )
            sampler = ReplayCountSampler(
                sampled_p_by_k,
                bundle,
                seed=int(args.seed) + 1009 * experiment_index,
                max_calls=int(args.replay_max_calls),
                extrapolate_probability=extrapolate_probability,
                extrapolated_cache=extrapolated_cache,
            )
            try:
                trace_rows, final_row = run_algorithm_once(
                    ALGORITHM_KEY,
                    sampler,
                    bundle,
                    run_kind="calibration_replay_final",
                    repetition=int(repetition),
                    epsilon_target=float(epsilon_target),
                    alpha=float(args.alpha),
                    n_shots=int(args.direct_shots),
                    max_queries=sys.maxsize,
                    t_eff=t_eff,
                    seed=int(args.seed) + experiment_index,
                    algorithm_labels=ALGORITHM_LABELS,
                    cap_kappa=float(args.cap_kappa),
                    disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
                    construct_circuit_mode="metadata_only",
                    solver_kwargs={
                        "noise_floor": float(contrast_baseline),
                        "contrast_prefactor": float(contrast_prefactor),
                    },
                    trace_extra={
                        **_state_extra(state),
                        "epsilon_target": float(epsilon_target),
                        "execution_context": execution_context,
                        "n_shots": int(args.direct_shots),
                        "calibration_source_run_dir": str(source_dir),
                        "replay_probability_mode": str(args.replay_probability_mode),
                    },
                    show_details=bool(args.show_details),
                )
                extrapolated_ks = sorted(int(k) for k in sampler.extrapolated_ks_used)
                for row in trace_rows:
                    row["replay_probability_source"] = (
                        "extrapolated"
                        if int(row["grover_power"]) in sampler.extrapolated_ks_used
                        else "measured"
                    )
                    row["replay_probability_extrapolated"] = (
                        int(row["grover_power"]) in sampler.extrapolated_ks_used
                    )
                final_row.update(
                    {
                        "run_kind": "calibration_replay_final",
                        "epsilon_target": float(epsilon_target),
                        "execution_context": execution_context,
                        "n_shots": int(args.direct_shots),
                        "n_trace_steps": len(trace_rows),
                        "query_budget_actual": final_row["final_queries"],
                        "estimate": final_row["final_estimate"],
                        "abs_error": final_row["final_abs_error"],
                        "normalized_abs_error": final_row[
                            "final_normalized_abs_error"
                        ],
                        "calibration_source_run_dir": str(source_dir),
                        "extrapolated_replay_ks_json": json.dumps(extrapolated_ks),
                        "n_extrapolated_replay_ks": len(extrapolated_ks),
                    }
                )
                state.replay_trace_rows.extend(add_cva_aliases(trace_rows))
                state.replay_final_rows.extend(add_cva_aliases([final_row]))
                print(
                    "[calibration_replay_final] "
                    f"eps={float(epsilon_target):.3g} "
                    f"rep={repetition + 1}/{int(args.final_repetitions)} "
                    f"estimate={_as_float(final_row['final_estimate']):.8g} "
                    f"normalized_abs_error="
                    f"{_as_float(final_row['final_normalized_abs_error']):.4g} "
                    f"queries={_as_float(final_row['final_queries']):.0f}",
                    flush=True,
                )
            except Exception as exc:
                cause = getattr(exc, "__cause__", None)
                cause_message = (
                    f"{type(cause).__name__}: {cause}" if cause is not None else ""
                )
                state.error_rows.append(
                    {
                        "phase": "calibration_replay_final",
                        "algorithm": ALGORITHM_KEY,
                        "algorithm_key": ALGORITHM_KEY,
                        "epsilon_target": float(epsilon_target),
                        "repetition": int(repetition),
                        "execution_context": execution_context,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "error_cause": cause_message,
                        "timestamp_epoch": time.time(),
                    }
                )
                detail = f" caused by {cause_message}" if cause_message else ""
                print(
                    "[calibration_replay_final] "
                    f"eps={float(epsilon_target):.3g} "
                    f"rep={repetition + 1}/{int(args.final_repetitions)} "
                    f"FAILED: {type(exc).__name__}: {exc}{detail}",
                    flush=True,
                )
                if bool(args.fail_fast):
                    _save_final_outputs(state, args, epsilon_targets)
                    raise
            _save_final_outputs(state, args, epsilon_targets)

    if extrapolated_cache:
        state.config["replay_extrapolated_probabilities"] = {
            str(k): float(value) for k, value in sorted(extrapolated_cache.items())
        }
    summary_rows = _save_final_outputs(state, args, epsilon_targets)
    try:
        _plot_final_error_vs_queries(state, summary_rows)
    except Exception as exc:
        state.error_rows.append(
            {
                "phase": "final_plot",
                "algorithm": ALGORITHM_KEY,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_epoch": time.time(),
            }
        )
        state.persist()
        print(f"[final_plot] FAILED: {type(exc).__name__}: {exc}", flush=True)
    if not state.replay_final_rows:
        raise RuntimeError("All replayed final CABIQAE estimators failed. Inspect errors.csv.")


def _initialize_experiment(
    args: argparse.Namespace,
    *,
    epsilon_targets: Sequence[float],
) -> tuple[ExperimentState, Any, Any, Any, int, str | float]:
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    contrast_baseline = _resolve_noise_floor(args.noise_floor, bundle)
    budgets = parse_int_list(args.budgets)
    run_dir = _create_run_dir(args)
    paths = RunPaths(run_dir)
    state = ExperimentState(
        paths=paths,
        config=_base_config(
            args=args,
            bundle=bundle,
            run_dir=run_dir,
            mode=str(args.mode),
            algorithms=(ALGORITHM_KEY,),
            budgets=budgets,
            contrast_baseline=contrast_baseline,
        ),
    )
    state.config["pipeline"] = "6q_cva_cabiqae_final_estimators"
    state.config["epsilon_target_grid"] = [float(value) for value in epsilon_targets]
    state.config["final_repetitions"] = int(args.final_repetitions)
    state.config["bootstrap_samples"] = int(args.bootstrap_samples)
    state.session_details = {"mode": str(args.mode), "created_at_epoch": time.time()}

    backend, backend_mode = _load_backend_for_mode(args, str(args.mode))
    state.config["backend_mode"] = backend_mode
    snapshot = backend_snapshot(backend, mode=backend_mode, channel=str(args.channel))
    snapshot["use_fractional_gates_requested"] = bool(args.use_fractional_gates)
    snapshot["hardware_executor"] = str(args.hardware_executor)
    snapshot["instance_name"] = str(args.instance_name)
    snapshot["backend_name_requested"] = str(args.backend_name)
    if str(args.hardware_executor) == "qctrl":
        snapshot["qctrl_submission_strategy"] = "abstract_logical_circuits"
        snapshot["qctrl_transpilation_policy"] = "fire_opal_managed"
        snapshot["local_preflight_role"] = "diagnostic_and_k_cap_only"
        state.config["qctrl_submission"] = _qctrl_submission_metadata(args)
    save_json(snapshot, paths.backend_snapshot)

    pass_manager, transpilation_metadata = build_pass_manager_for_backend(
        backend,
        bundle,
        mode=str(args.mode),
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=parse_nonnegative_int_list(args.reference_ks),
        routing_method=args.routing_method,
        layout_search_strategy=args.layout_search_strategy,
        verbose=bool(args.verbose),
    )
    state.config["transpilation"] = transpilation_metadata
    state.persist()

    _, allowed_max = run_preflight(
        bundle,
        pass_manager,
        state,
        max_grover_power=int(args.max_grover_power),
        max_isa_depth=int(args.max_isa_depth),
        max_isa_2q=int(args.max_isa_2q),
        verbose=bool(args.verbose),
    )
    max_experiment_k = min(int(args.max_grover_power), int(allowed_max))
    state.config["max_grover_power_after_preflight"] = int(max_experiment_k)
    state.persist()
    return state, bundle, backend, pass_manager, max_experiment_k, contrast_baseline


def _calibrate_and_sweep(
    args: argparse.Namespace,
    state: ExperimentState,
    sampler: Any,
    bundle: Any,
    max_experiment_k: int,
    contrast_baseline: str | float,
    *,
    epsilon_targets: Sequence[float],
    batch_scan_circuits: bool = False,
) -> None:
    _execute_non_replay_phases(
        args,
        state,
        sampler,
        bundle,
        max_experiment_k,
        algorithms=(ALGORITHM_KEY,),
        contrast_baseline=contrast_baseline,
        batch_scan_circuits=bool(batch_scan_circuits),
    )
    _run_cabiqae_final_sweep(
        state,
        sampler,
        bundle,
        args,
        epsilon_targets=epsilon_targets,
    )


def _run_dry_run(
    args: argparse.Namespace,
    state: ExperimentState,
    backend: Any,
    pass_manager: Any,
    bundle: Any,
    max_experiment_k: int,
    contrast_baseline: str | float,
    *,
    epsilon_targets: Sequence[float],
) -> None:
    noise_model = build_noise_model(
        float(args.dry_run_noise_scale),
        profile=str(args.dry_run_noise_profile),
    )
    aer = AerCountSampler(
        noise_model=noise_model,
        seed=int(args.seed),
        method=str(args.aer_method),
        transpile_backend=backend,
        pass_manager=pass_manager,
    )
    sampler = LoggedAerSampler(
        aer,
        state.job_rows,
        max_grover_power=int(max_experiment_k),
    )
    _calibrate_and_sweep(
        args,
        state,
        sampler,
        bundle,
        max_experiment_k,
        contrast_baseline,
        epsilon_targets=epsilon_targets,
    )


def _run_hardware(
    args: argparse.Namespace,
    state: ExperimentState,
    backend: Any,
    pass_manager: Any,
    bundle: Any,
    max_experiment_k: int,
    contrast_baseline: str | float,
    *,
    epsilon_targets: Sequence[float],
) -> None:
    global_start = time.perf_counter()
    if str(args.hardware_executor) == "qctrl":
        from qiskit_ibm_runtime import Session

        with Session(backend=backend, max_time=args.session_max_time) as session:
            session_id = getattr(session, "session_id", None)
            if not session_id:
                raise RuntimeError(
                    "Q-CTRL execution requires a Qiskit Runtime Session id, "
                    "but the opened session did not expose one."
                )
            state.session_details.update(
                {
                    "session_id": session_id,
                    "hardware_executor": "qctrl",
                    "instance_name": str(args.instance_name),
                    "backend_name": str(args.backend_name),
                    "qiskit_function_name": str(args.qiskit_function_name),
                    "qiskit_function_channel": str(args.qiskit_function_channel or ""),
                    "submitted_circuit_kind": "abstract_logical",
                    "qctrl_transpilation_policy": "fire_opal_managed",
                    "runtime_session_strategy": "existing_qiskit_runtime_session",
                    "amplification_scan_submission": "batched_pubs_single_job",
                    "session_started_at_epoch": time.time(),
                }
            )
            state.persist()
            sampler = QctrlPerformanceManagementSampler(
                instance_name=str(args.instance_name),
                backend_name=str(args.backend_name),
                pass_manager=pass_manager,
                job_rows=state.job_rows,
                soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                max_grover_power=int(max_experiment_k),
                max_calls_by_context={},
                start_time=global_start,
                function_name=str(args.qiskit_function_name),
                catalog_channel=getattr(args, "qiskit_function_channel", None),
                session_id=session_id,
                verbose=bool(args.verbose),
            )
            _calibrate_and_sweep(
                args,
                state,
                sampler,
                bundle,
                max_experiment_k,
                contrast_baseline,
                epsilon_targets=epsilon_targets,
                batch_scan_circuits=True,
            )
            state.session_details["session_finished_at_epoch"] = time.time()
            state.persist()
        return

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime import Session

    with Session(backend=backend, max_time=args.session_max_time) as session:
        state.session_details.update(
            {
                "session_id": getattr(session, "session_id", None),
                "hardware_executor": "runtime",
                "session_started_at_epoch": time.time(),
            }
        )
        state.persist()
        runtime_sampler = Sampler(mode=session)
        sampler = RuntimeCountSampler(
            backend,
            runtime_sampler,
            pass_manager,
            state.job_rows,
            soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
            max_grover_power=int(max_experiment_k),
            max_calls_by_context={},
            start_time=global_start,
        )
        _calibrate_and_sweep(
            args,
            state,
            sampler,
            bundle,
            max_experiment_k,
            contrast_baseline,
            epsilon_targets=epsilon_targets,
        )
        state.session_details["session_finished_at_epoch"] = time.time()
        state.persist()


def run(argv: list[str] | None = None) -> Path | None:
    parser = build_arg_parser()
    args = parser.parse_args(DEFAULT_RUN_ARGUMENTS + list(argv or []))
    epsilon_targets = _parse_epsilon_targets(args.epsilon_targets)
    _validate_args(args, epsilon_targets)
    _force_cabiqae_only(args, epsilon_targets)
    _print_plan(args, epsilon_targets)

    if bool(args.plan_only):
        return None
    if str(args.mode) == "hardware" and not bool(args.confirm_hardware_sweep):
        raise RuntimeError(
            "Live hardware sweep not confirmed. Review the printed workload and "
            "rerun with --confirm-hardware-sweep to submit QPU jobs."
        )

    if str(args.mode) == "replay-only":
        state, bundle = _initialize_reused_calibration_experiment(
            args,
            epsilon_targets=epsilon_targets,
        )
        _run_cabiqae_calibration_replay_sweep(
            state,
            bundle,
            args,
            epsilon_targets=epsilon_targets,
        )
        print(f"Final CABIQAE artifacts saved in: {state.paths.run_dir}")
        return state.paths.run_dir

    state, bundle, backend, pass_manager, max_experiment_k, contrast_baseline = (
        _initialize_experiment(args, epsilon_targets=epsilon_targets)
    )
    if str(args.mode) == "preflight":
        print(f"Preflight artifacts saved in: {state.paths.run_dir}")
        return state.paths.run_dir
    if str(args.mode) == "dry-run":
        _run_dry_run(
            args,
            state,
            backend,
            pass_manager,
            bundle,
            max_experiment_k,
            contrast_baseline,
            epsilon_targets=epsilon_targets,
        )
    else:
        _run_hardware(
            args,
            state,
            backend,
            pass_manager,
            bundle,
            max_experiment_k,
            contrast_baseline,
            epsilon_targets=epsilon_targets,
        )
    print(f"Final CABIQAE artifacts saved in: {state.paths.run_dir}")
    return state.paths.run_dir


def main(argv: list[str] | None = None) -> None:
    run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
