from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from dataclasses import replace
from typing import Any, Iterable

import numpy as np


def _bootstrap_src_path() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent
        for parent in current.parents
        if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    pipeline_dir = current.parents[1]
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))
    return repo_root


REPO_ROOT = _bootstrap_src_path()
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

from full_cva_pipeline import CONFIG as BASE_CONFIG  # noqa: E402
from quantum_cva.multi_asset.pipeline_cfg.cfg_utilities import (  # noqa: E402
    CVAPipelineRunner,
    InstrumentConfig,
    PipelineConfig,
    run_pipeline,
)

BASE_CALL_STRIKE = float(BASE_CONFIG.instruments[0].strike)
BASE_PUT_STRIKE = float(BASE_CONFIG.instruments[1].strike)

DEFAULT_CASES: tuple[tuple[str, float, float], ...] = (
    ("base", BASE_CALL_STRIKE, BASE_PUT_STRIKE),
    ("call_down_10pct", 0.90 * BASE_CALL_STRIKE, BASE_PUT_STRIKE),
    ("call_up_10pct", 1.10 * BASE_CALL_STRIKE, BASE_PUT_STRIKE),
    ("put_down_10pct", BASE_CALL_STRIKE, 0.90 * BASE_PUT_STRIKE),
    ("put_up_10pct", BASE_CALL_STRIKE, 1.10 * BASE_PUT_STRIKE),
    ("both_down_10pct", 0.90 * BASE_CALL_STRIKE, 0.90 * BASE_PUT_STRIKE),
    ("both_up_10pct", 1.10 * BASE_CALL_STRIKE, 1.10 * BASE_PUT_STRIKE),
)

CASE_FIELDNAMES = [
    "case_id",
    "repeat",
    "call_strike",
    "put_strike",
    "call_strike_scale",
    "put_strike_scale",
    "benchmark_path",
    "exposure_training_path",
    "results_dir",
    "cva_statevector",
    "cva_classical_reference",
    "absolute_error",
    "signed_error",
    "absolute_relative_error_pct",
    "signed_relative_error_pct",
    "p111_statevector",
    "exposure_l2_statevector",
    "qcbm_kl_statevector",
    "default_l2_statevector",
    "discount_l2_statevector",
    "exposure_best_l2",
    "exposure_best_l2_rechecked",
    "exposure_final_l2",
    "exposure_elapsed_s",
    "classical_cva_mc_continuous",
    "classical_cva_std_err_mc_continuous",
    "classical_cva_limit",
    "classical_cva_small_scaled",
    "classical_elapsed_s",
    "final_elapsed_s",
    "shots",
    "theta_seed",
    "shot_seed",
    "simulator_seed",
    "seed_transpiler",
    "status",
    "error_message",
]

SUMMARY_FIELDNAMES = [
    "scope",
    "n",
    "mean_abs_rel_error_pct",
    "std_abs_rel_error_pct",
    "stderr_abs_rel_error_pct",
    "median_abs_rel_error_pct",
    "max_abs_rel_error_pct",
    "mean_signed_rel_error_pct",
    "std_signed_rel_error_pct",
    "stderr_signed_rel_error_pct",
    "rmse_rel_error_pct",
    "mean_abs_error",
    "std_abs_error",
    "stderr_abs_error",
    "max_abs_error",
]


def _case_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip(
        "_"
    )


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _replace_instrument_strike(
    instrument: InstrumentConfig,
    strike: float,
) -> InstrumentConfig:
    return replace(instrument, strike=float(strike))


def _with_case_paths(
    config: PipelineConfig,
    *,
    case_dir: pathlib.Path,
) -> PipelineConfig:
    paths = replace(
        config.paths,
        benchmark_relative_path=str(case_dir / "benchmark" / "benchmark.npz"),
        crca_exposure_training_relative_path=str(
            case_dir
            / "quantum"
            / "training"
            / "crca"
            / "positive_exposure"
            / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
        ),
        results_dir_relative_path=str(case_dir / "pipeline_run"),
    )
    return replace(config, paths=paths)


def _with_case_strikes(
    config: PipelineConfig,
    *,
    call_strike: float,
    put_strike: float,
) -> PipelineConfig:
    if len(config.instruments) < 2:
        raise ValueError("Expected at least two instruments in BASE_CONFIG.")
    instruments = list(config.instruments)
    instruments[0] = _replace_instrument_strike(instruments[0], call_strike)
    instruments[1] = _replace_instrument_strike(instruments[1], put_strike)
    return replace(config, instruments=tuple(instruments))


def _with_repeat_seeds(
    config: PipelineConfig, *, repeat: int
) -> PipelineConfig:
    if repeat == 0:
        return config
    stride = int(config.crca_exposure_training.repeat_seed_stride)
    exposure_training = replace(
        config.crca_exposure_training,
        theta_seed=int(config.crca_exposure_training.theta_seed)
        + stride * repeat,
        shot_seed=int(config.crca_exposure_training.shot_seed)
        + stride * repeat,
    )
    backend_noise = replace(
        config.backend_noise,
        simulator_seed=int(config.backend_noise.simulator_seed)
        + stride * repeat,
        seed_transpiler=int(config.backend_noise.seed_transpiler) + repeat,
    )
    return replace(
        config,
        crca_exposure_training=exposure_training,
        backend_noise=backend_noise,
    )


def _without_amplitude_estimation(config: PipelineConfig) -> PipelineConfig:
    final_cva = replace(config.final_cva, run_qae=False, run_iqae=False)
    return replace(config, final_cva=final_cva)


def _with_exposure_training_overrides(
    config: PipelineConfig,
    *,
    no_warmstart: bool,
    maxiter: int | None,
    shots: int | None,
) -> PipelineConfig:
    exposure_training = config.crca_exposure_training
    if no_warmstart:
        exposure_training = replace(
            exposure_training,
            use_statevector_warmstart=False,
        )
    if maxiter is not None:
        exposure_training = replace(
            exposure_training,
            single_stage_maxiter=int(maxiter),
        )
    if shots is not None:
        exposure_training = replace(exposure_training, shots=int(shots))
    return replace(config, crca_exposure_training=exposure_training)


def _with_final_cva_flags(
    config: PipelineConfig,
    *,
    run_qae: bool,
    run_iqae: bool,
) -> PipelineConfig:
    final_cva = replace(config.final_cva, run_qae=run_qae, run_iqae=run_iqae)
    return replace(config, final_cva=final_cva)


def _read_cases_csv(path: pathlib.Path) -> list[tuple[str, float, float]]:
    cases: list[tuple[str, float, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "call_strike", "put_strike"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Cases CSV missing columns: {sorted(missing)}")
        for row in reader:
            cases.append(
                (
                    _case_slug(str(row["case_id"])),
                    float(row["call_strike"]),
                    float(row["put_strike"]),
                )
            )
    if not cases:
        raise ValueError(f"No cases found in {path}.")
    return cases


def _write_cases_template(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "call_strike", "put_strike"])
        writer.writerows(DEFAULT_CASES)


def _append_case_row(path: pathlib.Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CASE_FIELDNAMES})


def _write_csv(
    path: pathlib.Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_existing_case_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summary_row(scope: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    abs_rel = np.asarray(
        [
            _float_or_nan(row.get("absolute_relative_error_pct"))
            for row in successful
        ],
        dtype=float,
    )
    signed_rel = np.asarray(
        [
            _float_or_nan(row.get("signed_relative_error_pct"))
            for row in successful
        ],
        dtype=float,
    )
    abs_err = np.asarray(
        [_float_or_nan(row.get("absolute_error")) for row in successful],
        dtype=float,
    )
    abs_rel = abs_rel[np.isfinite(abs_rel)]
    signed_rel = signed_rel[np.isfinite(signed_rel)]
    abs_err = abs_err[np.isfinite(abs_err)]
    n = int(abs_rel.size)

    def std(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    def stderr(values: np.ndarray) -> float:
        return (
            std(values) / math.sqrt(values.size)
            if values.size
            else float("nan")
        )

    if n == 0:
        return {"scope": scope, "n": 0}

    return {
        "scope": scope,
        "n": n,
        "mean_abs_rel_error_pct": float(np.mean(abs_rel)),
        "std_abs_rel_error_pct": std(abs_rel),
        "stderr_abs_rel_error_pct": stderr(abs_rel),
        "median_abs_rel_error_pct": float(np.median(abs_rel)),
        "max_abs_rel_error_pct": float(np.max(abs_rel)),
        "mean_signed_rel_error_pct": float(np.mean(signed_rel)),
        "std_signed_rel_error_pct": std(signed_rel),
        "stderr_signed_rel_error_pct": stderr(signed_rel),
        "rmse_rel_error_pct": float(np.sqrt(np.mean(signed_rel**2))),
        "mean_abs_error": float(np.mean(abs_err)),
        "std_abs_error": std(abs_err),
        "stderr_abs_error": stderr(abs_err),
        "max_abs_error": float(np.max(abs_err)),
    }


def _write_summary(
    case_csv_path: pathlib.Path, summary_csv_path: pathlib.Path
) -> None:
    rows = _load_existing_case_rows(case_csv_path)
    summary_rows = [_summary_row("all_successful_runs", rows)]
    case_ids = sorted(
        {row.get("case_id", "") for row in rows if row.get("case_id")}
    )
    for case_id in case_ids:
        case_rows = [row for row in rows if row.get("case_id") == case_id]
        summary_rows.append(_summary_row(f"case:{case_id}", case_rows))
    _write_csv(summary_csv_path, summary_rows, SUMMARY_FIELDNAMES)


def _result_row(
    *,
    case_id: str,
    repeat: int,
    call_strike: float,
    put_strike: float,
    config: PipelineConfig,
    case_dir: pathlib.Path,
    classical_result: dict[str, Any],
    exposure_result: dict[str, Any],
    final_result: dict[str, Any],
) -> dict[str, Any]:
    cva_statevector = float(final_result["cva_statevector"])
    cva_classical = float(final_result["cva_classical_reference"])
    signed_error = cva_statevector - cva_classical
    signed_relative = (
        signed_error / abs(cva_classical) * 100.0
        if cva_classical != 0.0
        else float("nan")
    )
    return {
        "case_id": case_id,
        "repeat": repeat,
        "call_strike": call_strike,
        "put_strike": put_strike,
        "call_strike_scale": call_strike / BASE_CALL_STRIKE,
        "put_strike_scale": put_strike / BASE_PUT_STRIKE,
        "benchmark_path": str(case_dir / "benchmark" / "benchmark.npz"),
        "exposure_training_path": exposure_result.get("path", ""),
        "results_dir": str(case_dir / "pipeline_run"),
        "cva_statevector": cva_statevector,
        "cva_classical_reference": cva_classical,
        "absolute_error": abs(signed_error),
        "signed_error": signed_error,
        "absolute_relative_error_pct": abs(signed_relative),
        "signed_relative_error_pct": signed_relative,
        "p111_statevector": final_result.get("p111_statevector", ""),
        "exposure_l2_statevector": final_result.get(
            "exposure_l2_statevector", ""
        ),
        "qcbm_kl_statevector": final_result.get("qcbm_kl_statevector", ""),
        "default_l2_statevector": final_result.get(
            "default_l2_statevector", ""
        ),
        "discount_l2_statevector": final_result.get(
            "discount_l2_statevector", ""
        ),
        "exposure_best_l2": exposure_result.get("best_l2", ""),
        "exposure_best_l2_rechecked": exposure_result.get(
            "best_l2_rechecked", ""
        ),
        "exposure_final_l2": exposure_result.get("final_l2", ""),
        "exposure_elapsed_s": exposure_result.get("stage_elapsed_s", ""),
        "classical_cva_mc_continuous": classical_result.get(
            "cva_mc_continuous", ""
        ),
        "classical_cva_std_err_mc_continuous": classical_result.get(
            "cva_std_err_mc_continuous",
            "",
        ),
        "classical_cva_limit": classical_result.get("cva_limit", ""),
        "classical_cva_small_scaled": classical_result.get(
            "cva_small_scaled", ""
        ),
        "classical_elapsed_s": classical_result.get("stage_elapsed_s", ""),
        "final_elapsed_s": final_result.get("stage_elapsed_s", ""),
        "shots": config.crca_exposure_training.shots,
        "theta_seed": config.crca_exposure_training.theta_seed,
        "shot_seed": config.crca_exposure_training.shot_seed,
        "simulator_seed": config.backend_noise.simulator_seed,
        "seed_transpiler": config.backend_noise.seed_transpiler,
        "status": "ok",
        "error_message": "",
    }


def _error_row(
    *,
    case_id: str,
    repeat: int,
    call_strike: float,
    put_strike: float,
    config: PipelineConfig,
    case_dir: pathlib.Path,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "repeat": repeat,
        "call_strike": call_strike,
        "put_strike": put_strike,
        "call_strike_scale": call_strike / BASE_CALL_STRIKE,
        "put_strike_scale": put_strike / BASE_PUT_STRIKE,
        "benchmark_path": str(case_dir / "benchmark" / "benchmark.npz"),
        "results_dir": str(case_dir / "pipeline_run"),
        "shots": config.crca_exposure_training.shots,
        "theta_seed": config.crca_exposure_training.theta_seed,
        "shot_seed": config.crca_exposure_training.shot_seed,
        "simulator_seed": config.backend_noise.simulator_seed,
        "seed_transpiler": config.backend_noise.seed_transpiler,
        "status": "error",
        "error_message": repr(exc),
    }


def _build_case_config(
    *,
    call_strike: float,
    put_strike: float,
    case_dir: pathlib.Path,
    repeat: int,
    args: argparse.Namespace,
) -> PipelineConfig:
    config = _with_case_strikes(
        BASE_CONFIG,
        call_strike=call_strike,
        put_strike=put_strike,
    )
    config = _with_case_paths(config, case_dir=case_dir)
    config = _with_repeat_seeds(config, repeat=repeat)
    config = _with_exposure_training_overrides(
        config,
        no_warmstart=bool(args.no_warmstart),
        maxiter=args.exposure_maxiter,
        shots=args.exposure_shots,
    )
    config = _with_final_cva_flags(
        config,
        run_qae=bool(args.run_qae),
        run_iqae=bool(args.run_iqae),
    )
    if not args.run_qae and not args.run_iqae:
        config = _without_amplitude_estimation(config)
    return config


def _print_plan(
    *,
    output_dir: pathlib.Path,
    cases: list[tuple[str, float, float]],
    repeats: int,
) -> None:
    print("============================================================")
    print("CVA robustness test plan")
    print("============================================================")
    print(f"output_dir={output_dir}")
    print(f"n_cases={len(cases)}")
    print(f"repeats_per_case={repeats}")
    for case_id, call_strike, put_strike in cases:
        print(
            f"- {case_id}: call_strike={call_strike:.8g}, "
            f"put_strike={put_strike:.8g}"
        )


def _prepare_shared_artifacts(args: argparse.Namespace) -> None:
    if not args.prepare_shared:
        return
    print("============================================================")
    print("Preparing shared noise+shots artifacts")
    print("============================================================")
    for stage in (
        "classical",
        "train_qcbm",
        "train_crca_default",
        "train_crca_discount",
    ):
        run_pipeline(
            BASE_CONFIG,
            stage=stage,
            resume=True,
            force=False,
            dry_run=False,
        )


def run_sweep(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_cases_template:
        _write_cases_template(pathlib.Path(args.write_cases_template))
        print(f"Wrote cases template: {args.write_cases_template}")
        return

    cases = (
        _read_cases_csv(pathlib.Path(args.cases_csv))
        if args.cases_csv
        else list(DEFAULT_CASES)
    )
    repeats = int(args.repeats)
    _print_plan(output_dir=output_dir, cases=cases, repeats=repeats)

    if args.dry_run:
        return

    _prepare_shared_artifacts(args)

    case_csv_path = output_dir / "cva_robustness_case_results.csv"
    summary_csv_path = output_dir / "cva_robustness_summary_statistics.csv"
    config_path = output_dir / "robustness_sweep_config.json"
    config_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": case_id,
                        "call_strike": call_strike,
                        "put_strike": put_strike,
                    }
                    for case_id, call_strike, put_strike in cases
                ],
                "repeats": repeats,
                "resume": bool(args.resume),
                "force": bool(args.force),
                "run_qae": bool(args.run_qae),
                "run_iqae": bool(args.run_iqae),
                "exposure_shots": args.exposure_shots,
                "exposure_maxiter": args.exposure_maxiter,
                "no_warmstart": bool(args.no_warmstart),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for case_id, call_strike, put_strike in cases:
        for repeat in range(repeats):
            run_id = case_id if repeats == 1 else f"{case_id}_rep{repeat:02d}"
            case_dir = output_dir / _case_slug(run_id)
            config = _build_case_config(
                call_strike=call_strike,
                put_strike=put_strike,
                case_dir=case_dir,
                repeat=repeat,
                args=args,
            )
            runner = CVAPipelineRunner(config)
            print(
                "============================================================"
            )
            print(f"Running robustness case: {run_id}")
            print(
                "============================================================"
            )
            try:
                classical_result = runner.run(
                    stage="classical",
                    resume=bool(args.resume),
                    force=bool(args.force),
                )["classical"]
                exposure_result = runner.run(
                    stage="train_crca_exposure",
                    resume=bool(args.resume),
                    force=bool(args.force),
                )["train_crca_exposure"]
                final_result = runner.run(
                    stage="final_statevector_cva",
                    resume=False,
                    force=True,
                )["final_statevector_cva"]
                row = _result_row(
                    case_id=case_id,
                    repeat=repeat,
                    call_strike=call_strike,
                    put_strike=put_strike,
                    config=config,
                    case_dir=case_dir,
                    classical_result=classical_result,
                    exposure_result=exposure_result,
                    final_result=final_result,
                )
            except Exception as exc:
                row = _error_row(
                    case_id=case_id,
                    repeat=repeat,
                    call_strike=call_strike,
                    put_strike=put_strike,
                    config=config,
                    case_dir=case_dir,
                    exc=exc,
                )
                if not args.keep_going:
                    _append_case_row(case_csv_path, row)
                    _write_summary(case_csv_path, summary_csv_path)
                    raise
            _append_case_row(case_csv_path, row)
            _write_summary(case_csv_path, summary_csv_path)
            print(f"[CSV] appended: {case_csv_path}")
            print(f"[CSV] updated: {summary_csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run noisy-shots positive-exposure CRCA robustness tests over "
            "different strike compositions and evaluate CVA with statevector."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR),
        help="Directory where per-case artifacts and CSV outputs are written.",
    )
    parser.add_argument(
        "--cases-csv",
        default=None,
        help="CSV with columns: case_id, call_strike, put_strike.",
    )
    parser.add_argument(
        "--write-cases-template",
        default=None,
        help="Write a default cases CSV template to this path and exit.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each strike case with shifted noisy training seeds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-case classical and exposure artifacts if present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of per-case classical and exposure artifacts.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failed cases in the CSV and continue with remaining cases.",
    )
    parser.add_argument(
        "--prepare-shared",
        action="store_true",
        help=(
            "Ensure shared QCBM/default/discount noisy-shots artifacts exist "
            "before the strike sweep."
        ),
    )
    parser.add_argument(
        "--run-qae",
        action="store_true",
        help="Also run textbook QAE in the final statevector stage.",
    )
    parser.add_argument(
        "--run-iqae",
        action="store_true",
        help="Also run IQAE in the final statevector stage.",
    )
    parser.add_argument(
        "--exposure-shots",
        type=int,
        default=None,
        help="Override CRCA positive-exposure noisy training shots.",
    )
    parser.add_argument(
        "--exposure-maxiter",
        type=int,
        default=None,
        help="Override single-stage CRCA positive-exposure SPSA iterations.",
    )
    parser.add_argument(
        "--no-warmstart",
        action="store_true",
        help="Disable the statevector warmstart for positive-exposure CRCA.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved sweep plan without running stages.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if int(args.repeats) < 1:
        raise ValueError("--repeats must be >= 1.")
    run_sweep(args)


if __name__ == "__main__":
    main()