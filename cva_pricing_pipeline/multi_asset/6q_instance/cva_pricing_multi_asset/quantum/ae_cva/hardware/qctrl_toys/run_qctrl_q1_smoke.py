from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any


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
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)


from quantum_cva.amplitude_estimation.experiments.circuits import (  # noqa: E402
    construct_measured_circuit,
)
from quantum_cva.amplitude_estimation.experiments.cva import (  # noqa: E402
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_common import (  # noqa: E402
    load_config,
)
from quantum_cva.amplitude_estimation.experiments.io import save_csv, save_json  # noqa: E402
from quantum_cva.amplitude_estimation.experiments.samplers import (  # noqa: E402
    QctrlPerformanceManagementSampler,
    count_good_from_counts,
    extract_result_counts,
    ideal_good_probability_for_circuit,
)


DEFAULT_BACKEND_NAME = "ibm_pittsburgh"
DEFAULT_INSTANCE_NAME = "premium_new_usa"
DEFAULT_QISKIT_FUNCTION_CHANNEL = "ibm_quantum_platform"
DEFAULT_QISKIT_FUNCTION_NAME = "q-ctrl/performance-management"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal Q-CTRL Performance Management Sampler smoke test for "
            "the 6q CVA Q^1 amplified circuit."
        )
    )
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND_NAME)
    parser.add_argument("--instance-name", default=DEFAULT_INSTANCE_NAME)
    parser.add_argument(
        "--qiskit-function-channel",
        default=DEFAULT_QISKIT_FUNCTION_CHANNEL,
        help=(
            "QiskitFunctionsCatalog channel. Set to an empty string to use "
            "--instance-name as the saved catalog account name instead."
        ),
    )
    parser.add_argument("--qiskit-function-name", default=DEFAULT_QISKIT_FUNCTION_NAME)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline config object as 'module:CONFIG'. Defaults to full_cva_pipeline:CONFIG.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Python file containing a PipelineConfig object.",
    )
    parser.add_argument("--config-attr", default="CONFIG")
    parser.add_argument("--soft-wallclock-limit", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit the Q-CTRL job, save the job metadata, and exit without waiting.",
    )
    parser.add_argument("--show-counts", action="store_true")
    return parser


def _create_run_dir(raw: str | None) -> Path:
    if raw:
        run_dir = Path(raw).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = current_file.parent / "results" / f"qctrl_q1_smoke_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _job_id(job: Any) -> str:
    value = getattr(job, "job_id", None)
    if callable(value):
        return str(value())
    if value is not None:
        return str(value)
    return str(job)


def _job_status(job: Any) -> str:
    value = getattr(job, "status", None)
    if callable(value):
        try:
            return str(value())
        except Exception as exc:
            return f"status_unavailable:{type(exc).__name__}:{exc}"
    if value is not None:
        return str(value)
    return ""


def _save_submission(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    job: Any,
    job_rows: list[dict[str, Any]],
    submitted_at: float,
    circuit: Any,
    p_ideal: float,
) -> None:
    submission = {
        "run_id": run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "hardware_executor": "qctrl",
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "primitive": "sampler",
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "job_id": _job_id(job),
        "submitted_at_epoch": float(submitted_at),
        "initial_status": _job_status(job),
        "grover_power": 1,
        "amplification_factor": 3,
        "shots_requested": int(args.shots),
        "p_ideal": float(p_ideal),
        "logical_depth": int(circuit.depth() or 0),
        "logical_size": int(circuit.size()),
        "logical_num_qubits": int(circuit.num_qubits),
        "logical_num_clbits": int(circuit.num_clbits),
    }
    save_json(submission, run_dir / "submission.json")
    save_csv([submission], run_dir / "submission.csv")
    save_csv(job_rows, run_dir / "qctrl_jobs.csv")


def _wait_for_result(job: Any, poll_seconds: float) -> Any:
    success = {"DONE", "SUCCEEDED"}
    failed = {"ERROR", "FAILED", "CANCELLED", "CANCELED", "STOPPED"}
    while True:
        status = _job_status(job)
        if status:
            print(f"[{time.strftime('%H:%M:%S')}] Q-CTRL job {_job_id(job)} status={status}", flush=True)
        status_upper = status.upper()
        if status_upper in success:
            return job.result()
        if status_upper in failed:
            raise RuntimeError(f"Q-CTRL job {_job_id(job)} finished with status={status}.")
        time.sleep(max(1.0, float(poll_seconds)))


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if int(args.shots) <= 0:
        raise ValueError("--shots must be positive.")

    run_dir = _create_run_dir(args.run_dir)
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    bundle = build_6q_cva_problem_bundle(config, repo_root=repo_root)

    if str(bundle.good_bitstring) != "111" or list(bundle.problem.objective_qubits) != [6, 7, 8]:
        raise ValueError(
            "This smoke test is specialized to the 6q CVA objective "
            "qubits [6, 7, 8] and good_bitstring='111'."
        )

    circuit = construct_measured_circuit(
        bundle.problem,
        1,
        source="qctrl_q1_smoke",
    )
    p_ideal = ideal_good_probability_for_circuit(circuit, bundle)
    job_rows: list[dict[str, Any]] = []
    sampler = QctrlPerformanceManagementSampler(
        instance_name=str(args.instance_name),
        backend_name=str(args.backend_name),
        pass_manager=None,
        job_rows=job_rows,
        soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
        max_grover_power=1,
        max_calls_by_context={"qctrl_q1_smoke": 1},
        start_time=time.perf_counter(),
        function_name=str(args.qiskit_function_name),
        catalog_channel=str(args.qiskit_function_channel or "") or None,
    )
    sampler.set_context("qctrl_q1_smoke")

    submitted_at = time.time()
    job = sampler.run([circuit], shots=int(args.shots))
    print(f"Submitted Q-CTRL Q^1 smoke job: {_job_id(job)}")
    status = _job_status(job)
    if status:
        print(f"Initial status: {status}")
    _save_submission(
        run_dir=run_dir,
        args=args,
        job=job,
        job_rows=job_rows,
        submitted_at=submitted_at,
        circuit=circuit,
        p_ideal=p_ideal,
    )
    print(f"Saved submission metadata: {run_dir / 'submission.json'}")
    if args.submit_only:
        return

    result = _wait_for_result(job, poll_seconds=float(args.poll_seconds))
    finished_at = time.time()
    counts = extract_result_counts(result, 0)
    good = count_good_from_counts(counts, bundle)
    total = int(sum(counts.values()))
    p_raw = float(good / max(total, 1))
    cva_estimate = float(bundle.process(p_raw))

    summary = {
        "run_id": run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "hardware_executor": "qctrl",
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "primitive": "sampler",
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "submitted_circuit_kind": "abstract_logical",
        "qctrl_transpilation_policy": "fire_opal_managed",
        "local_pass_manager_applied": False,
        "local_preflight_role": "not_used_for_smoke_test",
        "grover_power": 1,
        "amplification_factor": 3,
        "shots_requested": int(args.shots),
        "shots_observed": total,
        "job_id": _job_id(job),
        "submitted_at_epoch": submitted_at,
        "finished_at_epoch": finished_at,
        "wall_seconds": float(finished_at - submitted_at),
        "good_bitstring": str(bundle.good_bitstring),
        "objective_qubits": [int(q) for q in bundle.problem.objective_qubits],
        "objective_width": int(bundle.objective_width),
        "good_counts": int(good),
        "bad_counts": int(total - good),
        "p_raw": p_raw,
        "p_ideal": float(p_ideal),
        "p_true_state_preparation": float(bundle.true_amplitude),
        "cva_estimate": cva_estimate,
        "cva_true": float(bundle.processed_true_value),
        "counts_json": json.dumps(counts, sort_keys=True),
        "logical_depth": int(circuit.depth() or 0),
        "logical_size": int(circuit.size()),
        "logical_num_qubits": int(circuit.num_qubits),
        "logical_num_clbits": int(circuit.num_clbits),
    }

    save_json(summary, run_dir / "summary.json")
    save_csv([summary], run_dir / "summary.csv")
    save_csv(job_rows, run_dir / "qctrl_jobs.csv")

    print(f"Observed shots: {total}")
    print(f"Good counts for {bundle.good_bitstring}: {good}")
    print(f"Raw success probability: {p_raw:.8f}")
    print(f"Ideal Q^1 success probability: {p_ideal:.8f}")
    print(f"CVA estimate from raw probability: {cva_estimate:.8f}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
    if args.show_counts:
        print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
