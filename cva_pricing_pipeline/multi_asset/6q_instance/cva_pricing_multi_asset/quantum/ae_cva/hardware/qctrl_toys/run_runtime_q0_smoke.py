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
    circuit_metrics,
    construct_measured_circuit,
    two_qubit_count,
)
from quantum_cva.amplitude_estimation.experiments.cva import (  # noqa: E402
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_common import (  # noqa: E402
    load_config,
)
from quantum_cva.amplitude_estimation.experiments.hardware import (  # noqa: E402
    backend_snapshot,
    build_pass_manager_for_backend,
)
from quantum_cva.amplitude_estimation.experiments.io import save_csv, save_json  # noqa: E402
from quantum_cva.amplitude_estimation.experiments.samplers import (  # noqa: E402
    RuntimeCountSampler,
    count_good_from_counts,
    extract_result_counts,
    ideal_good_probability_for_circuit,
)


DEFAULT_BACKEND_NAME = "ibm_aachen"
DEFAULT_INSTANCE_NAME = "premium_new"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal IBM Runtime Sampler smoke test for the 6q CVA Q^0 "
            "state-preparation circuit."
        )
    )
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND_NAME)
    parser.add_argument("--instance-name", default=DEFAULT_INSTANCE_NAME)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--session-max-time", default="30m")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--soft-wallclock-limit", type=float, default=1800.0)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=1234)
    parser.add_argument("--routing-method", default="sabre")
    parser.add_argument(
        "--layout-search-strategy",
        choices=("fast", "exhaustive", "preset"),
        default="fast",
    )
    parser.add_argument(
        "--use-fractional-gates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Request IBM fractional-gate targets. Disabled by default because "
            "Runtime rejects some transpiled RZZ angles outside [0, pi/2] on "
            "current fractional-gate targets."
        ),
    )
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
    parser.add_argument("--show-counts", action="store_true")
    return parser


def _create_run_dir(raw: str | None) -> Path:
    if raw:
        run_dir = Path(raw).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = current_file.parent / "results" / f"runtime_q0_smoke_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _runtime_service(args: argparse.Namespace) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    if str(args.instance_name or "").strip():
        return QiskitRuntimeService(name=str(args.instance_name))
    if str(args.channel or "").strip():
        return QiskitRuntimeService(channel=str(args.channel))
    return QiskitRuntimeService()


def _load_backend(args: argparse.Namespace) -> Any:
    service = _runtime_service(args)
    try:
        return service.backend(
            str(args.backend_name),
            use_fractional_gates=bool(args.use_fractional_gates),
        )
    except TypeError:
        return service.backend(str(args.backend_name))


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
            status = value()
        except Exception as exc:
            return f"status_unavailable:{type(exc).__name__}:{exc}"
    else:
        status = value
    name = getattr(status, "name", None)
    return str(name or status or "")


def _wait_for_result(job: Any, poll_seconds: float) -> Any:
    terminal = {"DONE", "ERROR", "CANCELLED", "CANCELED"}
    while True:
        status = _job_status(job)
        if status:
            print(f"[{time.strftime('%H:%M:%S')}] Runtime job {_job_id(job)} status={status}", flush=True)
        if status.upper() in terminal:
            return job.result()
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
        0,
        source="runtime_q0_smoke",
    )
    p_ideal = ideal_good_probability_for_circuit(circuit, bundle)

    backend = _load_backend(args)
    snapshot = backend_snapshot(
        backend,
        mode="runtime_q0_smoke",
        channel=str(args.channel or ""),
    )
    snapshot["hardware_executor"] = "runtime"
    snapshot["backend_name_requested"] = str(args.backend_name)
    snapshot["instance_name"] = str(args.instance_name)
    snapshot["use_fractional_gates_requested"] = bool(args.use_fractional_gates)
    save_json(snapshot, run_dir / "backend_snapshot.json")

    pass_manager, transpilation_metadata = build_pass_manager_for_backend(
        backend,
        bundle,
        mode="hardware",
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=(0,),
        routing_method=args.routing_method,
        layout_search_strategy=args.layout_search_strategy,
        verbose=True,
    )
    decomposed = circuit.decompose(reps=10)
    isa = pass_manager.run(decomposed)
    logical_metrics = circuit_metrics(circuit)
    isa_metrics = circuit_metrics(isa)
    transpilation_row = {
        "grover_power": 0,
        "amplification_factor": 1,
        "logical_depth": logical_metrics["depth"],
        "logical_2q": logical_metrics["two_qubit_gates"],
        "decomposed_depth": circuit_metrics(decomposed)["depth"],
        "decomposed_2q": circuit_metrics(decomposed)["two_qubit_gates"],
        "isa_depth": isa_metrics["depth"],
        "isa_size": isa_metrics["size"],
        "isa_2q": int(two_qubit_count(isa)),
        "isa_swap_count": isa_metrics["swap_count"],
    }
    save_json(
        {
            "strategy": "local_isa_for_runtime_sampler",
            "transpilation": transpilation_metadata,
            "metrics": transpilation_row,
        },
        run_dir / "transpilation_summary.json",
    )
    save_csv([transpilation_row], run_dir / "transpilation_summary.csv")

    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime import Session

    job_rows: list[dict[str, Any]] = []
    submitted_at = time.time()
    with Session(backend=backend, max_time=str(args.session_max_time)) as session:
        runtime_sampler = Sampler(mode=session)
        sampler = RuntimeCountSampler(
            backend,
            runtime_sampler,
            pass_manager,
            job_rows,
            soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
            max_grover_power=0,
            max_calls_by_context={"runtime_q0_smoke": 1},
            start_time=time.perf_counter(),
        )
        sampler.set_context("runtime_q0_smoke")
        job = sampler.run([circuit], shots=int(args.shots))
        print(f"Submitted Runtime Q^0 smoke job: {_job_id(job)}", flush=True)
        print(f"Session id: {getattr(session, 'session_id', None)}", flush=True)
        result = _wait_for_result(job, float(args.poll_seconds))
        session_id = getattr(session, "session_id", None)

    finished_at = time.time()
    counts = extract_result_counts(result, 0)
    good = count_good_from_counts(counts, bundle)
    total = int(sum(counts.values()))
    p_raw = float(good / max(total, 1))
    cva_estimate = float(bundle.process(p_raw))

    summary = {
        "run_id": run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "hardware_executor": "runtime",
        "primitive": "sampler",
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "session_id": str(session_id or ""),
        "submitted_circuit_kind": "local_isa",
        "local_pass_manager_applied": True,
        "grover_power": 0,
        "amplification_factor": 1,
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
        **transpilation_row,
    }

    save_json(summary, run_dir / "summary.json")
    save_csv([summary], run_dir / "summary.csv")
    save_csv(job_rows, run_dir / "runtime_jobs.csv")

    print(f"Observed shots: {total}")
    print(f"Good counts for {bundle.good_bitstring}: {good}")
    print(f"Raw success probability: {p_raw:.8f}")
    print(f"Ideal Q^0 success probability: {p_ideal:.8f}")
    print(f"CVA estimate from raw probability: {cva_estimate:.8f}")
    print(f"Saved summary: {run_dir / 'summary.json'}")
    if args.show_counts:
        print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
