from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector


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
    CURRENT_DIR,
    load_config,
)
from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (  # noqa: E402
    _load_backend_for_mode,
    _resolve_backend_defaults,
)
from quantum_cva.amplitude_estimation.experiments.hardware import (  # noqa: E402
    backend_snapshot,
)
from quantum_cva.amplitude_estimation.experiments.io import (  # noqa: E402
    save_csv,
    save_json,
)
from quantum_cva.amplitude_estimation.experiments.problems import (  # noqa: E402
    normalize_bitstring,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (  # noqa: E402
    QctrlPerformanceManagementSampler,
    extract_result_counts,
    ideal_good_probability_for_circuit,
)


DEFAULT_RUN_DIR_PREFIX = "qctrl_objective_readout_calibration"
DEFAULT_INSTANCE_NAME = "premium_new_usa"
DEFAULT_BACKEND_NAME = "ibm_pittsburgh"
DEFAULT_QISKIT_FUNCTION_NAME = "q-ctrl/performance-management"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Q-CTRL/Fire Opal assignment calibration for the three CVA "
            "objective qubits. The calibration prepares all logical objective "
            "bitstrings, measures [6, 7, 8] into c0, and highlights the "
            "110 -> 111 false-positive channel suspected in the k=0 data."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--config-attr", default="CONFIG")
    parser.add_argument("--instance-name", default=DEFAULT_INSTANCE_NAME)
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND_NAME)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--channel", default="ibm_cloud")
    parser.add_argument("--qiskit-function-name", default=DEFAULT_QISKIT_FUNCTION_NAME)
    parser.add_argument("--qiskit-function-channel", default="")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--session-max-time", default="24h")
    parser.add_argument("--soft-wallclock-limit", type=float, default=315360000.0)
    parser.add_argument("--use-fractional-gates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-k0-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.repo_root = repo_root
    args.mode = "hardware"
    args.hardware_executor = "qctrl"
    return args


def all_bitstrings(width: int) -> list[str]:
    return [format(index, f"0{int(width)}b") for index in range(2**int(width))]


def objective_assignment_circuit(
    *,
    num_qubits: int,
    objective_qubits: Sequence[int],
    prepared_bitstring: str,
) -> QuantumCircuit:
    """Prepare one objective-register basis state using CVA bitstring semantics.

    For Qiskit counts and Statevector.probabilities_dict(qargs=[6, 7, 8]), the
    rightmost character corresponds to objective_qubits[0]. Preparing "110"
    therefore sets objective qubits [6, 7, 8] to [0, 1, 1].
    """
    objective = [int(qubit) for qubit in objective_qubits]
    width = len(objective)
    bitstring = normalize_bitstring(prepared_bitstring, width)
    circuit = QuantumCircuit(int(num_qubits), name=f"objective_prepare_{bitstring}")
    for local_index, qubit in enumerate(objective):
        if bitstring[-1 - local_index] == "1":
            circuit.x(int(qubit))
    classical = ClassicalRegister(width, "c0")
    circuit.add_register(classical)
    circuit.measure(objective, classical[:])
    circuit.metadata = {
        "source": "objective_readout_assignment",
        "prepared_bitstring": bitstring,
        "objective_qubits": objective,
    }
    return circuit


def observed_probability(counts: Mapping[str, int], observed_bitstring: str, width: int) -> float:
    total = int(sum(int(value) for value in counts.values()))
    if total <= 0:
        return 0.0
    target = normalize_bitstring(observed_bitstring, width)
    good = sum(
        int(value)
        for key, value in counts.items()
        if normalize_bitstring(key, width) == target
    )
    return float(good / total)


def probability_se(probability: float, shots: int) -> float:
    if int(shots) <= 0:
        return 0.0
    p = float(np.clip(probability, 0.0, 1.0))
    return float(np.sqrt(p * (1.0 - p) / int(shots)))


def ideal_objective_distribution(bundle: Any) -> dict[str, float]:
    width = int(bundle.objective_width)
    probabilities = Statevector.from_instruction(
        bundle.problem.state_preparation
    ).probabilities_dict(qargs=list(bundle.problem.objective_qubits))
    distribution = {bitstring: 0.0 for bitstring in all_bitstrings(width)}
    for bitstring, probability in probabilities.items():
        distribution[normalize_bitstring(bitstring, width)] += float(probability)
    return distribution


def save_artifacts(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    session_details: Mapping[str, Any],
    backend_info: Mapping[str, Any],
    job_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    matrix_rows: Sequence[Mapping[str, Any]],
    prepared_rows: Sequence[Mapping[str, Any]],
    k0_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    save_json(config, run_dir / "config.json")
    save_json(session_details, run_dir / "session_details.json")
    save_json(backend_info, run_dir / "backend_snapshot.json")
    save_json(summary, run_dir / "calibration_summary.json")
    save_csv(job_rows, run_dir / "runtime_jobs.csv")
    save_csv(raw_rows, run_dir / "objective_assignment_raw_counts.csv")
    save_csv(matrix_rows, run_dir / "objective_assignment_matrix.csv")
    save_csv(prepared_rows, run_dir / "objective_assignment_summary.csv")
    if k0_rows:
        save_csv(k0_rows, run_dir / "k0_probe_counts.csv")
    save_json(
        {
            "run_dir": str(run_dir),
            "config_json": str(run_dir / "config.json"),
            "backend_snapshot_json": str(run_dir / "backend_snapshot.json"),
            "session_details_json": str(run_dir / "session_details.json"),
            "runtime_jobs_csv": str(run_dir / "runtime_jobs.csv"),
            "objective_assignment_raw_counts_csv": str(
                run_dir / "objective_assignment_raw_counts.csv"
            ),
            "objective_assignment_matrix_csv": str(
                run_dir / "objective_assignment_matrix.csv"
            ),
            "objective_assignment_summary_csv": str(
                run_dir / "objective_assignment_summary.csv"
            ),
            "k0_probe_counts_csv": str(run_dir / "k0_probe_counts.csv") if k0_rows else "",
            "calibration_summary_json": str(run_dir / "calibration_summary.json"),
        },
        run_dir / "manifest.json",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.shots) <= 0:
        raise ValueError("--shots must be positive.")
    if int(args.repeats) <= 0:
        raise ValueError("--repeats must be positive.")

    if args.verbose:
        print("[objective-calibration] loading 6q CVA config and bundle", flush=True)
    config_obj = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config_obj)
    bundle = build_6q_cva_problem_bundle(config_obj, repo_root=repo_root)
    objective = [int(qubit) for qubit in bundle.problem.objective_qubits]
    if objective != [6, 7, 8] or str(bundle.good_bitstring) != "111":
        raise ValueError(
            "This calibration is specialized to the 6q CVA objective register "
            f"[6, 7, 8] and good_bitstring='111'; got {objective}, "
            f"{bundle.good_bitstring!r}."
        )

    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir is not None
        else CURRENT_DIR / "runs" / f"{DEFAULT_RUN_DIR_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(
            f"[objective-calibration] loading backend={args.backend_name} "
            f"instance={args.instance_name}",
            flush=True,
        )
    backend, backend_mode = _load_backend_for_mode(args, "hardware")
    backend_info = backend_snapshot(backend, mode=backend_mode, channel=str(args.channel))
    backend_info.update(
        {
            "hardware_executor": "qctrl",
            "instance_name": str(args.instance_name),
            "backend_name_requested": str(args.backend_name),
            "qiskit_function_name": str(args.qiskit_function_name),
            "qiskit_function_channel": str(args.qiskit_function_channel or ""),
            "qctrl_submission_strategy": "abstract_logical_circuits",
            "qctrl_transpilation_policy": "fire_opal_managed",
            "local_pass_manager_applied": False,
            "calibration_type": "objective_assignment_8x8",
        }
    )

    width = int(bundle.objective_width)
    bitstrings = all_bitstrings(width)
    num_qubits = max(
        int(bundle.problem.state_preparation.num_qubits),
        int(bundle.problem.grover_operator.num_qubits),
    )
    rng = np.random.default_rng(int(args.seed))
    assignment_schedule = [
        (bitstring, repeat_index)
        for bitstring in bitstrings
        for repeat_index in range(int(args.repeats))
    ]
    rng.shuffle(assignment_schedule)

    assignment_circuits = [
        objective_assignment_circuit(
            num_qubits=num_qubits,
            objective_qubits=objective,
            prepared_bitstring=bitstring,
        )
        for bitstring, _ in assignment_schedule
    ]
    k0_circuits: list[QuantumCircuit] = []
    if bool(args.include_k0_probe):
        if args.verbose:
            print("[objective-calibration] building k=0 probe circuit", flush=True)
        k0_circuits = [
            construct_measured_circuit(bundle.problem, 0, source="k0_probe")
            for _ in range(int(args.repeats))
        ]
    circuits = assignment_circuits + k0_circuits

    job_rows: list[dict[str, Any]] = []
    session_details: dict[str, Any] = {
        "mode": "hardware",
        "created_at_epoch": time.time(),
        "hardware_executor": "qctrl",
        "instance_name": str(args.instance_name),
        "backend_name": str(args.backend_name),
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "submitted_circuit_kind": "abstract_logical",
        "qctrl_transpilation_policy": "fire_opal_managed",
        "runtime_session_strategy": "existing_qiskit_runtime_session",
        "calibration_submission": "batched_pubs_single_job",
        "shots_per_pub": int(args.shots),
        "assignment_repeats": int(args.repeats),
        "include_k0_probe": bool(args.include_k0_probe),
        "objective_qubits": objective,
        "bitstring_convention": (
            "rightmost character corresponds to objective_qubits[0]"
        ),
    }

    if args.verbose:
        print(
            f"[objective-calibration] submitting {len(circuits)} PUBs to "
            f"{args.qiskit_function_name} on {args.backend_name}",
            flush=True,
        )

    global_start = time.perf_counter()
    from qiskit_ibm_runtime import Session

    if args.verbose:
        print("[objective-calibration] opening IBM Runtime session", flush=True)
    with Session(backend=backend, max_time=args.session_max_time) as session:
        session_id = getattr(session, "session_id", None)
        if not session_id:
            raise RuntimeError("Opened Runtime Session did not expose a session_id.")
        session_details["session_id"] = session_id
        session_details["session_started_at_epoch"] = time.time()
        if args.verbose:
            print(
                f"[objective-calibration] session_id={session_id}; "
                "loading Q-CTRL function",
                flush=True,
            )
        sampler = QctrlPerformanceManagementSampler(
            instance_name=str(args.instance_name),
            backend_name=str(args.backend_name),
            pass_manager=None,
            job_rows=job_rows,
            soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
            max_grover_power=0,
            max_calls_by_context={},
            start_time=global_start,
            function_name=str(args.qiskit_function_name),
            catalog_channel=str(args.qiskit_function_channel or "") or None,
            session_id=session_id,
            verbose=bool(args.verbose),
        )
        sampler.set_context("objective_assignment_calibration")
        if args.verbose:
            print("[objective-calibration] submitting batched calibration job", flush=True)
        result = sampler.run(circuits, shots=int(args.shots)).result()
        session_details["session_finished_at_epoch"] = time.time()

    if args.verbose:
        print("[objective-calibration] processing returned counts", flush=True)
    raw_rows: list[dict[str, Any]] = []
    matrix_counts: dict[str, Counter[str]] = defaultdict(Counter)
    matrix_shots: Counter[str] = Counter()
    for batch_index, (prepared, repeat_index) in enumerate(assignment_schedule):
        counts = extract_result_counts(result, batch_index)
        total = int(sum(counts.values()))
        normalized_counts = {
            normalize_bitstring(key, width): int(value) for key, value in counts.items()
        }
        matrix_shots[prepared] += total
        for observed, count in normalized_counts.items():
            matrix_counts[prepared][observed] += int(count)
        raw_rows.append(
            {
                "batch_index": int(batch_index),
                "repeat_index": int(repeat_index),
                "prepared_bitstring": prepared,
                "shots": total,
                "p_observed_111": observed_probability(normalized_counts, "111", width),
                "p_observed_110": observed_probability(normalized_counts, "110", width),
                "counts_json": json.dumps(normalized_counts, sort_keys=True),
            }
        )

    k0_rows: list[dict[str, Any]] = []
    k0_start = len(assignment_schedule)
    if k0_circuits:
        p_ideal_k0 = ideal_good_probability_for_circuit(k0_circuits[0], bundle)
        for repeat_index in range(len(k0_circuits)):
            counts = extract_result_counts(result, k0_start + repeat_index)
            normalized_counts = {
                normalize_bitstring(key, width): int(value) for key, value in counts.items()
            }
            total = int(sum(normalized_counts.values()))
            p111 = observed_probability(normalized_counts, "111", width)
            k0_rows.append(
                {
                    "batch_index": int(k0_start + repeat_index),
                    "repeat_index": int(repeat_index),
                    "grover_power": 0,
                    "amplification_factor": 1,
                    "shots": total,
                    "p_ideal_111": float(p_ideal_k0),
                    "p_observed_111": p111,
                    "p_observed_111_se": probability_se(p111, total),
                    "counts_json": json.dumps(normalized_counts, sort_keys=True),
                }
            )

    matrix_rows: list[dict[str, Any]] = []
    prepared_rows: list[dict[str, Any]] = []
    for prepared in bitstrings:
        total = int(matrix_shots[prepared])
        top_observed = ""
        top_count = 0
        for observed in bitstrings:
            count = int(matrix_counts[prepared][observed])
            probability = float(count / max(total, 1))
            if count > top_count:
                top_observed = observed
                top_count = count
            matrix_rows.append(
                {
                    "prepared_bitstring": prepared,
                    "observed_bitstring": observed,
                    "shots": total,
                    "counts": count,
                    "probability": probability,
                    "probability_se": probability_se(probability, total),
                }
            )
        p111 = float(matrix_counts[prepared]["111"] / max(total, 1))
        p110 = float(matrix_counts[prepared]["110"] / max(total, 1))
        prepared_rows.append(
            {
                "prepared_bitstring": prepared,
                "shots": total,
                "top_observed_bitstring": top_observed,
                "top_observed_probability": float(top_count / max(total, 1)),
                "p_observed_111": p111,
                "p_observed_111_se": probability_se(p111, total),
                "p_observed_110": p110,
                "p_observed_110_se": probability_se(p110, total),
                "p_correct": float(matrix_counts[prepared][prepared] / max(total, 1)),
                "p_correct_se": probability_se(
                    float(matrix_counts[prepared][prepared] / max(total, 1)),
                    total,
                ),
            }
        )

    p_110_to_111 = float(matrix_counts["110"]["111"] / max(matrix_shots["110"], 1))
    p_110_to_111_se = probability_se(p_110_to_111, int(matrix_shots["110"]))
    p_111_to_111 = float(matrix_counts["111"]["111"] / max(matrix_shots["111"], 1))
    p_000_to_111 = float(matrix_counts["000"]["111"] / max(matrix_shots["000"], 1))

    ideal_k0_distribution = ideal_objective_distribution(bundle)
    predicted_k0_p111_from_110_leakage = float(
        ideal_k0_distribution["111"] * p_111_to_111
        + ideal_k0_distribution["110"] * p_110_to_111
    )
    summary = {
        "run_id": run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "calibration_type": "objective_assignment_8x8",
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "hardware_executor": "qctrl",
        "qiskit_function_name": str(args.qiskit_function_name),
        "shots_per_pub": int(args.shots),
        "repeats": int(args.repeats),
        "objective_qubits": objective,
        "good_bitstring": str(bundle.good_bitstring),
        "bitstring_convention": (
            "rightmost character corresponds to objective_qubits[0]"
        ),
        "p_110_to_111": p_110_to_111,
        "p_110_to_111_se": p_110_to_111_se,
        "p_111_to_111": p_111_to_111,
        "p_000_to_111": p_000_to_111,
        "ideal_k0_p111": float(ideal_k0_distribution["111"]),
        "ideal_k0_p110": ideal_k0_distribution["110"],
        "ideal_k0_distribution_json": json.dumps(
            ideal_k0_distribution,
            sort_keys=True,
        ),
        "predicted_k0_p111_from_110_leakage": predicted_k0_p111_from_110_leakage,
        "k0_probe_mean_p111": (
            float(np.mean([row["p_observed_111"] for row in k0_rows]))
            if k0_rows
            else None
        ),
    }
    config_payload = {
        "run_id": run_dir.name,
        "pipeline": "6q_cva_objective_readout_calibration",
        "backend": str(args.backend_name),
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "channel": str(args.channel),
        "hardware_executor": "qctrl",
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "shots": int(args.shots),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "include_k0_probe": bool(args.include_k0_probe),
        "submitted_circuit_kind": "abstract_logical",
        "local_pass_manager_applied": False,
        "objective_qubits": objective,
        "good_bitstring": str(bundle.good_bitstring),
        "true_amplitude": float(bundle.true_amplitude),
    }
    save_artifacts(
        run_dir=run_dir,
        args=args,
        config=config_payload,
        session_details=session_details,
        backend_info=backend_info,
        job_rows=job_rows,
        raw_rows=raw_rows,
        matrix_rows=matrix_rows,
        prepared_rows=prepared_rows,
        k0_rows=k0_rows,
        summary=summary,
    )

    print(f"Run directory: {run_dir}")
    print(f"P(observed 111 | prepared 110): {p_110_to_111:.8f} +/- {p_110_to_111_se:.8f}")
    print(f"P(observed 111 | prepared 111): {p_111_to_111:.8f}")
    print(f"P(observed 111 | prepared 000): {p_000_to_111:.8f}")
    if k0_rows:
        print(f"k=0 probe mean P(111): {summary['k0_probe_mean_p111']:.8f}")
    print(
        "Predicted k=0 P(111) from ideal 110/111 mass and assignment matrix: "
        f"{predicted_k0_p111_from_110_leakage:.8f}"
    )


if __name__ == "__main__":
    main()
