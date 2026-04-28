from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, qasm3
from qiskit_ibm_runtime import QiskitRuntimeService

from ae_pipeline_utils import (
    CANONICAL_OBJECTIVE_RY_OFFSET,
    PHYSICAL_BACKEND_NAME,
    TRANSPILER_OPTIMIZATION_LEVEL,
    TRANSPILER_SEED,
    build_ae_pass_manager,
    build_large_problem,
    construct_measured_circuit,
    ideal_good_probability,
)


DEFAULT_OBJECTIVE_RY_OFFSET = CANONICAL_OBJECTIVE_RY_OFFSET
DEFAULT_GROVER_POWERS = tuple(range(1, 10))
DEFAULT_REFERENCE_KS = (0, 1, 2)
DEFAULT_CHANNELS = ("ibm_quantum_platform", "ibm_cloud")

OUTPUT_DIR = Path(__file__).resolve().parent / "circuit_visualizations"


def _offset_slug(value: float) -> str:
    prefix = "minus_" if value < 0 else "plus_"
    return prefix + f"{abs(float(value)):.2f}".replace(".", "p")


def _parse_int_list(value: str) -> tuple[int, ...]:
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return tuple(items)


def _two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    return int(sum(1 for instruction in circuit.data if len(instruction.qubits) == 2))


def _count_ops(circuit: QuantumCircuit, op_names: Iterable[str]) -> dict[str, int]:
    ops = circuit.count_ops()
    return {name: int(ops.get(name, 0)) for name in op_names}


def _duration_seconds(circuit: QuantumCircuit, backend: Any) -> float | None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*QuantumCircuit.duration.*",
            category=DeprecationWarning,
        )
        duration = getattr(circuit, "duration", None)
    if duration is None:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*QuantumCircuit.unit.*",
            category=DeprecationWarning,
        )
        unit = getattr(circuit, "unit", None)
    if unit == "s":
        return float(duration)
    if unit == "dt":
        dt = getattr(backend, "dt", None)
        if dt is not None:
            return float(duration) * float(dt)
    return None


def _load_real_backend(backend_name: str, channels: Iterable[str]) -> tuple[Any, str]:
    errors: list[str] = []
    for channel in channels:
        try:
            service = QiskitRuntimeService(channel=channel)
            return service.backend(backend_name), channel
        except Exception as exc:
            errors.append(f"{channel}: {type(exc).__name__}: {exc}")

    try:
        service = QiskitRuntimeService()
        return service.backend(backend_name), "default"
    except Exception as exc:
        errors.append(f"default: {type(exc).__name__}: {exc}")

    joined_errors = "\n  - ".join(errors)
    raise RuntimeError(
        f"Could not load backend {backend_name!r}. Tried:\n  - {joined_errors}"
    )


def _load_fake_backend(fake_backend: str) -> tuple[Any, str]:
    from qiskit_ibm_runtime import fake_provider

    class_name = "".join(part.capitalize() for part in fake_backend.split("_"))
    if not class_name.startswith("Fake"):
        class_name = f"Fake{class_name.removeprefix('Fake')}"
    if not hasattr(fake_provider, class_name):
        raise ValueError(
            f"Unknown fake backend {fake_backend!r}; expected a class like {class_name} "
            "in qiskit_ibm_runtime.fake_provider."
        )
    return getattr(fake_provider, class_name)(), f"fake:{class_name}"


def load_backend(
    backend_name: str,
    *,
    channels: Iterable[str],
    fake_backend: str | None,
) -> tuple[Any, str]:
    if fake_backend:
        return _load_fake_backend(fake_backend)
    return _load_real_backend(backend_name, channels)


def build_report_rows(
    backend: Any,
    *,
    objective_ry_offset: float,
    grover_powers: Iterable[int],
    reference_ks: Iterable[int],
    optimization_level: int,
    seed_transpiler: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    problem, a_true = build_large_problem(objective_ry_offset=objective_ry_offset)

    pass_manager, transpilation_metadata = build_ae_pass_manager(
        backend,
        problem,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        reference_ks=reference_ks,
    )

    rows: list[dict[str, Any]] = []
    native_2q_names = ("cz", "ecr", "cx", "rzz", "swap")

    for grover_power in grover_powers:
        logical = construct_measured_circuit(problem, int(grover_power))
        decomposed = logical.decompose(reps=10)
        isa = pass_manager.run(decomposed)
        native_2q_counts = _count_ops(isa, native_2q_names)
        duration_seconds = _duration_seconds(isa, backend)

        rows.append(
            {
                "backend_name": getattr(backend, "name", None),
                "objective_ry_offset": float(objective_ry_offset),
                "a_true": float(a_true),
                "grover_power": int(grover_power),
                "amplification_factor": int(2 * int(grover_power) + 1),
                "p_ideal": float(ideal_good_probability(problem, int(grover_power))),
                "logical_num_qubits": int(logical.num_qubits),
                "logical_num_clbits": int(logical.num_clbits),
                "logical_depth_raw": int(logical.depth() or 0),
                "logical_size_raw": int(logical.size()),
                "logical_2q_raw": _two_qubit_gate_count(logical),
                "logical_depth_decomposed": int(decomposed.depth() or 0),
                "logical_size_decomposed": int(decomposed.size()),
                "logical_2q_decomposed": _two_qubit_gate_count(decomposed),
                "isa_num_qubits": int(isa.num_qubits),
                "isa_num_clbits": int(isa.num_clbits),
                "isa_depth": int(isa.depth() or 0),
                "isa_size": int(isa.size()),
                "isa_2q": _two_qubit_gate_count(isa),
                "isa_duration_seconds": duration_seconds,
                **{f"isa_{name}_count": count for name, count in native_2q_counts.items()},
                "isa_ops": dict(isa.count_ops()),
            }
        )

    metadata = {
        "a_true": float(a_true),
        "objective_qubits": [int(q) for q in problem.objective_qubits],
        "state_preparation_qubits": int(problem.state_preparation.num_qubits),
        "grover_operator_qubits": int(problem.grover_operator.num_qubits),
        "transpilation": transpilation_metadata,
    }
    return rows, metadata


def print_report(
    rows: list[dict[str, Any]],
    *,
    backend_source: str,
    metadata: dict[str, Any],
) -> None:
    trans = metadata["transpilation"]
    print("=" * 120)
    print("AE hardware circuit diagnostics")
    print("=" * 120)
    print(f"backend source              : {backend_source}")
    print(f"a_true                      : {metadata['a_true']:.12f}")
    print(f"objective qubits            : {metadata['objective_qubits']}")
    print(f"state-prep qubits           : {metadata['state_preparation_qubits']}")
    print(f"grover-operator qubits      : {metadata['grover_operator_qubits']}")
    print(f"transpilation strategy      : {trans.get('strategy')}")
    print(f"initial layout              : {trans.get('initial_layout')}")
    print(f"layout source               : {trans.get('candidate_source')}")
    print(f"seed_transpiler             : {trans.get('seed_transpiler')}")
    print(f"optimization_level          : {trans.get('optimization_level')}")
    print(f"routing_method              : {trans.get('routing_method')}")
    print(f"reference_ks                : {trans.get('reference_ks')}")
    print("=" * 120)
    header = (
        f"{'k':>3} {'K':>3} {'p_ideal':>10} "
        f"{'logD':>6} {'log2q':>6} "
        f"{'decD':>6} {'dec2q':>6} "
        f"{'isaD':>6} {'isa2q':>6} {'cz':>5} {'ecr':>5} "
        f"{'cx':>5} {'rzz':>5} {'swap':>5}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['grover_power']:>3} {row['amplification_factor']:>3} "
            f"{row['p_ideal']:>10.6f} "
            f"{row['logical_depth_raw']:>6} {row['logical_2q_raw']:>6} "
            f"{row['logical_depth_decomposed']:>6} "
            f"{row['logical_2q_decomposed']:>6} "
            f"{row['isa_depth']:>6} {row['isa_2q']:>6} "
            f"{row['isa_cz_count']:>5} {row['isa_ecr_count']:>5} "
            f"{row['isa_cx_count']:>5} {row['isa_rzz_count']:>5} "
            f"{row['isa_swap_count']:>5}"
        )
    print("=" * 120)


def save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_rows = []
    for row in rows:
        clean_row = dict(row)
        clean_row["isa_ops"] = ";".join(
            f"{name}:{count}" for name, count in sorted(row["isa_ops"].items())
        )
        serializable_rows.append(clean_row)

    fieldnames = list(serializable_rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable_rows)


def save_qasm(rows: list[dict[str, Any]], backend: Any, args: argparse.Namespace) -> None:
    if not args.save_qasm:
        return

    problem, _ = build_large_problem(objective_ry_offset=args.objective_ry_offset)
    pass_manager, _ = build_ae_pass_manager(
        backend,
        problem,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
        reference_ks=args.reference_ks,
    )

    out_dir = Path(args.qasm_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        k = int(row["grover_power"])
        logical = construct_measured_circuit(problem, k).decompose(reps=10)
        isa = pass_manager.run(logical)
        qasm_path = out_dir / f"ae_isa_k{k}_K{2 * k + 1}.qasm"
        qasm_path.write_text(qasm3.dumps(isa), encoding="utf-8")


def _save_circuit_image(circuit: QuantumCircuit, output_path: Path, *, idle_wires: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = circuit.draw("mpl", idle_wires=idle_wires, fold=-1)
    figure.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)


def save_canonical_reference_images(
    backend: Any,
    *,
    grover_power: int,
    objective_ry_offset: float,
    reference_ks: Iterable[int],
    optimization_level: int,
    seed_transpiler: int,
    output_dir: Path,
) -> list[Path]:
    problem, _ = build_large_problem(objective_ry_offset=objective_ry_offset)
    pass_manager, _ = build_ae_pass_manager(
        backend,
        problem,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        reference_ks=reference_ks,
    )

    state_preparation_logical = problem.state_preparation
    state_preparation_isa = pass_manager.run(state_preparation_logical.decompose(reps=10))

    num_qubits = max(
        problem.state_preparation.num_qubits,
        problem.grover_operator.num_qubits,
    )
    grover_circuit = QuantumCircuit(num_qubits, name="Q")
    grover_power_instruction = problem.grover_operator.power(int(grover_power))
    if hasattr(grover_power_instruction, "decompose"):
        grover_power_instruction = grover_power_instruction.decompose(reps=10)
    grover_circuit.compose(grover_power_instruction, inplace=True)
    grover_isa = pass_manager.run(grover_circuit.decompose(reps=10))

    paths = [
        output_dir / "canonical_A_logical.png",
        output_dir / "canonical_A_transpiled.png",
        output_dir / "canonical_Q_logical.png",
        output_dir / "canonical_Q_transpiled.png",
    ]
    _save_circuit_image(state_preparation_logical, paths[0], idle_wires=True)
    _save_circuit_image(state_preparation_isa, paths[1], idle_wires=False)
    _save_circuit_image(grover_circuit, paths[2], idle_wires=True)
    _save_circuit_image(grover_isa, paths[3], idle_wires=False)
    return paths


def draw_circuits(
    backend: Any,
    *,
    grover_power: int,
    objective_ry_offset: float,
    reference_ks: Iterable[int],
    optimization_level: int,
    seed_transpiler: int,
) -> None:
    problem, _ = build_large_problem(objective_ry_offset=objective_ry_offset)
    pass_manager, _ = build_ae_pass_manager(
        backend,
        problem,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        reference_ks=reference_ks,
    )
    logical = construct_measured_circuit(problem, grover_power)
    isa = pass_manager.run(logical.decompose(reps=10))
    logical.draw("mpl", fold=-1)
    isa.draw("mpl", idle_wires=False, fold=-1)
    plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report logical and ISA-transpiled depth/2Q metrics for the "
            "3-qubit AE circuits that would be submitted to IBM Runtime."
        )
    )
    parser.add_argument("--backend", default=PHYSICAL_BACKEND_NAME)
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help=(
            "IBM Runtime channel to try. Can be passed multiple times. "
            "Defaults to ibm_quantum_platform then ibm_cloud."
        ),
    )
    parser.add_argument(
        "--fake-backend",
        default=None,
        help="Optional fake backend class suffix, e.g. fake_fez or fake_marrakesh.",
    )
    parser.add_argument(
        "--objective-ry-offset",
        type=float,
        default=DEFAULT_OBJECTIVE_RY_OFFSET,
    )
    parser.add_argument(
        "--grover-powers",
        type=_parse_int_list,
        default=DEFAULT_GROVER_POWERS,
        help="Comma-separated Grover powers to report. Default: 1,2,...,9.",
    )
    parser.add_argument(
        "--reference-ks",
        type=_parse_int_list,
        default=DEFAULT_REFERENCE_KS,
        help="Comma-separated k values used to select the fixed layout.",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=TRANSPILER_OPTIMIZATION_LEVEL,
    )
    parser.add_argument("--seed-transpiler", type=int, default=TRANSPILER_SEED)
    parser.add_argument(
        "--output-csv",
        default=str(OUTPUT_DIR / "ae_hardware_depth_report.csv"),
    )
    parser.add_argument("--no-save-csv", action="store_true")
    parser.add_argument("--save-qasm", action="store_true")
    parser.add_argument(
        "--qasm-dir",
        default=str(OUTPUT_DIR / "qasm"),
    )
    parser.add_argument(
        "--draw-k",
        type=int,
        default=None,
        help="Optionally draw logical and ISA circuits for one k.",
    )
    parser.add_argument(
        "--save-reference-images-k",
        type=int,
        default=1,
        help=(
            "Save transpiled A and Q^k circuit images to circuit_visualizations. "
            "Use --no-save-reference-images to disable."
        ),
    )
    parser.add_argument("--no-save-reference-images", action="store_true")
    parser.add_argument(
        "--reference-image-dir",
        default=str(OUTPUT_DIR),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    channels = tuple(args.channel) if args.channel else DEFAULT_CHANNELS

    backend, backend_source = load_backend(
        args.backend,
        channels=channels,
        fake_backend=args.fake_backend,
    )

    rows, metadata = build_report_rows(
        backend,
        objective_ry_offset=args.objective_ry_offset,
        grover_powers=args.grover_powers,
        reference_ks=args.reference_ks,
        optimization_level=args.optimization_level,
        seed_transpiler=args.seed_transpiler,
    )
    print_report(rows, backend_source=backend_source, metadata=metadata)

    if not args.no_save_csv:
        output_path = Path(args.output_csv)
        save_csv(rows, output_path)
        print(f"Saved CSV report -> {output_path}")

    save_qasm(rows, backend, args)

    if not args.no_save_reference_images:
        paths = save_canonical_reference_images(
            backend,
            grover_power=int(args.save_reference_images_k),
            objective_ry_offset=args.objective_ry_offset,
            reference_ks=args.reference_ks,
            optimization_level=args.optimization_level,
            seed_transpiler=args.seed_transpiler,
            output_dir=Path(args.reference_image_dir),
        )
        for path in paths:
            print(f"Saved reference image -> {path}")

    if args.draw_k is not None:
        draw_circuits(
            backend,
            grover_power=int(args.draw_k),
            objective_ry_offset=args.objective_ry_offset,
            reference_ks=args.reference_ks,
            optimization_level=args.optimization_level,
            seed_transpiler=args.seed_transpiler,
        )


if __name__ == "__main__":
    os.environ.setdefault("QISKIT_SUPPRESS_PACKAGING_WARNINGS", "Y")
    main()
