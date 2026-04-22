from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[3]
SRC_DIR = ROOT_DIR / "src"
HARDWARE_UTILS_PATH = CURRENT_DIR / "hardware" / "realistic_utils.py"
SIMULATION_UTILS_PATH = CURRENT_DIR / "simulations" / "realistic_utils.py"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "circuit_visualizations"
DEFAULT_REFERENCE_KS = (0, 1, 2)

for path in (str(SRC_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from quantum_cva.quantum_hardware_utilities.transpile_utils import (
    stable_circuit_key,
    transpilation_metrics,
)


def _load_module(alias: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(alias, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _build_shared_artifacts(
    *,
    objective_ry_offset: float,
    k_value: int,
    reference_ks: tuple[int, ...],
) -> dict[str, Any]:
    hardware_utils = _load_module("ae_hw_realistic_utils", HARDWARE_UTILS_PATH)
    simulation_utils = _load_module("ae_sim_realistic_utils", SIMULATION_UTILS_PATH)

    hw_problem, hw_a_true = hardware_utils.build_large_problem(
        objective_ry_offset=float(objective_ry_offset)
    )
    sim_problem, sim_a_true = simulation_utils.build_large_problem(
        objective_ry_offset=float(objective_ry_offset)
    )

    if not np.isclose(hw_a_true, sim_a_true, atol=1e-12, rtol=0.0):
        raise RuntimeError(
            "Hardware and simulation builders do not agree on a_true: "
            f"{hw_a_true} != {sim_a_true}."
        )

    logical_hw = hardware_utils.construct_measured_circuit(hw_problem, int(k_value))
    logical_sim = simulation_utils.construct_measured_circuit(sim_problem, int(k_value))
    if stable_circuit_key(logical_hw) != stable_circuit_key(logical_sim):
        raise RuntimeError(
            "Hardware and simulation builders do not produce the same logical AE circuit."
        )

    reference_circuits_hw = [
        hardware_utils.construct_measured_circuit(hw_problem, int(ref_k))
        for ref_k in reference_ks
    ]
    reference_circuits_sim = [
        simulation_utils.construct_measured_circuit(sim_problem, int(ref_k))
        for ref_k in reference_ks
    ]

    for ref_k, hw_circuit, sim_circuit in zip(
        reference_ks,
        reference_circuits_hw,
        reference_circuits_sim,
        strict=True,
    ):
        if stable_circuit_key(hw_circuit) != stable_circuit_key(sim_circuit):
            raise RuntimeError(
                f"Mismatch between hardware and simulation AE circuits for k={ref_k}."
            )

    return {
        "hardware_utils": hardware_utils,
        "problem": hw_problem,
        "logical_circuit": logical_hw,
        "a_true": float(hw_a_true),
    }


def _save_circuit_image(circuit: QuantumCircuit, path: Path, *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = circuit.draw(output="mpl", fold=80, idle_wires=False)
    figure.suptitle(title, fontsize=12)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _summarize_circuit(label: str, circuit: QuantumCircuit) -> None:
    metrics = transpilation_metrics(circuit)
    count_ops = {str(name): int(count) for name, count in circuit.count_ops().items()}

    print(f"{label}:")
    print(f"  qubits            : {circuit.num_qubits}")
    print(f"  classical bits    : {circuit.num_clbits}")
    print(f"  depth             : {metrics['depth']}")
    print(f"  size              : {metrics['size']}")
    print(f"  2q gates          : {metrics['two_qubit_gates']}")
    print(f"  swap count        : {metrics['swap_count']}")
    print(f"  count_ops         : {count_ops}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, transpile, and visualize the shared 5-qubit AE circuit used by "
            "both simulations and hardware workflows."
        )
    )
    parser.add_argument("--backend-name", default="ibm_basquecountry")
    parser.add_argument("--channel", default="ibm_quantum_platform")
    parser.add_argument("--objective-ry-offset", type=float, default=-0.10)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the circuit images will be written.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    from qiskit_ibm_runtime import QiskitRuntimeService

    reference_ks = tuple(int(k) for k in DEFAULT_REFERENCE_KS)
    artifacts = _build_shared_artifacts(
        objective_ry_offset=float(args.objective_ry_offset),
        k_value=int(args.k),
        reference_ks=reference_ks,
    )

    logical_circuit = artifacts["logical_circuit"]
    hardware_utils = artifacts["hardware_utils"]
    problem = artifacts["problem"]
    a_true = float(artifacts["a_true"])

    print("=" * 100)
    print("AE CIRCUIT VISUALIZATION")
    print("=" * 100)
    print(f"Backend name              : {args.backend_name}")
    print(f"Channel                   : {args.channel}")
    print(f"Objective RY offset       : {float(args.objective_ry_offset):+.3f}")
    print(f"k value                   : {int(args.k)}")
    print(f"Reference ks for layout   : {list(reference_ks)}")
    print(f"Optimization level        : {int(args.optimization_level)}")
    print(f"a_true                    : {a_true:.6f}")
    print("=" * 100)

    service = QiskitRuntimeService(channel=str(args.channel))
    backend = service.backend(
        str(args.backend_name),
        use_fractional_gates=True
        )
    transpile_pm, transpilation_metadata = hardware_utils.build_ae_pass_manager(
        backend,
        problem,
        optimization_level=int(args.optimization_level),
        reference_ks=reference_ks,
    )
    transpiled_circuit = transpile_pm.run(logical_circuit)

    _summarize_circuit("Logical circuit", logical_circuit)
    print()
    _summarize_circuit("Transpiled circuit", transpiled_circuit)
    print()
    print(f"Selected initial layout   : {transpilation_metadata.get('initial_layout')}")
    print(f"Selected transpiler seed  : {int(transpilation_metadata['seed_transpiler'])}")
    print(f"Candidate source          : {transpilation_metadata['candidate_source']}")
    print(f"Fallback used             : {bool(transpilation_metadata['fallback_used'])}")

    output_dir = Path(args.output_dir).resolve()
    logical_png = output_dir / f"ae_logical_k{int(args.k)}_{args.backend_name}.png"
    transpiled_png = output_dir / f"ae_transpiled_k{int(args.k)}_{args.backend_name}.png"

    _save_circuit_image(
        logical_circuit,
        logical_png,
        title=(
            f"Logical AE circuit | k={int(args.k)} | "
            f"offset={float(args.objective_ry_offset):+.3f}"
        ),
    )
    _save_circuit_image(
        transpiled_circuit,
        transpiled_png,
        title=(
            f"Transpiled AE circuit | backend={args.backend_name} | "
            f"layout={transpilation_metadata.get('initial_layout')}"
        ),
    )

    print(f"Logical image saved to    : {logical_png}")
    print(f"Transpiled image saved to : {transpiled_png}")

    qc = hardware_utils.construct_measured_circuit(problem, 1)
    qc_dec = qc
    for _ in range(6):
        qc_dec = qc_dec.decompose()

    print(qc.count_ops())
    print(qc_dec.count_ops())
    print(qc_dec.depth(), qc_dec.size())


if __name__ == "__main__":
    main()
