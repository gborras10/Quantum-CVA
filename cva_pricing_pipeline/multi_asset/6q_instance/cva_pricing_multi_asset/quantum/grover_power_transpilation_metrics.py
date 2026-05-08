from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_algorithms import EstimationProblem
from qiskit_ibm_runtime import QiskitRuntimeService


def _bootstrap_paths() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    script_dir = current.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    instance_dir = current.parents[2]
    if str(instance_dir) not in sys.path:
        sys.path.insert(0, str(instance_dir))

    return repo_root


REPO_ROOT = _bootstrap_paths()

from full_cva_pipeline import CONFIG  # noqa: E402
from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (  # noqa: E402
    QuantumCVACircuit,
)
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (  # noqa: E402
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (  # noqa: E402
    MLQcbmCircuit,
)
from utils_run_ideal import (  # noqa: E402
    _as_1d_float,
    _assert_param_size,
    _build_improved_cva_initial_layout,
    _metadata_dict,
    _npz_int,
    _npz_str,
    _select_layout_for_training,
)


DEFAULT_GROVER_POWERS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_SABRE_SEEDS: tuple[int, ...] = (
    1,
    7,
    8, 
    17,
    18,
    22,
    42,
    73,
    101,
    202,
    404,
    777,
    1234,
)


def _resolve(path_like: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_npz(path_like: str | pathlib.Path) -> np.lib.npyio.NpzFile:
    path = _resolve(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    return np.load(path, allow_pickle=True)


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("Grover powers must be non-negative integers.")
        values.append(value)
    if not values:
        raise ValueError("At least one Grover power is required.")
    return values


def _count_by_qubit_arity(circuit: QuantumCircuit) -> tuple[int, int, int]:
    one_qubit = 0
    two_qubit = 0
    multi_qubit = 0
    for instruction in circuit.data:
        arity = int(instruction.operation.num_qubits)
        if arity == 1:
            one_qubit += 1
        elif arity == 2:
            two_qubit += 1
        elif arity > 2:
            multi_qubit += 1
    return one_qubit, two_qubit, multi_qubit


def _circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    one_qubit, two_qubit, multi_qubit = _count_by_qubit_arity(circuit)
    ops = {str(name): int(count) for name, count in circuit.count_ops().items()}
    return {
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
        "width": int(circuit.width()),
        "one_qubit_gates": int(one_qubit),
        "two_qubit_gates": int(two_qubit),
        "multi_qubit_gates": int(multi_qubit),
        "swap_count": int(ops.get("swap", 0)),
        "ops": ops,
    }


def _active_qubits(circuit: QuantumCircuit) -> list[int]:
    return sorted(
        {
            int(circuit.find_bit(qubit).index)
            for instruction in circuit.data
            for qubit in instruction.qubits
        }
    )


def _initial_layout_from_circuit(
    circuit: QuantumCircuit,
    *,
    logical_qubits: int,
) -> list[int] | None:
    layout = getattr(circuit, "layout", None)
    if layout is None or not hasattr(layout, "initial_index_layout"):
        return None
    try:
        indices = list(layout.initial_index_layout())
    except Exception:
        return None
    if len(indices) < int(logical_qubits):
        return None
    return [int(idx) for idx in indices[: int(logical_qubits)]]


def _score(metrics: dict[str, Any], seed: int) -> tuple[int, int, int, int, int]:
    return (
        int(metrics["depth"]),
        int(metrics["two_qubit_gates"]),
        int(metrics["size"]),
        int(metrics["swap_count"]),
        int(seed),
    )


def _decompose_repeated(circuit: QuantumCircuit, reps: int) -> QuantumCircuit:
    out = circuit
    for _ in range(max(0, int(reps))):
        out = out.decompose()
    return out


def _build_grover_query_circuit(
    problem: EstimationProblem,
    *,
    power: int,
) -> QuantumCircuit:
    power_i = int(power)
    if power_i < 0:
        raise ValueError("Grover power must be non-negative.")

    circuit = problem.state_preparation.copy()
    circuit.name = f"cva_A_Q_power_{power_i}"

    # Qiskit's EstimationProblem builds Q from A and the objective qubits.
    # The global minus sign in Q = -A S0 A^dagger Sf is depth-neutral.
    if power_i > 0:
        circuit.compose(problem.grover_operator.power(power_i), inplace=True)
    return circuit


def _build_cva_estimation_problem() -> tuple[EstimationProblem, QuantumCircuit, dict[str, Any]]:
    cfg = CONFIG

    benchmark = _load_npz(cfg.paths.benchmark_relative_path)
    qcbm_data = _load_npz(cfg.paths.qcbm_training_relative_path)
    default_data = _load_npz(cfg.paths.crca_default_training_relative_path)
    discount_data = _load_npz(cfg.paths.crca_discount_training_relative_path)
    exposure_data = _load_npz(cfg.paths.crca_exposure_training_relative_path)

    qcbm_theta = _as_1d_float(qcbm_data["theta_star"])
    default_theta = _as_1d_float(default_data["theta_star"])
    discount_theta = _as_1d_float(discount_data["theta_star"])
    exposure_theta = _as_1d_float(exposure_data["theta_star"])

    num_qubits_time = int(cfg.classical.m_time)
    num_qubits_underlying = int(cfg.quantum.n_underlying_qubits)
    total_state_qubits = num_qubits_time + num_qubits_underlying

    qcbm_topology = _npz_str(qcbm_data, "effective_topology", cfg.qcbm_training.topology)
    qcbm_n_layers = _npz_int(qcbm_data, "n_layers", cfg.qcbm_training.n_layers)
    qcbm = MLQcbmCircuit(
        n_qubits=total_state_qubits,
        n_layers=qcbm_n_layers,
        name="qcbm_state_prep_circuit_shots_noise_theta",
        entangler=cfg.qcbm_training.entangler,
        topology=qcbm_topology,
        backend=AerSimulator(method="statevector"),
        simulation_method="statevector",
        optimization_level=0,
    )

    exposure_meta = _metadata_dict(exposure_data)
    exposure_n_layers = int(
        exposure_meta.get(
            "n_layers",
            _npz_int(exposure_data, "n_layers", cfg.crca_exposure_training.n_layers),
        )
    )
    exposure_ansatz = _npz_str(
        exposure_data,
        "effective_topology",
        cfg.crca_exposure_training.topology,
    )
    crca_exposure = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=num_qubits_underlying,
        n_layers=exposure_n_layers,
        ansatz_type=exposure_ansatz,
        name="crca_positive_exposure_circuit_shots_noise_theta",
    )
    crca_default = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(default_data, "n_layers", cfg.crca_default_training.n_layers),
        ansatz_type=cfg.crca_default_training.ansatz_type,
        native_1q_order=cfg.crca_default_training.native_1q_order,
        name="crca_default_probabilities_circuit_shots_noise_theta",
    )
    crca_discount = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(
            discount_data,
            "n_layers",
            cfg.crca_discount_training.n_layers,
        ),
        ansatz_type=cfg.crca_discount_training.ansatz_type,
        native_1q_order=cfg.crca_discount_training.native_1q_order,
        name="crca_discount_factors_circuit_shots_noise_theta",
    )

    _assert_param_size("QCBM", qcbm_theta, qcbm.n_params)
    _assert_param_size("CRCA exposure", exposure_theta, crca_exposure.n_params)
    _assert_param_size("CRCA default", default_theta, crca_default.n_params)
    _assert_param_size("CRCA discount", discount_theta, crca_discount.n_params)

    quantum_cva_circuit = QuantumCVACircuit(
        num_qubits_time=num_qubits_time,
        num_qubits_underlying=num_qubits_underlying,
        qcbm_circuit=qcbm,
        crca_circuit_exposure=crca_exposure,
        crca_circuit_default_prob=crca_default,
        crca_circuit_discount_factor=crca_discount,
        recovery_rate=float(benchmark["R_cva"]),
        C_v=float(benchmark["C_v"]),
        C_p=float(benchmark["C_p"]),
        C_q=float(benchmark["C_q"]),
        name="quantum_cva_circuit_shots_noise_theta",
        backend=cfg.final_cva.statevector_backend_name,
    )

    state_preparation = quantum_cva_circuit.build_cva_circuit(
        qcbm_params=qcbm_theta,
        crca_exposure_params=exposure_theta,
        crca_default_params=default_theta,
        crca_discount_params=discount_theta,
        measured=False,
    )

    objective_qubits = [
        int(total_state_qubits),
        int(total_state_qubits + 1),
        int(total_state_qubits + 2),
    ]
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=objective_qubits,
        is_good_state=lambda bitstr: bitstr == "111",
        post_processing=quantum_cva_circuit.cva_from_prob,
    )
    problem.grover_operator = problem.grover_operator

    metadata = {
        "num_qubits_time": num_qubits_time,
        "num_qubits_underlying": num_qubits_underlying,
        "total_state_qubits": total_state_qubits,
        "objective_qubits": objective_qubits,
        "qcbm_topology": qcbm_topology,
        "qcbm_n_layers": qcbm_n_layers,
        "exposure_ansatz": exposure_ansatz,
        "exposure_n_layers": exposure_n_layers,
    }
    return problem, state_preparation, metadata


def _load_real_backend() -> Any:
    cfg = CONFIG.backend_noise
    service = QiskitRuntimeService(channel=cfg.runtime_channel)
    return service.backend(
        cfg.backend_name,
        use_fractional_gates=bool(cfg.use_fractional_gates),
    )


def _build_improved_fixed_layout(real_backend: Any) -> list[int]:
    cfg = CONFIG

    qcbm_data = _load_npz(cfg.paths.qcbm_training_relative_path)
    default_data = _load_npz(cfg.paths.crca_default_training_relative_path)
    discount_data = _load_npz(cfg.paths.crca_discount_training_relative_path)
    exposure_data = _load_npz(cfg.paths.crca_exposure_training_relative_path)

    qcbm_topology = _npz_str(qcbm_data, "effective_topology", cfg.qcbm_training.topology)
    default_topology = _npz_str(
        default_data,
        "effective_topology",
        cfg.crca_default_training.topology,
    )
    discount_topology = _npz_str(
        discount_data,
        "effective_topology",
        cfg.crca_discount_training.topology,
    )
    exposure_topology = _npz_str(
        exposure_data,
        "effective_topology",
        cfg.crca_exposure_training.topology,
    )

    total_state_qubits = int(cfg.classical.m_time + cfg.quantum.n_underlying_qubits)
    exposure_layout_length = int(total_state_qubits + 1)

    qcbm_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology=qcbm_topology,
        length=total_state_qubits,
        readout_quantile=cfg.backend_noise.readout_quantile,
        local_2q_quantile=cfg.backend_noise.local_2q_quantile,
        relax_if_needed=cfg.backend_noise.relax_if_needed,
    )
    default_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology=default_topology,
        length=3,
        readout_quantile=cfg.backend_noise.readout_quantile,
        local_2q_quantile=cfg.backend_noise.local_2q_quantile,
        relax_if_needed=cfg.backend_noise.relax_if_needed,
    )
    discount_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology=discount_topology,
        length=3,
        readout_quantile=cfg.backend_noise.readout_quantile,
        local_2q_quantile=cfg.backend_noise.local_2q_quantile,
        relax_if_needed=cfg.backend_noise.relax_if_needed,
    )
    exposure_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology=exposure_topology,
        length=exposure_layout_length,
        readout_quantile=cfg.backend_noise.readout_quantile,
        local_2q_quantile=cfg.backend_noise.local_2q_quantile,
        relax_if_needed=cfg.backend_noise.relax_if_needed,
    )

    cva_initial_layout, *_ = _build_improved_cva_initial_layout(
        real_backend=real_backend,
        qcbm_layout=qcbm_layout,
        positive_exposure_layout=exposure_layout,
        default_layout=default_layout,
        discount_layout=discount_layout,
    )
    return [int(qubit) for qubit in cva_initial_layout]


def _transpile_fixed_layout(
    circuit: QuantumCircuit,
    *,
    real_backend: Any,
    initial_layout: Sequence[int],
) -> tuple[QuantumCircuit, dict[str, Any]]:
    cfg = CONFIG.backend_noise
    pm = generate_preset_pass_manager(
        backend=real_backend,
        optimization_level=int(cfg.transpilation_opt_level),
        initial_layout=[int(q) for q in initial_layout],
        seed_transpiler=int(cfg.seed_transpiler),
        approximation_degree=float(cfg.approximation_degree),
    )
    transpiled = pm.run(circuit)
    metrics = _circuit_metrics(transpiled)
    return transpiled, {
        "method": "fixed_improved_cva_layout",
        "seed": int(cfg.seed_transpiler),
        "initial_layout": [int(q) for q in initial_layout],
        "metrics": metrics,
    }


def _transpile_sabre_multiseed(
    circuit: QuantumCircuit,
    *,
    real_backend: Any,
    seeds: Sequence[int],
) -> tuple[QuantumCircuit, dict[str, Any]]:
    cfg = CONFIG.backend_noise
    best_circuit: QuantumCircuit | None = None
    best_info: dict[str, Any] | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    all_metrics: list[dict[str, Any]] = []

    for seed in seeds:
        seed_i = int(seed)
        pm = generate_preset_pass_manager(
            backend=real_backend,
            optimization_level=int(cfg.transpilation_opt_level),
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=seed_i,
            approximation_degree=float(cfg.approximation_degree),
        )
        transpiled = pm.run(circuit)
        metrics = _circuit_metrics(transpiled)
        metrics_with_seed = {"seed": seed_i, **metrics}
        all_metrics.append(metrics_with_seed)
        score = _score(metrics, seed_i)
        if best_score is None or score < best_score:
            best_score = score
            best_circuit = transpiled
            best_info = {
                "method": "sabre_multiseed",
                "seed": seed_i,
                "initial_layout": _initial_layout_from_circuit(
                    transpiled,
                    logical_qubits=circuit.num_qubits,
                ),
                "metrics": metrics,
                "candidate_metrics": all_metrics,
            }

    if best_circuit is None or best_info is None:
        raise RuntimeError("SABRE multi-seed transpilation produced no circuit.")
    best_info["candidate_metrics"] = all_metrics
    return best_circuit, best_info


def _select_best_transpilation(
    circuit: QuantumCircuit,
    *,
    real_backend: Any,
    fixed_layout: Sequence[int],
    sabre_seeds: Sequence[int],
    mode: str,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    candidates: list[tuple[QuantumCircuit, dict[str, Any]]] = []

    if mode in {"best", "fixed"}:
        candidates.append(
            _transpile_fixed_layout(
                circuit,
                real_backend=real_backend,
                initial_layout=fixed_layout,
            )
        )
    if mode in {"best", "sabre"}:
        candidates.append(
            _transpile_sabre_multiseed(
                circuit,
                real_backend=real_backend,
                seeds=sabre_seeds,
            )
        )

    best_circuit: QuantumCircuit | None = None
    best_info: dict[str, Any] | None = None
    best_score: tuple[int, int, int, int, int] | None = None
    for transpiled, info in candidates:
        seed = int(info["seed"])
        score = _score(info["metrics"], seed)
        if best_score is None or score < best_score:
            best_score = score
            best_circuit = transpiled
            best_info = info

    if best_circuit is None or best_info is None:
        raise RuntimeError(f"No transpilation candidates were evaluated for mode={mode!r}.")
    return best_circuit, best_info


def _row_for_power(
    *,
    power: int,
    logical_circuit: QuantumCircuit,
    transpile_input_circuit: QuantumCircuit,
    transpiled_circuit: QuantumCircuit,
    transpilation_info: dict[str, Any],
    metadata: dict[str, Any],
    logical_decompose_reps: int,
    transpile_source: str,
) -> dict[str, Any]:
    logical = _circuit_metrics(logical_circuit)
    transpile_input = _circuit_metrics(transpile_input_circuit)
    transpiled = transpilation_info["metrics"]
    initial_layout = transpilation_info.get("initial_layout")

    return {
        "grover_power_k": int(power),
        "operator": "A Q^k",
        "a_calls_in_query": int(2 * int(power) + 1),
        "s0_calls_in_Q_power": int(power),
        "sf_calls_in_Q_power": int(power),
        "num_qubits": int(logical_circuit.num_qubits),
        "objective_qubits": json.dumps(metadata["objective_qubits"]),
        "logical_decompose_reps": int(logical_decompose_reps),
        "logical_depth": logical["depth"],
        "logical_size": logical["size"],
        "logical_width": logical["width"],
        "logical_1q_gates": logical["one_qubit_gates"],
        "logical_2q_gates": logical["two_qubit_gates"],
        "logical_multi_qubit_gates": logical["multi_qubit_gates"],
        "logical_swap_count": logical["swap_count"],
        "logical_ops": json.dumps(logical["ops"], sort_keys=True),
        "transpile_source": str(transpile_source),
        "transpile_input_depth": transpile_input["depth"],
        "transpile_input_size": transpile_input["size"],
        "transpile_input_width": transpile_input["width"],
        "transpile_input_1q_gates": transpile_input["one_qubit_gates"],
        "transpile_input_2q_gates": transpile_input["two_qubit_gates"],
        "transpile_input_multi_qubit_gates": transpile_input["multi_qubit_gates"],
        "transpile_input_ops": json.dumps(transpile_input["ops"], sort_keys=True),
        "transpilation_method": transpilation_info["method"],
        "selected_seed_transpiler": int(transpilation_info["seed"]),
        "selected_initial_layout": json.dumps(initial_layout),
        "transpiled_depth": transpiled["depth"],
        "transpiled_size": transpiled["size"],
        "transpiled_width": transpiled["width"],
        "transpiled_1q_gates": transpiled["one_qubit_gates"],
        "transpiled_2q_gates": transpiled["two_qubit_gates"],
        "transpiled_multi_qubit_gates": transpiled["multi_qubit_gates"],
        "transpiled_swap_count": transpiled["swap_count"],
        "active_physical_qubits": json.dumps(_active_qubits(transpiled_circuit)),
        "transpiled_ops": json.dumps(transpiled["ops"], sort_keys=True),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute logical and backend-transpiled metrics for CVA query circuits A Q^k."
        )
    )
    parser.add_argument(
        "--grover-powers",
        default=",".join(str(k) for k in DEFAULT_GROVER_POWERS),
        help="Comma/space separated non-negative Grover powers, e.g. '0,1,2,4,8'.",
    )
    parser.add_argument(
        "--sabre-seeds",
        default=",".join(str(seed) for seed in DEFAULT_SABRE_SEEDS),
        help="Comma/space separated SABRE seeds used when mode is sabre or best.",
    )
    parser.add_argument(
        "--transpilation-mode",
        choices=("best", "fixed", "sabre"),
        default="best",
        help=(
            "best compares the improved fixed CVA layout against SABRE multi-seed "
            "and keeps the lowest (depth, 2Q, size, swap, seed) circuit."
        ),
    )
    parser.add_argument(
        "--decompose-reps",
        type=int,
        default=4,
        help=(
            "Number of repeated decompose() passes for logical metrics. "
            "The transpiler input is controlled separately by --transpile-source."
        ),
    )
    parser.add_argument(
        "--transpile-source",
        choices=("raw", "decomposed", "both"),
        default="raw",
        help=(
            "raw transpiles A Q^k without pre-decompose; decomposed reproduces the "
            "old behavior; both writes one CSV row for each strategy."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("grover_power_transpilation_metrics.csv"),
        help="CSV output path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    powers = _parse_int_list(args.grover_powers)
    sabre_seeds = _parse_int_list(args.sabre_seeds)

    print("Building CVA estimation problem and Grover operator...")
    problem, state_preparation, metadata = _build_cva_estimation_problem()
    state_preparation_for_metrics = _decompose_repeated(
        state_preparation.copy(),
        reps=int(args.decompose_reps),
    )
    raw_a_metrics = _circuit_metrics(state_preparation)
    decomposed_a_metrics = _circuit_metrics(state_preparation_for_metrics)
    print(
        "State preparation A (raw): "
        f"num_qubits={state_preparation.num_qubits}, "
        f"depth={raw_a_metrics['depth']}, "
        f"2Q={raw_a_metrics['two_qubit_gates']}, "
        f"size={raw_a_metrics['size']}"
    )
    print(
        f"State preparation A after decompose_reps={int(args.decompose_reps)} "
        "(same basis used for A Q^k logical metrics): "
        f"depth={decomposed_a_metrics['depth']}, "
        f"2Q={decomposed_a_metrics['two_qubit_gates']}, "
        f"size={decomposed_a_metrics['size']}"
    )

    print(
        "Loading backend and building improved fixed CVA layout: "
        f"{CONFIG.backend_noise.backend_name}"
    )
    real_backend = _load_real_backend()
    fixed_layout = _build_improved_fixed_layout(real_backend)
    print(f"Fixed layout candidate: {fixed_layout}")

    rows: list[dict[str, Any]] = []
    for power in powers:
        print(f"\nGrover query A Q^k with k={power}")
        raw_query_circuit = _build_grover_query_circuit(problem, power=int(power))
        logical_circuit = _decompose_repeated(
            raw_query_circuit.copy(),
            reps=int(args.decompose_reps),
        )
        logical_metrics = _circuit_metrics(logical_circuit)
        print(
            "  logical: "
            f"depth={logical_metrics['depth']}, "
            f"2Q={logical_metrics['two_qubit_gates']}, "
            f"size={logical_metrics['size']}"
        )

        transpile_sources = (
            ("raw", raw_query_circuit),
            ("decomposed", logical_circuit),
        )
        if args.transpile_source != "both":
            transpile_sources = tuple(
                item for item in transpile_sources if item[0] == args.transpile_source
            )

        for transpile_source, transpile_input_circuit in transpile_sources:
            input_metrics = _circuit_metrics(transpile_input_circuit)
            print(
                f"  transpile input ({transpile_source}): "
                f"depth={input_metrics['depth']}, "
                f"2Q={input_metrics['two_qubit_gates']}, "
                f"multiQ={input_metrics['multi_qubit_gates']}, "
                f"size={input_metrics['size']}"
            )
            transpiled_circuit, transpilation_info = _select_best_transpilation(
                transpile_input_circuit,
                real_backend=real_backend,
                fixed_layout=fixed_layout,
                sabre_seeds=sabre_seeds,
                mode=str(args.transpilation_mode),
            )
            transpiled_metrics = transpilation_info["metrics"]
            print(
                f"  transpiled ({transpile_source}): "
                f"method={transpilation_info['method']}, "
                f"seed={transpilation_info['seed']}, "
                f"depth={transpiled_metrics['depth']}, "
                f"2Q={transpiled_metrics['two_qubit_gates']}, "
                f"size={transpiled_metrics['size']}"
            )
            rows.append(
                _row_for_power(
                    power=int(power),
                    logical_circuit=logical_circuit,
                    transpile_input_circuit=transpile_input_circuit,
                    transpiled_circuit=transpiled_circuit,
                    transpilation_info=transpilation_info,
                    metadata=metadata,
                    logical_decompose_reps=int(args.decompose_reps),
                    transpile_source=str(transpile_source),
                )
            )

    output_csv = args.output_csv
    if not output_csv.is_absolute():
        output_csv = pathlib.Path.cwd() / output_csv
    _write_csv(output_csv, rows)
    print(f"\n[OK] Grover transpilation metrics written to: {output_csv}")


if __name__ == "__main__":
    main()
