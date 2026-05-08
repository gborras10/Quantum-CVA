from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_algorithms import EstimationProblem

from quantum_cva.amplitude_estimation.experiments.problems import (
    AEProblemBundle,
)
from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (
    QuantumCVACircuit,
)
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)


def _as_1d_float(array_like: Any) -> np.ndarray:
    return np.asarray(array_like, dtype=float).ravel()


def _npz_int(npz_data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in npz_data:
        return int(default)
    return int(np.asarray(npz_data[key]).item())


def _npz_str(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz_data:
        return str(default)
    return str(np.asarray(npz_data[key]).item())


def _metadata_dict(npz_data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in npz_data:
        return {}
    raw = npz_data["metadata"]
    if hasattr(raw, "item"):
        raw = raw.item()
    return dict(raw) if isinstance(raw, Mapping) else {}


def _assert_param_size(label: str, theta: np.ndarray, expected_size: int) -> None:
    actual = int(np.asarray(theta).size)
    if actual != int(expected_size):
        raise ValueError(
            f"Parameter-size mismatch for {label}: expected {expected_size}, got {actual}."
        )


def _resolve(repo_root: str | Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else Path(repo_root) / path


def _load_npz(repo_root: str | Path, path_like: str | Path) -> np.lib.npyio.NpzFile:
    path = _resolve(repo_root, path_like)
    if not path.exists():
        raise FileNotFoundError(f"Required CVA artifact does not exist: {path}")
    return np.load(path, allow_pickle=True)


def build_cva_problem_bundle(
    quantum_cva_circuit: QuantumCVACircuit,
    *,
    qcbm_params: np.ndarray,
    exposure_params: np.ndarray,
    default_prob_params: np.ndarray,
    discount_factor_params: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> AEProblemBundle:
    """Build an AE problem for the CVA ``|111>`` ancilla amplitude."""
    num_state_qubits = (
        int(quantum_cva_circuit.num_qubits_time)
        + int(quantum_cva_circuit.num_qubits_underlying)
    )
    objective_qubits = [
        num_state_qubits,
        num_state_qubits + 1,
        num_state_qubits + 2,
    ]

    state_preparation = quantum_cva_circuit.build_cva_circuit(
        qcbm_params=qcbm_params,
        crca_exposure_params=exposure_params,
        crca_default_params=default_prob_params,
        crca_discount_params=discount_factor_params,
        measured=False,
    )
    true_prob = quantum_cva_circuit.prob_111(
        qcbm_params=qcbm_params,
        crca_exposure_params=exposure_params,
        crca_default_params=default_prob_params,
        crca_discount_params=discount_factor_params,
    )
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=objective_qubits,
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "") == "111",
        post_processing=quantum_cva_circuit.cva_from_prob,
    )
    problem.grover_operator = problem.grover_operator

    bundle_metadata = {
        "target_name": "cva",
        "good_bitstring": "111",
        "num_qubits_time": int(quantum_cva_circuit.num_qubits_time),
        "num_qubits_underlying": int(quantum_cva_circuit.num_qubits_underlying),
        "total_state_qubits": int(num_state_qubits),
        "objective_qubits": objective_qubits,
        "cva_scaling": {
            "recovery_rate": float(quantum_cva_circuit.recovery_rate),
            "C_v": float(quantum_cva_circuit.C_v),
            "C_q": float(quantum_cva_circuit.C_q),
            "C_p": float(quantum_cva_circuit.C_p),
        },
    }
    if metadata:
        bundle_metadata.update(dict(metadata))

    return AEProblemBundle(
        problem=problem,
        true_amplitude=float(true_prob),
        processed_true_value=float(quantum_cva_circuit.cva_from_prob(true_prob)),
        target_name="cva",
        good_bitstring="111",
        metadata=bundle_metadata,
    )


def build_6q_cva_problem_bundle(
    config: Any,
    repo_root: str | Path,
) -> AEProblemBundle:
    """Build the real CVA AE problem for the current 6-qubit pipeline instance.

    This is the importable version of the CVA construction logic that existed
    in the 6q pipeline scripts. It deliberately does not run any experiment or
    write outputs.
    """
    cfg = config
    benchmark = _load_npz(repo_root, cfg.paths.benchmark_relative_path)
    qcbm_data = _load_npz(repo_root, cfg.paths.qcbm_training_relative_path)
    default_data = _load_npz(repo_root, cfg.paths.crca_default_training_relative_path)
    discount_data = _load_npz(repo_root, cfg.paths.crca_discount_training_relative_path)
    exposure_data = _load_npz(repo_root, cfg.paths.crca_exposure_training_relative_path)

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
        n_layers=_npz_int(discount_data, "n_layers", cfg.crca_discount_training.n_layers),
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

    return build_cva_problem_bundle(
        quantum_cva_circuit,
        qcbm_params=qcbm_theta,
        exposure_params=exposure_theta,
        default_prob_params=default_theta,
        discount_factor_params=discount_theta,
        metadata={
            "builder": "6q_cva",
            "qcbm_topology": qcbm_topology,
            "qcbm_n_layers": int(qcbm_n_layers),
            "exposure_ansatz": exposure_ansatz,
            "exposure_n_layers": int(exposure_n_layers),
            "artifact_paths": {
                "benchmark": str(_resolve(repo_root, cfg.paths.benchmark_relative_path)),
                "qcbm": str(_resolve(repo_root, cfg.paths.qcbm_training_relative_path)),
                "crca_default": str(_resolve(repo_root, cfg.paths.crca_default_training_relative_path)),
                "crca_discount": str(_resolve(repo_root, cfg.paths.crca_discount_training_relative_path)),
                "crca_exposure": str(_resolve(repo_root, cfg.paths.crca_exposure_training_relative_path)),
            },
        },
    )
