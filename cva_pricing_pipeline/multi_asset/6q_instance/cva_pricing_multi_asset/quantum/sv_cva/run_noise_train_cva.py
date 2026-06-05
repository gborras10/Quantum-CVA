import pathlib

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    IterativeAmplitudeEstimation,
)
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import QuantumCVACircuit
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from utils_run_ideal import (
    _as_1d_float,
    _assert_param_size,
    _assert_vector_size,
    _bind_crca_eval,
    _bind_qcbm_ansatz,
    _build_improved_cva_initial_layout,
    _compute_statevector_submodel_diagnostics,
    _metadata_dict,
    _npz_int,
    _npz_str,
    _percent_relative_error,
    _print_circuit_summary,
    _print_layout_summary,
    _select_layout_for_training,
    _two_qubit_gate_count,
    _transpile_with_layout,
)


def _load_required_npz(path: pathlib.Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return np.load(path, allow_pickle=True)


def _load_qcbm_shots_npz(
    training_root: pathlib.Path,
) -> tuple[np.lib.npyio.NpzFile, pathlib.Path]:
    candidates = [
        training_root
        / "qcbm"
        / "shots"
        / "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz",
        training_root
        / "qcbm"
        / "shots"
        / "training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz",
    ]

    for path in candidates:
        if path.exists():
            return np.load(path, allow_pickle=True), path

    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No se encontro un archivo QCBM de shots+noise en ninguna ruta candidata:\n"
        f"{joined}"
    )


def main() -> None:
    backend_name = "ibm_basquecountry"
    seed_transpiler = 1234
    transpilation_opt_level = 3

    qcbm_topology_default = "qcbm_heavyhex6"
    crca_scalar_topology_default = "crca2"
    positive_exposure_topology_default = "heavy_hex_star"
    positive_exposure_layout_length = 7

    readout_quantile = 0.92
    local_2q_quantile = 0.85
    relax_if_needed = True
    approximation_degree = 1.0

    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data_root = repo_root / "data" / "multi_asset" / "6q_instance"
    benchmark_root = data_root / "benchmark"
    training_root = data_root / "quantum" / "training"

    classical_cva_data = _load_required_npz(benchmark_root / "three_asset_instance.npz")
    qcbm_data, qcbm_training_path = _load_qcbm_shots_npz(training_root)
    default_probabilities_path = (
        training_root
        / "crca"
        / "default_probabilities"
        / "training_crca2_shots_backend_noise_snapshot.npz"
    )
    discount_factors_path = (
        training_root
        / "crca"
        / "discount_factors"
        / "training_crca2_shots_backend_noise_snapshot.npz"
    )
    positive_exposure_path = (
        training_root
        / "crca"
        / "positive_exposure"
        / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
    )

    default_probabilities_data = _load_required_npz(default_probabilities_path)
    discount_factors_data = _load_required_npz(discount_factors_path)
    positive_exposure_data = _load_required_npz(positive_exposure_path)

    print("\n=== Fuentes de entrenamiento (shots + backend noise) ===")
    print(f"QCBM: {qcbm_training_path}")
    print(f"CRCA default_probabilities: {default_probabilities_path}")
    print(f"CRCA discount_factors: {discount_factors_path}")
    print(f"CRCA positive_exposure: {positive_exposure_path}")

    qcbm_parameters = _as_1d_float(qcbm_data["theta_star"])
    default_probabilities_parameters = _as_1d_float(default_probabilities_data["theta_star"])
    discount_factors_parameters = _as_1d_float(discount_factors_data["theta_star"])
    positive_exposure_parameters = _as_1d_float(positive_exposure_data["theta_star"])

    positive_exposure_metadata = _metadata_dict(positive_exposure_data)

    num_qubits_time = 2
    num_qubits_underlying = 4
    total_num_qubits = num_qubits_time + num_qubits_underlying

    recovery_rate = classical_cva_data["R_cva"]
    c_v = classical_cva_data["C_v"]
    c_p = classical_cva_data["C_p"]
    c_q = classical_cva_data["C_q"]

    qcbm_target_distribution = _as_1d_float(qcbm_data["p_target"])
    default_probabilities_target = _as_1d_float(classical_cva_data["q_t"]) / float(c_q)
    discount_factors_target = _as_1d_float(classical_cva_data["p_t"]) / float(c_p)
    positive_exposure_target = _as_1d_float(classical_cva_data["v_joint_t"]) / float(c_v)

    service = QiskitRuntimeService(channel="ibm_cloud", name="basquecountry_updated")
    real_backend = service.backend(backend_name, use_fractional_gates=True)

    qcbm_requested_topology = _npz_str(qcbm_data, "requested_topology", qcbm_topology_default)
    default_requested_topology = _npz_str(
        default_probabilities_data,
        "requested_topology",
        crca_scalar_topology_default,
    )
    discount_requested_topology = _npz_str(
        discount_factors_data,
        "requested_topology",
        crca_scalar_topology_default,
    )
    positive_exposure_requested_topology = _npz_str(
        positive_exposure_data,
        "requested_topology",
        positive_exposure_topology_default,
    )

    qcbm_layout, _, qcbm_layout_meta = _select_layout_for_training(
        real_backend,
        topology=qcbm_requested_topology,
        length=total_num_qubits,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )
    default_layout, _, default_layout_meta = _select_layout_for_training(
        real_backend,
        topology=default_requested_topology,
        length=3,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )
    discount_layout, _, discount_layout_meta = _select_layout_for_training(
        real_backend,
        topology=discount_requested_topology,
        length=3,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )
    positive_exposure_layout, _, positive_exposure_layout_meta = _select_layout_for_training(
        real_backend,
        topology=positive_exposure_requested_topology,
        length=positive_exposure_layout_length,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )

    qcbm_n_layers = _npz_int(qcbm_data, "n_layers", 6)
    positive_exposure_n_layers = int(
        positive_exposure_metadata.get(
            "n_layers",
            _npz_int(positive_exposure_data, "n_layers", 2),
        )
    )

    qcbm = MLQcbmCircuit(
        n_qubits=total_num_qubits,
        n_layers=qcbm_n_layers,
        name="qcbm_state_prep_circuit_shots_noise_theta",
        entangler="rzz",
        topology=qcbm_layout_meta["selected_topology"],
        backend=AerSimulator(method="statevector"),
        transpile_backend=real_backend,
        noise_model=None,
        simulation_method="statevector",
        optimization_level=transpilation_opt_level,
        initial_layout=[int(q) for q in qcbm_layout],
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=seed_transpiler,
    )

    crca_positive_exposure = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=num_qubits_underlying,
        n_layers=positive_exposure_n_layers,
        ansatz_type="heavy_hex_star",
        name="crca_positive_exposure_circuit_shots_noise_theta",
    )

    crca_default_probabilities = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(default_probabilities_data, "n_layers", 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_probabilities_circuit_shots_noise_theta",
    )

    crca_discount_factors = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(discount_factors_data, "n_layers", 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_discount_factors_circuit_shots_noise_theta",
    )

    _assert_param_size("QCBM", qcbm_parameters, qcbm.n_params)
    _assert_param_size(
        "CRCA positive_exposure",
        positive_exposure_parameters,
        crca_positive_exposure.n_params,
    )
    _assert_param_size(
        "CRCA default_probabilities",
        default_probabilities_parameters,
        crca_default_probabilities.n_params,
    )
    _assert_param_size(
        "CRCA discount_factors",
        discount_factors_parameters,
        crca_discount_factors.n_params,
    )

    _assert_vector_size("QCBM p_target", qcbm_target_distribution, qcbm.dim)
    _assert_vector_size(
        "CRCA default_probabilities target",
        default_probabilities_target,
        crca_default_probabilities.dim_controls,
    )
    _assert_vector_size(
        "CRCA discount_factors target",
        discount_factors_target,
        crca_discount_factors.dim_controls,
    )
    _assert_vector_size(
        "CRCA positive_exposure target",
        positive_exposure_target,
        crca_positive_exposure.dim_controls,
    )

    print("\n=== Dimensiones ===")
    print(f"QCBM: n_params={qcbm.n_params}, dim={qcbm.dim}")
    print(
        "CRCA positive_exposure: "
        f"n_params={crca_positive_exposure.n_params}, dim_controls={crca_positive_exposure.dim_controls}"
    )
    print(
        "CRCA default_probabilities: "
        f"n_params={crca_default_probabilities.n_params}, dim_controls={crca_default_probabilities.dim_controls}"
    )
    print(
        "CRCA discount_factors: "
        f"n_params={crca_discount_factors.n_params}, dim_controls={crca_discount_factors.dim_controls}"
    )

    quantum_cva_circuit = QuantumCVACircuit(
        num_qubits_time=num_qubits_time,
        num_qubits_underlying=num_qubits_underlying,
        qcbm_circuit=qcbm,
        crca_circuit_exposure=crca_positive_exposure,
        crca_circuit_default_prob=crca_default_probabilities,
        crca_circuit_discount_factor=crca_discount_factors,
        recovery_rate=recovery_rate,
        C_v=c_v,
        C_p=c_p,
        C_q=c_q,
        name="quantum_cva_circuit_shots_noise_theta",
        backend="statevector",
    )

    qc_cva_logical = quantum_cva_circuit.build_cva_circuit(
        qcbm_params=qcbm_parameters,
        crca_exposure_params=positive_exposure_parameters,
        crca_default_params=default_probabilities_parameters,
        crca_discount_params=discount_factors_parameters,
    )

    qc_qcbm_isa = _transpile_with_layout(
        _bind_qcbm_ansatz(qcbm, qcbm_parameters),
        backend=real_backend,
        initial_layout=qcbm_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )
    qc_positive_exposure_isa = _transpile_with_layout(
        _bind_crca_eval(crca_positive_exposure, positive_exposure_parameters),
        backend=real_backend,
        initial_layout=positive_exposure_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )
    qc_default_isa = _transpile_with_layout(
        _bind_crca_eval(crca_default_probabilities, default_probabilities_parameters),
        backend=real_backend,
        initial_layout=default_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )
    qc_discount_isa = _transpile_with_layout(
        _bind_crca_eval(crca_discount_factors, discount_factors_parameters),
        backend=real_backend,
        initial_layout=discount_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )

    (
        cva_initial_layout,
        cva_state_layout,
        time_phys,
        ancilla_exposure_phys,
        ancilla_default_phys,
        ancilla_discount_phys,
    ) = _build_improved_cva_initial_layout(
        real_backend=real_backend,
        qcbm_layout=qcbm_layout,
        positive_exposure_layout=positive_exposure_layout,
        default_layout=default_layout,
        discount_layout=discount_layout,
    )

    print("\n=== Improved aggregate ancilla placement ===")
    print(f"time qubits:               {time_phys}")
    print(f"state block:               {cva_state_layout}")
    print(f"ancilla_exposure_phys:     {ancilla_exposure_phys}")
    print(f"ancilla_default_phys:      {ancilla_default_phys}")
    print(f"ancilla_discount_phys:     {ancilla_discount_phys}")
    print(f"improved cva_initial_layout: {cva_initial_layout}")

    qc_cva_isa = _transpile_with_layout(
        qc_cva_logical,
        backend=real_backend,
        initial_layout=cva_initial_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )

    print("\n=== CVA Aggregate Transpilation (fixed layout) ===")
    print(f"depth={int(qc_cva_isa.depth())}")
    print(f"two_qubit_gates={_two_qubit_gate_count(qc_cva_isa)}")
    print(f"size={int(qc_cva_isa.size())}")

    _print_layout_summary(
        qcbm_requested_topology=qcbm_requested_topology,
        qcbm_layout_meta=qcbm_layout_meta,
        qcbm_layout=qcbm_layout,
        physical_positive_exposure_topology=positive_exposure_requested_topology,
        positive_exposure_layout_meta=positive_exposure_layout_meta,
        positive_exposure_layout=positive_exposure_layout,
        default_requested_topology=default_requested_topology,
        default_layout_meta=default_layout_meta,
        default_layout=default_layout,
        discount_requested_topology=discount_requested_topology,
        discount_layout_meta=discount_layout_meta,
        discount_layout=discount_layout,
        cva_initial_layout=cva_initial_layout,
    )

    _print_circuit_summary(
        qc_qcbm_isa=qc_qcbm_isa,
        qc_positive_exposure_isa=qc_positive_exposure_isa,
        qc_default_isa=qc_default_isa,
        qc_discount_isa=qc_discount_isa,
        qc_cva_logical=qc_cva_logical,
        qc_cva_isa=qc_cva_isa,
    )

    (
        qcbm_kl_statevector,
        default_probabilities_l2,
        discount_factors_l2,
        positive_exposure_l2,
    ) = _compute_statevector_submodel_diagnostics(
        qcbm=qcbm,
        qcbm_parameters=qcbm_parameters,
        qcbm_target_distribution=qcbm_target_distribution,
        crca_default_probabilities=crca_default_probabilities,
        default_probabilities_parameters=default_probabilities_parameters,
        default_probabilities_target=default_probabilities_target,
        crca_discount_factors=crca_discount_factors,
        discount_factors_parameters=discount_factors_parameters,
        discount_factors_target=discount_factors_target,
        crca_positive_exposure=crca_positive_exposure,
        positive_exposure_parameters=positive_exposure_parameters,
        positive_exposure_target=positive_exposure_target,
    )

    print("\n=== Statevector diagnostics con theta_star entrenado en shots+noise ===")
    print(f"QCBM KL(p_target || p_theta_star): {qcbm_kl_statevector:.8e}")
    print(f"CRCA default probabilities L2 error: {default_probabilities_l2:.8e}")
    print(f"CRCA discount factors L2 error: {discount_factors_l2:.8e}")
    print(f"CRCA positive exposure L2 error: {positive_exposure_l2:.8e}")

    cva_quantum_statevector = quantum_cva_circuit.cva(
        qcbm_params=qcbm_parameters,
        exposure_params=positive_exposure_parameters,
        default_prob_params=default_probabilities_parameters,
        discount_factor_params=discount_factors_parameters,
    )
    cva_classical = classical_cva_data["cva_by_grid_size_values"][1]

    ancilla_exposure_idx = num_qubits_time + num_qubits_underlying
    ancilla_default_idx = num_qubits_time + num_qubits_underlying + 1
    ancilla_discount_idx = num_qubits_time + num_qubits_underlying + 2

    problem = EstimationProblem(
        state_preparation=qc_cva_logical,
        objective_qubits=[
            ancilla_exposure_idx,
            ancilla_default_idx,
            ancilla_discount_idx,
        ],
        is_good_state=lambda bitstr: bitstr == "111",
        post_processing=quantum_cva_circuit.cva_from_prob,
    )

    ae = AmplitudeEstimation(num_eval_qubits=6, sampler=StatevectorSampler())
    ae_result = ae.estimate(problem)
    ae_cva = ae_result.estimation_processed

    iae = IterativeAmplitudeEstimation(
        epsilon_target=1e-3,
        alpha=0.05,
        sampler=StatevectorSampler(),
    )
    iae_result = iae.estimate(problem)
    iae_cva = iae_result.estimation_processed

    print("\n=== Quantum CVA Estimation Results ===\n")
    print(f"Classical CVA (n_s=2): {cva_classical}")
    print(f"Ideal quantum CVA usando theta_star shots+noise: {cva_quantum_statevector}")
    print(f"Estimated CVA from QAE: {ae_cva} in {ae_result.num_oracle_queries}")
    print(f"Estimated CVA from IQAE: {iae_cva} in {iae_result.num_oracle_queries}")

    print("\n=== Relative Errors ===")
    print(
        "Relative error between ideal quantum & classical CVA: "
        f"{_percent_relative_error(cva_quantum_statevector, cva_classical)} %"
    )
    print(
        "Relative error between QAE CVA and classical CVA: "
        f"{_percent_relative_error(ae_cva, cva_classical)} %"
    )
    print(
        "Relative error between IQAE CVA and classical CVA: "
        f"{_percent_relative_error(iae_cva, cva_classical)} %"
    )


if __name__ == "__main__":
    main()