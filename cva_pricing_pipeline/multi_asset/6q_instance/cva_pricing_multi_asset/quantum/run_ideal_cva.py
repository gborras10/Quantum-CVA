import pathlib

import matplotlib.pyplot as plt
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
    _choose_positive_exposure_physical_params,
    _compute_statevector_submodel_diagnostics,
    _configure_plot_style,
    _metadata_dict,
    _npz_int,
    _npz_str,
    _percent_relative_error,
    _print_circuit_summary,
    _print_layout_summary,
    _select_layout_for_training,
    _two_qubit_gate_count,
    _transpile_with_layout,
    _transpile_with_sabre_multiseed,
    _write_transpiled_cva_circuit_text_snapshot,
)


def main() -> None:
    backend_name = "ibm_basquecountry"
    seed_transpiler = 1234
    transpilation_opt_level = 3

    qcbm_topology_default = "qcbm_heavyhex6"
    crca_scalar_topology_default = "crca2"
    positive_exposure_physical_topology = "heavy_hex_star"
    positive_exposure_physical_length = 7
    positive_exposure_sabre_seeds = [1, 42, 202, 404, 1234]
    cva_aggregate_sabre_seeds = [1, 7, 8, 17, 18, 22, 42, 73, 101, 202, 404, 777, 1234]

    readout_quantile = 0.95
    local_2q_quantile = 0.95
    relax_if_needed = True
    approximation_degree = 1.0

    # ----- Data from classical computation loading -----
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data_root = repo_root / "data" / "multi_asset" / "6q_instance"
    benchmark_root = data_root / "benchmark"
    training_root = data_root / "quantum" / "training"

    classical_cva_data = np.load(
        benchmark_root / "three_asset_instance.npz",
        allow_pickle=True,
    )

    # ----- Data from training loading -----
    qcbm_data = np.load(
        training_root / "qcbm" / "statevector" / "training_qcbm_heavyhex6_6lay.npz",
        allow_pickle=True,
    )

    positive_exposure_data = np.load(
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "positive_exposure"
        / "training_heavy_hex_star.npz",
        allow_pickle=True,
    )

    default_probabilities_data = np.load(
        training_root / "crca" / "default_probabilities" / "training_crca2.npz",
        allow_pickle=True,
    )
    discount_factors_data = np.load(
        training_root / "crca" / "discount_factors" / "training_crca2.npz",
        allow_pickle=True,
    )

    # Trained parameters
    qcbm_parameters = _as_1d_float(qcbm_data["theta_star"])
    positive_exposure_parameters = _as_1d_float(positive_exposure_data["theta_star"])
    default_probabilities_parameters = _as_1d_float(default_probabilities_data["theta_star"])
    discount_factors_parameters = _as_1d_float(discount_factors_data["theta_star"])

    positive_exposure_metadata = _metadata_dict(positive_exposure_data)

    # ----- Circuit construction -----
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

    # ----- Real-backend config and training-like layouts -----
    service = QiskitRuntimeService(channel="ibm_cloud")
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

    qcbm_layout, qcbm_layout_score, qcbm_layout_meta = _select_layout_for_training(
        real_backend,
        topology=qcbm_requested_topology,
        length=total_num_qubits,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )
    default_layout, default_layout_score, default_layout_meta = _select_layout_for_training(
        real_backend,
        topology=default_requested_topology,
        length=3,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )
    discount_layout, discount_layout_score, discount_layout_meta = _select_layout_for_training(
        real_backend,
        topology=discount_requested_topology,
        length=3,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )

    # ----- Positive exposure physical circuit (heavy_hex_star: 8 controls + 1 ancilla) -----
    crca_positive_exposure_physical = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=num_qubits_underlying,
        n_layers=int(positive_exposure_metadata.get("n_layers", 2)),
        ansatz_type="heavy_hex_star",
        name="crca_positive_exposure_heavy_hex_star_physical",
    )

    (
        positive_exposure_layout,
        positive_exposure_layout_score,
        positive_exposure_layout_meta,
    ) = _select_layout_for_training(
        real_backend,
        topology=positive_exposure_physical_topology,
        length=positive_exposure_physical_length,
        readout_quantile=readout_quantile,
        local_2q_quantile=local_2q_quantile,
        relax_if_needed=relax_if_needed,
    )

    # ----- Sub-circuits (logical, aligned with training) -----
    qcbm_n_layers = _npz_int(qcbm_data, "n_layers", 6)
    qcbm = MLQcbmCircuit(
        n_qubits=total_num_qubits,
        n_layers=qcbm_n_layers,
        name="qcbm_state_prep_circuit",
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
        n_layers=int(positive_exposure_metadata.get("n_layers", 2)),
        ansatz_type="heavy_hex_star",
        name="crca_positive_exposure_circuit",
    )

    crca_default_probabilities = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(default_probabilities_data, "n_layers", 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_probabilities_circuit",
    )

    crca_discount_factors = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(discount_factors_data, "n_layers", 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_discount_factors_circuit",
    )

    # Early checks: loaded theta* must match circuit definitions.
    _assert_param_size("QCBM", qcbm_parameters, qcbm.n_params)
    _assert_param_size("CRCA positive_exposure", positive_exposure_parameters, crca_positive_exposure.n_params)
    _assert_param_size(
        "CRCA default_probabilities",
        default_probabilities_parameters,
        crca_default_probabilities.n_params,
    )
    _assert_param_size("CRCA discount_factors", discount_factors_parameters, crca_discount_factors.n_params)
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

    # Constructing the overall quantum CVA circuit
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
        name="quantum_cva_circuit",
        backend="statevector",
    )

    # ----- Logical and transpiled CVA circuits -----
    qc_cva_logical = quantum_cva_circuit.build_cva_circuit(
        qcbm_params=qcbm_parameters,
        crca_exposure_params=positive_exposure_parameters,
        crca_default_params=default_probabilities_parameters,
        crca_discount_params=discount_factors_parameters,
    )

    # Build and transpile each sub-circuit with training-like layout assumptions.
    qc_qcbm_bound = _bind_qcbm_ansatz(qcbm, qcbm_parameters)
    qc_qcbm_isa = _transpile_with_layout(
        qc_qcbm_bound,
        backend=real_backend,
        initial_layout=qcbm_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )

    positive_exposure_physical_params = _choose_positive_exposure_physical_params(
        positive_exposure_parameters,
        crca_positive_exposure_physical.n_params,
        fallback_seed=1234,
        fallback_scale=0.5,
    )
    qc_positive_exposure_eval_bound = _bind_crca_eval(
        crca_positive_exposure_physical,
        positive_exposure_physical_params,
    )
    qc_positive_exposure_isa, positive_exposure_best_seed, positive_exposure_best_metrics, positive_exposure_seed_metrics = _transpile_with_sabre_multiseed(
        qc_positive_exposure_eval_bound,
        backend=real_backend,
        optimization_level=transpilation_opt_level,
        seeds=positive_exposure_sabre_seeds,
        approximation_degree=approximation_degree,
    )
    print("\n=== Positive Exposure Sabre Multi-seed ===")
    print(f"candidate seeds: {positive_exposure_sabre_seeds}")
    for metrics in positive_exposure_seed_metrics:
        print(
            f"seed={metrics['seed']}: depth={metrics['depth']}, "
            f"two_qubit_gates={metrics['two_qubit_gates']}, size={metrics['size']}"
        )
    print(
        "selected seed: "
        f"{positive_exposure_best_seed} "
        f"(depth={positive_exposure_best_metrics['depth']}, "
        f"two_qubit_gates={positive_exposure_best_metrics['two_qubit_gates']}, "
        f"size={positive_exposure_best_metrics['size']})"
    )

    qc_default_eval_bound = _bind_crca_eval(
        crca_default_probabilities,
        default_probabilities_parameters,
    )
    qc_default_isa = _transpile_with_layout(
        qc_default_eval_bound,
        backend=real_backend,
        initial_layout=default_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )

    qc_discount_eval_bound = _bind_crca_eval(
        crca_discount_factors,
        discount_factors_parameters,
    )
    qc_discount_isa = _transpile_with_layout(
        qc_discount_eval_bound,
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

    qc_cva_isa_fixed = _transpile_with_layout(
        qc_cva_logical,
        backend=real_backend,
        initial_layout=cva_initial_layout,
        optimization_level=transpilation_opt_level,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )

    cva_fixed_metrics = {
        "depth": int(qc_cva_isa_fixed.depth()),
        "two_qubit_gates": _two_qubit_gate_count(qc_cva_isa_fixed),
        "size": int(qc_cva_isa_fixed.size()),
    }

    qc_cva_isa_sabre, cva_aggregate_best_seed, cva_aggregate_best_metrics, cva_aggregate_seed_metrics = _transpile_with_sabre_multiseed(
        qc_cva_logical,
        backend=real_backend,
        optimization_level=transpilation_opt_level,
        seeds=cva_aggregate_sabre_seeds,
        approximation_degree=approximation_degree,
    )

    cva_fixed_key = (
        cva_fixed_metrics["depth"],
        cva_fixed_metrics["two_qubit_gates"],
        cva_fixed_metrics["size"],
        -1,
    )
    cva_sabre_key = (
        cva_aggregate_best_metrics["depth"],
        cva_aggregate_best_metrics["two_qubit_gates"],
        cva_aggregate_best_metrics["size"],
        cva_aggregate_best_seed,
    )

    if cva_sabre_key < cva_fixed_key:
        qc_cva_isa = qc_cva_isa_sabre
        cva_aggregate_method = "sabre_multiseed"
        cva_selected_seed = cva_aggregate_best_seed
        cva_selected_metrics = cva_aggregate_best_metrics
    else:
        qc_cva_isa = qc_cva_isa_fixed
        cva_aggregate_method = "fixed_layout"
        cva_selected_seed = seed_transpiler
        cva_selected_metrics = cva_fixed_metrics

    transpiled_cva_text_snapshot_path = (
        data_root / "quantum" / "cva_pricing" / "transpiled_cva_circuit_text_snapshot.txt"
    )
    _write_transpiled_cva_circuit_text_snapshot(
        circuit=qc_cva_isa,
        output_path=transpiled_cva_text_snapshot_path,
        backend_name=backend_name,
        transpilation_method=cva_aggregate_method,
        selected_seed=cva_selected_seed,
        selected_metrics=cva_selected_metrics,
    )

    print("\n=== CVA Aggregate Transpilation Selection ===")
    print(
        "fixed layout candidate: "
        f"depth={cva_fixed_metrics['depth']}, "
        f"two_qubit_gates={cva_fixed_metrics['two_qubit_gates']}, "
        f"size={cva_fixed_metrics['size']}"
    )
    print(f"sabre candidate seeds: {cva_aggregate_sabre_seeds}")
    for metrics in cva_aggregate_seed_metrics:
        print(
            f"seed={metrics['seed']}: depth={metrics['depth']}, "
            f"two_qubit_gates={metrics['two_qubit_gates']}, size={metrics['size']}"
        )
    print(
        "selected aggregate transpilation: "
        f"method={cva_aggregate_method}, "
        f"seed={cva_selected_seed}, "
        f"depth={cva_selected_metrics['depth']}, "
        f"two_qubit_gates={cva_selected_metrics['two_qubit_gates']}, "
        f"size={cva_selected_metrics['size']}"
    )
    print(f"transpiled CVA text snapshot saved to: {transpiled_cva_text_snapshot_path}")

    _print_layout_summary(
        qcbm_requested_topology=qcbm_requested_topology,
        qcbm_layout_meta=qcbm_layout_meta,
        qcbm_layout=qcbm_layout,
        physical_positive_exposure_topology=positive_exposure_physical_topology,
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

    # ----- Statevector diagnostics with trained theta* -----
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

    print("\n=== Statevector Submodel Diagnostics (theta* trained in statevector) ===")
    print(f"QCBM KL(p_target || p_theta*): {qcbm_kl_statevector:.8e}")
    print(f"CRCA default probabilities L2 error: {default_probabilities_l2:.8e}")
    print(f"CRCA discount factors L2 error: {discount_factors_l2:.8e}")
    print(f"CRCA positive exposure L2 error: {positive_exposure_l2:.8e}")

    # ----- Circuit visualization (logical aggregate CVA) -----
    _configure_plot_style()
    fig = qc_cva_logical.draw(
        output="mpl",
        style={
            "name": "bw",
            "fontsize": 8,
            "subfontsize": 6,
            "figwidth": 36,
            "dpi": 300,
            "linecolor": "#1a1a1a",
            "textcolor": "#000000",
            "gatetextcolor": "#000000",
            "barrierfacecolor": "#cccccc",
            "backgroundcolor": "#FFFFFF",
        },
        fold=-1,
        scale=0.62,
        plot_barriers=True,
        initial_state=False,
        cregbundle=False,
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.8)
    plt.show()

    # =============================================================================
    #           Ideal (algebraic) CVA value: <xi|111><111|xi>
    # =============================================================================
    cva_quantum_statevector = quantum_cva_circuit.cva(
        qcbm_params=qcbm_parameters,
        exposure_params=positive_exposure_parameters,
        default_prob_params=default_probabilities_parameters,
        discount_factor_params=discount_factors_parameters,
    )

    # Relative error between classical & quantum estimation
    cva_classical = classical_cva_data["cva_by_grid_size_values"][1]

    # =============================================================================
    #           CVA estimation via Quantum Amplitude Estimation (QAE)
    # =============================================================================
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

    # =============================================================================
    #       CVA estimation via Iterative QAE (IQAE) - NISQ version of QAE
    # =============================================================================
    iae = IterativeAmplitudeEstimation(
        epsilon_target=1e-3,
        alpha=0.05,
        sampler=StatevectorSampler(),
    )
    iae_result = iae.estimate(problem)
    iae_cva = iae_result.estimation_processed

    # ------------------------ Print run results ------------------------
    print("\n=== Quantum CVA Estimation Results ===\n")
    print(f"Classical CVA (n_s=2): {cva_classical}")
    print(f"Exact (no shots) quantum CVA estimation: {cva_quantum_statevector}")
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
