# python utils
import numpy as np
import matplotlib.pyplot as plt
import pathlib

#qiskit utils
from qiskit_algorithms import AmplitudeEstimation, IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler

# quantum_cva utils
from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import QuantumCVACircuit
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit

# ----- Data from classical computation loading -----
data_dir: str = "data"
_out = pathlib.Path(data_dir)
_out.mkdir(parents=True, exist_ok=True)

classical_cva_data = np.load(
    _out / "single_asset/benchmark/run_classical_cva_single_asset.npz"
)

# ----- Data from training loading -----
qcbm_data_path = (
    "data/single_asset/qcbm/qcbm_training_results_shots.npz"
)
positive_exposure_data_path = (
    "data/single_asset/crca/positive_exposures/training_results_shots.npz"
)
default_probabilities_data_path = (
    "data/single_asset/crca/default_probabilities/training_results_shots.npz"
)
discount_factors_data_path = (
    "data/single_asset/crca/discount_factors/training_results_shots.npz"
)

qcbm_data = np.load(qcbm_data_path, allow_pickle=True)
positive_exposure_data = np.load(positive_exposure_data_path, allow_pickle=True)
default_probabilities_data = np.load(
    default_probabilities_data_path, allow_pickle=True
)
discount_factors_data = np.load(discount_factors_data_path, allow_pickle=True)

# Trained parameters
qcbm_parameters = qcbm_data["theta_star"]
positive_exposure_parameters = positive_exposure_data["theta_star"]
default_probabilities_parameters = default_probabilities_data["theta_star"]
discount_factors_parameters = discount_factors_data["theta_star"]


# ----- Circuit construction -----
# Global parameters
num_qubits_time = 2
num_qubits_underlying = 2
total_num_qubits = num_qubits_time + num_qubits_underlying

recovery_rate = classical_cva_data["R_cva"]
C_v = classical_cva_data["C_v"]
C_p = classical_cva_data["C_p"]
C_q = classical_cva_data["C_q"]

seed = 123  # for reproducibility of shots-based estimation


# Constructing the sub-circuits
qcbm = MLQcbmCircuit(
    n_qubits=total_num_qubits, n_layers=2, name="qcbm_state_prep_circuit"
)

crca_positive_exposure = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=num_qubits_underlying,
    n_layers=2,
    name="crca_positive_exposure_circuit",
)

crca_default_probabilities = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=0,
    n_layers=1,
    name="crca_default_probabilities_circuit",
)

crca_discount_factors = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=0,
    n_layers=1,
    name="crca_discount_factors_circuit",
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
    C_v=C_v,
    C_p=C_p,
    C_q=C_q,
    name="quantum_cva_circuit",
    backend="statevector",
)

# ----- Circuit visualization -----
qc = quantum_cva_circuit.build_cva_circuit(
    qcbm_params=qcbm_parameters,
    crca_exposure_params=positive_exposure_parameters,
    crca_default_params=default_probabilities_parameters,
    crca_discount_params=discount_factors_parameters,
)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

fig = qc.draw(
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
#           Ideal (algebraic) CVA value: <ξ|111><111|ξ>
# =============================================================================
cva_quantum_statevector = quantum_cva_circuit.cva(
    qcbm_params=qcbm_parameters,
    exposure_params=positive_exposure_parameters,
    default_prob_params=default_probabilities_parameters,
    discount_factor_params=discount_factors_parameters,
)
# Relative error between classical & quantum estimation
cva_classical = classical_cva_data["cva_by_grid_size_values"][
    1
]  
quantum_agregated_error = np.abs(cva_quantum_statevector - cva_classical)
quantum_relative_error = quantum_agregated_error / cva_classical

# =============================================================================
#           CVA estimation via Quantum Amplitude Estimation (QAE) 
# =============================================================================
ancilla_exposure_idx = num_qubits_time + num_qubits_underlying
ancilla_default_idx = num_qubits_time + num_qubits_underlying + 1
ancilla_discount_idx = num_qubits_time + num_qubits_underlying + 2

problem = EstimationProblem(
    state_preparation=qc,
    objective_qubits=[
        ancilla_exposure_idx, 
        ancilla_default_idx, 
        ancilla_discount_idx
        ],
    is_good_state=lambda bitstr: bitstr == "111",
    post_processing=quantum_cva_circuit.cva_from_prob,  # devuelve CVA directamente desde a
)


ae = AmplitudeEstimation(
    num_eval_qubits=6,
    sampler=StatevectorSampler()
)

ae_result = ae.estimate(problem)

p111_hat = ae_result.estimation
ae_cva = ae_result.estimation_processed

# =============================================================================
#       CVA estimation via Iterative QAE (IQAE) - NISQ version of QAE
# =============================================================================
iae = IterativeAmplitudeEstimation(epsilon_target=1e-3, alpha=0.05, sampler=StatevectorSampler())

iae_result = iae.estimate(problem)

p111_hat = iae_result.estimation
iae_cva = iae_result.estimation_processed

# ------------------------ Print run results ------------------------
print("\n=== Quantum CVA Estimation Results ===\n")
print(f"Classical CVA (n_s=2): {cva_classical}")
print(f"Exact (no shots) quantum CVA estimation: {cva_quantum_statevector}")
print(f"Estimated CVA from QAE: {ae_cva} in {ae_result.num_oracle_queries}")
print(f"Estimated CVA from IQAE: {iae_cva} in {iae_result.num_oracle_queries}")

print("\n=== Relative Errors ===")
print(f"Relative error between ideal quantum & classical CVA:"
      f"{quantum_relative_error * 100} %")
print(f"Relative error between QAE CVA and classical CVA:"
      f"{np.abs(ae_cva - cva_classical) / cva_classical * 100} %")
print(f"Relative error between IQAE CVA and classical CVA:"
      f"{np.abs(iae_cva - cva_classical) / cva_classical * 100} %")
