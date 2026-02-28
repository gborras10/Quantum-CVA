# python utils
import numpy as np

# quantum_cva utils
from quantum_cva.cva_circuit import QuantumCVACircuit
from quantum_cva.state_prep.qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.crca.crca_circuit import CrcaCircuit

# ----- Data from training loading -----
qcbm_data_path = "data/qcbm/training_results.npz"
positive_exposure_data_path = "data/crca/positive_exposures/training_results.npz"
default_probabilities_data_path = "data/crca/default_probabilities/training_results.npz"
discount_factors_data_path = "data/crca/discount_factors/training_results.npz"

qcbm_data = np.load(qcbm_data_path, allow_pickle=True)
positive_exposure_data = np.load(positive_exposure_data_path, allow_pickle=True)
default_probabilities_data = np.load(default_probabilities_data_path, allow_pickle=True)
discount_factors_data = np.load(discount_factors_data_path, allow_pickle=True)

# Trained parameters
qcbm_parameters = qcbm_data["theta_star"]
positive_exposure_parameters = positive_exposure_data["theta_star"]
default_probabilities_parameters = default_probabilities_data["theta_star"]
discount_factors_parameters = discount_factors_data["theta_star"]
# -------------------------------------


# ----- Circuit construction -----
# Global parameters
num_qubits_time = 2
num_qubits_underlying = 2
total_num_qubits = num_qubits_time + num_qubits_underlying

recovery_rate = 0.415


# Constructing the sub-circuits
qcbm = MLQcbmCircuit(
    n_qubits=total_num_qubits, 
    n_layers=2, 
    name="qcbm_state_prep_circuit"
)

crca_positive_exposure = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=num_qubits_underlying,
    n_layers=2,      
    name="crca_positive_exposure_circuit"
)

crca_default_probabilities = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=0,
    n_layers=1,
    name="crca_default_probabilities_circuit"
)

crca_discount_factors = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=0,
    n_layers=1,
    name="crca_discount_factors_circuit"
)

# Constructing the overall quantum CVA circuit
quantum_cva_circuit = QuantumCVACircuit(
    num_qubits_time=num_qubits_time,
    num_qubits_underlying=num_qubits_underlying,
    qcbm_circuit=qcbm,
    crca_circuit_exposure=crca_positive_exposure,
    crca_circuit_default_prob=crca_default_probabilities,
    crca_circuit_discount_factor=crca_discount_factors,
    recovery_rate= recovery_rate,
    C_v=
)


