# python utils
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# qiskit utils
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    IterativeAmplitudeEstimation,
)
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

# quantum_cva utils
from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import QuantumCVACircuit
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)

BACKEND_NAME = "ibm_basquecountry"
SEED_TRANSPILER = 1234
TRANSPILATION_OPT_LEVEL = 3

QCBM_TOPOLOGY_DEFAULT = "qcbm_heavyhex8"
CRCA_SCALAR_TOPOLOGY_DEFAULT = "crca2"
POSITIVE_EXPOSURE_PHYSICAL_TOPOLOGY = "heavy_hex_star"
POSITIVE_EXPOSURE_PHYSICAL_LENGTH = 9


def _as_1d_float(array_like: np.ndarray) -> np.ndarray:
    return np.asarray(array_like, dtype=float).ravel()


def _npz_int(npz_data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in npz_data:
        return int(default)
    return int(np.asarray(npz_data[key]).item())


def _npz_str(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz_data:
        return str(default)
    return str(np.asarray(npz_data[key]).item())


def _metadata_dict(npz_data: np.lib.npyio.NpzFile) -> dict:
    if "metadata" not in npz_data:
        return {}
    maybe_dict = npz_data["metadata"]
    if hasattr(maybe_dict, "item"):
        maybe_dict = maybe_dict.item()
    return maybe_dict if isinstance(maybe_dict, dict) else {}


def _assert_param_size(label: str, theta: np.ndarray, expected_size: int) -> None:
    actual_size = int(np.asarray(theta).size)
    if actual_size != int(expected_size):
        raise ValueError(
            f"Parameter-size mismatch for {label}: expected {expected_size}, got {actual_size}."
        )


def _bind_qcbm_ansatz(qcbm: MLQcbmCircuit, theta: np.ndarray) -> np.ndarray:
    bind_map = {qcbm.theta[i]: float(theta[i]) for i in range(qcbm.n_params)}
    return qcbm.qc.assign_parameters(bind_map, inplace=False)


def _bind_crca_eval(crca: CrcaCircuit, theta: np.ndarray) -> np.ndarray:
    bind_map = {crca.theta[i]: float(theta[i]) for i in range(crca.n_params)}
    return crca.qc_eval.assign_parameters(bind_map, inplace=False)


def _pick_distinct_qubit(
    preferred: int,
    used: set[int],
    backend_num_qubits: int,
) -> int:
    if preferred not in used:
        return preferred

    for q in range(int(backend_num_qubits)):
        if q not in used:
            return q

    raise RuntimeError("No available physical qubits left while building CVA initial layout.")


# ----- Data from classical computation loading -----
repo_root = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

classical_cva_data = np.load(
    repo_root / "data" / "multi_asset" / "8q_instance" / "benchmark" / "three_asset_instance.npz",
    allow_pickle=True,
)

# ----- Data from training loading -----
qcbm_data_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "8q_instance"
    / "quantum"
    / "training"
    / "qcbm"
    / "training_qcbm_heavyhex8.npz"
)

positive_exposure_data_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "8q_instance"
    / "quantum"
    / "training"
    / "crca"
    / "positive_exposure"
    / "training_heavy_hex_star.npz"
)

default_probabilities_data_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "8q_instance"
    / "quantum"
    / "training"
    / "crca"
    / "default_probabilities"
    / "training_crca2.npz"
)

discount_factors_data_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "8q_instance"
    / "quantum"
    / "training"
    / "crca"
    / "discount_factors"
    / "training_crca2.npz"
)

qcbm_data = np.load(qcbm_data_path, allow_pickle=True)
positive_exposure_data = np.load(positive_exposure_data_path, allow_pickle=True)
default_probabilities_data = np.load(default_probabilities_data_path, allow_pickle=True)
discount_factors_data = np.load(discount_factors_data_path, allow_pickle=True)

# Trained parameters
qcbm_parameters = _as_1d_float(qcbm_data["theta_star"])
positive_exposure_parameters = _as_1d_float(positive_exposure_data["theta_star"])
default_probabilities_parameters = _as_1d_float(default_probabilities_data["theta_star"])
discount_factors_parameters = _as_1d_float(discount_factors_data["theta_star"])

positive_exposure_metadata = _metadata_dict(positive_exposure_data)

# ----- Circuit construction -----
# Global parameters
num_qubits_time = 2
num_qubits_underlying = 6
total_num_qubits = num_qubits_time + num_qubits_underlying

recovery_rate = classical_cva_data["R_cva"]
C_v = classical_cva_data["C_v"]
C_p = classical_cva_data["C_p"]
C_q = classical_cva_data["C_q"]

# ----- Real-backend config and training-like layouts -----
service = QiskitRuntimeService(channel="ibm_cloud")
real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

qcbm_requested_topology = _npz_str(qcbm_data, "requested_topology", QCBM_TOPOLOGY_DEFAULT)
default_requested_topology = _npz_str(
    default_probabilities_data,
    "requested_topology",
    CRCA_SCALAR_TOPOLOGY_DEFAULT,
)
discount_requested_topology = _npz_str(
    discount_factors_data,
    "requested_topology",
    CRCA_SCALAR_TOPOLOGY_DEFAULT,
)

qcbm_layout, qcbm_layout_score, qcbm_layout_meta = select_best_layout(
    real_backend,
    topology=qcbm_requested_topology,
    length=total_num_qubits,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
    relax_if_needed=True,
)

default_layout, default_layout_score, default_layout_meta = select_best_layout(
    real_backend,
    topology=default_requested_topology,
    length=3,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
    relax_if_needed=True,
)

discount_layout, discount_layout_score, discount_layout_meta = select_best_layout(
    real_backend,
    topology=discount_requested_topology,
    length=3,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
    relax_if_needed=True,
)

# ----- Positive exposure physical circuit (heavy_hex_star: 8 controls + 1 ancilla) -----
physical_positive_exposure_topology = POSITIVE_EXPOSURE_PHYSICAL_TOPOLOGY
crca_positive_exposure_physical = CrcaCircuit(
    m_time=num_qubits_time,
    n_price=num_qubits_underlying,
    n_layers=int(positive_exposure_metadata.get("n_layers", 2)),
    ansatz_type="heavy_hex_star",
    name="crca_positive_exposure_heavy_hex_star_physical",
)

positive_exposure_layout, positive_exposure_layout_score, positive_exposure_layout_meta = (
    select_best_layout(
        real_backend,
        topology=physical_positive_exposure_topology,
        length=POSITIVE_EXPOSURE_PHYSICAL_LENGTH,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
    )
)

# ----- Sub-circuits (logical, aligned with training) -----
qcbm_n_layers = _npz_int(qcbm_data, "n_layers", 8)
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
    optimization_level=TRANSPILATION_OPT_LEVEL,
    initial_layout=list(map(int, qcbm_layout)),
    layout_method="trivial",
    routing_method="none",
    seed_transpiler=SEED_TRANSPILER,
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
_assert_param_size("CRCA default_probabilities", default_probabilities_parameters, crca_default_probabilities.n_params)
_assert_param_size("CRCA discount_factors", discount_factors_parameters, crca_discount_factors.n_params)

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

# ----- Logical and transpiled CVA circuits -----
qc_cva_logical = quantum_cva_circuit.build_cva_circuit(
    qcbm_params=qcbm_parameters,
    crca_exposure_params=positive_exposure_parameters,
    crca_default_params=default_probabilities_parameters,
    crca_discount_params=discount_factors_parameters,
)

# Build and transpile each sub-circuit with training-like layout assumptions.
qc_qcbm_bound = _bind_qcbm_ansatz(qcbm, qcbm_parameters)
pm_qcbm = generate_preset_pass_manager(
    backend=real_backend,
    optimization_level=TRANSPILATION_OPT_LEVEL,
    initial_layout=list(map(int, qcbm_layout)),
    seed_transpiler=SEED_TRANSPILER,
    approximation_degree=1.0,
)
qc_qcbm_isa = pm_qcbm.run(qc_qcbm_bound)

if positive_exposure_parameters.size == crca_positive_exposure_physical.n_params:
    positive_exposure_physical_params = positive_exposure_parameters
else:
    rng = np.random.default_rng(1234)
    positive_exposure_physical_params = 0.5 * rng.standard_normal(
        crca_positive_exposure_physical.n_params
    ).astype(float)

qc_positive_exposure_eval_bound = _bind_crca_eval(
    crca_positive_exposure_physical,
    positive_exposure_physical_params,
)
pm_positive_exposure = generate_preset_pass_manager(
    backend=real_backend,
    optimization_level=TRANSPILATION_OPT_LEVEL,
    initial_layout=list(map(int, positive_exposure_layout)),
    seed_transpiler=SEED_TRANSPILER,
    approximation_degree=1.0,
)
qc_positive_exposure_isa = pm_positive_exposure.run(qc_positive_exposure_eval_bound)

qc_default_eval_bound = _bind_crca_eval(
    crca_default_probabilities,
    default_probabilities_parameters,
)
pm_default = generate_preset_pass_manager(
    backend=real_backend,
    optimization_level=TRANSPILATION_OPT_LEVEL,
    initial_layout=list(map(int, default_layout)),
    seed_transpiler=SEED_TRANSPILER,
    approximation_degree=1.0,
)
qc_default_isa = pm_default.run(qc_default_eval_bound)

qc_discount_eval_bound = _bind_crca_eval(
    crca_discount_factors,
    discount_factors_parameters,
)
pm_discount = generate_preset_pass_manager(
    backend=real_backend,
    optimization_level=TRANSPILATION_OPT_LEVEL,
    initial_layout=list(map(int, discount_layout)),
    seed_transpiler=SEED_TRANSPILER,
    approximation_degree=1.0,
)
qc_discount_isa = pm_discount.run(qc_discount_eval_bound)

# Aggregate CVA physical layout from selected sub-layouts.
cva_state_layout = list(map(int, qcbm_layout))
used_physical = set(cva_state_layout)

ancilla_exposure_pref = int(positive_exposure_layout[-1])
ancilla_default_pref = int(default_layout[-1])
ancilla_discount_pref = int(discount_layout[-1])

backend_num_qubits = int(real_backend.configuration().num_qubits)
ancilla_exposure_phys = _pick_distinct_qubit(
    ancilla_exposure_pref,
    used_physical,
    backend_num_qubits,
)
used_physical.add(ancilla_exposure_phys)

ancilla_default_phys = _pick_distinct_qubit(
    ancilla_default_pref,
    used_physical,
    backend_num_qubits,
)
used_physical.add(ancilla_default_phys)

ancilla_discount_phys = _pick_distinct_qubit(
    ancilla_discount_pref,
    used_physical,
    backend_num_qubits,
)

cva_initial_layout = cva_state_layout + [
    ancilla_exposure_phys,
    ancilla_default_phys,
    ancilla_discount_phys,
]

pm_cva = generate_preset_pass_manager(
    backend=real_backend,
    optimization_level=TRANSPILATION_OPT_LEVEL,
    initial_layout=cva_initial_layout,
    seed_transpiler=SEED_TRANSPILER,
    approximation_degree=1.0,
)
qc_cva_isa = pm_cva.run(qc_cva_logical)

print("\n=== Layout Summary ===")
print(f"QCBM topology requested/effective: {qcbm_requested_topology} / {qcbm_layout_meta['selected_topology']}")
print(f"QCBM layout: {list(map(int, qcbm_layout))}")
print(
    "Positive exposure topology requested/effective: "
    f"{physical_positive_exposure_topology} / {positive_exposure_layout_meta['selected_topology']}"
)
print(f"Positive exposure layout: {list(map(int, positive_exposure_layout))}")
print(
    "Default probabilities topology requested/effective: "
    f"{default_requested_topology} / {default_layout_meta['selected_topology']}"
)
print(f"Default probabilities layout: {list(map(int, default_layout))}")
print(
    "Discount factors topology requested/effective: "
    f"{discount_requested_topology} / {discount_layout_meta['selected_topology']}"
)
print(f"Discount factors layout: {list(map(int, discount_layout))}")
print(f"CVA initial layout (state + ancillas): {cva_initial_layout}")

print("\n=== Circuit Summary ===")
summarize_circuit(qc_qcbm_isa, label="QCBM transpiled")
summarize_circuit(qc_positive_exposure_isa, label="Positive exposure transpiled")
summarize_circuit(qc_default_isa, label="Default probabilities transpiled")
summarize_circuit(qc_discount_isa, label="Discount factors transpiled")
summarize_circuit(qc_cva_logical, label="CVA logical (agregado)")
summarize_circuit(qc_cva_isa, label="CVA transpiled (agregado)")

# ----- Circuit visualization (logical aggregate CVA) -----
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "text.usetex": False,
    }
)

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
#           Ideal (algebraic) CVA value: <ξ|111><111|ξ>
# =============================================================================
cva_quantum_statevector = quantum_cva_circuit.cva(
    qcbm_params=qcbm_parameters,
    exposure_params=positive_exposure_parameters,
    default_prob_params=default_probabilities_parameters,
    discount_factor_params=discount_factors_parameters,
)

# Relative error between classical & quantum estimation
cva_classical = classical_cva_data["cva_by_grid_size_values"][1]
quantum_agregated_error = np.abs(cva_quantum_statevector - cva_classical)
quantum_relative_error = quantum_agregated_error / cva_classical

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
print(f"Relative error between ideal quantum & classical CVA: {quantum_relative_error * 100} %")
print(
    "Relative error between QAE CVA and classical CVA: "
    f"{np.abs(ae_cva - cva_classical) / cva_classical * 100} %"
)
print(
    "Relative error between IQAE CVA and classical CVA: "
    f"{np.abs(iae_cva - cva_classical) / cva_classical * 100} %"
)
