# single_eval_density_matrix_noise.py

# python utils
import pathlib
import time
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# quantum_cva utils
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)

# ===================== Load target =====================
repo_root = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

data = np.load(
    repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz",
    allow_pickle=True,
)

saving_path = (
    repo_root
    / "data"
    / "multi_asset"
    / "quantum"
    / "training"
    / "qcbm"
    / "single_eval_density_matrix_noise.npz"
)

ptg = np.asarray(data["p_target"], dtype=float).ravel()
ptg /= ptg.sum()

dim = ptg.size
n_qubits = int(np.log2(dim))
if 2**n_qubits != dim:
    raise ValueError(f"p_target length must be a power of two; got {dim}.")

# ===================== Backend / layout =====================
BACKEND_NAME = "ibm_basquecountry"
LOGICAL_TOPOLOGY = "circular"

service = QiskitRuntimeService(channel="ibm_cloud")
real_backend = service.backend(BACKEND_NAME)
real_noise_model = NoiseModel.from_backend(real_backend)

chosen_layout, layout_score, layout_meta = select_best_layout(
    real_backend,
    topology=LOGICAL_TOPOLOGY,
    length=n_qubits,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
)

effective_topology = layout_meta["selected_topology"]

print("backend_name       =", BACKEND_NAME)
print("requested_topology =", LOGICAL_TOPOLOGY)
print("effective_topology =", effective_topology)
print("chosen_layout      =", chosen_layout)
print("layout_score       =", layout_score)
print("fallback_used      =", layout_meta["fallback_used"])
print("tried              =", layout_meta["tried"])

# ===================== QCBM =====================
N_LAYERS = 6
EPS_COST = 1e-9
INIT_SCALE = 0.01
SEED = 355

qcbm = MLQcbmCircuit(
    n_qubits=n_qubits,
    n_layers=N_LAYERS,
    name="G_p_density_matrix_noise_single_eval",
    entangler="cz",
    topology=effective_topology,
    backend=AerSimulator(
        method="density_matrix",
        noise_model=real_noise_model,
    ),
    transpile_backend=real_backend,
    noise_model=real_noise_model,
    simulation_method="density_matrix",
    optimization_level=3,
    initial_layout=chosen_layout,
    layout_method="trivial",
    routing_method="none",
    seed_transpiler=1234,
)

summarize_circuit(qcbm._tqc, label="QCBM transpiled density-matrix noisy")


# ===================== Single evaluation =====================
rng = np.random.default_rng(SEED)
x0 = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)

cost_noisy = qcbm.cost_fn(ptg, eps=EPS_COST)

t0 = time.perf_counter()
cost_value = cost_noisy(x0)
eval_time = time.perf_counter() - t0

print("\nSingle noisy density-matrix evaluation")
print("cost(x0)           =", f"{cost_value:.12e}")
print("eval_time_s        =", f"{eval_time:.6f}")
print("n_qubits           =", n_qubits)
print("n_layers           =", N_LAYERS)
print("n_params           =", qcbm.n_params)
print("transpiled_depth   =", qcbm._tqc.depth())
print("transpiled_size    =", qcbm._tqc.size())
print("active_qubits      =", qcbm._active_qubit_indices(qcbm._tqc))
print("count_ops          =", dict(qcbm._tqc.count_ops()))

# ===================== Save =====================
np.savez(
    saving_path,
    cost_value=np.float64(cost_value),
    eval_time_s=np.float64(eval_time),
    theta_init=x0,
    p_target=ptg,
    n_qubits=np.int64(n_qubits),
    n_layers=np.int64(N_LAYERS),
    n_params=np.int64(qcbm.n_params),
    theta_seed=np.int64(SEED),
    init_scale=np.float64(INIT_SCALE),
    eps_cost=np.float64(EPS_COST),
    backend_name=np.array(BACKEND_NAME),
    requested_topology=np.array(LOGICAL_TOPOLOGY),
    effective_topology=np.array(effective_topology),
    chosen_layout=np.array(chosen_layout, dtype=int),
    layout_score=np.float64(layout_score),
    fallback_used=np.bool_(layout_meta["fallback_used"]),
    transpiled_depth=np.int64(qcbm._tqc.depth()),
    transpiled_size=np.int64(qcbm._tqc.size()),
    transpiled_ops=np.array(dict(qcbm._tqc.count_ops()), dtype=object),
    active_qubits=np.array(qcbm._active_qubit_indices(qcbm._tqc), dtype=int),
)

print(f"\nSaved single-eval result to: {saving_path}")