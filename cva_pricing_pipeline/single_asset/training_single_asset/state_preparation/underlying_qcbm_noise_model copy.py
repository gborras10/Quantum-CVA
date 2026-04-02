import pathlib
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    draw_local_subgraph,
    summarize_circuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    minimize_with_cost_history,
    plot_cost_evolution_cases,
)

# ------------------ Output path setup ------------------
out_path = pathlib.Path(
    "data/single_asset/qcbm/qcbm_training_results_transpile_method_noise_cz_gates_8_layers.npz"
)
out_path.parent.mkdir(parents=True, exist_ok=True)

# ------------------ Target distribution ------------------
ptg: np.ndarray = np.load(
    "data/single_asset/benchmark/run_classical_cva_single_asset.npz"
)["p_target"]

# ------------------ Problem size ------------------
num_qubits_price = 2
num_qubits_time = 2
num_qubits = num_qubits_price + num_qubits_time
TOPOLOGY = "circular"

# ------------------ Backend / layout ------------------
service = QiskitRuntimeService(channel="ibm_cloud")
real_backend = service.backend("ibm_basquecountry")
real_noise_model = NoiseModel.from_backend(real_backend)
empty_noise_model = NoiseModel()

chosen_layout, layout_score, layout_meta = select_best_layout(
    real_backend,
    topology=TOPOLOGY,
    length=num_qubits,
    readout_quantile=0.95,
    local_2q_quantile=0.95,
)

effective_topology = layout_meta["selected_topology"]


def build_bad_noise_model(
    backend,
    *,
    t1_us: float = 35.0,
    t2_us: float = 25.0,
    single_qubit_gate_time_ns: float = 80.0,
    two_qubit_gate_time_ns: float = 450.0,
    p1q: float = 0.008,
    p2q: float = 0.06,
    p_meas_01: float = 0.06,
    p_meas_10: float = 0.08,
) -> NoiseModel:
    """
    Artificially bad noise model for comparison purposes.

    Keeps the same basis gates and coupling structure implied by the transpilation
    backend, but injects substantially worse gate/readout/coherence errors.
    """
    noise_model = NoiseModel()

    backend_config = backend.configuration()
    basis_gates = list(getattr(backend_config, "basis_gates", []))
    coupling_map = list(getattr(backend_config, "coupling_map", []))

    n_qubits_backend = backend_config.n_qubits

    t1 = t1_us * 1e-6
    t2 = t2_us * 1e-6
    tg1 = single_qubit_gate_time_ns * 1e-9
    tg2 = two_qubit_gate_time_ns * 1e-9

    # -------- 1Q errors --------
    err_1q_thermal = thermal_relaxation_error(t1, t2, tg1)
    err_1q_depol = depolarizing_error(p1q, 1)
    err_1q = err_1q_depol.compose(err_1q_thermal)

    one_qubit_basis_candidates = ["rz", "sx", "x", "id"]
    one_qubit_basis = [g for g in one_qubit_basis_candidates if g in basis_gates]

    for gate in one_qubit_basis:
        for q in range(n_qubits_backend):
            noise_model.add_quantum_error(err_1q, gate, [q])

    # -------- 2Q errors --------
    err_2q_thermal = thermal_relaxation_error(t1, t2, tg2).tensor(
        thermal_relaxation_error(t1, t2, tg2)
    )
    err_2q_depol = depolarizing_error(p2q, 2)
    err_2q = err_2q_depol.compose(err_2q_thermal)

    two_qubit_basis_candidates = ["cz", "ecr", "cx"]
    two_qubit_basis = [g for g in two_qubit_basis_candidates if g in basis_gates]

    for gate in two_qubit_basis:
        for edge in coupling_map:
            if len(edge) == 2:
                noise_model.add_quantum_error(err_2q, gate, list(edge))

    # -------- Readout errors --------
    ro_error = ReadoutError(
        [
            [1.0 - p_meas_01, p_meas_01],
            [p_meas_10, 1.0 - p_meas_10],
        ]
    )
    for q in range(n_qubits_backend):
        noise_model.add_readout_error(ro_error, [q])

    return noise_model


bad_noise_model = build_bad_noise_model(real_backend)

# ------------------ Common hyperparameters ------------------
EPS_COST = 1e-12
THETA_SEED = 355
N_ITERS = 50
RHOBEG = 0.5
METHOD = "COBYLA"

rng = np.random.default_rng(THETA_SEED)

qcbm_ref = MLQcbmCircuit(
    n_qubits=num_qubits,
    n_layers=8,
    name="G_p_ref",
    entangler="cz",
    topology=effective_topology,
)
x0 = rng.standard_normal(qcbm_ref.n_params).astype(float)

qcbm_ref_all_to_all = MLQcbmCircuit(
    n_qubits=num_qubits,
    n_layers=8,
    name="G_p_ref_all_to_all",
    entangler="cz",
    topology="all-to-all",
)
x0_all_to_all = rng.standard_normal(qcbm_ref_all_to_all.n_params).astype(float)


def make_qcbm(case_name: str) -> MLQcbmCircuit:
    common = dict(
        n_qubits=num_qubits,
        n_layers=8,
        name=case_name,
        entangler="cz",
        optimization_level=3,
    )

    if case_name == "statevector_ideal_no_transpile":
        return MLQcbmCircuit(
            **common,
            topology=effective_topology,
            backend=AerSimulator(method="statevector"),
            transpile_backend=None,
            noise_model=None,
            simulation_method="statevector",
            initial_layout=None,
        )

    if case_name == "statevector_ideal_no_transpile_all_to_all":
        return MLQcbmCircuit(
            **common,
            topology="all-to-all",
            backend=AerSimulator(method="statevector"),
            transpile_backend=None,
            noise_model=None,
            simulation_method="statevector",
            initial_layout=None,
        )

    if case_name == "statevector_transpiled_no_noise":
        return MLQcbmCircuit(
            **common,
            topology=effective_topology,
            backend=AerSimulator(method="statevector"),
            transpile_backend=real_backend,
            noise_model=empty_noise_model,
            simulation_method="statevector",
            initial_layout=chosen_layout,
            layout_method="trivial",
            routing_method="none",
            seed_transpiler=1234,
        )

    if case_name == "density_matrix_transpiled_no_noise":
        return MLQcbmCircuit(
            **common,
            topology=effective_topology,
            backend=AerSimulator(method="density_matrix"),
            transpile_backend=real_backend,
            noise_model=empty_noise_model,
            simulation_method="density_matrix",
            initial_layout=chosen_layout,
            layout_method="trivial",
            routing_method="none",
            seed_transpiler=1234,
        )

    if case_name == "density_matrix_transpiled_with_noise":
        return MLQcbmCircuit(
            **common,
            topology=effective_topology,
            backend=AerSimulator(
                method="density_matrix",
                noise_model=real_noise_model,
            ),
            transpile_backend=real_backend,
            noise_model=real_noise_model,
            simulation_method="density_matrix",
            initial_layout=chosen_layout,
            layout_method="trivial",
            routing_method="none",
            seed_transpiler=1234,
        )

    if case_name == "density_matrix_transpiled_with_bad_noise":
        return MLQcbmCircuit(
            **common,
            topology=effective_topology,
            backend=AerSimulator(
                method="density_matrix",
                noise_model=bad_noise_model,
            ),
            transpile_backend=real_backend,
            noise_model=bad_noise_model,
            simulation_method="density_matrix",
            initial_layout=chosen_layout,
            layout_method="trivial",
            routing_method="none",
            seed_transpiler=1234,
        )

    raise ValueError(f"Unknown case: {case_name}")


def get_x0_for_case(case_name: str) -> np.ndarray:
    if case_name == "statevector_ideal_no_transpile_all_to_all":
        return x0_all_to_all.copy()
    return x0.copy()


def run_case(case_name: str) -> dict:
    qcbm = make_qcbm(case_name)
    summarize_circuit(qcbm._tqc, label=case_name)

    x0_case = get_x0_for_case(case_name)
    cost = qcbm.cost_fn(ptg, eps=EPS_COST)

    t0 = time.perf_counter()
    c0 = cost(x0_case)
    eval_time = time.perf_counter() - t0

    t_start = time.perf_counter()
    res, cost_history = minimize_with_cost_history(
        cost,
        x0=x0_case,
        minimize_fn=minimize,
        method=METHOD,
        options={"maxiter": int(N_ITERS), "rhobeg": RHOBEG, "disp": False},
    )
    elapsed = time.perf_counter() - t_start

    theta_star = np.asarray(res.x, dtype=float)
    p0 = qcbm.probabilities(x0_case)
    p_star = qcbm.probabilities(theta_star)
    metrics = qcbm.metrics(ptg, p_star)

    target_entropy = -np.sum(ptg * np.log(np.maximum(ptg, EPS_COST)))
    rescaled = np.maximum(np.asarray(cost_history) - target_entropy, 1e-12)

    return {
        "name": case_name,
        "initial_cost": float(c0),
        "elapsed_time_s": float(elapsed),
        "eval_time_s": float(eval_time),
        "success": bool(res.success),
        "message": str(res.message),
        "theta_init": np.asarray(x0_case, dtype=float),
        "theta_star": theta_star,
        "p_init": p0,
        "p_star": p_star,
        "cost_history": np.asarray(cost_history, dtype=float),
        "rescaled_cost_history": rescaled,
        "metrics": metrics,
        "ops": dict(qcbm._tqc.count_ops()),
        "depth": int(qcbm._tqc.depth()),
        "active_qubits": np.array(qcbm._active_qubit_indices(qcbm._tqc), dtype=int),
        "topology": qcbm.topology if hasattr(qcbm, "topology") else None,
        "n_params": int(len(qcbm.theta)),
    }


case_names = [
    "statevector_ideal_no_transpile",
    "statevector_ideal_no_transpile_all_to_all",
    "statevector_transpiled_no_noise",
    "density_matrix_transpiled_no_noise",
    "density_matrix_transpiled_with_noise",
    "density_matrix_transpiled_with_bad_noise",
]

results = [run_case(name) for name in case_names]


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


rows = []
for r in results:
    row = {}
    for key, value in r.items():
        serializable_value = _to_serializable(value)
        if isinstance(serializable_value, (dict, list)):
            row[key] = json.dumps(serializable_value)
        else:
            row[key] = serializable_value
    rows.append(row)

results_df = pd.DataFrame(rows)

# Save dataframe in the exact out_path configured above.
np.savez_compressed(
    out_path,
    records=np.array(results_df.to_dict(orient="records"), dtype=object),
    columns=np.array(results_df.columns.tolist(), dtype=object),
)

print(f"Results DataFrame saved to: {out_path.resolve()}")

print("\nSummary of the artificially bad noise:")
print("  T1 =", 35.0, "us")
print("  T2 =", 25.0, "us")
print("  1Q depolarizing error =", 0.008)
print("  2Q depolarizing error =", 0.06)
print("  P(0->1) readout =", 0.06)
print("  P(1->0) readout =", 0.08)

for r in results:
    print(
        f"{r['name']}: "
        f"success={r['success']} | "
        f"init_cost={r['initial_cost']:.6e} | "
        f"final_KL={r['metrics']['kl']:.6e} | "
        f"eval_time={r['eval_time_s']:.3f}s | "
        f"elapsed={r['elapsed_time_s']:.2f}s | "
        f"depth={r['depth']} | "
        f"n_params={r['n_params']} | "
        f"active_qubits={r['active_qubits'].tolist()}"
    )

fig = plot_cost_evolution_cases(
    results=results,
    y_key="rescaled_cost_history",
    title="Single-Asset QCBM Training Curves",
    ylabel="KL(p_target || p_theta)",
    smooth=False,
    marker_every=None,
    save_path=None,
)

plt.show()