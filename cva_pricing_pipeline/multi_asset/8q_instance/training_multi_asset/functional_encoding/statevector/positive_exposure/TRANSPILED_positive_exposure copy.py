import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit,
)

# ===================== Global Configuration =====================

BACKEND_NAME = "ibm_basquecountry"
SEARCH_TOPOLOGY = "linear"
TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 2
N_PRICE = 6
N_LAYERS = 1

THETA_SEED = 42
INIT_SCALE = 0.2

# "l2" -> loss original de Alcazar / wrapper
# "support_aware" -> pérdida que penaliza más el soporte no nulo
LOSS_MODE = "l2"

TARGET_THRESHOLD = 1e-10
RELATIVE_EPS = 1e-4
LAMBDA_POS = 50.0
LAMBDA_ZERO = 1.0

# COBYLA
COBYLA_MAXITER = 1000
COBYLA_TOL = 1e-6
COBYLA_RHOBEG = 0.5


def get_backend_coupling_map(backend):
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is not None:
        return coupling_map

    target = getattr(backend, "target", None)
    if target is not None and hasattr(target, "build_coupling_map"):
        return target.build_coupling_map()

    raise RuntimeError("Could not extract a coupling map from the backend.")


def get_backend_basis_gates(backend) -> list[str]:
    raw_ops = list(getattr(backend, "operation_names", []))
    if not raw_ops:
        target = getattr(backend, "target", None)
        if target is not None:
            raw_ops = list(target.operation_names)

    exclude = {
        "measure",
        "reset",
        "delay",
        "barrier",
        "if_else",
        "for_loop",
        "while_loop",
        "switch_case",
        "box",
        "store",
        "snapshot",
    }
    basis_gates = [name for name in raw_ops if name not in exclude]
    if not basis_gates:
        raise RuntimeError("Could not extract basis_gates from the backend.")
    return basis_gates


def build_local_coupling_map(coupling_map, chosen_layout: list[int]) -> CouplingMap:
    if len(chosen_layout) == 0:
        raise RuntimeError("chosen_layout is empty.")

    physical_to_local = {physical: local for local, physical in enumerate(chosen_layout)}
    local_edges = []
    for src, dst in coupling_map.get_edges():
        if src in physical_to_local and dst in physical_to_local:
            local_edges.append((physical_to_local[src], physical_to_local[dst]))

    if not local_edges:
        raise RuntimeError(
            "No coupling edges were found inside the selected physical layout. "
            "Try a different topology or layout length."
        )

    return CouplingMap(local_edges)


def _iter_circuit_data(qc):
    for item in qc.data:
        if hasattr(item, "operation"):
            yield item.operation, item.qubits, item.clbits
        else:
            yield item[0], item[1], item[2]


def extract_qubit_indices_from_transpiled_measured(qc_meas, n_controls: int):
    cregs = {cr.name: cr for cr in qc_meas.cregs}
    if "c" in cregs and "ca" in cregs:
        c_ctrl = cregs["c"]
        c_a = cregs["ca"]
        ctrl_clbit_indices = [qc_meas.clbits.index(c_ctrl[i]) for i in range(len(c_ctrl))]
        a_clbit_index = qc_meas.clbits.index(c_a[0])
    else:
        expected = n_controls + 1
        if len(qc_meas.clbits) < expected:
            raise RuntimeError(
                "Could not infer classical registers for controls/ancilla "
                f"(need at least {expected} clbits, got {len(qc_meas.clbits)})."
            )
        ctrl_clbit_indices = list(range(n_controls))
        a_clbit_index = n_controls

    measured_qubit_for_clbit: dict[int, int] = {}
    for op, qargs, cargs in _iter_circuit_data(qc_meas):
        if op.name != "measure":
            continue
        qubit_idx = qc_meas.qubits.index(qargs[0])
        clbit_idx = qc_meas.clbits.index(cargs[0])
        measured_qubit_for_clbit[clbit_idx] = qubit_idx

    missing_ctrl = [idx for idx in ctrl_clbit_indices if idx not in measured_qubit_for_clbit]
    if missing_ctrl:
        raise RuntimeError(f"Missing measure mapping for control clbits {missing_ctrl}.")

    if a_clbit_index not in measured_qubit_for_clbit:
        raise RuntimeError("Missing measure mapping for ancilla clbit.")

    ctrl_qubit_indices = [measured_qubit_for_clbit[idx] for idx in ctrl_clbit_indices]
    ancilla_qubit_index = measured_qubit_for_clbit[a_clbit_index]

    return ctrl_qubit_indices, ancilla_qubit_index


def bind_transpiled_eval_circuit(tqc_eval, theta_vec, original_theta):
    theta_vec = np.asarray(theta_vec, dtype=float).ravel()
    full_bind_map = {
        original_theta[i]: float(theta_vec[i])
        for i in range(len(original_theta))
    }
    present_params = set(tqc_eval.parameters)
    filtered_bind_map = {
        p: v for p, v in full_bind_map.items()
        if p in present_params
    }
    return tqc_eval.assign_parameters(filtered_bind_map, inplace=False)


def build_transpiled_function_values(
    tqc_eval,
    ctrl_qubit_indices,
    ancilla_qubit_index,
    original_theta,
    n_controls,
):
    dim_ctrl = 1 << n_controls

    def transpiled_function_values(theta_vec: np.ndarray) -> np.ndarray:
        qc_bound = bind_transpiled_eval_circuit(tqc_eval, theta_vec, original_theta)
        sv = Statevector.from_instruction(qc_bound)
        probs = np.asarray(sv.probabilities(), dtype=float)

        num = np.zeros(dim_ctrl, dtype=float)
        den = np.zeros(dim_ctrl, dtype=float)

        for basis_idx, pj in enumerate(probs):
            i_val = 0
            for local_idx, q_idx in enumerate(ctrl_qubit_indices):
                bit = (basis_idx >> q_idx) & 1
                i_val |= bit << local_idx

            a_bit = (basis_idx >> ancilla_qubit_index) & 1

            den[i_val] += pj
            if a_bit == 1:
                num[i_val] += pj

        out = np.zeros(dim_ctrl, dtype=float)
        mask = den > 0.0
        out[mask] = num[mask] / den[mask]
        return out

    return transpiled_function_values


def build_support_aware_cost(function_values_fn, f_target):
    pos_mask = f_target > TARGET_THRESHOLD
    zero_mask = ~pos_mask

    def cost_fn(theta_vec: np.ndarray) -> float:
        fx = np.asarray(function_values_fn(np.asarray(theta_vec, dtype=float)), dtype=float).reshape(-1)

        pos_term = 0.0
        zero_term = 0.0

        if np.any(pos_mask):
            rel_diff = (fx[pos_mask] - f_target[pos_mask]) / (f_target[pos_mask] + RELATIVE_EPS)
            pos_term = float(np.mean(rel_diff * rel_diff))

        if np.any(zero_mask):
            zero_term = float(np.mean(fx[zero_mask] * fx[zero_mask]))

        return LAMBDA_POS * pos_term + LAMBDA_ZERO * zero_term

    return cost_fn, pos_mask, zero_mask


def main() -> None:
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "8q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    v_t = benchmark["v_joint_t"]
    c_v = float(benchmark["C_v"])
    f_target_2d = np.asarray(v_t / c_v, dtype=float)
    f_target = f_target_2d.reshape(-1)

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "8q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "positive_exposure"
        / "positive_exposure_standard_transpiled_hardware_statevector.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="standard",
        name="crca_positive_exposure_standard_transpiled_hardware",
    )

    print("Original logical circuits")
    summarize_circuit(crca.qc)
    summarize_circuit(crca.qc_eval)

    service = QiskitRuntimeService(channel="ibm_cloud")
    backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    chosen_layout, layout_score, layout_meta = select_best_layout(
        backend,
        topology=SEARCH_TOPOLOGY,
        length=crca.qc.num_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
    )

    print("Selected layout:", chosen_layout)
    print("Layout score:", layout_score)
    print("Layout fallback used:", layout_meta["fallback_used"])
    if layout_meta["tried"]:
        print("Layout tries:", layout_meta["tried"])

    basis_gates = get_backend_basis_gates(backend)
    backend_coupling_map = get_backend_coupling_map(backend)
    local_coupling_map = build_local_coupling_map(
        backend_coupling_map,
        chosen_layout,
    )
    local_initial_layout = list(range(crca.qc.num_qubits))

    print("Using restricted local coupling map with", len(local_coupling_map.get_edges()), "edges")

    pm = generate_preset_pass_manager(
        optimization_level=TRANSPILATION_OPT_LEVEL,
        basis_gates=basis_gates,
        coupling_map=local_coupling_map,
        initial_layout=local_initial_layout,
        seed_transpiler=SEED_TRANSPILER,
    )

    tqc_ansatz = pm.run(crca.qc)
    tqc_eval = pm.run(crca.qc_eval)
    tqc_eval_meas = pm.run(crca._qc_eval_meas)

    print("Transpiled circuits against hardware basis + connectivity")
    summarize_circuit(tqc_ansatz)
    summarize_circuit(tqc_eval)

    ctrl_qubit_indices, ancilla_qubit_index = extract_qubit_indices_from_transpiled_measured(
        tqc_eval_meas,
        crca.n_controls,
    )

    print("Control qubit indices in transpiled eval circuit:", ctrl_qubit_indices)
    print("Ancilla qubit index in transpiled eval circuit:", ancilla_qubit_index)

    transpiled_function_values = build_transpiled_function_values(
        tqc_eval=tqc_eval,
        ctrl_qubit_indices=ctrl_qubit_indices,
        ancilla_qubit_index=ancilla_qubit_index,
        original_theta=crca.theta,
        n_controls=crca.n_controls,
    )

    rng = np.random.default_rng(THETA_SEED)
    theta = INIT_SCALE * rng.standard_normal(crca.n_params)
    theta_init = theta.copy()

    f0_statevector = np.asarray(transpiled_function_values(theta), dtype=float).reshape(-1)

    if LOSS_MODE == "l2":
        def objective(theta_vec: np.ndarray) -> float:
            fx = np.asarray(transpiled_function_values(theta_vec), dtype=float).reshape(-1)
            diff = fx - f_target
            return float(np.mean(diff * diff))

        cost_label = "L2 loss"
        metadata_loss = {
            "loss_name": "l2",
        }
        print("Using L2 loss on transpiled hardware-constrained circuit")
    elif LOSS_MODE == "support_aware":
        objective, pos_mask, zero_mask = build_support_aware_cost(
            transpiled_function_values,
            f_target,
        )
        cost_label = "Support-aware loss"
        metadata_loss = {
            "loss_name": "support_aware_relative_plus_zero_penalty",
            "target_threshold": TARGET_THRESHOLD,
            "relative_eps": RELATIVE_EPS,
            "lambda_pos": LAMBDA_POS,
            "lambda_zero": LAMBDA_ZERO,
            "n_positive_support_bins": int(np.count_nonzero(pos_mask)),
            "n_zero_support_bins": int(np.count_nonzero(zero_mask)),
        }
        print(
            "Using support-aware loss with "
            f"lambda_pos={LAMBDA_POS:.1f}, lambda_zero={LAMBDA_ZERO:.1f}, "
            f"relative_eps={RELATIVE_EPS:.1e}"
        )
    else:
        raise ValueError("LOSS_MODE must be 'l2' or 'support_aware'.")

    eval_cost_history: list[float] = []
    iter_cost_history: list[float] = []
    eval_counter = 0

    best_loss = float("inf")
    best_theta = theta.copy()

    def eval_cost(x: np.ndarray) -> float:
        nonlocal eval_counter, best_loss, best_theta

        x = np.asarray(x, dtype=float)
        fx = float(objective(x))

        eval_counter += 1
        eval_cost_history.append(fx)

        if fx < best_loss:
            best_loss = fx
            best_theta = x.copy()

        print(f"[eval {eval_counter:04d}][COBYLA] cost = {fx:.8e}")
        return fx

    def record_iter(xk: np.ndarray) -> None:
        xk = np.asarray(xk, dtype=float)
        fx = float(objective(xk))
        iter_cost_history.append(fx)
        print(f"[iter {len(iter_cost_history)-1:04d}][COBYLA] iter_cost = {fx:.8e}")

    initial_loss = float(objective(theta))
    iter_cost_history.append(initial_loss)
    best_loss = initial_loss
    best_theta = theta.copy()

    print(f"Initial {cost_label.lower()} = {initial_loss:.8e}")
    print("Training standard CRCA transpiled to hardware basis/connectivity (exact statevector on transpiled circuit)")

    t0 = time.perf_counter()

    res_cobyla = minimize(
        eval_cost,
        x0=theta,
        method="COBYLA",
        callback=record_iter,
        options={
            "maxiter": COBYLA_MAXITER,
            "tol": COBYLA_TOL,
            "rhobeg": COBYLA_RHOBEG,
            "disp": False,
        },
    )

    elapsed = time.perf_counter() - t0

    theta_last = np.asarray(res_cobyla.x, dtype=float)
    final_loss = float(objective(theta_last))
    iter_cost_history.append(final_loss)

    if final_loss < best_loss:
        best_loss = final_loss
        best_theta = theta_last.copy()

    theta_best = best_theta.copy()
    f_star_statevector = np.asarray(transpiled_function_values(theta_best), dtype=float).reshape(-1)

    cost_history_arr = np.array(iter_cost_history, dtype=float)
    eval_cost_history_arr = np.array(eval_cost_history, dtype=float)

    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(np.r_[True, best_so_far[1:] < best_so_far[:-1]])

    print(f"COBYLA finished | success = {res_cobyla.success} | message = {res_cobyla.message}")
    print(f"COBYLA final {cost_label.lower()} = {final_loss:.8e}")
    print(f"Best {cost_label.lower()} observed = {best_loss:.8e}")

    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0_statevector,
        after=f_star_statevector,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel=cost_label,
        title_before="Before training",
        title_after="After training",
        cost_log_x=False,
        cost_log_y=True,
    )

    plt.show()

    metadata = {
        "model": "CRCA",
        "task": "positive_exposure",
        "ansatz_type": "standard",
        "training_mode": "transpiled_hardware_constrained_statevector",
        "ancilla_observable": "P(a=1 | control=i)",
        "backend_name": BACKEND_NAME,
        "search_topology": SEARCH_TOPOLOGY,
        "basis_gates": np.array(basis_gates, dtype=object),
        "chosen_layout": np.array(chosen_layout, dtype=int),
        "local_coupling_edges": np.array(local_coupling_map.get_edges(), dtype=object),
        "layout_score": float(layout_score),
        "layout_fallback_used": bool(layout_meta["fallback_used"]),
        "transpile_optimization_level": TRANSPILATION_OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_layers": N_LAYERS,
        "n_controls": crca.n_controls,
        "n_parameters": crca.n_params,
        "optimizer": "COBYLA",
        "optimizer_library": "scipy",
        "init_scale": INIT_SCALE,
        "cobyla_maxiter": COBYLA_MAXITER,
        "cobyla_tol": COBYLA_TOL,
        "cobyla_rhobeg": COBYLA_RHOBEG,
        "shots": None,
        "stochastic_cost": False,
        "theta_seed": THETA_SEED,
        "result_success": bool(res_cobyla.success),
        "result_message": str(res_cobyla.message),
        "result_nfev": int(res_cobyla.nfev),
        "best_eval_cost_observed": float(np.min(eval_cost_history_arr)) if eval_cost_history_arr.size else float(initial_loss),
        "best_iter_cost_observed": float(best_loss),
        "transpiled_ansatz_depth": int(tqc_ansatz.depth()),
        "transpiled_ansatz_size": int(tqc_ansatz.size()),
        "transpiled_eval_depth": int(tqc_eval.depth()),
        "transpiled_eval_size": int(tqc_eval.size()),
        "ctrl_qubit_indices": np.array(ctrl_qubit_indices, dtype=int),
        "ancilla_qubit_index": int(ancilla_qubit_index),
        "note": (
            "Standard CRCA trained on an exact statevector of the transpiled circuit, "
            "compiled with preset pass manager over a local coupling-map restriction "
            "induced by the selected physical layout."
        ),
    }
    metadata.update(metadata_loss)

    np.savez(
        out_path,
        theta_star=theta_best,
        theta_last=theta_last,
        theta_init=theta_init,
        cost_history=cost_history_arr,
        eval_cost_history=eval_cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        f_target=f_target,
        f_target_2d=f_target_2d,
        f_init_statevector=f0_statevector,
        f_star_statevector=f_star_statevector,
        elapsed_time=np.float64(elapsed),
        best_cost=np.float64(best_loss),
        final_cost=np.float64(final_loss),
        C_v=np.float64(c_v),
        n_iters=np.int64(len(iter_cost_history) - 1),
        theta_seed=np.int64(THETA_SEED),
        transpiled_ansatz_ops=np.array(dict(tqc_ansatz.count_ops()), dtype=object),
        transpiled_eval_ops=np.array(dict(tqc_eval.count_ops()), dtype=object),
        metadata=np.array(metadata, dtype=object),
    )

    print("Training finished")
    print("Elapsed time:", elapsed)
    print("Total evaluations:", eval_counter)
    print("COBYLA nfev:", res_cobyla.nfev)
    print("Best iterate cost:", best_loss)


if __name__ == "__main__":
    main()
