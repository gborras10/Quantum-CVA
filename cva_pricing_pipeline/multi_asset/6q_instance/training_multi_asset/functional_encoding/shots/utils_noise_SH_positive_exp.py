import pathlib
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from qiskit import ClassicalRegister
from qiskit_algorithms.optimizers import SPSA

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)


# Default configuration for utility functions.
BACKEND_NAME = "ibm_basquecountry"
TARGET_THRESHOLD = 1e-10
RELATIVE_EPS = 1e-4
LAMBDA_POS = 10.0
LAMBDA_ZERO = 15.0
LAMBDA_L2_MIX = 25.0
ROBUST_REL_CLIP = 2.5
ROBUST_REL_HUBER_DELTA = 0.6
ROBUST_ZERO_HUBER_DELTA = 0.02
SHOT_SEED = 355
REPEAT_SEED_STRIDE = 10007
USE_STATEVECTOR_WARMSTART = True
WARMSTART_PATH_RELATIVE = (
    "data/multi_asset/6q_instance/quantum/training/crca/positive_exposure/"
    "training_heavy_hex_star.npz"
)
SPSA_LAST_AVG = 40

def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
    dt = datetime.fromisoformat(snapshot_iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _BackendSnapshotView:
    def __init__(self, backend, snapshot_props):
        self._backend = backend
        self._snapshot_props = snapshot_props

    def properties(self, *args, **kwargs):
        return self._snapshot_props

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _inject_missing_frequencies(snapshot_props, fallback_backend, default_frequency_ghz: float = 5.0):
    props_dict = snapshot_props.to_dict()
    fallback_qprops = getattr(getattr(fallback_backend, "target", None), "qubit_properties", None)
    injected = 0

    for qi, q_props in enumerate(props_dict.get("qubits", [])):
        if any(str(p.get("name", "")).lower() == "frequency" for p in q_props):
            continue

        freq_ghz = None
        if fallback_qprops is not None and qi < len(fallback_qprops):
            maybe_freq = getattr(fallback_qprops[qi], "frequency", None)
            if maybe_freq is not None and np.isfinite(maybe_freq):
                maybe_freq = float(maybe_freq)
                freq_ghz = maybe_freq / 1e9 if maybe_freq > 1e6 else maybe_freq

        q_props.append(
            {
                "date": props_dict.get("last_update_date"),
                "name": "frequency",
                "unit": "GHz",
                "value": float(freq_ghz if freq_ghz is not None else default_frequency_ghz),
            }
        )
        injected += 1

    return type(snapshot_props).from_dict(props_dict), injected


def _build_local_coupling_map(full_edges, chosen_layout: list[int]) -> list[list[int]]:
    phys_to_local = {int(q): i for i, q in enumerate(chosen_layout)}
    local_edges: list[list[int]] = []
    for a, b in full_edges:
        if int(a) in phys_to_local and int(b) in phys_to_local:
            local_edges.append([phys_to_local[int(a)], phys_to_local[int(b)]])
    return local_edges


def _subset_backend_properties(snapshot_props, chosen_layout: list[int]):
    props_dict = snapshot_props.to_dict()
    phys_to_local = {int(q): i for i, q in enumerate(chosen_layout)}
    subset_dict = {k: v for k, v in props_dict.items()}
    subset_dict["qubits"] = [props_dict["qubits"][int(q)] for q in chosen_layout]

    subset_gates = []
    for gate in props_dict.get("gates", []):
        qids = [int(q) for q in gate.get("qubits", [])]
        if all(q in phys_to_local for q in qids):
            new_gate = dict(gate)
            new_gate["qubits"] = [phys_to_local[q] for q in qids]
            subset_gates.append(new_gate)
    subset_dict["gates"] = subset_gates
    subset_dict["general_qlists"] = []
    subset_dict["backend_name"] = f"{props_dict.get('backend_name', BACKEND_NAME)}_subset_{len(chosen_layout)}q"
    return type(snapshot_props).from_dict(subset_dict)


def _as_1d_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _evaluate_function_values(crca: CrcaCircuit, theta: np.ndarray, shots: int, seed: int) -> np.ndarray:
    theta = _as_1d_float(theta)
    try:
        out = crca.function_values(theta, shots=shots, seed=seed)
    except TypeError:
        out = crca.function_values(theta, shots=shots)
    return _as_1d_float(out)


def _support_aware_cost(
    f_model: np.ndarray,
    f_target: np.ndarray,
    *,
    pos_mask: np.ndarray | None = None,
    zero_mask: np.ndarray | None = None,
    pos_denom: np.ndarray | None = None,
) -> float:
    if pos_mask is None or zero_mask is None:
        pos_mask = f_target > TARGET_THRESHOLD
        zero_mask = ~pos_mask

    loss = 0.0

    if np.any(pos_mask):
        denom = pos_denom
        if denom is None:
            denom = np.maximum(np.abs(f_target[pos_mask]), RELATIVE_EPS)
        rel_err_sq = ((f_model[pos_mask] - f_target[pos_mask]) / denom) ** 2
        loss += float(LAMBDA_POS * np.mean(rel_err_sq))

    if np.any(zero_mask):
        # L1 penalty is less brittle than squared penalty under shot noise.
        loss += float(LAMBDA_ZERO * np.mean(np.abs(f_model[zero_mask])))

    return float(loss)


def _huber_loss(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * abs_x**2, delta * (abs_x - 0.5 * delta))


def _support_aware_robust_cost(
    f_model: np.ndarray,
    f_target: np.ndarray,
    *,
    pos_mask: np.ndarray | None = None,
    zero_mask: np.ndarray | None = None,
    pos_denom: np.ndarray | None = None,
) -> float:
    if pos_mask is None or zero_mask is None:
        pos_mask = f_target > TARGET_THRESHOLD
        zero_mask = ~pos_mask

    # Keep stage-2 aligned with the validation metric to avoid objective mismatch.
    l2_term = float(np.mean((f_model - f_target) ** 2))
    loss = float(LAMBDA_L2_MIX * l2_term)

    if np.any(pos_mask):
        denom = pos_denom
        if denom is None:
            denom = np.maximum(np.abs(f_target[pos_mask]), RELATIVE_EPS)
        rel_err = (f_model[pos_mask] - f_target[pos_mask]) / denom
        rel_err = np.clip(rel_err, -ROBUST_REL_CLIP, ROBUST_REL_CLIP)
        loss += float(LAMBDA_POS * np.mean(_huber_loss(rel_err, ROBUST_REL_HUBER_DELTA)))

    if np.any(zero_mask):
        zero_vals = np.clip(f_model[zero_mask], -1.0, 1.0)
        loss += float(LAMBDA_ZERO * np.mean(_huber_loss(zero_vals, ROBUST_ZERO_HUBER_DELTA)))

    return float(loss)


def _build_objective(
    crca: CrcaCircuit,
    f_target: np.ndarray,
    *,
    mode: str,
    shots: int,
    eval_repeats: int,
):
    f_target = _as_1d_float(f_target)
    eval_repeats = max(1, int(eval_repeats))

    pos_mask = f_target > TARGET_THRESHOLD
    zero_mask = ~pos_mask
    pos_denom = np.maximum(np.abs(f_target[pos_mask]), RELATIVE_EPS)

    if mode not in {"l2", "support_aware", "support_aware_robust"}:
        raise ValueError("mode must be 'l2', 'support_aware', or 'support_aware_robust'.")

    def objective(theta: np.ndarray) -> float:
        theta_arr = _as_1d_float(theta)
        acc = 0.0
        for k in range(eval_repeats):
            seed_k = int(SHOT_SEED + REPEAT_SEED_STRIDE * k)
            f_model = _evaluate_function_values(crca, theta_arr, shots=shots, seed=seed_k)
            if mode == "l2":
                acc += float(np.mean((f_model - f_target) ** 2))
            elif mode == "support_aware":
                acc += _support_aware_cost(
                    f_model,
                    f_target,
                    pos_mask=pos_mask,
                    zero_mask=zero_mask,
                    pos_denom=pos_denom,
                )
            else:
                acc += _support_aware_robust_cost(
                    f_model,
                    f_target,
                    pos_mask=pos_mask,
                    zero_mask=zero_mask,
                    pos_denom=pos_denom,
                )
        return float(acc / eval_repeats)

    return objective


def _load_warmstart_theta(repo_root: pathlib.Path, n_params_expected: int) -> np.ndarray | None:
    if not USE_STATEVECTOR_WARMSTART:
        return None

    warmstart_path = repo_root / pathlib.Path(WARMSTART_PATH_RELATIVE)
    if not warmstart_path.exists():
        print(f"[INFO] Warm-start file not found: {warmstart_path}")
        return None

    data = np.load(warmstart_path, allow_pickle=True)
    if "theta_star" not in data:
        print("[INFO] Warm-start file exists but has no 'theta_star'; skipping warm-start.")
        return None

    theta = np.asarray(data["theta_star"], dtype=float).ravel()
    if theta.size != int(n_params_expected):
        print(
            "[WARNING] Warm-start theta size mismatch; expected "
            f"{n_params_expected}, got {theta.size}. Ignoring warm-start."
        )
        return None

    print(f"[INFO] Warm-start loaded from: {warmstart_path}")
    return theta


def _run_spsa_stage(
    *,
    stage_name: str,
    objective_mode: str,
    crca: CrcaCircuit,
    f_target: np.ndarray,
    x0: np.ndarray,
    shots: int,
    calibration_shots: int,
    maxiter: int,
    resamplings: int | dict[int, int],
    eval_repeats: int,
    second_order: bool,
    blocking: bool,
    trust_region: bool,
    regularization: float,
    hessian_delay: int,
    calibration_target_magnitude: float | None,
) -> dict[str, Any]:
    objective = _build_objective(
        crca,
        f_target,
        mode=objective_mode,
        shots=shots,
        eval_repeats=eval_repeats,
    )
    objective_calibration = _build_objective(
        crca,
        f_target,
        mode=objective_mode,
        shots=calibration_shots,
        eval_repeats=eval_repeats,
    )
    l2_objective = _build_objective(
        crca,
        f_target,
        mode="l2",
        shots=shots,
        eval_repeats=eval_repeats,
    )

    x0 = np.asarray(x0, dtype=float).copy()
    obj_history: list[float] = [float(objective(x0))]
    l2_history: list[float] = [float(l2_objective(x0))]
    theta_history: list[np.ndarray] = [x0.copy()]

    print(
        f"\n[{stage_name}] mode={objective_mode} | shots={shots} | "
        f"calibration_shots={calibration_shots} | maxiter={maxiter} | "
        f"resamplings={resamplings} | eval_repeats={eval_repeats} | "
        f"target_magnitude={calibration_target_magnitude}"
    )
    print(f"[{stage_name}] Calibrating SPSA hyperparameters...")
    lr, pert = SPSA.calibrate(
        objective_calibration,
        x0,
        target_magnitude=calibration_target_magnitude,
    )

    def callback(nfev, x, fx, step, accepted):
        x_arr = np.asarray(x, dtype=float).copy()
        fx_obj = float(fx)
        fx_l2 = float(l2_objective(x_arr))
        obj_history.append(fx_obj)
        l2_history.append(fx_l2)
        theta_history.append(x_arr)
        print(
            f"[{stage_name} iter {len(obj_history)-1:4d}] "
            f"obj={fx_obj:.6e} | l2={fx_l2:.6e} | "
            f"nfev={int(nfev):6d} | step={float(step):.3e} | accepted={bool(accepted)}"
        )

    spsa_kwargs = dict(
        maxiter=int(maxiter),
        learning_rate=lr,
        perturbation=pert,
        resamplings=resamplings,
        last_avg=SPSA_LAST_AVG,
        second_order=bool(second_order),
        blocking=bool(blocking),
        trust_region=bool(trust_region),
        callback=callback,
    )
    if second_order:
        spsa_kwargs["regularization"] = float(regularization)
        spsa_kwargs["hessian_delay"] = int(hessian_delay)

    opt = SPSA(**spsa_kwargs)

    t0 = time.perf_counter()
    res = opt.minimize(fun=objective, x0=x0)
    elapsed_s = float(time.perf_counter() - t0)

    theta_last = _as_1d_float(res.x)
    if not np.allclose(theta_history[-1], theta_last, rtol=0.0, atol=1e-15):
        obj_history.append(float(objective(theta_last)))
        l2_history.append(float(l2_objective(theta_last)))
        theta_history.append(theta_last.copy())

    obj_arr = np.asarray(obj_history, dtype=float)
    l2_arr = np.asarray(l2_history, dtype=float)
    theta_arr = np.asarray(theta_history, dtype=float)

    idx_best_obj = int(np.argmin(obj_arr))
    idx_best_l2 = int(np.argmin(l2_arr))

    result = {
        "stage_name": stage_name,
        "objective_mode": objective_mode,
        "eval_repeats": int(eval_repeats),
        "second_order": bool(second_order),
        "blocking": bool(blocking),
        "trust_region": bool(trust_region),
        "regularization": float(regularization),
        "hessian_delay": int(hessian_delay),
        "target_magnitude": calibration_target_magnitude,
        "learning_rate": lr,
        "perturbation": pert,
        "elapsed_s": elapsed_s,
        "theta_last": theta_last,
        "theta_best_obj": theta_arr[idx_best_obj].copy(),
        "theta_best_l2": theta_arr[idx_best_l2].copy(),
        "best_obj": float(obj_arr[idx_best_obj]),
        "best_l2": float(l2_arr[idx_best_l2]),
        "obj_history": obj_arr,
        "l2_history": l2_arr,
        "theta_history": theta_arr,
        "result_success": bool(getattr(res, "success", False)),
        "result_message": str(getattr(res, "message", "")),
    }

    print(
        f"[{stage_name}] done in {elapsed_s:.1f}s | "
        f"best_obj={result['best_obj']:.6e} | best_l2={result['best_l2']:.6e}"
    )
    return result


def _select_best_theta_by_recheck(
    crca: CrcaCircuit,
    f_target: np.ndarray,
    theta_history: np.ndarray,
    l2_history: np.ndarray,
    *,
    shots: int,
    top_k: int,
    eval_repeats: int,
) -> tuple[int, float]:
    l2_history = np.asarray(l2_history, dtype=float).reshape(-1)
    theta_history = np.asarray(theta_history, dtype=float)
    if theta_history.shape[0] != l2_history.size:
        raise ValueError("theta_history and l2_history must have the same length.")

    top_k = max(1, min(int(top_k), int(l2_history.size)))
    eval_repeats = max(1, int(eval_repeats))

    # Re-evaluate the top noisy candidates with stronger averaging before selecting theta_star.
    candidate_idx = np.argsort(l2_history)[:top_k]
    candidate_idx = np.unique(np.r_[candidate_idx, [l2_history.size - 1]])

    scorer = _build_objective(
        crca,
        f_target,
        mode="l2",
        shots=shots,
        eval_repeats=eval_repeats,
    )

    best_idx = int(candidate_idx[0])
    best_l2 = float("inf")
    for idx in candidate_idx:
        l2_val = float(scorer(theta_history[int(idx)]))
        if l2_val < best_l2:
            best_l2 = l2_val
            best_idx = int(idx)

    return best_idx, best_l2


def _build_transpiled_measured_eval_circuit(
    crca: CrcaCircuit,
    pass_manager,
):
    qc_meas = crca.qc_eval.copy()
    c_ctrl = ClassicalRegister(crca.n_controls, "c")
    c_a = ClassicalRegister(1, "ca")
    qc_meas.add_register(c_ctrl, c_a)
    qc_meas.measure(crca._control_qubit_indices, c_ctrl)
    qc_meas.measure([crca._ancilla_qubit_index], c_a)
    return pass_manager.run(qc_meas)


def _merge_stage_histories(
    stage1: dict[str, Any],
    stage2: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stage2_starts_from_stage1_tail = np.allclose(
        stage2["theta_history"][0],
        stage1["theta_history"][-1],
        rtol=0.0,
        atol=1e-15,
    )

    if stage2_starts_from_stage1_tail:
        theta_history_arr = np.vstack([stage1["theta_history"], stage2["theta_history"][1:]])
        cost_history_arr = np.r_[stage1["obj_history"], stage2["obj_history"][1:]]
        l2_history_arr = np.r_[stage1["l2_history"], stage2["l2_history"][1:]]
    else:
        theta_history_arr = np.vstack([stage1["theta_history"], stage2["theta_history"]])
        cost_history_arr = np.r_[stage1["obj_history"], stage2["obj_history"]]
        l2_history_arr = np.r_[stage1["l2_history"], stage2["l2_history"]]

    return theta_history_arr, cost_history_arr, l2_history_arr


def _mean_squared_error(values: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((values - target) ** 2))