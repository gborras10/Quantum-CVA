import pathlib
import time
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, qpy
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService

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
from utils_noise_SH_positive_exp import (
    _BackendSnapshotView,
    _as_1d_float,
    _build_local_coupling_map,
    _build_objective,
    _build_transpiled_measured_eval_circuit,
    _evaluate_function_values,
    _inject_missing_frequencies,
    _load_warmstart_theta,
    _mean_squared_error,
    _merge_stage_histories,
    _parse_snapshot_datetime,
    _run_spsa_stage,
    _select_best_theta_by_recheck,
    _subset_backend_properties,
)


BACKEND_NAME = "ibm_basquecountry"
REQUESTED_TOPOLOGY = "heavy_hex_star"
TRANSPILATION_OPT_LEVEL = 3

M_TIME, N_PRICE, N_LAYERS = 2, 4, 2

TARGET_THRESHOLD = 1e-10
RELATIVE_EPS = 1e-4
LAMBDA_POS = 10.0
LAMBDA_ZERO = 15.0

INIT_SCALE = 0.01
SEED = 12
SHOT_SEED = 355
REPEAT_SEED_STRIDE = 10007
SEED_TRANSPILER = 1234
SIMULATOR_SEED = 20260407

# High-quality external optimization (ansatz untouched)
SHOTS = 60000
STAGE1_CALIBRATION_SHOTS = SHOTS
STAGE2_CALIBRATION_SHOTS = SHOTS

STAGE1_MODE = "l2"
STAGE2_MODE = "support_aware"

STAGE1_MAXITER = 120
STAGE2_MAXITER = 150

STAGE1_RESAMPLINGS: int | dict[int, int] = {0: 3, 40: 4, 90: 5}
STAGE2_RESAMPLINGS: int | dict[int, int] = {0: 4, 120: 6, 280: 8}

STAGE1_EVAL_REPEATS = 2
STAGE2_EVAL_REPEATS = 3

STAGE1_TARGET_MAGNITUDE = 0.08
STAGE2_TARGET_MAGNITUDE = 0.05

SPSA_LAST_AVG = 40
STAGE1_SECOND_ORDER = True
STAGE2_SECOND_ORDER = False
STAGE1_BLOCKING = True
STAGE2_BLOCKING = True
STAGE1_TRUST_REGION = True
STAGE2_TRUST_REGION = True
STAGE1_REGULARIZATION = 0.02
STAGE2_REGULARIZATION = 0.0
STAGE1_HESSIAN_DELAY = 40
STAGE2_HESSIAN_DELAY = 0

# Recommended default after observing support-aware stage-2 plateau under noise.
USE_TWO_STAGE = False

SINGLE_STAGE_MODE = "l2"
SINGLE_STAGE_MAXITER = STAGE1_MAXITER + STAGE2_MAXITER
SINGLE_STAGE_CALIBRATION_SHOTS = SHOTS
SINGLE_STAGE_RESAMPLINGS: int | dict[int, int] = {0: 3, 120: 4, 260: 5, 420: 6}
SINGLE_STAGE_EVAL_REPEATS = 2
SINGLE_STAGE_TARGET_MAGNITUDE = 0.08
SINGLE_STAGE_SECOND_ORDER = True
SINGLE_STAGE_BLOCKING = True
SINGLE_STAGE_TRUST_REGION = True
SINGLE_STAGE_REGULARIZATION = 0.02
SINGLE_STAGE_HESSIAN_DELAY = 60

INIT_SELECTION_EVAL_REPEATS = 5
POSTSELECT_TOPK = 12
POSTSELECT_EVAL_REPEATS = 5

ROBUST_REL_CLIP = 2.5
ROBUST_REL_HUBER_DELTA = 0.6
ROBUST_ZERO_HUBER_DELTA = 0.02
LAMBDA_L2_MIX = 25.0

USE_STATEVECTOR_WARMSTART = True
WARMSTART_PATH_RELATIVE = (
    "data/multi_asset/6q_instance/quantum/training/crca/positive_exposure/"
    "training_heavy_hex_star.npz"
)

THERMAL_RELAXATION_REQUESTED = True
CHECKPOINT_TOL = 1e-15
NOISE_SNAPSHOT_ISO_UTC = "2026-04-07T12:10:00+00:00"

def main() -> None:
    repo_root = next(
        p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
    )
    data = np.load(
        repo_root / "data" / "multi_asset" / "6q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )
    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "positive_exposure"
        / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c_v = float(data["C_v"])
    f_target_2d = np.asarray(data["v_joint_t"], dtype=float) / c_v
    f_target = f_target_2d.reshape(-1)

    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)
    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    backend_props = real_backend.properties(datetime=snapshot_dt_utc)
    if backend_props is None:
        raise RuntimeError(f"Could not load backend properties at {NOISE_SNAPSHOT_ISO_UTC}.")

    logical_crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type=REQUESTED_TOPOLOGY,
    )
    n_total_logical_qubits = int(logical_crca.qc.num_qubits)
    snapshot_view = _BackendSnapshotView(real_backend, backend_props)

    chosen_layout, layout_score, layout_meta = select_best_layout(
        snapshot_view,
        topology=REQUESTED_TOPOLOGY,
        length=n_total_logical_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
    )
    effective_topology = layout_meta["selected_topology"]

    full_coupling = getattr(real_backend, "coupling_map", None)
    if full_coupling is None:
        full_coupling = real_backend.configuration().coupling_map
    full_edges = list(full_coupling.get_edges()) if hasattr(full_coupling, "get_edges") else list(full_coupling)

    local_layout = list(range(n_total_logical_qubits))
    local_coupling_map = _build_local_coupling_map(full_edges, chosen_layout)
    if not local_coupling_map:
        raise RuntimeError("Local coupling map is empty for the selected layout.")

    props_for_noise = backend_props
    injected_frequency_count = 0
    if THERMAL_RELAXATION_REQUESTED:
        props_for_noise, injected_frequency_count = _inject_missing_frequencies(backend_props, real_backend)

    subset_props = _subset_backend_properties(props_for_noise, chosen_layout)
    thermal_relaxation_effective = bool(THERMAL_RELAXATION_REQUESTED)
    noise_model_build = "snapshot_backend_properties"
    try:
        noise_model = NoiseModel.from_backend_properties(
            subset_props,
            thermal_relaxation=THERMAL_RELAXATION_REQUESTED,
        )
    except Exception as exc:
        print("[WARNING] thermal_relaxation=True failed on snapshot; falling back to False.")
        print(f"[WARNING] {exc}")
        thermal_relaxation_effective = False
        noise_model_build = "snapshot_backend_properties_without_thermal_relaxation"
        noise_model = NoiseModel.from_backend_properties(subset_props, thermal_relaxation=False)

    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=local_coupling_map,
        seed_simulator=SIMULATOR_SEED,
    )

    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type=effective_topology,
        name="crca_positive_exposure_heavy_hex_star_shots_backend_noise_snapshot",
    )

    pm = generate_preset_pass_manager(
        backend=noisy_backend,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        initial_layout=local_layout,
        seed_transpiler=SEED_TRANSPILER,
        approximation_degree=1.0,
    )
    tqc_ansatz = pm.run(crca.qc)
    tqc_eval = pm.run(crca.qc_eval)

    tqc_eval_meas_parametric = _build_transpiled_measured_eval_circuit(crca, pm)

    crca._backend = noisy_backend
    crca._tqc_eval_meas = tqc_eval_meas_parametric
    crca._tqc_eval_meas_param_set = set(tqc_eval_meas_parametric.parameters)
    crca._n_clbits = len(tqc_eval_meas_parametric.clbits)
    crca._ctrl_clbit_indices, crca._a_clbit_index = crca._extract_clbit_indices(tqc_eval_meas_parametric)
    crca._tqc = tqc_ansatz

    summarize_circuit(tqc_ansatz, label="CRCA ansatz transpiled")
    summarize_circuit(tqc_eval, label="CRCA eval transpiled")
    summarize_circuit(tqc_eval_meas_parametric, label="CRCA eval+measure transpiled")

    l2_objective_for_init_select = _build_objective(
        crca,
        f_target,
        mode="l2",
        shots=SHOTS,
        eval_repeats=INIT_SELECTION_EVAL_REPEATS,
    )

    rng = np.random.default_rng(SEED)
    theta0_random = INIT_SCALE * rng.standard_normal(crca.n_params).astype(float)
    theta0_warm = _load_warmstart_theta(repo_root, crca.n_params)
    l2_random = float(l2_objective_for_init_select(theta0_random))
    theta0 = theta0_random.copy()

    warmstart_used = False
    warmstart_l2 = float("nan")
    if theta0_warm is not None:
        warmstart_l2 = float(l2_objective_for_init_select(theta0_warm))
        if warmstart_l2 <= l2_random:
            theta0 = theta0_warm.copy()
            warmstart_used = True

    print("\nOptimization setup (high-quality mode):")
    if USE_TWO_STAGE:
        print(f"shots={SHOTS}, stage1_iters={STAGE1_MAXITER}, stage2_iters={STAGE2_MAXITER}")
        print(f"stage1_resamplings={STAGE1_RESAMPLINGS}, stage1_eval_repeats={STAGE1_EVAL_REPEATS}")
        print(f"stage2_resamplings={STAGE2_RESAMPLINGS}, stage2_eval_repeats={STAGE2_EVAL_REPEATS}")
        print(
            "stage1: "
            f"blocking={STAGE1_BLOCKING}, second_order={STAGE1_SECOND_ORDER}, trust_region={STAGE1_TRUST_REGION}"
        )
        print(
            "stage2: "
            f"blocking={STAGE2_BLOCKING}, second_order={STAGE2_SECOND_ORDER}, trust_region={STAGE2_TRUST_REGION}"
        )
    else:
        print(f"shots={SHOTS}, single_stage_iters={SINGLE_STAGE_MAXITER}")
        print(
            f"single_stage_resamplings={SINGLE_STAGE_RESAMPLINGS}, "
            f"single_stage_eval_repeats={SINGLE_STAGE_EVAL_REPEATS}"
        )
        print(
            "single_stage: "
            f"blocking={SINGLE_STAGE_BLOCKING}, second_order={SINGLE_STAGE_SECOND_ORDER}, "
            f"trust_region={SINGLE_STAGE_TRUST_REGION}"
        )
    print(f"shot_seed={SHOT_SEED}")
    print(f"l2(theta0_random, repeats={INIT_SELECTION_EVAL_REPEATS})={l2_random:.6e}")
    if theta0_warm is not None:
        print(
            f"l2(theta0_warm, repeats={INIT_SELECTION_EVAL_REPEATS})={warmstart_l2:.6e} "
            f"| warmstart_used={warmstart_used}"
        )

    if USE_TWO_STAGE:
        stage1 = _run_spsa_stage(
            stage_name="stage1",
            objective_mode=STAGE1_MODE,
            crca=crca,
            f_target=f_target,
            x0=theta0,
            shots=SHOTS,
            calibration_shots=STAGE1_CALIBRATION_SHOTS,
            maxiter=STAGE1_MAXITER,
            resamplings=STAGE1_RESAMPLINGS,
            eval_repeats=STAGE1_EVAL_REPEATS,
            second_order=STAGE1_SECOND_ORDER,
            blocking=STAGE1_BLOCKING,
            trust_region=STAGE1_TRUST_REGION,
            regularization=STAGE1_REGULARIZATION,
            hessian_delay=STAGE1_HESSIAN_DELAY,
            calibration_target_magnitude=STAGE1_TARGET_MAGNITUDE,
        )

        stage2 = _run_spsa_stage(
            stage_name="stage2",
            objective_mode=STAGE2_MODE,
            crca=crca,
            f_target=f_target,
            x0=stage1["theta_best_l2"],
            shots=SHOTS,
            calibration_shots=STAGE2_CALIBRATION_SHOTS,
            maxiter=STAGE2_MAXITER,
            resamplings=STAGE2_RESAMPLINGS,
            eval_repeats=STAGE2_EVAL_REPEATS,
            second_order=STAGE2_SECOND_ORDER,
            blocking=STAGE2_BLOCKING,
            trust_region=STAGE2_TRUST_REGION,
            regularization=STAGE2_REGULARIZATION,
            hessian_delay=STAGE2_HESSIAN_DELAY,
            calibration_target_magnitude=STAGE2_TARGET_MAGNITUDE,
        )

        theta_history_arr, cost_history_arr, l2_history_arr = _merge_stage_histories(stage1, stage2)
        elapsed_time = float(stage1["elapsed_s"] + stage2["elapsed_s"])
        theta_last = _as_1d_float(stage2["theta_last"])
    else:
        stage1 = _run_spsa_stage(
            stage_name="single_l2",
            objective_mode=SINGLE_STAGE_MODE,
            crca=crca,
            f_target=f_target,
            x0=theta0,
            shots=SHOTS,
            calibration_shots=SINGLE_STAGE_CALIBRATION_SHOTS,
            maxiter=SINGLE_STAGE_MAXITER,
            resamplings=SINGLE_STAGE_RESAMPLINGS,
            eval_repeats=SINGLE_STAGE_EVAL_REPEATS,
            second_order=SINGLE_STAGE_SECOND_ORDER,
            blocking=SINGLE_STAGE_BLOCKING,
            trust_region=SINGLE_STAGE_TRUST_REGION,
            regularization=SINGLE_STAGE_REGULARIZATION,
            hessian_delay=SINGLE_STAGE_HESSIAN_DELAY,
            calibration_target_magnitude=SINGLE_STAGE_TARGET_MAGNITUDE,
        )
        stage2 = None
        theta_history_arr = np.asarray(stage1["theta_history"], dtype=float)
        cost_history_arr = np.asarray(stage1["obj_history"], dtype=float)
        l2_history_arr = np.asarray(stage1["l2_history"], dtype=float)
        elapsed_time = float(stage1["elapsed_s"])
        theta_last = _as_1d_float(stage1["theta_last"])

    idx_best_l2_raw = int(np.argmin(l2_history_arr))
    idx_best_obj_global = int(np.argmin(cost_history_arr))
    idx_best_l2_global, best_l2_rechecked = _select_best_theta_by_recheck(
        crca,
        f_target,
        theta_history_arr,
        l2_history_arr,
        shots=SHOTS,
        top_k=POSTSELECT_TOPK,
        eval_repeats=POSTSELECT_EVAL_REPEATS,
    )
    theta_star = theta_history_arr[idx_best_l2_global].copy()
    best_fx = float(cost_history_arr[idx_best_obj_global])

    f0 = _evaluate_function_values(crca, theta0, shots=SHOTS, seed=SHOT_SEED)
    f_last = _evaluate_function_values(crca, theta_last, shots=SHOTS, seed=SHOT_SEED)
    f_star = _evaluate_function_values(crca, theta_star, shots=SHOTS, seed=SHOT_SEED)

    best_so_far = np.minimum.accumulate(np.maximum(l2_history_arr, 1e-15))
    best_idx = np.flatnonzero(np.r_[True, best_so_far[1:] < best_so_far[:-1] - CHECKPOINT_TOL])
    final_l2 = _mean_squared_error(f_last, f_target)
    best_l2 = _mean_squared_error(f_star, f_target)
    cost_plot_label = "L2 loss across stages (shots + frozen backend noise)"

    print(f"\nTraining complete in {elapsed_time:.1f} s")
    print(f"Best stage objective value: {best_fx:.6e}")
    print(
        f"Best L2 (rechecked, repeats={POSTSELECT_EVAL_REPEATS}): {best_l2_rechecked:.6e} | "
        f"Best L2 (single-seed): {best_l2:.6e} | Final L2: {final_l2:.6e}"
    )

    if USE_TWO_STAGE:
        loss_mode_stage1 = STAGE1_MODE
        loss_mode_stage2 = STAGE2_MODE
        stage1_calibration_shots = STAGE1_CALIBRATION_SHOTS
        stage2_calibration_shots = STAGE2_CALIBRATION_SHOTS
        stage1_resamplings = STAGE1_RESAMPLINGS
        stage2_resamplings = STAGE2_RESAMPLINGS
        stage1_eval_repeats = STAGE1_EVAL_REPEATS
        stage2_eval_repeats = STAGE2_EVAL_REPEATS
        stage1_target_magnitude = STAGE1_TARGET_MAGNITUDE
        stage2_target_magnitude = STAGE2_TARGET_MAGNITUDE
        stage1_maxiter = STAGE1_MAXITER
        stage2_maxiter = STAGE2_MAXITER
        stage1_second_order = STAGE1_SECOND_ORDER
        stage2_second_order = STAGE2_SECOND_ORDER
        stage1_blocking = STAGE1_BLOCKING
        stage2_blocking = STAGE2_BLOCKING
        stage1_trust_region = STAGE1_TRUST_REGION
        stage2_trust_region = STAGE2_TRUST_REGION
        stage1_regularization = STAGE1_REGULARIZATION
        stage2_regularization = STAGE2_REGULARIZATION
        stage1_hessian_delay = STAGE1_HESSIAN_DELAY
        stage2_hessian_delay = STAGE2_HESSIAN_DELAY
        stage2_best_obj = float(stage2["best_obj"])
        stage2_best_l2 = float(stage2["best_l2"])
        stage2_success = bool(stage2["result_success"])
        stage2_message = str(stage2["result_message"])
        stage2_obj_history = np.asarray(stage2["obj_history"], dtype=float)
        stage2_l2_history = np.asarray(stage2["l2_history"], dtype=float)
    else:
        loss_mode_stage1 = SINGLE_STAGE_MODE
        loss_mode_stage2 = "none"
        stage1_calibration_shots = SINGLE_STAGE_CALIBRATION_SHOTS
        stage2_calibration_shots = 0
        stage1_resamplings = SINGLE_STAGE_RESAMPLINGS
        stage2_resamplings = None
        stage1_eval_repeats = SINGLE_STAGE_EVAL_REPEATS
        stage2_eval_repeats = 0
        stage1_target_magnitude = SINGLE_STAGE_TARGET_MAGNITUDE
        stage2_target_magnitude = float("nan")
        stage1_maxiter = SINGLE_STAGE_MAXITER
        stage2_maxiter = 0
        stage1_second_order = SINGLE_STAGE_SECOND_ORDER
        stage2_second_order = False
        stage1_blocking = SINGLE_STAGE_BLOCKING
        stage2_blocking = False
        stage1_trust_region = SINGLE_STAGE_TRUST_REGION
        stage2_trust_region = False
        stage1_regularization = SINGLE_STAGE_REGULARIZATION
        stage2_regularization = 0.0
        stage1_hessian_delay = SINGLE_STAGE_HESSIAN_DELAY
        stage2_hessian_delay = 0
        stage2_best_obj = float("nan")
        stage2_best_l2 = float("nan")
        stage2_success = False
        stage2_message = "disabled (single-stage mode)"
        stage2_obj_history = np.asarray([], dtype=float)
        stage2_l2_history = np.asarray([], dtype=float)

    maxiter_total = int(stage1_maxiter + stage2_maxiter)
    overall_blocking = bool(stage1_blocking and stage2_blocking) if USE_TWO_STAGE else bool(stage1_blocking)
    overall_second_order = bool(stage1_second_order and stage2_second_order) if USE_TWO_STAGE else bool(stage1_second_order)
    overall_trust_region = bool(stage1_trust_region and stage2_trust_region) if USE_TWO_STAGE else bool(stage1_trust_region)

    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0,
        after=f_star,
        cost_history=np.maximum(l2_history_arr, 1e-15),
        best_so_far=best_so_far,
        best_idx=best_idx,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel=cost_plot_label,
        title_before="Before training (shots + backend noise)",
        title_after="After training (best iterate, shots + backend noise)",
        cost_log_x=False,
        cost_log_y=True,
    )
    plt.show()

    circuit_save_path = out_path.with_name(
        "trained_crca_positive_exposure_circuit_shots_backend_noise_snapshot.qpy"
    )
    with open(circuit_save_path, "wb") as f:
        qpy.dump(tqc_eval_meas_parametric, f)
    print(f"[OK] Circuit saved to: {circuit_save_path}")

    metadata = {
        "model": "CRCA",
        "task": "positive_exposure",
        "optimizer": "SPSA",
        "use_two_stage": USE_TWO_STAGE,
        "loss_mode_stage1": loss_mode_stage1,
        "loss_mode_stage2": loss_mode_stage2,
        "shots": SHOTS,
        "stage1_calibration_shots": stage1_calibration_shots,
        "stage2_calibration_shots": stage2_calibration_shots,
        "stage1_resamplings": stage1_resamplings,
        "stage2_resamplings": stage2_resamplings,
        "stage1_eval_repeats": stage1_eval_repeats,
        "stage2_eval_repeats": stage2_eval_repeats,
        "stage1_target_magnitude": stage1_target_magnitude,
        "stage2_target_magnitude": stage2_target_magnitude,
        "init_selection_eval_repeats": INIT_SELECTION_EVAL_REPEATS,
        "postselect_top_k": POSTSELECT_TOPK,
        "postselect_eval_repeats": POSTSELECT_EVAL_REPEATS,
        "stage1_maxiter": stage1_maxiter,
        "stage2_maxiter": stage2_maxiter,
        "maxiter_total": maxiter_total,
        "blocking": overall_blocking,
        "second_order": overall_second_order,
        "trust_region": overall_trust_region,
        "stage1_blocking": stage1_blocking,
        "stage2_blocking": stage2_blocking,
        "stage1_second_order": stage1_second_order,
        "stage2_second_order": stage2_second_order,
        "stage1_trust_region": stage1_trust_region,
        "stage2_trust_region": stage2_trust_region,
        "last_avg": SPSA_LAST_AVG,
        "regularization": stage1_regularization,
        "stage1_regularization": stage1_regularization,
        "stage2_regularization": stage2_regularization,
        "hessian_delay": stage1_hessian_delay,
        "stage1_hessian_delay": stage1_hessian_delay,
        "stage2_hessian_delay": stage2_hessian_delay,
        "robust_rel_clip": ROBUST_REL_CLIP,
        "robust_rel_huber_delta": ROBUST_REL_HUBER_DELTA,
        "robust_zero_huber_delta": ROBUST_ZERO_HUBER_DELTA,
        "lambda_l2_mix": LAMBDA_L2_MIX,
        "shot_seed": SHOT_SEED,
        "theta_seed": SEED,
        "init_scale": INIT_SCALE,
        "warmstart_enabled": USE_STATEVECTOR_WARMSTART,
        "warmstart_used": warmstart_used,
        "warmstart_l2": warmstart_l2,
        "random_start_l2": l2_random,
        "stage1_best_obj": float(stage1["best_obj"]),
        "stage1_best_l2": float(stage1["best_l2"]),
        "stage2_best_obj": stage2_best_obj,
        "stage2_best_l2": stage2_best_l2,
        "idx_best_l2_raw": idx_best_l2_raw,
        "idx_best_l2_rechecked": idx_best_l2_global,
        "best_l2_rechecked": float(best_l2_rechecked),
        "stage1_success": bool(stage1["result_success"]),
        "stage2_success": stage2_success,
        "stage1_message": str(stage1["result_message"]),
        "stage2_message": stage2_message,
        "backend_name": BACKEND_NAME,
        "requested_topology": REQUESTED_TOPOLOGY,
        "effective_topology": effective_topology,
        "layout_score": float(layout_score),
        "fallback_used": bool(layout_meta["fallback_used"]),
        "noise_snapshot_iso_utc": snapshot_dt_utc.isoformat(),
        "backend_props_last_update": str(getattr(backend_props, "last_update_date", None)),
        "thermal_relaxation_requested": THERMAL_RELAXATION_REQUESTED,
        "thermal_relaxation_effective": thermal_relaxation_effective,
        "injected_frequency_count": int(injected_frequency_count),
        "noise_model_build": noise_model_build,
    }

    np.savez(
        out_path,
        theta_star=theta_star,
        theta_last=theta_last,
        theta_init=theta0,
        theta_history=theta_history_arr,
        cost_history=cost_history_arr,
        l2_history=l2_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        f_target=f_target,
        f_target_2d=f_target_2d,
        f_init=f0,
        f_last=f_last,
        f_star=f_star,
        C_v=np.float64(c_v),
        elapsed_time=np.float64(elapsed_time),
        best_cost=np.float64(best_fx),
        final_l2=np.float64(final_l2),
        best_l2=np.float64(best_l2),
        best_l2_rechecked=np.float64(best_l2_rechecked),
        idx_best_l2_raw=np.int64(idx_best_l2_raw),
        idx_best_l2_rechecked=np.int64(idx_best_l2_global),
        n_iters=np.int64(maxiter_total),
        stage1_obj_history=np.asarray(stage1["obj_history"], dtype=float),
        stage1_l2_history=np.asarray(stage1["l2_history"], dtype=float),
        stage2_obj_history=stage2_obj_history,
        stage2_l2_history=stage2_l2_history,
        shots=np.int64(SHOTS),
        stage1_calibration_shots=np.int64(stage1_calibration_shots),
        stage2_calibration_shots=np.int64(stage2_calibration_shots),
        stage1_resamplings=np.array(stage1_resamplings, dtype=object),
        stage2_resamplings=np.array(stage2_resamplings, dtype=object),
        stage1_eval_repeats=np.int64(stage1_eval_repeats),
        stage2_eval_repeats=np.int64(stage2_eval_repeats),
        stage1_target_magnitude=np.float64(stage1_target_magnitude),
        stage2_target_magnitude=np.float64(stage2_target_magnitude),
        init_selection_eval_repeats=np.int64(INIT_SELECTION_EVAL_REPEATS),
        postselect_top_k=np.int64(POSTSELECT_TOPK),
        postselect_eval_repeats=np.int64(POSTSELECT_EVAL_REPEATS),
        stage1_second_order=np.bool_(stage1_second_order),
        stage2_second_order=np.bool_(stage2_second_order),
        stage1_blocking=np.bool_(stage1_blocking),
        stage2_blocking=np.bool_(stage2_blocking),
        stage1_trust_region=np.bool_(stage1_trust_region),
        stage2_trust_region=np.bool_(stage2_trust_region),
        theta_seed=np.int64(SEED),
        shot_seed=np.int64(SHOT_SEED),
        simulator_seed=np.int64(SIMULATOR_SEED),
        seed_transpiler=np.int64(SEED_TRANSPILER),
        transpile_optimization_level=np.int64(TRANSPILATION_OPT_LEVEL),
        requested_topology=np.array(REQUESTED_TOPOLOGY),
        effective_topology=np.array(effective_topology),
        chosen_layout=np.array(chosen_layout, dtype=int),
        chosen_layout_local=np.array(local_layout, dtype=int),
        local_coupling_map=np.array(local_coupling_map, dtype=int),
        layout_score=np.float64(layout_score),
        fallback_used=np.bool_(layout_meta["fallback_used"]),
        tried_layout_search=np.array(layout_meta["tried"], dtype=object),
        transpiled_ansatz_depth=np.int64(tqc_ansatz.depth()),
        transpiled_eval_depth=np.int64(tqc_eval.depth()),
        transpiled_eval_meas_depth=np.int64(tqc_eval_meas_parametric.depth()),
        noise_snapshot_iso_utc=np.array(snapshot_dt_utc.isoformat()),
        backend_props_last_update=np.array(str(getattr(backend_props, "last_update_date", None))),
        thermal_relaxation_requested=np.bool_(THERMAL_RELAXATION_REQUESTED),
        thermal_relaxation_effective=np.bool_(thermal_relaxation_effective),
        injected_frequency_count=np.int64(injected_frequency_count),
        noise_model_build=np.array(noise_model_build),
        noise_basis_gates=np.array(list(noise_model.basis_gates), dtype=object),
        metadata=np.array(metadata, dtype=object),
    )
    print(f"[OK] Results saved to: {out_path}")


if __name__ == "__main__":
    main()