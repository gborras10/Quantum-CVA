import pathlib
import time
from datetime import datetime, timezone

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

# ===================== Global Configuration =====================
BACKEND_NAME = "ibm_basquecountry"
REQUESTED_TOPOLOGY = "crca2"

TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234
SIMULATOR_SEED = 20260407

M_TIME = 2
N_PRICE = 0
N_LAYERS = 1

INIT_SCALE = 0.10
SEED = 42
SHOT_SEED = 355

SHOTS = 60000
N_ITERS = 70
RESAMPLINGS = 3

CHECKPOINT_TOL = 1e-15

# =====================================================================
# Fixed backend-noise snapshot
# =====================================================================
NOISE_SNAPSHOT_ISO_UTC = "2026-04-07T12:10:00+00:00"


def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
    dt = datetime.fromisoformat(snapshot_iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_property_value(props, qubit: int, name: str):
    try:
        out = props.qubit_property(qubit)
        value = out.get(name, (None, None))[0]
        return None if value is None else float(value)
    except Exception:
        return None


def main() -> None:
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "6q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    q_t = np.asarray(benchmark["q_t"], dtype=float).ravel()
    c_q = float(benchmark["C_q"])
    f_target = q_t / c_q

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "default_probabilities"
        / "training_crca2_shots_backend_noise_snapshot.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    statevector_training_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "default_probabilities"
        / "training_crca2.npz"
    )

    # ===================== Backend, historical properties & Layout =====================
    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    backend_props = real_backend.properties(datetime=snapshot_dt_utc)
    if backend_props is None:
        raise RuntimeError(
            "No se pudieron recuperar las propiedades historicas del backend para la fecha "
            f"{NOISE_SNAPSHOT_ISO_UTC}."
        )

    # Logical CRCA (same ansatz as statevector default_probabilities)
    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_probabilities_crca2_shots_backend_noise_snapshot",
    )

    n_total_logical_qubits = int(crca.qc.num_qubits)

    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=REQUESTED_TOPOLOGY,
        length=n_total_logical_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
    )

    effective_topology = layout_meta["selected_topology"]

    print(f"backend_name            = {BACKEND_NAME}")
    print(f"requested_topology      = {REQUESTED_TOPOLOGY}")
    print(f"effective_topology      = {effective_topology}")
    print(f"chosen_layout           = {chosen_layout}")
    print(f"layout_score            = {layout_score}")
    print(f"fallback_used           = {layout_meta['fallback_used']}")
    print(f"tried                   = {layout_meta['tried']}")
    print(f"noise_snapshot_utc      = {snapshot_dt_utc.isoformat()}")
    print(f"backend_props_last_update = {getattr(backend_props, 'last_update_date', None)}")

    # ===================== Fixed noise model from backend snapshot =====================
    used_noise_fallback = False
    try:
        noise_model = NoiseModel.from_backend_properties(
            backend_props,
            thermal_relaxation=False,
        )
    except AttributeError:
        used_noise_fallback = True
        print(
            "[WARNING] No disponible NoiseModel.from_backend_properties(). "
            "Se usa NoiseModel.from_backend(real_backend)."
        )
        noise_model = NoiseModel.from_backend(real_backend)

    coupling_map = getattr(real_backend, "coupling_map", None)
    if coupling_map is None:
        try:
            coupling_map = real_backend.configuration().coupling_map
        except Exception:
            coupling_map = None

    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=coupling_map,
        seed_simulator=SIMULATOR_SEED,
    )

    # ===================== Transpilation on real backend layout =====================
    pm = generate_preset_pass_manager(
        backend=real_backend,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        initial_layout=chosen_layout,
        seed_transpiler=SEED_TRANSPILER,
        approximation_degree=1.0,
    )

    tqc_ansatz = pm.run(crca.qc)
    tqc_eval = pm.run(crca.qc_eval)

    # Build parametric measured eval circuit then transpile
    qc_meas = crca.qc_eval.copy()
    c_ctrl = ClassicalRegister(crca.n_controls, "c")
    c_a = ClassicalRegister(1, "ca")
    qc_meas.add_register(c_ctrl, c_a)
    qc_meas.measure(crca._control_qubit_indices, c_ctrl)
    qc_meas.measure([crca._ancilla_qubit_index], c_a)

    tqc_eval_meas_parametric = pm.run(qc_meas)

    # Inject transpiled measured circuit + noisy backend in CRCA evaluator
    crca._backend = noisy_backend
    crca._tqc_eval_meas = tqc_eval_meas_parametric
    crca._tqc_eval_meas_param_set = set(tqc_eval_meas_parametric.parameters)
    crca._n_clbits = len(tqc_eval_meas_parametric.clbits)
    crca._ctrl_clbit_indices, crca._a_clbit_index = crca._extract_clbit_indices(
        tqc_eval_meas_parametric
    )

    summarize_circuit(tqc_ansatz, label="CRCA ansatz transpiled")
    summarize_circuit(tqc_eval, label="CRCA eval transpiled")
    summarize_circuit(tqc_eval_meas_parametric, label="CRCA eval+measure transpiled")

    # ===================== Baseline: statevector theta under noisy model =====================
    theta_statevector = None
    baseline_l2_statevector_under_noise = None
    f_statevector_under_noise = None

    if statevector_training_path.exists():
        statevector_training = np.load(statevector_training_path, allow_pickle=True)
        if "theta_star" in statevector_training:
            theta_statevector = np.asarray(statevector_training["theta_star"], dtype=float).ravel()
            if theta_statevector.size != crca.n_params:
                raise ValueError(
                    "Incompatibilidad de parametros entre theta_star statevector y CRCA actual. "
                    f"theta={theta_statevector.size}, crca.n_params={crca.n_params}."
                )

            f_statevector_under_noise = crca.function_values(
                theta_statevector,
                shots=SHOTS,
                seed=SHOT_SEED,
            )
            baseline_l2_statevector_under_noise = float(
                np.mean((f_statevector_under_noise - f_target) ** 2)
            )
            print(
                "Baseline L2 (theta statevector evaluada con ruido+shots): "
                f"{baseline_l2_statevector_under_noise:.8e}"
            )

    # ===================== Objective =====================
    cost_shots = crca.cost_fn(
        f_target,
        shots=SHOTS,
        seed=SHOT_SEED,
    )

    # ===================== Initialization =====================
    rng = np.random.default_rng(SEED)
    x0 = INIT_SCALE * rng.standard_normal(crca.n_params).astype(float)

    f0_shots = crca.function_values(x0, shots=SHOTS, seed=SHOT_SEED)

    cost_history: list[float] = []
    theta_history: list[np.ndarray] = []

    def callback(nfev, x, fx, step, accepted):
        cost_history.append(float(fx))
        theta_history.append(np.asarray(x, dtype=float).copy())
        iter_idx = len(cost_history)
        print(
            f"[iter {iter_idx:5d}] fx={float(fx):.6e} | "
            f"nfev={int(nfev):6d} | step={float(step):.3e} | accepted={bool(accepted)}"
        )

    print("\nCalibrating SPSA hyperparameters...")
    learning_rate, perturbation = SPSA.calibrate(cost_shots, x0)

    opt = SPSA(
        maxiter=int(N_ITERS),
        learning_rate=learning_rate,
        perturbation=perturbation,
        resamplings=RESAMPLINGS,
        last_avg=25,
        second_order=True,
        blocking=True,
        trust_region=True,
        callback=callback,
    )

    # ===================== Training =====================
    print("\nStarting default_probabilities training (shots + backend noise snapshot)...")
    t0 = time.perf_counter()
    res = opt.minimize(fun=cost_shots, x0=x0)
    elapsed_time = time.perf_counter() - t0

    cost_history_arr = np.asarray(cost_history, dtype=float)
    if cost_history_arr.size == 0:
        cost_history_arr = np.asarray([float(res.fun)], dtype=float)
        theta_best = np.asarray(res.x, dtype=float).copy()
        best_fx = float(res.fun)
    else:
        best_pos = int(np.argmin(cost_history_arr))
        theta_best = theta_history[best_pos].copy()
        best_fx = float(cost_history_arr[best_pos])

    theta_last = np.asarray(res.x, dtype=float)
    f_last_shots = crca.function_values(theta_last, shots=SHOTS, seed=SHOT_SEED)
    f_star_shots = crca.function_values(theta_best, shots=SHOTS, seed=SHOT_SEED)

    print(f"Training complete in {elapsed_time:.1f} s")
    print(f"Best L2 cost observed: {best_fx:.8e}")

    # ===================== Plots =====================
    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - CHECKPOINT_TOL]
    )

    time_labels = [format(i, f"0{M_TIME}b") for i in range(2**M_TIME)]
    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0_shots,
        after=f_star_shots,
        cost_history=np.maximum(cost_history_arr, 1e-15),
        best_so_far=np.maximum(best_so_far, 1e-15),
        best_idx=best_idx,
        labels=time_labels,
        xlabel="Time register |t>",
        ylabel="f(t)",
        cost_ylabel="L2 loss (shots + backend noise snapshot)",
        title_before="Before training (default probabilities, shots + backend noise)",
        title_after="After training (best iterate, default probabilities, shots + backend noise)",
        cost_log_x=False,
        cost_log_y=True,
    )
    plt.show()

    # ===================== Save circuit =====================
    circuit_save_path = out_path.with_name(
        "trained_crca_default_probabilities_shots_backend_noise_snapshot.qpy"
    )
    with open(circuit_save_path, "wb") as f:
        qpy.dump(tqc_eval_meas_parametric, f)
    print(f"[OK] Transpiled measured circuit saved to: {circuit_save_path}")

    # ===================== Save results =====================
    t1_values = [_safe_property_value(backend_props, int(q), "T1") for q in chosen_layout]
    t2_values = [_safe_property_value(backend_props, int(q), "T2") for q in chosen_layout]
    readout_values = [
        _safe_property_value(backend_props, int(q), "readout_error")
        for q in chosen_layout
    ]

    metadata = {
        "model": "CRCA",
        "task": "default_probability",
        "ansatz_type": "native_tree",
        "native_1q_order": ("rx", "rz"),
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_controls": crca.n_controls,
        "n_layers": N_LAYERS,
        "n_parameters": crca.n_params,
        "optimizer": "SPSA",
        "optimizer_library": "qiskit-algorithms",
        "maxiter": N_ITERS,
        "resamplings": RESAMPLINGS,
        "second_order": True,
        "blocking": True,
        "trust_region": True,
        "cost_function": "L2",
        "shots": SHOTS,
        "stochastic_cost": True,
        "shot_seed": SHOT_SEED,
        "best_iter_cost_observed": best_fx,
        "final_l2_cost": float(np.mean((f_last_shots - f_target) ** 2)),
        "stopping_criterion": "maxiter",
        "backend_name": BACKEND_NAME,
        "requested_topology": REQUESTED_TOPOLOGY,
        "effective_topology": effective_topology,
        "layout_score": float(layout_score),
        "fallback_used": bool(layout_meta["fallback_used"]),
        "tried_layout_search": list(layout_meta["tried"]),
        "transpile_optimization_level": TRANSPILATION_OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "noise_snapshot_iso_utc": snapshot_dt_utc.isoformat(),
        "backend_props_last_update": str(getattr(backend_props, "last_update_date", None)),
        "simulator_method": "density_matrix",
        "simulator_seed": SIMULATOR_SEED,
        "used_noise_fallback": used_noise_fallback,
        "statevector_reference_path": str(statevector_training_path),
        "note": (
            "CRCA default-probability training via shots + backend-noise snapshot. "
            "Ansatz/eval transpiled to backend layout and optimized with SPSA."
        ),
    }

    if f_statevector_under_noise is None:
        f_statevector_under_noise_arr = np.asarray([], dtype=float)
    else:
        f_statevector_under_noise_arr = np.asarray(f_statevector_under_noise, dtype=float)

    baseline_l2_value = (
        float(baseline_l2_statevector_under_noise)
        if baseline_l2_statevector_under_noise is not None
        else float("nan")
    )

    np.savez(
        out_path,
        theta_star=theta_best,
        theta_last=theta_last,
        theta_init=x0,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        f_target=f_target,
        f_init_shots=f0_shots,
        f_last_shots=f_last_shots,
        f_star_shots=f_star_shots,
        f_statevector_under_noise=f_statevector_under_noise_arr,
        baseline_l2_statevector_under_noise=np.float64(baseline_l2_value),
        elapsed_time=np.float64(elapsed_time),
        best_cost=np.float64(best_fx),
        final_cost=np.float64(np.mean((f_last_shots - f_target) ** 2)),
        C_q=np.float64(c_q),
        n_iters=np.int64(N_ITERS),
        shots=np.int64(SHOTS),
        resamplings=np.int64(RESAMPLINGS),
        theta_seed=np.int64(SEED),
        shot_seed=np.int64(SHOT_SEED),
        transpile_optimization_level=np.int64(TRANSPILATION_OPT_LEVEL),
        seed_transpiler=np.int64(SEED_TRANSPILER),
        backend_name=np.array(BACKEND_NAME),
        requested_topology=np.array(REQUESTED_TOPOLOGY),
        effective_topology=np.array(effective_topology),
        chosen_layout=np.array(chosen_layout, dtype=int),
        layout_score=np.float64(layout_score),
        fallback_used=np.bool_(layout_meta["fallback_used"]),
        tried_layout_search=np.array(layout_meta["tried"], dtype=object),
        transpiled_ansatz_depth=np.int64(tqc_ansatz.depth()),
        transpiled_ansatz_size=np.int64(tqc_ansatz.size()),
        transpiled_ansatz_ops=np.array(dict(tqc_ansatz.count_ops()), dtype=object),
        transpiled_eval_depth=np.int64(tqc_eval.depth()),
        transpiled_eval_size=np.int64(tqc_eval.size()),
        transpiled_eval_ops=np.array(dict(tqc_eval.count_ops()), dtype=object),
        transpiled_eval_meas_depth=np.int64(tqc_eval_meas_parametric.depth()),
        transpiled_eval_meas_size=np.int64(tqc_eval_meas_parametric.size()),
        transpiled_eval_meas_ops=np.array(dict(tqc_eval_meas_parametric.count_ops()), dtype=object),
        noise_snapshot_iso_utc=np.array(snapshot_dt_utc.isoformat()),
        backend_props_last_update=np.array(str(getattr(backend_props, "last_update_date", None))),
        simulator_seed=np.int64(SIMULATOR_SEED),
        simulator_method=np.array("density_matrix"),
        noise_basis_gates=np.array(list(noise_model.basis_gates), dtype=object),
        used_noise_fallback=np.bool_(used_noise_fallback),
        snapshot_t1_chosen_layout=np.array(t1_values, dtype=object),
        snapshot_t2_chosen_layout=np.array(t2_values, dtype=object),
        snapshot_readout_error_chosen_layout=np.array(readout_values, dtype=object),
        statevector_training_path=np.array(str(statevector_training_path)),
        metadata=np.array(metadata, dtype=object),
    )
    print(f"[OK] Results saved to: {out_path}")


if __name__ == "__main__":
    main()