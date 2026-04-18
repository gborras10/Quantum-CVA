import pathlib
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
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
LOGICAL_TOPOLOGY = "qcbm_heavyhex6"

N_LAYERS = 6 
EPS_COST = 1e-9
INIT_SCALE = 0.01
SEED = 42
SEED_TRANSPILER = 1234
SIMULATOR_SEED = 20260407

SHOTS = 60000
N_ITERS = 500
RESAMPLINGS = 3
DIRICHLET_ALPHA = 1.0

KL_EVAL_RUNS = 10
KL_EVAL_SHOTS = 100000
KL_EVAL_SEED_BASE = 42

CHECKPOINT_TOL = 1e-15

# =====================================================================
# Fixed backend-noise snapshot
# ---------------------------------------------------------------------
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
        parent for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    data = np.load(
        repo_root / "data" / "multi_asset" / "6q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )

    saving_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "qcbm"
        / "shots"
        / "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)

    ptg = np.asarray(data["p_target"], dtype=float).ravel()
    ptg /= ptg.sum()

    dim = ptg.size
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("Target distribution dimension must be a power of 2.")

    target_entropy = -np.sum(ptg * np.log(np.clip(ptg, EPS_COST, 1.0)))

    # ===================== Backend, historical properties & Layout =====================
    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)

    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    backend_props = real_backend.properties(datetime=snapshot_dt_utc)
    if backend_props is None:
        raise RuntimeError(
            "Could not retrieve historical backend properties for timestamp "
            f"{NOISE_SNAPSHOT_ISO_UTC}."
        )

    chosen_layout, layout_score, layout_meta = select_best_layout(
        real_backend,
        topology=LOGICAL_TOPOLOGY,
        length=n_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
    )

    effective_topology = layout_meta["selected_topology"]

    print(f"backend_name            = {BACKEND_NAME}")
    print(f"requested_topology      = {LOGICAL_TOPOLOGY}")
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
        noise_model = NoiseModel.from_backend_properties(backend_props, thermal_relaxation=False)
    except AttributeError:
        used_noise_fallback = True
        print("Noise model could not be created from backend properties. " \
        "Falling back to noise model from backend (non-snapshot).")
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

    # ===================== QCBM Configuration =====================
    qcbm = MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=N_LAYERS,
        name="G_p_shots_backend_noise_snapshot_transpiled",
        entangler="rzz",
        topology=effective_topology,
        backend=noisy_backend,
        transpile_backend=real_backend,
        noise_model=noise_model,
        basis_gates=list(noise_model.basis_gates),
        simulation_method="density_matrix",
        optimization_level=3,
        initial_layout=chosen_layout,
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=SEED_TRANSPILER,
    )

    summarize_circuit(qcbm._tqc, label="Transpiled QCBM Circuit")

    # ===================== Baseline KL: statevector-trained theta under noise =====================
    statevector_training_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "6q_instance"
        / "quantum"
        / "training"
        / "qcbm"
        / "statevector"
        / "training_qcbm_heavyhex6_6lay.npz"
    )
    statevector_training = np.load(statevector_training_path, allow_pickle=True)
    if "theta_star" not in statevector_training:
        raise KeyError(
            "'theta_star' was not found in the statevector training file: "
            f"{statevector_training_path}"
        )

    theta_statevector = np.asarray(statevector_training["theta_star"], dtype=float).ravel()
    if theta_statevector.size != qcbm.n_params:
        raise ValueError(
            "Parameter size mismatch between theta_statevector and the current QCBM. "
            f"theta_statevector={theta_statevector.size}, qcbm.n_params={qcbm.n_params}."
        )

    print("\nEvaluating KL under noise with statevector-trained parameters...")
    kl_eval_values: list[float] = []
    for run_idx in range(KL_EVAL_RUNS):
        run_seed = KL_EVAL_SEED_BASE + run_idx
        p_sv_noise = qcbm.probabilities(
            theta_statevector,
            shots=KL_EVAL_SHOTS,
            seed=run_seed,
        )
        kl_val = float(qcbm.metrics(ptg, p_sv_noise, eps=EPS_COST)["kl"])
        kl_eval_values.append(kl_val)
        print(
            f"[KL eval {run_idx + 1:02d}/{KL_EVAL_RUNS}] "
            f"seed={run_seed} | KL={kl_val:.6e}"
        )

    kl_eval_values_arr = np.asarray(kl_eval_values, dtype=float)
    kl_eval_mean = float(np.mean(kl_eval_values_arr))
    kl_eval_std = (
        float(np.std(kl_eval_values_arr, ddof=1))
        if kl_eval_values_arr.size > 1
        else 0.0
    )
    print(
        "KL mean (statevector theta under noisy model): "
        f"{kl_eval_mean:.6e}"
    )
    print(
        "KL std  (statevector theta under noisy model): "
        f"{kl_eval_std:.6e}"
    )

    # ===================== Cost function =====================
    cost_shots = qcbm.cost_fn(
        ptg,
        eps=EPS_COST,
        shots=SHOTS,
        seed=None,
        rescaled=True,
        smoothing="dirichlet",
        alpha=DIRICHLET_ALPHA,
    )

    # ===================== Initialization =====================
    rng = np.random.default_rng(SEED)
    theta0 = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)

    cost_history: list[float] = []
    theta_history: list[np.ndarray] = [theta0.copy()]
    best = {"fx": float("inf"), "x": theta0.copy()}

    iter_times: list[float] = []
    live_log: list[dict[str, float]] = []

    training_t0 = None
    last_callback_t = None

    def callback(nfev, x, fx, stepsize, accepted):
        nonlocal training_t0, last_callback_t

        now = time.perf_counter()

        if training_t0 is None:
            training_t0 = now
        if last_callback_t is None:
            iter_dt = 0.0
        else:
            iter_dt = now - last_callback_t

        last_callback_t = now

        fx = float(fx)
        x_arr = np.asarray(x, dtype=float).copy()

        cost_history.append(fx)
        theta_history.append(x_arr)
        iter_times.append(iter_dt)

        if fx < best["fx"]:
            best["fx"] = fx
            best["x"] = x_arr.copy()

        iter_idx = len(cost_history)
        elapsed_total = now - training_t0
        mean_iter_time = (
            float(np.mean(iter_times[1:])) if len(iter_times) > 1 else 0.0
        )

        live_log.append(
            {
                "iter": float(iter_idx),
                "nfev": float(nfev),
                "fx": fx,
                "best_fx": float(best["fx"]),
                "iter_time_s": float(iter_dt),
                "mean_iter_time_s": mean_iter_time,
                "elapsed_s": float(elapsed_total),
                "stepsize": float(stepsize),
                "accepted": float(bool(accepted)),
            }
        )

        print(
            f"[iter {iter_idx:5d}] "
            f"fx={fx:.6e} | best={best['fx']:.6e} | "
            f"dt={iter_dt:7.2f}s | mean_dt={mean_iter_time:7.2f}s | "
            f"nfev={nfev:6d} | step={stepsize:.3e}"
        )

    print("\nCalibrating SPSA hyperparameters...")
    learning_rate, perturbation = SPSA.calibrate(cost_shots, theta0)

    opt = SPSA(
        maxiter=int(N_ITERS),
        learning_rate=learning_rate,
        perturbation=perturbation,
        resamplings=RESAMPLINGS,
        blocking=False,
        callback=callback,
        trust_region=True,
        regularization=0.01,
    )

    # ===================== Training =====================
    print("\nStarting shots-based SPSA training with frozen backend noise model...")
    t0 = time.perf_counter()
    res = opt.minimize(fun=cost_shots, x0=theta0)
    elapsed_time = time.perf_counter() - t0

    theta_last = np.asarray(res.x, dtype=float)
    theta_star = best["x"].copy()

    print(f"Training complete. Best cost observed: {best['fx']:.6e}")
    print(f"Elapsed time (s): {elapsed_time:.2f}")
    print("success:", getattr(res, "success", None))
    print("message:", getattr(res, "message", None))

    # ===================== Probabilities & Metrics =====================
    p0 = qcbm.probabilities(theta0, shots=SHOTS, seed=None)
    p_last = qcbm.probabilities(theta_last, shots=SHOTS, seed=None)
    p_star = qcbm.probabilities(theta_star, shots=SHOTS, seed=None)

    cost_history_arr = np.asarray(cost_history, dtype=float)
    theta_history_arr = np.asarray(theta_history, dtype=float)
    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - CHECKPOINT_TOL]
    )
    kl_history = np.maximum(cost_history_arr, 1e-15)

    metrics_best = qcbm.metrics(ptg, p_star, eps=EPS_COST)
    metrics_last = qcbm.metrics(ptg, p_last, eps=EPS_COST)

    print("\nMetrics summary (best iterate):")
    print(metrics_best)

    # ===================== Plots =====================
    plot_training_diagnostics_multi_asset(
        target=ptg,
        before=p0,
        after=p_star,
        cost_history=kl_history,
        best_so_far=best_so_far,
        xlabel="Control basis state |i>",
        ylabel="f(i)",
        cost_ylabel="Rescaled CE (shots + frozen backend noise)",
        title_before="Before training (shots + backend noise)",
        title_after="After training (best iterate, shots + backend noise)",
        cost_log_x=False,
        cost_log_y=True,
    )
    plt.show()

    # ===================== Save circuit =====================
    circuit_save_path = saving_path.with_name(
        "trained_qcbm_circuit_shots_backend_noise_snapshot.qpy"
    )
    with open(circuit_save_path, "wb") as f:
        qpy.dump(qcbm._tqc, f)
    print(f"\n[OK] Transpiled circuit saved to: {circuit_save_path}")

    # ===================== Save results =====================
    t1_values = [_safe_property_value(backend_props, q, "T1") for q in chosen_layout]
    t2_values = [_safe_property_value(backend_props, q, "T2") for q in chosen_layout]
    readout_values = [
        _safe_property_value(backend_props, q, "readout_error") for q in chosen_layout
    ]

    np.savez(
        saving_path,
        theta_star=theta_star,
        theta_last=theta_last,
        theta_init=theta0,
        cost_history=cost_history_arr,
        kl_history=kl_history,
        best_so_far=best_so_far,
        best_idx=best_idx,
        theta_history=theta_history_arr,
        p_target=ptg,
        p_init=p0,
        p_last=p_last,
        p_star=p_star,
        elapsed_time=np.float64(elapsed_time),
        best_cost=np.float64(best["fx"]),
        target_entropy=np.float64(target_entropy),
        kl_eval_values_statevector=kl_eval_values_arr,
        kl_eval_mean_statevector=np.float64(kl_eval_mean),
        kl_eval_std_statevector=np.float64(kl_eval_std),
        kl_eval_runs=np.int64(KL_EVAL_RUNS),
        kl_eval_shots=np.int64(KL_EVAL_SHOTS),
        kl_eval_seed_base=np.int64(KL_EVAL_SEED_BASE),
        statevector_training_path=np.array(str(statevector_training_path)),
        n_iters=np.int64(N_ITERS),
        shots=np.int64(SHOTS),
        epsilon=np.float64(EPS_COST),
        theta_seed=np.int64(SEED),
        init_scale=np.float64(INIT_SCALE),
        resamplings=np.int64(RESAMPLINGS),
        dirichlet_alpha=np.float64(DIRICHLET_ALPHA),
        n_qubits=np.int64(n_qubits),
        n_layers=np.int64(N_LAYERS),
        backend_name=np.array(BACKEND_NAME),
        requested_topology=np.array(LOGICAL_TOPOLOGY),
        effective_topology=np.array(effective_topology),
        chosen_layout=np.array(chosen_layout, dtype=int),
        layout_score=np.float64(layout_score),
        fallback_used=np.bool_(layout_meta["fallback_used"]),
        tried_layout_search=np.array(layout_meta["tried"], dtype=object),
        transpiled_depth=np.int64(qcbm._tqc.depth()),
        transpiled_size=np.int64(qcbm._tqc.size()),
        transpiled_ops=np.array(dict(qcbm._tqc.count_ops()), dtype=object),
        metrics_best=np.array(metrics_best, dtype=object),
        metrics_last=np.array(metrics_last, dtype=object),
        noise_snapshot_iso_utc=np.array(snapshot_dt_utc.isoformat()),
        backend_props_last_update=np.array(str(getattr(backend_props, "last_update_date", None))),
        simulator_seed=np.int64(SIMULATOR_SEED),
        simulator_method=np.array("density_matrix"),
        noise_basis_gates=np.array(list(noise_model.basis_gates), dtype=object),
        used_noise_fallback=np.bool_(used_noise_fallback),
        snapshot_t1_chosen_layout=np.array(t1_values, dtype=object),
        snapshot_t2_chosen_layout=np.array(t2_values, dtype=object),
        snapshot_readout_error_chosen_layout=np.array(readout_values, dtype=object),
    )
    print(f"[OK] Results saved to: {saving_path}")


if __name__ == "__main__":
    main()
