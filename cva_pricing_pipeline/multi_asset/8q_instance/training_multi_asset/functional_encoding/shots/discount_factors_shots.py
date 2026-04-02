import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import summarize_circuit


# ===================== Global Configuration =====================
SIMULATOR_METHOD = "automatic"
TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 4
N_PRICE = 0
N_LAYERS = 1

THETA_SEED = 355
N_ITERS = 1000
SHOTS = 10000 
RESAMPLINGS = 1
BLOCKING = False
TRUST_REGION = False
SHOT_SEED = 355


def main() -> None:
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )
    p_t: np.ndarray = benchmark["p_t"]
    c_p: float = float(benchmark["C_p"])
    f_target: np.ndarray = p_t / c_p

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "quantum"
        / "training"
        / "crca"
        / "default_probabilities"
        / "shots_ideal"
        / "default_probabilities_time_tree4_shots_spsa.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ===================== CRCA Configuration =====================
    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_probabilities_time_tree4_shots",
    )

    # ===================== Simulator & Transpilation =====================
    sim_backend = AerSimulator(method=SIMULATOR_METHOD)

    print("simulator_backend =", sim_backend.name)
    print("simulator_method  =", SIMULATOR_METHOD)
    print("shots             =", SHOTS)

    tqc_ansatz = transpile(
        crca.qc,
        backend=sim_backend,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
    )

    tqc_eval = transpile(
        crca.qc_eval,
        backend=sim_backend,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
    )

    summarize_circuit(tqc_ansatz, label="CRCA ansatz transpiled for AerSimulator")
    summarize_circuit(tqc_eval, label="CRCA eval transpiled for AerSimulator")

    # ===================== Training (shots + SPSA) =====================
    rng = np.random.default_rng(THETA_SEED)
    #x0: np.ndarray = 0.5 * rng.standard_normal(crca.n_params).astype(float)
    x0: np.ndarray = 0.1 * rng.standard_normal(crca.n_params).astype(float)
    f0_shots: np.ndarray = crca.function_values(x0, shots=SHOTS, seed=SHOT_SEED)

    cost_shots = crca.cost_fn(
        f_target,
        shots=SHOTS,
        seed=SHOT_SEED,
    )

    cost_history: list[float] = []
    theta_history: list[np.ndarray] = []

    lr, pert = SPSA.calibrate(cost_shots, x0)

    shots_optimizer = SPSA(
        maxiter=int(N_ITERS),
        learning_rate=lr,
        perturbation=pert,
        resamplings={0: 1, 50: 2, 150: 4},
        last_avg=25,
        second_order=True,
        blocking=True,
        trust_region=True,
        callback=lambda nfev, x, fx, step, accepted: (
            cost_history.append(float(fx)),
            theta_history.append(np.asarray(x, dtype=float).copy()),
        ),
    )

    t0: float = time.perf_counter()
    res = shots_optimizer.minimize(fun=cost_shots, x0=x0)
    elapsed_time: float = time.perf_counter() - t0

    cost_history_arr = np.asarray(cost_history, dtype=float)
    if cost_history_arr.size == 0:
        cost_history_arr = np.asarray([float(res.fun)], dtype=float)
        theta_best = np.asarray(res.x, dtype=float).copy()
        best_fx = float(res.fun)
    else:
        best_pos = int(np.argmin(cost_history_arr))
        theta_best = theta_history[best_pos].copy()
        best_fx = float(cost_history_arr[best_pos])

    print(f"Training complete in {elapsed_time:.1f} s")
    print(f"Best L2 cost observed: {best_fx:.8f}")

    theta_last: np.ndarray = np.asarray(res.x, dtype=float)
    f_star_shots: np.ndarray = crca.function_values(theta_best, shots=SHOTS, seed=SHOT_SEED)

    # ===================== Plots =====================
    best_so_far = np.minimum.accumulate(cost_history_arr)
    best_idx = np.flatnonzero(
        np.r_[True, best_so_far[1:] < best_so_far[:-1] - 1e-15]
    )

    time_labels = [format(i, f"0{M_TIME}b") for i in range(2**M_TIME)]
    plot_training_diagnostics_multi_asset(
        target=f_target,
        before=f0_shots,
        after=f_star_shots,
        cost_history=cost_history_arr,
        best_so_far=best_so_far,
        best_idx=best_idx,
        labels=time_labels,
        xlabel="Time register |t>",
        ylabel="f(t)",
        cost_ylabel=("L2 loss"),
        title_before="Before training  (CRCA default probability, SPSA shots)",
        title_after="After training  (best-iter, CRCA discount factor, SPSA shots)",
        cost_log_x=True,
        cost_log_y=True,
    )
    plt.show()

    # ===================== Results Saving =====================
    metadata = {
        "model": "CRCA",
        "task": "default_probability",
        "ancilla_observable": "P(a=1 | control=i)",
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_controls": crca.n_controls,
        "n_layers": N_LAYERS,
        "n_parameters": crca.n_params,
        "optimizer": "SPSA",
        "optimizer_library": "qiskit-algorithms",
        "maxiter": N_ITERS,
        "resamplings": RESAMPLINGS,
        "blocking": BLOCKING,
        "trust_region": TRUST_REGION,
        "cost_function": "L2",
        "shots": SHOTS,
        "stochastic_cost": True,
        "shot_seed": SHOT_SEED,
        "best_iter_cost_observed": best_fx,
        "stopping_criterion": "maxiter",
        "simulator_backend": sim_backend.name,
        "simulator_method": SIMULATOR_METHOD,
        "transpile_optimization_level": TRANSPILATION_OPT_LEVEL,
        "seed_transpiler": SEED_TRANSPILER,
        "note": (
            "CRCA discount-factor training via shots-only SPSA on AerSimulator. "
            "theta_star = best-iteration parameters."
        ),
    }

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
        f_star_shots=f_star_shots,
        elapsed_time=np.float64(elapsed_time),
        best_cost=np.float64(best_fx),
        C_p=np.float64(c_p),
        n_iters=np.int64(N_ITERS),
        shots=np.int64(SHOTS),
        resamplings=np.int64(RESAMPLINGS),
        theta_seed=np.int64(THETA_SEED),
        transpiled_ansatz_depth=np.int64(tqc_ansatz.depth()),
        transpiled_ansatz_size=np.int64(tqc_ansatz.size()),
        transpiled_ansatz_ops=np.array(dict(tqc_ansatz.count_ops()), dtype=object),
        transpiled_eval_depth=np.int64(tqc_eval.depth()),
        transpiled_eval_size=np.int64(tqc_eval.size()),
        transpiled_eval_ops=np.array(dict(tqc_eval.count_ops()), dtype=object),
        transpile_optimization_level=np.int64(TRANSPILATION_OPT_LEVEL),
        seed_transpiler=np.int64(SEED_TRANSPILER),
        metadata=np.array(metadata, dtype=object),
    )


if __name__ == "__main__":
    main()