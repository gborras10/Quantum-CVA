import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit
from quantum_cva.multi_asset.quantum.training.utilities.circuit_training_tools import (
    plot_training_diagnostics_multi_asset,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (
    select_best_layout,
    summarize_circuit
)

# ===================== Global Configuration =====================
BACKEND_NAME = "ibm_basquecountry"

TRANSPILATION_OPT_LEVEL = 3
SEED_TRANSPILER = 1234

M_TIME = 2  # 2 time qubits
N_PRICE = 0 # 0 price qubits
N_LAYERS = 1

THETA_SEED = 355
N_ITERS = 400
SHOTS = 1024 
CALIB_SHOTS = SHOTS 

RESAMPLINGS = 1
BLOCKING = False
TRUST_REGION = False
SHOT_SEED = 355


def run_spsa_training(
    crca: CrcaCircuit, 
    f_target: np.ndarray, 
    x0: np.ndarray, 
    shots: int, 
    calib_shots: int, 
    n_iters: int, 
    label: str
):
    """Helper function to run SPSA training and return metrics."""
    print(f"\n{'-'*50}\nStarting {label} Training Setup...\n{'-'*50}")
    
    # 1. Define cost functions
    cost_shots = crca.cost_fn(f_target, shots=shots, seed=SHOT_SEED)
    cost_calib = crca.cost_fn(f_target, shots=calib_shots, seed=SHOT_SEED)

    # 2. Calibration
    print(f"[{label}] Calibrating SPSA hyperparameters (using {calib_shots} shots)...")
    t_calib_start = time.perf_counter()
    lr, pert = SPSA.calibrate(cost_calib, x0)    
    t_calib = time.perf_counter() - t_calib_start
    print(f"[{label}] Calibration complete in {t_calib:.1f}s.")

    # 3. Optimization Setup
    cost_history = []
    theta_history = []

    def spsa_callback(nfev, x, fx, step, accepted):
        cost_history.append(float(fx))
        theta_history.append(np.asarray(x, dtype=float).copy())
        if nfev % 20 == 0 or nfev <= 10:  # Print less frequently to avoid terminal clutter
            print(f"[{label}] Eval: {nfev:4d} | Cost: {fx:.6f} | Step: {step:.4f} | Accepted: {accepted}")

    optimizer = SPSA(
        maxiter=int(n_iters),
        learning_rate=lr,
        perturbation=pert,
        resamplings={0: 1, 50: 2, 150: 4},
        last_avg=25,
        second_order=True,
        blocking=True,
        trust_region=True,
        callback=spsa_callback,
    )

    # 4. Run Optimization
    print(f"\n[{label}] Starting SPSA optimization loop...")
    t_opt_start = time.perf_counter()
    res = optimizer.minimize(fun=cost_shots, x0=x0)
    t_opt = time.perf_counter() - t_opt_start

    # 5. Extract Results
    cost_history_arr = np.asarray(cost_history, dtype=float)
    if cost_history_arr.size == 0:
        cost_history_arr = np.asarray([float(res.fun)], dtype=float)
        theta_best = np.asarray(res.x, dtype=float).copy()
        best_fx = float(res.fun)
    else:
        best_pos = int(np.argmin(cost_history_arr))
        theta_best = theta_history[best_pos].copy()
        best_fx = float(cost_history_arr[best_pos])

    print(f"[{label}] Training complete in {t_opt:.1f}s. Best L2 Cost: {best_fx:.8f}")

    return {
        "best_cost": best_fx,
        "theta_best": theta_best,
        "cost_history": cost_history_arr,
        "time_calib": t_calib,
        "time_opt": t_opt,
        "lr": lr,
        "pert": pert
    }


def main() -> None:
    # ===================== Path & Data Loading =====================
    repo_root = next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )

    benchmark = np.load(
        repo_root / "data" / "multi_asset" / "8q_instance" / "benchmark" / "three_asset_instance.npz",
        allow_pickle=True,
    )
    q_t: np.ndarray = benchmark["q_t"]
    c_q: float = float(benchmark["C_q"])
    f_target: np.ndarray = q_t / c_q

    out_path = (
        repo_root
        / "data"
        / "multi_asset"
        / "8q_instance"
        / "quantum"
        / "training"
        / "crca"
        / "default_probabilities"
        / "shots_noise"
        / "default_probabilities_standard_shots_noise_comparison.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ==================== CRCA Configuration ====================
    crca = CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=N_LAYERS,
        ansatz_type="native_tree", 
        native_1q_order=("rx", "rz"),
        name="crca_standard",
    )

    # ==================== Hardware & Simulators Loading ====================
    print(f"Loading real hardware configuration and noise model from {BACKEND_NAME}...")
    service = QiskitRuntimeService()
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)
    
    # Create two separate simulators to swap later
    ideal_sim = AerSimulator() 
    noisy_sim = AerSimulator.from_backend(real_backend)
    
    # ==================== Layout Search ====================
    n_logical_qubits = crca.n_controls + 1 
    
    chosen_layout, _, _ = select_best_layout(
        real_backend,
        topology="crca2",
        length=n_logical_qubits,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
    )
    print(f"Best hardware layout found [c0, c1, a]: {chosen_layout}")

    # ==================== Transpilation ====================
    pm = generate_preset_pass_manager(
        backend=real_backend,
        optimization_level=TRANSPILATION_OPT_LEVEL,
        initial_layout=chosen_layout,
        seed_transpiler=SEED_TRANSPILER,
    )

    # Parametric Transpilation (Required for SPSA iterative training)
    qc_meas = crca.qc_eval.copy()
    c_ctrl = ClassicalRegister(crca.n_controls, "c")
    c_a = ClassicalRegister(1, "ca")
    qc_meas.add_register(c_ctrl, c_a)
    qc_meas.measure(crca._control_qubit_indices, c_ctrl)
    qc_meas.measure([crca._ancilla_qubit_index], c_a)

    tqc_meas_parametric = pm.run(qc_meas)

    # Inject the parametric physical circuit into CRCA (It will be shared by both trainings)
    crca._tqc_eval_meas = tqc_meas_parametric
    crca._tqc_eval_meas_param_set = set(tqc_meas_parametric.parameters)

    print("Generando imagen del circuito transpilado paramétrico...")
    tqc_meas_parametric.draw(
        output="mpl",            # Usa matplotlib para alta calidad
        idle_wires=False,        # Oculta los cables de los qubits del chip que no usamos
        fold=-1,                 # Evita que el circuito se parta en varias líneas (lo dibuja alargado)
        filename=out_path.parent / "transpiled_training_circuit.png"
    )
    print("Imagen guardada como 'transpiled_training_circuit.png'")

    # Initialize parameters
    rng = np.random.default_rng(THETA_SEED)
    x0: np.ndarray = 0.1 * rng.standard_normal(crca.n_params).astype(float)
    f0_ideal_shots = crca.function_values(x0, shots=SHOTS, seed=SHOT_SEED) # Will use AerSim by default first

    # ===================== TRAINING 1: IDEAL SIMULATOR =====================
    crca._backend = ideal_sim
    res_ideal = run_spsa_training(
        crca=crca, 
        f_target=f_target, 
        x0=x0, 
        shots=SHOTS, 
        calib_shots=SHOTS, # Ideal is fast, we can use full shots for calibration
        n_iters=N_ITERS, 
        label="IDEAL"
    )

    # ===================== TRAINING 2: NOISY SIMULATOR =====================
    crca._backend = noisy_sim
    res_noisy = run_spsa_training(
        crca=crca, 
        f_target=f_target, 
        x0=x0, 
        shots=SHOTS, 
        calib_shots=CALIB_SHOTS, # Noisy is slow, reduce shots for calibration
        n_iters=N_ITERS, 
        label="NOISY"
    )

    # ===================== QUANTITATIVE NOISE ANALYSIS =====================
    print(f"\n{'='*60}")
    print(f" QUANTITATIVE NOISE ANALYSIS REPORT ({BACKEND_NAME})")
    print(f"{'='*60}")
    
    # Time comparison
    t_ideal_total = res_ideal['time_calib'] + res_ideal['time_opt']
    t_noisy_total = res_noisy['time_calib'] + res_noisy['time_opt']
    time_multiplier = t_noisy_total / t_ideal_total if t_ideal_total > 0 else 0

    print("1. Execution Time (Wall-clock):")
    print(f"   - Ideal Training: {t_ideal_total:.2f} seconds")
    print(f"   - Noisy Training: {t_noisy_total:.2f} seconds")
    print(f"   -> Noise simulation is {time_multiplier:.2f}x slower than ideal simulation.\n")

    # Cost comparison
    cost_diff = res_noisy['best_cost'] - res_ideal['best_cost']
    cost_ratio = res_noisy['best_cost'] / res_ideal['best_cost'] if res_ideal['best_cost'] > 0 else 0

    print("2. Final Minimum L2 Cost (Convergence Quality):")
    print(f"   - Ideal Best Cost: {res_ideal['best_cost']:.8f}")
    print(f"   - Noisy Best Cost: {res_noisy['best_cost']:.8f}")
    print(f"   -> Hardware noise degrades the final L2 cost by +{cost_diff:.8f} (a {cost_ratio:.1f}x increase).\n")

    # ===================== Custom Comparison Plot =====================
    plt.figure(figsize=(10, 6))
    
    # Calculate "best so far" for both
    best_ideal = np.minimum.accumulate(res_ideal['cost_history'])
    best_noisy = np.minimum.accumulate(res_noisy['cost_history'])

    plt.plot(best_ideal, label=f"Ideal Simulator (Min: {res_ideal['best_cost']:.6f})", linewidth=2, color='blue')
    plt.plot(best_noisy, label=f"Noisy Simulator ({BACKEND_NAME}) (Min: {res_noisy['best_cost']:.6f})", linewidth=2, color='red', linestyle='--')
    
    plt.yscale("log")
    plt.title("SPSA Training Convergence: Ideal vs Hardware Noise")
    plt.xlabel("Function Evaluations (approx.)")
    plt.ylabel("Best L2 Loss Observed (Log Scale)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.parent / "noise_comparison_plot.png")
    plt.show()

    # ===================== Results Saving =====================
    # We save the noisy results as the primary output, but include ideal metrics in metadata
    metadata = {
        "model": "CRCA",
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "optimizer": "SPSA",
        "maxiter": N_ITERS,
        "shots": SHOTS,
        "hardware_noise_model": BACKEND_NAME,
        "chosen_layout": chosen_layout,
        "ideal_best_cost": res_ideal['best_cost'],
        "ideal_time_sec": t_ideal_total,
        "noisy_time_sec": t_noisy_total,
        "noise_time_multiplier": time_multiplier,
        "note": "Side-by-side comparison of Ideal vs Noisy training.",
    }

    f_star_noisy = crca.function_values(res_noisy['theta_best'], shots=SHOTS, seed=SHOT_SEED)

    np.savez(
        out_path,
        theta_star=res_noisy['theta_best'],
        cost_history_noisy=res_noisy['cost_history'],
        cost_history_ideal=res_ideal['cost_history'],
        f_target=f_target,
        f_init_shots=f0_ideal_shots,
        f_star_shots=f_star_noisy,
        best_cost_noisy=np.float64(res_noisy['best_cost']),
        best_cost_ideal=np.float64(res_ideal['best_cost']),
        n_iters=np.int64(N_ITERS),
        shots=np.int64(SHOTS),
        metadata=np.array(metadata, dtype=object),
    )

    plt.close('all')

if __name__ == "__main__":
    main()