from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.optimize import minimize_scalar
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# -----------------------------------------------------------------------------
# Improved effective-T calibration for the contrast-decay model
#
#     p_obs(k) = 0.5 + exp(-(2k+1)/T) * (q_k - 0.5)
#     q_k      = sin^2((2k+1) * theta),  a = sin^2(theta)
#
# Main fixes relative to the first version:
#   1) optimization_level=0 to prevent the transpiler from compressing Q^k A
#      into a single effective rotation.
#   2) Much wider K-range, so that a large T can still be identified.
#   3) Many more shots, since the previous fit was dominated by shot noise.
#   4) k-grid built from repeated-probability subsequences (theta = pi/10):
#         - k = 5m     -> q_k = sin^2(pi/10) ≈ 0.09549
#         - k = 5m + 2 -> q_k = 1
#      This isolates contrast decay more cleanly.
#   5) Diagnostics on transpiled depth / operation counts and likelihood profile.
#
# The output T is still an *effective* parameter for this primitive and model,
# not a physical T1 or T2.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    backend_name: str = "ibm_basquecountry"
    a_true: float = float(np.sin(np.pi / 10.0) ** 2)

    # Choose two repeated-probability branches:
    #   low branch:  k = 5m      -> q ≈ 0.09549
    #   high branch: k = 5m + 2  -> q = 1
    m_values: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14)
    use_low_branch: bool = True
    use_high_branch: bool = True

    # 22 circuits x 2048 shots = 45,056 shots total.
    # For this 1-qubit primitive this is still modest, but much more stable
    # than 256 shots.
    shots: int = 2048
    optimization_level: int = 0
    physical_qubit: int | None = None

    # Wider range because the first experiment saturated at the upper bound.
    T_bounds: tuple[float, float] = (5.0, 1000.0)
    bootstrap_samples: int = 500
    random_seed: int = 1234

    output_dir: str = "./decoherence_calibration_outputs"
    output_stem: str = "t_calibration_improved"


def build_k_grid(cfg: CalibrationConfig) -> tuple[int, ...]:
    ks: list[int] = []
    for m in cfg.m_values:
        if cfg.use_low_branch:
            ks.append(5 * int(m))
        if cfg.use_high_branch:
            ks.append(5 * int(m) + 2)
    return tuple(sorted(set(ks)))


def build_state_prep(a: float) -> QuantumCircuit:
    qc = QuantumCircuit(1, name="A")
    qc.ry(2.0 * np.arcsin(np.sqrt(a)), 0)
    return qc


def build_grover(a: float) -> QuantumCircuit:
    qc = QuantumCircuit(1, name="Q")
    theta_p = 2.0 * np.arcsin(np.sqrt(a))
    qc.ry(2.0 * theta_p, 0)
    return qc


def build_calibration_circuit(a: float, k: int) -> QuantumCircuit:
    qc = QuantumCircuit(1, name=f"cal_k{k}")
    qc.compose(build_state_prep(a), inplace=True)
    if k > 0:
        grover = build_grover(a)
        for _ in range(int(k)):
            qc.compose(grover, inplace=True)
    c = ClassicalRegister(1, "c0")
    qc.add_register(c)
    qc.barrier()
    qc.measure(0, c[0])
    qc.metadata = {"grover_power": int(k)}
    return qc


def ideal_probability(a: float, k: int) -> float:
    theta = math.asin(math.sqrt(a))
    q = math.sin((2 * k + 1) * theta) ** 2
    return float(np.clip(q, 1e-12, 1.0 - 1e-12))


def model_probability(a: float, k: int, T: float) -> float:
    q = ideal_probability(a, k)
    contrast = math.exp(-(2 * k + 1) / T)
    p = 0.5 + contrast * (q - 0.5)
    return float(np.clip(p, 1e-12, 1.0 - 1e-12))


def choose_best_qubit(backend) -> int | None:
    try:
        props = backend.properties()
    except Exception:
        return None
    if props is None:
        return None

    best_q = None
    best_score = float("inf")
    for q in range(backend.num_qubits):
        score = 0.0
        try:
            readout = props.readout_error(q)
            if readout is not None:
                score += float(readout)
        except Exception:
            score += 0.01
        gate_found = False
        for gate_name in ("sx", "x", "id"):
            try:
                err = props.gate_error(gate_name, [q])
                if err is not None:
                    score += float(err)
                    gate_found = True
                    break
            except Exception:
                continue
        if not gate_found:
            score += 0.01
        if score < best_score:
            best_score = score
            best_q = q
    return best_q


def extract_counts(pub_result) -> dict[str, int]:
    data = pub_result.data
    if hasattr(data, "c0"):
        return dict(data.c0.get_counts())

    for name in dir(data):
        if name.startswith("_"):
            continue
        obj = getattr(data, name)
        if hasattr(obj, "get_counts"):
            return dict(obj.get_counts())

    raise RuntimeError("Could not extract counts from sampler result.")


def fit_T_mle(a: float, ks: np.ndarray, ones: np.ndarray, shots: np.ndarray, bounds: tuple[float, float]) -> float:
    def nll(T: float) -> float:
        ps = np.array([model_probability(a, int(k), float(T)) for k in ks], dtype=float)
        return float(-np.sum(ones * np.log(ps) + (shots - ones) * np.log(1.0 - ps)))

    opt = minimize_scalar(nll, bounds=bounds, method="bounded")
    if not opt.success:
        raise RuntimeError(f"MLE fit failed: {opt}")
    return float(opt.x)


def bootstrap_T_ci(
    a: float,
    ks: np.ndarray,
    shots: np.ndarray,
    T_hat: float,
    bounds: tuple[float, float],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ps = np.array([model_probability(a, int(k), T_hat) for k in ks], dtype=float)

    samples: list[float] = []
    for _ in range(n_boot):
        ones_b = rng.binomial(shots.astype(int), ps)
        try:
            T_b = fit_T_mle(a=a, ks=ks, ones=ones_b, shots=shots, bounds=bounds)
        except Exception:
            continue
        samples.append(T_b)

    if len(samples) < max(50, n_boot // 5):
        return (float("nan"), float("nan"))

    low, high = np.quantile(np.asarray(samples, dtype=float), [0.025, 0.975])
    return float(low), float(high)


def wilson_interval(n_ones: int, n_shots: int, z: float = 1.96) -> tuple[float, float]:
    phat = n_ones / n_shots
    denom = 1.0 + z * z / n_shots
    center = (phat + z * z / (2.0 * n_shots)) / denom
    radius = z * math.sqrt(phat * (1.0 - phat) / n_shots + z * z / (4.0 * n_shots * n_shots)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def likelihood_profile(
    a: float,
    ks: np.ndarray,
    ones: np.ndarray,
    shots: np.ndarray,
    T_grid: np.ndarray,
) -> np.ndarray:
    vals = []
    for T in T_grid:
        ps = np.array([model_probability(a, int(k), float(T)) for k in ks], dtype=float)
        nll = -np.sum(ones * np.log(ps) + (shots - ones) * np.log(1.0 - ps))
        vals.append(float(nll))
    return np.asarray(vals, dtype=float)


def main() -> None:
    cfg = CalibrationConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ks_tuple = build_k_grid(cfg)
    if len(ks_tuple) < 4:
        raise ValueError("The k-grid is too small.")

    service = QiskitRuntimeService()
    backend = service.backend(cfg.backend_name)

    physical_qubit = cfg.physical_qubit
    if physical_qubit is None:
        physical_qubit = choose_best_qubit(backend)

    circuits = [build_calibration_circuit(cfg.a_true, k) for k in ks_tuple]
    if physical_qubit is None:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=cfg.optimization_level,
        )
    else:
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=cfg.optimization_level,
            initial_layout=[physical_qubit],
        )
    isa_circuits = pm.run(circuits)

    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=cfg.shots)
    job_result = job.result()

    ks = np.asarray(ks_tuple, dtype=int)
    K = 2 * ks + 1
    shots = np.full(len(ks), cfg.shots, dtype=int)
    ones = np.empty(len(ks), dtype=int)
    p_hats = np.empty(len(ks), dtype=float)
    q_ideals = np.empty(len(ks), dtype=float)
    ci_low_p = np.empty(len(ks), dtype=float)
    ci_high_p = np.empty(len(ks), dtype=float)
    depths = np.empty(len(ks), dtype=int)
    n_rz = np.empty(len(ks), dtype=int)
    n_sx = np.empty(len(ks), dtype=int)

    for i, (pub, qc) in enumerate(zip(job_result, isa_circuits)):
        counts = extract_counts(pub)
        one_counts = int(counts.get("1", 0))
        ones[i] = one_counts
        p_hats[i] = one_counts / cfg.shots
        q_ideals[i] = ideal_probability(cfg.a_true, int(ks[i]))
        lo, hi = wilson_interval(one_counts, cfg.shots)
        ci_low_p[i] = lo
        ci_high_p[i] = hi
        depths[i] = int(qc.depth())
        ops = qc.count_ops()
        n_rz[i] = int(ops.get("rz", 0))
        n_sx[i] = int(ops.get("sx", 0))

    T_hat = fit_T_mle(
        a=cfg.a_true,
        ks=ks,
        ones=ones,
        shots=shots,
        bounds=cfg.T_bounds,
    )
    ci_low_T, ci_high_T = bootstrap_T_ci(
        a=cfg.a_true,
        ks=ks,
        shots=shots,
        T_hat=T_hat,
        bounds=cfg.T_bounds,
        n_boot=cfg.bootstrap_samples,
        seed=cfg.random_seed,
    )

    fitted_ps = np.array([model_probability(cfg.a_true, int(k), T_hat) for k in ks], dtype=float)
    fitted_cs = np.exp(-K / T_hat)
    residuals = p_hats - fitted_ps
    rmse = float(np.sqrt(np.mean(residuals**2)))

    T_grid = np.linspace(cfg.T_bounds[0], cfg.T_bounds[1], 800)
    nll_profile = likelihood_profile(cfg.a_true, ks, ones, shots, T_grid)
    nll_min = float(np.min(nll_profile))

    csv_path = output_dir / f"{cfg.output_stem}_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "k,K,q_ideal,one_counts,shots,p_hat,p_ci_low,p_ci_high,p_fit,residual,contrast_fit,depth,n_rz,n_sx\n"
        )
        for row in zip(
            ks,
            K,
            q_ideals,
            ones,
            shots,
            p_hats,
            ci_low_p,
            ci_high_p,
            fitted_ps,
            residuals,
            fitted_cs,
            depths,
            n_rz,
            n_sx,
        ):
            f.write(
                f"{row[0]},{row[1]},{row[2]:.10f},{row[3]},{row[4]},{row[5]:.10f},"
                f"{row[6]:.10f},{row[7]:.10f},{row[8]:.10f},{row[9]:.10f},"
                f"{row[10]:.10f},{row[11]},{row[12]},{row[13]}\n"
            )

    profile_path = output_dir / f"{cfg.output_stem}_likelihood_profile.csv"
    with profile_path.open("w", encoding="utf-8") as f:
        f.write("T,nll,delta_nll\n")
        for T, nll in zip(T_grid, nll_profile):
            f.write(f"{T:.10f},{nll:.10f},{(nll - nll_min):.10f}\n")

    print("=" * 100)
    print("Improved effective-T calibration for the exponential contrast model")
    print("=" * 100)
    print(f"Backend                 : {backend.name}")
    print(f"Job ID                  : {job.job_id()}")
    print(f"Chosen physical qubit   : {physical_qubit}")
    print(f"a_true                  : {cfg.a_true:.10f}")
    print(f"m_values                : {list(cfg.m_values)}")
    print(f"k grid                  : {list(ks_tuple)}")
    print(f"shots per k             : {cfg.shots}")
    print(f"total circuits          : {len(ks)}")
    print(f"total shots             : {len(ks) * cfg.shots}")
    print(f"optimization level      : {cfg.optimization_level}")
    print(f"T bounds                : {cfg.T_bounds}")
    print(f"Fitted T_hat            : {T_hat:.6f}")
    print(f"Bootstrap 95% CI        : [{ci_low_T:.6f}, {ci_high_T:.6f}]")
    print(f"Probability RMSE        : {rmse:.6e}")
    print(f"Results CSV             : {csv_path.resolve()}")
    print(f"Likelihood profile CSV  : {profile_path.resolve()}")
    print("-" * 100)
    print(
        f"{'k':>3} {'K':>4} {'q_ideal':>10} {'p_hat':>10} {'p_fit':>10} {'resid':>10} "
        f"{'depth':>6} {'rz':>4} {'sx':>4}"
    )
    for row in zip(ks, K, q_ideals, p_hats, fitted_ps, residuals, depths, n_rz, n_sx):
        print(
            f"{row[0]:3d} {row[1]:4d} {row[2]:10.6f} {row[3]:10.6f} {row[4]:10.6f} "
            f"{row[5]:10.6f} {row[6]:6d} {row[7]:4d} {row[8]:4d}"
        )
    print("=" * 100)
    print(
        "Interpretation:\n"
        "- This T is an effective contrast-decay parameter for the 1-qubit Bernoulli/Grover\n"
        "  primitive transpiled with optimization_level=0 on the chosen backend/qubit.\n"
        "- It is appropriate for the model p_obs = 0.5 + exp(-K/T) (q - 0.5), K = 2k+1.\n"
        "- It is not the same thing as the backend T1 or T2.\n"
        "- The wider K-grid and higher shot count are meant to avoid boundary saturation and\n"
        "  reduce the instability seen in the first calibration run."
    )


if __name__ == "__main__":
    main()