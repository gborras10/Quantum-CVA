import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# python utils
import time
import numpy as np

# quantum_cva utils
from quantum_cva.multi_asset.classical.classical_cva.cva_auxiliar_functions import (
    P0,
    build_survival_from_cds,
)
from quantum_cva.multi_asset.classical.probability_and_underlying.multi_asset_dynamics_utils import (
    simulate_multi_asset_gbm,
)
from quantum_cva.multi_asset.classical.classical_cva.classical_cont_cva import (
    ContinuousUnderlyingCvaEngine,
)
from quantum_cva.multi_asset.classical.classical_cva.classical_discrete_cva import (
    DiscreteUnderlyingCvaEngine,
)
from quantum_cva.multi_asset.instruments.derivatives import Forward

#  -------------------------- Simulation parameters --------------------------
underlying_initial: float = 5.0
strike: float = 5.5
sigma: float = 0.25
mu: float = 0.02
r: float = mu  # approx flat curve equal to drift

N_paths: int = int(1e5)
M: int = int(4)
seed: int = 105
rng: np.random.Generator = np.random.default_rng(seed)

# 1D normals -> adapt to (N_paths, M, d) with d=1
Z_1d: np.ndarray = rng.standard_normal(size=(N_paths, M))
Z: np.ndarray = Z_1d[:, :, None]  # (N_paths, M, 1)

R_cva: float = 0.415
R_cds: float = 0.4125
lost_given_default: float = 1.0 - R_cva

T: float = 184 / 360
t: np.ndarray = np.linspace(0.0, T, M + 1)[1:]  # exposure dates only (no t=0)

P0_flat = lambda u: P0(u, r)

# CDS specifications
cds_tenors_years: list[int] = [1, 3, 5, 7, 10, 15]
cds_spreads: list[float] = [
    0.00093772,
    0.00184451,
    0.0032286,
    0.0047065,
    0.00574888,
    0.00574888,
]

#  -------------------------- CDS Bootstrapping --------------------------
_, _, survival_curve, q_interval = build_survival_from_cds(
    P0=P0_flat,
    tenors=cds_tenors_years,
    spreads=cds_spreads,
    R_cds=R_cds,
    pay_freq=4,
)

#  -------------------------- Simulate underlying  --------------------------
rho_1d = np.array([[1.0]], dtype=float)

S_by_time_multi_asset = simulate_multi_asset_gbm(
    S0=[underlying_initial],
    mu=[mu],
    sigma=[sigma],
    rho=rho_1d,
    t=t,
    Z=Z,
    antithetic=True,
    moment_match=True,
    replications=100,
    replication_seed=12345,
    pathwise=True,
)

#  -------------------------- Portfolio (same payoff as old Vpos) --------------------------
forward_instance = [Forward(asset_idx=0, position=+1.0, K=strike, T=T)]

continuous_cva_engine = ContinuousUnderlyingCvaEngine(
    instruments=forward_instance,
    P0_func=P0_flat,
    q_interval=q_interval,
    LGD=lost_given_default,
    r=r,
)

# =======================================================================
#        Compute CVA using continuous underlying distribution 
# =======================================================================
t0: float = time.perf_counter()

cva_mc_continuous, cva_std_err_mc_continuous = continuous_cva_engine.cva_from_paths(
    S_by_time=S_by_time_multi_asset,
    t=t,
)

t1: float = time.perf_counter()

print("==========================================================")
print("Continuous Underlying Distribution")
print("===========================================================")

print(
    f"\nCVA (continuous underlying, OOP): {cva_mc_continuous}"
    f" ± {cva_std_err_mc_continuous}"
)
print(f"Computation time: {t1 - t0:.2f} seconds")

# =======================================================================
#        Compute CVA using discrete underlying distribution
# =======================================================================
def make_engine(n_bits_1d: int) -> DiscreteUnderlyingCvaEngine:
    return DiscreteUnderlyingCvaEngine(
        instruments=forward_instance,
        P0_func=P0_flat,          
        q_interval=q_interval,    
        LGD=lost_given_default,
        r=r,
        n_bits=[int(n_bits_1d)],  # 1 asset => length-1 list
        n_sigma=3.0,
        payoff_repr="left",  
        order="time_major",
        time_weights=None,
    )

#  CVA by grid size (n_bits = 1..14) 
cva_by_grid_size: dict[int, float] = {}

for n in range(1, 15):
    discrete_engine = make_engine(n)
    cva_n = discrete_engine.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset,
        t=t,
        return_blocks=False,
    )
    cva_by_grid_size[n] = float(cva_n)

grid_sizes: np.ndarray = np.array(
    sorted(cva_by_grid_size.keys()), dtype=int
)
cva_values: np.ndarray = np.array(
    [cva_by_grid_size[n] for n in grid_sizes], dtype=float
)

#  Proxy for CVA(infinite grid) 
t_0: float = time.perf_counter()

grid_size_infinite: int = 20
engine_inf = make_engine(grid_size_infinite)


# Useful for QCBM
cva_limit, grid_inf, P_joint_t_inf, v_joint_t_inf, p_target_inf, w_t_inf = (
    engine_inf.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset,
        t=t,
        return_blocks=True,
    )
)

t_1: float = time.perf_counter()
elapsed_seconds: float = t_1 - t_0

# Discretization error vs continuous MC CVA 
discretization_relative_error: float = (
    np.abs(float(cva_limit) - float(cva_mc_continuous)) / np.abs(float(cva_mc_continuous)) * 100.0
)

engine_4 = make_engine(2)
cva_4, grid_4, P_joint_t_4, v_joint_t_4, p_target_4, w_t_4 = (
    engine_4.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset,
        t=t,
        return_blocks=True,
    )
)

p_t_4 = engine_4.discount_factors_on_grid(t)
q_t_4 = engine_4.default_increments_on_grid(t)

C_p: float = float(np.max(p_t_4))
C_q: float = float(np.max(q_t_4))
C_v: float = float(np.max(v_joint_t_4))

# Optional: if you want the “scaled” CVA (should match unscaled if you pass C_* consistently)
cva_4_scaled = engine_4.cva_from_discrete_blocks(
    P_joint_t=P_joint_t_4,
    v_joint_t=v_joint_t_4,
    t=t,
    C_p=C_p,
    C_q=C_q,
    C_v=C_v,
)

# print results of the discrete CVA computation
print("==========================================================")
print("Discrete Underlying Distribution")
print("===========================================================")
print("\nCVA by grid size:")
for n, cva_val in cva_by_grid_size.items():
    print(f"n={n}: CVA={cva_val}")

print(f"\nCVA_inf = {cva_limit}")
print(f"Elapsed time [s] = {elapsed_seconds}")
print(
    "\nRelative error between continuous CVA and CVA(inf):"
    f" {discretization_relative_error} %"
)

print(f"\nScaling constants for 4 bins: C_p={C_p}, C_q={C_q}, C_v={C_v}")
print(f"CVA (4 bins, scaled) = {cva_4_scaled}")

# -------------------- Save classical tables and reference values --------------------
path = pathlib.Path(__file__).resolve().parents[4] / "data" / "single_asset" / "benchmark" / "run_classical_cva_single_asset.npz"
path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    path,
    # ── time grid ────────────────────────────────────────────────
    t=t,
    # ── discount factors & default increments ───────────────────
    p_t=p_t_4,
    q_t=q_t_4,
    # ── discrete grid spec (d=1 here, but file format supports multi-asset) ──
    edges_list=np.array([grid_4.edges_list[0]], dtype=object),
    rep_list=np.array([grid_4.rep_list[0]], dtype=object),
    n_bins=np.array(grid_4.n_bins, dtype=int),
    N_joint=np.array(grid_4.N_joint, dtype=int),
    # ── core matrices (conditional + payoff on grid) ─────────────
    P_joint_t=P_joint_t_4,
    v_joint_t=v_joint_t_4,
    # ── QCBM target ──────────────────────────────────────────────
    p_target=p_target_4,
    w_t=w_t_4,
    # ── scaling constants ────────────────────────────────────────
    C_p=np.array(C_p),
    C_q=np.array(C_q),
    C_v=np.array(C_v),
    # ── CVA reference values ─────────────────────────────────────
    cva_mc_continuous=np.array(cva_mc_continuous),
    cva_std_err_mc_continuous=np.array(cva_std_err_mc_continuous),
    cva_limit=np.array(cva_limit),
    # ── Discrete CVA values ─────────────────────────────────────
    grid_sizes=grid_sizes,
    cva_by_grid_size_values=cva_values,
    # ── simulation / contract parameters ─────────────────────────
    S0=np.array(underlying_initial),
    K=np.array(strike),
    sigma=np.array(sigma),
    mu=np.array(mu),
    r=np.array(r),
    T=np.array(T),
    R_cva=np.array(R_cva),
    R_cds=np.array(R_cds),
    LGD=np.array(lost_given_default),
    M=np.array(M),
    n_bits=np.array(2),
    n_sigma=np.array(3.0),
)
