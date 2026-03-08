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


# ======================================================================
#                         Simulation parameters
# ======================================================================
# Three underlyings
S0_list: list[float] = [5.0, 6.5, 4.2]
mu_list: list[float] = [0.02, 0.02, 0.02]
sigma_list: list[float] = [0.25, 0.22, 0.30]

# Flat curve (keep same style as your single-asset script)
r: float = float(0.02)
P0_flat = lambda u: P0(u, r)

# Time grid
M: int = int(4)
T: float = 184 / 360
t: np.ndarray = np.linspace(0.0, T, M + 1)[1:]  # exposure dates only (no t=0)

# Monte Carlo controls
N_paths: int = int(1e5)
seed: int = 105
rng: np.random.Generator = np.random.default_rng(seed)

d: int = 3  # number of assets

# Normals Z with shape (N_paths, M, d)
Z: np.ndarray = rng.standard_normal(size=(N_paths, M, d))

# Correlation matrix (SPD, reasonable correlations)
rho_3d: np.ndarray = np.array(
    [
        [1.0, 0.55, 0.25],
        [0.55, 1.0, 0.35],
        [0.25, 0.35, 1.0],
    ],
    dtype=float,
)

# ======================================================================
#                         CDS / survival curve
# ======================================================================
R_cva: float = 0.415
R_cds: float = 0.4125
lost_given_default: float = 1.0 - R_cva

cds_tenors_years: list[int] = [1, 3, 5, 7, 10, 15]
cds_spreads: list[float] = [
    0.00093772,
    0.00184451,
    0.0032286,
    0.0047065,
    0.00574888,
    0.00574888,
]

_, _, survival_curve, q_interval = build_survival_from_cds(
    P0=P0_flat,
    tenors=cds_tenors_years,
    spreads=cds_spreads,
    R_cds=R_cds,
    pay_freq=4,
)

# ======================================================================
#                 Simulate multi-asset GBM (correlated)
# ======================================================================
S_by_time_multi_asset = simulate_multi_asset_gbm(
    S0=S0_list,
    mu=mu_list,
    sigma=sigma_list,
    rho=rho_3d,
    t=t,
    Z=Z,
    antithetic=True,
    moment_match=True,
    replications=100,
    replication_seed=12345,
    pathwise=True,
)

# ======================================================================
#                         Portfolio: 3 forwards
# ======================================================================
K_list: list[float] = [5.5, 6.7, 4.0]

instruments = [
    Forward(asset_idx=0, position=+1.0, K=K_list[0], T=T),
    Forward(asset_idx=1, position=+1.0, K=K_list[1], T=T),
    Forward(asset_idx=2, position=+1.0, K=K_list[2], T=T),
]

continuous_cva_engine = ContinuousUnderlyingCvaEngine(
    instruments=instruments,
    P0_func=P0_flat,
    q_interval=q_interval,
    LGD=lost_given_default,
    r=r,
)

# ======================================================================
#        Compute CVA using continuous underlying distribution
# ======================================================================
t0: float = time.perf_counter()

cva_mc_continuous, cva_std_err_mc_continuous = continuous_cva_engine.cva_from_paths(
    S_by_time=S_by_time_multi_asset,
    t=t,
)

t1: float = time.perf_counter()

print("==========================================================")
print("Continuous Underlying Distribution (3 assets)")
print("===========================================================")

print(
    f"\nCVA (continuous underlying, OOP): {cva_mc_continuous}"
    f" ± {cva_std_err_mc_continuous}"
)
print(f"Computation time: {t1 - t0:.2f} seconds")

# ======================================================================
#        Compute CVA using discrete underlying distribution
# ======================================================================
# utility for analysis of convergence as grid size increases
def make_engine(n_bits_1d: int) -> DiscreteUnderlyingCvaEngine:
    return DiscreteUnderlyingCvaEngine(
        instruments=instruments,
        P0_func=P0_flat,
        q_interval=q_interval,
        LGD=lost_given_default,
        r=r,
        # 3 assets => list length 3
        n_bits=[int(n_bits_1d), int(n_bits_1d), int(n_bits_1d)],
        n_sigma=3.0,
        payoff_repr="left",
        order="time_major",
        time_weights=None,
    )

# CVA by grid size (n_bits = 1..10)
cva_by_grid_size: dict[int, float] = {}

for n in range(1, 6):
    discrete_engine = make_engine(n)
    cva_n = discrete_engine.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset,
        t=t,
        return_blocks=False,
    )
    cva_by_grid_size[n] = float(cva_n)

grid_sizes: np.ndarray = np.array(sorted(cva_by_grid_size.keys()), dtype=int)
cva_values: np.ndarray = np.array([cva_by_grid_size[n] for n in grid_sizes], dtype=float)

# Proxy for CVA(infinite grid)
t_0: float = time.perf_counter()

grid_size_infinite: int = 6
engine_inf = make_engine(grid_size_infinite)

cva_limit, grid_inf, P_joint_t_inf, v_joint_t_inf, p_target_inf, w_t_inf = (
    engine_inf.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset,
        t=t,
        return_blocks=True,
    )
)

t_1: float = time.perf_counter()
elapsed_seconds: float = t_1 - t_0

discretization_relative_error: float = ( 
    np.abs(float(cva_limit) - float(cva_mc_continuous))
    / np.abs(float(cva_mc_continuous))
    * 100.0
)

engine_small = make_engine(2)  # 4 bins per asset -> N_joint = 64
cva_small, grid_small, P_joint_t_small, v_joint_t_small, p_target_small, w_t_small = (
    engine_small.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset,
        t=t,
        return_blocks=True,
    )
)

p_t_small = engine_small.discount_factors_on_grid(t)
q_t_small = engine_small.default_increments_on_grid(t)

C_p: float = float(np.max(p_t_small))
C_q: float = float(np.max(q_t_small))
C_v: float = float(np.max(v_joint_t_small))

cva_small_scaled = engine_small.cva_from_discrete_blocks(
    P_joint_t=P_joint_t_small,
    v_joint_t=v_joint_t_small,
    t=t,
    C_p=C_p,
    C_q=C_q,
    C_v=C_v,
)

print("==========================================================")
print("Discrete Underlying Distribution (3 assets)")
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

print(f"\nScaling constants for small grid: C_p={C_p}, C_q={C_q}, C_v={C_v}")
print(f"CVA (small grid, scaled) = {cva_small_scaled}")

# ======================================================================
#                         Save benchmark tables
# ======================================================================
path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "data"
    / "multi_asset"
    / "benchmark"
    / "three_asset_instance.npz"
)
path.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    path,
    # ── time grid ────────────────────────────────────────────────
    t=t,
    # ── discount factors & default increments ───────────────────
    p_t=p_t_small,
    q_t=q_t_small,
    # ── discrete grid spec ───────────────────────────────────────
    edges_list=np.array(grid_small.edges_list, dtype=object),
    rep_list=np.array(grid_small.rep_list, dtype=object),
    n_bins=np.array(grid_small.n_bins, dtype=int),
    N_joint=np.array(grid_small.N_joint, dtype=int),
    # ── core tensors (conditional/joint + payoff on grid) ─────────
    P_joint_t=P_joint_t_small,
    v_joint_t=v_joint_t_small,
    # ── QCBM target ──────────────────────────────────────────────
    p_target=p_target_small,
    w_t=w_t_small,
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
    S0=np.array(S0_list, dtype=float),
    K=np.array(K_list, dtype=float),
    sigma=np.array(sigma_list, dtype=float),
    mu=np.array(mu_list, dtype=float),
    r=np.array(r),
    T=np.array(T),
    rho=np.array(rho_3d, dtype=float),
    R_cva=np.array(R_cva),
    R_cds=np.array(R_cds),
    LGD=np.array(lost_given_default),
    M=np.array(M),
    n_bits_small=np.array(2),
    n_sigma=np.array(3.0),
)