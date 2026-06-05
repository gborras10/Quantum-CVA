import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# python utils
import time
import numpy as np

# quantum_cva utils
from quantum_cva.multi_asset.classical.classical_cva.cva_auxiliar_functions import (
    P0,
    Instrument,
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
from quantum_cva.multi_asset.instruments.derivatives import Call, Forward, Put
from quantum_cva.multi_asset.instruments.market_data import MarketData
from quantum_cva.multi_asset.classical.probability_and_underlying.piecewise_volatility_utils import (
    build_integrated_covariance_grid,
    build_piecewise_sigma_grid,
    build_piecewise_vol_curve_for_underlying,
    build_residual_volatility_function_for_underlying,
)

# ======================================================================
#                         Load market data
# ======================================================================
market_data = MarketData.load(
    discount_curve_path="data/loaded_market_data/discount_curve.xlsx",
    credit_data_path="data/loaded_market_data/iberdrola_data.xlsx",
    historical_series_path="data/loaded_market_data/time_series.xlsx",
    atm_vol_surfaces_path="data/loaded_market_data/vol_surfaces.xlsx",
    valuation_date="2026-03-15",
)

underlyings = [".STOXX50E", ".SSMI"]

# ======================================================================
#                         Real market data parameters
# ======================================================================
# Get real spot prices
S0_list: list[float] = market_data.get_spot_vector(underlyings)

# Get real ATM volatilities (average across time for stability)
atm_vol_curves = market_data.get_atm_vol_curves(underlyings=underlyings)

# Get real correlation matrix from log returns
rho_3d: np.ndarray = market_data.get_log_return_correlation(underlyings)

# Get real discount factor curve
P0_flat = lambda u: market_data.discount_factor(u)

# Calculate drift from discount factor and dividend yields
r: float = -np.log(P0_flat(1.0))

# Dividend yields for the 2 assets
div_yields =  [0.0224722, 0.0316306]

# Drift for simulation: mu = r - div_yield
mu_list: list[float] = [r - dy for dy in div_yields]

# Maturities
maturity_call: float = 3 / 12  # 3 months for the call on Eurostoxx
#maturity_fwd: float = 2 / 12  # 2 months for the forward on FTSE
maturity_put: float = 6 / 12  # 6 months for the put on SMI
T: float = max(maturity_call, maturity_put)

# Time grid
m: int = int(2)
M: int = 2**m
M_fine_grid: int = 252 # for fine grid convergence analysis
t: np.ndarray = np.linspace(0.0, T, M + 1)[1:]  # exposure dates only (no t=0)
t_fine: np.ndarray = np.linspace(0.0, T, M_fine_grid + 1)[1:]  # fine grid for convergence analysis

# Monte Carlo controls
N_paths: int = int(1e5)
seed: int = 105
rng: np.random.Generator = np.random.default_rng(seed)

d: int = 2  # number of assets

# False reproduces the historical right-endpoint volatility approximation.
# True integrates all market volatility buckets crossed by each time step.
USE_INTEGRATED_BUCKET_DYNAMICS: bool = False

# Normals Z with shape (N_paths, M, d)
Z: np.ndarray = rng.standard_normal(size=(N_paths, M, d))

# ======================================================================
#                         CDS / survival curve
# ======================================================================
R_cva: float = 0.415
R_cds: float = market_data.recovery_rate
lost_given_default: float = 1.0 - R_cva

cds_tenors_years, cds_spreads = market_data.get_cds_curve()

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
def build_right_endpoint_sigma_grid(sim_times: np.ndarray) -> np.ndarray:
    """Legacy approximation: use the bucket at each step's right endpoint."""
    sigma_cols: list[np.ndarray] = []
    for underlying in underlyings:
        maturities, sigma_pw = build_piecewise_vol_curve_for_underlying(
            atm_vol_curves=atm_vol_curves,
            underlying=underlying,
        )
        idx = np.searchsorted(maturities, sim_times, side="left")
        sigma_cols.append(sigma_pw[idx])
    return np.column_stack(sigma_cols)


if USE_INTEGRATED_BUCKET_DYNAMICS:
    sigma_grid = build_piecewise_sigma_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=underlyings,
        sim_times=t,
    )
    integrated_covariance_grid = build_integrated_covariance_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=underlyings,
        sim_times=t,
        rho=rho_3d,
    )
    sigma_grid_fine = build_piecewise_sigma_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=underlyings,
        sim_times=t_fine,
    )
    integrated_covariance_grid_fine = build_integrated_covariance_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=underlyings,
        sim_times=t_fine,
        rho=rho_3d,
    )
else:
    sigma_grid = build_right_endpoint_sigma_grid(t)
    integrated_covariance_grid = None
    sigma_grid_fine = build_right_endpoint_sigma_grid(t_fine)
    integrated_covariance_grid_fine = None

S_by_time_multi_asset = simulate_multi_asset_gbm(
    S0=S0_list,
    mu=mu_list,
    sigma=sigma_grid,
    rho=rho_3d,
    t=t,
    Z=Z,
    antithetic=True,
    moment_match=True,
    replications=1,
    replication_seed=12345,
    pathwise=True,
    integrated_covariances=integrated_covariance_grid,
)

S_by_time_multi_asset_fine = simulate_multi_asset_gbm(
    S0=S0_list,
    mu=mu_list,
    sigma=sigma_grid_fine,
    rho=rho_3d,
    t=t_fine,
    Z=rng.standard_normal(size=(N_paths, M_fine_grid, d)),
    antithetic=True,
    moment_match=True,
    replications=1,
    replication_seed=12345,
    pathwise=True,
    integrated_covariances=integrated_covariance_grid_fine,
)

# ======================================================================
#                         Portfolio of 2 derivatives instance
# ======================================================================
K_list: list[float] = [
    4500.0,  # strike for call on Eurostoxx
    12500.0,  # strike for put on SMI
]  

sigma_func_stoxx = build_residual_volatility_function_for_underlying(
    atm_vol_curves=atm_vol_curves,
    underlying=".STOXX50E",
)

sigma_func_ssmi = build_residual_volatility_function_for_underlying(
    atm_vol_curves=atm_vol_curves,
    underlying=".SSMI",
)

instruments = [
    Call(
        asset_idx=0,
        quantity=1,
        multiplier=4,
        K=K_list[0],
        T=maturity_call,
        sigma_func=sigma_func_stoxx,
    ),
    Put(
        asset_idx=1,
        quantity=-1,
        multiplier=2,
        K=K_list[1],
        T=maturity_put,
        sigma_func=sigma_func_ssmi,
    ),
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

elapsed_seconds = time.perf_counter() - t0

print("==========================================================")
print("Continuous Underlying Distribution (3 assets)")
print("===========================================================")

print(
    f"CVA (continuous underlying): {cva_mc_continuous}"
    f" ± {cva_std_err_mc_continuous}"
)
print(f"Computation time: {elapsed_seconds:.2f} seconds")

# =====================================================================
#       Compute CVA using continuous underlying distribution on fine grid
# =====================================================================
t0_fine: float = time.perf_counter()
cva_mc_continuous_fine, cva_std_err_mc_continuous_fine = continuous_cva_engine.cva_from_paths(
    S_by_time=S_by_time_multi_asset_fine,
    t=t_fine,
)
elapsed_seconds_fine = time.perf_counter() - t0_fine
print("\n==========================================================")
print("Continuous Underlying Distribution (3 assets) - Fine Grid")
print("===========================================================")
print(
    f"CVA (continuous underlying, fine grid): {cva_mc_continuous_fine}"
    f" ± {cva_std_err_mc_continuous_fine}"
)
print(f"Computation time (fine grid): {elapsed_seconds_fine:.2f} seconds")

# ======================================================================
#        Compute CVA using discrete underlying distribution
# ======================================================================
# utility for analysis of convergence as grid size increases
def make_engine(n_bits: int | list[int]) -> DiscreteUnderlyingCvaEngine:
    if isinstance(n_bits, int):
        n_bits_list = [int(n_bits), int(n_bits)]
    else:
        n_bits_list = [int(x) for x in n_bits]

    return DiscreteUnderlyingCvaEngine(
        instruments=instruments,
        P0_func=P0_flat,
        q_interval=q_interval,
        LGD=lost_given_default,
        r=r,
        n_bits=n_bits_list,
        n_sigma=3.0,
        payoff_repr="left",
        order="time_major",
        time_weights=None,
    )

# CVA by grid size (n_bits = 1..10)
cva_by_grid_size: dict[int, float] = {}

for n in range(1, 11):
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

grid_size_infinite: int = 13  # chosen as a proxy for infinite grid (2^13 = 8192 bins per asset, total 67 million joint bins)
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

t_fine: np.ndarray = np.linspace(0.0, T, 50)[1:]
Z_fine: np.ndarray = rng.standard_normal(size=(N_paths, t_fine.size, d))
if USE_INTEGRATED_BUCKET_DYNAMICS:
    sigma_grid_fine = build_piecewise_sigma_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=underlyings,
        sim_times=t_fine,
    )
    integrated_covariance_grid_fine = build_integrated_covariance_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=underlyings,
        sim_times=t_fine,
        rho=rho_3d,
    )
    sigma_times_fine = None
else:
    # Preserve the original small-grid construction: resample coarse rows.
    sigma_grid_fine = sigma_grid
    integrated_covariance_grid_fine = None
    sigma_times_fine = t

S_by_time_multi_asset_fine = simulate_multi_asset_gbm(
    S0=S0_list,
    mu=mu_list,
    sigma=sigma_grid_fine,
    sigma_times=sigma_times_fine,
    rho=rho_3d,
    t=t_fine,
    Z=Z_fine,
    antithetic=True,
    moment_match=True,
    replications=1,
    replication_seed=12345,
    pathwise=True,
    integrated_covariances=integrated_covariance_grid_fine,
)

engine_small = make_engine([2, 2])  # bins per asset = (4, 4)
cva_small, grid_small, P_joint_t_small, v_joint_t_small, p_target_small, w_t_small = (
    engine_small.cva_from_paths_discretized(
        S_by_time=S_by_time_multi_asset_fine,
        t=t_fine,
        t_output=t,
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

small_discretization_relative_error: float = (
    np.abs(float(cva_small) - float(cva_mc_continuous))    / np.abs(float(cva_mc_continuous))
    * 100.0
)

print("\n==========================================================")
print("Discrete Underlying Distribution (3 assets)")
print("===========================================================")
print("CVA by grid size:")
for n, cva_val in cva_by_grid_size.items():
    print(f"n={n}: CVA={cva_val}")

print(f"CVA_inf(n={grid_size_infinite}) = {cva_limit}")
print(f"Elapsed time [s] = {elapsed_seconds}")
print(
    "\nRelative error between continuous CVA and CVA(inf):"
    f" {discretization_relative_error} %"
)
print(f"Small grid (n={grid_small.N_joint}) relative error: {small_discretization_relative_error} %")
print(f"\nScaling constants for small grid: C_p={C_p}, C_q={C_q}, C_v={C_v}")
print(f"CVA (small grid, scaled) = {cva_small_scaled}")

# ======================================================================
#                         Save benchmark tables
# ======================================================================
repo_root = next(
    parent
    for parent in pathlib.Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

path = repo_root / "data" / "multi_asset" / "6q_instance" / "benchmark" / "three_asset_instance.npz"
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
    # ── CVA reference values ────────5─────────────────────────────
    cva_mc_continuous=np.array(cva_mc_continuous),
    cva_std_err_mc_continuous=np.array(cva_std_err_mc_continuous),
    cva_limit=np.array(cva_limit),
    # ── Discrete CVA values ─────────────────────────────────────
    grid_sizes=grid_sizes,
    cva_by_grid_size_values=cva_values,
    # ── simulation / contract parameters ─────────────────────────
    S0=np.array(S0_list, dtype=float),
    K=np.array(K_list, dtype=float),
    sigma=np.array(sigma_grid, dtype=float),
    mu=np.array(mu_list, dtype=float),
    r=np.array(r),
    T=np.array(T),
    rho=np.array(rho_3d, dtype=float),
    R_cva=np.array(R_cva),
    R_cds=np.array(R_cds),
    LGD=np.array(lost_given_default),
    M=np.array(M),
    n_bits_small=np.array([2, 2], dtype=int),
    n_sigma=np.array(3.0),
)
