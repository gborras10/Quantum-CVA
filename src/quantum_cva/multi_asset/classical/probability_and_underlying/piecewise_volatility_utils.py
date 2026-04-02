# python utils
import math
import numpy as np
import pandas as pd
from collections.abc import Callable

_EPS_TIME = 1e-12


def bootstrap_piecewise_vol_from_atm(
    maturities: np.ndarray,
    atm_vols: np.ndarray,
    *,
    var_floor: float = 1e-14,
) -> np.ndarray:
    """
    Convert ATM implied vols by maturity into piecewise-constant forward vols
    on intervals (T_{i-1}, T_i], with T_0 = 0.
    """
    maturities = np.asarray(maturities, dtype=float).ravel()
    atm_vols = np.asarray(atm_vols, dtype=float).ravel()

    if maturities.shape != atm_vols.shape:
        raise ValueError("maturities and atm_vols must have the same shape.")
    if maturities.size == 0:
        raise ValueError("Need at least one maturity.")
    if np.any(maturities <= 0.0):
        raise ValueError("All maturities must be > 0.")
    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("maturities must be strictly increasing.")
    if np.any(atm_vols < 0.0):
        raise ValueError("atm_vols must be non-negative.")

    total_var = atm_vols**2 * maturities

    T_prev = np.concatenate(([0.0], maturities[:-1]))
    w_prev = np.concatenate(([0.0], total_var[:-1]))
    dT = maturities - T_prev

    fwd_var = (total_var - w_prev) / dT
    fwd_var = np.maximum(fwd_var, var_floor)

    return np.sqrt(fwd_var)


def map_piecewise_vol_to_sim_grid(
    market_maturities: np.ndarray,
    sigma_pw: np.ndarray,
    sim_times: np.ndarray,
) -> np.ndarray:
    """
    Map market piecewise vols defined on intervals (T_{i-1}, T_i]
    onto simulation intervals (t_{j-1}, t_j].
    """
    market_maturities = np.asarray(market_maturities, dtype=float).ravel()
    sigma_pw = np.asarray(sigma_pw, dtype=float).ravel()
    sim_times = np.asarray(sim_times, dtype=float).ravel()

    if market_maturities.shape != sigma_pw.shape:
        raise ValueError("market_maturities and sigma_pw must have the same shape.")
    if sim_times.size == 0:
        raise ValueError("sim_times cannot be empty.")
    if np.any(np.diff(sim_times) <= 0.0):
        raise ValueError("sim_times must be strictly increasing.")
    if sim_times[-1] > market_maturities[-1]:
        raise ValueError("Simulation grid extends beyond last market maturity.")

    idx = np.searchsorted(market_maturities, sim_times, side="left")
    return sigma_pw[idx]


def build_piecewise_sigma_grid(
    atm_vol_curves: pd.DataFrame,
    underlyings: list[str],
    sim_times: np.ndarray,
) -> np.ndarray:
    sigma_cols: list[np.ndarray] = []

    for u in underlyings:
        df_u = (
            atm_vol_curves.loc[
                atm_vol_curves["underlying"] == u,
                ["t", "atm_vol"],
            ]
            .dropna()
            .sort_values("t")
            .copy()
        )

        maturities_u = df_u["t"].to_numpy(dtype=float)
        atm_vols_u = df_u["atm_vol"].to_numpy(dtype=float)

        sigma_pw_u = bootstrap_piecewise_vol_from_atm(
            maturities=maturities_u,
            atm_vols=atm_vols_u,
        )

        sigma_grid_u = map_piecewise_vol_to_sim_grid(
            market_maturities=maturities_u,
            sigma_pw=sigma_pw_u,
            sim_times=sim_times,
        )

        sigma_cols.append(sigma_grid_u)

    return np.column_stack(sigma_cols)


def build_piecewise_vol_curve_for_underlying(
    atm_vol_curves: pd.DataFrame,
    underlying: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return market maturities and piecewise-constant forward vols for one underlying.
    """
    df_u = (
        atm_vol_curves.loc[
            atm_vol_curves["underlying"] == underlying,
            ["t", "atm_vol"],
        ]
        .dropna()
        .sort_values("t")
        .copy()
    )

    if df_u.empty:
        raise ValueError(f"No ATM vol data found for underlying={underlying}.")

    maturities = df_u["t"].to_numpy(dtype=float)
    atm_vols = df_u["atm_vol"].to_numpy(dtype=float)

    sigma_pw = bootstrap_piecewise_vol_from_atm(
        maturities=maturities,
        atm_vols=atm_vols,
    )
    return maturities, sigma_pw


def integrated_piecewise_variance(
    t0: float,
    T: float,
    market_maturities: np.ndarray,
    sigma_pw: np.ndarray,
) -> float:
    """
    Integrated variance over (t0, T] for a piecewise-constant vol curve.
    """
    t0 = float(t0)
    T = float(T)

    market_maturities = np.asarray(market_maturities, dtype=float).ravel()
    sigma_pw = np.asarray(sigma_pw, dtype=float).ravel()

    if market_maturities.shape != sigma_pw.shape:
        raise ValueError("market_maturities and sigma_pw must have the same shape.")
    if T < t0 - _EPS_TIME:
        return 0.0
    if T <= t0 + _EPS_TIME:
        return 0.0
    if T > market_maturities[-1] + _EPS_TIME:
        raise ValueError("Requested maturity exceeds last point of the vol curve.")

    left_edges = np.concatenate(([0.0], market_maturities[:-1]))
    right_edges = market_maturities

    total_var = 0.0

    for left, right, sigma in zip(left_edges, right_edges, sigma_pw, strict=False):
        overlap_left = max(t0, float(left))
        overlap_right = min(T, float(right))

        if overlap_right > overlap_left:
            total_var += float(sigma)**2 * (overlap_right - overlap_left)

    return total_var


def residual_equivalent_vol(
    t0: float,
    T: float,
    market_maturities: np.ndarray,
    sigma_pw: np.ndarray,
    *,
    var_floor: float = 0.0,
) -> float:
    """
    Equivalent flat volatility on (t0, T] matching the integrated piecewise variance.
    """
    tau = float(T) - float(t0)

    if tau <= _EPS_TIME:
        return 0.0

    total_var = integrated_piecewise_variance(
        t0=t0,
        T=T,
        market_maturities=market_maturities,
        sigma_pw=sigma_pw,
    )
    total_var = max(total_var, float(var_floor))

    return math.sqrt(total_var / tau)


def make_residual_volatility_function(
    market_maturities: np.ndarray,
    sigma_pw: np.ndarray,
) -> Callable[[float, float], float]:
    """
    Build sigma_func(t, T) for option pricing under deterministic piecewise vol.
    """
    market_maturities = np.asarray(market_maturities, dtype=float).ravel()
    sigma_pw = np.asarray(sigma_pw, dtype=float).ravel()

    if market_maturities.shape != sigma_pw.shape:
        raise ValueError("market_maturities and sigma_pw must have the same shape.")

    def sigma_func(t: float, T: float) -> float:
        return residual_equivalent_vol(
            t0=float(t),
            T=float(T),
            market_maturities=market_maturities,
            sigma_pw=sigma_pw,
        )

    return sigma_func


def build_residual_volatility_function_for_underlying(
    atm_vol_curves: pd.DataFrame,
    underlying: str,
) -> Callable[[float, float], float]:
    """
    Convenience wrapper: build sigma_func(t, T) directly from ATM vol data.
    """
    market_maturities, sigma_pw = build_piecewise_vol_curve_for_underlying(
        atm_vol_curves=atm_vol_curves,
        underlying=underlying,
    )
    return make_residual_volatility_function(
        market_maturities=market_maturities,
        sigma_pw=sigma_pw,
    )