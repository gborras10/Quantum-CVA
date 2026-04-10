# python utils
import numpy as np
from collections.abc import Sequence
from scipy.stats import norm

def simulate_multi_asset_gbm(
    S0: Sequence[float],
    mu: Sequence[float],
    sigma: Sequence[float] | np.ndarray,
    rho: np.ndarray,
    t: np.ndarray,
    Z: np.ndarray,
    *,
    antithetic: bool = True,
    moment_match: bool = False,
    replications: int = 1,
    replication_seed: int = 12345,
    pathwise: bool = True,
    sigma_times: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns S_by_time with shape (N_paths_eff, M, d).

    sigma can be:
      - shape (d,)   : constant volatility per asset
      - shape (M, d) : piecewise-constant volatility per time interval and asset

        Optional
        --------
        sigma_times:
            - if sigma has shape (M_sigma, d) and M_sigma != M,
                provide the time grid associated with sigma rows.
            - if omitted, and M_sigma != M, a uniform grid on (0, t[-1]]
                with M_sigma points is assumed.
    """
    t = np.asarray(t, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)

    S0 = np.asarray(S0, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if Z.ndim != 3:
        raise ValueError("Z must have shape (N_paths, M, d).")

    N_paths, M, num_assets = Z.shape

    if t.shape != (M,):
        raise ValueError(
            "t and Z must have matching time dimension: "
            f"len(t)={t.size} but Z.shape[1]={M}."
        )

    if S0.shape != (num_assets,):
        raise ValueError("S0 must have shape (d,).")
    if mu.shape != (num_assets,):
        raise ValueError("mu must have shape (d,).")
    if sigma.ndim == 1:
        if sigma.shape != (num_assets,):
            raise ValueError("If sigma is 1D, it must have shape (d,).")
        sigma_grid = np.broadcast_to(sigma[None, :], (M, num_assets))
    elif sigma.ndim == 2:
        if sigma.shape[1] != num_assets:
            raise ValueError("If sigma is 2D, it must have shape (M_sigma, d).")

        M_sigma = int(sigma.shape[0])
        if M_sigma == M:
            sigma_grid = sigma
        elif M_sigma == 1:
            sigma_grid = np.broadcast_to(sigma, (M, num_assets))
        else:
            if sigma_times is None:
                sigma_time_grid = np.linspace(0.0, float(t[-1]), M_sigma + 1, dtype=float)[1:]
            else:
                sigma_time_grid = np.asarray(sigma_times, dtype=float).ravel()
                if sigma_time_grid.shape != (M_sigma,):
                    raise ValueError(
                        "sigma_times must have shape (M_sigma,) aligned with sigma rows."
                    )

            if np.any(np.diff(sigma_time_grid) <= 0.0):
                raise ValueError("sigma_times must be strictly increasing.")

            tol = 1e-12
            if float(t[-1]) > float(sigma_time_grid[-1]) + tol:
                raise ValueError(
                    "t extends beyond sigma_time_grid upper bound when resampling 2D sigma."
                )

            # Piecewise-constant assignment on intervals (T_{j-1}, T_j], with
            # left extension to (0, T_1] using the first sigma row.
            idx = np.searchsorted(sigma_time_grid, t, side="left")
            idx = np.clip(idx, 0, M_sigma - 1)
            sigma_grid = sigma[idx, :]
    else:
        raise ValueError("sigma must have shape (d,) or (M_sigma, d).")

    # variance reduction techniques
    if replications > 1:
        rng = np.random.default_rng(int(replication_seed))
        blocks = [Z]
        for _ in range(replications - 1):
            blocks.append(rng.standard_normal(size=Z.shape))
        Z = np.vstack(blocks)

    if antithetic:
        Z = np.vstack([Z, -Z])

    if moment_match:
        mean = Z.mean(axis=0, keepdims=True)
        std = Z.std(axis=0, ddof=0, keepdims=True)
        if np.any(std <= 0.0):
            raise ValueError("Moment matching failed: zero-variance slice.")
        Z = (Z - mean) / std

    # correlate normals using Cholesky
    L = np.linalg.cholesky(rho)
    Lt = np.ascontiguousarray(L.T)

    # increase memory contiguity for better performance in the matrix multiplication below
    Zc = (np.ascontiguousarray(Z).reshape(-1, num_assets) @ Lt).reshape(Z.shape) # shape (N_paths_eff, M, d)
    #Zc = Z @ L.T

    # simulate correlated GBM paths
    log_S0 = np.log(S0)
    if pathwise:
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("Exposure dates must be strictly increasing when pathwise=True.")

        dt = np.diff(np.concatenate(([0.0], t)))
        sqrt_dt = np.sqrt(dt)

        drift_increment = (mu[None, :] - 0.5 * sigma_grid**2)[None, :, :] * dt[None, :, None] # broadcast to shape (M, d) then (1, M, d)
        diff_increment  = sigma_grid[None, :, :] * sqrt_dt[None, :, None] * Zc # broadcast to shape (M, d) then (1, M, d)

        log_S = log_S0[None, None, :] + np.cumsum(drift_increment + diff_increment, axis=1)
        S = np.exp(log_S)

    else:
        if np.any(t <= 0.0):
            raise ValueError("All exposure dates must be > 0.")
        if sigma.ndim != 1:
            raise ValueError("pathwise=False only supports constant sigma with shape (d,).")

        sqrt_t = np.sqrt(t)
        drift  = (mu - 0.5 * sigma**2)[None, None, :] * t[None, :, None]
        diff   = sigma[None, None, :] * sqrt_t[None, :, None] * Zc

        log_S = log_S0[None, None, :] + drift + diff
        S = np.exp(log_S)

    return S