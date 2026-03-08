# python utils
import numpy as np
from collections.abc import Sequence
from scipy.stats import norm

def simulate_multi_asset_gbm(
    S0: Sequence[float],
    mu: Sequence[float],
    sigma: Sequence[float],
    rho: np.ndarray,
    t: np.ndarray,
    Z: np.ndarray,
    *,
    antithetic: bool = True,
    moment_match: bool = False,
    replications: int = 1,
    replication_seed: int = 12345,
    pathwise: bool = True,
) -> list[np.ndarray]:
    """
    Returns S_by_time: list length M, each array shape (N_paths_eff, d).
    """
    t = np.asarray(t, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float)

    S0 = np.asarray(S0, dtype=float).ravel()  # (3x1)
    mu = np.asarray(mu, dtype=float).ravel()  # (3x1)
    sigma = np.asarray(sigma, dtype=float).ravel()  # (3x1)
    rho = np.asarray(rho, dtype=float)  # (3x3)

    N_paths, M, num_assets = Z.shape

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
    Zc = (np.ascontiguousarray(Z).reshape(-1, num_assets) @ Lt).reshape(Z.shape)
    #Zc = Z @ L.T

    N_eff = Zc.shape[0]

    # simulate correlated GBM paths
    log_S0 = np.log(S0)
    if pathwise:
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("Exposure dates must be strictly increasing when pathwise=True.")

        dt = np.diff(np.concatenate(([0.0], t)))
        sqrt_dt = np.sqrt(dt)

        drift_inc = (mu - 0.5 * sigma**2)[None, None, :] * dt[None, :, None]          # (1,M,A)
        diff_inc  = sigma[None, None, :] * sqrt_dt[None, :, None] * Zc                 # (N,M,A)

        log_S = log_S0[None, None, :] + np.cumsum(drift_inc + diff_inc, axis=1)        # (N,M,A)
        S = np.exp(log_S)

    else:
        if np.any(t <= 0.0):
            raise ValueError("All exposure dates must be > 0.")

        sqrt_t = np.sqrt(t)
        drift  = (mu - 0.5 * sigma**2)[None, None, :] * t[None, :, None]               # (1,M,A)
        diff   = sigma[None, None, :] * sqrt_t[None, :, None] * Zc                     # (N,M,A)

        log_S = log_S0[None, None, :] + drift + diff
        S = np.exp(log_S)

    return S