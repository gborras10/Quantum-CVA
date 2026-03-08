# python utils
import math
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm

# global constants
_EPS_TAU = 1e-12  # threshold for treating tau=T-t as zero

# =========================
# Derivatives instruments 
# =========================
@dataclass(frozen=True, slots=True)
class Forward:
    asset_idx: int
    position: float  # +1 long, -1 short (or any notional scaling)
    K: float
    T: float

    def mtm_at_t(self, S_t: np.ndarray, *, r: float, t: float) -> np.ndarray:
        """PV at time t of a long forward."""
        S_t = np.asarray(S_t, dtype=float)
        tau = float(self.T) - float(t)

        if tau < -_EPS_TAU:
            return np.zeros_like(S_t, dtype=float)

        if abs(tau) <= _EPS_TAU:
            return S_t - float(self.K)

        return S_t - float(self.K) * np.exp(-float(r) * tau)

@dataclass(frozen=True, slots=True)
class Call:
    asset_idx: int
    position: float
    K: float
    T: float
    sigma: float

    def mtm_at_t(self, S_t: np.ndarray, *, r: float, t: float) -> np.ndarray:
        """PV at time t of a long European call."""
        S_t = np.asarray(S_t, dtype=float)
        K = float(self.K)
        r = float(r)
        sigma = float(self.sigma)

        tau = float(self.T) - float(t)

        if tau < -_EPS_TAU:
            return np.zeros_like(S_t, dtype=float)

        if abs(tau) <= _EPS_TAU:
            return np.maximum(S_t - K, 0.0)

        vol_sqrt = sigma * math.sqrt(tau)
        d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / vol_sqrt
        d2 = d1 - vol_sqrt
        return S_t * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2)


@dataclass(frozen=True, slots=True)
class Put:
    asset_idx: int
    position: float
    K: float
    T: float
    sigma: float

    def mtm_at_t(self, S_t: np.ndarray, *, r: float, t: float) -> np.ndarray:
        """PV at time t of a long European put."""
        S_t = np.asarray(S_t, dtype=float)
        K = float(self.K)
        r = float(r)
        sigma = float(self.sigma)

        tau = float(self.T) - float(t)

        if tau < -_EPS_TAU:
            return np.zeros_like(S_t, dtype=float)

        if abs(tau) <= _EPS_TAU:
            return np.maximum(K - S_t, 0.0)

        vol_sqrt = sigma * math.sqrt(tau)
        d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / vol_sqrt
        d2 = d1 - vol_sqrt
        return K * math.exp(-r * tau) * norm.cdf(-d2) - S_t * norm.cdf(-d1)