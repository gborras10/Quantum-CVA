# python utils
import math
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable
from scipy.stats import norm

# global constants
_EPS_TAU = 1e-12  # threshold for treating tau=T-t as zero

SigmaFunc = Callable[[float, float], float]

# =========================
# Derivatives instruments 
# =========================
@dataclass(frozen=True, slots=True)
class Forward:
    asset_idx: int
    quantity: float
    multiplier: float
    K: float
    T: float

    def mtm_at_t(self, S_t: np.ndarray, *, r: float, t: float) -> np.ndarray:
        """PV at time t of a forward."""
        S_t = np.asarray(S_t, dtype=float)
        tau = float(self.T) - float(t)
        scale = float(self.quantity) * float(self.multiplier)

        if tau < -_EPS_TAU:
            return np.zeros_like(S_t, dtype=float)

        if abs(tau) <= _EPS_TAU:
            return scale * (S_t - float(self.K))
        
        long_mtm = S_t - float(self.K) * np.exp(-float(r) * tau)

        return scale * long_mtm


@dataclass(frozen=True, slots=True)
class Call:
    asset_idx: int
    quantity: float
    multiplier: float
    K: float
    T: float
    sigma_func: SigmaFunc

    def mtm_at_t(self, S_t: np.ndarray, *, r: float, t: float) -> np.ndarray:
        """PV at time t of a European call."""
        S_t = np.asarray(S_t, dtype=float)
        K = float(self.K)
        r = float(r)
        tau = float(self.T) - float(t)
        scale = float(self.quantity) * float(self.multiplier)

        if tau < -_EPS_TAU:
            return np.zeros_like(S_t, dtype=float)

        if abs(tau) <= _EPS_TAU:
            return scale * np.maximum(S_t - K, 0.0)

        sigma = float(self.sigma_func(float(t), float(self.T)))

        if sigma < 0.0:
            raise ValueError("sigma_func must return a non-negative volatility.")

        if sigma <= _EPS_TAU:
            long_mtm = np.maximum(S_t - K * math.exp(-r * tau), 0.0)
            return scale * long_mtm

        vol_sqrt = sigma * math.sqrt(tau)
        d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / vol_sqrt
        d2 = d1 - vol_sqrt
        long_mtm = S_t * norm.cdf(d1) - K * math.exp(-r * tau) * norm.cdf(d2)

        return scale * long_mtm


@dataclass(frozen=True, slots=True)
class Put:
    asset_idx: int
    quantity: float
    multiplier: float
    K: float
    T: float
    sigma_func: SigmaFunc

    def mtm_at_t(self, S_t: np.ndarray, *, r: float, t: float) -> np.ndarray:
        """PV at time t of a European put."""
        S_t = np.asarray(S_t, dtype=float)
        K = float(self.K)
        r = float(r)
        tau = float(self.T) - float(t)
        scale = float(self.quantity) * float(self.multiplier)

        if tau < -_EPS_TAU:
            return np.zeros_like(S_t, dtype=float)

        if abs(tau) <= _EPS_TAU:
            return scale * np.maximum(K - S_t, 0.0)

        sigma = float(self.sigma_func(float(t), float(self.T)))

        if sigma < 0.0:
            raise ValueError("sigma_func must return a non-negative volatility.")

        if sigma <= _EPS_TAU:
            long_mtm = np.maximum(K * math.exp(-r * tau) - S_t, 0.0)
            return scale * long_mtm

        vol_sqrt = sigma * math.sqrt(tau)
        d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / vol_sqrt
        d2 = d1 - vol_sqrt
        long_mtm = K * math.exp(-r * tau) * norm.cdf(-d2) - S_t * norm.cdf(-d1)
        
        return scale * long_mtm