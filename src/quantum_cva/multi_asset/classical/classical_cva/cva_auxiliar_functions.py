# python utils
import numpy as np
from scipy.optimize import brentq
from collections.abc import Sequence, Callable

# quantum_cva utils
from quantum_cva.multi_asset.instruments.derivatives  import Forward, Call, Put

# defining accepted derivatives intrument types for the CVA calculation
Instrument = Forward | Call | Put

# ------------------- discount curve utils -------------------  
def P0(u: float, r: float) -> float:
    """
    Return the discount factor P(0,u) under a flat continuously-compounded rate.
    """
    return float(np.exp(-r * u))


def P_t_T(ti: float, T: float, r: float) -> float:
    """
    Return the forward discount factor P(ti,T) under a flat curve.
    """
    return float(np.exp(-r * (T - ti)))

def discount_factors_on_grid(
    t: np.ndarray,
    P0_func: Callable[[float], float],
) -> np.ndarray:
    """
    Build the discount-factor array p(t_i) = P(0, t_i) on the exposure time grid.
    """
    t = np.asarray(t, dtype=float)
    return np.array([P0_func(float(ti)) for ti in t], dtype=float)

# ---------------------- default curve utils ----------------------
def build_survival_from_cds(
    P0: Callable[[float], float],
    tenors: Sequence[float],
    spreads: Sequence[float],
    R_cds: float,
    pay_freq: int = 4,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Callable[[float], float],
    Callable[[float, float], float],
]:
    """
    Bootstrap a survival probability curve from CDS par spreads using a
    piecewise-constant hazard rate model. The calibration enforces, for each
    tenor, equality between the present value of the premium leg and the
    protection leg of a standard CDS contract.

    """
    # Sanity check
    if np.any(np.diff(tenors) <= 0):
        raise ValueError(
            "Los tenors deben estar estrictamente ordenados de menor a mayor."
        )

    tenors = np.asarray(tenors, dtype=float)
    spreads = np.asarray(spreads, dtype=float)

    dt = 1.0 / pay_freq
    breaks = np.r_[0.0, tenors]  # concatenate 0.0 at the beginning
    lambdas = np.zeros(len(tenors))  # one lambda per interval

    def survival(t: float, upto_idx: int) -> float:
        """
        Survival probability S(t) implied by the current hazard rates.
        """
        integral = 0.0
        for j in range(upto_idx + 1):
            a, b = breaks[j], breaks[j + 1]
            if t <= a:
                break
            integral += lambdas[j] * (min(t, b) - a)
            if t <= b:
                break

        # If t lies beyond the last calibrated break, extrapolate
        last_break = breaks[upto_idx + 1]
        if t > last_break:
            integral += lambdas[upto_idx] * (t - last_break)

        return float(np.exp(-integral))

    def pv_legs(Tm: float, spread: float, idx: int) -> tuple[float, float]:
        """
        Present value of premium and protection legs up to maturity Tm.
        """
        times = np.arange(dt, Tm, dt)
        LGD = 1.0 - R_cds

        prem = 0.0
        prot = 0.0

        tprev = 0.0
        for tk in times:
            S_prev = survival(tprev, upto_idx=idx)
            S_k = survival(tk, upto_idx=idx)

            # Regular premium payment
            prem += spread * dt * P0(tk) * S_k

            # Protection payment
            prot += LGD * P0(tk) * (S_prev - S_k)

            tprev = tk

        return float(prem), float(prot)

    for i in range(len(tenors)):
        Tm = tenors[i]
        s_i = spreads[i]

        def f(lam: float) -> float:
            """
            Objective function for root finding: difference between PVs of legs.
            """
            lambdas[i] = lam
            prem, prot = pv_legs(Tm, s_i, idx=i)
            return prem - prot

        # Find the root for lambda_i using Brent's method in
        # a reasonable interval
        max_hazard_rate = 5.0
        min_hazard_rate = 1e-12
        lambdas[i] = brentq(f, min_hazard_rate, max_hazard_rate)

    def survival_curve(t: float) -> float:
        """
        Survival probability using the fully calibrated hazard curve.
        """
        return survival(t, upto_idx=len(lambdas) - 1)

    def q_interval(t_prev: float, t_curr: float) -> float:
        """
        Default probability over the interval (t_prev, t_curr].
        """
        return float(survival_curve(t_prev) - survival_curve(t_curr))

    return breaks, lambdas, survival_curve, q_interval