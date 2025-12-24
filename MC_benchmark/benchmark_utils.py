# =============================
# benchmark_utils.py
# =============================
import numpy as np
from scipy.optimize import brentq
from typing import Callable, Sequence, Tuple


def P0(u: float, r: float) -> float:
    """
    Return the discount factor P(0,u) under a flat continuously-compounded rate.

    Parameters
    ----------
    u : float
        Maturity in years at which the discount factor is evaluated.
    r : float
        Flat risk-free rate (continuous compounding).

    Returns
    -------
    float
        Discount factor P(0,u) = exp{-r*u}.
    """
    return float(np.exp(-r * u))


def P_t_T(ti: float, T: float, r: float) -> float:
    """
    Return the forward discount factor P(ti,T) under a flat curve.

    Parameters
    ----------
    ti : float
        Start time in years.
    T : float
        End time in years.
    r : float
        Flat risk-free rate (continuous compounding).

    Returns
    -------
    float
        Forward discount factor P(ti,T) = P(0,T)/P(0,ti) = exp(-r*(T-ti)).
    """
    return float(np.exp(-r * (T - ti)))


# =============================
# Function to build survival curve from CDS quotes via bootstrapping
# =============================
def build_survival_from_cds(
    P0: Callable[[float], float],
    tenors: Sequence[float],
    spreads: Sequence[float],
    R_cds: float,
    pay_freq: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Callable[[float], float], Callable[[float, float], float]]:
    """
    Bootstrap a piecewise-constant hazard rate curve from CDS par spreads.

    Parameters
    ----------
    P0 : Callable[[float], float]
        Discount factor function P0(t) = P(0,t).
    tenors : Sequence[float]
        CDS maturities in years (strictly increasing).
    spreads : Sequence[float]
        CDS par spreads in decimal form.
    R_cds : float
        CDS recovery rate.
    pay_freq : int, optional
        Number of premium payments per year (e.g. 4 for quarterly).

    Returns
    -------
    breaks : np.ndarray
        Interval boundaries [0, T_1, ..., T_n].
    lambdas : np.ndarray
        Calibrated hazard rates for each interval.
    survival_curve : Callable[[float], float]
        Survival probability function S(t).
    q_interval : Callable[[float, float], float]
        Default probability over (t_prev, t_curr].
    """
    tenors = np.asarray(tenors, dtype=float)
    spreads = np.asarray(spreads, dtype=float)

    dt = 1.0 / pay_freq
    breaks = np.r_[0.0, tenors]  # concatenate 0.0 at the beginning
    lambdas = np.zeros(len(tenors))  # one lambda per interval

    def survival(t: float, upto_idx: int) -> float:
        """
        Survival probability S(t) implied by the current hazard rates.

        Parameters
        ----------
        t : float
            Time in years.
        upto_idx : int
            Index of the last interval (inclusive) used in the hazard integral.

        Returns
        -------
        float
            Survival probability S(t).
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

    def pv_legs(Tm: float, spread: float, idx: int) -> Tuple[float, float]:
        """
        Present value of premium and protection legs up to maturity Tm.

        Parameters
        ----------
        Tm : float
            CDS maturity.
        spread : float
            CDS par spread for maturity Tm.
        idx : int
            Index of the hazard rate currently being calibrated.

        Returns
        -------
        prem : float
            Present value of the premium leg.
        prot : float
            Present value of the protection leg.
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
        lambdas[i] = brentq(f, 1e-12, 5.0)

    def survival_curve(t: float) -> float:
        """
        Survival probability using the fully calibrated hazard curve.

        Parameters
        ----------
        t : float
            Time in years.

        Returns
        -------
        float
            Survival probability S(t).
        """
        return survival(t, upto_idx=len(lambdas) - 1)

    def q_interval(t_prev: float, t_curr: float) -> float:
        """
        Default probability over the interval (t_prev, t_curr].

        Parameters
        ----------
        t_prev : float
            Interval start time in years.
        t_curr : float
            Interval end time in years.

        Returns
        -------
        float
            Default probability over (t_prev, t_curr].
        """
        return float(survival_curve(t_prev) - survival_curve(t_curr))

    return breaks, lambdas, survival_curve, q_interval
