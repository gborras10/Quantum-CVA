# =============================
# benchmark_utils.py
# =============================
import numpy as np
from scipy.optimize import brentq
from collections.abc import Callable, Sequence

# ------------------------------
# Auxiliary functions for discount factors
# ------------------------------
#%%
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

#%%
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


# -----------------------------
# Function to build survival curve from CDS quotes via bootstrapping
# -----------------------------
#%%
def build_survival_from_cds(P0: Callable[[float], float],
    tenors: Sequence[float],
    spreads: Sequence[float],
    R_cds: float,
    pay_freq: int = 4,
) -> tuple[np.ndarray, np.ndarray, Callable[[float], float], Callable[[float, float], float]]:
    """
    Bootstrap a survival probability curve from CDS par spreads using a
    piecewise-constant hazard rate model. This function calibrates a hazard rate 
    curve λ(t) assumed constant on each interval (T_{i-1}, T_i), 
    where T_i are the CDS maturities. The calibration enforces, for each
    tenor, equality between the present value of the premium leg and the 
    protection leg of a standard CDS contract.

    Parameters
    ----------
    P0 : Callable[[float], float]
        Discount factor function P(0,t). It must return the present
        value of one unit of currency paid at time ``t`` (in years).
    tenors : Sequence[float]
        CDS maturities in years (strictly increasing), e.g.
        ``[1, 3, 5, 7, 10]``.
    spreads : Sequence[float]
        CDS par spreads corresponding to ``tenors``, expressed in decimal
        form (e.g. ``0.001`` for 100 bps).
    R_cds : float
        Recovery rate assumed in the CDS contracts.
    pay_freq : int, optional
        Number of premium payments per year (default is 4, i.e. quarterly).

    Returns
    -------
    breaks : np.ndarray
        Array of interval boundaries ``[0, T_1, ..., T_n]`` defining the
        piecewise-constant hazard rate structure.
    lambdas : np.ndarray
        Calibrated hazard rates (λ_i), one for each interval
        ``(T_{i-1}, T_i]``.
    survival_curve : Callable[[float], float]
        Survival probability function ``S(t)`` implied by the calibrated
        hazard curve.
    q_interval : Callable[[float, float], float]
        Function returning the default probability over an interval
        ``(t_prev, t_curr]``, i.e.
        ``S(t_prev) - S(t_curr)``.

    Examples
    --------
    Bootstrap a survival curve from CDS spreads:

    >>> tenors = [1, 3, 5, 7, 10]
    >>> spreads = [0.0010, 0.0018, 0.0032, 0.0047, 0.0057]
    >>> breaks, λs, S, q = build_survival_from_cds(
    ...     P0=P0_flat,
    ...     tenors=tenors,
    ...     spreads=spreads,
    ...     R_cds=0.4,
    ... )
    >>> S(5.0)
    0.96
    >>> q(2.0, 2.25)
    0.0008
    """
    # Sanity check
    if np.any(np.diff(tenors) <= 0):
        raise ValueError("Los tenors deben estar estrictamente ordenados de menor a mayor.")
    
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

    def pv_legs(Tm: float, spread: float, idx: int) -> tuple[float, float]:
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

            Parameters
            ----------
            lam : float
                Hazard rate for the current interval.
            
            Returns
            -------
            float
                Difference between premium leg and protection leg PVs.
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

# -----------------------------
# Functions to work with the discretized underlying distribution
# -----------------------------
#%%
def simulate_S(
    S0: float,
    mu: float,
    sigma: float,
    t: np.ndarray,
    Z: np.ndarray,   # shape (N_paths, M)
) -> list[np.ndarray]:
    """
    Simulate marginal samples of a Geometric Brownian Motion at given time points.

    Parameters
    ----------
    S0 : float
        Initial underlying price at time t = 0.
    mu : float
        Drift parameter of the GBM.
    sigma : float
        Volatility of the GBM.
    t : np.ndarray
        Time grid of length M+1 with t[0] = 0 and t[i] the i-th exposure date.
    Z : np.ndarray
        Standard normal samples with shape (N_paths, M). Column i-1 is used
        to generate samples at time t_i.

    Returns
    -------
    list[np.ndarray]
        List of length M. The i-th element is an array of shape (N_paths,)
        containing samples of S(t_i).

    Example
    -------
    >>> import numpy as np
    >>> S0, mu, sigma = 5.0, 0.02, 0.25
    >>> t = np.linspace(0.0, 0.5, 5)   # M = 4 time steps
    >>> N_paths = 100_000
    >>> Z = np.random.standard_normal(size=(N_paths, 4))
    >>> S_by_time = simulate_S(S0, mu, sigma, t, Z)
    >>> len(S_by_time)
    4
    >>> S_by_time[0]
    array([4.92017382, 5.58214855, 4.52283849, ..., 5.67746454, 4.53352251,
       5.00398244], shape=(100000,))
    """
    if Z.shape[1] != len(t) - 1:
        raise ValueError("Z shape and time grid t are inconsistent.")

    M = Z.shape[1]
    S_list = []
    for i in range(1, M + 1):
        ti = t[i]
        # Marginal GBM sampling at time t_i (no pathwise dynamics)
        Si = S0 * np.exp(
            (mu - 0.5 * sigma**2) * ti
            + sigma * np.sqrt(ti) * Z[:, i - 1]
        )
        S_list.append(Si)
    return S_list

#%%
def price_grid_from_samples(
    S_samples_by_time: list[np.ndarray],
    n: int,
    n_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a global uniform price grid with N = 2^n points from GBM samples.

    The grid range [s0, sN] is defined using the mean and standard deviation
    of the terminal distribution S(T), truncated to n_sigma standard deviations.

    Parameters
    ----------
    S_samples_by_time : list[np.ndarray]
        Samples of S(t_i) for each time step; only S(T) is used.
    n : int
        Number of qubits for the price register (N = 2^n grid points).
    n_sigma : float, optional
        Truncation width in standard deviations (default is 4.0).

    Returns
    -------
    edges : np.ndarray
        Array of length N+1 defining the bin edges of the price grid.
    s_mid : np.ndarray
        Array of length N with the representative price of each bin.

    Example
    -------
    >>> edges, s_mid = price_grid_from_samples(S_by_time, n=6)
    >>> len(s_mid)
    64
    """
    N = 2**n
    # use terminal time T (widest distribution) to define a global price grid
    X = S_samples_by_time[-1] 
    muhat = float(X.mean())
    sighat = float(X.std(ddof=1))
    s0 = max(muhat - n_sigma * sighat, 0.0)
    sN = muhat + n_sigma * sighat

    edges = np.linspace(s0, sN, N + 1)
    s_mid = 0.5 * (edges[:-1] + edges[1:])

    return edges, s_mid

#%%
def discrete_probs_from_samples(
    S: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Build a discrete probability distribution from continuous samples
    by histogramming over a fixed price grid and renormalizing.

    Parameters
    ----------
    S : np.ndarray
        Monte Carlo samples of the underlying price at a fixed time t_i.
    edges : np.ndarray
        Bin edges defining the price grid [s0, ..., sN].

    Returns
    -------
    p : np.ndarray
        Discrete probability vector of length len(edges)-1 such that
        p[j] ≈ P(S in bin j | S in [s0, sN]) and sum(p) = 1.

    Example
    -------
    >>> import numpy as np
    >>> S = np.array([1.2, 1.7, 2.1, 2.9])
    >>> edges = np.array([1.0, 2.0, 3.0])
    >>> p = discrete_probs_from_samples(S, edges)
    >>> p
    array([0.5, 0.5])
    """
    counts, _ = np.histogram(S, bins=edges)
    in_range = counts.sum()
    if in_range == 0:
        raise ValueError("No samples in range; widen [s0, sN].")
    return counts / in_range

#%%
#------------------------------
# Continuous-underlying-distribution CVA estimator 
#------------------------------
def classical_continuous_cva_MC(
    S0: float,
    mu: float,
    sigma: float,
    t: np.ndarray,
    K: float,
    r: float,
    T: float,
    LGD: float,
    P0_func: Callable[[float], float],
    q_interval: Callable[[float, float], float],
    Z: np.ndarray | None = None
) -> tuple[float, float]:
    """
    Classical Monte Carlo estimator of CVA with a continuous distribution
    of the underlying (no price discretization). This function computes the CVA
    contribution for a single Monte Carlo run using marginal sampling of a 
    geometric Brownian motion at each exposure date. It returns both the 
    CVA estimate and the Monte Carlo standard error for that run.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    mu : float
        Drift of the geometric Brownian motion.
    sigma : float
        Volatility of the geometric Brownian motion.
    t : np.ndarray
        Time grid of length ``M+1`` with ``t[0] = 0`` and exposure dates
        ``t[1], …, t[M]``.
    K : float
        Strike parameter of the forward-like exposure.
    r : float
        Flat continuously compounded risk-free rate used for discounting
        and for the forward strike adjustment.
    T : float
        Maturity of the contract.
    LGD : float
        Loss-given-default (typically ``1 - recovery``).
    P0_func : callable
        Discount factor function. Must accept a single maturity ``u`` and
        return ``P(0,u)``.
    q_interval : callable
        Incremental default probability function ``q(a,b)`` for the interval
        ``(a,b]``.
    Z : np.ndarray
        Standard normal shocks with shape ``(N_paths, M)``, where each column
        corresponds to the normal variates used to sample ``S(t_i)``.

    Returns
    -------
    cva : float
        Monte Carlo estimate of CVA for this run.
    std_err : float
        Monte Carlo standard error of the estimator for this run,
        computed from the sample variance of the pathwise CVA contributions.

    Examples
    --------
    Single Monte Carlo run:

    >>> rng = np.random.default_rng(42)
    >>> Z = rng.standard_normal(size=(100_000, M))
    >>> cva, std = classical_continuous_cva_MC(
    ...     S0=S0,
    ...     mu=mu,
    ...     sigma=sigma,
    ...     t=t,
    ...     K=K,
    ...     r=r,
    ...     T=T,
    ...     LGD=1.0,
    ...     P0_func=P0_flat,
    ...     q_interval=q_interval,
    ...     Z=Z,
    ... )
    >>> cva, std
    (5.6e-05, 1.2e-06)
    """
    
    N_paths, M = Z.shape

    # Precompute time scalars
    dq = np.array([q_interval(t[i - 1], t[i]) for i in range(1, M + 1)], dtype=float)
    p = np.array([P0_func(t[i]) for i in range(1, M + 1)], dtype=float)
    fwd_strike = np.array([K * np.exp(-r * (T - t[i])) for i in range(1, M + 1)], dtype=float)

    cva_path = np.zeros(N_paths, dtype=float)

    for i in range(1, M + 1):
        ti = float(t[i])
        Si = S0 * np.exp((mu - 0.5 * sigma**2) * ti + sigma * np.sqrt(ti) * Z[:, i - 1])
        Vpos = np.maximum(Si - fwd_strike[i - 1], 0.0)
        cva_path += Vpos * p[i - 1] * dq[i - 1]

    cva = float(LGD * cva_path.mean())
    std_err = float(np.sqrt((LGD**2) * cva_path.var(ddof=1) / N_paths))
    return cva, std_err

#%%
# -----------------------------
# Discrete-underlying-distribution CVA estimator    
# -----------------------------
def classical_discrete_cva_MC(
    S_by_time: list[np.ndarray],
    t: np.ndarray,
    K: float,
    r: float,
    T: float,
    LGD: float,
    P0_func,
    q_interval,
    n: int | np.ndarray,
    n_sigma: float = 3.0,
    C_v: float = 1.0,
    C_p: float = 1.0,
    C_q: float = 1.0,
    payoff_repr: str = "left",   # "left" | "right" | "midpoint"
) -> np.ndarray | float:
    """
    Discrete-time, discrete-price CVA estimator with explicit
    scaling constants introduced to rescale the payoff, 
    discount factor and default probability to 
    the unit interval [0,1] for amplitude encoding, and to recover and
    amplify small variations when rescaling back to the original magnitude.
    Designed to be run in quantum registers.

    Parameters
    ----------
    S_by_time : list[np.ndarray]
        Monte Carlo samples of the underlying price at each exposure date
        \\([S(t_1), \\dots, S(t_M)]\\). Each array has shape ``(N_paths,)``.
    t : np.ndarray
        Time grid of length ``M+1`` with ``t[0] = 0`` and exposure dates
        ``t[1], …, t[M]``.
    K : float
        Strike parameter of the forward-like exposure payoff. The effective
        strike at time ``t_i`` is ``K * exp(-r * (T - t_i))``.
    r : float
        Flat continuously compounded risk-free rate used in the forward
        strike adjustment.
    T : float
        Maturity of the deal (in years).
    LGD : float
        Loss-given-default (typically ``1 - recovery``).
    P0_func : callable
        Discount factor function. Must accept a single maturity ``u`` and
        return the discount factor ``P(0,u)``.
    q_interval : callable
        Incremental default probability function ``q(a, b)`` for the interval
        ``(a, b]``.
    n : int or array-like of int
        Price discretization level(s). For a given ``n``, the price grid has
        ``N = 2**n`` bins. If an array is provided, CVA is computed for all
        values in one call.
    n_sigma : float, optional
        Width of the global price truncation interval expressed in standard
        deviations of the terminal price distribution ``S(T)``.
    C_v : float, optional
        Scaling constant for the exposure/payoff function ``v``.
    C_p : float, optional
        Scaling constant for the discount factor ``p``.
    C_q : float, optional
        Scaling constant for the default probability ``q``.

    Returns
    -------
    float or np.ndarray
        If ``n`` is a scalar, returns ``CVA(n)`` as a float.
        If ``n`` is array-like, returns an array of ``CVA(n)`` values with
        the same shape as ``n``.

    Examples
    --------
    Compute CVA for a single discretization level:

    >>> cva_4 = discrete_cva(
    ...     S_by_time=S_by_time,
    ...     t=t,
    ...     K=K,
    ...     r=r,
    ...     T=T,
    ...     LGD=1.0,
    ...     P0_func=P0_flat,
    ...     q_interval=q_interval,
    ...     n=4,
    ...     n_sigma=4.0,
    ... )

    Compute CVA for several discretization levels in one call:

    >>> ns = np.arange(2, 10)
    >>> cva_ns = discrete_cva(
    ...     S_by_time=S_by_time,
    ...     t=t,
    ...     K=K,
    ...     r=r,
    ...     T=T,
    ...     LGD=1.0,
    ...     P0_func=P0_flat,
    ...     q_interval=q_interval,
    ...     n=ns,
    ...     n_sigma=4.0,
    ... )
    >>> cva_ns.shape
    (8,)
    """
    n_arr = np.atleast_1d(n).astype(int)
    M = len(S_by_time)

    p = np.array([P0_func(t[i]) for i in range(1, M + 1)], dtype=float)
    dq = np.array([q_interval(t[i - 1], t[i]) for i in range(1, M + 1)], dtype=float)
    p_tilde = p / C_p
    dq_tilde = dq / C_q

    out = np.zeros(len(n_arr), dtype=float)

    pr = payoff_repr.lower()

    for k, ni in enumerate(n_arr):
        edges, _ = price_grid_from_samples(S_by_time, n=int(ni), n_sigma=n_sigma)

        left_edges  = edges[:-1]
        right_edges = edges[1:]
        mid_edges   = 0.5 * (left_edges + right_edges)

        # parámetro de sesgo: 0=left, 0.5=mid, 1=right
        theta = 1.0

        if pr in ("left", "l"):
            s_rep = left_edges

        elif pr in ("right", "r"):
            s_rep = right_edges

        elif pr in ("mid", "midpoint", "m"):
            s_rep = mid_edges

        elif pr in ("theta", "tilted", "t"):
            # representante interior sesgado hacia la derecha
            s_rep = (1.0 - theta) * left_edges + theta * right_edges

        else:
            raise ValueError(
                "payoff_repr must be one of {'left','right','midpoint','theta'}"
            )

        bracket_sum = 0.0
        for i in range(1, M + 1):
            P_bin_given_t = discrete_probs_from_samples(S_by_time[i - 1], edges)

            ti = t[i]
            forward_strike = K * np.exp(-r * (T - ti))

            payoff = np.maximum(s_rep - forward_strike, 0.0)
            payoff_tilde = payoff / C_v

            E_tilde_ti = np.dot(P_bin_given_t, payoff_tilde)
            bracket_sum += E_tilde_ti * p_tilde[i - 1] * dq_tilde[i - 1]

        out[k] = LGD * C_v * C_p * C_q * bracket_sum

    return float(out[0]) if np.isscalar(n) else out