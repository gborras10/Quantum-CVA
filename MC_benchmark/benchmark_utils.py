# =============================
# benchmark_utils.py
# =============================
import numpy as np
from scipy.optimize import brentq
from typing import Callable, Sequence, Tuple, List

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
) -> List[np.ndarray]:
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
    List[np.ndarray]
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
def discrete_cva(
    S_by_time: list[np.ndarray],   # length M, each shape (N_paths,)
    t: np.ndarray,                 # length M+1
    K: float,
    r: float,
    T: float,
    LGD: float,
    P0_func,                       # callable: P0(0,ti) or P0(ti)
    q_interval,                    # callable: q(t_{i-1},t_i)
    n: int | np.ndarray,
    n_sigma: float = 3.0,
) -> np.ndarray | float:
    """
    Compute the discrete-time, discrete-price CVA approximation (Alcázar et al., 2022) for a single Monte Carlo run.

    Parameters
    ----------
    S_by_time : list[np.ndarray]
        List of Monte Carlo samples for each exposure date: [S(t_1), ..., S(t_M)].
        Each entry has shape (N_paths,).
    t : np.ndarray
        Time grid of length M+1 with t[0]=0 and exposure dates t[1],...,t[M].
    K : float
        Strike of the forward-like payoff used for exposure: max(S(t_i) - K*exp(-r*(T-t_i)), 0).
    r : float
        Flat continuously-compounded rate used in the forward strike adjustment.
    T : float
        Maturity of the deal (years).
    LGD : float
        Loss-given-default.
    P0_func : callable
        Discount factor function. Must accept a single time argument u and return P(0,u).
    q_interval : callable
        Incremental default probability function q(a,b) for interval (a,b].
    n : int or array-like of int
        Price discretization levels. Uses N=2^n bins. If an array is provided, CVA is computed for
        all values in one call.
    n_sigma : float, optional
        Truncation width for the global price range in standard deviations of S(T).

    Returns
    -------
    float or np.ndarray
        If n is an int, returns CVA(n) as a float.
        If n is array-like, returns an array of CVA(n) with the same shape as n.

    Examples
    --------
    Compute CVA for multiple discretization levels in one run:
    >>> ns = np.arange(1, 15)  # n=1..14
    >>> cva_ns = discrete_cva(
    ...     S_by_time=S_by_time, t=t, K=K, r=r, T=T, LGD=LGD,
    ...     P0_func=P0_flat, q_interval=q_interval, n=ns, n_sigma=4.0
    ... )
    >>> cva_ns.shape
    (14,)

    Proxy for CVA(infinity) using a fine grid:
    >>> cva_inf = discrete_cva(
    ...     S_by_time=S_by_time, t=t, K=K, r=r, T=T, LGD=LGD,
    ...     P0_func=P0_flat, q_interval=q_interval, n=20, n_sigma=4.0
    ... )
    >>> float(cva_inf)  # doctest: +SKIP
    5.5e-05
    """
    n_arr = np.atleast_1d(n).astype(int)
    n_times = len(S_by_time)

    discount_factors = np.array(
        [P0_func(t[i]) for i in range(1, n_times + 1)],
        dtype=float,
    )
    default_prob_increments = np.array(
        [q_interval(t[i - 1], t[i]) for i in range(1, n_times + 1)],
        dtype=float,
    )

    terminal_prices = S_by_time[-1]          # S(T)
    terminal_mean = float(terminal_prices.mean())
    terminal_std = float(terminal_prices.std(ddof=1))

    price_min = max(terminal_mean - n_sigma * terminal_std, 0.0)
    price_max = terminal_mean + n_sigma * terminal_std

    bin_edges_per_n = [
        np.linspace(price_min, price_max, 2**ni + 1) for ni in n_arr
    ]
    bin_midpoints_per_n = [
        0.5 * (edges[:-1] + edges[1:]) for edges in bin_edges_per_n
    ]

    cva_values = np.zeros(len(n_arr), dtype=float)

    for k, (bin_edges, bin_midpoints) in enumerate(
        zip(bin_edges_per_n, bin_midpoints_per_n)
    ):
        expected_exposure = np.zeros(n_times, dtype=float)

        for i in range(1, n_times + 1):
            spot_samples = S_by_time[i - 1]

            bin_counts, _ = np.histogram(spot_samples, bins=bin_edges)
            n_in_support = bin_counts.sum()
            if n_in_support == 0:
                raise ValueError(
                    "No samples in range; widen [price_min, price_max] or increase n_sigma."
                )

            discrete_price_prob = bin_counts / n_in_support

            ti = t[i]
            forward_strike = K * np.exp(-r * (T - ti))
            positive_payoff_midpoints = np.maximum(
                bin_midpoints - forward_strike, 0.0
            )

            expected_exposure[i - 1] = np.dot(
                discrete_price_prob, positive_payoff_midpoints
            )

        cva_values[k] = LGD * np.sum(
            expected_exposure * discount_factors * default_prob_increments
        )

    return float(cva_values[0]) if np.isscalar(n) else cva_values


def discrete_cva_paper_form(
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
) -> np.ndarray | float:
    """
    Same estimator as discrete_cva but written explicitly in the paper form:
      CVA = M*LGD*C_v*C_p*C_q * sum_{i,j} P(sj,ti) v~(sj,ti) p~(ti) q~(ti)
    where v = C_v v~, p = C_p p~, q = C_q q~.
    """

    n_arr = np.atleast_1d(n).astype(int)
    M = len(S_by_time)

    # p(t_i) and q(t_{i-1},t_i)
    p = np.array([P0_func(t[i]) for i in range(1, M + 1)], dtype=float)
    q = np.array([q_interval(t[i - 1], t[i]) for i in range(1, M + 1)], dtype=float)

    # scaled versions
    p_tilde = p / C_p
    q_tilde = q / C_q

    # global price support from terminal distribution
    X_T = S_by_time[-1]
    mu_T = float(X_T.mean())
    sigma_T = float(X_T.std(ddof=1))
    s_min = max(mu_T - n_sigma * sigma_T, 0.0)
    s_max = mu_T + n_sigma * sigma_T

    edges_list = [np.linspace(s_min, s_max, 2**ni + 1) for ni in n_arr]
    mids_list  = [0.5 * (e[:-1] + e[1:]) for e in edges_list]

    out = np.zeros(len(n_arr), dtype=float)

    for k, (edges, s_mid) in enumerate(zip(edges_list, mids_list)):
        bracket_sum = 0.0  # this is the bracketed double-sum (already includes time-sum)

        for i in range(1, M + 1):
            counts, _ = np.histogram(S_by_time[i - 1], bins=edges)
            in_range = counts.sum()
            if in_range == 0:
                raise ValueError("No samples in range; widen [s_min,s_max] or increase n_sigma.")
            P_bin_given_t = counts / in_range  # sums to 1 over bins

            ti = t[i]
            v_mid = np.maximum(s_mid - K * np.exp(-r * (T - ti)), 0.0)
            v_tilde_mid = v_mid / C_v

            # sum_j P(sj|ti) v~(sj,ti)
            E_tilde_ti = np.dot(P_bin_given_t, v_tilde_mid)

            # multiply by p~(ti) q~(ti) and accumulate over i
            bracket_sum += E_tilde_ti * p_tilde[i - 1] * q_tilde[i - 1]

        # prefactor M * LGD * C_v * C_p * C_q
        out[k] = M * LGD * C_v * C_p * C_q * bracket_sum

    return float(out[0]) if np.isscalar(n) else out
