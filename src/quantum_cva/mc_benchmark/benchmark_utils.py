# src/quantum_cva/mc_benchmark/benchmark_utils.py
import numpy as np
from scipy.optimize import brentq
from collections.abc import Callable, Sequence

# ------------------------------
# Auxiliary functions for computing discount factors
# ------------------------------

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

# -----------------------------
# Function to build survival curve from CDS quotes via bootstrapping
# -----------------------------

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
        ---------
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

# ===========================================
# MAIN CVA DRIVERS FUNCTIONS
# ===========================================

def simulate_S(
    S0: float,
    mu: float,
    sigma: float,
    t: np.ndarray,
    Z: np.ndarray,   # shape (N_paths, M) where M = len(t) - 1
    antithetic: bool = True,
    moment_match: bool = False,
    replications: int = 1,
    replication_seed: int = 12345,
    pathwise: bool = False,
) -> list[np.ndarray]:
    """
    Simulate samples of a Geometric Brownian Motion (GBM) at the time grid
    `t`, including the initial time t[0] = 0.

    The function returns samples of the underlying at all grid points:
    S_by_time[i] contains samples of S(t[i]) for i = 0, ..., M, where M = len(t) - 1.
    The first entry S_by_time[0] is deterministic and equal to S0 replicated across
    all Monte Carlo paths; stochastic sampling starts at t[1].

    The simulation can be either marginal (pathwise=False) or pathwise (pathwise=True).
    - Marginal: samples at different times are generated independently using columns of Z.
    - Pathwise: samples follow a cumulative path, S(t_i) = S(t_{i-1}) * exp(...).

    Parameters
    ----------
    S0 : float
        Initial value of the underlying at time t[0].
    mu : float
        Drift parameter of the GBM.
    sigma : float
        Volatility parameter of the GBM.
    t : np.ndarray
        One-dimensional array of times with t[0] = 0 and length M+1.
    Z : np.ndarray
        Array of standard normal variates with shape (N_paths, M), where column
        i-1 is used to generate samples at time t[i].
    antithetic : bool, optional
        If True, use antithetic variates by stacking Z and -Z. Default is True.
    moment_match : bool, optional
        If True, apply per-time-step moment matching to the normal variates after
        antithetic expansion. Default is False.
    replications : int, optional
        Number of independent normal blocks to concatenate in order to increase the
        effective number of Monte Carlo paths. Default is 1.
    replication_seed : int, optional
        Seed used to generate additional normal blocks when replications > 1.
        Default is 12345.
    pathwise : bool, optional
        If True, simulate pathwise where S(t_i) = S(t_{i-1}) * exp(...).
        If False, simulate marginally where S(t_i) = S0 * exp(...).
        Default is False.

    Returns
    -------
    list[np.ndarray]
        List of length M+1. Each element is a one-dimensional array of shape
        (N_paths,) containing samples of the underlying at the corresponding time.
        The first element corresponds to t[0] and is a constant array equal to S0.
    """

    t = np.asarray(t, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1D array with at least two points (t[0]=0 plus exposure dates).")
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array with shape (N_paths, M).")

    M = len(t) - 1
    if Z.shape[1] != M:
        raise ValueError("Z shape and time grid t are inconsistent: need Z.shape[1] == len(t) - 1.")
    if replications < 1:
        raise ValueError("replications must be >= 1.")

    # --- concatenate independent blocks to increase effective N ---
    if replications > 1:
        rng = np.random.default_rng(int(replication_seed))
        blocks = [Z]
        for _ in range(replications - 1):
            blocks.append(rng.standard_normal(size=Z.shape))
        Z = np.vstack(blocks)

    # --- antithetic expansion ---
    if antithetic:
        Z = np.vstack([Z, -Z])

    # --- moment matching (after antithetic) ---
    if moment_match:
        col_mean = Z.mean(axis=0, keepdims=True)
        col_std = Z.std(axis=0, ddof=0, keepdims=True)
        if np.any(col_std <= 0.0):
            bad = np.where(col_std.ravel() <= 0.0)[0].tolist()
            raise ValueError(f"Moment matching failed: zero-variance columns {bad}.")
        Z = (Z - col_mean) / col_std

    N_paths = int(Z.shape[0])

    # i=0: deterministic S0 replicated across paths
    S_list: list[np.ndarray] = [np.full(N_paths, float(S0), dtype=float)]

    # i=1..M: exposure dates
    if pathwise:
        # Pathwise simulation: S(t_i) = S(t_{i-1}) * exp(...)
        for i in range(1, M + 1):
            ti_prev = float(t[i - 1])
            ti = float(t[i])
            dt = ti - ti_prev
            Si = S_list[i - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt
                + sigma * np.sqrt(dt) * Z[:, i - 1]
            )
            S_list.append(Si.astype(float, copy=False))
    else:
        # Marginal simulation: S(t_i) = S0 * exp(...) (independent at each time)
        for i in range(1, M + 1):
            ti = float(t[i])
            Si = float(S0) * np.exp(
                (mu - 0.5 * sigma**2) * ti
                + sigma * np.sqrt(ti) * Z[:, i - 1]
            )
            S_list.append(Si.astype(float, copy=False))

    return S_list

def discount_factors_on_grid(
    t: np.ndarray,
    P0_func: Callable[[float], float],
) -> np.ndarray:
    """
    Build the discount-factor array p(t_i) = P(0, t_i) on the full time grid.

    Parameters
    ----------
    t : np.ndarray
        Time grid of length M+1 with t[0] = 0.
    P0_func : Callable[[float], float]
        Discount factor function P(0, u) as a one-argument callable.

    Returns
    -------
    np.ndarray
        Array of length M+1 with entries p_i = P(0, t[i]) for i=0..M.
    """
    t = np.asarray(t, dtype=float)
    return np.array([P0_func(float(ti)) for ti in t], dtype=float)

def default_increments_on_grid(
    t: np.ndarray,
    q_interval: Callable[[float, float], float],
) -> np.ndarray:
    """
    Build the default-increment array on the full time grid, including t0.

    Convention:
        q_full[0] = 0.0
        q_full[i] = q_interval(t[i-1], t[i]) for i=1..M

    Parameters
    ----------
    t : np.ndarray
        Time grid of length M+1 with t[0] = 0.
    q_interval : Callable[[float, float], float]
        Function returning default probability over (a, b].

    Returns
    -------
    np.ndarray
        Array of length M+1 with q_full[0]=0 and interval increments in positions 1..M.
    """
    t = np.asarray(t, dtype=float)
    q_full = np.zeros(len(t), dtype=float)
    for i in range(1, len(t)):
        q_full[i] = float(q_interval(float(t[i - 1]), float(t[i])))
    return q_full


def positive_exposure_matrix_from_samples(
    S_by_time: list[np.ndarray],
    t: np.ndarray,
    *,
    K: float,
    r: float,
    T: float,
) -> np.ndarray:
    """
    Build the positive exposure matrix Vpos(paths, time) from continuous samples,
    aligned with the full time grid (including t0).

    The forward-adjusted strike is computed internally for each time:
        fwd_strike(t_i) = K * exp(-r * (T - t_i))

    Parameters
    ----------
    S_by_time : list[np.ndarray]
        Samples aligned with `t`, length M+1 with S_by_time[i] ~ S(t[i]).
        Typically produced by `simulate_S` that includes t0.
    t : np.ndarray
        Time grid of length M+1 aligned with S_by_time.
    K : float
        Strike parameter.
    r : float
        Flat risk-free rate.
    T : float
        Contract maturity.

    Returns
    -------
    np.ndarray
        Positive exposure matrix of shape (N_paths, M+1).
    """
    t = np.asarray(t, dtype=float)
    n_times = len(S_by_time)
    if t.shape != (n_times,):
        raise ValueError("t must have shape (M+1,) matching the length of S_by_time.")

    # forward-adjusted strike aligned with t[0..M]
    fwd_strike = K * np.exp(-r * (T - t))  # (M+1,)

    N_paths = int(np.asarray(S_by_time[0]).shape[0])
    Vpos = np.empty((N_paths, n_times), dtype=float)

    for k, Sk in enumerate(S_by_time):
        Sk = np.asarray(Sk, dtype=float)
        if Sk.shape != (N_paths,):
            raise ValueError("All arrays in S_by_time must have the same shape (N_paths,).")
        Vpos[:, k] = np.maximum(Sk - float(fwd_strike[k]), 0.0)

    return Vpos

# ---------------------------------------
# Classical cva estimation in the continuous underlying regime
# ---------------------------------------
#%%
def cva_from_continuous_blocks(
    Vpos: np.ndarray,  # (N_paths, M+1)
    p_t: np.ndarray,   # (M+1,)
    q_t: np.ndarray,   # (M+1,) with q_t[0]=0
    *,
    LGD: float,
) -> tuple[float, float]:
    """
    Compute CVA (and MC standard error) from continuous-sampling building blocks,
    aligned with the full time grid (including t0).

    Convention:
        q_t[0] = 0 so the t0 column contributes nothing automatically.

    Parameters
    ----------
    Vpos : np.ndarray
        Positive exposure matrix of shape (N_paths, M+1) aligned with t[0..M].
    p_t : np.ndarray
        Discount factors of length M+1 aligned with t[0..M].
    q_t : np.ndarray
        Default increments of length M+1 aligned with t[0..M], with q_t[0]=0.
    LGD : float
        Loss-given-default.

    Returns
    -------
    cva : float
        Monte Carlo estimate of CVA.
    std_err : float
        Monte Carlo standard error computed from the sample variance of the
        pathwise CVA contributions.
    """
    Vpos = np.asarray(Vpos, dtype=float)
    p_t = np.asarray(p_t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)

    if Vpos.ndim != 2:
        raise ValueError("Vpos must be a 2D array of shape (N_paths, M+1).")

    N_paths, n_times = Vpos.shape
    if p_t.shape != (n_times,) or q_t.shape != (n_times,):
        raise ValueError("p_t and q_t must have shape (M+1,) matching Vpos.shape[1].")
    if N_paths <= 1:
        raise ValueError("Need at least 2 paths to estimate a standard error.")
    if float(q_t[0]) != 0.0:
        raise ValueError("q_t[0] must be 0.0 to ensure t0 contributes nothing.")

    w = p_t * q_t
    cva_path = Vpos @ w

    cva = float(LGD * cva_path.mean())
    std_err = float(abs(LGD) * np.sqrt(cva_path.var(ddof=1) / N_paths))
    return cva, std_err

# ===========================================
# DISCRETE UNDERLYING SETTING FUNCTIONS
# ===========================================

def grid_and_prob_matrix(
    S_by_time: list[np.ndarray],
    n: int,
    *,
    n_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (edges, s_mid, P_s_t) in one pass.

    - Global uniform grid from terminal samples (mean ± n_sigma * std, clipped at 0).
    - Probability matrix by histogramming samples at each time and renormalizing.

    Parameters
    ----------
    S_by_time : list[np.ndarray]
        Samples aligned with the time grid, length M+1 (including t0), each (N_paths,).
    n : int
        Price discretization level (N = 2**n bins).
    n_sigma : float, optional
        Truncation width in terminal std devs.

    Returns
    -------
    edges : np.ndarray
        Bin edges, shape (N+1,).
    s_mid : np.ndarray
        Bin midpoints, shape (N,).
    P_s_t : np.ndarray
        Probability matrix, shape (M+1, N).
    """
    N = 2 ** int(n)

    X = np.asarray(S_by_time[-1], dtype=float)
    muhat = float(X.mean())
    sighat = float(X.std(ddof=1))
    s0 = max(muhat - float(n_sigma) * sighat, 0.0)
    sN = muhat + float(n_sigma) * sighat

    edges = np.linspace(s0, sN, N + 1, dtype=float)
    s_mid = 0.5 * (edges[:-1] + edges[1:])

    P_rows = []
    for Si in S_by_time:
        counts, _ = np.histogram(np.asarray(Si, dtype=float), bins=edges)
        tot = int(counts.sum())
        if tot == 0:
            raise ValueError("No samples in range for at least one time; widen [s0, sN] or increase n_sigma.")
        P_rows.append(counts / tot)

    P_s_t = np.asarray(P_rows, dtype=float)  # (M+1, N)
    return edges, s_mid, P_s_t


def build_p_target_from_samples(
    S_by_time: list[np.ndarray],
    *,
    n: int,
    n_sigma: float = 3.0,
    order: str = "time_major",
    drop_t0: bool = False,
    time_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build quantum-ready joint target p_target over (time, price-bin) from samples.

    Inputs
    ------
    S_by_time:
        Samples aligned with the time grid, length len(t) (includes t0 if present).
    n:
        Price register bits (N = 2**n).
    drop_t0:
        If True, removes the first time row (t0) from the target.
        (Only use this if the remaining #times is still a power of two.)
    time_weights:
        Optional P(t_i). If None -> uniform over the kept time points.

    Returns
    -------
    p_target:
        Flattened joint probability vector of length M_used * N (quantum-ready).
    P_s_t:
        Conditional table P(s_j | t_i), shape (M_used, N).
    s_grid:
        Representative price grid (midpoints), shape (N,).
    w_t:
        Time weights, shape (M_used,), sums to 1.
    """
    # Build grid + conditional probability rows from histograms
    edges, s_grid, P_full = grid_and_prob_matrix(S_by_time, n=int(n), n_sigma=float(n_sigma))
    # P_full is row-stochastic by construction (each row sums to 1)

    if drop_t0:
        P = P_full[1:, :]
    else:
        P = P_full

    M_used, N = P.shape

    # Require powers of two for quantum registers
    if (M_used & (M_used - 1)) != 0 or (N & (N - 1)) != 0:
        raise ValueError(f"M_used={M_used} and N={N} must be powers of two for quantum registers.")

    # Time weights
    if time_weights is None:
        w_t = np.full(M_used, 1.0 / M_used, dtype=float)
    else:
        w_t = np.asarray(time_weights, dtype=float).ravel()
        if w_t.shape != (M_used,):
            raise ValueError("time_weights must have shape (M_used,)")
        s = float(w_t.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError("time_weights must have a positive finite sum.")
        w_t = w_t / s

    # Joint = P(t_i) * P(s_j|t_i)
    joint = w_t[:, None] * P  # (M_used, N)

    # Flatten
    if order == "time_major":
        p_target = joint.reshape(M_used * N)
    elif order == "price_major":
        p_target = joint.T.reshape(M_used * N)
    else:
        raise ValueError("order must be 'time_major' or 'price_major'.")

    # Final sanitize/normalize (numerical safety)
    p_target = np.clip(p_target, 0.0, None)
    p_target = p_target / float(p_target.sum())

    return p_target, P, s_grid, w_t

def payoff_matrix_forward_call(
    edges: np.ndarray,
    t: np.ndarray,
    *,
    K: float,
    r: float,
    T: float,
    payoff_repr: str = "mid",
) -> np.ndarray:
    """
    Build payoff matrix v(s_j, t_i) on full time grid using representative bin prices.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges, shape (N+1,).
    t : np.ndarray
        Time grid, shape (M+1,), including t0.
    K, r, T : float
        Contract/payoff parameters.
    payoff_repr : str, optional
        Representative price per bin: "left", "right", or "mid" (default "mid").

    Returns
    -------
    np.ndarray
        Payoff matrix v_s_t, shape (M+1, N).
    """
    edges = np.asarray(edges, dtype=float)
    t = np.asarray(t, dtype=float)

    left = edges[:-1]
    right = edges[1:]
    mid = 0.5 * (left + right)

    pr = payoff_repr.lower()
    if pr in ("left", "l"):
        s_rep = left
    elif pr in ("right", "r"):
        s_rep = right
    elif pr in ("mid", "midpoint", "m"):
        s_rep = mid
    else:
        raise ValueError("payoff_repr must be one of {'left','right','midpoint'}")

    fwd_strike = K * np.exp(-r * (T - t))                       # (M+1,)
    v_s_t = np.maximum(s_rep[None, :] - fwd_strike[:, None], 0.0)  # (M+1, N)
    return v_s_t


def cva_discrete_from_blocks(
    P_s_t: np.ndarray,   # (M+1, N)
    v_s_t: np.ndarray,   # (M+1, N)
    p_t: np.ndarray,     # (M+1,)
    q_t: np.ndarray,     # (M+1,) with q_t[0]=0
    *,
    LGD: float,
    C_v: float = 1.0,
    C_p: float = 1.0,
    C_q: float = 1.0,
) -> float:
    """
    Compute discrete-price CVA from precomputed blocks on the full time grid.

    Parameters
    ----------
    P_s_t : np.ndarray
        Probability matrix, shape (M+1, N), aligned with t[0..M].
    v_s_t : np.ndarray
        Payoff matrix, shape (M+1, N), aligned with t[0..M].
    p_t : np.ndarray
        Discount factors, shape (M+1,), aligned with t[0..M].
    q_t : np.ndarray
        Default increments, shape (M+1,), aligned with t[0..M], with q_t[0]=0.
    LGD : float
        Loss-given-default.
    C_v, C_p, C_q : float, optional
        Scaling constants (amplitude-encoding style). Default 1.0.

    Returns
    -------
    float
        CVA value computed from the blocks.
    """
    P_s_t = np.asarray(P_s_t, dtype=float)
    v_s_t = np.asarray(v_s_t, dtype=float)
    p_t = np.asarray(p_t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)

    if P_s_t.shape != v_s_t.shape:
        raise ValueError("P_s_t and v_s_t must have the same shape (M+1, N).")
    if p_t.shape != (P_s_t.shape[0],) or q_t.shape != (P_s_t.shape[0],):
        raise ValueError("p_t and q_t must have shape (M+1,) aligned with P_s_t.")
    if float(q_t[0]) != 0.0:
        raise ValueError("q_t[0] must be 0.0 so t0 contributes nothing.")

    p_tilde = p_t / C_p
    q_tilde = q_t / C_q
    v_tilde = v_s_t / C_v

    E_tilde = np.sum(P_s_t * v_tilde, axis=1)          # (M+1,)
    bracket = float(np.sum(E_tilde * p_tilde * q_tilde))
    return float(LGD * C_v * C_p * C_q * bracket)

# ===========================
# LEGACY FUNCTIONS
# ===========================
def price_grid_from_samples(
    S_samples_by_time: list[np.ndarray],
    n: int,
    n_sigma: float = 3.0,
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