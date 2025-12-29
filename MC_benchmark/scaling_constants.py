from __future__ import annotations
from dataclasses import dataclass
from benchmark_utils import price_grid_from_samples, discrete_probs_from_samples

import numpy as np

@dataclass
class DiscreteCvaTables:
    n: int
    edges: np.ndarray          # (N+1,)
    left_edges: np.ndarray     # (N,)
    p: np.ndarray              # (M,)
    dq: np.ndarray             # (M,)
    P_bin: np.ndarray          # (M, N)
    v_left: np.ndarray         # (M, N)
    C_p: float
    C_q: float
    C_v: float


def extract_tables_and_scalings(
    S_by_time: list[np.ndarray],
    t: np.ndarray,
    K: float,
    r: float,
    T: float,
    P0_func,
    q_interval,
    n: int,
    n_sigma: float = 3.0,
    eps: float = 1e-6,
    payoff_repr: str = "left",
) -> DiscreteCvaTables:
    """
    Precompute discrete CVA tables (time, price and payoff) and normalization
    constants used in a discretized CVA estimator, with Cp fixed to 1.

    From Monte Carlo samples of the underlying at discrete times, this function:
      - builds a common price grid with N = 2**n bins,
      - estimates marginal price probabilities per time step,
      - evaluates discounted call payoffs on a bin representative,
      - computes discount factors and default probabilities,
      - defines scaling constants (Cp, Cq, Cv) as in Alcázar et al. (2022). :contentReference[oaicite:0]{index=0}

    Parameters
    ----------
    S_by_time : list[np.ndarray]
        Samples of the underlying at times t[1:], length M.
    t : np.ndarray
        Time grid [t0, ..., tM] in years.
    K : float
        Option strike at maturity T.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    P0_func : callable
        Discount factor P0(0,u).
    q_interval : callable
        Default probability over (t_{i-1}, t_i).
    n : int
        Price discretization level (N = 2**n bins).
    n_sigma : float
        Width of the price grid in standard deviations.
    eps : float
        Safety buffer for Cq and Cv.
    payoff_repr : {"left","right","midpoint"}
        Representative price per bin for payoff evaluation.

    Returns
    -------
    DiscreteCvaTables
        Dataclass with price grid, probabilities, payoff table and scaling
        constants.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(0.0, 0.5, 5)          # 4 time steps
    >>> S_by_time = [5.0 * np.exp(0.2 * rng.standard_normal(20_000))
    ...              for _ in range(4)]
    >>> P0 = lambda u: np.exp(-0.02 * u)
    >>> q_interval = lambda a, b: 0.01 * (b - a)
    >>> tables = extract_tables_and_scalings(
    ...     S_by_time, t, K=5.5, r=0.02, T=0.5,
    ...     P0_func=P0, q_interval=q_interval, n=2
    ... )
    >>> tables.P_bin.shape
    (4, 4)
    >>> tables.C_p
    1.0
    """
    M = len(S_by_time)

    # time-only tables
    p = np.array([P0_func(t[i]) for i in range(1, M + 1)], dtype=float)
    dq = np.array([q_interval(t[i - 1], t[i]) for i in range(1, M + 1)], dtype=float)

    # price grid
    edges, _ = price_grid_from_samples(S_by_time, n=int(n), n_sigma=n_sigma)
    left_edges = edges[:-1]
    right_edges = edges[1:]
    mid_edges = 0.5 * (left_edges + right_edges)

    pr = payoff_repr.lower()

    if pr == "left":
        s_rep = left_edges
    elif pr == "right":
        s_rep = right_edges
    elif pr == "midpoint":
        s_rep = mid_edges
    elif pr.startswith("tilted"):
        # admite "tilted" o "tilted:0.75"
        theta = 1.0
        if ":" in pr:
            theta = float(pr.split(":", 1)[1])
        s_rep = (1.0 - theta) * left_edges + theta * right_edges
    else:
        raise ValueError("payoff_repr must be one of {'left','right','midpoint','tilted' (or 'tilted:theta')}")

    N = len(s_rep)

    P_bin = np.empty((M, N), dtype=float)
    v_rep = np.empty((M, N), dtype=float)

    for i in range(1, M + 1):
        P_bin[i - 1] = discrete_probs_from_samples(S_by_time[i - 1], edges)

        ti = t[i]
        forward_strike = K * np.exp(-r * (T - ti))
        v_rep[i - 1] = np.maximum(s_rep - forward_strike, 0.0)

    # scalings (Alcázar-style)
    # - Ensure p_tilde, dq_tilde, v_tilde are in [0,1]
    # - Avoid extremely small scalings that would kill resolution after encoding
    # - Keep C_p, C_q, C_v >= 1

    p_max  = float(np.max(p))  if np.size(p)  else 0.0
    dq_max = float(np.max(dq)) if np.size(dq) else 0.0
    v_max  = float(np.max(v_rep)) if np.size(v_rep) else 0.0

    # If you want the *tight* normalization to [0,1], choose max.
    # The eps buffer avoids hitting 1 exactly due to floating error.
    C_p = (1.0 + eps) * p_max  if p_max  > 0 else 1.0
    C_q = (1.0 + eps) * dq_max if dq_max > 0 else 1.0
    C_v = (1.0 + eps) * v_max  if v_max  > 0 else 1.0


    return DiscreteCvaTables(
        n=int(n),
        edges=edges,
        left_edges=left_edges,
        p=p,
        dq=dq,
        P_bin=P_bin,
        v_left=v_rep,   
        C_p=float(C_p),
        C_q=float(C_q),
        C_v=float(C_v),
    )

