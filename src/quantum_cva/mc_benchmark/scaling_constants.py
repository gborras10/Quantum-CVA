# src/quantum_cva/mc_benchmark/scaling_constants.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class DiscreteCvaTables:
    n: int
    edges: np.ndarray      # (N+1,)
    s_rep: np.ndarray      # (N,)
    p_t: np.ndarray        # (M+1,)
    q_t: np.ndarray        # (M+1,) with q_t[0]=0
    P_s_t: np.ndarray      # (M+1, N)
    v_s_t: np.ndarray      # (M+1, N)
    C_p: float
    C_q: float
    C_v: float


def extract_tables_and_scalings(
    S_by_time: list[np.ndarray],
    t: np.ndarray,
    *,
    K: float,
    r: float,
    T: float,
    p_t: np.ndarray,
    q_t: np.ndarray,
    n: int,
    n_sigma: float = 3.0,
    eps: float = 1e-6,
    payoff_repr: str = "left",  # "left" | "right" | "mid" | "midpoint" | "tilted[:theta]"
) -> DiscreteCvaTables:
    """
    Precompute discrete CVA blocks (grid, probabilities, payoff) and scaling constants.

    Assumes p_t and q_t are already computed on the full time grid t[0..M]
    (with q_t[0]=0). This function only builds:
      - global price grid edges
      - discrete probability matrix P_s_t (histograms per time)
      - payoff matrix v_s_t on representative bin prices
      - scaling constants C_p, C_q, C_v

    Parameters
    ----------
    S_by_time : list[np.ndarray]
        Samples of the underlying aligned with t[0..M], length M+1.
    t : np.ndarray
        Time grid of length M+1 including t[0]=0.
    K, r, T : float
        Payoff parameters (forward-like call exposure).
    p_t : np.ndarray
        Discount factors on the full grid, shape (M+1,).
    q_t : np.ndarray
        Default increments on the full grid, shape (M+1,), with q_t[0]=0.
    n : int
        Price discretization level (N = 2**n bins).
    n_sigma : float, optional
        Truncation width (terminal mean ± n_sigma * std) for the global grid.
    eps : float, optional
        Safety buffer for scaling constants.
    payoff_repr : str, optional
        Representative price per bin for payoff evaluation. Supports:
        "left", "right", "mid"/"midpoint", "tilted" or "tilted:theta" (0..1).

    Returns
    -------
    DiscreteCvaTables
        Precomputed blocks and scaling constants.
    """
    t = np.asarray(t, dtype=float)
    p_t = np.asarray(p_t, dtype=float)
    q_t = np.asarray(q_t, dtype=float)

    if len(S_by_time) != len(t):
        raise ValueError("S_by_time must have length len(t) (include t0).")
    if p_t.shape != (len(t),) or q_t.shape != (len(t),):
        raise ValueError("p_t and q_t must have shape (M+1,) aligned with t.")
    if float(q_t[0]) != 0.0:
        raise ValueError("q_t[0] must be 0.0 (t0 contributes nothing).")

    # --- grid + prob matrix (concise, no external helpers) ---
    N = 2 ** int(n)
    X = np.asarray(S_by_time[-1], dtype=float)
    muhat = float(X.mean())
    sighat = float(X.std(ddof=1))
    s0 = max(muhat - float(n_sigma) * sighat, 0.0)
    sN = muhat + float(n_sigma) * sighat

    edges = np.linspace(s0, sN, N + 1, dtype=float)
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
    elif pr.startswith("tilted"):
        theta = 1.0
        if ":" in pr:
            theta = float(pr.split(":", 1)[1])
        s_rep = (1.0 - theta) * left + theta * right
    else:
        raise ValueError("payoff_repr must be one of {'left','right','midpoint','tilted' (or 'tilted:theta')}")

    P_rows = []
    for Si in S_by_time:
        counts, _ = np.histogram(np.asarray(Si, dtype=float), bins=edges)
        tot = int(counts.sum())
        if tot == 0:
            raise ValueError("No samples in range for at least one time; widen grid (increase n_sigma).")
        P_rows.append(counts / tot)
    P_s_t = np.asarray(P_rows, dtype=float)  # (M+1, N)

    # --- payoff matrix (vectorized) ---
    fwd_strike = K * np.exp(-r * (T - t))                       # (M+1,)
    v_s_t = np.maximum(s_rep[None, :] - fwd_strike[:, None], 0.0)  # (M+1, N)

    # --- scaling constants ---
    p_max = float(np.max(p_t)) if p_t.size else 0.0
    q_max = float(np.max(q_t)) if q_t.size else 0.0
    v_max = float(np.max(v_s_t)) if v_s_t.size else 0.0

    C_p = (1.0 + eps) * p_max if p_max > 0 else 1.0
    C_q = (1.0 + eps) * q_max if q_max > 0 else 1.0
    C_v = (1.0 + eps) * v_max if v_max > 0 else 1.0

    return DiscreteCvaTables(
        n=int(n),
        edges=edges,
        s_rep=s_rep,
        p_t=p_t,
        q_t=q_t,
        P_s_t=P_s_t,
        v_s_t=v_s_t,
        C_p=float(C_p),
        C_q=float(C_q),
        C_v=float(C_v),
    )
