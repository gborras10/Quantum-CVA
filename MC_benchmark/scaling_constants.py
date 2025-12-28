from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Importa las utilidades que ya tienes
from benchmark_utils import price_grid_from_samples, discrete_probs_from_samples


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
    payoff_repr: str = "left",  # "left" | "right" | "midpoint"
) -> DiscreteCvaTables:
    """
    Build tables (p_i, dq_i, P_{i,j}, v_{i,j}) and scaling constants,
    forcing C_p = 1 as in the reference implementation.
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
    if pr in ("left"):
        s_rep = left_edges
    elif pr in ("right"):
        s_rep = right_edges
    elif pr in ("midpoint"):
        s_rep = mid_edges
    else:
        raise ValueError("payoff_repr must be one of {'left','right','midpoint'}")

    N = len(s_rep)

    P_bin = np.empty((M, N), dtype=float)
    v_rep = np.empty((M, N), dtype=float)

    for i in range(1, M + 1):
        P_bin[i - 1] = discrete_probs_from_samples(S_by_time[i - 1], edges)

        ti = t[i]
        forward_strike = K * np.exp(-r * (T - ti))
        v_rep[i - 1] = np.maximum(s_rep - forward_strike, 0.0)

    # scalings
    dq_max = float(np.max(dq)) if np.size(dq) else 0.0
    v_max = float(np.max(v_rep)) if np.size(v_rep) else 0.0

    C_p = 1.0  
    C_q = (1.0 + eps) * dq_max if dq_max > 0 else 1.0
    C_v = (1.0 + eps) * v_max if v_max > 0 else 1.0

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

