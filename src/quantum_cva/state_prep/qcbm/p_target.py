# src/quantum_cva/qcbm/target.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Order = Literal["time_major", "price_major"]


@dataclass(frozen=True)
class JointQcbmTarget:
    """
    Joint target distribution for QCBM training over (time, price-bin).

    Attributes
    ----------
    p_tg : np.ndarray
        Flattened joint probability vector of length M * N.
    M, N : int
        Original sizes: M time steps, N price bins.
    m, n : int
        Number of qubits for time and price registers: M = 2**m, N = 2**n.
    order : {"time_major", "price_major"}
        Flattening convention.
        - time_major: index x = i * N + j  (time bits first, then price bits)
        - price_major: index x = j * M + i (price bits first, then time bits)
    time_weights : np.ndarray
        The weights pi_i used for P(t_i). Length M, sums to 1.
    """

    p_tg: np.ndarray
    M: int
    N: int
    m: int
    n: int
    order: Order
    time_weights: np.ndarray


def _check_and_normalize_prob_vector(p: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Validate and normalize a probability vector to sum to 1 within tolerance."""
    p = np.asarray(p, dtype=float).ravel()
    if np.any(p < -tol):
        raise ValueError("Probability vector has negative entries (beyond tolerance).")
    p = np.clip(p, 0.0, None) # remove small negatives
    prob_sum = float(p.sum())
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError("Probability vector has non-finite or non-positive sum.")
    if abs(prob_sum - 1.0) > tol:
        p = p / prob_sum
    return p


def build_joint_target_from_P_bin(
    P_bin: np.ndarray,
    *,
    order: Order = "time_major",
) -> JointQcbmTarget:
    r"""
    Build the joint target distribution used to train a QCBM, following:

    p_{tg}(i,j) = P(t_i, s_j) = P(s_j|t_i) · P(t_i)

    where P(s_j|t_i)\) is given by `P_bin[i, j]` (row-stochastic), and P(t_i) is
    uniform.

    Parameters
    ----------
    P_bin : np.ndarray, shape (M, N)
        Conditional probabilities per time: P(s_bin=j | t_i).
        Each row is renormalized internally to sum to 1.
    order : {"time_major", "price_major"}
        Flattening convention to map (i,j) -> x.
        - "time_major": x = i*N + j  (time register first, then price register)
        - "price_major": x = j*M + i

    Returns
    -------
    JointQcbmTarget
        Contains flattened p_tg and metadata (M, N, m, n, etc.).
    """
    P_bin = np.asarray(P_bin, dtype=float)
    if P_bin.ndim != 2:
        raise ValueError("P_bin must be a 2D array of shape (M, N).")

    M, N = P_bin.shape

    # Sanitize/renormalize each conditional row P(s|t_i)
    P = np.clip(P_bin, 0.0, None)
    row_sums = P.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        bad = np.where(row_sums.ravel() <= 0.0)[0]
        raise ValueError(f"At least one row in P_bin has zero mass. Bad rows: {bad.tolist()}")
    P = P / row_sums

    # Define P(t_i) = pi_i
    w = np.full(M, 1.0 / M, dtype=float)

    # Require sizes compatible with qubit registers
    if (M & (M - 1)) != 0 or (N & (N - 1)) != 0:
        raise ValueError("M and N must be powers of two.")

    # Joint p(t_i, s_j) = w_i * P(s_j|t_i)
    joint = w[:, None] * P  # shape (M, N)

    # Flatten in the chosen order
    if order == "time_major":
        p_tg = joint.reshape(M * N)
    elif order == "price_major":
        p_tg = joint.T.reshape(M * N)
    else:
        raise ValueError("order must be 'time_major' or 'price_major'.")

    p_tg = _check_and_normalize_prob_vector(p_tg)

    # Qubit counts
    # (safe because M and N are powers of two)
    m = int(np.log2(M))
    n = int(np.log2(N))

    return JointQcbmTarget(
        p_tg=p_tg,
        M=M,
        N=N,
        m=m,
        n=n,
        order=order,
        time_weights=w,
    )
