# src/quantum_cva/qcbm/target.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Order = Literal["time_major", "price_major"]


@dataclass(frozen=True)
class JointQcbmTarget:
    """
    Joint target distribution for QCBM training over (time, price-bin).

    Attributes
    ----------
    p_tg : np.ndarray
        Flattened joint probability vector of length M_pad * N_pad.
    M, N : int
        Original sizes: M time steps, N price bins.
    M_pad, N_pad : int
        Padded sizes (powers of two if padding enabled).
    m, n : int
        Number of qubits for time and price registers: M_pad = 2**m, N_pad = 2**n.
    order : {"time_major", "price_major"}
        Flattening convention.
        - time_major: index x = i * N_pad + j  (time bits first, then price bits)
        - price_major: index x = j * M_pad + i (price bits first, then time bits)
    time_weights : np.ndarray
        The weights π_i used for P(t_i). Length M, sums to 1.
    """

    p_tg: np.ndarray
    M: int
    N: int
    M_pad: int
    N_pad: int
    m: int
    n: int
    order: Order
    time_weights: np.ndarray


def _next_pow2(k: int) -> int:
    if k < 1:
        raise ValueError("k must be >= 1.")
    return 1 << (k - 1).bit_length()


def _check_and_normalize_prob_vector(p: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if np.any(p < -tol):
        raise ValueError("Probability vector has negative entries (beyond tolerance).")
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Probability vector has non-finite or non-positive sum.")
    if abs(s - 1.0) > tol:
        p = p / s
    return p


def build_joint_target_from_P_bin(
    P_bin: np.ndarray,
    *,
    time_weights: Optional[np.ndarray] = None,
    order: Order = "time_major",
    pad_to_pow2: bool = True,
) -> JointQcbmTarget:
    r"""
    Build the joint target distribution used to train a QCBM, following Alcázar et al.:

    \[
    p_{\text{tg}}(i,j) = P(t_i, s_j) = P(s_j \mid t_i)\,P(t_i)
    \]

    where \(P(s_j \mid t_i)\) is given by `P_bin[i, j]` (row-stochastic), and \(P(t_i)\) is
    taken uniform by default (or provided via `time_weights`).

    Parameters
    ----------
    P_bin : np.ndarray, shape (M, N)
        Conditional probabilities per time: P(s_bin=j | t_i).
        Each row is renormalized internally to sum to 1.
    time_weights : np.ndarray, shape (M,), optional
        Weights π_i for times (non-negative). If None, use uniform π_i = 1/M.
        Internally normalized to sum to 1.
    order : {"time_major", "price_major"}
        Flattening convention to map (i,j) -> x.
        - "time_major": x = i*N_pad + j  (time register first, then price register)
        - "price_major": x = j*M_pad + i
    pad_to_pow2 : bool
        If True, pad M and N to next powers of two (adds zero-probability bins/times).

    Returns
    -------
    JointQcbmTarget
        Contains flattened p_tg and metadata (M_pad, N_pad, m, n, etc.).

    Notes
    -----
    - Padding is typically required because the QCBM lives on \(2^{m+n}\) basis states.
    - If you later want a *single* register of size M*N without separate registers, keep
      "time_major" so bits are (time bits) then (price bits).
    """
    P_bin = np.asarray(P_bin, dtype=float)
    if P_bin.ndim != 2:
        raise ValueError("P_bin must be a 2D array of shape (M, N).")

    M, N = P_bin.shape

    # 1) sanitize/renormalize each conditional row P(s|t_i)
    P = np.clip(P_bin, 0.0, None)
    row_sums = P.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        bad = np.where(row_sums.ravel() <= 0.0)[0]
        raise ValueError(f"At least one row in P_bin has zero mass. Bad rows: {bad.tolist()}")
    P = P / row_sums

    # 2) define P(t_i) = π_i
    if time_weights is None:
        w = np.full(M, 1.0 / M, dtype=float)
    else:
        w = np.asarray(time_weights, dtype=float).ravel()
        if w.size != M:
            raise ValueError(f"time_weights must have length M={M}. Got {w.size}.")
        if np.any(w < 0.0):
            raise ValueError("time_weights must be non-negative.")
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("time_weights sum must be finite and > 0.")
        w = w / w_sum

    # 3) padding to powers of two (optional)
    M_pad = _next_pow2(M) if pad_to_pow2 else M
    N_pad = _next_pow2(N) if pad_to_pow2 else N

    P_pad = np.zeros((M_pad, N_pad), dtype=float)
    P_pad[:M, :N] = P

    w_pad = np.zeros(M_pad, dtype=float)
    w_pad[:M] = w

    # 4) joint p(t_i, s_j) = w_i * P(s_j|t_i)
    joint = w_pad[:, None] * P_pad  # shape (M_pad, N_pad)

    # 5) flatten in the chosen order
    if order == "time_major":
        p_tg = joint.reshape(M_pad * N_pad)
    elif order == "price_major":
        p_tg = joint.T.reshape(M_pad * N_pad)
    else:
        raise ValueError("order must be 'time_major' or 'price_major'.")

    p_tg = _check_and_normalize_prob_vector(p_tg)

    # 6) qubit counts
    # (safe because M_pad, N_pad are powers of two if pad_to_pow2=True)
    m = int(np.log2(M_pad))
    n = int(np.log2(N_pad))

    return JointQcbmTarget(
        p_tg=p_tg,
        M=M,
        N=N,
        M_pad=M_pad,
        N_pad=N_pad,
        m=m,
        n=n,
        order=order,
        time_weights=w,
    )


def build_joint_target_from_tables(
    tables,
    *,
    time_weights: Optional[np.ndarray] = None,
    order: Order = "time_major",
    pad_to_pow2: bool = True,
) -> JointQcbmTarget:
    """
    Convenience wrapper if you pass the DiscreteCvaTables object (from mc_benchmark.scaling_constants).

    Expects `tables.P_bin` to exist with shape (M, N).
    """
    if not hasattr(tables, "P_bin"):
        raise AttributeError("tables must have attribute 'P_bin'.")
    return build_joint_target_from_P_bin(
        tables.P_bin,
        time_weights=time_weights,
        order=order,
        pad_to_pow2=pad_to_pow2,
    )