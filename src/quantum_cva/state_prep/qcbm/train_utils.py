# src/quantum_cva/qcbm/train_utils.py
from __future__ import annotations

from collections.abc import Callable
import numpy as np
from scipy.optimize import OptimizeResult


def minimize_with_cost_history(
    cost_fn: Callable[[np.ndarray], float],
    *,
    x0: np.ndarray,
    minimize_fn: Callable[..., OptimizeResult],
    method: str,
    options: dict,
) -> tuple[OptimizeResult, np.ndarray]:
    """
    Run scipy.optimize.minimize (or a compatible function) and record the cost
    at each iterate via callback. Works with COBYLA.

    Returns
    -------
    res : OptimizeResult
    cost_history : np.ndarray
    """
    cost_history: list[float] = []

    def callback(xk: np.ndarray) -> None:
        cost_history.append(float(cost_fn(xk)))

    res = minimize_fn(
        cost_fn,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    return res, np.asarray(cost_history, dtype=float)
