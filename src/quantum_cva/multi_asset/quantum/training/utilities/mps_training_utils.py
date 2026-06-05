from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

import numpy as np


def cross_entropy(ptg: np.ndarray, p: np.ndarray, *, eps: float = 1e-12) -> float:
    """Return CE(ptg, p) = -sum_x ptg[x] log(p[x])."""
    ptg = np.asarray(ptg, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if ptg.shape != p.shape:
        raise ValueError(f"Shape mismatch: ptg{ptg.shape} vs p{p.shape}.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    ptg = np.clip(ptg, 0.0, None)
    p = np.clip(p, 0.0, None)
    ptg /= float(ptg.sum())
    p /= float(p.sum())
    return float(-np.sum(ptg * np.log(np.clip(p, eps, 1.0))))


def run_mps_fit(
    ptg: np.ndarray,
    *,
    mps,
    target_entropy: float,
    eps: float,
    cutoff: float = 0.0,
    rebuild_circuit: bool = True,
    p_init: np.ndarray | None = None,
    method: str = "gradient",
    init: str = "tt_svd",
    optimizer: str = "adam",
    maxiter: int = 1000,
    lr: float = 5e-2,
    tol: float = 1e-10,
    miniter: int = 25,
    seed: int | None = 42,
    init_scale: float = 1e-2,
) -> dict[str, object]:
    """
    Fit an MLMpsCircuit to a target distribution and return a QCBM-like
    training dictionary.

    By default this function runs gradient-based MPS training. The returned
    histories contain the initial uniform baseline followed by the iterative
    MPS loss history. To keep downstream code compatible with the QCBM
    training scripts, theta arrays are still empty because the trained
    parameters are MPS tensors, not circuit angles.
    """
    ptg = np.asarray(ptg, dtype=float).ravel()
    ptg = np.clip(ptg, 0.0, None)
    ptg /= float(ptg.sum())

    if p_init is None:
        p_init = np.full_like(ptg, 1.0 / ptg.size, dtype=float)
    else:
        p_init = np.asarray(p_init, dtype=float).ravel()
        if p_init.shape != ptg.shape:
            raise ValueError(f"p_init must have shape {ptg.shape}; got {p_init.shape}.")
        p_init = np.clip(p_init, 0.0, None)
        p_init /= float(p_init.sum())

    ce_init = cross_entropy(ptg, p_init, eps=eps)
    kl_init = float(ce_init - target_entropy)

    t0 = time.perf_counter()
    result = mps.fit_target(
        ptg,
        method=method,
        cutoff=float(cutoff),
        rebuild_circuit=bool(rebuild_circuit),
        init=init,
        optimizer=optimizer,
        maxiter=int(maxiter),
        lr=float(lr),
        tol=float(tol),
        miniter=int(miniter),
        seed=seed,
        init_scale=float(init_scale),
        eps=float(eps),
    )
    elapsed = time.perf_counter() - t0

    p_star = mps.probabilities()
    ce_final = cross_entropy(ptg, p_star, eps=eps)
    kl_final = float(ce_final - target_entropy)

    theta_init = np.empty(0, dtype=float)
    theta_star = np.empty(0, dtype=float)

    result_losses = np.asarray(getattr(result, "loss_history", []), dtype=float).ravel()
    if result_losses.size == 0:
        result_losses = np.array([ce_final], dtype=float)

    # Initial uniform baseline + iterative MPS training history.
    cost_history = np.r_[ce_init, result_losses]
    kl_history = np.maximum(cost_history - float(target_entropy), 0.0)
    theta_history = np.empty((cost_history.size, 0), dtype=float)

    return {
        "result": result,
        "theta_init": theta_init,
        "theta_star": theta_star,
        "theta_history": theta_history,
        "p_init": p_init,
        "p_star": p_star,
        "cost_history": cost_history,
        "kl_history": kl_history,
        "elapsed_time": float(elapsed),
        "ce_init": float(ce_init),
        "kl_init": float(kl_init),
        "ce_final": float(ce_final),
        "kl_final": float(kl_final),
        "truncation_error": float(getattr(result, "truncation_error", np.nan)),
        "effective_bond_dim": int(getattr(result, "effective_bond_dim", -1)),
        "circuit_bond_dim": int(getattr(result, "circuit_bond_dim", -1)),
    }


def select_checkpoint_indices(
    cost_history: np.ndarray,
    *,
    every: int = 25,
    top_k: int = 20,
    n_recent: int = 10,
) -> np.ndarray:
    """Same checkpoint selector used in the QCBM training utilities."""
    cost_history = np.asarray(cost_history, dtype=float).ravel()
    n = cost_history.size
    if n == 0:
        return np.empty(0, dtype=int)

    idx = {0, n - 1}
    every = max(1, int(every))
    idx.update(range(0, n, every))

    top_k = min(int(top_k), n)
    idx.update(np.argsort(cost_history)[:top_k].tolist())

    n_recent = min(int(n_recent), n)
    idx.update(range(n - n_recent, n))

    return np.array(sorted(idx), dtype=int)


def serializable_result_dict(result) -> dict[str, object]:
    """Convert a dataclass/dict/simple object result to a np.savez-friendly dict."""
    if result is None:
        return {}
    if is_dataclass(result):
        raw = asdict(result)
    elif isinstance(result, Mapping):
        raw = dict(result)
    else:
        raw = {
            name: getattr(result, name)
            for name in dir(result)
            if not name.startswith("_") and not callable(getattr(result, name))
        }

    out: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, (str, bytes)):
            out[key] = np.array(value)
        elif np.isscalar(value):
            out[key] = value
        else:
            out[key] = np.array(value, dtype=object)
    return out


def save_experiment(saving_path, checkpoint_path, main_data_dict, checkpoint_data_dict):
    np.savez(saving_path, **main_data_dict)
    np.savez(checkpoint_path, **checkpoint_data_dict)
    print(f"\nResultados guardados en: {saving_path}")
    print(f"Checkpoints guardados en: {checkpoint_path}")
