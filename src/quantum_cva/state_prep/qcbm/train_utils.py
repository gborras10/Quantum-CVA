# src/quantum_cva/qcbm/train_utils.py
from __future__ import annotations

import numpy as np
import json
import quantum_cva

from collections.abc import Callable, Mapping
from scipy.optimize import OptimizeResult
from pathlib import Path



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

def _repo_root() -> Path:
    # .../Quantum-CVA/src/quantum_cva/__init__.py -> parents[2] = repo root
    return Path(quantum_cva.__file__).resolve().parents[2]


def save_qcbm_distributions(
    *,
    qcbm,
    p_trained: np.ndarray,
    p_target: np.ndarray,
    filename_stem: str = "qcbm_output_distribution",
    subdir: str = "data/qcbm",
    metadata: Mapping[str, object] | None = None,
    tol: float = 1e-10,
) -> tuple[Path, Path]:
    """
    Save QCBM trained + target distributions as:
      - NPZ: machine-consumable (QAE-ready)
      - JSON: human-readable + metrics

    Parameters
    ----------
    qcbm
        Instance exposing: n_qubits, dim, metrics(ptg, p).
    p_trained, p_target
        Probability vectors of length 2**n_qubits.
    filename_stem
        Base name (without extension).
    subdir
        Output folder relative to repo root.
    metadata
        Extra fields to store in JSON under "metadata" (e.g. seed, optimizer, n_iters).
    tol
        Tolerance for sum-to-one checks.

    Returns
    -------
    (npz_path, json_path)
    """
    p_trained = np.asarray(p_trained, dtype=float).ravel()
    p_target = np.asarray(p_target, dtype=float).ravel()

    if p_trained.shape != p_target.shape:
        raise ValueError(f"Shape mismatch: trained{p_trained.shape} vs target{p_target.shape}.")

    dim = int(getattr(qcbm, "dim"))
    if p_trained.size != dim:
        raise ValueError(f"Expected vectors of length {dim}; got {p_trained.size}.")

    s_tr = float(p_trained.sum())
    s_tg = float(p_target.sum())
    if abs(s_tr - 1.0) > tol:
        raise ValueError(f"p_trained must sum to 1 (got {s_tr}).")
    if abs(s_tg - 1.0) > tol:
        raise ValueError(f"p_target must sum to 1 (got {s_tg}).")

    out_dir = _repo_root() / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{filename_stem}.npz"
    json_path = out_dir / f"{filename_stem}.json"

    # --- NPZ (QAE-ready) ---
    np.savez(
        npz_path,
        p_trained=p_trained,
        p_target=p_target,
        n_qubits=int(getattr(qcbm, "n_qubits")),
        dim=dim,
    )

    # --- JSON (readable + metrics) ---
    ms = qcbm.metrics(p_target, p_trained)
    n_total = int(getattr(qcbm, "n_qubits"))
    bitstrings = [format(i, f"0{n_total}b") for i in range(2**n_total)]

    md = {
        "n_qubits": n_total,
        "dim": dim,
    }
    if metadata:
        md.update({str(k): v for k, v in metadata.items()})

    payload = {
        "metadata": md,
        "metrics": {
            "kl_ptg_ptrained": float(ms["kl"]),
            "l1": float(ms["l1"]),
            "tv": float(ms["tv"]),
            "linf": float(ms["linf"]),
        },
        "distributions": {
            "trained": {b: float(p) for b, p in zip(bitstrings, p_trained)},
            "target": {b: float(p) for b, p in zip(bitstrings, p_target)},
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return npz_path, json_path

# src/quantum_cva/state_prep/qcbm/utils_plot.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_qcbm_training_diagnostics(
    *,
    ptg: np.ndarray,
    p0: np.ndarray,
    p_star: np.ndarray,
    cost_history: np.ndarray,
    best_so_far: np.ndarray,
    best_idx: np.ndarray,
    m: int,
    n: int,
    bar_width: float = 0.85,
    figsize: tuple[float, float] = (13.5, 5.2),
) -> None:
    """
    Plot QCBM training diagnostics:
      - cost curve with best-so-far
      - histograms before and after training

    Parameters
    ----------
    ptg : np.ndarray
        Target probability distribution.
    p0 : np.ndarray
        Distribution before training.
    p_star : np.ndarray
        Distribution after training.
    cost_history : np.ndarray
        Raw or rescaled cost values per iteration.
    best_so_far : np.ndarray
        Best-so-far cost curve (same length as cost_history).
    best_idx : np.ndarray
        Indices where best-so-far improves.
    m, n : int
        Number of qubits for time and price registers.
    bar_width : float, optional
        Width of histogram bars.
    figsize : tuple, optional
        Figure size.
    """
    n_total = m + n
    dim = 2 ** n_total

    if ptg.shape[0] != dim:
        raise ValueError("ptg has incompatible dimension.")
    if p0.shape[0] != dim or p_star.shape[0] != dim:
        raise ValueError("p0 / p_star have incompatible dimension.")

    bitstrings = [format(i, f"0{n_total}b") for i in range(dim)]
    x = np.arange(dim)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1, 1],
        wspace=0.30,
        hspace=0.35,
    )

    ax_cost   = fig.add_subplot(gs[:, 0])
    ax_top    = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_top)

    # ---- Cost curve ----
    ax_cost.semilogy(
        best_idx,
        best_so_far[best_idx],
        linewidth=1.5,
        label="best-so-far",
    )
    ax_cost.semilogy(
        cost_history,
        linestyle="none",
        marker=".",
        markersize=3,
        alpha=0.7,
        label="cost",
    )
    ax_cost.set_xlabel("Optimization Step")
    ax_cost.set_ylabel("Rescaled Cost Function")
    ax_cost.grid(True, which="both", alpha=0.25)
    ax_cost.legend(frameon=True)

    # ---- Before training ----
    ax_top.bar(x, ptg, width=bar_width, alpha=0.70, label="target", zorder=2)
    ax_top.bar(x, p0,  width=bar_width, alpha=0.85, label="measured", zorder=1)
    ax_top.set_title("Before training")
    ax_top.set_ylabel("Measured Probability")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(bitstrings, rotation=90, fontsize=8)
    ax_top.grid(True, axis="y", alpha=0.25)
    ax_top.legend(frameon=True)

    # ---- After training ----
    ax_bottom.bar(x, ptg,    width=bar_width, alpha=0.70, label="target", zorder=2)
    ax_bottom.bar(x, p_star, width=bar_width, alpha=0.85, label="measured", zorder=1)
    ax_bottom.set_title("After training")
    ax_bottom.set_ylabel("Measured Probability")
    ax_bottom.set_xlabel("Bitstring")
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(bitstrings, rotation=90, fontsize=8)
    ax_bottom.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

def plot_joint_and_marginals(
    *,
    ptg: np.ndarray,
    p_star: np.ndarray,
    M: int,
    N: int,
    order: str = "time_major",
    title_suffix: str = "",
) -> None:
    """
    Plot joint distributions (heatmaps) and marginals from flattened vectors.

    Parameters
    ----------
    ptg, p_star : np.ndarray
        Flattened joint probability vectors of length M*N.
    M, N : int
        Joint grid shape: M time steps, N price bins.
    order : {"time_major","price_major"}
        Flattening convention:
          - time_major: x = i*N + j  (reshape to (M,N))
          - price_major: x = j*M + i (reshape to (N,M).T)
    title_suffix : str, optional
        Extra string appended to plot titles.
    """
    ptg = np.asarray(ptg, dtype=float).ravel()
    p_star = np.asarray(p_star, dtype=float).ravel()

    if ptg.size != M * N or p_star.size != M * N:
        raise ValueError(f"ptg and p_star must have length M*N = {M*N}.")

    if order == "time_major":
        Ptg_2d = ptg.reshape(M, N)
        Pst_2d = p_star.reshape(M, N)
    elif order == "price_major":
        Ptg_2d = ptg.reshape(N, M).T
        Pst_2d = p_star.reshape(N, M).T
    else:
        raise ValueError("order must be 'time_major' or 'price_major'.")

    # Marginals
    ptg_time  = Ptg_2d.sum(axis=1)
    pst_time  = Pst_2d.sum(axis=1)
    ptg_price = Ptg_2d.sum(axis=0)
    pst_price = Pst_2d.sum(axis=0)

    # Heatmaps: target
    plt.figure()
    plt.title(f"ptg (target) heatmap {title_suffix}".strip())
    plt.imshow(Ptg_2d, aspect="auto")
    plt.colorbar()
    plt.xlabel("price bin j")
    plt.ylabel("time i")
    plt.show()

    # Heatmaps: learned
    plt.figure()
    plt.title(f"p* (learned) heatmap {title_suffix}".strip())
    plt.imshow(Pst_2d, aspect="auto")
    plt.colorbar()
    plt.xlabel("price bin j")
    plt.ylabel("time i")
    plt.show()

    # Time marginal
    plt.figure()
    plt.title("Time marginal: ptg vs p*")
    plt.plot(ptg_time, marker="o", label="ptg")
    plt.plot(pst_time, marker="o", label="p*")
    plt.xlabel("i (time)")
    plt.ylabel("probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Price marginal
    plt.figure()
    plt.title("Price marginal: ptg vs p*")
    plt.plot(ptg_price, marker="o", label="ptg")
    plt.plot(pst_price, marker="o", label="p*")
    plt.xlabel("j (price)")
    plt.ylabel("probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()