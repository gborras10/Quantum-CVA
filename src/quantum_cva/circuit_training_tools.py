# src/quantum_cva/circuit_training_tools.py
from __future__ import annotations
import numpy as np
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

def minimize_with_cost_history(
    cost_fn,
    *,
    x0,
    minimize_fn,
    method,
    options,
):
    cost_history: list[float] = []
    last_val: float | None = None

    def wrapped(x):
        nonlocal last_val
        last_val = float(cost_fn(x))
        return last_val

    def callback(xk):
        if last_val is not None:
            cost_history.append(last_val)

    res = minimize_fn(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    return res, np.asarray(cost_history, dtype=float)

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

def plot_training_diagnostics(
    *,
    target: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    cost_history: np.ndarray,
    best_so_far: np.ndarray | None = None,
    best_idx: np.ndarray | None = None,
    labels: list[str] | np.ndarray | None = None,
    x_values: np.ndarray | None = None,
    grid_info: dict | None = None,
    xlabel: str | None = None,
    ylabel: str = "Probability",
    title_before: str | None = None,
    title_after: str | None = None,
    cost_xlabel: str = "Optimization Step",
    cost_ylabel: str = "Rescaled Cost Function",
    bar_width: float = 0.85,
    figsize_dist: tuple[float, float] = (14, 5),
    figsize_cost: tuple[float, float] = (10, 5),
    max_states: int | None = None,
    cost_log_x: bool = False,
    cost_log_y: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Plot training diagnostics: distributions comparison and cost evolution.
    """
    target = np.asarray(target, dtype=float).ravel()
    before = np.asarray(before, dtype=float).ravel()
    after = np.asarray(after, dtype=float).ravel()
    cost_history = np.asarray(cost_history, dtype=float).ravel()

    dim = target.shape[0]
    if before.shape[0] != dim or after.shape[0] != dim:
        raise ValueError(
            f"Incompatible dimensions: target({dim}), before({before.shape[0]}), after({after.shape[0]})."
        )
    if cost_history.size == 0:
        raise ValueError("cost_history must be non-empty.")

    if best_so_far is None:
        best_so_far = np.minimum.accumulate(cost_history)
    else:
        best_so_far = np.asarray(best_so_far, dtype=float).ravel()
        if best_so_far.shape != cost_history.shape:
            raise ValueError("best_so_far must have the same shape as cost_history.")

    if best_idx is None:
        tol = 1e-15
        improved = np.r_[True, best_so_far[1:] < best_so_far[:-1] - tol]
        best_idx = np.flatnonzero(improved)
    else:
        best_idx = np.asarray(best_idx, dtype=int).ravel()

    # ============================================================
    # Scenario detection: Price-only vs Time+Price
    # ============================================================
    if grid_info is not None and "s_mid" in grid_info:
        scenario = "price_only"
        s_mid = np.asarray(grid_info["s_mid"], dtype=float)
        if s_mid.shape[0] != dim:
            raise ValueError(f"grid_info['s_mid'] must have length {dim}; got {s_mid.shape[0]}.")
        use_price_values = True
    else:
        scenario = "time_price"
        use_price_values = False

    if xlabel is None:
        xlabel = "Underlying Price" if use_price_values else r"Computational basis state"

    if title_before is None:
        title_before = "S(T) Distribution - Before Training" if use_price_values else "Before Training"

    if title_after is None:
        title_after = "S(T) Distribution - After Training" if use_price_values else "After Training"

    if max_states is not None and dim > int(max_states):
        dim_plot = int(max_states)
        sl = slice(0, dim_plot)
    else:
        dim_plot = dim
        sl = slice(None)

    if use_price_values:
        x = s_mid[sl]
        ticklabels = [f"{val:.3g}" for val in x]
        use_x_values = True
    elif x_values is not None:
        x_values = np.asarray(x_values, dtype=float).ravel()
        if x_values.shape[0] != dim:
            raise ValueError(f"x_values must have length {dim}; got {x_values.shape[0]}.")
        x = x_values[sl]
        ticklabels = [f"{val:.3g}" for val in x]
        use_x_values = True
    else:
        x = np.arange(dim_plot)
        use_x_values = False

        if labels is None:
            n_qubits = int(np.log2(dim))
            if 2**n_qubits == dim:
                labels = [format(i, f"0{n_qubits}b") for i in range(dim)]
            else:
                labels = [str(i) for i in range(dim)]

        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if len(labels) != dim:
            raise ValueError(f"labels must have length {dim}; got {len(labels)}.")
        ticklabels = labels[sl]

    if use_x_values and dim_plot > 1:
        spacing = np.diff(x).mean()
        bar_width_actual = spacing * bar_width
    else:
        bar_width_actual = bar_width


    rc = {
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
    }

    col_target = "steelblue"
    col_meas = "darkorange"
    col_cost_pts = "#b94d95"  
    col_best = "steelblue"

    target_alpha = 0.85
    meas_alpha = 0.90
    bar_edge = (0, 0, 0, 0.28)   
    bar_lw = 0.35
    meas_width_factor = 0.55    

    with mpl.rc_context(rc):

        # High-Quality histogram 
        mpl.rcParams["figure.dpi"] = 200         
        mpl.rcParams["path.simplify"] = False     
        mpl.rcParams["patch.antialiased"] = True 

        # ============================================================
        # Figure 1: Distribution Comparison
        # ============================================================
        fig_dist, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize_dist)

        # Common tick strategy for continuous x: sparse + rotated
        if use_x_values:
            n_ticks = min(10, dim_plot)
            tick_indices = np.linspace(0, dim_plot - 1, n_ticks, dtype=int)
            xticks = x[tick_indices]
            xticklabels = [ticklabels[i] for i in tick_indices]

        # ---- Before training ----
        ax_before.bar(
            x,
            target[sl],
            width=bar_width_actual,
            alpha=target_alpha,
            label="target",
            zorder=3,
            color=col_target,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_before.bar(
            x,
            before[sl],
            width=bar_width_actual * meas_width_factor,
            alpha=meas_alpha,
            label="measured",
            zorder=2,
            color=col_meas,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_before.set_title(title_before)
        ax_before.set_ylabel(ylabel)
        ax_before.set_xlabel(xlabel)
        ax_before.grid(True, axis="y")
        ax_before.legend(loc="upper right")

        if use_x_values:
            ax_before.set_xticks(xticks)
            ax_before.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
        else:
            ax_before.set_xticks(x)
            ax_before.set_xticklabels(ticklabels, rotation=90, fontsize=8)

        # ---- After training ----
        ax_after.bar(
            x,
            target[sl],
            width=bar_width_actual,
            alpha=target_alpha,
            label="target",
            zorder=3,
            color=col_target,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_after.bar(
            x,
            after[sl],
            width=bar_width_actual * meas_width_factor,
            alpha=meas_alpha,
            label="measured",
            zorder=2,
            color=col_meas,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_after.set_title(title_after)
        ax_after.set_ylabel(ylabel)
        ax_after.set_xlabel(xlabel)
        ax_after.grid(True, axis="y")
        ax_after.legend(loc="upper right")

        if use_x_values:
            ax_after.set_xticks(xticks)
            ax_after.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
        else:
            ax_after.set_xticks(x)
            ax_after.set_xticklabels(ticklabels, rotation=90, fontsize=8)

        if max_states is not None and dim > dim_plot:
            ax_after.annotate(
                f"Showing first {dim_plot} of {dim} states",
                xy=(0.99, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=9,
                alpha=0.85,
            )

        fig_dist.tight_layout()

        # ============================================================
        # Figure 2: Cost Evolution
        # ============================================================
        fig_cost, ax_cost = plt.subplots(figsize=figsize_cost)

        steps = np.arange(cost_history.size)

        ax_cost.plot(
            best_idx,
            best_so_far[best_idx],
            linewidth=2.2,
            marker="o",
            markersize=3.2,
            label="best-so-far",
            color=col_best,
            zorder=2,
        )

        ax_cost.plot(
            steps,
            cost_history,
            linestyle="none",
            marker=".",
            markersize=2.2,
            alpha=0.55,
            label="cost",
            color=col_cost_pts,
            zorder=3,
        )

        if cost_log_x:
            ax_cost.set_xscale("log")
        if cost_log_y:
            ax_cost.set_yscale("log")

        ax_cost.set_xlabel(cost_xlabel)
        ax_cost.set_ylabel(cost_ylabel)
        ax_cost.set_title("Training Cost Evolution")
        ax_cost.grid(True, which="both")
        ax_cost.legend(loc="upper right")

        fig_cost.tight_layout()

    return fig_dist, fig_cost

def save_crca_training(
    *,
    f_trained: np.ndarray,
    f_target: np.ndarray,
    filename_stem: str,
    metadata: dict[str, Any],
    outdir: str | Path,
    f_statevector: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """
    Save CRCA training results (function values).

    Always saves:
      - f_trained (typically shot-based)
      - f_target

    Optionally saves:
      - f_statevector (ideal / statevector)

    Returns
    -------
    npz_path, json_path
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / f"{filename_stem}.npz"
    json_path = outdir / f"{filename_stem}.json"

    arrays = {
        "f_trained": np.asarray(f_trained, dtype=float),
        "f_target": np.asarray(f_target, dtype=float),
    }

    if f_statevector is not None:
        arrays["f_statevector"] = np.asarray(f_statevector, dtype=float)

    np.savez(npz_path, **arrays)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npz_path, json_path