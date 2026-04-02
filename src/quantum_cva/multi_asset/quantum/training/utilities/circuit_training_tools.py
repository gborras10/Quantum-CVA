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
    ptg_time = Ptg_2d.sum(axis=1)
    pst_time = Pst_2d.sum(axis=1)
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


def plot_training_diagnostics_multi_asset(
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
    figsize_dist: tuple[float, float] = (16, 5),
    figsize_cost: tuple[float, float] = (8, 4),
    max_states: int | None = None,
    cost_log_x: bool = False,
    cost_log_y: bool = True,
) -> tuple[plt.Figure, plt.Figure]:
    """
    Plot training diagnostics:
      - Figure 1: target vs trained distribution (single histogram)
      - Figure 2: cost evolution

    Notes
    -----
    - The argument `before` is kept for interface compatibility, but it is not
      used in the distribution plot.
    - For large discrete spaces (e.g. 256 states), the x-axis shows sparse ticks
      for readability while still plotting all states.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    target = np.asarray(target, dtype=float).ravel()
    before = np.asarray(before, dtype=float).ravel()  # kept for compatibility
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
            raise ValueError(
                "best_so_far must have the same shape as cost_history."
            )

    if best_idx is None:
        tol = 1e-15
        improved = np.r_[True, best_so_far[1:] < best_so_far[:-1] - tol]
        best_idx = np.flatnonzero(improved)
    else:
        best_idx = np.asarray(best_idx, dtype=int).ravel()

    # ============================================================
    # Scenario detection: Price-only vs generic basis states
    # ============================================================
    if grid_info is not None and "s_mid" in grid_info:
        s_mid = np.asarray(grid_info["s_mid"], dtype=float).ravel()
        if s_mid.shape[0] != dim:
            raise ValueError(
                f"grid_info['s_mid'] must have length {dim}; got {s_mid.shape[0]}."
            )
        use_price_values = True
    else:
        use_price_values = False

    if xlabel is None:
        xlabel = (
            "Underlying Price"
            if use_price_values
            else "Computational basis state"
        )

    # `title_before` is intentionally unused; kept for API compatibility.
    if title_after is None:
        title_after = (
            "Target vs trained S(T) distribution"
            if use_price_values
            else "Target vs trained distribution"
        )

    # ============================================================
    # Slice for plotting
    # ============================================================
    if max_states is not None and dim > int(max_states):
        dim_plot = int(max_states)
        sl = slice(0, dim_plot)
    else:
        dim_plot = dim
        sl = slice(None)

    # x-axis values and tick labels
    if use_price_values:
        x = s_mid[sl]
        ticklabels = [f"{val:.3g}" for val in x]
        use_x_values = True
    elif x_values is not None:
        x_values = np.asarray(x_values, dtype=float).ravel()
        if x_values.shape[0] != dim:
            raise ValueError(
                f"x_values must have length {dim}; got {x_values.shape[0]}."
            )
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
            raise ValueError(
                f"labels must have length {dim}; got {len(labels)}."
            )
        ticklabels = labels[sl]

    # Histogram widths
    if use_x_values and dim_plot > 1:
        spacing = float(np.diff(x).mean())
        width_target = spacing * min(0.95, max(0.05, bar_width))
        width_after = width_target * 0.60
    else:
        width_target = min(0.95, max(0.05, bar_width))
        width_after = width_target * 0.60

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

    col_target = "lightgray"
    col_after = "darkorange"
    col_cost_pts = "#b94d95"
    col_best = "steelblue"

    target_alpha = 0.85
    after_alpha = 0.95
    bar_edge = (0, 0, 0, 0.20)
    bar_lw = 0.25

    with mpl.rc_context(rc):
        mpl.rcParams["figure.dpi"] = 100
        mpl.rcParams["path.simplify"] = False
        mpl.rcParams["patch.antialiased"] = True

        # ============================================================
        # Figure 1: Distribution comparison (target vs after only)
        # ============================================================
        fig_dist, ax_dist = plt.subplots(figsize=figsize_dist)

        ax_dist.bar(
            x,
            target[sl],
            width=width_target,
            alpha=target_alpha,
            label="target",
            zorder=2,
            color=col_target,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )
        ax_dist.bar(
            x,
            after[sl],
            width=width_after,
            alpha=after_alpha,
            label="trained",
            zorder=3,
            color=col_after,
            edgecolor=bar_edge,
            linewidth=bar_lw,
        )

        ax_dist.set_title(title_after)
        ax_dist.set_ylabel(ylabel)
        ax_dist.set_xlabel(xlabel)
        ax_dist.grid(True, axis="y")
        ax_dist.legend(loc="upper right")

        # Sparse ticks for readability
        if use_x_values:
            n_ticks = min(10, dim_plot)
            tick_indices = np.linspace(0, dim_plot - 1, n_ticks, dtype=int)
            xticks = x[tick_indices]
            xticklabels = [ticklabels[i] for i in tick_indices]
            ax_dist.set_xticks(xticks)
            ax_dist.set_xticklabels(
                xticklabels, rotation=45, ha="right", fontsize=9
            )
        else:
            if dim_plot <= 16:
                xticks = x
                xticklabels_sparse = ticklabels
                rotation = 90
                fontsize = 8
            else:
                tick_step = max(1, dim_plot // 8)  # e.g. 256 -> 32
                xticks = np.arange(0, dim_plot, tick_step)
                if xticks[-1] != dim_plot - 1:
                    xticks = np.r_[xticks, dim_plot - 1]
                xticklabels_sparse = [str(i) for i in xticks]
                rotation = 0
                fontsize = 9

            ax_dist.set_xticks(xticks)
            ax_dist.set_xticklabels(
                xticklabels_sparse, rotation=rotation, fontsize=fontsize
            )

        if max_states is not None and dim > dim_plot:
            ax_dist.annotate(
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
        # Figure 2: Cost evolution
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