# python utils
import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence

@dataclass(frozen=True, slots=True)
class GridSpec:
    """
    Specification of a multi-asset tensor-product price grid.

    edges_list[k] : bin edges for asset k, shape (N_k+1,)
    rep_list[k]   : representative price per bin for asset k, shape (N_k,)
    n_bins        : (N_1, ..., N_d)
    N_joint       : total number of joint states = prod_k N_k
    """

    edges_list: tuple[np.ndarray, ...]
    rep_list: tuple[np.ndarray, ...]
    n_bins: tuple[int, ...]
    N_joint: int

    @property
    def num_assets(self) -> int:
        return len(self.n_bins)

    def validate(self) -> None:
        d = self.num_assets

        if d < 1:
            raise ValueError("GridSpec must contain at least one asset.")

        if len(self.edges_list) != d or len(self.rep_list) != d:
            raise ValueError("edges_list and rep_list must have length equal to num_assets.")

        if len(self.n_bins) != d:
            raise ValueError("n_bins must have length equal to num_assets.")

        if int(np.prod(self.n_bins)) != int(self.N_joint):
            raise ValueError("N_joint must equal product of n_bins.")

        for k in range(d):
            edges = np.asarray(self.edges_list[k], dtype=float)
            reps = np.asarray(self.rep_list[k], dtype=float)
            Nk = int(self.n_bins[k])

            if edges.ndim != 1 or reps.ndim != 1:
                raise ValueError("Edges and representatives must be 1D arrays.")

            if edges.shape[0] != Nk + 1:
                raise ValueError(f"edges_list[{k}] must have length N_k+1.")

            if reps.shape[0] != Nk:
                raise ValueError(f"rep_list[{k}] must have length N_k.")

            if not np.all(np.diff(edges) > 0.0):
                raise ValueError(f"edges_list[{k}] must be strictly increasing.")


# Internal helpers
def _representatives_from_edges(edges: np.ndarray, payoff_repr: str) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)

    left = edges[:-1]
    right = edges[1:]
    mid = 0.5 * (left + right)

    pr = payoff_repr.lower()

    if pr in ("left", "l"):
        return left
    if pr in ("right", "r"):
        return right
    if pr in ("mid", "midpoint", "m"):
        return mid

    raise ValueError("payoff_repr must be one of {'left','right','mid'}.")


def joint_representatives_matrix(grid: GridSpec) -> np.ndarray:
    """
    Return joint representative states S_rep_joint of shape (N_joint, d).
    """
    reps = [np.asarray(r, dtype=float) for r in grid.rep_list]

    meshes = np.meshgrid(*reps, indexing="ij")
    cols = [m.ravel(order="C") for m in meshes]

    return np.stack(cols, axis=1)


# ------------------ Discretization from Monte Carlo samples ------------------

def grid_and_prob_tensor_multiasset(
    S_by_time: list[np.ndarray],      
    n_bits: Sequence[int],            
    *,
    n_sigma: float = 3.0,
    payoff_repr: str = "mid",
) -> tuple[GridSpec, np.ndarray]:
    """
    Build:

        - GridSpec
        - P_joint_t  (M, N_joint)

    using multi-dimensional histogramming.
    """
    X_T = np.asarray(S_by_time[-1], dtype=float)

    if X_T.ndim != 2:
        raise ValueError("Each S_by_time[i] must have shape (N_paths, d).")

    N_paths, num_assets = X_T.shape

    n_bits = tuple(int(nb) for nb in n_bits)

    if len(n_bits) != num_assets:
        raise ValueError("n_bits must match number of assets.")

    edges_list = []
    rep_list = []
    n_bins = []

    for k in range(num_assets):
        N_k = 2 ** n_bits[k]

        xk = X_T[:, k]
        muhat = float(xk.mean())
        sighat = float(xk.std(ddof=1))

        s0 = max(muhat - n_sigma * sighat, 0.0)
        sN = muhat + n_sigma * sighat

        edges = np.linspace(s0, sN, N_k + 1)
        reps = _representatives_from_edges(edges, payoff_repr)

        edges_list.append(edges)
        rep_list.append(reps)
        n_bins.append(N_k)

    N_joint = int(np.prod(n_bins))

    grid = GridSpec(
        edges_list=tuple(edges_list),
        rep_list=tuple(rep_list),
        n_bins=tuple(n_bins),
        N_joint=N_joint,
    )

    grid.validate()

    P_rows = []

    for Si in S_by_time:
        Si = np.asarray(Si, dtype=float)

        if Si.shape != (N_paths, num_assets):
            raise ValueError("All S_by_time[i] must have same shape.")

        counts, _ = np.histogramdd(Si, bins=list(grid.edges_list))

        total = float(counts.sum())

        if total <= 0.0:
            raise ValueError("No samples in grid range; increase n_sigma.")

        P_rows.append((counts / total).reshape(N_joint))

    P_joint_t = np.asarray(P_rows, dtype=float)

    P_joint_t = np.clip(P_joint_t, 0.0, None)
    P_joint_t /= P_joint_t.sum(axis=1, keepdims=True)

    return grid, P_joint_t


# -------------- Quantum-ready joint probability construction ----------------

def build_joint_p_target(
    P_joint_t: np.ndarray,           
    *,
    order: str = "time_major",
    time_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build flattened joint distribution:
    p_target(i, b) = w_t[i] * P_joint_t[i, b]
    """

    P_joint_t = np.asarray(P_joint_t, dtype=float)

    if P_joint_t.ndim != 2:
        raise ValueError("P_joint_t must be 2D (M, N_joint).")

    M, N_joint = P_joint_t.shape

    # Ensure powers of two (for quantum registers)
    if (M & (M - 1)) != 0:
        raise ValueError("Number of time points M must be a power of two.")
    
    if (N_joint & (N_joint - 1)) != 0:
        raise ValueError("N_joint must be a power of two.")
    
    # Time weights defauult to uniform if not provided
    if time_weights is None:
        w_t = np.full(M, 1.0 / M, dtype=float)
    else:
        w_t = np.asarray(time_weights, dtype=float).ravel()
        if w_t.shape != (M,):
            raise ValueError(f"time_weights must have shape (M,), got {w_t.shape}.")
        s = float(w_t.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("time_weights must have a positive finite sum.")
        w_t = w_t / s

    joint_probability = w_t[:, None] * P_joint_t  # (M, N_joint)

    if order == "time_major":
        p_target = joint_probability.reshape(M * N_joint)
    elif order == "price_major":
        p_target = joint_probability.T.reshape(M * N_joint)
    else:
        raise ValueError("order must be 'time_major' or 'price_major'.")

    p_target = np.clip(p_target, 0.0, None) # avoid negative values due to numerical issues
    p_target /= float(p_target.sum()) # ensure normalization 

    return p_target, w_t