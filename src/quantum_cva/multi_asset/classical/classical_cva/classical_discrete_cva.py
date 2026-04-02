# python utils
import numpy as np
from dataclasses import dataclass
from collections.abc import Callable, Sequence

# quantum_cva utils
from quantum_cva.multi_asset.instruments.derivatives import (
    Forward,
    Call,
    Put,
)

from quantum_cva.multi_asset.classical.probability_and_underlying.multi_asset_discrete_probability_utils import (
    GridSpec,
    grid_and_prob_tensor_multiasset,
    joint_representatives_matrix,
    build_joint_p_target,
)

Instrument = Forward | Call | Put

@dataclass(frozen=True, slots=True)
class DiscreteUnderlyingCvaEngine:
    """
    Object-oriented CVA engine for the discrete-underlying (Alcázar-style) setting.

    Design:
      - Keeps CDS bootstrapping external 
      - Stores portfolio + curves + LGD as attributes.
      - Discretizes the multi-asset state on a tensor grid.
      - Estimates P(state | t_i) by histogramming Monte Carlo samples.
      - Builds portfolio positive exposure on the grid using representative prices:
            v(t_i, state) = max(V_portfolio(t_i, s_rep(state)), 0)
      - Computes:
            CVA = LGD * sum_i p(t_i) q(t_i) * sum_state P(state|t_i) v(t_i,state)

    Netting:
        - Netting is applied at portfolio level before taking the positive part.
    """

    instruments: Sequence[Instrument]
    P0_func: Callable[[float], float]                 # discount curve P(0,t)
    q_interval: Callable[[float, float], float]       # default prob on (a,b]
    LGD: float
    r: float                                          # flat r used inside instrument pricing

    # discretization config
    n_bits: Sequence[int]                             # per asset bits (N_k=2**n_bits[k])
    n_sigma: float = 3.0
    payoff_repr: str = "mid"                          # "mid" | "left" | "right"
    order: str = "time_major"                         # flattening for p_target
    time_weights: np.ndarray | None = None            # optional P(t_i) weights for p_target

    # -------------------------------------------------
    # Shared curve utilities (same style as continuous)
    # -------------------------------------------------
    def discount_factors_on_grid(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).ravel()
        return np.array([float(self.P0_func(float(ti))) for ti in t], dtype=float)

    def default_increments_on_grid(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float).ravel()
        q = np.zeros_like(t, dtype=float)
        t_prev = 0.0
        for i, ti in enumerate(t):
            q[i] = float(self.q_interval(float(t_prev), float(ti)))
            t_prev = float(ti)
        return q

    # -------------------------------------------------
    # Discrete distribution + QCBM target
    # -------------------------------------------------
    def fit_discrete_distribution(
        self,
        *,
        S_by_time: list[np.ndarray],  # len M, each (N_paths, d)
    ) -> tuple[GridSpec, np.ndarray]:
        """
        Estimate the discrete conditional distribution P(state | t_i).

        Returns
        -------
        grid : GridSpec
        P_joint_t : np.ndarray
            Shape (M, N_joint). Each row sums to 1.
        """
        return grid_and_prob_tensor_multiasset(
            S_by_time,
            n_bits=self.n_bits,
            n_sigma=float(self.n_sigma),
            payoff_repr=str(self.payoff_repr),
        )

    def build_p_target(
        self,
        *,
        P_joint_t: np.ndarray,  # (M, N_joint)
        time_weights: np.ndarray | None = None,
        order: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build QCBM-ready flattened joint distribution over (time, joint-state).

        Returns
        -------
        p_target : (M*N_joint,)
        w_t      : (M,)
        """
        return build_joint_p_target(
            P_joint_t,
            order=self.order if order is None else str(order),
            time_weights=self.time_weights if time_weights is None else time_weights,
        )

    # -------------------------------------------------
    # Portfolio payoff (positive exposure) on the grid
    # -------------------------------------------------
    def payoff_matrix_portfolio_on_grid(
        self,
        *,
        grid: GridSpec,
        t: np.ndarray,  # (M,)
        return_mtm: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Build:
            V(t_i, state)    = portfolio MTM (with sign) on representative prices
            Vpos(t_i, state) = max(V(t_i, state), 0)

        Returns
        -------
        Vpos_joint_t : (M, N_joint)
        Optionally, V_joint_t : (M, N_joint)
        """
        t = np.asarray(t, dtype=float).ravel()
        if t.size < 1:
            raise ValueError("Need at least one exposure date.")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("t must be strictly increasing and > 0.")

        S_rep_joint = joint_representatives_matrix(grid)  # (N_joint, d)
        N_joint, d = S_rep_joint.shape

        V_joint_t = np.empty((t.size, N_joint), dtype=float)

        for i, ti in enumerate(t):
            V = np.zeros(N_joint, dtype=float)

            for inst in self.instruments:
                asset_index = int(inst.asset_idx)
                if asset_index < 0 or asset_index >= d:
                    raise ValueError(
                        f"asset_idx {asset_index} out of bounds for d={d}."
                    )

                S = S_rep_joint[:, asset_index]  # (N_joint,)

                # inst.mtm_at_t is MTM of a LONG instrument
                npv = inst.mtm_at_t(S, r=float(self.r), t=float(ti))
                V += np.asarray(npv, dtype=float)

            V_joint_t[i, :] = V

        Vpos_joint_t = np.maximum(V_joint_t, 0.0)

        if return_mtm:
            return Vpos_joint_t, V_joint_t
        return Vpos_joint_t

    # -------------------------------------------------
    # CVA from discrete blocks (method, not free function)
    # -------------------------------------------------

    def cva_from_discrete_blocks(
        self,
        *,
        P_joint_t: np.ndarray,  # (M, N_joint)
        v_joint_t: np.ndarray,  # (M, N_joint)
        t: np.ndarray,          # (M,)
        C_v: float = 1.0,
        C_p: float = 1.0,
        C_q: float = 1.0,
    ) -> float:
        """
        Compute discrete CVA from precomputed blocks aligned with exposure dates.
        """
        t = np.asarray(t, dtype=float).ravel()
        p_t = self.discount_factors_on_grid(t)          # (M,)
        q_t = self.default_increments_on_grid(t)        # (M,)

        P_joint_t = np.asarray(P_joint_t, dtype=np.float32)
        v_joint_t = np.asarray(v_joint_t, dtype=np.float32)

        if P_joint_t.ndim != 2 or v_joint_t.ndim != 2:
            raise ValueError("P_joint_t and v_joint_t must be 2D arrays.")
        if P_joint_t.shape != v_joint_t.shape:
            raise ValueError("P_joint_t and v_joint_t must have the same shape (M, N_joint).")

        M, _ = P_joint_t.shape
        if p_t.shape != (M,) or q_t.shape != (M,):
            raise ValueError("p_t and q_t must have shape (M,) aligned with P_joint_t.")

        p_tilde = p_t / float(C_p)
        q_tilde = q_t / float(C_q)

        prefactor = float(self.LGD) * float(C_v) * float(C_p) * float(C_q)
        inv_Cv = 1.0 / float(C_v)

        bracket = 0.0
        for i in range(M):
            # E_tilde[i] = sum_b P(state|t_i) * (v(t_i,state)/C_v)
            E_tilde_i = float(np.dot(P_joint_t[i], v_joint_t[i]) * inv_Cv)
            bracket += E_tilde_i * float(p_tilde[i]) * float(q_tilde[i])

        return float(prefactor * bracket)

    # -------------------------------------------------
    # Full pipeline convenience method
    # -------------------------------------------------
    def cva_from_paths_discretized(
            self,
            *,
            S_by_time: np.ndarray,   # (N_paths, M, d)   <-- antes list[np.ndarray]
            t: np.ndarray,           # (M,)
            C_v: float = 1.0,
            C_p: float = 1.0,
            C_q: float = 1.0,
            return_blocks: bool = False,
        ) -> float | tuple[float, GridSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        t = np.asarray(t, dtype=float).ravel()
        S_by_time = np.asarray(S_by_time, dtype=float)

        if S_by_time.ndim != 3:
            raise ValueError("S_by_time must have shape (N_paths, M, d).")
        N_paths, M, d = S_by_time.shape

        if M != t.size:
            raise ValueError("S_by_time.shape[1] must match len(t).")
        if t.size < 1:
            raise ValueError("Need at least one exposure date.")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("t must be strictly increasing and > 0.")

        # Si tus funciones internas esperan list[np.ndarray], conviértelo aquí (vista, sin copies grandes)
        S_list = [S_by_time[:, i, :] for i in range(M)]  # len M, each (N_paths, d)

        grid, P_joint_t = self.fit_discrete_distribution(S_by_time=S_list)
        v_joint_t = self.payoff_matrix_portfolio_on_grid(grid=grid, t=t)

        cva = self.cva_from_discrete_blocks(
            P_joint_t=P_joint_t,
            v_joint_t=v_joint_t,
            t=t,
            C_v=C_v,
            C_p=C_p,
            C_q=C_q,
        )

        if not return_blocks:
            return float(cva)

        p_target, w_t = self.build_p_target(P_joint_t=P_joint_t)
        return float(cva), grid, P_joint_t, v_joint_t, p_target, w_t