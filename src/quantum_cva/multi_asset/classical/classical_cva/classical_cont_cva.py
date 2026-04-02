# python utils
from dataclasses import dataclass
from collections.abc import Sequence, Callable
import numpy as np

# quantum_cva utils
from quantum_cva.multi_asset.instruments.derivatives import Forward, Call, Put    

Instrument = Forward | Call | Put

@dataclass(frozen=True, slots=True)
class ContinuousUnderlyingCvaEngine:
    """
    Object-oriented CVA engine for the continuous-underlying setting.

    Design:
      - Keeps CDS bootstrapping external (you pass q_interval in).
      - Stores portfolio + curves + LGD as attributes.
      - Computes:
          * discount factors p(t_i)
          * default increments q(t_i)
          * portfolio MtM V(paths, time)
          * positive exposure Vpos(paths, time)
          * CVA and MC standard error

    Netting:
        - Vpos is computed at portfolio level:
            Vpos = max(V_portfolio, 0)
        - NOT sum_k max(V_k, 0).
    """

    instruments: Sequence[Instrument]
    P0_func: Callable[[float], float]                 # discount curve P(0,t)
    q_interval: Callable[[float, float], float]       # default prob on (a,b]
    LGD: float
    r: float                                          # flat r used inside instrument pricing

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

    def portfolio_mtm_matrix(
            self,
            *,
            S_by_time: np.ndarray,  # (N_paths, M, num_assets)  <-- antes list[np.ndarray]
            t: np.ndarray,          # (M,)
        ) -> np.ndarray:
        """
        Return V(paths, time): portfolio MtM (with sign) at each exposure date.
        """
        t = np.asarray(t, dtype=float).ravel()
        S_by_time = np.asarray(S_by_time, dtype=float)

        if S_by_time.ndim != 3:
            raise ValueError("S_by_time must have shape (N_paths, M, num_assets).")

        N_paths, M, num_assets = S_by_time.shape
        if M != t.size:
            raise ValueError("S_by_time.shape[1] must match len(t).")

        V = np.zeros((N_paths, M), dtype=float)

        for i, ti in enumerate(t):
            S_ti = S_by_time[:, i, :]  # (N_paths, num_assets)

            V_ti = np.zeros(N_paths, dtype=float)
            for inst in self.instruments:
                asset_index = int(inst.asset_idx)
                if asset_index < 0 or asset_index >= num_assets:
                    raise ValueError(
                        f"asset_idx {asset_index} out of bounds for num_assets={num_assets}."
                    )

                S = S_ti[:, asset_index]
                npv = inst.mtm_at_t(S, r=float(self.r), t=float(ti))
                V_ti += npv

            V[:, i] = V_ti

        return V

    def positive_exposure_matrix(
        self,
        *,
        S_by_time: list[np.ndarray],
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Return Vpos(paths, time) = max(V(paths,time), 0) with portfolio netting.
        """
        V = self.portfolio_mtm_matrix(S_by_time=S_by_time, t=t)
        return np.maximum(V, 0.0)

    def cva_from_paths(
        self,
        *,
        S_by_time: list[np.ndarray],
        t: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute (cva, std_err) using:
            CVA = LGD * E[ sum_i p(t_i) q(t_i) Vpos(t_i) ].
        """
        t = np.asarray(t, dtype=float).ravel()
        if t.size < 1:
            raise ValueError("Need at least one exposure date.")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError("t must be strictly increasing and > 0.")

        p_t = self.discount_factors_on_grid(t)
        q_t = self.default_increments_on_grid(t)
        Vpos = self.positive_exposure_matrix(S_by_time=S_by_time, t=t)

        Vpos = np.asarray(Vpos, dtype=float)
        if Vpos.ndim != 2:
            raise ValueError("Vpos must be a 2D array of shape (N_paths, M).")

        N_paths, M = Vpos.shape
        if p_t.shape != (M,) or q_t.shape != (M,):
            raise ValueError("p_t and q_t must have shape (M,) aligned with t.")
        if N_paths <= 1:
            raise ValueError("Need at least 2 paths to estimate a standard error.")

        w = p_t * q_t  # (M,)
        cva_path = Vpos @ w  # (N_paths,)

        cva = float(self.LGD * cva_path.mean())
        std_err = float(abs(self.LGD) * np.sqrt(cva_path.var(ddof=1) / N_paths))
        return cva, std_err