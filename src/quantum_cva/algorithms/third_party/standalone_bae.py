from __future__ import annotations

import io
import math
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector

from quantum_cva.amplitude_estimation.algorithms.BAE import BAE as LegacyBAE
from quantum_cva.amplitude_estimation.algorithms.samplers import get_sampler
from quantum_cva.amplitude_estimation.utils.models import QAEmodel


class ProbingTimeQAEmodel(QAEmodel):
    """QAEmodel variant with contrast decay based on probing time 2k+1."""

    def damp_fun(self, ctrls: Any, fun: Any, whichTc: str):
        assert whichTc in ["real", "est"]
        Tc = self.Tc if whichTc == "real" else self.Tc_est
        probing_times = 2.0 * np.asarray(ctrls, dtype=float) + 1.0
        exps = np.exp(-probing_times / float(Tc))
        return exps * fun + (1.0 - exps) / 2.0


@dataclass
class BAEResult:
    estimation: float | None = None
    confidence_interval: tuple[float, float] | None = None
    num_oracle_queries: int = 0
    # Kept for compatibility with consumers that prioritize this attribute.
    num_state_prep_calls: int = 0
    powers: list[int] = field(default_factory=list)
    circuit_depths: list[int] = field(default_factory=list)
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"queries": [], "estimations": []}
    )


class StandaloneBAE:
    """Adapter around the original BAE repository implementation."""

    def __init__(
        self,
        epsilon_target: float,
        alpha: float,
        sampler: Any,
        noise_model: str = "ideal",
        T_known: float | None = None,
        cap_kappa: float = 1.0,
        max_shots_same_k: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.epsilon_target = float(epsilon_target)
        self.alpha = float(alpha)
        self._sampler = sampler  # maintained for API compatibility
        self.noise_model = noise_model
        self.T_known = T_known
        self.cap_kappa = float(cap_kappa)
        self.max_shots_same_k = max_shots_same_k

        # If True, emulate the original BAE notebook setup with Tc learning.
        default_estimate_T = bool(
            self.T_known is not None
            and not np.isinf(self.T_known)
            and noise_model != "ideal"
        )
        self.estimate_T = bool(kwargs.get("estimate_T", default_estimate_T))

        self.T_range = kwargs.get("T_range")
        self.sampler_kind = kwargs.get("sampler_kind", "RWM")

        self.strategy = {
            "wNs": int(kwargs.get("wNs", 100)),
            "Ns": int(kwargs.get("Ns", 1)),
            "TNs": int(kwargs.get("TNs", 500)),
            "k": int(kwargs.get("k", 1)),
            "Nevals": int(kwargs.get("Nevals", 50)),
            "erefs": int(kwargs.get("erefs", 3)),
            "ethr": int(kwargs.get("ethr", 3)),
            "cap": bool(kwargs.get("cap", False)),
            "capk": float(kwargs.get("capk", 2.0)),
        }
        self.strategy.update(kwargs.get("strategy", {}))

        self.sampler_kwargs = {
            "Npart": int(kwargs.get("Npart", 800)),
            "thr": float(kwargs.get("thr", 0.5)),
            "var": kwargs.get("var", "theta"),
            "ut": kwargs.get("ut", "var"),
            "c": float(kwargs.get("c", 2.38)),
            "log": bool(kwargs.get("log", True)),
            "res_ut": bool(kwargs.get("res_ut", False)),
            "plot": bool(kwargs.get("plot", False)),
        }
        self.sampler_kwargs.update(kwargs.get("sampler_kwargs", {}))

    @staticmethod
    def _infer_good_amplitude(problem: Any) -> float:
        objective_qubits = tuple(int(q) for q in getattr(problem, "objective_qubits", [0]))
        state = Statevector.from_instruction(problem.state_preparation)
        probs = state.probabilities_dict(qargs=list(objective_qubits))
        good_key = "1" * len(objective_qubits)
        amplitude = float(probs.get(good_key, 0.0))
        return float(np.clip(amplitude, 1e-12, 1.0 - 1e-12))

    def _resolve_T_range(self) -> tuple[float, float] | None:
        if self.T_range is not None:
            return tuple(self.T_range)

        if self.T_known is None or np.isinf(self.T_known):
            return None

        t = float(self.T_known)
        low = max(1.0, 0.5 * t)
        high = max(low + 1.0, 1.5 * t)
        return (low, high)

    def _build_confidence_interval(self, mean: float, std: float) -> tuple[float, float]:
        z = NormalDist().inv_cdf(1.0 - self.alpha / 2.0)
        low = max(0.0, float(mean) - z * float(std))
        high = min(1.0, float(mean) + z * float(std))
        return (low, high)

    def estimate(
        self,
        problem: Any,
        bayes: Any = None,
        show_details: bool = False,
        n_shots: int = 20,
        max_queries: int = 200000,
    ) -> BAEResult:
        del bayes  # preserved for API compatibility
        result = BAEResult()

        a_true = self._infer_good_amplitude(problem)
        finite_T = self.T_known is not None and not np.isinf(self.T_known)
        Tc_real = float(self.T_known) if finite_T else None
        T_range = self._resolve_T_range()

        # BAE expects either:
        # - Tc_precalc=False (ideal),
        # - Tc_precalc=True (learn Tc),
        # - Tc_precalc=<float> (known Tc).
        if finite_T and self.estimate_T:
            Tc_precalc: bool | float = True
            TNs = max(1, int(self.strategy.get("TNs", 500)))
        elif finite_T:
            Tc_precalc = float(Tc_real)
            TNs = 0
        else:
            Tc_precalc = False
            TNs = 0

        strategy = dict(self.strategy)
        strategy["Ns"] = max(1, int(n_shots))
        strategy["TNs"] = TNs
        strategy["capk"] = float(max(strategy.get("capk", 1.0), self.cap_kappa))

        def _execute_inference():
            model = ProbingTimeQAEmodel(a_true, Tc=Tc_real, Tcrange=T_range)
            estimator = LegacyBAE(model, Tc_precalc=Tc_precalc, Tcrange=T_range)
            sampler = get_sampler(self.sampler_kind, model, dict(self.sampler_kwargs))

            run_args = {
                "sampler": sampler,
                "strat": strategy,
                "maxPT": int(max_queries),
                "print_evol": False,
                "plot_all": False,
            }
            means, stds, nqs = estimator.adapt_inference(**run_args)
            return means, stds, nqs, estimator

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
            if show_details:
                means, stds, nqs, estimator = _execute_inference()
            else:
                with redirect_stdout(io.StringIO()):
                    means, stds, nqs, estimator = _execute_inference()

        means = [float(m) for m in means]
        stds = [float(s) for s in stds]
        nqs = [int(q) for q in nqs]

        if means:
            estimation = means[-1]
            std = stds[-1] if stds else 0.0
        else:
            estimation = a_true
            std = 0.0

        result.estimation = float(estimation)
        result.confidence_interval = self._build_confidence_interval(estimation, std)
        result.num_oracle_queries = int(nqs[-1]) if nqs else 0
        result.num_state_prep_calls = result.num_oracle_queries

        powers = [int(round(k)) for k in getattr(estimator, "ctrls_list", [])]
        result.powers = powers
        result.circuit_depths = [int(2 * k + 1) for k in powers]
        result.history = {
            "queries": nqs,
            "estimations": means,
        }

        return result
