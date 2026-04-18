from __future__ import annotations

import io
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

from quantum_cva.amplitude_estimation.algorithms.BAE import BAE as LegacyBAE
from quantum_cva.amplitude_estimation.algorithms.samplers import get_sampler
from quantum_cva.amplitude_estimation.utils.models import QAEmodel


CountsProvider = Callable[[QuantumCircuit, int], Mapping[str, int]]


@dataclass
class _MeasurementRecord:
    control: int
    shots: int
    one_counts: int
    var: str


@dataclass
class BAEResult:
    """Result container compatible with existing benchmark scripts."""

    estimation: float | None = None
    confidence_interval: tuple[float, float] | None = None
    num_oracle_queries: int = 0
    # Kept for compatibility with consumers that prioritize this attribute.
    num_state_prep_calls: int = 0
    powers: list[int] = field(default_factory=list)
    circuit_depths: list[int] = field(default_factory=list)
    K_sequence: list[int] = field(default_factory=list)
    K_max: int = 0
    history: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "queries": [],
            "estimations": [],
            "stds": [],
            "controls": [],
            "one_counts": [],
            "shots": [],
            "K_sequence": [],
            "circuit_depths": [],
        }
    )


def _is_sequence_but_not_str(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _normalize_counts(counts: Mapping[Any, Any]) -> dict[str, int]:
    """Normalize counts to a plain ``dict[str, int]`` and validate values."""

    normalized: dict[str, int] = {}
    for key, value in counts.items():
        if isinstance(value, Integral):
            ivalue = int(value)
        elif isinstance(value, Real) and float(value).is_integer():
            ivalue = int(round(float(value)))
        else:
            raise TypeError(
                "Counts values must be integer-like shot counts; received "
                f"{value!r} for key {key!r}."
            )

        if ivalue < 0:
            raise ValueError(f"Counts must be non-negative; received {ivalue} for key {key!r}.")

        normalized[str(key).replace(" ", "")] = ivalue

    if not normalized:
        raise ValueError("Received empty counts dictionary.")

    return normalized


def _extract_counts_from_data_bin(data_bin: Any) -> Mapping[Any, Any] | None:
    """Extract counts from the data payload returned by sampler primitives."""

    if data_bin is None:
        return None

    if hasattr(data_bin, "get_counts"):
        return data_bin.get_counts()

    # Typical SamplerV2 shape: data.c0.get_counts() or data.c.get_counts().
    for attr in ("c0", "c"):
        maybe_register = getattr(data_bin, attr, None)
        if maybe_register is not None and hasattr(maybe_register, "get_counts"):
            return maybe_register.get_counts()

    for attr in dir(data_bin):
        if attr.startswith("_"):
            continue
        maybe_register = getattr(data_bin, attr)
        if hasattr(maybe_register, "get_counts"):
            return maybe_register.get_counts()

    return None


def _extract_single_counts(payload: Any, circuit: QuantumCircuit | None = None) -> dict[str, int]:
    """Extract one-circuit counts from sampler/backends with different return shapes."""

    if isinstance(payload, Mapping):
        return _normalize_counts(payload)

    if _is_sequence_but_not_str(payload):
        if len(payload) == 0:
            raise ValueError("Sampler returned an empty sequence of results.")
        return _extract_single_counts(payload[0], circuit=circuit)

    data_bin = getattr(payload, "data", None)
    counts_from_data = _extract_counts_from_data_bin(data_bin)
    if counts_from_data is not None:
        return _normalize_counts(counts_from_data)

    if hasattr(payload, "get_counts"):
        getter = payload.get_counts
        try:
            counts = getter()
        except TypeError:
            if circuit is None:
                raise
            counts = getter(circuit)
        return _extract_single_counts(counts, circuit=circuit)

    quasi = getattr(payload, "quasi_dists", None)
    if quasi is not None:
        raise TypeError(
            "Received quasi-distributions instead of counts. "
            "Provide a sampler/backend that returns measured counts."
        )

    raise TypeError(
        "Unsupported sampler/backend result format; could not extract counts from "
        f"object of type {type(payload).__name__}."
    )


class SamplerCountsAdapter:
    """Adapter that turns a sampler/backend into ``CountsProvider``."""

    def __init__(self, sampler_or_backend: Any):
        self._sampler_or_backend = sampler_or_backend

    def __call__(self, circuit: QuantumCircuit, shots: int) -> Mapping[str, int]:
        source = self._sampler_or_backend

        if callable(source) and not hasattr(source, "run"):
            return _normalize_counts(source(circuit, shots))

        if not hasattr(source, "run"):
            raise TypeError(
                "Expected a callable counts adapter or an object exposing run(...); "
                f"received {type(source).__name__}."
            )

        run_method = source.run
        try:
            job_or_result = run_method([circuit], shots=shots)
        except TypeError:
            try:
                job_or_result = run_method(circuit, shots=shots)
            except TypeError:
                job_or_result = run_method([circuit], shots)

        result_payload = job_or_result.result() if hasattr(job_or_result, "result") else job_or_result
        return _extract_single_counts(result_payload, circuit=circuit)


class HardwareQAEmodel(QAEmodel):
    """
    QAEmodel variant where measurements come from external counts.

    The likelihood logic is inherited from QAEmodel. Only ``measure`` is replaced,
    so posterior updates rely on real (or externally supplied) circuit counts.
    """

    def __init__(
        self,
        problem: Any,
        counts_provider: CountsProvider,
        Tcrange: tuple[float, float] | None,
    ) -> None:
        # ``a`` and ``Tc`` are not used for data generation here.
        super().__init__(a=0.5, Tc=None, Tcrange=Tcrange)
        self.problem = problem
        self._counts_provider = counts_provider
        self._circuit_cache: dict[int, QuantumCircuit] = {}
        self.execution_log: list[_MeasurementRecord] = []

        objective_qubits = getattr(problem, "objective_qubits", None)
        if objective_qubits is None:
            objective_qubits = [0]
        self.objective_qubits = tuple(int(q) for q in objective_qubits)

    def damp_fun(self, ctrls: Any, fun: Any, whichTc: str):
        """Apply exponential contrast decay using probing time 2k+1."""
        assert whichTc in ["real", "est"]
        Tc = self.Tc if whichTc == "real" else self.Tc_est
        probing_times = 2.0 * np.asarray(ctrls, dtype=float) + 1.0
        exps = np.exp(-probing_times / float(Tc))
        return exps * fun + (1.0 - exps) / 2.0

    def _build_measured_circuit(self, control: int) -> QuantumCircuit:
        if control in self._circuit_cache:
            return self._circuit_cache[control].copy()

        if not hasattr(self.problem, "state_preparation"):
            raise ValueError("Estimation problem does not expose state_preparation.")
        if not hasattr(self.problem, "grover_operator"):
            raise ValueError("Estimation problem does not expose grover_operator.")

        num_qubits = max(
            self.problem.state_preparation.num_qubits,
            self.problem.grover_operator.num_qubits,
        )

        circuit = QuantumCircuit(num_qubits, name="bae_hardware_circuit")
        circuit.compose(self.problem.state_preparation, inplace=True)

        if control > 0:
            grover_power = self.problem.grover_operator.power(control)
            if hasattr(grover_power, "decompose"):
                grover_power = grover_power.decompose()
            circuit.compose(grover_power, inplace=True)

        classical = ClassicalRegister(len(self.objective_qubits), "c0")
        circuit.add_register(classical)
        circuit.barrier()
        circuit.measure(self.objective_qubits, classical[:])

        circuit.metadata = {
            "bae_control": int(control),
            "grover_power": int(control),
            "K_value": int(2 * control + 1),
        }

        self._circuit_cache[control] = circuit
        return circuit.copy()

    def _count_good_states(self, counts: Mapping[str, int]) -> int:
        one_counts = 0
        objective_width = len(self.objective_qubits)
        is_good_state = getattr(self.problem, "is_good_state", None)

        for raw_state, raw_count in counts.items():
            state = str(raw_state).replace(" ", "")
            if state.startswith("0x"):
                state = format(int(state, 16), f"0{objective_width}b")

            if callable(is_good_state):
                if is_good_state(state):
                    one_counts += int(raw_count)
            else:
                if state == ("1" * objective_width):
                    one_counts += int(raw_count)

        return one_counts

    def measure(self, m: int, nshots: int, var: str = "a", prt: bool = False) -> int:
        if nshots <= 0:
            raise ValueError(f"nshots must be positive, received {nshots}.")

        control = max(0, int(round(float(m))))
        circuit = self._build_measured_circuit(control)
        counts = _normalize_counts(self._counts_provider(circuit, int(nshots)))
        one_counts = self._count_good_states(counts)

        self.execution_log.append(
            _MeasurementRecord(control=control, shots=int(nshots), one_counts=int(one_counts), var=var)
        )

        if prt:
            p1_hat = one_counts / float(nshots)
            print(f"> Measured control={control}, p1_hat={p1_hat:.6f}. [HardwareQAEmodel.measure]")

        return int(one_counts)


class StandaloneBAEHardware:
    """Hardware-ready adapter around the original BAE implementation."""

    def __init__(
        self,
        epsilon_target: float,
        alpha: float,
        sampler: Any,
        noise_model: str = "ideal",
        T_known: float | None = None,
        cap_kappa: float = 1.0,
        max_shots_same_k: int | None = None,
        counts_adapter: CountsProvider | None = None,
        **kwargs: Any,
    ) -> None:
        self.epsilon_target = float(epsilon_target)
        self.alpha = float(alpha)
        self._sampler = sampler
        self._counts_adapter = counts_adapter
        self.noise_model = noise_model
        self.T_known = T_known
        self.cap_kappa = float(cap_kappa)
        self.max_shots_same_k = max_shots_same_k

        default_estimate_T = bool(
            self.T_known is not None
            and not np.isinf(self.T_known)
            and noise_model != "ideal"
        )
        self.estimate_T = bool(kwargs.get("estimate_T", default_estimate_T))

        self.T_range = kwargs.get("T_range")

        # Legacy BAE naming: sampler_kind refers to internal SMC sampler,
        # not to the external quantum sampler.
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
            "Npart": int(kwargs.get("Npart", 2000)),
            "thr": float(kwargs.get("thr", 0.5)),
            "var": kwargs.get("var", "theta"),
            "ut": kwargs.get("ut", "var"),
            "c": float(kwargs.get("c", 2.38)),
            "log": bool(kwargs.get("log", True)),
            "res_ut": bool(kwargs.get("res_ut", False)),
            "plot": bool(kwargs.get("plot", False)),
        }
        self.sampler_kwargs.update(kwargs.get("sampler_kwargs", {}))

    def _resolve_T_range(self) -> tuple[float, float] | None:
        if self.T_range is not None:
            return tuple(self.T_range)

        if self.T_known is None or np.isinf(self.T_known):
            return None

        tval = float(self.T_known)
        low = max(1.0, 0.5 * tval)
        high = max(low + 1.0, 1.5 * tval)
        return (low, high)

    def _build_confidence_interval(self, mean: float, std: float) -> tuple[float, float]:
        z_value = NormalDist().inv_cdf(1.0 - self.alpha / 2.0)
        low = max(0.0, float(mean) - z_value * float(std))
        high = min(1.0, float(mean) + z_value * float(std))
        return (low, high)

    def _resolve_counts_provider(
        self,
        sampler: Any | None,
        counts_adapter: CountsProvider | None,
    ) -> CountsProvider:
        if counts_adapter is not None:
            return counts_adapter
        if self._counts_adapter is not None and sampler is None:
            return self._counts_adapter

        source = sampler if sampler is not None else self._sampler

        if source is None:
            warnings.warn(
                "No sampler/counts adapter provided; defaulting to local Aer SamplerV2."
            )
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime import SamplerV2

            source = SamplerV2(mode=AerSimulator())

        return SamplerCountsAdapter(source)

    @staticmethod
    def _extract_control_history(
        nqs: list[int],
        estimator: LegacyBAE,
        model: HardwareQAEmodel,
    ) -> tuple[list[int], list[int], list[int]]:
        # Only amplitude/phase measurements contribute to query trajectory.
        records = [r for r in model.execution_log if r.var != "Tc"]

        controls = [int(r.control) for r in records]
        shot_sequence = [int(r.shots) for r in records]
        one_counts = [int(r.one_counts) for r in records]

        if len(controls) != len(nqs):
            adaptive_ctrls = [int(round(float(k))) for k in getattr(estimator, "ctrls_list", [])]
            if len(adaptive_ctrls) + 1 == len(nqs):
                controls = [0] + adaptive_ctrls
            elif len(adaptive_ctrls) == len(nqs):
                controls = adaptive_ctrls

            if len(shot_sequence) != len(nqs):
                increments = np.diff(np.asarray([0, *nqs], dtype=float)).astype(int)
                inferred_shots = []
                for idx, delta in enumerate(increments.tolist()):
                    kval = max(1, 2 * controls[idx] + 1) if idx < len(controls) else 1
                    inferred_shots.append(max(1, int(round(delta / kval))))
                shot_sequence = inferred_shots

            if len(one_counts) != len(nqs):
                one_counts = [0 for _ in nqs]

        return controls, shot_sequence, one_counts

    def estimate(
        self,
        problem: Any,
        bayes: Any = None,
        show_details: bool = False,
        n_shots: int = 20,
        max_queries: int = 200000,
        sampler: Any | None = None,
        counts_adapter: CountsProvider | None = None,
    ) -> BAEResult:
        del bayes  # preserved for API compatibility
        result = BAEResult()

        counts_provider = self._resolve_counts_provider(sampler=sampler, counts_adapter=counts_adapter)

        finite_T = self.T_known is not None and not np.isinf(self.T_known)
        T_range = self._resolve_T_range()

        # Legacy BAE expects:
        # - False: ideal/no-noise model
        # - True: estimate Tc online
        # - float: use provided Tc directly
        if finite_T and self.estimate_T:
            Tc_precalc: bool | float = True
            TNs = max(1, int(self.strategy.get("TNs", 500)))
        elif finite_T:
            Tc_precalc = float(self.T_known)
            TNs = 0
        else:
            Tc_precalc = False
            TNs = 0

        strategy = dict(self.strategy)
        strategy["Ns"] = max(1, int(n_shots))
        strategy["TNs"] = TNs
        strategy["capk"] = float(max(strategy.get("capk", 1.0), self.cap_kappa))

        def _execute_inference() -> tuple[list[float], list[float], list[int], LegacyBAE, HardwareQAEmodel]:
            model = HardwareQAEmodel(problem=problem, counts_provider=counts_provider, Tcrange=T_range)
            estimator = LegacyBAE(model, Tc_precalc=Tc_precalc, Tcrange=T_range)
            smc_sampler = get_sampler(self.sampler_kind, model, dict(self.sampler_kwargs))

            run_args = {
                "sampler": smc_sampler,
                "strat": strategy,
                "maxPT": int(max_queries),
                "print_evol": False,
                "plot_all": False,
            }
            means, stds, nqs = estimator.adapt_inference(**run_args)
            return means, stds, nqs, estimator, model

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
            if show_details:
                means, stds, nqs, estimator, model = _execute_inference()
            else:
                with redirect_stdout(io.StringIO()):
                    means, stds, nqs, estimator, model = _execute_inference()

        means = [float(mval) for mval in means]
        stds = [float(sval) for sval in stds]
        nqs = [int(qval) for qval in nqs]

        if means:
            estimation = float(means[-1])
            std = float(stds[-1]) if stds else 0.0
        else:
            estimation = 0.5
            std = 0.0

        controls, shot_sequence, one_counts = self._extract_control_history(nqs, estimator, model)
        k_sequence = [int(2 * ctrl + 1) for ctrl in controls]

        result.estimation = estimation
        result.confidence_interval = self._build_confidence_interval(estimation, std)
        result.num_oracle_queries = int(nqs[-1]) if nqs else 0
        result.num_state_prep_calls = result.num_oracle_queries

        result.powers = controls
        result.circuit_depths = k_sequence
        result.K_sequence = k_sequence
        result.K_max = int(max(k_sequence)) if k_sequence else 0

        result.history = {
            "queries": nqs,
            "estimations": means,
            "stds": stds,
            "controls": controls,
            "one_counts": one_counts,
            "shots": shot_sequence,
            "K_sequence": k_sequence,
            "circuit_depths": k_sequence,
        }

        return result
