from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from quantum_cva.amplitude_estimation.experiments.circuits import (
    circuit_k,
    stable_circuit_key,
)
from quantum_cva.amplitude_estimation.experiments.problems import (
    AEProblemBundle,
    count_good_states,
    normalize_bitstring,
)


class _CountsRegister:
    def __init__(self, counts: Mapping[str, int]):
        self._counts = {str(k): int(v) for k, v in counts.items()}

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _PubData:
    def __init__(self, counts: Mapping[str, int]):
        self.c0 = _CountsRegister(counts)


class _PubResult:
    def __init__(self, counts: Mapping[str, int]):
        self.data = _PubData(counts)


class _SamplerJob:
    def __init__(self, pub_results: list[_PubResult], job_id: str | None = None):
        self._pub_results = pub_results
        self._job_id = str(job_id or uuid.uuid4())

    def job_id(self) -> str:
        return self._job_id

    def result(self) -> list[_PubResult]:
        return self._pub_results


def normalize_counts(counts: Mapping[Any, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in counts.items():
        out[normalize_bitstring(key)] = int(value)
    return out


def extract_result_counts(result_payload: Any, index: int) -> dict[str, int]:
    item = result_payload[index]
    data = getattr(item, "data", None)
    if data is not None:
        for attr in ("c0", "c"):
            register = getattr(data, attr, None)
            if register is not None and hasattr(register, "get_counts"):
                return normalize_counts(register.get_counts())
        for attr in dir(data):
            if attr.startswith("_"):
                continue
            register = getattr(data, attr)
            if hasattr(register, "get_counts"):
                return normalize_counts(register.get_counts())
    if hasattr(item, "get_counts"):
        return normalize_counts(item.get_counts())
    raise TypeError(f"Cannot extract counts from sampler result item {index}.")


def count_good_from_counts(counts: Mapping[str, int], bundle: AEProblemBundle) -> int:
    return count_good_states(
        counts,
        problem=bundle.problem,
        good_bitstring=bundle.good_bitstring,
        width=bundle.objective_width,
    )


def _remove_measurements_and_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    clean = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"measure", "barrier"}:
            continue
        q_indices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        c_indices = [circuit.find_bit(clbit).index for clbit in instruction.clbits]
        clean.append(operation, q_indices, c_indices)
    return clean


def ideal_good_probability_for_circuit(
    circuit: QuantumCircuit,
    bundle: AEProblemBundle,
) -> float:
    clean = _remove_measurements_and_barriers(circuit)
    state = Statevector.from_instruction(clean)
    probabilities = state.probabilities_dict(qargs=list(bundle.problem.objective_qubits))
    total = 0.0
    for bitstring, probability in probabilities.items():
        state_bits = normalize_bitstring(bitstring, bundle.objective_width)
        if bundle.problem.is_good_state(state_bits):
            total += float(probability)
    return float(np.clip(total, 0.0, 1.0))


def infer_grover_power(circuit: QuantumCircuit) -> int:
    k = circuit_k(circuit)
    if k is not None:
        return int(k)
    count_ops = circuit.count_ops()
    if "Q" in count_ops:
        return int(count_ops.get("Q", 0))
    return max(0, int(count_ops.get("ry", 0)) - 1)


def contrast_decay(k: int, T: float | None) -> float:
    if T is None or np.isinf(T):
        return 1.0
    return float(np.exp(-(2.0 * float(k) + 1.0) / float(T)))


def apply_contrast_decay(
    p_ideal: float,
    k: int,
    T: float | None,
    *,
    baseline: float = 0.5,
) -> float:
    contrast = contrast_decay(k, T)
    floor = float(baseline)
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError(f"baseline must be a finite probability in [0, 1], got {baseline}.")
    return float(np.clip(floor + contrast * (float(p_ideal) - floor), 0.0, 1.0))


def _bad_bitstring(good_bitstring: str) -> str:
    width = len(good_bitstring)
    candidate = "0" * width
    if candidate != good_bitstring:
        return candidate
    return "1" * width


def counts_from_good_probability(
    probability: float,
    shots: int,
    *,
    good_bitstring: str,
    rng: np.random.Generator,
) -> dict[str, int]:
    one = int(rng.binomial(int(shots), float(np.clip(probability, 0.0, 1.0))))
    bad = int(shots) - one
    return {
        str(good_bitstring): int(one),
        _bad_bitstring(str(good_bitstring)): int(bad),
    }


def ideal_amplified_good_probability(
    true_amplitude: float,
    k: int,
) -> float:
    """Ideal amplitude-amplification probability for Grover power ``k``.

    This avoids statevector simulation of ``A Q^k`` while preserving the ideal
    AE probability model. It is valid for the noiseless canonical Grover
    operator built from the same ``EstimationProblem`` amplitude.
    """
    amplitude = float(np.clip(true_amplitude, 0.0, 1.0))
    theta = float(np.arcsin(np.sqrt(amplitude)))
    probability = np.sin((2.0 * int(k) + 1.0) * theta) ** 2
    return float(np.clip(probability, 0.0, 1.0))


class FastIdealAmplificationSampler:
    """SamplerV2-compatible ideal sampler using the closed-form AE law."""

    def __init__(
        self,
        bundle: AEProblemBundle,
        T: float | None = None,
        seed: int | None = None,
        contrast_baseline: float = 0.5,
    ) -> None:
        self.bundle = bundle
        self._T = T
        self._rng = np.random.default_rng(seed)
        self._contrast_baseline = float(contrast_baseline)

    def probability_for_grover_power(self, k: int) -> float:
        p_ideal = ideal_amplified_good_probability(
            self.bundle.true_amplitude,
            int(k),
        )
        return apply_contrast_decay(
            p_ideal,
            int(k),
            self._T,
            baseline=self._contrast_baseline,
        )

    def counts_for_grover_power(self, k: int, shots: int) -> dict[str, int]:
        return counts_from_good_probability(
            self.probability_for_grover_power(int(k)),
            int(shots),
            good_bitstring=str(self.bundle.good_bitstring),
            rng=self._rng,
        )

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _SamplerJob:
        pub_results: list[_PubResult] = []
        for circuit in circuits:
            k = infer_grover_power(circuit)
            counts = self.counts_for_grover_power(k, int(shots))
            pub_results.append(_PubResult(counts))
        return _SamplerJob(pub_results, "fast-ideal-amplification")


class ContrastDecaySampler:
    """SamplerV2-compatible synthetic sampler with exponential contrast decay."""

    def __init__(
        self,
        bundle: AEProblemBundle,
        T: float | None = None,
        seed: int | None = None,
        contrast_baseline: float = 0.5,
    ) -> None:
        self.bundle = bundle
        self._T = T
        self._rng = np.random.default_rng(seed)
        self._contrast_baseline = float(contrast_baseline)

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _SamplerJob:
        pub_results: list[_PubResult] = []
        for circuit in circuits:
            k = infer_grover_power(circuit)
            p_ideal = ideal_good_probability_for_circuit(circuit, self.bundle)
            p_obs = apply_contrast_decay(
                p_ideal,
                k,
                self._T,
                baseline=self._contrast_baseline,
            )
            counts = counts_from_good_probability(
                p_obs,
                int(shots),
                good_bitstring=str(self.bundle.good_bitstring),
                rng=self._rng,
            )
            pub_results.append(_PubResult(counts))
        return _SamplerJob(pub_results, "contrast-decay")


class AerCountSampler:
    """Minimal SamplerV2-compatible counts wrapper around ``AerSimulator``."""

    def __init__(
        self,
        *,
        noise_model: Any | None = None,
        seed: int | None = None,
        method: str = "density_matrix",
        transpile_backend: object | None = None,
        initial_layout: list[int] | None = None,
        pass_manager: Any | None = None,
    ) -> None:
        from qiskit_aer import AerSimulator

        self._rng = np.random.default_rng(seed)
        self._sim = AerSimulator(noise_model=noise_model, method=method)
        self._transpile_backend = transpile_backend
        self._initial_layout = list(initial_layout) if initial_layout is not None else None
        self._pass_manager = pass_manager
        self._cache: dict[str, QuantumCircuit] = {}

    def _transpiled(self, circuit: QuantumCircuit) -> QuantumCircuit:
        key = stable_circuit_key(circuit)
        if key not in self._cache:
            if self._pass_manager is not None:
                self._cache[key] = self._pass_manager.run(circuit.decompose(reps=10))
            else:
                kwargs: dict[str, Any] = {
                    "backend": self._transpile_backend or self._sim,
                    "optimization_level": 3,
                    "seed_transpiler": 1234,
                }
                if self._initial_layout is not None:
                    kwargs["initial_layout"] = list(self._initial_layout)
                self._cache[key] = transpile(circuit.decompose(reps=10), **kwargs)
        return self._cache[key]

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _SamplerJob:
        pub_results: list[_PubResult] = []
        for circuit in circuits:
            seed_simulator = int(self._rng.integers(1, 2**31 - 1))
            result = self._sim.run(
                self._transpiled(circuit),
                shots=int(shots),
                seed_simulator=seed_simulator,
            ).result()
            counts = result.get_counts()
            pub_results.append(_PubResult(normalize_counts(counts)))
        return _SamplerJob(pub_results, "aer-counts")


class LoggedAerSampler:
    def __init__(
        self,
        sampler: AerCountSampler,
        job_rows: list[dict[str, Any]],
        *,
        max_grover_power: int | None = None,
    ) -> None:
        self.sampler = sampler
        self.job_rows = job_rows
        self.max_grover_power = max_grover_power
        self.context = "unknown"
        self._call_index: dict[str, int] = {}

    def set_context(self, context: str) -> None:
        self.context = str(context)

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _SamplerJob:
        self._check_ks(circuits)
        job_id = f"local-{uuid.uuid4()}"
        idx = self._call_index.get(self.context, 0)
        self._call_index[self.context] = idx + 1
        self.job_rows.append(
            {
                "backend_mode": "dry_run_aer",
                "context": self.context,
                "sampler_call_index": idx,
                "n_circuits": len(circuits),
                "shots": int(shots),
                "job_id": job_id,
                "submitted_at_epoch": time.time(),
            }
        )
        job = self.sampler.run(circuits, shots=int(shots))
        return _SamplerJob(job.result(), job_id)

    def _check_ks(self, circuits: list[QuantumCircuit]) -> None:
        if self.max_grover_power is None:
            return
        for circuit in circuits:
            k = circuit_k(circuit)
            if k is not None and int(k) > int(self.max_grover_power):
                raise RuntimeError(
                    f"Refusing circuit with grover_power={k}; cap is {self.max_grover_power}."
                )


class RuntimeCountSampler:
    """Runtime Sampler wrapper with layout cache and metadata logging."""

    def __init__(
        self,
        backend: Any,
        sampler: Any,
        pass_manager: Any,
        job_rows: list[dict[str, Any]],
        *,
        soft_wallclock_limit_seconds: float,
        max_grover_power: int | None = None,
        max_calls_by_context: Mapping[str, int] | None = None,
        start_time: float | None = None,
    ) -> None:
        self.backend = backend
        self.sampler = sampler
        self.pass_manager = pass_manager
        self.job_rows = job_rows
        self.soft_wallclock_limit_seconds = float(soft_wallclock_limit_seconds)
        self.max_grover_power = max_grover_power
        self.max_calls_by_context = dict(max_calls_by_context or {})
        self.start_time = time.perf_counter() if start_time is None else float(start_time)
        self.context = "unknown"
        self._cache: dict[str, QuantumCircuit] = {}
        self._call_index: dict[str, int] = {}

    def set_context(self, context: str) -> None:
        self.context = str(context)

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> Any:
        self._check_budget()
        self._check_ks(circuits)
        isa_circuits = [self._isa(circuit) for circuit in circuits]
        job = self.sampler.run(isa_circuits, shots=int(shots))
        idx = self._call_index.get(self.context, 0)
        self._call_index[self.context] = idx + 1
        self.job_rows.append(
            {
                "backend_mode": "runtime",
                "context": self.context,
                "sampler_call_index": idx,
                "n_circuits": len(isa_circuits),
                "shots": int(shots),
                "job_id": str(job.job_id()),
                "submitted_at_epoch": time.time(),
            }
        )
        return job

    def _check_budget(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        if elapsed > self.soft_wallclock_limit_seconds:
            raise TimeoutError(
                f"Soft wall-clock limit exceeded: {elapsed:.1f}s > "
                f"{self.soft_wallclock_limit_seconds:.1f}s."
            )
        limit = self.max_calls_by_context.get(self.context)
        used = self._call_index.get(self.context, 0)
        if limit is not None and used >= int(limit):
            raise TimeoutError(f"Sampler call cap reached for {self.context}: {used} >= {limit}.")

    def _check_ks(self, circuits: list[QuantumCircuit]) -> None:
        if self.max_grover_power is None:
            return
        for circuit in circuits:
            k = circuit_k(circuit)
            if k is not None and int(k) > int(self.max_grover_power):
                raise RuntimeError(
                    f"Refusing circuit with grover_power={k}; cap is {self.max_grover_power}."
                )

    def _isa(self, circuit: QuantumCircuit) -> QuantumCircuit:
        decomposed = circuit.decompose(reps=10)
        key = stable_circuit_key(decomposed)
        if key not in self._cache:
            self._cache[key] = self.pass_manager.run(decomposed)
        return self._cache[key]


class ReplayCountSampler:
    """Sampler fed by premeasured or extrapolated good-state probabilities."""

    def __init__(
        self,
        p_by_k: Mapping[int, float],
        bundle: AEProblemBundle,
        *,
        seed: int,
        max_calls: int = 128,
        extrapolate_probability: Callable[[int], float] | None = None,
        extrapolated_cache: dict[int, float] | None = None,
    ) -> None:
        self.p_by_k = {int(k): float(v) for k, v in p_by_k.items()}
        self.bundle = bundle
        self.rng = np.random.default_rng(int(seed))
        self.max_calls = int(max_calls)
        self.context = "replay"
        self.calls = 0
        self.extrapolate_probability = extrapolate_probability
        self.extrapolated_cache = extrapolated_cache if extrapolated_cache is not None else {}
        self.extrapolated_ks_used: set[int] = set()

    def set_context(self, context: str) -> None:
        self.context = str(context)

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _SamplerJob:
        if self.calls >= self.max_calls:
            raise TimeoutError(f"Replay sampler call cap reached: {self.calls} >= {self.max_calls}.")
        pub_results: list[_PubResult] = []
        for circuit in circuits:
            k = circuit_k(circuit)
            if k is None:
                raise RuntimeError("Replay sampler requires circuit.metadata['grover_power'].")
            if int(k) not in self.p_by_k:
                if self.extrapolate_probability is None:
                    available = ", ".join(str(x) for x in sorted(self.p_by_k))
                    raise KeyError(
                        f"Replay requested k={k}, but no probability exists. "
                        f"Available k values: [{available}]."
                    )
                if int(k) not in self.extrapolated_cache:
                    self.extrapolated_cache[int(k)] = float(self.extrapolate_probability(int(k)))
                p_good = float(self.extrapolated_cache[int(k)])
                self.extrapolated_ks_used.add(int(k))
            else:
                p_good = float(self.p_by_k[int(k)])
            counts = counts_from_good_probability(
                p_good,
                int(shots),
                good_bitstring=str(self.bundle.good_bitstring),
                rng=self.rng,
            )
            pub_results.append(_PubResult(counts))
        self.calls += 1
        return _SamplerJob(pub_results, f"replay-{uuid.uuid4()}")


def build_noise_model(scale: float, *, profile: str = "projected") -> Any:
    """Build the projected/baseline toy-style Aer noise model."""
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error

    noise_model = NoiseModel()
    profile = str(profile)
    if profile == "baseline":
        p1 = min(1.88e-4 * scale, 5e-3)
        p2 = min(1.93e-3 * scale, 5e-2)
        p_10 = min(4.83e-3 * scale, 0.2)
        p_01 = min(4.30e-3 * scale, 0.2)
        t1 = 279_400.0 / scale
        t2 = 220_400.0 / scale
        t_id = 5.0
        t_sx = 32.0
        t_x = 32.0
        t_cx = 132.0
    elif profile in {"projected", "mild", "realistic"}:
        p1 = min(9.00e-5 * scale, 5e-3)
        p2 = min(8.00e-4 * scale, 5e-2)
        p_10 = min(1.20e-3 * scale, 0.2)
        p_01 = min(1.00e-3 * scale, 0.2)
        t1 = 600_000.0 / scale
        t2 = 550_000.0 / scale
        t_id = 5.0
        t_sx = 5.0
        t_x = 20.0
        t_cx = 50.0
    else:
        raise ValueError(f"Unknown noise profile: {profile}")

    err_id = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, t_id))
    err_sx = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, t_sx))
    err_x = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, t_x))
    err_cx = depolarizing_error(p2, 2).compose(
        thermal_relaxation_error(t1, t2, t_cx).tensor(
            thermal_relaxation_error(t1, t2, t_cx)
        )
    )
    readout = ReadoutError([[1.0 - p_10, p_10], [p_01, 1.0 - p_01]])
    noise_model.add_all_qubit_quantum_error(err_id, ["id"])
    noise_model.add_all_qubit_quantum_error(err_sx, ["sx"])
    noise_model.add_all_qubit_quantum_error(err_x, ["x"])
    noise_model.add_all_qubit_quantum_error(err_cx, ["cx", "cz", "ecr"])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model
