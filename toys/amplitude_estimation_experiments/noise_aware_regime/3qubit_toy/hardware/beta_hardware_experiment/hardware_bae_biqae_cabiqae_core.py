from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator


BETA_DIR = Path(__file__).resolve().parent
HARDWARE_DIR = BETA_DIR.parent
TOY_DIR = HARDWARE_DIR.parent
REPO_ROOT = next(parent for parent in BETA_DIR.parents if (parent / "pyproject.toml").exists())
SRC_DIR = REPO_ROOT / "src"

for path in (SRC_DIR, TOY_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from ae_pipeline_utils import (  # noqa: E402
    AerCountSampler,
    BAE_KIND,
    build_ae_pass_manager,
    build_large_problem,
    build_noise_model,
    circuit_cache_key,
    construct_measured_circuit,
    extract_trace,
    ideal_good_probability,
)
from toys.amplitude_estimation_experiments.common_utils.plotting_utils import (  # noqa: E402
    log_query_bin_indices,
    standard_error,
)
from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta  # noqa: E402
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE  # noqa: E402

try:  # noqa: E402
    from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
        StandaloneBAEHardware,
    )
except Exception:  # pragma: no cover - kept for environments without hardware adapter
    StandaloneBAEHardware = None


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
    def __init__(self, pub_results: list[_PubResult], job_id: str):
        self._pub_results = pub_results
        self._job_id = str(job_id)

    def job_id(self) -> str:
        return self._job_id

    def result(self) -> list[_PubResult]:
        return self._pub_results


def parse_int_list(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return [int(x) for x in raw]


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(dict(obj)), f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    fieldnames: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    fieldnames = list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_algorithms(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(str(x) for x in args.algorithms)


def get_algorithm_labels(args: argparse.Namespace) -> dict[str, str]:
    return {str(k): str(v) for k, v in vars(args)["algorithm_labels"].items()}


def normalize_counts(counts: Mapping[Any, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in counts.items():
        out[str(key).replace(" ", "")] = int(value)
    return out


def count_ones(counts: Mapping[str, int]) -> int:
    total = 0
    for state, value in counts.items():
        normalized = str(state).replace(" ", "")
        if normalized.startswith("0x"):
            normalized = format(int(normalized, 16), "b")
        if normalized.endswith("1") or normalized == "1":
            total += int(value)
    return total


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


def two_qubit_count(circuit: QuantumCircuit) -> int:
    return int(sum(1 for instruction in circuit.data if len(instruction.qubits) == 2))


def circuit_k(circuit: QuantumCircuit) -> int | None:
    metadata = getattr(circuit, "metadata", None) or {}
    for key in ("grover_power", "bae_control"):
        if key in metadata:
            return int(metadata[key])
    return None


def set_circuit_metadata(circuit: QuantumCircuit, k: int, source: str) -> QuantumCircuit:
    metadata = dict(getattr(circuit, "metadata", None) or {})
    metadata.update(
        {
            "source": source,
            "grover_power": int(k),
            "amplification_factor": int(2 * int(k) + 1),
        }
    )
    circuit.metadata = metadata
    return circuit


def build_tagged_measured_circuit(problem: Any, k: int, source: str) -> QuantumCircuit:
    return set_circuit_metadata(construct_measured_circuit(problem, int(k)), int(k), source)


def build_unmeasured_circuit(problem: Any, k: int, source: str) -> QuantumCircuit:
    num_qubits = max(problem.state_preparation.num_qubits, problem.grover_operator.num_qubits)
    circuit = QuantumCircuit(num_qubits, name=f"AE_k_{int(k)}")
    circuit.compose(problem.state_preparation, inplace=True)
    if int(k) > 0:
        grover_power = problem.grover_operator.power(int(k))
        if hasattr(grover_power, "decompose"):
            grover_power = grover_power.decompose(reps=10)
        circuit.compose(grover_power, inplace=True)
    return set_circuit_metadata(circuit, int(k), source)


def patch_construct_circuit(solver: Any, source: str) -> None:
    def _construct(self: Any, estimation_problem: Any, k: int = 0, measurement: bool = False):
        if measurement:
            return build_tagged_measured_circuit(estimation_problem, int(k), source)
        return build_unmeasured_circuit(estimation_problem, int(k), source)

    solver.construct_circuit = types.MethodType(_construct, solver)


def disable_cabiqae_hard_k_cap(solver: Any) -> None:
    def _uncapped(self: Any) -> int:
        return 10**100

    solver._k_cap = types.MethodType(_uncapped, solver)


@dataclass
class RunPaths:
    run_dir: Path

    @property
    def config(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def backend_snapshot(self) -> Path:
        return self.run_dir / "backend_snapshot.json"

    @property
    def session_details(self) -> Path:
        return self.run_dir / "session_details.json"

    @property
    def transpilation_report(self) -> Path:
        return self.run_dir / "transpilation_report.csv"

    @property
    def amplification_counts(self) -> Path:
        return self.run_dir / "amplification_counts.csv"

    @property
    def amplification_points(self) -> Path:
        return self.run_dir / "amplification_points.csv"

    @property
    def readout_calibration(self) -> Path:
        return self.run_dir / "readout_calibration.csv"

    @property
    def calibration_summary(self) -> Path:
        return self.run_dir / "calibration_summary.json"

    @property
    def direct_trace(self) -> Path:
        return self.run_dir / "direct_trace_rows.csv"

    @property
    def direct_final(self) -> Path:
        return self.run_dir / "direct_final_rows.csv"

    @property
    def replay_trace(self) -> Path:
        return self.run_dir / "replay_trace_rows.csv"

    @property
    def replay_final(self) -> Path:
        return self.run_dir / "replay_final_rows.csv"

    @property
    def budget_summary(self) -> Path:
        return self.run_dir / "budget_summary.csv"

    @property
    def runtime_jobs(self) -> Path:
        return self.run_dir / "runtime_jobs.csv"

    @property
    def errors(self) -> Path:
        return self.run_dir / "errors.csv"

    @property
    def trace_bundle(self) -> Path:
        return self.run_dir / "trace_bundle.npz"

    @property
    def qasm_dir(self) -> Path:
        return self.run_dir / "qasm3_isa"


@dataclass
class ExperimentState:
    paths: RunPaths
    config: dict[str, Any]
    job_rows: list[dict[str, Any]] = field(default_factory=list)
    error_rows: list[dict[str, Any]] = field(default_factory=list)
    readout_rows: list[dict[str, Any]] = field(default_factory=list)
    amplification_count_rows: list[dict[str, Any]] = field(default_factory=list)
    amplification_point_rows: list[dict[str, Any]] = field(default_factory=list)
    direct_trace_rows: list[dict[str, Any]] = field(default_factory=list)
    direct_final_rows: list[dict[str, Any]] = field(default_factory=list)
    replay_trace_rows: list[dict[str, Any]] = field(default_factory=list)
    replay_final_rows: list[dict[str, Any]] = field(default_factory=list)
    budget_summary_rows: list[dict[str, Any]] = field(default_factory=list)
    calibration_summary: dict[str, Any] = field(default_factory=dict)
    session_details: dict[str, Any] = field(default_factory=dict)

    def persist(self) -> None:
        save_json(self.config, self.paths.config)
        save_json(self.session_details, self.paths.session_details)
        save_json(self.calibration_summary, self.paths.calibration_summary)
        save_csv(self.job_rows, self.paths.runtime_jobs)
        save_csv(self.error_rows, self.paths.errors)
        save_csv(self.readout_rows, self.paths.readout_calibration)
        save_csv(self.amplification_count_rows, self.paths.amplification_counts)
        save_csv(self.amplification_point_rows, self.paths.amplification_points)
        save_csv(self.direct_trace_rows, self.paths.direct_trace)
        save_csv(self.direct_final_rows, self.paths.direct_final)
        save_csv(self.replay_trace_rows, self.paths.replay_trace)
        save_csv(self.replay_final_rows, self.paths.replay_final)
        save_csv(self.budget_summary_rows, self.paths.budget_summary)
        manifest = {
            "run_dir": str(self.paths.run_dir),
            "config_json": str(self.paths.config),
            "backend_snapshot_json": str(self.paths.backend_snapshot),
            "session_details_json": str(self.paths.session_details),
            "transpilation_report_csv": str(self.paths.transpilation_report),
            "readout_calibration_csv": str(self.paths.readout_calibration),
            "amplification_counts_csv": str(self.paths.amplification_counts),
            "amplification_points_csv": str(self.paths.amplification_points),
            "calibration_summary_json": str(self.paths.calibration_summary),
            "direct_trace_rows_csv": str(self.paths.direct_trace),
            "direct_final_rows_csv": str(self.paths.direct_final),
            "replay_trace_rows_csv": str(self.paths.replay_trace),
            "replay_final_rows_csv": str(self.paths.replay_final),
            "budget_summary_csv": str(self.paths.budget_summary),
            "runtime_jobs_csv": str(self.paths.runtime_jobs),
            "errors_csv": str(self.paths.errors),
            "trace_bundle_npz": str(self.paths.trace_bundle),
            "qasm3_isa_dir": str(self.paths.qasm_dir),
        }
        save_json(manifest, self.paths.manifest)
        write_trace_bundle(self)


def write_trace_bundle(state: ExperimentState) -> None:
    payload: dict[str, np.ndarray] = {}
    if state.amplification_point_rows:
        payload["amplification_grover_power"] = np.asarray(
            [float(r["grover_power"]) for r in state.amplification_point_rows],
            dtype=float,
        )
        payload["amplification_p_hw_mitigated"] = np.asarray(
            [float(r["p_hw_mitigated"]) for r in state.amplification_point_rows],
            dtype=float,
        )
        payload["amplification_p_ideal"] = np.asarray(
            [float(r["p_ideal"]) for r in state.amplification_point_rows],
            dtype=float,
        )
    for prefix, rows in (
        ("direct", state.direct_trace_rows),
        ("replay", state.replay_trace_rows),
    ):
        if rows:
            payload[f"{prefix}_query_budget"] = np.asarray(
                [float(r["query_budget"]) for r in rows],
                dtype=float,
            )
            payload[f"{prefix}_estimate"] = np.asarray(
                [float(r["estimate"]) for r in rows],
                dtype=float,
            )
    state.paths.trace_bundle.parent.mkdir(parents=True, exist_ok=True)
    if payload:
        np.savez(state.paths.trace_bundle, **payload)
    else:
        np.savez(state.paths.trace_bundle, empty=np.asarray([], dtype=float))


class LoggedAerSampler:
    def __init__(
        self,
        sampler: AerCountSampler,
        state: ExperimentState,
        *,
        max_grover_power: int | None = None,
    ):
        self.sampler = sampler
        self.state = state
        self.max_grover_power = max_grover_power
        self.context = "unknown"
        self._call_index: dict[str, int] = {}

    def set_context(self, context: str) -> None:
        self.context = str(context)

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> Any:
        self._check_ks(circuits)
        job_id = f"local-{uuid.uuid4()}"
        idx = self._call_index.get(self.context, 0)
        self._call_index[self.context] = idx + 1
        self.state.job_rows.append(
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
    def __init__(
        self,
        backend: Any,
        sampler: Any,
        pass_manager: Any,
        state: ExperimentState,
        *,
        soft_wallclock_limit_seconds: float,
        max_grover_power: int | None = None,
        max_calls_by_context: Mapping[str, int] | None = None,
        start_time: float | None = None,
    ):
        self.backend = backend
        self.sampler = sampler
        self.pass_manager = pass_manager
        self.state = state
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
        self.state.job_rows.append(
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
        key = circuit_cache_key(decomposed)
        if key not in self._cache:
            self._cache[key] = self.pass_manager.run(decomposed)
        return self._cache[key]


class ReplayCountSampler:
    def __init__(
        self,
        p_by_k: Mapping[int, float],
        state: ExperimentState,
        *,
        seed: int,
        max_calls: int = 128,
        extrapolate_probability: Callable[[int], float] | None = None,
        extrapolated_cache: dict[int, float] | None = None,
    ):
        self.p_by_k = {int(k): float(v) for k, v in p_by_k.items()}
        self.state = state
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
            if k not in self.p_by_k:
                if self.extrapolate_probability is None:
                    available = ", ".join(str(x) for x in sorted(self.p_by_k))
                    raise KeyError(
                        f"Replay requested k={k}, but no premeasured hardware probability exists "
                        f"for that Grover power. Available k values: [{available}]."
                    )
                if int(k) not in self.extrapolated_cache:
                    self.extrapolated_cache[int(k)] = float(self.extrapolate_probability(int(k)))
                p_one = float(np.clip(self.extrapolated_cache[int(k)], 0.0, 1.0))
                self.extrapolated_ks_used.add(int(k))
            else:
                p_one = float(np.clip(self.p_by_k[k], 0.0, 1.0))
            one = int(self.rng.binomial(int(shots), p_one))
            pub_results.append(_PubResult({"0": int(shots) - one, "1": one}))
        self.calls += 1
        return _SamplerJob(pub_results, f"replay-{uuid.uuid4()}")


def load_fake_backend(fake_backend: str) -> Any:
    from qiskit_ibm_runtime import fake_provider

    normalized = "".join(part.capitalize() for part in fake_backend.replace("-", "_").split("_"))
    candidates = [normalized]
    if not normalized.startswith("Fake"):
        candidates.append(f"Fake{normalized}")
    for name in candidates:
        cls = getattr(fake_provider, name, None)
        if cls is not None:
            return cls()
    available = sorted(x for x in dir(fake_provider) if x.startswith("Fake"))
    raise ValueError(f"Unknown fake backend {fake_backend!r}. Available examples: {available[:10]}")


def backend_snapshot(backend: Any, *, mode: str, channel: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "mode": mode,
        "channel": channel,
        "backend_name": getattr(backend, "name", None),
        "num_qubits": getattr(backend, "num_qubits", None),
        "backend_version": getattr(backend, "backend_version", None),
        "timestamp_epoch": time.time(),
    }
    try:
        snapshot["basis_gates"] = list(getattr(backend.target, "operation_names", []))
    except Exception:
        snapshot["basis_gates"] = None
    try:
        status = backend.status()
        snapshot["status"] = {
            "operational": getattr(status, "operational", None),
            "pending_jobs": getattr(status, "pending_jobs", None),
            "status_msg": getattr(status, "status_msg", None),
        }
    except Exception as exc:
        snapshot["status_error"] = str(exc)
    try:
        coupling = getattr(backend, "coupling_map", None)
        snapshot["coupling_edges_sample"] = list(coupling.get_edges()[:50]) if coupling else None
    except Exception:
        snapshot["coupling_edges_sample"] = None
    return snapshot


def build_pass_manager_for_mode(
    backend: Any,
    problem: Any,
    *,
    mode: str,
    optimization_level: int,
    seed_transpiler: int,
    reference_ks: Iterable[int],
) -> tuple[Any, dict[str, Any]]:
    if mode == "dry-run" and isinstance(backend, AerSimulator):
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=int(optimization_level),
            seed_transpiler=int(seed_transpiler),
        )
        return pass_manager, {
            "strategy": "aer_preset",
            "fallback_used": False,
            "initial_layout": None,
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "reference_ks": list(reference_ks),
        }
    return build_ae_pass_manager(
        backend,
        problem,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed_transpiler),
        reference_ks=tuple(reference_ks),
    )


def write_isa_qasm(circuit: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = qasm3.dumps(circuit)
    except Exception as exc:
        text = f"// QASM3 export failed: {exc}\n"
    path.write_text(text, encoding="utf-8")


def run_preflight(
    problem: Any,
    pass_manager: Any,
    state: ExperimentState,
    *,
    max_grover_power: int,
    max_isa_depth: int,
    max_isa_2q: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    allowed_max = -1
    contiguous_allowed = True
    state.paths.qasm_dir.mkdir(parents=True, exist_ok=True)
    for k in range(int(max_grover_power) + 1):
        logical = build_tagged_measured_circuit(problem, k, "preflight")
        decomposed = logical.decompose(reps=10)
        isa = pass_manager.run(decomposed)
        ops = {str(name): int(count) for name, count in isa.count_ops().items()}
        isa_2q = two_qubit_count(isa)
        row = {
            "grover_power": int(k),
            "amplification_factor": int(2 * k + 1),
            "p_ideal": ideal_good_probability(problem, k),
            "logical_depth": logical.depth(),
            "logical_2q": two_qubit_count(logical),
            "decomposed_depth": decomposed.depth(),
            "decomposed_2q": two_qubit_count(decomposed),
            "isa_depth": isa.depth(),
            "isa_size": isa.size(),
            "isa_2q": isa_2q,
            "cz": ops.get("cz", 0),
            "ecr": ops.get("ecr", 0),
            "cx": ops.get("cx", 0),
            "rzz": ops.get("rzz", 0),
            "swap": ops.get("swap", 0),
            "within_depth_limit": bool(isa.depth() <= int(max_isa_depth)),
            "within_2q_limit": bool(isa_2q <= int(max_isa_2q)),
        }
        if contiguous_allowed and row["within_depth_limit"] and row["within_2q_limit"]:
            allowed_max = int(k)
        else:
            contiguous_allowed = False
        rows.append(row)
        write_isa_qasm(isa, state.paths.qasm_dir / f"ae_k_{k:02d}_isa.qasm3")
    save_csv(rows, state.paths.transpilation_report)
    if allowed_max < 0:
        raise RuntimeError("No Grover power passed the ISA preflight limits.")
    return rows, allowed_max


def build_readout_circuits(problem: Any) -> list[QuantumCircuit]:
    circuits: list[QuantumCircuit] = []
    num_qubits = max(problem.state_preparation.num_qubits, problem.grover_operator.num_qubits)
    objective = list(problem.objective_qubits)
    for prepared_state in (0, 1):
        circuit = QuantumCircuit(num_qubits, name=f"readout_prepare_{prepared_state}")
        if prepared_state == 1:
            for q in objective:
                circuit.x(int(q))
        classical = ClassicalRegister(len(objective), "c0")
        circuit.add_register(classical)
        circuit.measure(objective, classical[:])
        circuit.metadata = {
            "source": "readout_calibration",
            "prepared_state": int(prepared_state),
            "grover_power": 0,
            "amplification_factor": 1,
        }
        circuits.append(circuit)
    return circuits


def run_readout_calibration(
    sampler: Any,
    problem: Any,
    *,
    shots: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    sampler.set_context("readout_calibration")
    circuits = build_readout_circuits(problem)
    result = sampler.run(circuits, shots=int(shots)).result()
    rows: list[dict[str, Any]] = []
    p_obs: dict[int, float] = {}
    for idx, prepared_state in enumerate((0, 1)):
        counts = extract_result_counts(result, idx)
        one = count_ones(counts)
        total = int(sum(counts.values()))
        p1 = one / max(total, 1)
        p_obs[int(prepared_state)] = float(p1)
        rows.append(
            {
                "prepared_state": int(prepared_state),
                "shots": total,
                "counts_json": json.dumps(counts, sort_keys=True),
                "one_counts": one,
                "p_observed_1": p1,
            }
        )
    denom = p_obs[1] - p_obs[0]
    params = {
        "p_obs_1_given_0": float(p_obs[0]),
        "p_obs_1_given_1": float(p_obs[1]),
        "readout_denom": float(denom),
        "readout_usable": float(abs(denom) > 0.05),
    }
    return rows, params


def mitigate_readout_probability(p_raw: float, readout: Mapping[str, float]) -> float:
    denom = float(readout.get("readout_denom", 1.0))
    p0 = float(readout.get("p_obs_1_given_0", 0.0))
    if abs(denom) <= 0.05:
        return float(np.clip(p_raw, 0.0, 1.0))
    return float(np.clip((float(p_raw) - p0) / denom, 0.0, 1.0))


def run_amplification_scan(
    sampler: Any,
    problem: Any,
    *,
    grover_powers: list[int],
    repeats: int,
    shots: int,
    seed: int,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    schedule = [(int(k), int(r)) for k in grover_powers for r in range(int(repeats))]
    rng.shuffle(schedule)
    rows: list[dict[str, Any]] = []
    sampler.set_context("amplification_scan")
    for batch_index, (k, repeat_index) in enumerate(schedule):
        if verbose:
            print(
                f"[amplification_scan] batch {batch_index + 1}/{len(schedule)} "
                f"k={k} repeat={repeat_index} shots={int(shots)}",
                flush=True,
            )
        circuit = build_tagged_measured_circuit(problem, k, "amplification_scan")
        result = sampler.run([circuit], shots=int(shots)).result()
        counts = extract_result_counts(result, 0)
        one = count_ones(counts)
        total = int(sum(counts.values()))
        if verbose:
            print(
                f"[amplification_scan] k={k} repeat={repeat_index} "
                f"one_counts={one}/{total} p_raw={one / max(total, 1):.6f}",
                flush=True,
            )
        rows.append(
            {
                "batch_index": int(batch_index),
                "repeat_index": int(repeat_index),
                "grover_power": int(k),
                "amplification_factor": int(2 * k + 1),
                "shots": total,
                "one_counts": int(one),
                "zero_counts": int(total - one),
                "p_hw_raw": float(one / max(total, 1)),
                "counts_json": json.dumps(counts, sort_keys=True),
            }
        )
    return rows


def analyze_amplification(
    count_rows: list[dict[str, Any]],
    problem: Any,
    readout: Mapping[str, float],
    *,
    min_ideal_offset: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, float]]:
    by_k: dict[int, list[dict[str, Any]]] = {}
    for row in count_rows:
        by_k.setdefault(int(row["grover_power"]), []).append(row)

    points: list[dict[str, Any]] = []
    fit_x: list[float] = []
    fit_y: list[float] = []
    p_replay_by_k: dict[int, float] = {}
    for k in sorted(by_k):
        rows = by_k[k]
        one = int(sum(int(r["one_counts"]) for r in rows))
        shots = int(sum(int(r["shots"]) for r in rows))
        p_raw = one / max(shots, 1)
        p_mitigated = mitigate_readout_probability(p_raw, readout)
        p_replay_by_k[k] = p_mitigated
        p_ideal = ideal_good_probability(problem, k)
        denom = p_ideal - 0.5
        raw_se = math.sqrt(max(p_raw * (1.0 - p_raw), 0.0) / max(shots, 1))
        readout_denom = abs(float(readout.get("readout_denom", 1.0)))
        mitigated_se = raw_se / max(readout_denom, 0.05)
        contrast = np.nan
        contrast_se = np.nan
        if abs(denom) > 1e-12:
            contrast = (p_mitigated - 0.5) / denom
            contrast_se = mitigated_se / abs(denom)
        used = (
            abs(denom) >= float(min_ideal_offset)
            and np.isfinite(contrast)
            and 0.0 < contrast < 1.0
            and contrast > 2.0 * contrast_se
        )
        if used:
            fit_x.append(float(2 * k + 1))
            fit_y.append(float(np.log(contrast)))
        signal_z = abs(p_mitigated - 0.5) / max(mitigated_se, 1e-12)
        points.append(
            {
                "grover_power": int(k),
                "amplification_factor": int(2 * k + 1),
                "shots": int(shots),
                "p_ideal": float(p_ideal),
                "p_hw_raw": float(p_raw),
                "p_hw_mitigated": float(p_mitigated),
                "p_hw_raw_se": float(raw_se),
                "p_hw_mitigated_se": float(mitigated_se),
                "contrast_mitigated": float(contrast) if np.isfinite(contrast) else np.nan,
                "contrast_mitigated_se": float(contrast_se) if np.isfinite(contrast_se) else np.nan,
                "signal_z_from_half": float(signal_z),
                "used_in_fit": bool(used),
                "fit_exclusion_reason": "" if used else contrast_exclusion_reason(denom, contrast, contrast_se),
            }
        )

    summary: dict[str, Any] = {
        "calibration_status": "insufficient_fit_points",
        "fit_points": len(fit_x),
        "t_eff_zero_intercept": None,
        "t_eff_free_intercept": None,
        "contrast_prefactor": None,
        "free_intercept_slope": None,
        "k_visible": 0,
    }
    visible = [int(p["grover_power"]) for p in points if float(p["signal_z_from_half"]) >= 3.0]
    if visible:
        summary["k_visible"] = int(max(visible))
    if len(fit_x) >= 2:
        x = np.asarray(fit_x, dtype=float)
        y = np.asarray(fit_y, dtype=float)
        slope_zero = float(np.sum(x * y) / np.sum(x * x))
        if slope_zero < 0.0:
            summary["t_eff_zero_intercept"] = float(-1.0 / slope_zero)
            summary["calibration_status"] = "ok"
        slope_free, intercept_free = np.polyfit(x, y, deg=1)
        summary["free_intercept_slope"] = float(slope_free)
        summary["contrast_prefactor"] = float(np.exp(intercept_free))
        if slope_free < 0.0:
            summary["t_eff_free_intercept"] = float(-1.0 / slope_free)
    return points, summary, p_replay_by_k


def contrast_exclusion_reason(denom: float, contrast: float, contrast_se: float) -> str:
    if abs(denom) < 0.15:
        return "ideal_probability_too_close_to_half"
    if not np.isfinite(contrast):
        return "non_finite_contrast"
    if contrast <= 0.0:
        return "negative_or_zero_contrast"
    if contrast >= 1.0:
        return "saturated_contrast"
    if np.isfinite(contrast_se) and contrast <= 2.0 * contrast_se:
        return "contrast_not_distinguishable_from_zero"
    return "unknown"


def effective_t_for_algorithms(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("t_eff_zero_intercept")
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def empirical_contrast_model(summary: Mapping[str, Any]) -> tuple[str, float, float]:
    prefactor = summary.get("contrast_prefactor")
    t_free = summary.get("t_eff_free_intercept")
    if prefactor is not None and t_free is not None:
        prefactor_f = float(prefactor)
        t_free_f = float(t_free)
        if np.isfinite(prefactor_f) and prefactor_f > 0.0 and np.isfinite(t_free_f) and t_free_f > 0.0:
            return "free_intercept", prefactor_f, t_free_f

    t_zero = summary.get("t_eff_zero_intercept")
    if t_zero is not None:
        t_zero_f = float(t_zero)
        if np.isfinite(t_zero_f) and t_zero_f > 0.0:
            return "zero_intercept", 1.0, t_zero_f

    raise ValueError(
        "Cannot extrapolate replay probabilities without a valid empirical contrast model "
        "in calibration_summary.json."
    )


def make_replay_probability_extrapolator(
    *,
    a_true: float,
    problem: Any,
    calibration_summary: Mapping[str, Any],
) -> tuple[Callable[[int], float], dict[str, Any]]:
    model_name, prefactor, t_eff = empirical_contrast_model(calibration_summary)
    a = float(np.clip(float(a_true), 0.0, 1.0))
    theta = float(np.arcsin(np.sqrt(a))) if np.isfinite(a) else np.nan

    def _ideal_probability(k: int) -> float:
        if np.isfinite(theta):
            return float(np.sin((2 * int(k) + 1) * theta) ** 2)
        return float(ideal_good_probability(problem, int(k)))

    def _extrapolate(k: int) -> float:
        amplification_factor = 2 * int(k) + 1
        contrast = float(np.clip(prefactor * np.exp(-float(amplification_factor) / float(t_eff)), 0.0, 1.0))
        p_ideal = _ideal_probability(int(k))
        return float(np.clip(0.5 + contrast * (p_ideal - 0.5), 0.0, 1.0))

    metadata = {
        "model": model_name,
        "contrast_prefactor": float(prefactor),
        "t_eff": float(t_eff),
        "a_true": float(a),
    }
    return _extrapolate, metadata


def build_solver(
    algorithm: str,
    sampler: Any,
    *,
    epsilon_target: float,
    alpha: float,
    n_shots: int,
    t_eff: float | None,
    seed: int,
) -> Any:
    if algorithm == "biqae":
        solver = BIQAE(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            min_ratio=2,
            confint_method="beta",
            max_shots_same_k=None,
        )
        patch_construct_circuit(solver, "biqae")
        return solver

    if algorithm == "cabiqae_latentt":
        solver = CABIQAELatentTheta(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            min_ratio=2,
            confint_method="beta",
            noise_model="ideal" if t_eff is None else "exponential_contrast",
            T_known=None if t_eff is None else float(t_eff),
            cap_kappa=1.0,
            use_noise_cap=True,
            max_shots_same_k=None,
            random_seed=int(seed),
        )
        patch_construct_circuit(solver, "cabiqae_latentt")
        disable_cabiqae_hard_k_cap(solver)
        return solver

    if algorithm == "bae":
        if StandaloneBAEHardware is None:
            raise RuntimeError("StandaloneBAEHardware is not available in this environment.")
        return StandaloneBAEHardware(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            noise_model="exponential_contrast" if t_eff is not None else "ideal",
            T_known=None if t_eff is None else float(t_eff),
            estimate_T=False,
            wNs=max(1, int(n_shots)),
            Ns=max(1, int(n_shots)),
            Npart=300,
            thr=0.4,
            k=1,
            erefs=1,
            ethr=1,
            stoch=False,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def trace_rows_from_result(
    result: Any,
    *,
    algorithm: str,
    algorithm_labels: Mapping[str, str],
    run_kind: str,
    repetition: int,
    a_true: float,
    objective_ry_offset: float,
    n_shots: int,
    elapsed_wall_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries, estimates, amp_factors = extract_trace(algorithm, result, int(n_shots))
    final_queries_for_timing = float(
        getattr(result, "num_state_prep_calls", queries[-1] if len(queries) else np.nan)
    )
    rows: list[dict[str, Any]] = []
    for idx, (query, estimate, amp) in enumerate(zip(queries, estimates, amp_factors)):
        amp_int = int(round(float(amp)))
        k = max(0, (amp_int - 1) // 2)
        prefix_amplification = np.asarray(amp_factors[: idx + 1], dtype=float)
        k_max_budget = max(
            0,
            int((int(round(float(np.nanmax(prefix_amplification)))) - 1) // 2),
        )
        if elapsed_wall_seconds is None or not np.isfinite(final_queries_for_timing) or final_queries_for_timing <= 0:
            runtime_wall_seconds = np.nan
        else:
            runtime_wall_seconds = float(elapsed_wall_seconds) * min(
                max(float(query) / float(final_queries_for_timing), 0.0),
                1.0,
            )
        rows.append(
            {
                "run_kind": run_kind,
                "repetition": int(repetition),
                "algorithm": algorithm_labels.get(algorithm, algorithm),
                "algorithm_key": algorithm,
                "step_index": int(idx),
                "budget": int(round(float(query))),
                "query_budget": float(query),
                "query_budget_actual": float(query),
                "estimate": float(estimate),
                "abs_error": float(abs(float(estimate) - float(a_true))),
                "normalized_abs_error": float(abs(float(estimate) - float(a_true)) / max(float(a_true), 1e-12)),
                "grover_power": int(k),
                "k_max_budget": int(k_max_budget),
                "amplification_factor": int(amp_int),
                "a_true": float(a_true),
                "objective_ry_offset": float(objective_ry_offset),
                "runtime_wall_seconds": runtime_wall_seconds,
                "time_to_budget_seconds": runtime_wall_seconds,
            }
        )

    final_est = float(getattr(result, "estimation", rows[-1]["estimate"] if rows else np.nan))
    ci = getattr(result, "confidence_interval", None)
    coverage = np.nan
    ci_low = np.nan
    ci_high = np.nan
    if ci is not None:
        ci_low = float(ci[0])
        ci_high = float(ci[1])
        coverage = float(ci_low <= float(a_true) <= ci_high)
    final_queries = float(getattr(result, "num_state_prep_calls", queries[-1] if len(queries) else np.nan))
    k_max = int(max([r["grover_power"] for r in rows], default=0))
    final_row = {
        "run_kind": run_kind,
        "repetition": int(repetition),
        "algorithm": algorithm_labels.get(algorithm, algorithm),
        "algorithm_key": algorithm,
        "a_true": float(a_true),
        "objective_ry_offset": float(objective_ry_offset),
        "final_queries": final_queries,
        "final_estimate": final_est,
        "final_abs_error": float(abs(final_est - float(a_true))) if np.isfinite(final_est) else np.nan,
        "final_normalized_abs_error": float(abs(final_est - float(a_true)) / max(float(a_true), 1e-12))
        if np.isfinite(final_est)
        else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "coverage": coverage,
        "k_max": k_max,
        "amplification_factor_max": int(2 * k_max + 1),
        "runtime_wall_seconds": float(elapsed_wall_seconds)
        if elapsed_wall_seconds is not None and np.isfinite(elapsed_wall_seconds)
        else np.nan,
    }
    return rows, final_row


def run_algorithm_once(
    algorithm: str,
    algorithm_labels: Mapping[str, str],
    sampler: Any,
    problem: Any,
    *,
    run_kind: str,
    repetition: int,
    a_true: float,
    objective_ry_offset: float,
    n_shots: int,
    epsilon_target: float,
    alpha: float,
    t_eff: float | None,
    max_queries: int,
    seed: int,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if hasattr(sampler, "set_context"):
        sampler.set_context(f"{run_kind}_{algorithm}")
    np.random.seed(int(seed))
    solver = build_solver(
        algorithm,
        sampler,
        epsilon_target=epsilon_target,
        alpha=alpha,
        n_shots=n_shots,
        t_eff=t_eff,
        seed=seed,
    )
    start = time.perf_counter()
    if algorithm == "bae":
        result = solver.estimate(
            problem,
            n_shots=int(n_shots),
            max_queries=int(max_queries),
            show_details=False,
        )
    else:
        result = solver.estimate(problem, bayes=True, n_shots=int(n_shots), show_details=False)
    elapsed_wall_seconds = time.perf_counter() - start
    return trace_rows_from_result(
        result,
        algorithm=algorithm,
        algorithm_labels=algorithm_labels,
        run_kind=run_kind,
        repetition=repetition,
        a_true=a_true,
        objective_ry_offset=objective_ry_offset,
        n_shots=n_shots,
        elapsed_wall_seconds=elapsed_wall_seconds,
    )


def print_compact_trace(
    trace_rows: list[dict[str, Any]],
    final_row: Mapping[str, Any],
    *,
    prefix: str,
    repetition: int,
    total_repetitions: int,
) -> None:
    rep_width = max(2, len(str(max(1, int(total_repetitions)))))
    alg = str(final_row.get("algorithm_key") or final_row.get("algorithm") or "unknown")
    total_steps = max(1, len(trace_rows))
    for row in trace_rows:
        step = int(row.get("step_index", 0)) + 1
        print(
            f"[{prefix}] rep {int(repetition) + 1:0{rep_width}d}/{int(total_repetitions)} "
            f"alg={alg} iter={step:02d}/{total_steps:02d} "
            f"k={int(row['grover_power'])} amp={int(row['amplification_factor'])} "
            f"queries={float(row['query_budget']):.0f} "
            f"runtime={float(row.get('runtime_wall_seconds', np.nan)):.3f}s "
            f"estimate={float(row['estimate']):.8f} "
            f"abs_error={float(row['abs_error']):.8f} "
            f"normalized_abs_error={float(row['normalized_abs_error']):.6f}",
            flush=True,
        )


def run_direct_live(
    state: ExperimentState,
    sampler: Any,
    problem: Any,
    *,
    algorithms: tuple[str, ...],
    algorithm_labels: Mapping[str, str],
    a_true: float,
    objective_ry_offset: float,
    n_shots: int,
    epsilon_target: float,
    alpha: float,
    t_eff: float | None,
    max_direct_calls: int,
    seed: int,
    verbose: bool = False,
) -> None:
    del max_direct_calls
    max_queries = sys.maxsize
    for offset, algorithm in enumerate(algorithms):
        try:
            trace_rows, final_row = run_algorithm_once(
                algorithm,
                algorithm_labels,
                sampler,
                problem,
                run_kind="direct_live",
                repetition=0,
                a_true=a_true,
                objective_ry_offset=objective_ry_offset,
                n_shots=n_shots,
                epsilon_target=epsilon_target,
                alpha=alpha,
                t_eff=t_eff,
                max_queries=max_queries,
                seed=int(seed) + offset,
                verbose=verbose,
            )
            if verbose:
                print_compact_trace(
                    trace_rows,
                    final_row,
                    prefix="direct_live",
                    repetition=0,
                    total_repetitions=1,
                )
            state.direct_trace_rows.extend(trace_rows)
            state.direct_final_rows.append(final_row)
            state.persist()
        except Exception as exc:
            state.error_rows.append(
                {
                    "phase": "direct_live",
                    "algorithm": algorithm,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "timestamp_epoch": time.time(),
                }
            )
            state.persist()


def load_replay_probabilities_from_counts(path: Path) -> dict[int, float]:
    rows = load_csv(path)
    by_k: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        k = int(row["grover_power"])
        one = int(float(row["one_counts"]))
        shots = int(float(row["shots"]))
        by_k.setdefault(k, []).append((one, shots))
    return {
        k: float(sum(one for one, _ in values) / max(sum(shots for _, shots in values), 1))
        for k, values in by_k.items()
    }


def sample_replay_probabilities(
    p_by_k: Mapping[int, float],
    p_se_by_k: Mapping[int, float] | None,
    *,
    mode: str,
    rng: np.random.Generator,
    se_scale: float,
) -> dict[int, float]:
    mode = str(mode)
    if mode == "fixed":
        return {int(k): float(v) for k, v in p_by_k.items()}
    if mode != "normal":
        raise ValueError(f"Unknown replay probability mode: {mode!r}.")
    if not p_se_by_k:
        raise ValueError("Replay probability mode 'normal' requires p_hw_mitigated_se values.")

    sampled: dict[int, float] = {}
    missing: list[int] = []
    for k, p in p_by_k.items():
        key = int(k)
        if key not in p_se_by_k:
            missing.append(key)
            continue
        sigma = max(0.0, float(se_scale) * float(p_se_by_k[key]))
        sampled[key] = float(np.clip(rng.normal(float(p), sigma), 0.0, 1.0))
    if missing:
        raise ValueError(
            "Replay probability mode 'normal' is missing standard errors for k values: "
            + ", ".join(str(k) for k in sorted(missing))
        )
    return sampled


def run_replay(
    state: ExperimentState,
    problem: Any,
    *,
    algorithms: tuple[str, ...],
    algorithm_labels: Mapping[str, str],
    p_by_k: Mapping[int, float],
    p_se_by_k: Mapping[int, float] | None,
    replay_probability_mode: str,
    replay_probability_se_scale: float,
    a_true: float,
    objective_ry_offset: float,
    budgets: list[int],
    repetitions: int,
    n_shots: int,
    epsilon_target: float,
    alpha: float,
    t_eff: float | None,
    seed: int,
    extrapolate: bool = False,
    calibration_summary: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> None:
    state.replay_trace_rows.clear()
    state.replay_final_rows.clear()
    state.budget_summary_rows.clear()
    state.error_rows = [r for r in state.error_rows if str(r.get("phase")) != "hardware_replay"]
    budget_rows: list[dict[str, Any]] = []
    max_queries = max(int(x) for x in budgets)
    extrapolate_probability: Callable[[int], float] | None = None
    extrapolated_cache: dict[int, float] = {}
    extrapolation_metadata: dict[str, Any] = {}
    if bool(extrapolate):
        extrapolate_probability, extrapolation_metadata = make_replay_probability_extrapolator(
            a_true=float(a_true),
            problem=problem,
            calibration_summary=calibration_summary or state.calibration_summary,
        )
    state.config["replay_extrapolate"] = bool(extrapolate)
    if extrapolation_metadata:
        state.config["replay_extrapolation_model"] = extrapolation_metadata
    for rep in range(int(repetitions)):
        replay_rng = np.random.default_rng(int(seed) + 7919 * rep)
        rep_p_by_k = sample_replay_probabilities(
            p_by_k,
            p_se_by_k,
            mode=replay_probability_mode,
            rng=replay_rng,
            se_scale=replay_probability_se_scale,
        )
        for alg_index, algorithm in enumerate(algorithms):
            sampler = ReplayCountSampler(
                rep_p_by_k,
                state,
                seed=int(seed) + 1009 * rep + 17 * alg_index,
                max_calls=128,
                extrapolate_probability=extrapolate_probability,
                extrapolated_cache=extrapolated_cache,
            )
            try:
                trace_rows, final_row = run_algorithm_once(
                    algorithm,
                    algorithm_labels,
                    sampler,
                    problem,
                    run_kind="hardware_replay",
                    repetition=rep,
                    a_true=a_true,
                    objective_ry_offset=objective_ry_offset,
                    n_shots=n_shots,
                    epsilon_target=epsilon_target,
                    alpha=alpha,
                    t_eff=t_eff,
                    max_queries=max_queries,
                    seed=int(seed) + rep + alg_index,
                    verbose=verbose,
                )
                if bool(extrapolate) and getattr(sampler, "extrapolated_ks_used", None):
                    used_ks = set(int(k) for k in sampler.extrapolated_ks_used)
                    for row in trace_rows:
                        row["replay_probability_source"] = (
                            "extrapolated" if int(row["grover_power"]) in used_ks else "measured"
                        )
                        row["replay_probability_extrapolated"] = int(row["grover_power"]) in used_ks
                    final_row["extrapolated_replay_ks_json"] = json.dumps(sorted(used_ks))
                    final_row["n_extrapolated_replay_ks"] = len(used_ks)
                elif bool(extrapolate):
                    for row in trace_rows:
                        row["replay_probability_source"] = "measured"
                        row["replay_probability_extrapolated"] = False
                    final_row["extrapolated_replay_ks_json"] = "[]"
                    final_row["n_extrapolated_replay_ks"] = 0
                if verbose:
                    print_compact_trace(
                        trace_rows,
                        final_row,
                        prefix="hardware_replay",
                        repetition=rep,
                        total_repetitions=int(repetitions),
                )
                state.replay_trace_rows.extend(trace_rows)
                state.replay_final_rows.append(final_row)
                budget_rows.extend(trace_rows)
            except Exception as exc:
                state.error_rows.append(
                    {
                        "phase": "hardware_replay",
                        "algorithm": algorithm,
                        "repetition": rep,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp_epoch": time.time(),
                    }
                )
        if rep % 10 == 0:
            state.budget_summary_rows = aggregate_budget_summary(
                budget_rows,
                total_repetitions=int(repetitions),
            )
            state.persist()
            if verbose:
                print_budget_runtime_summary(
                    state.budget_summary_rows,
                    prefix=f"hardware_replay runtime summary after rep {rep + 1}/{int(repetitions)}",
                )
    state.budget_summary_rows = aggregate_budget_summary(
        budget_rows,
        total_repetitions=int(repetitions),
    )
    if bool(extrapolate):
        state.config["replay_extrapolated_probabilities"] = {
            str(k): float(v) for k, v in sorted(extrapolated_cache.items())
        }
        state.config["replay_extrapolated_k_values"] = sorted(int(k) for k in extrapolated_cache)
    state.persist()
    print_budget_runtime_summary(
        state.budget_summary_rows,
        prefix="hardware_replay final runtime summary by budget",
    )


def rows_at_budgets(trace_rows: list[dict[str, Any]], budgets: list[int]) -> list[dict[str, Any]]:
    if not trace_rows:
        return []
    ordered = sorted(trace_rows, key=lambda r: float(r["query_budget"]))
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        if float(ordered[-1]["query_budget"]) < float(budget):
            continue
        candidates = [r for r in ordered if float(r["query_budget"]) <= float(budget)]
        if not candidates:
            continue
        chosen = candidates[-1]
        out = {
            "run_kind": "hardware_replay",
            "repetition": int(chosen["repetition"]),
            "algorithm": chosen["algorithm"],
            "algorithm_key": chosen["algorithm_key"],
            "budget": int(budget),
            "query_budget_actual": float(chosen["query_budget"]),
            "estimate": float(chosen["estimate"]),
            "abs_error": float(chosen["abs_error"]),
            "normalized_abs_error": float(chosen["normalized_abs_error"]),
            "grover_power": int(chosen["grover_power"]),
            "amplification_factor": int(chosen["amplification_factor"]),
            "a_true": float(chosen["a_true"]),
            "runtime_wall_seconds": float(chosen.get("runtime_wall_seconds", np.nan)),
            "replay_probability_source": str(chosen.get("replay_probability_source", "measured")),
            "replay_probability_extrapolated": bool(chosen.get("replay_probability_extrapolated", False)),
        }
        rows.append(out)
    return rows


def print_budget_runtime_summary(rows: list[dict[str, Any]], *, prefix: str) -> None:
    if not rows:
        print(f"[{prefix}] no budget rows available", flush=True)
        return
    print(f"[{prefix}]", flush=True)
    for row in sorted(rows, key=lambda r: (int(r["budget"]), str(r["algorithm"]))):
        runtime = float(row.get("runtime_wall_seconds_median", np.nan))
        runtime_mean = float(row.get("runtime_wall_seconds_mean", np.nan))
        err = float(row.get("normalized_abs_error_median", np.nan))
        success = float(row.get("success_rate", np.nan))
        n_runs = int(float(row.get("n_runs", 0)))
        print(
            f"  budget={int(row['budget']):>7d} alg={str(row['algorithm']):<8s} "
            f"runtime_median={runtime:.3f}s runtime_mean={runtime_mean:.3f}s "
            f"nae_median={err:.6g} success={success:.3f} n={n_runs}",
            flush=True,
        )


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, rng: Any = None) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean

    if rng is None:
        rng = np.random.default_rng(12345)

    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = np.mean(rng.choice(values, size=len(values), replace=True))

    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return mean, low, high


def bootstrap_median_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, rng: Any = None) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    median = float(np.median(values))
    if len(values) == 1:
        return median, median, median

    if rng is None:
        rng = np.random.default_rng(12345)

    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_medians[i] = np.median(rng.choice(values, size=len(values), replace=True))

    low = float(np.quantile(boot_medians, alpha / 2))
    high = float(np.quantile(boot_medians, 1 - alpha / 2))
    return median, low, high


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _query_budget(row: Mapping[str, Any]) -> float:
    for key in ("query_budget", "query_budget_actual", "budget", "final_queries"):
        value = _as_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def aggregate_budget_summary(
    rows: list[dict[str, Any]],
    *,
    total_repetitions: int | None = None,
    max_bins: int = 12,
    min_points_per_bin: int = 100,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    if not rows:
        return summary
    algorithms = sorted({str(r["algorithm"]) for r in rows})
    for algorithm in algorithms:
        alg_rows = [row for row in rows if str(row["algorithm"]) == algorithm]
        query_budget = np.asarray([_query_budget(row) for row in alg_rows], dtype=float)
        error_values = np.asarray([_as_float(row.get("normalized_abs_error")) for row in alg_rows], dtype=float)
        valid = (
            np.isfinite(query_budget)
            & np.isfinite(error_values)
            & (query_budget > 0.0)
            & (error_values > 0.0)
        )
        if not np.any(valid):
            continue

        bin_indices = log_query_bin_indices(
            query_budget,
            error_values,
            max_bins=max_bins,
            min_points_per_bin=min_points_per_bin,
        )

        for indices in bin_indices:
            if indices.size == 0:
                continue
            subset = [alg_rows[int(idx)] for idx in indices]
            if not subset:
                continue
            normalized_abs_error = np.asarray(
                [_as_float(r.get("normalized_abs_error")) for r in subset],
                dtype=float,
            )
            abs_error = np.asarray([_as_float(r.get("abs_error")) for r in subset], dtype=float)
            estimates = np.asarray([_as_float(r.get("estimate")) for r in subset], dtype=float)
            k_vals = np.asarray(
                [_as_float(r.get("k_max_budget"), _as_float(r.get("grover_power"))) for r in subset],
                dtype=float,
            )
            amplification_factors = np.asarray(
                [_as_float(r.get("amplification_factor")) for r in subset],
                dtype=float,
            )
            query_budget_actual = np.asarray(
                [_query_budget(r) for r in subset],
                dtype=float,
            )
            runtime_wall_seconds = np.asarray(
                [_as_float(r.get("time_to_budget_seconds", r.get("runtime_wall_seconds", np.nan))) for r in subset],
                dtype=float,
            )
            extrapolated_flags = np.asarray(
                [bool(r.get("replay_probability_extrapolated", False)) for r in subset],
                dtype=bool,
            )
            n_subset = int(len(subset))
            repetitions = {
                int(_as_float(r.get("repetition")))
                for r in subset
                if np.isfinite(_as_float(r.get("repetition")))
            }
            n_runs = int(len(repetitions)) if repetitions else n_subset
            estimate_std = float(np.nanstd(estimates, ddof=1)) if n_subset > 1 else 0.0
            abs_error_std = float(np.nanstd(abs_error, ddof=1)) if n_subset > 1 else 0.0
            normalized_abs_error_std = (
                float(np.nanstd(normalized_abs_error, ddof=1)) if n_subset > 1 else 0.0
            )
            runtime_std = float(np.nanstd(runtime_wall_seconds, ddof=1)) if n_subset > 1 else 0.0
            _, mae_low, mae_high = bootstrap_mean_ci(abs_error)
            nae_mean, nae_low, nae_high = bootstrap_mean_ci(normalized_abs_error)
            nae_median, nae_median_low, nae_median_high = bootstrap_median_ci(normalized_abs_error)
            runtime_mean, runtime_mean_low, runtime_mean_high = bootstrap_mean_ci(runtime_wall_seconds)
            runtime_median, runtime_median_low, runtime_median_high = bootstrap_median_ci(runtime_wall_seconds)
            summary.append(
                {
                    "run_kind": "hardware_replay",
                    "budget": int(round(float(np.nanmedian(query_budget_actual)))),
                    "algorithm": algorithm,
                    "algorithm_key": str(subset[0].get("algorithm_key", algorithm)),
                    "n_points": n_subset,
                    "n_runs": n_runs,
                    "total_repetitions": np.nan if total_repetitions is None else int(total_repetitions),
                    "success_rate": np.nan
                    if total_repetitions is None or int(total_repetitions) <= 0
                    else float(n_runs / int(total_repetitions)),
                    "estimate_mean": float(np.nanmean(estimates)),
                    "estimate_std": estimate_std,
                    "estimate_se": float(estimate_std / math.sqrt(n_subset)) if n_subset > 0 else np.nan,
                    "query_budget_actual_mean": float(np.nanmean(query_budget_actual)),
                    "query_budget_actual_median": float(np.nanmedian(query_budget_actual)),
                    "query_budget_actual_q25": float(np.nanquantile(query_budget_actual, 0.25)),
                    "query_budget_actual_q75": float(np.nanquantile(query_budget_actual, 0.75)),
                    "abs_error_mean": float(np.nanmean(abs_error)),
                    "abs_error_median": float(np.nanmedian(abs_error)),
                    "abs_error_std": abs_error_std,
                    "abs_error_se": float(abs_error_std / math.sqrt(n_subset)) if n_subset > 0 else np.nan,
                    "mae_ci_low": mae_low,
                    "mae_ci_high": mae_high,
                    "normalized_abs_error_mean": nae_mean,
                    "normalized_abs_error_std": normalized_abs_error_std,
                    "normalized_abs_error_se": standard_error(normalized_abs_error),
                    "normalized_abs_error_ci_low": nae_low,
                    "normalized_abs_error_ci_high": nae_high,
                    "normalized_abs_error_median": nae_median,
                    "normalized_abs_error_median_ci_low": nae_median_low,
                    "normalized_abs_error_median_ci_high": nae_median_high,
                    "normalized_abs_error_q25": float(np.nanquantile(normalized_abs_error, 0.25)),
                    "normalized_abs_error_q75": float(np.nanquantile(normalized_abs_error, 0.75)),
                    "grover_power_max_median": float(np.nanmedian(k_vals)),
                    "amplification_factor_median": float(np.nanmedian(amplification_factors)),
                    "runtime_wall_seconds_mean": runtime_mean,
                    "runtime_wall_seconds_std": runtime_std,
                    "runtime_wall_seconds_se": float(runtime_std / math.sqrt(n_subset))
                    if n_subset > 0
                    else np.nan,
                    "runtime_wall_seconds_ci_low": runtime_mean_low,
                    "runtime_wall_seconds_ci_high": runtime_mean_high,
                    "runtime_wall_seconds_median": runtime_median,
                    "runtime_wall_seconds_median_ci_low": runtime_median_low,
                    "runtime_wall_seconds_median_ci_high": runtime_median_high,
                    "runtime_wall_seconds_q25": float(np.nanquantile(runtime_wall_seconds, 0.25)),
                    "runtime_wall_seconds_q75": float(np.nanquantile(runtime_wall_seconds, 0.75)),
                    "replay_extrapolated_fraction": float(np.mean(extrapolated_flags))
                    if n_subset > 0
                    else np.nan,
                }
            )
    return summary


def run_plotter(run_dir: Path) -> None:
    import subprocess

    plotter = BETA_DIR / "plot_hardware.py"
    subprocess.run([sys.executable, str(plotter), "--run-dir", str(run_dir)], check=False)


def create_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = BETA_DIR / "runs" / f"hardware_amplification_bae_biqae_cabiqae_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_replay_only(args: argparse.Namespace) -> None:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode replay-only.")
    run_dir = Path(args.run_dir).expanduser().resolve()
    paths = RunPaths(run_dir)
    config = json.loads(paths.config.read_text(encoding="utf-8"))
    state = ExperimentState(paths=paths, config=config)
    state.job_rows = load_csv(paths.runtime_jobs) if paths.runtime_jobs.exists() else []
    state.error_rows = load_csv(paths.errors) if paths.errors.exists() else []
    state.readout_rows = load_csv(paths.readout_calibration) if paths.readout_calibration.exists() else []
    state.amplification_count_rows = (
        load_csv(paths.amplification_counts) if paths.amplification_counts.exists() else []
    )
    state.amplification_point_rows = (
        load_csv(paths.amplification_points) if paths.amplification_points.exists() else []
    )
    state.direct_trace_rows = load_csv(paths.direct_trace) if paths.direct_trace.exists() else []
    state.direct_final_rows = load_csv(paths.direct_final) if paths.direct_final.exists() else []
    state.calibration_summary = (
        json.loads(paths.calibration_summary.read_text(encoding="utf-8"))
        if paths.calibration_summary.exists()
        else {}
    )
    state.session_details = (
        json.loads(paths.session_details.read_text(encoding="utf-8"))
        if paths.session_details.exists()
        else {}
    )
    objective_ry_offset = float(config.get("objective_ry_offset", args.objective_ry_offset))
    problem, a_true = build_large_problem(objective_ry_offset)
    if paths.amplification_points.exists():
        points = load_csv(paths.amplification_points)
        p_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated"]) for r in points}
        p_se_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated_se"]) for r in points}
    else:
        p_by_k = load_replay_probabilities_from_counts(paths.amplification_counts)
        p_se_by_k = None
    summary = state.calibration_summary
    t_eff = effective_t_for_algorithms(summary)
    state.config["replay_repetitions"] = int(args.replay_repetitions)
    state.config["replay_probability_mode"] = str(args.replay_probability_mode)
    state.config["replay_probability_se_scale"] = float(args.replay_probability_se_scale)
    state.config["replay_extrapolate"] = bool(getattr(args, "extrapolate", False))
    state.config["epsilon_target"] = float(args.epsilon_target)
    state.config["verbose"] = bool(args.verbose)
    run_replay(
        state,
        problem,
        algorithms=get_algorithms(args),
        algorithm_labels=get_algorithm_labels(args),
        p_by_k=p_by_k,
        p_se_by_k=p_se_by_k,
        replay_probability_mode=str(args.replay_probability_mode),
        replay_probability_se_scale=float(args.replay_probability_se_scale),
        a_true=float(config.get("a_true", a_true)),
        objective_ry_offset=objective_ry_offset,
        budgets=parse_int_list(args.budgets),
        repetitions=int(args.replay_repetitions),
        n_shots=int(args.direct_shots),
        epsilon_target=float(args.epsilon_target),
        alpha=float(args.alpha),
        t_eff=t_eff,
        seed=int(args.seed),
        extrapolate=bool(getattr(args, "extrapolate", False)),
        calibration_summary=summary,
        verbose=bool(args.verbose),
    )
    if not args.skip_plots:
        run_plotter(run_dir)


def load_existing_state(run_dir: Path) -> ExperimentState:
    paths = RunPaths(run_dir)
    config = json.loads(paths.config.read_text(encoding="utf-8"))
    state = ExperimentState(paths=paths, config=config)
    state.job_rows = load_csv(paths.runtime_jobs) if paths.runtime_jobs.exists() else []
    state.error_rows = load_csv(paths.errors) if paths.errors.exists() else []
    state.readout_rows = load_csv(paths.readout_calibration) if paths.readout_calibration.exists() else []
    state.amplification_count_rows = (
        load_csv(paths.amplification_counts) if paths.amplification_counts.exists() else []
    )
    state.amplification_point_rows = (
        load_csv(paths.amplification_points) if paths.amplification_points.exists() else []
    )
    state.direct_trace_rows = load_csv(paths.direct_trace) if paths.direct_trace.exists() else []
    state.direct_final_rows = load_csv(paths.direct_final) if paths.direct_final.exists() else []
    state.replay_trace_rows = load_csv(paths.replay_trace) if paths.replay_trace.exists() else []
    state.replay_final_rows = load_csv(paths.replay_final) if paths.replay_final.exists() else []
    state.budget_summary_rows = load_csv(paths.budget_summary) if paths.budget_summary.exists() else []
    state.calibration_summary = (
        json.loads(paths.calibration_summary.read_text(encoding="utf-8"))
        if paths.calibration_summary.exists()
        else {}
    )
    state.session_details = (
        json.loads(paths.session_details.read_text(encoding="utf-8"))
        if paths.session_details.exists()
        else {}
    )
    return state


def run_hardware_topup(args: argparse.Namespace) -> None:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode hardware-topup.")
    if not args.scan_grover_powers:
        raise ValueError("--scan-grover-powers is required for --mode hardware-topup.")

    run_dir = Path(args.run_dir).expanduser().resolve()
    state = load_existing_state(run_dir)
    paths = state.paths
    objective_ry_offset = float(state.config.get("objective_ry_offset", args.objective_ry_offset))
    problem, a_true = build_large_problem(objective_ry_offset)
    state.config["a_true"] = float(state.config.get("a_true", a_true))

    readout_params = state.calibration_summary.get("readout")
    if not isinstance(readout_params, Mapping):
        raise RuntimeError("Existing run has no readout calibration in calibration_summary.json.")

    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit_ibm_runtime import Session

    service = QiskitRuntimeService(channel=args.channel)
    backend = service.backend(args.backend)
    pass_manager, transpilation_metadata = build_pass_manager_for_mode(
        backend,
        problem,
        mode="hardware",
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=parse_int_list(args.reference_ks),
    )
    state.config["topup_transpilation"] = transpilation_metadata
    state.config["topup_scan_grover_powers"] = parse_int_list(args.scan_grover_powers)
    state.config["topup_scan_shots"] = int(args.scan_shots)
    state.config["topup_scan_repeats"] = int(args.scan_repeats)
    state.config["verbose"] = bool(args.verbose)
    state.config["topup_updated_at_epoch"] = time.time()
    state.persist()

    global_start = time.perf_counter()
    with Session(backend=backend, max_time=args.session_max_time) as session:
        topup_sessions = list(state.session_details.get("topup_sessions", []))
        topup_sessions.append(
            {
                "session_id": getattr(session, "session_id", None),
                "session_started_at_epoch": time.time(),
                "scan_grover_powers": parse_int_list(args.scan_grover_powers),
            }
        )
        state.session_details["topup_sessions"] = topup_sessions
        runtime_sampler = Sampler(mode=session)
        sampler = RuntimeCountSampler(
            backend,
            runtime_sampler,
            pass_manager,
            state,
            soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
            max_grover_power=None,
            max_calls_by_context={},
            start_time=global_start,
        )
        new_rows = run_amplification_scan(
            sampler,
            problem,
            grover_powers=parse_int_list(args.scan_grover_powers),
            repeats=int(args.scan_repeats),
            shots=int(args.scan_shots),
            seed=int(args.seed),
            verbose=bool(args.verbose),
        )
        state.amplification_count_rows.extend(new_rows)
        points, calibration_summary, _ = analyze_amplification(
            state.amplification_count_rows,
            problem,
            readout_params,
        )
        state.amplification_point_rows = points
        state.calibration_summary.update(calibration_summary)
        state.session_details["topup_sessions"][-1]["session_finished_at_epoch"] = time.time()
        state.persist()

    if not args.skip_replay:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        p_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated"]) for r in state.amplification_point_rows}
        p_se_by_k = {
            int(r["grover_power"]): float(r["p_hw_mitigated_se"])
            for r in state.amplification_point_rows
        }
        run_replay(
            state,
            problem,
            algorithms=get_algorithms(args),
            algorithm_labels=get_algorithm_labels(args),
            p_by_k=p_by_k,
            p_se_by_k=p_se_by_k,
            replay_probability_mode=str(args.replay_probability_mode),
            replay_probability_se_scale=float(args.replay_probability_se_scale),
            a_true=float(state.config.get("a_true", a_true)),
            objective_ry_offset=objective_ry_offset,
            budgets=parse_int_list(args.budgets),
            repetitions=int(args.replay_repetitions),
            n_shots=int(args.direct_shots),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            t_eff=t_eff,
            seed=int(args.seed),
            extrapolate=bool(getattr(args, "extrapolate", False)),
            calibration_summary=state.calibration_summary,
            verbose=bool(args.verbose),
        )

    if not args.skip_plots:
        run_plotter(paths.run_dir)


def run_experiment(args: argparse.Namespace) -> None:
    algorithms = get_algorithms(args)
    algorithm_labels = get_algorithm_labels(args)
    run_dir = create_run_dir(args)
    paths = RunPaths(run_dir)
    run_id = run_dir.name
    config = {
        "run_id": run_id,
        "run_uuid": str(uuid.uuid4()),
        "mode": args.mode,
        "backend": args.backend,
        "channel": args.channel,
        "objective_ry_offset": float(args.objective_ry_offset),
        "max_grover_power_requested": int(args.max_grover_power),
        "external_k_cap_enabled": False,
        "scan_repeats": int(args.scan_repeats),
        "scan_shots": int(args.scan_shots),
        "readout_shots": int(args.readout_shots),
        "direct_shots": int(args.direct_shots),
        "max_direct_sampler_calls_per_algorithm": int(args.max_direct_calls),
        "replay_repetitions": int(args.replay_repetitions),
        "replay_probability_mode": str(args.replay_probability_mode),
        "replay_probability_se_scale": float(args.replay_probability_se_scale),
        "budgets": parse_int_list(args.budgets),
        "session_max_time": args.session_max_time,
        "soft_wallclock_limit_seconds": float(args.soft_wallclock_limit),
        "max_isa_depth": int(args.max_isa_depth),
        "max_isa_2q": int(args.max_isa_2q),
        "optimization_level": int(args.optimization_level),
        "seed_transpiler": int(args.seed_transpiler),
        "seed": int(args.seed),
        "verbose": bool(args.verbose),
        "algorithms": list(algorithms),
        "bae_adapter_kind": BAE_KIND,
    }
    state = ExperimentState(paths=paths, config=config)
    state.session_details = {"mode": args.mode, "created_at_epoch": time.time()}

    problem, a_true = build_large_problem(float(args.objective_ry_offset))
    state.config["a_true"] = float(a_true)

    if args.mode == "dry-run":
        try:
            backend = load_fake_backend(args.fake_backend)
            backend_mode = f"fake:{args.fake_backend}"
        except Exception:
            backend = AerSimulator()
            backend_mode = "aer_simulator"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel=args.channel)
        backend = service.backend(args.backend)
        backend_mode = "runtime_backend"

    save_json(backend_snapshot(backend, mode=backend_mode, channel=args.channel), paths.backend_snapshot)
    pass_manager, transpilation_metadata = build_pass_manager_for_mode(
        backend,
        problem,
        mode=args.mode,
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=parse_int_list(args.reference_ks),
    )
    state.config["transpilation"] = transpilation_metadata
    state.persist()

    preflight_rows, allowed_max = run_preflight(
        problem,
        pass_manager,
        state,
        max_grover_power=int(args.max_grover_power),
        max_isa_depth=int(args.max_isa_depth),
        max_isa_2q=int(args.max_isa_2q),
    )
    del preflight_rows
    max_experiment_k = min(int(args.max_grover_power), int(allowed_max))
    state.config["max_grover_power_after_preflight"] = int(max_experiment_k)
    state.persist()

    if args.mode == "preflight":
        return

    if args.mode == "dry-run":
        noise_model = build_noise_model(float(args.dry_run_noise_scale), profile=args.dry_run_noise_profile)
        aer = AerCountSampler(
            noise_model=noise_model,
            seed=int(args.seed),
            method=args.aer_method,
            transpile_backend=backend,
        )
        sampler = LoggedAerSampler(aer, state, max_grover_power=None)
        execute_non_replay_phases(
            args,
            state,
            sampler,
            problem,
            a_true,
            max_experiment_k,
            algorithms=algorithms,
            algorithm_labels=algorithm_labels,
        )
    elif args.mode == "hardware":
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit_ibm_runtime import Session

        global_start = time.perf_counter()
        with Session(backend=backend, max_time=args.session_max_time) as session:
            state.session_details.update(
                {
                    "session_id": getattr(session, "session_id", None),
                    "session_started_at_epoch": time.time(),
                }
            )
            runtime_sampler = Sampler(mode=session)
            max_calls_by_context = {
                f"direct_live_{algorithm}": int(args.max_direct_calls)
                for algorithm in algorithms
            }
            sampler = RuntimeCountSampler(
                backend,
                runtime_sampler,
                pass_manager,
                state,
                soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                max_grover_power=None,
                max_calls_by_context=max_calls_by_context,
                start_time=global_start,
            )
            execute_non_replay_phases(
                args,
                state,
                sampler,
                problem,
                a_true,
                max_experiment_k,
                algorithms=algorithms,
                algorithm_labels=algorithm_labels,
            )
            state.session_details["session_finished_at_epoch"] = time.time()
            state.persist()
    else:
        raise ValueError(f"Unsupported mode for run_experiment: {args.mode}")

    if not args.skip_replay:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        p_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated"]) for r in state.amplification_point_rows}
        p_se_by_k = {
            int(r["grover_power"]): float(r["p_hw_mitigated_se"])
            for r in state.amplification_point_rows
        }
        run_replay(
            state,
            problem,
            algorithms=algorithms,
            algorithm_labels=algorithm_labels,
            p_by_k=p_by_k,
            p_se_by_k=p_se_by_k,
            replay_probability_mode=str(args.replay_probability_mode),
            replay_probability_se_scale=float(args.replay_probability_se_scale),
            a_true=a_true,
            objective_ry_offset=float(args.objective_ry_offset),
            budgets=parse_int_list(args.budgets),
            repetitions=int(args.replay_repetitions),
            n_shots=int(args.direct_shots),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            t_eff=t_eff,
            seed=int(args.seed),
            extrapolate=bool(getattr(args, "extrapolate", False)),
            calibration_summary=state.calibration_summary,
            verbose=bool(args.verbose),
        )

    if not args.skip_plots:
        run_plotter(paths.run_dir)


def execute_non_replay_phases(
    args: argparse.Namespace,
    state: ExperimentState,
    sampler: Any,
    problem: Any,
    a_true: float,
    max_experiment_k: int,
    *,
    algorithms: tuple[str, ...],
    algorithm_labels: Mapping[str, str],
) -> None:
    readout_rows, readout_params = run_readout_calibration(
        sampler,
        problem,
        shots=int(args.readout_shots),
    )
    state.readout_rows = readout_rows
    state.calibration_summary["readout"] = readout_params
    state.persist()

    grover_powers = list(range(int(max_experiment_k) + 1))
    state.amplification_count_rows = run_amplification_scan(
        sampler,
        problem,
        grover_powers=grover_powers,
        repeats=int(args.scan_repeats),
        shots=int(args.scan_shots),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )
    points, calibration_summary, _ = analyze_amplification(
        state.amplification_count_rows,
        problem,
        readout_params,
    )
    state.amplification_point_rows = points
    state.calibration_summary.update(calibration_summary)

    if calibration_summary.get("calibration_status") == "ok":
        k_visible = int(calibration_summary.get("k_visible", 0))
        max_direct_k = min(int(max_experiment_k), max(1, k_visible))
    else:
        max_direct_k = min(3, int(max_experiment_k))
    state.config["max_grover_power_direct"] = int(max_direct_k)
    state.config["calibration_status"] = calibration_summary.get("calibration_status")
    state.persist()

    if not args.skip_direct:
        run_direct_live(
            state,
            sampler,
            problem,
            algorithms=algorithms,
            algorithm_labels=algorithm_labels,
            a_true=float(a_true),
            objective_ry_offset=float(args.objective_ry_offset),
            n_shots=int(args.direct_shots),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            t_eff=effective_t_for_algorithms(state.calibration_summary),
            max_direct_calls=int(args.max_direct_calls),
            seed=int(args.seed),
            verbose=bool(args.verbose),
        )
