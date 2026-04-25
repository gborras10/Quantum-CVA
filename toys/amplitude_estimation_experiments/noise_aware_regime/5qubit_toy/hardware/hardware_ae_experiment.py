from __future__ import annotations

import csv
import json
import os
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
hardware_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")

for path in [hardware_dir, src_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --------------------------------------------------------------------------------------
# Imports from project
# --------------------------------------------------------------------------------------
try:
    from ae_pipeline_utils import (
        DEFAULT_AE_REFERENCE_KS,
        build_ae_pass_manager,
        build_solver as build_common_solver,
        build_large_problem,
        circuit_cache_key,
        extract_trace,
    )
except ImportError as e:
    print(f"Error importing realistic_utils: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Qiskit Runtime imports
# --------------------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Session,
        SamplerV2 as Sampler,
    )
except ImportError as e:
    print(f"Error importing Qiskit Runtime modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
BACKEND_NAME = "ibm_basquecountry"

# Same problem family that has been used already
OBJECTIVE_RY_OFFSET = -0.10

# Comparison
ALGORITHMS = ("bae", "cabiqae")

# Budget
SESSION_MAX_TIME = "5m"
SOFT_WALLCLOCK_LIMIT_SECONDS = 270.0   # margin before forced shutdown
MAX_SAMPLER_CALLS_PER_ALGORITHM = 18   # safety cutoff

# Benchmark parameters: deliberately modest
ALPHA = 0.10
NUM_SHOTS = 16
EPSILON_TARGET = 0.08

# Use  known calibration; change it if needed
T_EFF_KNOWN = 4.96
CAP_KAPPA = 1000.0 # let cabiqae decide based on Fisher info

# transpilation
OPTIMIZATION_LEVEL = 3
SEED_TRANSPILE = 1234
TRANSPILATION_REFERENCE_KS = DEFAULT_AE_REFERENCE_KS

# outputs
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_UUID = str(uuid.uuid4())

OUTPUT_DIR = os.path.join(current_dir, f"hardware_bae_vs_cabiqae_{RUN_ID}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_JSON = os.path.join(OUTPUT_DIR, "config.json")
TRACE_CSV = os.path.join(OUTPUT_DIR, "trace_rows.csv")
FINAL_CSV = os.path.join(OUTPUT_DIR, "final_rows.csv")
JOB_CSV = os.path.join(OUTPUT_DIR, "runtime_jobs.csv")
SESSION_JSON = os.path.join(OUTPUT_DIR, "session_details.json")
MANIFEST_JSON = os.path.join(OUTPUT_DIR, "manifest.json")
NPZ_PATH = os.path.join(OUTPUT_DIR, "trace_bundle.npz")


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def save_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def save_json(obj: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def persist_all(
    config: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
    session_details: dict[str, Any] | None,
    npz_payload: dict[str, np.ndarray] | None,
) -> None:
    save_json(config, CONFIG_JSON)
    if trace_rows:
        save_csv(trace_rows, TRACE_CSV)
    if final_rows:
        save_csv(final_rows, FINAL_CSV)
    if job_rows:
        save_csv(job_rows, JOB_CSV)
    if session_details is not None:
        save_json(session_details, SESSION_JSON)
    if npz_payload:
        np.savez(NPZ_PATH, **npz_payload)

    manifest = {
        "run_id": RUN_ID,
        "run_uuid": RUN_UUID,
        "output_dir": OUTPUT_DIR,
        "config_json": CONFIG_JSON,
        "trace_csv": TRACE_CSV if trace_rows else None,
        "final_csv": FINAL_CSV if final_rows else None,
        "job_csv": JOB_CSV if job_rows else None,
        "session_json": SESSION_JSON if session_details is not None else None,
        "npz_path": NPZ_PATH if npz_payload else None,
    }
    save_json(manifest, MANIFEST_JSON)


# --------------------------------------------------------------------------------------
# Runtime sampler wrapper
# --------------------------------------------------------------------------------------
@dataclass
class RuntimeCountSampler:
    backend: Any
    sampler: Any
    optimization_level: int
    seed_transpiler: int
    global_start_time: float
    soft_wallclock_limit_seconds: float
    max_sampler_calls_per_algorithm: int

    pass_manager: Any | None = None
    transpilation_metadata: dict[str, Any] = field(default_factory=dict)
    current_algorithm: str = "unknown"
    pm: Any = field(init=False)
    _cache: dict[str, QuantumCircuit] = field(default_factory=dict, init=False)
    submitted_jobs: list[dict[str, Any]] = field(default_factory=list, init=False)
    _calls_this_algorithm: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.pm = self.pass_manager
        if self.pm is None:
            self.pm = generate_preset_pass_manager(
                backend=self.backend,
                optimization_level=self.optimization_level,
                seed_transpiler=self.seed_transpiler,
            )
            if not self.transpilation_metadata:
                self.transpilation_metadata = {
                    "strategy": "preset_default_fallback",
                    "fallback_used": True,
                    "fallback_reason": "pass_manager_not_provided",
                    "initial_layout": None,
                    "seed_transpiler": int(self.seed_transpiler),
                    "optimization_level": int(self.optimization_level),
                    "routing_method": None,
                    "candidate_source": "legacy_default",
                    "reference_ks": list(TRANSPILATION_REFERENCE_KS),
                }

    def set_algorithm(self, algorithm_name: str) -> None:
        self.current_algorithm = str(algorithm_name)
        self._calls_this_algorithm = 0

    def _cache_key(self, circuit: QuantumCircuit) -> str:
        return circuit_cache_key(circuit)

    def _isa_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit = circuit.decompose(reps=10)
        key = self._cache_key(circuit)
        if key not in self._cache:
            self._cache[key] = self.pm.run(circuit)
        return self._cache[key]

    def _check_budget(self) -> None:
        elapsed = time.perf_counter() - self.global_start_time
        if elapsed > self.soft_wallclock_limit_seconds:
            raise TimeoutError(
                f"Soft wall-clock limit exceeded: {elapsed:.1f}s > "
                f"{self.soft_wallclock_limit_seconds:.1f}s"
            )
        if self._calls_this_algorithm >= self.max_sampler_calls_per_algorithm:
            raise TimeoutError(
                f"Max sampler calls reached for {self.current_algorithm}: "
                f"{self._calls_this_algorithm} >= {self.max_sampler_calls_per_algorithm}"
            )

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> Any:
        self._check_budget()

        isa_circuits = [self._isa_circuit(c) for c in circuits]
        job = self.sampler.run(isa_circuits, shots=int(shots))

        job_row = {
            "run_id": RUN_ID,
            "run_uuid": RUN_UUID,
            "algorithm": self.current_algorithm,
            "sampler_call_index_for_algorithm": int(self._calls_this_algorithm),
            "n_circuits": int(len(isa_circuits)),
            "shots": int(shots),
            "job_id": str(job.job_id()),
            "submitted_at_epoch": float(time.time()),
        }
        self.submitted_jobs.append(job_row)
        self._calls_this_algorithm += 1

        print(
            f"[Runtime] alg={self.current_algorithm:8s} | "
            f"call={self._calls_this_algorithm:02d} | "
            f"job_id={job.job_id()} | ncirc={len(isa_circuits)} | shots={shots}"
        )
        return job


# --------------------------------------------------------------------------------------
# Solver construction
# --------------------------------------------------------------------------------------
def build_solver(
    algorithm: str,
    epsilon_target: float,
    alpha: float,
    n_shots: int,
    sampler: RuntimeCountSampler,
    t_eff: float | None,
) -> tuple[Any, bool]:
    canonical_algorithm = "cabiqae_latentt" if algorithm == "cabiqae" else algorithm
    return build_common_solver(
        canonical_algorithm,
        epsilon_target,
        alpha,
        n_shots,
        seed=0,
        noisy_sampler=sampler,
        t_eff=t_eff,
        cap_kappa=CAP_KAPPA,
    )


# --------------------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------------------
def run_experiment() -> None:
    config = {
        "run_id": RUN_ID,
        "run_uuid": RUN_UUID,
        "backend_name": BACKEND_NAME,
        "objective_ry_offset": OBJECTIVE_RY_OFFSET,
        "algorithms": list(ALGORITHMS),
        "session_max_time": SESSION_MAX_TIME,
        "soft_wallclock_limit_seconds": SOFT_WALLCLOCK_LIMIT_SECONDS,
        "max_sampler_calls_per_algorithm": MAX_SAMPLER_CALLS_PER_ALGORITHM,
        "alpha": ALPHA,
        "num_shots": NUM_SHOTS,
        "epsilon_target": EPSILON_TARGET,
        "t_eff_known": T_EFF_KNOWN,
        "cap_kappa": CAP_KAPPA,
        "optimization_level": OPTIMIZATION_LEVEL,
        "seed_transpile": SEED_TRANSPILE,
        "transpilation_reference_ks": list(TRANSPILATION_REFERENCE_KS),
        "channel": "ibm_quantum_platform",
    }

    problem, a_true = build_large_problem(objective_ry_offset=float(OBJECTIVE_RY_OFFSET))
    config["a_true"] = float(a_true)

    trace_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    session_details: dict[str, Any] | None = None
    npz_payload: dict[str, np.ndarray] = {}

    print("=" * 100)
    print("HARDWARE BAE vs CABIQAE (SESSION MODE)")
    print("=" * 100)
    print(f"Backend                    : {BACKEND_NAME}")
    print(f"Objective offset           : {OBJECTIVE_RY_OFFSET:+.3f}")
    print(f"a_true                     : {a_true:.6f}")
    print(f"Algorithms                 : {ALGORITHMS}")
    print(f"Session max_time           : {SESSION_MAX_TIME}")
    print(f"Soft wall-clock limit      : {SOFT_WALLCLOCK_LIMIT_SECONDS:.1f}s")
    print(f"Shots per circuit          : {NUM_SHOTS}")
    print(f"Epsilon target             : {EPSILON_TARGET}")
    print(f"T_eff known                : {T_EFF_KNOWN}")
    print(f"Noise cap kappa (CABIQAE)  : {CAP_KAPPA}")
    print("=" * 100)

    global_t0 = time.perf_counter()

    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.backend(BACKEND_NAME)
    transpile_pm, transpilation_metadata = build_ae_pass_manager(
        backend,
        problem,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        reference_ks=TRANSPILATION_REFERENCE_KS,
    )
    config["optimization_level"] = int(transpilation_metadata["optimization_level"])
    config["seed_transpile"] = int(transpilation_metadata["seed_transpiler"])
    config["transpilation"] = transpilation_metadata

    runtime_sampler_wrapper: RuntimeCountSampler | None = None

    try:
        with Session(backend=backend, max_time=SESSION_MAX_TIME) as session:
            sampler = Sampler(mode=session)

            runtime_sampler_wrapper = RuntimeCountSampler(
                backend=backend,
                sampler=sampler,
                optimization_level=int(transpilation_metadata["optimization_level"]),
                seed_transpiler=int(transpilation_metadata["seed_transpiler"]),
                global_start_time=global_t0,
                soft_wallclock_limit_seconds=SOFT_WALLCLOCK_LIMIT_SECONDS,
                max_sampler_calls_per_algorithm=MAX_SAMPLER_CALLS_PER_ALGORITHM,
                pass_manager=transpile_pm,
                transpilation_metadata=transpilation_metadata,
            )

            print(f"Session id: {session.session_id}")
            print(
                "Transpilation               : "
                f"{transpilation_metadata['strategy']} | "
                f"layout={transpilation_metadata.get('initial_layout')} | "
                f"seed={transpilation_metadata['seed_transpiler']} | "
                f"fallback={transpilation_metadata['fallback_used']}"
            )

            for alg_key in ALGORITHMS:
                elapsed = time.perf_counter() - global_t0
                if elapsed > SOFT_WALLCLOCK_LIMIT_SECONDS:
                    print(f"Skipping {alg_key}: soft time budget exhausted.")
                    break

                runtime_sampler_wrapper.set_algorithm(alg_key)

                solver, is_bayes = build_solver(
                    algorithm=alg_key,
                    epsilon_target=EPSILON_TARGET,
                    alpha=ALPHA,
                    n_shots=NUM_SHOTS,
                    sampler=runtime_sampler_wrapper,
                    t_eff=T_EFF_KNOWN,
                )

                print(f"\nRunning algorithm: {alg_key}")

                try:
                    alg_t0 = time.perf_counter()

                    if alg_key == "bae":
                        # The hardware BAE implementation exposes max_queries
                        np.random.seed(12345)
                        result = solver.estimate(
                            problem,
                            n_shots=NUM_SHOTS,
                            max_queries=1200,
                        )
                    else:
                        result = solver.estimate(
                            problem,
                            bayes=is_bayes,
                            n_shots=NUM_SHOTS,
                            show_details=False,
                        )

                    runtime_seconds = float(time.perf_counter() - alg_t0)

                    queries, estimations, k_sequence = extract_trace(
                        "bae" if alg_key == "bae" else "cabiqae_latentt",
                        result,
                        NUM_SHOTS,
                    )

                    if len(queries) == 0:
                        print(f"  {alg_key}: no trajectory")
                        continue

                    final_est = float(estimations[-1])
                    final_abs_error = abs(final_est - a_true)
                    final_nrmse = final_abs_error / a_true
                    final_queries = int(getattr(result, "num_state_prep_calls", queries[-1]))
                    k_max = int(np.max(k_sequence)) if len(k_sequence) > 0 else 0

                    ci = getattr(result, "confidence_interval", None)
                    coverage = (
                        np.nan
                        if ci is None
                        else float(float(ci[0]) <= a_true <= float(ci[1]))
                    )

                    for i in range(len(queries)):
                        trace_rows.append(
                            {
                                "run_id": RUN_ID,
                                "run_uuid": RUN_UUID,
                                "algorithm": alg_key,
                                "step_index": int(i),
                                "query_budget": float(queries[i]),
                                "estimate": float(estimations[i]),
                                "abs_error": float(abs(float(estimations[i]) - a_true)),
                                "nrmse": float(abs(float(estimations[i]) - a_true) / a_true),
                                "k_value": int(k_sequence[i]),
                                "a_true": float(a_true),
                                "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
                                "runtime_seconds_total_algorithm": runtime_seconds,
                            }
                        )

                    final_rows.append(
                        {
                            "run_id": RUN_ID,
                            "run_uuid": RUN_UUID,
                            "algorithm": alg_key,
                            "a_true": float(a_true),
                            "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
                            "final_queries": int(final_queries),
                            "final_estimate": float(final_est),
                            "final_abs_error": float(final_abs_error),
                            "final_nrmse": float(final_nrmse),
                            "coverage": float(coverage) if np.isfinite(coverage) else np.nan,
                            "runtime_seconds": float(runtime_seconds),
                            "k_max": int(k_max),
                            "n_trace_points": int(len(queries)),
                        }
                    )

                    npz_payload[f"{alg_key}_queries"] = np.asarray(queries, dtype=float)
                    npz_payload[f"{alg_key}_estimations"] = np.asarray(estimations, dtype=float)
                    npz_payload[f"{alg_key}_k_sequence"] = np.asarray(k_sequence, dtype=int)

                    print(
                        f"  {alg_key:8s} | "
                        f"Qmax={int(queries[-1]):5d} | "
                        f"final_est={final_est:.6f} | "
                        f"nRMSE={final_nrmse:.3e} | "
                        f"time={runtime_seconds:.2f}s | "
                        f"Kmax={k_max:3d}"
                    )

                    session_details = session.details()
                    persist_all(
                        config=config,
                        trace_rows=trace_rows,
                        final_rows=final_rows,
                        job_rows=runtime_sampler_wrapper.submitted_jobs,
                        session_details=session_details,
                        npz_payload=npz_payload if npz_payload else None,
                    )

                except TimeoutError as e:
                    print(f"  {alg_key}: stopped by budget guard -> {e}")
                    session_details = session.details()
                    persist_all(
                        config=config,
                        trace_rows=trace_rows,
                        final_rows=final_rows,
                        job_rows=runtime_sampler_wrapper.submitted_jobs,
                        session_details=session_details,
                        npz_payload=npz_payload if npz_payload else None,
                    )
                    break

                except Exception as e:
                    print(f"  Error in {alg_key}: {e}")
                    session_details = session.details()
                    persist_all(
                        config=config,
                        trace_rows=trace_rows,
                        final_rows=final_rows,
                        job_rows=runtime_sampler_wrapper.submitted_jobs,
                        session_details=session_details,
                        npz_payload=npz_payload if npz_payload else None,
                    )

            session_details = session.details()

    finally:
        if runtime_sampler_wrapper is not None and session_details is None:
            try:
                # If the session no longer exists or is already closed, this may fail
                session_details = {}
            except Exception:
                session_details = {}

        persist_all(
            config=config,
            trace_rows=trace_rows,
            final_rows=final_rows,
            job_rows=runtime_sampler_wrapper.submitted_jobs if runtime_sampler_wrapper else [],
            session_details=session_details,
            npz_payload=npz_payload if npz_payload else None,
        )

    total_elapsed = time.perf_counter() - global_t0

    print("\n" + "=" * 100)
    print("FINISHED")
    print("=" * 100)
    print(f"Output dir               : {OUTPUT_DIR}")
    print(f"Total wall-clock         : {total_elapsed:.2f}s")

    if session_details:
        usage_time = session_details.get("usage_time", None)
        state = session_details.get("state", None)
        print(f"Session state            : {state}")
        print(f"Session usage_time [s]   : {usage_time}")

    if final_rows:
        print("\nFinal summary:")
        for row in final_rows:
            print(
                f"  {row['algorithm']:8s} | "
                f"Q={row['final_queries']:5d} | "
                f"est={row['final_estimate']:.6f} | "
                f"nRMSE={row['final_nrmse']:.3e} | "
                f"time={row['runtime_seconds']:.2f}s | "
                f"Kmax={row['k_max']:3d}"
            )
    else:
        print("No final rows were produced.")


if __name__ == "__main__":
    run_experiment()
