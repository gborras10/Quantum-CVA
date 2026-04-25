from __future__ import annotations

import csv
import json
import os
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")
hardware_dir = os.path.abspath(os.path.join(current_dir, ".."))
toy_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

for path in [toy_dir, src_dir, root_dir, hardware_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --------------------------------------------------------------------------------------
# Imports from project
# --------------------------------------------------------------------------------------
try:
    from ae_pipeline_utils import (
        DEFAULT_AE_REFERENCE_KS,
        build_ae_pass_manager,
        build_large_problem,
        circuit_cache_key,
        construct_measured_circuit,
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Qiskit Runtime imports
# --------------------------------------------------------------------------------------
try:
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime import SamplerV2 as Sampler
except ImportError as e:
    print(f"Error importing Qiskit Runtime modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
BACKEND_NAME = "ibm_basquecountry"
OBJECTIVE_RY_OFFSET = -0.10

PROBE_KS = [0, 1, 2, 3, 4, 6, 8]
PROBE_SHOTS = 256
N_REPEATS_PER_K = 6

OPTIMIZATION_LEVEL = 3
SEED_TRANSPILE = 1234
TRANSPILATION_REFERENCE_KS = DEFAULT_AE_REFERENCE_KS

CALIB_ID = time.strftime("%Y%m%d_%H%M%S")
CALIB_UUID = str(uuid.uuid4())

JOB_MAP_CSV = os.path.join(current_dir, f"t_eff_single_job_map_{CALIB_ID}.csv")
CONFIG_JSON = os.path.join(current_dir, f"t_eff_single_job_config_{CALIB_ID}.json")


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def save_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        print(f"No rows to save for {path}")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_meas_register(circuit: QuantumCircuit) -> QuantumCircuit:
    if circuit.num_clbits > 0:
        return circuit
    qc = circuit.copy()
    creg = ClassicalRegister(qc.num_qubits, "meas")
    qc.add_register(creg)
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))
    return qc


@dataclass
class RuntimeSingleJobBuilder:
    backend: Any
    optimization_level: int = 0
    seed_transpiler: int = 1234
    pass_manager: Any | None = None
    transpilation_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.pm = self.pass_manager
        if self.pm is None:
            self.pm = generate_preset_pass_manager(
                backend=self.backend,
                optimization_level=self.optimization_level,
                seed_transpiler=self.seed_transpiler,
            )
            if self.transpilation_metadata is None:
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
        self._cache: dict[str, QuantumCircuit] = {}

    def _cache_key(self, circuit: QuantumCircuit) -> str:
        return circuit_cache_key(circuit)

    def isa_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        qc = ensure_meas_register(circuit).decompose(reps=10)
        key = self._cache_key(qc)
        if key not in self._cache:
            self._cache[key] = self.pm.run(qc)
        return self._cache[key]


def config_dict(transpilation_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    effective_transpilation = transpilation_metadata or {
        "strategy": "preset_default_fallback",
        "fallback_used": True,
        "fallback_reason": "not_computed_yet",
        "initial_layout": None,
        "seed_transpiler": int(SEED_TRANSPILE),
        "optimization_level": int(OPTIMIZATION_LEVEL),
        "routing_method": None,
        "candidate_source": "legacy_default",
        "reference_ks": list(TRANSPILATION_REFERENCE_KS),
    }
    return {
        "calib_id": CALIB_ID,
        "calib_uuid": CALIB_UUID,
        "backend_name": BACKEND_NAME,
        "objective_ry_offset": OBJECTIVE_RY_OFFSET,
        "probe_ks": PROBE_KS,
        "probe_shots": PROBE_SHOTS,
        "n_repeats_per_k": N_REPEATS_PER_K,
        "optimization_level": int(effective_transpilation["optimization_level"]),
        "seed_transpile": int(effective_transpilation["seed_transpiler"]),
        "transpilation_reference_ks": list(TRANSPILATION_REFERENCE_KS),
        "transpilation": effective_transpilation,
        "execution_mode": "job",
        "submission_style": "single_sampler_run_single_job",
        "job_map_csv": JOB_MAP_CSV,
    }


def main() -> None:
    problem, a_true = build_large_problem(objective_ry_offset=float(OBJECTIVE_RY_OFFSET))

    print("=" * 100)
    print("T_eff HARDWARE CALIBRATION - SINGLE JOB SUBMISSION")
    print("=" * 100)
    print(f"Backend               : {BACKEND_NAME}")
    print(f"Objective offset      : {OBJECTIVE_RY_OFFSET:+.3f}")
    print(f"a_true                : {a_true:.6f}")
    print(f"probe ks              : {PROBE_KS}")
    print(f"shots / repeat        : {PROBE_SHOTS}")
    print(f"repeats / k           : {N_REPEATS_PER_K}")
    print(f"total pubs            : {len(PROBE_KS) * N_REPEATS_PER_K}")
    
    service = QiskitRuntimeService(channel="ibm_cloud")
    backend = service.backend(BACKEND_NAME)
    sampler = Sampler(mode=backend)
    transpile_pm, transpilation_metadata = build_ae_pass_manager(
        backend,
        problem,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=SEED_TRANSPILE,
        reference_ks=TRANSPILATION_REFERENCE_KS,
    )
    print(
        "Transpilation         : "
        f"{transpilation_metadata['strategy']} | "
        f"layout={transpilation_metadata.get('initial_layout')} | "
        f"seed={transpilation_metadata['seed_transpiler']} | "
        f"fallback={transpilation_metadata['fallback_used']}"
    )
    print("=" * 100)

    save_json(config_dict(transpilation_metadata), CONFIG_JSON)

    builder = RuntimeSingleJobBuilder(
        backend=backend,
        optimization_level=int(transpilation_metadata["optimization_level"]),
        seed_transpiler=int(transpilation_metadata["seed_transpiler"]),
        pass_manager=transpile_pm,
        transpilation_metadata=transpilation_metadata,
    )

    pubs: list[QuantumCircuit] = []
    rows: list[dict[str, Any]] = []

    pub_index = 0
    for k in PROBE_KS:
        logical_circuit = construct_measured_circuit(problem, k)
        isa_qc = builder.isa_circuit(logical_circuit)

        for rep in range(1, N_REPEATS_PER_K + 1):
            pubs.append(isa_qc)

            rows.append(
                {
                    "calib_id": CALIB_ID,
                    "calib_uuid": CALIB_UUID,
                    "backend": BACKEND_NAME,
                    "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
                    "pub_index": int(pub_index),
                    "k": int(k),
                    "K": int(2 * k + 1),
                    "rep": int(rep),
                    "shots": int(PROBE_SHOTS),
                    "logical_num_qubits": int(logical_circuit.num_qubits),
                    "logical_depth": int(logical_circuit.depth()),
                    "logical_size": int(logical_circuit.size()),
                    "isa_num_qubits": int(isa_qc.num_qubits),
                    "isa_depth": int(isa_qc.depth()),
                    "isa_size": int(isa_qc.size()),
                    "isa_cregs": ",".join(creg.name for creg in isa_qc.cregs),
                }
            )
            pub_index += 1

    if not pubs:
        raise RuntimeError("No PUBs were built.")

    print("Submitting single Sampler job...")
    job = sampler.run(pubs, shots=int(PROBE_SHOTS))
    job_id = job.job_id()

    submission_epoch = time.time()
    for row in rows:
        row["job_id"] = str(job_id)
        row["submitted_at_epoch"] = float(submission_epoch)

    save_csv(rows, JOB_MAP_CSV)

    print(f"Submitted single job_id = {job_id}")
    print(f"Saved job map -> {JOB_MAP_CSV}")
    print(f"Saved config  -> {CONFIG_JSON}")

if __name__ == "__main__":
    main()
