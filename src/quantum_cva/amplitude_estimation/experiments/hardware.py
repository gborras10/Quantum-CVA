from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, qasm3
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from quantum_cva.amplitude_estimation.experiments.circuits import (
    active_qubits,
    build_reference_circuits,
    circuit_metrics,
    construct_measured_circuit,
    two_qubit_count,
)
from quantum_cva.amplitude_estimation.experiments.io import (
    RunPaths,
    load_csv,
    load_json,
    save_csv,
    save_json,
    write_trace_bundle,
)
from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle
from quantum_cva.amplitude_estimation.experiments.samplers import (
    AerCountSampler,
    LoggedAerSampler,
    ReplayCountSampler,
    build_noise_model,
    count_good_from_counts,
    extract_result_counts,
    ideal_good_probability_for_circuit,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    run_algorithm_once,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (
    aggregate_budget_summary,
    as_float,
)
from quantum_cva.amplitude_estimation.experiments.traces import rows_at_budgets
from quantum_cva.quantum_hardware_utilities.transpile_utils import (
    DEFAULT_TRANSPILER_SEEDS,
    select_best_fixed_transpilation_plan,
)


def parse_int_list(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    return [int(x) for x in raw]


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
    replay_budget_rows: list[dict[str, Any]] = field(default_factory=list)
    budget_summary_rows: list[dict[str, Any]] = field(default_factory=list)
    calibration_summary: dict[str, Any] = field(default_factory=dict)
    session_details: dict[str, Any] = field(default_factory=dict)

    def persist(self) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
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
        save_csv(self.replay_budget_rows, self.paths.replay_budget)
        save_csv(self.budget_summary_rows, self.paths.budget_summary)
        self.paths.write_manifest()
        write_trace_bundle(
            self.paths.trace_bundle,
            trace_rows=self.replay_trace_rows or self.direct_trace_rows,
            budget_rows=self.replay_budget_rows,
            amplification_rows=self.amplification_point_rows,
        )


def create_run_dir(base_dir: str | Path, prefix: str = "ae_experiment") -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_existing_state(run_dir: str | Path) -> ExperimentState:
    paths = RunPaths(Path(run_dir))
    state = ExperimentState(paths=paths, config=load_json(paths.config, default={}) or {})
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
    state.replay_budget_rows = load_csv(paths.replay_budget) if paths.replay_budget.exists() else []
    state.budget_summary_rows = load_csv(paths.budget_summary) if paths.budget_summary.exists() else []
    state.calibration_summary = load_json(paths.calibration_summary, default={}) or {}
    state.session_details = load_json(paths.session_details, default={}) or {}
    return state


def backend_snapshot(backend: Any, *, mode: str, channel: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "mode": str(mode),
        "channel": str(channel),
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


def build_pass_manager_for_backend(
    backend: Any,
    bundle: AEProblemBundle,
    *,
    mode: str,
    optimization_level: int = 3,
    seed_transpiler: int = 1234,
    reference_ks: Sequence[int] = (0, 1, 2, 3, 4),
    routing_method: str | None = "sabre",
    discovery_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    evaluation_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    layout_search_strategy: str = "fast",
    verbose: bool = True,
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
            "reference_ks": [int(k) for k in reference_ks],
        }

    if verbose:
        print(
            "[pass_manager] building reference circuits for "
            f"k={list(int(k) for k in reference_ks)}",
            flush=True,
        )
    reference_circuits = build_reference_circuits(bundle.problem, reference_ks)
    strategy = str(layout_search_strategy).strip().lower()
    if strategy not in {"fast", "exhaustive", "preset"}:
        raise ValueError(
            "layout_search_strategy must be one of 'fast', 'exhaustive', or 'preset', "
            f"got {layout_search_strategy!r}."
        )

    try:
        if strategy == "preset":
            if verbose:
                print("[pass_manager] using Qiskit preset pass manager without fixed layout", flush=True)
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=int(optimization_level),
                routing_method=routing_method,
                seed_transpiler=int(seed_transpiler),
            )
            return pass_manager, {
                "strategy": "qiskit_preset",
                "fallback_used": False,
                "initial_layout": None,
                "seed_transpiler": int(seed_transpiler),
                "optimization_level": int(optimization_level),
                "routing_method": routing_method,
                "reference_ks": [int(k) for k in reference_ks],
            }

        if strategy == "fast":
            scoring_pairs = [
                (int(k), circuit)
                for k, circuit in zip(reference_ks, reference_circuits)
                if int(k) > 0
            ]
            if not scoring_pairs:
                scoring_pairs = [
                    (int(k), circuit)
                    for k, circuit in zip(reference_ks, reference_circuits)
                ]
            scoring_ks = [int(k) for k, _ in scoring_pairs]
            scoring_circuits = [circuit for _, circuit in scoring_pairs]
            if verbose:
                print(
                    "[pass_manager] fast layout selection: "
                    "single-seed Qiskit SABRE candidate search scored over "
                    f"k={scoring_ks}",
                    flush=True,
                )
            plan = select_best_fixed_transpilation_plan(
                backend,
                scoring_circuits,
                candidate_layouts=(),
                optimization_level=int(optimization_level),
                routing_method=routing_method,
                discovery_seeds=(int(seed_transpiler),),
                evaluation_seeds=(int(seed_transpiler),),
                include_sabre_candidates=True,
                verbose=bool(verbose),
            )
            if verbose:
                print(
                    "[pass_manager] fast layout selected: "
                    f"layout={plan.initial_layout}, "
                    f"swaps={plan.aggregate_swap_count}, "
                    f"2q={plan.aggregate_two_qubit_gates}, "
                    f"depth={plan.aggregate_depth}",
                    flush=True,
                )
            return plan.build_pass_manager(backend), {
                "strategy": "qiskit_sabre_fast_multik_fixed_layout",
                "fallback_used": False,
                "reference_ks": [int(k) for k in reference_ks],
                "scoring_ks": scoring_ks,
                "discovery_seeds": [int(seed_transpiler)],
                "evaluation_seeds": [int(seed_transpiler)],
                **plan.metadata(),
            }

        if verbose:
            print("[pass_manager] selecting best fixed-layout transpilation plan", flush=True)
        plan = select_best_fixed_transpilation_plan(
            backend,
            reference_circuits,
            candidate_layouts=(),
            optimization_level=int(optimization_level),
            routing_method=routing_method,
            discovery_seeds=tuple(int(seed) for seed in discovery_seeds),
            evaluation_seeds=tuple(int(seed) for seed in evaluation_seeds),
            include_sabre_candidates=True,
            verbose=bool(verbose),
        )
        return plan.build_pass_manager(backend), {
            "strategy": "fixed_layout_search",
            "fallback_used": False,
            "reference_ks": [int(k) for k in reference_ks],
            "discovery_seeds": [int(seed) for seed in discovery_seeds],
            "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
            **plan.metadata(),
        }
    except Exception as exc:
        if verbose:
            print(
                "[pass_manager] fixed-layout search failed; falling back to preset pass manager: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=int(optimization_level),
            seed_transpiler=int(seed_transpiler),
        )
        return pass_manager, {
            "strategy": "preset_default_fallback",
            "fallback_used": True,
            "fallback_reason": str(exc),
            "initial_layout": None,
            "seed_transpiler": int(seed_transpiler),
            "optimization_level": int(optimization_level),
            "routing_method": None,
            "reference_ks": [int(k) for k in reference_ks],
        }


def write_isa_qasm(circuit: QuantumCircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = qasm3.dumps(circuit)
    except Exception as exc:
        text = f"// QASM3 export failed: {exc}\n"
    path.write_text(text, encoding="utf-8")


def run_preflight(
    bundle: AEProblemBundle,
    pass_manager: Any,
    state: ExperimentState,
    *,
    max_grover_power: int,
    max_isa_depth: int,
    max_isa_2q: int,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    allowed_max = -1
    contiguous_allowed = True
    state.paths.qasm_dir.mkdir(parents=True, exist_ok=True)
    for k in range(int(max_grover_power) + 1):
        if verbose:
            print(
                f"[preflight] k={k}/{int(max_grover_power)}: "
                "constructing, transpiling, measuring metrics",
                flush=True,
            )
        logical = construct_measured_circuit(bundle.problem, k, source="preflight")
        decomposed = logical.decompose(reps=10)
        isa = pass_manager.run(decomposed)
        logical_metrics = circuit_metrics(logical)
        decomposed_metrics = circuit_metrics(decomposed)
        isa_metrics = circuit_metrics(isa)
        isa_2q = two_qubit_count(isa)
        p_ideal = ideal_good_probability_for_circuit(logical, bundle)
        row = {
            "grover_power": int(k),
            "amplification_factor": int(2 * k + 1),
            "p_ideal": float(p_ideal),
            "logical_depth": logical_metrics["depth"],
            "logical_2q": logical_metrics["two_qubit_gates"],
            "decomposed_depth": decomposed_metrics["depth"],
            "decomposed_2q": decomposed_metrics["two_qubit_gates"],
            "isa_depth": isa_metrics["depth"],
            "isa_size": isa_metrics["size"],
            "isa_2q": int(isa_2q),
            "isa_swap_count": isa_metrics["swap_count"],
            "active_physical_qubits": json.dumps(active_qubits(isa)),
            "within_depth_limit": bool(isa_metrics["depth"] <= int(max_isa_depth)),
            "within_2q_limit": bool(isa_2q <= int(max_isa_2q)),
        }
        if contiguous_allowed and row["within_depth_limit"] and row["within_2q_limit"]:
            allowed_max = int(k)
        else:
            contiguous_allowed = False
        rows.append(row)
        write_isa_qasm(isa, state.paths.qasm_dir / f"ae_k_{k:02d}_isa.qasm3")
        if verbose:
            print(
                f"[preflight] k={k}/{int(max_grover_power)} done: "
                f"isa_depth={isa_metrics['depth']}, isa_2q={int(isa_2q)}, "
                f"swaps={isa_metrics['swap_count']}, p_ideal={p_ideal:.6g}",
                flush=True,
            )
    state.config["max_grover_power_after_preflight"] = int(allowed_max)
    state.config["preflight_limits"] = {
        "max_isa_depth": int(max_isa_depth),
        "max_isa_2q": int(max_isa_2q),
    }
    state.amplification_point_rows = state.amplification_point_rows
    save_csv(rows, state.paths.transpilation_report)
    if allowed_max < 0:
        raise RuntimeError("No Grover power passed the ISA preflight limits.")
    return rows, allowed_max


def build_readout_circuits(bundle: AEProblemBundle) -> list[QuantumCircuit]:
    objective = list(bundle.problem.objective_qubits)
    num_qubits = max(
        int(bundle.problem.state_preparation.num_qubits),
        int(bundle.problem.grover_operator.num_qubits),
    )
    bad_bits = "0" * bundle.objective_width
    good_bits = str(bundle.good_bitstring)
    circuits: list[QuantumCircuit] = []
    for label, bitstring in (("bad", bad_bits), ("good", good_bits)):
        circuit = QuantumCircuit(num_qubits, name=f"readout_prepare_{label}")
        for qubit, bit in zip(objective, bitstring):
            if bit == "1":
                circuit.x(int(qubit))
        classical = ClassicalRegister(len(objective), "c0")
        circuit.add_register(classical)
        circuit.measure(objective, classical[:])
        circuit.metadata = {
            "source": "readout_calibration",
            "prepared_label": label,
            "prepared_bitstring": bitstring,
            "grover_power": 0,
            "amplification_factor": 1,
        }
        circuits.append(circuit)
    return circuits


def run_readout_calibration(
    sampler: Any,
    bundle: AEProblemBundle,
    *,
    shots: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if hasattr(sampler, "set_context"):
        sampler.set_context("readout_calibration")
    circuits = build_readout_circuits(bundle)
    result = sampler.run(circuits, shots=int(shots)).result()
    rows: list[dict[str, Any]] = []
    p_obs: dict[str, float] = {}
    for idx, label in enumerate(("bad", "good")):
        counts = extract_result_counts(result, idx)
        good = count_good_from_counts(counts, bundle)
        total = int(sum(counts.values()))
        p_good = float(good / max(total, 1))
        p_obs[label] = p_good
        rows.append(
            {
                "prepared_label": label,
                "prepared_bitstring": circuits[idx].metadata["prepared_bitstring"],
                "shots": total,
                "counts_json": json.dumps(counts, sort_keys=True),
                "good_counts": int(good),
                "p_observed_good": p_good,
            }
        )
    denom = p_obs["good"] - p_obs["bad"]
    return rows, {
        "p_obs_good_given_bad": float(p_obs["bad"]),
        "p_obs_good_given_good": float(p_obs["good"]),
        "readout_denom": float(denom),
        "readout_usable": float(abs(denom) > 0.05),
    }


def mitigate_readout_probability(p_raw: float, readout: Mapping[str, float]) -> float:
    denom = float(readout.get("readout_denom", 1.0))
    p_bad = float(readout.get("p_obs_good_given_bad", 0.0))
    if abs(denom) <= 0.05:
        return float(np.clip(p_raw, 0.0, 1.0))
    return float(np.clip((float(p_raw) - p_bad) / denom, 0.0, 1.0))


def run_amplification_scan(
    sampler: Any,
    bundle: AEProblemBundle,
    *,
    grover_powers: Sequence[int],
    repeats: int,
    shots: int,
    seed: int,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    schedule = [(int(k), int(rep)) for k in grover_powers for rep in range(int(repeats))]
    rng.shuffle(schedule)
    if hasattr(sampler, "set_context"):
        sampler.set_context("amplification_scan")
    rows: list[dict[str, Any]] = []
    for batch_index, (k, repeat_index) in enumerate(schedule):
        if verbose:
            print(
                f"[amplification_scan] batch {batch_index + 1}/{len(schedule)} "
                f"k={k} repeat={repeat_index} shots={int(shots)}",
                flush=True,
            )
        circuit = construct_measured_circuit(bundle.problem, k, source="amplification_scan")
        result = sampler.run([circuit], shots=int(shots)).result()
        counts = extract_result_counts(result, 0)
        good = count_good_from_counts(counts, bundle)
        total = int(sum(counts.values()))
        rows.append(
            {
                "batch_index": int(batch_index),
                "repeat_index": int(repeat_index),
                "grover_power": int(k),
                "amplification_factor": int(2 * k + 1),
                "shots": total,
                "good_counts": int(good),
                "bad_counts": int(total - good),
                "p_raw": float(good / max(total, 1)),
                "counts_json": json.dumps(counts, sort_keys=True),
            }
        )
    return rows


def _contrast_diagnostics_for_baseline(
    point: Mapping[str, Any],
    baseline: float,
    *,
    min_ideal_offset: float,
    min_fit_contrast_z: float,
) -> dict[str, Any]:
    p_ideal = float(point["p_ideal"])
    p_mitigated = float(point["p_hw_mitigated"])
    mitigated_se = float(point["p_hw_mitigated_se"])
    denom = p_ideal - float(baseline)
    contrast = np.nan
    contrast_se = np.nan
    contrast_z = np.nan
    contrast_relative_se = np.nan
    if abs(denom) > 1e-12:
        contrast = (p_mitigated - float(baseline)) / denom
        contrast_se = mitigated_se / abs(denom)
        if np.isfinite(contrast) and contrast > 0.0:
            contrast_z = contrast / max(contrast_se, 1e-12)
            contrast_relative_se = contrast_se / max(contrast, 1e-12)
    used = (
        abs(denom) >= float(min_ideal_offset)
        and np.isfinite(contrast)
        and 0.0 < contrast < 1.0
        and np.isfinite(contrast_z)
        and contrast_z >= float(min_fit_contrast_z)
    )
    return {
        "contrast_mitigated": float(contrast) if np.isfinite(contrast) else np.nan,
        "contrast_mitigated_se": float(contrast_se) if np.isfinite(contrast_se) else np.nan,
        "contrast_signal_z": float(contrast_z) if np.isfinite(contrast_z) else np.nan,
        "contrast_relative_se": float(contrast_relative_se)
        if np.isfinite(contrast_relative_se)
        else np.nan,
        "used_in_fit": bool(used),
    }


def _weighted_log_contrast_fit(
    points: Sequence[Mapping[str, Any]],
    *,
    baseline: float,
    min_ideal_offset: float,
    min_fit_contrast_z: float,
) -> dict[str, Any] | None:
    fit_x: list[float] = []
    fit_y: list[float] = []
    fit_w: list[float] = []
    fit_ks: list[int] = []
    for point in points:
        diagnostics = _contrast_diagnostics_for_baseline(
            point,
            float(baseline),
            min_ideal_offset=float(min_ideal_offset),
            min_fit_contrast_z=float(min_fit_contrast_z),
        )
        if not bool(diagnostics["used_in_fit"]):
            continue
        contrast = float(diagnostics["contrast_mitigated"])
        contrast_se = float(diagnostics["contrast_mitigated_se"])
        log_se = max(contrast_se / max(contrast, 1e-12), 1e-6)
        fit_x.append(float(point["amplification_factor"]))
        fit_y.append(float(np.log(contrast)))
        fit_w.append(float(min(1.0 / (log_se * log_se), 1e6)))
        fit_ks.append(int(point["grover_power"]))

    if len(fit_x) < 2:
        return None
    x = np.asarray(fit_x, dtype=float)
    y = np.asarray(fit_y, dtype=float)
    w = np.asarray(fit_w, dtype=float)
    slope_free, intercept_free = np.polyfit(x, y, deg=1, w=np.sqrt(w))
    if not np.isfinite(slope_free) or slope_free >= 0.0:
        return None
    residual = y - (slope_free * x + intercept_free)
    dof = max(1, len(x) - 2)
    reduced_weighted_sse = float(np.sum(w * residual * residual) / float(dof))
    slope_zero = float(np.sum(x * y) / np.sum(x * x))
    t_zero = float(-1.0 / slope_zero) if slope_zero < 0.0 else None
    return {
        "fit_points": int(len(x)),
        "fit_ks": fit_ks,
        "t_eff_zero_intercept": t_zero,
        "t_eff_free_intercept": float(-1.0 / slope_free),
        "contrast_prefactor": float(np.exp(intercept_free)),
        "free_intercept_slope": float(slope_free),
        "fit_reduced_weighted_sse": reduced_weighted_sse,
    }


def _fit_contrast_baseline(
    points: Sequence[Mapping[str, Any]],
    *,
    min_ideal_offset: float,
    min_fit_contrast_z: float,
    min_baseline_fit_points: int,
) -> tuple[float, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for baseline in np.linspace(0.0, 1.0, 1001):
        fit = _weighted_log_contrast_fit(
            points,
            baseline=float(baseline),
            min_ideal_offset=float(min_ideal_offset),
            min_fit_contrast_z=float(min_fit_contrast_z),
        )
        if fit is None or int(fit["fit_points"]) < int(min_baseline_fit_points):
            continue
        candidate = dict(fit)
        candidate["contrast_baseline"] = float(baseline)
        candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "Could not fit contrast baseline: not enough valid contrast points. "
            "Increase --max-grover-power, --scan-shots, or lower --min-ideal-offset."
        )

    max_points = max(int(candidate["fit_points"]) for candidate in candidates)
    min_points_to_keep = max(
        int(min_baseline_fit_points),
        int(np.ceil(0.8 * float(max_points))),
    )
    stable_candidates = [
        candidate
        for candidate in candidates
        if int(candidate["fit_points"]) >= min_points_to_keep
    ]
    best = min(
        stable_candidates,
        key=lambda candidate: float(candidate["fit_reduced_weighted_sse"]),
    )

    best_baseline = float(best["contrast_baseline"])
    refined_low = max(0.0, best_baseline - 0.002)
    refined_high = min(1.0, best_baseline + 0.002)
    refined_candidates: list[dict[str, Any]] = []
    for baseline in np.linspace(refined_low, refined_high, 401):
        fit = _weighted_log_contrast_fit(
            points,
            baseline=float(baseline),
            min_ideal_offset=float(min_ideal_offset),
            min_fit_contrast_z=float(min_fit_contrast_z),
        )
        if (
            fit is None
            or int(fit["fit_points"]) < min_points_to_keep
        ):
            continue
        candidate = dict(fit)
        candidate["contrast_baseline"] = float(baseline)
        refined_candidates.append(candidate)
    if refined_candidates:
        best = min(
            refined_candidates,
            key=lambda candidate: float(candidate["fit_reduced_weighted_sse"]),
        )

    return float(best["contrast_baseline"]), {
        "contrast_baseline_fit_points": int(best["fit_points"]),
        "contrast_baseline_fit_ks": [int(k) for k in best["fit_ks"]],
        "contrast_baseline_fit_reduced_weighted_sse": float(
            best["fit_reduced_weighted_sse"]
        ),
        "contrast_baseline_fit_candidate_count": int(len(candidates)),
        "contrast_baseline_fit_min_points_kept": int(min_points_to_keep),
    }


def analyze_amplification(
    count_rows: Sequence[Mapping[str, Any]],
    bundle: AEProblemBundle,
    readout: Mapping[str, float],
    *,
    min_ideal_offset: float = 0.15,
    contrast_baseline: float | str = 0.5,
    min_fit_contrast_z: float = 2.0,
    min_visible_contrast_z: float = 3.0,
    min_baseline_fit_points: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, float]]:
    baseline_mode = str(contrast_baseline).strip().lower()
    fit_baseline = baseline_mode in {"fit", "fitted", "estimate", "estimated"}
    by_k: dict[int, list[Mapping[str, Any]]] = {}
    for row in count_rows:
        by_k.setdefault(int(as_float(row["grover_power"])), []).append(row)

    raw_points: list[dict[str, Any]] = []
    p_replay_by_k: dict[int, float] = {}
    for k in sorted(by_k):
        rows = by_k[k]
        good = int(sum(int(as_float(r.get("good_counts"))) for r in rows))
        shots = int(sum(int(as_float(r.get("shots"))) for r in rows))
        p_raw = float(good / max(shots, 1))
        p_mitigated = mitigate_readout_probability(p_raw, readout)
        p_replay_by_k[k] = p_mitigated
        circuit = construct_measured_circuit(bundle.problem, k, source="amplification_analysis")
        p_ideal = ideal_good_probability_for_circuit(circuit, bundle)
        raw_se = float(np.sqrt(max(p_raw * (1.0 - p_raw), 0.0) / max(shots, 1)))
        readout_denom = abs(float(readout.get("readout_denom", 1.0)))
        mitigated_se = raw_se / max(readout_denom, 0.05)
        raw_points.append(
            {
                "grover_power": int(k),
                "amplification_factor": int(2 * k + 1),
                "shots": int(shots),
                "p_ideal": float(p_ideal),
                "p_raw": float(p_raw),
                "p_hw_mitigated": float(p_mitigated),
                "p_raw_se": raw_se,
                "p_hw_mitigated_se": float(mitigated_se),
            }
        )

    baseline_fit_metadata: dict[str, Any] = {}
    if fit_baseline:
        baseline, baseline_fit_metadata = _fit_contrast_baseline(
            raw_points,
            min_ideal_offset=float(min_ideal_offset),
            min_fit_contrast_z=float(min_fit_contrast_z),
            min_baseline_fit_points=int(min_baseline_fit_points),
        )
        baseline_mode_out = "fitted"
    else:
        baseline = float(contrast_baseline)
        if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise ValueError(
                "contrast_baseline must be a finite probability in [0, 1] "
                f"or 'fit', got {contrast_baseline}."
            )
        baseline_mode_out = "fixed"

    points: list[dict[str, Any]] = []
    for raw in raw_points:
        diagnostics = _contrast_diagnostics_for_baseline(
            raw,
            float(baseline),
            min_ideal_offset=float(min_ideal_offset),
            min_fit_contrast_z=float(min_fit_contrast_z),
        )
        signal_z_from_baseline = abs(
            float(raw["p_hw_mitigated"]) - float(baseline)
        ) / max(float(raw["p_hw_mitigated_se"]), 1e-12)
        signal_z_from_half = abs(float(raw["p_hw_mitigated"]) - 0.5) / max(
            float(raw["p_hw_mitigated_se"]),
            1e-12,
        )
        visible_by_contrast = bool(
            diagnostics["used_in_fit"]
            and np.isfinite(float(diagnostics["contrast_signal_z"]))
            and float(diagnostics["contrast_signal_z"]) >= float(min_visible_contrast_z)
        )
        points.append(
            {
                **raw,
                "contrast_baseline": float(baseline),
                **diagnostics,
                "signal_z_from_baseline": float(signal_z_from_baseline),
                "signal_z_from_half": float(signal_z_from_half),
                "visible_by_contrast": bool(visible_by_contrast),
            }
        )

    fit = _weighted_log_contrast_fit(
        raw_points,
        baseline=float(baseline),
        min_ideal_offset=float(min_ideal_offset),
        min_fit_contrast_z=float(min_fit_contrast_z),
    )
    fit_ks = [] if fit is None else [int(k) for k in fit["fit_ks"]]
    robust_visible = [
        int(p["grover_power"])
        for p in points
        if bool(p["visible_by_contrast"])
    ]
    signal_visible = [
        int(p["grover_power"])
        for p in points
        if float(p["signal_z_from_baseline"]) >= 3.0
    ]
    summary: dict[str, Any] = {
        "calibration_status": "insufficient_fit_points",
        "fit_points": 0 if fit is None else int(fit["fit_points"]),
        "t_eff_zero_intercept": None,
        "t_eff_free_intercept": None,
        "contrast_prefactor": None,
        "free_intercept_slope": None,
        "k_visible": 0,
        "k_visible_criterion": (
            "max k with valid contrast, contrast_signal_z >= "
            f"{float(min_visible_contrast_z):.6g}, and "
            f"|p_ideal - baseline| >= {float(min_ideal_offset):.6g}"
        ),
        "k_signal_from_baseline": int(max(signal_visible)) if signal_visible else 0,
        "k_contrast_fit_max": int(max(fit_ks)) if fit_ks else 0,
        "min_fit_contrast_z": float(min_fit_contrast_z),
        "min_visible_contrast_z": float(min_visible_contrast_z),
        "min_ideal_offset": float(min_ideal_offset),
        "contrast_baseline": float(baseline),
        "contrast_baseline_mode": baseline_mode_out,
        **baseline_fit_metadata,
    }
    if robust_visible:
        summary["k_visible"] = int(max(robust_visible))
    if fit is not None:
        summary["t_eff_zero_intercept"] = fit["t_eff_zero_intercept"]
        summary["t_eff_free_intercept"] = fit["t_eff_free_intercept"]
        summary["contrast_prefactor"] = fit["contrast_prefactor"]
        summary["free_intercept_slope"] = fit["free_intercept_slope"]
        summary["fit_reduced_weighted_sse"] = fit["fit_reduced_weighted_sse"]
        if fit["t_eff_zero_intercept"] is not None:
            summary["calibration_status"] = "ok"
    return points, summary, p_replay_by_k


def effective_t_for_algorithms(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("t_eff_zero_intercept")
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f) or value_f <= 0.0:
        return None
    return value_f


def make_replay_probability_extrapolator(
    bundle: AEProblemBundle,
    calibration_summary: Mapping[str, Any],
) -> tuple[Callable[[int], float], dict[str, Any]]:
    baseline = float(calibration_summary.get("contrast_baseline", 0.5))
    prefactor = calibration_summary.get("contrast_prefactor")
    t_free = calibration_summary.get("t_eff_free_intercept")
    if prefactor is not None and t_free is not None:
        prefactor_f = float(prefactor)
        t_eff = float(t_free)
        model = "free_intercept"
    else:
        t_zero = calibration_summary.get("t_eff_zero_intercept")
        if t_zero is None:
            raise ValueError("Cannot extrapolate replay probabilities without valid T_eff.")
        prefactor_f = 1.0
        t_eff = float(t_zero)
        model = "zero_intercept"
    theta = float(np.arcsin(np.sqrt(np.clip(bundle.true_amplitude, 0.0, 1.0))))

    def _extrapolate(k: int) -> float:
        amplification_factor = 2 * int(k) + 1
        contrast = float(np.clip(prefactor_f * np.exp(-amplification_factor / t_eff), 0.0, 1.0))
        p_ideal = float(np.sin(amplification_factor * theta) ** 2)
        return float(np.clip(baseline + contrast * (p_ideal - baseline), 0.0, 1.0))

    return _extrapolate, {
        "model": model,
        "contrast_prefactor": float(prefactor_f),
        "t_eff": float(t_eff),
        "contrast_baseline": float(baseline),
        "a_true": float(bundle.true_amplitude),
    }


def sample_replay_probabilities(
    p_by_k: Mapping[int, float],
    p_se_by_k: Mapping[int, float] | None,
    *,
    mode: str,
    rng: np.random.Generator,
    se_scale: float,
) -> dict[int, float]:
    if mode == "fixed":
        return {int(k): float(v) for k, v in p_by_k.items()}
    if mode != "normal":
        raise ValueError(f"Unknown replay probability mode: {mode!r}.")
    if not p_se_by_k:
        raise ValueError("Replay probability mode 'normal' requires standard errors.")
    sampled: dict[int, float] = {}
    for k, p in p_by_k.items():
        sigma = max(0.0, float(se_scale) * float(p_se_by_k[int(k)]))
        sampled[int(k)] = float(np.clip(rng.normal(float(p), sigma), 0.0, 1.0))
    return sampled


def run_replay(
    state: ExperimentState,
    bundle: AEProblemBundle,
    *,
    algorithms: Sequence[str],
    algorithm_labels: Mapping[str, str] = ALGORITHM_LABELS,
    p_by_k: Mapping[int, float],
    p_se_by_k: Mapping[int, float] | None,
    replay_probability_mode: str,
    replay_probability_se_scale: float,
    budgets: Sequence[int],
    repetitions: int,
    n_shots: int,
    epsilon_target: float,
    alpha: float,
    t_eff: float | None,
    seed: int,
    extrapolate: bool = False,
    verbose: bool = False,
) -> None:
    state.replay_trace_rows.clear()
    state.replay_final_rows.clear()
    state.replay_budget_rows.clear()
    state.budget_summary_rows.clear()
    max_queries = max(int(x) for x in budgets)
    extrapolate_probability: Callable[[int], float] | None = None
    extrapolated_cache: dict[int, float] = {}
    if extrapolate:
        extrapolate_probability, metadata = make_replay_probability_extrapolator(
            bundle,
            state.calibration_summary,
        )
        state.config["replay_extrapolation_model"] = metadata
    contrast_baseline = float(
        state.calibration_summary.get(
            "contrast_baseline",
            state.config.get("contrast_baseline", 0.5),
        )
    )
    for rep in range(int(repetitions)):
        replay_rng = np.random.default_rng(int(seed) + 7919 * rep)
        rep_p_by_k = sample_replay_probabilities(
            p_by_k,
            p_se_by_k,
            mode=str(replay_probability_mode),
            rng=replay_rng,
            se_scale=float(replay_probability_se_scale),
        )
        for alg_index, algorithm in enumerate(algorithms):
            sampler = ReplayCountSampler(
                rep_p_by_k,
                bundle,
                seed=int(seed) + 1009 * rep + 17 * alg_index,
                extrapolate_probability=extrapolate_probability,
                extrapolated_cache=extrapolated_cache,
            )
            try:
                trace_rows, final_row = run_algorithm_once(
                    algorithm,
                    sampler,
                    bundle,
                    run_kind="hardware_replay",
                    repetition=rep,
                    epsilon_target=float(epsilon_target),
                    alpha=float(alpha),
                    n_shots=int(n_shots),
                    max_queries=int(max_queries),
                    t_eff=t_eff,
                    seed=int(seed) + rep + alg_index,
                    algorithm_labels=algorithm_labels,
                    solver_kwargs={"noise_floor": contrast_baseline},
                )
                if extrapolate and sampler.extrapolated_ks_used:
                    used = {int(k) for k in sampler.extrapolated_ks_used}
                    for row in trace_rows:
                        row["replay_probability_source"] = (
                            "extrapolated" if int(row["grover_power"]) in used else "measured"
                        )
                        row["replay_probability_extrapolated"] = int(row["grover_power"]) in used
                    final_row["extrapolated_replay_ks_json"] = json.dumps(sorted(used))
                    final_row["n_extrapolated_replay_ks"] = len(used)
                state.replay_trace_rows.extend(trace_rows)
                state.replay_final_rows.append(final_row)
                state.replay_budget_rows.extend(
                    rows_at_budgets(trace_rows, budgets, run_kind="hardware_replay")
                )
            except Exception as exc:
                state.error_rows.append(
                    {
                        "phase": "hardware_replay",
                        "algorithm": str(algorithm),
                        "repetition": rep,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "timestamp_epoch": time.time(),
                    }
                )
        if verbose and rep % 10 == 0:
            print(f"[hardware_replay] completed repetition {rep + 1}/{int(repetitions)}")
    state.budget_summary_rows = aggregate_budget_summary(
        state.replay_budget_rows,
        total_repetitions=int(repetitions),
        group_by_budget=True,
    )
    if extrapolated_cache:
        state.config["replay_extrapolated_probabilities"] = {
            str(k): float(v) for k, v in sorted(extrapolated_cache.items())
        }
    state.persist()


def run_dry_run_experiment(
    bundle: AEProblemBundle,
    *,
    run_dir: str | Path,
    algorithms: Sequence[str] = ("cabiqae_latentt", "biqae", "bae"),
    budgets: Sequence[int] = (128, 256, 512, 1024, 2048),
    max_grover_power: int = 8,
    scan_repeats: int = 1,
    scan_shots: int = 256,
    readout_shots: int = 512,
    direct_shots: int = 64,
    replay_repetitions: int = 20,
    epsilon_target: float = 0.08,
    alpha: float = 0.10,
    seed: int = 12345,
    noise_scale: float = 1.0,
    noise_profile: str = "projected",
    contrast_baseline: float = 0.5,
) -> ExperimentState:
    paths = RunPaths(Path(run_dir))
    state = ExperimentState(
        paths=paths,
        config={
            "run_id": paths.run_dir.name,
            "run_uuid": str(uuid.uuid4()),
            "mode": "dry-run",
            "target_name": bundle.target_name,
            "a_true": float(bundle.true_amplitude),
            "processed_true_value": float(bundle.processed_true_value),
            "algorithms": list(algorithms),
            "budgets": [int(x) for x in budgets],
            "seed": int(seed),
            "contrast_baseline": float(contrast_baseline),
        },
    )
    backend = AerSimulator()
    save_json(backend_snapshot(backend, mode="aer_simulator", channel="local"), paths.backend_snapshot)
    pass_manager, transpilation_metadata = build_pass_manager_for_backend(
        backend,
        bundle,
        mode="dry-run",
        reference_ks=tuple(range(min(int(max_grover_power), 4) + 1)),
    )
    state.config["transpilation"] = transpilation_metadata
    run_preflight(
        bundle,
        pass_manager,
        state,
        max_grover_power=int(max_grover_power),
        max_isa_depth=10**9,
        max_isa_2q=10**9,
    )
    noise_model = build_noise_model(float(noise_scale), profile=noise_profile)
    aer = AerCountSampler(
        noise_model=noise_model,
        seed=int(seed),
        method="density_matrix",
        transpile_backend=backend,
        pass_manager=pass_manager,
    )
    sampler = LoggedAerSampler(aer, state.job_rows)
    readout_rows, readout_params = run_readout_calibration(
        sampler,
        bundle,
        shots=int(readout_shots),
    )
    state.readout_rows = readout_rows
    state.calibration_summary["readout"] = readout_params
    state.amplification_count_rows = run_amplification_scan(
        sampler,
        bundle,
        grover_powers=list(range(int(max_grover_power) + 1)),
        repeats=int(scan_repeats),
        shots=int(scan_shots),
        seed=int(seed),
    )
    points, calibration, _ = analyze_amplification(
        state.amplification_count_rows,
        bundle,
        readout_params,
        contrast_baseline=float(contrast_baseline),
    )
    state.amplification_point_rows = points
    state.calibration_summary.update(calibration)
    t_eff = effective_t_for_algorithms(state.calibration_summary)
    p_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated"]) for r in points}
    p_se_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated_se"]) for r in points}
    run_replay(
        state,
        bundle,
        algorithms=algorithms,
        p_by_k=p_by_k,
        p_se_by_k=p_se_by_k,
        replay_probability_mode="normal",
        replay_probability_se_scale=1.0,
        budgets=budgets,
        repetitions=int(replay_repetitions),
        n_shots=int(direct_shots),
        epsilon_target=float(epsilon_target),
        alpha=float(alpha),
        t_eff=t_eff,
        seed=int(seed),
    )
    return state
