from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, LogFormatterMathtext, LogLocator
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize


def _bootstrap_src_path() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


REPO_ROOT = _bootstrap_src_path()

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (  # noqa: E402
    MLQcbmCircuit,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (  # noqa: E402
    build_undirected_coupling_graph,
    find_best_qcbm_heavyhex6,
)


# ======================================================================
# Experiment configuration
# ======================================================================

BACKEND_NAME = "ibm_basquecountry"
RUNTIME_CHANNEL = "ibm_cloud"
USE_FRACTIONAL_GATES = True
NOISE_SNAPSHOT_ISO_UTC = "2026-04-07T12:10:00+00:00"

LOGICAL_TOPOLOGY = "qcbm_heavyhex6"
ENTANGLER = "rzz"
LAYERS_GRID = tuple(range(2, 17, 2))

EPS_COST = 1e-9
INIT_SCALE = 0.01
THETA_SEED = 42
SEED_TRANSPILER = 1234

OPTIMIZATION_LEVEL = 3
LAYOUT_METHOD = "trivial"
ROUTING_METHOD = "none"

# Keep the same optimizer family and objective as the current 6q
# statevector training script.
STAGE1_MAXITER = 600  
STAGE1_RHOBEG = 0.25
STAGE1_TOL = 1e-6
STAGE2_MAXITER = 10000
STAGE2_MAXFUN = 5000000

SIMULATOR_SEED = 20260407
NOISY_EVAL_RUNS = 10
NOISY_EVAL_SHOTS = 100000
NOISY_EVAL_SEED_BASE = 42
DIRICHLET_ALPHA = 1.0

READOUT_QUANTILE = 0.95
LOCAL_2Q_QUANTILE = 0.95

BENCHMARK_RELATIVE_PATH = (
    "data/multi_asset/6q_instance/benchmark/three_asset_instance.npz"
)
RESULTS_RELATIVE_DIR = (
    "cva_pricing_pipeline/multi_asset/6q_instance/training_multi_asset/"
    "state_preparation/layers_comparison/results/ideal"
)

PLOT_SELECTED_LAYERS = (2, 4, 6, 8, 10, 12, 14, 16)
LAYER_COLORS = {
    2: "#1f77b4",
    4: "#ff7f0e",
    6: "#2ca02c",
    8: "#d62728",
    10: "#9467bd",
    12: "#8c564b",
    14: "#e377c2",
    16: "#7f7f7f",
}
PAPER_LAYER_COLORS = {
    2: "#0072B2",
    4: "#D55E00",
    6: "#009E73",
    8: "#CC79A7",
    10: "#56B4E9",
    12: "#E69F00",
    14: "#332288",
    16: "#117733",
}
TRAINING_CURVE_MAX_ITERATION = 2500
TRAINING_CURVE_XMAX = 2625
TRAINING_CURVE_CONTINUATION_STEM = "training_curve_continuation"


@dataclass
class LayerArtifact:
    n_layers: int
    result_path: pathlib.Path
    kl_history: np.ndarray
    best_kl_history: np.ndarray
    p_star: np.ndarray
    p_init: np.ndarray
    p_target: np.ndarray
    kl_noisy_values: np.ndarray | None = None
    p_noisy_mean: np.ndarray | None = None


# ======================================================================
# Numerical and serialization helpers
# ======================================================================


def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
    dt = datetime.fromisoformat(snapshot_iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _as_1d_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _normalize_distribution(values: np.ndarray, *, eps: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if eps is not None:
        arr = np.maximum(arr, float(eps))
    else:
        arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot normalize an empty or non-positive distribution.")
    return arr / total


def kl_divergence(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    eps: float = EPS_COST,
) -> float:
    p = _normalize_distribution(target, eps=eps)
    q = _normalize_distribution(predicted, eps=eps)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch for KL: {p.shape} vs {q.shape}.")
    return float(np.sum(p * np.log(p / q)))


def target_entropy(target: np.ndarray, *, eps: float = EPS_COST) -> float:
    p = _normalize_distribution(target, eps=None)
    return float(-np.sum(p * np.log(np.clip(p, eps, 1.0))))


def _dirichlet_smooth(
    p: np.ndarray,
    *,
    shots: int,
    alpha: float,
) -> np.ndarray:
    """Match the finite-shot smoothing used in noisy MPS/QCBM evaluations."""
    p = np.asarray(p, dtype=float).ravel()
    if alpha <= 0.0:
        return _normalize_distribution(p, eps=None)

    dim = int(p.size)
    out = (float(shots) * p + float(alpha)) / (
        float(shots) + float(alpha) * float(dim)
    )
    return _normalize_distribution(out, eps=None)


def kl_contributions(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    eps: float = EPS_COST,
) -> np.ndarray:
    p = _normalize_distribution(target, eps=eps)
    q = _normalize_distribution(predicted, eps=eps)
    return p * np.log(p / q)


def kl_mass_contribution(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    mass_threshold: float,
    eps: float = EPS_COST,
) -> tuple[float, float, int]:
    p = _normalize_distribution(target, eps=None)
    contrib = kl_contributions(target, predicted, eps=eps)
    order = np.argsort(p)[::-1]
    cumulative = np.cumsum(p[order])
    cutoff = int(np.searchsorted(cumulative, mass_threshold, side="left"))
    selected = order[: min(cutoff + 1, order.size)]
    mask = np.zeros(order.size, dtype=bool)
    mask[selected] = True
    top = float(np.sum(contrib[mask]))
    tail = float(np.sum(contrib[~mask]))
    return top, tail, int(mask.sum())


def kl_time_decomposition(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    n_time: int,
    eps: float = EPS_COST,
) -> dict[str, float]:
    p = _normalize_distribution(target, eps=eps)
    q = _normalize_distribution(predicted, eps=eps)
    if p.size % int(n_time) != 0:
        return {
            "kl_time_marginal": math.nan,
            "kl_time_conditional": math.nan,
            "kl_time_conditional_max": math.nan,
        }

    n_state = p.size // int(n_time)
    p2 = p.reshape(int(n_time), n_state)
    q2 = q.reshape(int(n_time), n_state)

    p_t = _normalize_distribution(p2.sum(axis=1), eps=eps)
    q_t = _normalize_distribution(q2.sum(axis=1), eps=eps)
    kl_time = kl_divergence(p_t, q_t, eps=eps)

    conditional_terms: list[float] = []
    weighted_conditional = 0.0
    for t_idx in range(int(n_time)):
        p_weight = float(p_t[t_idx])
        p_cond = _normalize_distribution(p2[t_idx], eps=eps)
        q_cond = _normalize_distribution(q2[t_idx], eps=eps)
        kl_cond_t = kl_divergence(p_cond, q_cond, eps=eps)
        conditional_terms.append(kl_cond_t)
        weighted_conditional += p_weight * kl_cond_t

    return {
        "kl_time_marginal": float(kl_time),
        "kl_time_conditional": float(weighted_conditional),
        "kl_time_conditional_max": float(np.max(conditional_terms)),
    }


def _format_float(x: Any, precision: int = 4) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return ""
    if xf == 0:
        return "0"
    if abs(xf) < 1e-3 or abs(xf) >= 1e4:
        return f"{xf:.{precision}e}"
    return f"{xf:.{precision}f}"


# ======================================================================
# Snapshot-aware layout helpers
# ======================================================================


def _safe_log_pos(x: float, floor: float = 1e-16) -> float:
    return math.log(max(float(x), floor))


def _extract_error_from_gate_entry(gate_entry: dict[str, Any]) -> float:
    params = gate_entry.get("parameters", [])
    for param in params:
        name = str(param.get("name", "")).lower()
        if "error" in name:
            return float(param["value"])
    if params:
        return float(params[0]["value"])
    return math.nan


def _qubit_param_map(qubit_entry: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for param in qubit_entry:
        try:
            out[str(param["name"])] = float(param["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _snapshot_quality_maps(
    backend,
    backend_props,
    *,
    readout_quantile: float,
    local_2q_quantile: float,
) -> tuple[dict[int, float], set[int], dict[tuple[int, int], float], dict[str, Any]]:
    props = backend_props.to_dict()
    coupling_map = backend.configuration().coupling_map
    graph = build_undirected_coupling_graph(coupling_map)

    qubit_metrics: dict[int, dict[str, float]] = {}
    for qubit, entry in enumerate(props["qubits"]):
        values = _qubit_param_map(entry)
        readout_error = values.get("readout_error", math.nan)
        if not np.isfinite(readout_error):
            p01 = values.get("prob_meas0_prep1", math.nan)
            p10 = values.get("prob_meas1_prep0", math.nan)
            if np.isfinite(p01) and np.isfinite(p10):
                readout_error = 0.5 * (p01 + p10)

        qubit_metrics[qubit] = {
            "T1": values.get("T1", math.nan),
            "T2": values.get("T2", math.nan),
            "readout_error": readout_error,
        }

    edge_error_map: dict[tuple[int, int], float] = {}
    for gate in props.get("gates", []):
        qubits = gate.get("qubits", [])
        if len(qubits) != 2:
            continue
        edge = tuple(sorted((int(qubits[0]), int(qubits[1]))))
        error = _extract_error_from_gate_entry(gate)
        if edge not in edge_error_map or (
            np.isfinite(error) and error < edge_error_map[edge]
        ):
            edge_error_map[edge] = float(error)

    local_mean_2q: dict[int, float] = {}
    for qubit in graph.nodes:
        errors = [
            edge_error_map.get(tuple(sorted((qubit, neighbor))), math.nan)
            for neighbor in graph.neighbors(qubit)
        ]
        finite_errors = [err for err in errors if np.isfinite(err)]
        local_mean_2q[qubit] = (
            float(np.mean(finite_errors)) if finite_errors else math.nan
        )

    preferred_scores: dict[int, float] = {}
    for qubit in graph.nodes:
        t1 = qubit_metrics[qubit]["T1"]
        t2 = qubit_metrics[qubit]["T2"]
        readout_error = qubit_metrics[qubit]["readout_error"]
        mean_2q_error = local_mean_2q[qubit]

        score = 0.0
        if np.isfinite(t1):
            score += 0.35 * _safe_log_pos(t1)
        if np.isfinite(t2):
            score += 0.35 * _safe_log_pos(t2)
        if np.isfinite(readout_error):
            score += 0.20 * (-_safe_log_pos(readout_error))
        if np.isfinite(mean_2q_error):
            score += 0.10 * (-_safe_log_pos(mean_2q_error))
        preferred_scores[qubit] = float(score)

    edge_scores = {
        edge: float(-_safe_log_pos(error)) if np.isfinite(error) else -1e6
        for edge, error in edge_error_map.items()
    }

    readout_values = np.asarray(
        [
            qubit_metrics[q]["readout_error"]
            for q in graph.nodes
            if np.isfinite(qubit_metrics[q]["readout_error"])
        ],
        dtype=float,
    )
    local_2q_values = np.asarray(
        [local_mean_2q[q] for q in graph.nodes if np.isfinite(local_mean_2q[q])],
        dtype=float,
    )
    readout_threshold = (
        float(np.quantile(readout_values, readout_quantile))
        if readout_values.size
        else math.inf
    )
    local_2q_threshold = (
        float(np.quantile(local_2q_values, local_2q_quantile))
        if local_2q_values.size
        else math.inf
    )

    avoided_qubits = {
        q
        for q in graph.nodes
        if (
            (
                np.isfinite(qubit_metrics[q]["readout_error"])
                and qubit_metrics[q]["readout_error"] >= readout_threshold
            )
            or (
                np.isfinite(local_mean_2q[q])
                and local_mean_2q[q] >= local_2q_threshold
            )
        )
    }

    diagnostics = {
        "qubit_metrics": qubit_metrics,
        "local_mean_2q": local_mean_2q,
        "edge_error_map": edge_error_map,
        "readout_threshold": readout_threshold,
        "local_2q_threshold": local_2q_threshold,
    }
    return preferred_scores, avoided_qubits, edge_scores, diagnostics


def select_qcbm_heavyhex6_layout_from_snapshot(
    backend,
    backend_props,
    *,
    readout_quantile: float,
    local_2q_quantile: float,
    relax_if_needed: bool = True,
) -> tuple[list[int], float, dict[str, Any]]:
    graph = build_undirected_coupling_graph(backend.configuration().coupling_map)
    preferred_scores, avoided_qubits, edge_scores, diagnostics = (
        _snapshot_quality_maps(
            backend,
            backend_props,
            readout_quantile=readout_quantile,
            local_2q_quantile=local_2q_quantile,
        )
    )

    tried: list[str] = []
    try:
        layout, score = find_best_qcbm_heavyhex6(
            graph,
            preferred_scores,
            avoided_qubits,
            edge_scores=edge_scores,
        )
        fallback_used = False
    except RuntimeError:
        tried.append("strict qcbm_heavyhex6 failed")
        if not relax_if_needed:
            raise
        avoided_sorted = sorted(
            avoided_qubits,
            key=lambda q: preferred_scores.get(q, -1e9),
        )
        relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
        layout, score = find_best_qcbm_heavyhex6(
            graph,
            preferred_scores,
            relaxed_avoided,
            edge_scores=edge_scores,
        )
        avoided_qubits = relaxed_avoided
        fallback_used = True
        tried.append("relaxed qcbm_heavyhex6 succeeded")

    metadata = {
        "selected_topology": "qcbm_heavyhex6",
        "fallback_used": fallback_used,
        "tried": tried,
        "preferred_scores": preferred_scores,
        "avoided_qubits": sorted(int(q) for q in avoided_qubits),
        "edge_scores": edge_scores,
        "diagnostics": diagnostics,
    }
    return [int(q) for q in layout], float(score), metadata


def _snapshot_property_value(backend_props, qubit: int, name: str) -> float:
    try:
        out = backend_props.qubit_property(int(qubit))
        value = out.get(name, (None, None))[0]
        return math.nan if value is None else float(value)
    except Exception:
        return math.nan


def _snapshot_gate_error(
    layout_meta: dict[str, Any],
    physical_a: int,
    physical_b: int,
) -> float:
    edge = tuple(sorted((int(physical_a), int(physical_b))))
    return float(
        layout_meta["diagnostics"]["edge_error_map"].get(edge, math.nan)
    )


def snapshot_qubit_table(
    backend_props,
    chosen_layout: list[int],
    layout_meta: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for logical, physical in enumerate(chosen_layout):
        rows.append(
            {
                "logical_qubit": logical,
                "physical_qubit": int(physical),
                "T1_us": 1e6 * _snapshot_property_value(backend_props, physical, "T1"),
                "T2_us": 1e6 * _snapshot_property_value(backend_props, physical, "T2"),
                "readout_error": _snapshot_property_value(
                    backend_props, physical, "readout_error"
                ),
                "local_mean_2q_error": layout_meta["diagnostics"][
                    "local_mean_2q"
                ].get(int(physical), math.nan),
                "layout_score_component": layout_meta["preferred_scores"].get(
                    int(physical), math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def snapshot_edge_table(
    chosen_layout: list[int],
    layout_meta: dict[str, Any],
) -> pd.DataFrame:
    logical_pairs = MLQcbmCircuit._build_pairs(6, "qcbm_heavyhex6")
    rows = []
    for logical_a, logical_b in logical_pairs:
        physical_a = int(chosen_layout[logical_a])
        physical_b = int(chosen_layout[logical_b])
        rows.append(
            {
                "logical_edge": f"{logical_a}-{logical_b}",
                "physical_edge": f"{physical_a}-{physical_b}",
                "snapshot_2q_error": _snapshot_gate_error(
                    layout_meta, physical_a, physical_b
                ),
                "edge_score": layout_meta["edge_scores"].get(
                    tuple(sorted((physical_a, physical_b))),
                    math.nan,
                ),
            }
        )
    return pd.DataFrame(rows)


# ======================================================================
# Circuit summaries
# ======================================================================


def _iter_instructions(circuit):
    for item in circuit.data:
        if hasattr(item, "operation"):
            operation = item.operation
            qubits = item.qubits
        else:
            operation = item[0]
            qubits = item[1]
        yield operation, qubits


def _two_qubit_depth(circuit) -> int:
    levels = [0] * int(circuit.num_qubits)
    for operation, qubits in _iter_instructions(circuit):
        if len(qubits) != 2:
            continue
        qubit_indices = [circuit.find_bit(q).index for q in qubits]
        next_level = max(levels[idx] for idx in qubit_indices) + 1
        for idx in qubit_indices:
            levels[idx] = next_level
    return int(max(levels, default=0))


def summarize_transpiled_circuit(
    circuit,
    *,
    n_layers: int,
    n_params: int,
    circuit_kind: str,
) -> dict[str, Any]:
    counts = {str(k): int(v) for k, v in dict(circuit.count_ops()).items()}

    one_qubit_ops = 0
    two_qubit_ops = 0
    multi_qubit_ops = 0
    non_unitary_ops = 0
    for operation, qubits in _iter_instructions(circuit):
        name = str(operation.name)
        arity = len(qubits)
        if name in {"measure", "barrier", "delay", "reset"}:
            non_unitary_ops += 1
        elif arity == 1:
            one_qubit_ops += 1
        elif arity == 2:
            two_qubit_ops += 1
        elif arity > 2:
            multi_qubit_ops += 1

    active_qubits = sorted(
        {
            int(circuit.find_bit(q).index)
            for _, qubits in _iter_instructions(circuit)
            for q in qubits
        }
    )

    return {
        "n_layers": int(n_layers),
        "circuit_kind": circuit_kind,
        "n_params": int(n_params),
        "width": int(circuit.num_qubits),
        "active_qubits": ",".join(str(q) for q in active_qubits),
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "one_qubit_ops": int(one_qubit_ops),
        "two_qubit_ops": int(two_qubit_ops),
        "multi_qubit_ops": int(multi_qubit_ops),
        "non_unitary_ops": int(non_unitary_ops),
        "two_qubit_depth": _two_qubit_depth(circuit),
        "rx": counts.get("rx", 0),
        "rz": counts.get("rz", 0),
        "sx": counts.get("sx", 0),
        "x": counts.get("x", 0),
        "rzz": counts.get("rzz", 0),
        "cz": counts.get("cz", 0),
        "ecr": counts.get("ecr", 0),
        "cx": counts.get("cx", 0),
        "swap": counts.get("swap", 0),
        "measure": counts.get("measure", 0),
        "raw_ops_json": json.dumps(counts, sort_keys=True),
    }


# ======================================================================
# Training routines
# ======================================================================


def minimize_with_cost_history(
    cost_fn,
    *,
    x0: np.ndarray,
    method: str,
    options: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    f0 = float(cost_fn(x0))
    cost_history: list[float] = [f0]
    theta_history: list[np.ndarray] = [x0.copy()]

    def wrapped(x: np.ndarray) -> float:
        return float(cost_fn(np.asarray(x, dtype=float)))

    def callback(xk: np.ndarray) -> None:
        xk_arr = np.asarray(xk, dtype=float)
        fk = float(cost_fn(xk_arr))
        cost_history.append(fk)
        theta_history.append(xk_arr.copy())

    result = minimize(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    x_final = np.asarray(result.x, dtype=float)
    f_final = float(result.fun)
    same_theta = np.allclose(theta_history[-1], x_final, rtol=0.0, atol=1e-15)
    same_cost = abs(cost_history[-1] - f_final) <= 1e-15
    if not (same_theta and same_cost):
        cost_history.append(f_final)
        theta_history.append(x_final.copy())

    return result, np.asarray(cost_history, dtype=float), np.vstack(theta_history)


def run_stage1(
    x0: np.ndarray,
    *,
    maxiter: int,
    rhobeg: float,
    tol: float,
    cost_fn,
    qcbm: MLQcbmCircuit,
    target_h: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    result, cost_history, theta_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        method="COBYLA",
        options={
            "maxiter": int(maxiter),
            "rhobeg": float(rhobeg),
            "tol": float(tol),
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - start
    theta_star = np.asarray(result.x, dtype=float)
    p_star = qcbm.probabilities(theta_star)
    ce_final = float(result.fun)
    return {
        "result": result,
        "theta_star": theta_star,
        "p_star": p_star,
        "cost_history": cost_history,
        "theta_history": theta_history,
        "elapsed_time": float(elapsed),
        "ce_final": ce_final,
        "kl_final": float(ce_final - target_h),
    }


def run_stage2(
    x0: np.ndarray,
    *,
    maxiter: int,
    maxfun: int,
    cost_fn,
    qcbm: MLQcbmCircuit,
    target_h: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    result, cost_history, theta_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        method="L-BFGS-B",
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "ftol": 1e-12,
            "gtol": 1e-10,
            "eps": 1e-6,
            "maxls": 50,
            "maxcor": 20,
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - start
    theta_star = np.asarray(result.x, dtype=float)
    p_star = qcbm.probabilities(theta_star)
    ce_final = float(result.fun)
    return {
        "result": result,
        "theta_star": theta_star,
        "p_star": p_star,
        "cost_history": cost_history,
        "theta_history": theta_history,
        "elapsed_time": float(elapsed),
        "ce_final": ce_final,
        "kl_final": float(ce_final - target_h),
    }


def build_qcbm(
    *,
    n_qubits: int,
    n_layers: int,
    backend,
    chosen_layout: list[int],
) -> MLQcbmCircuit:
    return MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=int(n_layers),
        name=f"G_p_ideal_L{int(n_layers):02d}",
        entangler=ENTANGLER,
        topology=LOGICAL_TOPOLOGY,
        backend=AerSimulator(method="statevector"),
        transpile_backend=backend,
        noise_model=None,
        simulation_method="statevector",
        optimization_level=OPTIMIZATION_LEVEL,
        initial_layout=chosen_layout,
        layout_method=LAYOUT_METHOD,
        routing_method=ROUTING_METHOD,
        seed_transpiler=SEED_TRANSPILER,
    )


def build_noisy_qcbm(
    *,
    n_qubits: int,
    n_layers: int,
    transpile_backend,
    noisy_backend,
    noise_model,
    chosen_layout: list[int],
) -> MLQcbmCircuit:
    return MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=int(n_layers),
        name=f"G_p_shots_noise_eval_L{int(n_layers):02d}",
        entangler=ENTANGLER,
        topology=LOGICAL_TOPOLOGY,
        backend=noisy_backend,
        transpile_backend=transpile_backend,
        noise_model=noise_model,
        basis_gates=list(noise_model.basis_gates),
        simulation_method="density_matrix",
        optimization_level=OPTIMIZATION_LEVEL,
        initial_layout=chosen_layout,
        layout_method=LAYOUT_METHOD,
        routing_method=ROUTING_METHOD,
        seed_transpiler=SEED_TRANSPILER,
    )


def evaluate_theta_under_shots_noise(
    *,
    n_layers: int,
    theta_star: np.ndarray,
    ptg: np.ndarray,
    n_time: int,
    transpile_backend,
    noisy_backend,
    noise_model,
    chosen_layout: list[int],
    output_dir: pathlib.Path,
    shots: int,
    eval_runs: int,
    seed_base: int,
    alpha: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    print(
        f"Evaluating theta_star under sample noise + backend noise | "
        f"L={n_layers:02d} | runs={eval_runs} | shots={shots}"
    )
    qcbm_noisy = build_noisy_qcbm(
        n_qubits=int(round(math.log2(ptg.size))),
        n_layers=int(n_layers),
        transpile_backend=transpile_backend,
        noisy_backend=noisy_backend,
        noise_model=noise_model,
        chosen_layout=chosen_layout,
    )
    theta_star = np.asarray(theta_star, dtype=float).ravel()
    if theta_star.size != qcbm_noisy.n_params:
        raise ValueError(
            f"Noisy QCBM parameter mismatch for L={n_layers}: "
            f"theta_star={theta_star.size}, qcbm.n_params={qcbm_noisy.n_params}."
        )

    kl_values: list[float] = []
    p_raw_values: list[np.ndarray] = []
    p_smooth_values: list[np.ndarray] = []
    seeds: list[int] = []

    for run_idx in range(int(eval_runs)):
        run_seed = int(seed_base) + run_idx
        p_raw = qcbm_noisy.probabilities(
            theta_star,
            shots=int(shots),
            seed=run_seed,
        )
        p_smooth = _dirichlet_smooth(
            p_raw,
            shots=int(shots),
            alpha=float(alpha),
        )
        kl_val = kl_divergence(ptg, p_smooth, eps=EPS_COST)

        seeds.append(run_seed)
        kl_values.append(kl_val)
        p_raw_values.append(p_raw)
        p_smooth_values.append(p_smooth)
        print(
            f"  [noise eval {run_idx + 1:02d}/{eval_runs:02d}] "
            f"seed={run_seed} | KL={kl_val:.8e}"
        )

    kl_arr = np.asarray(kl_values, dtype=float)
    p_raw_arr = np.vstack(p_raw_values)
    p_smooth_arr = np.vstack(p_smooth_values)
    p_mean_raw = _normalize_distribution(np.mean(p_raw_arr, axis=0), eps=None)
    p_mean_smooth = _normalize_distribution(np.mean(p_smooth_arr, axis=0), eps=None)
    mean_prob_kl = kl_divergence(ptg, p_mean_smooth, eps=EPS_COST)
    noisy_decomp = kl_time_decomposition(
        ptg,
        p_mean_smooth,
        n_time=n_time,
        eps=EPS_COST,
    )

    row = {
        "kl_shots_noise_mean": float(np.mean(kl_arr)),
        "kl_shots_noise_std": float(np.std(kl_arr, ddof=1))
        if kl_arr.size > 1
        else 0.0,
        "kl_shots_noise_sem": float(np.std(kl_arr, ddof=1) / math.sqrt(kl_arr.size))
        if kl_arr.size > 1
        else 0.0,
        "kl_shots_noise_min": float(np.min(kl_arr)),
        "kl_shots_noise_max": float(np.max(kl_arr)),
        "kl_shots_noise_median": float(np.median(kl_arr)),
        "kl_shots_noise_mean_probability": float(mean_prob_kl),
        "kl_shots_noise_time_marginal": noisy_decomp["kl_time_marginal"],
        "kl_shots_noise_time_conditional": noisy_decomp["kl_time_conditional"],
        "kl_shots_noise_time_conditional_max": noisy_decomp[
            "kl_time_conditional_max"
        ],
        "noisy_eval_runs": int(eval_runs),
        "noisy_eval_shots": int(shots),
        "noisy_eval_seed_base": int(seed_base),
        "noisy_eval_dirichlet_alpha": float(alpha),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"noisy_eval_L{int(n_layers):02d}.npz",
        theta_star=theta_star,
        p_target=ptg,
        p_noisy_raw_values=p_raw_arr,
        p_noisy_smooth_values=p_smooth_arr,
        p_noisy_mean_raw=p_mean_raw,
        p_noisy_mean_smooth=p_mean_smooth,
        kl_shots_noise_values=kl_arr,
        noisy_eval_seeds=np.asarray(seeds, dtype=int),
        noisy_eval_shots=np.int64(shots),
        noisy_eval_runs=np.int64(eval_runs),
        noisy_eval_seed_base=np.int64(seed_base),
        noisy_eval_dirichlet_alpha=np.float64(alpha),
        summary_row_json=np.array(json.dumps(_json_ready(row), sort_keys=True)),
    )
    return row, kl_arr, p_mean_smooth


def train_one_layer(
    *,
    n_layers: int,
    ptg: np.ndarray,
    n_time: int,
    backend,
    noisy_backend,
    noise_model,
    chosen_layout: list[int],
    layout_score: float,
    layout_meta: dict[str, Any],
    snapshot_dt_utc: datetime,
    output_dir: pathlib.Path,
    stage1_maxiter: int,
    stage2_maxiter: int,
    stage2_maxfun: int,
    run_noisy_eval: bool,
    noisy_eval_shots: int,
    noisy_eval_runs: int,
    noisy_eval_seed_base: int,
    noisy_eval_alpha: float,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], LayerArtifact]:
    layer_dir = output_dir / f"L{int(n_layers):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    result_path = layer_dir / f"qcbm_ideal_L{int(n_layers):02d}.npz"
    summary_path = layer_dir / "summary_row.json"

    if result_path.exists() and summary_path.exists() and not force:
        data = np.load(result_path, allow_pickle=True)
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        transpile_rows = json.loads(
            (layer_dir / "transpile_rows.json").read_text(encoding="utf-8")
        )
        kl_noisy_values = None
        p_noisy_mean = None
        noisy_eval_path = layer_dir / f"noisy_eval_L{int(n_layers):02d}.npz"
        if run_noisy_eval and (
            "kl_shots_noise_mean" not in row or not noisy_eval_path.exists()
        ):
            noisy_row, kl_noisy_values, p_noisy_mean = (
                evaluate_theta_under_shots_noise(
                    n_layers=int(n_layers),
                    theta_star=np.asarray(data["theta_star"], dtype=float),
                    ptg=ptg,
                    n_time=n_time,
                    transpile_backend=backend,
                    noisy_backend=noisy_backend,
                    noise_model=noise_model,
                    chosen_layout=chosen_layout,
                    output_dir=layer_dir,
                    shots=int(noisy_eval_shots),
                    eval_runs=int(noisy_eval_runs),
                    seed_base=int(noisy_eval_seed_base),
                    alpha=float(noisy_eval_alpha),
                )
            )
            row.update(noisy_row)
            row["kl_noise_penalty_mean"] = float(
                row["kl_shots_noise_mean"] - row["kl_final"]
            )
            _write_json(summary_path, row)
        elif noisy_eval_path.exists():
            noisy_data = np.load(noisy_eval_path, allow_pickle=True)
            kl_noisy_values = np.asarray(
                noisy_data["kl_shots_noise_values"], dtype=float
            )
            p_noisy_mean = np.asarray(
                noisy_data["p_noisy_mean_smooth"], dtype=float
            )

        artifact = LayerArtifact(
            n_layers=int(n_layers),
            result_path=result_path,
            kl_history=np.asarray(data["kl_history"], dtype=float),
            best_kl_history=np.asarray(data["best_kl_history"], dtype=float),
            p_star=np.asarray(data["p_star"], dtype=float),
            p_init=np.asarray(data["p_init"], dtype=float),
            p_target=np.asarray(data["p_target"], dtype=float),
            kl_noisy_values=kl_noisy_values,
            p_noisy_mean=p_noisy_mean,
        )
        print(f"[resume] L={n_layers:02d} loaded from {result_path}")
        return row, transpile_rows, artifact

    print(f"\n=== Training ideal statevector QCBM | layers={n_layers:02d} ===")
    qcbm = build_qcbm(
        n_qubits=int(round(math.log2(ptg.size))),
        n_layers=int(n_layers),
        backend=backend,
        chosen_layout=chosen_layout,
    )

    ansatz_summary = summarize_transpiled_circuit(
        qcbm._tqc,
        n_layers=int(n_layers),
        n_params=qcbm.n_params,
        circuit_kind="ansatz",
    )
    measured_summary = summarize_transpiled_circuit(
        qcbm._tqc_meas,
        n_layers=int(n_layers),
        n_params=qcbm.n_params,
        circuit_kind="measured",
    )
    transpile_rows = [ansatz_summary, measured_summary]

    cost_statevector = qcbm.cost_fn(ptg, eps=EPS_COST)
    target_h = target_entropy(ptg, eps=EPS_COST)

    rng = np.random.default_rng(THETA_SEED)
    theta_init = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)
    p_init = qcbm.probabilities(theta_init)
    kl_init = kl_divergence(ptg, p_init, eps=EPS_COST)

    stage1 = run_stage1(
        theta_init,
        maxiter=stage1_maxiter,
        rhobeg=STAGE1_RHOBEG,
        tol=STAGE1_TOL,
        cost_fn=cost_statevector,
        qcbm=qcbm,
        target_h=target_h,
    )
    print(
        f"L={n_layers:02d} stage1 | CE={stage1['ce_final']:.8e} "
        f"| KL={stage1['kl_final']:.8e} | t={stage1['elapsed_time']:.2f}s"
    )

    stage2 = run_stage2(
        stage1["theta_star"],
        maxiter=stage2_maxiter,
        maxfun=stage2_maxfun,
        cost_fn=cost_statevector,
        qcbm=qcbm,
        target_h=target_h,
    )
    print(
        f"L={n_layers:02d} stage2 | CE={stage2['ce_final']:.8e} "
        f"| KL={stage2['kl_final']:.8e} | t={stage2['elapsed_time']:.2f}s"
    )

    cost_history = np.r_[stage1["cost_history"], stage2["cost_history"][1:]]
    theta_history = np.vstack(
        [stage1["theta_history"], stage2["theta_history"][1:]]
    )
    kl_history = np.maximum(cost_history - target_h, 1e-15)
    best_kl_history = np.minimum.accumulate(kl_history)

    theta_star = np.asarray(stage2["theta_star"], dtype=float)
    p_star = np.asarray(stage2["p_star"], dtype=float)
    metrics = qcbm.metrics(ptg, p_star, eps=EPS_COST)
    final_kl = float(metrics["kl"])
    decomp = kl_time_decomposition(ptg, p_star, n_time=n_time, eps=EPS_COST)
    kl_top90, kl_tail10, n_top90 = kl_mass_contribution(
        ptg, p_star, mass_threshold=0.90, eps=EPS_COST
    )
    kl_top99, kl_tail01, n_top99 = kl_mass_contribution(
        ptg, p_star, mass_threshold=0.99, eps=EPS_COST
    )

    elapsed_total = float(stage1["elapsed_time"] + stage2["elapsed_time"])
    n_entangling_pairs = len(qcbm.pairs)
    n_rot_layers = (int(n_layers) + 1) // 2
    n_ent_layers = int(n_layers) // 2

    row = {
        "n_layers": int(n_layers),
        "n_qubits": int(qcbm.n_qubits),
        "n_params": int(qcbm.n_params),
        "n_rot_layers": int(n_rot_layers),
        "n_ent_layers": int(n_ent_layers),
        "n_entangling_pairs": int(n_entangling_pairs),
        "theta_seed": int(THETA_SEED),
        "init_scale": float(INIT_SCALE),
        "kl_init": float(kl_init),
        "kl_stage1": float(stage1["kl_final"]),
        "kl_final": final_kl,
        "kl_best_history": float(np.min(best_kl_history)),
        "kl_time_marginal": decomp["kl_time_marginal"],
        "kl_time_conditional": decomp["kl_time_conditional"],
        "kl_time_conditional_max": decomp["kl_time_conditional_max"],
        "kl_top90_contribution": kl_top90,
        "kl_tail10_contribution": kl_tail10,
        "n_states_for_90pct_target_mass": int(n_top90),
        "kl_top99_contribution": kl_top99,
        "kl_tail01_contribution": kl_tail01,
        "n_states_for_99pct_target_mass": int(n_top99),
        "kl_improvement_init_to_final": float(kl_init - final_kl),
        "kl_reduction_factor": float(kl_init / final_kl)
        if final_kl > 0
        else math.inf,
        "ce_final": float(stage2["ce_final"]),
        "target_entropy": float(target_h),
        "elapsed_total_s": elapsed_total,
        "stage1_elapsed_s": float(stage1["elapsed_time"]),
        "stage2_elapsed_s": float(stage2["elapsed_time"]),
        "stage1_nit": int(getattr(stage1["result"], "nit", -1)),
        "stage2_nit": int(getattr(stage2["result"], "nit", -1)),
        "n_history_points": int(kl_history.size),
        "stage2_success": bool(getattr(stage2["result"], "success", False)),
        "stage2_message": str(getattr(stage2["result"], "message", "")),
        "backend_name": BACKEND_NAME,
        "noise_snapshot_iso_utc": snapshot_dt_utc.isoformat(),
        "requested_topology": LOGICAL_TOPOLOGY,
        "effective_topology": layout_meta["selected_topology"],
        "entangler": ENTANGLER,
        "chosen_layout": ",".join(str(q) for q in chosen_layout),
        "layout_score": float(layout_score),
        "layout_fallback_used": bool(layout_meta["fallback_used"]),
        "ansatz_depth": int(ansatz_summary["depth"]),
        "ansatz_size": int(ansatz_summary["size"]),
        "ansatz_width": int(ansatz_summary["width"]),
        "ansatz_one_qubit_ops": int(ansatz_summary["one_qubit_ops"]),
        "ansatz_two_qubit_ops": int(ansatz_summary["two_qubit_ops"]),
        "ansatz_two_qubit_depth": int(ansatz_summary["two_qubit_depth"]),
        "ansatz_swap": int(ansatz_summary["swap"]),
        "ansatz_rzz": int(ansatz_summary["rzz"]),
        "measured_depth": int(measured_summary["depth"]),
        "measured_size": int(measured_summary["size"]),
        "measured_one_qubit_ops": int(measured_summary["one_qubit_ops"]),
        "measured_two_qubit_ops": int(measured_summary["two_qubit_ops"]),
        "measured_non_unitary_ops": int(measured_summary["non_unitary_ops"]),
    }

    kl_noisy_values = None
    p_noisy_mean = None
    if run_noisy_eval:
        noisy_row, kl_noisy_values, p_noisy_mean = evaluate_theta_under_shots_noise(
            n_layers=int(n_layers),
            theta_star=theta_star,
            ptg=ptg,
            n_time=n_time,
            transpile_backend=backend,
            noisy_backend=noisy_backend,
            noise_model=noise_model,
            chosen_layout=chosen_layout,
            output_dir=layer_dir,
            shots=int(noisy_eval_shots),
            eval_runs=int(noisy_eval_runs),
            seed_base=int(noisy_eval_seed_base),
            alpha=float(noisy_eval_alpha),
        )
        row.update(noisy_row)
        row["kl_noise_penalty_mean"] = float(
            row["kl_shots_noise_mean"] - row["kl_final"]
        )

    with open(layer_dir / f"trained_qcbm_L{int(n_layers):02d}.qpy", "wb") as f:
        qpy.dump(qcbm._tqc, f)
    with open(layer_dir / f"trained_qcbm_measured_L{int(n_layers):02d}.qpy", "wb") as f:
        qpy.dump(qcbm._tqc_meas, f)

    np.savez(
        result_path,
        theta_star=theta_star,
        theta_init=theta_init,
        theta_stage1=np.asarray(stage1["theta_star"], dtype=float),
        theta_history=theta_history,
        cost_history=cost_history,
        kl_history=kl_history,
        best_kl_history=best_kl_history,
        p_target=ptg,
        p_init=p_init,
        p_stage1=np.asarray(stage1["p_star"], dtype=float),
        p_star=p_star,
        target_entropy=np.float64(target_h),
        kl_init=np.float64(kl_init),
        kl_stage1=np.float64(stage1["kl_final"]),
        kl_final=np.float64(final_kl),
        metrics=np.array(metrics, dtype=object),
        summary_row_json=np.array(json.dumps(_json_ready(row), sort_keys=True)),
        transpile_rows_json=np.array(
            json.dumps(_json_ready(transpile_rows), sort_keys=True)
        ),
    )
    _write_json(summary_path, row)
    _write_json(layer_dir / "transpile_rows.json", {"rows": transpile_rows})

    artifact = LayerArtifact(
        n_layers=int(n_layers),
        result_path=result_path,
        kl_history=kl_history,
        best_kl_history=best_kl_history,
        p_star=p_star,
        p_init=p_init,
        p_target=ptg,
        kl_noisy_values=kl_noisy_values,
        p_noisy_mean=p_noisy_mean,
    )
    return row, transpile_rows, artifact


# ======================================================================
# Table outputs
# ======================================================================


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(_format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_table_bundle(
    df: pd.DataFrame,
    *,
    output_dir: pathlib.Path,
    stem: str,
    title: str,
    image_columns: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(dataframe_to_markdown(df), encoding="utf-8")
    try:
        (output_dir / f"{stem}.tex").write_text(
            df.to_latex(index=False, escape=True),
            encoding="utf-8",
        )
    except Exception as exc:
        (output_dir / f"{stem}.tex.error.txt").write_text(str(exc), encoding="utf-8")

    if image_columns is None:
        image_columns = list(df.columns)
    image_df = df[image_columns].copy()
    formatted = image_df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: _format_float(x, 3))

    fig_width = max(8.0, 1.1 * len(formatted.columns))
    fig_height = max(2.6, 0.42 * len(formatted) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    table = ax.table(
        cellText=formatted.astype(str).values,
        colLabels=formatted.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#263238")
        else:
            cell.set_facecolor("#f7f9fb" if row_idx % 2 else "#ffffff")
        cell.set_edgecolor("#d7dde2")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=350, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Plotting
# ======================================================================


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 350,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "lines.linewidth": 2.0,
            "patch.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, output_dir: pathlib.Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_kl_vs_layers(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = summary_df["n_layers"].to_numpy()
    ax.semilogy(
        x,
        summary_df["kl_init"],
        marker="o",
        color="#7b8794",
        linestyle="--",
        label="Initial",
    )
    ax.semilogy(
        x,
        summary_df["kl_stage1"],
        marker="s",
        color="#2f80ed",
        linestyle="-.",
        label="After COBYLA",
    )
    ax.semilogy(
        x,
        summary_df["kl_final"],
        marker="D",
        color="#c0392b",
        label="After L-BFGS-B",
    )
    best_idx = int(summary_df["kl_final"].idxmin())
    best_row = summary_df.loc[best_idx]
    ax.scatter(
        [best_row["n_layers"]],
        [best_row["kl_final"]],
        s=90,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        f"best L={int(best_row['n_layers'])}\nKL={best_row['kl_final']:.2e}",
        xy=(best_row["n_layers"], best_row["kl_final"]),
        xytext=(8, 18),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.8},
    )
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel(r"$D_{KL}(p_{\mathrm{target}}\Vert p_{\theta})$")
    ax.set_title("Ideal statevector training improves with QCBM depth")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "kl_vs_layers")


def plot_ideal_vs_shots_noise_kl(
    summary_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    if "kl_shots_noise_mean" not in summary_df.columns:
        return

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Latin Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.25,
            "lines.solid_capstyle": "round",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        x = summary_df["n_layers"].to_numpy()
        yerr = (
            summary_df["kl_shots_noise_std"].to_numpy()
            if "kl_shots_noise_std" in summary_df.columns
            else None
        )
        ax.semilogy(
            x,
            summary_df["kl_final"],
            marker="^",
            markersize=5.5,
            color="#111827",
            linewidth=1.75,
            label="Ideal statevector eval",
        )
        ax.errorbar(
            x,
            summary_df["kl_shots_noise_mean"],
            yerr=yerr,
            marker="o",
            markersize=5.5,
            color="#c2410c",
            capsize=3,
            linewidth=1.75,
            label="Shots + backend noise eval",
        )
        ax.set_xlabel("QCBM layers")
        ax.set_ylabel(r"$\mathrm{KL}_{\epsilon}(P_{\mathrm{target}}\Vert P_{\theta})$")
        ax.set_xticks(x)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax.minorticks_on()
        ax.grid(True, which="major", axis="both", color="#c7c7c7", linewidth=0.8, alpha=0.72)
        ax.grid(True, which="minor", axis="x", color="#e7e7e7", linewidth=0.45, alpha=0.85)
        ax.grid(True, which="minor", axis="y", color="#dcdcdc", linewidth=0.55, alpha=0.95)
        ax.tick_params(axis="both", which="major", direction="in", length=6.0, width=1.05, top=False, right=False)
        ax.tick_params(axis="both", which="minor", direction="in", length=3.2, width=0.85, top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(1.25)
        ax.legend(
            frameon=False,
            loc="upper right",
            handlelength=2.7,
        )
        fig.tight_layout(pad=0.45)
        save_figure(fig, output_dir, "ideal_vs_shots_noise_kl")


def plot_kl_decomposition(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = summary_df["n_layers"].to_numpy()
    ax.semilogy(
        x,
        summary_df["kl_final"],
        marker="D",
        color="#111827",
        label="Joint KL",
    )
    ax.semilogy(
        x,
        summary_df["kl_time_marginal"],
        marker="o",
        color="#0f766e",
        label="Time marginal KL",
    )
    ax.semilogy(
        x,
        summary_df["kl_time_conditional"],
        marker="s",
        color="#b45309",
        label="Weighted conditional KL",
    )
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel("KL contribution")
    ax.set_title("KL decomposition under the time-major CVA factorization")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "kl_time_decomposition_vs_layers")


def plot_resource_scaling(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    x = summary_df["n_layers"].to_numpy()

    axes[0].plot(
        x,
        summary_df["ansatz_depth"],
        marker="o",
        color="#1f6feb",
        label="Depth",
    )
    axes[0].plot(
        x,
        summary_df["ansatz_two_qubit_depth"],
        marker="s",
        color="#a21caf",
        label="2q depth",
    )
    axes[0].set_xlabel("QCBM layers")
    axes[0].set_ylabel("Transpiled depth")
    axes[0].set_title("Depth scaling")
    axes[0].legend(frameon=True)

    axes[1].plot(
        x,
        summary_df["ansatz_one_qubit_ops"],
        marker="o",
        color="#0f766e",
        label="1q gates",
    )
    axes[1].plot(
        x,
        summary_df["ansatz_two_qubit_ops"],
        marker="s",
        color="#c2410c",
        label="2q gates",
    )
    axes[1].plot(
        x,
        summary_df["ansatz_swap"],
        marker="^",
        color="#475569",
        label="SWAP",
    )
    axes[1].set_xlabel("QCBM layers")
    axes[1].set_ylabel("Transpiled operation count")
    axes[1].set_title("Gate-count scaling")
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "transpiled_resources_vs_layers")


def plot_kl_vs_resources(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    y = summary_df["kl_final"].to_numpy()
    layers = summary_df["n_layers"].to_numpy()
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(layers.min()), vmax=float(layers.max()))

    x_specs = [
        ("n_params", "Parameters"),
        ("ansatz_depth", "Transpiled depth"),
        ("ansatz_two_qubit_ops", "2q gates"),
    ]
    for ax, (col, xlabel) in zip(axes, x_specs, strict=True):
        sc = ax.scatter(
            summary_df[col],
            y,
            c=layers,
            cmap=cmap,
            norm=norm,
            s=52,
            edgecolor="#1f2937",
            linewidth=0.5,
        )
        ax.plot(summary_df[col], y, color="#9ca3af", linewidth=1.0, zorder=0)
        for _, row in summary_df.iterrows():
            ax.annotate(
                str(int(row["n_layers"])),
                (row[col], row["kl_final"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$D_{KL}$")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_title("Accuracy-resource frontier")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.84, pad=0.02)
    cbar.set_label("Layers")
    fig.tight_layout()
    save_figure(fig, output_dir, "kl_vs_training_resources")


def plot_training_curves(
    artifacts: list[LayerArtifact],
    output_dir: pathlib.Path,
) -> None:
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "Latin Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.25,
            "lines.solid_capstyle": "round",
        }
    ):
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for artifact in artifacts:
            curve_history = artifact.best_kl_history
            continuation_path = (
                artifact.result_path.parent
                / f"{TRAINING_CURVE_CONTINUATION_STEM}_L{artifact.n_layers:02d}.npz"
            )
            if continuation_path.exists():
                with np.load(continuation_path, allow_pickle=False) as continuation:
                    curve_history = np.asarray(
                        continuation["best_kl_history"], dtype=float
                    )
            # After convergence, keep the best-so-far KL constant so every
            # visual trajectory reaches the same callback-index horizon.
            y = np.pad(
                curve_history,
                (0, max(0, TRAINING_CURVE_MAX_ITERATION + 1 - curve_history.size)),
                mode="edge",
            )[: TRAINING_CURVE_MAX_ITERATION + 1]
            x = np.arange(y.size)
            color = PAPER_LAYER_COLORS.get(int(artifact.n_layers), "#444444")
            ax.semilogy(
                x,
                y,
                color=color,
                alpha=0.96,
                linewidth=1.75,
                label=f"L={artifact.n_layers}",
            )
        ax.set_xlim(0, TRAINING_CURVE_XMAX)
        ax.set_xlabel("iterations")
        ax.set_ylabel(r"$\mathrm{KL}_{\epsilon}(P_{\mathrm{target}}\Vert P_{\theta})$")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.minorticks_on()
        ax.grid(True, which="major", axis="both", color="#c7c7c7", linewidth=0.8, alpha=0.72)
        ax.grid(True, which="minor", axis="both", color="#e7e7e7", linewidth=0.45, alpha=0.85)
        ax.tick_params(axis="both", which="major", direction="in", length=6.0, width=1.05, top=False, right=False)
        ax.tick_params(axis="both", which="minor", direction="in", length=3.2, width=0.85, top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(1.25)
        ax.legend(
            ncol=4,
            frameon=False,
            loc="upper right",
            handlelength=2.7,
            columnspacing=1.05,
        )
        fig.tight_layout(pad=0.45)
        save_figure(fig, output_dir, "training_kl_trajectories")


def _select_artifacts_for_plots(
    artifacts: list[LayerArtifact],
    selected_layers: tuple[int, ...] = PLOT_SELECTED_LAYERS,
) -> list[LayerArtifact]:
    by_layer = {a.n_layers: a for a in artifacts}
    selected = [by_layer[layer] for layer in selected_layers if layer in by_layer]
    if selected:
        return selected
    return artifacts


def plot_top_state_fit(
    artifacts: list[LayerArtifact],
    output_dir: pathlib.Path,
    *,
    top_n: int = 24,
) -> None:
    selected = _select_artifacts_for_plots(artifacts)
    target = selected[0].p_target
    order = np.argsort(target)[::-1][:top_n]
    x = np.arange(order.size)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.2]},
    )
    axes[0].plot(
        x,
        target[order],
        color="#111111",
        marker="o",
        linewidth=2.4,
        label="target",
    )
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.12, 0.92, len(selected)))
    for artifact, color in zip(selected, colors, strict=True):
        axes[0].plot(
            x,
            artifact.p_star[order],
            marker=".",
            color=color,
            alpha=0.9,
            label=f"L={artifact.n_layers}",
        )
        axes[1].plot(
            x,
            artifact.p_star[order] - target[order],
            marker=".",
            color=color,
            alpha=0.9,
            label=f"L={artifact.n_layers}",
        )
    axes[1].axhline(0.0, color="#111111", linewidth=0.9)
    axes[0].set_ylabel("Probability")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("States sorted by target probability")
    axes[0].set_title("Fit on dominant probability-mass states")
    axes[0].legend(ncol=5, frameon=True)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(idx)) for idx in order], rotation=90)
    fig.tight_layout()
    save_figure(fig, output_dir, "top_state_fit_and_residuals")


def plot_time_major_heatmaps(
    artifacts: list[LayerArtifact],
    output_dir: pathlib.Path,
    *,
    n_time: int,
) -> None:
    selected = _select_artifacts_for_plots(artifacts, (2, 6, 10, 14, 16))
    target = selected[0].p_target
    if target.size % int(n_time) != 0:
        return
    n_state = target.size // int(n_time)
    target_2d = target.reshape(int(n_time), n_state)
    residuals = [artifact.p_star.reshape(int(n_time), n_state) - target_2d for artifact in selected]
    vmax = max(float(np.max(np.abs(res))) for res in residuals)
    vmax = max(vmax, 1e-12)

    nrows = 1 + len(selected)
    fig, axes = plt.subplots(nrows, 1, figsize=(10.5, 1.7 * nrows), sharex=True)
    target_im = axes[0].imshow(target_2d, aspect="auto", cmap="viridis")
    axes[0].set_title("Target distribution reshaped as time-major p(t, state)")
    axes[0].set_ylabel("t")
    fig.colorbar(target_im, ax=axes[0], fraction=0.018, pad=0.015)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for ax, artifact, residual in zip(axes[1:], selected, residuals, strict=True):
        im = ax.imshow(residual, aspect="auto", cmap="coolwarm", norm=norm)
        ax.set_ylabel("t")
        ax.set_title(f"Residual p_theta - p_target | L={artifact.n_layers}")
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
    axes[-1].set_xlabel("Joint asset-grid state index")
    fig.tight_layout()
    save_figure(fig, output_dir, "time_major_distribution_residual_heatmaps")


def plot_cumulative_kl_concentration(
    artifacts: list[LayerArtifact],
    output_dir: pathlib.Path,
) -> None:
    selected = _select_artifacts_for_plots(artifacts)
    target = _normalize_distribution(selected[0].p_target, eps=None)
    order = np.argsort(target)[::-1]
    cumulative_mass = np.cumsum(target[order])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cmap = plt.get_cmap("magma")
    colors = cmap(np.linspace(0.16, 0.9, len(selected)))
    for artifact, color in zip(selected, colors, strict=True):
        contrib = kl_contributions(target, artifact.p_star, eps=EPS_COST)[order]
        cumulative_kl = np.cumsum(contrib)
        total_kl = float(np.sum(contrib))
        if abs(total_kl) > 0:
            cumulative_kl = cumulative_kl / total_kl
        ax.plot(
            cumulative_mass,
            cumulative_kl,
            color=color,
            label=f"L={artifact.n_layers}",
        )
    ax.axvline(0.90, color="#475569", linestyle="--", linewidth=1.0)
    ax.axvline(0.99, color="#475569", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Cumulative target probability mass")
    ax.set_ylabel("Cumulative fraction of final KL")
    ax.set_title("Where the KL error is concentrated")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "cumulative_kl_concentration")


def generate_plots(
    summary_df: pd.DataFrame,
    artifacts: list[LayerArtifact],
    *,
    output_dir: pathlib.Path,
    n_time: int,
) -> None:
    plot_dir = output_dir / "figures"
    configure_matplotlib()
    plot_kl_vs_layers(summary_df, plot_dir)
    plot_ideal_vs_shots_noise_kl(summary_df, plot_dir)
    plot_kl_decomposition(summary_df, plot_dir)
    plot_resource_scaling(summary_df, plot_dir)
    plot_kl_vs_resources(summary_df, plot_dir)
    plot_training_curves(artifacts, plot_dir)
    plot_top_state_fit(artifacts, plot_dir)
    plot_time_major_heatmaps(artifacts, plot_dir, n_time=n_time)
    plot_cumulative_kl_concentration(artifacts, plot_dir)


# ======================================================================
# Main
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ideal statevector QCBM layer comparison for the 6q multi-asset "
            "CVA instance. The training objective is the same clipped "
            "cross-entropy used in the existing statevector QCBM training; "
            "reported performance is KL."
        )
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(LAYERS_GRID),
        help="Layer values to train. Default: all integers from 2 to 16.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain layers even if per-layer NPZ files already exist.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only train and write tables; do not regenerate figures.",
    )
    parser.add_argument(
        "--skip-noisy-eval",
        action="store_true",
        help="Skip the post-training shots + backend-noise KL evaluation.",
    )
    parser.add_argument(
        "--noisy-eval-runs",
        type=int,
        default=NOISY_EVAL_RUNS,
        help="Number of independent noisy shot evaluations per trained theta.",
    )
    parser.add_argument(
        "--noisy-eval-shots",
        type=int,
        default=NOISY_EVAL_SHOTS,
        help="Shots per noisy evaluation run.",
    )
    parser.add_argument(
        "--noisy-eval-seed-base",
        type=int,
        default=NOISY_EVAL_SEED_BASE,
        help="Base simulator seed for noisy evaluation repetitions.",
    )
    parser.add_argument(
        "--stage1-maxiter",
        type=int,
        default=STAGE1_MAXITER,
        help="COBYLA maxiter per layer.",
    )
    parser.add_argument(
        "--stage2-maxiter",
        type=int,
        default=STAGE2_MAXITER,
        help="L-BFGS-B maxiter per layer.",
    )
    parser.add_argument(
        "--stage2-maxfun",
        type=int,
        default=STAGE2_MAXFUN,
        help="L-BFGS-B maxfun per layer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = sorted({int(layer) for layer in args.layers})
    if any(layer < 2 or layer > 16 for layer in layers):
        raise ValueError("This comparison is intentionally scoped to layers 2..16.")

    output_dir = REPO_ROOT / RESULTS_RELATIVE_DIR
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(REPO_ROOT / BENCHMARK_RELATIVE_PATH, allow_pickle=True)
    ptg = _normalize_distribution(_as_1d_float(data["p_target"]), eps=None)
    dim = int(ptg.size)
    n_qubits = int(round(math.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError("p_target length must be a power of two.")

    n_time = int(np.asarray(data["t"]).size) if "t" in data else 1
    if dim % n_time != 0:
        raise ValueError("p_target length is not divisible by the time grid size.")

    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    print(f"Backend: {BACKEND_NAME}")
    print(f"Noise snapshot UTC: {snapshot_dt_utc.isoformat()}")
    print(f"Output directory: {output_dir}")

    service = QiskitRuntimeService(channel=RUNTIME_CHANNEL)
    real_backend = service.backend(
        BACKEND_NAME,
        use_fractional_gates=USE_FRACTIONAL_GATES,
    )
    backend_props = real_backend.properties(datetime=snapshot_dt_utc)
    if backend_props is None:
        raise RuntimeError(
            "Could not retrieve backend properties for snapshot "
            f"{snapshot_dt_utc.isoformat()}."
        )

    # Same frozen backend-noise construction used in
    # shots/noise_SH_training_6q_hh6_MPS.py: historical backend properties,
    # no thermal relaxation term, density-matrix Aer simulation, finite shots.
    used_noise_fallback = False
    try:
        noise_model = NoiseModel.from_backend_properties(
            backend_props,
            thermal_relaxation=False,
        )
    except AttributeError:
        used_noise_fallback = True
        print(
            "Noise model could not be created from backend properties. "
            "Falling back to noise model from backend (non-snapshot)."
        )
        noise_model = NoiseModel.from_backend(real_backend)

    coupling_map = getattr(real_backend, "coupling_map", None)
    if coupling_map is None:
        try:
            coupling_map = real_backend.configuration().coupling_map
        except Exception:
            coupling_map = None

    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=coupling_map,
        seed_simulator=SIMULATOR_SEED,
    )

    chosen_layout, layout_score, layout_meta = (
        select_qcbm_heavyhex6_layout_from_snapshot(
            real_backend,
            backend_props,
            readout_quantile=READOUT_QUANTILE,
            local_2q_quantile=LOCAL_2Q_QUANTILE,
        )
    )

    qubit_df = snapshot_qubit_table(backend_props, chosen_layout, layout_meta)
    edge_df = snapshot_edge_table(chosen_layout, layout_meta)
    save_table_bundle(
        qubit_df,
        output_dir=tables_dir,
        stem="snapshot_selected_qubits",
        title="Selected physical qubits from backend snapshot",
        image_columns=[
            "logical_qubit",
            "physical_qubit",
            "T1_us",
            "T2_us",
            "readout_error",
            "local_mean_2q_error",
        ],
    )
    save_table_bundle(
        edge_df,
        output_dir=tables_dir,
        stem="snapshot_selected_edges",
        title="Selected QCBM heavy-hex6 edges from backend snapshot",
        image_columns=[
            "logical_edge",
            "physical_edge",
            "snapshot_2q_error",
            "edge_score",
        ],
    )

    run_config = {
        "backend_name": BACKEND_NAME,
        "runtime_channel": RUNTIME_CHANNEL,
        "use_fractional_gates": USE_FRACTIONAL_GATES,
        "noise_snapshot_iso_utc": snapshot_dt_utc.isoformat(),
        "logical_topology": LOGICAL_TOPOLOGY,
        "entangler": ENTANGLER,
        "layers": layers,
        "eps_cost": EPS_COST,
        "init_scale": INIT_SCALE,
        "theta_seed": THETA_SEED,
        "seed_transpiler": SEED_TRANSPILER,
        "optimization_level": OPTIMIZATION_LEVEL,
        "layout_method": LAYOUT_METHOD,
        "routing_method": ROUTING_METHOD,
        "stage1_maxiter": int(args.stage1_maxiter),
        "stage1_rhobeg": STAGE1_RHOBEG,
        "stage1_tol": STAGE1_TOL,
        "stage2_maxiter": int(args.stage2_maxiter),
        "stage2_maxfun": int(args.stage2_maxfun),
        "noisy_evaluation_enabled": not bool(args.skip_noisy_eval),
        "noisy_eval_runs": int(args.noisy_eval_runs),
        "noisy_eval_shots": int(args.noisy_eval_shots),
        "noisy_eval_seed_base": int(args.noisy_eval_seed_base),
        "noisy_eval_dirichlet_alpha": float(DIRICHLET_ALPHA),
        "simulator_seed": int(SIMULATOR_SEED),
        "simulator_method_noisy_eval": "density_matrix",
        "noise_thermal_relaxation": False,
        "noise_basis_gates": list(noise_model.basis_gates),
        "used_noise_fallback": bool(used_noise_fallback),
        "chosen_layout": chosen_layout,
        "layout_score": layout_score,
        "layout_meta": layout_meta,
        "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
        "output_dir": str(output_dir),
        "training_regime": "ideal_statevector_no_sample_noise",
        "cost_function": "MLQcbmCircuit.cost_fn(ptg, eps=EPS_COST)",
        "reported_metric": "KL(target || qcbm)",
        "flattening_order": "time_major",
    }
    _write_json(output_dir / "run_config.json", run_config)

    rows: list[dict[str, Any]] = []
    transpile_rows_all: list[dict[str, Any]] = []
    artifacts: list[LayerArtifact] = []

    for layer in layers:
        row, transpile_rows, artifact = train_one_layer(
            n_layers=layer,
            ptg=ptg,
            n_time=n_time,
            backend=real_backend,
            noisy_backend=noisy_backend,
            noise_model=noise_model,
            chosen_layout=chosen_layout,
            layout_score=layout_score,
            layout_meta=layout_meta,
            snapshot_dt_utc=snapshot_dt_utc,
            output_dir=output_dir,
            stage1_maxiter=int(args.stage1_maxiter),
            stage2_maxiter=int(args.stage2_maxiter),
            stage2_maxfun=int(args.stage2_maxfun),
            run_noisy_eval=not bool(args.skip_noisy_eval),
            noisy_eval_shots=int(args.noisy_eval_shots),
            noisy_eval_runs=int(args.noisy_eval_runs),
            noisy_eval_seed_base=int(args.noisy_eval_seed_base),
            noisy_eval_alpha=float(DIRICHLET_ALPHA),
            force=bool(args.force),
        )
        rows.append(row)
        if isinstance(transpile_rows, dict) and "rows" in transpile_rows:
            transpile_rows_all.extend(transpile_rows["rows"])
        else:
            transpile_rows_all.extend(transpile_rows)
        artifacts.append(artifact)

    summary_df = pd.DataFrame(rows).sort_values("n_layers").reset_index(drop=True)
    transpile_df = (
        pd.DataFrame(transpile_rows_all)
        .sort_values(["n_layers", "circuit_kind"])
        .reset_index(drop=True)
    )

    summary_columns_for_image = [
        "n_layers",
        "n_params",
        "ansatz_depth",
        "ansatz_one_qubit_ops",
        "ansatz_two_qubit_ops",
        "ansatz_swap",
        "kl_init",
        "kl_stage1",
        "kl_final",
        "kl_shots_noise_mean",
        "kl_shots_noise_std",
        "kl_noise_penalty_mean",
        "kl_time_marginal",
        "kl_time_conditional",
        "elapsed_total_s",
    ]
    summary_columns_for_image = [
        col for col in summary_columns_for_image if col in summary_df.columns
    ]
    save_table_bundle(
        summary_df,
        output_dir=tables_dir,
        stem="ideal_layer_sweep_summary",
        title="Ideal statevector QCBM layer sweep summary",
        image_columns=summary_columns_for_image,
    )
    noisy_table_columns = [
        "n_layers",
        "n_params",
        "ansatz_depth",
        "ansatz_two_qubit_ops",
        "kl_final",
        "kl_shots_noise_mean",
        "kl_shots_noise_std",
        "kl_shots_noise_sem",
        "kl_noise_penalty_mean",
        "noisy_eval_runs",
        "noisy_eval_shots",
    ]
    noisy_table_columns = [
        col for col in noisy_table_columns if col in summary_df.columns
    ]
    if "kl_shots_noise_mean" in summary_df.columns:
        save_table_bundle(
            summary_df[noisy_table_columns],
            output_dir=tables_dir,
            stem="ideal_theta_shots_noise_evaluation",
            title="Ideal-trained parameters evaluated with shots + backend noise",
            image_columns=noisy_table_columns,
        )
    save_table_bundle(
        transpile_df,
        output_dir=tables_dir,
        stem="ideal_layer_sweep_transpile_table",
        title="Backend-transpiled circuit resources by layer",
        image_columns=[
            "n_layers",
            "circuit_kind",
            "n_params",
            "depth",
            "size",
            "one_qubit_ops",
            "two_qubit_ops",
            "two_qubit_depth",
            "rzz",
            "swap",
            "measure",
        ],
    )

    aggregate_payload = {
        "summary_json": np.array(
            json.dumps(_json_ready(summary_df.to_dict(orient="records")))
        ),
        "transpile_json": np.array(
            json.dumps(_json_ready(transpile_df.to_dict(orient="records")))
        ),
        "layers": summary_df["n_layers"].to_numpy(dtype=int),
        "kl_final": summary_df["kl_final"].to_numpy(dtype=float),
        "kl_init": summary_df["kl_init"].to_numpy(dtype=float),
        "kl_stage1": summary_df["kl_stage1"].to_numpy(dtype=float),
        "ansatz_depth": summary_df["ansatz_depth"].to_numpy(dtype=int),
        "ansatz_two_qubit_ops": summary_df["ansatz_two_qubit_ops"].to_numpy(dtype=int),
        "ansatz_one_qubit_ops": summary_df["ansatz_one_qubit_ops"].to_numpy(dtype=int),
        "p_target": ptg,
    }
    for optional_col in [
        "kl_shots_noise_mean",
        "kl_shots_noise_std",
        "kl_shots_noise_sem",
        "kl_shots_noise_mean_probability",
        "kl_noise_penalty_mean",
    ]:
        if optional_col in summary_df.columns:
            aggregate_payload[optional_col] = summary_df[optional_col].to_numpy(
                dtype=float
            )
    np.savez(
        output_dir / "ideal_layer_sweep_aggregate.npz",
        **aggregate_payload,
    )

    if not args.skip_plots:
        generate_plots(
            summary_df,
            artifacts,
            output_dir=output_dir,
            n_time=n_time,
        )

    print("\n=== Ideal QCBM layer comparison complete ===")
    print(f"Results: {output_dir}")
    print("\nFinal KL summary:")
    final_print_columns = [
        "n_layers",
        "n_params",
        "ansatz_depth",
        "ansatz_two_qubit_ops",
        "kl_final",
        "kl_shots_noise_mean",
        "kl_noise_penalty_mean",
        "elapsed_total_s",
    ]
    final_print_columns = [
        col for col in final_print_columns if col in summary_df.columns
    ]
    print(
        summary_df[final_print_columns].to_string(
            index=False,
            float_format=lambda x: _format_float(x, 4),
        )
    )


if __name__ == "__main__":
    main()
