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
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from networkx.algorithms import isomorphism
from qiskit import qpy, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import minimize


def _bootstrap_paths() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    positive_exposure_dir = current.parents[1]
    for path in (src_path, positive_exposure_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _bootstrap_paths()

from exposure_utils import build_support_aware_cost  # noqa: E402
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (  # noqa: E402
    CrcaCircuit,
)
from quantum_cva.quantum_hardware_utilities.layout_utils import (  # noqa: E402
    build_backend_quality_maps,
    build_undirected_coupling_graph,
)


# ======================================================================
# Experiment configuration
# ======================================================================

BENCHMARK_RELATIVE_PATH = (
    "data/multi_asset/6q_instance/benchmark/three_asset_instance.npz"
)
RESULTS_RELATIVE_DIR = (
    "cva_pricing_pipeline/multi_asset/6q_instance/training_multi_asset/"
    "functional_encoding/statevector/positive_exposure/layers_comparison/"
    "results/statevector"
)

BACKEND_NAME = "ibm_basquecountry"
RUNTIME_CHANNEL = "ibm_cloud"
USE_FRACTIONAL_GATES = True
NOISE_SNAPSHOT_ISO_UTC = "2026-04-07T12:10:00+00:00"

M_TIME = 2
N_PRICE = 4
ANSATZ_TYPE = "heavy_hex_star"
LAYERS_GRID = (1, 2, 3, 4, 6, 8)

LOSS_MODE = "l2"
TARGET_THRESHOLD = 1e-10
RELATIVE_EPS = 1e-4
LAMBDA_POS = 10.0
LAMBDA_ZERO = 15.0

INIT_SCALE = 1.0
THETA_SEED = 12
SEED_STRIDE = 1009
SEED_TRANSPILER = 1234

OPTIMIZATION_LEVEL = 3
LAYOUT_METHOD = "trivial"
ROUTING_METHOD = "none"

STAGE1_MAXITER = 600
STAGE1_RHOBEG = 0.20
STAGE1_TOL = 1e-6
STAGE2_MAXITER = 10000
STAGE2_MAXFUN = 50000
STAGE2_FTOL = 1e-12
STAGE2_GTOL = 1e-10
STAGE2_EPS = 1e-6
STAGE2_MAXLS = 50
STAGE2_MAXCOR = 20

PLOT_SELECTED_LAYERS = (1, 2, 4, 6, 8)
LAYER_COLORS = {
    1: "#111111",
    2: "#1f77b4",
    3: "#2ca02c",
    4: "#ff7f0e",
    6: "#d62728",
    8: "#9467bd",
}
SIMULATOR_SEED = 20260407
NOISY_EVAL_SHOTS = 100_000
NOISY_EVAL_RUNS = 10
NOISY_EVAL_SEED_BASE = 42
READOUT_QUANTILE = 0.95
LOCAL_2Q_QUANTILE = 0.95


@dataclass
class LayerArtifact:
    n_layers: int
    result_path: pathlib.Path
    train_cost_history: np.ndarray
    l2_history: np.ndarray
    best_l2_history: np.ndarray
    f_target: np.ndarray
    f_target_2d: np.ndarray
    f_init: np.ndarray
    f_stage1: np.ndarray
    f_star: np.ndarray
    row: dict[str, Any]
    mse_noisy_values: np.ndarray | None = None
    f_noisy_mean: np.ndarray | None = None


# ======================================================================
# Numerical and serialization helpers
# ======================================================================


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


def _safe_result_attr(result: Any, name: str, default: Any) -> Any:
    value = getattr(result, name, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
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


def summarize_circuit(
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
        "ry": counts.get("ry", 0),
        "rz": counts.get("rz", 0),
        "sx": counts.get("sx", 0),
        "x": counts.get("x", 0),
        "cx": counts.get("cx", 0),
        "cz": counts.get("cz", 0),
        "ecr": counts.get("ecr", 0),
        "swap": counts.get("swap", 0),
        "measure": counts.get("measure", 0),
        "raw_ops_json": json.dumps(counts, sort_keys=True),
    }


# ======================================================================
# Target, objective and training
# ======================================================================


def load_positive_exposure_target() -> tuple[np.ndarray, np.ndarray, float, int, int]:
    data = np.load(REPO_ROOT / BENCHMARK_RELATIVE_PATH, allow_pickle=True)
    v_t = np.asarray(data["v_joint_t"], dtype=float)
    c_v = float(data["C_v"])
    if not np.isfinite(c_v) or c_v <= 0.0:
        raise ValueError("C_v must be finite and positive.")

    f_target_2d = np.asarray(v_t / c_v, dtype=float)
    f_target = _as_1d_float(f_target_2d)
    n_time = int(np.asarray(data["t"]).size) if "t" in data else int(2**M_TIME)
    n_price_states = int(f_target.size // n_time)

    expected_dim = 2 ** (M_TIME + N_PRICE)
    if f_target.size != expected_dim:
        raise ValueError(
            f"Positive-exposure target length is {f_target.size}, "
            f"expected {expected_dim} for m_time={M_TIME}, n_price={N_PRICE}."
        )
    if n_time != 2**M_TIME:
        raise ValueError(f"Target has {n_time} time states, expected {2**M_TIME}.")
    if n_price_states != 2**N_PRICE:
        raise ValueError(
            f"Target has {n_price_states} price states, expected {2**N_PRICE}."
        )
    if np.any((f_target < -1e-12) | (f_target > 1.0 + 1e-12)):
        raise ValueError("Normalized positive exposure target must lie in [0, 1].")
    f_target = np.clip(f_target, 0.0, 1.0)
    f_target_2d = f_target.reshape(n_time, n_price_states)
    return f_target, f_target_2d, c_v, n_time, n_price_states


def build_crca(*, n_layers: int, ansatz_type: str) -> CrcaCircuit:
    return CrcaCircuit(
        m_time=M_TIME,
        n_price=N_PRICE,
        n_layers=int(n_layers),
        ansatz_type=str(ansatz_type),
        name=f"crca_positive_exposure_{ansatz_type}_L{int(n_layers):02d}",
    )


def make_objectives(
    crca: CrcaCircuit,
    f_target: np.ndarray,
    *,
    loss_mode: str,
    target_threshold: float,
    relative_eps: float,
    lambda_pos: float,
    lambda_zero: float,
) -> tuple[Any, Any, dict[str, Any]]:
    l2_objective = crca.cost_fn(f_target, shots=None)
    if loss_mode == "l2":
        metadata = {
            "loss_name": "l2",
            "objective_shots": None,
            "stochastic_cost": False,
        }
        return l2_objective, l2_objective, metadata
    if loss_mode == "support_aware":
        objective, pos_mask, zero_mask = build_support_aware_cost(
            target_threshold,
            relative_eps,
            lambda_pos,
            lambda_zero,
            crca,
            f_target,
        )
        metadata = {
            "loss_name": "support_aware_relative_plus_zero_penalty",
            "objective_shots": None,
            "stochastic_cost": False,
            "target_threshold": float(target_threshold),
            "relative_eps": float(relative_eps),
            "lambda_pos": float(lambda_pos),
            "lambda_zero": float(lambda_zero),
            "n_positive_support_bins": int(np.count_nonzero(pos_mask)),
            "n_zero_support_bins": int(np.count_nonzero(zero_mask)),
        }
        return objective, l2_objective, metadata
    raise ValueError("loss_mode must be 'l2' or 'support_aware'.")


def exposure_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    target_threshold: float,
    relative_eps: float,
) -> dict[str, float | int]:
    target = _as_1d_float(target)
    predicted = np.clip(_as_1d_float(predicted), 0.0, 1.0)
    diff = predicted - target
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff * diff))
    pos_mask = target > float(target_threshold)
    zero_mask = ~pos_mask

    metrics: dict[str, float | int] = {
        "mse": mse,
        "rmse": float(math.sqrt(max(mse, 0.0))),
        "mae": float(np.mean(abs_diff)),
        "linf": float(np.max(abs_diff)),
        "n_positive_support_bins": int(np.count_nonzero(pos_mask)),
        "n_zero_support_bins": int(np.count_nonzero(zero_mask)),
    }

    if np.any(pos_mask):
        pos_diff = diff[pos_mask]
        pos_rel = pos_diff / (target[pos_mask] + float(relative_eps))
        metrics.update(
            {
                "positive_mse": float(np.mean(pos_diff * pos_diff)),
                "positive_rmse": float(math.sqrt(float(np.mean(pos_diff * pos_diff)))),
                "positive_relative_rmse": float(
                    math.sqrt(float(np.mean(pos_rel * pos_rel)))
                ),
                "positive_relative_mae": float(np.mean(np.abs(pos_rel))),
            }
        )
    else:
        metrics.update(
            {
                "positive_mse": math.nan,
                "positive_rmse": math.nan,
                "positive_relative_rmse": math.nan,
                "positive_relative_mae": math.nan,
            }
        )

    if np.any(zero_mask):
        zero_pred = np.abs(predicted[zero_mask])
        metrics.update(
            {
                "zero_leakage_mean_abs": float(np.mean(zero_pred)),
                "zero_leakage_max_abs": float(np.max(zero_pred)),
            }
        )
    else:
        metrics.update(
            {
                "zero_leakage_mean_abs": math.nan,
                "zero_leakage_max_abs": math.nan,
            }
        )

    return metrics


def minimize_with_histories(
    *,
    train_objective,
    l2_objective,
    x0: np.ndarray,
    method: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    x0 = np.asarray(x0, dtype=float)
    train_history: list[float] = [float(train_objective(x0))]
    l2_history: list[float] = [float(l2_objective(x0))]
    theta_history: list[np.ndarray] = [x0.copy()]
    eval_train_history: list[float] = []

    best_cost = float(train_history[0])
    best_theta = x0.copy()

    def wrapped(x: np.ndarray) -> float:
        nonlocal best_cost, best_theta
        x_arr = np.asarray(x, dtype=float)
        fx = float(train_objective(x_arr))
        eval_train_history.append(fx)
        if fx < best_cost:
            best_cost = fx
            best_theta = x_arr.copy()
        return fx

    def callback(xk: np.ndarray) -> None:
        xk_arr = np.asarray(xk, dtype=float)
        train_history.append(float(train_objective(xk_arr)))
        l2_history.append(float(l2_objective(xk_arr)))
        theta_history.append(xk_arr.copy())

    start = time.perf_counter()
    result = minimize(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )
    elapsed = time.perf_counter() - start

    x_final = np.asarray(result.x, dtype=float)
    final_train = float(train_objective(x_final))
    final_l2 = float(l2_objective(x_final))
    same_theta = np.allclose(theta_history[-1], x_final, rtol=0.0, atol=1e-15)
    same_train = abs(train_history[-1] - final_train) <= 1e-15
    if not (same_theta and same_train):
        train_history.append(final_train)
        l2_history.append(final_l2)
        theta_history.append(x_final.copy())
    if final_train < best_cost:
        best_cost = final_train
        best_theta = x_final.copy()

    return {
        "result": result,
        "theta_star": best_theta,
        "theta_last": x_final,
        "train_cost_history": np.asarray(train_history, dtype=float),
        "l2_history": np.asarray(l2_history, dtype=float),
        "theta_history": np.vstack(theta_history),
        "eval_train_cost_history": np.asarray(eval_train_history, dtype=float),
        "elapsed_time": float(elapsed),
        "best_train_cost": float(best_cost),
        "final_train_cost": float(final_train),
        "final_l2_cost": float(final_l2),
    }


def _concat_without_duplicate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    if np.isclose(a[-1], b[0], rtol=1e-12, atol=1e-15):
        return np.r_[a, b[1:]]
    return np.r_[a, b]


def _concat_theta_without_duplicate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return b
    if b.size == 0:
        return a
    if np.allclose(a[-1], b[0], rtol=0.0, atol=1e-15):
        return np.vstack([a, b[1:]])
    return np.vstack([a, b])


def bind_circuit(crca: CrcaCircuit, theta: np.ndarray, *, eval_circuit: bool):
    bind_map = {crca.theta[i]: float(theta[i]) for i in range(crca.n_params)}
    circuit = crca.qc_eval if eval_circuit else crca.qc
    return circuit.assign_parameters(bind_map, inplace=False)


def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
    dt = datetime.fromisoformat(snapshot_iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _BackendSnapshotView:
    def __init__(self, backend, snapshot_props):
        self._backend = backend
        self._snapshot_props = snapshot_props

    def properties(self, *args, **kwargs):
        return self._snapshot_props

    def __getattr__(self, name):
        return getattr(self._backend, name)


def _layout_meta_brief(layout_meta: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(layout_meta, dict):
        layout_meta = {}
    diagnostics = layout_meta.get("diagnostics", {})
    return {
        "selected_topology": str(layout_meta.get("selected_topology", "")),
        "fallback_used": bool(layout_meta.get("fallback_used", False)),
        "tried": list(layout_meta.get("tried", [])),
        "readout_threshold": diagnostics.get("readout_threshold"),
        "local_2q_threshold": diagnostics.get("local_2q_threshold"),
    }


def _crca_logical_interaction_graph(ansatz_type: str) -> nx.Graph:
    crca = build_crca(n_layers=max(2, min(LAYERS_GRID)), ansatz_type=ansatz_type)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(crca._qc_eval_meas.num_qubits)))
    for instruction in crca._qc_eval_meas.data:
        qubits = instruction.qubits
        if len(qubits) != 2:
            continue
        a = int(crca._qc_eval_meas.find_bit(qubits[0]).index)
        b = int(crca._qc_eval_meas.find_bit(qubits[1]).index)
        graph.add_edge(a, b)
    return graph


def select_crca_eval_layout_from_snapshot(
    backend,
    *,
    ansatz_type: str,
    readout_quantile: float,
    local_2q_quantile: float,
    default_score: float = 0.15,
) -> tuple[list[int], float, dict[str, Any]]:
    coupling_map = backend.configuration().coupling_map
    physical_graph = build_undirected_coupling_graph(coupling_map)
    logical_graph = _crca_logical_interaction_graph(ansatz_type)
    preferred_scores, avoided_qubits, edge_scores, diagnostics = (
        build_backend_quality_maps(
            backend,
            readout_quantile=readout_quantile,
            local_2q_quantile=local_2q_quantile,
        )
    )

    tried: list[str] = []

    def search(current_avoided: set[int]) -> tuple[list[int], float] | None:
        valid_nodes = [q for q in physical_graph.nodes if q not in current_avoided]
        physical_subgraph = physical_graph.subgraph(valid_nodes)
        matcher = isomorphism.GraphMatcher(physical_subgraph, logical_graph)
        best_layout: list[int] | None = None
        best_score = -math.inf

        for mapping in matcher.subgraph_isomorphisms_iter():
            inv_map = {logical_q: physical_q for physical_q, logical_q in mapping.items()}
            if any(q not in inv_map for q in logical_graph.nodes):
                continue
            node_term = sum(
                preferred_scores.get(inv_map[q], default_score)
                for q in logical_graph.nodes
            )
            edge_term = 0.0
            for u, v in logical_graph.edges:
                physical_edge = tuple(sorted((inv_map[u], inv_map[v])))
                edge_term += edge_scores.get(physical_edge, 0.0)
            score = float(node_term + 0.35 * edge_term)
            if score > best_score:
                best_score = score
                best_layout = [int(inv_map[i]) for i in range(logical_graph.number_of_nodes())]

        if best_layout is None:
            return None
        return best_layout, float(best_score)

    strict = search(set(avoided_qubits))
    if strict is not None:
        layout, score = strict
        fallback_used = False
    else:
        tried.append("strict crca_eval_interaction_graph failed")
        avoided_sorted = sorted(
            avoided_qubits,
            key=lambda q: preferred_scores.get(q, -1e9),
        )
        relaxed_avoided = set(avoided_sorted[: len(avoided_sorted) // 2])
        relaxed = search(relaxed_avoided)
        if relaxed is not None:
            layout, score = relaxed
            avoided_qubits = relaxed_avoided
            fallback_used = True
            tried.append("relaxed crca_eval_interaction_graph succeeded")
        else:
            tried.append("relaxed crca_eval_interaction_graph failed")
            full = search(set())
            if full is None:
                raise RuntimeError("No backend subgraph matches the CRCA eval interactions.")
            layout, score = full
            avoided_qubits = set()
            fallback_used = True
            tried.append("unfiltered crca_eval_interaction_graph succeeded")

    metadata = {
        "selected_topology": "crca_eval_interaction_graph",
        "fallback_used": bool(fallback_used),
        "tried": tried,
        "preferred_scores": preferred_scores,
        "avoided_qubits": sorted(int(q) for q in avoided_qubits),
        "edge_scores": edge_scores,
        "diagnostics": diagnostics,
        "logical_edges": [
            [int(u), int(v)] for u, v in sorted(logical_graph.edges())
        ],
    }
    return layout, float(score), metadata


def prepare_backend_noise_context(ansatz_type: str) -> dict[str, Any]:
    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    print(f"Backend: {BACKEND_NAME}")
    print(f"Noise snapshot UTC: {snapshot_dt_utc.isoformat()}")

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

    used_noise_fallback = False
    try:
        noise_model = NoiseModel.from_backend_properties(
            backend_props,
            thermal_relaxation=False,
        )
        noise_model_build = "snapshot_backend_properties_no_thermal_relaxation"
    except AttributeError:
        used_noise_fallback = True
        print(
            "Noise model could not be created from backend properties. "
            "Falling back to noise model from backend (non-snapshot)."
        )
        noise_model = NoiseModel.from_backend(real_backend)
        noise_model_build = "backend_current_properties_fallback"

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

    snapshot_backend = _BackendSnapshotView(real_backend, backend_props)
    chosen_layout, layout_score, layout_meta = select_crca_eval_layout_from_snapshot(
        snapshot_backend,
        ansatz_type=str(ansatz_type),
        readout_quantile=READOUT_QUANTILE,
        local_2q_quantile=LOCAL_2Q_QUANTILE,
    )
    chosen_layout = [int(q) for q in chosen_layout]

    print(
        "Selected noisy-eval layout: "
        f"{chosen_layout} | topology={layout_meta.get('selected_topology')} "
        f"| score={float(layout_score):.6g}"
    )

    return {
        "transpile_backend": real_backend,
        "noisy_backend": noisy_backend,
        "noise_model": noise_model,
        "chosen_layout": chosen_layout,
        "layout_score": float(layout_score),
        "layout_meta": layout_meta,
        "snapshot_dt_utc": snapshot_dt_utc,
        "used_noise_fallback": bool(used_noise_fallback),
        "noise_model_build": noise_model_build,
    }


def _bind_transpiled_measured_eval(
    crca: CrcaCircuit,
    transpiled_circuit,
    theta: np.ndarray,
):
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size != crca.n_params:
        raise ValueError(
            f"theta must have length {crca.n_params}; got {theta.size}."
        )
    param_set = set(transpiled_circuit.parameters)
    bind_map = {
        crca.theta[i]: float(theta[i])
        for i in range(crca.n_params)
        if crca.theta[i] in param_set
    }
    bound = transpiled_circuit.assign_parameters(bind_map, inplace=False)
    if bound.parameters:
        remaining = ", ".join(str(p) for p in sorted(bound.parameters, key=str))
        raise RuntimeError(f"Unbound CRCA noisy-eval parameters remain: {remaining}")
    return bound


def _counts_to_function_values_from_layout(
    counts: dict[str, int],
    *,
    dim_ctrl: int,
    n_controls: int,
    n_clbits: int,
    ctrl_clbit_indices: list[int],
    a_clbit_index: int,
) -> np.ndarray:
    n_i = np.zeros(dim_ctrl, dtype=float)
    n_i1 = np.zeros(dim_ctrl, dtype=float)

    for raw_bs, count in counts.items():
        bs = str(raw_bs).replace(" ", "")
        if len(bs) != int(n_clbits):
            raise RuntimeError(
                f"Unexpected bitstring length {len(bs)} "
                f"(expected {int(n_clbits)})."
            )

        def bit_at_clbit_index(cl_idx: int) -> int:
            pos_from_left = (int(n_clbits) - 1) - int(cl_idx)
            return 1 if bs[pos_from_left] == "1" else 0

        i_val = 0
        for q in range(int(n_controls)):
            bit = bit_at_clbit_index(ctrl_clbit_indices[q])
            i_val |= bit << q

        count_f = float(count)
        n_i[i_val] += count_f
        if bit_at_clbit_index(a_clbit_index) == 1:
            n_i1[i_val] += count_f

    out = np.zeros(dim_ctrl, dtype=float)
    mask = n_i > 0.0
    out[mask] = n_i1[mask] / n_i[mask]
    return out


def evaluate_theta_under_shots_noise(
    *,
    artifact: LayerArtifact,
    ansatz_type: str,
    noise_context: dict[str, Any],
    shots: int,
    eval_runs: int,
    seed_base: int,
    target_threshold: float,
    relative_eps: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if int(shots) <= 0:
        raise ValueError("noisy eval shots must be positive.")
    if int(eval_runs) <= 0:
        raise ValueError("noisy eval runs must be positive.")

    n_layers = int(artifact.n_layers)
    layer_dir = artifact.result_path.parent
    data = np.load(artifact.result_path, allow_pickle=True)
    theta_star = np.asarray(data["theta_star"], dtype=float).ravel()
    f_target = _as_1d_float(artifact.f_target)

    print(
        f"Evaluating CRCA theta_star under sample noise + backend noise | "
        f"L={n_layers:02d} | runs={int(eval_runs)} | shots={int(shots)}"
    )

    crca = build_crca(n_layers=n_layers, ansatz_type=ansatz_type)
    if theta_star.size != crca.n_params:
        raise ValueError(
            f"Noisy CRCA parameter mismatch for L={n_layers}: "
            f"theta_star={theta_star.size}, crca.n_params={crca.n_params}."
        )

    tqc_eval_meas = transpile(
        crca._qc_eval_meas,
        backend=noise_context["transpile_backend"],
        initial_layout=list(noise_context["chosen_layout"]),
        optimization_level=OPTIMIZATION_LEVEL,
        layout_method=LAYOUT_METHOD,
        routing_method=ROUTING_METHOD,
        seed_transpiler=SEED_TRANSPILER,
    )
    ctrl_clbit_indices, a_clbit_index = crca._extract_clbit_indices(tqc_eval_meas)
    transpiled_summary = summarize_circuit(
        tqc_eval_meas,
        n_layers=n_layers,
        n_params=crca.n_params,
        circuit_kind="eval_measured_backend_transpiled",
    )

    with open(layer_dir / f"crca_eval_measured_noisy_L{n_layers:02d}.qpy", "wb") as f:
        qpy.dump(tqc_eval_meas, f)

    mse_values: list[float] = []
    f_noisy_values: list[np.ndarray] = []
    seeds: list[int] = []

    for run_idx in range(int(eval_runs)):
        run_seed = int(seed_base) + int(run_idx)
        bound = _bind_transpiled_measured_eval(crca, tqc_eval_meas, theta_star)
        result = noise_context["noisy_backend"].run(
            bound,
            shots=int(shots),
            seed_simulator=run_seed,
        ).result()
        counts = result.get_counts()
        f_noisy = _counts_to_function_values_from_layout(
            counts,
            dim_ctrl=crca.dim_controls,
            n_controls=crca.n_controls,
            n_clbits=len(tqc_eval_meas.clbits),
            ctrl_clbit_indices=ctrl_clbit_indices,
            a_clbit_index=a_clbit_index,
        )
        diff = f_noisy - f_target
        mse_val = float(np.mean(diff * diff))

        seeds.append(run_seed)
        mse_values.append(mse_val)
        f_noisy_values.append(f_noisy)
        print(
            f"  [noise eval {run_idx + 1:02d}/{int(eval_runs):02d}] "
            f"seed={run_seed} | MSE={mse_val:.8e}"
        )

    mse_arr = np.asarray(mse_values, dtype=float)
    f_noisy_arr = np.vstack(f_noisy_values)
    f_noisy_mean = np.mean(f_noisy_arr, axis=0)
    mean_metrics = exposure_metrics(
        f_target,
        f_noisy_mean,
        target_threshold=target_threshold,
        relative_eps=relative_eps,
    )
    snapshot_dt_utc = noise_context["snapshot_dt_utc"]
    layout_meta = _layout_meta_brief(noise_context["layout_meta"])

    row = {
        "mse_shots_noise_mean": float(np.mean(mse_arr)),
        "mse_shots_noise_std": float(np.std(mse_arr, ddof=1))
        if mse_arr.size > 1
        else 0.0,
        "mse_shots_noise_sem": float(np.std(mse_arr, ddof=1) / math.sqrt(mse_arr.size))
        if mse_arr.size > 1
        else 0.0,
        "mse_shots_noise_min": float(np.min(mse_arr)),
        "mse_shots_noise_max": float(np.max(mse_arr)),
        "mse_shots_noise_median": float(np.median(mse_arr)),
        "mse_shots_noise_mean_function": float(mean_metrics["mse"]),
        "rmse_shots_noise_mean_function": float(mean_metrics["rmse"]),
        "mae_shots_noise_mean_function": float(mean_metrics["mae"]),
        "positive_relative_rmse_shots_noise_mean_function": float(
            mean_metrics["positive_relative_rmse"]
        ),
        "zero_leakage_mean_abs_shots_noise_mean_function": float(
            mean_metrics["zero_leakage_mean_abs"]
        ),
        "noisy_eval_runs": int(eval_runs),
        "noisy_eval_shots": int(shots),
        "noisy_eval_seed_base": int(seed_base),
        "backend_name": BACKEND_NAME,
        "runtime_channel": RUNTIME_CHANNEL,
        "use_fractional_gates": bool(USE_FRACTIONAL_GATES),
        "noise_snapshot_iso_utc": snapshot_dt_utc.isoformat(),
        "noise_model_build": str(noise_context["noise_model_build"]),
        "noise_model_fallback_used": bool(noise_context["used_noise_fallback"]),
        "simulator_seed": int(SIMULATOR_SEED),
        "seed_transpiler": int(SEED_TRANSPILER),
        "optimization_level": int(OPTIMIZATION_LEVEL),
        "layout_method": LAYOUT_METHOD,
        "routing_method": ROUTING_METHOD,
        "requested_topology": str(ansatz_type),
        "effective_topology": layout_meta["selected_topology"],
        "chosen_layout": ",".join(str(q) for q in noise_context["chosen_layout"]),
        "layout_score": float(noise_context["layout_score"]),
        "layout_fallback_used": bool(layout_meta["fallback_used"]),
        "layout_tried": "; ".join(str(x) for x in layout_meta["tried"]),
        "readout_threshold": layout_meta["readout_threshold"],
        "local_2q_threshold": layout_meta["local_2q_threshold"],
        "noisy_eval_transpiled_depth": int(transpiled_summary["depth"]),
        "noisy_eval_transpiled_size": int(transpiled_summary["size"]),
        "noisy_eval_transpiled_width": int(transpiled_summary["width"]),
        "noisy_eval_transpiled_two_qubit_ops": int(
            transpiled_summary["two_qubit_ops"]
        ),
        "noisy_eval_transpiled_two_qubit_depth": int(
            transpiled_summary["two_qubit_depth"]
        ),
        "noisy_eval_transpiled_measure": int(transpiled_summary["measure"]),
    }

    np.savez(
        layer_dir / f"noisy_eval_L{n_layers:02d}.npz",
        theta_star=theta_star,
        f_target=f_target,
        f_noisy_values=f_noisy_arr,
        f_noisy_mean=f_noisy_mean,
        mse_shots_noise_values=mse_arr,
        noisy_eval_seeds=np.asarray(seeds, dtype=int),
        noisy_eval_shots=np.int64(shots),
        noisy_eval_runs=np.int64(eval_runs),
        noisy_eval_seed_base=np.int64(seed_base),
        summary_row_json=np.array(json.dumps(_json_ready(row), sort_keys=True)),
    )
    return row, mse_arr, f_noisy_mean


def load_cached_noisy_evaluation(artifact: LayerArtifact) -> bool:
    noisy_path = artifact.result_path.parent / f"noisy_eval_L{int(artifact.n_layers):02d}.npz"
    if not noisy_path.exists():
        return False

    data = np.load(noisy_path, allow_pickle=True)
    if "summary_row_json" in data.files:
        artifact.row.update(json.loads(str(data["summary_row_json"].item())))
    if "mse_shots_noise_values" in data.files:
        artifact.mse_noisy_values = np.asarray(
            data["mse_shots_noise_values"],
            dtype=float,
        )
    if "f_noisy_mean" in data.files:
        artifact.f_noisy_mean = np.asarray(data["f_noisy_mean"], dtype=float)
    if (
        "mse_shots_noise_mean" in artifact.row
        and "mse_noise_penalty_mean" not in artifact.row
        and "mse_final" in artifact.row
    ):
        artifact.row["mse_noise_penalty_mean"] = float(
            artifact.row["mse_shots_noise_mean"] - artifact.row["mse_final"]
        )
    return "mse_shots_noise_mean" in artifact.row


def complete_noisy_evaluation(
    artifacts: list[LayerArtifact],
    *,
    ansatz_type: str,
    noise_context: dict[str, Any],
    shots: int,
    eval_runs: int,
    seed_base: int,
    target_threshold: float,
    relative_eps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in sorted(artifacts, key=lambda item: int(item.n_layers)):
        summary_path = artifact.result_path.parent / "summary_row.json"
        noisy_path = (
            artifact.result_path.parent
            / f"noisy_eval_L{int(artifact.n_layers):02d}.npz"
        )
        if (
            "mse_shots_noise_mean" in artifact.row
            and noisy_path.exists()
            and load_cached_noisy_evaluation(artifact)
        ):
            _write_json(summary_path, artifact.row)
            rows.append(artifact.row)
            continue

        noisy_row, mse_noisy_values, f_noisy_mean = evaluate_theta_under_shots_noise(
            artifact=artifact,
            ansatz_type=ansatz_type,
            noise_context=noise_context,
            shots=int(shots),
            eval_runs=int(eval_runs),
            seed_base=int(seed_base),
            target_threshold=float(target_threshold),
            relative_eps=float(relative_eps),
        )
        artifact.row.update(noisy_row)
        artifact.row["mse_noise_penalty_mean"] = float(
            artifact.row["mse_shots_noise_mean"] - artifact.row["mse_final"]
        )
        artifact.mse_noisy_values = mse_noisy_values
        artifact.f_noisy_mean = f_noisy_mean
        _write_json(summary_path, artifact.row)
        rows.append(artifact.row)

    return pd.DataFrame(rows).sort_values("n_layers").reset_index(drop=True)


def train_one_layer(
    *,
    n_layers: int,
    ansatz_type: str,
    f_target: np.ndarray,
    f_target_2d: np.ndarray,
    c_v: float,
    n_time: int,
    n_price_states: int,
    output_dir: pathlib.Path,
    loss_mode: str,
    init_scale: float,
    theta_seed: int,
    seed_stride: int,
    target_threshold: float,
    relative_eps: float,
    lambda_pos: float,
    lambda_zero: float,
    stage1_maxiter: int,
    stage1_rhobeg: float,
    stage1_tol: float,
    stage2_maxiter: int,
    stage2_maxfun: int,
    stage2_ftol: float,
    stage2_gtol: float,
    stage2_eps: float,
    stage2_maxls: int,
    stage2_maxcor: int,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], LayerArtifact]:
    layer_dir = output_dir / f"L{int(n_layers):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    result_path = layer_dir / f"crca_positive_exposure_statevector_L{int(n_layers):02d}.npz"
    summary_path = layer_dir / "summary_row.json"
    resource_path = layer_dir / "resource_rows.json"

    if result_path.exists() and summary_path.exists() and resource_path.exists() and not force:
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        resource_rows = json.loads(resource_path.read_text(encoding="utf-8"))
        data = np.load(result_path, allow_pickle=True)
        artifact = LayerArtifact(
            n_layers=int(n_layers),
            result_path=result_path,
            train_cost_history=np.asarray(data["train_cost_history"], dtype=float),
            l2_history=np.asarray(data["l2_history"], dtype=float),
            best_l2_history=np.asarray(data["best_l2_history"], dtype=float),
            f_target=np.asarray(data["f_target"], dtype=float),
            f_target_2d=np.asarray(data["f_target_2d"], dtype=float),
            f_init=np.asarray(data["f_init_statevector"], dtype=float),
            f_stage1=np.asarray(data["f_stage1_statevector"], dtype=float),
            f_star=np.asarray(data["f_star_statevector"], dtype=float),
            row=row,
        )
        return row, resource_rows[0], resource_rows[1], artifact

    crca = build_crca(n_layers=int(n_layers), ansatz_type=ansatz_type)
    ansatz_summary = summarize_circuit(
        crca.qc,
        n_layers=int(n_layers),
        n_params=crca.n_params,
        circuit_kind="ansatz_logical",
    )
    eval_summary = summarize_circuit(
        crca.qc_eval,
        n_layers=int(n_layers),
        n_params=crca.n_params,
        circuit_kind="eval_logical",
    )
    for resource_row in (ansatz_summary, eval_summary):
        resource_row["ansatz_type"] = ansatz_type
        resource_row["training_mode"] = "statevector"

    train_objective, l2_objective, loss_metadata = make_objectives(
        crca,
        f_target,
        loss_mode=loss_mode,
        target_threshold=target_threshold,
        relative_eps=relative_eps,
        lambda_pos=lambda_pos,
        lambda_zero=lambda_zero,
    )

    rng = np.random.default_rng(int(theta_seed) + int(seed_stride) * int(n_layers))
    theta_init = float(init_scale) * rng.standard_normal(crca.n_params).astype(float)
    f_init = np.asarray(crca.function_values(theta_init, shots=None), dtype=float).ravel()

    stage1 = minimize_with_histories(
        train_objective=train_objective,
        l2_objective=l2_objective,
        x0=theta_init,
        method="COBYLA",
        options={
            "maxiter": int(stage1_maxiter),
            "rhobeg": float(stage1_rhobeg),
            "tol": float(stage1_tol),
            "disp": False,
        },
    )

    theta_stage1 = np.asarray(stage1["theta_star"], dtype=float)
    f_stage1 = np.asarray(crca.function_values(theta_stage1, shots=None), dtype=float).ravel()

    stage2 = minimize_with_histories(
        train_objective=train_objective,
        l2_objective=l2_objective,
        x0=theta_stage1,
        method="L-BFGS-B",
        options={
            "maxiter": int(stage2_maxiter),
            "maxfun": int(stage2_maxfun),
            "ftol": float(stage2_ftol),
            "gtol": float(stage2_gtol),
            "eps": float(stage2_eps),
            "maxls": int(stage2_maxls),
            "maxcor": int(stage2_maxcor),
            "disp": False,
        },
    )

    theta_star = np.asarray(stage2["theta_star"], dtype=float)
    theta_last = np.asarray(stage2["theta_last"], dtype=float)
    f_star = np.asarray(crca.function_values(theta_star, shots=None), dtype=float).ravel()

    train_cost_history = _concat_without_duplicate(
        np.asarray(stage1["train_cost_history"], dtype=float),
        np.asarray(stage2["train_cost_history"], dtype=float),
    )
    l2_history = _concat_without_duplicate(
        np.asarray(stage1["l2_history"], dtype=float),
        np.asarray(stage2["l2_history"], dtype=float),
    )
    theta_history = _concat_theta_without_duplicate(
        np.asarray(stage1["theta_history"], dtype=float),
        np.asarray(stage2["theta_history"], dtype=float),
    )
    eval_train_cost_history = np.r_[
        np.asarray(stage1["eval_train_cost_history"], dtype=float),
        np.asarray(stage2["eval_train_cost_history"], dtype=float),
    ]
    best_l2_history = np.minimum.accumulate(l2_history)
    best_l2_idx = np.flatnonzero(
        np.r_[True, best_l2_history[1:] < best_l2_history[:-1]]
    )

    metrics_init = exposure_metrics(
        f_target,
        f_init,
        target_threshold=target_threshold,
        relative_eps=relative_eps,
    )
    metrics_stage1 = exposure_metrics(
        f_target,
        f_stage1,
        target_threshold=target_threshold,
        relative_eps=relative_eps,
    )
    metrics_final = exposure_metrics(
        f_target,
        f_star,
        target_threshold=target_threshold,
        relative_eps=relative_eps,
    )

    elapsed_total = float(stage1["elapsed_time"] + stage2["elapsed_time"])
    final_train_cost = float(train_objective(theta_star))
    initial_train_cost = float(train_objective(theta_init))
    stage1_train_cost = float(train_objective(theta_stage1))

    row = {
        "n_layers": int(n_layers),
        "ansatz_type": ansatz_type,
        "training_mode": "statevector",
        "loss_mode": loss_mode,
        "theta_seed": int(theta_seed) + int(seed_stride) * int(n_layers),
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_controls": int(crca.n_controls),
        "n_work": int(crca.n_work),
        "n_params": int(crca.n_params),
        "n_params_per_layer": int(crca.n_params_per_layer),
        "target_dim": int(f_target.size),
        "n_time_states": int(n_time),
        "n_price_states": int(n_price_states),
        "target_positive_max": float(np.max(f_target)),
        "target_positive_mean": float(np.mean(f_target)),
        "target_positive_support": int(np.count_nonzero(f_target > target_threshold)),
        "train_cost_init": initial_train_cost,
        "train_cost_stage1": stage1_train_cost,
        "train_cost_final": final_train_cost,
        "mse_init": float(metrics_init["mse"]),
        "mse_stage1": float(metrics_stage1["mse"]),
        "mse_final": float(metrics_final["mse"]),
        "rmse_init": float(metrics_init["rmse"]),
        "rmse_stage1": float(metrics_stage1["rmse"]),
        "rmse_final": float(metrics_final["rmse"]),
        "mae_final": float(metrics_final["mae"]),
        "linf_final": float(metrics_final["linf"]),
        "positive_rmse_final": float(metrics_final["positive_rmse"]),
        "positive_relative_rmse_final": float(
            metrics_final["positive_relative_rmse"]
        ),
        "positive_relative_mae_final": float(metrics_final["positive_relative_mae"]),
        "zero_leakage_mean_abs_final": float(metrics_final["zero_leakage_mean_abs"]),
        "zero_leakage_max_abs_final": float(metrics_final["zero_leakage_max_abs"]),
        "elapsed_total_s": elapsed_total,
        "stage1_elapsed_s": float(stage1["elapsed_time"]),
        "stage2_elapsed_s": float(stage2["elapsed_time"]),
        "stage1_nfev": int(_safe_result_attr(stage1["result"], "nfev", -1)),
        "stage2_nfev": int(_safe_result_attr(stage2["result"], "nfev", -1)),
        "stage1_success": bool(_safe_result_attr(stage1["result"], "success", False)),
        "stage2_success": bool(_safe_result_attr(stage2["result"], "success", False)),
        "stage1_message": str(_safe_result_attr(stage1["result"], "message", "")),
        "stage2_message": str(_safe_result_attr(stage2["result"], "message", "")),
        "n_history_points": int(l2_history.size),
        "logical_ansatz_depth": int(ansatz_summary["depth"]),
        "logical_ansatz_size": int(ansatz_summary["size"]),
        "logical_ansatz_width": int(ansatz_summary["width"]),
        "logical_ansatz_one_qubit_ops": int(ansatz_summary["one_qubit_ops"]),
        "logical_ansatz_two_qubit_ops": int(ansatz_summary["two_qubit_ops"]),
        "logical_ansatz_two_qubit_depth": int(ansatz_summary["two_qubit_depth"]),
        "logical_ansatz_swap": int(ansatz_summary["swap"]),
        "logical_eval_depth": int(eval_summary["depth"]),
        "logical_eval_size": int(eval_summary["size"]),
        "logical_eval_width": int(eval_summary["width"]),
        "logical_eval_one_qubit_ops": int(eval_summary["one_qubit_ops"]),
        "logical_eval_two_qubit_ops": int(eval_summary["two_qubit_ops"]),
        "logical_eval_two_qubit_depth": int(eval_summary["two_qubit_depth"]),
        "logical_eval_swap": int(eval_summary["swap"]),
        "neg_log10_mse": float(-math.log10(max(float(metrics_final["mse"]), 1e-15))),
        "neg_log10_mse_per_eval_two_qubit_depth": float(
            -math.log10(max(float(metrics_final["mse"]), 1e-15))
            / max(int(eval_summary["two_qubit_depth"]), 1)
        ),
        "mse_times_eval_two_qubit_depth": float(
            float(metrics_final["mse"]) * max(int(eval_summary["two_qubit_depth"]), 1)
        ),
    }

    metadata = {
        "model": "CRCA",
        "task": "positive_exposure",
        "ansatz_type": ansatz_type,
        "training_mode": "statevector",
        "ancilla_observable": "P(a=1 | control=i)",
        "flattening_order": "time_major",
        "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
        "m_time": M_TIME,
        "n_price": N_PRICE,
        "n_layers": int(n_layers),
        "n_controls": int(crca.n_controls),
        "n_parameters": int(crca.n_params),
        "optimizer": "2-stage (COBYLA -> L-BFGS-B)",
        "stage1_optimizer": "COBYLA",
        "stage1_maxiter": int(stage1_maxiter),
        "stage1_tol": float(stage1_tol),
        "stage1_rhobeg": float(stage1_rhobeg),
        "stage2_optimizer": "L-BFGS-B",
        "stage2_maxiter": int(stage2_maxiter),
        "stage2_maxfun": int(stage2_maxfun),
        "stage2_ftol": float(stage2_ftol),
        "stage2_gtol": float(stage2_gtol),
        "stage2_eps": float(stage2_eps),
        "shots": None,
        "statevector_training": True,
        "theta_seed": int(theta_seed) + int(seed_stride) * int(n_layers),
        "init_scale": float(init_scale),
        "C_v": float(c_v),
        "note": (
            "CRCA positive-exposure layer comparison. All optimization "
            "objectives and evaluations use exact statevector function values "
            "with shots=None."
        ),
    }
    metadata.update(loss_metadata)

    with open(layer_dir / f"crca_template_L{int(n_layers):02d}.qpy", "wb") as f:
        qpy.dump(crca.qc, f)
    with open(layer_dir / f"crca_eval_template_L{int(n_layers):02d}.qpy", "wb") as f:
        qpy.dump(crca.qc_eval, f)
    with open(layer_dir / f"crca_eval_bound_L{int(n_layers):02d}.qpy", "wb") as f:
        qpy.dump(bind_circuit(crca, theta_star, eval_circuit=True), f)

    np.savez(
        result_path,
        theta_init=theta_init,
        theta_stage1=theta_stage1,
        theta_last=theta_last,
        theta_star=theta_star,
        theta_history=theta_history,
        train_cost_history=train_cost_history,
        l2_history=l2_history,
        best_l2_history=best_l2_history,
        best_l2_idx=best_l2_idx,
        stage1_train_cost_history=np.asarray(stage1["train_cost_history"], dtype=float),
        stage1_l2_history=np.asarray(stage1["l2_history"], dtype=float),
        stage2_train_cost_history=np.asarray(stage2["train_cost_history"], dtype=float),
        stage2_l2_history=np.asarray(stage2["l2_history"], dtype=float),
        eval_train_cost_history=eval_train_cost_history,
        f_target=f_target,
        f_target_2d=f_target_2d,
        f_init_statevector=f_init,
        f_stage1_statevector=f_stage1,
        f_star_statevector=f_star,
        metrics_init=np.array(metrics_init, dtype=object),
        metrics_stage1=np.array(metrics_stage1, dtype=object),
        metrics_final=np.array(metrics_final, dtype=object),
        summary_row_json=np.array(json.dumps(_json_ready(row)), dtype=object),
        metadata=np.array(metadata, dtype=object),
        C_v=np.float64(c_v),
        elapsed_time=np.float64(elapsed_total),
        n_layers=np.int64(n_layers),
        theta_seed=np.int64(int(theta_seed) + int(seed_stride) * int(n_layers)),
    )

    _write_json(summary_path, row)
    _write_json(resource_path, [ansatz_summary, eval_summary])
    _write_json(layer_dir / "metadata.json", metadata)

    artifact = LayerArtifact(
        n_layers=int(n_layers),
        result_path=result_path,
        train_cost_history=train_cost_history,
        l2_history=l2_history,
        best_l2_history=best_l2_history,
        f_target=f_target,
        f_target_2d=f_target_2d,
        f_init=f_init,
        f_stage1=f_stage1,
        f_star=f_star,
        row=row,
    )
    return row, ansatz_summary, eval_summary, artifact


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output_dir: pathlib.Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_layers(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = summary_df["n_layers"].to_numpy()
    ax.semilogy(
        x,
        summary_df["mse_init"],
        marker="o",
        color="#7b8794",
        linestyle="--",
        label="Initial",
    )
    ax.semilogy(
        x,
        summary_df["mse_stage1"],
        marker="s",
        color="#2f80ed",
        linestyle="-.",
        label="After COBYLA",
    )
    ax.semilogy(
        x,
        summary_df["mse_final"],
        marker="D",
        color="#c0392b",
        label="After L-BFGS-B",
    )
    best_idx = int(summary_df["mse_final"].idxmin())
    best_row = summary_df.loc[best_idx]
    ax.scatter(
        [best_row["n_layers"]],
        [best_row["mse_final"]],
        s=90,
        facecolor="white",
        edgecolor="#111111",
        linewidth=1.3,
        zorder=5,
    )
    ax.annotate(
        f"best L={int(best_row['n_layers'])}\nMSE={best_row['mse_final']:.2e}",
        xy=(best_row["n_layers"], best_row["mse_final"]),
        xytext=(8, 18),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.8},
    )
    ax.set_xlabel("CRCA layers")
    ax.set_ylabel("Mean-squared exposure error")
    ax.set_title("Statevector CRCA positive-exposure training")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "positive_exposure_mse_vs_layers")


def plot_error_decomposition(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = summary_df["n_layers"].to_numpy()
    ax.semilogy(
        x,
        summary_df["rmse_final"],
        marker="D",
        color="#111827",
        label="Global RMSE",
    )
    ax.semilogy(
        x,
        summary_df["positive_rmse_final"],
        marker="o",
        color="#0f766e",
        label="Positive-support RMSE",
    )
    ax.semilogy(
        x,
        summary_df["zero_leakage_mean_abs_final"],
        marker="s",
        color="#b45309",
        label="Mean zero-support leakage",
    )
    ax.set_xlabel("CRCA layers")
    ax.set_ylabel("Exposure error")
    ax.set_title("Error components across positive and zero support")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "positive_exposure_error_components_vs_layers")


def plot_statevector_vs_shots_noise_mse(
    summary_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    if "mse_shots_noise_mean" not in summary_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    x = summary_df["n_layers"].to_numpy(dtype=int)
    yerr = (
        summary_df["mse_shots_noise_std"].to_numpy(dtype=float)
        if "mse_shots_noise_std" in summary_df.columns
        else None
    )

    axes[0].semilogy(
        x,
        summary_df["mse_final"],
        marker="D",
        color="#111827",
        label="Ideal statevector evaluation",
    )
    axes[0].errorbar(
        x,
        summary_df["mse_shots_noise_mean"],
        yerr=yerr,
        marker="o",
        color="#c2410c",
        capsize=3,
        label="Shots + backend noise",
    )
    axes[0].set_xlabel("CRCA layers")
    axes[0].set_ylabel("Mean-squared exposure error")
    axes[0].set_title("Ideal parameters evaluated under noise")
    axes[0].legend(frameon=True)

    penalty = (
        summary_df["mse_shots_noise_mean"].to_numpy(dtype=float)
        - summary_df["mse_final"].to_numpy(dtype=float)
    )
    axes[1].bar(
        x,
        penalty,
        width=0.62,
        color="#f97316",
        edgecolor="#7c2d12",
        alpha=0.86,
    )
    axes[1].axhline(0.0, color="#111827", linewidth=0.9)
    axes[1].set_xlabel("CRCA layers")
    axes[1].set_ylabel("MSE noise penalty")
    axes[1].set_title("Additional MSE from sample noise + noise model")

    for ax in axes:
        ax.set_xticks(x)
        ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    save_figure(fig, output_dir, "positive_exposure_ideal_vs_shots_noise_mse")


def plot_resource_scaling(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharex=True)
    x = summary_df["n_layers"].to_numpy()

    axes[0].plot(
        x,
        summary_df["logical_eval_depth"],
        marker="o",
        color="#1f6feb",
        label="Eval depth",
    )
    axes[0].plot(
        x,
        summary_df["logical_eval_two_qubit_depth"],
        marker="s",
        color="#a21caf",
        label="Eval 2q depth",
    )
    axes[0].plot(
        x,
        summary_df["logical_ansatz_two_qubit_depth"],
        marker="^",
        color="#64748b",
        label="Ansatz 2q depth",
    )
    axes[0].set_xlabel("CRCA layers")
    axes[0].set_ylabel("Logical circuit depth")
    axes[0].set_title("Depth scaling")
    axes[0].legend(frameon=True)

    axes[1].plot(
        x,
        summary_df["logical_eval_one_qubit_ops"],
        marker="o",
        color="#0f766e",
        label="1q gates",
    )
    axes[1].plot(
        x,
        summary_df["logical_eval_two_qubit_ops"],
        marker="s",
        color="#c2410c",
        label="2q gates",
    )
    axes[1].plot(
        x,
        summary_df["logical_eval_swap"],
        marker="^",
        color="#475569",
        label="SWAP",
    )
    axes[1].set_xlabel("CRCA layers")
    axes[1].set_ylabel("Logical eval operation count")
    axes[1].set_title("Gate-count scaling")
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.set_xticks(x)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "positive_exposure_resources_vs_layers")


def plot_error_vs_resources(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    y = summary_df["mse_final"].to_numpy()
    layers = summary_df["n_layers"].to_numpy()
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(layers.min()), vmax=float(layers.max()))

    x_specs = [
        ("n_params", "Parameters"),
        ("logical_eval_depth", "Logical eval depth"),
        ("logical_eval_two_qubit_ops", "Logical eval 2q gates"),
    ]
    sc = None
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
                (row[col], row["mse_final"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MSE")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_title("Accuracy-resource frontier")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.84, pad=0.02)
        cbar.set_label("Layers")
    save_figure(fig, output_dir, "positive_exposure_mse_vs_resources")


def plot_training_curves(
    artifacts: list[LayerArtifact],
    output_dir: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for artifact in artifacts:
        x = np.arange(artifact.best_l2_history.size)
        color = LAYER_COLORS.get(int(artifact.n_layers), "#444444")
        ax.semilogy(
            x,
            np.maximum(artifact.best_l2_history, 1e-15),
            color=color,
            alpha=0.9,
            label=f"L={artifact.n_layers}",
        )
    ax.set_xlabel("Recorded optimizer checkpoint")
    ax.set_ylabel("Best-so-far MSE")
    ax.set_title("Optimization trajectories across CRCA depths")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=3, frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "positive_exposure_training_trajectories")


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
    target = selected[0].f_target
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
            artifact.f_star[order],
            marker=".",
            color=color,
            alpha=0.9,
            label=f"L={artifact.n_layers}",
        )
        axes[1].plot(
            x,
            artifact.f_star[order] - target[order],
            marker=".",
            color=color,
            alpha=0.9,
            label=f"L={artifact.n_layers}",
        )
    axes[0].set_ylabel("Normalized exposure")
    axes[0].set_title("Fit on largest positive-exposure states")
    axes[0].legend(ncol=4, frameon=True)
    axes[1].axhline(0.0, color="#111111", linewidth=0.9)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Ranked target state")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "positive_exposure_top_state_fit_and_residuals")


def plot_heatmaps(artifacts: list[LayerArtifact], output_dir: pathlib.Path) -> None:
    best = min(artifacts, key=lambda artifact: float(artifact.row["mse_final"]))
    target = best.f_target_2d
    pred = best.f_star.reshape(target.shape)
    residual = pred - target
    vmax = max(float(np.max(np.abs(residual))), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.8), constrained_layout=True)
    im0 = axes[0, 0].imshow(target, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Target positive exposure")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.82)

    im1 = axes[0, 1].imshow(pred, aspect="auto", cmap="viridis")
    axes[0, 1].set_title(f"CRCA prediction, L={best.n_layers}")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.82)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im2 = axes[1, 0].imshow(residual, aspect="auto", cmap="coolwarm", norm=norm)
    axes[1, 0].set_title("Signed residual")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.82)

    im3 = axes[1, 1].imshow(np.abs(residual), aspect="auto", cmap="magma")
    axes[1, 1].set_title("Absolute residual")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.82)

    for ax in axes.ravel():
        ax.set_xlabel("Price-state index")
        ax.set_ylabel("Time index")
    save_figure(fig, output_dir, "positive_exposure_time_price_residual_heatmaps")


def generate_plots(
    summary_df: pd.DataFrame,
    artifacts: list[LayerArtifact],
    *,
    output_dir: pathlib.Path,
) -> None:
    tables_dir = output_dir.parent / "tables"
    if "mse_shots_noise_mean" in summary_df.columns:
        noisy_columns = [
            "n_layers",
            "mse_final",
            "mse_shots_noise_mean",
            "mse_shots_noise_std",
            "mse_noise_penalty_mean",
            "noisy_eval_transpiled_depth",
            "noisy_eval_transpiled_two_qubit_depth",
            "noisy_eval_transpiled_two_qubit_ops",
            "noisy_eval_shots",
            "noisy_eval_runs",
            "backend_name",
            "noise_snapshot_iso_utc",
            "chosen_layout",
        ]
        save_table_bundle(
            summary_df[[col for col in noisy_columns if col in summary_df.columns]],
            output_dir=tables_dir,
            stem="positive_exposure_statevector_vs_backend_noise_eval",
            title=(
                "CRCA positive exposure statevector parameters under "
                "shots and backend noise"
            ),
            image_columns=[col for col in noisy_columns if col in summary_df.columns],
        )
    backend_resource_columns = [
        "n_layers",
        "backend_name",
        "noise_snapshot_iso_utc",
        "requested_topology",
        "effective_topology",
        "chosen_layout",
        "layout_score",
        "layout_fallback_used",
        "optimization_level",
        "layout_method",
        "routing_method",
        "logical_eval_depth",
        "logical_eval_two_qubit_depth",
        "logical_eval_two_qubit_ops",
        "noisy_eval_transpiled_depth",
        "noisy_eval_transpiled_two_qubit_depth",
        "noisy_eval_transpiled_two_qubit_ops",
        "noisy_eval_transpiled_size",
        "noisy_eval_transpiled_width",
        "noisy_eval_transpiled_measure",
    ]
    backend_resource_columns = [
        col for col in backend_resource_columns if col in summary_df.columns
    ]
    if "noisy_eval_transpiled_depth" in backend_resource_columns:
        save_table_bundle(
            summary_df[backend_resource_columns],
            output_dir=tables_dir,
            stem="positive_exposure_backend_transpiled_resource_table",
            title="CRCA positive exposure resources after backend transpilation",
            image_columns=backend_resource_columns,
        )
    plot_error_vs_layers(summary_df, output_dir)
    plot_error_decomposition(summary_df, output_dir)
    plot_statevector_vs_shots_noise_mse(summary_df, output_dir)
    plot_resource_scaling(summary_df, output_dir)
    plot_error_vs_resources(summary_df, output_dir)
    plot_training_curves(artifacts, output_dir)
    plot_top_state_fit(artifacts, output_dir)
    plot_heatmaps(artifacts, output_dir)


# ======================================================================
# CLI and orchestration
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Layer sweep for CRCA positive-exposure functional encoding. "
            "All training and evaluation objectives use exact statevector "
            "function values, i.e. shots=None."
        )
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(LAYERS_GRID),
        help="CRCA layer counts to train.",
    )
    parser.add_argument(
        "--ansatz-type",
        default=ANSATZ_TYPE,
        choices=["standard", "native_tree", "heavy_hex_star"],
        help="CRCA ansatz topology.",
    )
    parser.add_argument(
        "--loss-mode",
        default=LOSS_MODE,
        choices=["l2", "support_aware"],
        help="Training objective. Both modes are evaluated with statevector.",
    )
    parser.add_argument("--init-scale", type=float, default=INIT_SCALE)
    parser.add_argument("--theta-seed", type=int, default=THETA_SEED)
    parser.add_argument("--seed-stride", type=int, default=SEED_STRIDE)
    parser.add_argument("--target-threshold", type=float, default=TARGET_THRESHOLD)
    parser.add_argument("--relative-eps", type=float, default=RELATIVE_EPS)
    parser.add_argument("--lambda-pos", type=float, default=LAMBDA_POS)
    parser.add_argument("--lambda-zero", type=float, default=LAMBDA_ZERO)
    parser.add_argument("--stage1-maxiter", type=int, default=STAGE1_MAXITER)
    parser.add_argument("--stage1-rhobeg", type=float, default=STAGE1_RHOBEG)
    parser.add_argument("--stage1-tol", type=float, default=STAGE1_TOL)
    parser.add_argument("--stage2-maxiter", type=int, default=STAGE2_MAXITER)
    parser.add_argument("--stage2-maxfun", type=int, default=STAGE2_MAXFUN)
    parser.add_argument("--stage2-ftol", type=float, default=STAGE2_FTOL)
    parser.add_argument("--stage2-gtol", type=float, default=STAGE2_GTOL)
    parser.add_argument("--stage2-eps", type=float, default=STAGE2_EPS)
    parser.add_argument("--stage2-maxls", type=int, default=STAGE2_MAXLS)
    parser.add_argument("--stage2-maxcor", type=int, default=STAGE2_MAXCOR)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=REPO_ROOT / RESULTS_RELATIVE_DIR,
        help="Directory for results, tables and figures.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Regenerate aggregate tables and figures from existing layer NPZ files. "
            "Unless --skip-noisy-eval is set, missing noisy evaluations are run."
        ),
    )
    parser.add_argument(
        "--skip-noisy-eval",
        action="store_true",
        help="Skip backend-noise finite-shot evaluation of the trained statevector theta.",
    )
    parser.add_argument(
        "--noisy-eval-runs",
        type=int,
        default=NOISY_EVAL_RUNS,
        help="Number of repeated backend-noise finite-shot evaluations per layer.",
    )
    parser.add_argument(
        "--noisy-eval-shots",
        type=int,
        default=NOISY_EVAL_SHOTS,
        help="Shots per backend-noise finite-shot evaluation.",
    )
    parser.add_argument(
        "--noisy-eval-seed-base",
        type=int,
        default=NOISY_EVAL_SEED_BASE,
        help="Base seed for repeated backend-noise finite-shot evaluations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate target/circuit construction and print resource rows; no training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain layers even if result files already exist.",
    )
    return parser.parse_args()


def load_artifact_from_result(result_path: pathlib.Path) -> LayerArtifact:
    data = np.load(result_path, allow_pickle=True)
    summary_path = result_path.parent / "summary_row.json"
    if summary_path.exists():
        row = json.loads(summary_path.read_text(encoding="utf-8"))
    elif "summary_row_json" in data.files:
        row = json.loads(str(data["summary_row_json"].item()))
    else:
        raise FileNotFoundError(f"No summary row found for {result_path}")
    return LayerArtifact(
        n_layers=int(row["n_layers"]),
        result_path=result_path,
        train_cost_history=np.asarray(data["train_cost_history"], dtype=float),
        l2_history=np.asarray(data["l2_history"], dtype=float),
        best_l2_history=np.asarray(data["best_l2_history"], dtype=float),
        f_target=np.asarray(data["f_target"], dtype=float),
        f_target_2d=np.asarray(data["f_target_2d"], dtype=float),
        f_init=np.asarray(data["f_init_statevector"], dtype=float),
        f_stage1=np.asarray(data["f_stage1_statevector"], dtype=float),
        f_star=np.asarray(data["f_star_statevector"], dtype=float),
        row=row,
    )


def regenerate_from_existing(output_dir: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, list[LayerArtifact]]:
    result_paths = sorted(output_dir.glob("L*/crca_positive_exposure_statevector_L*.npz"))
    if not result_paths:
        raise FileNotFoundError(f"No layer result NPZ files found under {output_dir}")

    artifacts = [load_artifact_from_result(path) for path in result_paths]
    summary_df = pd.DataFrame([artifact.row for artifact in artifacts]).sort_values(
        "n_layers"
    )

    resource_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        resource_path = artifact.result_path.parent / "resource_rows.json"
        if resource_path.exists():
            resource_rows.extend(json.loads(resource_path.read_text(encoding="utf-8")))
    resource_df = pd.DataFrame(resource_rows)
    if not resource_df.empty:
        resource_df = resource_df.sort_values(["n_layers", "circuit_kind"])
    return summary_df.reset_index(drop=True), resource_df.reset_index(drop=True), artifacts


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    layers = sorted({int(layer) for layer in args.layers})
    if not layers:
        raise ValueError("At least one layer must be provided.")

    output_dir = pathlib.Path(args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    f_target, f_target_2d, c_v, n_time, n_price_states = load_positive_exposure_target()

    if args.dry_run:
        rows = []
        for n_layers in layers:
            crca = build_crca(n_layers=n_layers, ansatz_type=args.ansatz_type)
            rows.append(
                summarize_circuit(
                    crca.qc_eval,
                    n_layers=n_layers,
                    n_params=crca.n_params,
                    circuit_kind="eval_logical",
                )
            )
        print(pd.DataFrame(rows).to_string(index=False))
        return

    run_noisy_eval = not bool(args.skip_noisy_eval)
    noise_context = None
    if args.plots_only:
        summary_df, resource_df, artifacts = regenerate_from_existing(output_dir)
        needs_noisy_eval = run_noisy_eval and any(
            "mse_shots_noise_mean" not in artifact.row
            or not (
                artifact.result_path.parent
                / f"noisy_eval_L{int(artifact.n_layers):02d}.npz"
            ).exists()
            for artifact in artifacts
        )
        if needs_noisy_eval:
            noise_context = prepare_backend_noise_context(args.ansatz_type)
        elif run_noisy_eval:
            for artifact in artifacts:
                load_cached_noisy_evaluation(artifact)
            summary_df = pd.DataFrame(
                [artifact.row for artifact in artifacts]
            ).sort_values("n_layers")
            summary_df = summary_df.reset_index(drop=True)
    else:
        if run_noisy_eval:
            noise_context = prepare_backend_noise_context(args.ansatz_type)
        run_config = {
            "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
            "output_dir": str(output_dir),
            "task": "crca_positive_exposure_layer_comparison",
            "training_mode": "statevector",
            "shots": None,
            "statevector_training": True,
            "ansatz_type": args.ansatz_type,
            "layers": layers,
            "m_time": M_TIME,
            "n_price": N_PRICE,
            "target_dim": int(f_target.size),
            "n_time_states": int(n_time),
            "n_price_states": int(n_price_states),
            "C_v": float(c_v),
            "loss_mode": args.loss_mode,
            "target_threshold": float(args.target_threshold),
            "relative_eps": float(args.relative_eps),
            "lambda_pos": float(args.lambda_pos),
            "lambda_zero": float(args.lambda_zero),
            "init_scale": float(args.init_scale),
            "theta_seed": int(args.theta_seed),
            "seed_stride": int(args.seed_stride),
            "stage1_maxiter": int(args.stage1_maxiter),
            "stage1_rhobeg": float(args.stage1_rhobeg),
            "stage1_tol": float(args.stage1_tol),
            "stage2_maxiter": int(args.stage2_maxiter),
            "stage2_maxfun": int(args.stage2_maxfun),
            "stage2_ftol": float(args.stage2_ftol),
            "stage2_gtol": float(args.stage2_gtol),
            "stage2_eps": float(args.stage2_eps),
            "stage2_maxls": int(args.stage2_maxls),
            "stage2_maxcor": int(args.stage2_maxcor),
            "flattening_order": "time_major",
            "reported_metric": "MSE between target exposure and CRCA statevector function values",
            "noisy_evaluation_enabled": bool(run_noisy_eval),
            "noisy_eval_runs": int(args.noisy_eval_runs),
            "noisy_eval_shots": int(args.noisy_eval_shots),
            "noisy_eval_seed_base": int(args.noisy_eval_seed_base),
            "backend_name": BACKEND_NAME,
            "runtime_channel": RUNTIME_CHANNEL,
            "use_fractional_gates": bool(USE_FRACTIONAL_GATES),
            "noise_snapshot_iso_utc": NOISE_SNAPSHOT_ISO_UTC,
            "seed_transpiler": int(SEED_TRANSPILER),
            "optimization_level": int(OPTIMIZATION_LEVEL),
            "layout_method": LAYOUT_METHOD,
            "routing_method": ROUTING_METHOD,
            "simulator_seed": int(SIMULATOR_SEED),
        }
        if noise_context is not None:
            layout_meta = _layout_meta_brief(noise_context["layout_meta"])
            run_config.update(
                {
                    "chosen_layout": list(noise_context["chosen_layout"]),
                    "layout_score": float(noise_context["layout_score"]),
                    "effective_topology": layout_meta["selected_topology"],
                    "layout_fallback_used": bool(layout_meta["fallback_used"]),
                    "noise_model_build": str(noise_context["noise_model_build"]),
                    "noise_model_fallback_used": bool(
                        noise_context["used_noise_fallback"]
                    ),
                }
            )
        _write_json(output_dir / "run_config.json", run_config)

        rows: list[dict[str, Any]] = []
        resource_rows: list[dict[str, Any]] = []
        artifacts: list[LayerArtifact] = []

        for n_layers in layers:
            print(f"Training CRCA positive exposure L={n_layers} with statevector")
            row, ansatz_summary, eval_summary, artifact = train_one_layer(
                n_layers=n_layers,
                ansatz_type=args.ansatz_type,
                f_target=f_target,
                f_target_2d=f_target_2d,
                c_v=c_v,
                n_time=n_time,
                n_price_states=n_price_states,
                output_dir=output_dir,
                loss_mode=args.loss_mode,
                init_scale=float(args.init_scale),
                theta_seed=int(args.theta_seed),
                seed_stride=int(args.seed_stride),
                target_threshold=float(args.target_threshold),
                relative_eps=float(args.relative_eps),
                lambda_pos=float(args.lambda_pos),
                lambda_zero=float(args.lambda_zero),
                stage1_maxiter=int(args.stage1_maxiter),
                stage1_rhobeg=float(args.stage1_rhobeg),
                stage1_tol=float(args.stage1_tol),
                stage2_maxiter=int(args.stage2_maxiter),
                stage2_maxfun=int(args.stage2_maxfun),
                stage2_ftol=float(args.stage2_ftol),
                stage2_gtol=float(args.stage2_gtol),
                stage2_eps=float(args.stage2_eps),
                stage2_maxls=int(args.stage2_maxls),
                stage2_maxcor=int(args.stage2_maxcor),
                force=bool(args.force),
            )
            rows.append(row)
            resource_rows.extend([ansatz_summary, eval_summary])
            artifacts.append(artifact)

        summary_df = pd.DataFrame(rows).sort_values("n_layers").reset_index(drop=True)
        resource_df = (
            pd.DataFrame(resource_rows)
            .sort_values(["n_layers", "circuit_kind"])
            .reset_index(drop=True)
        )

        np.savez(
            output_dir / "positive_exposure_layer_sweep_aggregate.npz",
            summary_json=np.array(json.dumps(_json_ready(rows)), dtype=object),
            resource_json=np.array(json.dumps(_json_ready(resource_rows)), dtype=object),
            layers=np.asarray(layers, dtype=int),
            f_target=f_target,
            f_target_2d=f_target_2d,
            C_v=np.float64(c_v),
        )

    if noise_context is not None:
        summary_df = complete_noisy_evaluation(
            artifacts,
            ansatz_type=args.ansatz_type,
            noise_context=noise_context,
            shots=int(args.noisy_eval_shots),
            eval_runs=int(args.noisy_eval_runs),
            seed_base=int(args.noisy_eval_seed_base),
            target_threshold=float(args.target_threshold),
            relative_eps=float(args.relative_eps),
        )

    np.savez(
        output_dir / "positive_exposure_layer_sweep_aggregate.npz",
        summary_json=np.array(
            json.dumps(_json_ready(summary_df.to_dict("records"))),
            dtype=object,
        ),
        resource_json=np.array(
            json.dumps(
                _json_ready(
                    resource_df.to_dict("records") if not resource_df.empty else []
                )
            ),
            dtype=object,
        ),
        layers=summary_df["n_layers"].to_numpy(dtype=int),
        f_target=f_target,
        f_target_2d=f_target_2d,
        C_v=np.float64(c_v),
    )

    save_table_bundle(
        summary_df,
        output_dir=tables_dir,
        stem="positive_exposure_layer_sweep_summary",
        title="CRCA positive exposure statevector layer sweep",
        image_columns=[
            col
            for col in [
            "n_layers",
            "ansatz_type",
            "loss_mode",
            "n_params",
            "backend_name",
            "noise_snapshot_iso_utc",
            "chosen_layout",
            "mse_init",
            "mse_stage1",
            "mse_final",
            "mse_shots_noise_mean",
            "rmse_final",
            "positive_relative_rmse_final",
            "zero_leakage_mean_abs_final",
            "logical_eval_depth",
            "logical_eval_two_qubit_depth",
            "noisy_eval_transpiled_depth",
            "noisy_eval_transpiled_two_qubit_depth",
            "noisy_eval_transpiled_two_qubit_ops",
            "elapsed_total_s",
            ]
            if col in summary_df.columns
        ],
    )
    if not resource_df.empty:
        save_table_bundle(
            resource_df,
            output_dir=tables_dir,
            stem="positive_exposure_layer_sweep_resource_table",
            title="CRCA positive exposure logical resources by layer",
            image_columns=[
                "n_layers",
                "circuit_kind",
                "n_params",
                "width",
                "depth",
                "two_qubit_ops",
                "two_qubit_depth",
                "rx",
                "ry",
                "rz",
                "cx",
                "swap",
                "measure",
            ],
        )

    generate_plots(summary_df, artifacts, output_dir=figures_dir)
    print(f"Saved CRCA positive-exposure layer sweep results to: {output_dir}")


if __name__ == "__main__":
    main()
