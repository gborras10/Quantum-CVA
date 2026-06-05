from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.stats import kruskal, mannwhitneyu


def _bootstrap_paths() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    layers_dir = current.parents[1] / "layers_comparison"
    for path in (src_path, layers_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _bootstrap_paths()

from qcbm_comparative_ideal import (  # noqa: E402
    BACKEND_NAME,
    BENCHMARK_RELATIVE_PATH,
    EPS_COST,
    INIT_SCALE,
    LAYOUT_METHOD,
    LOCAL_2Q_QUANTILE,
    LOGICAL_TOPOLOGY,
    NOISE_SNAPSHOT_ISO_UTC,
    OPTIMIZATION_LEVEL,
    READOUT_QUANTILE,
    ROUTING_METHOD,
    RUNTIME_CHANNEL,
    SEED_TRANSPILER,
    STAGE1_MAXITER,
    STAGE1_RHOBEG,
    STAGE1_TOL,
    STAGE2_MAXFUN,
    STAGE2_MAXITER,
    THETA_SEED,
    USE_FRACTIONAL_GATES,
    _as_1d_float,
    _format_float,
    _json_ready,
    _normalize_distribution,
    _parse_snapshot_datetime,
    _write_json,
    configure_matplotlib,
    kl_divergence,
    kl_mass_contribution,
    kl_time_decomposition,
    run_stage1,
    run_stage2,
    save_figure,
    save_table_bundle,
    select_qcbm_heavyhex6_layout_from_snapshot,
    snapshot_edge_table,
    snapshot_qubit_table,
    summarize_transpiled_circuit,
    target_entropy,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (  # noqa: E402
    MLQcbmCircuit,
)


RESULTS_RELATIVE_DIR = (
    "cva_pricing_pipeline/multi_asset/6q_instance/training_multi_asset/"
    "state_preparation/ansatz_comparison/results/entangler_comparison"
)

DEFAULT_LAYERS = (4, 6, 8, 10)
DEFAULT_ENTANGLERS = ("rzz", "cz", "rxx")
VALID_EFFECTIVE_ENTANGLERS = {"rzz", "rxx", "cz"}


def normalize_entangler(requested: str) -> tuple[str, str, str]:
    label = str(requested).strip().lower()
    if label in VALID_EFFECTIVE_ENTANGLERS:
        return label, label, ""
    if label == "rz":
        return (
            "rz_requested_cz_effective",
            "cz",
            (
                "RZ is a single-qubit rotation already present in the QCBM "
                "rotation layers; it is not a two-qubit entangler. This run "
                "uses CZ as the fixed two-qubit entangler baseline."
            ),
        )
    raise ValueError(
        f"Unsupported entangler '{requested}'. Use rzz, rxx or cz. "
        "If you meant a non-parametric entangler baseline, use cz."
    )


def build_qcbm(
    *,
    n_qubits: int,
    n_layers: int,
    entangler: str,
    backend,
    chosen_layout: list[int],
) -> MLQcbmCircuit:
    return MLQcbmCircuit(
        n_qubits=n_qubits,
        n_layers=int(n_layers),
        name=f"G_p_{entangler}_L{int(n_layers):02d}",
        entangler=entangler,
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


def _load_target_distribution() -> tuple[np.ndarray, int]:
    data = np.load(REPO_ROOT / BENCHMARK_RELATIVE_PATH, allow_pickle=True)
    ptg = _normalize_distribution(_as_1d_float(data["p_target"]), eps=None)
    dim = int(ptg.size)
    n_qubits = int(round(math.log2(dim)))
    if 2**n_qubits != dim:
        raise ValueError("p_target length must be a power of two.")

    n_time = int(np.asarray(data["t"]).size) if "t" in data else 1
    if dim % n_time != 0:
        raise ValueError("p_target length is not divisible by the time grid size.")
    return ptg, n_time


def _prepare_backend_context(snapshot_dt_utc: datetime) -> dict[str, Any]:
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
    chosen_layout, layout_score, layout_meta = (
        select_qcbm_heavyhex6_layout_from_snapshot(
            real_backend,
            backend_props,
            readout_quantile=READOUT_QUANTILE,
            local_2q_quantile=LOCAL_2Q_QUANTILE,
        )
    )
    return {
        "real_backend": real_backend,
        "backend_props": backend_props,
        "chosen_layout": chosen_layout,
        "layout_score": layout_score,
        "layout_meta": layout_meta,
    }


def _safe_result_attr(result: Any, name: str, default: Any) -> Any:
    value = getattr(result, name, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _raw_ops(summary: dict[str, Any]) -> dict[str, int]:
    try:
        return {str(k): int(v) for k, v in json.loads(summary["raw_ops_json"]).items()}
    except Exception:
        return {}


def train_one_configuration(
    *,
    qcbm: MLQcbmCircuit,
    n_layers: int,
    entangler_label: str,
    effective_entangler: str,
    alias_note: str,
    theta_seed: int,
    ptg: np.ndarray,
    n_time: int,
    target_h: float,
    ansatz_summary: dict[str, Any],
    measured_summary: dict[str, Any],
    output_dir: pathlib.Path,
    stage1_maxiter: int,
    stage2_maxiter: int,
    stage2_maxfun: int,
    force: bool,
) -> dict[str, Any]:
    run_dir = (
        output_dir
        / str(entangler_label)
        / f"L{int(n_layers):02d}"
        / f"seed_{int(theta_seed):06d}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        run_dir
        / (
            f"qcbm_{entangler_label}_L{int(n_layers):02d}_"
            f"seed{int(theta_seed):06d}.npz"
        )
    )
    summary_path = run_dir / "summary_row.json"

    if result_path.exists() and summary_path.exists() and not force:
        print(
            f"[resume] entangler={entangler_label} L={n_layers:02d} "
            f"seed={theta_seed} loaded"
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(int(theta_seed))
    theta_init = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)
    p_init = qcbm.probabilities(theta_init)
    kl_init = kl_divergence(ptg, p_init, eps=EPS_COST)
    cost_statevector = qcbm.cost_fn(ptg, eps=EPS_COST)

    print(
        f"\n=== Training entangler={entangler_label} "
        f"(effective={effective_entangler}) | L={n_layers:02d} "
        f"| theta_seed={theta_seed} ==="
    )
    t0 = time.perf_counter()
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
        f"entangler={entangler_label} L={n_layers:02d} seed={theta_seed} "
        f"stage1 | KL={stage1['kl_final']:.8e} | "
        f"t={stage1['elapsed_time']:.2f}s"
    )

    stage2 = run_stage2(
        stage1["theta_star"],
        maxiter=stage2_maxiter,
        maxfun=stage2_maxfun,
        cost_fn=cost_statevector,
        qcbm=qcbm,
        target_h=target_h,
    )
    elapsed_total = time.perf_counter() - t0
    print(
        f"entangler={entangler_label} L={n_layers:02d} seed={theta_seed} "
        f"stage2 | KL={stage2['kl_final']:.8e} | "
        f"t={stage2['elapsed_time']:.2f}s"
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
    ansatz_ops = _raw_ops(ansatz_summary)

    two_qubit_depth = int(ansatz_summary["two_qubit_depth"])
    depth = int(ansatz_summary["depth"])
    log_accuracy = -math.log10(max(final_kl, 1e-15))

    row = {
        "entangler_label": entangler_label,
        "effective_entangler": effective_entangler,
        "entangler_alias_note": alias_note,
        "n_layers": int(n_layers),
        "theta_seed": int(theta_seed),
        "n_qubits": int(qcbm.n_qubits),
        "n_params": int(qcbm.n_params),
        "n_entangling_pairs": int(len(qcbm.pairs)),
        "topology": LOGICAL_TOPOLOGY,
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
        "metric_l1": float(metrics["l1"]),
        "metric_tv": float(metrics["tv"]),
        "metric_linf": float(metrics["linf"]),
        "ce_final": float(stage2["ce_final"]),
        "target_entropy": float(target_h),
        "elapsed_total_s": float(elapsed_total),
        "stage1_elapsed_s": float(stage1["elapsed_time"]),
        "stage2_elapsed_s": float(stage2["elapsed_time"]),
        "stage1_nit": int(_safe_result_attr(stage1["result"], "nit", -1)),
        "stage2_nit": int(_safe_result_attr(stage2["result"], "nit", -1)),
        "n_history_points": int(kl_history.size),
        "stage2_success": bool(_safe_result_attr(stage2["result"], "success", False)),
        "stage2_message": str(_safe_result_attr(stage2["result"], "message", "")),
        "ansatz_depth": depth,
        "ansatz_size": int(ansatz_summary["size"]),
        "ansatz_width": int(ansatz_summary["width"]),
        "ansatz_one_qubit_ops": int(ansatz_summary["one_qubit_ops"]),
        "ansatz_two_qubit_ops": int(ansatz_summary["two_qubit_ops"]),
        "ansatz_two_qubit_depth": two_qubit_depth,
        "ansatz_swap": int(ansatz_summary["swap"]),
        "ansatz_rzz": int(ansatz_ops.get("rzz", 0)),
        "ansatz_rxx": int(ansatz_ops.get("rxx", 0)),
        "ansatz_cz": int(ansatz_ops.get("cz", 0)),
        "ansatz_ecr": int(ansatz_ops.get("ecr", 0)),
        "ansatz_cx": int(ansatz_ops.get("cx", 0)),
        "measured_depth": int(measured_summary["depth"]),
        "measured_two_qubit_ops": int(measured_summary["two_qubit_ops"]),
        "kl_times_depth": float(final_kl * max(depth, 1)),
        "kl_times_two_qubit_depth": float(final_kl * max(two_qubit_depth, 1)),
        "neg_log10_kl": float(log_accuracy),
        "neg_log10_kl_per_depth": float(log_accuracy / max(depth, 1)),
        "neg_log10_kl_per_two_qubit_depth": float(
            log_accuracy / max(two_qubit_depth, 1)
        ),
    }

    np.savez(
        result_path,
        entangler_label=np.array(entangler_label),
        effective_entangler=np.array(effective_entangler),
        theta_seed=np.int64(theta_seed),
        theta_init=theta_init,
        theta_stage1=np.asarray(stage1["theta_star"], dtype=float),
        theta_star=theta_star,
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
    )
    _write_json(summary_path, row)
    return row


def _metric_stats(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "sem": math.nan,
            "median": math.nan,
            "q25": math.nan,
            "q75": math.nan,
            "iqr": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    q25, q75 = np.quantile(x, [0.25, 0.75])
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return {
        "mean": float(np.mean(x)),
        "std": std,
        "sem": float(std / math.sqrt(x.size)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _mark_pareto(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    efficient: list[bool] = []
    for idx, row in out.iterrows():
        kl = float(row["kl_final_median"])
        depth = float(row["ansatz_two_qubit_depth"])
        dominated = False
        for other_idx, other in out.iterrows():
            if idx == other_idx:
                continue
            other_kl = float(other["kl_final_median"])
            other_depth = float(other["ansatz_two_qubit_depth"])
            if (
                other_kl <= kl
                and other_depth <= depth
                and (other_kl < kl or other_depth < depth)
            ):
                dominated = True
                break
        efficient.append(not dominated)
    out["pareto_efficient_kl_vs_two_qubit_depth"] = efficient
    return out


def summarize_entanglers(
    run_df: pd.DataFrame,
    *,
    convergence_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["entangler_label", "effective_entangler", "n_layers"]
    for keys, group in run_df.groupby(group_cols, sort=True):
        entangler_label, effective_entangler, n_layers = keys
        kl_stats = _metric_stats(group["kl_final"])
        elapsed_stats = _metric_stats(group["elapsed_total_s"])
        converged = group["kl_final"].astype(float) <= float(convergence_threshold)
        base = {
            "entangler_label": entangler_label,
            "effective_entangler": effective_entangler,
            "n_layers": int(n_layers),
            "n_runs": int(len(group)),
            "n_params": int(group["n_params"].iloc[0]),
            "ansatz_depth": int(group["ansatz_depth"].iloc[0]),
            "ansatz_two_qubit_ops": int(group["ansatz_two_qubit_ops"].iloc[0]),
            "ansatz_two_qubit_depth": int(group["ansatz_two_qubit_depth"].iloc[0]),
            "ansatz_swap": int(group["ansatz_swap"].iloc[0]),
            "ansatz_rzz": int(group["ansatz_rzz"].iloc[0]),
            "ansatz_rxx": int(group["ansatz_rxx"].iloc[0]),
            "ansatz_cz": int(group["ansatz_cz"].iloc[0]),
            "convergence_threshold": float(convergence_threshold),
            "convergence_rate": float(np.mean(converged)),
            "stage2_success_rate": float(np.mean(group["stage2_success"].astype(bool))),
            "elapsed_total_mean_s": elapsed_stats["mean"],
            "elapsed_total_median_s": elapsed_stats["median"],
            "neg_log10_kl_per_two_qubit_depth_median": float(
                np.median(group["neg_log10_kl_per_two_qubit_depth"])
            ),
            "kl_times_two_qubit_depth_median": float(
                np.median(group["kl_times_two_qubit_depth"])
            ),
        }
        for key, value in kl_stats.items():
            base[f"kl_final_{key}"] = value
        rows.append(base)
    summary_df = pd.DataFrame(rows).sort_values(
        ["n_layers", "entangler_label"]
    ).reset_index(drop=True)
    return _mark_pareto(summary_df)


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = float(p_values[idx])
        adj = min(1.0, (m - rank) * raw)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


def nonparametric_entangler_tests(
    run_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for n_layers, layer_df in run_df.groupby("n_layers", sort=True):
        groups = [
            (
                str(label),
                group["kl_final"].astype(float).to_numpy(dtype=float),
            )
            for label, group in layer_df.groupby("entangler_label", sort=True)
        ]
        if len(groups) >= 2 and all(values.size > 0 for _, values in groups):
            try:
                stat, p_value = kruskal(*(values for _, values in groups))
                error = ""
            except ValueError as exc:
                stat, p_value = math.nan, math.nan
                error = str(exc)
            global_rows.append(
                {
                    "n_layers": int(n_layers),
                    "test": "kruskal",
                    "metric": "kl_final",
                    "n_groups": int(len(groups)),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "error": error,
                }
            )

        layer_pair_rows: list[dict[str, Any]] = []
        raw_p_values: list[float] = []
        for i, (label_a, values_a) in enumerate(groups):
            for label_b, values_b in groups[i + 1 :]:
                try:
                    stat, p_value = mannwhitneyu(
                        values_a,
                        values_b,
                        alternative="two-sided",
                    )
                    error = ""
                except ValueError as exc:
                    stat, p_value = math.nan, math.nan
                    error = str(exc)
                raw_p_values.append(float(p_value) if np.isfinite(p_value) else 1.0)
                layer_pair_rows.append(
                    {
                        "n_layers": int(n_layers),
                        "entangler_a": label_a,
                        "entangler_b": label_b,
                        "test": "mannwhitneyu",
                        "metric": "kl_final",
                        "n_a": int(values_a.size),
                        "n_b": int(values_b.size),
                        "median_a": float(np.median(values_a)),
                        "median_b": float(np.median(values_b)),
                        "delta_median_a_minus_b": float(
                            np.median(values_a) - np.median(values_b)
                        ),
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "error": error,
                    }
                )
        adjusted = _holm_adjust(raw_p_values)
        for row, p_adj in zip(layer_pair_rows, adjusted, strict=True):
            row["p_value_holm_within_layer"] = float(p_adj)
        pair_rows.extend(layer_pair_rows)

    return pd.DataFrame(global_rows), pd.DataFrame(pair_rows)


def plot_kl_vs_layer(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, group in summary_df.groupby("entangler_label", sort=True):
        group = group.sort_values("n_layers")
        ax.semilogy(
            group["n_layers"],
            np.maximum(group["kl_final_median"], 1e-15),
            marker="o",
            label=str(label),
        )
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel("Median final KL")
    ax.set_title("Ansatz entangler comparison under equal layer counts")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "ansatz_median_kl_vs_layers")


def plot_kl_vs_depth(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for label, group in summary_df.groupby("entangler_label", sort=True):
        group = group.sort_values("ansatz_two_qubit_depth")
        ax.semilogy(
            group["ansatz_two_qubit_depth"],
            np.maximum(group["kl_final_median"], 1e-15),
            marker="o",
            label=str(label),
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"L={int(row['n_layers'])}",
                xy=(
                    row["ansatz_two_qubit_depth"],
                    max(row["kl_final_median"], 1e-15),
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    pareto = summary_df[summary_df["pareto_efficient_kl_vs_two_qubit_depth"]]
    if not pareto.empty:
        ax.scatter(
            pareto["ansatz_two_qubit_depth"],
            np.maximum(pareto["kl_final_median"], 1e-15),
            s=95,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.4,
            label="Pareto efficient",
        )
    ax.set_xlabel("Transpiled two-qubit depth")
    ax.set_ylabel("Median final KL")
    ax.set_title("Accuracy versus hardware-relevant depth")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "ansatz_median_kl_vs_two_qubit_depth")


def plot_resource_adjusted_score(
    summary_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [
        f"{row.entangler_label}\nL={int(row.n_layers)}"
        for row in summary_df.itertuples(index=False)
    ]
    x = np.arange(len(summary_df))
    ax.bar(
        x,
        summary_df["neg_log10_kl_per_two_qubit_depth_median"].to_numpy(dtype=float),
        color="#2563eb",
        edgecolor="#1e3a8a",
        alpha=0.88,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel(r"median $-\log_{10}(KL)$ / two-qubit depth")
    ax.set_title("Depth-adjusted training quality")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "ansatz_depth_adjusted_score")


def generate_plots(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    plot_dir = output_dir / "figures"
    configure_matplotlib()
    plot_kl_vs_layer(summary_df, plot_dir)
    plot_kl_vs_depth(summary_df, plot_dir)
    plot_resource_adjusted_score(summary_df, plot_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare QCBM heavy-hex6 ansatz entanglers for the 6q multi-asset "
            "CVA target. Defaults compare RZZ, CZ and RXX on layers 4, 6, 8 "
            "and 10, reporting both training quality and transpiled depth."
        )
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="QCBM layer values to compare. Default: 4 6 8 10.",
    )
    parser.add_argument(
        "--entanglers",
        type=str,
        nargs="+",
        default=list(DEFAULT_ENTANGLERS),
        help=(
            "Entanglers to compare. Valid effective entanglers: rzz, cz, rxx. "
            "If 'rz' is provided, the script records it as a request but uses "
            "CZ because RZ is single-qubit and does not entangle."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Independent theta seeds per entangler/layer pair.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=THETA_SEED,
        help="First theta seed. Default matches the layer-comparison seed.",
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="Stride between theta seeds.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=1e-3,
        help="KL threshold used to report convergence rate.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain cached runs.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figures.")
    parser.add_argument(
        "--stage1-maxiter",
        type=int,
        default=STAGE1_MAXITER,
        help="COBYLA maxiter per run.",
    )
    parser.add_argument(
        "--stage2-maxiter",
        type=int,
        default=STAGE2_MAXITER,
        help="L-BFGS-B maxiter per run.",
    )
    parser.add_argument(
        "--stage2-maxfun",
        type=int,
        default=STAGE2_MAXFUN,
        help="L-BFGS-B maxfun per run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.n_seeds) <= 0:
        raise ValueError("--n-seeds must be positive.")
    if int(args.seed_stride) <= 0:
        raise ValueError("--seed-stride must be positive.")
    if float(args.convergence_threshold) <= 0.0:
        raise ValueError("--convergence-threshold must be positive.")

    layers = sorted({int(layer) for layer in args.layers})
    if any(layer <= 0 for layer in layers):
        raise ValueError("Layer values must be positive.")

    entangler_specs = [normalize_entangler(item) for item in args.entanglers]
    seen_labels: set[str] = set()
    deduped_specs: list[tuple[str, str, str]] = []
    for spec in entangler_specs:
        if spec[0] not in seen_labels:
            seen_labels.add(spec[0])
            deduped_specs.append(spec)
    entangler_specs = deduped_specs
    for label, effective, note in entangler_specs:
        if note:
            print(f"[entangler note] requested={label} effective={effective}: {note}")

    theta_seeds = [
        int(args.seed_base) + idx * int(args.seed_stride)
        for idx in range(int(args.n_seeds))
    ]

    output_dir = REPO_ROOT / RESULTS_RELATIVE_DIR
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    ptg, n_time = _load_target_distribution()
    n_qubits = int(round(math.log2(ptg.size)))
    target_h = target_entropy(ptg, eps=EPS_COST)

    snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
    print(f"Backend: {BACKEND_NAME}")
    print(f"Noise snapshot UTC: {snapshot_dt_utc.isoformat()}")
    print(f"Layers: {layers}")
    print(f"Entanglers: {[(label, effective) for label, effective, _ in entangler_specs]}")
    print(f"Theta seeds: {theta_seeds}")
    print(f"Output directory: {output_dir}")

    backend_ctx = _prepare_backend_context(snapshot_dt_utc)
    real_backend = backend_ctx["real_backend"]
    backend_props = backend_ctx["backend_props"]
    chosen_layout = backend_ctx["chosen_layout"]
    layout_score = backend_ctx["layout_score"]
    layout_meta = backend_ctx["layout_meta"]

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
        "requested_entanglers": list(args.entanglers),
        "entangler_specs": [
            {"label": label, "effective": effective, "note": note}
            for label, effective, note in entangler_specs
        ],
        "layers": layers,
        "theta_seeds": theta_seeds,
        "n_seeds": int(args.n_seeds),
        "seed_base": int(args.seed_base),
        "seed_stride": int(args.seed_stride),
        "init_scale": float(INIT_SCALE),
        "eps_cost": EPS_COST,
        "stage1_maxiter": int(args.stage1_maxiter),
        "stage1_rhobeg": STAGE1_RHOBEG,
        "stage1_tol": STAGE1_TOL,
        "stage2_maxiter": int(args.stage2_maxiter),
        "stage2_maxfun": int(args.stage2_maxfun),
        "convergence_threshold": float(args.convergence_threshold),
        "seed_transpiler": SEED_TRANSPILER,
        "optimization_level": OPTIMIZATION_LEVEL,
        "layout_method": LAYOUT_METHOD,
        "routing_method": ROUTING_METHOD,
        "chosen_layout": chosen_layout,
        "layout_score": layout_score,
        "layout_meta": layout_meta,
        "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
        "output_dir": str(output_dir),
        "training_regime": "ideal_statevector_ansatz_entangler_comparison",
        "cost_function": "MLQcbmCircuit.cost_fn(ptg, eps=EPS_COST)",
        "reported_metric": "KL(target || qcbm)",
        "flattening_order": "time_major",
    }
    _write_json(output_dir / "run_config.json", run_config)

    run_rows: list[dict[str, Any]] = []
    transpile_rows: list[dict[str, Any]] = []

    for entangler_label, effective_entangler, alias_note in entangler_specs:
        for n_layers in layers:
            layer_dir = output_dir / str(entangler_label) / f"L{int(n_layers):02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            qcbm = build_qcbm(
                n_qubits=n_qubits,
                n_layers=int(n_layers),
                entangler=effective_entangler,
                backend=real_backend,
                chosen_layout=chosen_layout,
            )
            ansatz_summary = summarize_transpiled_circuit(
                qcbm._tqc,
                n_layers=int(n_layers),
                n_params=qcbm.n_params,
                circuit_kind=f"ansatz_{entangler_label}",
            )
            measured_summary = summarize_transpiled_circuit(
                qcbm._tqc_meas,
                n_layers=int(n_layers),
                n_params=qcbm.n_params,
                circuit_kind=f"measured_{entangler_label}",
            )
            ansatz_summary["entangler_label"] = entangler_label
            ansatz_summary["effective_entangler"] = effective_entangler
            measured_summary["entangler_label"] = entangler_label
            measured_summary["effective_entangler"] = effective_entangler
            transpile_rows.extend([ansatz_summary, measured_summary])

            with open(
                layer_dir / f"qcbm_{entangler_label}_template_L{int(n_layers):02d}.qpy",
                "wb",
            ) as f:
                qpy.dump(qcbm._tqc, f)
            with open(
                layer_dir
                / f"qcbm_{entangler_label}_template_measured_L{int(n_layers):02d}.qpy",
                "wb",
            ) as f:
                qpy.dump(qcbm._tqc_meas, f)

            for theta_seed in theta_seeds:
                row = train_one_configuration(
                    qcbm=qcbm,
                    n_layers=int(n_layers),
                    entangler_label=entangler_label,
                    effective_entangler=effective_entangler,
                    alias_note=alias_note,
                    theta_seed=int(theta_seed),
                    ptg=ptg,
                    n_time=n_time,
                    target_h=target_h,
                    ansatz_summary=ansatz_summary,
                    measured_summary=measured_summary,
                    output_dir=output_dir,
                    stage1_maxiter=int(args.stage1_maxiter),
                    stage2_maxiter=int(args.stage2_maxiter),
                    stage2_maxfun=int(args.stage2_maxfun),
                    force=bool(args.force),
                )
                run_rows.append(row)

    run_df = pd.DataFrame(run_rows).sort_values(
        ["n_layers", "entangler_label", "theta_seed"]
    ).reset_index(drop=True)
    transpile_df = pd.DataFrame(transpile_rows).sort_values(
        ["n_layers", "entangler_label", "circuit_kind"]
    ).reset_index(drop=True)
    summary_df = summarize_entanglers(
        run_df,
        convergence_threshold=float(args.convergence_threshold),
    )
    global_tests_df, pairwise_tests_df = nonparametric_entangler_tests(run_df)

    save_table_bundle(
        run_df,
        output_dir=tables_dir,
        stem="ansatz_entangler_per_run_results",
        title="Per-run QCBM ansatz entangler comparison",
        image_columns=[
            "entangler_label",
            "effective_entangler",
            "n_layers",
            "theta_seed",
            "n_params",
            "ansatz_two_qubit_depth",
            "kl_init",
            "kl_stage1",
            "kl_final",
            "neg_log10_kl_per_two_qubit_depth",
            "elapsed_total_s",
        ],
    )
    save_table_bundle(
        summary_df,
        output_dir=tables_dir,
        stem="ansatz_entangler_summary",
        title="QCBM ansatz entangler summary with depth-adjusted metrics",
        image_columns=[
            "entangler_label",
            "effective_entangler",
            "n_layers",
            "n_runs",
            "n_params",
            "ansatz_two_qubit_depth",
            "kl_final_median",
            "kl_final_iqr",
            "kl_final_min",
            "kl_final_max",
            "neg_log10_kl_per_two_qubit_depth_median",
            "pareto_efficient_kl_vs_two_qubit_depth",
        ],
    )
    save_table_bundle(
        transpile_df,
        output_dir=tables_dir,
        stem="ansatz_entangler_transpile_table",
        title="Backend-transpiled resources by QCBM ansatz entangler",
        image_columns=[
            "entangler_label",
            "effective_entangler",
            "n_layers",
            "circuit_kind",
            "n_params",
            "depth",
            "two_qubit_ops",
            "two_qubit_depth",
            "rzz",
            "cz",
            "ecr",
            "cx",
            "swap",
        ],
    )
    if not global_tests_df.empty:
        save_table_bundle(
            global_tests_df,
            output_dir=tables_dir,
            stem="ansatz_entangler_global_nonparametric_tests",
            title="Global non-parametric entangler tests within each layer",
        )
    if not pairwise_tests_df.empty:
        save_table_bundle(
            pairwise_tests_df,
            output_dir=tables_dir,
            stem="ansatz_entangler_pairwise_nonparametric_tests",
            title="Pairwise entangler tests within each layer",
            image_columns=[
                "n_layers",
                "entangler_a",
                "entangler_b",
                "n_a",
                "n_b",
                "median_a",
                "median_b",
                "delta_median_a_minus_b",
                "p_value",
                "p_value_holm_within_layer",
            ],
        )

    np.savez(
        output_dir / "ansatz_entangler_aggregate.npz",
        run_results_json=np.array(
            json.dumps(_json_ready(run_df.to_dict(orient="records")))
        ),
        summary_json=np.array(
            json.dumps(_json_ready(summary_df.to_dict(orient="records")))
        ),
        pairwise_tests_json=np.array(
            json.dumps(_json_ready(pairwise_tests_df.to_dict(orient="records")))
        ),
        layers=np.asarray(layers, dtype=int),
        theta_seeds=np.asarray(theta_seeds, dtype=int),
        kl_final=run_df["kl_final"].to_numpy(dtype=float),
        p_target=ptg,
    )

    if not args.skip_plots:
        generate_plots(summary_df, output_dir)

    print("\n=== QCBM ansatz entangler comparison complete ===")
    print(f"Results: {output_dir}")
    print("\nDepth-aware ansatz summary:")
    final_cols = [
        "entangler_label",
        "effective_entangler",
        "n_layers",
        "n_params",
        "ansatz_two_qubit_depth",
        "kl_final_median",
        "kl_final_iqr",
        "neg_log10_kl_per_two_qubit_depth_median",
        "pareto_efficient_kl_vs_two_qubit_depth",
    ]
    print(
        summary_df[final_cols].to_string(
            index=False,
            float_format=lambda x: _format_float(x, 4),
        )
    )


if __name__ == "__main__":
    main()
