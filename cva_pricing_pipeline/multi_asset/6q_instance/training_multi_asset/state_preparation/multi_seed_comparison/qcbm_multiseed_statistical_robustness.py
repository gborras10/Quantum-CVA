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
from qiskit_aer.noise import NoiseModel
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
    DIRICHLET_ALPHA,
    ENTANGLER,
    EPS_COST,
    INIT_SCALE,
    LAYOUT_METHOD,
    LOCAL_2Q_QUANTILE,
    LOGICAL_TOPOLOGY,
    NOISE_SNAPSHOT_ISO_UTC,
    NOISY_EVAL_RUNS,
    NOISY_EVAL_SEED_BASE,
    NOISY_EVAL_SHOTS,
    OPTIMIZATION_LEVEL,
    READOUT_QUANTILE,
    ROUTING_METHOD,
    RUNTIME_CHANNEL,
    SEED_TRANSPILER,
    SIMULATOR_SEED,
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
    evaluate_theta_under_shots_noise,
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
    "state_preparation/multi_seed_comparison/results/ideal_multiseed"
)

DEFAULT_LAYERS = (4, 6, 8, 10)
DEFAULT_N_SEEDS = 20
DEFAULT_CONVERGENCE_THRESHOLD = 1e-3


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
        name=f"G_p_multiseed_L{int(n_layers):02d}",
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


def _prepare_backend_context(snapshot_dt_utc: datetime):
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

    return {
        "real_backend": real_backend,
        "backend_props": backend_props,
        "noise_model": noise_model,
        "noisy_backend": noisy_backend,
        "chosen_layout": chosen_layout,
        "layout_score": layout_score,
        "layout_meta": layout_meta,
        "used_noise_fallback": used_noise_fallback,
    }


def _safe_result_attr(result: Any, name: str, default: Any) -> Any:
    value = getattr(result, name, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_one_seed(
    *,
    qcbm: MLQcbmCircuit,
    n_layers: int,
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
    run_noisy_eval: bool,
    real_backend,
    noisy_backend,
    noise_model,
    chosen_layout: list[int],
    noisy_eval_shots: int,
    noisy_eval_runs: int,
    noisy_eval_seed_base: int,
    noisy_eval_alpha: float,
) -> dict[str, Any]:
    seed_dir = output_dir / f"L{int(n_layers):02d}" / f"seed_{int(theta_seed):06d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        seed_dir
        / f"qcbm_multiseed_L{int(n_layers):02d}_seed{int(theta_seed):06d}.npz"
    )
    summary_path = seed_dir / "summary_row.json"

    if result_path.exists() and summary_path.exists() and not force:
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if run_noisy_eval and "kl_shots_noise_mean" not in row:
            data = np.load(result_path, allow_pickle=True)
            noisy_row, _, _ = evaluate_theta_under_shots_noise(
                n_layers=int(n_layers),
                theta_star=np.asarray(data["theta_star"], dtype=float),
                ptg=ptg,
                n_time=n_time,
                transpile_backend=real_backend,
                noisy_backend=noisy_backend,
                noise_model=noise_model,
                chosen_layout=chosen_layout,
                output_dir=seed_dir,
                shots=int(noisy_eval_shots),
                eval_runs=int(noisy_eval_runs),
                seed_base=int(noisy_eval_seed_base) + int(theta_seed) * 1000,
                alpha=float(noisy_eval_alpha),
            )
            row.update(noisy_row)
            row["kl_noise_penalty_mean"] = float(
                row["kl_shots_noise_mean"] - row["kl_final"]
            )
            _write_json(summary_path, row)
        print(f"[resume] L={n_layers:02d} seed={theta_seed} loaded")
        return row

    rng = np.random.default_rng(int(theta_seed))
    theta_init = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)
    p_init = qcbm.probabilities(theta_init)
    kl_init = kl_divergence(ptg, p_init, eps=EPS_COST)
    cost_statevector = qcbm.cost_fn(ptg, eps=EPS_COST)

    print(f"\n=== Training L={n_layers:02d} | theta_seed={theta_seed} ===")
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
        f"L={n_layers:02d} seed={theta_seed} stage1 | "
        f"KL={stage1['kl_final']:.8e} | t={stage1['elapsed_time']:.2f}s"
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
        f"L={n_layers:02d} seed={theta_seed} stage2 | "
        f"KL={stage2['kl_final']:.8e} | t={stage2['elapsed_time']:.2f}s"
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

    row = {
        "n_layers": int(n_layers),
        "theta_seed": int(theta_seed),
        "n_qubits": int(qcbm.n_qubits),
        "n_params": int(qcbm.n_params),
        "n_entangling_pairs": int(len(qcbm.pairs)),
        "entangler": ENTANGLER,
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
        "ansatz_depth": int(ansatz_summary["depth"]),
        "ansatz_size": int(ansatz_summary["size"]),
        "ansatz_width": int(ansatz_summary["width"]),
        "ansatz_one_qubit_ops": int(ansatz_summary["one_qubit_ops"]),
        "ansatz_two_qubit_ops": int(ansatz_summary["two_qubit_ops"]),
        "ansatz_two_qubit_depth": int(ansatz_summary["two_qubit_depth"]),
        "ansatz_swap": int(ansatz_summary["swap"]),
        "ansatz_rzz": int(ansatz_summary["rzz"]),
        "measured_depth": int(measured_summary["depth"]),
        "measured_two_qubit_ops": int(measured_summary["two_qubit_ops"]),
    }

    if run_noisy_eval:
        noisy_row, _, _ = evaluate_theta_under_shots_noise(
            n_layers=int(n_layers),
            theta_star=theta_star,
            ptg=ptg,
            n_time=n_time,
            transpile_backend=real_backend,
            noisy_backend=noisy_backend,
            noise_model=noise_model,
            chosen_layout=chosen_layout,
            output_dir=seed_dir,
            shots=int(noisy_eval_shots),
            eval_runs=int(noisy_eval_runs),
            seed_base=int(noisy_eval_seed_base) + int(theta_seed) * 1000,
            alpha=float(noisy_eval_alpha),
        )
        row.update(noisy_row)
        row["kl_noise_penalty_mean"] = float(
            row["kl_shots_noise_mean"] - row["kl_final"]
        )

    np.savez(
        result_path,
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


def summarize_layers(
    seed_df: pd.DataFrame,
    *,
    convergence_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_layers, group in seed_df.groupby("n_layers", sort=True):
        kl_stats = _metric_stats(group["kl_final"])
        elapsed_stats = _metric_stats(group["elapsed_total_s"])
        stage2_success_rate = float(np.mean(group["stage2_success"].astype(bool)))
        converged = group["kl_final"].astype(float) <= float(convergence_threshold)
        base = {
            "n_layers": int(n_layers),
            "n_runs": int(len(group)),
            "n_params": int(group["n_params"].iloc[0]),
            "ansatz_depth": int(group["ansatz_depth"].iloc[0]),
            "ansatz_two_qubit_ops": int(group["ansatz_two_qubit_ops"].iloc[0]),
            "ansatz_two_qubit_depth": int(group["ansatz_two_qubit_depth"].iloc[0]),
            "ansatz_swap": int(group["ansatz_swap"].iloc[0]),
            "kl_convergence_threshold": float(convergence_threshold),
            "convergence_rate": float(np.mean(converged)),
            "stage2_success_rate": stage2_success_rate,
            "elapsed_total_mean_s": elapsed_stats["mean"],
            "elapsed_total_median_s": elapsed_stats["median"],
        }
        for key, value in kl_stats.items():
            base[f"kl_final_{key}"] = value
        rows.append(base)
    return pd.DataFrame(rows).sort_values("n_layers").reset_index(drop=True)


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


def nonparametric_layer_tests(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [
        (
            int(n_layers),
            group["kl_final"].astype(float).to_numpy(dtype=float),
        )
        for n_layers, group in seed_df.groupby("n_layers", sort=True)
    ]

    global_rows: list[dict[str, Any]] = []
    if len(groups) >= 2 and all(values.size > 0 for _, values in groups):
        try:
            stat, p_value = kruskal(*(values for _, values in groups))
        except ValueError as exc:
            stat, p_value = math.nan, math.nan
            global_rows.append({"test": "kruskal", "error": str(exc)})
        else:
            global_rows.append(
                {
                    "test": "kruskal",
                    "metric": "kl_final",
                    "n_groups": int(len(groups)),
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "error": "",
                }
            )

    pair_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for i, (layer_a, values_a) in enumerate(groups):
        for layer_b, values_b in groups[i + 1 :]:
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
            pair_rows.append(
                {
                    "layer_a": int(layer_a),
                    "layer_b": int(layer_b),
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
    for row, p_adj in zip(pair_rows, adjusted, strict=True):
        row["p_value_holm"] = float(p_adj)

    return pd.DataFrame(global_rows), pd.DataFrame(pair_rows)


def plot_multiseed_kl_boxplot(seed_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    layers = sorted(int(v) for v in seed_df["n_layers"].unique())
    data = [
        np.maximum(
            seed_df.loc[seed_df["n_layers"] == layer, "kl_final"].to_numpy(dtype=float),
            1e-15,
        )
        for layer in layers
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.boxplot(
        data,
        labels=[f"L={layer}" for layer in layers],
        showmeans=True,
        patch_artist=True,
        medianprops={"color": "#111827", "linewidth": 1.4},
        boxprops={"facecolor": "#dbeafe", "edgecolor": "#1d4ed8"},
        whiskerprops={"color": "#1d4ed8"},
        capprops={"color": "#1d4ed8"},
        meanprops={
            "marker": "D",
            "markerfacecolor": "#c2410c",
            "markeredgecolor": "#7c2d12",
            "markersize": 5,
        },
    )
    ax.set_yscale("log")
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel(r"$D_{KL}(p_{\mathrm{target}}\Vert p_{\theta})$")
    ax.set_title("Multi-start robustness of final QCBM KL")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "multiseed_final_kl_boxplot")


def plot_convergence_rate(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = summary_df["n_layers"].to_numpy(dtype=int)
    ax.bar(
        x,
        summary_df["convergence_rate"].to_numpy(dtype=float),
        width=0.65,
        color="#0f766e",
        edgecolor="#134e4a",
        alpha=0.88,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("QCBM layers")
    ax.set_ylabel("Convergence rate")
    ax.set_title("Fraction of seeds below the KL convergence threshold")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "multiseed_convergence_rate")


def plot_median_kl_vs_depth(summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.semilogy(
        summary_df["ansatz_two_qubit_depth"],
        np.maximum(summary_df["kl_final_median"], 1e-15),
        marker="o",
        color="#7c3aed",
    )
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"L={int(row['n_layers'])}",
            xy=(row["ansatz_two_qubit_depth"], max(row["kl_final_median"], 1e-15)),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Transpiled two-qubit depth")
    ax.set_ylabel("Median final KL across seeds")
    ax.set_title("Robust accuracy versus hardware-relevant depth")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "multiseed_median_kl_vs_two_qubit_depth")


def generate_plots(seed_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    plot_dir = output_dir / "figures"
    configure_matplotlib()
    plot_multiseed_kl_boxplot(seed_df, plot_dir)
    plot_convergence_rate(summary_df, plot_dir)
    plot_median_kl_vs_depth(summary_df, plot_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-start statistical robustness experiment for the 6q "
            "multi-asset CVA QCBM. Defaults test layers 4, 6, 8 and 10 "
            "with 20 independent theta initializations."
        )
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="QCBM layer values to test. Default: 4 6 8 10.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Number of independent theta initialization seeds per layer.",
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
        default=DEFAULT_CONVERGENCE_THRESHOLD,
        help="KL threshold used to report convergence rate.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain cached runs.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figures.")
    parser.add_argument(
        "--stage1-maxiter",
        type=int,
        default=STAGE1_MAXITER,
        help="COBYLA maxiter per seed.",
    )
    parser.add_argument(
        "--stage2-maxiter",
        type=int,
        default=STAGE2_MAXITER,
        help="L-BFGS-B maxiter per seed.",
    )
    parser.add_argument(
        "--stage2-maxfun",
        type=int,
        default=STAGE2_MAXFUN,
        help="L-BFGS-B maxfun per seed.",
    )
    parser.add_argument(
        "--run-noisy-eval",
        action="store_true",
        help=(
            "Evaluate each final theta under finite shots and backend noise. "
            "This multiplies the runtime by n_seeds and is disabled by default."
        ),
    )
    parser.add_argument(
        "--noisy-eval-runs",
        type=int,
        default=NOISY_EVAL_RUNS,
        help="Independent noisy shot evaluations per trained theta.",
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
        help="Base simulator seed for noisy evaluations.",
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
    print(f"Theta seeds: {theta_seeds}")
    print(f"Output directory: {output_dir}")

    backend_ctx = _prepare_backend_context(snapshot_dt_utc)
    real_backend = backend_ctx["real_backend"]
    backend_props = backend_ctx["backend_props"]
    noisy_backend = backend_ctx["noisy_backend"]
    noise_model = backend_ctx["noise_model"]
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
        "entangler": ENTANGLER,
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
        "run_noisy_eval": bool(args.run_noisy_eval),
        "noisy_eval_runs": int(args.noisy_eval_runs),
        "noisy_eval_shots": int(args.noisy_eval_shots),
        "noisy_eval_seed_base": int(args.noisy_eval_seed_base),
        "dirichlet_alpha": float(DIRICHLET_ALPHA),
        "simulator_seed": int(SIMULATOR_SEED),
        "noise_thermal_relaxation": False,
        "noise_basis_gates": list(noise_model.basis_gates),
        "used_noise_fallback": bool(backend_ctx["used_noise_fallback"]),
        "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
        "output_dir": str(output_dir),
        "training_regime": "ideal_statevector_multistart",
        "cost_function": "MLQcbmCircuit.cost_fn(ptg, eps=EPS_COST)",
        "reported_metric": "KL(target || qcbm)",
        "flattening_order": "time_major",
    }
    _write_json(output_dir / "run_config.json", run_config)

    seed_rows: list[dict[str, Any]] = []
    transpile_rows: list[dict[str, Any]] = []

    for n_layers in layers:
        layer_dir = output_dir / f"L{int(n_layers):02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        qcbm = build_qcbm(
            n_qubits=n_qubits,
            n_layers=int(n_layers),
            backend=real_backend,
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
        transpile_rows.extend([ansatz_summary, measured_summary])

        with open(layer_dir / f"qcbm_multiseed_template_L{int(n_layers):02d}.qpy", "wb") as f:
            qpy.dump(qcbm._tqc, f)
        with open(
            layer_dir / f"qcbm_multiseed_template_measured_L{int(n_layers):02d}.qpy",
            "wb",
        ) as f:
            qpy.dump(qcbm._tqc_meas, f)

        for theta_seed in theta_seeds:
            row = train_one_seed(
                qcbm=qcbm,
                n_layers=int(n_layers),
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
                run_noisy_eval=bool(args.run_noisy_eval),
                real_backend=real_backend,
                noisy_backend=noisy_backend,
                noise_model=noise_model,
                chosen_layout=chosen_layout,
                noisy_eval_shots=int(args.noisy_eval_shots),
                noisy_eval_runs=int(args.noisy_eval_runs),
                noisy_eval_seed_base=int(args.noisy_eval_seed_base),
                noisy_eval_alpha=float(DIRICHLET_ALPHA),
            )
            seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows).sort_values(
        ["n_layers", "theta_seed"]
    ).reset_index(drop=True)
    transpile_df = pd.DataFrame(transpile_rows).sort_values(
        ["n_layers", "circuit_kind"]
    ).reset_index(drop=True)
    summary_df = summarize_layers(
        seed_df,
        convergence_threshold=float(args.convergence_threshold),
    )
    global_tests_df, pairwise_tests_df = nonparametric_layer_tests(seed_df)

    save_table_bundle(
        seed_df,
        output_dir=tables_dir,
        stem="multiseed_per_run_results",
        title="Per-run QCBM multi-start results",
        image_columns=[
            "n_layers",
            "theta_seed",
            "n_params",
            "ansatz_depth",
            "ansatz_two_qubit_ops",
            "kl_init",
            "kl_stage1",
            "kl_final",
            "elapsed_total_s",
            "stage2_success",
        ],
    )
    save_table_bundle(
        summary_df,
        output_dir=tables_dir,
        stem="multiseed_layer_robustness_summary",
        title="Layer-level QCBM multi-start robustness summary",
        image_columns=[
            "n_layers",
            "n_runs",
            "n_params",
            "ansatz_two_qubit_depth",
            "kl_final_mean",
            "kl_final_median",
            "kl_final_iqr",
            "kl_final_min",
            "kl_final_max",
            "convergence_rate",
            "stage2_success_rate",
        ],
    )
    save_table_bundle(
        transpile_df,
        output_dir=tables_dir,
        stem="multiseed_transpile_table",
        title="Backend-transpiled QCBM resources for multi-start layers",
        image_columns=[
            "n_layers",
            "circuit_kind",
            "n_params",
            "depth",
            "two_qubit_ops",
            "two_qubit_depth",
            "rzz",
            "swap",
        ],
    )
    if not global_tests_df.empty:
        save_table_bundle(
            global_tests_df,
            output_dir=tables_dir,
            stem="multiseed_global_nonparametric_tests",
            title="Global non-parametric layer comparison tests",
        )
    if not pairwise_tests_df.empty:
        save_table_bundle(
            pairwise_tests_df,
            output_dir=tables_dir,
            stem="multiseed_pairwise_nonparametric_tests",
            title="Pairwise Mann-Whitney layer comparison tests",
            image_columns=[
                "layer_a",
                "layer_b",
                "n_a",
                "n_b",
                "median_a",
                "median_b",
                "delta_median_a_minus_b",
                "p_value",
                "p_value_holm",
            ],
        )

    np.savez(
        output_dir / "multiseed_aggregate.npz",
        seed_results_json=np.array(
            json.dumps(_json_ready(seed_df.to_dict(orient="records")))
        ),
        summary_json=np.array(
            json.dumps(_json_ready(summary_df.to_dict(orient="records")))
        ),
        pairwise_tests_json=np.array(
            json.dumps(_json_ready(pairwise_tests_df.to_dict(orient="records")))
        ),
        layers=summary_df["n_layers"].to_numpy(dtype=int),
        theta_seeds=np.asarray(theta_seeds, dtype=int),
        kl_final=seed_df["kl_final"].to_numpy(dtype=float),
        p_target=ptg,
    )

    if not args.skip_plots:
        generate_plots(seed_df, summary_df, output_dir)

    print("\n=== QCBM multi-start robustness experiment complete ===")
    print(f"Results: {output_dir}")
    print("\nLayer robustness summary:")
    final_cols = [
        "n_layers",
        "n_runs",
        "n_params",
        "ansatz_two_qubit_depth",
        "kl_final_median",
        "kl_final_iqr",
        "kl_final_min",
        "kl_final_max",
        "convergence_rate",
    ]
    print(
        summary_df[final_cols].to_string(
            index=False,
            float_format=lambda x: _format_float(x, 4),
        )
    )


if __name__ == "__main__":
    main()
