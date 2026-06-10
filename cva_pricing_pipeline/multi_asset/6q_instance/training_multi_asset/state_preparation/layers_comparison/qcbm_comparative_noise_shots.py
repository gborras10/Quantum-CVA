from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_algorithms.optimizers import SPSA
from qiskit_ibm_runtime import QiskitRuntimeService


def _bootstrap_paths() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    current_dir = current.parent
    for path in (src_path, current_dir):
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
    THETA_SEED,
    USE_FRACTIONAL_GATES,
    _dirichlet_smooth,
    _format_float,
    _json_ready,
    _normalize_distribution,
    _parse_snapshot_datetime,
    _write_json,
    build_qcbm,
    build_noisy_qcbm,
    evaluate_theta_under_shots_noise,
    kl_divergence,
    kl_mass_contribution,
    kl_time_decomposition,
    plot_cumulative_kl_concentration,
    plot_kl_vs_resources,
    plot_resource_scaling,
    plot_time_major_heatmaps,
    plot_top_state_fit,
    save_figure,
    save_table_bundle,
    select_qcbm_heavyhex6_layout_from_snapshot,
    snapshot_edge_table,
    snapshot_qubit_table,
    summarize_transpiled_circuit,
)


# ======================================================================
# Noisy training configuration
# ======================================================================

RESULTS_RELATIVE_DIR = (
    "cva_pricing_pipeline/multi_asset/6q_instance/training_multi_asset/"
    "state_preparation/layers_comparison/results/noise_shots"
)

LAYERS_GRID = [2, 6, 10, 16]
TRAIN_SHOTS = 60000
N_ITERS = 200
RESAMPLINGS = 3
SPSA_BLOCKING = False
SPSA_TRUST_REGION = True
SPSA_REGULARIZATION = 0.01

CHECKPOINT_TOL = 1e-15
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
TRAINING_CURVE_RIGHT_MARGIN_FRACTION = 0.05


@dataclass
class NoisyLayerArtifact:
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
# Noisy training
# ======================================================================


def _safe_metric(metrics: dict[str, float], key: str) -> float:
    value = metrics.get(key, math.nan)
    return float(value) if value is not None else math.nan


def train_one_layer_noise(
    *,
    n_layers: int,
    ptg: np.ndarray,
    n_time: int,
    real_backend,
    noisy_backend,
    noise_model,
    chosen_layout: list[int],
    layout_score: float,
    layout_meta: dict[str, Any],
    snapshot_dt_utc: datetime,
    output_dir: pathlib.Path,
    train_shots: int,
    n_iters: int,
    resamplings: int,
    noisy_eval_shots: int,
    noisy_eval_runs: int,
    noisy_eval_seed_base: int,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], NoisyLayerArtifact]:
    layer_dir = output_dir / f"L{int(n_layers):02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    result_path = layer_dir / f"qcbm_noise_shots_L{int(n_layers):02d}.npz"
    summary_path = layer_dir / "summary_row.json"
    noisy_eval_path = layer_dir / f"noisy_eval_L{int(n_layers):02d}.npz"

    if result_path.exists() and summary_path.exists() and not force:
        data = np.load(result_path, allow_pickle=True)
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        transpile_rows = json.loads(
            (layer_dir / "transpile_rows.json").read_text(encoding="utf-8")
        )

        kl_noisy_values = None
        p_noisy_mean = None
        if noisy_eval_path.exists():
            noisy_data = np.load(noisy_eval_path, allow_pickle=True)
            kl_noisy_values = np.asarray(
                noisy_data["kl_shots_noise_values"], dtype=float
            )
            p_noisy_mean = np.asarray(
                noisy_data["p_noisy_mean_smooth"], dtype=float
            )
        else:
            noisy_row, kl_noisy_values, p_noisy_mean = evaluate_theta_under_shots_noise(
                n_layers=int(n_layers),
                theta_star=np.asarray(data["theta_star"], dtype=float),
                ptg=ptg,
                n_time=n_time,
                transpile_backend=real_backend,
                noisy_backend=noisy_backend,
                noise_model=noise_model,
                chosen_layout=chosen_layout,
                output_dir=layer_dir,
                shots=int(noisy_eval_shots),
                eval_runs=int(noisy_eval_runs),
                seed_base=int(noisy_eval_seed_base),
                alpha=float(DIRICHLET_ALPHA),
            )
            row.update(noisy_row)
            row["kl_final"] = float(row["kl_shots_noise_mean"])
            row["kl_noise_eval_minus_train_best"] = float(
                row["kl_shots_noise_mean"] - row["kl_train_best_observed"]
            )
            _write_json(summary_path, row)

        artifact = NoisyLayerArtifact(
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

    print(f"\n=== Training shots+noise QCBM | layers={n_layers:02d} ===")
    qcbm = build_noisy_qcbm(
        n_qubits=int(round(math.log2(ptg.size))),
        n_layers=int(n_layers),
        transpile_backend=real_backend,
        noisy_backend=noisy_backend,
        noise_model=noise_model,
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

    cost_shots = qcbm.cost_fn(
        ptg,
        eps=EPS_COST,
        shots=int(train_shots),
        seed=None,
        rescaled=True,
        smoothing="dirichlet",
        alpha=DIRICHLET_ALPHA,
    )

    rng = np.random.default_rng(THETA_SEED)
    theta0 = INIT_SCALE * rng.standard_normal(qcbm.n_params).astype(float)

    p_init_raw = qcbm.probabilities(
        theta0,
        shots=int(noisy_eval_shots),
        seed=int(noisy_eval_seed_base),
    )
    p_init = _dirichlet_smooth(
        p_init_raw,
        shots=int(noisy_eval_shots),
        alpha=DIRICHLET_ALPHA,
    )
    kl_init = kl_divergence(ptg, p_init, eps=EPS_COST)

    cost_history: list[float] = []
    theta_history: list[np.ndarray] = [theta0.copy()]
    best = {"fx": float("inf"), "x": theta0.copy()}
    iter_times: list[float] = []
    live_log: list[dict[str, float]] = []
    training_t0: float | None = None
    last_callback_t: float | None = None

    def callback(nfev, x, fx, stepsize, accepted):
        nonlocal training_t0, last_callback_t

        now = time.perf_counter()
        if training_t0 is None:
            training_t0 = now
        iter_dt = 0.0 if last_callback_t is None else now - last_callback_t
        last_callback_t = now

        fx = float(fx)
        x_arr = np.asarray(x, dtype=float).copy()
        cost_history.append(fx)
        theta_history.append(x_arr)
        iter_times.append(iter_dt)

        if fx < best["fx"]:
            best["fx"] = fx
            best["x"] = x_arr.copy()

        iter_idx = len(cost_history)
        elapsed_total = now - training_t0
        mean_iter_time = (
            float(np.mean(iter_times[1:])) if len(iter_times) > 1 else 0.0
        )
        live_log.append(
            {
                "iter": float(iter_idx),
                "nfev": float(nfev),
                "fx": fx,
                "best_fx": float(best["fx"]),
                "iter_time_s": float(iter_dt),
                "mean_iter_time_s": mean_iter_time,
                "elapsed_s": float(elapsed_total),
                "stepsize": float(stepsize),
                "accepted": float(bool(accepted)),
            }
        )
        print(
            f"[L={n_layers:02d} iter {iter_idx:5d}] "
            f"fx={fx:.8e} | best={best['fx']:.8e} | "
            f"dt={iter_dt:7.2f}s | mean_dt={mean_iter_time:7.2f}s | "
            f"nfev={nfev:6d} | step={float(stepsize):.3e}"
        )

    print(f"L={n_layers:02d} calibrating SPSA...")
    learning_rate, perturbation = SPSA.calibrate(cost_shots, theta0)

    opt = SPSA(
        maxiter=int(n_iters),
        learning_rate=learning_rate,
        perturbation=perturbation,
        resamplings=int(resamplings),
        blocking=SPSA_BLOCKING,
        callback=callback,
        trust_region=SPSA_TRUST_REGION,
        regularization=SPSA_REGULARIZATION,
    )

    t0 = time.perf_counter()
    result = opt.minimize(fun=cost_shots, x0=theta0)
    elapsed_time = time.perf_counter() - t0

    theta_last = np.asarray(result.x, dtype=float)
    theta_star = np.asarray(best["x"], dtype=float).copy()
    cost_history_arr = np.asarray(cost_history, dtype=float)
    theta_history_arr = np.asarray(theta_history, dtype=float)
    if cost_history_arr.size == 0:
        final_fx = float(cost_shots(theta_star))
        cost_history_arr = np.asarray([final_fx], dtype=float)
        theta_history_arr = np.vstack([theta_history_arr, theta_star])
        best["fx"] = min(best["fx"], final_fx)

    kl_history = np.maximum(cost_history_arr, 1e-15)
    best_kl_history = np.minimum.accumulate(kl_history)
    best_idx = np.flatnonzero(
        np.r_[True, best_kl_history[1:] < best_kl_history[:-1] - CHECKPOINT_TOL]
    )

    print(
        f"L={n_layers:02d} training complete | "
        f"best noisy objective={float(best['fx']):.8e} | "
        f"elapsed={elapsed_time:.2f}s"
    )

    noisy_row, kl_noisy_values, p_noisy_mean = evaluate_theta_under_shots_noise(
        n_layers=int(n_layers),
        theta_star=theta_star,
        ptg=ptg,
        n_time=n_time,
        transpile_backend=real_backend,
        noisy_backend=noisy_backend,
        noise_model=noise_model,
        chosen_layout=chosen_layout,
        output_dir=layer_dir,
        shots=int(noisy_eval_shots),
        eval_runs=int(noisy_eval_runs),
        seed_base=int(noisy_eval_seed_base),
        alpha=float(DIRICHLET_ALPHA),
    )

    qcbm_clean = build_qcbm(
        n_qubits=int(round(math.log2(ptg.size))),
        n_layers=int(n_layers),
        backend=real_backend,
        chosen_layout=chosen_layout,
    )
    p_clean = qcbm_clean.probabilities(theta_star)
    kl_clean = kl_divergence(ptg, p_clean, eps=EPS_COST)
    clean_metrics = qcbm_clean.metrics(ptg, p_clean, eps=EPS_COST)

    p_last_raw = qcbm.probabilities(
        theta_last,
        shots=int(noisy_eval_shots),
        seed=int(noisy_eval_seed_base) + 1000,
    )
    p_last = _dirichlet_smooth(
        p_last_raw,
        shots=int(noisy_eval_shots),
        alpha=DIRICHLET_ALPHA,
    )
    kl_last = kl_divergence(ptg, p_last, eps=EPS_COST)
    noisy_metrics = qcbm.metrics(ptg, p_noisy_mean, eps=EPS_COST)
    noisy_decomp = kl_time_decomposition(
        ptg,
        p_noisy_mean,
        n_time=n_time,
        eps=EPS_COST,
    )
    kl_top90, kl_tail10, n_top90 = kl_mass_contribution(
        ptg,
        p_noisy_mean,
        mass_threshold=0.90,
        eps=EPS_COST,
    )
    kl_top99, kl_tail01, n_top99 = kl_mass_contribution(
        ptg,
        p_noisy_mean,
        mass_threshold=0.99,
        eps=EPS_COST,
    )

    n_rot_layers = (int(n_layers) + 1) // 2
    n_ent_layers = int(n_layers) // 2
    row = {
        "n_layers": int(n_layers),
        "n_qubits": int(qcbm.n_qubits),
        "n_params": int(qcbm.n_params),
        "n_rot_layers": int(n_rot_layers),
        "n_ent_layers": int(n_ent_layers),
        "n_entangling_pairs": int(len(qcbm.pairs)),
        "theta_seed": int(THETA_SEED),
        "init_scale": float(INIT_SCALE),
        "train_shots": int(train_shots),
        "n_iters": int(n_iters),
        "resamplings": int(resamplings),
        "dirichlet_alpha": float(DIRICHLET_ALPHA),
        "kl_init": float(kl_init),
        "kl_train_best_observed": float(np.min(best_kl_history)),
        "kl_last_noisy_eval": float(kl_last),
        "kl_final": float(noisy_row["kl_shots_noise_mean"]),
        "kl_clean_statevector": float(kl_clean),
        "kl_noise_eval_minus_clean": float(
            noisy_row["kl_shots_noise_mean"] - kl_clean
        ),
        "kl_noise_eval_minus_train_best": float(
            noisy_row["kl_shots_noise_mean"] - np.min(best_kl_history)
        ),
        "kl_top90_contribution": kl_top90,
        "kl_tail10_contribution": kl_tail10,
        "n_states_for_90pct_target_mass": int(n_top90),
        "kl_top99_contribution": kl_top99,
        "kl_tail01_contribution": kl_tail01,
        "n_states_for_99pct_target_mass": int(n_top99),
        "kl_time_marginal": noisy_decomp["kl_time_marginal"],
        "kl_time_conditional": noisy_decomp["kl_time_conditional"],
        "kl_time_conditional_max": noisy_decomp["kl_time_conditional_max"],
        "elapsed_total_s": float(elapsed_time),
        "n_history_points": int(kl_history.size),
        "best_idx_count": int(best_idx.size),
        "optimizer_success": bool(getattr(result, "success", False)),
        "optimizer_message": str(getattr(result, "message", "")),
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
        "clean_metric_l1": _safe_metric(clean_metrics, "l1"),
        "clean_metric_tv": _safe_metric(clean_metrics, "tv"),
        "clean_metric_linf": _safe_metric(clean_metrics, "linf"),
        "noisy_metric_l1": _safe_metric(noisy_metrics, "l1"),
        "noisy_metric_tv": _safe_metric(noisy_metrics, "tv"),
        "noisy_metric_linf": _safe_metric(noisy_metrics, "linf"),
    }
    row.update(noisy_row)

    with open(layer_dir / f"trained_qcbm_noise_L{int(n_layers):02d}.qpy", "wb") as f:
        qpy.dump(qcbm._tqc, f)
    with open(
        layer_dir / f"trained_qcbm_noise_measured_L{int(n_layers):02d}.qpy",
        "wb",
    ) as f:
        qpy.dump(qcbm._tqc_meas, f)

    np.savez(
        result_path,
        theta_star=theta_star,
        theta_last=theta_last,
        theta_init=theta0,
        theta_history=theta_history_arr,
        cost_history=cost_history_arr,
        kl_history=kl_history,
        best_kl_history=best_kl_history,
        best_idx=best_idx,
        live_log=np.array(live_log, dtype=object),
        p_target=ptg,
        p_init=p_init,
        p_init_raw=p_init_raw,
        p_last=p_last,
        p_last_raw=p_last_raw,
        p_star=p_noisy_mean,
        p_clean_statevector=p_clean,
        kl_init=np.float64(kl_init),
        kl_train_best_observed=np.float64(np.min(best_kl_history)),
        kl_final=np.float64(row["kl_final"]),
        kl_clean_statevector=np.float64(kl_clean),
        kl_shots_noise_values=kl_noisy_values,
        elapsed_time=np.float64(elapsed_time),
        train_shots=np.int64(train_shots),
        n_iters=np.int64(n_iters),
        resamplings=np.int64(resamplings),
        dirichlet_alpha=np.float64(DIRICHLET_ALPHA),
        metrics_clean=np.array(clean_metrics, dtype=object),
        metrics_noisy=np.array(noisy_metrics, dtype=object),
        summary_row_json=np.array(json.dumps(_json_ready(row), sort_keys=True)),
        transpile_rows_json=np.array(
            json.dumps(_json_ready(transpile_rows), sort_keys=True)
        ),
    )
    _write_json(summary_path, row)
    _write_json(layer_dir / "transpile_rows.json", {"rows": transpile_rows})

    artifact = NoisyLayerArtifact(
        n_layers=int(n_layers),
        result_path=result_path,
        kl_history=kl_history,
        best_kl_history=best_kl_history,
        p_star=p_noisy_mean,
        p_init=p_init,
        p_target=ptg,
        kl_noisy_values=kl_noisy_values,
        p_noisy_mean=p_noisy_mean,
    )
    return row, transpile_rows, artifact


# ======================================================================
# Plots
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


def plot_noise_training_kl_vs_layers(
    summary_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x = summary_df["n_layers"].to_numpy()
    ax.semilogy(
        x,
        summary_df["kl_init"],
        marker="o",
        linestyle="--",
        color="#7b8794",
        label="Initial noisy KL",
    )
    ax.semilogy(
        x,
        summary_df["kl_train_best_observed"],
        marker="s",
        linestyle="-.",
        color="#2563eb",
        label="Best noisy training objective",
    )
    ax.errorbar(
        x,
        summary_df["kl_final"],
        yerr=summary_df["kl_shots_noise_std"],
        marker="D",
        color="#c2410c",
        capsize=3,
        label="Post-training noisy KL",
    )
    if "kl_clean_statevector" in summary_df.columns:
        ax.semilogy(
            x,
            summary_df["kl_clean_statevector"],
            marker="^",
            color="#111827",
            label="Clean statevector eval of noisy theta",
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
    ax.set_title("Shots + backend-noise QCBM training across layers")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "noise_training_kl_vs_layers")


def plot_noise_decomposition(
    summary_df: pd.DataFrame,
    output_dir: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = summary_df["n_layers"].to_numpy()
    ax.semilogy(
        x,
        summary_df["kl_final"],
        marker="D",
        color="#111827",
        label="Noisy joint KL",
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
    ax.set_title("Noisy KL decomposition under time-major CVA factorization")
    ax.set_xticks(x)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "noise_kl_time_decomposition_vs_layers")


def plot_noise_penalty(
    summary_df: pd.DataFrame,
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
        x = summary_df["n_layers"].to_numpy()
        ax.semilogy(
            x,
            summary_df["kl_clean_statevector"],
            marker="^",
            markersize=5.5,
            color="#111827",
            linewidth=1.75,
            label="Noiseless evaluation",
        )
        ax.errorbar(
            x,
            summary_df["kl_final"],
            yerr=summary_df["kl_shots_noise_std"],
            marker="o",
            markersize=5.5,
            color="#c2410c",
            capsize=3,
            linewidth=1.75,
            label="Noisy evaluation",
        )
        ax.set_xlabel("QCBM layers")
        ax.set_ylabel(r"$\mathrm{KL}_{\epsilon}(P_{\mathrm{target}}\Vert P_{\theta})$")
        ax.set_xticks(x)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
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
            frameon=False,
            loc="upper right",
            handlelength=2.7,
        )
        fig.tight_layout(pad=0.45)
        save_figure(fig, output_dir, "noise_training_clean_vs_noisy_eval")


def plot_noise_training_curves(
    artifacts: list[NoisyLayerArtifact],
    output_dir: pathlib.Path,
) -> None:
    max_iteration = max(int(artifact.best_kl_history.size - 1) for artifact in artifacts)
    xmax = max_iteration * (1.0 + TRAINING_CURVE_RIGHT_MARGIN_FRACTION)

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
            x = np.arange(artifact.best_kl_history.size)
            color = PAPER_LAYER_COLORS.get(int(artifact.n_layers), "#444444")
            ax.semilogy(
                x,
                artifact.best_kl_history,
                color=color,
                alpha=0.96,
                linewidth=1.75,
                label=f"L={artifact.n_layers}",
            )
        ax.set_xlim(0, xmax)
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
            ncol=2,
            frameon=False,
            loc="upper right",
            handlelength=2.7,
            columnspacing=1.05,
        )
        fig.tight_layout(pad=0.45)
        save_figure(fig, output_dir, "training_kl_trajectories")


def generate_noise_plots(
    summary_df: pd.DataFrame,
    artifacts: list[NoisyLayerArtifact],
    *,
    output_dir: pathlib.Path,
    n_time: int,
) -> None:
    plot_dir = output_dir / "figures"
    configure_matplotlib()
    plot_noise_training_kl_vs_layers(summary_df, plot_dir)
    plot_noise_decomposition(summary_df, plot_dir)
    plot_noise_penalty(summary_df, plot_dir)
    plot_noise_training_curves(artifacts, plot_dir)
    plot_resource_scaling(summary_df, plot_dir)
    plot_kl_vs_resources(summary_df, plot_dir)
    plot_top_state_fit(artifacts, plot_dir)
    plot_time_major_heatmaps(artifacts, plot_dir, n_time=n_time)
    plot_cumulative_kl_concentration(artifacts, plot_dir)


# ======================================================================
# Main
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shots + backend-noise QCBM layer comparison for the 6q "
            "multi-asset CVA instance. The noise model, snapshot and SPSA "
            "training setup match the current noisy 6q QCBM training."
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
        "--train-shots",
        type=int,
        default=TRAIN_SHOTS,
        help="Shots per noisy SPSA objective evaluation.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=N_ITERS,
        help="SPSA iterations per layer.",
    )
    parser.add_argument(
        "--resamplings",
        type=int,
        default=RESAMPLINGS,
        help="SPSA resamplings per perturbation.",
    )
    parser.add_argument(
        "--noisy-eval-runs",
        type=int,
        default=NOISY_EVAL_RUNS,
        help="Independent post-training noisy evaluations per layer.",
    )
    parser.add_argument(
        "--noisy-eval-shots",
        type=int,
        default=NOISY_EVAL_SHOTS,
        help="Shots per post-training noisy evaluation.",
    )
    parser.add_argument(
        "--noisy-eval-seed-base",
        type=int,
        default=NOISY_EVAL_SEED_BASE,
        help="Base simulator seed for post-training noisy evaluations.",
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
    ptg = _normalize_distribution(np.asarray(data["p_target"], dtype=float).ravel())
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
        "train_shots": int(args.train_shots),
        "n_iters": int(args.n_iters),
        "resamplings": int(args.resamplings),
        "spsa_blocking": SPSA_BLOCKING,
        "spsa_trust_region": SPSA_TRUST_REGION,
        "spsa_regularization": SPSA_REGULARIZATION,
        "noisy_eval_runs": int(args.noisy_eval_runs),
        "noisy_eval_shots": int(args.noisy_eval_shots),
        "noisy_eval_seed_base": int(args.noisy_eval_seed_base),
        "dirichlet_alpha": float(DIRICHLET_ALPHA),
        "simulator_seed": int(SIMULATOR_SEED),
        "simulator_method": "density_matrix",
        "noise_thermal_relaxation": False,
        "noise_basis_gates": list(noise_model.basis_gates),
        "used_noise_fallback": bool(used_noise_fallback),
        "chosen_layout": chosen_layout,
        "layout_score": layout_score,
        "layout_meta": layout_meta,
        "benchmark_relative_path": BENCHMARK_RELATIVE_PATH,
        "output_dir": str(output_dir),
        "training_regime": "shots_plus_backend_noise",
        "cost_function": (
            "MLQcbmCircuit.cost_fn(ptg, eps=EPS_COST, shots=train_shots, "
            "rescaled=True, smoothing='dirichlet', alpha=DIRICHLET_ALPHA)"
        ),
        "reported_metric": "KL(target || qcbm)",
        "flattening_order": "time_major",
    }
    _write_json(output_dir / "run_config.json", run_config)

    rows: list[dict[str, Any]] = []
    transpile_rows_all: list[dict[str, Any]] = []
    artifacts: list[NoisyLayerArtifact] = []

    for layer in layers:
        row, transpile_rows, artifact = train_one_layer_noise(
            n_layers=layer,
            ptg=ptg,
            n_time=n_time,
            real_backend=real_backend,
            noisy_backend=noisy_backend,
            noise_model=noise_model,
            chosen_layout=chosen_layout,
            layout_score=layout_score,
            layout_meta=layout_meta,
            snapshot_dt_utc=snapshot_dt_utc,
            output_dir=output_dir,
            train_shots=int(args.train_shots),
            n_iters=int(args.n_iters),
            resamplings=int(args.resamplings),
            noisy_eval_shots=int(args.noisy_eval_shots),
            noisy_eval_runs=int(args.noisy_eval_runs),
            noisy_eval_seed_base=int(args.noisy_eval_seed_base),
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
        "kl_train_best_observed",
        "kl_final",
        "kl_shots_noise_std",
        "kl_clean_statevector",
        "kl_noise_eval_minus_clean",
        "elapsed_total_s",
    ]
    summary_columns_for_image = [
        col for col in summary_columns_for_image if col in summary_df.columns
    ]
    save_table_bundle(
        summary_df,
        output_dir=tables_dir,
        stem="noise_shots_layer_sweep_summary",
        title="Shots + backend-noise QCBM layer sweep summary",
        image_columns=summary_columns_for_image,
    )
    save_table_bundle(
        transpile_df,
        output_dir=tables_dir,
        stem="noise_shots_layer_sweep_transpile_table",
        title="Backend-transpiled noisy-training circuit resources by layer",
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
        "kl_train_best_observed": summary_df["kl_train_best_observed"].to_numpy(
            dtype=float
        ),
        "kl_shots_noise_mean": summary_df["kl_shots_noise_mean"].to_numpy(
            dtype=float
        ),
        "kl_shots_noise_std": summary_df["kl_shots_noise_std"].to_numpy(
            dtype=float
        ),
        "kl_clean_statevector": summary_df["kl_clean_statevector"].to_numpy(
            dtype=float
        ),
        "ansatz_depth": summary_df["ansatz_depth"].to_numpy(dtype=int),
        "ansatz_two_qubit_ops": summary_df["ansatz_two_qubit_ops"].to_numpy(
            dtype=int
        ),
        "ansatz_one_qubit_ops": summary_df["ansatz_one_qubit_ops"].to_numpy(
            dtype=int
        ),
        "p_target": ptg,
    }
    np.savez(
        output_dir / "noise_shots_layer_sweep_aggregate.npz",
        **aggregate_payload,
    )

    if not args.skip_plots:
        generate_noise_plots(
            summary_df,
            artifacts,
            output_dir=output_dir,
            n_time=n_time,
        )

    print("\n=== Shots + backend-noise QCBM layer comparison complete ===")
    print(f"Results: {output_dir}")
    print("\nFinal KL summary:")
    final_print_columns = [
        "n_layers",
        "n_params",
        "ansatz_depth",
        "ansatz_two_qubit_ops",
        "kl_train_best_observed",
        "kl_final",
        "kl_shots_noise_std",
        "kl_clean_statevector",
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
