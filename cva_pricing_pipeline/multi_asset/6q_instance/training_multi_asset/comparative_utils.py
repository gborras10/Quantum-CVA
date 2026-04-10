from __future__ import annotations

import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy import stats

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)


BACKEND_NAME = "ibm_basquecountry"
EPS = 1e-12
BASE_SEED = 1234
DEFAULT_SHOTS_GRID = (2048, 8192)
DEFAULT_REPETITIONS = 2
DEFAULT_BOOTSTRAP_REPS = 2000
SCENARIO_LABELS = {
    "statevector": "Statevector (ideal)",
    "shots_ideal": "Finite shots (noiseless)",
    "shots_noise": "Finite shots + backend noise model",
}
SCENARIO_COLORS = {
    "statevector": "black",
    "shots_ideal": "tab:blue",
    "shots_noise": "tab:red",
}


@dataclass(frozen=True)
class QcbmCheckpoint:
    npz_path: pathlib.Path
    name: str
    n_layers: int
    topology: str
    theta_star: np.ndarray
    p_target: np.ndarray
    n_qubits: int
    entangler: str


@dataclass
class EvaluationBundle:
    qcbm: MLQcbmCircuit
    backend_label: str
    simulation_method: str
    depth: float
    size: float
    width: float
    two_qubit_gates: float
    swap_count: float


def as_1d_float(array_like: np.ndarray) -> np.ndarray:
    return np.asarray(array_like, dtype=float).ravel()


def npz_int(npz_data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in npz_data:
        return int(default)
    return int(np.asarray(npz_data[key]).item())


def npz_str(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz_data:
        return str(default)
    return str(np.asarray(npz_data[key]).item())


def metadata_dict(npz_data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in npz_data:
        return {}
    maybe_dict = npz_data["metadata"]
    if hasattr(maybe_dict, "item"):
        maybe_dict = maybe_dict.item()
    return maybe_dict if isinstance(maybe_dict, dict) else {}


def normalize_for_kl(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Empty vector cannot be normalized for KL divergence.")
    arr = np.maximum(arr, eps)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("Vector has non-finite or non-positive sum.")
    return arr / s


def kl_divergence(target: np.ndarray, predicted: np.ndarray, eps: float = EPS) -> float:
    p = normalize_for_kl(target, eps=eps)
    q = normalize_for_kl(predicted, eps=eps)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch for KL: {p.shape} vs {q.shape}.")
    return float(np.sum(p * np.log(p / q)))


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def fit_mean_asymptote(shots: np.ndarray, means: np.ndarray) -> tuple[float, float, float]:
    x = 1.0 / np.asarray(shots, dtype=float)
    y = np.asarray(means, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return float(beta[0]), float(beta[1]), float(r2)


def fit_std_inverse_sqrt(shots: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
    x = 1.0 / np.sqrt(np.asarray(shots, dtype=float))
    y = np.asarray(stds, dtype=float)
    coef = float(np.dot(x, y) / np.dot(x, x))
    y_hat = coef * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return coef, r2


def repo_root_from_script(script_path: pathlib.Path) -> pathlib.Path:
    return next(
        parent
        for parent in script_path.resolve().parents
        if (parent / "pyproject.toml").exists()
    )


def parse_layers_from_name(path: pathlib.Path) -> int | None:
    match = re.search(r"_(\d+)lay(?:\.|$)", path.name)
    if match:
        return int(match.group(1))
    return None


def discover_qcbm_checkpoints(checkpoint_dir: pathlib.Path) -> list[QcbmCheckpoint]:
    npz_files = sorted(checkpoint_dir.glob("training_qcbm_*.npz"))
    checkpoints: list[QcbmCheckpoint] = []
    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        meta = metadata_dict(data)
        theta_star = as_1d_float(data["theta_star"])
        p_target = as_1d_float(data["p_target"])
        n_layers = int(meta.get("n_layers", npz_int(data, "n_layers", parse_layers_from_name(path) or 0)))
        topology = str(meta.get("requested_topology", npz_str(data, "requested_topology", "qcbm_heavyhex6")))
        n_qubits = int(meta.get("n_qubits", npz_int(data, "n_qubits", int(round(math.log2(p_target.size))))))
        entangler = str(meta.get("entangler", npz_str(data, "entangler", "rzz")))
        checkpoints.append(
            QcbmCheckpoint(
                npz_path=path,
                name=path.stem,
                n_layers=n_layers,
                topology=topology,
                theta_star=theta_star,
                p_target=p_target,
                n_qubits=n_qubits,
                entangler=entangler,
            )
        )
    if not checkpoints:
        raise FileNotFoundError(f"No training_qcbm_*.npz files found in {checkpoint_dir}")
    checkpoints.sort(key=lambda x: (x.n_layers, x.name))
    return checkpoints


def _extract_qcbm_circuit(qcbm: MLQcbmCircuit):
    candidate_names = [
        "_tqc_meas",
        "_qc_meas",
        "_tqc",
        "_qc",
        "circuit_meas",
        "circuit",
    ]
    for name in candidate_names:
        if hasattr(qcbm, name):
            candidate = getattr(qcbm, name)
            if candidate is not None and hasattr(candidate, "depth"):
                return candidate
    return None


def summarize_qiskit_circuit(circuit) -> dict[str, float]:
    if circuit is None:
        return {
            "depth": math.nan,
            "size": math.nan,
            "width": math.nan,
            "two_qubit_gates": math.nan,
            "swap_count": math.nan,
        }
    count_ops = circuit.count_ops()
    two_qubit_gates = 0
    for gate_name, count in count_ops.items():
        if gate_name in {"cx", "cz", "ecr", "swap", "rxx", "ryy", "rzz", "crx", "cry", "crz", "cp", "cu", "cu1", "cu3"}:
            two_qubit_gates += int(count)
    return {
        "depth": float(circuit.depth()),
        "size": float(circuit.size()),
        "width": float(circuit.num_qubits),
        "two_qubit_gates": float(two_qubit_gates),
        "swap_count": float(count_ops.get("swap", 0)),
    }


def build_qcbm_bundle(
    checkpoint: QcbmCheckpoint,
    *,
    backend,
    simulation_method: str,
    noise_model=None,
    seed_transpiler: int = BASE_SEED,
) -> EvaluationBundle:
    qcbm = MLQcbmCircuit(
        n_qubits=checkpoint.n_qubits,
        n_layers=checkpoint.n_layers,
        name=f"qcbm_{checkpoint.n_layers}lay",
        entangler=checkpoint.entangler,
        topology=checkpoint.topology,
        backend=backend,
        noise_model=noise_model,
        simulation_method=simulation_method,
        optimization_level=1,
        seed_transpiler=seed_transpiler,
    )
    if checkpoint.theta_star.size != qcbm.n_params:
        raise ValueError(
            f"Checkpoint {checkpoint.npz_path.name}: theta_star has size {checkpoint.theta_star.size}, "
            f"but circuit expects {qcbm.n_params}."
        )
    circuit = _extract_qcbm_circuit(qcbm)
    metrics = summarize_qiskit_circuit(circuit)
    return EvaluationBundle(
        qcbm=qcbm,
        backend_label="none" if backend is None else getattr(backend, "name", str(backend)),
        simulation_method=simulation_method,
        depth=metrics["depth"],
        size=metrics["size"],
        width=metrics["width"],
        two_qubit_gates=metrics["two_qubit_gates"],
        swap_count=metrics["swap_count"],
    )


def evaluate_checkpoint_kl(
    bundle: EvaluationBundle,
    checkpoint: QcbmCheckpoint,
    *,
    shots: int | None,
    seed: int,
) -> float:
    pred = bundle.qcbm.probabilities(checkpoint.theta_star, shots=shots, seed=seed)
    return kl_divergence(checkpoint.p_target, pred)


def collect_qcbm_family_results(
    checkpoints: list[QcbmCheckpoint],
    *,
    shots_grid: list[int],
    repetitions: int,
    backend_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    ideal_shots_backend = AerSimulator(method="automatic", seed_simulator=BASE_SEED)

    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(backend_name, use_fractional_gates=True)
    noise_model = NoiseModel.from_backend(real_backend, thermal_relaxation=True)
    noisy_backend = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        seed_simulator=BASE_SEED,
    )

    for checkpoint in checkpoints:
        print(f"\n[INFO] Evaluating checkpoint: {checkpoint.name} | layers={checkpoint.n_layers}")
        sv_bundle = build_qcbm_bundle(
            checkpoint,
            backend=None,
            simulation_method="statevector",
            noise_model=None,
        )
        shots_bundle = build_qcbm_bundle(
            checkpoint,
            backend=ideal_shots_backend,
            simulation_method="automatic",
            noise_model=None,
        )
        noisy_bundle = build_qcbm_bundle(
            checkpoint,
            backend=noisy_backend,
            simulation_method="density_matrix",
            noise_model=noise_model,
        )

        inventory_rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "checkpoint_file": checkpoint.npz_path.name,
                "n_layers": checkpoint.n_layers,
                "topology": checkpoint.topology,
                "n_qubits": checkpoint.n_qubits,
                "entangler": checkpoint.entangler,
                "logical_or_transpiled_depth_noiseless": shots_bundle.depth,
                "logical_or_transpiled_size_noiseless": shots_bundle.size,
                "logical_or_transpiled_width_noiseless": shots_bundle.width,
                "two_qubit_gates_noiseless": shots_bundle.two_qubit_gates,
                "swap_count_noiseless": shots_bundle.swap_count,
                "logical_or_transpiled_depth_noisy": noisy_bundle.depth,
                "logical_or_transpiled_size_noisy": noisy_bundle.size,
                "logical_or_transpiled_width_noisy": noisy_bundle.width,
                "two_qubit_gates_noisy": noisy_bundle.two_qubit_gates,
                "swap_count_noisy": noisy_bundle.swap_count,
            }
        )

        kl_sv = evaluate_checkpoint_kl(sv_bundle, checkpoint, shots=None, seed=BASE_SEED)
        rows.append(
            {
                "checkpoint_name": checkpoint.name,
                "checkpoint_file": checkpoint.npz_path.name,
                "n_layers": checkpoint.n_layers,
                "topology": checkpoint.topology,
                "scenario": "statevector",
                "scenario_label": SCENARIO_LABELS["statevector"],
                "shots": math.nan,
                "repetition": 0,
                "seed": BASE_SEED,
                "kl": kl_sv,
            }
        )

        for shots in shots_grid:
            print(f"[INFO]   shots={shots}")
            for repetition in range(repetitions):
                print(f"[INFO]     repetition {repetition+1}/{repetitions}", end="\r")
                seed = BASE_SEED + 10_000 * checkpoint.n_layers + 100 * shots + repetition
                kl_shots = evaluate_checkpoint_kl(shots_bundle, checkpoint, shots=shots, seed=seed)
                kl_noise = evaluate_checkpoint_kl(noisy_bundle, checkpoint, shots=shots, seed=seed)
                rows.append(
                    {
                        "checkpoint_name": checkpoint.name,
                        "checkpoint_file": checkpoint.npz_path.name,
                        "n_layers": checkpoint.n_layers,
                        "topology": checkpoint.topology,
                        "scenario": "shots_ideal",
                        "scenario_label": SCENARIO_LABELS["shots_ideal"],
                        "shots": int(shots),
                        "repetition": repetition,
                        "seed": seed,
                        "kl": kl_shots,
                    }
                )
                rows.append(
                    {
                        "checkpoint_name": checkpoint.name,
                        "checkpoint_file": checkpoint.npz_path.name,
                        "n_layers": checkpoint.n_layers,
                        "topology": checkpoint.topology,
                        "scenario": "shots_noise",
                        "scenario_label": SCENARIO_LABELS["shots_noise"],
                        "shots": int(shots),
                        "repetition": repetition,
                        "seed": seed,
                        "kl": kl_noise,
                    }
                )

    raw_df = pd.DataFrame(rows)
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["n_layers", "checkpoint_name"]).reset_index(drop=True)
    backend_info = {
        "backend_name": backend_name,
        "noise_model_basis_gates": ",".join(noise_model.basis_gates),
        "real_backend_name": getattr(real_backend, "name", backend_name),
        "shots_grid": ",".join(str(x) for x in shots_grid),
        "repetitions": repetitions,
    }
    return raw_df, inventory_df, backend_info


def make_summary_tables(
    raw_df: pd.DataFrame,
    *,
    shots_grid: list[int],
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    statevector_ref = (
        raw_df[raw_df["scenario"] == "statevector"][
            ["checkpoint_name", "checkpoint_file", "n_layers", "kl"]
        ]
        .rename(columns={"kl": "kl_statevector_exact"})
        .copy()
    )

    noisy_df = raw_df[raw_df["scenario"] != "statevector"].copy()
    summary_rows: list[dict[str, Any]] = []
    for (checkpoint_name, checkpoint_file, n_layers, scenario, shots), grp in noisy_df.groupby(
        ["checkpoint_name", "checkpoint_file", "n_layers", "scenario", "shots"], sort=True
    ):
        vals = grp["kl"].to_numpy(dtype=float)
        ci_lo, ci_hi = bootstrap_mean_ci(
            vals,
            n_resamples=bootstrap_reps,
            seed=BASE_SEED + int(n_layers) + int(shots),
        )
        summary_rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_file": checkpoint_file,
                "n_layers": n_layers,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "shots": int(shots),
                "n_repetitions": int(vals.size),
                "kl_mean": float(np.mean(vals)),
                "kl_median": float(np.median(vals)),
                "kl_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "kl_min": float(np.min(vals)),
                "kl_max": float(np.max(vals)),
                "kl_ci95_lo": ci_lo,
                "kl_ci95_hi": ci_hi,
            }
        )
    summary_df = pd.DataFrame(summary_rows).merge(
        statevector_ref,
        on=["checkpoint_name", "checkpoint_file", "n_layers"],
        how="left",
    )
    summary_df["excess_over_ideal"] = summary_df["kl_mean"] - summary_df["kl_statevector_exact"]
    summary_df["ratio_to_ideal"] = summary_df["kl_mean"] / summary_df["kl_statevector_exact"]
    summary_df = summary_df.sort_values(["n_layers", "scenario", "shots"]).reset_index(drop=True)

    fit_rows: list[dict[str, Any]] = []
    for (checkpoint_name, checkpoint_file, n_layers, scenario), grp in summary_df.groupby(
        ["checkpoint_name", "checkpoint_file", "n_layers", "scenario"], sort=True
    ):
        grp = grp.sort_values("shots")
        asymptote, c_over_n, r2_mean = fit_mean_asymptote(grp["shots"].to_numpy(), grp["kl_mean"].to_numpy())
        std_coef, r2_std = fit_std_inverse_sqrt(grp["shots"].to_numpy(), grp["kl_std"].to_numpy())
        kl_exact = float(grp["kl_statevector_exact"].iloc[0])
        fit_rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_file": checkpoint_file,
                "n_layers": n_layers,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "fit_asymptote": asymptote,
                "fit_c_over_n": c_over_n,
                "fit_r2_mean": r2_mean,
                "fit_std_coef": std_coef,
                "fit_r2_std": r2_std,
                "kl_statevector_exact": kl_exact,
                "fit_asymptote_over_ideal": asymptote / kl_exact if kl_exact > 0.0 else math.nan,
                "fit_excess_over_ideal": asymptote - kl_exact,
            }
        )
    fits_df = pd.DataFrame(fit_rows).sort_values(["n_layers", "scenario"]).reset_index(drop=True)

    max_shots = max(shots_grid)
    paired_rows: list[dict[str, Any]] = []
    for (checkpoint_name, checkpoint_file, n_layers), grp in noisy_df[noisy_df["shots"] == max_shots].groupby(
        ["checkpoint_name", "checkpoint_file", "n_layers"], sort=True
    ):
        ideal_vals = grp[grp["scenario"] == "shots_ideal"].sort_values("repetition")["kl"].to_numpy(dtype=float)
        noise_vals = grp[grp["scenario"] == "shots_noise"].sort_values("repetition")["kl"].to_numpy(dtype=float)
        if ideal_vals.size == 0 or noise_vals.size == 0:
            continue
        if ideal_vals.size != noise_vals.size:
            raise ValueError("Paired comparison requires the same number of repetitions for ideal and noisy cases.")
        try:
            wilcoxon_res = stats.wilcoxon(noise_vals, ideal_vals, alternative="greater", zero_method="wilcox")
            pvalue = float(wilcoxon_res.pvalue)
            statistic = float(wilcoxon_res.statistic)
        except ValueError:
            pvalue = math.nan
            statistic = math.nan
        paired_rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_file": checkpoint_file,
                "n_layers": n_layers,
                "shots": max_shots,
                "mean_kl_shots_ideal": float(np.mean(ideal_vals)),
                "mean_kl_shots_noise": float(np.mean(noise_vals)),
                "median_kl_shots_ideal": float(np.median(ideal_vals)),
                "median_kl_shots_noise": float(np.median(noise_vals)),
                "mean_difference_noise_minus_ideal": float(np.mean(noise_vals - ideal_vals)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue_greater": pvalue,
            }
        )
    paired_df = pd.DataFrame(paired_rows).sort_values(["n_layers", "checkpoint_name"]).reset_index(drop=True)

    max_summary_df = summary_df[summary_df["shots"] == max_shots].copy().sort_values(["n_layers", "scenario"])
    return summary_df, fits_df, paired_df, max_summary_df


def _setup_layer_axis(ax, layers: np.ndarray) -> None:
    ax.set_xticks(layers)
    ax.set_xlabel("QCBM layers")
    ax.grid(alpha=0.25, linestyle="--")


def plot_kl_vs_layers(max_summary_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    statevector_line = (
        max_summary_df[["n_layers", "kl_statevector_exact"]]
        .drop_duplicates()
        .sort_values("n_layers")
    )
    layers = statevector_line["n_layers"].to_numpy(dtype=int)
    ax.plot(
        layers,
        statevector_line["kl_statevector_exact"].to_numpy(dtype=float),
        marker="o",
        linewidth=2.0,
        color=SCENARIO_COLORS["statevector"],
        label=SCENARIO_LABELS["statevector"],
    )
    for scenario in ["shots_ideal", "shots_noise"]:
        grp = max_summary_df[max_summary_df["scenario"] == scenario].sort_values("n_layers")
        ax.errorbar(
            grp["n_layers"].to_numpy(dtype=int),
            grp["kl_mean"].to_numpy(dtype=float),
            yerr=[
                grp["kl_mean"].to_numpy(dtype=float) - grp["kl_ci95_lo"].to_numpy(dtype=float),
                grp["kl_ci95_hi"].to_numpy(dtype=float) - grp["kl_mean"].to_numpy(dtype=float),
            ],
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    _setup_layer_axis(ax, layers)
    ax.set_ylabel("KL(target || prediction)")
    ax.set_title("QCBM family: KL at maximum shot budget")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_excess_over_ideal(max_summary_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    layers = np.sort(max_summary_df["n_layers"].unique())
    for scenario in ["shots_ideal", "shots_noise"]:
        grp = max_summary_df[max_summary_df["scenario"] == scenario].sort_values("n_layers")
        ax.plot(
            grp["n_layers"].to_numpy(dtype=int),
            grp["excess_over_ideal"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            color=SCENARIO_COLORS[scenario],
            label=scenario,
        )
    _setup_layer_axis(ax, layers)
    ax.set_ylabel("Mean KL - exact statevector KL")
    ax.set_title("Excess over ideal at maximum shot budget")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_noise_floor_ratio(fits_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    grp = fits_df[fits_df["scenario"] == "shots_noise"].sort_values("n_layers")
    layers = grp["n_layers"].to_numpy(dtype=int)
    ax.plot(
        layers,
        grp["fit_asymptote_over_ideal"].to_numpy(dtype=float),
        marker="o",
        linewidth=2.0,
        color="tab:red",
    )
    _setup_layer_axis(ax, layers)
    ax.set_ylabel("Fitted noisy asymptote / exact ideal KL")
    ax.set_title("Estimated noisy floor relative to ideal KL")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_scatter_ideal_vs_noisy(max_summary_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    statevector_ref = (
        max_summary_df[["checkpoint_name", "n_layers", "kl_statevector_exact"]]
        .drop_duplicates()
        .sort_values("n_layers")
    )
    noisy = max_summary_df[max_summary_df["scenario"] == "shots_noise"][
        ["checkpoint_name", "n_layers", "kl_mean"]
    ].rename(columns={"kl_mean": "kl_noise_mean"})
    merged = statevector_ref.merge(noisy, on=["checkpoint_name", "n_layers"], how="inner")
    ax.scatter(
        merged["kl_statevector_exact"].to_numpy(dtype=float),
        merged["kl_noise_mean"].to_numpy(dtype=float),
        s=60,
        color="tab:red",
    )
    for _, row in merged.iterrows():
        ax.annotate(f"{int(row['n_layers'])}L", (row["kl_statevector_exact"], row["kl_noise_mean"]), xytext=(5, 3), textcoords="offset points")
    lo = min(merged["kl_statevector_exact"].min(), merged["kl_noise_mean"].min())
    hi = max(merged["kl_statevector_exact"].max(), merged["kl_noise_mean"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Exact statevector KL")
    ax.set_ylabel("Noisy mean KL at max shots")
    ax.set_title("Ideal vs noisy performance across QCBM family")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_kl_vs_shots_by_layer(summary_df: pd.DataFrame, fits_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    layers = np.sort(summary_df["n_layers"].unique())
    n_layers = len(layers)
    ncols = min(3, n_layers)
    nrows = int(math.ceil(n_layers / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.3 * ncols, 4.1 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, layer in zip(axes_flat, layers):
        layer_df = summary_df[summary_df["n_layers"] == layer]
        exact = float(layer_df["kl_statevector_exact"].iloc[0])
        ax.axhline(exact, color="black", linestyle="-", linewidth=1.8, label="statevector")
        for scenario in ["shots_ideal", "shots_noise"]:
            grp = layer_df[layer_df["scenario"] == scenario].sort_values("shots")
            ax.errorbar(
                grp["shots"].to_numpy(dtype=int),
                grp["kl_mean"].to_numpy(dtype=float),
                yerr=grp["kl_std"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=SCENARIO_COLORS[scenario],
                label=scenario,
            )
            fit_row = fits_df[(fits_df["n_layers"] == layer) & (fits_df["scenario"] == scenario)]
            if not fit_row.empty:
                asym = float(fit_row["fit_asymptote"].iloc[0])
                c = float(fit_row["fit_c_over_n"].iloc[0])
                x = grp["shots"].to_numpy(dtype=float)
                ax.plot(x, asym + c / x, linestyle="--", color=SCENARIO_COLORS[scenario], alpha=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{int(layer)} layers")
        ax.set_xlabel("Shots")
        ax.set_ylabel("KL")
        ax.grid(alpha=0.25, linestyle="--")
    for ax in axes_flat[n_layers:]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.suptitle("QCBM family: KL vs shots", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_depth_and_2q(inventory_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    layers = inventory_df["n_layers"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(layers, inventory_df["logical_or_transpiled_depth_noisy"].to_numpy(dtype=float), marker="o", linewidth=2.0)
    _setup_layer_axis(axes[0], np.sort(np.unique(layers)))
    axes[0].set_ylabel("Depth")
    axes[0].set_title("Transpiled depth vs layers (noisy backend)")

    axes[1].plot(layers, inventory_df["two_qubit_gates_noisy"].to_numpy(dtype=float), marker="o", linewidth=2.0)
    _setup_layer_axis(axes[1], np.sort(np.unique(layers)))
    axes[1].set_ylabel("Two-qubit gates")
    axes[1].set_title("Two-qubit gates vs layers (noisy backend)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
