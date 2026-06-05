from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService


def _bootstrap_repo() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(parent for parent in current.parents if (parent / "pyproject.toml").exists())
    src_path = repo_root / "src"
    sv_cva_path = current.parents[1]
    for path in (src_path, sv_cva_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _bootstrap_repo()

from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (  # noqa: E402
    QuantumCVACircuit,
)
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (  # noqa: E402
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (  # noqa: E402
    MLQcbmCircuit,
)


DATA_ROOT = REPO_ROOT / "data" / "multi_asset" / "6q_instance"
TRAINING_ROOT = DATA_ROOT / "quantum" / "training"
BENCHMARK_PATH = DATA_ROOT / "benchmark" / "three_asset_instance.npz"

OUTPUT_ROOT = pathlib.Path(__file__).resolve().parent / "paper_results"

QCBM_IDEAL_PATH = TRAINING_ROOT / "qcbm" / "statevector" / "training_qcbm_heavyhex6_6lay.npz"
QCBM_NOISE_PATH = (
    TRAINING_ROOT
    / "qcbm"
    / "shots"
    / "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz"
)

DEFAULT_IDEAL_PATH = TRAINING_ROOT / "crca" / "default_probabilities" / "training_crca2.npz"
DEFAULT_NOISE_PATH = (
    TRAINING_ROOT
    / "crca"
    / "default_probabilities"
    / "training_crca2_shots_backend_noise_snapshot.npz"
)
DISCOUNT_IDEAL_PATH = TRAINING_ROOT / "crca" / "discount_factors" / "training_crca2.npz"
DISCOUNT_NOISE_PATH = (
    TRAINING_ROOT
    / "crca"
    / "discount_factors"
    / "training_crca2_shots_backend_noise_snapshot.npz"
)
EXPOSURE_IDEAL_PATH = (
    TRAINING_ROOT / "crca" / "positive_exposure" / "training_heavy_hex_star.npz"
)
EXPOSURE_NOISE_PATH = (
    TRAINING_ROOT
    / "crca"
    / "positive_exposure"
    / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
)

EXPOSURE_IDEAL_NOISE_TABLE = (
    REPO_ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "training_multi_asset"
    / "functional_encoding"
    / "statevector"
    / "positive_exposure"
    / "layers_comparison"
    / "results"
    / "statevector"
    / "tables"
    / "positive_exposure_statevector_vs_backend_noise_eval.csv"
)

EPS = 1e-12


@dataclass(frozen=True)
class RegimeBundle:
    label: str
    training_regime: str
    qcbm: np.lib.npyio.NpzFile
    default_probabilities: np.lib.npyio.NpzFile
    discount_factors: np.lib.npyio.NpzFile
    positive_exposure: np.lib.npyio.NpzFile


def _load_npz(path: pathlib.Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return np.load(path, allow_pickle=True)


def _as_1d_float(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _scalar(npz: np.lib.npyio.NpzFile, key: str, default: float = math.nan) -> float:
    if key not in npz:
        return float(default)
    arr = np.asarray(npz[key])
    if arr.size != 1:
        return float(default)
    return float(arr.item())


def _int_scalar(npz: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    value = _scalar(npz, key, float(default))
    if not np.isfinite(value):
        return int(default)
    return int(value)


def _str_scalar(npz: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz:
        return str(default)
    return str(np.asarray(npz[key]).item())


def _metadata(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in npz:
        return {}
    obj = npz["metadata"]
    if hasattr(obj, "item"):
        obj = obj.item()
    return obj if isinstance(obj, dict) else {}


def _mean_squared_error(target: np.ndarray, prediction: np.ndarray) -> float:
    target = _as_1d_float(target)
    prediction = _as_1d_float(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f"Shape mismatch for MSE: {target.shape} vs {prediction.shape}.")
    diff = prediction - target
    return float(np.mean(diff * diff))


def _l2_error(target: np.ndarray, prediction: np.ndarray) -> float:
    target = _as_1d_float(target)
    prediction = _as_1d_float(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f"Shape mismatch for L2: {target.shape} vs {prediction.shape}.")
    return float(np.linalg.norm(prediction - target, ord=2))


def _normalize_distribution(values: np.ndarray, eps: float = EPS) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = np.maximum(arr, eps)
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Cannot normalize distribution.")
    return arr / total


def _kl_divergence(target: np.ndarray, prediction: np.ndarray, eps: float = EPS) -> float:
    p = _normalize_distribution(target, eps=eps)
    q = _normalize_distribution(prediction, eps=eps)
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch for KL: {p.shape} vs {q.shape}.")
    return float(np.sum(p * np.log(p / q)))


def _relative_error(estimate: float, reference: float) -> float:
    if not np.isfinite(reference) or abs(reference) <= 0.0:
        return math.nan
    return float(abs(float(estimate) - float(reference)) / abs(float(reference)))


def _format_float(x: Any) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return ""
    if xf == 0.0:
        return "0"
    if abs(xf) < 1e-3 or abs(xf) >= 1e4:
        return f"{xf:.3e}"
    return f"{xf:.6f}"


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "patch.linewidth": 0.7,
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


def _markdown_escape(value: Any) -> str:
    if isinstance(value, float):
        text = _format_float(value)
    else:
        text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _write_markdown_table(df: pd.DataFrame, path: pathlib.Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_tables(df: pd.DataFrame, output_dir: pathlib.Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    _write_markdown_table(df, output_dir / f"{stem}.md")
    df.to_latex(
        output_dir / f"{stem}.tex",
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.6g}",
    )


def load_regimes() -> tuple[RegimeBundle, RegimeBundle]:
    ideal = RegimeBundle(
        label="Ideal-trained",
        training_regime="ideal_statevector",
        qcbm=_load_npz(QCBM_IDEAL_PATH),
        default_probabilities=_load_npz(DEFAULT_IDEAL_PATH),
        discount_factors=_load_npz(DISCOUNT_IDEAL_PATH),
        positive_exposure=_load_npz(EXPOSURE_IDEAL_PATH),
    )
    noisy = RegimeBundle(
        label="Noise-trained",
        training_regime="shots_backend_noise",
        qcbm=_load_npz(QCBM_NOISE_PATH),
        default_probabilities=_load_npz(DEFAULT_NOISE_PATH),
        discount_factors=_load_npz(DISCOUNT_NOISE_PATH),
        positive_exposure=_load_npz(EXPOSURE_NOISE_PATH),
    )
    return ideal, noisy


def qcbm_subblock_rows(ideal: RegimeBundle, noisy: RegimeBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = _as_1d_float(ideal.qcbm["p_target"])
    ideal_eval_ideal = _kl_divergence(target, _as_1d_float(ideal.qcbm["p_star"]))

    rows.append(
        {
            "component": "QCBM",
            "metric": "KL(target || p_theta)",
            "metric_family": "KL",
            "training_regime": "ideal_statevector",
            "evaluation_regime": "ideal_statevector",
            "value": ideal_eval_ideal,
            "std": math.nan,
            "shots": math.nan,
            "repetitions": math.nan,
            "source": str(QCBM_IDEAL_PATH.relative_to(REPO_ROOT)),
        }
    )
    rows.append(
        {
            "component": "QCBM",
            "metric": "KL(target || p_theta)",
            "metric_family": "KL",
            "training_regime": "ideal_statevector",
            "evaluation_regime": "shots_backend_noise",
            "value": _scalar(noisy.qcbm, "kl_eval_mean_statevector"),
            "std": _scalar(noisy.qcbm, "kl_eval_std_statevector"),
            "shots": _scalar(noisy.qcbm, "kl_eval_shots"),
            "repetitions": _scalar(noisy.qcbm, "kl_eval_runs"),
            "source": str(QCBM_NOISE_PATH.relative_to(REPO_ROOT)),
        }
    )
    rows.append(
        {
            "component": "QCBM",
            "metric": "KL(target || p_theta)",
            "metric_family": "KL",
            "training_regime": "shots_backend_noise",
            "evaluation_regime": "shots_backend_noise",
            "value": _kl_divergence(_as_1d_float(noisy.qcbm["p_target"]), _as_1d_float(noisy.qcbm["p_star"])),
            "std": math.nan,
            "shots": _scalar(noisy.qcbm, "shots"),
            "repetitions": math.nan,
            "source": str(QCBM_NOISE_PATH.relative_to(REPO_ROOT)),
        }
    )
    return rows


def scalar_crca_rows(
    *,
    component: str,
    ideal_npz: np.lib.npyio.NpzFile,
    noisy_npz: np.lib.npyio.NpzFile,
    ideal_path: pathlib.Path,
    noisy_path: pathlib.Path,
) -> list[dict[str, Any]]:
    target = _as_1d_float(ideal_npz["f_target"])
    rows: list[dict[str, Any]] = [
        {
            "component": component,
            "metric": "MSE(f_target, f_theta)",
            "metric_family": "MSE",
            "training_regime": "ideal_statevector",
            "evaluation_regime": "ideal_statevector",
            "value": _mean_squared_error(target, ideal_npz["f_star_statevector"]),
            "std": math.nan,
            "shots": math.nan,
            "repetitions": math.nan,
            "source": str(ideal_path.relative_to(REPO_ROOT)),
        },
        {
            "component": component,
            "metric": "MSE(f_target, f_theta)",
            "metric_family": "MSE",
            "training_regime": "ideal_statevector",
            "evaluation_regime": "shots_backend_noise",
            "value": _scalar(noisy_npz, "baseline_l2_statevector_under_noise"),
            "std": math.nan,
            "shots": _scalar(noisy_npz, "shots"),
            "repetitions": math.nan,
            "source": str(noisy_path.relative_to(REPO_ROOT)),
        },
        {
            "component": component,
            "metric": "MSE(f_target, f_theta)",
            "metric_family": "MSE",
            "training_regime": "shots_backend_noise",
            "evaluation_regime": "shots_backend_noise",
            "value": _mean_squared_error(noisy_npz["f_target"], noisy_npz["f_star_shots"]),
            "std": math.nan,
            "shots": _scalar(noisy_npz, "shots"),
            "repetitions": math.nan,
            "source": str(noisy_path.relative_to(REPO_ROOT)),
        },
    ]
    return rows


def positive_exposure_rows(ideal: RegimeBundle, noisy: RegimeBundle) -> list[dict[str, Any]]:
    target = _as_1d_float(ideal.positive_exposure["f_target"])
    ideal_metadata = _metadata(ideal.positive_exposure)
    n_layers = int(ideal_metadata.get("n_layers", 2))

    ideal_noise_value = math.nan
    ideal_noise_std = math.nan
    ideal_noise_shots = math.nan
    ideal_noise_runs = math.nan
    source = str(EXPOSURE_IDEAL_NOISE_TABLE.relative_to(REPO_ROOT))
    if EXPOSURE_IDEAL_NOISE_TABLE.exists():
        noise_df = pd.read_csv(EXPOSURE_IDEAL_NOISE_TABLE)
        layer_rows = noise_df[noise_df["n_layers"].astype(int) == int(n_layers)]
        if not layer_rows.empty:
            row = layer_rows.iloc[0]
            ideal_noise_value = float(row["mse_shots_noise_mean"])
            ideal_noise_std = float(row["mse_shots_noise_std"])
            ideal_noise_shots = float(row["noisy_eval_shots"])
            ideal_noise_runs = float(row["noisy_eval_runs"])

    rows = [
        {
            "component": "CRCA positive exposure",
            "metric": "MSE(f_target, f_theta)",
            "metric_family": "MSE",
            "training_regime": "ideal_statevector",
            "evaluation_regime": "ideal_statevector",
            "value": _mean_squared_error(target, ideal.positive_exposure["f_star_statevector"]),
            "std": math.nan,
            "shots": math.nan,
            "repetitions": math.nan,
            "source": str(EXPOSURE_IDEAL_PATH.relative_to(REPO_ROOT)),
        },
        {
            "component": "CRCA positive exposure",
            "metric": "MSE(f_target, f_theta)",
            "metric_family": "MSE",
            "training_regime": "ideal_statevector",
            "evaluation_regime": "shots_backend_noise",
            "value": ideal_noise_value,
            "std": ideal_noise_std,
            "shots": ideal_noise_shots,
            "repetitions": ideal_noise_runs,
            "source": source,
        },
        {
            "component": "CRCA positive exposure",
            "metric": "MSE(f_target, f_theta)",
            "metric_family": "MSE",
            "training_regime": "shots_backend_noise",
            "evaluation_regime": "shots_backend_noise",
            "value": _scalar(noisy.positive_exposure, "best_l2_rechecked", _scalar(noisy.positive_exposure, "best_l2")),
            "std": math.nan,
            "shots": _scalar(noisy.positive_exposure, "shots"),
            "repetitions": math.nan,
            "source": str(EXPOSURE_NOISE_PATH.relative_to(REPO_ROOT)),
        },
    ]
    return rows


def build_subblock_table(ideal: RegimeBundle, noisy: RegimeBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(qcbm_subblock_rows(ideal, noisy))
    rows.extend(
        scalar_crca_rows(
            component="CRCA default probabilities",
            ideal_npz=ideal.default_probabilities,
            noisy_npz=noisy.default_probabilities,
            ideal_path=DEFAULT_IDEAL_PATH,
            noisy_path=DEFAULT_NOISE_PATH,
        )
    )
    rows.extend(
        scalar_crca_rows(
            component="CRCA discount factors",
            ideal_npz=ideal.discount_factors,
            noisy_npz=noisy.discount_factors,
            ideal_path=DISCOUNT_IDEAL_PATH,
            noisy_path=DISCOUNT_NOISE_PATH,
        )
    )
    rows.extend(positive_exposure_rows(ideal, noisy))
    return pd.DataFrame(rows)


def _crca_layers(npz: np.lib.npyio.NpzFile, default: int) -> int:
    meta = _metadata(npz)
    if "n_layers" in meta:
        return int(meta["n_layers"])
    return _int_scalar(npz, "n_layers", default)


def build_cva_model(bundle: RegimeBundle, *, backend: str = "statevector") -> QuantumCVACircuit:
    benchmark = _load_npz(BENCHMARK_PATH)
    num_qubits_time = 2
    num_qubits_underlying = 4
    total_num_qubits = num_qubits_time + num_qubits_underlying

    qcbm = MLQcbmCircuit(
        n_qubits=total_num_qubits,
        n_layers=_int_scalar(bundle.qcbm, "n_layers", 6),
        name=f"qcbm_{bundle.training_regime}",
        entangler="rzz",
        topology=_str_scalar(bundle.qcbm, "effective_topology", _str_scalar(bundle.qcbm, "requested_topology", "qcbm_heavyhex6")),
        backend=AerSimulator(method="statevector"),
        simulation_method="statevector",
    )
    crca_positive = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=num_qubits_underlying,
        n_layers=_crca_layers(bundle.positive_exposure, 2),
        ansatz_type="heavy_hex_star",
        name=f"crca_positive_exposure_{bundle.training_regime}",
    )
    crca_default = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_crca_layers(bundle.default_probabilities, 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name=f"crca_default_probabilities_{bundle.training_regime}",
    )
    crca_discount = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_crca_layers(bundle.discount_factors, 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name=f"crca_discount_factors_{bundle.training_regime}",
    )
    return QuantumCVACircuit(
        num_qubits_time=num_qubits_time,
        num_qubits_underlying=num_qubits_underlying,
        qcbm_circuit=qcbm,
        crca_circuit_exposure=crca_positive,
        crca_circuit_default_prob=crca_default,
        crca_circuit_discount_factor=crca_discount,
        recovery_rate=float(benchmark["R_cva"]),
        C_v=float(benchmark["C_v"]),
        C_p=float(benchmark["C_p"]),
        C_q=float(benchmark["C_q"]),
        name=f"quantum_cva_{bundle.training_regime}",
        backend=backend,
    )


def exact_cva_rows(ideal: RegimeBundle, noisy: RegimeBundle) -> pd.DataFrame:
    benchmark = _load_npz(BENCHMARK_PATH)
    cva_reference = float(benchmark["cva_by_grid_size_values"][1])
    rows: list[dict[str, Any]] = []
    for bundle in (ideal, noisy):
        model = build_cva_model(bundle, backend="statevector")
        estimate = model.cva(
            qcbm_params=_as_1d_float(bundle.qcbm["theta_star"]),
            exposure_params=_as_1d_float(bundle.positive_exposure["theta_star"]),
            default_prob_params=_as_1d_float(bundle.default_probabilities["theta_star"]),
            discount_factor_params=_as_1d_float(bundle.discount_factors["theta_star"]),
        )
        rows.append(
            {
                "quantity": "CVA",
                "training_regime": bundle.training_regime,
                "evaluation_regime": "ideal_statevector",
                "cva_estimate": estimate,
                "cva_reference": cva_reference,
                "absolute_error": abs(estimate - cva_reference),
                "relative_error": _relative_error(estimate, cva_reference),
                "std": math.nan,
                "shots": math.nan,
                "repetitions": math.nan,
                "source": "statevector recomputation from training artifacts",
            }
        )
    return pd.DataFrame(rows)


def _snapshot_iso(noisy: RegimeBundle) -> str | None:
    for npz in (
        noisy.qcbm,
        noisy.default_probabilities,
        noisy.discount_factors,
        noisy.positive_exposure,
    ):
        if "noise_snapshot_iso_utc" in npz:
            return str(np.asarray(npz["noise_snapshot_iso_utc"]).item())
    return None


def _transpile_for_noisy_aer(
    circuit,
    *,
    noise_model: NoiseModel,
    coupling_map,
    optimization_level: int,
    seed_transpiler: int,
):
    """Transpile for local Aer noise simulation without using Aer's synthetic Target.

    Some Qiskit/Aer versions build an inconsistent average-error map when
    transpile() receives an AerSimulator constructed from a backend-derived
    noise model plus a backend coupling map. Supplying basis_gates/coupling_map
    explicitly preserves the same noisy execution assumptions while avoiding
    the backend Target path that triggers VF2 layout failures.
    """
    basis_gates = list(getattr(noise_model, "basis_gates", []) or [])
    return transpile(
        circuit,
        basis_gates=basis_gates or None,
        coupling_map=coupling_map,
        optimization_level=int(optimization_level),
        layout_method="sabre" if coupling_map is not None else None,
        routing_method="sabre" if coupling_map is not None else None,
        seed_transpiler=int(seed_transpiler),
    )


def run_noisy_cva_evaluation(
    ideal: RegimeBundle,
    noisy: RegimeBundle,
    *,
    shots: int,
    repetitions: int,
    seed_base: int,
    backend_name: str,
    runtime_channel: str,
    use_fractional_gates: bool,
    thermal_relaxation: bool,
) -> pd.DataFrame:
    service = QiskitRuntimeService(channel=runtime_channel)
    real_backend = service.backend(backend_name, use_fractional_gates=use_fractional_gates)
    snapshot_iso = _snapshot_iso(noisy)
    backend_props = None
    if snapshot_iso:
        from datetime import datetime, timezone

        snapshot_dt = datetime.fromisoformat(snapshot_iso)
        if snapshot_dt.tzinfo is None:
            snapshot_dt = snapshot_dt.replace(tzinfo=timezone.utc)
        backend_props = real_backend.properties(datetime=snapshot_dt)
    if backend_props is None:
        backend_props = real_backend.properties()

    noise_model = NoiseModel.from_backend_properties(
        backend_props,
        thermal_relaxation=bool(thermal_relaxation),
    )
    coupling_map = getattr(real_backend, "coupling_map", None)
    if coupling_map is None:
        coupling_map = real_backend.configuration().coupling_map

    simulator = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=coupling_map,
        seed_simulator=int(seed_base),
    )

    benchmark = _load_npz(BENCHMARK_PATH)
    cva_reference = float(benchmark["cva_by_grid_size_values"][1])
    rows: list[dict[str, Any]] = []

    for bundle in (ideal, noisy):
        model = build_cva_model(bundle, backend="statevector")
        measured = model.build_cva_circuit(
            qcbm_params=_as_1d_float(bundle.qcbm["theta_star"]),
            crca_exposure_params=_as_1d_float(bundle.positive_exposure["theta_star"]),
            crca_default_params=_as_1d_float(bundle.default_probabilities["theta_star"]),
            crca_discount_params=_as_1d_float(bundle.discount_factors["theta_star"]),
            measured=True,
        )
        transpiled = _transpile_for_noisy_aer(
            measured,
            noise_model=noise_model,
            coupling_map=coupling_map,
            optimization_level=3,
            seed_transpiler=int(seed_base),
        )
        cva_values: list[float] = []
        p111_values: list[float] = []
        for rep in range(int(repetitions)):
            seed = int(seed_base) + rep
            counts = simulator.run(
                transpiled,
                shots=int(shots),
                seed_simulator=seed,
            ).result().get_counts()
            p111 = model._prob_111_from_counts(counts)
            p111_values.append(float(p111))
            cva_values.append(model.cva_from_prob(p111))

        arr = np.asarray(cva_values, dtype=float)
        rows.append(
            {
                "quantity": "CVA",
                "training_regime": bundle.training_regime,
                "evaluation_regime": "shots_backend_noise",
                "cva_estimate": float(np.mean(arr)),
                "cva_reference": cva_reference,
                "absolute_error": abs(float(np.mean(arr)) - cva_reference),
                "relative_error": _relative_error(float(np.mean(arr)), cva_reference),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "shots": int(shots),
                "repetitions": int(repetitions),
                "p111_mean": float(np.mean(p111_values)),
                "p111_std": float(np.std(p111_values, ddof=1)) if len(p111_values) > 1 else 0.0,
                "backend_name": backend_name,
                "noise_snapshot_iso_utc": snapshot_iso,
                "thermal_relaxation": bool(thermal_relaxation),
                "source": "Aer density-matrix simulation of full measured CVA circuit",
            }
        )
    return pd.DataFrame(rows)


def build_comparison_summary(subblocks: pd.DataFrame, cva: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, group in subblocks.groupby("component", sort=False):
        ideal_noise = group[
            (group["training_regime"] == "ideal_statevector")
            & (group["evaluation_regime"] == "shots_backend_noise")
        ]
        noisy_noise = group[
            (group["training_regime"] == "shots_backend_noise")
            & (group["evaluation_regime"] == "shots_backend_noise")
        ]
        if ideal_noise.empty or noisy_noise.empty:
            continue
        ideal_value = float(ideal_noise["value"].iloc[0])
        noisy_value = float(noisy_noise["value"].iloc[0])
        rows.append(
            {
                "quantity": component,
                "metric": str(group["metric"].iloc[0]),
                "ideal_train_noise_eval": ideal_value,
                "noise_train_noise_eval": noisy_value,
                "absolute_delta_noise_minus_ideal": noisy_value - ideal_value,
                "ratio_noise_train_over_ideal_train": noisy_value / ideal_value if ideal_value > 0 else math.nan,
            }
        )

    cva_noise = cva[cva["evaluation_regime"] == "shots_backend_noise"]
    if not cva_noise.empty:
        ideal_cva = cva_noise[cva_noise["training_regime"] == "ideal_statevector"]
        noisy_cva = cva_noise[cva_noise["training_regime"] == "shots_backend_noise"]
        if not ideal_cva.empty and not noisy_cva.empty:
            ideal_err = float(ideal_cva["relative_error"].iloc[0])
            noisy_err = float(noisy_cva["relative_error"].iloc[0])
            rows.append(
                {
                    "quantity": "CVA",
                    "metric": "relative error vs classical CVA",
                    "ideal_train_noise_eval": ideal_err,
                    "noise_train_noise_eval": noisy_err,
                    "absolute_delta_noise_minus_ideal": noisy_err - ideal_err,
                    "ratio_noise_train_over_ideal_train": noisy_err / ideal_err if ideal_err > 0 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_subblock_noise_comparison(subblocks: pd.DataFrame, output_dir: pathlib.Path) -> None:
    df = subblocks[subblocks["evaluation_regime"] == "shots_backend_noise"].copy()
    df["label"] = df["training_regime"].map(
        {
            "ideal_statevector": "Train ideal",
            "shots_backend_noise": "Train noise",
        }
    )
    component_order = [
        "QCBM",
        "CRCA default probabilities",
        "CRCA discount factors",
        "CRCA positive exposure",
    ]
    colors = {"Train ideal": "#0072B2", "Train noise": "#D55E00"}

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), constrained_layout=True)
    for ax, family, title, ylabel in [
        (axes[0], "KL", "QCBM distribution fit under noise", "KL"),
        (axes[1], "MSE", "CRCA function fits under noise", "MSE"),
    ]:
        sub = df[df["metric_family"] == family].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        components = [c for c in component_order if c in set(sub["component"])]
        x = np.arange(len(components), dtype=float)
        width = 0.34
        for offset, label in [(-width / 2, "Train ideal"), (width / 2, "Train noise")]:
            vals = []
            errs = []
            for component in components:
                row = sub[(sub["component"] == component) & (sub["label"] == label)]
                vals.append(float(row["value"].iloc[0]) if not row.empty else math.nan)
                errs.append(float(row["std"].iloc[0]) if not row.empty and np.isfinite(row["std"].iloc[0]) else 0.0)
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[label],
                edgecolor="#222222",
                linewidth=0.45,
                label=label,
                yerr=errs if any(err > 0 for err in errs) else None,
                capsize=2.0,
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                c.replace("CRCA ", "").replace(" probabilities", "").replace(" factors", "")
                for c in components
            ],
            rotation=24,
            ha="right",
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    save_figure(fig, output_dir, "fig_subblock_noise_eval_comparison")


def plot_noise_penalty(summary: pd.DataFrame, output_dir: pathlib.Path) -> None:
    if summary.empty:
        return
    df = summary.copy()
    df["plot_label"] = (
        df["quantity"]
        .str.replace("CRCA ", "", regex=False)
        .str.replace(" probabilities", "", regex=False)
        .str.replace(" factors", "", regex=False)
        .str.replace("positive exposure", "positive exp.", regex=False)
    )
    fig, ax = plt.subplots(figsize=(6.1, 3.0), constrained_layout=True)
    x = np.arange(len(df))
    ratios = df["ratio_noise_train_over_ideal_train"].to_numpy(dtype=float)
    colors = np.where(ratios <= 1.0, "#009E73", "#D55E00")
    ax.bar(x, ratios, color=colors, edgecolor="#222222", linewidth=0.45)
    ax.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(df["plot_label"], rotation=25, ha="right")
    ax.set_ylabel("Noise-trained / ideal-trained")
    ax.set_title("Effect of training objective when evaluation includes noise")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    save_figure(fig, output_dir, "fig_noise_training_ratio")


def plot_cva_comparison(cva_df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    if cva_df.empty:
        return
    df = cva_df.copy()
    df["training_label"] = df["training_regime"].map(
        {
            "ideal_statevector": "Train ideal",
            "shots_backend_noise": "Train noise",
        }
    )
    df["eval_label"] = df["evaluation_regime"].map(
        {
            "ideal_statevector": "Exact statevector",
            "shots_backend_noise": "Backend-noise shots",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85), constrained_layout=True)
    colors = {"Train ideal": "#0072B2", "Train noise": "#D55E00"}
    for ax, y_col, title, ylabel in [
        (axes[0], "cva_estimate", "CVA estimate", "CVA"),
        (axes[1], "relative_error", "CVA relative error", "Relative error"),
    ]:
        evals = list(df["eval_label"].dropna().unique())
        train_labels = ["Train ideal", "Train noise"]
        x = np.arange(len(evals), dtype=float)
        width = 0.34
        for offset, train_label in [(-width / 2, "Train ideal"), (width / 2, "Train noise")]:
            vals = []
            errs = []
            for eval_label in evals:
                row = df[(df["training_label"] == train_label) & (df["eval_label"] == eval_label)]
                vals.append(float(row[y_col].iloc[0]) if not row.empty else math.nan)
                errs.append(float(row["std"].iloc[0]) if y_col == "cva_estimate" and not row.empty and np.isfinite(row["std"].iloc[0]) else 0.0)
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[train_label],
                edgecolor="#222222",
                linewidth=0.45,
                label=train_label,
                yerr=errs if any(err > 0 for err in errs) else None,
                capsize=2.0,
            )
        if y_col == "cva_estimate" and "cva_reference" in df:
            ax.axhline(float(df["cva_reference"].iloc[0]), color="#333333", linestyle="--", linewidth=0.9, label="Classical")
        if y_col == "relative_error":
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(evals, rotation=12, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    save_figure(fig, output_dir, "fig_cva_training_regime_comparison")


def write_manifest(
    *,
    output_dir: pathlib.Path,
    subblocks: pd.DataFrame,
    cva: pd.DataFrame,
    run_noisy_cva: bool,
    args: argparse.Namespace,
) -> None:
    cli_args = {
        key: str(value) if isinstance(value, pathlib.Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "description": (
            "Paper-ready comparison between parameters trained in ideal statevector "
            "and parameters trained with shots plus backend noise."
        ),
        "important_semantics": {
            "QCBM_metric": "KL(target || predicted)",
            "CRCA_metric": "mean squared error over encoded function values",
            "CVA_exact": "statevector recomputation of the aggregate CVA circuit",
            "CVA_noisy": (
                "Aer density-matrix simulation of measured aggregate CVA circuit"
                if run_noisy_cva
                else "not run; pass --run-noisy-cva to generate"
            ),
        },
        "generated_tables": [
            "tables/subblock_training_regime_metrics.csv",
            "tables/cva_training_regime_metrics.csv",
            "tables/noise_eval_comparison_summary.csv",
        ],
        "generated_figures": [
            "figures/fig_subblock_noise_eval_comparison.{png,pdf,svg}",
            "figures/fig_noise_training_ratio.{png,pdf,svg}",
            "figures/fig_cva_training_regime_comparison.{png,pdf,svg}",
        ],
        "source_artifacts": sorted(set(subblocks["source"].dropna().astype(str).tolist() + cva["source"].dropna().astype(str).tolist())),
        "cli_args": cli_args,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready tables and figures comparing ideal-trained vs "
            "noise-trained CVA pipeline components."
        )
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--run-noisy-cva",
        action="store_true",
        help="Also simulate the full measured CVA circuit with backend noise. This may be slow and requires IBM backend metadata access.",
    )
    parser.add_argument("--cva-shots", type=int, default=100_000)
    parser.add_argument("--cva-repetitions", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--backend-name", default="ibm_basquecountry")
    parser.add_argument("--runtime-channel", default="ibm_cloud")
    parser.add_argument("--use-fractional-gates", action="store_true", default=True)
    parser.add_argument(
        "--thermal-relaxation",
        action="store_true",
        help="Include thermal relaxation when constructing the optional full-CVA noise model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    ideal, noisy = load_regimes()

    subblocks = build_subblock_table(ideal, noisy)
    cva_df = exact_cva_rows(ideal, noisy)
    if args.run_noisy_cva:
        noisy_cva = run_noisy_cva_evaluation(
            ideal,
            noisy,
            shots=int(args.cva_shots),
            repetitions=int(args.cva_repetitions),
            seed_base=int(args.seed_base),
            backend_name=str(args.backend_name),
            runtime_channel=str(args.runtime_channel),
            use_fractional_gates=bool(args.use_fractional_gates),
            thermal_relaxation=bool(args.thermal_relaxation),
        )
        cva_df = pd.concat([cva_df, noisy_cva], ignore_index=True)

    summary = build_comparison_summary(subblocks, cva_df)

    save_tables(subblocks, table_dir, "subblock_training_regime_metrics")
    save_tables(cva_df, table_dir, "cva_training_regime_metrics")
    save_tables(summary, table_dir, "noise_eval_comparison_summary")

    plot_subblock_noise_comparison(subblocks, figure_dir)
    plot_noise_penalty(summary, figure_dir)
    plot_cva_comparison(cva_df, figure_dir)

    write_manifest(
        output_dir=output_dir,
        subblocks=subblocks,
        cva=cva_df,
        run_noisy_cva=bool(args.run_noisy_cva),
        args=args,
    )

    print(f"Saved paper-ready comparison outputs to: {output_dir}")
    print("\nNoise-evaluation summary:")
    if summary.empty:
        print("No paired noise-evaluation summary could be built.")
    else:
        print(summary.to_string(index=False, formatters={col: _format_float for col in summary.select_dtypes(include=[float]).columns}))
    if not args.run_noisy_cva:
        print("\nFull noisy CVA simulation was not run. Use --run-noisy-cva to add CVA under backend noise.")


if __name__ == "__main__":
    main()
