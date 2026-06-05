
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

BENCHMARK_FILE = (
    REPO_ROOT
    / "data"
    / "multi_asset"
    / "6q_instance"
    / "benchmark"
    / "three_asset_instance.npz"
)


@dataclass(frozen=True)
class RegimeConfig:
    regime: str
    results_dir: Path
    qcbm_file: Path
    crca_files: dict[str, Path]
    qcbm_hat_key: str
    crca_hat_keys: dict[str, list[str]]
    ae_algorithm_filter: tuple[str, ...] | None


@dataclass
class LoadedInputs:
    regime: str
    results_dir: Path
    benchmark_file: Path
    qcbm_file: Path
    crca_files: dict[str, Path]
    cva_mc_hat: float
    cva_mc_std_err: float | None
    p_target: np.ndarray
    p_hat: np.ndarray
    v_target: np.ndarray
    v_hat: np.ndarray
    p_func_target: np.ndarray
    p_func_hat: np.ndarray
    q_func_target: np.ndarray
    q_func_hat: np.ndarray
    c_scale: float
    num_time_steps: int
    ae_algorithm_filter: tuple[str, ...] | None


def load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def scalar(data: dict[str, Any], key: str, source: Path) -> float:
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {source}")
    arr = np.asarray(data[key])
    if arr.size != 1:
        raise ValueError(f"Expected scalar key '{key}' in {source}, got shape {arr.shape}")
    return float(arr.reshape(()))


def require_array(data: dict[str, Any], key: str, source: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {source}")
    return np.asarray(data[key], dtype=float)


def first_available_array(
    data: dict[str, Any], keys: list[str], source: Path, label: str
) -> tuple[np.ndarray, str]:
    for key in keys:
        if key in data:
            return np.asarray(data[key], dtype=float), key
    raise KeyError(f"Missing {label}. Tried keys {keys} in {source}")


def normalize_distribution(values: np.ndarray, label: str, source: Path) -> np.ndarray:
    dist = np.asarray(values, dtype=float).ravel()
    total = float(dist.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} in {source} has invalid sum {total}")
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        print(f"WARNING: normalizing {label} from {source}; sum was {total:.16g}")
        dist = dist / total
    if np.any(dist < -1e-14):
        min_value = float(dist.min())
        raise ValueError(f"{label} in {source} contains negative values; min={min_value}")
    return np.clip(dist, 0.0, None)


def benchmark_targets(benchmark: dict[str, Any], source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_joint = require_array(benchmark, "v_joint_t", source).ravel()
    p_t = require_array(benchmark, "p_t", source).ravel()
    q_t = require_array(benchmark, "q_t", source).ravel()
    c_v = scalar(benchmark, "C_v", source)
    c_p = scalar(benchmark, "C_p", source)
    c_q = scalar(benchmark, "C_q", source)
    return v_joint / c_v, p_t / c_p, q_t / c_q


def report_target_mismatch(
    label: str, benchmark_target: np.ndarray, artifact_target: np.ndarray, source: Path
) -> None:
    benchmark_target = np.asarray(benchmark_target, dtype=float).ravel()
    artifact_target = np.asarray(artifact_target, dtype=float).ravel()
    if benchmark_target.shape != artifact_target.shape:
        print(
            f"WARNING: {label} target shape mismatch. "
            f"benchmark={benchmark_target.shape}, artifact={artifact_target.shape}, file={source}"
        )
        return
    max_abs = float(np.max(np.abs(benchmark_target - artifact_target)))
    if max_abs > 1e-10:
        print(
            f"WARNING: {label} artifact f_target differs from benchmark-scaled target; "
            f"max_abs_diff={max_abs:.6g}, file={source}"
        )


def load_inputs(config: RegimeConfig) -> LoadedInputs:
    benchmark = load_npz(BENCHMARK_FILE)
    qcbm = load_npz(config.qcbm_file)
    crca = {name: load_npz(path) for name, path in config.crca_files.items()}

    if "cva_mc_continuous" in benchmark:
        cva_mc_hat = scalar(benchmark, "cva_mc_continuous", BENCHMARK_FILE)
    elif "cva_by_grid_size_values" in benchmark:
        values = require_array(benchmark, "cva_by_grid_size_values", BENCHMARK_FILE).ravel()
        cva_mc_hat = float(values[-1])
    else:
        raise KeyError(
            f"Missing CVA MC benchmark in {BENCHMARK_FILE}. "
            "Tried 'cva_mc_continuous' and 'cva_by_grid_size_values'."
        )

    cva_mc_std_err = None
    if "cva_std_err_mc_continuous" in benchmark:
        cva_mc_std_err = scalar(benchmark, "cva_std_err_mc_continuous", BENCHMARK_FILE)

    p_target = normalize_distribution(
        require_array(qcbm, "p_target", config.qcbm_file),
        "P_target",
        config.qcbm_file,
    )
    p_hat = normalize_distribution(
        require_array(qcbm, config.qcbm_hat_key, config.qcbm_file),
        "P_hat",
        config.qcbm_file,
    )
    if p_target.shape != p_hat.shape:
        raise ValueError(
            f"P_target and P_hat shape mismatch in {config.qcbm_file}: "
            f"{p_target.shape} vs {p_hat.shape}"
        )

    v_target, p_func_target, q_func_target = benchmark_targets(benchmark, BENCHMARK_FILE)

    v_hat, v_hat_key = first_available_array(
        crca["v"], config.crca_hat_keys["v"], config.crca_files["v"], "v_hat"
    )
    p_func_hat, p_hat_key = first_available_array(
        crca["p"], config.crca_hat_keys["p"], config.crca_files["p"], "p_hat"
    )
    q_func_hat, q_hat_key = first_available_array(
        crca["q"], config.crca_hat_keys["q"], config.crca_files["q"], "q_hat"
    )

    v_hat = np.asarray(v_hat, dtype=float).ravel()
    p_func_hat = np.asarray(p_func_hat, dtype=float).ravel()
    q_func_hat = np.asarray(q_func_hat, dtype=float).ravel()

    report_target_mismatch(
        "v", v_target, require_array(crca["v"], "f_target", config.crca_files["v"]), config.crca_files["v"]
    )
    report_target_mismatch(
        "p",
        p_func_target,
        require_array(crca["p"], "f_target", config.crca_files["p"]),
        config.crca_files["p"],
    )
    report_target_mismatch(
        "q",
        q_func_target,
        require_array(crca["q"], "f_target", config.crca_files["q"]),
        config.crca_files["q"],
    )

    num_time_steps = int(len(p_func_target))
    if len(q_func_target) != num_time_steps:
        raise ValueError(
            f"p_target and q_target time lengths differ: {len(p_func_target)} vs {len(q_func_target)}"
        )
    if len(p_target) % num_time_steps != 0:
        raise ValueError(
            f"P_target length {len(p_target)} is not divisible by M={num_time_steps}"
        )
    if len(v_target) != len(p_target) or len(v_hat) != len(p_target):
        raise ValueError(
            "Exposure arrays must match the QCBM grid length. "
            f"len(P_target)={len(p_target)}, len(v_target)={len(v_target)}, len(v_hat)={len(v_hat)}"
        )
    if len(p_func_hat) != num_time_steps or len(q_func_hat) != num_time_steps:
        raise ValueError(
            "Temporal rotation arrays must match M. "
            f"M={num_time_steps}, len(p_hat)={len(p_func_hat)}, len(q_hat)={len(q_func_hat)}"
        )

    for label, arr in [
        ("v_target", v_target),
        (f"v_hat[{v_hat_key}]", v_hat),
        ("p_target", p_func_target),
        (f"p_hat[{p_hat_key}]", p_func_hat),
        ("q_target", q_func_target),
        (f"q_hat[{q_hat_key}]", q_func_hat),
    ]:
        arr = np.asarray(arr, dtype=float)
        if np.any(arr < -1e-12) or np.any(arr > 1.0 + 1e-12):
            print(
                f"WARNING: {label} has values outside [0, 1]; "
                f"min={float(np.min(arr)):.6g}, max={float(np.max(arr)):.6g}"
            )

    c_v = scalar(benchmark, "C_v", BENCHMARK_FILE)
    c_p = scalar(benchmark, "C_p", BENCHMARK_FILE)
    c_q = scalar(benchmark, "C_q", BENCHMARK_FILE)
    recovery = scalar(benchmark, "R_cva", BENCHMARK_FILE)
    c_scale = num_time_steps * (1.0 - recovery) * c_v * c_p * c_q

    return LoadedInputs(
        regime=config.regime,
        results_dir=resolve_results_dir(config.results_dir),
        benchmark_file=BENCHMARK_FILE,
        qcbm_file=config.qcbm_file,
        crca_files=config.crca_files,
        cva_mc_hat=cva_mc_hat,
        cva_mc_std_err=cva_mc_std_err,
        p_target=p_target,
        p_hat=p_hat,
        v_target=v_target,
        v_hat=v_hat,
        p_func_target=p_func_target,
        p_func_hat=p_func_hat,
        q_func_target=q_func_target,
        q_func_hat=q_func_hat,
        c_scale=c_scale,
        num_time_steps=num_time_steps,
        ae_algorithm_filter=config.ae_algorithm_filter,
    )


def resolve_results_dir(results_dir: Path) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    result_file_names = {
        "direct_final_rows.csv",
        "replay_final_rows.csv",
        "direct_budget_rows.csv",
        "replay_budget_rows.csv",
        "budget_summary.csv",
    }
    if any((results_dir / name).exists() for name in result_file_names):
        return results_dir
    experiment_results = results_dir / "experiment_results"
    if experiment_results.exists() and any(
        (experiment_results / name).exists() for name in result_file_names
    ):
        return experiment_results

    candidates = []
    for child in results_dir.rglob("*"):
        if child.is_dir() and any((child / name).exists() for name in result_file_names):
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No AE result CSV files found under {results_dir}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    print(f"Multiple result directories found under {results_dir}; using latest: {candidates[0]}")
    return candidates[0]


def check_absolute_continuity(
    p_target: np.ndarray, p_hat: np.ndarray
) -> dict[str, float | int | bool]:
    support = p_target > 0.0
    violations = support & (p_hat == 0.0)
    min_p_hat = float(np.min(p_hat[support])) if np.any(support) else float("nan")
    return {
        "ok": bool(not np.any(violations)),
        "num_support_violations": int(np.count_nonzero(violations)),
        "target_mass_on_violations": float(np.sum(p_target[violations])),
        "min_P_hat_on_target_support": min_p_hat,
    }


def kl_divergence(p_target: np.ndarray, p_hat: np.ndarray) -> float:
    continuity = check_absolute_continuity(p_target, p_hat)
    if not continuity["ok"]:
        return float("inf")
    support = p_target > 0.0
    return float(np.sum(p_target[support] * np.log(p_target[support] / p_hat[support])))


def reshape_time_grid(values: np.ndarray, num_time_steps: int, label: str) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    if len(flat) % num_time_steps != 0:
        raise ValueError(f"{label} length {len(flat)} is not divisible by M={num_time_steps}")
    return flat.reshape(num_time_steps, -1)


def compute_cva_delta(inputs: LoadedInputs) -> tuple[float, float]:
    p_target_grid = reshape_time_grid(inputs.p_target, inputs.num_time_steps, "P_target")
    v_target_grid = reshape_time_grid(inputs.v_target, inputs.num_time_steps, "v_target")
    amplitude_delta = float(
        np.sum(
            p_target_grid
            * v_target_grid
            * inputs.p_func_target[:, None]
            * inputs.q_func_target[:, None]
        )
    )
    return inputs.c_scale * amplitude_delta, amplitude_delta


def compute_weighted_rotation_errors(inputs: LoadedInputs) -> dict[str, float]:
    p_hat_grid = reshape_time_grid(inputs.p_hat, inputs.num_time_steps, "P_hat")
    v_target_grid = reshape_time_grid(inputs.v_target, inputs.num_time_steps, "v_target")
    v_hat_grid = reshape_time_grid(inputs.v_hat, inputs.num_time_steps, "v_hat")
    p_hat_time = np.sum(p_hat_grid, axis=1)

    delta_v = math.sqrt(float(np.sum(p_hat_grid * (v_hat_grid - v_target_grid) ** 2)))
    delta_p = math.sqrt(float(np.sum(p_hat_time * (inputs.p_func_hat - inputs.p_func_target) ** 2)))
    delta_q = math.sqrt(float(np.sum(p_hat_time * (inputs.q_func_hat - inputs.q_func_target) ** 2)))
    return {
        "delta_v": delta_v,
        "delta_p": delta_p,
        "delta_q": delta_q,
        "rotation_weighting": "P_hat for v; temporal marginal P_hat_T for p and q",
    }


def compute_artifact_oracle_cva(inputs: LoadedInputs) -> tuple[float, float]:
    p_hat_grid = reshape_time_grid(inputs.p_hat, inputs.num_time_steps, "P_hat")
    v_hat_grid = reshape_time_grid(inputs.v_hat, inputs.num_time_steps, "v_hat")
    amplitude = float(
        np.sum(
            p_hat_grid
            * v_hat_grid
            * inputs.p_func_hat[:, None]
            * inputs.q_func_hat[:, None]
        )
    )
    return inputs.c_scale * amplitude, amplitude


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_estimate_columns(df: pd.DataFrame, c_scale: float) -> pd.DataFrame:
    df = df.copy()

    cva_candidates = [
        "processed_estimate",
        "cva_estimate",
        "final_processed_estimate",
        "estimate_cva",
    ]
    for column in cva_candidates:
        if column in df.columns:
            df["_CVA_AE_hat"] = finite_series(df, column)
            df["_a_AE_hat"] = df["_CVA_AE_hat"] / c_scale
            df["_CVA_AE_source"] = column
            break
    else:
        amplitude_candidates = ["final_estimate", "estimate", "mean", "median"]
        for column in amplitude_candidates:
            if column in df.columns:
                df["_a_AE_hat"] = finite_series(df, column)
                df["_CVA_AE_hat"] = df["_a_AE_hat"] * c_scale
                df["_CVA_AE_source"] = column
                break

    if "_CVA_AE_hat" not in df.columns:
        return df

    if "processed_true_value" in df.columns:
        df["_CVA_SV_hat"] = finite_series(df, "processed_true_value")
        df["_a_SV_hat"] = df["_CVA_SV_hat"] / c_scale
        df["_CVA_SV_source"] = "processed_true_value"
    elif "cva_true" in df.columns:
        df["_CVA_SV_hat"] = finite_series(df, "cva_true")
        df["_a_SV_hat"] = df["_CVA_SV_hat"] / c_scale
        df["_CVA_SV_source"] = "cva_true"
    elif "a_true" in df.columns:
        df["_a_SV_hat"] = finite_series(df, "a_true")
        df["_CVA_SV_hat"] = df["_a_SV_hat"] * c_scale
        df["_CVA_SV_source"] = "a_true"

    return df


def candidate_result_files(results_dir: Path) -> list[Path]:
    names = [
        "replay_final_rows.csv",
        "direct_final_rows.csv",
        "final_rows.csv",
        "budget_summary.csv",
        "replay_budget_rows.csv",
        "direct_budget_rows.csv",
    ]
    files = [results_dir / name for name in names if (results_dir / name).exists()]
    if not files:
        raise FileNotFoundError(f"No AE result CSV files found in {results_dir}")
    return files


def select_best_ae_result(
    results_dir: Path,
    cva_mc_hat: float,
    c_scale: float,
    algorithm_filter: tuple[str, ...] | None,
) -> dict[str, Any]:
    rows = []
    used_files: list[Path] = []
    for csv_file in candidate_result_files(results_dir):
        if csv_file.stat().st_size <= 1:
            continue
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df = add_estimate_columns(df, c_scale)
        if "_CVA_AE_hat" not in df.columns:
            continue

        if "algorithm_key" in df.columns:
            algo = df["algorithm_key"].astype(str)
        elif "algorithm" in df.columns:
            algo = df["algorithm"].astype(str)
        else:
            algo = pd.Series(["unknown"] * len(df), index=df.index)
        df["_selected_AE_algorithm"] = algo

        if algorithm_filter is not None:
            pattern = "|".join(algorithm_filter)
            df = df[algo.str.lower().str.contains(pattern, regex=True, na=False)]
            if df.empty:
                continue

        df["_source_file"] = str(csv_file)
        df["_abs_error_vs_mc"] = (df["_CVA_AE_hat"] - cva_mc_hat).abs()
        df = df[np.isfinite(df["_abs_error_vs_mc"])]
        if df.empty:
            continue
        rows.append(df)
        used_files.append(csv_file)

        if csv_file.name.endswith("final_rows.csv") or csv_file.name == "budget_summary.csv":
            break

    if not rows:
        tried = ", ".join(str(path) for path in candidate_result_files(results_dir))
        raise ValueError(
            "Could not find AE estimates with usable CVA/amplitude columns. "
            f"Tried: {tried}"
        )

    all_rows = pd.concat(rows, ignore_index=True)
    idx = all_rows["_abs_error_vs_mc"].idxmin()
    selected = all_rows.loc[idx].to_dict()
    source_file = Path(str(selected["_source_file"]))
    print(f"Using AE result file for {results_dir}: {source_file}")

    return {
        "selected_AE_algorithm": str(selected["_selected_AE_algorithm"]),
        "CVA_AE_hat": float(selected["_CVA_AE_hat"]),
        "a_AE_hat": float(selected["_a_AE_hat"]),
        "CVA_SV_hat": (
            float(selected["_CVA_SV_hat"])
            if "_CVA_SV_hat" in selected and pd.notna(selected["_CVA_SV_hat"])
            else None
        ),
        "a_SV_hat": (
            float(selected["_a_SV_hat"])
            if "_a_SV_hat" in selected and pd.notna(selected["_a_SV_hat"])
            else None
        ),
        "source_file": source_file,
    }


def compute_bound(inputs: LoadedInputs) -> dict[str, Any]:
    continuity = check_absolute_continuity(inputs.p_target, inputs.p_hat)
    k_qcbm = kl_divergence(inputs.p_target, inputs.p_hat)
    cva_delta, _ = compute_cva_delta(inputs)
    rotations = compute_weighted_rotation_errors(inputs)
    artifact_cva_sv, artifact_a_sv = compute_artifact_oracle_cva(inputs)

    ae = select_best_ae_result(
        inputs.results_dir,
        inputs.cva_mc_hat,
        inputs.c_scale,
        algorithm_filter=inputs.ae_algorithm_filter,
    )
    cva_sv_hat = ae["CVA_SV_hat"] if ae["CVA_SV_hat"] is not None else artifact_cva_sv
    a_sv_hat = ae["a_SV_hat"] if ae["a_SV_hat"] is not None else artifact_a_sv

    error_observed = abs(inputs.cva_mc_hat - ae["CVA_AE_hat"])
    b_mc_delta = abs(inputs.cva_mc_hat - cva_delta)

    if math.isinf(k_qcbm):
        sqrt_2_k = float("inf")
        b_qcbm = float("inf")
        bound_total = float("inf")
        bound_check = "INVALID_ABSOLUTE_CONTINUITY"
    else:
        sqrt_2_k = math.sqrt(2.0 * k_qcbm)
        b_qcbm = inputs.c_scale * sqrt_2_k

    b_rot = inputs.c_scale * (
        rotations["delta_v"] + rotations["delta_p"] + rotations["delta_q"]
    )
    delta_ae = abs(ae["a_AE_hat"] - a_sv_hat)
    b_ae = inputs.c_scale * delta_ae

    if not math.isinf(k_qcbm):
        bound_total = b_mc_delta + b_qcbm + b_rot + b_ae
        bound_check = "PASS" if error_observed <= bound_total + 1e-12 else "FAIL"

    return {
        "regime": inputs.regime,
        "selected_AE_algorithm": ae["selected_AE_algorithm"],
        "CVA_MC_hat": inputs.cva_mc_hat,
        "CVA_Delta": cva_delta,
        "CVA_SV_hat": cva_sv_hat,
        "CVA_AE_hat": ae["CVA_AE_hat"],
        "Error_observed": error_observed,
        "C_scale": inputs.c_scale,
        "K_QCBM": k_qcbm,
        "sqrt_2_K_QCBM": sqrt_2_k,
        "delta_v": rotations["delta_v"],
        "delta_p": rotations["delta_p"],
        "delta_q": rotations["delta_q"],
        "Delta_AE": delta_ae,
        "B_MC_Delta": b_mc_delta,
        "B_QCBM": b_qcbm,
        "B_rot": b_rot,
        "B_AE": b_ae,
        "Bound_total": bound_total,
        "bound_check": bound_check,
        "absolute_continuity_check": "PASS" if continuity["ok"] else "FAIL",
        "num_support_violations": continuity["num_support_violations"],
        "target_mass_on_violations": continuity["target_mass_on_violations"],
        "min_P_hat_on_target_support": continuity["min_P_hat_on_target_support"],
        "_ae_source_file": str(ae["source_file"]),
        "_rotation_weighting": rotations["rotation_weighting"],
        "_CVA_MC_std_err": inputs.cva_mc_std_err,
        "_artifact_CVA_SV_hat": artifact_cva_sv,
    }


def print_files_used(inputs_list: list[LoadedInputs], rows: list[dict[str, Any]]) -> None:
    print("\nFiles used:")
    for inputs, row in zip(inputs_list, rows):
        print(f"- regime={inputs.regime}")
        print(f"  benchmark: {inputs.benchmark_file}")
        if inputs.cva_mc_std_err is not None:
            print(f"  CVA_MC_std_err: {inputs.cva_mc_std_err:.12g}")
        print(f"  qcbm: {inputs.qcbm_file}")
        print(f"  crca_v: {inputs.crca_files['v']}")
        print(f"  crca_p: {inputs.crca_files['p']}")
        print(f"  crca_q: {inputs.crca_files['q']}")
        print(f"  ae: {row['_ae_source_file']}")
        print(f"  rotation weighting: {row['_rotation_weighting']}")
