from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Path management
# --------------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
hardware_dir = os.path.abspath(os.path.join(current_dir, ".."))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", "..", ".."))
src_dir = os.path.join(root_dir, "src")

for path in [hardware_dir, src_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --------------------------------------------------------------------------------------
# Imports from project
# --------------------------------------------------------------------------------------
try:
    from realistic_utils import build_large_problem, ideal_good_probability
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Qiskit Runtime imports
# --------------------------------------------------------------------------------------
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
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

DENOM_THRESHOLD = 2.0e-2
MIN_C_FOR_LOG = 1.0e-6
MAX_C_FOR_LOG = 1.0
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_ALPHA = 0.05
RNG_SEED = 12345

CALIB_ID = time.strftime("%Y%m%d_%H%M%S")
CALIB_UUID = str(uuid.uuid4())

JOB_ID = "d7ism9tpv4fs739rm7qg"

SUMMARY_CSV = os.path.join(current_dir, f"t_eff_postjob_summary_{CALIB_ID}.csv")
POINTS_CSV = os.path.join(current_dir, f"t_eff_postjob_points_{CALIB_ID}.csv")
RAW_CSV = os.path.join(current_dir, f"t_eff_postjob_raw_{CALIB_ID}.csv")
CONFIG_JSON = os.path.join(current_dir, f"t_eff_postjob_config_{CALIB_ID}.json")


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


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """
    Weighted least squares for y = a + b x.
    Returns (a, b).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    sw = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x * x)
    sxy = np.sum(w * x * y)

    det = sw * sxx - sx * sx
    if not np.isfinite(det) or abs(det) < 1e-15:
        raise ValueError("Degenerate weighted fit.")

    a = (sxx * sy - sx * sxy) / det
    b = (sw * sxy - sx * sy) / det
    return float(a), float(b)


def find_counts_container(pub_result: Any) -> Any:
    """
    Robustly locate the classical register container in a SamplerV2 pub result.
    Your uploaded JSON shows the register is named 'c0', not 'meas'. :contentReference[oaicite:1]{index=1}
    """
    data = pub_result.data

    if hasattr(data, "meas"):
        return data.meas

    if hasattr(data, "c0"):
        return data.c0

    for name in dir(data):
        if name.startswith("_"):
            continue
        obj = getattr(data, name)
        if hasattr(obj, "get_counts"):
            return obj

    raise RuntimeError("Could not locate counts container in SamplerV2 result.")


def estimate_prob_from_counts(counts: dict[str, int], shots: int) -> float:
    """
    Estimate P(good) from counts.

    In your uploaded JSON each pub has num_bits = 1, so the keys should be '0' and '1'. :contentReference[oaicite:2]{index=2}
    """
    one = int(counts.get("1", 0))
    return float(one / shots)


def build_single_job_rows(
    job_id: str,
    probe_ks: list[int],
    shots: int,
    n_repeats_per_k: int,
) -> list[dict[str, str]]:
    """
    Reconstruct the pub ordering used in the single-job submission:
    for each k in PROBE_KS, reps 1..N_REPEATS_PER_K were appended in order.
    """
    rows: list[dict[str, str]] = []
    pub_index = 0

    for k in probe_ks:
        for rep in range(1, n_repeats_per_k + 1):
            rows.append(
                {
                    "job_id": str(job_id),
                    "pub_index": str(pub_index),
                    "k": str(k),
                    "K": str(2 * k + 1),
                    "rep": str(rep),
                    "shots": str(shots),
                }
            )
            pub_index += 1

    return rows


def build_job_map_by_k(job_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in job_rows:
        k = int(row["k"])
        grouped.setdefault(k, []).append(row)

    for k in grouped:
        grouped[k].sort(key=lambda r: int(r["rep"]))
    return grouped


@dataclass
class RuntimeJobResultFetcher:
    service: Any

    def fetch_result(self, job_id: str) -> Any:
        job = self.service.job(job_id)
        return job.result()

    def fetch_counts_by_pub_index(self, primitive_result: Any, pub_index: int) -> dict[str, int]:
        pub_result = primitive_result[pub_index]
        counts_container = find_counts_container(pub_result)
        return dict(counts_container.get_counts())


def robust_calibrate_t_eff_from_single_job(
    fetcher: RuntimeJobResultFetcher,
    problem: Any,
    job_id: str,
    probe_ks: list[int],
    shots: int,
    n_repeats_per_k: int,
    denom_threshold: float,
    bootstrap_samples: int,
    bootstrap_alpha: float,
    rng_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(rng_seed)

    job_rows = build_single_job_rows(
        job_id=job_id,
        probe_ks=probe_ks,
        shots=shots,
        n_repeats_per_k=n_repeats_per_k,
    )
    grouped = build_job_map_by_k(job_rows)

    primitive_result = fetcher.fetch_result(job_id)

    raw_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    x_fit = []
    y_fit = []
    w_fit = []

    bootstrap_payload: list[dict[str, Any]] = []

    for k in sorted(grouped.keys()):
        rows_k = grouped[k]
        K = 2 * k + 1

        p_ideal = float(ideal_good_probability(problem, k))
        denom = float(p_ideal - 0.5)

        p_noisy_reps: list[float] = []

        for row in rows_k:
            pub_index = int(row["pub_index"])
            rep = int(row["rep"])

            counts = fetcher.fetch_counts_by_pub_index(primitive_result, pub_index)
            p_noisy = estimate_prob_from_counts(counts, shots=shots)
            p_noisy_reps.append(p_noisy)

            numer = float(p_noisy - 0.5)
            if abs(denom) < denom_threshold:
                c_est = np.nan
            else:
                c_est = numer / denom

            if np.isfinite(c_est):
                c_est = float(np.clip(c_est, MIN_C_FOR_LOG, MAX_C_FOR_LOG))
            else:
                c_est = np.nan

            raw_rows.append(
                {
                    "calib_id": CALIB_ID,
                    "calib_uuid": CALIB_UUID,
                    "backend": BACKEND_NAME,
                    "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
                    "job_id": str(job_id),
                    "pub_index": int(pub_index),
                    "k": int(k),
                    "K": int(K),
                    "rep": int(rep),
                    "shots": int(shots),
                    "p_ideal": float(p_ideal),
                    "p_noisy": float(p_noisy),
                    "denom": float(denom),
                    "c_est": None if not np.isfinite(c_est) else float(c_est),
                }
            )

            print(
                f"Recovered job_id={job_id} | pub_index={pub_index:02d} | "
                f"k={k:2d} rep={rep:02d} p_noisy={p_noisy:.6f}"
            )

        p_mean = float(np.mean(p_noisy_reps))
        p_std = float(np.std(p_noisy_reps, ddof=1)) if len(p_noisy_reps) >= 2 else 0.0
        p_se = p_std / math.sqrt(len(p_noisy_reps)) if len(p_noisy_reps) >= 1 else np.nan

        numer_mean = float(p_mean - 0.5)
        if abs(denom) < denom_threshold:
            c_mean = np.nan
            log_c_mean = np.nan
            log_c_se = np.nan
            include_fit = False
        else:
            c_mean = numer_mean / denom
            if np.isfinite(c_mean):
                c_mean = float(np.clip(c_mean, MIN_C_FOR_LOG, MAX_C_FOR_LOG))
            else:
                c_mean = np.nan

            if np.isfinite(c_mean) and (c_mean > 0.0):
                log_c_mean = float(np.log(c_mean))
                if p_se > 0.0 and np.isfinite(c_mean):
                    log_c_se = float(p_se / (abs(denom) * c_mean))
                else:
                    log_c_se = np.nan

                include_fit = np.isfinite(log_c_se) and log_c_se > 0.0 and c_mean < 1.0
            else:
                log_c_mean = np.nan
                log_c_se = np.nan
                include_fit = False

        point_rows.append(
            {
                "calib_id": CALIB_ID,
                "calib_uuid": CALIB_UUID,
                "backend": BACKEND_NAME,
                "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
                "job_id": str(job_id),
                "k": int(k),
                "K": int(K),
                "shots_per_rep": int(shots),
                "n_repeats": int(len(rows_k)),
                "p_ideal": float(p_ideal),
                "denom": float(denom),
                "p_noisy_mean": float(p_mean),
                "p_noisy_std": float(p_std),
                "p_noisy_se": float(p_se),
                "c_mean": None if not np.isfinite(c_mean) else float(c_mean),
                "log_c_mean": None if not np.isfinite(log_c_mean) else float(log_c_mean),
                "log_c_se": None if not np.isfinite(log_c_se) else float(log_c_se),
                "used_in_fit": bool(include_fit),
            }
        )

        if include_fit:
            x_fit.append(float(K))
            y_fit.append(float(log_c_mean))
            w_fit.append(float(1.0 / (log_c_se**2)))

            bootstrap_payload.append(
                {
                    "k": int(k),
                    "K": int(K),
                    "p_ideal": float(p_ideal),
                    "denom": float(denom),
                    "p_noisy_reps": list(map(float, p_noisy_reps)),
                }
            )

    if len(x_fit) < 2:
        summary = {
            "calib_id": CALIB_ID,
            "calib_uuid": CALIB_UUID,
            "backend": BACKEND_NAME,
            "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
            "job_id": str(job_id),
            "n_fit_points": int(len(x_fit)),
            "t_eff": None,
            "t_eff_ci_low": None,
            "t_eff_ci_high": None,
            "slope": None,
            "intercept": None,
            "bootstrap_samples": int(bootstrap_samples),
            "fit_kind": "weighted_log_contrast",
            "status": "insufficient_fit_points",
        }
        return summary, point_rows, raw_rows

    x_fit_arr = np.asarray(x_fit, dtype=float)
    y_fit_arr = np.asarray(y_fit, dtype=float)
    w_fit_arr = np.asarray(w_fit, dtype=float)

    intercept, slope = weighted_linear_fit(x_fit_arr, y_fit_arr, w_fit_arr)
    t_eff = None if slope >= 0.0 else float(-1.0 / slope)

    t_boot = []
    for _ in range(bootstrap_samples):
        xb = []
        yb = []
        wb = []

        for payload in bootstrap_payload:
            reps = np.asarray(payload["p_noisy_reps"], dtype=float)
            boot = rng.choice(reps, size=len(reps), replace=True)

            p_mean_b = float(np.mean(boot))
            p_std_b = float(np.std(boot, ddof=1)) if len(boot) >= 2 else 0.0
            p_se_b = p_std_b / math.sqrt(len(boot)) if len(boot) >= 1 else np.nan

            denom = float(payload["denom"])
            numer_b = float(p_mean_b - 0.5)
            if abs(denom) < denom_threshold:
                continue

            c_b = numer_b / denom
            if not np.isfinite(c_b):
                continue
            c_b = float(np.clip(c_b, MIN_C_FOR_LOG, MAX_C_FOR_LOG))
            if c_b <= 0.0 or c_b >= 1.0:
                continue

            log_c_b = float(np.log(c_b))
            if not np.isfinite(p_se_b) or p_se_b <= 0.0:
                continue

            log_c_se_b = float(p_se_b / (abs(denom) * c_b))
            if not np.isfinite(log_c_se_b) or log_c_se_b <= 0.0:
                continue

            xb.append(float(payload["K"]))
            yb.append(log_c_b)
            wb.append(float(1.0 / (log_c_se_b**2)))

        if len(xb) < 2:
            continue

        try:
            _, slope_b = weighted_linear_fit(
                np.asarray(xb, dtype=float),
                np.asarray(yb, dtype=float),
                np.asarray(wb, dtype=float),
            )
            if slope_b < 0.0:
                t_boot.append(float(-1.0 / slope_b))
        except Exception:
            continue

    if len(t_boot) >= 20:
        ci_low = float(np.quantile(t_boot, bootstrap_alpha / 2.0))
        ci_high = float(np.quantile(t_boot, 1.0 - bootstrap_alpha / 2.0))
    else:
        ci_low = None
        ci_high = None

    summary = {
        "calib_id": CALIB_ID,
        "calib_uuid": CALIB_UUID,
        "backend": BACKEND_NAME,
        "objective_ry_offset": float(OBJECTIVE_RY_OFFSET),
        "job_id": str(job_id),
        "n_probe_ks": int(len(probe_ks)),
        "probe_ks": ",".join(str(k) for k in probe_ks),
        "shots_per_rep": int(shots),
        "n_repeats_per_k": int(n_repeats_per_k),
        "n_fit_points": int(len(x_fit)),
        "fit_kind": "weighted_log_contrast",
        "slope": float(slope),
        "intercept": float(intercept),
        "t_eff": t_eff,
        "t_eff_ci_low": ci_low,
        "t_eff_ci_high": ci_high,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_valid_samples": int(len(t_boot)),
        "status": "ok" if t_eff is not None else "nonnegative_slope",
    }

    return summary, point_rows, raw_rows


def config_dict() -> dict[str, Any]:
    return {
        "calib_id": CALIB_ID,
        "calib_uuid": CALIB_UUID,
        "backend_name": BACKEND_NAME,
        "objective_ry_offset": OBJECTIVE_RY_OFFSET,
        "probe_ks": PROBE_KS,
        "probe_shots": PROBE_SHOTS,
        "n_repeats_per_k": N_REPEATS_PER_K,
        "job_id": JOB_ID,
        "denom_threshold": DENOM_THRESHOLD,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_alpha": BOOTSTRAP_ALPHA,
        "rng_seed": RNG_SEED,
        "execution_mode": "post_job_single_job_id",
    }


def main() -> None:
    save_json(config_dict(), CONFIG_JSON)

    if JOB_ID == "REPLACE_WITH_YOUR_SINGLE_JOB_ID":
        raise ValueError("Set JOB_ID to the single runtime job id.")

    problem, a_true = build_large_problem(objective_ry_offset=float(OBJECTIVE_RY_OFFSET))

    print("=" * 100)
    print("POST-JOB T_eff HARDWARE CALIBRATION")
    print("=" * 100)
    print(f"Backend               : {BACKEND_NAME}")
    print(f"Objective offset      : {OBJECTIVE_RY_OFFSET:+.3f}")
    print(f"a_true                : {a_true:.6f}")
    print(f"job_id                : {JOB_ID}")
    print(f"expected pubs         : {len(PROBE_KS) * N_REPEATS_PER_K}")
    print(f"bootstrap samples     : {BOOTSTRAP_SAMPLES}")
    print("=" * 100)

    service = QiskitRuntimeService(channel="ibm_cloud")
    fetcher = RuntimeJobResultFetcher(service=service)

    t0 = time.perf_counter()

    summary, point_rows, raw_rows = robust_calibrate_t_eff_from_single_job(
        fetcher=fetcher,
        problem=problem,
        job_id=JOB_ID,
        probe_ks=PROBE_KS,
        shots=PROBE_SHOTS,
        n_repeats_per_k=N_REPEATS_PER_K,
        denom_threshold=DENOM_THRESHOLD,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_alpha=BOOTSTRAP_ALPHA,
        rng_seed=RNG_SEED,
    )

    runtime_seconds = float(time.perf_counter() - t0)
    summary["runtime_seconds"] = runtime_seconds
    summary["a_true"] = float(a_true)

    save_csv([summary], SUMMARY_CSV)
    save_csv(point_rows, POINTS_CSV)
    save_csv(raw_rows, RAW_CSV)

    print(f"\nSaved summary -> {SUMMARY_CSV}")
    print(f"Saved points  -> {POINTS_CSV}")
    print(f"Saved raw     -> {RAW_CSV}")
    print(f"Saved config  -> {CONFIG_JSON}")

    if summary["t_eff"] is not None:
        print(f"T_eff = {summary['t_eff']:.6f}")
        if summary["t_eff_ci_low"] is not None:
            print(
                f"{100*(1-BOOTSTRAP_ALPHA):.1f}% bootstrap CI = "
                f"[{summary['t_eff_ci_low']:.6f}, {summary['t_eff_ci_high']:.6f}]"
            )
    else:
        print("Calibration finished, but T_eff could not be inferred reliably.")


if __name__ == "__main__":
    main()