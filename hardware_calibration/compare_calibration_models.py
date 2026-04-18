from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# -----------------------------------------------------------------------------
# Model-comparison script for one-qubit calibration data.
#
# Goal:
#   Determine whether the calibration data are better explained by
#   (i) pure contrast decay,
#   (ii) coherent phase drift / over-rotation,
#   (iii) readout/SPAM bias,
#   or (iv) a combination of the above.
#
# Input CSV expected columns (from the previous calibration script):
#   k,K,q_ideal,one_counts,shots,p_hat,...
#
# Output:
#   - terminal summary with RMSE / NLL / AIC / BIC and fitted parameters
#   - CSV with pointwise fitted probabilities / residuals for every model
#   - CSV with model ranking
#
# Interpretation:
#   - If the best model still wants T -> large and delta ~ 0, the hardware may
#     simply be too coherent in this regime to reveal a strong decay-to-0.5.
#   - If a model with nonzero delta wins strongly, coherent error is dominating
#     and CABIQAE's exponential-contrast-only likelihood is likely mis-specified.
#   - If nonzero r01 / r10 are selected, readout/SPAM is contributing materially.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareConfig:
    BASE_DIR = Path(__file__).resolve().parent
    input_csv: str = str(BASE_DIR.parent /"hardware_calibration" / "t_calibration_improved_results.csv")
    output_dir: str = str(BASE_DIR.parent /"hardware_calibration" / "model_comparison")
    # Search range for contrast time.
    T_bounds: tuple[float, float] = (5.0, 5_000.0)
    # Bound for per-K coherent phase drift (radians per K unit).
    delta_bound: float = 0.02
    # Bound for readout error probabilities.
    readout_bound: float = 0.2
    # Multi-start grid for delta in the drift models.
    delta_grid: tuple[float, ...] = (-0.010, -0.005, -0.002, 0.0, 0.002, 0.005, 0.010)


@dataclass
class FitResult:
    model_name: str
    success: bool
    n_params: int
    nll: float
    rmse: float
    aic: float
    bic: float
    params: dict[str, float]
    fitted_probs: np.ndarray
    residuals: np.ndarray
    message: str


# -------------------------------
# Helpers
# -------------------------------


def _theta_from_a(a_true: float) -> float:
    if not (0.0 < a_true < 1.0):
        raise ValueError("a_true must be in (0,1).")
    return float(np.arcsin(np.sqrt(a_true)))


def _stable_clip_prob(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    if np.ndim(p) == 0:
        return float(arr)
    return arr


def _binomial_nll(ones: np.ndarray, shots: np.ndarray, p: np.ndarray) -> float:
    p = _stable_clip_prob(p)
    return float(-np.sum(ones * np.log(p) + (shots - ones) * np.log(1.0 - p)))


def _rmse_prob(p_hat: np.ndarray, p_fit: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p_hat - p_fit) ** 2)))


def _aic(nll: float, n_params: int) -> float:
    return float(2.0 * n_params + 2.0 * nll)


def _bic(nll: float, n_params: int, n_obs: int) -> float:
    return float(n_params * np.log(n_obs) + 2.0 * nll)


def _infer_a_true(q_ideal_at_k0: float) -> float:
    return float(q_ideal_at_k0)


# -------------------------------
# Model family
# -------------------------------


def prob_contrast_only(K: np.ndarray, q_ideal: np.ndarray, T: float) -> np.ndarray:
    contrast = np.exp(-K / T)
    return _stable_clip_prob(0.5 + contrast * (q_ideal - 0.5))


def prob_spam_only(q_ideal: np.ndarray, r01: float, r10: float) -> np.ndarray:
    # 0 -> 1 with prob r01 ; 1 -> 0 with prob r10.
    p = (1.0 - r01 - r10) * q_ideal + r01
    return _stable_clip_prob(p)


def prob_drift_spam(K: np.ndarray, theta: float, delta: float, r01: float, r10: float) -> np.ndarray:
    q = np.sin(K * theta + delta * K) ** 2
    p = (1.0 - r01 - r10) * q + r01
    return _stable_clip_prob(p)


def prob_drift_contrast_spam(
    K: np.ndarray,
    theta: float,
    delta: float,
    T: float,
    r01: float,
    r10: float,
) -> np.ndarray:
    q = np.sin(K * theta + delta * K) ** 2
    q_decay = 0.5 + np.exp(-K / T) * (q - 0.5)
    p = (1.0 - r01 - r10) * q_decay + r01
    return _stable_clip_prob(p)


# -------------------------------
# Fitters
# -------------------------------


def fit_contrast_only(K: np.ndarray, q_ideal: np.ndarray, ones: np.ndarray, shots: np.ndarray, p_hat: np.ndarray, cfg: CompareConfig) -> FitResult:
    log_low = math.log(cfg.T_bounds[0])
    log_high = math.log(cfg.T_bounds[1])

    def objective(x: np.ndarray) -> float:
        T = float(np.exp(x[0]))
        p = prob_contrast_only(K, q_ideal, T)
        return _binomial_nll(ones, shots, p)

    res = minimize(
        objective,
        x0=np.array([math.log(200.0)]),
        method="L-BFGS-B",
        bounds=[(log_low, log_high)],
    )
    T = float(np.exp(res.x[0]))
    p_fit = prob_contrast_only(K, q_ideal, T)
    nll = _binomial_nll(ones, shots, p_fit)
    rmse = _rmse_prob(p_hat, p_fit)
    return FitResult(
        model_name="contrast_only",
        success=bool(res.success),
        n_params=1,
        nll=nll,
        rmse=rmse,
        aic=_aic(nll, 1),
        bic=_bic(nll, 1, len(K)),
        params={"T": T},
        fitted_probs=p_fit,
        residuals=p_hat - p_fit,
        message=str(res.message),
    )


def fit_spam_only(q_ideal: np.ndarray, ones: np.ndarray, shots: np.ndarray, p_hat: np.ndarray, cfg: CompareConfig) -> FitResult:
    rb = cfg.readout_bound

    def objective(x: np.ndarray) -> float:
        r01, r10 = x
        if r01 + r10 >= 0.999:
            return 1e12
        p = prob_spam_only(q_ideal, float(r01), float(r10))
        return _binomial_nll(ones, shots, p)

    res = minimize(
        objective,
        x0=np.array([0.002, 0.01]),
        method="L-BFGS-B",
        bounds=[(0.0, rb), (0.0, rb)],
    )
    r01, r10 = map(float, res.x)
    p_fit = prob_spam_only(q_ideal, r01, r10)
    nll = _binomial_nll(ones, shots, p_fit)
    rmse = _rmse_prob(p_hat, p_fit)
    return FitResult(
        model_name="spam_only",
        success=bool(res.success),
        n_params=2,
        nll=nll,
        rmse=rmse,
        aic=_aic(nll, 2),
        bic=_bic(nll, 2, len(q_ideal)),
        params={"r01": r01, "r10": r10},
        fitted_probs=p_fit,
        residuals=p_hat - p_fit,
        message=str(res.message),
    )


def fit_drift_spam(K: np.ndarray, theta: float, ones: np.ndarray, shots: np.ndarray, p_hat: np.ndarray, cfg: CompareConfig) -> FitResult:
    db = cfg.delta_bound
    rb = cfg.readout_bound
    best: FitResult | None = None

    def objective(x: np.ndarray) -> float:
        delta, r01, r10 = map(float, x)
        if r01 + r10 >= 0.999:
            return 1e12
        p = prob_drift_spam(K, theta, delta, r01, r10)
        return _binomial_nll(ones, shots, p)

    for delta0 in cfg.delta_grid:
        res = minimize(
            objective,
            x0=np.array([delta0, 0.002, 0.01]),
            method="L-BFGS-B",
            bounds=[(-db, db), (0.0, rb), (0.0, rb)],
        )
        delta, r01, r10 = map(float, res.x)
        p_fit = prob_drift_spam(K, theta, delta, r01, r10)
        nll = _binomial_nll(ones, shots, p_fit)
        rmse = _rmse_prob(p_hat, p_fit)
        cand = FitResult(
            model_name="drift_spam",
            success=bool(res.success),
            n_params=3,
            nll=nll,
            rmse=rmse,
            aic=_aic(nll, 3),
            bic=_bic(nll, 3, len(K)),
            params={"delta": delta, "r01": r01, "r10": r10},
            fitted_probs=p_fit,
            residuals=p_hat - p_fit,
            message=str(res.message),
        )
        if best is None or cand.nll < best.nll:
            best = cand

    assert best is not None
    return best


def fit_drift_contrast_spam(K: np.ndarray, theta: float, ones: np.ndarray, shots: np.ndarray, p_hat: np.ndarray, cfg: CompareConfig) -> FitResult:
    db = cfg.delta_bound
    rb = cfg.readout_bound
    log_low = math.log(cfg.T_bounds[0])
    log_high = math.log(cfg.T_bounds[1])
    best: FitResult | None = None

    def objective(x: np.ndarray) -> float:
        delta = float(x[0])
        T = float(np.exp(x[1]))
        r01 = float(x[2])
        r10 = float(x[3])
        if r01 + r10 >= 0.999:
            return 1e12
        p = prob_drift_contrast_spam(K, theta, delta, T, r01, r10)
        return _binomial_nll(ones, shots, p)

    for delta0 in cfg.delta_grid:
        for logT0 in (math.log(50.0), math.log(200.0), math.log(1000.0)):
            res = minimize(
                objective,
                x0=np.array([delta0, logT0, 0.002, 0.01]),
                method="L-BFGS-B",
                bounds=[(-db, db), (log_low, log_high), (0.0, rb), (0.0, rb)],
            )
            delta = float(res.x[0])
            T = float(np.exp(res.x[1]))
            r01 = float(res.x[2])
            r10 = float(res.x[3])
            p_fit = prob_drift_contrast_spam(K, theta, delta, T, r01, r10)
            nll = _binomial_nll(ones, shots, p_fit)
            rmse = _rmse_prob(p_hat, p_fit)
            cand = FitResult(
                model_name="drift_contrast_spam",
                success=bool(res.success),
                n_params=4,
                nll=nll,
                rmse=rmse,
                aic=_aic(nll, 4),
                bic=_bic(nll, 4, len(K)),
                params={"delta": delta, "T": T, "r01": r01, "r10": r10},
                fitted_probs=p_fit,
                residuals=p_hat - p_fit,
                message=str(res.message),
            )
            if best is None or cand.nll < best.nll:
                best = cand

    assert best is not None
    return best


# -------------------------------
# Reporting
# -------------------------------


def print_summary(results: list[FitResult]) -> None:
    print("=" * 110)
    print("Model comparison for calibration data")
    print("=" * 110)
    print(f"{'model':<24} {'ok':<4} {'nll':>12} {'rmse':>12} {'aic':>12} {'bic':>12}  parameters")
    for r in results:
        params_str = ", ".join(f"{k}={v:.6g}" for k, v in r.params.items())
        print(f"{r.model_name:<24} {str(r.success):<4} {r.nll:12.6f} {r.rmse:12.6f} {r.aic:12.6f} {r.bic:12.6f}  {params_str}")
    print("-" * 110)
    best = results[0]
    print(f"Best by AIC: {best.model_name}")
    print()
    print("Interpretation guide:")
    print("- If contrast_only wins cleanly with finite T, the original CABIQAE noise model is supported.")
    print("- If drift_spam wins, coherent drift / over-rotation dominates and the original model is mis-specified.")
    print("- If drift_contrast_spam wins and T is finite, an extended CABIQAE likelihood may be worthwhile.")
    print("- If best-fit T runs to the upper boundary while delta is nonzero, the data are telling you 'phase drift, not contrast decay'.")
    print("=" * 110)


# -------------------------------
# Main
# -------------------------------


def main() -> None:
    cfg = CompareConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.input_csv)
    required = {"k", "K", "q_ideal", "one_counts", "shots", "p_hat"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    K = df["K"].to_numpy(dtype=float)
    q_ideal = df["q_ideal"].to_numpy(dtype=float)
    ones = df["one_counts"].to_numpy(dtype=float)
    shots = df["shots"].to_numpy(dtype=float)
    p_hat = df["p_hat"].to_numpy(dtype=float)

    # Since k=0 implies q_ideal = a_true in this experiment.
    row_k0 = df.loc[df["k"] == 0]
    if row_k0.empty:
        raise ValueError("Could not infer a_true because k=0 is not present in the CSV.")
    a_true = _infer_a_true(float(row_k0.iloc[0]["q_ideal"]))
    theta = _theta_from_a(a_true)

    results = [
        fit_contrast_only(K, q_ideal, ones, shots, p_hat, cfg),
        fit_spam_only(q_ideal, ones, shots, p_hat, cfg),
        fit_drift_spam(K, theta, ones, shots, p_hat, cfg),
        fit_drift_contrast_spam(K, theta, ones, shots, p_hat, cfg),
    ]
    results.sort(key=lambda r: r.aic)

    # Save ranking.
    ranking_path = output_dir / "model_ranking.csv"
    ranking_rows = []
    for r in results:
        row = {
            "model": r.model_name,
            "success": r.success,
            "n_params": r.n_params,
            "nll": r.nll,
            "rmse": r.rmse,
            "aic": r.aic,
            "bic": r.bic,
            "message": r.message,
        }
        for k, v in r.params.items():
            row[k] = v
        ranking_rows.append(row)
    pd.DataFrame(ranking_rows).to_csv(ranking_path, index=False)

    # Save pointwise fits.
    pointwise = df.copy()
    for r in results:
        pointwise[f"p_fit__{r.model_name}"] = r.fitted_probs
        pointwise[f"resid__{r.model_name}"] = r.residuals
    pointwise_path = output_dir / "pointwise_fits.csv"
    pointwise.to_csv(pointwise_path, index=False)

    print(f"Input CSV            : {Path(cfg.input_csv).resolve()}")
    print(f"Output directory     : {output_dir.resolve()}")
    print(f"a_true inferred      : {a_true:.10f}")
    print(f"theta inferred       : {theta:.10f}")
    print_summary(results)
    print(f"Ranking CSV          : {ranking_path.resolve()}")
    print(f"Pointwise fits CSV   : {pointwise_path.resolve()}")


if __name__ == "__main__":
    main()
