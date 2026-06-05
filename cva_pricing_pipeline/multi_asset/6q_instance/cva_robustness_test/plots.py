from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "plots"
RESULTS_CSV = ROOT / "cva_robustness_case_results.csv"
SUMMARY_CSV = ROOT / "cva_robustness_summary_statistics.csv"
CONFIG_JSON = ROOT / "robustness_sweep_config.json"
REPO_ROOT = next(parent for parent in ROOT.parents if (parent / "pyproject.toml").exists())

TRAINING_REL = Path(
    "quantum/training/crca/positive_exposure/"
    "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
)
SHARED_TRAINING_PATH = (
    REPO_ROOT
    / "data/multi_asset/6q_instance/quantum/training/crca/positive_exposure"
    / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
)

CASE_LABELS = {
    "base": "Base",
    "call_down_10pct": "Call -10%",
    "call_up_10pct": "Call +10%",
    "put_down_10pct": "Put -10%",
    "put_up_10pct": "Put +10%",
    "both_down_10pct": "Both -10%",
    "both_up_10pct": "Both +10%",
}

PLOT_STYLE = {
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def case_order() -> list[str]:
    if CONFIG_JSON.exists():
        with CONFIG_JSON.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cases = [str(c["case_id"]) for c in cfg.get("cases", [])]
        if cases:
            return cases
    if RESULTS_CSV.exists():
        return [str(x) for x in pd.read_csv(RESULTS_CSV)["case_id"].drop_duplicates()]
    return [p.name for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]


def label(case_id: str) -> str:
    return CASE_LABELS.get(case_id, case_id.replace("_", " "))


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    numeric = [
        "repeat",
        "cva_statevector",
        "cva_classical_reference",
        "absolute_error",
        "signed_error",
        "absolute_relative_error_pct",
        "signed_relative_error_pct",
        "classical_cva_std_err_mc_continuous",
        "exposure_l2_statevector",
        "qcbm_kl_statevector",
        "exposure_best_l2",
        "exposure_best_l2_rechecked",
        "exposure_final_l2",
        "exposure_elapsed_s",
        "final_elapsed_s",
        "shots",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    order = {case_id: idx for idx, case_id in enumerate(case_order())}
    df["_case_order"] = df["case_id"].map(order).fillna(len(order)).astype(int)
    df = df.sort_values(["_case_order", "repeat"]).reset_index(drop=True)
    df["case_label"] = df["case_id"].map(label)
    df["execution"] = np.arange(1, len(df) + 1)
    return df


def resolve_path(path_like: str | Path) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_training(path: str | Path | None = None) -> dict[str, np.ndarray]:
    path = resolve_path(path) if path else SHARED_TRAINING_PATH
    with np.load(path, allow_pickle=True) as raw:
        return {key: raw[key] for key in raw.files}


def training_path_for_row(row: pd.Series) -> Path:
    raw = row.get("exposure_training_path", "")
    if isinstance(raw, str) and raw.strip():
        return resolve_path(raw)
    case_path = DATA_DIR / str(row["case_id"]) / TRAINING_REL
    return case_path if case_path.exists() else SHARED_TRAINING_PATH


def load_case_exposure_target(row: pd.Series) -> np.ndarray:
    benchmark_path = resolve_path(str(row["benchmark_path"]))
    with np.load(benchmark_path, allow_pickle=True) as raw:
        return np.asarray(raw["v_joint_t"], dtype=float).ravel() / float(raw["C_v"])


def save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cva_by_execution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = df["execution"].to_numpy()
    xlabels = [
        f"{row.case_label}\nr{int(row.repeat)}" for row in df.itertuples(index=False)
    ]
    classical = df["cva_classical_reference"].to_numpy(dtype=float)
    quantum = df["cva_statevector"].to_numpy(dtype=float)
    classical_se = df["classical_cva_std_err_mc_continuous"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        classical,
        yerr=classical_se,
        fmt="o-",
        color="#1f77b4",
        capsize=4,
        linewidth=1.8,
        markersize=5,
        label="classical reference +/- MC SE",
    )
    ax.plot(
        x,
        quantum,
        "s-",
        color="#ff7f0e",
        linewidth=1.8,
        markersize=5,
        label="statevector CVA",
    )
    ax.set_title("CVA robustness by execution")
    ax.set_xlabel("Execution")
    ax.set_ylabel("CVA")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.legend()
    return save(fig, "robustness_cva_by_execution.png")


def plot_errors_by_execution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = df["execution"].to_numpy()
    labels = [label(case_id) for case_id in df["case_id"]]
    abs_rel = df["absolute_relative_error_pct"].to_numpy(dtype=float)
    signed_rel = df["signed_relative_error_pct"].to_numpy(dtype=float)

    ax.bar(x, abs_rel, color="#ff7f0e", alpha=0.78, label="absolute relative error")
    ax.plot(x, signed_rel, "o-", color="#444444", linewidth=1.5, label="signed relative error")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Relative CVA error by execution")
    ax.set_xlabel("Execution")
    ax.set_ylabel("Error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    return save(fig, "robustness_relative_errors_by_execution.png")


def plot_error_vs_classical_se(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = df["execution"].to_numpy()
    labels = [label(case_id) for case_id in df["case_id"]]
    abs_error = df["absolute_error"].to_numpy(dtype=float)
    classical_se = df["classical_cva_std_err_mc_continuous"].to_numpy(dtype=float)

    ax.bar(x - 0.18, abs_error, width=0.36, color="#d62728", alpha=0.78, label="|quantum - classical|")
    ax.bar(x + 0.18, classical_se, width=0.36, color="#1f77b4", alpha=0.78, label="classical MC SE")
    ax.set_yscale("log")
    ax.set_title("Absolute CVA error vs classical Monte Carlo standard error")
    ax.set_xlabel("Execution")
    ax.set_ylabel("Value (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    return save(fig, "robustness_error_vs_standard_error.png")


def plot_case_summary(df: pd.DataFrame) -> Path:
    grouped = (
        df.groupby(["_case_order", "case_id", "case_label"], as_index=False)
        .agg(
            n=("cva_statevector", "count"),
            cva_statevector_mean=("cva_statevector", "mean"),
            cva_statevector_std=("cva_statevector", "std"),
            cva_classical_reference_mean=("cva_classical_reference", "mean"),
            classical_cva_std_err_mc_continuous_mean=(
                "classical_cva_std_err_mc_continuous",
                "mean",
            ),
            absolute_relative_error_pct_mean=("absolute_relative_error_pct", "mean"),
        )
        .sort_values("_case_order")
    )
    quantum_stderr = (
        grouped["cva_statevector_std"].fillna(0.0)
        / np.sqrt(grouped["n"].clip(lower=1).to_numpy(dtype=float))
    )

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(grouped))
    ax.errorbar(
        x,
        grouped["cva_classical_reference_mean"],
        yerr=grouped["classical_cva_std_err_mc_continuous_mean"],
        fmt="o",
        capsize=4,
        color="#1f77b4",
        label="classical reference +/- MC SE",
    )
    ax.errorbar(
        x,
        grouped["cva_statevector_mean"],
        yerr=quantum_stderr,
        fmt="s",
        capsize=4,
        color="#ff7f0e",
        label="statevector mean +/- repeat SE",
    )
    ax.set_title("CVA robustness summary by scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("CVA")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["case_label"], rotation=35, ha="right")
    ax.legend()
    return save(fig, "robustness_cva_summary_by_scenario.png")


def plot_training_histories(df: pd.DataFrame) -> list[Path]:
    fig_l2, ax_l2 = plt.subplots(figsize=(11.5, 5.2))
    fig_cost, ax_cost = plt.subplots(figsize=(11.5, 5.2))

    paths = []
    for _, row in df.drop_duplicates("exposure_training_path").iterrows():
        path = training_path_for_row(row)
        if path not in paths:
            paths.append(path)

    for path in paths:
        data = load_training(path)
        line_label = "Shared positive-exposure theta_star"
        if len(paths) > 1:
            line_label = path.parent.parent.parent.name
        iterations = np.arange(len(data["l2_history"]))
        ax_l2.plot(iterations, data["l2_history"], linewidth=1.5, label=line_label)
        if "best_so_far" in data:
            ax_l2.plot(
                iterations,
                data["best_so_far"],
                linewidth=1.0,
                linestyle="--",
                alpha=0.65,
            )
        ax_cost.plot(
            np.arange(len(data["cost_history"])),
            data["cost_history"],
            linewidth=1.5,
            label=line_label,
        )

    ax_l2.set_title("Shared positive-exposure training L2 history")
    ax_l2.set_xlabel("Iteration")
    ax_l2.set_ylabel("L2 error")
    ax_l2.set_yscale("log")
    ax_l2.legend(ncol=1, fontsize=8)

    ax_cost.set_title("Shared positive-exposure training objective history")
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Objective")
    ax_cost.set_yscale("log")
    ax_cost.legend(ncol=1, fontsize=8)

    return [
        save(fig_l2, "training_l2_histories.png"),
        save(fig_cost, "training_objective_histories.png"),
    ]


def plot_training_quality_by_scenario(df: pd.DataFrame) -> Path:
    grouped = (
        df.groupby(["_case_order", "case_id", "case_label"], as_index=False)
        .agg(
            exposure_l2_statevector=("exposure_l2_statevector", "mean"),
            qcbm_kl_statevector=("qcbm_kl_statevector", "mean"),
        )
        .sort_values("_case_order")
    )
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(
        x,
        grouped["exposure_l2_statevector"],
        width=0.55,
        label="positive-exposure L2",
        color="#ff7f0e",
        alpha=0.82,
    )
    ax.set_yscale("log")
    ax.set_title("Shared positive-exposure approximation error by scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Statevector L2 error (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["case_label"], rotation=35, ha="right")
    ax.legend()
    return save(fig, "training_quality_by_scenario.png")


def histogram_plot(row: pd.Series, trained: bool) -> Path:
    case_id = str(row["case_id"])
    data = load_training(training_path_for_row(row))
    target = load_case_exposure_target(row)
    other_key = "f_star" if trained else "f_init"
    other = np.asarray(data[other_key], dtype=float)
    x = np.arange(target.size)
    title = (
        f"{label(case_id)} - After training (best iterate, shots + backend noise)"
        if trained
        else f"{label(case_id)} - Before training (initial iterate)"
    )
    other_label = "trained" if trained else "initial"
    filename = (
        f"hist_{case_id}_after_training.png"
        if trained
        else f"hist_{case_id}_before_training.png"
    )

    fig, ax = plt.subplots(figsize=(14.5, 4.8))
    ax.bar(x, other, width=0.52, color="#ff7f0e", edgecolor="#c45f00", alpha=0.88, label=other_label, zorder=3)
    ax.bar(x, target, width=0.86, color="#4169e1", edgecolor="#4169e1", alpha=0.45, label="target", zorder=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Control basis state |i>")
    ax.set_ylabel("f(i)")
    ax.set_xlim(-3.5, target.size + 2.5)
    ax.set_ylim(0.0, max(1.05, float(np.nanmax([target.max(), other.max()])) * 1.08))
    ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 63])
    ax.legend(loc="upper right")
    return save(fig, filename)


def plot_histograms(cases: list[str]) -> list[Path]:
    paths = []
    df = load_results()
    ok = df[df["status"].fillna("") == "ok"].copy()
    if ok.empty:
        ok = df.copy()
    for case_id in cases:
        case_rows = ok[ok["case_id"] == case_id]
        if case_rows.empty:
            continue
        row = case_rows.iloc[0]
        paths.append(histogram_plot(row, trained=False))
        paths.append(histogram_plot(row, trained=True))
    return paths


def write_manifest(paths: list[Path]) -> Path:
    manifest = OUT_DIR / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("Generated CVA robustness plots\n")
        f.write(f"Root: {ROOT}\n\n")
        for path in paths:
            f.write(f"{path.name}\n")
    return manifest


def main() -> None:
    plt.rcParams.update(PLOT_STYLE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = case_order()
    df = load_results()
    ok = df[df["status"].fillna("") == "ok"].copy()
    if ok.empty:
        ok = df.copy()

    paths = [
        plot_cva_by_execution(ok),
        plot_errors_by_execution(ok),
        plot_error_vs_classical_se(ok),
        plot_case_summary(ok),
        plot_training_quality_by_scenario(ok),
    ]
    paths.extend(plot_training_histories(ok))
    paths.extend(plot_histograms(cases))
    paths.append(write_manifest(paths))

    print(f"Generated {len(paths) - 1} plots in {OUT_DIR}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
