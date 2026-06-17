"""Generic error-budget plotter for the paper AE/CVA curves.

Reference scripts used to mirror the existing plots:
- toys/.../ideal_regime/ideal_bae_biqae_iqae_cabiqae_latentt_experiment.py
- cva_pricing_pipeline/.../noiseless_simulation/plot_noiseless_cva_ae.py
- toys/.../beta_hardware_experiment/paper_replay_error_plot.py
- cva_pricing_pipeline/.../hardware/paper_contrast_plots.py
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "error_budget"

IDEAL_TOY_DIR = (
    ROOT
    / "toys"
    / "amplitude_estimation_experiments"
    / "ideal_regime"
    / "experiment_results"
)
HARDWARE_TOY_DIR = (
    ROOT
    / "toys"
    / "amplitude_estimation_experiments"
    / "noise_aware_regime"
    / "3qubit_toy"
    / "hardware"
    / "beta_hardware_experiment"
    / "experiment_results"
)
CVA_NOISELESS_DIR = (
    ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "cva_pricing_multi_asset"
    / "quantum"
    / "ae_cva"
    / "noiseless_simulation"
    / "experiment_results"
)
CVA_HARDWARE_DIR = (
    ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "cva_pricing_multi_asset"
    / "quantum"
    / "ae_cva"
    / "hardware"
    / "results"
    / "q-ctrl_hardware_basquecountry"
)
CVA_HARDWARE_DCS_DIR = CVA_HARDWARE_DIR.with_name("q_ctrl_hardware")


PAPER_STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.top": True,
    "ytick.right": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

SERIES_ALIASES = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "cabiqae": "CABIQAE",
    "cabiqae_latentt": "CABIQAE",
    "cabiqae_known_t": "CABIQAE",
    "cabiqae-latentt": "CABIQAE",
    "classical_mc": "DCS",
    "classical mc": "DCS",
    "dcs": "DCS",
}


@dataclass(frozen=True, slots=True)
class SummarySource:
    paths: tuple[Path, ...]
    rename_algorithm: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CurveSpec:
    key: str
    output_basename: str
    sources: tuple[SummarySource, ...]
    algorithm_order: tuple[str, ...]
    styles: Mapping[str, Mapping[str, str]]
    y_col: str
    y_low_col: str
    y_high_col: str
    ylabel: str
    x_col: str = "query_budget_actual_mean"
    x_fallback_cols: tuple[str, ...] = ("query_budget_actual_median", "budget")
    xlabel: str = r"$N_q$"
    title: str = ""
    include_linear_guide: bool = False
    include_sqrt_guide: bool = True
    guide_mode: str = "fit"
    fixed_guide_anchor_x: float | None = None
    fixed_guide_anchor_y: float | None = None
    max_x: float | None = None
    y_limits: tuple[float, float] | None = None
    x_margin_factor: float = 1.18
    collapse_identical_points: bool = False
    summary_output_suffix: str = "_summary"


def parse_args() -> argparse.Namespace:
    plot_keys = tuple(spec.key for spec in curve_specs())
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-style error-budget curves from existing CSV result summaries. "
            "No repository-local plotting modules are imported."
        )
    )
    parser.add_argument(
        "--plot",
        default="all",
        help=f"Comma-separated plot keys or 'all'. Available: {', '.join(plot_keys)}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sqrt-guide-coefficient",
        type=float,
        default=None,
        help=(
            "Draw the square-root guide as coefficient/sqrt(N) instead of an "
            "anchored O(1/sqrt(N)) guide."
        ),
    )
    return parser.parse_args()


def as_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def configure_matplotlib() -> None:
    mpl.rcParams.update(PAPER_STYLE)


def normalise_series(value: object) -> str:
    raw = str(value).strip()
    if raw in {"BAE", "BIQAE", "CABIQAE", "DCS"}:
        return raw
    key = raw.lower().replace("_", " ").replace("-", " ").strip()
    if key in SERIES_ALIASES:
        return SERIES_ALIASES[key]
    compact = key.replace(" ", "_")
    return SERIES_ALIASES.get(compact, raw)


def series_from_row(row: pd.Series) -> str:
    plot_label = row.get("plot_label")
    if isinstance(plot_label, str) and plot_label.strip():
        return normalise_series(plot_label)

    algorithm_key = row.get("algorithm_key")
    if isinstance(algorithm_key, str) and algorithm_key.strip():
        return normalise_series(algorithm_key)

    return normalise_series(row.get("algorithm", ""))


def load_summary_source(source: SummarySource) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in source.paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing summary CSV: {path}")
        frame = pd.read_csv(path)
        if source.rename_algorithm and "algorithm" in frame.columns:
            frame = frame.copy()
            frame["algorithm"] = frame["algorithm"].replace(dict(source.rename_algorithm))
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("SummarySource must contain at least one path.")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_curve_rows(spec: CurveSpec) -> pd.DataFrame:
    rows = pd.concat(
        [load_summary_source(source) for source in spec.sources],
        ignore_index=True,
        sort=False,
    )
    rows = rows.copy()
    rows["series"] = rows.apply(series_from_row, axis=1)
    rows = rows[rows["series"].isin(spec.algorithm_order)].copy()

    rows["x"] = numeric_with_fallback(rows, spec.x_col, spec.x_fallback_cols)
    for column in ("budget", spec.y_col, spec.y_low_col, spec.y_high_col):
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["y"] = pd.to_numeric(rows[spec.y_col], errors="coerce")
    rows["ci_low"] = pd.to_numeric(rows[spec.y_low_col], errors="coerce")
    rows["ci_high"] = pd.to_numeric(rows[spec.y_high_col], errors="coerce")

    rows = rows[np.isfinite(rows["x"]) & np.isfinite(rows["y"])]
    rows = rows[(rows["x"] > 0.0) & (rows["y"] > 0.0)].copy()
    if spec.max_x is not None:
        rows = rows[rows["x"] <= float(spec.max_x)].copy()

    rows["order"] = rows["series"].map({name: idx for idx, name in enumerate(spec.algorithm_order)})
    rows = rows.sort_values(["order", "x", "budget"], kind="mergesort").copy()
    if spec.collapse_identical_points:
        rows = (
            rows.sort_values(["order", "budget"], kind="mergesort")
            .drop_duplicates(subset=["series", "x", "y"], keep="last")
            .sort_values(["order", "x", "budget"], kind="mergesort")
            .copy()
        )
    return rows


def numeric_with_fallback(
    rows: pd.DataFrame,
    primary: str,
    fallbacks: Sequence[str],
) -> pd.Series:
    values = (
        pd.to_numeric(rows[primary], errors="coerce")
        if primary in rows.columns
        else pd.Series(np.nan, index=rows.index, dtype=float)
    )
    for fallback in fallbacks:
        if fallback not in rows.columns:
            continue
        values = values.fillna(pd.to_numeric(rows[fallback], errors="coerce"))
    return values


def ci_errorbar(center: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    lower = np.where(np.isfinite(ci_low), center - ci_low, 0.0)
    upper = np.where(np.isfinite(ci_high), ci_high - center, 0.0)
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)
    lower = np.minimum(lower, np.maximum(0.0, 0.95 * center))
    return np.vstack([lower, upper])


def finite_positive(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def add_scaling_guides(
    ax: plt.Axes,
    rows: pd.DataFrame,
    spec: CurveSpec,
    *,
    sqrt_guide_coefficient: float | None = None,
) -> None:
    if not spec.include_linear_guide and not spec.include_sqrt_guide:
        return

    x_values = finite_positive(rows["x"])
    y_values = finite_positive(rows["y"])
    if x_values.size == 0 or y_values.size == 0:
        return

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    if x_max <= x_min:
        return

    if spec.guide_mode == "fixed":
        if spec.fixed_guide_anchor_x is None or spec.fixed_guide_anchor_y is None:
            raise ValueError(f"{spec.key} requested fixed guides without an anchor.")
        anchor_x = float(spec.fixed_guide_anchor_x)
        anchor_y = float(spec.fixed_guide_anchor_y)
    elif spec.guide_mode == "leftmost":
        anchor_x = x_min
        at_left = rows[np.isclose(rows["x"], x_min)]["y"].to_numpy(dtype=float)
        anchor_y = float(np.nanmin(at_left)) if at_left.size else float(np.nanmedian(y_values))
    elif spec.guide_mode == "fit":
        anchor_x = x_min
        order = np.argsort(rows["x"].to_numpy(dtype=float))
        x_fit = rows["x"].to_numpy(dtype=float)[order]
        y_fit = rows["y"].to_numpy(dtype=float)[order]
        valid = np.isfinite(x_fit) & np.isfinite(y_fit) & (x_fit > 0.0) & (y_fit > 0.0)
        x_fit = x_fit[valid]
        y_fit = y_fit[valid]
        if x_fit.size >= 2 and float(np.max(x_fit)) > float(np.min(x_fit)):
            slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), deg=1)
            anchor_y = float(np.exp(intercept) * anchor_x**slope)
        else:
            anchor_y = float(np.nanmedian(y_values))
    else:
        raise ValueError(f"Unsupported guide mode: {spec.guide_mode!r}")

    if not np.isfinite(anchor_y) or anchor_y <= 0.0:
        return

    guide_x = np.geomspace(x_min, x_max, num=200)
    if spec.include_linear_guide:
        ax.loglog(
            guide_x,
            anchor_y * (anchor_x / guide_x),
            color="#262626",
            linestyle="--",
            linewidth=1.25,
            alpha=0.82,
            label=r"$O(1/N)$",
            zorder=1,
        )
    if spec.include_sqrt_guide:
        if sqrt_guide_coefficient is None:
            sqrt_guide_y = anchor_y * np.sqrt(anchor_x / guide_x)
            sqrt_guide_label = r"$O(1/\sqrt{N})$"
        else:
            if not np.isfinite(sqrt_guide_coefficient) or sqrt_guide_coefficient <= 0.0:
                raise ValueError("sqrt_guide_coefficient must be positive and finite.")
            sqrt_guide_y = float(sqrt_guide_coefficient) / np.sqrt(guide_x)
            sqrt_guide_label = rf"${sqrt_guide_coefficient:g}/\sqrt{{N}}$"
        ax.loglog(
            guide_x,
            sqrt_guide_y,
            color="#262626",
            linestyle=":",
            linewidth=1.45,
            alpha=0.82,
            label=sqrt_guide_label,
            zorder=1,
        )


def multiplicative_limits(values: Sequence[float], *, margin_factor: float) -> tuple[float, float]:
    finite = finite_positive(values)
    if finite.size == 0:
        raise ValueError("Cannot compute log limits without positive finite values.")
    if float(margin_factor) <= 1.0:
        raise ValueError("margin_factor must be greater than 1.0.")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high):
        return low / 1.6, high * 1.6
    return low / float(margin_factor), high * float(margin_factor)


def plot_curve(
    spec: CurveSpec,
    output_dir: Path,
    *,
    sqrt_guide_coefficient: float | None = None,
) -> Path:
    rows = load_curve_rows(spec)
    if rows.empty:
        raise ValueError(f"{spec.key} has no valid rows to plot.")

    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
        add_scaling_guides(
            ax,
            rows,
            spec,
            sqrt_guide_coefficient=sqrt_guide_coefficient,
        )

        for series in spec.algorithm_order:
            group = rows[rows["series"] == series].sort_values("x")
            if group.empty:
                continue
            style = spec.styles[series]
            x_values = group["x"].to_numpy(dtype=float)
            y_values = group["y"].to_numpy(dtype=float)
            ax.errorbar(
                x_values,
                y_values,
                yerr=ci_errorbar(
                    y_values,
                    group["ci_low"].to_numpy(dtype=float),
                    group["ci_high"].to_numpy(dtype=float),
                ),
                fmt=style["marker"],
                color=style["color"],
                linestyle="-",
                linewidth=2.0,
                markersize=5.6,
                elinewidth=1.0,
                capsize=2.8,
                label=series,
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(spec.xlabel)
        ax.set_ylabel(spec.ylabel)
        if spec.title:
            ax.set_title(spec.title)
        ax.set_xlim(
            *multiplicative_limits(
                rows["x"].to_numpy(dtype=float),
                margin_factor=spec.x_margin_factor,
            )
        )
        if spec.y_limits is not None:
            ax.set_ylim(*spec.y_limits)
        ax.grid(True, which="major", color="#CFCFCF", linewidth=0.8, alpha=0.72)
        ax.grid(True, which="minor", color="#E7E7E7", linewidth=0.45, alpha=0.85)
        ax.minorticks_on()
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(1.15)
        ax.legend(frameon=False, loc="lower left", handlelength=2.55, borderpad=0.2)

        output_base = output_dir / spec.output_basename
        output_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.03)
        fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

    summary_path = output_base.with_name(f"{output_base.name}{spec.summary_output_suffix}.csv")
    export_columns = [
        column
        for column in (
            "series",
            "algorithm",
            "algorithm_key",
            "plot_label",
            "budget",
            "x",
            "y",
            "ci_low",
            "ci_high",
            "n_points",
            "n_runs",
            "source_file",
        )
        if column in rows.columns
    ]
    rows[export_columns].to_csv(summary_path, index=False)
    return output_base


def curve_specs() -> tuple[CurveSpec, ...]:
    ideal_styles = {
        "BAE": {"color": "#E07A5F", "marker": "^"},
        "BIQAE": {"color": "#A23B72", "marker": "s"},
        "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
        "DCS": {"color": "#2A9D8F", "marker": "X"},
    }
    hardware_toy_styles = {
        "BAE": {"color": "#E76F51", "marker": "^"},
        "BIQAE": {"color": "#B23A86", "marker": "s"},
        "CABIQAE": {"color": "#1F7A99", "marker": "o"},
        "DCS": {"color": "#2A9D8F", "marker": "X"},
    }
    cva_styles = {
        "BIQAE": {"color": "#A23B72", "marker": "s"},
        "CABIQAE": {"color": "#1F6F8B", "marker": "o"},
        "DCS": {"color": "#2A9D8F", "marker": "X"},
    }

    return (
        CurveSpec(
            key="ideal-toy",
            output_basename="bae_biqae_cabiqae_latentt_ideal_rmse",
            sources=(
                SummarySource((IDEAL_TOY_DIR / "bae_biqae_cabiqae_latentt_ideal_rmse_summary.csv",)),
            ),
            algorithm_order=("BAE", "BIQAE", "CABIQAE", "DCS"),
            styles=ideal_styles,
            y_col="normalized_abs_error_median",
            y_low_col="normalized_abs_error_median_ci_low",
            y_high_col="normalized_abs_error_median_ci_high",
            ylabel="Median relative error",
            x_col="query_budget_actual_mean",
            x_fallback_cols=("query_budget_actual_median", "budget"),
            include_linear_guide=True,
            include_sqrt_guide=True,
            guide_mode="fit",
            x_margin_factor=1.25,
        ),
        CurveSpec(
            key="noiseless-cva",
            output_basename="noseless_cva_ae_cva_relative_error_vs_queries_paper",
            sources=(
                SummarySource((CVA_NOISELESS_DIR / "budget_summary.csv",)),
                SummarySource((CVA_NOISELESS_DIR / "montecarlo_budget_summary.csv",)),
            ),
            algorithm_order=("BIQAE", "CABIQAE", "DCS"),
            styles=cva_styles,
            y_col="processed_relative_error_median",
            y_low_col="processed_relative_error_median_ci_low",
            y_high_col="processed_relative_error_median_ci_high",
            ylabel="Median relative CVA error",
            include_linear_guide=True,
            include_sqrt_guide=True,
            guide_mode="leftmost",
            x_margin_factor=1.18,
        ),
        CurveSpec(
            key="hardware-toy",
            output_basename="hardware_replay_median_relative_error_paper",
            sources=(
                SummarySource((HARDWARE_TOY_DIR / "hardware_replay_median_relative_error_paper_summary.csv",)),
            ),
            algorithm_order=("BAE", "BIQAE", "CABIQAE", "DCS"),
            styles=hardware_toy_styles,
            y_col="normalized_abs_error_median",
            y_low_col="normalized_abs_error_median_ci_low",
            y_high_col="normalized_abs_error_median_ci_high",
            ylabel="Median relative error",
            include_linear_guide=True,
            include_sqrt_guide=True,
            guide_mode="fixed",
            fixed_guide_anchor_x=256.0,
            fixed_guide_anchor_y=0.052,
            y_limits=(1.5e-4, 1.5e-1),
            x_margin_factor=1.25,
        ),
        CurveSpec(
            key="hardware-cva",
            output_basename="hardware_replay_cva_budget",
            sources=(
                SummarySource((CVA_HARDWARE_DIR / "budget_summary.csv",)),
                SummarySource(
                    (CVA_HARDWARE_DCS_DIR / "montecarlo_budget_summary.csv",),
                    rename_algorithm={"Classical MC": "DCS"},
                ),
            ),
            algorithm_order=("BIQAE", "CABIQAE", "DCS"),
            styles=cva_styles,
            y_col="processed_relative_error_median",
            y_low_col="processed_relative_error_median_ci_low",
            y_high_col="processed_relative_error_median_ci_high",
            ylabel="Median relative CVA error",
            include_linear_guide=False,
            include_sqrt_guide=True,
            guide_mode="leftmost",
            max_x=100000.0,
            collapse_identical_points=True,
            x_margin_factor=1.18,
        ),
    )


def selected_specs(raw_plot_arg: str, specs: Sequence[CurveSpec]) -> tuple[CurveSpec, ...]:
    requested = {
        item.strip()
        for token in raw_plot_arg.split(",")
        for item in token.split()
        if item.strip()
    }
    if not requested or requested == {"all"}:
        return tuple(specs)
    by_key = {spec.key: spec for spec in specs}
    unknown = sorted(requested.difference(by_key))
    if unknown:
        raise ValueError(
            "Unknown plot key(s): "
            + ", ".join(unknown)
            + ". Available keys: "
            + ", ".join(by_key)
        )
    return tuple(by_key[key] for key in by_key if key in requested)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    specs = selected_specs(str(args.plot), curve_specs())
    output_dir = args.output_dir.resolve()
    written: list[Path] = []
    for spec in specs:
        output_base = plot_curve(
            spec,
            output_dir,
            sqrt_guide_coefficient=args.sqrt_guide_coefficient,
        )
        written.extend([output_base.with_suffix(".png"), output_base.with_suffix(".pdf")])
        print(f"[{spec.key}] saved {output_base.with_suffix('.png')}")
        print(f"[{spec.key}] saved {output_base.with_suffix('.pdf')}")

    index = pd.DataFrame(
        {
            "plot_key": [spec.key for spec in specs],
            "png": [str(output_dir / f"{spec.output_basename}.png") for spec in specs],
            "pdf": [str(output_dir / f"{spec.output_basename}.pdf") for spec in specs],
        }
    )
    index.to_csv(output_dir / "plot_index.csv", index=False)
    print(f"Wrote {len(written)} figure files to {output_dir}")


if __name__ == "__main__":
    main()
