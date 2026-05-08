from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_common import DEFAULT_RUN_DIR, add_cva_aliases, parse_name_list
from quantum_cva.amplitude_estimation.experiments.io import (
    RunPaths,
    load_csv,
    save_csv,
)
from quantum_cva.amplitude_estimation.experiments.plotting import (
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    normalize_algorithm_key,
)
from quantum_cva.amplitude_estimation.experiments.statistics import as_float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot noisy 6q CVA amplitude-estimation outputs."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated algorithm keys. Defaults to algorithms in the CSVs.",
    )
    parser.add_argument(
        "--max-points-per-algorithm",
        type=int,
        default=14,
        help="Maximum points per algorithm to plot (use 0 or -1 for all).",
    )
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return add_cva_aliases(load_csv(path))


def _normalise_algorithms(
    algorithms: Sequence[str] | str | None,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...] | None:
    if algorithms is not None:
        return tuple(
            normalize_algorithm_key(name) for name in parse_name_list(algorithms)
        )
    keys = sorted(
        {
            str(row.get("algorithm_key", ""))
            for row in rows
            if row.get("algorithm_key")
        }
    )
    return tuple(keys) if keys else None


def _summary_projected_to_metric(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    ci_low_key = f"{metric_key}_ci_low"
    ci_high_key = f"{metric_key}_ci_high"
    se_key = (
        f"{metric_key.removesuffix('_median')}_se"
        if metric_key.endswith("_median")
        else f"{metric_key}_se"
    )
    for row in rows:
        value = as_float(row.get(metric_key))
        if not np.isfinite(value) or value <= 0.0:
            continue
        out = dict(row)
        out["normalized_abs_error_median"] = value
        out["normalized_abs_error_median_ci_low"] = as_float(row.get(ci_low_key))
        out["normalized_abs_error_median_ci_high"] = as_float(row.get(ci_high_key))
        out["normalized_abs_error_se"] = as_float(row.get(se_key), 0.0)
        projected.append(out)
    return projected


def _budget_rows_projected_to_metric(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        value = as_float(row.get(metric_key))
        if not np.isfinite(value) or value <= 0.0:
            continue
        out = dict(row)
        out["normalized_abs_error"] = value
        projected.append(out)
    return projected


def _budget_like_rows_from_final_rows(
    final_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for final in final_rows:
        queries = as_float(final.get("final_queries"))
        if not np.isfinite(queries) or queries <= 0.0:
            continue
        rows.append(
            {
                "run_kind": final.get("run_kind", "simulated_noise"),
                "repetition": final.get("repetition", 0),
                "algorithm": final.get("algorithm", ""),
                "algorithm_key": final.get("algorithm_key", ""),
                "budget": int(round(queries)),
                "query_budget_actual": queries,
                "runtime_wall_seconds": final.get("runtime_wall_seconds", np.nan),
                "normalized_abs_error": final.get(
                    "final_normalized_abs_error",
                    final.get("normalized_abs_error", np.nan),
                ),
                "processed_relative_error": final.get(
                    "processed_relative_error",
                    final.get("cva_relative_error", np.nan),
                ),
            }
        )
    return rows


def _plot_summary_pair(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    algorithms: Sequence[str] | None,
    title: str,
    ylabel: str,
    max_points_per_algorithm: int | None = 14,
) -> Path | None:
    if not summary_rows:
        return None
    # Normalize sentinel values: allow 0 or negative to mean "all points"
    mp = None if (max_points_per_algorithm is None or int(max_points_per_algorithm) <= 0) else int(max_points_per_algorithm)
    plot_budget_summary(
        summary_rows,
        output_path=output_path,
        algorithms=algorithms,
        algorithm_labels=ALGORITHM_LABELS,
        title=title,
        metric_key="normalized_abs_error_median",
        ylabel=ylabel,
        max_points_per_algorithm=mp,
    )
    return output_path


def make_plots(
    run_dir: str | Path,
    *,
    algorithms: Sequence[str] | str | None = None,
    max_points_per_algorithm: int | None = 14,
) -> list[Path]:
    paths = RunPaths(Path(run_dir))
    summary_rows = _load_rows(paths.budget_summary)
    actual_query_summary_rows = _load_rows(paths.run_dir / "actual_query_summary.csv")
    budget_rows = _load_rows(paths.run_dir / "budget_rows.csv")
    if not budget_rows:
        budget_rows = _load_rows(paths.replay_budget)
    final_rows = _load_rows(paths.direct_final)
    if not budget_rows:
        budget_rows = _budget_like_rows_from_final_rows(final_rows)

    algorithm_keys = _normalise_algorithms(
        algorithms,
        budget_rows or summary_rows or actual_query_summary_rows,
    )
    plots_dir = paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fixed_amp = _plot_summary_pair(
        summary_rows,
        output_path=plots_dir / "noisy_cva_ae_budget_amplitude_error_vs_queries.png",
        algorithms=algorithm_keys,
        title="6q CVA AE, simulated noise, fixed budgets",
        ylabel="Median relative amplitude error",
        max_points_per_algorithm=max_points_per_algorithm,
    )
    if fixed_amp is not None:
        written.append(fixed_amp)

    cva_summary_rows = _summary_projected_to_metric(
        summary_rows,
        "processed_relative_error_median",
    )
    fixed_cva = _plot_summary_pair(
        cva_summary_rows,
        output_path=plots_dir / "noisy_cva_ae_budget_cva_relative_error_vs_queries.png",
        algorithms=algorithm_keys,
        title="6q CVA AE, simulated noise, fixed budgets",
        ylabel="Median relative CVA error",
        max_points_per_algorithm=max_points_per_algorithm,
    )
    if fixed_cva is not None:
        written.append(fixed_cva)

    actual_amp = _plot_summary_pair(
        actual_query_summary_rows,
        output_path=plots_dir
        / "noisy_cva_ae_actual_query_bins_amplitude_error_vs_queries.png",
        algorithms=algorithm_keys,
        title="6q CVA AE, simulated noise, actual-query bins",
        ylabel="Median relative amplitude error",
        max_points_per_algorithm=max_points_per_algorithm,
    )
    if actual_amp is not None:
        written.append(actual_amp)

    actual_cva_summary_rows = _summary_projected_to_metric(
        actual_query_summary_rows,
        "processed_relative_error_median",
    )
    actual_cva = _plot_summary_pair(
        actual_cva_summary_rows,
        output_path=plots_dir
        / "noisy_cva_ae_actual_query_bins_cva_relative_error_vs_queries.png",
        algorithms=algorithm_keys,
        title="6q CVA AE, simulated noise, actual-query bins",
        ylabel="Median relative CVA error",
        max_points_per_algorithm=max_points_per_algorithm,
    )
    if actual_cva is not None:
        written.append(actual_cva)

    amp_runtime_plot = plots_dir / "noisy_cva_ae_final_amplitude_error_vs_runtime.png"
    amp_runtime_summary = paths.run_dir / "final_amplitude_error_vs_runtime_summary.csv"
    plot_final_runtime_scatter_from_budget_rows(
        budget_rows,
        output_path=amp_runtime_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        summary_path=amp_runtime_summary,
        x_kind="runtime",
        title="6q CVA AE final amplitude error, simulated noise",
    )
    written.append(amp_runtime_plot)

    amp_query_plot = plots_dir / "noisy_cva_ae_final_amplitude_error_vs_queries.png"
    amp_query_summary = paths.run_dir / "final_amplitude_error_vs_queries_summary.csv"
    plot_final_runtime_scatter_from_budget_rows(
        budget_rows,
        output_path=amp_query_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        summary_path=amp_query_summary,
        x_kind="queries",
        title="6q CVA AE final amplitude error, simulated noise",
    )
    written.append(amp_query_plot)

    cva_budget_rows = _budget_rows_projected_to_metric(
        budget_rows,
        "processed_relative_error",
    )
    cva_runtime_plot = plots_dir / "noisy_cva_ae_final_cva_error_vs_runtime.png"
    cva_runtime_summary = paths.run_dir / "final_cva_error_vs_runtime_summary.csv"
    plot_final_runtime_scatter_from_budget_rows(
        cva_budget_rows,
        output_path=cva_runtime_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        summary_path=cva_runtime_summary,
        x_kind="runtime",
        title="6q CVA AE final CVA error, simulated noise",
    )
    written.append(cva_runtime_plot)

    cva_query_plot = plots_dir / "noisy_cva_ae_final_cva_error_vs_queries.png"
    cva_query_summary = paths.run_dir / "final_cva_error_vs_queries_summary.csv"
    plot_final_runtime_scatter_from_budget_rows(
        cva_budget_rows,
        output_path=cva_query_plot,
        algorithms=algorithm_keys,
        algorithm_labels=ALGORITHM_LABELS,
        summary_path=cva_query_summary,
        x_kind="queries",
        title="6q CVA AE final CVA error, simulated noise",
    )
    written.append(cva_query_plot)

    plot_index = [
        {
            "plot_png": str(path),
            "plot_pdf": str(path.with_suffix(".pdf")),
        }
        for path in written
    ]
    save_csv(plot_index, paths.run_dir / "plot_index.csv")
    return written


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    algorithms = args.algorithms if args.algorithms else None
    written = make_plots(
        args.run_dir, algorithms=algorithms, max_points_per_algorithm=args.max_points_per_algorithm
    )
    print(f"Wrote {len(written)} plot families to {RunPaths(Path(args.run_dir)).plots_dir}")


if __name__ == "__main__":
    main()

