from __future__ import annotations

import argparse
import csv
import math
import pathlib
from collections import defaultdict
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EXPECTED_STATEVECTOR_EVALUATION_METHODOLOGY = "run_ideal_cva_logical_statevector_v1"
EXPECTED_CLASSICAL_BENCHMARK_METHODOLOGY = "legacy_right_endpoint_small_grid_n_bits_2_v1"
EXPECTED_RELATIVE_ERROR_REFERENCE = "classical_small_grid_n_bits_2_per_scenario"
FAMILY_COLORS = {
    "baseline": "#252525",
    "call_strike": "#1b9e77",
    "put_strike": "#7570b3",
    "volatility": "#d95f02",
    "interest_rate": "#1f78b4",
    "default_spread": "#e31a1c",
    "combined": "#6a3d9a",
}


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else math.nan


def _validate_results_methodology(rows: Sequence[dict[str, str]]) -> None:
    expected = {
        "statevector_evaluation_methodology": EXPECTED_STATEVECTOR_EVALUATION_METHODOLOGY,
        "classical_benchmark_methodology": EXPECTED_CLASSICAL_BENCHMARK_METHODOLOGY,
        "relative_error_reference": EXPECTED_RELATIVE_ERROR_REFERENCE,
    }
    for row in rows:
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    "Results were generated with an obsolete robustness methodology. "
                    "Rerun run_ideal_cva_robustness.py with --profile focused "
                    "--force --overwrite-results before plotting."
                )
        for field in (
            "cva_statevector",
            "cva_classical_small_grid_n_bits_2",
            "absolute_relative_error_vs_reference_pct",
        ):
            if row.get(field) in ("", None):
                raise ValueError(
                    f"Results are missing required field {field!r}. Rerun "
                    "run_ideal_cva_robustness.py with --profile focused "
                    "--force --overwrite-results before plotting."
                )


def _save(fig: plt.Figure, plots_dir: pathlib.Path, stem: str) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{stem}.png", dpi=240)
    fig.savefig(plots_dir / f"{stem}.pdf")
    plt.close(fig)


def _short_labels(rows: Sequence[dict[str, str]]) -> list[str]:
    return [row["case_id"] for row in rows]


def _set_case_ticks(ax: plt.Axes, rows: Sequence[dict[str, str]]) -> None:
    x = np.arange(len(rows))
    if len(rows) <= 35:
        ax.set_xticks(x)
        ax.set_xticklabels(_short_labels(rows), rotation=70, ha="right", fontsize=7)
    else:
        ax.set_xlabel(f"Scenario catalog index ({len(rows)} cases)")


def _scenario_code(index: int) -> str:
    return f"S{index}"


def _scenario_group(row: dict[str, str]) -> str:
    case_id = row.get("case_id", "")
    if case_id == "base":
        return "Baseline"
    if case_id.startswith(("call_k_", "put_k_", "both_k_")):
        return "Strike perturbations"
    if case_id.startswith("vol_"):
        return "Volatility perturbations"
    if case_id.startswith("rate_"):
        return "Interest-rate perturbations"
    if case_id.startswith("default_"):
        return "Default-spread perturbations"
    return "Combined stress scenarios"


def _scenario_group_short(row: dict[str, str]) -> str:
    return {
        "Baseline": "Base",
        "Strike perturbations": "Strikes",
        "Volatility perturbations": "Volatility",
        "Interest-rate perturbations": "Rates",
        "Default-spread perturbations": "CDS",
        "Combined stress scenarios": "Joint",
    }[_scenario_group(row)]


def _scenario_blocks(rows: Sequence[dict[str, str]]) -> list[tuple[int, int, str]]:
    if not rows:
        return []
    blocks: list[tuple[int, int, str]] = []
    start = 1
    current_group = _scenario_group_short(rows[0])
    for index, row in enumerate(rows[1:], start=2):
        group = _scenario_group_short(row)
        if group != current_group:
            blocks.append((start, index - 1, current_group))
            start = index
            current_group = group
    blocks.append((start, len(rows), current_group))
    return blocks


def _format_scale(value: float) -> str:
    return f"{value:.2f}"


def _scenario_math_perturbation(row: dict[str, str]) -> str:
    terms = []
    call_scale = _number(row, "call_strike_scale")
    put_scale = _number(row, "put_strike_scale")
    volatility_scale = _number(row, "volatility_scale")
    rate_shift_bps = _number(row, "rate_shift_bps")
    default_spread_scale = _number(row, "default_spread_scale")
    if not math.isclose(call_scale, 1.0):
        terms.append(
            rf"$K_1\mapsto {_format_scale(call_scale)}K_1$"
        )
    if not math.isclose(put_scale, 1.0):
        terms.append(
            rf"$K_2\mapsto {_format_scale(put_scale)}K_2$"
        )
    if not math.isclose(volatility_scale, 1.0):
        terms.append(rf"$\sigma\mapsto {_format_scale(volatility_scale)}\sigma$")
    if not math.isclose(rate_shift_bps, 0.0):
        sign = "+" if rate_shift_bps > 0.0 else "-"
        terms.append(rf"$r(t)\mapsto r(t){sign}{abs(rate_shift_bps):.0f}\,\mathrm{{bp}}$")
    if not math.isclose(default_spread_scale, 1.0):
        terms.append(
            rf"$s_{{\mathrm{{CDS}}}}\mapsto {_format_scale(default_spread_scale)}s_{{\mathrm{{CDS}}}}$"
        )
    return "; ".join(terms) if terms else "Baseline configuration"


def _latex_text(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _chunked(items: Sequence[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [list(items[index : index + chunk_size]) for index in range(0, len(items), chunk_size)]


def _column_blocks(items: Sequence[dict[str, Any]], n_columns: int) -> list[list[dict[str, Any]]]:
    block_size = math.ceil(len(items) / n_columns)
    return [
        list(items[index : index + block_size])
        for index in range(0, len(items), block_size)
    ]


def _latex_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return r"\textemdash{}"
    return f"{value:.{digits}f}"


def _latex_signed_float(value: float, digits: int = 1) -> str:
    if math.isnan(value):
        return r"\textemdash{}"
    return f"{value:+.{digits}f}"


def plot_relative_error_by_case(rows: Sequence[dict[str, str]], plots_dir: pathlib.Path) -> None:
    ordered = list(rows)
    x = np.arange(1, len(ordered) + 1)
    absolute_relative_error = np.asarray(
        [_number(row, "absolute_relative_error_vs_reference_pct") for row in ordered],
        dtype=float,
    )
    labels = [_scenario_code(index) for index in x]

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(
        x,
        absolute_relative_error,
        color="#153e75",
        marker="o",
        markersize=4.3,
        linewidth=1.35,
        markerfacecolor="#153e75",
        markeredgecolor="white",
        markeredgewidth=0.55,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(r"Relative error (%)")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, len(ordered) + 0.5)
    y_max = max(float(np.nanmax(absolute_relative_error)), 1e-12)
    ax.set_ylim(0.0, 1.12 * y_max)
    for start, end, group in _scenario_blocks(ordered):
        if start > 1:
            ax.axvline(start - 0.5, color="#bdbdbd", linewidth=0.55, zorder=1)
        if len(ordered) <= 35:
            ax.text(
                0.5 * (start + end),
                1.025,
                group,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.1,
                color="#4d4d4d",
            )
    _save(fig, plots_dir, "ideal_cva_absolute_relative_error_by_scenario")


def write_absolute_relative_error_table(
    rows: Sequence[dict[str, str]],
    tables_dir: pathlib.Path,
) -> None:
    values = []
    for index, row in enumerate(rows, start=1):
        values.append(
            {
                "scenario": _scenario_code(index),
                "case_id": row["case_id"],
                "cva_statevector": _number(row, "cva_statevector"),
                "cva_classical_small_grid_n_bits_2": _number(
                    row,
                    "cva_classical_small_grid_n_bits_2",
                ),
                "absolute_relative_error_pct": _number(
                    row,
                    "absolute_relative_error_vs_reference_pct",
                ),
            }
        )
    fields = [
        "scenario",
        "case_id",
        "cva_statevector",
        "cva_classical_small_grid_n_bits_2",
        "absolute_relative_error_pct",
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "ideal_cva_absolute_relative_error_by_scenario.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    (tables_dir / "ideal_cva_absolute_relative_error_by_scenario.md").write_text(
        _markdown_table(values, fields),
        encoding="utf-8",
    )


def write_scenario_definition_table(
    rows: Sequence[dict[str, str]],
    tables_dir: pathlib.Path,
) -> None:
    definitions = []
    for index, row in enumerate(rows, start=1):
        definitions.append(
            {
                "scenario": _scenario_code(index),
                "case_id": row["case_id"],
                "description": row["label"],
                "call_strike_scale": row["call_strike_scale"],
                "put_strike_scale": row["put_strike_scale"],
                "volatility_scale": row["volatility_scale"],
                "rate_shift_bps": row["rate_shift_bps"],
                "default_spread_scale": row["default_spread_scale"],
            }
        )
    fields = [
        "scenario",
        "case_id",
        "description",
        "call_strike_scale",
        "put_strike_scale",
        "volatility_scale",
        "rate_shift_bps",
        "default_spread_scale",
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "ideal_cva_scenario_definitions.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(definitions)
    (tables_dir / "ideal_cva_scenario_definitions.md").write_text(
        _markdown_table(definitions, fields),
        encoding="utf-8",
    )
    caption = (
        "Figure caption. Absolute relative error of the exact ideal-statevector "
        "CVA estimate with respect to the classical small-grid CVA across "
        f"the {len(definitions)} robustness scenarios indexed as S1, S2, .... "
        "Scenario definitions are reported "
        "in the accompanying scenario-definition table.\n"
    )
    (tables_dir / "ideal_cva_absolute_relative_error_caption.md").write_text(
        caption,
        encoding="utf-8",
    )
    tex_lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        (
            r"Scenario & Case ID & Description & Call $K$ scale & Put $K$ scale "
            r"& Vol. scale & Rate shift (bp) & CDS scale \\"
        ),
        r"\midrule",
    ]
    for row in definitions:
        escaped = {
            key: str(value).replace("_", r"\_").replace("%", r"\%")
            for key, value in row.items()
        }
        tex_lines.append(
            f"{escaped['scenario']} & {escaped['case_id']} & {escaped['description']} & "
            f"{escaped['call_strike_scale']} & {escaped['put_strike_scale']} & "
            f"{escaped['volatility_scale']} & {escaped['rate_shift_bps']} & "
            f"{escaped['default_spread_scale']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables_dir / "ideal_cva_scenario_definitions.tex").write_text(
        "\n".join(tex_lines),
        encoding="utf-8",
    )


def write_scenario_summary_table(
    rows: Sequence[dict[str, str]],
    tables_dir: pathlib.Path,
) -> None:
    summary = []
    for index, row in enumerate(rows, start=1):
        summary.append(
            {
                "scenario": _scenario_code(index),
                "group": _scenario_group(row),
                "case_id": row["case_id"],
                "description": row["label"],
                "mathematical_perturbation": _scenario_math_perturbation(row),
            }
        )
    fields = [
        "scenario",
        "group",
        "case_id",
        "description",
        "mathematical_perturbation",
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "ideal_cva_scenario_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    (tables_dir / "ideal_cva_scenario_summary.md").write_text(
        _markdown_table(summary, fields),
        encoding="utf-8",
    )
    tex_lines = [
        r"\begin{tabular}{llllp{0.48\textwidth}}",
        r"\toprule",
        r"Scenario & Group & Case ID & Description & Perturbation \\",
        r"\midrule",
    ]
    for row in summary:
        tex_lines.append(
            f"{_latex_text(row['scenario'])} & "
            f"{_latex_text(row['group'])} & "
            f"{_latex_text(row['case_id'])} & "
            f"{_latex_text(row['description'])} & "
            f"{row['mathematical_perturbation']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables_dir / "ideal_cva_scenario_summary.tex").write_text(
        "\n".join(tex_lines),
        encoding="utf-8",
    )


def write_compact_scenario_correspondence_table(
    rows: Sequence[dict[str, str]],
    tables_dir: pathlib.Path,
) -> None:
    scenarios = [
        {
            "scenario": _scenario_code(index),
            "perturbation": _scenario_math_perturbation(row),
        }
        for index, row in enumerate(rows, start=1)
    ]
    lines = [
        r"\begin{table}[H]",
        r"\scriptsize",
        r"\centering",
        r"\caption{Scenario definitions for the ideal robustness analysis.}",
        r"\label{tab:ideal_cva_scenario_correspondence}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\makebox[\textwidth][c]{%",
        r"\begin{tabular}{@{}lp{0.25\textwidth}lp{0.25\textwidth}lp{0.25\textwidth}@{}}",
        r"\toprule",
        r"\rowcolor{gray!10}",
        (
            r"\textbf{Scenario} & \textbf{Perturbation} & "
            r"\textbf{Scenario} & \textbf{Perturbation} & "
            r"\textbf{Scenario} & \textbf{Perturbation} \\"
        ),
        r"\midrule",
    ]
    blocks = _column_blocks(scenarios, 3)
    row_count = max(len(block) for block in blocks)
    for row_index in range(row_count):
        cells = []
        for block in blocks:
            if row_index < len(block):
                item = block[row_index]
                cells.extend([_latex_text(item["scenario"]), item["perturbation"]])
            else:
                cells.extend(["", ""])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    (tables_dir / "ideal_cva_scenario_correspondence_compact.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_compact_robustness_results_table(
    rows: Sequence[dict[str, str]],
    tables_dir: pathlib.Path,
) -> None:
    results = [
        {
            "scenario": _scenario_code(index),
            "cva_classical": _number(row, "cva_classical_small_grid_n_bits_2"),
            "cva_statevector": _number(row, "cva_statevector"),
            "relative_error": _number(row, "relative_error_vs_reference_pct"),
        }
        for index, row in enumerate(rows, start=1)
    ]
    lines = [
        r"\begin{table}[H]",
        r"\scriptsize",
        r"\centering",
        (
            r"\caption{Ideal robustness-test CVA results. The tabulated benchmark "
            r"is the scenario-specific classical small-grid CVA with \(n=2\), and "
            r"the statevector value is evaluated using the ideal trained quantum "
            r"pipeline.}"
        ),
        r"\label{tab:ideal_cva_robustness_results_compact}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\aboverulesep}{0pt}",
        r"\setlength{\belowrulesep}{0pt}",
        r"\makebox[\textwidth][c]{%",
        r"\begin{tabular}{@{}lrrrlrrrlrrr@{}}",
        r"\toprule",
        r"\rowcolor{gray!10}",
        (
            r"\multicolumn{4}{c}{\textbf{Scenarios S1--S8}} & "
            r"\multicolumn{4}{c}{\textbf{Scenarios S9--S16}} & "
            r"\multicolumn{4}{c@{}}{\textbf{Scenarios S17--S22}} \\"
        ),
        r"\cmidrule(lr){1-4}\cmidrule(lr){5-8}\cmidrule(l){9-12}",
        r"\rowcolor{gray!10}",
        (
            r"\textbf{Sc.} & \textbf{\(\mathrm{CVA}_{\mathrm{tab}}\)} "
            r"& \textbf{\(\mathrm{CVA}_{\mathrm{SV}}\)} "
            r"& \textbf{\(\varepsilon_{\mathrm{rel}}\)(\%)} & "
            r"\textbf{Sc.} & \textbf{\(\mathrm{CVA}_{\mathrm{tab}}\)} "
            r"& \textbf{\(\mathrm{CVA}_{\mathrm{SV}}\)} "
            r"& \textbf{\(\varepsilon_{\mathrm{rel}}\)(\%)} & "
            r"\textbf{Sc.} & \textbf{\(\mathrm{CVA}_{\mathrm{tab}}\)} "
            r"& \textbf{\(\mathrm{CVA}_{\mathrm{SV}}\)} "
            r"& \multicolumn{1}{r@{}}{\textbf{\(\varepsilon_{\mathrm{rel}}\)(\%)}} \\"
        ),
        r"\midrule",
    ]
    blocks = _column_blocks(results, 3)
    row_count = max(len(block) for block in blocks)
    for row_index in range(row_count):
        cells = []
        for block in blocks:
            if row_index < len(block):
                item = block[row_index]
                cells.extend(
                    [
                        _latex_text(item["scenario"]),
                        _latex_float(item["cva_classical"], digits=3),
                        _latex_float(item["cva_statevector"], digits=3),
                        _latex_signed_float(item["relative_error"], digits=1),
                    ]
                )
            else:
                cells.extend(["", "", "", ""])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    (tables_dir / "ideal_cva_robustness_results_compact.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def plot_one_factor_relative_error_sensitivity(
    rows: Sequence[dict[str, str]],
    plots_dir: pathlib.Path,
) -> None:
    factor_specs = [
        ("call_strike_scale", "call_strike", "Call strike scale"),
        ("put_strike_scale", "put_strike", "Put strike scale"),
        ("volatility_scale", "volatility", "Volatility scale"),
        ("rate_shift_bps", "interest_rate", "Rate shift (bp)"),
        ("default_spread_scale", "default_spread", "CDS spread scale"),
    ]
    fig, axes = plt.subplots(1, len(factor_specs), figsize=(18, 4.5), sharey=True)
    for ax, (factor, family, xlabel) in zip(axes, factor_specs):
        selected = [
            row
            for row in rows
            if row["family"] in {"baseline", family}
        ]
        selected = sorted(selected, key=lambda row: _number(row, factor))
        x = np.asarray([_number(row, factor) for row in selected], dtype=float)
        reference_error = np.asarray(
            [_number(row, "relative_error_vs_reference_pct") for row in selected],
            dtype=float,
        )
        encoded_error = np.asarray(
            [_number(row, "relative_error_vs_encoded_target_pct") for row in selected],
            dtype=float,
        )
        ax.axhline(0.0, color="#252525", linewidth=0.8)
        ax.plot(x, reference_error, "o-", color=FAMILY_COLORS[family], label="vs classical reference")
        ax.plot(x, encoded_error, "D--", color="#636363", markersize=4, label="vs encoded target")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.7)
    axes[0].set_ylabel("Signed relative error (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Ideal CVA robustness: one-factor sensitivity slices", fontsize=15)
    _save(fig, plots_dir, "ideal_cva_one_factor_relative_error_sensitivity")


def plot_overview(rows: Sequence[dict[str, str]], plots_dir: pathlib.Path) -> None:
    ordered = sorted(rows, key=lambda row: abs(_number(row, "relative_error_vs_reference_pct")))
    labels = _short_labels(ordered)
    reference = np.asarray([_number(row, "cva_reference_grid_2") for row in ordered])
    estimated = np.asarray([_number(row, "cva_statevector") for row in ordered])
    relative = np.asarray([_number(row, "relative_error_vs_reference_pct") for row in ordered])
    absolute_encoded = np.asarray([_number(row, "absolute_error_vs_encoded_target") for row in ordered])
    x = np.arange(len(ordered))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(x, reference, "o-", label="Classical 2-bit reference", markersize=3)
    axes[0, 0].plot(x, estimated, "s-", label="Exact statevector CVA", markersize=3)
    axes[0, 0].set_ylabel("CVA")
    axes[0, 0].set_title("Classical reference vs exact ideal statevector")
    axes[0, 0].legend()

    colors = np.where(relative >= 0.0, "#b2182b", "#2166ac")
    axes[0, 1].bar(x, relative, color=colors)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Signed relative error (%)")
    axes[0, 1].set_title("Statevector error vs classical 2-bit reference")

    axes[1, 0].bar(x, absolute_encoded, color="#4d9221")
    axes[1, 0].set_ylabel("Absolute CVA error")
    axes[1, 0].set_title("Representation error vs encoded target")

    family_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family_values[row["family"]].append(abs(_number(row, "relative_error_vs_reference_pct")))
    families = sorted(family_values)
    means = [float(np.nanmean(family_values[family])) for family in families]
    maxima = [float(np.nanmax(family_values[family])) for family in families]
    y = np.arange(len(families))
    axes[1, 1].barh(y, maxima, color="#fdae61", label="max")
    axes[1, 1].barh(y, means, color="#2b83ba", label="mean")
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels(families)
    axes[1, 1].set_xlabel("Absolute relative error (%)")
    axes[1, 1].set_title("Error envelope by stress family")
    axes[1, 1].legend()

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        if len(labels) <= 35:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        else:
            ax.set_xlabel(f"{len(labels)} scenarios, ordered by absolute relative error")
    _save(fig, plots_dir, "ideal_cva_robustness_overview")


def plot_component_errors(rows: Sequence[dict[str, str]], plots_dir: pathlib.Path) -> None:
    ordered = sorted(rows, key=lambda row: row["case_id"])
    labels = _short_labels(ordered)
    x = np.arange(len(ordered))
    metrics = [
        ("qcbm_kl", "QCBM KL divergence"),
        ("default_mse", "Default CRCA MSE"),
        ("discount_mse", "Discount CRCA MSE"),
        ("exposure_mse", "Exposure CRCA MSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        values = np.asarray([_number(row, key) for row in ordered], dtype=float)
        ax.bar(x, values, color="#5e3c99")
        ax.set_title(title)
        ax.set_yscale("log")
        if len(labels) <= 35:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
        else:
            ax.set_xlabel(f"{len(labels)} scenarios")
    _save(fig, plots_dir, "ideal_cva_component_errors")


def plot_vol_rate_heatmap(rows: Sequence[dict[str, str]], plots_dir: pathlib.Path) -> None:
    selected = [
        row
        for row in rows
        if math.isclose(_number(row, "call_strike_scale"), 1.0)
        and math.isclose(_number(row, "put_strike_scale"), 1.0)
        and math.isclose(_number(row, "default_spread_scale"), 1.0)
    ]
    vols = sorted({_number(row, "volatility_scale") for row in selected})
    rates = sorted({_number(row, "rate_shift_bps") for row in selected})
    if len(vols) < 2 or len(rates) < 2:
        return
    matrix = np.full((len(vols), len(rates)), np.nan)
    for row in selected:
        i = vols.index(_number(row, "volatility_scale"))
        j = rates.index(_number(row, "rate_shift_bps"))
        matrix[i, j] = _number(row, "relative_error_vs_reference_pct")
    max_abs = max(float(np.nanmax(np.abs(matrix))), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([f"{rate:g}" for rate in rates])
    ax.set_yticks(range(len(vols)))
    ax.set_yticklabels([f"{vol:g}" for vol in vols])
    ax.set_xlabel("Parallel interest-rate shift (bp)")
    ax.set_ylabel("Volatility scale")
    ax.set_title("Ideal statevector relative error: volatility-rate slice")
    fig.colorbar(image, ax=ax, label="Signed relative error (%)")
    _save(fig, plots_dir, "ideal_cva_volatility_rate_heatmap")


def _markdown_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def write_ranked_error_table(rows: Sequence[dict[str, str]], tables_dir: pathlib.Path) -> None:
    ranked = sorted(rows, key=lambda row: abs(_number(row, "relative_error_vs_reference_pct")), reverse=True)
    fields = [
        "case_id",
        "family",
        "cva_reference_grid_2",
        "cva_statevector",
        "relative_error_vs_reference_pct",
        "qcbm_kl",
        "default_mse",
        "discount_mse",
        "exposure_mse",
    ]
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "ideal_cva_ranked_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    (tables_dir / "ideal_cva_ranked_errors.md").write_text(
        _markdown_table(ranked, fields),
        encoding="utf-8",
    )
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Case & Family & Classical CVA & Statevector CVA & Rel. error (\%) & Exposure MSE \\",
        r"\midrule",
    ]
    for row in ranked:
        case = row["case_id"].replace("_", r"\_")
        family = row["family"].replace("_", r"\_")
        lines.append(
            f"{case} & {family} & "
            f"{_number(row, 'cva_reference_grid_2'):.6f} & "
            f"{_number(row, 'cva_statevector'):.6f} & "
            f"{_number(row, 'relative_error_vs_reference_pct'):.4f} & "
            f"{_number(row, 'exposure_mse'):.3e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (tables_dir / "ideal_cva_ranked_errors.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_analysis_outputs(
    output_dir: pathlib.Path,
    *,
    include_diagnostics: bool = False,
) -> None:
    results_path = output_dir / "results" / "ideal_cva_robustness_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")
    rows = [row for row in _read_csv(results_path) if row.get("status") == "ok"]
    if not rows:
        raise ValueError("No successful robustness rows are available for plotting.")
    _apply_paper_style()
    _validate_results_methodology(rows)
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plot_relative_error_by_case(rows, plots_dir)
    write_absolute_relative_error_table(rows, tables_dir)
    write_scenario_definition_table(rows, tables_dir)
    write_scenario_summary_table(rows, tables_dir)
    write_compact_scenario_correspondence_table(rows, tables_dir)
    write_compact_robustness_results_table(rows, tables_dir)
    if include_diagnostics:
        plot_one_factor_relative_error_sensitivity(rows, plots_dir)
        plot_overview(rows, plots_dir)
        plot_component_errors(rows, plots_dir)
        plot_vol_rate_heatmap(rows, plots_dir)
    write_ranked_error_table(rows, tables_dir)
    print(f"[OK] Plots and ranked tables created under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots and tables for ideal CVA robustness results.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "analysis_output"))
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also create the detailed diagnostic plots in addition to the single publication figure.",
    )
    args = parser.parse_args()
    generate_analysis_outputs(
        pathlib.Path(args.output_dir).resolve(),
        include_diagnostics=bool(args.include_diagnostics),
    )


if __name__ == "__main__":
    main()
