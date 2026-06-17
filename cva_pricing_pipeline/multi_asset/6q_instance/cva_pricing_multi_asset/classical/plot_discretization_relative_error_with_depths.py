from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    parent for parent in SCRIPT_PATH.parents if (parent / "pyproject.toml").exists()
)
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_BENCHMARK = (
    REPO_ROOT
    / "data"
    / "multi_asset"
    / "6q_instance"
    / "benchmark"
    / "three_asset_instance.npz"
)
DEFAULT_HARDWARE_REPORT = (
    REPO_ROOT
    / "cva_pricing_pipeline"
    / "multi_asset"
    / "6q_instance"
    / "cva_pricing_multi_asset"
    / "quantum"
    / "ae_cva"
    / "hardware"
    / "results"
    / "q_ctrl_hardware"
    / "transpilation_report.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "results" / "discretization_convergence"
DEFAULT_REFERENCE_BITS = 13
DEFAULT_MAX_PLOT_BITS = 10
TIME_QUBITS = 2

PALETTE = ["#1f4e79", "#8f3f2f", "#5b6f3a", "#4c4c4c", "#b07d2c"]

TRAINING_DIR = (
    REPO_ROOT / "data" / "multi_asset" / "6q_instance" / "quantum" / "training"
)
TRAINED_PARAMS = {
    "qcbm": (
        TRAINING_DIR
        / "qcbm"
        / "shots"
        / "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz"
    ),
    "exposure": (
        TRAINING_DIR
        / "crca"
        / "positive_exposure"
        / "training_heavy_hex_star_shots_backend_noise_snapshot.npz"
    ),
    "default": (
        TRAINING_DIR
        / "crca"
        / "default_probabilities"
        / "training_crca2_shots_backend_noise_snapshot.npz"
    ),
    "discount": (
        TRAINING_DIR
        / "crca"
        / "discount_factors"
        / "training_crca2_shots_backend_noise_snapshot.npz"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the two-asset CVA discretization error and export a table "
            "with explicitly labelled Q^1 A proxy circuit depths and 2Q counts."
        )
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-bits", type=int, default=DEFAULT_REFERENCE_BITS)
    parser.add_argument("--max-plot-bits", type=int, default=DEFAULT_MAX_PLOT_BITS)
    parser.add_argument(
        "--hardware-report",
        type=Path,
        default=DEFAULT_HARDWARE_REPORT,
        help="Hardware preflight CSV used to anchor the exact n=2 Q^1 A metrics.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#222222",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "lines.linewidth": 1.45,
            "patch.linewidth": 0.5,
        }
    )


def load_benchmark(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {path}. Run "
            "run_classical_cva-two_assets_instance.py first."
        )
    with np.load(path, allow_pickle=True) as payload:
        benchmark = {key: payload[key] for key in payload.files}

    required = {"grid_sizes", "cva_by_grid_size_values", "cva_limit", "M"}
    missing = sorted(required - set(benchmark))
    if missing:
        raise KeyError(f"Benchmark {path} is missing required keys: {missing}")
    return benchmark


def scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.shape == () else value


def load_trained_parameters() -> dict[str, np.ndarray]:
    parameters: dict[str, np.ndarray] = {}
    for name, path in TRAINED_PARAMS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing trained n=2 artifact: {path}")
        with np.load(path, allow_pickle=True) as payload:
            parameters[name] = np.asarray(payload["theta_star"], dtype=float).reshape(-1)
    return parameters


def representative_parameters(size: int, *, phase: float) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=float)
    return np.linspace(0.173 + phase, 1.219 + phase, int(size), dtype=float)


def build_cva_depth_proxy(
    asset_bits_per_underlying: int,
    trained_parameters: dict[str, np.ndarray],
) -> tuple[Any, str, str]:
    from qiskit_algorithms import EstimationProblem

    from quantum_cva.amplitude_estimation.experiments.circuits import (
        construct_measured_circuit,
    )
    from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (
        QuantumCVACircuit,
    )
    from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
        CrcaCircuit,
    )
    from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
        MLQcbmCircuit,
    )

    n = int(asset_bits_per_underlying)
    underlying_qubits = 2 * n
    state_qubits = TIME_QUBITS + underlying_qubits
    qcbm_topology = "qcbm_heavyhex6" if state_qubits == 6 else "linear"

    qcbm = MLQcbmCircuit(
        n_qubits=state_qubits,
        n_layers=6,
        entangler="rzz",
        topology=qcbm_topology,
        optimization_level=1,
    )
    exposure = CrcaCircuit(
        m_time=TIME_QUBITS,
        n_price=underlying_qubits,
        n_layers=2,
        ansatz_type="heavy_hex_star",
        name="crca_exposure_proxy",
    )
    default = CrcaCircuit(
        m_time=TIME_QUBITS,
        n_price=0,
        n_layers=1,
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_proxy",
    )
    discount = CrcaCircuit(
        m_time=TIME_QUBITS,
        n_price=0,
        n_layers=1,
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_discount_proxy",
    )

    if n == 2:
        parameter_source = "trained_n2_artifacts"
        qcbm_params = trained_parameters["qcbm"]
        exposure_params = trained_parameters["exposure"]
        default_params = trained_parameters["default"]
        discount_params = trained_parameters["discount"]
    else:
        parameter_source = "representative_nonzero_proxy"
        qcbm_params = representative_parameters(qcbm.n_params, phase=0.00)
        exposure_params = representative_parameters(exposure.n_params, phase=0.11)
        default_params = representative_parameters(default.n_params, phase=0.23)
        discount_params = representative_parameters(discount.n_params, phase=0.37)

    cva = QuantumCVACircuit(
        num_qubits_time=TIME_QUBITS,
        num_qubits_underlying=underlying_qubits,
        qcbm_circuit=qcbm,
        crca_circuit_exposure=exposure,
        crca_circuit_default_prob=default,
        crca_circuit_discount_factor=discount,
        recovery_rate=0.0,
        C_v=1.0,
        C_q=1.0,
        C_p=1.0,
    )
    state_preparation = cva.build_cva_circuit(
        qcbm_params=qcbm_params,
        crca_exposure_params=exposure_params,
        crca_default_params=default_params,
        crca_discount_params=discount_params,
        measured=False,
    )
    objective_qubits = [state_qubits, state_qubits + 1, state_qubits + 2]
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=objective_qubits,
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "") == "111",
    )
    q1a_circuit = construct_measured_circuit(
        problem,
        1,
        source="discretization_depth_q1a_proxy",
    )
    return q1a_circuit, qcbm_topology, parameter_source


def load_hardware_anchor(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"Hardware transpilation report not found: {report_path}")
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["grover_power"]) == 1]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one grover_power=1 row in {report_path}; got {len(matches)}."
        )
    row = matches[0]
    return {
        "q1a_logical_depth": int(row["logical_depth"]),
        "q1a_logical_2q_gates": int(row["logical_2q"]),
        "q1a_decomposed_depth": int(row["decomposed_depth"]),
        "q1a_decomposed_2q_gates": int(row["decomposed_2q"]),
        "q1a_isa_depth": int(row["isa_depth"]),
        "q1a_isa_2q_gates": int(row["isa_2q"]),
    }


def scaled_metric(anchor_value: int, proxy_value: int, proxy_anchor: int) -> int:
    if int(proxy_anchor) <= 0:
        raise ValueError("Proxy anchor metric must be positive.")
    return int(round(float(anchor_value) * float(proxy_value) / float(proxy_anchor)))


def depth_rows(
    grid_bits: np.ndarray,
    *,
    hardware_anchor: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    from quantum_cva.amplitude_estimation.experiments.circuits import circuit_metrics

    trained_parameters = load_trained_parameters()
    structural_rows: dict[int, dict[str, Any]] = {}
    required_bits = sorted({int(bits) for bits in grid_bits} | {2})
    for n in required_bits:
        print(f"[depth] building structural Q^1 A proxy for n={n}", flush=True)
        q1a_logical, qcbm_topology, parameter_source = build_cva_depth_proxy(
            n, trained_parameters
        )
        logical_metrics = circuit_metrics(q1a_logical)
        decomposed_metrics = circuit_metrics(q1a_logical.decompose(reps=10))
        structural_rows[n] = {
            "q1a_circuit_qubits": int(q1a_logical.num_qubits),
            "qcbm_topology_proxy": qcbm_topology,
            "proxy_parameter_source": parameter_source,
            "q1a_structural_logical_depth": int(logical_metrics["depth"]),
            "q1a_structural_logical_2q_gates": int(
                logical_metrics["two_qubit_gates"]
            ),
            "q1a_structural_decomposed_depth": int(decomposed_metrics["depth"]),
            "q1a_structural_decomposed_2q_gates": int(
                decomposed_metrics["two_qubit_gates"]
            ),
        }

    proxy_anchor = structural_rows[2]
    rows: dict[int, dict[str, Any]] = {}
    for bits in grid_bits:
        n = int(bits)
        structural = structural_rows[n]
        if n == 2:
            metrics = dict(hardware_anchor)
            source = "ibm_pittsburgh_preflight_exact"
            is_exact = True
        else:
            metrics = {
                "q1a_logical_depth": scaled_metric(
                    hardware_anchor["q1a_logical_depth"],
                    structural["q1a_structural_logical_depth"],
                    proxy_anchor["q1a_structural_logical_depth"],
                ),
                "q1a_logical_2q_gates": scaled_metric(
                    hardware_anchor["q1a_logical_2q_gates"],
                    structural["q1a_structural_logical_2q_gates"],
                    proxy_anchor["q1a_structural_logical_2q_gates"],
                ),
                "q1a_decomposed_depth": scaled_metric(
                    hardware_anchor["q1a_decomposed_depth"],
                    structural["q1a_structural_decomposed_depth"],
                    proxy_anchor["q1a_structural_decomposed_depth"],
                ),
                "q1a_decomposed_2q_gates": scaled_metric(
                    hardware_anchor["q1a_decomposed_2q_gates"],
                    structural["q1a_structural_decomposed_2q_gates"],
                    proxy_anchor["q1a_structural_decomposed_2q_gates"],
                ),
                "q1a_isa_depth": scaled_metric(
                    hardware_anchor["q1a_isa_depth"],
                    structural["q1a_structural_decomposed_depth"],
                    proxy_anchor["q1a_structural_decomposed_depth"],
                ),
                "q1a_isa_2q_gates": scaled_metric(
                    hardware_anchor["q1a_isa_2q_gates"],
                    structural["q1a_structural_decomposed_2q_gates"],
                    proxy_anchor["q1a_structural_decomposed_2q_gates"],
                ),
            }
            source = "structural_ratio_projection_anchored_to_n2"
            is_exact = False
        rows[n] = {
            **structural,
            **metrics,
            "q1a_metrics_source": source,
            "q1a_metrics_are_exact": is_exact,
        }
    return rows


def convergence_rows(
    benchmark: dict[str, Any],
    *,
    reference_bits: int,
    max_plot_bits: int,
    depths: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    grid_bits = np.asarray(benchmark["grid_sizes"], dtype=int).reshape(-1)
    grid_cva = np.asarray(
        benchmark["cva_by_grid_size_values"], dtype=float
    ).reshape(-1)
    if grid_bits.shape != grid_cva.shape:
        raise ValueError(
            "grid_sizes and cva_by_grid_size_values must align: "
            f"{grid_bits.shape} != {grid_cva.shape}"
        )

    mask = grid_bits <= int(max_plot_bits)
    selected_bits = grid_bits[mask]
    selected_cva = grid_cva[mask]
    if not np.any(selected_bits == int(max_plot_bits)):
        raise ValueError(
            f"Benchmark must include n={max_plot_bits} to reproduce the figure."
        )

    cva_reference = float(scalar(benchmark["cva_limit"]))
    if cva_reference == 0.0:
        raise ValueError("CVA reference is zero; relative error is undefined.")

    n_underlyings = int(np.asarray(benchmark.get("n_bits_small", [2, 2])).size)
    if n_underlyings != 2:
        raise ValueError(
            f"This report is for the two-asset instance, got {n_underlyings} assets."
        )

    rows: list[dict[str, Any]] = []
    for bits, cva in zip(selected_bits, selected_cva):
        n = int(bits)
        total_asset_qubits = n_underlyings * n
        absolute_error = float(abs(float(cva) - cva_reference))
        relative_error = float(absolute_error / abs(cva_reference))
        row: dict[str, Any] = {
            "time_qubits_m": TIME_QUBITS,
            "asset_qubits_per_underlying_n": n,
            "x_axis_n_plus_m": total_asset_qubits + TIME_QUBITS,
            "total_asset_qubits": total_asset_qubits,
            "joint_asset_grid_cells": int((2**n) ** n_underlyings),
            "cva_n": float(cva),
            "cva_reference_n13": cva_reference,
            "reference_bits_per_underlying": int(reference_bits),
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "relative_error_pct": 100.0 * relative_error,
        }
        row.update(depths.get(n, {}))
        rows.append(row)
    return rows


def save_plot(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    configure_plot_style()
    x = np.asarray([row["x_axis_n_plus_m"] for row in rows], dtype=int)
    pct_err = np.asarray([row["relative_error_pct"] for row in rows], dtype=float)
    qubits_per_point = rows[0]["total_asset_qubits"] // rows[0][
        "asset_qubits_per_underlying_n"
    ]
    reference_total_asset_qubits = (
        qubits_per_point * rows[0]["reference_bits_per_underlying"]
    )

    fig, ax = plt.subplots(figsize=(4.7, 2.55))
    ax.plot(
        x,
        pct_err,
        marker="o",
        markersize=3.8,
        linewidth=1.05,
        color=PALETTE[1],
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="Relative error",
    )
    ax.set_xlabel(r"$n + m$")
    ax.set_ylabel(
        rf"$100\,|\mathrm{{CVA}}_n-\mathrm{{CVA}}_{{{reference_total_asset_qubits}}}|"
        rf"/|\mathrm{{CVA}}_{{{reference_total_asset_qubits}}}|$",
        fontsize=8.5,
    )
    ax.set_yscale("linear")
    ax.set_xticks(x)
    ax.minorticks_on()
    ax.grid(True, axis="both", which="major", color="0.78", linewidth=0.55)
    ax.grid(True, axis="y", which="minor", color="0.90", linewidth=0.35)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=3.0,
        width=0.7,
        labelsize=8,
    )
    ax.tick_params(
        axis="both", which="minor", direction="in", length=1.8, width=0.5
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.6)
    fig.tight_layout(pad=0.35)

    paths = []
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"figure_discretization_relative_error.{extension}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "asset_qubits_per_underlying_n",
        "x_axis_n_plus_m",
        "joint_asset_grid_cells",
        "cva_n",
        "cva_reference_n13",
        "relative_error_pct",
        "q1a_circuit_qubits",
        "q1a_logical_depth",
        "q1a_logical_2q_gates",
        "q1a_isa_depth",
        "q1a_isa_2q_gates",
        "q1a_metrics_source",
    ]
    lines = [
        (
            "Caption: For n=2, the Q^1 A logical and ISA metrics are read "
            "directly from the ibm_pittsburgh hardware preflight report. For "
            "n != 2, no trained circuits exist; the reported values are "
            "structural projections obtained by scaling the n=2 hardware "
            "metrics with the growth of same-family Q^1 A proxy circuits."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(format_cell(row.get(column, "")) for column in columns) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_latex(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        ("asset_qubits_per_underlying_n", "$n$"),
        ("x_axis_n_plus_m", "$n+m$"),
        ("joint_asset_grid_cells", "Grid cells"),
        ("cva_n", "$\\mathrm{CVA}_n$"),
        ("relative_error_pct", "Rel. error (\\%)"),
        ("q1a_logical_depth", "$Q^1A$ logical depth"),
        ("q1a_logical_2q_gates", "$Q^1A$ logical 2Q gates"),
        ("q1a_isa_depth", "$Q^1A$ ISA depth"),
        ("q1a_isa_2q_gates", "$Q^1A$ ISA 2Q gates"),
    ]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{For $n=2$, the $Q^1A$ logical and ISA metrics are read "
        "directly from the \\texttt{ibm\\_pittsburgh} hardware preflight "
        "report. For $n\\neq2$, no trained circuits exist; the reported values "
        "are structural projections obtained by scaling the $n=2$ hardware "
        "metrics with the growth of same-family $Q^1A$ proxy circuits.}",
        "\\label{tab:cva-discretization-q1a-resources}",
        "\\begin{tabular}{rrrrrrrrr}",
        "\\hline",
        " & ".join(label for _, label in columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(format_cell(row.get(key, "")) for key, _ in columns)
            + " \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(args.benchmark.resolve())
    grid_bits = np.asarray(benchmark["grid_sizes"], dtype=int).reshape(-1)
    plotted_bits = grid_bits[grid_bits <= int(args.max_plot_bits)]

    hardware_report = args.hardware_report.resolve()
    hardware_anchor = load_hardware_anchor(hardware_report)
    depths = depth_rows(plotted_bits, hardware_anchor=hardware_anchor)

    rows = convergence_rows(
        benchmark,
        reference_bits=args.reference_bits,
        max_plot_bits=args.max_plot_bits,
        depths=depths,
    )
    figure_paths = save_plot(rows, output_dir)
    csv_path = output_dir / "table_discretization_relative_error_with_depths.csv"
    markdown_path = output_dir / "table_discretization_relative_error_with_depths.md"
    latex_path = output_dir / "table_discretization_relative_error_with_depths.tex"
    metadata_path = output_dir / "report_metadata.json"
    save_csv(rows, csv_path)
    save_markdown(rows, markdown_path)
    save_latex(rows, latex_path)

    metadata = {
        "benchmark_path": str(args.benchmark.resolve()),
        "output_dir": str(output_dir),
        "reference_definition": (
            f"abs(CVA(n) - CVA(n={args.reference_bits})) / "
            f"abs(CVA(n={args.reference_bits}))"
        ),
        "cva_reference": float(scalar(benchmark["cva_limit"])),
        "reference_bits_per_underlying": int(args.reference_bits),
        "max_plotted_bits_per_underlying": int(args.max_plot_bits),
        "time_qubits": TIME_QUBITS,
        "x_axis_definition": (
            "n+m, where n is the total number of underlying qubits "
            "(n=n1+n2) and m is the number of time qubits."
        ),
        "depth_method": (
            "For n=2, the Q^1 A logical and ISA metrics are read directly from "
            "the ibm_pittsburgh hardware preflight report. For n!=2, no trained "
            "circuits exist: same-family Q^1 A proxy circuits are built and "
            "decomposed with reps=10, then every reported metric is projected "
            "from its exact n=2 hardware anchor using the corresponding "
            "structural-growth ratio. The QCBM proxy uses the trained "
            "qcbm_heavyhex6 topology at n=2 and an explicit linear extension "
            "outside the trained point."
        ),
        "hardware_anchor_report": str(hardware_report),
        "hardware_anchor_n2": hardware_anchor,
        "figures": [str(path) for path in figure_paths],
        "tables": [str(csv_path), str(markdown_path), str(latex_path)],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[done] wrote figure: {figure_paths[0]}")
    print(f"[done] wrote table: {csv_path}")
    print(f"[done] wrote metadata: {metadata_path}")
    print(
        "[note] n=2 metrics are exact hardware-preflight values; "
        "n!=2 metrics are anchored structural projections."
    )


if __name__ == "__main__":
    main()
