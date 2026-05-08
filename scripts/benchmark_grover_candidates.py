from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms import EstimationProblem


def _bootstrap_repo() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


REPO_ROOT = _bootstrap_repo()

from quantum_cva.amplitude_estimation.experiments.cva import (  # noqa: E402
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.problems import (  # noqa: E402
    bundle_from_problem,
)
from quantum_cva.amplitude_estimation.grover import (  # noqa: E402
    GroverCandidate,
    build_grover_candidate,
    circuit_metrics,
    good_probability_from_statevector,
    validate_grover_amplification,
)
from quantum_cva.amplitude_estimation.grover import (  # noqa: E402
    candidates as grover_candidates,
)


DEFAULT_CANDIDATES: tuple[GroverCandidate, ...] = (
    "qiskit_standard",
    "qiskit_standard_clean",
    "custom_full",
    "custom_full_flat",
)


def _parse_int_list(raw: str) -> list[int]:
    values = [
        int(token)
        for token in raw.replace(";", ",").replace(" ", ",").split(",")
        if token.strip()
    ]
    if not values or any(value < 0 for value in values):
        raise ValueError("Powers must be a non-empty list of non-negative integers.")
    return values


def _parse_candidates(raw: str) -> list[GroverCandidate]:
    values = [
        token.strip()
        for token in raw.replace(";", ",").replace(" ", ",").split(",")
        if token.strip()
    ]
    if not values:
        raise ValueError("At least one candidate is required.")
    allowed = set(DEFAULT_CANDIDATES)
    bad = [value for value in values if value not in allowed]
    if bad:
        raise ValueError(f"Unsupported candidates: {bad}.")
    return [value for value in values]  # type: ignore[list-item]


def _synthetic_bundle():
    state_preparation = QuantumCircuit(3, name="synthetic_A")
    state_preparation.ry(0.72, 0)
    state_preparation.ry(0.51, 1)
    state_preparation.ry(0.65, 2)
    state_preparation.cx(0, 2)
    state_preparation.ry(-0.21, 2)
    state_preparation.cx(0, 2)
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=[0, 1, 2],
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "") == "111",
    )
    problem.grover_operator = problem.grover_operator
    return bundle_from_problem(
        problem,
        target_name="synthetic",
        good_bitstring="111",
        metadata={"source": "synthetic"},
    )


def _load_cva_bundle() -> Any:
    config_path = (
        REPO_ROOT
        / "cva_pricing_pipeline"
        / "multi_asset"
        / "6q_instance"
        / "full_cva_pipeline.py"
    )
    if not config_path.exists():
        print("[warn] CVA config not found; using synthetic problem.")
        return _synthetic_bundle()
    spec = importlib.util.spec_from_file_location("full_cva_pipeline_6q", config_path)
    if spec is None or spec.loader is None:
        print("[warn] CVA config cannot be loaded; using synthetic problem.")
        return _synthetic_bundle()
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return build_6q_cva_problem_bundle(module.CONFIG, repo_root=REPO_ROOT)
    except Exception as exc:
        print(f"[warn] CVA bundle unavailable ({exc}); using synthetic problem.")
        return _synthetic_bundle()


def _load_backend(name: str, runtime_channel: str) -> tuple[Any, str]:
    if str(name).lower() in {"none", "aer", "local"}:
        return _basis_fallback_backend(), "aer_basis_fallback"

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel=str(runtime_channel))
        return service.backend(str(name)), str(name)
    except Exception as exc:
        print(f"[warn] Could not load IBM backend {name!r}: {exc}")

    fake_name = "FakeBrisbane"
    try:
        import qiskit_ibm_runtime.fake_provider as fake_provider

        fake_cls = getattr(fake_provider, fake_name)
        print(f"[warn] Falling back to {fake_name}.")
        return fake_cls(), fake_name
    except Exception as exc:
        print(f"[warn] Could not load fake backend: {exc}")

    print("[warn] Falling back to Aer basis backend.")
    return _basis_fallback_backend(), "aer_basis_fallback"


def _basis_fallback_backend() -> AerSimulator:
    return AerSimulator(
        basis_gates=["id", "rz", "rx", "sx", "x", "cx", "cz", "rzz"],
    )


def _candidate_state_preparation(
    state_preparation: QuantumCircuit,
    candidate: GroverCandidate,
) -> QuantumCircuit:
    if candidate == "qiskit_standard_clean":
        return grover_candidates._clean_state_preparation(state_preparation)
    return state_preparation


def _build_query(
    state_preparation: QuantumCircuit,
    grover_operator: QuantumCircuit,
    k: int,
    candidate: GroverCandidate,
) -> QuantumCircuit:
    circuit = QuantumCircuit(state_preparation.num_qubits, name=f"{candidate}_k_{k}")
    circuit.compose(state_preparation, inplace=True)
    if int(k) <= 0:
        return circuit
    if candidate == "custom_full_flat":
        for _ in range(int(k)):
            circuit.compose(grover_operator, inplace=True)
    else:
        circuit.compose(grover_operator.power(int(k)), inplace=True)
    return circuit


def _active_physical_qubits(circuit: QuantumCircuit) -> list[int]:
    return sorted(
        {
            int(circuit.find_bit(qubit).index)
            for instruction in circuit.data
            for qubit in instruction.qubits
        }
    )


def _layout_indices(circuit: QuantumCircuit, logical_qubits: int) -> list[int] | None:
    layout = getattr(circuit, "layout", None)
    if layout is None or not hasattr(layout, "initial_index_layout"):
        return None
    try:
        indices = list(layout.initial_index_layout())
    except Exception:
        return None
    if len(indices) < int(logical_qubits):
        return None
    return [int(index) for index in indices[: int(logical_qubits)]]


def _ops(circuit: QuantumCircuit) -> dict[str, int]:
    return {str(name): int(count) for name, count in circuit.count_ops().items()}


def _twoq_total(circuit: QuantumCircuit) -> int:
    return int(
        sum(
            1
            for instruction in circuit.data
            if int(instruction.operation.num_qubits) == 2
        )
    )


def _target_summary(backend: Any) -> dict[str, Any]:
    out = {"backend_class": type(backend).__name__}
    try:
        out["backend_name"] = str(backend.name)
    except Exception:
        out["backend_name"] = type(backend).__name__
    try:
        out["basis_gates"] = list(getattr(backend, "operation_names", []))
    except Exception:
        out["basis_gates"] = []
    return out


def _p_ideal(a_true: float, k: int) -> float:
    theta = float(np.arcsin(np.sqrt(float(np.clip(a_true, 0.0, 1.0)))))
    return float(np.sin((2 * int(k) + 1) * theta) ** 2)


def _contrast_fields(
    *,
    row: dict[str, Any],
    baseline_cost: float | None,
    t_eff_base: float | None,
    cost_kind: str,
    a_true: float,
) -> dict[str, float]:
    key = (
        "depth_per_grover_step"
        if str(cost_kind) == "depth"
        else "twoq_per_grover_step"
    )
    cost = float(row[key])
    if cost <= 0.0 or not math.isfinite(cost):
        return {
            "relative_contrast_budget": float("nan"),
            "t_eff_candidate": float("nan"),
            "p_ideal": _p_ideal(a_true, int(row["k"])),
            "p_obs": float("nan"),
        }
    relative = 1.0 / cost
    p_ideal = _p_ideal(a_true, int(row["k"]))
    if baseline_cost is None or t_eff_base is None or baseline_cost <= 0.0:
        return {
            "relative_contrast_budget": float(relative),
            "t_eff_candidate": float("nan"),
            "p_ideal": float(p_ideal),
            "p_obs": float("nan"),
        }
    t_eff = float(t_eff_base) * float(baseline_cost) / cost
    p_obs = 0.5 + math.exp(-(2 * int(row["k"]) + 1) / t_eff) * (p_ideal - 0.5)
    return {
        "relative_contrast_budget": float(relative),
        "t_eff_candidate": float(t_eff),
        "p_ideal": float(p_ideal),
        "p_obs": float(p_obs),
    }


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No benchmark rows to write.")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _rank(rows: list[dict[str, Any]], powers: Sequence[int]) -> list[dict[str, Any]]:
    max_power = max(int(power) for power in powers)
    candidates = sorted({str(row["candidate"]) for row in rows})
    ranked: list[dict[str, Any]] = []
    standard = next(
        (
            row
            for row in rows
            if row["candidate"] == "qiskit_standard" and int(row["k"]) == max_power
        ),
        None,
    )
    standard_depth = float(standard["transpiled_depth"]) if standard else float("nan")
    standard_twoq = float(standard["transpiled_twoq_total"]) if standard else float("nan")
    for candidate in candidates:
        row = next(
            row
            for row in rows
            if row["candidate"] == candidate and int(row["k"]) == max_power
        )
        depth = float(row["transpiled_depth"])
        twoq = float(row["transpiled_twoq_total"])
        ranked.append(
            {
                "candidate": candidate,
                "k": max_power,
                "correctness_passed": bool(row["correctness_passed"]),
                "transpiled_depth": int(depth),
                "transpiled_twoq_total": int(twoq),
                "depth_reduction_vs_standard": float(standard_depth - depth),
                "twoq_reduction_vs_standard": float(standard_twoq - twoq),
                "relative_contrast_budget": float(row["relative_contrast_budget"]),
            }
        )
    ranked.sort(
        key=lambda item: (
            not item["correctness_passed"],
            -item["depth_reduction_vs_standard"],
            -item["twoq_reduction_vs_standard"],
            -item["relative_contrast_budget"],
        )
    )
    return ranked


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark validated CVA Grover candidates."
    )
    parser.add_argument("--powers", default="0,1,2,3,4")
    parser.add_argument("--backend", default="ibm_basquecountry")
    parser.add_argument("--runtime-channel", default="ibm_cloud")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=1234)
    parser.add_argument(
        "--candidate",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated candidates.",
    )
    parser.add_argument("--t-eff-base", type=float, default=None)
    parser.add_argument(
        "--t-eff-base-candidate",
        default="qiskit_standard",
        choices=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--cost-kind", choices=("depth", "twoq"), default="twoq")
    parser.add_argument(
        "--output-csv",
        type=pathlib.Path,
        default=REPO_ROOT / "results" / "grover_candidates" / "metrics.csv",
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=REPO_ROOT / "results" / "grover_candidates" / "summary.json",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    powers = _parse_int_list(args.powers)
    candidates = _parse_candidates(args.candidate)
    bundle = _load_cva_bundle()
    backend, backend_label = _load_backend(args.backend, args.runtime_channel)
    state_preparation = bundle.problem.state_preparation
    objective_qubits = list(bundle.problem.objective_qubits)
    good_bitstring = str(bundle.good_bitstring or "111")
    a_true = good_probability_from_statevector(
        state_preparation,
        objective_qubits,
        good_bitstring,
    )

    print(f"Problem source: {bundle.metadata.get('source', bundle.target_name)}")
    print(f"Backend: {backend_label}")
    print(f"a_true: {a_true:.12f}")

    rows: list[dict[str, Any]] = []
    baseline_by_candidate: dict[str, dict[str, float]] = {}

    for candidate in candidates:
        candidate_state = _candidate_state_preparation(state_preparation, candidate)
        grover = build_grover_candidate(
            state_preparation,
            objective_qubits,
            good_bitstring,
            candidate=candidate,
        )
        validation = validate_grover_amplification(
            candidate_state,
            grover,
            objective_qubits,
            a_true,
            powers,
            good_bitstring,
            atol=1e-8,
        )

        for k in powers:
            query = _build_query(candidate_state, grover, int(k), candidate)
            logical = circuit_metrics(query)
            transpiled = transpile(
                query,
                backend=backend,
                optimization_level=int(args.optimization_level),
                seed_transpiler=int(args.seed_transpiler),
            )
            transpiled_ops = _ops(transpiled)
            transpiled_depth = int(transpiled.depth() or 0)
            transpiled_size = int(transpiled.size())
            transpiled_twoq = _twoq_total(transpiled)
            if int(k) == 0:
                baseline_by_candidate[str(candidate)] = {
                    "depth": float(transpiled_depth),
                    "twoq": float(transpiled_twoq),
                }
            base = baseline_by_candidate.get(str(candidate), {})
            depth_per_step = (
                (float(transpiled_depth) - float(base.get("depth", transpiled_depth)))
                / max(int(k), 1)
            )
            twoq_per_step = (
                (float(transpiled_twoq) - float(base.get("twoq", transpiled_twoq)))
                / max(int(k), 1)
            )
            row = {
                "candidate": str(candidate),
                "k": int(k),
                "correctness_passed": bool(validation["passed"]),
                "logical_depth": int(logical["depth"]),
                "logical_size": int(logical["size"]),
                "logical_twoq_total": int(logical["two_qubit_gates"]),
                "logical_multiq_total": int(logical["multi_qubit_gates"]),
                "logical_ops": json.dumps(logical["ops"], sort_keys=True),
                "transpiled_depth": int(transpiled_depth),
                "transpiled_size": int(transpiled_size),
                "transpiled_twoq_total": int(transpiled_twoq),
                "cz_count": int(transpiled_ops.get("cz", 0)),
                "rzz_count": int(transpiled_ops.get("rzz", 0)),
                "cx_count": int(transpiled_ops.get("cx", 0)),
                "ecr_count": int(transpiled_ops.get("ecr", 0)),
                "swap_count": int(transpiled_ops.get("swap", 0)),
                "transpiled_ops": json.dumps(transpiled_ops, sort_keys=True),
                "initial_layout": json.dumps(
                    _layout_indices(transpiled, query.num_qubits)
                ),
                "active_physical_qubits": json.dumps(
                    _active_physical_qubits(transpiled)
                ),
                "optimization_level": int(args.optimization_level),
                "seed_transpiler": int(args.seed_transpiler),
                "backend": str(backend_label),
                "target_summary": json.dumps(_target_summary(backend), sort_keys=True),
                "depth_per_grover_step": float(depth_per_step),
                "twoq_per_grover_step": float(twoq_per_step),
            }
            rows.append(row)

    base_cost = None
    base_row = next(
        (
            row
            for row in rows
            if row["candidate"] == str(args.t_eff_base_candidate)
            and int(row["k"]) == max(powers)
        ),
        None,
    )
    if base_row is not None:
        base_cost = (
            float(base_row["depth_per_grover_step"])
            if args.cost_kind == "depth"
            else float(base_row["twoq_per_grover_step"])
        )

    for row in rows:
        row.update(
            _contrast_fields(
                row=row,
                baseline_cost=base_cost,
                t_eff_base=args.t_eff_base,
                cost_kind=args.cost_kind,
                a_true=a_true,
            )
        )

    ranking = _rank(rows, powers)
    _write_csv(args.output_csv, rows)
    _write_json(
        args.output_json,
        {
            "backend": backend_label,
            "powers": powers,
            "candidates": candidates,
            "a_true": a_true,
            "ranking": ranking,
        },
    )

    print("\nRanking at max k:")
    for item in ranking:
        print(
            f"  {item['candidate']:<22} "
            f"ok={item['correctness_passed']} "
            f"depth={item['transpiled_depth']} "
            f"2q={item['transpiled_twoq_total']} "
            f"rel_contrast={item['relative_contrast_budget']:.6g}"
        )
    print(f"\nCSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
