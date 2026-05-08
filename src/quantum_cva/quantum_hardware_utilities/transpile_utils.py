from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from qiskit import QuantumCircuit, qasm3
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


DEFAULT_TRANSPILER_SEEDS: tuple[int, ...] = (1234,)


@dataclass(frozen=True)
class LayoutCandidate:
    initial_layout: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class FixedTranspilationPlan:
    initial_layout: tuple[int, ...]
    seed_transpiler: int
    optimization_level: int
    routing_method: str | None
    candidate_source: str
    aggregate_swap_count: int
    aggregate_two_qubit_gates: int
    aggregate_depth: int
    aggregate_size: int
    evaluated_layouts: int
    evaluated_plans: int
    reference_circuit_count: int

    def build_pass_manager(self, backend: Any) -> Any:
        return generate_preset_pass_manager(
            backend=backend,
            optimization_level=self.optimization_level,
            initial_layout=list(self.initial_layout),
            routing_method=self.routing_method,
            seed_transpiler=self.seed_transpiler,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "initial_layout": list(self.initial_layout),
            "seed_transpiler": int(self.seed_transpiler),
            "optimization_level": int(self.optimization_level),
            "routing_method": self.routing_method,
            "candidate_source": self.candidate_source,
            "aggregate_swap_count": int(self.aggregate_swap_count),
            "aggregate_two_qubit_gates": int(self.aggregate_two_qubit_gates),
            "aggregate_depth": int(self.aggregate_depth),
            "aggregate_size": int(self.aggregate_size),
            "evaluated_layouts": int(self.evaluated_layouts),
            "evaluated_plans": int(self.evaluated_plans),
            "reference_circuit_count": int(self.reference_circuit_count),
        }


def stable_circuit_key(circuit: QuantumCircuit) -> str:
    return qasm3.dumps(circuit)


def _reference_label(circuit: QuantumCircuit, index: int) -> str:
    metadata = getattr(circuit, "metadata", None) or {}
    grover_power = metadata.get("grover_power")
    if grover_power is not None:
        return f"k={int(grover_power)}"
    amplification_factor = metadata.get("amplification_factor")
    if amplification_factor is not None:
        return f"K={int(amplification_factor)}"
    return f"ref={int(index)}"


def transpilation_metrics(circuit: QuantumCircuit) -> dict[str, int]:
    ops = circuit.count_ops()
    two_qubit_gates = sum(1 for inst in circuit.data if len(inst.qubits) == 2)
    return {
        "swap_count": int(ops.get("swap", 0)),
        "two_qubit_gates": int(two_qubit_gates),
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size()),
    }


def extract_initial_layout(
    transpiled_circuit: QuantumCircuit,
    *,
    logical_qubit_count: int,
) -> tuple[int, ...]:
    layout = getattr(transpiled_circuit, "layout", None)
    if layout is None or not hasattr(layout, "initial_index_layout"):
        raise ValueError("Transpiled circuit does not expose an initial layout.")

    indices = list(layout.initial_index_layout())
    if len(indices) < logical_qubit_count:
        raise ValueError(
            "Initial layout is shorter than the number of logical qubits: "
            f"{len(indices)} < {logical_qubit_count}."
        )

    initial_layout = tuple(int(idx) for idx in indices[:logical_qubit_count])
    if len(set(initial_layout)) != logical_qubit_count:
        raise ValueError(f"Invalid initial layout with repeated qubits: {initial_layout}.")

    return initial_layout


def collect_sabre_layout_candidates(
    backend: Any,
    reference_circuits: Sequence[QuantumCircuit],
    *,
    optimization_level: int = 3,
    routing_method: str | None = "sabre",
    seed_candidates: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    verbose: bool = True,
) -> list[LayoutCandidate]:
    num_qubits = _validate_reference_circuits(reference_circuits)
    candidates: dict[tuple[int, ...], LayoutCandidate] = {}
    if verbose:
        print(
            "[pass_manager] discovering SABRE layouts: "
            f"{len(seed_candidates)} seeds x {len(reference_circuits)} reference circuits",
            flush=True,
        )

    for seed in seed_candidates:
        if verbose:
            print(f"[pass_manager][discovery] seed={int(seed)}: building SABRE pass manager", flush=True)
        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
            layout_method="sabre",
            routing_method=routing_method,
            seed_transpiler=int(seed),
        )
        for ref_idx, circuit in enumerate(reference_circuits):
            label = _reference_label(circuit, ref_idx)
            if verbose:
                print(
                    "[pass_manager][discovery] "
                    f"seed={int(seed)} {label} ({ref_idx + 1}/{len(reference_circuits)}): transpiling",
                    flush=True,
                )
            transpiled = pm.run(circuit)
            initial_layout = extract_initial_layout(
                transpiled,
                logical_qubit_count=num_qubits,
            )
            if verbose:
                metrics = transpilation_metrics(transpiled)
                print(
                    "[pass_manager][discovery] "
                    f"seed={int(seed)} {label}: "
                    f"layout={initial_layout}, swaps={metrics['swap_count']}, "
                    f"2q={metrics['two_qubit_gates']}, depth={metrics['depth']}",
                    flush=True,
                )
            candidates.setdefault(
                initial_layout,
                LayoutCandidate(
                    initial_layout=initial_layout,
                    source=f"sabre_discovery(seed={int(seed)}, ref={int(ref_idx)})",
                ),
            )

    if verbose:
        print(f"[pass_manager] discovered {len(candidates)} unique layout candidates", flush=True)
    return list(candidates.values())


def select_best_fixed_transpilation_plan(
    backend: Any,
    reference_circuits: Sequence[QuantumCircuit],
    *,
    candidate_layouts: Iterable[LayoutCandidate | tuple[Sequence[int], str] | Sequence[int]] = (),
    optimization_level: int = 3,
    routing_method: str | None = "sabre",
    discovery_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    evaluation_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    include_sabre_candidates: bool = True,
    verbose: bool = True,
) -> FixedTranspilationPlan:
    num_qubits = _validate_reference_circuits(reference_circuits)
    normalized_candidates = _normalize_candidate_layouts(candidate_layouts, num_qubits)

    if include_sabre_candidates:
        for candidate in collect_sabre_layout_candidates(
            backend,
            reference_circuits,
            optimization_level=optimization_level,
            routing_method=routing_method,
            seed_candidates=discovery_seeds,
            verbose=verbose,
        ):
            normalized_candidates.setdefault(candidate.initial_layout, candidate)

    if not normalized_candidates:
        raise RuntimeError("No transpilation layout candidates were produced.")

    best_score: tuple[int, ...] | None = None
    best_plan: FixedTranspilationPlan | None = None
    evaluated_plans = 0
    layout_count = len(normalized_candidates)
    if verbose:
        print(
            "[pass_manager] evaluating fixed layouts: "
            f"{layout_count} layouts x {len(evaluation_seeds)} seeds x "
            f"{len(reference_circuits)} reference circuits",
            flush=True,
        )

    for layout_idx, (initial_layout, candidate) in enumerate(normalized_candidates.items(), start=1):
        layout_span = max(initial_layout) - min(initial_layout)

        for seed in evaluation_seeds:
            evaluated_plans += 1
            aggregate_metrics = {
                "swap_count": 0,
                "two_qubit_gates": 0,
                "depth": 0,
                "size": 0,
            }

            pm = generate_preset_pass_manager(
                backend=backend,
                optimization_level=optimization_level,
                initial_layout=list(initial_layout),
                routing_method=routing_method,
                seed_transpiler=int(seed),
            )

            for ref_idx, circuit in enumerate(reference_circuits):
                label = _reference_label(circuit, ref_idx)
                if verbose:
                    print(
                        "[pass_manager][eval] "
                        f"layout={layout_idx}/{layout_count} seed={int(seed)} "
                        f"{label} ({ref_idx + 1}/{len(reference_circuits)}): transpiling",
                        flush=True,
                    )
                transpiled = pm.run(circuit)
                metrics = transpilation_metrics(transpiled)
                for key, value in metrics.items():
                    aggregate_metrics[key] += int(value)

            score = (
                int(aggregate_metrics["swap_count"]),
                int(aggregate_metrics["two_qubit_gates"]),
                int(aggregate_metrics["depth"]),
                int(aggregate_metrics["size"]),
                int(layout_span),
                int(sum(initial_layout)),
                int(seed),
            )
            if verbose:
                print(
                    "[pass_manager][eval] "
                    f"plan={evaluated_plans}: swaps={aggregate_metrics['swap_count']}, "
                    f"2q={aggregate_metrics['two_qubit_gates']}, "
                    f"depth={aggregate_metrics['depth']}, size={aggregate_metrics['size']}",
                    flush=True,
                )

            if best_score is not None and score >= best_score:
                continue

            best_score = score
            best_plan = FixedTranspilationPlan(
                initial_layout=initial_layout,
                seed_transpiler=int(seed),
                optimization_level=int(optimization_level),
                routing_method=routing_method,
                candidate_source=str(candidate.source),
                aggregate_swap_count=int(aggregate_metrics["swap_count"]),
                aggregate_two_qubit_gates=int(aggregate_metrics["two_qubit_gates"]),
                aggregate_depth=int(aggregate_metrics["depth"]),
                aggregate_size=int(aggregate_metrics["size"]),
                evaluated_layouts=int(len(normalized_candidates)),
                evaluated_plans=int(evaluated_plans),
                reference_circuit_count=int(len(reference_circuits)),
            )
            if verbose:
                print(
                    "[pass_manager][eval] "
                    f"new best plan={evaluated_plans}: layout={initial_layout}, "
                    f"seed={int(seed)}, swaps={aggregate_metrics['swap_count']}, "
                    f"2q={aggregate_metrics['two_qubit_gates']}, "
                    f"depth={aggregate_metrics['depth']}",
                    flush=True,
                )

    if best_plan is None:
        raise RuntimeError("Failed to select a fixed transpilation plan.")

    return best_plan


def _validate_reference_circuits(reference_circuits: Sequence[QuantumCircuit]) -> int:
    if not reference_circuits:
        raise ValueError("At least one reference circuit is required.")

    num_qubits = int(reference_circuits[0].num_qubits)
    for idx, circuit in enumerate(reference_circuits):
        if int(circuit.num_qubits) != num_qubits:
            raise ValueError(
                "All reference circuits must have the same number of qubits. "
                f"Mismatch at index {idx}: {circuit.num_qubits} != {num_qubits}."
            )

    return num_qubits


def _normalize_layout(layout: Sequence[int], num_qubits: int) -> tuple[int, ...]:
    initial_layout = tuple(int(qubit) for qubit in layout)
    if len(initial_layout) != num_qubits:
        raise ValueError(
            "Initial layout has the wrong length: "
            f"{len(initial_layout)} != {num_qubits}."
        )
    if len(set(initial_layout)) != num_qubits:
        raise ValueError(f"Initial layout contains repeated physical qubits: {initial_layout}.")
    return initial_layout


def _normalize_candidate_layouts(
    candidate_layouts: Iterable[LayoutCandidate | tuple[Sequence[int], str] | Sequence[int]],
    num_qubits: int,
) -> dict[tuple[int, ...], LayoutCandidate]:
    normalized: dict[tuple[int, ...], LayoutCandidate] = {}

    for candidate in candidate_layouts:
        if isinstance(candidate, LayoutCandidate):
            initial_layout = _normalize_layout(candidate.initial_layout, num_qubits)
            normalized.setdefault(initial_layout, LayoutCandidate(initial_layout, candidate.source))
            continue

        if (
            isinstance(candidate, tuple)
            and len(candidate) == 2
            and isinstance(candidate[1], str)
        ):
            initial_layout = _normalize_layout(candidate[0], num_qubits)
            normalized.setdefault(initial_layout, LayoutCandidate(initial_layout, candidate[1]))
            continue

        initial_layout = _normalize_layout(candidate, num_qubits)
        normalized.setdefault(initial_layout, LayoutCandidate(initial_layout, "external_candidate"))

    return normalized
