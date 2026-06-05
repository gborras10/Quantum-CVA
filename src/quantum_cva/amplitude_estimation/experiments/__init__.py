"""Reusable amplitude-estimation experiment infrastructure.

These modules are intentionally problem-agnostic: demo circuits, CVA circuits,
and hardware replay data all enter through an :class:`AEProblemBundle`.
"""

from quantum_cva.amplitude_estimation.experiments.problems import (
    AEProblemBundle,
    bundle_from_problem,
    count_good_states,
    normalize_bitstring,
    true_amplitude,
)
from quantum_cva.amplitude_estimation.experiments.circuits import (
    build_reference_circuits,
    build_unmeasured_query_circuit,
    construct_measured_circuit,
)
from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
    build_cva_problem_bundle,
)

__all__ = [
    "AEProblemBundle",
    "build_6q_cva_problem_bundle",
    "build_cva_problem_bundle",
    "build_reference_circuits",
    "build_unmeasured_query_circuit",
    "bundle_from_problem",
    "construct_measured_circuit",
    "count_good_states",
    "normalize_bitstring",
    "true_amplitude",
]
