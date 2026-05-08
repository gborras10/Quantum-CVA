from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem


def normalize_bitstring(value: object, width: int | None = None) -> str:
    """Normalize sampler bitstrings to a compact binary string.

    Qiskit counts can include spaces between classical registers or hexadecimal
    keys. The returned string is left-padded to ``width`` when supplied.
    """
    text = str(value).replace(" ", "").strip()
    if text.startswith(("0x", "0X")):
        digits = max(1, int(width or 1))
        return format(int(text, 16), f"0{digits}b")
    if width is not None:
        return text.zfill(int(width))
    return text


def objective_width(problem: EstimationProblem) -> int:
    return int(len(problem.objective_qubits))


def default_good_bitstring(problem: EstimationProblem) -> str:
    return "1" * objective_width(problem)


def make_good_state(good_bitstring: str) -> Callable[[str], bool]:
    width = len(str(good_bitstring))
    expected = normalize_bitstring(good_bitstring, width)

    def _is_good_state(bitstring: str) -> bool:
        return normalize_bitstring(bitstring, width) == expected

    return _is_good_state


def ensure_good_state(
    problem: EstimationProblem,
    good_bitstring: str | None,
) -> str:
    """Install an explicit good-state predicate on ``problem``.

    Qiskit's default is "all objective bits are one"; making the predicate
    explicit avoids silent regressions when moving from one objective qubit to
    CVA's three objective ancillas.
    """
    width = objective_width(problem)
    good = default_good_bitstring(problem) if good_bitstring is None else str(good_bitstring)
    good = normalize_bitstring(good, width)
    if len(good) != width:
        raise ValueError(
            "good_bitstring length must match objective qubit count: "
            f"{len(good)} != {width}."
        )
    problem.is_good_state = make_good_state(good)
    return good


def true_amplitude(
    problem: EstimationProblem,
    good_bitstring: str | None = None,
) -> float:
    """Return the exact good-state probability of ``problem.state_preparation``."""
    good = ensure_good_state(problem, good_bitstring)
    state = Statevector.from_instruction(problem.state_preparation)
    probabilities = state.probabilities_dict(qargs=list(problem.objective_qubits))
    probability = 0.0
    for bitstring, value in probabilities.items():
        if normalize_bitstring(bitstring, len(good)) == good:
            probability += float(value)
    return float(np.clip(probability, 0.0, 1.0))


def apply_post_processing(problem: EstimationProblem, amplitude: float) -> float:
    return float(problem.post_processing(float(amplitude)))


def count_good_states(
    counts: Mapping[Any, Any],
    *,
    problem: EstimationProblem | None = None,
    good_bitstring: str | None = None,
    width: int | None = None,
) -> int:
    """Count outcomes classified as good.

    If ``problem`` is supplied, its ``is_good_state`` callable is the authority.
    Otherwise ``good_bitstring`` must be supplied.
    """
    if problem is None and good_bitstring is None:
        raise ValueError("Either problem or good_bitstring must be provided.")

    if width is None:
        width = objective_width(problem) if problem is not None else len(str(good_bitstring))
    if problem is not None:
        is_good = problem.is_good_state
    else:
        is_good = make_good_state(str(good_bitstring))

    total = 0
    for raw_state, raw_count in counts.items():
        state = normalize_bitstring(raw_state, int(width))
        if is_good(state):
            total += int(raw_count)
    return int(total)


@dataclass
class AEProblemBundle:
    """Problem plus the metadata experiment code needs to stay generic."""

    problem: EstimationProblem
    true_amplitude: float
    processed_true_value: float
    target_name: str = "amplitude"
    good_bitstring: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.good_bitstring = ensure_good_state(self.problem, self.good_bitstring)
        self.true_amplitude = float(np.clip(float(self.true_amplitude), 0.0, 1.0))
        self.processed_true_value = float(self.processed_true_value)
        self.target_name = str(self.target_name)
        self.metadata = dict(self.metadata or {})
        self.metadata.setdefault("objective_qubits", list(self.problem.objective_qubits))
        self.metadata.setdefault("good_bitstring", self.good_bitstring)
        self.metadata.setdefault("target_name", self.target_name)

    @property
    def objective_width(self) -> int:
        return objective_width(self.problem)

    def process(self, amplitude: float) -> float:
        return apply_post_processing(self.problem, float(amplitude))


def bundle_from_problem(
    problem: EstimationProblem,
    *,
    true_amplitude_value: float | None = None,
    processed_true_value: float | None = None,
    target_name: str = "amplitude",
    good_bitstring: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AEProblemBundle:
    """Build an :class:`AEProblemBundle` from any Qiskit estimation problem."""
    good = ensure_good_state(problem, good_bitstring)
    amplitude = (
        true_amplitude(problem, good)
        if true_amplitude_value is None
        else float(true_amplitude_value)
    )
    processed = (
        apply_post_processing(problem, amplitude)
        if processed_true_value is None
        else float(processed_true_value)
    )
    return AEProblemBundle(
        problem=problem,
        true_amplitude=amplitude,
        processed_true_value=processed,
        target_name=target_name,
        good_bitstring=good,
        metadata=dict(metadata or {}),
    )
