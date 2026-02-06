# src/quantum_cva/crca/rotations.py
from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.parameterexpression import ParameterExpression


Angle = float | int | ParameterExpression

_ALLOWED = {"rx", "ry", "rz"}


def apply_su2_block(
    qc: QuantumCircuit,
    *,
    target: Qubit,
    thetas: Sequence[Angle],
    order: Sequence[str] = ("ry", "rz", "ry"),
) -> None:
    """
    Apply an SU(2) block on `target` as a sequence of rotations.

    This helper is kept strict (3 rotations) since it represents the
    initial (uncontrolled) SU(2) in the CRCA figure.

    Parameters
    ----------
    qc:
        Circuit to append gates to.
    target:
        Target qubit (ancilla).
    thetas:
        Iterable of 3 angles (can be Parameters).
    order:
        Iterable of 3 strings in {'rx','ry','rz'} defining the rotation sequence.
    """
    if len(thetas) != 3:
        raise ValueError("thetas must have length 3.")
    if len(order) != 3:
        raise ValueError("order must have length 3.")

    gates = [str(g).lower() for g in order]
    for g in gates:
        _validate_gate(g)

    _apply_rotation(qc, gates[0], thetas[0], target)
    _apply_rotation(qc, gates[1], thetas[1], target)
    _apply_rotation(qc, gates[2], thetas[2], target)


def apply_controlled_su2_block(
    qc: QuantumCircuit,
    *,
    control: Qubit,
    target: Qubit,
    thetas: Sequence[Angle],
    order: Sequence[str] = ("rx", "ry", "rz"),
) -> None:
    """
    Apply a controlled rotation block on `target` with a single control qubit.

    Supports either 3 controlled rotations (default) or 2 controlled rotations
    (e.g. dropping the final Rz as a global phase w.r.t. QAE).

    Parameters
    ----------
    qc:
        Circuit to append gates to.
    control:
        Single control qubit.
    target:
        Target qubit (ancilla).
    thetas:
        Iterable of angles; must match len(order) (2 or 3).
    order:
        Iterable of rotation labels in {'rx','ry','rz'} defining the sequence.
        Typical:
          - ("rx","ry","rz")  -> 3 rotations
          - ("rx","ry")       -> 2 rotations
    """
    if len(order) not in (2, 3):
        raise ValueError("order must have length 2 or 3.")
    if len(thetas) != len(order):
        raise ValueError("thetas length must match order length.")

    gates = [str(g).lower() for g in order]
    for g in gates:
        _validate_gate(g)

    for g, th in zip(gates, thetas):
        _apply_controlled_rotation(qc, g, th, control, target)


# =========================================================
# Internals
# =========================================================
def _validate_gate(g: str) -> None:
    if g not in _ALLOWED:
        raise ValueError(f"Unsupported gate '{g}'. Use one of {_ALLOWED}.")


def _apply_rotation(qc: QuantumCircuit, g: str, theta: Angle, target: Qubit) -> None:
    if g == "rx":
        qc.rx(theta, target)
    elif g == "ry":
        qc.ry(theta, target)
    elif g == "rz":
        qc.rz(theta, target)
    else:
        raise RuntimeError("Unreachable: gate already validated.")


def _apply_controlled_rotation(
    qc: QuantumCircuit,
    g: str,
    theta: Angle,
    control: Qubit,
    target: Qubit,
) -> None:
    if g == "rx":
        qc.crx(theta, control, target)
    elif g == "ry":
        qc.cry(theta, control, target)
    elif g == "rz":
        qc.crz(theta, control, target)
    else:
        raise RuntimeError("Unreachable: gate already validated.")