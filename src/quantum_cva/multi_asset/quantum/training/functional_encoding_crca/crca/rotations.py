# python utils
from collections.abc import Sequence
from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.parameterexpression import ParameterExpression

Angle = float | int | ParameterExpression

_ALLOWED = {"rx", "ry", "rz"}

# ============== Alcazar CRCA rotation blocks ===============
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
    initial (uncontrolled) SU(2) in the CRCA ansatz, which is always 3 rotations.

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

# =============== Heron r2 - Native CRCA blocks ===============
def apply_1q_block(
    qc: QuantumCircuit,
    target: Qubit,
    thetas: Sequence,
    *,
    order: Sequence[str] = ("rx", "rz"),
) -> None:
    if len(thetas) != len(order):
        raise ValueError("thetas and order must have same length.")

    for gate, theta in zip(order, thetas):
        gate = gate.lower()
        if gate == "rx":
            qc.rx(theta, target)
        elif gate == "ry":
            qc.ry(theta, target)
        elif gate == "rz":
            qc.rz(theta, target)
        elif gate == "sx":
            qc.sx(target)
        elif gate == "x":
            qc.x(target)
        else:
            raise ValueError(f"Unsupported 1Q gate in native block: {gate}")


def apply_native_pair_block(
    qc: QuantumCircuit,
    left: Qubit,
    right: Qubit,
    target: Qubit,
    thetas: Sequence,
    *,
    one_q_order: Sequence[str] = ("rx", "rz"),
) -> None:
    """
    Pair-compression block:
        1Q(target) -> RZZ(left,target) -> RZZ(right,target) -> 1Q(target)

    Parameter count:
        2*len(one_q_order) + 2
    """
    n1 = len(one_q_order)
    expected = 2 * n1 + 2
    if len(thetas) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(thetas)}.")

    apply_1q_block(
        qc,
        target,
        thetas[:n1],
        order=one_q_order,
    )
    qc.rzz(thetas[n1], left, target)
    qc.rzz(thetas[n1 + 1], right, target)
    apply_1q_block(
        qc,
        target,
        thetas[n1 + 2 : n1 + 2 + n1],
        order=one_q_order,
    )


def apply_native_final_block(
    qc: QuantumCircuit,
    sources: Sequence[Qubit],
    target: Qubit,
    thetas: Sequence,
    *,
    one_q_order: Sequence[str] = ("rx", "rz"),
) -> None:
    """
    Final fusion block:
        1Q(target) -> prod_j RZZ(source_j, target) -> 1Q(target)

    Supports 1, 2 or 3 sources.
    Parameter count:
        2*len(one_q_order) + len(sources)
    """
    if not (1 <= len(sources) <= 3):
        raise ValueError("sources must have length 1, 2, or 3.")

    n1 = len(one_q_order)
    expected = 2 * n1 + len(sources)
    if len(thetas) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(thetas)}.")

    apply_1q_block(
        qc,
        target,
        thetas[:n1],
        order=one_q_order,
    )

    offset = n1
    for j, src in enumerate(sources):
        qc.rzz(thetas[offset + j], src, target)

    apply_1q_block(
        qc,
        target,
        thetas[offset + len(sources) : offset + len(sources) + n1],
        order=one_q_order,
    )

def apply_native_single_block(
    qc: QuantumCircuit,
    source: Qubit,
    target: Qubit,
    thetas: Sequence,
    *,
    one_q_order: Sequence[str] = ("rx", "rz"),
) -> None:
    """
    Single-source transfer block:
        1Q(target) -> RZZ(source, target) -> 1Q(target)

    Parameter count:
        2*len(one_q_order) + 1
    """
    n1 = len(one_q_order)
    expected = 2 * n1 + 1
    if len(thetas) != expected:
        raise ValueError(f"Expected {expected} parameters, got {len(thetas)}.")

    apply_1q_block(
        qc,
        target,
        thetas[:n1],
        order=one_q_order,
    )
    qc.rzz(thetas[n1], source, target)
    apply_1q_block(
        qc,
        target,
        thetas[n1 + 1 : n1 + 1 + n1],
        order=one_q_order,
    )

# =========================================================
# Internals
# =========================================================
def _validate_gate(g: str) -> None:
    if g not in _ALLOWED:
        raise ValueError(f"Unsupported gate '{g}'. Use one of {_ALLOWED}.")


def _apply_rotation(
    qc: QuantumCircuit, g: str, theta: Angle, target: Qubit
) -> None:
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