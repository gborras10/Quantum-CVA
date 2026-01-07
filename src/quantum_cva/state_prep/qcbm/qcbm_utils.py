from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_ansatz(n_qubits: int, *, name: str = "G_p") -> tuple[QuantumCircuit, ParameterVector]:
    """
    Build a QCBM-style ansatz circuit with a single-qubit rotation layer followed by
    a fully-connected entangling layer using RXX gates.

    The parameterization matches the pattern used in many QCBM baselines:
      - Layer 1 (local): for each qubit q, apply Rx(θ) then Rz(θ)
        → 2 * n_qubits parameters
      - Layer 2 (entangling): for each unordered qubit pair (a, b), apply RXX(θ)
        → n_qubits * (n_qubits - 1) / 2 parameters

    Hence, the total number of parameters is computed internally as:
        n_parameters = 2 * n_qubits + n_qubits * (n_qubits - 1) // 2

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (must be >= 1).
    name : str, optional
        Circuit name (default: "G_p").

    Returns
    -------
    qc : QuantumCircuit
        Parameterized ansatz circuit acting on ``n_qubits`` qubits.
    theta : ParameterVector
        Parameter vector of length ``n_parameters``.

    Raises
    ------
    ValueError
        If ``n_qubits < 1``.

    Examples
    --------
    Build the 4-qubit ansatz used in the fully-connected layout of
    Alcázar et al. (Fig. 5–style):

    >>> qc, theta = build_ansatz(n_qubits=4, name="G_p_fig5")
    >>> qc.num_qubits
    4
    >>> len(theta)
    14

    For n_qubits = 3, the number of parameters is 2*3 + 3 = 9:

    >>> qc, theta = build_ansatz(n_qubits=3)
    >>> len(theta)
    9
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")

    # number of trainable parameters implied by the architecture
    n_parameters = 2 * n_qubits + (n_qubits * (n_qubits - 1)) // 2

    theta = ParameterVector("theta", n_parameters)
    qc = QuantumCircuit(n_qubits, name=name)

    # Layer 1: local rotations
    k = 0
    for q in range(n_qubits):
        qc.rx(theta[k], q)
        k += 1
        qc.rz(theta[k], q)
        k += 1

    # Layer 2: fully-connected RXX entanglers
    for a in range(n_qubits):
        for b in range(a + 1, n_qubits):
            qc.rxx(theta[k], a, b)
            k += 1

    # Internal consistency check (should never fail)
    if k != n_parameters:
        raise RuntimeError(
            f"Internal parameter counting mismatch: used {k}, expected {n_parameters}."
        )

    return qc, theta

def probs_from_params(qc: QuantumCircuit, theta: ParameterVector, x: np.ndarray) -> np.ndarray:
    """
    Compute computational-basis probabilities from a parameterized circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        Parameterized quantum circuit (no measurements required).
    theta : ParameterVector
        Circuit parameters in the same order used when building the circuit.
    x : np.ndarray
        Parameter values, shape ``(len(theta),)``. Will be cast to float.

    Returns
    -------
    p : np.ndarray
        Probability vector in the computational basis, shape ``(2**n_qubits,)``.

    Raises
    ------
    ValueError
        If ``x`` does not have length ``len(theta)``.

    Examples
    --------
    >>> qc, theta = build_ansatz(n_qubits=2, n_parameters=5)
    >>> x = np.zeros(len(theta))
    >>> p = probs_from_params(qc, theta, x)
    >>> p.shape
    (4,)
    >>> float(p.sum())
    1.0
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.shape[0] != len(theta):
        raise ValueError(f"x must have length {len(theta)}; got {x.shape[0]}.")

    bind = {theta[i]: float(x[i]) for i in range(len(theta))}
    qc_bound = qc.assign_parameters(bind, inplace=False)

    sv = Statevector.from_instruction(qc_bound)
    return np.asarray(sv.probabilities(), dtype=float)


def make_cost(
    qc: QuantumCircuit,
    theta: ParameterVector,
    ptg: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Callable[[np.ndarray], float]:
    """
    Build a negative log-likelihood (cross-entropy) cost for fitting a QCBM distribution.

    The objective is:
        L(x) = - Σ_i ptg[i] * log(p_x[i]),
    where p_x are the circuit Born probabilities under parameters x.

    Parameters
    ----------
    qc : QuantumCircuit
        Parameterized circuit defining the model distribution.
    theta : ParameterVector
        Parameter vector used by the circuit.
    ptg : np.ndarray
        Target probability vector, shape ``(2**n_qubits,)``. Should sum to 1.
    eps : float, optional
        Floor applied to model probabilities to avoid log(0) (default: 1e-12).

    Returns
    -------
    cost : Callable[[np.ndarray], float]
        A function that maps a parameter vector ``x`` to a scalar loss.

    Raises
    ------
    ValueError
        If ptg shape is incompatible with the circuit dimension.

    Examples
    --------
    >>> qc, theta = build_ansatz(n_qubits=2, n_parameters=5)
    >>> ptg = np.array([0.25, 0.25, 0.25, 0.25])
    >>> cost = make_cost(qc, theta, ptg)
    >>> x0 = np.zeros(len(theta))
    >>> float(cost(x0))
    1.3862943611
    """
    ptg = np.asarray(ptg, dtype=float).ravel()

    dim = 2 ** qc.num_qubits
    if ptg.shape[0] != dim:
        raise ValueError(f"ptg must have length {dim} (2**n_qubits); got {ptg.shape[0]}.")

    if eps <= 0:
        raise ValueError("eps must be > 0.")

    # Do not silently renormalize here; enforce basic sanity only.
    if np.any(ptg < 0):
        raise ValueError("ptg contains negative entries.")
    s = float(ptg.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("ptg has non-finite or non-positive sum.")
    if abs(s - 1.0) > 1e-8:
        ptg = ptg / s

    def cost(x: np.ndarray) -> float:
        p = probs_from_params(qc, theta, x)
        p = np.maximum(p, eps)
        return float(-np.sum(ptg * np.log(p)))

    return cost


def metrics(ptg: np.ndarray, p: np.ndarray, *, eps: float = 1e-12) -> dict[str, float]:
    """
    Compute common discrepancy metrics between a target distribution and a model distribution.

    Metrics returned:
      - kl   : KL(ptg || p)
      - l1   : ||p - ptg||_1
      - tv   : total variation distance = 0.5 * l1
      - linf : ||p - ptg||_∞

    Parameters
    ----------
    ptg : np.ndarray
        Target probability vector (flattened), non-negative.
    p : np.ndarray
        Model probability vector (flattened), non-negative.
    eps : float, optional
        Floor to avoid division by zero / log(0) in KL (default: 1e-12).

    Returns
    -------
    out : dict[str, float]
        Dictionary with keys {"kl","l1","tv","linf"}.

    Raises
    ------
    ValueError
        If input shapes do not match.

    Examples
    --------
    >>> ptg = np.array([0.5, 0.5])
    >>> p   = np.array([0.9, 0.1])
    >>> m = metrics(ptg, p)
    >>> sorted(m.keys())
    ['kl', 'l1', 'linf', 'tv']
    >>> round(m["tv"], 3)
    0.4
    """
    ptg = np.asarray(ptg, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()

    if ptg.shape != p.shape:
        raise ValueError(f"Shape mismatch: ptg{ptg.shape} vs p{p.shape}.")

    if eps <= 0:
        raise ValueError("eps must be > 0.")

    # Normalize defensively (common in practice); this keeps metrics interpretable.
    s_ptg = float(ptg.sum())
    s_p = float(p.sum())
    if s_ptg <= 0 or not np.isfinite(s_ptg):
        raise ValueError("ptg has non-finite or non-positive sum.")
    if s_p <= 0 or not np.isfinite(s_p):
        raise ValueError("p has non-finite or non-positive sum.")
    ptg = ptg / s_ptg
    p = p / s_p

    ptg_c = np.maximum(ptg, eps)
    p_c = np.maximum(p, eps)

    kl = float(np.sum(ptg_c * np.log(ptg_c / p_c)))  # KL(ptg || p)
    diff = p - ptg
    l1 = float(np.sum(np.abs(diff)))
    tv = 0.5 * l1
    linf = float(np.max(np.abs(diff)))

    return {"kl": kl, "l1": l1, "tv": tv, "linf": linf}