# src/quantum_cva/qcbm/qcbm_circuit.py
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer


class QcbmCircuit:
    """
    Quantum Circuit Born Machine (QCBM) circuit wrapper.

    This class encapsulates:
      - construction of a fixed QCBM ansatz,
      - parameter binding,
      - probability evaluation (ideal or shot-based),
      - cross-entropy / negative log-likelihood cost,
      - standard distance metrics between distributions.

    A single interface is used for both deterministic (statevector)
    and stochastic (shot-based) evaluations, controlled via the
    `shots` argument.

    Notes
    -----
    * shots=None  -> exact Born probabilities via Statevector
    * shots=int   -> Monte Carlo estimate via Aer simulator

    The circuit topology follows a standard QCBM layout:
      - one layer of local Rx–Rz rotations,
      - one fully-connected layer of RXX entangling gates.
    """

    def __init__(self, n_qubits: int, *, name: str = "G_p") -> None:
        """
        Initialize the QCBM circuit.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the QCBM register.
        name : str, optional
            Name of the quantum circuit (used for display/debugging).

        Raises
        ------
        ValueError
            If n_qubits < 1.
        RuntimeError
            If the internally counted number of parameters is inconsistent.
        """
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1.")

        self._n_qubits = int(n_qubits)
        self._name = str(name)

        # Number of parameters:
        #   - 2 per qubit (Rx, Rz)
        #   - one RXX per unordered qubit pair
        n_params = (
            2 * self._n_qubits
            + (self._n_qubits * (self._n_qubits - 1)) // 2
        )
        self.theta = ParameterVector("theta", n_params)

        # -------------------------
        # Build parameterized circuit
        # -------------------------
        qc = QuantumCircuit(self._n_qubits, name=self._name)

        # Layer 1: local single-qubit rotations
        k = 0
        for q in range(self._n_qubits):
            qc.rx(self.theta[k], q); k += 1
            qc.rz(self.theta[k], q); k += 1

        # Layer 2: fully-connected RXX entanglers
        for a in range(self._n_qubits):
            for b in range(a + 1, self._n_qubits):
                qc.rxx(self.theta[k], a, b); k += 1

        if k != n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {n_params}."
            )

        self.qc = qc

        # -------------------------
        # Measurement template (shot-based path)
        # -------------------------
        qc_meas = qc.copy()
        qc_meas.measure_all()
        self._qc_meas = qc_meas

        # -------------------------
        # Backend and transpilation cache
        # -------------------------
        self._backend = Aer.get_backend("aer_simulator")
        self._tqc_meas = transpile(self._qc_meas, self._backend)

    # =========================================================
    # Basic properties
    # =========================================================
    @property
    def n_qubits(self) -> int:
        """Number of qubits in the circuit."""
        return self._n_qubits

    @property
    def dim(self) -> int:
        """Hilbert space dimension (2**n_qubits)."""
        return 2 ** self._n_qubits

    @property
    def n_params(self) -> int:
        """Total number of variational parameters."""
        return len(self.theta)

    # =========================================================
    # Core operations
    # =========================================================
    def bind(self, x: np.ndarray, *, measured: bool = False) -> QuantumCircuit:
        """
        Bind a numerical parameter vector to the circuit.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector of shape (n_params,).
        measured : bool, optional
            If True, return the measured version of the circuit
            (used for shot-based evaluation).

        Returns
        -------
        QuantumCircuit
            A new circuit instance with parameters bound.

        Raises
        ------
        ValueError
            If the length of x is inconsistent.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}
        template = self._qc_meas if measured else self.qc
        return template.assign_parameters(bind_map, inplace=False)

    def probabilities(
        self,
        x: np.ndarray,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Compute Born probabilities for a given parameter vector.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector.
        shots : int or None, optional
            If None, compute exact probabilities via statevector.
            If int, estimate probabilities from `shots` samples.
        seed : int or None, optional
            Random seed for the simulator (shot-based only).

        Returns
        -------
        np.ndarray
            Probability vector of length 2**n_qubits.
        """
        # Ideal (deterministic) path
        if shots is None:
            qc_bound = self.bind(x, measured=False)
            sv = Statevector.from_instruction(qc_bound)
            return np.asarray(sv.probabilities(), dtype=float)

        # Shot-based path
        shots_i = int(shots)
        if shots_i <= 0:
            raise ValueError("shots must be a positive integer.")

        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}

        # Bind parameters on the already-transpiled circuit
        tqc_bound = self._tqc_meas.assign_parameters(
            bind_map, inplace=False
        )

        run_kwargs = {"shots": shots_i}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        counts = (
            self._backend
            .run(tqc_bound, **run_kwargs)
            .result()
            .get_counts()
        )

        p = np.zeros(self.dim, dtype=float)
        for bitstring, c in counts.items():
            p[int(bitstring, 2)] += float(c)
        p /= float(shots_i)

        return p

    # =========================================================
    # Cost (negative log-likelihood / cross-entropy)
    # =========================================================
    @staticmethod
    def _validate_target(ptg: np.ndarray, *, dim: int) -> np.ndarray:
        """
        Validate and normalize a target probability distribution.

        Parameters
        ----------
        ptg : np.ndarray
            Target probability vector.
        dim : int
            Expected dimension (2**n_qubits).

        Returns
        -------
        np.ndarray
            Validated and normalized target distribution.
        """
        ptg = np.asarray(ptg, dtype=float).ravel()

        if ptg.shape[0] != dim:
            raise ValueError(
                f"ptg must have length {dim}; got {ptg.shape[0]}."
            )
        if np.any(ptg < 0):
            raise ValueError("ptg contains negative entries.")

        s = float(ptg.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("ptg has non-finite or non-positive sum.")

        if abs(s - 1.0) > 1e-8:
            ptg = ptg / s

        return ptg

    def cost_value(
        self,
        x: np.ndarray,
        ptg: np.ndarray,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Evaluate the cross-entropy cost at a given parameter vector.

        This is equivalent to:
            -sum_j ptg[j] * log(p[j])

        Parameters
        ----------
        x : np.ndarray
            Parameter vector.
        ptg : np.ndarray
            Target probability distribution.
        eps : float, optional
            Lower cutoff for probabilities (numerical stability).
        shots : int or None, optional
            Controls deterministic vs shot-based evaluation.
        seed : int or None, optional
            Simulator seed (shot-based only).

        Returns
        -------
        float
            Cost value.
        """
        if eps <= 0:
            raise ValueError("eps must be > 0.")

        ptg_v = self._validate_target(ptg, dim=self.dim)
        p = self.probabilities(x, shots=shots, seed=seed)
        p = np.maximum(p, eps)

        return float(-np.sum(ptg_v * np.log(p)))

    def cost_fn(
        self,
        ptg: np.ndarray,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Callable[[np.ndarray], float]:
        """
        Return a callable cost function f(x).

        Useful for optimizers that expect a function of x only.

        Parameters
        ----------
        ptg : np.ndarray
            Target probability distribution.
        eps : float, optional
            Probability cutoff for numerical stability.
        shots : int or None, optional
            Deterministic vs shot-based objective.
        seed : int or None, optional
            Simulator seed (shot-based only).

        Returns
        -------
        Callable[[np.ndarray], float]
            Cost function f(x).
        """
        if eps <= 0:
            raise ValueError("eps must be > 0.")

        ptg_v = self._validate_target(ptg, dim=self.dim)

        def cost(x: np.ndarray) -> float:
            p = self.probabilities(x, shots=shots, seed=seed)
            p = np.maximum(p, eps)
            return float(-np.sum(ptg_v * np.log(p)))

        return cost

    # =========================================================
    # Metrics
    # =========================================================
    @staticmethod
    def metrics(
        ptg: np.ndarray,
        p: np.ndarray,
        *,
        eps: float = 1e-12,
    ) -> dict[str, float]:
        """
        Compute standard distance metrics between two distributions.

        Returned metrics:
          - KL divergence (ptg || p)
          - L1 distance
          - Total variation distance
          - L-infinity distance

        Parameters
        ----------
        ptg : np.ndarray
            Target distribution.
        p : np.ndarray
            Estimated distribution.
        eps : float, optional
            Cutoff for numerical stability.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: {"kl", "l1", "tv", "linf"}.
        """
        ptg = np.asarray(ptg, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()

        if ptg.shape != p.shape:
            raise ValueError(
                f"Shape mismatch: ptg{ptg.shape} vs p{p.shape}."
            )
        if eps <= 0:
            raise ValueError("eps must be > 0.")

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

        kl = float(np.sum(ptg_c * np.log(ptg_c / p_c)))
        diff = p - ptg
        l1 = float(np.sum(np.abs(diff)))
        tv = 0.5 * l1
        linf = float(np.max(np.abs(diff)))

        return {
            "kl": kl,
            "l1": l1,
            "tv": tv,
            "linf": linf,
        }