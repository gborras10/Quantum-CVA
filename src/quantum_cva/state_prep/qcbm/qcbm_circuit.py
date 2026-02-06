from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

class MLQcbmCircuit:
    r"""
    Quantum Circuit Born Machine (QCBM) circuit wrapper with an arbitrary number of layers.

    This wrapper provides:
      - construction of a parameterized QCBM ansatz with `n_layers`,
      - parameter binding,
      - probability evaluation (exact statevector or shot-based sampling),
      - (clipped) cross-entropy / negative log-likelihood cost,
      - optional rescaled cost (ideal \(\approx 0\) when \(p_\theta = p_{\text{tg}}\)),
      - optional Dirichlet/Laplace smoothing for shot-based objectives,
      - standard distance metrics between distributions.

    Layering convention
    -------------------
    The circuit is built as an *alternating* sequence of layers:

        rotations - XX - rotations - XX - ... (sequentially)

    where:
      - a "rotations" layer applies, on every qubit \(q_i\), two parametrized rotations
        \(R_x(\theta)\) and \(R_z(\theta)\),
      - an "XX" layer applies a fully-connected set of two-qubit entanglers \(R_{XX}(\theta)\)
        over all unordered pairs \((a,b)\) with \(a<b\).

    The first layer is always a rotations layer. Therefore:
      - `n_layers = 1`  -> rotations
      - `n_layers = 2`  -> rotations + XX
      - `n_layers = 3`  -> rotations + XX + rotations
      - etc.

    Parameter counting
    ------------------
    Let:
      - \(n\) be the number of qubits,
      - \(L\) be `n_layers`,
      - \(L_{\text{rot}} = \lceil L/2 \rceil\) (number of rotation layers),
      - \(L_{XX} = \lfloor L/2 \rfloor\) (number of XX layers),
      - \(E = n(n-1)/2\) be the number of unordered pairs.

    Then the total number of parameters is:

        n_params = L_rot * (2*n) + L_xx * E

    Notes
    -----
    * shots=None  -> exact Born probabilities via Statevector
    * shots=int   -> Monte Carlo estimate via Aer simulator

    Important implementation detail (shots path)
    -------------------------------------------
    Qiskit count bitstrings can be tricky due to classical-bit ordering.
    To make the shot-based probability vector consistent with the statevector
    ordering, this class:
      (i) uses an explicit ClassicalRegister c[i] for each qubit q[i],
      (ii) measures q[i] -> c[i] explicitly,
      (iii) parses the returned count bitstrings using clbit indices.

    The resulting index convention matches Statevector:
        index = sum_{q=0}^{n-1} bit(q) * 2^q
    (little-endian w.r.t. qubit index).

    Dirichlet/Laplace smoothing (shot-based objectives)
    ---------------------------------------------------
    When training with finite shots, the empirical distribution can contain
    zeros, and the log-likelihood / cross-entropy becomes high-variance.
    Optional Dirichlet smoothing replaces the empirical probabilities p(x) by

        p_s(x) = (N * p(x) + alpha) / (N + alpha * dim),

    where N is the number of shots, dim = 2**n_qubits, and alpha > 0 is the
    symmetric Dirichlet prior strength (alpha=1 corresponds to Laplace add-one).
    """

    def __init__(self, n_qubits: int, *, n_layers: int = 2, name: str = "G_p") -> None:
        """
        Initialize the QCBM circuit.

        Parameters
        ----------
        n_qubits:
            Number of qubits in the QCBM register.
        n_layers:
            Total number of layers in the alternating stack:
            rotations - XX - rotations - XX - ...
            Must be >= 1. The first layer is always rotations.
        name:
            Circuit name (useful for debugging/plots).

        Raises
        ------
        ValueError
            If n_qubits < 1 or n_layers < 1.
        RuntimeError
            If internal parameter counting is inconsistent.
        """
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")

        self._n_qubits = int(n_qubits)
        self._n_layers = int(n_layers)
        self._name = str(name)

        # -------------------------
        # Parameter count
        # -------------------------
        n_rot_layers = (self._n_layers + 1) // 2          
        n_xx_layers = self._n_layers // 2                
        n_pairs = (self._n_qubits * (self._n_qubits - 1)) // 2

        n_params = n_rot_layers * (2 * self._n_qubits) + n_xx_layers * n_pairs
        self.theta = ParameterVector("theta", int(n_params))

        # -------------------------
        # Build parameterized ansatz (no measurements)
        # -------------------------
        q = QuantumRegister(self._n_qubits, "q")
        qc = QuantumCircuit(q, name=self._name)

        k = 0
        for layer_idx in range(1, self._n_layers + 1):
            is_rotation_layer = (layer_idx % 2 == 1)  # 1,3,5,... are rotations
            if is_rotation_layer:
                # Local single-qubit rotations: Rx, Rz per qubit
                for qi in range(self._n_qubits):
                    qc.rx(self.theta[k], q[qi])
                    k += 1
                    qc.rz(self.theta[k], q[qi])
                    k += 1
            else:
                # Fully-connected RXX entanglers: one parameter per unordered pair
                for a in range(self._n_qubits):
                    for b in range(a + 1, self._n_qubits):
                        qc.rxx(self.theta[k], q[a], q[b])
                        k += 1

        if k != n_params:
            raise RuntimeError(f"Parameter counting mismatch: used {k}, expected {n_params}.")

        self.qc = qc

        # -------------------------
        # Measured circuit template (shot-based path)
        # -------------------------
        # Use explicit classical register so that measurement mapping is unambiguous:
        #   measure q[i] -> c[i]
        c = ClassicalRegister(self._n_qubits, "c")
        qc_meas = QuantumCircuit(q, c, name=f"{self._name}_meas")
        qc_meas.compose(self.qc, qubits=list(qc_meas.qubits)[: self._n_qubits], inplace=True)

        for i in range(self._n_qubits):
            qc_meas.measure(q[i], c[i])

        self._qc_meas = qc_meas

        # -------------------------
        # Backend and transpilation cache
        # -------------------------
        self._backend = Aer.get_backend("aer_simulator")
        self._tqc_meas = transpile(self._qc_meas, self._backend)

        # Cache clbit indices for robust bitstring parsing
        self._n_clbits = len(self._qc_meas.clbits)
        self._clbit_indices = [self._qc_meas.clbits.index(c[i]) for i in range(self._n_qubits)]

    # =========================================================
    # Basic properties
    # =========================================================
    @property
    def n_qubits(self) -> int:
        """Number of qubits in the QCBM register."""
        return self._n_qubits

    @property
    def n_layers(self) -> int:
        """
        Total number of layers in the ansatz (rotations/XX alternating).

        The first layer is always rotations.
        """
        return self._n_layers

    @property
    def dim(self) -> int:
        """Hilbert space dimension (2**n_qubits)."""
        return 2**self._n_qubits

    @property
    def n_params(self) -> int:
        """Total number of variational parameters."""
        return len(self.theta)

    # =========================================================
    # Parameter binding
    # =========================================================
    def bind(self, x: np.ndarray, *, measured: bool = False) -> QuantumCircuit:
        """
        Bind a numerical parameter vector to the circuit.

        Parameters
        ----------
        x:
            Parameter vector of shape (n_params,).
        measured:
            If True, return the measured circuit (shot-based template).
            If False, return the ansatz circuit (no measurements).

        Returns
        -------
        QuantumCircuit
            A new circuit instance with parameters bound.

        Raises
        ------
        ValueError
            If x has inconsistent length.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(f"x must have length {self.n_params}; got {x.shape[0]}.")

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}
        template = self._qc_meas if measured else self.qc
        return template.assign_parameters(bind_map, inplace=False)

    # =========================================================
    # Probabilities
    # =========================================================
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
        x:
            Parameter vector.
        shots:
            If None, compute exact probabilities via Statevector.
            If int, estimate probabilities from `shots` samples on Aer.
        seed:
            Simulator seed (shot-based only).

        Returns
        -------
        np.ndarray
            Probability vector of length 2**n_qubits, indexed consistently with
            Statevector: index = sum_q bit(q) * 2^q.
        """
        # Exact (deterministic) path
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
            raise ValueError(f"x must have length {self.n_params}; got {x.shape[0]}.")

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}

        # Bind parameters on the transpiled measured circuit
        tqc_bound = self._tqc_meas.assign_parameters(bind_map, inplace=False)

        run_kwargs: dict[str, object] = {"shots": shots_i}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        counts = self._backend.run(tqc_bound, **run_kwargs).result().get_counts()
        return self._counts_to_probabilities(counts, shots=shots_i)

    def _counts_to_probabilities(self, counts: dict[str, int], *, shots: int) -> np.ndarray:
        """
        Convert raw counts into a probability vector p over computational basis.

        This parser is robust to Qiskit classical-bit ordering and guarantees
        the returned p matches Statevector indexing:
            x = sum_{q=0}^{n-1} bit(q) * 2^q

        Parameters
        ----------
        counts:
            Dictionary {bitstring: counts} from Aer.
        shots:
            Total number of shots.

        Returns
        -------
        np.ndarray
            Probability vector of length 2**n_qubits.
        """
        p = np.zeros(self.dim, dtype=float)

        for raw_bs, c in counts.items():
            bs = raw_bs.replace(" ", "")
            if len(bs) != self._n_clbits:
                raise RuntimeError(
                    f"Unexpected bitstring length {len(bs)} (expected {self._n_clbits})."
                )

            # Qiskit convention: leftmost char corresponds to the highest clbit index.
            def bit_at_clbit_index(cl_idx: int) -> int:
                pos_from_left = (self._n_clbits - 1) - cl_idx
                return 1 if bs[pos_from_left] == "1" else 0

            x_val = 0
            for q in range(self._n_qubits):
                b = bit_at_clbit_index(self._clbit_indices[q])
                x_val |= (b << q)

            p[x_val] += float(c)

        p /= float(shots)
        return p

    # =========================================================
    # Dirichlet smoothing (optional, shot-based losses)
    # =========================================================
    def _smooth_probabilities_dirichlet(
        self,
        p: np.ndarray,
        *,
        shots: int,
        alpha: float,
    ) -> np.ndarray:
        """
        Apply symmetric Dirichlet/Laplace smoothing to a probability vector.

        Given empirical probabilities p(x) = n_x / N, return
            p_s(x) = (n_x + alpha) / (N + alpha * dim)
                 = (N * p(x) + alpha) / (N + alpha * dim).

        Parameters
        ----------
        p:
            Probability vector (typically empirical, sum ~= 1).
        shots:
            Total number of shots N used to form p.
        alpha:
            Symmetric Dirichlet prior strength (alpha > 0).

        Returns
        -------
        np.ndarray
            Smoothed probability vector of same shape as p.

        Raises
        ------
        ValueError
            If shots <= 0 or alpha <= 0.
        """
        shots_i = int(shots)
        if shots_i <= 0:
            raise ValueError("shots must be a positive integer for Dirichlet smoothing.")
        if alpha <= 0.0 or not np.isfinite(float(alpha)):
            raise ValueError("alpha must be a finite positive number.")

        p = np.asarray(p, dtype=float).ravel()
        if p.shape[0] != self.dim:
            raise ValueError(f"p must have length {self.dim}; got {p.shape[0]}.")

        N = float(shots_i)
        denom = N + float(alpha) * float(self.dim)
        out = (p * N + float(alpha)) / denom

        # Ensure numerical sanity
        out = np.maximum(out, np.finfo(float).tiny)
        out /= float(out.sum())
        return out

    # =========================================================
    # Cross-entropy cost and rescaled cost
    # =========================================================
    @staticmethod
    def _validate_target(ptg: np.ndarray, *, dim: int) -> np.ndarray:
        """
        Validate and normalize a target probability distribution.

        Parameters
        ----------
        ptg:
            Target probability vector.
        dim:
            Expected dimension (2**n_qubits).

        Returns
        -------
        np.ndarray
            Validated and normalized target distribution.
        """
        ptg = np.asarray(ptg, dtype=float).ravel()
        if ptg.shape[0] != dim:
            raise ValueError(f"ptg must have length {dim}; got {ptg.shape[0]}.")
        if np.any(ptg < 0.0):
            raise ValueError("ptg contains negative entries.")

        s = float(ptg.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("ptg has non-finite or non-positive sum.")

        if abs(s - 1.0) > 1e-12:
            ptg = ptg / s

        return ptg

    @staticmethod
    def entropy(ptg: np.ndarray, *, eps: float = 1e-12) -> float:
        """
        Compute the Shannon entropy H(ptg) = -sum_j ptg[j] log(ptg[j]).

        Parameters
        ----------
        ptg:
            Probability vector (will be normalized internally).
        eps:
            Lower cutoff for numerical stability inside log().

        Returns
        -------
        float
            Entropy value.
        """
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")

        ptg = np.asarray(ptg, dtype=float).ravel()
        s = float(ptg.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("ptg has non-finite or non-positive sum.")
        ptg = ptg / s

        ptg_c = np.maximum(ptg, eps)
        return float(-np.sum(ptg * np.log(ptg_c)))

    def cost_value(
        self,
        x: np.ndarray,
        ptg: np.ndarray,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
        smoothing: str | None = None,
        alpha: float = 1.0,
    ) -> float:
        """
        Evaluate the cross-entropy (negative log-likelihood) cost:
            CE(ptg, p) = -sum_j ptg[j] log(p[j])

        Parameters
        ----------
        x:
            Parameter vector.
        ptg:
            Target probability distribution.
        eps:
            Lower cutoff for p[j] to avoid log(0). Used when smoothing is None.
        shots:
            If None -> exact; if int -> shot-based.
        seed:
            Simulator seed (shot-based only).
        smoothing:
            None        -> use clipping with eps (original behaviour).
            "dirichlet" -> apply Dirichlet/Laplace smoothing to p (shot-based only).
        alpha:
            Dirichlet prior strength (alpha > 0).

        Returns
        -------
        float
            Cross-entropy value.
        """
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        if smoothing not in (None, "dirichlet"):
            raise ValueError("smoothing must be None or 'dirichlet'.")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0.")

        ptg_v = self._validate_target(ptg, dim=self.dim)
        p = self.probabilities(x, shots=shots, seed=seed)

        if smoothing == "dirichlet" and shots is not None:
            p_use = self._smooth_probabilities_dirichlet(p, shots=int(shots), alpha=alpha)
        else:
            p_use = np.maximum(p, eps)

        return float(-np.sum(ptg_v * np.log(p_use)))

    def cost_value_rescaled(
        self,
        x: np.ndarray,
        ptg: np.ndarray,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
        smoothing: str | None = None,
        alpha: float = 1.0,
    ) -> float:
        """
        Evaluate a rescaled objective:
            CE(ptg, p_theta) - H(ptg)

        This is ideally ~0 when p_theta == ptg (up to eps/smoothing effects).

        Parameters
        ----------
        x:
            Parameter vector.
        ptg:
            Target probability distribution.
        eps:
            Lower cutoff for numerical stability.
        shots:
            If None -> exact; if int -> shot-based.
        seed:
            Simulator seed (shot-based only).
        smoothing:
            None        -> use clipping with eps (original behaviour).
            "dirichlet" -> apply Dirichlet/Laplace smoothing to p (shot-based only).
        alpha:
            Dirichlet prior strength (alpha > 0).

        Returns
        -------
        float
            Rescaled cost value.
        """
        ptg_v = self._validate_target(ptg, dim=self.dim)
        ce = self.cost_value(
            x,
            ptg_v,
            eps=eps,
            shots=shots,
            seed=seed,
            smoothing=smoothing,
            alpha=alpha,
        )
        h = self.entropy(ptg_v, eps=eps)
        return float(ce - h)

    def cost_fn(
        self,
        ptg: np.ndarray,
        *,
        eps: float = 1e-12,
        shots: int | None = None,
        seed: int | None = None,
        rescaled: bool = False,
        smoothing: str | None = None,
        alpha: float = 1.0,
    ) -> Callable[[np.ndarray], float]:
        """
        Return a callable objective f(x) for optimizers (SPSA, COBYLA, etc.).

        Parameters
        ----------
        ptg:
            Target probability distribution.
        eps:
            Lower cutoff for p[j] inside log(). Used when smoothing is None.
        shots:
            If None -> exact objective; if int -> shot-based objective.
        seed:
            Simulator seed (shot-based only).
        rescaled:
            If True, return CE(ptg, p_theta) - H(ptg) (ideal ~ 0).
            If False, return CE(ptg, p_theta).
        smoothing:
            None        -> use clipping with eps (original behaviour).
            "dirichlet" -> apply Dirichlet/Laplace smoothing to p (shot-based only).
        alpha:
            Dirichlet prior strength (alpha > 0).

        Returns
        -------
        Callable[[np.ndarray], float]
            A function f(x) that returns the chosen cost.
        """
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")
        if smoothing not in (None, "dirichlet"):
            raise ValueError("smoothing must be None or 'dirichlet'.")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0.")

        ptg_v = self._validate_target(ptg, dim=self.dim)

        if rescaled:
            h = self.entropy(ptg_v, eps=eps)

            def cost(x: np.ndarray) -> float:
                p = self.probabilities(x, shots=shots, seed=seed)
                if smoothing == "dirichlet" and shots is not None:
                    p_use = self._smooth_probabilities_dirichlet(p, shots=int(shots), alpha=alpha)
                else:
                    p_use = np.maximum(p, eps)
                ce = float(-np.sum(ptg_v * np.log(p_use)))
                return float(ce - h)

            return cost

        def cost(x: np.ndarray) -> float:
            p = self.probabilities(x, shots=shots, seed=seed)
            if smoothing == "dirichlet" and shots is not None:
                p_use = self._smooth_probabilities_dirichlet(p, shots=int(shots), alpha=alpha)
            else:
                p_use = np.maximum(p, eps)
            return float(-np.sum(ptg_v * np.log(p_use)))

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
        ptg:
            Target distribution.
        p:
            Estimated distribution.
        eps:
            Cutoff for numerical stability.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: {"kl", "l1", "tv", "linf"}.
        """
        ptg = np.asarray(ptg, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()

        if ptg.shape != p.shape:
            raise ValueError(f"Shape mismatch: ptg{ptg.shape} vs p{p.shape}.")
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")

        s_ptg = float(ptg.sum())
        s_p = float(p.sum())
        if s_ptg <= 0.0 or not np.isfinite(s_ptg):
            raise ValueError("ptg has non-finite or non-positive sum.")
        if s_p <= 0.0 or not np.isfinite(s_p):
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

        return {"kl": kl, "l1": l1, "tv": tv, "linf": linf}