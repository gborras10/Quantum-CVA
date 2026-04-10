from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


class MLQcbmCircuit:
    r"""
    Quantum Circuit Born Machine (QCBM) circuit wrapper with configurable
    entangler and topology.

    Main features
    -------------
    This wrapper provides:
      - construction of a parameterized QCBM ansatz with `n_layers`,
      - selectable two-qubit entangler: `rxx`, `rzz` or `cz`,
      - selectable topology: "all-to-all", "linear", "circular",
        "tree_bus", "snowflake", "optimized_snowflake", "qcbm_heavyhex8" or "qcbm_heavyhex6"
      - parameter binding,
      - probability evaluation (exact statevector or shot-based sampling),
      - (clipped) cross-entropy / negative log-likelihood cost,
      - optional rescaled cost,
      - optional Dirichlet/Laplace smoothing for shot-based objectives,
      - standard distance metrics between distributions.

    Layering convention
    -------------------
    The circuit is built as an alternating sequence of layers:

        rotations - entanglers - rotations - entanglers - ... (sequentially)

    where:
      - a "rotations" layer applies, on every qubit q_i, two parametrized rotations
        Rx(theta) and Rz(theta),
      - an "entangling" layer applies either:
          * parametrized RZZ(theta) gates over the selected qubit pairs, or
          * fixed CZ gates over the selected qubit pairs.

    The first layer is always a rotations layer. Therefore:
      - `n_layers = 1`  -> rotations
      - `n_layers = 2`  -> rotations + entanglers
      - `n_layers = 3`  -> rotations + entanglers + rotations
      - etc.

    Supported topologies
    --------------------
      - "all-to-all": all unordered pairs (a, b) with a < b
      - "linear":     pairs (0,1), (1,2), ..., (n-2,n-1)
      - "circular":   linear topology plus (n-1, 0) when n >= 3
      - "brickwork_ring": circular topology ordered in two disjoint sublayers
      - "qcbm_heavyhex8": native qcbm-heavyhex8-friendly 8-qubit motif formed by one
        6-cycle plus two opposite leaves
    Parameter counting
    ------------------
    Let:
      - n = number of qubits,
      - L = n_layers,
      - L_rot = ceil(L/2),
      - L_ent = floor(L/2),
      - P = number of entangling pairs for the chosen topology.

    Then:
      - if entangler == "rxx":
            n_params = L_rot * (2*n) + L_ent * P
      - if entangler == "rzz":
            n_params = L_rot * (2*n) + L_ent * P
      - if entangler == "cz":
            n_params = L_rot * (2*n)

    Notes
    -----
    * shots=None  -> exact Born probabilities via Statevector
    * shots=int   -> Monte Carlo estimate via Aer or the provided backend

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

    def __init__(
        self,
        n_qubits: int,
        *,
        n_layers: int = 2,
        name: str = "G_p",
        entangler: str = "rzz",
        topology: str = "all-to-all",
        backend=None,
        transpile_backend=None,
        noise_model=None,
        coupling_map=None,
        basis_gates: list[str] | None = None,
        simulation_method: str = "automatic",
        optimization_level: int = 1,
        initial_layout: list[int] | None = None,
        layout_method: str | None = None,
        routing_method: str | None = None,
        seed_transpiler: int | None = None,
    ) -> None:
        ...
        self._transpile_backend = transpile_backend
        self._layout_method = layout_method
        self._routing_method = routing_method
        self._seed_transpiler = seed_transpiler
        self._n_qubits = int(n_qubits)
        self._n_layers = int(n_layers)
        self._name = str(name)
        self._entangler = str(entangler)
        self._topology = str(topology)
        self._optimization_level = int(optimization_level)
        self._noise_model = noise_model
        self._coupling_map = coupling_map
        self._basis_gates = (
            list(basis_gates)
            if basis_gates is not None
            else ["cz", "id", "rx", "rz", "rzz", "sx", "x"]
        )
        self._simulation_method = str(simulation_method)
        self._initial_layout = list(initial_layout) if initial_layout is not None else None

        if backend is None:
            self._backend = AerSimulator(
                method=self._simulation_method,
                noise_model=self._noise_model,
                coupling_map=self._coupling_map,
                basis_gates=self._basis_gates,
            )
        else:
            self._backend = backend

        # -------------------------
        # Build entangler map from topology
        # -------------------------
        self._pairs = self._build_pairs(self._n_qubits, self._topology)
        n_pairs = len(self._pairs)

        # -------------------------
        # Parameter count
        # -------------------------
        n_rot_layers = (self._n_layers + 1) // 2
        n_ent_layers = self._n_layers // 2

        # 2) el conteo de parámetros:
        if self._entangler in {"rzz", "rxx"}:
            n_params = n_rot_layers * (2 * self._n_qubits) + n_ent_layers * n_pairs
        else:  # cz
            n_params = n_rot_layers * (2 * self._n_qubits)
                   
        self.theta = ParameterVector("theta", int(n_params))

        # -------------------------
        # Build parameterized ansatz (no measurements)
        # -------------------------
        q = QuantumRegister(self._n_qubits, "q")
        qc = QuantumCircuit(q, name=self._name)

        k = 0
        for layer_idx in range(1, self._n_layers + 1):
            is_rotation_layer = layer_idx % 2 == 1  # 1,3,5,... are rotations

            if is_rotation_layer:
                for qi in range(self._n_qubits):
                    qc.rx(self.theta[k], q[qi])
                    k += 1
                    qc.rz(self.theta[k], q[qi])
                    k += 1
            else:
                for a, b in self._pairs:
                    if self._entangler == "rzz":
                        qc.rzz(self.theta[k], q[a], q[b])
                        k += 1
                    elif self._entangler == "rxx":
                        qc.rxx(self.theta[k], q[a], q[b])
                        k += 1
                    else:
                        qc.cz(q[a], q[b])

        if k != n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {n_params}."
            )

        self.qc = qc

        # -------------------------
        # Measured circuit template (shot-based path)
        # -------------------------
        c = ClassicalRegister(self._n_qubits, "c")
        qc_meas = QuantumCircuit(q, c, name=f"{self._name}_meas")
        qc_meas.compose(
            self.qc,
            qubits=list(qc_meas.qubits)[: self._n_qubits],
            inplace=True,
        )

        for i in range(self._n_qubits):
            qc_meas.measure(q[i], c[i])

        self._qc_meas = qc_meas
        # -------------------------
        # Transpilation cache
        # -------------------------
        self._tqc = self._transpile_for_backend(self.qc)
        self._tqc_meas = self._transpile_for_backend(self._qc_meas)
        self._tqc_param_set = set(self._tqc.parameters)
        self._tqc_meas_param_set = set(self._tqc_meas.parameters)
        
        # Cache clbit indices for robust bitstring parsing
        self._n_clbits = len(self._qc_meas.clbits)
        self._clbit_indices = [
            self._qc_meas.clbits.index(c[i]) for i in range(self._n_qubits)
        ]

    # =========================================================
    # Internal helpers
    # =========================================================
    @staticmethod
    def _build_pairs(n_qubits: int, topology: str) -> list[tuple[int, int]]:
        """
        Build unordered entangling pairs according to the selected topology.
        """
        if topology == "all-to-all":
            return [
                (a, b)
                for a in range(n_qubits)
                for b in range(a + 1, n_qubits)
            ]

        if topology == "linear":
            return [(i, i + 1) for i in range(n_qubits - 1)]
        
        if topology == "tree_bus":
            if n_qubits != 10:
                raise ValueError("Tree bus topology specific for 10 qubits (1 central bus + 3 branches of 3 qubits each).")

            return [
                # Central bus (Time variables: logical qubits 0, 1, 2, 3)
                (0, 1), (1, 2), (2, 3),

                # Underlying branch 1 (connected to time 0)
                (0, 4), (4, 5),

                # Underlying branch 2 (connected to time 1)
                (1, 6), (6, 7),

                # Underlying branch 3 (connected to time 3)
                (3, 8), (8, 9),
            ]
        
        if topology == "snowflake":
            if n_qubits != 10:
                raise ValueError("Snowflake topology specific for 10 qubits (1 Central Hub).")
            
            return [
                # Central star of Time (Hub = 0)
                (0, 1), (0, 2), (0, 3),
                
                # Rama Subyacente 1 (Conectada a la punta de tiempo 1)
                (1, 4), (4, 5),
                
                # Rama Subyacente 2 (Conectada a la punta de tiempo 2)
                (2, 6), (6, 7),
                
                # Rama Subyacente 3 (Conectada a la punta de tiempo 3)
                (3, 8), (8, 9)
            ]
        
        if topology == "optimized_snowflake":
            if n_qubits != 10:
                raise ValueError("Solo para 10 qubits.")

            # El Hub lógico ahora es q5 (Subyacente 3).
            # Los activos se agrupan en el centro, el Tiempo va a las ramas exteriores.
            return [
                # Conexiones desde el Hub (q5)
                (5, 1), (5, 3), (5, 4),

                # Rama 1 (Continúa hacia el Tiempo)
                (1, 0), (0, 9),

                # Rama 2 (Continúa hacia el Tiempo)
                (3, 2), (2, 6),

                # Rama 3 (Continúa hacia el Tiempo)
                (4, 8), (8, 7)
            ]

        if topology == "qcbm_heavyhex8":
            if n_qubits != 8:
                raise ValueError(
                    "qcbm_heavyhex8 topology is only supported for 8 qubits."
                )

            # Native heavy-hex-friendly 8-qubit caterpillar tree:
            #
            #   q0 - q1 - q2 - q3 - q4 - q5
            #               |         |
            #               q6        q7
            #
            # Pair order is arranged in approximate disjoint sublayers to help
            # the scheduler/transpiler preserve a low 2Q depth.
            return [
                (0, 1),
                (2, 6),
                (4, 7),
                (1, 2),
                (3, 4),
                (2, 3),
                (4, 5),
            ]
        
        if topology == "qcbm_heavyhex6":
            if n_qubits != 6:
                raise ValueError(
                    "qcbm_heavyhex6 topology is only supported for 6 qubits."
                )

            # Grafo en estrella ramificada para maximizar el entrelazamiento
            # en 6 qubits sobre topología Heavy Hex.
            # Hub central en q0, ramas primarias en q1, q2, q3.
            # Extensiones en q4 (desde q1) y q5 (desde q2).
            return [
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (2, 5),
            ]

        # circular
        if n_qubits == 1:
            return []
        if n_qubits == 2:
            return [(0, 1)]

        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        pairs.append((0, n_qubits - 1))
        return pairs

    def _transpile_for_backend(self, circuit: QuantumCircuit) -> QuantumCircuit:
        compile_backend = (
            self._transpile_backend
            if self._transpile_backend is not None
            else self._backend
        )

        kwargs: dict[str, object] = {
            "backend": compile_backend,
            "optimization_level": self._optimization_level,
        }

        if self._initial_layout is not None:
            kwargs["initial_layout"] = self._initial_layout
        if self._layout_method is not None:
            kwargs["layout_method"] = self._layout_method
        if self._routing_method is not None:
            kwargs["routing_method"] = self._routing_method
        if self._seed_transpiler is not None:
            kwargs["seed_transpiler"] = self._seed_transpiler

        return transpile(circuit, **kwargs)

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
        Total number of layers in the ansatz (rotations / entanglers alternating).

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

    @property
    def entangler(self) -> str:
        """Two-qubit entangler used in the ansatz: 'rzz', 'rxx' or 'cz'."""
        return self._entangler

    @property
    def topology(self) -> str:
        """Entangling topology: 'all-to-all', 'linear', 'circular', 'tree_bus',
          'snowflake', 'optimized_snowflake', 'qcbm_heavyhex8' or 
          'qcbm_heavyhex6'."""
        return self._topology

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """List of entangling pairs used in the circuit."""
        return list(self._pairs)

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
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

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

        Modes
        -----
        * shots=None and noise_model is None:
            exact ideal probabilities via Statevector.
        * shots=None and noise_model is not None:
            exact noisy probabilities via AerSimulator + save_probabilities().
        * shots=int:
            sampled probabilities from backend / simulator counts.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}

        use_statevector_shortcut = (
            shots is None
            and self._simulation_method == "statevector"
            and self._noise_model is None
            and self._transpile_backend is None
        )

        if use_statevector_shortcut:
            qc_bound = self.qc.assign_parameters(bind_map, inplace=False)
            sv = Statevector.from_instruction(qc_bound)
            return np.asarray(sv.probabilities(), dtype=float)

        if shots is None:
            tqc_bound = self._bind_transpiled_cached(
                self._tqc,
                self._tqc_param_set,
                x,
            )

            qc_run = tqc_bound.copy()

            if self._initial_layout is not None:
                active_qubits = self._initial_layout
            else:
                active_qubits = list(range(self.n_qubits))
            
            qc_run.save_probabilities(qubits=active_qubits)

            run_kwargs: dict[str, object] = {}
            if seed is not None:
                run_kwargs["seed_simulator"] = int(seed)

            result = self._backend.run(qc_run, **run_kwargs).result()
            p = np.asarray(result.data(0)["probabilities"], dtype=float)
            p = np.maximum(p, 0.0)
            p /= float(p.sum())
            return p

        shots_i = int(shots)
        if shots_i <= 0:
            raise ValueError("shots must be a positive integer.")

        tqc_bound = self._bind_transpiled_cached(
            self._tqc_meas,
            self._tqc_meas_param_set,
            x,
        )

        run_kwargs: dict[str, object] = {"shots": shots_i}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        counts = self._backend.run(tqc_bound, **run_kwargs).result().get_counts()
        return self._counts_to_probabilities(counts, shots=shots_i)
    
    def _active_qubit_indices(self, circuit: QuantumCircuit) -> list[int]:
        """
        Return the physical qubit indices that are actually used by the transpiled circuit.
        This is useful when a backend-transpiled circuit has a large width but only a small
        subset of qubits is active.
        """
        return sorted(
            {
                circuit.find_bit(q).index
                for inst in circuit.data
                for q in inst.qubits
            }
        )

    def _counts_to_probabilities(
        self,
        counts: dict[str, int],
        *,
        shots: int,
    ) -> np.ndarray:
        """
        Convert raw counts into a probability vector p over computational basis.

        This parser is robust to Qiskit classical-bit ordering and guarantees
        the returned p matches Statevector indexing:
            x = sum_{q=0}^{n-1} bit(q) * 2^q

        Parameters
        ----------
        counts:
            Dictionary {bitstring: counts}.
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
                x_val |= b << q

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
            raise ValueError(
                "shots must be a positive integer for Dirichlet smoothing."
            )
        if alpha <= 0.0 or not np.isfinite(float(alpha)):
            raise ValueError("alpha must be a finite positive number.")

        p = np.asarray(p, dtype=float).ravel()
        if p.shape[0] != self.dim:
            raise ValueError(
                f"p must have length {self.dim}; got {p.shape[0]}."
            )

        N = float(shots_i)
        denom = N + float(alpha) * float(self.dim)
        out = (p * N + float(alpha)) / denom

        out = np.maximum(out, np.finfo(float).tiny)
        out /= float(out.sum())
        return out
    
    def _bind_transpiled_cached(
        self,
        circuit: QuantumCircuit,
        circuit_param_set: set,
        x: np.ndarray,
    ) -> QuantumCircuit:
        """
        Bind only the parameters that actually survive in a cached transpiled circuit.

        This avoids CircuitError when the transpiler has removed or fused away some
        original symbolic parameters at high optimization levels.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        full_bind_map = {
            self.theta[i]: float(x[i])
            for i in range(self.n_params)
        }

        filtered_bind_map = {
            param: value
            for param, value in full_bind_map.items()
            if param in circuit_param_set
        }

        return circuit.assign_parameters(filtered_bind_map, inplace=False)

    # =========================================================
    # Cross-entropy cost and rescaled cost
    # =========================================================
    @staticmethod
    def _validate_target(ptg: np.ndarray, *, dim: int) -> np.ndarray:
        """
        Validate and normalize a target probability distribution.
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
            p_use = self._smooth_probabilities_dirichlet(
                p, shots=int(shots), alpha=alpha
            )
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
                    p_use = self._smooth_probabilities_dirichlet(
                        p, shots=int(shots), alpha=alpha
                    )
                else:
                    p_use = np.maximum(p, eps)
                ce = float(-np.sum(ptg_v * np.log(p_use)))
                return float(ce - h)

            return cost

        def cost(x: np.ndarray) -> float:
            p = self.probabilities(x, shots=shots, seed=seed)
            if smoothing == "dirichlet" and shots is not None:
                p_use = self._smooth_probabilities_dirichlet(
                    p, shots=int(shots), alpha=alpha
                )
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
