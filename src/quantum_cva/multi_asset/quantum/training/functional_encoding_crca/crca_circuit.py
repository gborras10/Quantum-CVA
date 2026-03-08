# python utils
import numpy as np
from collections.abc import Callable, Sequence
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector, Qubit
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from .rotations import apply_su2_block, apply_controlled_su2_block


class CrcaCircuit:
    """Controlled Rotations Circuit Ansatz (CRCA) wrapper with multiple layers.

    One *layer* is the original CRCA block:
      - Initial SU(2) block on ancilla: 3 single-qubit rotations
      - For each control qubit: controlled rotations
          * all but the last control: 3 controlled rotations (default: Rx, Ry, Rz)
          * last control: 2 controlled rotations (default: Rx, Ry) 

    With ``n_layers=L``, the circuit repeats this block L times sequentially.

    This wrapper also provides utilities to *train* CRCA to approximate a
    target function f(i) in the sense of Alcazar et al. (Eq. (29)–(33)):

        f~(i, theta) = P(a=1 | controls=i)

    Training is performed by preparing the control registers in a uniform
    superposition (Hadamards), applying CRCA, and estimating the conditional
    probabilities from measurement outcomes.

    Registers:
      - t: time register (m qubits)
      - s: price register (n qubits)
      - a: ancilla (1 qubit)

    Parameter convention (layer-major):
      Let P_layer be the number of parameters in one layer.
      Then layer \ell uses:
        theta[\ell*P_layer : (\ell+1)*P_layer]

      Within a layer:
        - 3 params for initial ancilla SU(2) block
        - 3 per control, except last control uses only 2 (if controls exist)
    """

    def __init__(
        self,
        m_time: int,
        n_price: int,
        *,
        n_layers: int = 1,
        name: str = "CRCA",
        init_order: Sequence[str] = ("ry", "rz", "ry"),
        ctrl_order: Sequence[str] = ("rx", "ry", "rz"),
        last_ctrl_order: Sequence[str] = ("rx", "ry"),
    ) -> None:
        if m_time < 0:
            raise ValueError("m_time must be >= 0.")
        if n_price < 0:
            raise ValueError("n_price must be >= 0.")
        if n_layers <= 0:
            raise ValueError("n_layers must be >= 1.")
        if len(init_order) != 3:
            raise ValueError("init_order must have length 3.")
        if len(ctrl_order) != 3:
            raise ValueError("ctrl_order must have length 3.")
        if len(last_ctrl_order) != 2:
            raise ValueError("last_ctrl_order must have length 2.")

        self._m_time = int(m_time)
        self._n_price = int(n_price)
        self._n_layers = int(n_layers)

        self._name = str(name)
        self._init_order = tuple(str(g).lower() for g in init_order)
        self._ctrl_order = tuple(str(g).lower() for g in ctrl_order)
        self._last_ctrl_order = tuple(str(g).lower() for g in last_ctrl_order)

        self._n_controls = self._m_time + self._n_price

        # Params per layer:
        #   - 3 for initial SU(2) block
        #   - 3 per control, except last control uses only 2 
        self._n_params_per_layer = self._count_params_per_layer()
        n_params_total = self._n_layers * self._n_params_per_layer
        self.theta = ParameterVector("theta", n_params_total)

        # Core ansatz circuit (no measurements)
        self.qc = self._build_ansatz()

        # Evaluation circuits for training f~(i,theta):
        #   qc_eval: Hadamards on controls + ansatz (no measurements)
        #   qc_eval_meas: same, with explicit measurements (controls + ancilla)
        self.qc_eval, self._qc_eval_meas = self._build_eval_circuits()

        # Backend and transpilation cache (shot-based path)
        self._backend = Aer.get_backend("aer_simulator")
        self._tqc_eval_meas = transpile(self._qc_eval_meas, self._backend)

        # Cache classical-bit indices for robust parsing of counts
        self._n_clbits = len(self._qc_eval_meas.clbits)
        self._ctrl_clbit_indices, self._a_clbit_index = (
            self._extract_clbit_indices(self._qc_eval_meas)
        )

    # =========================================================
    # Properties
    # =========================================================
    @property
    def m_time(self) -> int:
        return self._m_time

    @property
    def n_price(self) -> int:
        return self._n_price

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def dim_controls(self) -> int:
        return 2**self._n_controls

    @property
    def n_params_per_layer(self) -> int:
        return self._n_params_per_layer

    @property
    def n_params(self) -> int:
        return len(self.theta)

    # =========================================================
    # Parameter binding
    # =========================================================
    def bind(self, x: np.ndarray, *, measured: bool = False) -> QuantumCircuit:
        """Bind parameters to the ansatz (or evaluation) circuit."""
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}
        template = self._qc_eval_meas if measured else self.qc
        return template.assign_parameters(bind_map, inplace=False)

    # =========================================================
    # Training primitives: f~(i,theta) and L2 loss
    # =========================================================
    def function_values(
        self,
        x: np.ndarray,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Estimate f~(i,theta) = P(a=1 | controls=i) for all i.

        Parameters
        ----------
        x:
            Parameter vector.
        shots:
            If None, compute exactly via statevector.
            If int, estimate via sampling on Aer.
        seed:
            Simulator seed for shot-based evaluation.

        Returns
        -------
        np.ndarray
            Vector p of length 2**n_controls with p[i] = P(a=1 | controls=i).
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        # Ideal path: exact conditional probabilities from statevector
        if shots is None:
            bind_map = {self.theta[k]: float(x[k]) for k in range(self.n_params)}
            qc_bound = self.qc_eval.assign_parameters(bind_map, inplace=False)
            sv = Statevector.from_instruction(qc_bound)
            probs = np.asarray(
                sv.probabilities(), dtype=float
            )  # length 2**(n_controls+1)

            n_ctrl = self._n_controls
            dim_ctrl = 1 << n_ctrl

            num = np.zeros(dim_ctrl, dtype=float)
            den = np.zeros(dim_ctrl, dtype=float)

            # Qubit ordering for qc_eval: controls are qubits 0..n_ctrl-1, ancilla is qubit n_ctrl.
            # Statevector basis index uses little-endian: bit q corresponds to qubit q.
            for j, pj in enumerate(probs):
                i = j & (dim_ctrl - 1)
                a = (j >> n_ctrl) & 1
                den[i] += pj
                if a == 1:
                    num[i] += pj

            out = np.zeros(dim_ctrl, dtype=float)
            mask = den > 0.0
            out[mask] = num[mask] / den[mask]
            return out

        # Shot-based path
        shots_i = int(shots)
        if shots_i <= 0:
            raise ValueError("shots must be a positive integer.")

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}
        tqc_bound = self._tqc_eval_meas.assign_parameters(
            bind_map, inplace=False
        )

        run_kwargs: dict[str, object] = {"shots": shots_i}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        counts = (
            self._backend.run(tqc_bound, **run_kwargs).result().get_counts()
        )

        return self._counts_to_function_values(counts)

    @staticmethod
    def _validate_target_f(f_tg: np.ndarray, *, dim: int) -> np.ndarray:
        f = np.asarray(f_tg, dtype=float).ravel()
        if f.shape[0] != dim:
            raise ValueError(f"f_tg must have length {dim}; got {f.shape[0]}.")
        if not np.all(np.isfinite(f)):
            raise ValueError("f_tg contains non-finite entries.")
        if np.any((f < 0.0) | (f > 1.0)):
            raise ValueError("f_tg must lie in [0,1] elementwise.")
        return f

    def cost_value(
        self,
        x: np.ndarray,
        f_tg: np.ndarray,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        """L2 loss: mean_i (f~(i,theta) - f_tg[i])^2."""
        f = self._validate_target_f(f_tg, dim=self.dim_controls)
        p = self.function_values(x, shots=shots, seed=seed)
        diff = p - f
        return float(np.mean(diff * diff))

    def cost_fn(
        self,
        f_tg: np.ndarray,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Callable[[np.ndarray], float]:
        """Return a callable cost(x) for optimizers."""
        f = self._validate_target_f(f_tg, dim=self.dim_controls)

        def cost(x: np.ndarray) -> float:
            p = self.function_values(x, shots=shots, seed=seed)
            diff = p - f
            return float(np.mean(diff * diff))

        return cost

    # =========================================================
    # Internal helpers/builders
    # =========================================================
    def _count_params_per_layer(self) -> int:
        if self._n_controls == 0:
            return 3
        return 3 + 3 * (self._n_controls - 1) + 2

    def _build_ansatz(self) -> QuantumCircuit:
        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        a = QuantumRegister(1, "a")

        qc = QuantumCircuit(t, s, a, name=self._name)
        controls: list[Qubit] = [*t, *s]

        k = 0
        for _layer in range(self._n_layers):
            apply_su2_block(
                qc,
                target=a[0],
                thetas=(self.theta[k], self.theta[k + 1], self.theta[k + 2]),
                order=self._init_order,
            )
            k += 3

            if not controls:
                continue

            for ctrl in controls[:-1]:
                apply_controlled_su2_block(
                    qc,
                    control=ctrl,
                    target=a[0],
                    thetas=(
                        self.theta[k],
                        self.theta[k + 1],
                        self.theta[k + 2],
                    ),
                    order=self._ctrl_order,
                )
                k += 3

            last = controls[-1]
            apply_controlled_su2_block(
                qc,
                control=last,
                target=a[0],
                thetas=(self.theta[k], self.theta[k + 1]),
                order=self._last_ctrl_order,
            )
            k += 2

        if k != self.n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {self.n_params}."
            )

        return qc

    def _build_eval_circuits(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        """Build qc_eval and qc_eval_meas for training f~(i,theta)."""
        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        a = QuantumRegister(1, "a")

        c_ctrl = ClassicalRegister(self._n_controls, "c")
        c_a = ClassicalRegister(1, "ca")

        qc_eval = QuantumCircuit(t, s, a, name=f"{self._name}_eval")
        controls_q: list[Qubit] = [*t, *s]

        # Uniform superposition over control register
        for q in controls_q:
            qc_eval.h(q)

        # Compose the parameterized ansatz on the same qubit ordering [t..., s..., a]
        qc_eval.compose(self.qc, qubits=list(qc_eval.qubits), inplace=True)

        # Measured version with explicit classical registers
        qc_meas = QuantumCircuit(
            t, s, a, c_ctrl, c_a, name=f"{self._name}_eval_meas"
        )
        for q in controls_q:
            qc_meas.h(q)
        qc_meas.compose(
            self.qc,
            qubits=list(qc_meas.qubits)[: len(qc_eval.qubits)],
            inplace=True,
        )

        # Measure controls in *qubit-index order* into c_ctrl[0..n_controls-1]
        for idx, q in enumerate(controls_q):
            qc_meas.measure(q, c_ctrl[idx])
        qc_meas.measure(a[0], c_a[0])

        return qc_eval, qc_meas

    def _extract_clbit_indices(
        self, qc_meas: QuantumCircuit
    ) -> tuple[list[int], int]:
        """Return (control_clbit_indices, ancilla_clbit_index) in qc_meas.clbits indexing."""
        # Find the two classical registers by name convention used above.
        # This stays robust even if qiskit changes how it prints count bitstrings.
        cregs = {cr.name: cr for cr in qc_meas.cregs}
        if "c" not in cregs or "ca" not in cregs:
            raise RuntimeError(
                "Expected classical registers 'c' and 'ca' in eval circuit."
            )

        c_ctrl = cregs["c"]
        c_a = cregs["ca"]

        ctrl_idx = [qc_meas.clbits.index(c_ctrl[i]) for i in range(len(c_ctrl))]
        a_idx = qc_meas.clbits.index(c_a[0])
        return ctrl_idx, a_idx

    def _counts_to_function_values(self, counts: dict[str, int]) -> np.ndarray:
        """Convert raw counts over (controls, ancilla) into p[i]=P(a=1|i)."""
        dim_ctrl = self.dim_controls
        n_ctrl = self._n_controls

        n_i = np.zeros(dim_ctrl, dtype=float)
        n_i1 = np.zeros(dim_ctrl, dtype=float)

        for raw_bs, c in counts.items():
            bs = raw_bs.replace(" ", "")
            if len(bs) != self._n_clbits:
                raise RuntimeError(
                    f"Unexpected bitstring length {len(bs)} (expected {self._n_clbits})."
                )

            # Qiskit convention: leftmost char is highest classical bit index.
            def bit_at_clbit_index(cl_idx: int) -> int:
                pos_from_left = (self._n_clbits - 1) - cl_idx
                return 1 if bs[pos_from_left] == "1" else 0

            # Build integer i using control qubit indices as significance (idx matches qubit position).
            i_val = 0
            for q in range(n_ctrl):
                b = bit_at_clbit_index(self._ctrl_clbit_indices[q])
                i_val |= b << q

            a_bit = bit_at_clbit_index(self._a_clbit_index)

            c_f = float(c)
            n_i[i_val] += c_f
            if a_bit == 1:
                n_i1[i_val] += c_f

        out = np.zeros(dim_ctrl, dtype=float)
        mask = n_i > 0.0
        out[mask] = n_i1[mask] / n_i[mask]
        return out