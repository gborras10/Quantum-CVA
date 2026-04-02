# python utils
import numpy as np
from collections.abc import Callable, Sequence
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import ParameterVector, Qubit
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from .rotations import (
    apply_su2_block, 
    apply_controlled_su2_block,
    apply_native_pair_block,
    apply_native_single_block,
    apply_native_final_block
)


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
        ansatz_type: str = "standard",
        native_1q_order: Sequence[str] = ("rx", "rz"),
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

        self._ansatz_type = str(ansatz_type).lower()
        self._native_1q_order = tuple(str(g).lower() for g in native_1q_order)

        if self._ansatz_type not in {
            "standard",
            "native_tree",
            "native_heavyhex8",
            "native_heavyhex10",
        }:
            raise ValueError(
                "ansatz_type must be 'standard', 'native_tree', "
                "'native_heavyhex8', or 'native_heavyhex10'."
            )

        if self._ansatz_type == "native_tree":
            self._n_work = self._native_required_work_qubits()
        elif self._ansatz_type == "native_heavyhex8":
            self._n_work = self._native_heavyhex8_required_work_qubits()
        elif self._ansatz_type == "native_heavyhex10":
            self._n_work = self._native_heavyhex10_required_work_qubits()
        else:
            self._n_work = 0
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
        self._tqc_eval_meas_param_set = set(self._tqc_eval_meas.parameters)

        # Cache classical-bit indices for robust parsing of counts
        # IMPORTANT: these must correspond to the circuit that is actually run.
        self._n_clbits = len(self._tqc_eval_meas.clbits)
        self._ctrl_clbit_indices, self._a_clbit_index = (
            self._extract_clbit_indices(self._tqc_eval_meas)
        )
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

        if measured:
            return self._bind_transpiled_eval_meas(x)

        bind_map = {self.theta[i]: float(x[i]) for i in range(self.n_params)}
        return self.qc.assign_parameters(bind_map, inplace=False)
    
    def _bind_transpiled_eval_meas(self, x: np.ndarray) -> QuantumCircuit:
        """
        Bind only the parameters that are still present in the cached transpiled
        measured circuit. This avoids CircuitError if transpilation removed/fused
        some symbolic parameters.
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
            if param in self._tqc_eval_meas_param_set
        }

        return self._tqc_eval_meas.assign_parameters(
            filtered_bind_map,
            inplace=False,
        )

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
            probs = np.asarray(sv.probabilities(), dtype=float)

            dim_ctrl = 1 << self._n_controls
            num = np.zeros(dim_ctrl, dtype=float)
            den = np.zeros(dim_ctrl, dtype=float)

            for basis_idx, pj in enumerate(probs):
                i_val = 0
                for local_idx, q_idx in enumerate(self._control_qubit_indices):
                    bit = (basis_idx >> q_idx) & 1
                    i_val |= bit << local_idx

                a_bit = (basis_idx >> self._ancilla_qubit_index) & 1

                den[i_val] += pj
                if a_bit == 1:
                    num[i_val] += pj

            out = np.zeros(dim_ctrl, dtype=float)
            mask = den > 0.0
            out[mask] = num[mask] / den[mask]
            return out

        # Shot-based path
        shots_i = int(shots)
        if shots_i <= 0:
            raise ValueError("shots must be a positive integer.")

        qc_bound = self._bind_transpiled_eval_meas(x)

        run_kwargs: dict[str, object] = {"shots": shots_i}
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        counts = self._backend.run(qc_bound, **run_kwargs).result().get_counts()

        return self._counts_to_function_values(counts)
            
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
    
    def _native_required_work_qubits(self) -> int:
        n = self._n_controls
        work = 0
        while n > 3:
            pairs = n // 2
            carry = n % 2
            work += pairs
            n = pairs + carry
        return work
    

    def _native_heavyhex8_required_work_qubits(self) -> int:
        if self._n_controls != 8:
            raise ValueError(
                "ansatz_type='native_heavyhex8' requires exactly 8 controls "
                "(m_time + n_price = 8)."
            )
        # h0..h3, b0..b3, g0..g1
        return 10

    def _native_heavyhex10_required_work_qubits(self) -> int:
        if self._n_controls != 10:
            raise ValueError(
                "ansatz_type='native_heavyhex10' requires exactly 10 controls "
                "(m_time + n_price = 10)."
            )
        # h0..h4, b0..b4, g0..g1, b5..b6
        return 14

    def _count_params_per_layer(self) -> int:
        if self._ansatz_type == "standard":
            if self._n_controls == 0:
                return 3
            return 3 + 3 * (self._n_controls - 1) + 2

        n1 = len(self._native_1q_order)

        if self._ansatz_type == "native_tree":
            n = self._n_controls
            total = 0

            while n > 3:
                pairs = n // 2
                carry = n % 2
                total += pairs * (2 * n1 + 2)
                n = pairs + carry

            total += 2 * n1 + n
            return total

        if self._ansatz_type == "native_heavyhex8":
            # 4 pair blocks de hojas
            # 4 single blocks h_i -> b_i
            # 2 pair blocks de segundo nivel
            # 1 final block de 2 fuentes sobre ancilla
            pair_params = 2 * n1 + 2
            single_params = 2 * n1 + 1
            final2_params = 2 * n1 + 2

            return (
                4 * pair_params
                + 4 * single_params
                + 2 * pair_params
                + final2_params
            )

        if self._ansatz_type == "native_heavyhex10":
            # 5 pair blocks de hojas
            # 5 single blocks h_i -> b_i
            # 2 pair blocks de segundo nivel
            # 2 single blocks g_i -> b_{5+i}
            # 1 final block de 3 fuentes sobre ancilla
            pair_params = 2 * n1 + 2
            single_params = 2 * n1 + 1
            final3_params = 2 * n1 + 3

            return (
                5 * pair_params
                + 5 * single_params
                + 2 * pair_params
                + 2 * single_params
                + final3_params
            )

        raise RuntimeError(f"Unsupported ansatz_type: {self._ansatz_type}")
    def _build_ansatz(self) -> QuantumCircuit:
        if self._ansatz_type == "standard":
            return self._build_standard_ansatz()
        if self._ansatz_type == "native_tree":
            return self._build_native_tree_ansatz()
        if self._ansatz_type == "native_heavyhex8":
            return self._build_native_heavyhex8_ansatz()
        if self._ansatz_type == "native_heavyhex10":
            return self._build_native_heavyhex10_ansatz()
        raise RuntimeError(f"Unsupported ansatz_type: {self._ansatz_type}")
    def _build_standard_ansatz(self) -> QuantumCircuit:
        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        a = QuantumRegister(1, "a")

        qc = QuantumCircuit(t, s, a, name=self._name)
        controls: list[Qubit] = [*t, *s]

        ancilla = a[0]

        self._control_qubit_indices = list(range(len(controls)))
        self._ancilla_qubit_index = qc.qubits.index(ancilla)

        k = 0
        for _layer in range(self._n_layers):
            apply_su2_block(
                qc,
                target=ancilla,
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
                    target=ancilla,
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
                target=ancilla,
                thetas=(self.theta[k], self.theta[k + 1]),
                order=self._last_ctrl_order,
            )
            k += 2

        if k != self.n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {self.n_params}."
            )

        return qc
    
    def _build_native_tree_ansatz(self) -> QuantumCircuit:
        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        w = QuantumRegister(self._n_work, "w") if self._n_work > 0 else None
        a = QuantumRegister(1, "a")

        if w is None:
            qc = QuantumCircuit(t, s, a, name=self._name)
            work_qubits: list[Qubit] = []
        else:
            qc = QuantumCircuit(t, s, w, a, name=self._name)
            work_qubits = list(w)

        controls: list[Qubit] = [*t, *s]
        ancilla = a[0]

        # metadata for exact marginalization later
        self._control_qubit_indices = list(range(len(controls)))
        self._ancilla_qubit_index = qc.qubits.index(ancilla)

        n1 = len(self._native_1q_order)
        pair_block_params = 2 * n1 + 2

        k = 0
        for _layer in range(self._n_layers):
            current_nodes: list[Qubit] = controls.copy()
            work_ptr = 0

            while len(current_nodes) > 3:
                next_nodes: list[Qubit] = []
                i = 0
                while i + 1 < len(current_nodes):
                    target = work_qubits[work_ptr]
                    work_ptr += 1

                    apply_native_pair_block(
                        qc,
                        current_nodes[i],
                        current_nodes[i + 1],
                        target,
                        thetas=self.theta[k : k + pair_block_params],
                        one_q_order=self._native_1q_order,
                    )
                    k += pair_block_params
                    next_nodes.append(target)
                    i += 2

                if i < len(current_nodes):
                    next_nodes.append(current_nodes[i])

                current_nodes = next_nodes

            final_params = 2 * n1 + len(current_nodes)
            apply_native_final_block(
                qc,
                current_nodes,
                ancilla,
                thetas=self.theta[k : k + final_params],
                one_q_order=self._native_1q_order,
            )
            k += final_params

        if k != self.n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {self.n_params}."
            )

        return qc
    

    def _build_native_heavyhex8_ansatz(self) -> QuantumCircuit:
        if self._n_controls != 8:
            raise ValueError(
                "native_heavyhex8 requires exactly 8 controls "
                "(m_time + n_price = 8)."
            )

        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        w = QuantumRegister(self._n_work, "w")
        a = QuantumRegister(1, "a")

        qc = QuantumCircuit(t, s, w, a, name=self._name)

        controls: list[Qubit] = [*t, *s]
        ancilla = a[0]
        work_qubits = list(w)

        self._control_qubit_indices = list(range(len(controls)))
        self._ancilla_qubit_index = qc.qubits.index(ancilla)

        # Work layout:
        # w[0:4]  = h0..h3
        # w[4:8]  = b0..b3
        # w[8:10] = g0..g1
        n1 = len(self._native_1q_order)
        pair_params = 2 * n1 + 2
        single_params = 2 * n1 + 1
        final2_params = 2 * n1 + 2

        k = 0
        for _layer in range(self._n_layers):
            h0, h1, h2, h3 = work_qubits[0:4]
            b0, b1, b2, b3 = work_qubits[4:8]
            g0, g1 = work_qubits[8:10]

            # 4 leaf pair blocks
            leaf_pairs = [
                (controls[0], controls[1], h0),
                (controls[2], controls[3], h1),
                (controls[4], controls[5], h2),
                (controls[6], controls[7], h3),
            ]
            for left, right, target in leaf_pairs:
                apply_native_pair_block(
                    qc,
                    left,
                    right,
                    target,
                    thetas=self.theta[k : k + pair_params],
                    one_q_order=self._native_1q_order,
                )
                k += pair_params

            # bridge singles h_i -> b_i
            bridge_pairs = [
                (h0, b0),
                (h1, b1),
                (h2, b2),
                (h3, b3),
            ]
            for source, target in bridge_pairs:
                apply_native_single_block(
                    qc,
                    source,
                    target,
                    thetas=self.theta[k : k + single_params],
                    one_q_order=self._native_1q_order,
                )
                k += single_params

            # second-level pair blocks
            upper_pairs = [
                (b0, b1, g0),
                (b2, b3, g1),
            ]
            for left, right, target in upper_pairs:
                apply_native_pair_block(
                    qc,
                    left,
                    right,
                    target,
                    thetas=self.theta[k : k + pair_params],
                    one_q_order=self._native_1q_order,
                )
                k += pair_params

            # final 2-source fusion onto ancilla
            apply_native_final_block(
                qc,
                [g0, g1],
                ancilla,
                thetas=self.theta[k : k + final2_params],
                one_q_order=self._native_1q_order,
            )
            k += final2_params

        if k != self.n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {self.n_params}."
            )

        return qc

    def _build_native_heavyhex10_ansatz(self) -> QuantumCircuit:
        if self._n_controls != 10:
            raise ValueError(
                "native_heavyhex10 requires exactly 10 controls "
                "(m_time + n_price = 10)."
            )

        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        w = QuantumRegister(self._n_work, "w")
        a = QuantumRegister(1, "a")

        qc = QuantumCircuit(t, s, w, a, name=self._name)

        controls: list[Qubit] = [*t, *s]
        ancilla = a[0]
        work_qubits = list(w)

        self._control_qubit_indices = list(range(len(controls)))
        self._ancilla_qubit_index = qc.qubits.index(ancilla)

        # Work layout:
        # w[0:5]   = h0..h4
        # w[5:10]  = b0..b4
        # w[10:12] = g0..g1
        # w[12:14] = b5..b6
        n1 = len(self._native_1q_order)
        pair_params = 2 * n1 + 2
        single_params = 2 * n1 + 1
        final3_params = 2 * n1 + 3

        k = 0
        for _layer in range(self._n_layers):
            h0, h1, h2, h3, h4 = work_qubits[0:5]
            b0, b1, b2, b3, b4 = work_qubits[5:10]
            g0, g1 = work_qubits[10:12]
            b5, b6 = work_qubits[12:14]

            # 5 leaf pair blocks
            leaf_pairs = [
                (controls[0], controls[1], h0),  # t0,t1 -> h0
                (controls[2], controls[3], h1),  # t2,t3 -> h1
                (controls[4], controls[5], h2),  # s0,s1 -> h2
                (controls[6], controls[7], h3),  # s2,s3 -> h3
                (controls[8], controls[9], h4),  # s4,s5 -> h4
            ]
            for left, right, target in leaf_pairs:
                apply_native_pair_block(
                    qc,
                    left,
                    right,
                    target,
                    thetas=self.theta[k : k + pair_params],
                    one_q_order=self._native_1q_order,
                )
                k += pair_params

            # bridge singles h_i -> b_i
            bridge_pairs = [
                (h0, b0),
                (h1, b1),
                (h2, b2),
                (h3, b3),
                (h4, b4),
            ]
            for source, target in bridge_pairs:
                apply_native_single_block(
                    qc,
                    source,
                    target,
                    thetas=self.theta[k : k + single_params],
                    one_q_order=self._native_1q_order,
                )
                k += single_params

            # second-level pair blocks
            upper_pairs = [
                (b0, b1, g0),
                (b2, b3, g1),
            ]
            for left, right, target in upper_pairs:
                apply_native_pair_block(
                    qc,
                    left,
                    right,
                    target,
                    thetas=self.theta[k : k + pair_params],
                    one_q_order=self._native_1q_order,
                )
                k += pair_params

            # bridge singles g_i -> b_{5+i}
            top_bridges = [
                (g0, b5),
                (g1, b6),
            ]
            for source, target in top_bridges:
                apply_native_single_block(
                    qc,
                    source,
                    target,
                    thetas=self.theta[k : k + single_params],
                    one_q_order=self._native_1q_order,
                )
                k += single_params

            # final 3-source fusion onto ancilla
            apply_native_final_block(
                qc,
                [b5, b6, b4],
                ancilla,
                thetas=self.theta[k : k + final3_params],
                one_q_order=self._native_1q_order,
            )
            k += final3_params

        if k != self.n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {self.n_params}."
            )

        return qc

    def _build_eval_circuits(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        w = QuantumRegister(self._n_work, "w") if self._n_work > 0 else None
        a = QuantumRegister(1, "a")

        c_ctrl = ClassicalRegister(self._n_controls, "c")
        c_a = ClassicalRegister(1, "ca")

        if w is None:
            qc_eval = QuantumCircuit(t, s, a, name=f"{self._name}_eval")
            qc_meas = QuantumCircuit(t, s, a, c_ctrl, c_a, name=f"{self._name}_eval_meas")
            all_qubits_eval = list(qc_eval.qubits)
            all_qubits_meas = list(qc_meas.qubits)[: len(all_qubits_eval)]
        else:
            qc_eval = QuantumCircuit(t, s, w, a, name=f"{self._name}_eval")
            qc_meas = QuantumCircuit(
                t, s, w, a, c_ctrl, c_a, name=f"{self._name}_eval_meas"
            )
            all_qubits_eval = list(qc_eval.qubits)
            all_qubits_meas = list(qc_meas.qubits)[: len(all_qubits_eval)]

        controls_q: list[Qubit] = [*t, *s]

        for q in controls_q:
            qc_eval.h(q)

        qc_eval.compose(self.qc, qubits=all_qubits_eval, inplace=True)

        for q in controls_q:
            qc_meas.h(q)

        qc_meas.compose(self.qc, qubits=all_qubits_meas, inplace=True)

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