import numpy as np
from collections.abc import Callable, Sequence

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector, Qubit
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

from .rotations import (
    apply_controlled_su2_block,
    apply_native_final_block,
    apply_native_pair_block,
    apply_su2_block,
)


class CrcaCircuit:
    r"""
        Controlled Rotations Circuit Ansatz (CRCA) wrapper.
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
        
        # Validaciones restrictivas de longitud de tuplas eliminadas para permitir flexibilidad

        self._m_time = int(m_time)
        self._n_price = int(n_price)
        self._n_layers = int(n_layers)
        self._n_controls = self._m_time + self._n_price

        self._name = str(name)
        self._init_order = tuple(str(g).lower() for g in init_order)
        self._ctrl_order = tuple(str(g).lower() for g in ctrl_order)
        self._last_ctrl_order = tuple(str(g).lower() for g in last_ctrl_order)

        self._ansatz_type = str(ansatz_type).lower()
        self._native_1q_order = tuple(str(g).lower() for g in native_1q_order)

        if self._ansatz_type not in {
            "standard",
            "native_tree",
            "heavy_hex_star",
        }:
            raise ValueError(
                "ansatz_type must be one of {'standard', 'native_tree', 'heavy_hex_star'}."
            )

        if self._ansatz_type == "native_tree":
            self._n_work = self._native_required_work_qubits()
        else:
            self._n_work = 0
        
        self._n_params_per_layer = self._count_params_per_layer()

        self.theta = ParameterVector(
            "theta",
            self._n_layers * self._n_params_per_layer,
        )

        self.qc = self._build_ansatz()
        self.qc_eval, self._qc_eval_meas = self._build_eval_circuits()

        self._backend = Aer.get_backend("aer_simulator")
        self._tqc_eval_meas = transpile(self._qc_eval_meas, self._backend)
        self._tqc_eval_meas_param_set = set(self._tqc_eval_meas.parameters)

        self._n_clbits = len(self._tqc_eval_meas.clbits)
        self._ctrl_clbit_indices, self._a_clbit_index = self._extract_clbit_indices(
            self._tqc_eval_meas
        )

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
    def n_work(self) -> int:
        return self._n_work

    @property
    def dim_controls(self) -> int:
        return 2**self._n_controls

    @property
    def n_params_per_layer(self) -> int:
        return self._n_params_per_layer

    @property
    def n_params(self) -> int:
        return len(self.theta)

    def bind(self, x: np.ndarray, *, measured: bool = False) -> QuantumCircuit:
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
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

        full_bind_map = {
            self.theta[i]: float(x[i]) for i in range(self.n_params)
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

    def function_values(
        self,
        x: np.ndarray,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_params:
            raise ValueError(
                f"x must have length {self.n_params}; got {x.shape[0]}."
            )

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
        f = self._validate_target_f(f_tg, dim=self.dim_controls)

        def cost(x: np.ndarray) -> float:
            p = self.function_values(x, shots=shots, seed=seed)
            diff = p - f
            return float(np.mean(diff * diff))

        return cost

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

    def _count_params_per_layer(self) -> int:
        if self._ansatz_type in {"standard", "heavy_hex_star"}:
            if self._n_controls == 0:
                return len(self._init_order)
            return len(self._init_order) + len(self._ctrl_order) * (self._n_controls - 1) + len(self._last_ctrl_order)

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

        raise RuntimeError(f"Unsupported ansatz_type: {self._ansatz_type}")

    def _build_ansatz(self) -> QuantumCircuit:
        if self._ansatz_type == "standard":
            return self._build_standard_ansatz()
        if self._ansatz_type == "native_tree":
            return self._build_native_tree_ansatz()
        if self._ansatz_type == "heavy_hex_star":
            return self._build_heavy_hex_star_ansatz()
        
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

        n_init = len(self._init_order)
        n_ctrl = len(self._ctrl_order)
        n_last = len(self._last_ctrl_order)

        k = 0
        for _layer in range(self._n_layers):
            apply_su2_block(
                qc,
                target=ancilla,
                thetas=self.theta[k : k + n_init],
                order=self._init_order,
            )
            k += n_init

            if not controls:
                continue

            for ctrl in controls[:-1]:
                apply_controlled_su2_block(
                    qc,
                    control=ctrl,
                    target=ancilla,
                    thetas=self.theta[k : k + n_ctrl],
                    order=self._ctrl_order,
                )
                k += n_ctrl

            apply_controlled_su2_block(
                qc,
                control=controls[-1],
                target=ancilla,
                thetas=self.theta[k : k + n_last],
                order=self._last_ctrl_order,
            )
            k += n_last

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
    
    def _build_heavy_hex_star_ansatz(self) -> QuantumCircuit:
        if self._m_time < 2 or self._n_price < 1:
            raise ValueError("El ansatz 'heavy_hex_star' requiere al menos m_time >= 2 y n_price >= 1.")

        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        a = QuantumRegister(1, "a")

        qc = QuantumCircuit(t, s, a, name=self._name)
        controls: list[Qubit] = [*t, *s]
        ancilla = a[0]

        self._control_qubit_indices = list(range(len(controls)))
        self._ancilla_qubit_index = qc.qubits.index(ancilla)

        direct_ctrls = [t[0], s[0], s[1]]
        sliding_ctrls = [q for q in controls if q not in direct_ctrls]
        ordered_controls = direct_ctrls + sliding_ctrls

        n_init = len(self._init_order)
        n_ctrl = len(self._ctrl_order)
        n_last = len(self._last_ctrl_order)

        k = 0
        current_ancilla = ancilla
        swaps_tracker = []

        for _layer in range(self._n_layers):
            apply_su2_block(
                qc,
                target=current_ancilla,
                thetas=self.theta[k : k + n_init],
                order=self._init_order,
            )
            k += n_init

            if _layer % 2 == 0:
                # Capa Par: Movimiento hacia afuera (Outward)
                for i, ctrl in enumerate(ordered_controls):
                    is_last = (i == len(ordered_controls) - 1)
                    
                    thetas = self.theta[k : k + n_ctrl] if not is_last else self.theta[k : k + n_last]
                    order = self._ctrl_order if not is_last else self._last_ctrl_order
                    
                    apply_controlled_su2_block(
                        qc,
                        control=ctrl,
                        target=current_ancilla,
                        thetas=thetas,
                        order=order,
                    )
                    k += n_ctrl if not is_last else n_last
                    
                    if i >= 2 and not is_last:
                        qc.swap(ctrl, current_ancilla)
                        swaps_tracker.append((ctrl, current_ancilla))
                        current_ancilla = ctrl 
            else:
                # Capa Impar: Movimiento hacia adentro (Snake Pattern)
                # 1. Operar con el último control de la cadena (que ya está adyacente)
                tip_ctrl = ordered_controls[-1]
                apply_controlled_su2_block(
                    qc,
                    control=tip_ctrl,
                    target=current_ancilla,
                    thetas=self.theta[k : k + n_ctrl],
                    order=self._ctrl_order,
                )
                k += n_ctrl
                
                # 2. Desandar la cadena, aplicando rotaciones ANTES de cada swap de regreso
                for q_ctrl, q_anc in reversed(swaps_tracker):
                    # OJO: Por el swap previo, q_ctrl tiene la Ancilla física y q_anc el Control lógico
                    apply_controlled_su2_block(
                        qc,
                        control=q_anc,
                        target=q_ctrl,
                        thetas=self.theta[k : k + n_ctrl],
                        order=self._ctrl_order,
                    )
                    k += n_ctrl
                    
                    qc.swap(q_ctrl, q_anc)
                    current_ancilla = q_anc
                
                swaps_tracker.clear()
                
                # 3. Operar con los controles directos en el centro
                for i in [1, 0]:
                    is_last = (i == 0)
                    ctrl = ordered_controls[i]
                    thetas = self.theta[k : k + n_ctrl] if not is_last else self.theta[k : k + n_last]
                    order = self._ctrl_order if not is_last else self._last_ctrl_order
                    
                    apply_controlled_su2_block(
                        qc,
                        control=ctrl,
                        target=current_ancilla,
                        thetas=thetas,
                        order=order,
                    )
                    k += n_ctrl if not is_last else n_last

        if self._n_layers % 2 != 0:
            for q1, q2 in reversed(swaps_tracker):
                qc.swap(q1, q2)

        if k != self.n_params:
            raise RuntimeError(
                f"Parameter counting mismatch: used {k}, expected {self.n_params}."
            )

        return qc

    def _build_eval_circuits(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        t = QuantumRegister(self._m_time, "t")
        s = QuantumRegister(self._n_price, "s")
        a = QuantumRegister(1, "a")
        c_ctrl = ClassicalRegister(self._n_controls, "c")
        c_a = ClassicalRegister(1, "ca")

        if self._n_work > 0:
            w = QuantumRegister(self._n_work, "w")
            qc_eval = QuantumCircuit(t, s, w, a, name=f"{self._name}_eval")
            qc_meas = QuantumCircuit(
                t, s, w, a, c_ctrl, c_a, name=f"{self._name}_eval_meas"
            )
        else:
            qc_eval = QuantumCircuit(t, s, a, name=f"{self._name}_eval")
            qc_meas = QuantumCircuit(
                t, s, a, c_ctrl, c_a, name=f"{self._name}_eval_meas"
            )

        controls_q: list[Qubit] = [*t, *s]

        for q in controls_q:
            qc_eval.h(q)
        qc_eval.compose(self.qc, qubits=qc_eval.qubits, inplace=True)

        for q in controls_q:
            qc_meas.h(q)
        qc_meas.compose(
            self.qc,
            qubits=qc_meas.qubits[: len(self.qc.qubits)],
            inplace=True,
        )

        for idx, q in enumerate(controls_q):
            qc_meas.measure(q, c_ctrl[idx])
        qc_meas.measure(a[0], c_a[0])

        return qc_eval, qc_meas

    def _extract_clbit_indices(
        self,
        qc_meas: QuantumCircuit,
    ) -> tuple[list[int], int]:
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
        dim_ctrl = self.dim_controls
        n_ctrl = self._n_controls

        n_i = np.zeros(dim_ctrl, dtype=float)
        n_i1 = np.zeros(dim_ctrl, dtype=float)

        for raw_bs, c in counts.items():
            bs = raw_bs.replace(" ", "")
            if len(bs) != self._n_clbits:
                raise RuntimeError(
                    f"Unexpected bitstring length {len(bs)} "
                    f"(expected {self._n_clbits})."
                )

            def bit_at_clbit_index(cl_idx: int) -> int:
                pos_from_left = (self._n_clbits - 1) - cl_idx
                return 1 if bs[pos_from_left] == "1" else 0

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