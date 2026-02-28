# Python Utils
from typing import Mapping
import numpy as np

# Quantum - CVA utils
from quantum_cva.state_prep.qcbm.qcbm_circuit import MLQcbmCircuit
from quantum_cva.crca.crca_circuit import CrcaCircuit

# Qiskit utils
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

class QuantumCVACircuit:
    def __init__(
            self, 
            num_qubits_time: int,
            num_qubits_underlying: int,
            qcbm_circuit: MLQcbmCircuit,
            crca_circuit_exposure: CrcaCircuit,
            crca_circuit_default_prob: CrcaCircuit,
            crca_circuit_discount_factor: CrcaCircuit,
            recovery_rate: float,
            C_v: float,
            C_q: float,
            C_p: float,
            name: str="CVA_circuit",
            backend: str="statevector"
    ) -> None:

        self.num_qubits_time = int(num_qubits_time)
        self.num_qubits_underlying = int(num_qubits_underlying)

        self.qcbm_circuit: MLQcbmCircuit = qcbm_circuit
        self.crca_circuit_exposure: CrcaCircuit = crca_circuit_exposure
        self.crca_circuit_default_prob: CrcaCircuit = crca_circuit_default_prob
        self.crca_circuit_discount_factor: CrcaCircuit = crca_circuit_discount_factor

        self.recovery_rate = float(recovery_rate)
        self.C_v = float(C_v)
        self.C_q = float(C_q)
        self.C_p = float(C_p)   

        self.name = str(name)

        self.backend = str(backend)

        self.template = self._build_template() # void circuit to be filled
        self.meas_template, self._meas_map = self._build_meas_template()

        self._aer_backend = Aer.get_backend("aer_simulator")  if self.backend == "aer" else None

    # Circuit construction
    def build_cva_circuit(
            self,
            qcbm_params: np.ndarray[float],
            crca_exposure_params: np.ndarray[float],
            crca_default_params: np.ndarray[float],
            crca_discount_params: np.ndarray[float],
            measured: bool=False 
    ) -> float:
        """ Build CVA circuit with fixed parameters"""
        qc_qcbm = self.qcbm_circuit.bind(
            qcbm_params, 
            measured=False)
        qc_exposure = self.crca_circuit_exposure.bind(
            crca_exposure_params,
            measured=False)
        qc_default = self.crca_circuit_default_prob.bind(
            crca_default_params,
            measured=False
        )
        qc_discount = self.crca_circuit_discount_factor.bind(
            crca_discount_params,
            measured=False
        )
        cva_quantum_circuit = (self.meas_template if measured 
                               else self.template).copy()
        self._compose(cva_quantum_circuit, qc_qcbm, qc_exposure, qc_default, qc_discount)

        return cva_quantum_circuit
    
    # Probability estimation of |111><111| projector
    def prob_111(
            self,
            qcbm_params: np.ndarray[float],
            crca_exposure_params: np.ndarray[float],    
            crca_default_params: np.ndarray[float],
            crca_discount_params: np.ndarray[float],
    ) -> float:
        """ Returns p111
        - backend="statevector": exact via Statevector
        - backend="aer": requires measure_prob_111(shots=...)
        """
        if self.backend == "aer":
            raise RuntimeError("backend='aer' not supported for prob_111. Use measure_prob_111 instead.")
        
        cva_quantum_circuit = self.build_cva_circuit(
            qcbm_params, 
            crca_exposure_params, 
            crca_default_params, 
            crca_discount_params, 
            measured=False
        )

        return self._prob_111_statevector(cva_quantum_circuit)
    
    def measure_prob_111(
            self,
            qcbm_params: np.ndarray[float],
            crca_exposure_params: np.ndarray[float],    
            crca_default_params: np.ndarray[float],
            crca_discount_params: np.ndarray[float],
            shots: int,
            seed: int | None = None
    ) -> float:
        """ Returns p111 via measurement on aer simulator (mandatory shots)."""
        if self.backend != "aer":
            raise RuntimeError("measure_prob_111 only supported for backend='aer'. Use prob_111 instead.")
        integer_shots = int(shots)

        if integer_shots <= 0:
            raise ValueError("shots must be a positive integer.")
 
        cva_quantum_circuit = self.build_cva_circuit(
            qcbm_params, 
            crca_exposure_params, 
            crca_default_params, 
            crca_discount_params, 
            measured=True
        )

        transpiled_circuit = transpile(cva_quantum_circuit, 
                                       self._aer_backend, 
                                       )

        run_kwargs: dict[str, object] = int(seed)
        if seed is not None:
            run_kwargs["seed_simulator"] = int(seed)

        counts = self._aer_backend.run(transpiled_circuit, **run_kwargs).result().get_counts()
       
        return self._prob_111_from_counts(counts)

        # CVA
    def cva_from_prob(
            self, 
            p111: float
    ) -> float:
        M = float(2 ** self.num_qubits_time)
        return float(M * (1.0 - self.recovery_rate) * self.C_v * self.C_q * self.C_p * float(p111))

    def cva(
        self,
        *,
        qcbm_params: np.ndarray,
        exposure_params: np.ndarray,
        default_prob_params: np.ndarray,
        discount_factor_params: np.ndarray,
    ) -> float:
        """Exact CVA (only if backend='statevector')."""
        p111 = self.prob_111(
            qcbm_params=qcbm_params,
            crca_exposure_params=exposure_params,
            crca_default_params=default_prob_params,
            crca_discount_params=discount_factor_params,
        )
        return self.cva_from_prob(p111)

    def measure_cva(
        self,
        *,
        qcbm_params: np.ndarray,
        exposure_params: np.ndarray,
        default_prob_params: np.ndarray,
        discount_factor_params: np.ndarray,
        shots: int,
        seed: int | None = None,
    ) -> float:
        """Estimated CVA via measurement (only if backend='aer')."""
        p111 = self.measure_prob_111(
            qcbm_params=qcbm_params,
            crca_exposure_params=exposure_params,
            crca_default_params=default_prob_params,
            crca_discount_params=discount_factor_params,
            shots=shots,
            seed=seed,
        )
        return self.cva_from_prob(p111)

    # -------------------------
    # Internals
    # -------------------------
    def _build_template(self) -> QuantumCircuit:
        t = QuantumRegister(self.num_qubits_time, "t")
        s = QuantumRegister(self.num_qubits_underlying, "s")
        a_v = QuantumRegister(1, "a_v")
        a_q = QuantumRegister(1, "a_q")
        a_p = QuantumRegister(1, "a_p")
        return QuantumCircuit(t, s, a_v, a_q, a_p, name=self.name)

    def _build_meas_template(self) -> tuple[QuantumCircuit, dict[str, int]]:
        qc = self._build_template()

        c_v = ClassicalRegister(1, "c_v")
        c_q = ClassicalRegister(1, "c_q")
        c_p = ClassicalRegister(1, "c_p")
        qc.add_register(c_v, c_q, c_p)

        qc.measure(qc.get_register("a_v")[0], c_v[0])
        qc.measure(qc.get_register("a_q")[0], c_q[0])
        qc.measure(qc.get_register("a_p")[0], c_p[0])

        clbits = list(qc.clbits)
        return qc, {"c_v": clbits.index(c_v[0]), "c_q": clbits.index(c_q[0]), "c_p": clbits.index(c_p[0])}

    def _compose(
        self,
        qc: QuantumCircuit,
        *,
        qc_qcbm: QuantumCircuit,
        qc_v: QuantumCircuit,
        qc_q: QuantumCircuit,
        qc_p: QuantumCircuit,
    ) -> None:
        t = list(qc.get_register("t"))
        s = list(qc.get_register("s"))
        a_v = qc.get_register("a_v")[0]
        a_q = qc.get_register("a_q")[0]
        a_p = qc.get_register("a_p")[0]

        qc.compose(qc_qcbm, qubits=[*t, *s], inplace=True)
        qc.compose(qc_v, qubits=[*t, *s, a_v], inplace=True)
        qc.compose(qc_q, qubits=[*t, a_q], inplace=True)
        qc.compose(qc_p, qubits=[*t, a_p], inplace=True)

    def _prob_111_statevector(
            self, 
            quantum_circuit: QuantumCircuit
    ) -> float:
        qcircuit_statevctor = Statevector.from_instruction(quantum_circuit)
        probs = np.asarray(qcircuit_statevctor.probabilities(), dtype=float)

        # order: [t..., s..., a_v, a_q, a_p]
        num_state_qubits = self.num_qubits_time + self.num_qubits_underlying
        ancilla_exposure_pos = num_state_qubits
        ancilla_default_pos = num_state_qubits + 1
        ancilla_discount_pos = num_state_qubits + 2

        p111 = 0.0
        for idx, p in enumerate(probs):
            exposure_bit = (idx >> ancilla_exposure_pos) & 1
            default_bit = (idx >> ancilla_default_pos) & 1
            discount_bit = (idx >> ancilla_discount_pos) & 1
            if (exposure_bit, default_bit, discount_bit) == (1, 1, 1):
                p111 += float(p)
        return float(p111)

    def _prob_111_from_counts(
            self,
            measurement_counts: Mapping[str, int]
    ) -> float:
        total_shots = float(sum(int(count) for count in measurement_counts.values()))
        if total_shots <= 0.0:
            return 0.0

        num_classical_bits = len(self.meas_template.clbits)

        exposure_bit_index = int(self._meas_map["c_v"])
        default_bit_index = int(self._meas_map["c_q"])
        discount_bit_index = int(self._meas_map["c_p"])

        def extract_bit(bitstring: str, *, classical_index: int) -> int:
            position_from_left = (num_classical_bits - 1) - classical_index
            return 1 if bitstring[position_from_left] == "1" else 0

        counts_111 = 0.0

        for raw_bitstring, count in measurement_counts.items():
            bitstring = raw_bitstring.replace(" ", "")
            if len(bitstring) != num_classical_bits:
                raise RuntimeError(
                    f"Unexpected bitstring length {len(bitstring)} "
                    f"(expected {num_classical_bits})."
                )

            exposure_bit = extract_bit(bitstring, classical_index=exposure_bit_index)
            default_bit = extract_bit(bitstring, classical_index=default_bit_index)
            discount_bit = extract_bit(bitstring, classical_index=discount_bit_index)

            if (exposure_bit, default_bit, discount_bit) == (1, 1, 1):
                counts_111 += float(count)

        return float(counts_111 / total_shots)