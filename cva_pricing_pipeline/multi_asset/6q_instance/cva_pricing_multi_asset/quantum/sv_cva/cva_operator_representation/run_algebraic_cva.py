from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (
    QuantumCVACircuit,
)
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
SV_CVA_DIR = SCRIPT_DIR.parent
if str(SV_CVA_DIR) not in sys.path:
    sys.path.insert(0, str(SV_CVA_DIR))

from utils_run_ideal import (
    _as_1d_float,
    _assert_param_size,
    _build_improved_cva_initial_layout,
    _metadata_dict,
    _npz_int,
    _npz_str,
    _select_layout_for_training,
    _transpile_with_layout,
    _two_qubit_gate_count,
)


BACKEND_NAME = "ibm_basquecountry"
SEED_TRANSPILER = 1234
TRANSPILATION_OPT_LEVEL = 3
APPROXIMATION_DEGREE = 1.0
MAX_TRANSPILED_MATRIX_ACTIVE_QUBITS = 10


def _repo_root() -> pathlib.Path:
    return next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _draw_text(circuit: QuantumCircuit) -> str:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The encoding .* has a limited charset.*",
            category=RuntimeWarning,
        )
        return str(
            circuit.draw(
                output="text",
                fold=-1,
                idle_wires=False,
                cregbundle=False,
            )
        )


def _draw_latex_source(circuit: QuantumCircuit) -> str:
    try:
        return str(circuit.draw(output="latex_source", idle_wires=False))
    except Exception as exc:
        return f"% Could not generate Qiskit latex_source: {exc}\n"


def _qasm3(circuit: QuantumCircuit) -> str:
    try:
        return qasm3.dumps(circuit)
    except Exception as exc:
        return f"// Could not generate OpenQASM 3: {exc}\n"


def _format_complex(value: complex, *, precision: int = 10) -> str:
    z = complex(value)
    real = 0.0 if abs(z.real) < 10 ** (-(precision - 1)) else z.real
    imag = 0.0 if abs(z.imag) < 10 ** (-(precision - 1)) else z.imag
    return f"{real:.{precision}e}{imag:+.{precision}e}j"


def _format_complex_latex(value: complex, *, precision: int = 6) -> str:
    z = complex(value)
    real = 0.0 if abs(z.real) < 10 ** (-(precision - 1)) else z.real
    imag = 0.0 if abs(z.imag) < 10 ** (-(precision - 1)) else z.imag
    if imag == 0.0:
        return f"{real:.{precision}e}"
    if real == 0.0:
        return f"{imag:.{precision}e}i"
    sign = "+" if imag >= 0.0 else "-"
    return f"{real:.{precision}e}{sign}{abs(imag):.{precision}e}i"


def _write_complex_matrix_txt(path: pathlib.Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"shape: {matrix.shape}\n")
        for row in matrix:
            handle.write(" ".join(_format_complex(value) for value in row))
            handle.write("\n")


def _write_complex_matrix_latex(path: pathlib.Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\[\n")
        handle.write("\\begin{bmatrix}\n")
        for row_index, row in enumerate(matrix):
            cells = " & ".join(_format_complex_latex(value) for value in row)
            suffix = " \\\\\n" if row_index + 1 < matrix.shape[0] else "\n"
            handle.write(cells + suffix)
        handle.write("\\end{bmatrix}\n")
        handle.write("\\]\n")


def _rename_parameters(
    circuit: QuantumCircuit,
    old_parameters: ParameterVector,
    new_name: str,
) -> tuple[QuantumCircuit, ParameterVector]:
    new_parameters = ParameterVector(new_name, len(old_parameters))
    bind_map = {
        old_parameters[index]: new_parameters[index]
        for index in range(len(old_parameters))
    }
    return circuit.assign_parameters(bind_map, inplace=False), new_parameters


def _build_unbound_cva_circuit(
    quantum_cva_circuit: QuantumCVACircuit,
) -> tuple[QuantumCircuit, dict[str, ParameterVector]]:
    qcbm_qc, theta_qcbm = _rename_parameters(
        quantum_cva_circuit.qcbm_circuit.qc,
        quantum_cva_circuit.qcbm_circuit.theta,
        "theta_qcbm",
    )
    exposure_qc, theta_v = _rename_parameters(
        quantum_cva_circuit.crca_circuit_exposure.qc,
        quantum_cva_circuit.crca_circuit_exposure.theta,
        "theta_v",
    )
    default_qc, theta_q = _rename_parameters(
        quantum_cva_circuit.crca_circuit_default_prob.qc,
        quantum_cva_circuit.crca_circuit_default_prob.theta,
        "theta_q",
    )
    discount_qc, theta_p = _rename_parameters(
        quantum_cva_circuit.crca_circuit_discount_factor.qc,
        quantum_cva_circuit.crca_circuit_discount_factor.theta,
        "theta_p",
    )

    circuit = quantum_cva_circuit.template.copy()
    quantum_cva_circuit._compose(
        circuit,
        qc_qcbm=qcbm_qc,
        qc_v=exposure_qc,
        qc_q=default_qc,
        qc_p=discount_qc,
    )
    return circuit, {
        "theta_qcbm": theta_qcbm,
        "theta_v": theta_v,
        "theta_q": theta_q,
        "theta_p": theta_p,
    }


def _active_qubit_indices(circuit: QuantumCircuit) -> list[int]:
    return sorted(
        {
            int(circuit.find_bit(qubit).index)
            for instruction in circuit.data
            for qubit in instruction.qubits
        }
    )


def _reduced_active_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    active = _active_qubit_indices(circuit)
    index_map = {old_index: new_index for new_index, old_index in enumerate(active)}
    reduced = QuantumCircuit(len(active), name=f"{circuit.name}_active")

    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"barrier", "delay", "measure"}:
            continue
        qargs = [
            reduced.qubits[index_map[int(circuit.find_bit(qubit).index)]]
            for qubit in instruction.qubits
        ]
        reduced.append(operation.copy(), qargs)

    return reduced


def _projection_mask(
    num_qubits: int,
    objective_qubits: Sequence[int],
    bitstring: str,
) -> np.ndarray:
    mask = np.zeros(2**num_qubits, dtype=bool)
    expected = [int(bit) for bit in bitstring]
    for basis_index in range(mask.size):
        bits = [
            (basis_index >> int(qubit_index)) & 1
            for qubit_index in objective_qubits
        ]
        mask[basis_index] = bits == expected
    return mask


def _param_text(param: Any) -> str:
    return str(param).replace("[", "_").replace("]", "")


def _param_latex(param: Any) -> str:
    raw = str(param)
    if "[" not in raw or not raw.endswith("]"):
        return raw.replace("_", r"\_")
    name, index = raw[:-1].split("[", maxsplit=1)
    if name.startswith("theta_"):
        family = name.removeprefix("theta_")
        return rf"\theta_{{\mathrm{{{family}}},{index}}}"
    return rf"{name.replace('_', r'\_')}_{{{index}}}"


def _qubit_labels(circuit: QuantumCircuit, qubits: Sequence[Any]) -> list[str]:
    labels: list[str] = []
    for qubit in qubits:
        bit = circuit.find_bit(qubit)
        labels.append(f"q{int(bit.index)}")
    return labels


def _gate_matrix_text(operation: Any) -> str:
    name = operation.name
    params = operation.params
    theta = _param_text(params[0]) if params else "theta"

    if name == "rx":
        return (
            f"Rx({theta}) = [[cos({theta}/2), -i sin({theta}/2)], "
            f"[-i sin({theta}/2), cos({theta}/2)]]"
        )
    if name == "ry":
        return (
            f"Ry({theta}) = [[cos({theta}/2), -sin({theta}/2)], "
            f"[sin({theta}/2), cos({theta}/2)]]"
        )
    if name == "rz":
        return f"Rz({theta}) = diag(exp(-i {theta}/2), exp(i {theta}/2))"
    if name == "rzz":
        return (
            f"Rzz({theta}) = diag(exp(-i {theta}/2), exp(i {theta}/2), "
            f"exp(i {theta}/2), exp(-i {theta}/2))"
        )
    if name == "crx":
        return (
            f"CRx({theta}) = [[1,0,0,0], [0,1,0,0], "
            f"[0,0,cos({theta}/2),-i sin({theta}/2)], "
            f"[0,0,-i sin({theta}/2),cos({theta}/2)]]"
        )
    if name == "cry":
        return (
            f"CRy({theta}) = [[1,0,0,0], [0,1,0,0], "
            f"[0,0,cos({theta}/2),-sin({theta}/2)], "
            f"[0,0,sin({theta}/2),cos({theta}/2)]]"
        )
    if name == "crz":
        return (
            f"CRz({theta}) = diag(1, 1, exp(-i {theta}/2), "
            f"exp(i {theta}/2))"
        )
    if name == "swap":
        return "SWAP = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]"
    if name == "cx":
        return "CX = |0><0| tensor I + |1><1| tensor X"
    if name == "cz":
        return "CZ = diag(1, 1, 1, -1)"
    return f"{name}: local symbolic matrix not listed explicitly"


def _gate_matrix_latex(operation: Any) -> str:
    name = operation.name
    params = operation.params
    theta = _param_latex(params[0]) if params else r"\theta"

    if name == "rx":
        return (
            rf"R_x({theta})=\begin{{bmatrix}}"
            rf"\cos({theta}/2)&-i\sin({theta}/2)\\"
            rf"-i\sin({theta}/2)&\cos({theta}/2)"
            rf"\end{{bmatrix}}"
        )
    if name == "ry":
        return (
            rf"R_y({theta})=\begin{{bmatrix}}"
            rf"\cos({theta}/2)&-\sin({theta}/2)\\"
            rf"\sin({theta}/2)&\cos({theta}/2)"
            rf"\end{{bmatrix}}"
        )
    if name == "rz":
        return (
            rf"R_z({theta})=\begin{{bmatrix}}"
            rf"e^{{-i{theta}/2}}&0\\0&e^{{i{theta}/2}}"
            rf"\end{{bmatrix}}"
        )
    if name == "rzz":
        return (
            rf"R_{{zz}}({theta})=\operatorname{{diag}}"
            rf"(e^{{-i{theta}/2}},e^{{i{theta}/2}},"
            rf"e^{{i{theta}/2}},e^{{-i{theta}/2}})"
        )
    if name == "crx":
        return (
            rf"CR_x({theta})=\begin{{bmatrix}}"
            rf"1&0&0&0\\0&1&0&0\\"
            rf"0&0&\cos({theta}/2)&-i\sin({theta}/2)\\"
            rf"0&0&-i\sin({theta}/2)&\cos({theta}/2)"
            rf"\end{{bmatrix}}"
        )
    if name == "cry":
        return (
            rf"CR_y({theta})=\begin{{bmatrix}}"
            rf"1&0&0&0\\0&1&0&0\\"
            rf"0&0&\cos({theta}/2)&-\sin({theta}/2)\\"
            rf"0&0&\sin({theta}/2)&\cos({theta}/2)"
            rf"\end{{bmatrix}}"
        )
    if name == "crz":
        return (
            rf"CR_z({theta})=\operatorname{{diag}}"
            rf"(1,1,e^{{-i{theta}/2}},e^{{i{theta}/2}})"
        )
    if name == "swap":
        return (
            r"\operatorname{SWAP}=\begin{bmatrix}"
            r"1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1"
            r"\end{bmatrix}"
        )
    if name == "cx":
        return r"CX=|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes X"
    if name == "cz":
        return r"CZ=\operatorname{diag}(1,1,1,-1)"
    return name.replace("_", r"\_")


def _single_qubit_recurrence_text(
    *,
    gate_name: str,
    theta: str,
    qubit: int,
) -> list[str]:
    c = f"cos({theta}/2)"
    s = f"sin({theta}/2)"
    flip = f"z xor 2^{qubit}"

    if gate_name == "rx":
        return [
            f"if b_{qubit}(z)=0: psi_next(z) = {c} psi(z) - i {s} psi({flip})",
            f"if b_{qubit}(z)=1: psi_next(z) = -i {s} psi({flip}) + {c} psi(z)",
        ]
    if gate_name == "ry":
        return [
            f"if b_{qubit}(z)=0: psi_next(z) = {c} psi(z) - {s} psi({flip})",
            f"if b_{qubit}(z)=1: psi_next(z) = {s} psi({flip}) + {c} psi(z)",
        ]
    if gate_name == "rz":
        return [
            f"if b_{qubit}(z)=0: psi_next(z) = exp(-i {theta}/2) psi(z)",
            f"if b_{qubit}(z)=1: psi_next(z) = exp(i {theta}/2) psi(z)",
        ]
    raise ValueError(f"Unsupported single-qubit recurrence gate: {gate_name}")


def _controlled_recurrence_text(
    *,
    gate_name: str,
    theta: str,
    control: int,
    target: int,
) -> list[str]:
    c = f"cos({theta}/2)"
    s = f"sin({theta}/2)"
    flip = f"z xor 2^{target}"

    if gate_name == "crx":
        active = [
            f"if b_{target}(z)=0: psi_next(z) = {c} psi(z) - i {s} psi({flip})",
            f"if b_{target}(z)=1: psi_next(z) = -i {s} psi({flip}) + {c} psi(z)",
        ]
    elif gate_name == "cry":
        active = [
            f"if b_{target}(z)=0: psi_next(z) = {c} psi(z) - {s} psi({flip})",
            f"if b_{target}(z)=1: psi_next(z) = {s} psi({flip}) + {c} psi(z)",
        ]
    elif gate_name == "crz":
        active = [
            f"if b_{target}(z)=0: psi_next(z) = exp(-i {theta}/2) psi(z)",
            f"if b_{target}(z)=1: psi_next(z) = exp(i {theta}/2) psi(z)",
        ]
    else:
        raise ValueError(f"Unsupported controlled recurrence gate: {gate_name}")

    return [
        f"if b_{control}(z)=0: psi_next(z) = psi(z)",
        f"if b_{control}(z)=1 and " + active[0].removeprefix("if "),
        f"if b_{control}(z)=1 and " + active[1].removeprefix("if "),
    ]


def _gate_recurrence_text(
    circuit: QuantumCircuit,
    instruction: Any,
    *,
    step_index: int,
) -> list[str]:
    operation = instruction.operation
    name = operation.name
    qubits = [int(circuit.find_bit(qubit).index) for qubit in instruction.qubits]
    theta = _param_text(operation.params[0]) if operation.params else "theta"
    header = (
        f"Step {step_index:03d}: G_{step_index} = {name} "
        f"on qubits {qubits}; psi = psi_{step_index}, "
        f"psi_next = psi_{step_index + 1}"
    )

    if name in {"rx", "ry", "rz"}:
        return [header, *_single_qubit_recurrence_text(gate_name=name, theta=theta, qubit=qubits[0])]
    if name == "rzz":
        q0, q1 = qubits
        return [
            header,
            f"if b_{q0}(z)=b_{q1}(z): psi_next(z) = exp(-i {theta}/2) psi(z)",
            f"if b_{q0}(z)!=b_{q1}(z): psi_next(z) = exp(i {theta}/2) psi(z)",
        ]
    if name in {"crx", "cry", "crz"}:
        control, target = qubits
        return [
            header,
            *_controlled_recurrence_text(
                gate_name=name,
                theta=theta,
                control=control,
                target=target,
            ),
        ]
    if name == "swap":
        q0, q1 = qubits
        return [
            header,
            f"psi_next(z) = psi(swap_bits(z, {q0}, {q1}))",
        ]
    if name == "cx":
        control, target = qubits
        return [
            header,
            f"if b_{control}(z)=0: psi_next(z) = psi(z)",
            f"if b_{control}(z)=1: psi_next(z) = psi(z xor 2^{target})",
        ]
    if name == "cz":
        control, target = qubits
        return [
            header,
            f"if b_{control}(z)=1 and b_{target}(z)=1: psi_next(z) = -psi(z)",
            "otherwise: psi_next(z) = psi(z)",
        ]

    return [header, f"recurrence not listed for gate '{name}'"]


def _write_symbolic_cva_amplitudes(
    output_dir: pathlib.Path,
    circuit: QuantumCircuit,
) -> None:
    final_step = len(circuit.data)

    text_lines = [
        "Symbolic CVA objective amplitudes",
        "=================================",
        "",
        "This is the symbolic amplitude of the CVA objective subspace.",
        "The full expanded expressions are intentionally not expanded: with 103",
        "parameters they become enormous. The recurrence below is exact and keeps",
        "the sin/cos structure visible gate by gate.",
        "",
        "Little-endian convention:",
        "  z = sum_j b_j(z) 2^j for logical qubits q0..q8.",
        "  q0,q1 are time; q2..q5 are underlying/state; q6,q7,q8 are",
        "  a_exposure,a_default,a_discount.",
        "",
        "Initial amplitude:",
        "  psi_0(0) = 1",
        "  psi_0(z != 0) = 0",
        "",
        "CVA objective amplitudes:",
        "  alpha_x(theta) = <x,111|A(theta)|0^9>",
        "  x = sum_{j=0}^5 x_j 2^j",
        "  alpha_x(theta) = psi_N(x + 2^6 + 2^7 + 2^8)",
        f"  N = {final_step}",
        "",
        "Gate-by-gate symbolic recurrence",
        "--------------------------------",
        "",
    ]

    for step_index, instruction in enumerate(circuit.data):
        text_lines.extend(_gate_recurrence_text(circuit, instruction, step_index=step_index))
        text_lines.append("")

    text_lines.extend(
        [
            "Final CVA-subspace amplitudes",
            "-----------------------------",
            "",
        ]
    )
    for x_value in range(2**6):
        bits = "".join(str((x_value >> bit) & 1) for bit in range(6))
        basis_index = x_value + (1 << 6) + (1 << 7) + (1 << 8)
        full_bits = bits + "111"
        text_lines.append(
            f"alpha_{bits}(theta) = <q0..q8={full_bits}|A(theta)|0^9> "
            f"= psi_{final_step}({basis_index})"
        )

    text_lines.extend(
        [
            "",
            "CVA probability and value",
            "-------------------------",
            "",
            "p_111(theta) = sum_{x=0}^{63} |alpha_x(theta)|^2",
            "CVA(theta) = 2^m (1-R) C_v C_q C_p p_111(theta)",
            "",
        ]
    )

    latex_lines = [
        r"\[",
        r"\psi_0(0)=1,\qquad \psi_0(z\ne0)=0",
        r"\]",
        r"\[",
        r"\alpha_x(\theta)=\langle x,111|A(\theta)|0^9\rangle"
        rf"=\psi_{{{final_step}}}(x+2^6+2^7+2^8)",
        r"\]",
        r"\[",
        r"p_{111}(\theta)=\sum_{x=0}^{63}|\alpha_x(\theta)|^2,\qquad "
        r"\mathrm{CVA}(\theta)=2^m(1-R)C_vC_qC_p\,p_{111}(\theta)",
        r"\]",
        "",
        r"% The text file contains the full gate-by-gate recurrence with sin/cos.",
        r"\begin{align*}",
    ]
    for x_value in range(2**6):
        bits = "".join(str((x_value >> bit) & 1) for bit in range(6))
        basis_index = x_value + (1 << 6) + (1 << 7) + (1 << 8)
        latex_lines.append(
            rf"\alpha_{{{bits}}}(\theta)&=\psi_{{{final_step}}}({basis_index})\\"
        )
    latex_lines.append(r"\end{align*}")
    latex_lines.append("")

    _write_text(
        output_dir / "cva_objective_symbolic_amplitudes.txt",
        "\n".join(text_lines),
    )
    _write_text(
        output_dir / "cva_objective_symbolic_amplitudes.tex",
        "\n".join(latex_lines),
    )


def _write_symbolic_gate_products(
    output_dir: pathlib.Path,
    circuit: QuantumCircuit,
) -> None:
    operation_symbols = [f"G_{{{idx}}}" for idx, _ in enumerate(circuit.data)]
    product_latex = " ".join(reversed(operation_symbols))
    product_text = " ".join(f"G_{idx}" for idx in reversed(range(len(circuit.data))))

    text_lines = [
        "Symbolic gate-level CVA operator",
        "================================",
        "",
        "Qiskit does not expand parameterized circuits into symbolic dense matrices.",
        "This file writes the exact factored symbolic operator instead.",
        "",
        f"A(theta) = {product_text}",
        "",
        "Each embedded operator G_k acts as the listed local matrix on its qubits,",
        "tensored with identity on the remaining logical qubits, with Qiskit's",
        "little-endian qubit indexing convention.",
        "",
    ]
    latex_lines = [
        r"\[",
        rf"A(\theta)={product_latex}",
        r"\]",
        "",
        r"\begin{align*}",
    ]

    for idx, instruction in enumerate(circuit.data):
        operation = instruction.operation
        qlabels = _qubit_labels(circuit, instruction.qubits)
        params = ", ".join(_param_text(param) for param in operation.params) or "-"
        text_lines.extend(
            [
                f"G_{idx}: gate={operation.name}, qubits={qlabels}, params={params}",
                f"  { _gate_matrix_text(operation) }",
                "",
            ]
        )

        qlatex = ",".join(qlabels)
        latex_lines.append(
            rf"G_{{{idx}}}&:\ \text{{{operation.name} on {qlatex}}},\quad "
            + _gate_matrix_latex(operation)
            + r"\\"
        )

    latex_lines.append(r"\end{align*}")
    latex_lines.append("")

    _write_text(output_dir / "logical_unbound_symbolic_gates.txt", "\n".join(text_lines))
    _write_text(output_dir / "logical_unbound_symbolic_gates.tex", "\n".join(latex_lines))


def _write_algebraic_forms(
    output_dir: pathlib.Path,
    quantum_cva_circuit: QuantumCVACircuit,
) -> None:
    num_state_qubits = (
        quantum_cva_circuit.num_qubits_time
        + quantum_cva_circuit.num_qubits_underlying
    )
    scale = (
        2 ** quantum_cva_circuit.num_qubits_time
        * (1.0 - quantum_cva_circuit.recovery_rate)
        * quantum_cva_circuit.C_v
        * quantum_cva_circuit.C_q
        * quantum_cva_circuit.C_p
    )

    text = "\n".join(
        [
            "Algebraic CVA operator form",
            "===========================",
            "",
            "Register order:",
            "  [t0, t1, s0, s1, s2, s3, a_exposure, a_default, a_discount]",
            "",
            "State-preparation operator:",
            "  A(theta) = U_p(theta_p) U_q(theta_q) U_v(theta_v) U_qcbm(theta_qcbm)",
            "",
            "Composition order in Qiskit:",
            "  1. U_qcbm on [t, s]",
            "  2. U_v on [t, s, a_exposure]",
            "  3. U_q on [t, a_default]",
            "  4. U_p on [t, a_discount]",
            "",
            "CVA projector:",
            f"  Pi_111 = I_(2^{num_state_qubits}) tensor |111><111|",
            "  objective_qubits = [6, 7, 8]",
            "",
            "Projected state and amplitude:",
            "  |xi> = A(theta*) |0^9>",
            "  |xi_111> = Pi_111 |xi>",
            "  p_111 = <xi| Pi_111 |xi> = || |xi_111> ||^2",
            "",
            "CVA post-processing:",
            "  CVA = 2^m (1 - R) C_v C_q C_p p_111",
            f"  scale = {scale:.16e}",
            "",
        ]
    )
    _write_text(output_dir / "algebraic_operator_form.txt", text)

    latex = "\n".join(
        [
            "\\[",
            r"A(\theta) = U_p(\theta_p) U_q(\theta_q) "
            r"U_v(\theta_v) U_{\mathrm{QCBM}}(\theta_{\mathrm{qcbm}})",
            "\\]",
            "\\[",
            r"\Pi_{111} = I_{2^{" + str(num_state_qubits) + r"}} "
            r"\otimes |111\rangle\langle 111|",
            "\\]",
            "\\[",
            r"|\xi\rangle = A(\theta^\star)|0^9\rangle,\quad "
            r"|\xi_{111}\rangle = \Pi_{111}|\xi\rangle",
            "\\]",
            "\\[",
            r"p_{111} = \langle \xi|\Pi_{111}|\xi\rangle "
            r"= \|\Pi_{111}|\xi\rangle\|_2^2",
            "\\]",
            "\\[",
            r"\mathrm{CVA} = 2^m(1-R)C_vC_qC_p\,p_{111}",
            "\\]",
            f"\\[\n\\mathrm{{scale}} = {scale:.16e}\n\\]\n",
        ]
    )
    _write_text(output_dir / "algebraic_operator_form.tex", latex)


def _write_projection_forms(
    output_dir: pathlib.Path,
    *,
    scale: float,
) -> None:
    text = "\n".join(
        [
            "Projection on the CVA objective subspace",
            "========================================",
            "",
            "Let |xi> = A |0^9>.",
            "The CVA objective qubits are [6, 7, 8].",
            "",
            "Pi_111 = I_(2^6) tensor |111><111|",
            "|xi_111> = Pi_111 |xi>",
            "p_111 = <xi|Pi_111|xi> = |||xi_111>||^2",
            f"CVA = {scale:.16e} * p_111",
            "",
        ]
    )
    _write_text(output_dir / "projection_111_form.txt", text)

    latex = "\n".join(
        [
            "\\[",
            r"\Pi_{111} = I_{2^6}\otimes |111\rangle\langle 111|",
            "\\]",
            "\\[",
            r"|\xi_{111}\rangle = \Pi_{111}A|0^9\rangle",
            "\\]",
            "\\[",
            r"p_{111} = \langle 0^9|A^\dagger\Pi_{111}A|0^9\rangle "
            r"= \|\Pi_{111}A|0^9\rangle\|_2^2",
            "\\]",
            "\\[",
            rf"\mathrm{{CVA}} = {scale:.16e}\,p_{{111}}",
            "\\]",
            "",
        ]
    )
    _write_text(output_dir / "projection_111_form.tex", latex)


def _load_inputs(repo_root: pathlib.Path) -> dict[str, Any]:
    data_root = repo_root / "data" / "multi_asset" / "6q_instance"
    training_root = data_root / "quantum" / "training"
    benchmark_root = data_root / "benchmark"

    return {
        "data_root": data_root,
        "classical": np.load(
            benchmark_root / "three_asset_instance.npz",
            allow_pickle=True,
        ),
        "qcbm": np.load(
            training_root / "qcbm" / "statevector" / "training_qcbm_heavyhex6_6lay.npz",
            allow_pickle=True,
        ),
        "exposure": np.load(
            training_root / "crca" / "positive_exposure" / "training_heavy_hex_star.npz",
            allow_pickle=True,
        ),
        "default": np.load(
            training_root / "crca" / "default_probabilities" / "training_crca2.npz",
            allow_pickle=True,
        ),
        "discount": np.load(
            training_root / "crca" / "discount_factors" / "training_crca2.npz",
            allow_pickle=True,
        ),
    }


def _build_cva_objects(inputs: dict[str, Any], real_backend: Any) -> dict[str, Any]:
    qcbm_data = inputs["qcbm"]
    exposure_data = inputs["exposure"]
    default_data = inputs["default"]
    discount_data = inputs["discount"]
    classical = inputs["classical"]

    num_qubits_time = 2
    num_qubits_underlying = 4
    total_num_qubits = num_qubits_time + num_qubits_underlying

    qcbm_parameters = _as_1d_float(qcbm_data["theta_star"])
    exposure_parameters = _as_1d_float(exposure_data["theta_star"])
    default_parameters = _as_1d_float(default_data["theta_star"])
    discount_parameters = _as_1d_float(discount_data["theta_star"])
    exposure_metadata = _metadata_dict(exposure_data)

    qcbm_layout, _, qcbm_layout_meta = _select_layout_for_training(
        real_backend,
        topology=_npz_str(qcbm_data, "requested_topology", "qcbm_heavyhex6"),
        length=total_num_qubits,
        readout_quantile=0.92,
        local_2q_quantile=0.85,
        relax_if_needed=True,
    )
    default_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology=_npz_str(default_data, "requested_topology", "crca2"),
        length=3,
        readout_quantile=0.92,
        local_2q_quantile=0.85,
        relax_if_needed=True,
    )
    discount_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology=_npz_str(discount_data, "requested_topology", "crca2"),
        length=3,
        readout_quantile=0.92,
        local_2q_quantile=0.85,
        relax_if_needed=True,
    )
    exposure_layout, _, _ = _select_layout_for_training(
        real_backend,
        topology="heavy_hex_star",
        length=7,
        readout_quantile=0.92,
        local_2q_quantile=0.85,
        relax_if_needed=True,
    )

    qcbm = MLQcbmCircuit(
        n_qubits=total_num_qubits,
        n_layers=_npz_int(qcbm_data, "n_layers", 6),
        name="qcbm_state_prep_circuit",
        entangler="rzz",
        topology=qcbm_layout_meta["selected_topology"],
        backend=AerSimulator(method="statevector"),
        transpile_backend=real_backend,
        noise_model=None,
        simulation_method="statevector",
        optimization_level=TRANSPILATION_OPT_LEVEL,
        initial_layout=[int(q) for q in qcbm_layout],
        layout_method="trivial",
        routing_method="none",
        seed_transpiler=SEED_TRANSPILER,
    )
    crca_exposure = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=num_qubits_underlying,
        n_layers=int(exposure_metadata.get("n_layers", 2)),
        ansatz_type="heavy_hex_star",
        name="crca_positive_exposure_circuit",
    )
    crca_default = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(default_data, "n_layers", 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_default_probabilities_circuit",
    )
    crca_discount = CrcaCircuit(
        m_time=num_qubits_time,
        n_price=0,
        n_layers=_npz_int(discount_data, "n_layers", 1),
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name="crca_discount_factors_circuit",
    )

    _assert_param_size("QCBM", qcbm_parameters, qcbm.n_params)
    _assert_param_size("CRCA exposure", exposure_parameters, crca_exposure.n_params)
    _assert_param_size("CRCA default", default_parameters, crca_default.n_params)
    _assert_param_size("CRCA discount", discount_parameters, crca_discount.n_params)

    quantum_cva_circuit = QuantumCVACircuit(
        num_qubits_time=num_qubits_time,
        num_qubits_underlying=num_qubits_underlying,
        qcbm_circuit=qcbm,
        crca_circuit_exposure=crca_exposure,
        crca_circuit_default_prob=crca_default,
        crca_circuit_discount_factor=crca_discount,
        recovery_rate=float(classical["R_cva"]),
        C_v=float(classical["C_v"]),
        C_p=float(classical["C_p"]),
        C_q=float(classical["C_q"]),
        name="quantum_cva_circuit",
        backend="statevector",
    )

    cva_initial_layout, *_ = _build_improved_cva_initial_layout(
        real_backend=real_backend,
        qcbm_layout=qcbm_layout,
        positive_exposure_layout=exposure_layout,
        default_layout=default_layout,
        discount_layout=discount_layout,
    )

    return {
        "quantum_cva_circuit": quantum_cva_circuit,
        "qcbm_parameters": qcbm_parameters,
        "exposure_parameters": exposure_parameters,
        "default_parameters": default_parameters,
        "discount_parameters": discount_parameters,
        "cva_initial_layout": cva_initial_layout,
    }


def main() -> None:
    repo_root = _repo_root()
    inputs = _load_inputs(repo_root)
    output_dir = (
        inputs["data_root"] / "quantum" / "cva_pricing" / "algebraic_cva"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    service = QiskitRuntimeService(channel="ibm_cloud")
    real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)
    objects = _build_cva_objects(inputs, real_backend)

    quantum_cva_circuit = objects["quantum_cva_circuit"]
    qcbm_parameters = objects["qcbm_parameters"]
    exposure_parameters = objects["exposure_parameters"]
    default_parameters = objects["default_parameters"]
    discount_parameters = objects["discount_parameters"]

    qc_unbound, parameter_vectors = _build_unbound_cva_circuit(quantum_cva_circuit)
    _write_text(output_dir / "logical_unbound_circuit.txt", _draw_text(qc_unbound))
    _write_text(output_dir / "logical_unbound_circuit.qasm3", _qasm3(qc_unbound))
    _write_text(
        output_dir / "logical_unbound_circuit.tex",
        _draw_latex_source(qc_unbound),
    )
    _write_symbolic_gate_products(output_dir, qc_unbound)
    _write_symbolic_cva_amplitudes(output_dir, qc_unbound)
    _write_algebraic_forms(output_dir, quantum_cva_circuit)

    qc_bound = quantum_cva_circuit.build_cva_circuit(
        qcbm_params=qcbm_parameters,
        crca_exposure_params=exposure_parameters,
        crca_default_params=default_parameters,
        crca_discount_params=discount_parameters,
        measured=False,
    )
    logical_matrix = Operator(qc_bound).data
    np.save(output_dir / "logical_bound_matrix.npy", logical_matrix)
    np.savez_compressed(
        output_dir / "logical_bound_matrix.npz",
        matrix=logical_matrix,
    )
    _write_complex_matrix_txt(output_dir / "logical_bound_matrix.txt", logical_matrix)
    _write_complex_matrix_latex(output_dir / "logical_bound_matrix.tex", logical_matrix)

    qc_transpiled = _transpile_with_layout(
        qc_bound,
        backend=real_backend,
        initial_layout=objects["cva_initial_layout"],
        optimization_level=TRANSPILATION_OPT_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
        approximation_degree=APPROXIMATION_DEGREE,
    )
    _write_text(
        output_dir / "transpiled_bound_circuit.txt",
        _draw_text(qc_transpiled),
    )
    _write_text(
        output_dir / "transpiled_bound_circuit.qasm3",
        _qasm3(qc_transpiled),
    )
    _write_text(
        output_dir / "transpiled_bound_circuit.tex",
        _draw_latex_source(qc_transpiled),
    )

    transpiled_matrix_status = "skipped"
    active_qubits = _active_qubit_indices(qc_transpiled)
    if len(active_qubits) <= MAX_TRANSPILED_MATRIX_ACTIVE_QUBITS:
        reduced_transpiled = _reduced_active_circuit(qc_transpiled)
        transpiled_matrix = Operator(reduced_transpiled).data
        np.save(output_dir / "transpiled_bound_active_matrix.npy", transpiled_matrix)
        np.savez_compressed(
            output_dir / "transpiled_bound_active_matrix.npz",
            matrix=transpiled_matrix,
            active_physical_qubits=np.asarray(active_qubits, dtype=int),
        )
        _write_complex_matrix_txt(
            output_dir / "transpiled_bound_active_matrix.txt",
            transpiled_matrix,
        )
        transpiled_matrix_status = "written"

    state = Statevector.from_instruction(qc_bound).data
    objective_qubits = [6, 7, 8]
    mask_111 = _projection_mask(qc_bound.num_qubits, objective_qubits, "111")
    projected_state = np.zeros_like(state)
    projected_state[mask_111] = state[mask_111]
    p111_projector = float(np.vdot(projected_state, projected_state).real)

    p111_reference = quantum_cva_circuit.prob_111(
        qcbm_params=qcbm_parameters,
        crca_exposure_params=exposure_parameters,
        crca_default_params=default_parameters,
        crca_discount_params=discount_parameters,
    )
    scale = (
        2 ** quantum_cva_circuit.num_qubits_time
        * (1.0 - quantum_cva_circuit.recovery_rate)
        * quantum_cva_circuit.C_v
        * quantum_cva_circuit.C_q
        * quantum_cva_circuit.C_p
    )
    cva_projector = float(scale * p111_projector)
    cva_reference = quantum_cva_circuit.cva(
        qcbm_params=qcbm_parameters,
        exposure_params=exposure_parameters,
        default_prob_params=default_parameters,
        discount_factor_params=discount_parameters,
    )

    _write_projection_forms(output_dir, scale=scale)
    np.save(output_dir / "projected_state_111.npy", projected_state)
    np.savez_compressed(
        output_dir / "projected_state_111.npz",
        projected_state=projected_state,
        mask_111=mask_111,
        objective_qubits=np.asarray(objective_qubits, dtype=int),
        p111=p111_projector,
        cva=cva_projector,
    )

    summary = "\n".join(
        [
            "Projection numeric summary",
            "==========================",
            "",
            f"objective_qubits: {objective_qubits}",
            "good_state: 111",
            f"state_norm: {float(np.vdot(state, state).real):.16e}",
            f"projected_state_norm: {float(np.linalg.norm(projected_state)):.16e}",
            f"p111_projector: {p111_projector:.16e}",
            f"p111_reference: {float(p111_reference):.16e}",
            f"abs_p111_difference: {abs(p111_projector - p111_reference):.16e}",
            f"scale: {scale:.16e}",
            f"cva_projector: {cva_projector:.16e}",
            f"cva_reference: {float(cva_reference):.16e}",
            f"abs_cva_difference: {abs(cva_projector - cva_reference):.16e}",
            "",
        ]
    )
    _write_text(output_dir / "projection_111_numeric_summary.txt", summary)

    manifest = {
        "backend_name": BACKEND_NAME,
        "seed_transpiler": SEED_TRANSPILER,
        "transpilation_opt_level": TRANSPILATION_OPT_LEVEL,
        "approximation_degree": APPROXIMATION_DEGREE,
        "output_dir": str(output_dir),
        "logical_unbound": {
            "num_qubits": qc_unbound.num_qubits,
            "num_parameters": qc_unbound.num_parameters,
            "symbolic_gate_product_files": [
                "logical_unbound_symbolic_gates.txt",
                "logical_unbound_symbolic_gates.tex",
            ],
            "symbolic_objective_amplitude_files": [
                "cva_objective_symbolic_amplitudes.txt",
                "cva_objective_symbolic_amplitudes.tex",
            ],
            "parameter_vectors": {
                name: len(vector) for name, vector in parameter_vectors.items()
            },
            "depth": qc_unbound.depth(),
            "size": qc_unbound.size(),
        },
        "logical_bound": {
            "num_qubits": qc_bound.num_qubits,
            "num_parameters": qc_bound.num_parameters,
            "matrix_shape": list(logical_matrix.shape),
        },
        "transpiled_bound": {
            "num_qubits": qc_transpiled.num_qubits,
            "active_physical_qubits": active_qubits,
            "num_active_physical_qubits": len(active_qubits),
            "depth": qc_transpiled.depth(),
            "size": qc_transpiled.size(),
            "two_qubit_gates": _two_qubit_gate_count(qc_transpiled),
            "active_matrix_status": transpiled_matrix_status,
        },
        "projection": {
            "objective_qubits": objective_qubits,
            "good_state": "111",
            "p111_projector": p111_projector,
            "p111_reference": float(p111_reference),
            "cva_projector": cva_projector,
            "cva_reference": float(cva_reference),
            "scale": scale,
        },
    }
    _write_text(output_dir / "manifest.json", json.dumps(manifest, indent=2))

    if qc_unbound.num_parameters != 103:
        raise RuntimeError(
            f"Expected 103 unbound parameters; got {qc_unbound.num_parameters}."
        )
    if qc_bound.num_parameters != 0:
        raise RuntimeError("The logical bound CVA circuit still has parameters.")
    if not np.isclose(p111_projector, p111_reference, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Projector p111 does not match QuantumCVACircuit.prob_111.")
    if not np.isclose(cva_projector, cva_reference, rtol=1e-12, atol=1e-12):
        raise RuntimeError("Projector CVA does not match QuantumCVACircuit.cva.")

    print(f"Algebraic CVA artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
