from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
	NoiseModel,
	ReadoutError,
	depolarizing_error,
	thermal_relaxation_error,
)

from quantum_cva.algorithms.proposed_algorithms.cabiae import (
	CABIQAELatentTheta,
)
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE

try:
	from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
		StandaloneBAEHardware as StandaloneBAEModel,
	)

	BAE_KIND = "hardware"
except Exception:
	from quantum_cva.algorithms.third_party.standalone_bae import (
		StandaloneBAE as StandaloneBAEModel,
	)

	BAE_KIND = "legacy"


class _FakeCounts:
	def __init__(self, counts: dict[str, int]):
		self._counts = dict(counts)

	def get_counts(self) -> dict[str, int]:
		return dict(self._counts)


class _FakeData:
	def __init__(self, counts: dict[str, int]):
		self.c0 = _FakeCounts(counts)


class _FakePubResult:
	def __init__(self, counts: dict[str, int]):
		self.data = _FakeData(counts)


class _FakeSamplerJob:
	def __init__(self, pub_results: list[_FakePubResult]):
		self._pub_results = pub_results

	def result(self) -> list[_FakePubResult]:
		return self._pub_results


def build_large_state_preparation(objective_ry_offset: float = 0.0) -> QuantumCircuit:
    """
    Five-qubit hardware-friendly state preparation.

    Diseño:
      - solo usa puertas nativas o muy cercanas al backend: rx, rz, cz
      - solo usa conectividad de cadena 0-1-2-3-4
      - mantiene un estado no trivial, correlacionado y sensible al ruido
      - evita controlled rotations y acoplos largos que disparan el routing
    """
    qc = QuantumCircuit(5, name="A_large")

    # ------------------------------------------------------------------
    # Layer 1: local single-qubit dressing
    # ------------------------------------------------------------------
    rx_1 = [0.88, 0.54, 1.02, 0.71, 0.36]
    rz_1 = [0.31, -0.27, 0.18, 0.41, -0.22]

    for q, (ax, az) in enumerate(zip(rx_1, rz_1)):
        qc.rz(az, q)
        qc.rx(ax, q)

    # ------------------------------------------------------------------
    # Layer 2: nearest-neighbour entanglement on a chain
    # 0--1--2--3--4
    # ------------------------------------------------------------------
    for q0, q1 in ((0, 1), (1, 2), (2, 3), (3, 4)):
        qc.cz(q0, q1)

    # ------------------------------------------------------------------
    # Layer 3: second local dressing
    # ------------------------------------------------------------------
    rx_2 = [-0.24, 0.17, -0.28, 0.23, 0.39]
    rz_2 = [0.16, 0.09, -0.14, 0.07, 0.12]

    for q, (ax, az) in enumerate(zip(rx_2, rz_2)):
        qc.rz(az, q)
        qc.rx(ax, q)

    # ------------------------------------------------------------------
    # Layer 4: focused entanglement near the objective qubit
    # mantiene no trivialidad pero sin meter acoplos largos
    # ------------------------------------------------------------------
    for q0, q1 in ((1, 2), (2, 3), (3, 4)):
        qc.cz(q0, q1)

    # ------------------------------------------------------------------
    # Final local dressing on the "business end"
    # ------------------------------------------------------------------
    qc.rz(0.11, 2)
    qc.rx(-0.18, 2)

    qc.rz(-0.09, 3)
    qc.rx(0.14, 3)

    qc.rz(0.08, 4)
    qc.rx(0.26, 4)

    # Sweep of the final amplitude on the objective qubit
    if objective_ry_offset != 0.0:
        qc.rx(float(objective_ry_offset), 4)

    return qc

def build_noise_model(scale: float) -> NoiseModel:
    """
    Effective noise model tailored for a projected 2028-2029 superconducting QPU.
    
    Interpretation (Projected from ibm_basquecountry):
      - 1q noise halved.
      - 2q (CX) error drops below the critical 0.1% threshold.
      - Coherence times (T1/T2) more than doubled (hitting ~600us).
      - Readout fidelity dramatically improved to ~99.9%.
    """
    noise_model = NoiseModel()

    # ------------------------------------------------------------------
    # Parámetros proyectados a 2-3 años vista
    # ------------------------------------------------------------------

    # Error depolarizante 1q (
    p1 = min(1.88e-4 * scale, 5e-3)

    # Error efectivo CX:
    p2 = min(1.93e-3 * scale, 5e-2)

    # Errores de lectura asimétricos promedios de los 5 qubits
    p_10 = min(4.83e-3 * scale, 0.2)   # P(1|0) (Prob meas1 prep0)
    p_01 = min(4.30e-3 * scale, 0.2)   # P(0|1) (Prob meas0 prep1)

    # Tiempos de coherencia (ns) promedios de los 5 qubits
    t1 = 279_400.0 / scale
    t2 = 220_400.0 / scale

    # Duraciones de puertas (ns)
    t_id = 5.0
    t_sx = 5.0   # En base al "Single-qubit gate length" general del CSV
    t_x = 32.0

    # Duración CX: CZ original dura 68ns, asumiendo 2x sx gates de penalización
    t_cx = 70.0

    # ------------------------------------------------------------------
    # 1-qubit errors
    # ------------------------------------------------------------------
    err_id = depolarizing_error(p1, 1).compose(
        thermal_relaxation_error(t1, t2, t_id)
    )
    err_sx = depolarizing_error(p1, 1).compose(
        thermal_relaxation_error(t1, t2, t_sx)
    )
    err_x = depolarizing_error(p1, 1).compose(
        thermal_relaxation_error(t1, t2, t_x)
    )

    # ------------------------------------------------------------------
    # 2-qubit CX error
    # ------------------------------------------------------------------
    therm_cx = thermal_relaxation_error(t1, t2, t_cx).tensor(
        thermal_relaxation_error(t1, t2, t_cx)
    )
    err_cx = depolarizing_error(p2, 2).compose(therm_cx)

    # ------------------------------------------------------------------
    # 2-qubit CX error 
    # ------------------------------------------------------------------
    therm_cx = thermal_relaxation_error(t1, t2, t_cx).tensor(
        thermal_relaxation_error(t1, t2, t_cx)
    )
    err_cx = depolarizing_error(p2, 2).compose(therm_cx)

    # ------------------------------------------------------------------
    # Readout error
    # [[P(0|0), P(1|0)],
    #  [P(0|1), P(1|1)]]
    # ------------------------------------------------------------------
    ro = ReadoutError([
        [1.0 - p_10, p_10],
        [p_01, 1.0 - p_01],
    ])

    noise_model.add_all_qubit_quantum_error(err_id, ["id"])
    noise_model.add_all_qubit_quantum_error(err_sx, ["sx"])
    noise_model.add_all_qubit_quantum_error(err_x, ["x"])
    noise_model.add_all_qubit_quantum_error(err_cx, ["cx"])
    noise_model.add_all_qubit_readout_error(ro)

    return noise_model

# mild noise conditions
'''
def build_noise_model(scale: float) -> NoiseModel:
    """
    Effective noise model tailored for a projected 2028-2029 superconducting QPU.
    
    Interpretation (Projected from ibm_basquecountry):
      - 1q noise halved.
      - 2q (CX) error drops below the critical 0.1% threshold.
      - Coherence times (T1/T2) more than doubled (hitting ~600us).
      - Readout fidelity dramatically improved to ~99.9%.
    """
    noise_model = NoiseModel()

    # ------------------------------------------------------------------
    # Parámetros proyectados a 2-3 años vista
    # ------------------------------------------------------------------

    # Error depolarizante 1q (Reducido aprox. a la mitad)
    p1 = min(9.00e-5 * scale, 5e-3)

    # Error efectivo CX:
    # Cruza la barrera del 0.1%, fundamental para la utilidad cuántica profunda
    p2 = min(8.00e-4 * scale, 5e-2)

    # Errores de lectura (Mejora muy significativa, cerca del 99.9% de fidelidad)
    p_10 = min(1.20e-3 * scale, 0.2)   # P(1|0)
    p_01 = min(1.00e-3 * scale, 0.2)   # P(0|1)

    # Tiempos de coherencia (ns) (Escalan al rango de los 500-600 microsegundos)
    t1 = 600_000.0 / scale
    t2 = 550_000.0 / scale

    # Duraciones de puertas (ns)
    # Ligeramente más rápidas por optimización de pulsos, pero limitadas físicamente
    t_id = 5.0
    t_sx = 5.0   
    t_x = 20.0   # Antes 32.0
    t_cx = 50.0  # Antes 70.0 

    # ------------------------------------------------------------------
    # 1-qubit errors
    # ------------------------------------------------------------------
    err_id = depolarizing_error(p1, 1).compose(
        thermal_relaxation_error(t1, t2, t_id)
    )
    err_sx = depolarizing_error(p1, 1).compose(
        thermal_relaxation_error(t1, t2, t_sx)
    )
    err_x = depolarizing_error(p1, 1).compose(
        thermal_relaxation_error(t1, t2, t_x)
    )

    # ------------------------------------------------------------------
    # 2-qubit CX error 
    # ------------------------------------------------------------------
    therm_cx = thermal_relaxation_error(t1, t2, t_cx).tensor(
        thermal_relaxation_error(t1, t2, t_cx)
    )
    err_cx = depolarizing_error(p2, 2).compose(therm_cx)

    # ------------------------------------------------------------------
    # Readout error
    # [[P(0|0), P(1|0)],
    #  [P(0|1), P(1|1)]]
    # ------------------------------------------------------------------
    ro = ReadoutError([
        [1.0 - p_10, p_10],
        [p_01, 1.0 - p_01],
    ])

    noise_model.add_all_qubit_quantum_error(err_id, ["id"])
    noise_model.add_all_qubit_quantum_error(err_sx, ["sx"])
    noise_model.add_all_qubit_quantum_error(err_x, ["x"])
    noise_model.add_all_qubit_quantum_error(err_cx, ["cx"])
    noise_model.add_all_qubit_readout_error(ro)

    return noise_model
'''
def build_large_problem(objective_ry_offset: float = 0.0) -> tuple[EstimationProblem, float]:
	"""
	Build an amplitude-estimation problem on 5 qubits.
	The good state is defined by the objective qubit (qubit 4) being |1>.
	"""
	state_preparation = build_large_state_preparation(objective_ry_offset=objective_ry_offset)
	oracle = QuantumCircuit(5, name="oracle_good")
	oracle.z(4)

	grover_operator = GroverOperator(oracle, state_preparation=state_preparation)
	problem = EstimationProblem(
		state_preparation=state_preparation,
		grover_operator=grover_operator,
		objective_qubits=[4],
	)

	state = Statevector.from_instruction(state_preparation)
	probs = state.probabilities_dict(qargs=[4])
	a_true = float(probs.get("1", 0.0))
	return problem, a_true


def construct_measured_circuit(problem: EstimationProblem, k: int) -> QuantumCircuit:
	num_qubits = max(
		problem.state_preparation.num_qubits,
		problem.grover_operator.num_qubits,
	)
	circuit = QuantumCircuit(num_qubits, name=f"AE_k_{k}")
	circuit.compose(problem.state_preparation, inplace=True)
	if k > 0:
		circuit.compose(problem.grover_operator.power(k).decompose(), inplace=True)

	creg = ClassicalRegister(len(problem.objective_qubits), "c0")
	circuit.add_register(creg)
	circuit.barrier()
	circuit.measure(problem.objective_qubits, creg[:])
	return circuit


def ideal_good_probability(problem: EstimationProblem, k: int) -> float:
	circuit = QuantumCircuit(problem.state_preparation.num_qubits)
	circuit.compose(problem.state_preparation, inplace=True)
	if k > 0:
		circuit.compose(problem.grover_operator.power(k).decompose(), inplace=True)
	state = Statevector.from_instruction(circuit)
	probs = state.probabilities_dict(qargs=list(problem.objective_qubits))
	good_key = "1" * len(problem.objective_qubits)
	return float(probs.get(good_key, 0.0))


from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error

class AerCountSampler:
	"""
	Minimal SamplerV2-compatible wrapper around AerSimulator with a realistic noise model.
	"""

	def __init__(
		self,
		noise_model: NoiseModel | None = None,
		seed: int | None = None,
		method: str = "density_matrix",
	):
		self._rng = np.random.default_rng(seed)
		self._sim = AerSimulator(noise_model=noise_model, method=method)
		self._basis_gates = noise_model.basis_gates if noise_model is not None else None
		self._cache: dict[tuple[Any, ...], QuantumCircuit] = {}

	def _cache_key(self, circuit: QuantumCircuit) -> tuple[Any, ...]:
		ops = tuple(sorted((str(k), int(v)) for k, v in circuit.count_ops().items()))
		return (circuit.num_qubits, circuit.size(), circuit.depth(), ops)

	def _transpiled(self, circuit: QuantumCircuit) -> QuantumCircuit:
		key = self._cache_key(circuit)
		if key not in self._cache:
			self._cache[key] = transpile(
				circuit,
				self._sim,
				basis_gates=self._basis_gates,
				optimization_level=0,
				seed_transpiler=1234,
			)
		return self._cache[key]

	def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _FakeSamplerJob:
		pub_results: list[_FakePubResult] = []
		for circuit in circuits:
			transpiled_circuit = self._transpiled(circuit)
			seed_sim = int(self._rng.integers(1, 2**31 - 1))
			result = self._sim.run(
				transpiled_circuit,
				shots=shots,
				seed_simulator=seed_sim,
			).result()
			counts = result.get_counts()
			if not isinstance(counts, dict):
				counts = dict(counts)
			counts = {str(k): int(v) for k, v in counts.items()}
			pub_results.append(_FakePubResult(counts))
		return _FakeSamplerJob(pub_results)


@dataclass
class CalibrationResult:
	profile_name: str
	t_eff: float | None
	slope: float | None
	intercept: float | None
	used_points: int
	rows: list[dict[str, float]]


def estimate_prob_from_sampler(
	sampler: AerCountSampler,
	problem: EstimationProblem,
	k: int,
	shots: int,
) -> float:
	circuit = construct_measured_circuit(problem, k)
	ret = sampler.run([circuit], shots=shots).result()
	counts = ret[0].data.c0.get_counts()
	one = int(counts.get("1", 0))
	return float(one / shots)


def calibrate_effective_T(
	profile_name: str,
	sampler: AerCountSampler,
	problem: EstimationProblem,
	probe_ks: list[int],
	probe_shots: int,
) -> CalibrationResult:
	rows: list[dict[str, float]] = []
	fit_k_values: list[float] = []
	fit_log_c_values: list[float] = []

	for k in probe_ks:
		K = 2 * k + 1
		p_ideal = ideal_good_probability(problem, k)
		p_noisy = estimate_prob_from_sampler(sampler, problem, k, probe_shots)

		denom = p_ideal - 0.5
		numer = p_noisy - 0.5

		if abs(denom) < 2.0e-2:
			c_est = np.nan
		else:
			c_est = numer / denom

		if np.isfinite(c_est):
			c_est = float(np.clip(c_est, 1e-6, 1.0))
		else:
			c_est = np.nan

		rows.append(
			{
				"k": float(k),
				"K": float(K),
				"p_ideal": float(p_ideal),
				"p_noisy": float(p_noisy),
				"c_est": float(c_est) if np.isfinite(c_est) else np.nan,
			}
		)

		if np.isfinite(c_est) and c_est > 0.0 and c_est < 1.0 and abs(denom) >= 2.0e-2:
			fit_k_values.append(float(K))
			fit_log_c_values.append(float(np.log(c_est)))

	if len(fit_k_values) < 2:
		return CalibrationResult(
			profile_name=profile_name,
			t_eff=None,
			slope=None,
			intercept=None,
			used_points=len(fit_k_values),
			rows=rows,
		)

	slope, intercept = np.polyfit(np.asarray(fit_k_values), np.asarray(fit_log_c_values), deg=1)
	if slope >= 0:
		t_eff = None
	else:
		t_eff = float(-1.0 / slope)

	return CalibrationResult(
		profile_name=profile_name,
		t_eff=t_eff,
		slope=float(slope),
		intercept=float(intercept),
		used_points=len(fit_k_values),
		rows=rows,
	)


def build_solver(
	algorithm: str,
	epsilon_target: float,
	alpha: float,
	n_shots: int,
	seed: int,
	noisy_sampler: AerCountSampler,
	t_eff: float | None,
) -> tuple[Any, bool]:
	if algorithm == "bae":
		if BAE_KIND == "hardware":
			solver = StandaloneBAEModel(
				epsilon_target=epsilon_target,
				alpha=alpha,
				sampler=noisy_sampler,
				noise_model="ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast",
				T_known=None if t_eff is None or np.isinf(t_eff) else float(t_eff),
				cap_kappa=1000.0,
				max_shots_same_k=None,
				estimate_T=False,
			)
		else:
			solver = StandaloneBAEModel(
				epsilon_target=epsilon_target,
				alpha=alpha,
				sampler=noisy_sampler,
				noise_model="ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast",
				T_known=None if t_eff is None or np.isinf(t_eff) else float(t_eff),
				cap_kappa=3.0,
				estimate_T=False,
				T_range=None if t_eff is None or np.isinf(t_eff) else (0.5 * float(t_eff), 1.5 * float(t_eff)),
				TNs=0,
				wNs=60,
				Ns=n_shots,
			)
		return solver, True

	if algorithm == "biqae":
		solver = BIQAE(
			epsilon_target=epsilon_target,
			alpha=alpha,
			sampler=noisy_sampler,
			min_ratio=2,
			confint_method="beta",
			max_shots_same_k=None,
		)
		return solver, True

	if algorithm == "cabiqae_latentt":
		solver = CABIQAELatentTheta(
			epsilon_target=epsilon_target,
			alpha=alpha,
			sampler=noisy_sampler,
			min_ratio=2,
			confint_method="beta",
			noise_model="ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast",
			T_known=None if t_eff is None or np.isinf(t_eff) else float(t_eff),
			cap_kappa=1000.0,
			use_noise_cap=True,
			max_shots_same_k=None,
		)
		return solver, True

	raise ValueError(f"Unknown algorithm: {algorithm}")


def _extract_bae_k_sequence(
	result: Any,
	history: dict[str, Any],
	queries: np.ndarray,
	n_shots: int,
) -> np.ndarray:
	k_seq_candidates = []

	obj_seq = getattr(result, "circuit_depths", None)
	if obj_seq is not None:
		arr = np.asarray(obj_seq, dtype=float).ravel()
		if arr.size > 0:
			k_seq_candidates.append(arr)

	hist_seq = history.get("circuit_depths", None)
	if hist_seq is not None:
		arr = np.asarray(hist_seq, dtype=float).ravel()
		if arr.size > 0:
			k_seq_candidates.append(arr)

	if len(k_seq_candidates) > 0:
		return np.asarray(np.rint(k_seq_candidates[0]), dtype=int)

	if len(queries) == 0:
		return np.asarray([], dtype=int)

	dq = np.diff(np.r_[0.0, queries])
	inferred = dq / float(n_shots)
	return np.asarray(np.rint(inferred), dtype=int)


def extract_trace(
	algorithm: str,
	result: Any,
	n_shots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	if algorithm == "bae":
		history = getattr(result, "history", {}) or {}
		queries = np.asarray(history.get("queries", []), dtype=float)
		estimations = np.asarray(history.get("estimations", []), dtype=float)
		k_sequence = _extract_bae_k_sequence(result, history, queries, n_shots)

		usable = min(len(queries), len(estimations), len(k_sequence))
		if usable <= 0:
			return (
				np.asarray([], dtype=float),
				np.asarray([], dtype=float),
				np.asarray([], dtype=int),
			)

		return (
			queries[:usable].astype(float),
			estimations[:usable].astype(float),
			k_sequence[:usable].astype(int),
		)

	powers = np.asarray(getattr(result, "powers", []) or [], dtype=float)
	estimate_intervals = getattr(result, "estimate_intervals", []) or []

	usable = min(len(powers), max(0, len(estimate_intervals) - 1))
	if usable <= 0:
		return (
			np.asarray([], dtype=float),
			np.asarray([], dtype=float),
			np.asarray([], dtype=int),
		)

	interval_array = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
	estimations = np.mean(interval_array, axis=1)
	k_sequence = (2.0 * powers[:usable] + 1.0).astype(int)
	queries = np.cumsum(n_shots * k_sequence)

	return queries.astype(float), estimations.astype(float), k_sequence.astype(int)


def estimate_at_budget(
	queries: np.ndarray,
	estimations: np.ndarray,
	budget: int,
) -> float | None:
	if len(queries) == 0:
		return None
	idx = np.searchsorted(queries, budget, side="right") - 1
	if idx < 0:
		return None
	return float(estimations[idx])


def _budget_index_at_or_below(queries: np.ndarray, budget: int) -> int:
	if len(queries) == 0:
		return -1
	return int(np.searchsorted(queries, budget, side="right") - 1)


def kmax_at_budget(
	queries: np.ndarray,
	k_sequence: np.ndarray,
	budget: int,
) -> int | None:
	idx = _budget_index_at_or_below(queries, budget)
	if idx < 0:
		return None
	prefix = np.asarray(k_sequence[: idx + 1], dtype=float)
	if prefix.size == 0:
		return None
	return int(np.max(prefix))


def time_to_budget(
	queries: np.ndarray,
	total_runtime_seconds: float,
	budget: int,
) -> float | None:
	if len(queries) == 0:
		return None
	q_final = float(queries[-1])
	if not np.isfinite(q_final) or q_final <= 0.0:
		return None
	budget_clipped = float(np.clip(float(budget), 0.0, q_final))
	runtime = max(0.0, float(total_runtime_seconds))
	return float(runtime * (budget_clipped / q_final))


def save_csv(rows: list[dict[str, Any]], out_path: str) -> None:
	if not rows:
		return
	fieldnames = list(rows[0].keys())
	with open(out_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def bootstrap_mean_ci(
	values,
	n_boot=2000,
	alpha=0.05,
	rng=None,
):
	import numpy as np

	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]

	if len(values) == 0:
		return np.nan, np.nan, np.nan

	mean = float(np.mean(values))

	if len(values) == 1:
		return mean, mean, mean

	if rng is None:
		rng = np.random.default_rng(12345)

	n = len(values)
	boot_means = np.empty(n_boot, dtype=float)

	for i in range(n_boot):
		sample = rng.choice(values, size=n, replace=True)
		boot_means[i] = np.mean(sample)

	low = float(np.quantile(boot_means, alpha / 2))
	high = float(np.quantile(boot_means, 1 - alpha / 2))

	return mean, low, high


def bootstrap_median_ci(
	values,
	n_boot=2000,
	alpha=0.05,
	rng=None,
):
	import numpy as np

	values = np.asarray(values, dtype=float)
	values = values[np.isfinite(values)]

	if len(values) == 0:
		return np.nan, np.nan, np.nan

	median = float(np.median(values))

	if len(values) == 1:
		return median, median, median

	if rng is None:
		rng = np.random.default_rng(12345)

	n = len(values)
	boot_medians = np.empty(n_boot, dtype=float)

	for i in range(n_boot):
		sample = rng.choice(values, size=n, replace=True)
		boot_medians[i] = np.median(sample)

	low = float(np.quantile(boot_medians, alpha / 2))
	high = float(np.quantile(boot_medians, 1 - alpha / 2))

	return median, low, high


def aggregate_budget_rows(
	rows: list[dict[str, Any]],
	algorithms: Iterable[str],
	algorithm_labels: dict[str, str],
) -> list[dict[str, Any]]:
	summary_rows: list[dict[str, Any]] = []
	profile_names = sorted({str(r["profile"]) for r in rows})
	budgets = sorted({int(r["budget"]) for r in rows})

	for profile in profile_names:
		for budget in budgets:
			for alg in algorithms:
				alg_name = algorithm_labels[alg]
				subset = [
					r
					for r in rows
					if str(r["profile"]) == profile
					and int(r["budget"]) == budget
					and r["algorithm"] == alg_name
				]
				if not subset:
					continue

				abs_err = np.asarray([float(r["abs_error"]) for r in subset], dtype=float)
				nrmse = np.asarray([float(r["nrmse"]) for r in subset], dtype=float)
				time_budget = np.asarray(
					[float(r.get("time_to_budget_seconds", np.nan)) for r in subset],
					dtype=float,
				)
				kmax_budget = np.asarray(
					[float(r.get("k_max_budget", np.nan)) for r in subset],
					dtype=float,
				)
				mae_mean, mae_low, mae_high = bootstrap_mean_ci(abs_err)
				nrmse_mean, nrmse_low, nrmse_high = bootstrap_mean_ci(nrmse)
				nrmse_median, nrmse_median_low, nrmse_median_high = bootstrap_median_ci(nrmse)
				time_mean, time_low, time_high = bootstrap_mean_ci(time_budget)
				kmax_mean, kmax_low, kmax_high = bootstrap_mean_ci(kmax_budget)

				summary_rows.append(
					{
						"profile": profile,
						"budget": int(budget),
						"algorithm": alg_name,
						"n_points": int(len(subset)),
						"n_runs": int(len(subset)),
						"abs_error_median": float(np.nanmedian(abs_err)),
						"mae_mean": mae_mean,
						"mae_ci_low": mae_low,
						"mae_ci_high": mae_high,
						"nrmse_median": nrmse_median,
						"nrmse_median_ci_low": nrmse_median_low,
						"nrmse_median_ci_high": nrmse_median_high,
						"nrmse_mean": nrmse_mean,
						"nrmse_ci_low": nrmse_low,
						"nrmse_ci_high": nrmse_high,
						"time_to_budget_seconds_median": float(np.nanmedian(time_budget)),
						"time_mean": time_mean,
						"time_ci_low": time_low,
						"time_ci_high": time_high,
						"k_max_budget_median": float(np.nanmedian(kmax_budget)),
						"kmax_mean": kmax_mean,
						"kmax_ci_low": kmax_low,
						"kmax_ci_high": kmax_high,
					}
				)
	return summary_rows


def print_budget_summary(
	summary_rows: list[dict[str, Any]],
	profile_names: Iterable[str],
) -> None:
	print("\n=== Budget-aligned summary on larger noisy circuit ===")
	for profile_name in profile_names:
		print(f"profile={profile_name}")
		prof_rows = [r for r in summary_rows if str(r["profile"]) == profile_name]
		budgets = sorted({int(r["budget"]) for r in prof_rows})
		for budget in budgets:
			b_rows = [r for r in prof_rows if int(r["budget"]) == budget]
			b_rows.sort(key=lambda x: float(x["nrmse_median"]))
			print(f"  budget={budget}")
			for row in b_rows:
				time_med = float(row.get("time_to_budget_seconds_median", np.nan))
				kmax_med = float(row.get("k_max_budget_median", np.nan))
				time_part = f" | t_budget_med={time_med:.3f}s" if np.isfinite(time_med) else ""
				k_part = f" | Kmax_budget_med={int(np.rint(kmax_med))}" if np.isfinite(kmax_med) else ""
				print(
					"    "
					+ f"{row['algorithm']:16s} "
					+ f"nRMSE_med={row['nrmse_median']:.3e} | "
					+ f"AbsErr_med={row['abs_error_median']:.3e} | "
					+ f"n={row['n_points']}"
					+ time_part
					+ k_part
				)


def print_final_summary(
	final_rows: list[dict[str, Any]],
	profile_names: Iterable[str],
	algorithms: Iterable[str],
	algorithm_labels: dict[str, str],
) -> None:
	print("\n=== Final-stop auxiliary summary ===")
	for profile_name in profile_names:
		print(f"profile={profile_name}")
		for alg in algorithms:
			alg_name = algorithm_labels[alg]
			subset = [
				r
				for r in final_rows
				if str(r["profile"]) == profile_name and r["algorithm"] == alg_name
			]
			if not subset:
				continue

			q = np.asarray([float(r["final_queries"]) for r in subset], dtype=float)
			nr = np.asarray([float(r["final_nrmse"]) for r in subset], dtype=float)
			cov = np.asarray([float(r["coverage"]) for r in subset], dtype=float)
			kmax = np.asarray([float(r["k_max"]) for r in subset], dtype=float)

			runtime_part = ""
			if any("runtime_seconds" in r for r in subset):
				runtime = np.asarray([float(r.get("runtime_seconds", np.nan)) for r in subset], dtype=float)
				runtime_part = f"time_med={np.nanmedian(runtime):.3f}s | "

			print(
				"  "
				+ f"{alg_name:16s} "
				+ f"Q_med={int(np.nanmedian(q)):6d} | "
				+ f"nRMSE_med={np.nanmedian(nr):.3e} | "
				+ f"coverage={np.nanmean(cov):.2f} | "
				+ runtime_part
				+ f"K_med={np.nanmedian(kmax):.0f}"
			)


def plot_budget_panels(
	summary_rows: list[dict[str, Any]],
	output_path: str,
	profile_names: Iterable[str],
	algorithms: Iterable[str],
	algorithm_labels: dict[str, str],
	algorithm_styles: dict[str, dict[str, str]],
	budgets_reference: np.ndarray,
) -> None:
	profile_names = list(profile_names)
	fig, axes = plt.subplots(1, len(profile_names), figsize=(5.4 * len(profile_names), 4.8), squeeze=False)

	for j, profile in enumerate(profile_names):
		ax = axes[0, j]
		prof_rows = [r for r in summary_rows if str(r["profile"]) == profile]

		for alg in algorithms:
			alg_name = algorithm_labels[alg]
			subset = [r for r in prof_rows if r["algorithm"] == alg_name]
			if not subset:
				continue

			budgets = np.asarray([int(r["budget"]) for r in subset], dtype=float)
			nrmse = np.asarray([float(r["nrmse_median"]) for r in subset], dtype=float)
			kmax = np.asarray([float(r.get("k_max_budget_median", np.nan)) for r in subset], dtype=float)
			nrmse_median_ci_low = np.asarray(
				[float(r.get("nrmse_median_ci_low", np.nan)) for r in subset],
				dtype=float,
			)
			nrmse_median_ci_high = np.asarray(
				[float(r.get("nrmse_median_ci_high", np.nan)) for r in subset],
				dtype=float,
			)
			order = np.argsort(budgets)
			budgets = budgets[order]
			nrmse = nrmse[order]
			kmax = kmax[order]
			nrmse_median_ci_low = nrmse_median_ci_low[order]
			nrmse_median_ci_high = nrmse_median_ci_high[order]

			style = algorithm_styles[alg]
			valid_band = (
				np.isfinite(nrmse_median_ci_low)
				& np.isfinite(nrmse_median_ci_high)
				& (nrmse_median_ci_low > 0.0)
				& (nrmse_median_ci_high > 0.0)
			)
			if np.any(valid_band):
				ax.fill_between(
					budgets[valid_band],
					nrmse_median_ci_low[valid_band],
					nrmse_median_ci_high[valid_band],
					color=style["color"],
					alpha=0.14,
					linewidth=0.0,
					zorder=1,
				)
			ax.loglog(
				budgets,
				nrmse,
				color=style["color"],
				marker=style["marker"],
				linewidth=1.8,
				markersize=4,
				label=alg_name,
			)

			for x_val, y_val, k_val in zip(budgets, nrmse, kmax):
				if not np.isfinite(k_val):
					continue
				ax.annotate(
					f"K={int(np.rint(k_val))}",
					xy=(x_val, y_val),
					xytext=(4, 4),
					textcoords="offset points",
					fontsize=7,
					color=style["color"],
					alpha=0.9,
				)

		ax.loglog(
			budgets_reference,
			1.0 / np.sqrt(budgets_reference),
			"--",
			color="gray",
			alpha=0.6,
			label=r"$\mathcal{O}(1/\sqrt{N_q})$",
		)
		ax.loglog(
			budgets_reference,
			3.0 / budgets_reference,
			"-.",
			color="black",
			alpha=0.5,
			label=r"$\mathcal{O}(1/N_q)$",
		)

		ax.set_xlabel("Common query budget")
		ax.set_ylabel("Median normalized RMSE")
		ax.grid(True, which="both", alpha=0.2)

	handles, labels = axes[0, 0].get_legend_handles_labels()
	legend_ncol = min(len(labels), 4) if labels else 1
	fig.legend(
		handles,
		labels,
		loc="upper center",
		bbox_to_anchor=(0.5, 0.99),
		ncol=legend_ncol,
		frameon=False,
		columnspacing=1.6,
		handlelength=2.2,
	)
	fig.tight_layout(rect=(0, 0, 1, 0.86))
	fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.2)
	plt.close(fig)