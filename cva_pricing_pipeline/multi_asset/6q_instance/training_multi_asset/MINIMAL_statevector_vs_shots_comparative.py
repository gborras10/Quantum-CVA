from __future__ import annotations

import pathlib
from datetime import datetime, timezone

import numpy as np
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
	CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
	MLQcbmCircuit,
)

BACKEND_NAME = "ibm_basquecountry"
NOISE_SNAPSHOT_ISO_UTC = "2026-04-07T12:10:00+00:00"
SHOTS = 60000
SEED = 1234
EPS = 1e-9


def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
	dt = datetime.fromisoformat(snapshot_iso_utc)
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc)


def _as_1d_float(array_like: np.ndarray) -> np.ndarray:
	return np.asarray(array_like, dtype=float).ravel()


def _npz_int(npz_data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
	if key not in npz_data:
		return int(default)
	return int(np.asarray(npz_data[key]).item())


def _npz_str(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
	if key not in npz_data:
		return str(default)
	return str(np.asarray(npz_data[key]).item())


def _npz_float(npz_data: np.lib.npyio.NpzFile, key: str, default: float) -> float:
	if key not in npz_data:
		return float(default)
	return float(np.asarray(npz_data[key]).item())


def _metadata_dict(npz_data: np.lib.npyio.NpzFile) -> dict:
	if "metadata" not in npz_data:
		return {}
	maybe_dict = npz_data["metadata"]
	if hasattr(maybe_dict, "item"):
		maybe_dict = maybe_dict.item()
	return maybe_dict if isinstance(maybe_dict, dict) else {}


def _npz_optional_int_list(npz_data: np.lib.npyio.NpzFile, key: str) -> list[int] | None:
	if key not in npz_data:
		return None
	arr = np.asarray(npz_data[key]).ravel()
	if arr.size == 0:
		return None
	return [int(x) for x in arr]


def _normalize_for_kl(values: np.ndarray, eps: float = EPS) -> np.ndarray:
	arr = np.asarray(values, dtype=float).ravel()
	if arr.size == 0:
		raise ValueError("Empty vector cannot be normalized for KL divergence.")
	arr = np.maximum(arr, eps)
	s = float(arr.sum())
	if not np.isfinite(s) or s <= 0.0:
		raise ValueError("Vector has non-finite or non-positive sum.")
	return arr / s


def _kl_divergence(target: np.ndarray, predicted: np.ndarray, eps: float = EPS) -> float:
	p = _normalize_for_kl(target, eps=eps)
	q = _normalize_for_kl(predicted, eps=eps)
	if p.shape != q.shape:
		raise ValueError(f"Shape mismatch for KL: {p.shape} vs {q.shape}.")
	return float(np.sum(p * np.log(p / q)))


def _l2_mse(target: np.ndarray, predicted: np.ndarray) -> float:
	t = np.asarray(target, dtype=float).ravel()
	p = np.asarray(predicted, dtype=float).ravel()
	if t.shape != p.shape:
		raise ValueError(f"Shape mismatch for L2/MSE: {t.shape} vs {p.shape}.")
	return float(np.mean((p - t) ** 2))


def _set_crca_backend(crca: CrcaCircuit, backend) -> None:
	crca._backend = backend
	crca._tqc_eval_meas = transpile(crca._qc_eval_meas, backend)
	crca._tqc_eval_meas_param_set = set(crca._tqc_eval_meas.parameters)
	crca._n_clbits = len(crca._tqc_eval_meas.clbits)
	crca._ctrl_clbit_indices, crca._a_clbit_index = crca._extract_clbit_indices(
		crca._tqc_eval_meas
	)


def _set_crca_backend_training_like(
	crca: CrcaCircuit,
	backend,
	*,
	transpile_opt_level: int,
	seed_transpiler: int,
) -> None:
	local_layout = list(range(crca.qc.num_qubits))
	pm = generate_preset_pass_manager(
		backend=backend,
		optimization_level=int(transpile_opt_level),
		initial_layout=local_layout,
		seed_transpiler=int(seed_transpiler),
		approximation_degree=1.0,
	)
	crca._backend = backend
	crca._tqc_eval_meas = pm.run(crca._qc_eval_meas)
	crca._tqc_eval_meas_param_set = set(crca._tqc_eval_meas.parameters)
	crca._n_clbits = len(crca._tqc_eval_meas.clbits)
	crca._ctrl_clbit_indices, crca._a_clbit_index = crca._extract_clbit_indices(
		crca._tqc_eval_meas
	)


def _inject_missing_frequencies(snapshot_props, fallback_backend, default_frequency_ghz: float = 5.0):
	props_dict = snapshot_props.to_dict()
	fallback_qprops = getattr(getattr(fallback_backend, "target", None), "qubit_properties", None)

	for qi, q_props in enumerate(props_dict.get("qubits", [])):
		if any(str(p.get("name", "")).lower() == "frequency" for p in q_props):
			continue

		freq_ghz = None
		if fallback_qprops is not None and qi < len(fallback_qprops):
			maybe_freq = getattr(fallback_qprops[qi], "frequency", None)
			if maybe_freq is not None and np.isfinite(maybe_freq):
				maybe_freq = float(maybe_freq)
				freq_ghz = maybe_freq / 1e9 if maybe_freq > 1e6 else maybe_freq

		q_props.append(
			{
				"date": props_dict.get("last_update_date"),
				"name": "frequency",
				"unit": "GHz",
				"value": float(freq_ghz if freq_ghz is not None else default_frequency_ghz),
			}
		)

	return type(snapshot_props).from_dict(props_dict)


def _build_local_coupling_map(full_edges, chosen_layout: list[int]) -> list[list[int]]:
	phys_to_local = {int(q): i for i, q in enumerate(chosen_layout)}
	local_edges: list[list[int]] = []
	for a, b in full_edges:
		if int(a) in phys_to_local and int(b) in phys_to_local:
			local_edges.append([phys_to_local[int(a)], phys_to_local[int(b)]])
	return local_edges


def _subset_backend_properties(snapshot_props, chosen_layout: list[int]):
	props_dict = snapshot_props.to_dict()
	phys_to_local = {int(q): i for i, q in enumerate(chosen_layout)}
	subset_dict = {k: v for k, v in props_dict.items()}
	subset_dict["qubits"] = [props_dict["qubits"][int(q)] for q in chosen_layout]

	subset_gates = []
	for gate in props_dict.get("gates", []):
		qids = [int(q) for q in gate.get("qubits", [])]
		if all(q in phys_to_local for q in qids):
			new_gate = dict(gate)
			new_gate["qubits"] = [phys_to_local[q] for q in qids]
			subset_gates.append(new_gate)
	subset_dict["gates"] = subset_gates
	subset_dict["general_qlists"] = []
	subset_dict["backend_name"] = f"{props_dict.get('backend_name', BACKEND_NAME)}_subset_{len(chosen_layout)}q"
	return type(snapshot_props).from_dict(subset_dict)


def _build_positive_exposure_training_like_backend(
	*,
	real_backend,
	backend_props,
	chosen_layout: list[int],
	thermal_relaxation_requested: bool,
	seed_simulator: int,
):
	full_coupling = getattr(real_backend, "coupling_map", None)
	if full_coupling is None:
		full_coupling = real_backend.configuration().coupling_map
	full_edges = list(full_coupling.get_edges()) if hasattr(full_coupling, "get_edges") else list(full_coupling)
	local_coupling_map = _build_local_coupling_map(full_edges, chosen_layout)
	if not local_coupling_map:
		raise RuntimeError("Local coupling map vacio para positive exposure.")

	props_for_noise = backend_props
	if thermal_relaxation_requested:
		props_for_noise = _inject_missing_frequencies(backend_props, real_backend)

	subset_props = _subset_backend_properties(props_for_noise, chosen_layout)
	try:
		noise_model = NoiseModel.from_backend_properties(
			subset_props,
			thermal_relaxation=thermal_relaxation_requested,
		)
	except Exception:
		noise_model = NoiseModel.from_backend_properties(
			subset_props,
			thermal_relaxation=False,
		)

	return AerSimulator(
		method="density_matrix",
		noise_model=noise_model,
		coupling_map=local_coupling_map,
		seed_simulator=int(seed_simulator),
	)


def main() -> None:
	repo_root = next(
		parent
		for parent in pathlib.Path(__file__).resolve().parents
		if (parent / "pyproject.toml").exists()
	)

	classical_data = np.load(
		repo_root / "data" / "multi_asset" / "6q_instance" / "benchmark" / "three_asset_instance.npz",
		allow_pickle=True,
	)

	qcbm_data = np.load(
		repo_root
		/ "data"
		/ "multi_asset"
		/ "6q_instance"
		/ "quantum"
		/ "training"
		/ "qcbm"
		/ "shots"
		/ "training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz",
		allow_pickle=True,
	)
	positive_exposure_data = np.load(
		repo_root
		/ "data"
		/ "multi_asset"
		/ "6q_instance"
		/ "quantum"
		/ "training"
		/ "crca"
		/ "positive_exposure"
		/ "training_heavy_hex_star_shots_backend_noise_snapshot.npz",
		allow_pickle=True,
	)
	discount_factors_data = np.load(
		repo_root
		/ "data"
		/ "multi_asset"
		/ "6q_instance"
		/ "quantum"
		/ "training"
		/ "crca"
		/ "discount_factors"
		/ "training_crca2.npz",
		allow_pickle=True,
	)
	default_probabilities_data = np.load(
		repo_root
		/ "data"
		/ "multi_asset"
		/ "6q_instance"
		/ "quantum"
		/ "training"
		/ "crca"
		/ "default_probabilities"
		/ "training_crca2.npz",
		allow_pickle=True,
	)

	qcbm_theta = _as_1d_float(qcbm_data["theta_star"])
	positive_exposure_theta = _as_1d_float(positive_exposure_data["theta_star"])
	discount_factors_theta = _as_1d_float(discount_factors_data["theta_star"])
	default_probabilities_theta = _as_1d_float(default_probabilities_data["theta_star"])

	qcbm_target = _as_1d_float(qcbm_data["p_target"])
	positive_exposure_target = _as_1d_float(classical_data["v_joint_t"]) / float(classical_data["C_v"])
	discount_factors_target = _as_1d_float(classical_data["p_t"]) / float(classical_data["C_p"])
	default_probabilities_target = _as_1d_float(classical_data["q_t"]) / float(classical_data["C_q"])

	qcbm_n_layers = _npz_int(qcbm_data, "n_layers", 6)
	qcbm_topology = _npz_str(
		qcbm_data,
		"effective_topology",
		_npz_str(qcbm_data, "requested_topology", "qcbm_heavyhex6"),
	)
	qcbm_layout = _npz_optional_int_list(qcbm_data, "chosen_layout")
	qcbm_transpile_opt_level = _npz_int(qcbm_data, "transpile_optimization_level", 1)
	qcbm_seed_transpiler = _npz_int(qcbm_data, "seed_transpiler", SEED)
	qcbm_used_noise_fallback = None
	if "used_noise_fallback" in qcbm_data:
		qcbm_used_noise_fallback = bool(np.asarray(qcbm_data["used_noise_fallback"]).item())

	positive_exposure_meta = _metadata_dict(positive_exposure_data)
	default_probabilities_meta = _metadata_dict(default_probabilities_data)
	discount_factors_meta = _metadata_dict(discount_factors_data)
	positive_exposure_effective_topology = _npz_str(
		positive_exposure_data,
		"effective_topology",
		"heavy_hex_star",
	)
	positive_exposure_layout = _npz_optional_int_list(positive_exposure_data, "chosen_layout")
	if positive_exposure_layout is None:
		raise RuntimeError(
			"No se encontro chosen_layout en training de positive exposure; "
			"no puedo reproducir la comparacion justa con snapshot."
		)
	positive_exposure_transpile_opt_level = _npz_int(
		positive_exposure_data,
		"transpile_optimization_level",
		3,
	)
	positive_exposure_seed_transpiler = _npz_int(
		positive_exposure_data,
		"seed_transpiler",
		SEED,
	)
	positive_exposure_thermal_requested = bool(
		np.asarray(
			positive_exposure_data["thermal_relaxation_requested"]
			if "thermal_relaxation_requested" in positive_exposure_data
			else False
		).item()
	)

	positive_exposure_layers = int(
		positive_exposure_meta.get(
			"n_layers",
			_npz_int(positive_exposure_data, "n_layers", 2),
		)
	)
	default_probabilities_layers = int(
		default_probabilities_meta.get(
			"n_layers",
			_npz_int(default_probabilities_data, "n_layers", 1),
		)
	)
	discount_factors_layers = int(
		discount_factors_meta.get(
			"n_layers",
			_npz_int(discount_factors_data, "n_layers", 1),
		)
	)

	num_qubits_time = 2
	num_qubits_underlying = 4
	total_num_qubits = num_qubits_time + num_qubits_underlying

	def build_qcbm(
		*,
		backend,
		simulation_method: str,
		noise_model=None,
		transpile_backend=None,
		initial_layout: list[int] | None = None,
	) -> MLQcbmCircuit:
		qcbm = MLQcbmCircuit(
			n_qubits=total_num_qubits,
			n_layers=qcbm_n_layers,
			name="qcbm_state_prep_circuit",
			entangler="rzz",
			topology=qcbm_topology,
			backend=backend,
			transpile_backend=transpile_backend,
			noise_model=noise_model,
			simulation_method=simulation_method,
			optimization_level=qcbm_transpile_opt_level,
			initial_layout=initial_layout,
			seed_transpiler=qcbm_seed_transpiler,
		)
		if qcbm_theta.size != qcbm.n_params:
			raise ValueError(
				f"QCBM theta_star has size {qcbm_theta.size}, but circuit expects {qcbm.n_params}."
			)
		return qcbm

	def build_crca_triplet() -> tuple[CrcaCircuit, CrcaCircuit, CrcaCircuit]:
		crca_positive_exposure = CrcaCircuit(
			m_time=num_qubits_time,
			n_price=num_qubits_underlying,
			n_layers=positive_exposure_layers,
			ansatz_type=positive_exposure_effective_topology,
			name="crca_positive_exposure",
		)
		crca_discount_factors = CrcaCircuit(
			m_time=num_qubits_time,
			n_price=0,
			n_layers=discount_factors_layers,
			ansatz_type="native_tree",
			native_1q_order=("rx", "rz"),
			name="crca_discount_factors",
		)
		crca_default_probabilities = CrcaCircuit(
			m_time=num_qubits_time,
			n_price=0,
			n_layers=default_probabilities_layers,
			ansatz_type="native_tree",
			native_1q_order=("rx", "rz"),
			name="crca_default_probabilities",
		)

		if positive_exposure_theta.size != crca_positive_exposure.n_params:
			raise ValueError(
				"positive_exposure theta_star size mismatch: "
				f"{positive_exposure_theta.size} vs {crca_positive_exposure.n_params}"
			)
		if discount_factors_theta.size != crca_discount_factors.n_params:
			raise ValueError(
				"discount_factors theta_star size mismatch: "
				f"{discount_factors_theta.size} vs {crca_discount_factors.n_params}"
			)
		if default_probabilities_theta.size != crca_default_probabilities.n_params:
			raise ValueError(
				"default_probabilities theta_star size mismatch: "
				f"{default_probabilities_theta.size} vs {crca_default_probabilities.n_params}"
			)

		return crca_positive_exposure, crca_discount_factors, crca_default_probabilities

	def evaluate_metrics_bundle(
		*,
		qcbm: MLQcbmCircuit,
		crca_positive_exposure: CrcaCircuit,
		crca_discount_factors: CrcaCircuit,
		crca_default_probabilities: CrcaCircuit,
		shots: int | None,
	) -> dict[str, dict[str, float]]:
		qcbm_pred = qcbm.probabilities(qcbm_theta, shots=shots, seed=SEED)
		positive_exposure_pred = crca_positive_exposure.function_values(
			positive_exposure_theta,
			shots=shots,
			seed=SEED,
		)
		discount_factors_pred = crca_discount_factors.function_values(
			discount_factors_theta,
			shots=shots,
			seed=SEED,
		)
		default_probabilities_pred = crca_default_probabilities.function_values(
			default_probabilities_theta,
			shots=shots,
			seed=SEED,
		)

		return {
			"kl": {
				"qcbm": _kl_divergence(qcbm_target, qcbm_pred),
				"positive_exposure": _kl_divergence(positive_exposure_target, positive_exposure_pred),
				"discount_factors": _kl_divergence(discount_factors_target, discount_factors_pred),
				"default_probabilities": _kl_divergence(default_probabilities_target, default_probabilities_pred),
			},
			"l2": {
				"qcbm": _l2_mse(qcbm_target, qcbm_pred),
				"positive_exposure": _l2_mse(positive_exposure_target, positive_exposure_pred),
				"discount_factors": _l2_mse(discount_factors_target, discount_factors_pred),
				"default_probabilities": _l2_mse(default_probabilities_target, default_probabilities_pred),
			},
		}

	qcbm_sv = build_qcbm(backend=None, simulation_method="statevector", noise_model=None)
	crca_pos_sv, crca_disc_sv, crca_def_sv = build_crca_triplet()
	metrics_statevector = evaluate_metrics_bundle(
		qcbm=qcbm_sv,
		crca_positive_exposure=crca_pos_sv,
		crca_discount_factors=crca_disc_sv,
		crca_default_probabilities=crca_def_sv,
		shots=None,
	)

	ideal_shots_backend = AerSimulator(method="automatic", seed_simulator=SEED)
	qcbm_shots = build_qcbm(
		backend=ideal_shots_backend,
		simulation_method="automatic",
		noise_model=None,
	)
	crca_pos_shots, crca_disc_shots, crca_def_shots = build_crca_triplet()
	_set_crca_backend(crca_pos_shots, ideal_shots_backend)
	_set_crca_backend(crca_disc_shots, ideal_shots_backend)
	_set_crca_backend(crca_def_shots, ideal_shots_backend)
	metrics_shots = evaluate_metrics_bundle(
		qcbm=qcbm_shots,
		crca_positive_exposure=crca_pos_shots,
		crca_discount_factors=crca_disc_shots,
		crca_default_probabilities=crca_def_shots,
		shots=SHOTS,
	)

	service = QiskitRuntimeService(channel="ibm_cloud")
	real_backend = service.backend(BACKEND_NAME, use_fractional_gates=True)
	snapshot_dt_utc = _parse_snapshot_datetime(NOISE_SNAPSHOT_ISO_UTC)
	backend_props_snapshot = real_backend.properties(datetime=snapshot_dt_utc)
	if backend_props_snapshot is None:
		raise RuntimeError(
			"No se pudieron recuperar propiedades del backend para el snapshot "
			f"{NOISE_SNAPSHOT_ISO_UTC}."
		)

	backend_props_current = real_backend.properties()
	if backend_props_current is None:
		raise RuntimeError("No se pudieron recuperar propiedades actuales del backend.")

	try:
		noise_model_current = NoiseModel.from_backend_properties(
			backend_props_current,
			thermal_relaxation=False,
		)
		noise_model_snapshot = NoiseModel.from_backend_properties(
			backend_props_snapshot,
			thermal_relaxation=False,
		)
	except AttributeError:
		print(
			"[WARNING] No disponible NoiseModel.from_backend_properties(). "
			"Se usa from_backend(real_backend, thermal_relaxation=False) para ambas comparaciones."
		)
		noise_model_current = NoiseModel.from_backend(real_backend, thermal_relaxation=False)
		noise_model_snapshot = NoiseModel.from_backend(real_backend, thermal_relaxation=False)

	coupling_map = getattr(real_backend, "coupling_map", None)
	if coupling_map is None:
		try:
			coupling_map = real_backend.configuration().coupling_map
		except Exception:
			coupling_map = None

	noisy_backend = AerSimulator(
		method="density_matrix",
		noise_model=noise_model_current,
		coupling_map=coupling_map,
		seed_simulator=SEED,
	)

	qcbm_noisy = build_qcbm(
		backend=noisy_backend,
		simulation_method="density_matrix",
		noise_model=noise_model_current,
		transpile_backend=real_backend,
		initial_layout=qcbm_layout,
	)
	crca_pos_noisy, crca_disc_noisy, crca_def_noisy = build_crca_triplet()
	_set_crca_backend(crca_pos_noisy, noisy_backend)
	_set_crca_backend(crca_disc_noisy, noisy_backend)
	_set_crca_backend(crca_def_noisy, noisy_backend)
	metrics_shots_noise = evaluate_metrics_bundle(
		qcbm=qcbm_noisy,
		crca_positive_exposure=crca_pos_noisy,
		crca_discount_factors=crca_disc_noisy,
		crca_default_probabilities=crca_def_noisy,
		shots=SHOTS,
	)

	# QCBM KL with snapshot-based backend noise model
	noisy_backend_snapshot = AerSimulator(
		method="density_matrix",
		noise_model=noise_model_snapshot,
		coupling_map=coupling_map,
		seed_simulator=SEED,
	)

	qcbm_noisy_snapshot = build_qcbm(
		backend=noisy_backend_snapshot,
		simulation_method="density_matrix",
		noise_model=noise_model_snapshot,
		transpile_backend=real_backend,
		initial_layout=qcbm_layout,
	)
	qcbm_prob_snapshot = qcbm_noisy_snapshot.probabilities(qcbm_theta, shots=SHOTS, seed=SEED)
	qcbm_kl_noise_snapshot = _kl_divergence(qcbm_target, qcbm_prob_snapshot)
	qcbm_kl_noise_current = metrics_shots_noise["kl"]["qcbm"]
	qcbm_l2_noise_snapshot = _l2_mse(qcbm_target, qcbm_prob_snapshot)
	qcbm_l2_noise_current = metrics_shots_noise["l2"]["qcbm"]

	# Positive exposure KL with current vs snapshot using the same backend path as training.
	positive_exposure_backend_current = _build_positive_exposure_training_like_backend(
		real_backend=real_backend,
		backend_props=backend_props_current,
		chosen_layout=positive_exposure_layout,
		thermal_relaxation_requested=positive_exposure_thermal_requested,
		seed_simulator=SEED,
	)
	positive_exposure_backend_snapshot = _build_positive_exposure_training_like_backend(
		real_backend=real_backend,
		backend_props=backend_props_snapshot,
		chosen_layout=positive_exposure_layout,
		thermal_relaxation_requested=positive_exposure_thermal_requested,
		seed_simulator=SEED,
	)

	crca_pos_current_training_like = CrcaCircuit(
		m_time=num_qubits_time,
		n_price=num_qubits_underlying,
		n_layers=positive_exposure_layers,
		ansatz_type=positive_exposure_effective_topology,
		name="crca_positive_exposure_current_training_like",
	)
	_set_crca_backend_training_like(
		crca_pos_current_training_like,
		positive_exposure_backend_current,
		transpile_opt_level=positive_exposure_transpile_opt_level,
		seed_transpiler=positive_exposure_seed_transpiler,
	)
	positive_exposure_pred_current_training_like = crca_pos_current_training_like.function_values(
		positive_exposure_theta,
		shots=SHOTS,
		seed=SEED,
	)
	positive_exposure_kl_noise_current = _kl_divergence(
		positive_exposure_target,
		positive_exposure_pred_current_training_like,
	)
	positive_exposure_l2_noise_current = _l2_mse(
		positive_exposure_target,
		positive_exposure_pred_current_training_like,
	)

	crca_pos_snapshot_training_like = CrcaCircuit(
		m_time=num_qubits_time,
		n_price=num_qubits_underlying,
		n_layers=positive_exposure_layers,
		ansatz_type=positive_exposure_effective_topology,
		name="crca_positive_exposure_snapshot_training_like",
	)
	_set_crca_backend_training_like(
		crca_pos_snapshot_training_like,
		positive_exposure_backend_snapshot,
		transpile_opt_level=positive_exposure_transpile_opt_level,
		seed_transpiler=positive_exposure_seed_transpiler,
	)
	positive_exposure_pred_snapshot_training_like = crca_pos_snapshot_training_like.function_values(
		positive_exposure_theta,
		shots=SHOTS,
		seed=SEED,
	)
	positive_exposure_kl_noise_snapshot = _kl_divergence(
		positive_exposure_target,
		positive_exposure_pred_snapshot_training_like,
	)
	positive_exposure_l2_noise_snapshot = _l2_mse(
		positive_exposure_target,
		positive_exposure_pred_snapshot_training_like,
	)
	positive_exposure_l2_best_training = _npz_float(positive_exposure_data, "best_l2", float("nan"))
	positive_exposure_l2_final_training = _npz_float(positive_exposure_data, "final_l2", float("nan"))

	print("\n=== KL(target || salida(theta_star)) ===")
	print(f"Shots finitos usados: {SHOTS}")
	print(f"Backend para ruido: {BACKEND_NAME}")
	print(f"Topologia QCBM evaluada: {qcbm_topology}")
	print(f"Layout QCBM evaluado: {qcbm_layout}")
	print(f"Topologia positive exposure evaluada: {positive_exposure_effective_topology}")
	print(f"Layout positive exposure (training): {positive_exposure_layout}")
	if qcbm_used_noise_fallback is not None:
		print(f"Training QCBM uso fallback no-snapshot: {qcbm_used_noise_fallback}")
	print("\nNota: en CRCA se normaliza f_target y f_theta a distribuciones para computar KL.")
	print("\n=== QCBM KL: ruido actual vs snapshot ===")
	print(f"QCBM KL con ruido actual backend:  {qcbm_kl_noise_current:.8e}")
	print(f"QCBM KL con ruido snapshot ({NOISE_SNAPSHOT_ISO_UTC}): {qcbm_kl_noise_snapshot:.8e}")
	print("\n=== QCBM L2(MSE): ruido actual vs snapshot ===")
	print(f"QCBM L2(MSE) con ruido actual backend:  {qcbm_l2_noise_current:.8e}")
	print(f"QCBM L2(MSE) con ruido snapshot ({NOISE_SNAPSHOT_ISO_UTC}): {qcbm_l2_noise_snapshot:.8e}")
	print("\n=== Positive exposure KL: ruido actual vs snapshot (training-like) ===")
	print(f"Positive exposure KL con ruido actual backend:  {positive_exposure_kl_noise_current:.8e}")
	print(
		"Positive exposure KL con ruido snapshot "
		f"({NOISE_SNAPSHOT_ISO_UTC}): {positive_exposure_kl_noise_snapshot:.8e}"
	)
	print("\n=== Positive exposure L2(MSE): ruido actual vs snapshot (training-like) ===")
	print(f"Positive exposure L2(MSE) con ruido actual backend:  {positive_exposure_l2_noise_current:.8e}")
	print(
		"Positive exposure L2(MSE) con ruido snapshot "
		f"({NOISE_SNAPSHOT_ISO_UTC}): {positive_exposure_l2_noise_snapshot:.8e}"
	)
	print(
		"Positive exposure training reported L2(MSE) best/final: "
		f"{positive_exposure_l2_best_training:.8e} / {positive_exposure_l2_final_training:.8e}"
	)

	results = {
		"1) Statevector (ideal)": metrics_statevector,
		"2) Shots finitos (sin ruido)": metrics_shots,
		"3) Shots finitos + ruido backend": metrics_shots_noise,
	}

	for scenario, values in results.items():
		kl_values = values["kl"]
		l2_values = values["l2"]
		print(f"\n{scenario}")
		print(f"  QCBM KL:                  {kl_values['qcbm']:.8e}")
		print(f"  QCBM L2(MSE):             {l2_values['qcbm']:.8e}")
		print(f"  Positive exposure KL:     {kl_values['positive_exposure']:.8e}")
		print(f"  Positive exposure L2(MSE):{l2_values['positive_exposure']:.8e}")
		print(f"  Discount factors KL:      {kl_values['discount_factors']:.8e}")
		print(f"  Discount factors L2(MSE): {l2_values['discount_factors']:.8e}")
		print(f"  Default probabilities KL: {kl_values['default_probabilities']:.8e}")
		print(f"  Default probabilities L2(MSE): {l2_values['default_probabilities']:.8e}")


if __name__ == "__main__":
	main()
 