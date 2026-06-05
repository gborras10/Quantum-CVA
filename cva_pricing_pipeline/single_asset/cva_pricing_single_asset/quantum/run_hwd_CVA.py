from __future__ import annotations

# ruff: noqa: E402

import argparse
import math
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_aer import AerSimulator


# ======================================================================
#                       Project import paths
# ======================================================================
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    parent for parent in CURRENT_FILE.parents if (parent / "pyproject.toml").exists()
)
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from quantum_cva.amplitude_estimation.experiments.cva import (
    build_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.hardware import (
    ExperimentState,
    analyze_amplification,
    backend_snapshot,
    build_pass_manager_for_backend,
    effective_contrast_prefactor_for_algorithms,
    effective_t_for_algorithms,
    load_existing_state,
    run_amplification_scan,
    run_preflight,
    run_readout_calibration,
    run_replay,
)
from quantum_cva.amplitude_estimation.experiments.io import RunPaths, save_json
from quantum_cva.amplitude_estimation.experiments.samplers import (
    AerCountSampler,
    LoggedAerSampler,
    QctrlPerformanceManagementSampler,
    RuntimeCountSampler,
    build_noise_model,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    normalize_algorithm_key,
    run_algorithm_once,
)
from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (
    QuantumCVACircuit,
)
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)


# ======================================================================
#                       Single-asset experiment defaults
# ======================================================================
DEFAULT_ALGORITHMS = "cabiqae,biqae"
DEFAULT_BUDGETS = (
    "128,256,384,512,640,768,896,1024,1280,1536,1792,2048,2560,"
    "3072,3584,4096,4608,5120,6144,7168,8192,9216,10000"
)
DEFAULT_REFERENCE_KS = "0,1,2,3,4"
DEFAULT_MAX_GROVER_POWER = 4
DEFAULT_INSTANCE_NAME = "premium_new_usa"
DEFAULT_BACKEND_NAME = "ibm_pittsburgh"
DEFAULT_QISKIT_FUNCTION_NAME = "q-ctrl/performance-management"
DEFAULT_RUN_BASE_DIR = CURRENT_FILE.parent / "ae_cva" / "hardware" / "runs"

NUM_QUBITS_TIME = 2
NUM_QUBITS_UNDERLYING = 2
OBJECTIVE_QUBITS = [4, 5, 6]
GOOD_BITSTRING = "111"

SINGLE_ASSET_ARTIFACTS = {
    "benchmark": Path("data/single_asset/benchmark/run_classical_cva_single_asset.npz"),
    "qcbm": Path("data/single_asset/qcbm/qcbm_training_results_shots.npz"),
    "crca_exposure": Path(
        "data/single_asset/crca/positive_exposures/training_results_shots.npz"
    ),
    "crca_default": Path(
        "data/single_asset/crca/default_probabilities/training_results_shots.npz"
    ),
    "crca_discount": Path(
        "data/single_asset/crca/discount_factors/training_results_shots.npz"
    ),
}


# ======================================================================
#                       Single-asset CVA construction
# ======================================================================
def _load_npz(repo_root: Path, relative_path: Path) -> np.lib.npyio.NpzFile:
    artifact_path = repo_root / relative_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Required single-asset artifact not found: {artifact_path}")
    return np.load(artifact_path, allow_pickle=True)


def _theta(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.asarray(data["theta_star"], dtype=float).ravel()


def _assert_parameter_size(label: str, theta: np.ndarray, expected: int) -> None:
    actual = int(theta.size)
    if actual != int(expected):
        raise ValueError(
            f"Parameter-size mismatch for {label}: expected {expected}, got {actual}."
        )


def build_single_asset_cva_problem_bundle(
    repo_root: str | Path = REPO_ROOT,
) -> Any:
    """Build the single-asset CVA amplitude-estimation problem from trained data."""
    root = Path(repo_root)
    benchmark = _load_npz(root, SINGLE_ASSET_ARTIFACTS["benchmark"])
    qcbm_data = _load_npz(root, SINGLE_ASSET_ARTIFACTS["qcbm"])
    exposure_data = _load_npz(root, SINGLE_ASSET_ARTIFACTS["crca_exposure"])
    default_data = _load_npz(root, SINGLE_ASSET_ARTIFACTS["crca_default"])
    discount_data = _load_npz(root, SINGLE_ASSET_ARTIFACTS["crca_discount"])

    qcbm_theta = _theta(qcbm_data)
    exposure_theta = _theta(exposure_data)
    default_theta = _theta(default_data)
    discount_theta = _theta(discount_data)

    qcbm = MLQcbmCircuit(
        n_qubits=NUM_QUBITS_TIME + NUM_QUBITS_UNDERLYING,
        n_layers=2,
        name="single_asset_qcbm_state_preparation",
    )
    crca_exposure = CrcaCircuit(
        m_time=NUM_QUBITS_TIME,
        n_price=NUM_QUBITS_UNDERLYING,
        n_layers=2,
        name="single_asset_crca_positive_exposure",
    )
    crca_default = CrcaCircuit(
        m_time=NUM_QUBITS_TIME,
        n_price=0,
        n_layers=1,
        name="single_asset_crca_default_probabilities",
    )
    crca_discount = CrcaCircuit(
        m_time=NUM_QUBITS_TIME,
        n_price=0,
        n_layers=1,
        name="single_asset_crca_discount_factors",
    )

    _assert_parameter_size("QCBM", qcbm_theta, qcbm.n_params)
    _assert_parameter_size("CRCA exposure", exposure_theta, crca_exposure.n_params)
    _assert_parameter_size("CRCA default", default_theta, crca_default.n_params)
    _assert_parameter_size("CRCA discount", discount_theta, crca_discount.n_params)

    cva_circuit = QuantumCVACircuit(
        num_qubits_time=NUM_QUBITS_TIME,
        num_qubits_underlying=NUM_QUBITS_UNDERLYING,
        qcbm_circuit=qcbm,
        crca_circuit_exposure=crca_exposure,
        crca_circuit_default_prob=crca_default,
        crca_circuit_discount_factor=crca_discount,
        recovery_rate=float(benchmark["R_cva"]),
        C_v=float(benchmark["C_v"]),
        C_p=float(benchmark["C_p"]),
        C_q=float(benchmark["C_q"]),
        name="single_asset_quantum_cva_circuit",
        backend="statevector",
    )
    bundle = build_cva_problem_bundle(
        cva_circuit,
        qcbm_params=qcbm_theta,
        exposure_params=exposure_theta,
        default_prob_params=default_theta,
        discount_factor_params=discount_theta,
        metadata={
            "builder": "single_asset_cva",
            "training_regime": "shots",
            "state_register_order": "time_then_underlying",
            "execution_backend_note": (
                "ibm_pittsburgh is the default execution backend; trained single-asset "
                "artifacts were loaded unchanged and were not trained on this backend."
            ),
            "artifact_paths": {
                key: str(root / value) for key, value in SINGLE_ASSET_ARTIFACTS.items()
            },
        },
    )
    _validate_single_asset_bundle(bundle)
    return bundle


def _validate_single_asset_bundle(bundle: Any) -> None:
    if list(bundle.problem.objective_qubits) != OBJECTIVE_QUBITS:
        raise ValueError(
            "The single-asset CVA hardware experiment expects objective_qubits="
            f"{OBJECTIVE_QUBITS}, got {list(bundle.problem.objective_qubits)!r}."
        )
    if str(bundle.good_bitstring) != GOOD_BITSTRING:
        raise ValueError(
            "The single-asset CVA hardware experiment expects good_bitstring="
            f"{GOOD_BITSTRING!r}, got {bundle.good_bitstring!r}."
        )
    if int(bundle.problem.state_preparation.num_qubits) != 7:
        raise ValueError(
            "The single-asset CVA hardware experiment expects a 7-qubit circuit, "
            f"got {bundle.problem.state_preparation.num_qubits}."
        )


# ======================================================================
#                       Command-line configuration
# ======================================================================
def _parse_bool(raw: Any) -> bool:
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, or 1/0.")


def _parse_noise_floor(raw: Any) -> str | float:
    text = str(raw).strip().lower()
    if text in {"fit", "fitted", "estimate", "estimated"}:
        return "fit"
    if text in {"uniform_objective", "uniform"}:
        return "uniform_objective"
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("Expected a probability in [0, 1] or 'fit'.")
    return value


def _parse_names(raw: str | Sequence[str]) -> tuple[str, ...]:
    tokens = raw.replace(";", ",").replace(" ", ",").split(",") if isinstance(raw, str) else raw
    result = tuple(str(token).strip() for token in tokens if str(token).strip())
    if not result:
        raise ValueError("At least one value is required.")
    return result


def _parse_ints(raw: str, *, allow_zero: bool = False) -> list[int]:
    values = [int(token) for token in _parse_names(raw)]
    floor = 0 if allow_zero else 1
    if any(value < floor for value in values):
        raise ValueError(f"Integer values must be >= {floor}.")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hardware amplitude-estimation experiments for single-asset CVA."
    )
    parser.add_argument(
        "--mode",
        choices=("hardware", "preflight", "dry-run", "replay-only"),
        default="hardware",
    )
    parser.add_argument(
        "--hardware-executor",
        choices=("qctrl", "runtime"),
        default="qctrl",
    )
    parser.add_argument("--instance-name", default=DEFAULT_INSTANCE_NAME)
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND_NAME)
    parser.add_argument("--channel", default="ibm_cloud")
    parser.add_argument("--qiskit-function-name", default=DEFAULT_QISKIT_FUNCTION_NAME)
    parser.add_argument("--qiskit-function-channel", default=None)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--max-grover-power", type=int, default=DEFAULT_MAX_GROVER_POWER)
    parser.add_argument("--scan-grover-powers", default="0,1,2,3,4")
    parser.add_argument("--scan-repeats", type=int, default=2)
    parser.add_argument("--scan-shots", type=int, default=4096)
    parser.add_argument("--readout-shots", type=int, default=8192)
    parser.add_argument("--skip-readout-calibration", action="store_true")
    parser.add_argument("--direct-shots", type=int, default=128)
    parser.add_argument("--max-direct-calls", type=int, default=4)
    parser.add_argument("--direct", dest="skip_direct", action="store_false")
    parser.add_argument("--skip-direct", dest="skip_direct", action="store_true")
    parser.set_defaults(skip_direct=True)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--replay-repetitions", type=int, default=100)
    parser.add_argument("--replay-max-calls", type=int, default=4096)
    parser.add_argument(
        "--replay-probability-mode",
        choices=("fixed", "normal"),
        default="normal",
    )
    parser.add_argument("--replay-probability-se-scale", type=float, default=1.0)
    parser.add_argument("--extrapolate", type=_parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--epsilon-target", type=float, default=0.08)
    parser.add_argument("--cabiqae-epsilon-target", type=float, default=0.05)
    parser.add_argument("--biqae-epsilon-target", type=float, default=0.002)
    parser.add_argument("--session-max-time", default="24h")
    parser.add_argument("--soft-wallclock-limit", type=float, default=315360000.0)
    parser.add_argument("--max-isa-depth", type=int, default=12_000)
    parser.add_argument("--max-isa-2q", type=int, default=6_000)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=1234)
    parser.add_argument("--routing-method", default="sabre")
    parser.add_argument(
        "--layout-search-strategy",
        choices=("fast", "exhaustive", "preset"),
        default="exhaustive",
    )
    parser.add_argument("--reference-ks", default=DEFAULT_REFERENCE_KS)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dry-run-noise-scale", type=float, default=1.0)
    parser.add_argument("--dry-run-noise-profile", default="projected")
    parser.add_argument("--aer-method", default="density_matrix")
    parser.add_argument("--noise-floor", type=_parse_noise_floor, default="fit")
    parser.add_argument("--min-ideal-offset", type=float, default=0.08)
    parser.add_argument("--min-fit-contrast-z", type=float, default=2.0)
    parser.add_argument("--min-visible-contrast-z", type=float, default=3.0)
    parser.add_argument("--min-baseline-fit-points", type=int, default=3)
    parser.add_argument("--t-eff", type=float, default=None)
    parser.add_argument("--cap-kappa", type=float, default=1000.0)
    parser.add_argument(
        "--cabiqae-hard-k-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--execution-max-grover-power", type=int, default=None)
    parser.add_argument(
        "--use-fractional-gates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--show-details", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


# ======================================================================
#                       Experiment orchestration
# ======================================================================
def _create_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = DEFAULT_RUN_BASE_DIR / f"hardware_cva_ae_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_noise_floor(raw: str | float, bundle: Any) -> str | float:
    if str(raw) == "fit":
        return "fit"
    if str(raw) == "uniform_objective":
        return float(1.0 / (2.0**int(bundle.objective_width)))
    return float(raw)


def _runtime_service(args: argparse.Namespace) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    if str(args.instance_name).strip():
        return QiskitRuntimeService(name=str(args.instance_name))
    return QiskitRuntimeService(channel=str(args.channel))


def _load_backend(args: argparse.Namespace) -> tuple[Any, str]:
    if args.mode == "dry-run":
        return AerSimulator(), "aer_simulator"
    service = _runtime_service(args)
    try:
        backend = service.backend(
            str(args.backend_name),
            use_fractional_gates=bool(args.use_fractional_gates),
        )
    except TypeError:
        backend = service.backend(str(args.backend_name))
    return backend, "ibm_runtime_backend"


def _algorithms(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(normalize_algorithm_key(name) for name in _parse_names(args.algorithms))


def _epsilon_targets(args: argparse.Namespace, algorithms: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for algorithm in algorithms:
        if algorithm == "biqae" and args.biqae_epsilon_target is not None:
            result[algorithm] = float(args.biqae_epsilon_target)
        elif algorithm.startswith("cabiqae") and args.cabiqae_epsilon_target is not None:
            result[algorithm] = float(args.cabiqae_epsilon_target)
        else:
            result[algorithm] = float(args.epsilon_target)
    return result


def _alias_cva_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases = {
        "processed_true_value": "cva_true",
        "processed_estimate": "cva_estimate",
        "processed_abs_error": "cva_abs_error",
        "processed_relative_error": "cva_relative_error",
        "processed_ci_low": "cva_ci_low",
        "processed_ci_high": "cva_ci_high",
        "processed_abs_error_median": "cva_abs_error_median",
        "processed_relative_error_median": "cva_relative_error_median",
    }
    for row in rows:
        for source, target in aliases.items():
            if source in row and target not in row:
                row[target] = row[source]
    return rows


def _trace_extra(state: ExperimentState) -> dict[str, Any]:
    return {
        "hardware_mode": state.config["mode"],
        "backend_name": state.config["backend_name"],
        "instance_name": state.config["instance_name"],
        "backend_mode": state.config.get("backend_mode", ""),
        "hardware_executor": state.config["hardware_executor"],
        "channel": state.config["channel"],
        "execution_max_grover_power": state.config.get("execution_max_grover_power"),
    }


def _base_config(
    args: argparse.Namespace,
    bundle: Any,
    run_dir: Path,
    algorithms: Sequence[str],
    budgets: Sequence[int],
    contrast_baseline: str | float,
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "pipeline": "single_asset_cva_ae_hardware",
        "mode": str(args.mode),
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "channel": str(args.channel),
        "hardware_executor": str(args.hardware_executor),
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "use_fractional_gates_requested": bool(args.use_fractional_gates),
        "target_name": str(bundle.target_name),
        "good_bitstring": str(bundle.good_bitstring),
        "objective_qubits": [int(qubit) for qubit in bundle.problem.objective_qubits],
        "objective_width": int(bundle.objective_width),
        "a_true": float(bundle.true_amplitude),
        "processed_true_value": float(bundle.processed_true_value),
        "cva_true": float(bundle.processed_true_value),
        "bundle_metadata": dict(bundle.metadata),
        "max_grover_power_requested": int(args.max_grover_power),
        "scan_grover_powers": _parse_ints(args.scan_grover_powers, allow_zero=True),
        "scan_repeats": int(args.scan_repeats),
        "scan_shots": int(args.scan_shots),
        "readout_shots": int(args.readout_shots),
        "direct_shots": int(args.direct_shots),
        "skip_direct": bool(args.skip_direct),
        "replay_repetitions": int(args.replay_repetitions),
        "replay_probability_mode": str(args.replay_probability_mode),
        "replay_probability_se_scale": float(args.replay_probability_se_scale),
        "replay_extrapolate": bool(args.extrapolate),
        "budgets": [int(value) for value in budgets],
        "algorithms": list(algorithms),
        "epsilon_target": float(args.epsilon_target),
        "epsilon_targets": _epsilon_targets(args, algorithms),
        "noise_floor": contrast_baseline,
        "contrast_baseline": contrast_baseline,
        "session_max_time": str(args.session_max_time),
        "soft_wallclock_limit_seconds": float(args.soft_wallclock_limit),
        "optimization_level": int(args.optimization_level),
        "seed_transpiler": int(args.seed_transpiler),
        "layout_search_strategy": str(args.layout_search_strategy),
        "reference_ks": _parse_ints(args.reference_ks, allow_zero=True),
        "seed": int(args.seed),
        "hardware_risks": [
            "Grover-power contrast loss",
            "Q-CTRL queue/runtime cost",
            "backend calibration drift",
            "trained-model bias relative to the classical CVA reference",
        ],
    }


def _run_direct_live(
    state: ExperimentState,
    sampler: Any,
    bundle: Any,
    args: argparse.Namespace,
    algorithms: Sequence[str],
    t_eff: float | None,
    contrast_prefactor: float,
) -> None:
    epsilon_targets = _epsilon_targets(args, algorithms)
    for offset, algorithm in enumerate(algorithms):
        try:
            trace_rows, final_row = run_algorithm_once(
                algorithm,
                sampler,
                bundle,
                run_kind="direct_live",
                repetition=0,
                epsilon_target=epsilon_targets[algorithm],
                alpha=float(args.alpha),
                n_shots=int(args.direct_shots),
                max_queries=sys.maxsize,
                t_eff=t_eff,
                seed=int(args.seed) + offset,
                algorithm_labels=ALGORITHM_LABELS,
                cap_kappa=float(args.cap_kappa),
                disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
                solver_kwargs={
                    "noise_floor": float(state.calibration_summary["contrast_baseline"]),
                    "contrast_prefactor": float(contrast_prefactor),
                },
                trace_extra=_trace_extra(state),
                show_details=bool(args.show_details),
            )
            state.direct_trace_rows.extend(_alias_cva_rows(trace_rows))
            state.direct_final_rows.extend(_alias_cva_rows([final_row]))
            state.persist()
        except Exception as exc:
            state.error_rows.append(
                {
                    "phase": "direct_live",
                    "algorithm": str(algorithm),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "timestamp_epoch": time.time(),
                }
            )
            state.persist()


def _collect_and_analyze(
    args: argparse.Namespace,
    state: ExperimentState,
    sampler: Any,
    bundle: Any,
    max_experiment_k: int,
    algorithms: Sequence[str],
    contrast_baseline: str | float,
) -> None:
    if args.skip_readout_calibration:
        readout_rows: list[dict[str, Any]] = []
        readout = {
            "p_obs_good_given_bad": 0.0,
            "p_obs_good_given_good": 1.0,
            "readout_denom": 1.0,
            "readout_usable": 0.0,
            "readout_skipped": True,
        }
    else:
        readout_rows, readout = run_readout_calibration(
            sampler, bundle, shots=int(args.readout_shots)
        )
    state.readout_rows = readout_rows
    state.calibration_summary["readout"] = readout
    state.persist()

    grover_powers = _parse_ints(args.scan_grover_powers, allow_zero=True)
    too_high = [k for k in grover_powers if k > max_experiment_k]
    if too_high:
        raise ValueError(
            f"Scan Grover powers exceed the preflight cap {max_experiment_k}: {too_high}."
        )
    state.amplification_count_rows = run_amplification_scan(
        sampler,
        bundle,
        grover_powers=grover_powers,
        repeats=int(args.scan_repeats),
        shots=int(args.scan_shots),
        seed=int(args.seed),
        verbose=bool(args.verbose),
    )
    state.persist()

    try:
        points, calibration, _ = analyze_amplification(
            state.amplification_count_rows,
            bundle,
            readout,
            min_ideal_offset=float(args.min_ideal_offset),
            contrast_baseline=contrast_baseline,
            min_fit_contrast_z=float(args.min_fit_contrast_z),
            min_visible_contrast_z=float(args.min_visible_contrast_z),
            min_baseline_fit_points=int(args.min_baseline_fit_points),
        )
    except Exception as exc:
        state.error_rows.append(
            {
                "phase": "amplification_analysis",
                "algorithm": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_epoch": time.time(),
            }
        )
        state.calibration_summary["calibration_status"] = "analysis_failed"
        state.persist()
        raise
    state.amplification_point_rows = points
    state.calibration_summary.update(calibration)
    if args.t_eff is not None:
        t_eff = float(args.t_eff)
        contrast_prefactor = 1.0
    else:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        contrast_prefactor = effective_contrast_prefactor_for_algorithms(
            state.calibration_summary
        )
    if args.execution_max_grover_power is not None:
        direct_cap = int(args.execution_max_grover_power)
    elif calibration.get("calibration_status") == "ok":
        direct_cap = min(max_experiment_k, max(1, int(calibration.get("k_visible", 0))))
    else:
        direct_cap = min(1, max_experiment_k)
    state.config["execution_max_grover_power"] = int(direct_cap)
    state.config["calibration_status"] = calibration.get("calibration_status")
    state.persist()
    if hasattr(sampler, "max_grover_power"):
        sampler.max_grover_power = int(direct_cap)
    if not args.skip_direct:
        _run_direct_live(
            state,
            sampler,
            bundle,
            args,
            algorithms,
            t_eff,
            contrast_prefactor,
        )


def _run_replay(
    state: ExperimentState,
    bundle: Any,
    args: argparse.Namespace,
    algorithms: Sequence[str],
    budgets: Sequence[int],
) -> None:
    p_by_k = {
        int(row["grover_power"]): float(row["p_hw_mitigated"])
        for row in state.amplification_point_rows
    }
    p_se_by_k = {
        int(row["grover_power"]): float(row["p_hw_mitigated_se"])
        for row in state.amplification_point_rows
    }
    if args.t_eff is not None:
        t_eff = float(args.t_eff)
        contrast_prefactor = 1.0
    else:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        contrast_prefactor = effective_contrast_prefactor_for_algorithms(
            state.calibration_summary
        )
    run_replay(
        state,
        bundle,
        algorithms=algorithms,
        algorithm_labels=ALGORITHM_LABELS,
        p_by_k=p_by_k,
        p_se_by_k=p_se_by_k,
        replay_probability_mode=str(args.replay_probability_mode),
        replay_probability_se_scale=float(args.replay_probability_se_scale),
        budgets=budgets,
        repetitions=int(args.replay_repetitions),
        n_shots=int(args.direct_shots),
        epsilon_target=float(args.epsilon_target),
        epsilon_targets=_epsilon_targets(args, algorithms),
        alpha=float(args.alpha),
        t_eff=t_eff,
        contrast_prefactor=contrast_prefactor,
        seed=int(args.seed),
        replay_max_calls=int(args.replay_max_calls),
        extrapolate=bool(args.extrapolate),
        cap_kappa=float(args.cap_kappa),
        disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
        trace_extra=_trace_extra(state),
        verbose=bool(args.verbose),
    )
    state.direct_trace_rows = _alias_cva_rows(state.direct_trace_rows)
    state.direct_final_rows = _alias_cva_rows(state.direct_final_rows)
    state.replay_trace_rows = _alias_cva_rows(state.replay_trace_rows)
    state.replay_final_rows = _alias_cva_rows(state.replay_final_rows)
    state.replay_budget_rows = _alias_cva_rows(state.replay_budget_rows)
    state.budget_summary_rows = _alias_cva_rows(state.budget_summary_rows)
    state.persist()


def run_replay_only(args: argparse.Namespace) -> Path:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode replay-only.")
    bundle = build_single_asset_cva_problem_bundle(args.repo_root)
    algorithms = _algorithms(args)
    budgets = _parse_ints(args.budgets)
    run_dir = Path(args.run_dir).expanduser().resolve()
    state = load_existing_state(run_dir)
    if not state.amplification_point_rows:
        raise RuntimeError(
            "Replay-only requires existing amplification_points.csv calibration data."
        )
    _run_replay(state, bundle, args, algorithms, budgets)
    return run_dir


def run_experiment(args: argparse.Namespace) -> Path:
    bundle = build_single_asset_cva_problem_bundle(args.repo_root)
    algorithms = _algorithms(args)
    budgets = _parse_ints(args.budgets)
    contrast_baseline = _resolve_noise_floor(args.noise_floor, bundle)
    run_dir = _create_run_dir(args)
    paths = RunPaths(run_dir)
    state = ExperimentState(
        paths=paths,
        config=_base_config(
            args, bundle, run_dir, algorithms, budgets, contrast_baseline
        ),
    )
    state.session_details = {"mode": str(args.mode), "created_at_epoch": time.time()}

    backend, backend_mode = _load_backend(args)
    state.config["backend_mode"] = backend_mode
    snapshot = backend_snapshot(backend, mode=backend_mode, channel=str(args.channel))
    snapshot.update(
        {
            "hardware_executor": str(args.hardware_executor),
            "instance_name": str(args.instance_name),
            "backend_name_requested": str(args.backend_name),
            "qiskit_function_name": str(args.qiskit_function_name),
            "qctrl_submission_strategy": (
                "abstract_logical_circuits"
                if args.hardware_executor == "qctrl"
                else None
            ),
        }
    )
    save_json(snapshot, paths.backend_snapshot)

    pass_manager, transpilation = build_pass_manager_for_backend(
        backend,
        bundle,
        mode=str(args.mode),
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=_parse_ints(args.reference_ks, allow_zero=True),
        routing_method=str(args.routing_method),
        layout_search_strategy=str(args.layout_search_strategy),
        verbose=bool(args.verbose),
    )
    state.config["transpilation"] = transpilation
    if args.hardware_executor == "qctrl":
        state.config["qctrl_submission"] = {
            "primitive": "sampler",
            "transpilation_policy": "fire_opal_managed",
            "submitted_circuit_kind": "abstract_logical",
            "local_preflight_role": "diagnostic_and_k_cap_only",
        }
    state.persist()

    _, allowed_max = run_preflight(
        bundle,
        pass_manager,
        state,
        max_grover_power=int(args.max_grover_power),
        max_isa_depth=int(args.max_isa_depth),
        max_isa_2q=int(args.max_isa_2q),
        verbose=bool(args.verbose),
    )
    max_experiment_k = min(int(args.max_grover_power), int(allowed_max))
    state.config["max_grover_power_after_preflight"] = int(max_experiment_k)
    state.persist()
    if args.mode == "preflight":
        return run_dir

    if args.mode == "dry-run":
        aer = AerCountSampler(
            noise_model=build_noise_model(
                float(args.dry_run_noise_scale),
                profile=str(args.dry_run_noise_profile),
            ),
            seed=int(args.seed),
            method=str(args.aer_method),
            transpile_backend=backend,
            pass_manager=pass_manager,
        )
        sampler: Any = LoggedAerSampler(
            aer, state.job_rows, max_grover_power=max_experiment_k
        )
        _collect_and_analyze(
            args,
            state,
            sampler,
            bundle,
            max_experiment_k,
            algorithms,
            contrast_baseline,
        )
    elif args.hardware_executor == "qctrl":
        state.session_details.update(
            {
                "hardware_executor": "qctrl",
                "instance_name": str(args.instance_name),
                "backend_name": str(args.backend_name),
                "qiskit_function_name": str(args.qiskit_function_name),
                "submitted_circuit_kind": "abstract_logical",
                "session_started_at_epoch": time.time(),
            }
        )
        sampler = QctrlPerformanceManagementSampler(
            instance_name=str(args.instance_name),
            backend_name=str(args.backend_name),
            pass_manager=pass_manager,
            job_rows=state.job_rows,
            soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
            max_grover_power=max_experiment_k,
            max_calls_by_context={
                f"direct_live_{algorithm}": int(args.max_direct_calls)
                for algorithm in algorithms
            },
            start_time=time.perf_counter(),
            function_name=str(args.qiskit_function_name),
            catalog_channel=args.qiskit_function_channel,
            verbose=bool(args.verbose),
        )
        _collect_and_analyze(
            args,
            state,
            sampler,
            bundle,
            max_experiment_k,
            algorithms,
            contrast_baseline,
        )
        state.session_details["session_finished_at_epoch"] = time.time()
        state.persist()
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit_ibm_runtime import Session

        with Session(backend=backend, max_time=str(args.session_max_time)) as session:
            state.session_details.update(
                {
                    "hardware_executor": "runtime",
                    "session_id": getattr(session, "session_id", None),
                    "session_started_at_epoch": time.time(),
                }
            )
            sampler = RuntimeCountSampler(
                backend,
                Sampler(mode=session),
                pass_manager,
                state.job_rows,
                soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                max_grover_power=max_experiment_k,
                max_calls_by_context={
                    f"direct_live_{algorithm}": int(args.max_direct_calls)
                    for algorithm in algorithms
                },
                start_time=time.perf_counter(),
            )
            _collect_and_analyze(
                args,
                state,
                sampler,
                bundle,
                max_experiment_k,
                algorithms,
                contrast_baseline,
            )
            state.session_details["session_finished_at_epoch"] = time.time()
            state.persist()

    if not args.skip_replay:
        _run_replay(state, bundle, args, algorithms, budgets)
    return run_dir


class HardwareSingleAssetCvaExperimentRunner:
    """CLI-facing runner for new single-asset hardware CVA AE experiments."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @classmethod
    def from_cli(
        cls, argv: list[str] | None = None
    ) -> "HardwareSingleAssetCvaExperimentRunner":
        return cls(build_arg_parser().parse_args(argv))

    def run(self) -> Path:
        if self.args.mode == "replay-only":
            return run_replay_only(self.args)
        return run_experiment(self.args)


def main(argv: list[str] | None = None) -> None:
    runner = HardwareSingleAssetCvaExperimentRunner.from_cli(argv)
    run_dir = runner.run()
    print(f"Single-asset hardware CVA AE artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
