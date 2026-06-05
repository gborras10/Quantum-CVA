"""Run orchestration for the 6q hardware CVA amplitude-estimation experiment.

This module owns the experiment workflow.  The local `run_hwd_CVA.py` file only
parses command-line arguments through `HardwareCvaExperimentRunner` and calls
`run()`.

The core phases are:
- build the 6q CVA `EstimationProblem` from the pipeline config;
- load a fake, Aer, IBM Runtime, or Qiskit Functions/Q-CTRL backend path;
- build a pass manager and run ISA preflight diagnostics;
- collect readout and amplification calibration data;
- optionally run live direct AE on hardware or replay algorithms from measured
  probabilities;
- persist every artifact needed to reproduce the run.

The code intentionally preserves the existing probability semantics:
`good_bitstring='111'` on objective qubits `[6, 7, 8]`, CVA post-processing from
the problem bundle, and the existing replay CSV schema.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_aer import AerSimulator

from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_common import (
    CURRENT_DIR,
    REPO_ROOT,
    add_cva_aliases,
    load_config,
    parse_int_list,
    parse_name_list,
    parse_nonnegative_int_list,
)
from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.hardware import (
    ExperimentState,
    analyze_amplification,
    backend_snapshot,
    build_pass_manager_for_backend,
    effective_contrast_prefactor_for_algorithms,
    effective_t_for_algorithms,
    load_existing_state,
    load_fake_backend,
    make_replay_probability_extrapolator,
    run_amplification_scan,
    run_preflight,
    run_readout_calibration,
    run_replay,
)
from quantum_cva.amplitude_estimation.experiments.io import (
    RunPaths,
    save_csv,
    save_json,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (
    AerCountSampler,
    LoggedAerSampler,
    QctrlPerformanceManagementSampler,
    RuntimeCountSampler,
    build_noise_model,
    count_good_from_counts,
    extract_result_counts,
)
from quantum_cva.amplitude_estimation.experiments.solvers import (
    ALGORITHM_LABELS,
    normalize_algorithm_key,
    run_algorithm_once,
)


DEFAULT_ALGORITHMS = "cabiqae_latentt,biqae,bae"
DEFAULT_BUDGETS = "128,256,512,1024,2048,4096,8192,16384"
DEFAULT_REFERENCE_KS = "0,1,2,3,4"
DEFAULT_MAX_GROVER_POWER = 4
DEFAULT_FAKE_BACKEND = "fake_fez"
DEFAULT_INSTANCE_NAME = ""
DEFAULT_BACKEND_NAME = "ibm_aachen"
DEFAULT_QISKIT_FUNCTION_NAME = "q-ctrl/performance-management"
DEFAULT_HARDWARE_EXECUTOR = "runtime"

def _parse_bool(raw: Any) -> bool:
    # Argparse helper used for options that support explicit true/false values.
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean value: true/false, yes/no, 1/0."
    )


def _parse_noise_floor(raw: Any) -> str | float:
    text = str(raw).strip().lower()
    if text in {"fit", "fitted", "estimate", "estimated"}:
        return "fit"
    if text in {"uniform", "uniform_objective", "objective_uniform"}:
        return "uniform_objective"
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Expected a probability in [0, 1], 'uniform_objective', or 'fit'."
        ) from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("Expected a finite probability in [0, 1].")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    # Keep the CLI compatible with the original `run_hardware_cva_ae.py`.
    # `run_hwd_CVA.py` is now the preferred launcher, but the experiment modes
    # and flags intentionally have the same meaning.
    parser = argparse.ArgumentParser(
        description=(
            "Run hardware/replay amplitude-estimation experiments for the real "
            "6q CVA EstimationProblem."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "preflight",
            "hardware",
            "hardware-topup",
            "recover-session",
            "reanalyze-replay",
            "replay-only",
            "dry-run",
        ),
        default="dry-run",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            "Backward-compatible alias for --backend-name. Used for backend "
            "loading/transpilation when provided."
        ),
    )
    parser.add_argument(
        "--backend-name",
        default=DEFAULT_BACKEND_NAME,
        help=(
            "IBM backend name used for Runtime backend loading/transpilation. "
            "If --hardware-executor qctrl is explicitly selected, it is also "
            "passed as Qiskit Functions backend_name. Default: ibm_aachen."
        ),
    )
    parser.add_argument(
        "--instance-name",
        default=DEFAULT_INSTANCE_NAME,
        help=(
            "Saved IBM Quantum account name used as QiskitRuntimeService(name=...) "
            "for --hardware-executor runtime. Also used as QiskitFunctionsCatalog(name=...) "
            "when --hardware-executor qctrl is explicitly selected. Default: premium_new."
        ),
    )
    parser.add_argument(
        "--hardware-executor",
        choices=("qctrl", "runtime"),
        default=DEFAULT_HARDWARE_EXECUTOR,
        help=(
            "Live hardware submission path. Default: runtime. Use 'runtime' "
            "for IBM Runtime Session + SamplerV2. 'qctrl' is optional and only "
            "used when explicitly requested."
        ),
    )
    parser.add_argument(
        "--qiskit-function-name",
        default=DEFAULT_QISKIT_FUNCTION_NAME,
        help="Qiskit Function loaded from the catalog for --hardware-executor qctrl.",
    )
    parser.add_argument(
        "--qiskit-function-channel",
        default=None,
        help=(
            "Optional QiskitFunctionsCatalog channel for --hardware-executor qctrl, "
            "for example ibm_quantum_platform. If omitted, --instance-name is used "
            "as the saved catalog account name for backward compatibility."
        ),
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="IBM Runtime channel. Defaults to CONFIG.backend_noise.runtime_channel.",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline config object as 'module:CONFIG'. Defaults to full_cva_pipeline:CONFIG.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Python file containing a PipelineConfig object.",
    )
    parser.add_argument("--config-attr", default="CONFIG")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--max-grover-power", type=int, default=DEFAULT_MAX_GROVER_POWER)
    parser.add_argument("--scan-grover-powers", default=None)
    parser.add_argument("--session-max-time", default="5m")
    parser.add_argument(
        "--recover-session-id",
        default=None,
        help=(
            "IBM Runtime session id to recover. For Q-CTRL, this identifies the "
            "Qiskit Function job through its associated runtime_sessions metadata. "
            "Defaults to the session id saved in --run-dir/session_details.json."
        ),
    )
    parser.add_argument(
        "--recover-qctrl-job-id",
        default=None,
        help=(
            "Optional Qiskit Function job id to recover directly. Use this when "
            "--recover-session-id is unavailable or matches multiple Q-CTRL jobs."
        ),
    )
    parser.add_argument("--scan-repeats", type=int, default=1)
    parser.add_argument("--scan-shots", type=int, default=512)
    parser.add_argument(
        "--topup-replace-existing-powers",
        action="store_true",
        help=(
            "In --mode hardware-topup, replace the persisted amplification-count "
            "rows for the requested --scan-grover-powers instead of appending to "
            "them. Removed rows are saved to a timestamped backup CSV."
        ),
    )
    parser.add_argument("--readout-shots", type=int, default=1024)
    parser.add_argument(
        "--skip-readout-calibration",
        action="store_true",
        help=(
            "Skip live readout calibration and apply identity correction; "
            "amplification probabilities remain unmitigated raw estimates."
        ),
    )
    parser.add_argument("--direct-shots", type=int, default=128)
    parser.add_argument("--max-direct-calls", type=int, default=4)
    parser.add_argument("--replay-repetitions", type=int, default=200)
    parser.add_argument(
        "--replay-max-calls",
        type=int,
        default=128,
        help="Maximum sampler calls allowed per algorithm repetition in replay mode.",
    )
    parser.add_argument(
        "--replay-probability-mode",
        choices=("fixed", "normal"),
        default="normal",
    )
    parser.add_argument("--replay-probability-se-scale", type=float, default=1.0)
    parser.add_argument("--extrapolate", type=_parse_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--cabiqae-replay-contrast-model-all-k",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synthetic calibrated study: feed CABIQAE the fitted contrast-model "
            "probability for every k instead of empirical hardware scan points. "
            "BIQAE continues to use empirical hardware replay probabilities."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--epsilon-target", type=float, default=0.08)
    parser.add_argument(
        "--cabiqae-epsilon-target",
        type=float,
        default=None,
        help=(
            "Optional epsilon_target override for CABIQAE variants. "
            "Defaults to --epsilon-target."
        ),
    )
    parser.add_argument(
        "--biqae-epsilon-target",
        type=float,
        default=None,
        help=(
            "Optional epsilon_target override for BIQAE. "
            "Defaults to --epsilon-target."
        ),
    )
    parser.add_argument("--soft-wallclock-limit", type=float, default=7000.0)
    parser.add_argument(
        "--max-isa-depth",
        type=int,
        default=12_000,
        help="Preflight depth limit. Existing CVA k=4 layouts are around 7.5k.",
    )
    parser.add_argument(
        "--max-isa-2q",
        type=int,
        default=6_000,
        help="Preflight 2q-gate limit. Existing CVA k=4 layouts are around 3.8k.",
    )
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=1234)
    parser.add_argument("--routing-method", default="sabre")
    parser.add_argument(
        "--layout-search-strategy",
        choices=("fast", "exhaustive", "preset"),
        default="fast",
    )
    parser.add_argument("--reference-ks", default=DEFAULT_REFERENCE_KS)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fake-backend", default=DEFAULT_FAKE_BACKEND)
    parser.add_argument("--dry-run-noise-scale", type=float, default=1.0)
    parser.add_argument("--dry-run-noise-profile", default="projected")
    parser.add_argument("--aer-method", default="density_matrix")
    parser.add_argument(
        "--noise-floor",
        type=_parse_noise_floor,
        default="uniform_objective",
        help=(
            "Asymptotic good-state probability under full contrast loss. "
            "Default 'uniform_objective' is 1 / 2**objective_width = 0.125 for CVA."
        ),
    )
    parser.add_argument("--min-ideal-offset", type=float, default=0.08)
    parser.add_argument("--min-fit-contrast-z", type=float, default=2.0)
    parser.add_argument("--min-visible-contrast-z", type=float, default=3.0)
    parser.add_argument("--min-baseline-fit-points", type=int, default=3)
    parser.add_argument(
        "--allow-negative-contrast-fit-points",
        action="store_true",
        help=(
            "Fit the calibrated exponential contrast model in probability space "
            "so wrong-side observations can contribute as weighted residuals. "
            "The modeled contrast remains positive."
        ),
    )
    parser.add_argument("--t-eff", type=float, default=None)
    parser.add_argument("--cap-kappa", type=float, default=1000.0)
    parser.add_argument(
        "--cabiqae-hard-k-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable CABIQAE's hard k cap. By default only the hard cap is "
            "disabled; the noise-aware scheduler still receives T_eff."
        ),
    )
    parser.add_argument(
        "--execution-max-grover-power",
        type=int,
        default=None,
        help=(
            "Hard k cap for live direct AE. If omitted, the cap is selected "
            "from preflight and contrast calibration."
        ),
    )
    parser.add_argument(
        "--use-fractional-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request IBM backend targets with use_fractional_gates=True.",
    )
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-max-queries", type=float, default=None)
    parser.add_argument("--include-monte-carlo-plots", action="store_true")
    parser.add_argument("--show-details", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_backend_defaults(args: argparse.Namespace, config: Any) -> None:
    backend_cfg = getattr(config, "backend_noise", None)
    if args.backend is None:
        args.backend = str(args.backend_name or getattr(backend_cfg, "backend_name", DEFAULT_BACKEND_NAME))
    else:
        args.backend_name = str(args.backend)
    if args.channel is None:
        args.channel = getattr(backend_cfg, "runtime_channel", "ibm_cloud")


def _parse_algorithms(raw: str | Sequence[str]) -> tuple[str, ...]:
    return tuple(normalize_algorithm_key(token) for token in parse_name_list(raw))


def _epsilon_target_for_algorithm(args: argparse.Namespace, algorithm: str) -> float:
    key = normalize_algorithm_key(algorithm)
    if key == "biqae" and args.biqae_epsilon_target is not None:
        return float(args.biqae_epsilon_target)
    if key in {"cabiqae", "cabiqae_known_t", "cabiqae_latentt"}:
        if args.cabiqae_epsilon_target is not None:
            return float(args.cabiqae_epsilon_target)
    return float(args.epsilon_target)


def _epsilon_target_config(
    args: argparse.Namespace,
    algorithms: Sequence[str],
) -> dict[str, float]:
    return {
        normalize_algorithm_key(algorithm): _epsilon_target_for_algorithm(args, algorithm)
        for algorithm in algorithms
    }


def _validate_bundle(bundle: Any) -> None:
    # This runner is specialized to the current 6q CVA construction.  The
    # measurement and readout mitigation code assumes these objective qubits and
    # this good bitstring, so fail loudly if the config points to a different
    # circuit family.
    if str(bundle.good_bitstring) != "111":
        raise ValueError(
            "The 6q CVA AE hardware pipeline expects good_bitstring='111', got "
            f"{bundle.good_bitstring!r}."
        )
    objective_qubits = list(bundle.problem.objective_qubits)
    if objective_qubits != [6, 7, 8]:
        raise ValueError(
            "The 6q CVA AE hardware pipeline expects objective_qubits=[6, 7, 8], "
            f"got {objective_qubits!r}."
        )


def _resolve_noise_floor(raw: str | float, bundle: Any) -> str | float:
    if str(raw) == "fit":
        return "fit"
    if str(raw) == "uniform_objective":
        return float(1.0 / (2.0 ** int(bundle.objective_width)))
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"noise-floor must be in [0, 1], got {raw!r}.")
    return value


def _create_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = CURRENT_DIR / "runs" / f"hardware_cva_ae_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_service(channel: str | None) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    if channel:
        return QiskitRuntimeService(channel=channel)
    return QiskitRuntimeService()


def _runtime_service_from_args(args: argparse.Namespace) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service_name = str(getattr(args, "instance_name", "") or "").strip()
    if service_name:
        # Match the working notebook pattern:
        # QiskitRuntimeService(name="premium_new")
        return QiskitRuntimeService(name=service_name)
    return _runtime_service(getattr(args, "channel", None))


def _load_runtime_backend(
    *,
    backend_name: str,
    args: argparse.Namespace,
    use_fractional_gates: bool,
) -> Any:
    service = _runtime_service_from_args(args)
    try:
        return service.backend(
            str(backend_name),
            use_fractional_gates=bool(use_fractional_gates),
        )
    except TypeError:
        return service.backend(str(backend_name))


def _load_backend_for_mode(args: argparse.Namespace, mode: str) -> tuple[Any, str]:
    if mode == "dry-run":
        if args.fake_backend and str(args.fake_backend).strip().lower() not in {
            "",
            "none",
            "aer",
        }:
            try:
                return load_fake_backend(str(args.fake_backend)), f"fake:{args.fake_backend}"
            except Exception as exc:
                if args.verbose:
                    print(
                        "[backend] fake backend unavailable; using local AerSimulator: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        return AerSimulator(), "aer_simulator"

    backend = _load_runtime_backend(
        backend_name=str(args.backend),
        args=args,
        use_fractional_gates=bool(args.use_fractional_gates),
    )
    return backend, "ibm_runtime_backend"


def _build_problem_bundle(args: argparse.Namespace, config: Any) -> Any:
    bundle = build_6q_cva_problem_bundle(config, repo_root=args.repo_root)
    _validate_bundle(bundle)
    return bundle


def _base_config(
    *,
    args: argparse.Namespace,
    bundle: Any,
    run_dir: Path,
    mode: str,
    algorithms: Sequence[str],
    budgets: Sequence[int],
    contrast_baseline: str | float,
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "run_uuid": str(uuid.uuid4()),
        "pipeline": "6q_cva_ae_hardware",
        "mode": mode,
        "backend": str(args.backend),
        "backend_name": str(args.backend_name),
        "instance_name": str(args.instance_name),
        "runtime_service_name": str(args.instance_name),
        "channel": str(args.channel),
        "hardware_executor": str(args.hardware_executor),
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "use_fractional_gates_requested": bool(args.use_fractional_gates),
        "target_name": str(bundle.target_name),
        "good_bitstring": str(bundle.good_bitstring),
        "objective_qubits": [int(q) for q in bundle.problem.objective_qubits],
        "objective_width": int(bundle.objective_width),
        "a_true": float(bundle.true_amplitude),
        "processed_true_value": float(bundle.processed_true_value),
        "cva_true": float(bundle.processed_true_value),
        "bundle_metadata": dict(bundle.metadata),
        "max_grover_power_requested": int(args.max_grover_power),
        "scan_repeats": int(args.scan_repeats),
        "scan_shots": int(args.scan_shots),
        "readout_shots": int(args.readout_shots),
        "skip_readout_calibration": bool(args.skip_readout_calibration),
        "readout_mitigation_mode": (
            "identity_no_calibration"
            if bool(args.skip_readout_calibration)
            else "measured_linear_correction"
        ),
        "direct_shots": int(args.direct_shots),
        "max_direct_sampler_calls_per_algorithm": int(args.max_direct_calls),
        "replay_repetitions": int(args.replay_repetitions),
        "replay_probability_mode": str(args.replay_probability_mode),
        "replay_probability_se_scale": float(args.replay_probability_se_scale),
        "replay_extrapolate": bool(args.extrapolate),
        "cabiqae_replay_contrast_model_all_k": bool(
            args.cabiqae_replay_contrast_model_all_k
        ),
        "budgets": [int(x) for x in budgets],
        "session_max_time": str(args.session_max_time),
        "soft_wallclock_limit_seconds": float(args.soft_wallclock_limit),
        "max_isa_depth": int(args.max_isa_depth),
        "max_isa_2q": int(args.max_isa_2q),
        "optimization_level": int(args.optimization_level),
        "seed_transpiler": int(args.seed_transpiler),
        "routing_method": str(args.routing_method),
        "layout_search_strategy": str(args.layout_search_strategy),
        "reference_ks": parse_nonnegative_int_list(args.reference_ks),
        "seed": int(args.seed),
        "algorithms": list(algorithms),
        "algorithm_labels": {
            key: ALGORITHM_LABELS.get(key, key) for key in algorithms
        },
        "epsilon_target": float(args.epsilon_target),
        "epsilon_targets": _epsilon_target_config(args, algorithms),
        "cabiqae_epsilon_target": None
        if args.cabiqae_epsilon_target is None
        else float(args.cabiqae_epsilon_target),
        "biqae_epsilon_target": None
        if args.biqae_epsilon_target is None
        else float(args.biqae_epsilon_target),
        "noise_floor": contrast_baseline,
        "contrast_baseline": contrast_baseline,
        "min_ideal_offset": float(args.min_ideal_offset),
        "min_fit_contrast_z": float(args.min_fit_contrast_z),
        "min_visible_contrast_z": float(args.min_visible_contrast_z),
        "min_baseline_fit_points": int(args.min_baseline_fit_points),
        "allow_negative_contrast_fit_points": bool(
            args.allow_negative_contrast_fit_points
        ),
        "cap_kappa": float(args.cap_kappa),
        "cabiqae_hard_k_cap": bool(args.cabiqae_hard_k_cap),
    }


def _qctrl_submission_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primitive": "sampler",
        "qiskit_function_name": str(args.qiskit_function_name),
        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
        "backend_name": str(args.backend_name),
        "submitted_circuit_kind": "abstract_logical",
        "local_pass_manager_applied_to_submitted_circuits": False,
        "transpilation_policy": "fire_opal_managed",
        "local_preflight_role": "diagnostic_and_k_cap_only",
        "pub_shape": "(circuit, None, shots)",
        "runtime_session_strategy": "existing_qiskit_runtime_session",
        "amplification_scan_submission": "batched_pubs_single_job",
        "result_counts_registers_checked": [
            "c0",
            "c",
            "<first register with get_counts>",
        ],
    }


def _apply_cva_aliases_to_state(state: ExperimentState) -> None:
    state.direct_trace_rows = add_cva_aliases(state.direct_trace_rows)
    state.direct_final_rows = add_cva_aliases(state.direct_final_rows)
    state.replay_trace_rows = add_cva_aliases(state.replay_trace_rows)
    state.replay_final_rows = add_cva_aliases(state.replay_final_rows)
    state.replay_budget_rows = add_cva_aliases(state.replay_budget_rows)
    state.budget_summary_rows = add_cva_aliases(state.budget_summary_rows)


def _state_extra(state: ExperimentState) -> dict[str, Any]:
    return {
        "hardware_mode": str(state.config.get("mode", "")),
        "backend": str(state.config.get("backend", "")),
        "backend_name": str(state.config.get("backend_name", "")),
        "instance_name": str(state.config.get("instance_name", "")),
        "backend_mode": str(state.config.get("backend_mode", "")),
        "hardware_executor": str(state.config.get("hardware_executor", "")),
        "channel": str(state.config.get("channel", "")),
        "execution_max_grover_power": state.config.get("execution_max_grover_power"),
    }


def _run_plotter(args: argparse.Namespace, run_dir: Path) -> None:
    # Plot generation is intentionally a subprocess.  If plotting fails, the
    # experiment artifacts are still written and can be inspected manually.
    if args.skip_plots:
        return
    command = [
        sys.executable,
        str(CURRENT_DIR / "plot_hwd_CVA.py"),
        "--kind",
        "standard",
        "--run-dir",
        str(run_dir),
    ]
    if args.plot_max_queries is not None:
        command.extend(["--max-queries", str(float(args.plot_max_queries))])
    if args.include_monte_carlo_plots:
        command.append("--include-monte-carlo")
    subprocess.run(command, check=False)


def _load_replay_probabilities_from_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, float]:
    by_k: dict[int, list[tuple[int, int]]] = {}
    for row in rows:
        k = int(float(row["grover_power"]))
        good = int(float(row.get("good_counts", row.get("one_counts", 0))))
        shots = int(float(row["shots"]))
        by_k.setdefault(k, []).append((good, shots))
    return {
        k: float(sum(good for good, _ in values) / max(sum(shots for _, shots in values), 1))
        for k, values in by_k.items()
    }


def _replay_inputs_from_state(
    state: ExperimentState,
) -> tuple[dict[int, float], dict[int, float] | None]:
    if state.amplification_point_rows:
        return (
            {
                int(float(row["grover_power"])): float(row["p_hw_mitigated"])
                for row in state.amplification_point_rows
            },
            {
                int(float(row["grover_power"])): float(row["p_hw_mitigated_se"])
                for row in state.amplification_point_rows
            },
        )
    if state.amplification_count_rows:
        return _load_replay_probabilities_from_counts(state.amplification_count_rows), None
    raise RuntimeError(
        "No amplification_points.csv or amplification_counts.csv data are available "
        "for replay."
    )


def _run_replay_from_state(
    state: ExperimentState,
    bundle: Any,
    args: argparse.Namespace,
    *,
    algorithms: Sequence[str],
    budgets: Sequence[int],
) -> None:
    # Replay reuses measured hardware probabilities.  This is the statistically
    # cheap way to compare adaptive AE algorithms without submitting new jobs.
    p_by_k, p_se_by_k = _replay_inputs_from_state(state)
    if args.replay_probability_mode == "normal" and p_se_by_k is None:
        raise ValueError(
            "--replay-probability-mode normal requires p_hw_mitigated_se values "
            "from amplification_points.csv."
        )
    if args.t_eff is not None:
        t_eff = float(args.t_eff)
        contrast_prefactor = 1.0
    else:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        contrast_prefactor = effective_contrast_prefactor_for_algorithms(
            state.calibration_summary
        )
    algorithm_p_by_k: dict[str, dict[int, float]] = {}
    algorithm_p_se_by_k: dict[str, dict[int, float]] = {}
    algorithm_probability_sources: dict[str, str] = {}
    if bool(args.cabiqae_replay_contrast_model_all_k):
        if not bool(args.extrapolate):
            raise ValueError(
                "--cabiqae-replay-contrast-model-all-k requires --extrapolate true "
                "so CABIQAE can use the fitted model beyond scanned k values."
            )
        model_probability, model_metadata = make_replay_probability_extrapolator(
            bundle,
            state.calibration_summary,
        )
        modeled_probabilities = {
            int(k): float(model_probability(int(k))) for k in p_by_k
        }
        modeled_standard_errors = {int(k): 0.0 for k in p_by_k}
        for algorithm in algorithms:
            algorithm_key = normalize_algorithm_key(algorithm)
            if algorithm_key in {"cabiqae", "cabiqae_known_t", "cabiqae_latentt"}:
                algorithm_p_by_k[algorithm_key] = modeled_probabilities
                algorithm_p_se_by_k[algorithm_key] = modeled_standard_errors
                algorithm_probability_sources[algorithm_key] = "contrast_model_all_k"
        state.config["cabiqae_replay_contrast_model"] = {
            **model_metadata,
            "probabilities_by_k": {
                str(k): float(value) for k, value in sorted(modeled_probabilities.items())
            },
            "standard_errors_by_k": {
                str(k): 0.0 for k in sorted(modeled_standard_errors)
            },
            "study_kind": "synthetic_calibrated_contrast_model",
        }
    state.config["replay_repetitions"] = int(args.replay_repetitions)
    state.config["replay_probability_mode"] = str(args.replay_probability_mode)
    state.config["replay_probability_se_scale"] = float(args.replay_probability_se_scale)
    state.config["replay_extrapolate"] = bool(args.extrapolate)
    state.config["budgets"] = [int(value) for value in budgets]
    state.config["cabiqae_replay_contrast_model_all_k"] = bool(
        args.cabiqae_replay_contrast_model_all_k
    )
    state.config["epsilon_target"] = float(args.epsilon_target)
    state.config["epsilon_targets"] = _epsilon_target_config(args, algorithms)
    state.config["cabiqae_epsilon_target"] = (
        None
        if args.cabiqae_epsilon_target is None
        else float(args.cabiqae_epsilon_target)
    )
    state.config["biqae_epsilon_target"] = (
        None
        if args.biqae_epsilon_target is None
        else float(args.biqae_epsilon_target)
    )
    run_replay(
        state,
        bundle,
        algorithms=tuple(algorithms),
        algorithm_labels=ALGORITHM_LABELS,
        p_by_k=p_by_k,
        p_se_by_k=p_se_by_k,
        algorithm_p_by_k=algorithm_p_by_k,
        algorithm_p_se_by_k=algorithm_p_se_by_k,
        algorithm_probability_sources=algorithm_probability_sources,
        replay_probability_mode=str(args.replay_probability_mode),
        replay_probability_se_scale=float(args.replay_probability_se_scale),
        budgets=tuple(int(x) for x in budgets),
        repetitions=int(args.replay_repetitions),
        n_shots=int(args.direct_shots),
        epsilon_target=float(args.epsilon_target),
        epsilon_targets=_epsilon_target_config(args, algorithms),
        alpha=float(args.alpha),
        t_eff=t_eff,
        contrast_prefactor=contrast_prefactor,
        seed=int(args.seed),
        replay_max_calls=int(args.replay_max_calls),
        extrapolate=bool(args.extrapolate),
        cap_kappa=float(args.cap_kappa),
        disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
        trace_extra=_state_extra(state),
        verbose=bool(args.verbose),
    )
    _apply_cva_aliases_to_state(state)
    state.persist()


def _reanalyze_amplification_from_state(
    state: ExperimentState,
    bundle: Any,
    args: argparse.Namespace,
) -> None:
    """Rebuild calibration artifacts from local hardware counts only.

    This is the post-flight recovery path used when the QPU jobs completed but
    the original contrast analysis was too strict.  It intentionally avoids
    backend loading, sessions, and job submission.
    """
    if not state.amplification_count_rows:
        raise RuntimeError(
            "Existing run has no amplification_counts.csv rows to reanalyze."
        )

    readout_params = state.calibration_summary.get("readout")
    if not isinstance(readout_params, Mapping):
        readout_params = _readout_params_from_rows(state.readout_rows)
        state.calibration_summary["readout"] = readout_params

    contrast_baseline = _resolve_noise_floor(args.noise_floor, bundle)
    state.config["postflight_reanalysis"] = {
        "started_at_epoch": time.time(),
        "source": "local_amplification_counts",
        "noise_floor": contrast_baseline,
        "min_ideal_offset": float(args.min_ideal_offset),
        "min_fit_contrast_z": float(args.min_fit_contrast_z),
        "min_visible_contrast_z": float(args.min_visible_contrast_z),
        "min_baseline_fit_points": int(args.min_baseline_fit_points),
        "allow_negative_contrast_fit_points": bool(
            args.allow_negative_contrast_fit_points
        ),
    }
    state.config["noise_floor"] = contrast_baseline
    state.config["contrast_baseline"] = contrast_baseline
    state.config["min_ideal_offset"] = float(args.min_ideal_offset)
    state.config["min_fit_contrast_z"] = float(args.min_fit_contrast_z)
    state.config["min_visible_contrast_z"] = float(args.min_visible_contrast_z)
    state.config["min_baseline_fit_points"] = int(args.min_baseline_fit_points)
    state.config["allow_negative_contrast_fit_points"] = bool(
        args.allow_negative_contrast_fit_points
    )

    try:
        points, calibration_summary, _ = analyze_amplification(
            state.amplification_count_rows,
            bundle,
            readout_params,
            min_ideal_offset=float(args.min_ideal_offset),
            contrast_baseline=contrast_baseline,
            min_fit_contrast_z=float(args.min_fit_contrast_z),
            min_visible_contrast_z=float(args.min_visible_contrast_z),
            min_baseline_fit_points=int(args.min_baseline_fit_points),
            allow_negative_contrast_fit_points=bool(
                args.allow_negative_contrast_fit_points
            ),
        )
    except Exception as exc:
        state.error_rows.append(
            {
                "phase": "postflight_reanalysis",
                "algorithm": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_epoch": time.time(),
            }
        )
        state.calibration_summary["calibration_status"] = "analysis_failed"
        state.calibration_summary["analysis_error_type"] = type(exc).__name__
        state.calibration_summary["analysis_error"] = str(exc)
        state.persist()
        raise

    state.amplification_point_rows = points
    for stale_key in ("analysis_error_type", "analysis_error"):
        state.calibration_summary.pop(stale_key, None)
    state.calibration_summary.update(calibration_summary)
    if args.t_eff is not None:
        state.calibration_summary["t_eff_manual_override"] = float(args.t_eff)
    state.config["calibration_status"] = calibration_summary.get("calibration_status")
    state.config["postflight_reanalysis"]["finished_at_epoch"] = time.time()
    state.persist()


def _run_direct_live(
    state: ExperimentState,
    sampler: Any,
    bundle: Any,
    args: argparse.Namespace,
    *,
    algorithms: Sequence[str],
    t_eff: float | None,
    contrast_baseline: float,
    contrast_prefactor: float,
) -> None:
    for offset, algorithm in enumerate(algorithms):
        epsilon_target = _epsilon_target_for_algorithm(args, algorithm)
        try:
            trace_rows, final_row = run_algorithm_once(
                algorithm,
                sampler,
                bundle,
                run_kind="direct_live",
                repetition=0,
                epsilon_target=epsilon_target,
                alpha=float(args.alpha),
                n_shots=int(args.direct_shots),
                max_queries=sys.maxsize,
                t_eff=t_eff,
                seed=int(args.seed) + int(offset),
                algorithm_labels=ALGORITHM_LABELS,
                cap_kappa=float(args.cap_kappa),
                disable_hard_k_cap=not bool(args.cabiqae_hard_k_cap),
                solver_kwargs={
                    "noise_floor": float(contrast_baseline),
                    "contrast_prefactor": float(contrast_prefactor),
                },
                trace_extra={
                    **_state_extra(state),
                    "epsilon_target": epsilon_target,
                },
                show_details=bool(args.show_details),
            )
            state.direct_trace_rows.extend(add_cva_aliases(trace_rows))
            state.direct_final_rows.append(add_cva_aliases([final_row])[0])
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


def _execute_non_replay_phases(
    args: argparse.Namespace,
    state: ExperimentState,
    sampler: Any,
    bundle: Any,
    max_experiment_k: int,
    *,
    algorithms: Sequence[str],
    contrast_baseline: str | float,
    batch_scan_circuits: bool = False,
) -> None:
    # Shared body for dry-run and live hardware modes:
    # optional readout calibration -> amplification scan -> contrast analysis ->
    # optional direct live AE. Replay is handled afterwards by `run_experiment`.
    if bool(args.skip_readout_calibration):
        readout_rows = []
        readout_params: dict[str, Any] = {
            "p_obs_good_given_bad": 0.0,
            "p_obs_good_given_good": 1.0,
            "readout_denom": 1.0,
            "readout_usable": 0.0,
            "readout_skipped": True,
            "readout_mitigation_mode": "identity_no_calibration",
        }
        if bool(args.verbose):
            print(
                "[readout_calibration] skipped: using identity correction; "
                "scan probabilities are unmitigated.",
                flush=True,
            )
    else:
        readout_rows, measured_readout_params = run_readout_calibration(
            sampler,
            bundle,
            shots=int(args.readout_shots),
        )
        readout_params = {
            **measured_readout_params,
            "readout_skipped": False,
            "readout_mitigation_mode": "measured_linear_correction",
        }
    state.readout_rows = readout_rows
    state.calibration_summary["readout"] = readout_params
    state.persist()

    if args.scan_grover_powers:
        grover_powers = parse_nonnegative_int_list(args.scan_grover_powers)
        too_high = [k for k in grover_powers if int(k) > int(max_experiment_k)]
        if too_high:
            raise ValueError(
                "Requested scan Grover powers exceed the preflight cap "
                f"{max_experiment_k}: {too_high}"
            )
    else:
        grover_powers = list(range(int(max_experiment_k) + 1))

    state.amplification_count_rows = run_amplification_scan(
        sampler,
        bundle,
        grover_powers=grover_powers,
        repeats=int(args.scan_repeats),
        shots=int(args.scan_shots),
        seed=int(args.seed),
        batch_circuits=bool(batch_scan_circuits),
        verbose=bool(args.verbose),
    )
    state.persist()
    try:
        points, calibration_summary, _ = analyze_amplification(
            state.amplification_count_rows,
            bundle,
            readout_params,
            min_ideal_offset=float(args.min_ideal_offset),
            contrast_baseline=contrast_baseline,
            min_fit_contrast_z=float(args.min_fit_contrast_z),
            min_visible_contrast_z=float(args.min_visible_contrast_z),
            min_baseline_fit_points=int(args.min_baseline_fit_points),
            allow_negative_contrast_fit_points=bool(
                args.allow_negative_contrast_fit_points
            ),
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
        state.calibration_summary["analysis_error_type"] = type(exc).__name__
        state.calibration_summary["analysis_error"] = str(exc)
        state.persist()
        raise
    state.amplification_point_rows = points
    state.calibration_summary.update(calibration_summary)

    if args.t_eff is not None:
        state.calibration_summary["t_eff_manual_override"] = float(args.t_eff)
        t_eff = float(args.t_eff)
        contrast_prefactor = 1.0
    else:
        t_eff = effective_t_for_algorithms(state.calibration_summary)
        contrast_prefactor = effective_contrast_prefactor_for_algorithms(
            state.calibration_summary
        )

    if args.execution_max_grover_power is not None:
        direct_cap = int(args.execution_max_grover_power)
    elif calibration_summary.get("calibration_status") == "ok":
        direct_cap = min(
            int(max_experiment_k),
            max(1, int(calibration_summary.get("k_visible", 0))),
        )
    else:
        direct_cap = min(1, int(max_experiment_k))
    state.config["execution_max_grover_power"] = int(direct_cap)
    state.config["max_grover_power_direct"] = int(direct_cap)
    state.config["calibration_status"] = calibration_summary.get("calibration_status")
    state.persist()

    if hasattr(sampler, "max_grover_power"):
        sampler.max_grover_power = int(direct_cap)

    if not args.skip_direct:
        baseline_for_solver = float(state.calibration_summary["contrast_baseline"])
        _run_direct_live(
            state,
            sampler,
            bundle,
            args,
            algorithms=algorithms,
            t_eff=t_eff,
            contrast_baseline=baseline_for_solver,
            contrast_prefactor=contrast_prefactor,
        )


def run_replay_only(args: argparse.Namespace) -> None:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode replay-only.")
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    algorithms = _parse_algorithms(args.algorithms)
    budgets = parse_int_list(args.budgets)
    state = load_existing_state(args.run_dir)
    _run_replay_from_state(
        state,
        bundle,
        args,
        algorithms=algorithms,
        budgets=budgets,
    )
    _run_plotter(args, state.paths.run_dir)


def run_reanalyze_replay(args: argparse.Namespace) -> None:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode reanalyze-replay.")
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    algorithms = _parse_algorithms(args.algorithms)
    budgets = parse_int_list(args.budgets)
    state = load_existing_state(args.run_dir)

    _reanalyze_amplification_from_state(state, bundle, args)
    if not args.skip_replay:
        _run_replay_from_state(
            state,
            bundle,
            args,
            algorithms=algorithms,
            budgets=budgets,
        )
    _run_plotter(args, state.paths.run_dir)


def run_hardware_topup(args: argparse.Namespace) -> None:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode hardware-topup.")
    if not args.scan_grover_powers:
        raise ValueError("--scan-grover-powers is required for --mode hardware-topup.")

    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    algorithms = _parse_algorithms(args.algorithms)
    budgets = parse_int_list(args.budgets)
    state = load_existing_state(args.run_dir)
    readout_params = state.calibration_summary.get("readout")
    if not isinstance(readout_params, Mapping):
        raise RuntimeError("Existing run has no readout calibration metadata.")

    backend, backend_mode = _load_backend_for_mode(args, "hardware")
    pass_manager, transpilation_metadata = build_pass_manager_for_backend(
        backend,
        bundle,
        mode="hardware",
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=parse_nonnegative_int_list(args.reference_ks),
        routing_method=args.routing_method,
        layout_search_strategy=args.layout_search_strategy,
        verbose=bool(args.verbose),
    )
    state.config["backend_mode"] = backend_mode
    state.config["topup_transpilation"] = transpilation_metadata
    state.config["topup_scan_grover_powers"] = parse_nonnegative_int_list(
        args.scan_grover_powers
    )
    state.config["topup_scan_repeats"] = int(args.scan_repeats)
    state.config["topup_scan_shots"] = int(args.scan_shots)
    state.config["topup_replace_existing_powers"] = bool(
        args.topup_replace_existing_powers
    )
    if str(args.hardware_executor) == "qctrl":
        state.config["qctrl_submission"] = _qctrl_submission_metadata(args)
    state.persist()

    global_start = time.perf_counter()
    if str(args.hardware_executor) == "qctrl":
        from qiskit_ibm_runtime import Session

        with Session(backend=backend, max_time=args.session_max_time) as session:
            session_id = getattr(session, "session_id", None)
            if not session_id:
                raise RuntimeError(
                    "Q-CTRL execution requires a Qiskit Runtime Session id, "
                    "but the opened session did not expose one."
                )
            topup_sessions = list(state.session_details.get("topup_sessions", []))
            topup_sessions.append(
                {
                    "session_id": session_id,
                    "hardware_executor": "qctrl",
                    "instance_name": str(args.instance_name),
                    "backend_name": str(args.backend_name),
                    "qiskit_function_name": str(args.qiskit_function_name),
                    "qiskit_function_channel": str(args.qiskit_function_channel or ""),
                    "submitted_circuit_kind": "abstract_logical",
                    "qctrl_transpilation_policy": "fire_opal_managed",
                    "runtime_session_strategy": "existing_qiskit_runtime_session",
                    "amplification_scan_submission": "batched_pubs_single_job",
                    "session_started_at_epoch": time.time(),
                    "scan_grover_powers": parse_nonnegative_int_list(args.scan_grover_powers),
                    "scan_repeats": int(args.scan_repeats),
                    "scan_shots": int(args.scan_shots),
                    "replace_existing_powers": bool(args.topup_replace_existing_powers),
                }
            )
            state.session_details["topup_sessions"] = topup_sessions
            state.persist()
            sampler = QctrlPerformanceManagementSampler(
                instance_name=str(args.instance_name),
                backend_name=str(args.backend_name),
                pass_manager=pass_manager,
                job_rows=state.job_rows,
                soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                max_grover_power=max(parse_nonnegative_int_list(args.scan_grover_powers)),
                max_calls_by_context={},
                start_time=global_start,
                function_name=str(args.qiskit_function_name),
                catalog_channel=getattr(args, "qiskit_function_channel", None),
                session_id=session_id,
                verbose=bool(args.verbose),
            )
            new_rows = run_amplification_scan(
                sampler,
                bundle,
                grover_powers=parse_nonnegative_int_list(args.scan_grover_powers),
                repeats=int(args.scan_repeats),
                shots=int(args.scan_shots),
                seed=int(args.seed),
                batch_circuits=True,
                verbose=bool(args.verbose),
            )
            _merge_topup_amplification_rows(state, new_rows, args)
            contrast_baseline = _resolve_noise_floor(
                state.config.get("noise_floor", args.noise_floor),
                bundle,
            )
            points, calibration_summary, _ = analyze_amplification(
                state.amplification_count_rows,
                bundle,
                readout_params,
                min_ideal_offset=float(args.min_ideal_offset),
                contrast_baseline=contrast_baseline,
                min_fit_contrast_z=float(args.min_fit_contrast_z),
                min_visible_contrast_z=float(args.min_visible_contrast_z),
                min_baseline_fit_points=int(args.min_baseline_fit_points),
                allow_negative_contrast_fit_points=bool(
                    args.allow_negative_contrast_fit_points
                ),
            )
            state.amplification_point_rows = points
            state.calibration_summary.update(calibration_summary)
            state.session_details["topup_sessions"][-1]["session_finished_at_epoch"] = time.time()
            state.persist()
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit_ibm_runtime import Session

        with Session(backend=backend, max_time=args.session_max_time) as session:
            topup_sessions = list(state.session_details.get("topup_sessions", []))
            topup_sessions.append(
                {
                    "session_id": getattr(session, "session_id", None),
                    "hardware_executor": "runtime",
                    "session_started_at_epoch": time.time(),
                    "scan_grover_powers": parse_nonnegative_int_list(args.scan_grover_powers),
                    "scan_repeats": int(args.scan_repeats),
                    "scan_shots": int(args.scan_shots),
                    "replace_existing_powers": bool(args.topup_replace_existing_powers),
                }
            )
            state.session_details["topup_sessions"] = topup_sessions
            runtime_sampler = Sampler(mode=session)
            sampler = RuntimeCountSampler(
                backend,
                runtime_sampler,
                pass_manager,
                state.job_rows,
                soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                max_grover_power=max(parse_nonnegative_int_list(args.scan_grover_powers)),
                max_calls_by_context={},
                start_time=global_start,
            )
            new_rows = run_amplification_scan(
                sampler,
                bundle,
                grover_powers=parse_nonnegative_int_list(args.scan_grover_powers),
                repeats=int(args.scan_repeats),
                shots=int(args.scan_shots),
                seed=int(args.seed),
                verbose=bool(args.verbose),
            )
            _merge_topup_amplification_rows(state, new_rows, args)
            contrast_baseline = _resolve_noise_floor(
                state.config.get("noise_floor", args.noise_floor),
                bundle,
            )
            points, calibration_summary, _ = analyze_amplification(
                state.amplification_count_rows,
                bundle,
                readout_params,
                min_ideal_offset=float(args.min_ideal_offset),
                contrast_baseline=contrast_baseline,
                min_fit_contrast_z=float(args.min_fit_contrast_z),
                min_visible_contrast_z=float(args.min_visible_contrast_z),
                min_baseline_fit_points=int(args.min_baseline_fit_points),
                allow_negative_contrast_fit_points=bool(
                    args.allow_negative_contrast_fit_points
                ),
            )
            state.amplification_point_rows = points
            state.calibration_summary.update(calibration_summary)
            state.session_details["topup_sessions"][-1]["session_finished_at_epoch"] = time.time()
            state.persist()

    if not args.skip_replay:
        _run_replay_from_state(
            state,
            bundle,
            args,
            algorithms=algorithms,
            budgets=budgets,
        )
    _run_plotter(args, state.paths.run_dir)


def _merge_topup_amplification_rows(
    state: ExperimentState,
    new_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not bool(args.topup_replace_existing_powers):
        state.amplification_count_rows.extend(dict(row) for row in new_rows)
        return

    requested_powers = set(parse_nonnegative_int_list(args.scan_grover_powers))
    retained_rows: list[dict[str, Any]] = []
    replaced_rows: list[dict[str, Any]] = []
    for row in state.amplification_count_rows:
        destination = (
            replaced_rows
            if int(float(row["grover_power"])) in requested_powers
            else retained_rows
        )
        destination.append(dict(row))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = (
        state.paths.run_dir / f"amplification_counts_replaced_{timestamp}.csv"
    )
    save_csv(replaced_rows, backup_path)
    replacements = list(state.config.get("topup_replacements", []))
    replacements.append(
        {
            "replaced_at_epoch": time.time(),
            "grover_powers": sorted(requested_powers),
            "replaced_rows": len(replaced_rows),
            "replacement_rows": len(new_rows),
            "backup_csv": backup_path.name,
        }
    )
    state.config["topup_replacements"] = replacements
    state.amplification_count_rows = retained_rows + [
        dict(row) for row in new_rows
    ]


def _job_id(job: Any) -> str:
    value = getattr(job, "job_id", None)
    if callable(value):
        return str(value())
    if value is not None:
        return str(value)
    return str(job)


def _job_created_sort_key(job: Any) -> float:
    for name in ("creation_date", "created_at"):
        value = getattr(job, name, None)
        if callable(value):
            value = value()
        if isinstance(value, datetime):
            return float(value.timestamp())
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def _scan_schedule(grover_powers: Sequence[int], repeats: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    schedule = [(int(k), int(rep)) for k in grover_powers for rep in range(int(repeats))]
    rng.shuffle(schedule)
    return [(int(k), int(rep)) for k, rep in schedule]


def _readout_params_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_label = {str(row.get("prepared_label")): row for row in rows}
    if "bad" not in by_label or "good" not in by_label:
        raise RuntimeError("Existing run has no usable readout rows.")
    p_bad = float(by_label["bad"]["p_observed_good"])
    p_good = float(by_label["good"]["p_observed_good"])
    denom = p_good - p_bad
    return {
        "p_obs_good_given_bad": float(p_bad),
        "p_obs_good_given_good": float(p_good),
        "readout_denom": float(denom),
        "readout_usable": float(abs(denom) > 0.05),
    }


def _qctrl_catalog_from_args(args: argparse.Namespace) -> Any:
    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    if args.qiskit_function_channel:
        return QiskitFunctionsCatalog(channel=str(args.qiskit_function_channel))
    return QiskitFunctionsCatalog(name=str(args.instance_name))


def _associated_job_ids(job: Any, method_name: str) -> tuple[str, ...]:
    method = getattr(job, method_name, None)
    if not callable(method):
        return ()
    return tuple(str(item) for item in method())


def _find_qctrl_job(
    catalog: Any,
    *,
    qctrl_job_id: str | None,
    session_id: str | None,
) -> Any:
    if qctrl_job_id:
        return catalog.get_job_by_id(str(qctrl_job_id))
    if not session_id:
        raise RuntimeError(
            "No Q-CTRL Function job id or Runtime session id is available. "
            "Use --recover-qctrl-job-id or --recover-session-id."
        )

    jobs = list(catalog.jobs())
    matches = [
        job
        for job in jobs
        if str(session_id) in _associated_job_ids(job, "runtime_sessions")
    ]
    if len(matches) != 1:
        candidate_ids = [_job_id(job) for job in jobs[:10]]
        raise RuntimeError(
            "Expected exactly one Q-CTRL Function job associated with Runtime "
            f"session {session_id}, found {len(matches)}. "
            "Use --recover-qctrl-job-id with the Function job id. "
            f"Recent Function job ids: {candidate_ids}"
        )
    return matches[0]


def _last_qctrl_topup_session_id(state: ExperimentState) -> str | None:
    sessions = list(state.session_details.get("topup_sessions", []))
    for details in reversed(sessions):
        if str(details.get("hardware_executor")) == "qctrl" and details.get("session_id"):
            return str(details["session_id"])
    return None


def _run_recover_qctrl_topup(
    state: ExperimentState,
    bundle: Any,
    args: argparse.Namespace,
) -> None:
    session_id = str(args.recover_session_id) if args.recover_session_id else None
    if session_id is None and not args.recover_qctrl_job_id:
        session_id = _last_qctrl_topup_session_id(state)
    catalog = _qctrl_catalog_from_args(args)
    job = _find_qctrl_job(
        catalog,
        qctrl_job_id=args.recover_qctrl_job_id,
        session_id=session_id,
    )
    job_id = _job_id(job)
    if args.verbose:
        print(
            f"[recover-session] retrieving Q-CTRL Function job={job_id} "
            f"runtime_session={session_id or '<direct-job-id>'}",
            flush=True,
        )

    schedule = _scan_schedule(
        parse_nonnegative_int_list(args.scan_grover_powers),
        int(args.scan_repeats),
        int(args.seed),
    )
    result = job.result()
    if len(result) != len(schedule):
        raise RuntimeError(
            f"Q-CTRL Function job {job_id} returned {len(result)} PUB results, "
            f"but the requested recovery schedule expects {len(schedule)}. "
            "Check --scan-grover-powers, --scan-repeats, and --seed."
        )

    rows: list[dict[str, Any]] = []
    for batch_index, (k, repeat_index) in enumerate(schedule):
        counts = extract_result_counts(result, batch_index)
        good = count_good_from_counts(counts, bundle)
        total = int(sum(counts.values()))
        rows.append(
            {
                "batch_index": int(batch_index),
                "repeat_index": int(repeat_index),
                "grover_power": int(k),
                "amplification_factor": int(2 * int(k) + 1),
                "shots": total,
                "good_counts": int(good),
                "bad_counts": int(total - good),
                "p_raw": float(good / max(total, 1)),
                "counts_json": json.dumps(counts, sort_keys=True),
                "recovered_job_id": job_id,
            }
        )

    _merge_topup_amplification_rows(state, rows, args)
    existing_job_ids = {
        str(row.get("job_id")) for row in state.job_rows if row.get("job_id")
    }
    if job_id not in existing_job_ids:
        state.job_rows.append(
            {
                "backend_mode": "qctrl_performance_management_recovered",
                "context": "amplification_scan",
                "sampler_call_index": 0,
                "n_circuits": len(rows),
                "shots": int(args.scan_shots),
                "job_id": job_id,
                "submitted_at_epoch": _job_created_sort_key(job),
                "instance_name": str(args.instance_name),
                "catalog_channel": str(args.qiskit_function_channel or ""),
                "backend_name": str(args.backend_name),
                "session_id": str(session_id or ""),
                "qiskit_function": str(args.qiskit_function_name),
                "primitive": "sampler",
                "submitted_circuit_kind": "abstract_logical",
                "qctrl_transpilation_policy": "fire_opal_managed",
            }
        )
    state.config["recovered_qctrl_job_id"] = job_id
    state.config["recovered_session_id"] = str(session_id or "")
    state.config["recovered_scan_grover_powers"] = parse_nonnegative_int_list(
        args.scan_grover_powers
    )
    state.config["recovered_scan_repeats"] = int(args.scan_repeats)
    state.config["recovered_scan_shots"] = int(args.scan_shots)
    state.persist()
    _reanalyze_amplification_from_state(state, bundle, args)

    if not args.skip_replay:
        _run_replay_from_state(
            state,
            bundle,
            args,
            algorithms=_parse_algorithms(args.algorithms),
            budgets=parse_int_list(args.budgets),
        )
    _run_plotter(args, state.paths.run_dir)


def run_recover_session(args: argparse.Namespace) -> None:
    if not args.run_dir:
        raise ValueError("--run-dir is required for --mode recover-session.")
    if not args.scan_grover_powers:
        raise ValueError("--scan-grover-powers is required for --mode recover-session.")

    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    state = load_existing_state(args.run_dir)
    if str(state.config.get("hardware_executor", "runtime")) == "qctrl":
        _run_recover_qctrl_topup(state, bundle, args)
        return

    session_id = args.recover_session_id or state.session_details.get("session_id")
    if not session_id:
        raise RuntimeError(
            "No session id provided. Use --recover-session-id or a run_dir with "
            "session_details.json."
        )

    service = _runtime_service_from_args(args)
    jobs = service.jobs(
        backend_name=str(args.backend),
        session_id=str(session_id),
        limit=1000,
        descending=False,
    )
    jobs = sorted(jobs, key=_job_created_sort_key)
    if args.verbose:
        print(
            f"[recover-session] found {len(jobs)} jobs for session {session_id}",
            flush=True,
        )

    known_readout_ids = {
        str(row.get("job_id"))
        for row in state.job_rows
        if str(row.get("context")) == "readout_calibration" and row.get("job_id")
    }
    scan_jobs = [job for job in jobs if _job_id(job) not in known_readout_ids]
    schedule = _scan_schedule(
        parse_nonnegative_int_list(args.scan_grover_powers),
        int(args.scan_repeats),
        int(args.seed),
    )
    if len(scan_jobs) < len(schedule):
        raise RuntimeError(
            "Not enough recoverable scan jobs for the requested schedule: "
            f"found {len(scan_jobs)}, expected {len(schedule)}. "
            "Check --scan-grover-powers, --scan-repeats, --seed, and session id."
        )
    if len(scan_jobs) > len(schedule) and args.verbose:
        print(
            f"[recover-session] found {len(scan_jobs)} non-readout jobs; "
            f"using the first {len(schedule)} for the amplification scan",
            flush=True,
        )
    scan_jobs = scan_jobs[: len(schedule)]

    rows: list[dict[str, Any]] = []
    existing_job_ids = {str(row.get("job_id")) for row in state.job_rows if row.get("job_id")}
    recovered_job_rows: list[dict[str, Any]] = []
    for batch_index, (job, (k, repeat_index)) in enumerate(zip(scan_jobs, schedule)):
        job_id = _job_id(job)
        if args.verbose:
            print(
                f"[recover-session] batch {batch_index + 1}/{len(schedule)} "
                f"job={job_id} k={k} repeat={repeat_index}",
                flush=True,
            )
        result = job.result()
        counts = extract_result_counts(result, 0)
        good = count_good_from_counts(counts, bundle)
        total = int(sum(counts.values()))
        rows.append(
            {
                "batch_index": int(batch_index),
                "repeat_index": int(repeat_index),
                "grover_power": int(k),
                "amplification_factor": int(2 * int(k) + 1),
                "shots": total,
                "good_counts": int(good),
                "bad_counts": int(total - good),
                "p_raw": float(good / max(total, 1)),
                "counts_json": json.dumps(counts, sort_keys=True),
                "recovered_job_id": job_id,
            }
        )
        if job_id not in existing_job_ids:
            recovered_job_rows.append(
                {
                    "backend_mode": "runtime_recovered",
                    "context": "amplification_scan",
                    "sampler_call_index": int(batch_index),
                    "n_circuits": 1,
                    "shots": int(args.scan_shots),
                    "job_id": job_id,
                    "submitted_at_epoch": _job_created_sort_key(job),
                }
            )

    state.amplification_count_rows = rows
    state.job_rows.extend(recovered_job_rows)
    state.config["recovered_session_id"] = str(session_id)
    state.config["recovered_scan_grover_powers"] = parse_nonnegative_int_list(
        args.scan_grover_powers
    )
    state.config["recovered_scan_repeats"] = int(args.scan_repeats)
    state.config["recovered_scan_shots"] = int(args.scan_shots)
    state.persist()

    readout_params = state.calibration_summary.get("readout")
    if not isinstance(readout_params, Mapping):
        readout_params = _readout_params_from_rows(state.readout_rows)
        state.calibration_summary["readout"] = readout_params

    contrast_baseline = _resolve_noise_floor(args.noise_floor, bundle)
    try:
        points, calibration_summary, _ = analyze_amplification(
            state.amplification_count_rows,
            bundle,
            readout_params,
            min_ideal_offset=float(args.min_ideal_offset),
            contrast_baseline=contrast_baseline,
            min_fit_contrast_z=float(args.min_fit_contrast_z),
            min_visible_contrast_z=float(args.min_visible_contrast_z),
            min_baseline_fit_points=int(args.min_baseline_fit_points),
            allow_negative_contrast_fit_points=bool(
                args.allow_negative_contrast_fit_points
            ),
        )
    except Exception as exc:
        state.error_rows.append(
            {
                "phase": "recover_session_analysis",
                "algorithm": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_epoch": time.time(),
            }
        )
        state.calibration_summary["calibration_status"] = "analysis_failed"
        state.calibration_summary["analysis_error_type"] = type(exc).__name__
        state.calibration_summary["analysis_error"] = str(exc)
        state.persist()
        raise
    state.amplification_point_rows = points
    state.calibration_summary.update(calibration_summary)
    state.persist()

    if not args.skip_replay:
        _run_replay_from_state(
            state,
            bundle,
            args,
            algorithms=_parse_algorithms(args.algorithms),
            budgets=parse_int_list(args.budgets),
        )
    _run_plotter(args, state.paths.run_dir)


def run_experiment(args: argparse.Namespace) -> None:
    # Main fresh-run entrypoint for preflight, dry-run, and live hardware.
    # Session recovery and top-up modes have separate functions because they
    # start from already persisted artifacts.
    config = load_config(
        config=args.config,
        config_path=args.config_path,
        config_attr=args.config_attr,
    )
    _resolve_backend_defaults(args, config)
    bundle = _build_problem_bundle(args, config)
    algorithms = _parse_algorithms(args.algorithms)
    budgets = parse_int_list(args.budgets)
    contrast_baseline = _resolve_noise_floor(args.noise_floor, bundle)

    run_dir = _create_run_dir(args)
    paths = RunPaths(run_dir)
    state = ExperimentState(
        paths=paths,
        config=_base_config(
            args=args,
            bundle=bundle,
            run_dir=run_dir,
            mode=str(args.mode),
            algorithms=algorithms,
            budgets=budgets,
            contrast_baseline=contrast_baseline,
        ),
    )
    state.session_details = {"mode": str(args.mode), "created_at_epoch": time.time()}

    backend, backend_mode = _load_backend_for_mode(args, str(args.mode))
    state.config["backend_mode"] = backend_mode
    snapshot = backend_snapshot(backend, mode=backend_mode, channel=str(args.channel))
    snapshot["use_fractional_gates_requested"] = bool(args.use_fractional_gates)
    snapshot["use_fractional_gates_applied"] = bool(
        backend_mode == "ibm_runtime_backend" and args.use_fractional_gates
    )
    snapshot["hardware_executor"] = str(args.hardware_executor)
    snapshot["instance_name"] = str(args.instance_name)
    snapshot["runtime_service_name"] = str(args.instance_name)
    snapshot["backend_name_requested"] = str(args.backend_name)
    snapshot["qiskit_function_name"] = str(args.qiskit_function_name)
    snapshot["qiskit_function_channel"] = str(args.qiskit_function_channel or "")
    if str(args.hardware_executor) == "qctrl":
        snapshot["qctrl_submission_strategy"] = "abstract_logical_circuits"
        snapshot["qctrl_transpilation_policy"] = "fire_opal_managed"
        snapshot["local_preflight_role"] = "diagnostic_and_k_cap_only"
    save_json(snapshot, paths.backend_snapshot)

    pass_manager, transpilation_metadata = build_pass_manager_for_backend(
        backend,
        bundle,
        mode=str(args.mode),
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler),
        reference_ks=parse_nonnegative_int_list(args.reference_ks),
        routing_method=args.routing_method,
        layout_search_strategy=args.layout_search_strategy,
        verbose=bool(args.verbose),
    )
    state.config["transpilation"] = transpilation_metadata
    if str(args.hardware_executor) == "qctrl":
        state.config["qctrl_submission"] = _qctrl_submission_metadata(args)
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
        return

    if args.mode == "dry-run":
        noise_model = build_noise_model(
            float(args.dry_run_noise_scale),
            profile=str(args.dry_run_noise_profile),
        )
        aer = AerCountSampler(
            noise_model=noise_model,
            seed=int(args.seed),
            method=str(args.aer_method),
            transpile_backend=backend,
            pass_manager=pass_manager,
        )
        sampler = LoggedAerSampler(
            aer,
            state.job_rows,
            max_grover_power=int(max_experiment_k),
        )
        _execute_non_replay_phases(
            args,
            state,
            sampler,
            bundle,
            max_experiment_k,
            algorithms=algorithms,
            contrast_baseline=contrast_baseline,
        )
    elif args.mode == "hardware":
        global_start = time.perf_counter()
        if str(args.hardware_executor) == "qctrl":
            from qiskit_ibm_runtime import Session

            with Session(backend=backend, max_time=args.session_max_time) as session:
                session_id = getattr(session, "session_id", None)
                if not session_id:
                    raise RuntimeError(
                        "Q-CTRL execution requires a Qiskit Runtime Session id, "
                        "but the opened session did not expose one."
                    )
                state.session_details.update(
                    {
                        "session_id": session_id,
                        "hardware_executor": "qctrl",
                        "instance_name": str(args.instance_name),
                        "backend_name": str(args.backend_name),
                        "qiskit_function_name": str(args.qiskit_function_name),
                        "qiskit_function_channel": str(args.qiskit_function_channel or ""),
                        "submitted_circuit_kind": "abstract_logical",
                        "qctrl_transpilation_policy": "fire_opal_managed",
                        "runtime_session_strategy": "existing_qiskit_runtime_session",
                        "amplification_scan_submission": "batched_pubs_single_job",
                        "session_started_at_epoch": time.time(),
                    }
                )
                state.persist()
                sampler = QctrlPerformanceManagementSampler(
                    instance_name=str(args.instance_name),
                    backend_name=str(args.backend_name),
                    pass_manager=pass_manager,
                    job_rows=state.job_rows,
                    soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                    max_grover_power=int(max_experiment_k),
                    max_calls_by_context={
                        f"direct_live_{algorithm}": int(args.max_direct_calls)
                        for algorithm in algorithms
                    },
                    start_time=global_start,
                    function_name=str(args.qiskit_function_name),
                    catalog_channel=getattr(args, "qiskit_function_channel", None),
                    session_id=session_id,
                    verbose=bool(args.verbose),
                )
                _execute_non_replay_phases(
                    args,
                    state,
                    sampler,
                    bundle,
                    max_experiment_k,
                    algorithms=algorithms,
                    contrast_baseline=contrast_baseline,
                    batch_scan_circuits=True,
                )
                state.session_details["session_finished_at_epoch"] = time.time()
                state.persist()
        else:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit_ibm_runtime import Session

            with Session(backend=backend, max_time=args.session_max_time) as session:
                state.session_details.update(
                    {
                        "session_id": getattr(session, "session_id", None),
                        "hardware_executor": "runtime",
                        "session_started_at_epoch": time.time(),
                    }
                )
                runtime_sampler = Sampler(mode=session)
                sampler = RuntimeCountSampler(
                    backend,
                    runtime_sampler,
                    pass_manager,
                    state.job_rows,
                    soft_wallclock_limit_seconds=float(args.soft_wallclock_limit),
                    max_grover_power=int(max_experiment_k),
                    max_calls_by_context={
                        f"direct_live_{algorithm}": int(args.max_direct_calls)
                        for algorithm in algorithms
                    },
                    start_time=global_start,
                )
                _execute_non_replay_phases(
                    args,
                    state,
                    sampler,
                    bundle,
                    max_experiment_k,
                    algorithms=algorithms,
                    contrast_baseline=contrast_baseline,
                )
                state.session_details["session_finished_at_epoch"] = time.time()
                state.persist()
    else:
        raise ValueError(f"Unsupported mode for run_experiment: {args.mode!r}")

    if not args.skip_replay:
        _run_replay_from_state(
            state,
            bundle,
            args,
            algorithms=algorithms,
            budgets=budgets,
        )
    _run_plotter(args, paths.run_dir)


class HardwareCvaExperimentRunner:
    """CLI-facing runner for the 6q CVA hardware amplitude-estimation experiment."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "HardwareCvaExperimentRunner":
        args: argparse.Namespace = build_arg_parser().parse_args(argv)
        return cls(args)

    def run(self) -> None:
        # Dispatch is intentionally small: all mode-specific behaviour remains
        # in named functions above so each operational workflow is searchable.
        args: argparse.Namespace = self.args
        if args.mode == "replay-only":
            run_replay_only(args)
        elif args.mode == "hardware-topup":
            run_hardware_topup(args)
        elif args.mode == "recover-session":
            run_recover_session(args)
        elif args.mode == "reanalyze-replay":
            run_reanalyze_replay(args)
        else:
            run_experiment(args)


def run_cli(argv: list[str] | None = None) -> None:
    runner: HardwareCvaExperimentRunner = HardwareCvaExperimentRunner.from_cli(argv)
    runner.run()


def main(argv: list[str] | None = None) -> None:
    run_cli(argv)


if __name__ == "__main__":
    main()
