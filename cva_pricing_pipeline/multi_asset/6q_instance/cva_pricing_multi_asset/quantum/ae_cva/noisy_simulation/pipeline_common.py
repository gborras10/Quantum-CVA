from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any


def bootstrap_paths() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    instance_dir = next(
        parent for parent in current.parents if parent.name == "6q_instance"
    )
    if str(instance_dir) not in sys.path:
        sys.path.insert(0, str(instance_dir))

    script_dir = current.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return repo_root


REPO_ROOT = bootstrap_paths()
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_RUN_DIR = CURRENT_DIR / "experiment_results"


def _resolve_from_repo(path_like: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path_like).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_object(spec: str) -> Any:
    if ":" not in str(spec):
        raise ValueError("Object specs must use 'module:attribute' syntax.")
    module_name, attr_name = str(spec).split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def load_object_from_file(path: str | pathlib.Path, attr_name: str = "CONFIG") -> Any:
    module_path = _resolve_from_repo(path)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj: Any = module
    for part in str(attr_name).split("."):
        obj = getattr(obj, part)
    return obj


def load_config(
    *,
    config: str | None = None,
    config_path: str | pathlib.Path | None = None,
    config_attr: str = "CONFIG",
) -> Any:
    if config_path is not None:
        return load_object_from_file(config_path, config_attr)
    if config is not None:
        return load_object(config)
    return load_object("full_cva_pipeline:CONFIG")


def parse_name_list(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        tokens = raw.replace(";", ",").replace(" ", ",").split(",")
    else:
        tokens = list(raw)
    values = tuple(str(token).strip() for token in tokens if str(token).strip())
    if not values:
        raise ValueError("At least one value is required.")
    return values


def parse_int_list(raw: str | Sequence[int | str]) -> list[int]:
    values = [int(token) for token in parse_name_list(raw)]
    if any(value <= 0 for value in values):
        raise ValueError("Integer list values must be positive.")
    return values


def _copy_float_alias(
    out: dict[str, Any],
    *,
    source: str,
    target: str,
) -> None:
    if source in out and target not in out:
        out[target] = out[source]


def add_cva_alias_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    _copy_float_alias(out, source="processed_true_value", target="cva_true")
    _copy_float_alias(out, source="processed_estimate", target="cva_estimate")
    _copy_float_alias(out, source="processed_abs_error", target="cva_abs_error")
    _copy_float_alias(
        out,
        source="processed_relative_error",
        target="cva_relative_error",
    )
    _copy_float_alias(out, source="processed_ci_low", target="cva_ci_low")
    _copy_float_alias(out, source="processed_ci_high", target="cva_ci_high")
    _copy_float_alias(
        out,
        source="processed_abs_error_median",
        target="cva_abs_error_median",
    )
    _copy_float_alias(
        out,
        source="processed_relative_error_median",
        target="cva_relative_error_median",
    )
    return out


def add_cva_aliases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [add_cva_alias_columns(row) for row in rows]


def preferred_field_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "run_kind",
        "simulation_regime",
        "noise_profile",
        "noise_scale",
        "noise_floor",
        "contrast_baseline",
        "contrast_baseline_mode",
        "instance_name",
        "target_name",
        "phase",
        "repetition",
        "algorithm",
        "algorithm_key",
        "step_index",
        "budget",
        "query_budget",
        "query_budget_actual",
        "query_budget_actual_median",
        "final_queries",
        "max_queries",
        "estimate",
        "final_estimate",
        "a_true",
        "abs_error",
        "final_abs_error",
        "normalized_abs_error",
        "final_normalized_abs_error",
        "normalized_abs_error_median",
        "min_abs_error",
        "min_normalized_abs_error",
        "processed_true_value",
        "processed_estimate",
        "processed_abs_error",
        "processed_relative_error",
        "processed_relative_error_median",
        "min_processed_abs_error",
        "min_processed_relative_error",
        "cva_true",
        "cva_estimate",
        "cva_abs_error",
        "cva_relative_error",
        "cva_relative_error_median",
        "ci_low",
        "ci_high",
        "processed_ci_low",
        "processed_ci_high",
        "cva_ci_low",
        "cva_ci_high",
        "coverage",
        "grover_power",
        "grover_power_exceeds_calibration",
        "k_max",
        "k_max_exceeds_calibration",
        "k_max_budget",
        "amplification_factor",
        "amplification_factor_max",
        "contrast_mitigated",
        "contrast_mitigated_se",
        "contrast_signal_z",
        "contrast_relative_se",
        "visible_by_contrast",
        "signal_z_from_baseline",
        "t_eff",
        "calibration_status",
        "runtime_wall_seconds",
        "time_to_budget_seconds",
        "circuit_prepare_wall_seconds",
        "construct_circuit_wall_seconds",
        "construct_circuit_cache_hits",
        "construct_circuit_cache_misses",
        "construct_circuit_cache_size",
        "sampler_call_index",
        "n_circuits",
        "shots",
        "n_shots",
        "direct_shots",
        "max_queries_requested",
        "max_queries_reached",
        "max_grover_power",
        "calibration_max_grover_power",
        "execution_max_grover_power",
        "execution_k_cap_enforced",
        "cabiqae_hard_k_cap",
        "cabiqae_scheduler",
        "transpile_backend_name",
        "use_fractional_gates_applied",
        "transpilation_strategy",
    ]
    available = {key for row in rows for key in row}
    seen = set()
    ordered = [key for key in preferred if key in available]
    seen.update(ordered)
    ordered.extend(sorted({key for row in rows for key in row if key not in seen}))
    return ordered
