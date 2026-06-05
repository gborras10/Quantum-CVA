from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


def parse_name_csv(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = raw.replace(";", ",").split(",")
    else:
        values = raw
    parsed = tuple(str(value).strip() for value in values if str(value).strip())
    if not parsed:
        raise ValueError("At least one name is required.")
    return parsed


def parse_int_csv(raw: str | Sequence[int | str]) -> tuple[int, ...]:
    values = tuple(int(value) for value in parse_name_csv(raw))
    if any(value <= 0 for value in values):
        raise ValueError("Integer values must be positive.")
    return values


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AlgorithmRunConfig:
    algorithms: tuple[str, ...]
    epsilon_target: float
    alpha: float
    seed: int = 12345

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class IdealExperimentConfig:
    run_dir: Path
    algorithm: AlgorithmRunConfig
    repetitions: int = 20
    n_shots: int = 10
    max_queries: int = 10_000
    budgets: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class NoiseSimulationConfig:
    run_dir: Path
    algorithm: AlgorithmRunConfig
    budgets: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    max_grover_power: int = 8
    scan_repeats: int = 1
    scan_shots: int = 256
    readout_shots: int = 512
    direct_shots: int = 64
    replay_repetitions: int = 20
    noise_scale: float = 1.0
    noise_profile: str = "projected"
    contrast_baseline: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class HardwareReplayConfig:
    run_dir: Path
    algorithm: AlgorithmRunConfig
    mode: Literal["dry-run", "replay-only"] = "dry-run"
    budgets: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    replay_repetitions: int = 100
    direct_shots: int = 64
    max_grover_power: int = 8
    contrast_baseline: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class PlotConfig:
    run_dir: Path
    prefix: str = "ae"
    include_final_scatter: bool = True

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))
