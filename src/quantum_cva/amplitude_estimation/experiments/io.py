from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def save_json(obj: Mapping[str, Any], path: str | Path, *, fsync: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(obj)), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())


def load_json(path: str | Path, default: Any = None) -> Any:
    input_path = Path(path)
    if not input_path.exists():
        return default
    return json.loads(input_path.read_text(encoding="utf-8"))


def save_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    fieldnames: Iterable[str] | None = None,
    fsync: bool = False,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    field_list = list(fieldnames)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_list, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in field_list})
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class RunPaths:
    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)

    @property
    def config(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def backend_snapshot(self) -> Path:
        return self.run_dir / "backend_snapshot.json"

    @property
    def session_details(self) -> Path:
        return self.run_dir / "session_details.json"

    @property
    def transpilation_report(self) -> Path:
        return self.run_dir / "transpilation_report.csv"

    @property
    def amplification_counts(self) -> Path:
        return self.run_dir / "amplification_counts.csv"

    @property
    def amplification_points(self) -> Path:
        return self.run_dir / "amplification_points.csv"

    @property
    def readout_calibration(self) -> Path:
        return self.run_dir / "readout_calibration.csv"

    @property
    def calibration_summary(self) -> Path:
        return self.run_dir / "calibration_summary.json"

    @property
    def direct_trace(self) -> Path:
        return self.run_dir / "direct_trace_rows.csv"

    @property
    def direct_final(self) -> Path:
        return self.run_dir / "direct_final_rows.csv"

    @property
    def replay_trace(self) -> Path:
        return self.run_dir / "replay_trace_rows.csv"

    @property
    def replay_final(self) -> Path:
        return self.run_dir / "replay_final_rows.csv"

    @property
    def replay_budget(self) -> Path:
        return self.run_dir / "replay_budget_rows.csv"

    @property
    def budget_summary(self) -> Path:
        return self.run_dir / "budget_summary.csv"

    @property
    def runtime_jobs(self) -> Path:
        return self.run_dir / "runtime_jobs.csv"

    @property
    def errors(self) -> Path:
        return self.run_dir / "errors.csv"

    @property
    def trace_bundle(self) -> Path:
        return self.run_dir / "trace_bundle.npz"

    @property
    def qasm_dir(self) -> Path:
        return self.run_dir / "qasm3_isa"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    def manifest_payload(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "config_json": str(self.config),
            "backend_snapshot_json": str(self.backend_snapshot),
            "session_details_json": str(self.session_details),
            "transpilation_report_csv": str(self.transpilation_report),
            "readout_calibration_csv": str(self.readout_calibration),
            "amplification_counts_csv": str(self.amplification_counts),
            "amplification_points_csv": str(self.amplification_points),
            "calibration_summary_json": str(self.calibration_summary),
            "direct_trace_rows_csv": str(self.direct_trace),
            "direct_final_rows_csv": str(self.direct_final),
            "replay_trace_rows_csv": str(self.replay_trace),
            "replay_final_rows_csv": str(self.replay_final),
            "replay_budget_rows_csv": str(self.replay_budget),
            "budget_summary_csv": str(self.budget_summary),
            "runtime_jobs_csv": str(self.runtime_jobs),
            "errors_csv": str(self.errors),
            "trace_bundle_npz": str(self.trace_bundle),
            "qasm3_isa_dir": str(self.qasm_dir),
            "plots_dir": str(self.plots_dir),
        }

    def write_manifest(self) -> None:
        save_json(self.manifest_payload(), self.manifest)


def write_trace_bundle(
    path: str | Path,
    *,
    trace_rows: Sequence[Mapping[str, Any]] = (),
    budget_rows: Sequence[Mapping[str, Any]] = (),
    amplification_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    payload: dict[str, np.ndarray] = {}
    if trace_rows:
        payload["trace_query_budget"] = np.asarray(
            [float(r.get("query_budget", r.get("query_budget_actual", np.nan))) for r in trace_rows],
            dtype=float,
        )
        payload["trace_estimate"] = np.asarray(
            [float(r.get("estimate", np.nan)) for r in trace_rows],
            dtype=float,
        )
    if budget_rows:
        payload["budget_query_budget"] = np.asarray(
            [float(r.get("budget", np.nan)) for r in budget_rows],
            dtype=float,
        )
        payload["budget_estimate"] = np.asarray(
            [float(r.get("estimate", np.nan)) for r in budget_rows],
            dtype=float,
        )
    if amplification_rows:
        payload["amplification_grover_power"] = np.asarray(
            [float(r.get("grover_power", np.nan)) for r in amplification_rows],
            dtype=float,
        )
        payload["amplification_p_observed"] = np.asarray(
            [
                float(
                    r.get(
                        "p_hw_mitigated",
                        r.get("p_observed_mitigated", r.get("p_raw", np.nan)),
                    )
                )
                for r in amplification_rows
            ],
            dtype=float,
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if payload:
        np.savez(output, **payload)
    else:
        np.savez(output, empty=np.asarray([], dtype=float))
