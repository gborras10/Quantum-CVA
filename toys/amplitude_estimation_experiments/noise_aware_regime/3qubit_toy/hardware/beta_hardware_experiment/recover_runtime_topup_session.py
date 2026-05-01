from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from hardware_bae_biqae_cabiqae_core import (
    RunPaths,
    analyze_amplification,
    build_large_problem,
    count_ones,
    extract_result_counts,
    load_csv,
    parse_int_list,
    save_csv,
    save_json,
)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _backup_existing(run_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = run_dir / f"recovery_backup_before_session_restore_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "amplification_counts.csv",
        "amplification_points.csv",
        "calibration_summary.json",
        "config.json",
        "runtime_jobs.csv",
        "session_details.json",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    return backup_dir


def _schedule(grover_powers: list[int], repeats: int, seed: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(int(seed))
    schedule = [(int(k), int(r)) for k in grover_powers for r in range(int(repeats))]
    rng.shuffle(schedule)
    return schedule


def _job_created_epoch(job: Any) -> float:
    created_attr = getattr(job, "creation_date", None)
    created = created_attr() if callable(created_attr) else created_attr
    if created is None:
        return time.time()
    return float(created.timestamp())


def _job_status_name(job: Any) -> str:
    status_attr = getattr(job, "status", None)
    status = status_attr() if callable(status_attr) else status_attr
    return str(getattr(status, "name", status))


def recover(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    paths = RunPaths(run_dir)
    config = _load_json(paths.config, {})
    session_details = _load_json(paths.session_details, {})
    calibration_summary = _load_json(paths.calibration_summary, {})

    channel = str(args.channel or config.get("channel", "ibm_quantum_platform"))
    backend_name = str(args.backend or config.get("backend", "ibm_basquecountry"))
    grover_powers = parse_int_list(args.scan_grover_powers or config["topup_scan_grover_powers"])
    repeats = int(args.scan_repeats or config["topup_scan_repeats"])
    shots = int(args.scan_shots or config["topup_scan_shots"])
    seed = int(args.seed or config.get("seed", 12345))
    objective_ry_offset = float(args.objective_ry_offset or config.get("objective_ry_offset", -0.10))

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel=channel)
    jobs = service.jobs(
        limit=max(int(args.limit), len(grover_powers) * repeats + 10),
        backend_name=backend_name,
        session_id=str(args.session_id),
        descending=False,
    )
    jobs = sorted(jobs, key=_job_created_epoch)
    schedule = _schedule(grover_powers, repeats, seed)

    recovery_dir = run_dir / f"recovered_session_{args.session_id}"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    raw_job_rows: list[dict[str, Any]] = []
    recovered_count_rows: list[dict[str, Any]] = []
    recovered_job_rows: list[dict[str, Any]] = []
    recovered_schedule: list[dict[str, Any]] = []

    for index, job in enumerate(jobs):
        job_id = str(job.job_id())
        status = _job_status_name(job)
        created_epoch = _job_created_epoch(job)
        raw_job_rows.append(
            {
                "job_id": job_id,
                "status": status,
                "created_at_epoch": created_epoch,
                "session_id": str(getattr(job, "session_id", None) or args.session_id),
                "primitive_id": str(getattr(job, "primitive_id", None)),
            }
        )
        if index >= len(schedule):
            continue
        if status.upper() not in {"DONE", "COMPLETED"} and not job.done():
            continue
        k, repeat_index = schedule[index]
        try:
            result = job.result()
            counts = extract_result_counts(result, 0)
        except Exception as exc:
            raw_job_rows[-1]["result_error"] = repr(exc)
            continue

        one = count_ones(counts)
        total = int(sum(counts.values()))
        recovered_count_rows.append(
            {
                "recovered_from_session_id": str(args.session_id),
                "job_id": job_id,
                "batch_index": int(index),
                "repeat_index": int(repeat_index),
                "grover_power": int(k),
                "amplification_factor": int(2 * k + 1),
                "shots": int(total),
                "one_counts": int(one),
                "zero_counts": int(total - one),
                "p_hw_raw": float(one / max(total, 1)),
                "counts_json": json.dumps(counts, sort_keys=True),
            }
        )
        recovered_job_rows.append(
            {
                "backend_mode": "runtime_backend",
                "context": "amplification_scan_topup_recovered",
                "job_id": job_id,
                "n_circuits": 1,
                "sampler_call_index": int(index),
                "shots": int(shots),
                "submitted_at_epoch": created_epoch,
            }
        )
        recovered_schedule.append(
            {
                "job_id": job_id,
                "grover_power": int(k),
                "repeat_index": int(repeat_index),
            }
        )

    save_json(
        {
            "session_id": str(args.session_id),
            "backend": backend_name,
            "channel": channel,
            "expected_schedule": [
                {"grover_power": int(k), "repeat_index": int(r)} for k, r in schedule
            ],
            "jobs": raw_job_rows,
            "recovered_rows": len(recovered_count_rows),
        },
        recovery_dir / "recovery_manifest.json",
    )
    save_csv(recovered_count_rows, recovery_dir / "amplification_counts_recovered.csv")
    save_csv(recovered_job_rows, recovery_dir / "runtime_jobs_recovered.csv")

    if args.no_merge:
        print(f"Recovered {len(recovered_count_rows)} rows into {recovery_dir}")
        return

    backup_dir = _backup_existing(run_dir)
    existing_counts = load_csv(paths.amplification_counts) if paths.amplification_counts.exists() else []
    existing_jobs = load_csv(paths.runtime_jobs) if paths.runtime_jobs.exists() else []
    existing_job_ids = {str(row.get("job_id", "")) for row in existing_counts}
    merged_counts = existing_counts + [
        row for row in recovered_count_rows if str(row["job_id"]) not in existing_job_ids
    ]
    existing_runtime_job_ids = {str(row.get("job_id", "")) for row in existing_jobs}
    merged_jobs = existing_jobs + [
        row for row in recovered_job_rows if str(row["job_id"]) not in existing_runtime_job_ids
    ]

    problem, _ = build_large_problem(objective_ry_offset)
    readout = calibration_summary.get("readout", {})
    points, recovered_calibration_summary, _ = analyze_amplification(merged_counts, problem, readout)
    calibration_summary.update(recovered_calibration_summary)

    topup_sessions = list(session_details.get("topup_sessions", []))
    topup_sessions.append(
        {
            "session_id": str(args.session_id),
            "recovered_from_runtime": True,
            "recovered_at_epoch": time.time(),
            "scan_grover_powers": grover_powers,
            "scan_repeats": repeats,
            "scan_shots": shots,
            "job_ids": [row["job_id"] for row in recovered_job_rows],
            "schedule": recovered_schedule,
        }
    )
    session_details["topup_sessions"] = topup_sessions
    config["topup_recovered_from_session_id"] = str(args.session_id)
    config["topup_recovered_at_epoch"] = time.time()
    config["topup_recovery_backup_dir"] = str(backup_dir)

    save_csv(merged_counts, paths.amplification_counts)
    save_csv(points, paths.amplification_points)
    save_csv(merged_jobs, paths.runtime_jobs)
    save_json(calibration_summary, paths.calibration_summary)
    save_json(session_details, paths.session_details)
    save_json(config, paths.config)
    print(
        f"Recovered {len(recovered_count_rows)} rows from {len(jobs)} jobs. "
        f"Backup copied to {backup_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover completed IBM Runtime top-up jobs by session id.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--scan-grover-powers", default=None)
    parser.add_argument("--scan-repeats", type=int, default=None)
    parser.add_argument("--scan-shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objective-ry-offset", type=float, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--no-merge", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    recover(parse_args())
