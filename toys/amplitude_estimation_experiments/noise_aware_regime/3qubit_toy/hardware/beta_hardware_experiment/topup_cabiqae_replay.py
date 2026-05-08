from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hardware_bae_biqae_cabiqae_core import (
    ExperimentState,
    ReplayCountSampler,
    RunPaths,
    build_large_problem,
    effective_t_for_algorithms,
    load_csv,
    load_replay_probabilities_from_counts,
    make_replay_probability_extrapolator,
    parse_int_list,
    print_compact_trace,
    run_algorithm_once,
    rows_at_budgets,
    sample_replay_probabilities,
    save_csv,
    save_json,
)


BETA_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = BETA_DIR / "experiment_results" / "csv_results"
ALGORITHM = "cabiqae_latentt"
ALGORITHM_LABELS = {ALGORITHM: "CABIQAE"}
DEFAULT_BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fieldnames(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                out.append(str(key))
    return out


def _max_repetition(rows: Iterable[Mapping[str, Any]], algorithm: str) -> int:
    max_rep = -1
    for row in rows:
        key = str(row.get("algorithm_key") or "")
        label = str(row.get("algorithm") or "")
        if key != algorithm and label != ALGORITHM_LABELS[algorithm]:
            continue
        try:
            max_rep = max(max_rep, int(float(row.get("repetition", -1))))
        except (TypeError, ValueError):
            continue
    return max_rep


def _backup_existing(paths: RunPaths) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = paths.run_dir / f"backup_before_cabiqae_replay_topup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        paths.config,
        paths.replay_trace,
        paths.replay_final,
        paths.replay_budget,
        paths.budget_summary,
        paths.errors,
    ):
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def _load_state(paths: RunPaths) -> ExperimentState:
    config = _read_json(paths.config)
    state = ExperimentState(paths=paths, config=config)
    state.error_rows = load_csv(paths.errors) if paths.errors.exists() else []
    state.replay_trace_rows = load_csv(paths.replay_trace) if paths.replay_trace.exists() else []
    state.replay_final_rows = load_csv(paths.replay_final) if paths.replay_final.exists() else []
    state.replay_budget_rows = load_csv(paths.replay_budget) if paths.replay_budget.exists() else []
    state.amplification_count_rows = (
        load_csv(paths.amplification_counts) if paths.amplification_counts.exists() else []
    )
    state.amplification_point_rows = (
        load_csv(paths.amplification_points) if paths.amplification_points.exists() else []
    )
    state.calibration_summary = _read_json(paths.calibration_summary)
    state.session_details = _read_json(paths.session_details)
    return state


def _load_probabilities(paths: RunPaths) -> tuple[dict[int, float], dict[int, float] | None]:
    if paths.amplification_points.exists():
        points = load_csv(paths.amplification_points)
        p_by_k = {int(float(r["grover_power"])): float(r["p_hw_mitigated"]) for r in points}
        p_se_by_k = {
            int(float(r["grover_power"])): float(r["p_hw_mitigated_se"])
            for r in points
            if str(r.get("p_hw_mitigated_se", "")).strip() != ""
        }
        return p_by_k, p_se_by_k
    return load_replay_probabilities_from_counts(paths.amplification_counts), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append extra hardware-replay samples for CABIQAE only."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--extra-repetitions", type=int, default=1000)
    parser.add_argument("--budgets", default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--direct-shots", type=int, default=None)
    parser.add_argument("--epsilon-target", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--replay-probability-mode",
        choices=("fixed", "normal"),
        default=None,
    )
    parser.add_argument("--replay-probability-se-scale", type=float, default=None)
    parser.add_argument("--persist-every", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    extrapolate = parser.add_mutually_exclusive_group()
    extrapolate.add_argument("--extrapolate", dest="extrapolate", action="store_true")
    extrapolate.add_argument("--no-extrapolate", dest="extrapolate", action="store_false")
    parser.set_defaults(extrapolate=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    paths = RunPaths(run_dir)
    state = _load_state(paths)
    config = state.config
    if not config:
        raise FileNotFoundError(f"Missing config.json in {run_dir}")

    backup_dir = _backup_existing(paths)
    objective_ry_offset = float(config.get("objective_ry_offset", 0.0))
    problem, fallback_a_true = build_large_problem(objective_ry_offset)
    a_true = float(config.get("a_true", fallback_a_true))
    p_by_k, p_se_by_k = _load_probabilities(paths)
    budgets = parse_int_list(args.budgets)
    max_queries = max(int(x) for x in budgets)
    seed = int(args.seed if args.seed is not None else config.get("seed", 12345))
    n_shots = int(args.direct_shots if args.direct_shots is not None else config.get("direct_shots", 256))
    epsilon_target = float(
        args.epsilon_target if args.epsilon_target is not None else config.get("epsilon_target", 0.08)
    )
    replay_probability_mode = str(
        args.replay_probability_mode
        if args.replay_probability_mode is not None
        else config.get("replay_probability_mode", "normal")
    )
    replay_probability_se_scale = float(
        args.replay_probability_se_scale
        if args.replay_probability_se_scale is not None
        else config.get("replay_probability_se_scale", 1.0)
    )
    extrapolate = bool(
        args.extrapolate if args.extrapolate is not None else config.get("replay_extrapolate", False)
    )
    t_eff = effective_t_for_algorithms(state.calibration_summary)
    extrapolate_probability = None
    extrapolated_cache: dict[int, float] = {}
    extrapolation_metadata: dict[str, Any] = {}
    if extrapolate:
        extrapolate_probability, extrapolation_metadata = make_replay_probability_extrapolator(
            a_true=a_true,
            problem=problem,
            calibration_summary=state.calibration_summary,
        )

    start_rep = _max_repetition(state.replay_trace_rows, ALGORITHM) + 1
    existing_budget_rows = paths.replay_budget.exists()
    new_trace_rows: list[dict[str, Any]] = []
    new_final_rows: list[dict[str, Any]] = []
    new_budget_rows: list[dict[str, Any]] = []
    started_at = time.time()

    print(
        f"Appending {int(args.extra_repetitions)} CABIQAE replay repetitions "
        f"starting at repetition={start_rep}. Backup: {backup_dir}",
        flush=True,
    )

    for offset in range(int(args.extra_repetitions)):
        rep = start_rep + offset
        replay_rng = np.random.default_rng(seed + 7919 * rep)
        rep_p_by_k = sample_replay_probabilities(
            p_by_k,
            p_se_by_k,
            mode=replay_probability_mode,
            rng=replay_rng,
            se_scale=replay_probability_se_scale,
        )
        sampler = ReplayCountSampler(
            rep_p_by_k,
            state,
            seed=seed + 1009 * rep,
            max_calls=128,
            extrapolate_probability=extrapolate_probability,
            extrapolated_cache=extrapolated_cache,
        )
        try:
            trace_rows, final_row = run_algorithm_once(
                ALGORITHM,
                ALGORITHM_LABELS,
                sampler,
                problem,
                run_kind="hardware_replay",
                repetition=rep,
                a_true=a_true,
                objective_ry_offset=objective_ry_offset,
                n_shots=n_shots,
                epsilon_target=epsilon_target,
                alpha=float(args.alpha),
                t_eff=t_eff,
                max_queries=max_queries,
                seed=seed + rep,
                verbose=bool(args.verbose),
            )
            if extrapolate:
                used_ks = set(int(k) for k in getattr(sampler, "extrapolated_ks_used", set()))
                for row in trace_rows:
                    row["replay_probability_source"] = (
                        "extrapolated" if int(row["grover_power"]) in used_ks else "measured"
                    )
                    row["replay_probability_extrapolated"] = int(row["grover_power"]) in used_ks
                final_row["extrapolated_replay_ks_json"] = json.dumps(sorted(used_ks))
                final_row["n_extrapolated_replay_ks"] = len(used_ks)
            if args.verbose:
                print_compact_trace(
                    trace_rows,
                    final_row,
                    prefix="cabiqae_replay_topup",
                    repetition=rep,
                    total_repetitions=start_rep + int(args.extra_repetitions),
                )
            state.replay_trace_rows.extend(trace_rows)
            state.replay_final_rows.append(final_row)
            new_trace_rows.extend(trace_rows)
            new_final_rows.append(final_row)
            if existing_budget_rows:
                budget_rows = rows_at_budgets(trace_rows, budgets)
                state.replay_budget_rows.extend(budget_rows)
                new_budget_rows.extend(budget_rows)
        except Exception as exc:
            state.error_rows.append(
                {
                    "phase": "hardware_replay_topup",
                    "algorithm": ALGORITHM,
                    "repetition": rep,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "timestamp_epoch": time.time(),
                }
            )

        if int(args.persist_every) > 0 and (offset + 1) % int(args.persist_every) == 0:
            save_csv(state.replay_trace_rows, paths.replay_trace, fieldnames=_fieldnames(state.replay_trace_rows))
            save_csv(state.replay_final_rows, paths.replay_final, fieldnames=_fieldnames(state.replay_final_rows))
            save_csv(state.error_rows, paths.errors, fieldnames=_fieldnames(state.error_rows))
            if existing_budget_rows:
                save_csv(state.replay_budget_rows, paths.replay_budget, fieldnames=_fieldnames(state.replay_budget_rows))
            print(f"  persisted {offset + 1}/{int(args.extra_repetitions)} top-up reps", flush=True)

    topups = list(config.get("cabiqae_replay_topups", []))
    topups.append(
        {
            "started_at_epoch": started_at,
            "finished_at_epoch": time.time(),
            "algorithm": ALGORITHM,
            "start_repetition": int(start_rep),
            "extra_repetitions_requested": int(args.extra_repetitions),
            "extra_final_rows_added": len(new_final_rows),
            "extra_trace_rows_added": len(new_trace_rows),
            "extra_budget_rows_added": len(new_budget_rows),
            "seed": seed,
            "direct_shots": n_shots,
            "epsilon_target": epsilon_target,
            "alpha": float(args.alpha),
            "replay_probability_mode": replay_probability_mode,
            "replay_probability_se_scale": replay_probability_se_scale,
            "extrapolate": extrapolate,
            "extrapolation_model": extrapolation_metadata,
            "backup_dir": str(backup_dir),
        }
    )
    config["cabiqae_replay_topups"] = topups
    save_json(config, paths.config)
    save_csv(state.replay_trace_rows, paths.replay_trace, fieldnames=_fieldnames(state.replay_trace_rows))
    save_csv(state.replay_final_rows, paths.replay_final, fieldnames=_fieldnames(state.replay_final_rows))
    save_csv(state.error_rows, paths.errors, fieldnames=_fieldnames(state.error_rows))
    if existing_budget_rows:
        save_csv(state.replay_budget_rows, paths.replay_budget, fieldnames=_fieldnames(state.replay_budget_rows))

    print(
        f"Added {len(new_final_rows)} CABIQAE final rows and {len(new_trace_rows)} trace rows. "
        f"Files updated in {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
