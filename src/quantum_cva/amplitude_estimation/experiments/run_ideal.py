from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from quantum_cva.amplitude_estimation.experiments.io import RunPaths, save_csv, save_json
from quantum_cva.amplitude_estimation.experiments.runner_utils import (
    add_problem_builder_args,
    problem_bundle_from_args,
)
from quantum_cva.amplitude_estimation.experiments.samplers import ContrastDecaySampler
from quantum_cva.amplitude_estimation.experiments.solvers import ALGORITHM_LABELS, run_algorithm_once
from quantum_cva.amplitude_estimation.experiments.statistics import aggregate_budget_summary
from quantum_cva.amplitude_estimation.experiments.traces import rows_at_budgets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ideal reusable AE experiments.")
    add_problem_builder_args(parser)
    parser.add_argument("--run-dir", default="experiment_results/ideal")
    parser.add_argument("--algorithms", default="cabiqae_latentt,biqae,iqae,bae")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--epsilon-target", type=float, default=2e-3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-shots", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=10_000)
    parser.add_argument("--budgets", default="128,256,512,1024,2048,4096")
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    bundle = problem_bundle_from_args(args)
    run_dir = Path(args.run_dir)
    paths = RunPaths(run_dir)
    algorithms = tuple(x.strip() for x in args.algorithms.split(",") if x.strip())
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "mode": "ideal",
            "target_name": bundle.target_name,
            "a_true": float(bundle.true_amplitude),
            "processed_true_value": float(bundle.processed_true_value),
            "algorithms": list(algorithms),
            "repetitions": int(args.repetitions),
            "epsilon_target": float(args.epsilon_target),
            "alpha": float(args.alpha),
            "n_shots": int(args.n_shots),
            "max_queries": int(args.max_queries),
            "budgets": budgets,
            "created_at_epoch": time.time(),
            "problem_metadata": bundle.metadata,
        },
        paths.config,
    )

    trace_rows: list[dict] = []
    final_rows: list[dict] = []
    error_rows: list[dict] = []
    budget_rows: list[dict] = []
    for rep in range(int(args.repetitions)):
        for alg_index, algorithm in enumerate(algorithms):
            sampler = ContrastDecaySampler(
                bundle,
                T=None,
                seed=int(args.seed) + 1009 * rep + alg_index,
            )
            try:
                rows, final = run_algorithm_once(
                    algorithm,
                    sampler,
                    bundle,
                    run_kind="ideal_simulation",
                    repetition=rep,
                    epsilon_target=float(args.epsilon_target),
                    alpha=float(args.alpha),
                    n_shots=int(args.n_shots),
                    max_queries=int(args.max_queries),
                    seed=int(args.seed) + rep + alg_index,
                    algorithm_labels=ALGORITHM_LABELS,
                )
                trace_rows.extend(rows)
                final_rows.append(final)
                budget_rows.extend(rows_at_budgets(rows, budgets, run_kind="ideal_simulation"))
            except Exception as exc:
                error_rows.append(
                    {
                        "repetition": rep,
                        "algorithm": algorithm,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    summary_rows = aggregate_budget_summary(
        budget_rows,
        total_repetitions=int(args.repetitions),
        group_by_budget=True,
    )
    save_csv(trace_rows, paths.direct_trace)
    save_csv(final_rows, paths.direct_final)
    save_csv(error_rows, paths.errors)
    save_csv(budget_rows, paths.replay_budget)
    save_csv(summary_rows, paths.budget_summary)
    paths.write_manifest()


if __name__ == "__main__":
    main()
