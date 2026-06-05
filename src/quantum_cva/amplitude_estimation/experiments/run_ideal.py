from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.configs import (
    AlgorithmRunConfig,
    IdealExperimentConfig,
    parse_int_csv,
    parse_name_csv,
)
from quantum_cva.amplitude_estimation.experiments.ideal_runner import (
    IdealExperimentRunner,
)
from quantum_cva.amplitude_estimation.experiments.runner_utils import (
    add_problem_builder_args,
    problem_bundle_from_args,
)


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
    config = IdealExperimentConfig(
        run_dir=Path(args.run_dir),
        algorithm=AlgorithmRunConfig(
            algorithms=parse_name_csv(args.algorithms),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            seed=int(args.seed),
        ),
        repetitions=int(args.repetitions),
        n_shots=int(args.n_shots),
        max_queries=int(args.max_queries),
        budgets=parse_int_csv(args.budgets),
    )
    IdealExperimentRunner(config, bundle).run()


if __name__ == "__main__":
    main()
