from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.configs import (
    AlgorithmRunConfig,
    HardwareReplayConfig,
    parse_int_csv,
    parse_name_csv,
)
from quantum_cva.amplitude_estimation.experiments.hardware_runner import (
    HardwareReplayRunner,
)
from quantum_cva.amplitude_estimation.experiments.runner_utils import (
    add_problem_builder_args,
    problem_bundle_from_args,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hardware-aware AE runner. The default mode is dry-run; replay-only "
            "reuses an existing hardware/dry-run directory."
        )
    )
    add_problem_builder_args(parser)
    parser.add_argument("--mode", choices=("dry-run", "replay-only"), default="dry-run")
    parser.add_argument("--run-dir", default="experiment_results/hardware_dry_run")
    parser.add_argument("--algorithms", default="cabiqae_latentt,biqae,bae")
    parser.add_argument("--budgets", default="128,256,512,1024,2048")
    parser.add_argument("--replay-repetitions", type=int, default=100)
    parser.add_argument("--direct-shots", type=int, default=64)
    parser.add_argument("--epsilon-target", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-grover-power", type=int, default=8)
    parser.add_argument(
        "--contrast-baseline",
        type=float,
        default=0.5,
        help="Asymptotic good-state probability under full contrast loss.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    bundle = problem_bundle_from_args(args)
    config = HardwareReplayConfig(
        run_dir=Path(args.run_dir),
        mode=args.mode,
        algorithm=AlgorithmRunConfig(
            algorithms=parse_name_csv(args.algorithms),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            seed=int(args.seed),
        ),
        budgets=parse_int_csv(args.budgets),
        replay_repetitions=int(args.replay_repetitions),
        direct_shots=int(args.direct_shots),
        max_grover_power=int(args.max_grover_power),
        contrast_baseline=float(args.contrast_baseline),
    )
    HardwareReplayRunner(config, bundle).run()


if __name__ == "__main__":
    main()
