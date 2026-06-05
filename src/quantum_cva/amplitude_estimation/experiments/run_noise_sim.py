from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.configs import (
    AlgorithmRunConfig,
    NoiseSimulationConfig,
    parse_int_csv,
    parse_name_csv,
)
from quantum_cva.amplitude_estimation.experiments.noise_runner import (
    NoiseSimulationRunner,
)
from quantum_cva.amplitude_estimation.experiments.runner_utils import (
    add_problem_builder_args,
    problem_bundle_from_args,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run noisy Aer AE replay experiment.")
    add_problem_builder_args(parser)
    parser.add_argument("--run-dir", default="experiment_results/noise_sim")
    parser.add_argument("--algorithms", default="cabiqae_latentt,biqae,bae")
    parser.add_argument("--budgets", default="128,256,512,1024,2048")
    parser.add_argument("--max-grover-power", type=int, default=8)
    parser.add_argument("--scan-repeats", type=int, default=1)
    parser.add_argument("--scan-shots", type=int, default=256)
    parser.add_argument("--readout-shots", type=int, default=512)
    parser.add_argument("--direct-shots", type=int, default=64)
    parser.add_argument("--replay-repetitions", type=int, default=20)
    parser.add_argument("--epsilon-target", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--noise-profile", default="projected")
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
    config = NoiseSimulationConfig(
        run_dir=Path(args.run_dir),
        algorithm=AlgorithmRunConfig(
            algorithms=parse_name_csv(args.algorithms),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            seed=int(args.seed),
        ),
        budgets=parse_int_csv(args.budgets),
        max_grover_power=int(args.max_grover_power),
        scan_repeats=int(args.scan_repeats),
        scan_shots=int(args.scan_shots),
        readout_shots=int(args.readout_shots),
        direct_shots=int(args.direct_shots),
        replay_repetitions=int(args.replay_repetitions),
        noise_scale=float(args.noise_scale),
        noise_profile=str(args.noise_profile),
        contrast_baseline=float(args.contrast_baseline),
    )
    NoiseSimulationRunner(config, bundle).run()


if __name__ == "__main__":
    main()
