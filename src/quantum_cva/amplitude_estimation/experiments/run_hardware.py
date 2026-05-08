from __future__ import annotations

import argparse
from pathlib import Path

from quantum_cva.amplitude_estimation.experiments.hardware import (
    load_existing_state,
    run_dry_run_experiment,
    run_replay,
    effective_t_for_algorithms,
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
    algorithms = tuple(x.strip() for x in args.algorithms.split(",") if x.strip())
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())

    if args.mode == "dry-run":
        run_dry_run_experiment(
            bundle,
            run_dir=Path(args.run_dir),
            algorithms=algorithms,
            budgets=budgets,
            max_grover_power=int(args.max_grover_power),
            direct_shots=int(args.direct_shots),
            replay_repetitions=int(args.replay_repetitions),
            epsilon_target=float(args.epsilon_target),
            alpha=float(args.alpha),
            seed=int(args.seed),
            contrast_baseline=float(args.contrast_baseline),
        )
        return

    state = load_existing_state(args.run_dir)
    p_by_k = {int(r["grover_power"]): float(r["p_hw_mitigated"]) for r in state.amplification_point_rows}
    p_se_by_k = {
        int(r["grover_power"]): float(r["p_hw_mitigated_se"])
        for r in state.amplification_point_rows
    }
    run_replay(
        state,
        bundle,
        algorithms=algorithms,
        p_by_k=p_by_k,
        p_se_by_k=p_se_by_k,
        replay_probability_mode="normal",
        replay_probability_se_scale=1.0,
        budgets=budgets,
        repetitions=int(args.replay_repetitions),
        n_shots=int(args.direct_shots),
        epsilon_target=float(args.epsilon_target),
        alpha=float(args.alpha),
        t_eff=effective_t_for_algorithms(state.calibration_summary),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
