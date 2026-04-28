from __future__ import annotations

import argparse

'''


'''

DEFAULT_BACKEND = "ibm_basquecountry"
DEFAULT_CHANNEL = "ibm_quantum_platform"
DEFAULT_OBJECTIVE_RY_OFFSET = -0.10
DEFAULT_MAX_GROVER_POWER = 50
DEFAULT_REFERENCE_KS = (0, 1, 2, 3, 4, 5, 6, 7)
DEFAULT_BUDGETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
ALGORITHMS = ("cabiqae_latentt", "biqae", "bae")
ALGORITHM_LABELS = {
    "cabiqae_latentt": "CABIQAE",
    "biqae": "BIQAE",
    "bae": "BAE",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid hardware/replay BIQAE vs BAE vs CABIQAE experiment for the 3-qubit AE toy."
    )
    parser.add_argument(
        "--mode",
        choices=("preflight", "hardware", "hardware-topup", "replay-only", "dry-run"),
        default="dry-run",
    )
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-grover-power", type=int, default=DEFAULT_MAX_GROVER_POWER)
    parser.add_argument("--session-max-time", default="5m")
    parser.add_argument("--objective-ry-offset", type=float, default=DEFAULT_OBJECTIVE_RY_OFFSET)
    parser.add_argument("--scan-repeats", type=int, default=2)
    parser.add_argument("--scan-shots", type=int, default=1024)
    parser.add_argument("--scan-grover-powers", default=None)
    parser.add_argument("--readout-shots", type=int, default=2048)
    parser.add_argument("--direct-shots", type=int, default=128)
    parser.add_argument("--max-direct-calls", type=int, default=4)
    parser.add_argument("--replay-repetitions", type=int, default=200)
    parser.add_argument("--replay-probability-mode", choices=("fixed", "normal"), default="normal")
    parser.add_argument("--replay-probability-se-scale", type=float, default=1.0)
    parser.add_argument("--budgets", default=",".join(str(x) for x in DEFAULT_BUDGETS))
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--epsilon-target", type=float, default=0.08)
    parser.add_argument("--soft-wallclock-limit", type=float, default=240.0)
    parser.add_argument("--max-isa-depth", type=int, default=2700)
    parser.add_argument("--max-isa-2q", type=int, default=750)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--seed-transpiler", type=int, default=1234)
    parser.add_argument("--reference-ks", default=",".join(str(x) for x in DEFAULT_REFERENCE_KS))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fake-backend", default="fake_fez")
    parser.add_argument("--dry-run-noise-scale", type=float, default=1.0)
    parser.add_argument("--dry-run-noise-profile", default="projected")
    parser.add_argument("--aer-method", default="density_matrix")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.algorithms = ALGORITHMS
    args.algorithm_labels = ALGORITHM_LABELS
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from hardware_bae_biqae_cabiqae_core import run_experiment, run_hardware_topup, run_replay_only

    if args.mode == "replay-only":
        run_replay_only(args)
    elif args.mode == "hardware-topup":
        run_hardware_topup(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
