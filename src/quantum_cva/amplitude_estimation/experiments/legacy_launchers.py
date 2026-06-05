from __future__ import annotations

import sys
from collections.abc import Callable, Sequence


CliMain = Callable[[list[str] | None], None]
DefaultArgs = Sequence[tuple[str, str | None]]


def _has_option(args: Sequence[str], option: str) -> bool:
    return option in args or any(arg.startswith(f"{option}=") for arg in args)


def _with_defaults(argv: Sequence[str] | None, defaults: DefaultArgs) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    prefix: list[str] = []
    for option, value in defaults:
        if _has_option(args, option):
            continue
        prefix.append(option)
        if value is not None:
            prefix.append(value)
    return [*prefix, *args]


def _run(main: CliMain, argv: Sequence[str] | None, defaults: DefaultArgs = ()) -> None:
    main(_with_defaults(argv, defaults))


def run_ideal(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.run_ideal import main

    _run(main, argv)


def run_ideal_with_elf(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.run_ideal import main

    _run(
        main,
        argv,
        (
            ("--algorithms", "cabiqae_latentt,biqae,iqae,bae,elf_qae"),
            ("--run-dir", "experiment_results/ideal_elf"),
        ),
    )


def plot_ideal(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.plot_results import main

    _run(main, argv, (("--run-dir", "experiment_results/ideal"),))


def run_noise_sim(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.run_noise_sim import main

    _run(main, argv)


def plot_noise_sim(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.plot_results import main

    _run(main, argv, (("--run-dir", "experiment_results/noise_sim"),))


def run_hardware(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.run_hardware import main

    _run(main, argv)


def plot_hardware(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.plot_results import main

    _run(main, argv, (("--run-dir", "experiment_results/hardware_dry_run"),))


def run_cva_hardware(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (
        main,
    )

    _run(main, argv)


def plot_cva_hardware(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_plot_cli import (
        main,
    )

    _run(main, argv, (("--run-dir", "experiment_results/hardware_dry_run"),))


def plot_cva_calibration(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_plot_cli import (
        main,
    )

    _run(
        main,
        argv,
        (
            ("--run-dir", "experiment_results/hardware_dry_run"),
            ("--kind", "calibration-paper"),
        ),
    )


def run_cva_hardware_topup(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (
        main,
    )

    _run(main, argv, (("--mode", "hardware-topup"),))


def recover_cva_hardware_session(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (
        main,
    )

    _run(main, argv, (("--mode", "recover-session"),))


def reanalyze_cva_replay(argv: Sequence[str] | None = None) -> None:
    from quantum_cva.amplitude_estimation.experiments.cva_hwd_experiments.cva_hardware_runner import (
        main,
    )

    _run(main, argv, (("--mode", "reanalyze-replay"),))

