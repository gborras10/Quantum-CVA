from __future__ import annotations

import time
import types
from collections.abc import Mapping
from typing import Any

import numpy as np

from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta
from quantum_cva.algorithms.proposed_algorithms.cabiae_known_t import CABIQAE
from quantum_cva.algorithms.proposed_algorithms.elf_qae import ELFQAE
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE
from quantum_cva.amplitude_estimation.experiments.circuits import patch_construct_circuit
from quantum_cva.amplitude_estimation.experiments.problems import AEProblemBundle
from quantum_cva.amplitude_estimation.experiments.traces import (
    default_effective_n_shots,
    trace_rows_from_result,
)

try:
    from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
        StandaloneBAEHardware as StandaloneBAE,
    )

    BAE_KIND = "hardware"
except Exception:  # pragma: no cover - kept for minimal environments
    from quantum_cva.algorithms.third_party.standalone_bae import (
        StandaloneBAE,
    )

    BAE_KIND = "legacy"


ALGORITHM_LABELS: dict[str, str] = {
    "bae": "BAE",
    "biqae": "BIQAE",
    "iqae": "IQAE",
    "cabiqae": "CABIQAE",
    "cabiqae_known_t": "CABIQAE",
    "cabiqae_latentt": "CABIQAE",
    "cabiaelatent": "CABIQAE-LTheta",
    "elf_qae": "ELF-QAE",
}

ALGORITHM_STYLES: dict[str, dict[str, str]] = {
    "bae": {"color": "#E07A5F", "marker": "^"},
    "biqae": {"color": "#A23B72", "marker": "s"},
    "iqae": {"color": "#4C78A8", "marker": "D"},
    "cabiqae": {"color": "#1F6F8B", "marker": "o"},
    "cabiqae_known_t": {"color": "#1F6F8B", "marker": "o"},
    "cabiqae_latentt": {"color": "#1F6F8B", "marker": "o"},
    "cabiaelatent": {"color": "#1F6F8B", "marker": "o"},
    "elf_qae": {"color": "#6D597A", "marker": "v"},
}


def normalize_algorithm_key(algorithm: str) -> str:
    key = str(algorithm).strip().lower().replace("-", "_")
    aliases = {
        "cabiqaev2": "cabiqae_latentt",
        "cabiqae_ltheta": "cabiqae_latentt",
        "cabiqae_latent_theta": "cabiqae_latentt",
        "cabiaelatent": "cabiqae_latentt",
        "elf": "elf_qae",
    }
    return aliases.get(key, key)


def disable_cabiqae_hard_k_cap(solver: Any) -> None:
    if not hasattr(solver, "_k_cap"):
        return

    def _uncapped(self: Any) -> int:
        return 10**100

    solver._k_cap = types.MethodType(_uncapped, solver)


def _noise_model_name(t_eff: float | None) -> str:
    return "ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast"


def build_solver(
    algorithm: str,
    sampler: Any,
    *,
    epsilon_target: float,
    alpha: float,
    n_shots: int | None = None,
    t_eff: float | None = None,
    seed: int | None = None,
    cap_kappa: float = 2.0,
    max_shots_same_k: int | None = None,
    max_queries: int | None = None,
    layers: int | None = None,
    layer_selection: str = "fixed",
    optimizer_restarts: int = 0,
    disable_hard_k_cap: bool = False,
    construct_circuit_cache: dict[tuple[Any, ...], Any] | None = None,
    construct_circuit_mode: str = "full",
    solver_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, bool]:
    """Create a configured AE solver and whether to call it in Bayesian mode."""
    del n_shots
    algorithm = normalize_algorithm_key(algorithm)
    kwargs = dict(solver_kwargs or {})
    noise_floor = float(kwargs.pop("noise_floor", 0.5))
    noise_model = _noise_model_name(t_eff)
    T_known = None if t_eff is None or np.isinf(t_eff) else float(t_eff)

    if algorithm == "biqae":
        solver = BIQAE(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            min_ratio=kwargs.pop("min_ratio", 2),
            confint_method=kwargs.pop("confint_method", "beta"),
            max_shots_same_k=max_shots_same_k,
            **kwargs,
        )
        patch_construct_circuit(
            solver,
            source="biqae",
            circuit_cache=construct_circuit_cache,
            construction_mode=construct_circuit_mode,
        )
        return solver, True

    if algorithm == "iqae":
        solver = BIQAE(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            min_ratio=kwargs.pop("min_ratio", 2),
            confint_method=kwargs.pop("confint_method", "beta"),
            max_shots_same_k=max_shots_same_k,
            **kwargs,
        )
        patch_construct_circuit(
            solver,
            source="iqae",
            circuit_cache=construct_circuit_cache,
            construction_mode=construct_circuit_mode,
        )
        return solver, False

    if algorithm in {"cabiqae", "cabiqae_known_t"}:
        solver = CABIQAE(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            min_ratio=kwargs.pop("min_ratio", 2),
            confint_method=kwargs.pop("confint_method", "beta"),
            noise_model=noise_model,
            T_known=T_known,
            cap_kappa=float(cap_kappa),
            use_noise_cap=kwargs.pop("use_noise_cap", True),
            max_shots_same_k=max_shots_same_k,
            noise_floor=noise_floor,
            **kwargs,
        )
        patch_construct_circuit(
            solver,
            source="cabiqae",
            circuit_cache=construct_circuit_cache,
            construction_mode=construct_circuit_mode,
        )
        if disable_hard_k_cap:
            disable_cabiqae_hard_k_cap(solver)
        return solver, True

    if algorithm == "cabiqae_latentt":
        solver = CABIQAELatentTheta(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            min_ratio=kwargs.pop("min_ratio", 2),
            confint_method=kwargs.pop("confint_method", "beta"),
            noise_model=noise_model,
            T_known=T_known,
            cap_kappa=float(cap_kappa),
            use_noise_cap=kwargs.pop("use_noise_cap", True),
            max_shots_same_k=max_shots_same_k,
            random_seed=seed,
            noise_floor=noise_floor,
            **kwargs,
        )
        patch_construct_circuit(
            solver,
            source="cabiqae_latentt",
            circuit_cache=construct_circuit_cache,
            construction_mode=construct_circuit_mode,
        )
        if disable_hard_k_cap:
            disable_cabiqae_hard_k_cap(solver)
        return solver, True

    if algorithm == "bae":
        bae_kwargs: dict[str, Any] = {
            "epsilon_target": float(epsilon_target),
            "alpha": float(alpha),
            "sampler": sampler,
            "noise_model": noise_model,
            "T_known": T_known,
            "cap_kappa": float(cap_kappa),
            "max_shots_same_k": max_shots_same_k,
            "noise_floor": noise_floor,
            "estimate_T": False,
            "wNs": kwargs.pop("wNs", 100),
            "Nevals": kwargs.pop("Nevals", 50),
            "Npart": kwargs.pop("Npart", 800),
            "thr": kwargs.pop("thr", 0.5),
        }
        bae_kwargs.update(kwargs)
        solver = StandaloneBAE(**bae_kwargs)
        return solver, True

    if algorithm == "elf_qae":
        elf_layers = int(layers if layers is not None else 1)
        solver = ELFQAE(
            epsilon_target=float(epsilon_target),
            alpha=float(alpha),
            sampler=sampler,
            layers=elf_layers,
            max_layers=elf_layers,
            layer_selection=str(layer_selection),
            circuit_fidelity=kwargs.pop("circuit_fidelity", 1.0),
            max_state_prep_calls=max_queries,
            max_rounds=kwargs.pop(
                "max_rounds",
                max(1, int(float(max_queries or 10_000) // (2 * elf_layers + 1))),
            ),
            optimizer_restarts=int(optimizer_restarts),
            random_seed=seed,
            **kwargs,
        )
        return solver, False

    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_algorithm_once(
    algorithm: str,
    sampler: Any,
    bundle: AEProblemBundle,
    *,
    run_kind: str,
    repetition: int,
    epsilon_target: float,
    alpha: float,
    n_shots: int | None,
    max_queries: int | None = None,
    t_eff: float | None = None,
    seed: int = 12345,
    algorithm_labels: Mapping[str, str] | None = None,
    cap_kappa: float = 2.0,
    disable_hard_k_cap: bool = False,
    construct_circuit_cache: dict[tuple[Any, ...], Any] | None = None,
    construct_circuit_mode: str = "full",
    solver_kwargs: Mapping[str, Any] | None = None,
    trace_extra: Mapping[str, Any] | None = None,
    show_details: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm = normalize_algorithm_key(algorithm)
    if hasattr(sampler, "set_context"):
        sampler.set_context(f"{run_kind}_{algorithm}")
    np.random.seed(int(seed))
    solver, bayes = build_solver(
        algorithm,
        sampler,
        epsilon_target=float(epsilon_target),
        alpha=float(alpha),
        n_shots=n_shots,
        max_queries=max_queries,
        t_eff=t_eff,
        seed=int(seed),
        cap_kappa=float(cap_kappa),
        disable_hard_k_cap=bool(disable_hard_k_cap),
        construct_circuit_cache=construct_circuit_cache,
        construct_circuit_mode=construct_circuit_mode,
        solver_kwargs=solver_kwargs,
    )

    start = time.perf_counter()
    if algorithm == "bae":
        result = solver.estimate(
            bundle.problem,
            n_shots=default_effective_n_shots(algorithm, n_shots),
            max_queries=int(max_queries or 200_000),
            show_details=show_details,
        )
    elif algorithm == "elf_qae":
        kwargs: dict[str, Any] = {"show_details": show_details}
        if n_shots is not None:
            kwargs["n_shots"] = int(n_shots)
        result = solver.estimate(bundle.problem, **kwargs)
    else:
        kwargs = {"bayes": bayes, "show_details": show_details}
        if n_shots is not None:
            kwargs["n_shots"] = int(n_shots)
        result = solver.estimate(bundle.problem, **kwargs)
    elapsed_total = time.perf_counter() - start
    construct_metrics = dict(getattr(solver, "_construct_circuit_metrics", {}) or {})
    construct_wall = float(construct_metrics.get("construct_circuit_wall_seconds", 0.0))
    elapsed_algorithm = max(0.0, float(elapsed_total) - max(0.0, construct_wall))
    timing_extra = dict(trace_extra or {})
    timing_extra.update(
        {
            "circuit_prepare_wall_seconds": float(construct_wall),
            "construct_circuit_wall_seconds": float(construct_wall),
            "construct_circuit_cache_hits": int(
                construct_metrics.get("construct_circuit_cache_hits", 0)
            ),
            "construct_circuit_cache_misses": int(
                construct_metrics.get("construct_circuit_cache_misses", 0)
            ),
            "construct_circuit_cache_size": int(
                len(getattr(solver, "_construct_circuit_cache", {}) or {})
            ),
            "construct_circuit_mode": str(construct_circuit_mode),
        }
    )

    return trace_rows_from_result(
        result,
        bundle=bundle,
        algorithm=algorithm,
        algorithm_labels=algorithm_labels or ALGORITHM_LABELS,
        repetition=int(repetition),
        n_shots=n_shots,
        elapsed_wall_seconds=float(elapsed_algorithm),
        run_kind=run_kind,
        effective_n_shots=default_effective_n_shots,
        extra=timing_extra,
    )
