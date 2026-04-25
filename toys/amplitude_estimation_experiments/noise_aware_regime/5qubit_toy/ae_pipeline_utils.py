from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Statevector
from qiskit_algorithms import EstimationProblem
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_aer.noise.errors import depolarizing_error, thermal_relaxation_error


TOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in TOY_DIR.parents if (parent / "pyproject.toml").exists())
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, TOY_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from ae_circuit_utils import (  # noqa: E402
    DEFAULT_AE_REFERENCE_KS,
    DEFAULT_ROUTING_METHOD,
    INITIAL_LAYOUT,
    PHYSICAL_BACKEND_NAME,
    TRANSPILER_OPTIMIZATION_LEVEL,
    TRANSPILER_SEED,
    build_estimation_problem,
    build_problem_with_true_amplitude,
    build_state_preparation,
    choose_transpilation_plan,
    construct_measured_circuit,
    true_amplitude,
)
from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta  # noqa: E402
from quantum_cva.algorithms.third_party.biae import BayesianIQAE as BIQAE  # noqa: E402
from quantum_cva.quantum_hardware_utilities.transpile_utils import (  # noqa: E402
    DEFAULT_TRANSPILER_SEEDS,
    stable_circuit_key,
)

try:
    from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
        StandaloneBAEHardware as StandaloneBAEModel,
    )

    BAE_KIND = "hardware"
except Exception:
    from quantum_cva.algorithms.third_party.standalone_bae import (
        StandaloneBAE as StandaloneBAEModel,
    )

    BAE_KIND = "legacy"


NOISE_PROFILE_PROJECTED = "projected"
NOISE_PROFILE_BASELINE = "baseline"
NOISE_PROFILE_MILD = NOISE_PROFILE_PROJECTED
NOISE_PROFILE_REALISTIC = NOISE_PROFILE_PROJECTED


class _FakeCounts:
    def __init__(self, counts: dict[str, int]):
        self._counts = dict(counts)

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeData:
    def __init__(self, counts: dict[str, int]):
        self.c0 = _FakeCounts(counts)


class _FakePubResult:
    def __init__(self, counts: dict[str, int]):
        self.data = _FakeData(counts)


class _FakeSamplerJob:
    def __init__(self, pub_results: list[_FakePubResult]):
        self._pub_results = pub_results

    def result(self) -> list[_FakePubResult]:
        return self._pub_results


@dataclass
class CalibrationResult:
    profile_name: str
    t_eff: float | None
    slope: float | None
    intercept: float | None
    used_points: int
    rows: list[dict[str, float]]


# Compatibility aliases. The old directory name says "5qubit_toy", but the canonical
# problem for this toy now lives in ae_circuit_utils.py and uses 3 logical qubits.
def build_large_state_preparation(objective_ry_offset: float = 0.0) -> QuantumCircuit:
    return build_state_preparation(objective_ry_offset)


def build_large_problem(objective_ry_offset: float = 0.0) -> tuple[EstimationProblem, float]:
    return build_problem_with_true_amplitude(objective_ry_offset)


def circuit_cache_key(circuit: QuantumCircuit) -> str:
    try:
        return stable_circuit_key(circuit)
    except Exception:
        ops = tuple(sorted((str(name), int(count)) for name, count in circuit.count_ops().items()))
        return str((circuit.num_qubits, circuit.num_clbits, circuit.size(), circuit.depth(), ops))


def build_ae_pass_manager(
    backend: Any,
    problem: EstimationProblem,
    *,
    optimization_level: int = TRANSPILER_OPTIMIZATION_LEVEL,
    seed_transpiler: int = TRANSPILER_SEED,
    reference_ks: Iterable[int] = DEFAULT_AE_REFERENCE_KS,
    routing_method: str | None = DEFAULT_ROUTING_METHOD,
    discovery_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
    evaluation_seeds: Sequence[int] = DEFAULT_TRANSPILER_SEEDS,
) -> tuple[Any, dict[str, Any]]:
    normalized_reference_ks = tuple(sorted({int(k) for k in reference_ks}))

    try:
        plan = choose_transpilation_plan(
            backend,
            problem,
            optimization_level=int(optimization_level),
            reference_ks=normalized_reference_ks,
            routing_method=routing_method,
            discovery_seeds=discovery_seeds,
            evaluation_seeds=evaluation_seeds,
        )
    except Exception as exc:
        return (
            generate_preset_pass_manager(
                backend=backend,
                optimization_level=int(optimization_level),
                seed_transpiler=int(seed_transpiler),
            ),
            {
                "strategy": "preset_default_fallback",
                "fallback_used": True,
                "fallback_reason": str(exc),
                "initial_layout": None,
                "seed_transpiler": int(seed_transpiler),
                "optimization_level": int(optimization_level),
                "routing_method": None,
                "candidate_source": "preset_default",
                "reference_ks": list(normalized_reference_ks),
                "discovery_seeds": [int(seed) for seed in discovery_seeds],
                "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
            },
        )

    return (
        plan.build_pass_manager(backend),
        {
            "strategy": "fixed_layout_search",
            "fallback_used": False,
            "fallback_reason": None,
            "reference_ks": list(normalized_reference_ks),
            "discovery_seeds": [int(seed) for seed in discovery_seeds],
            "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
            **plan.metadata(),
        },
    )


def build_noise_model(scale: float, *, profile: str = NOISE_PROFILE_PROJECTED) -> NoiseModel:
    noise_model = NoiseModel()

    if profile == NOISE_PROFILE_BASELINE:
        p1 = min(1.88e-4 * scale, 5e-3)
        p2 = min(1.93e-3 * scale, 5e-2)
        p_10 = min(4.83e-3 * scale, 0.2)
        p_01 = min(4.30e-3 * scale, 0.2)
        t1 = 279_400.0 / scale
        t2 = 220_400.0 / scale
        t_id = 5.0
        t_sx = 32.0
        t_x = 32.0
        t_cx = 132.0
    elif profile in {NOISE_PROFILE_PROJECTED, NOISE_PROFILE_MILD, NOISE_PROFILE_REALISTIC, "mild", "realistic"}:
        p1 = min(9.00e-5 * scale, 5e-3)
        p2 = min(8.00e-4 * scale, 5e-2)
        p_10 = min(1.20e-3 * scale, 0.2)
        p_01 = min(1.00e-3 * scale, 0.2)
        t1 = 600_000.0 / scale
        t2 = 550_000.0 / scale
        t_id = 5.0
        t_sx = 5.0
        t_x = 20.0
        t_cx = 50.0
    else:
        raise ValueError(f"Unknown noise profile: {profile}")

    err_id = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, t_id))
    err_sx = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, t_sx))
    err_x = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, t_x))
    err_cx = depolarizing_error(p2, 2).compose(
        thermal_relaxation_error(t1, t2, t_cx).tensor(
            thermal_relaxation_error(t1, t2, t_cx)
        )
    )
    ro = ReadoutError([[1.0 - p_10, p_10], [p_01, 1.0 - p_01]])

    noise_model.add_all_qubit_quantum_error(err_id, ["id"])
    noise_model.add_all_qubit_quantum_error(err_sx, ["sx"])
    noise_model.add_all_qubit_quantum_error(err_x, ["x"])
    noise_model.add_all_qubit_quantum_error(err_cx, ["cx", "cz", "ecr"])
    noise_model.add_all_qubit_readout_error(ro)

    return noise_model


def ideal_good_probability(problem: EstimationProblem, k: int) -> float:
    circuit = QuantumCircuit(problem.state_preparation.num_qubits)
    circuit.compose(problem.state_preparation, inplace=True)
    if k > 0:
        grover_power = problem.grover_operator.power(k)
        if hasattr(grover_power, "decompose"):
            grover_power = grover_power.decompose(reps=10)
        circuit.compose(grover_power, inplace=True)
    state = Statevector.from_instruction(circuit)
    probs = state.probabilities_dict(qargs=list(problem.objective_qubits))
    good_key = "1" * len(problem.objective_qubits)
    return float(probs.get(good_key, 0.0))


class AerCountSampler:
    """
    Minimal SamplerV2-compatible counts wrapper around AerSimulator.

    It always executes the canonical measured AE circuits after transpiling them
    through either a fixed hardware-aware pass manager or the provided backend.
    """

    def __init__(
        self,
        noise_model: NoiseModel | None = None,
        seed: int | None = None,
        method: str = "density_matrix",
        transpile_backend: object | None = None,
        initial_layout: list[int] | None = None,
        transpilation_plan: Any | None = None,
    ):
        self._rng = np.random.default_rng(seed)
        self._sim = AerSimulator(noise_model=noise_model, method=method)
        self._transpile_backend = transpile_backend
        self._initial_layout = list(initial_layout) if initial_layout is not None else None
        self._transpilation_plan = transpilation_plan
        self._pass_manager = (
            transpilation_plan.build_pass_manager(transpile_backend)
            if transpilation_plan is not None and transpile_backend is not None
            else None
        )
        self._cache: dict[str, QuantumCircuit] = {}

    def _transpiled(self, circuit: QuantumCircuit) -> QuantumCircuit:
        circuit = circuit.decompose(reps=10)
        key = circuit_cache_key(circuit)
        if key not in self._cache:
            if self._pass_manager is not None:
                self._cache[key] = self._pass_manager.run(circuit)
            else:
                from ae_circuit_utils import transpile_for_execution

                transpile_backend = self._transpile_backend or self._sim
                initial_layout = self._initial_layout
                if initial_layout is None and self._transpile_backend is not None:
                    initial_layout = INITIAL_LAYOUT
                self._cache[key] = transpile_for_execution(
                    circuit,
                    backend=transpile_backend,
                    initial_layout=initial_layout,
                )
        return self._cache[key]

    def run(self, circuits: list[QuantumCircuit], shots: int = 1024) -> _FakeSamplerJob:
        pub_results: list[_FakePubResult] = []
        for circuit in circuits:
            transpiled_circuit = self._transpiled(circuit)
            seed_sim = int(self._rng.integers(1, 2**31 - 1))
            result = self._sim.run(
                transpiled_circuit,
                shots=int(shots),
                seed_simulator=seed_sim,
            ).result()
            counts = result.get_counts()
            if not isinstance(counts, dict):
                counts = dict(counts)
            counts = {str(key): int(value) for key, value in counts.items()}
            pub_results.append(_FakePubResult(counts))
        return _FakeSamplerJob(pub_results)


def estimate_prob_from_sampler(
    sampler: AerCountSampler,
    problem: EstimationProblem,
    k: int,
    shots: int,
) -> float:
    circuit = construct_measured_circuit(problem, k)
    ret = sampler.run([circuit], shots=shots).result()
    counts = ret[0].data.c0.get_counts()
    one = int(counts.get("1", 0))
    return float(one / shots)


def calibrate_effective_T(
    profile_name: str,
    sampler: AerCountSampler,
    problem: EstimationProblem,
    probe_ks: list[int],
    probe_shots: int,
) -> CalibrationResult:
    rows: list[dict[str, float]] = []
    fit_k_values: list[float] = []
    fit_log_c_values: list[float] = []

    for k in probe_ks:
        K = 2 * k + 1
        p_ideal = ideal_good_probability(problem, k)
        p_noisy = estimate_prob_from_sampler(sampler, problem, k, probe_shots)

        denom = p_ideal - 0.5
        numer = p_noisy - 0.5

        if abs(denom) < 2.0e-2:
            c_est = np.nan
        else:
            c_est = numer / denom

        if np.isfinite(c_est):
            c_est = float(np.clip(c_est, 1e-6, 1.0))
        else:
            c_est = np.nan

        rows.append(
            {
                "k": float(k),
                "K": float(K),
                "p_ideal": float(p_ideal),
                "p_noisy": float(p_noisy),
                "c_est": float(c_est) if np.isfinite(c_est) else np.nan,
            }
        )

        if np.isfinite(c_est) and 0.0 < c_est < 1.0 and abs(denom) >= 2.0e-2:
            fit_k_values.append(float(K))
            fit_log_c_values.append(float(np.log(c_est)))

    if len(fit_k_values) < 2:
        return CalibrationResult(
            profile_name=profile_name,
            t_eff=None,
            slope=None,
            intercept=None,
            used_points=len(fit_k_values),
            rows=rows,
        )

    slope, intercept = np.polyfit(np.asarray(fit_k_values), np.asarray(fit_log_c_values), deg=1)
    t_eff = None if slope >= 0 else float(-1.0 / slope)

    return CalibrationResult(
        profile_name=profile_name,
        t_eff=t_eff,
        slope=float(slope),
        intercept=float(intercept),
        used_points=len(fit_k_values),
        rows=rows,
    )


def build_solver(
    algorithm: str,
    epsilon_target: float,
    alpha: float,
    n_shots: int,
    seed: int,
    noisy_sampler: Any,
    t_eff: float | None,
    *,
    cap_kappa: float = 1000.0,
) -> tuple[Any, bool]:
    del seed

    if algorithm == "bae":
        if BAE_KIND == "hardware":
            solver = StandaloneBAEModel(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=noisy_sampler,
                noise_model="ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast",
                T_known=None if t_eff is None or np.isinf(t_eff) else float(t_eff),
                cap_kappa=float(cap_kappa),
                max_shots_same_k=None,
                estimate_T=False,
                wNs=50,
            )
        else:
            solver = StandaloneBAEModel(
                epsilon_target=epsilon_target,
                alpha=alpha,
                sampler=noisy_sampler,
                noise_model="ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast",
                T_known=None if t_eff is None or np.isinf(t_eff) else float(t_eff),
                cap_kappa=3.0,
                estimate_T=False,
                T_range=None
                if t_eff is None or np.isinf(t_eff)
                else (0.5 * float(t_eff), 1.5 * float(t_eff)),
                TNs=0,
                wNs=50,
                Ns=n_shots,
            )
        return solver, True

    if algorithm == "biqae":
        solver = BIQAE(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=noisy_sampler,
            min_ratio=2,
            confint_method="beta",
            max_shots_same_k=None,
        )
        return solver, True

    if algorithm in {"cabiqae", "cabiqae_latentt"}:
        solver = CABIQAELatentTheta(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=noisy_sampler,
            min_ratio=2,
            confint_method="beta",
            noise_model="ideal" if t_eff is None or np.isinf(t_eff) else "exponential_contrast",
            T_known=None if t_eff is None or np.isinf(t_eff) else float(t_eff),
            cap_kappa=float(cap_kappa),
            use_noise_cap=True,
            max_shots_same_k=None,
        )
        return solver, True

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _extract_bae_k_sequence(
    result: Any,
    history: dict[str, Any],
    queries: np.ndarray,
    n_shots: int,
) -> np.ndarray:
    k_seq_candidates = []

    obj_seq = getattr(result, "circuit_depths", None)
    if obj_seq is not None:
        arr = np.asarray(obj_seq, dtype=float).ravel()
        if arr.size > 0:
            k_seq_candidates.append(arr)

    hist_seq = history.get("circuit_depths", None)
    if hist_seq is not None:
        arr = np.asarray(hist_seq, dtype=float).ravel()
        if arr.size > 0:
            k_seq_candidates.append(arr)

    if k_seq_candidates:
        return np.asarray(np.rint(k_seq_candidates[0]), dtype=int)

    if len(queries) == 0:
        return np.asarray([], dtype=int)

    dq = np.diff(np.r_[0.0, queries])
    inferred = dq / float(n_shots)
    return np.asarray(np.rint(inferred), dtype=int)


def extract_trace(
    algorithm: str,
    result: Any,
    n_shots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if algorithm == "bae":
        history = getattr(result, "history", {}) or {}
        queries = np.asarray(history.get("queries", []), dtype=float)
        estimations = np.asarray(history.get("estimations", []), dtype=float)
        k_sequence = _extract_bae_k_sequence(result, history, queries, n_shots)

        usable = min(len(queries), len(estimations), len(k_sequence))
        if usable <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=int)

        return (
            queries[:usable].astype(float),
            estimations[:usable].astype(float),
            k_sequence[:usable].astype(int),
        )

    powers = np.asarray(getattr(result, "powers", []) or [], dtype=float)
    estimate_intervals = getattr(result, "estimate_intervals", []) or []

    usable = min(len(powers), max(0, len(estimate_intervals) - 1))
    if usable <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=int)

    interval_array = np.asarray(estimate_intervals[1 : 1 + usable], dtype=float)
    estimations = np.mean(interval_array, axis=1)
    k_sequence = (2.0 * powers[:usable] + 1.0).astype(int)
    queries = np.cumsum(n_shots * k_sequence)

    return queries.astype(float), estimations.astype(float), k_sequence.astype(int)


def estimate_at_budget(
    queries: np.ndarray,
    estimations: np.ndarray,
    budget: int,
) -> float | None:
    if len(queries) == 0:
        return None
    idx = np.searchsorted(queries, budget, side="right") - 1
    if idx < 0:
        return None
    return float(estimations[idx])


def _budget_index_at_or_below(queries: np.ndarray, budget: int) -> int:
    if len(queries) == 0:
        return -1
    return int(np.searchsorted(queries, budget, side="right") - 1)


def kmax_at_budget(
    queries: np.ndarray,
    k_sequence: np.ndarray,
    budget: int,
) -> int | None:
    idx = _budget_index_at_or_below(queries, budget)
    if idx < 0:
        return None
    prefix = np.asarray(k_sequence[: idx + 1], dtype=float)
    if prefix.size == 0:
        return None
    return int(np.max(prefix))


def time_to_budget(
    queries: np.ndarray,
    total_runtime_seconds: float,
    budget: int,
) -> float | None:
    if len(queries) == 0:
        return None
    q_final = float(queries[-1])
    if not np.isfinite(q_final) or q_final <= 0.0:
        return None
    budget_clipped = float(np.clip(float(budget), 0.0, q_final))
    runtime = max(0.0, float(total_runtime_seconds))
    return float(runtime * (budget_clipped / q_final))


def save_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean

    if rng is None:
        rng = np.random.default_rng(12345)

    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_means[i] = np.mean(rng.choice(values, size=len(values), replace=True))

    low = float(np.quantile(boot_means, alpha / 2))
    high = float(np.quantile(boot_means, 1 - alpha / 2))
    return mean, low, high


def bootstrap_median_ci(values, n_boot=2000, alpha=0.05, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    median = float(np.median(values))
    if len(values) == 1:
        return median, median, median

    if rng is None:
        rng = np.random.default_rng(12345)

    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot_medians[i] = np.median(rng.choice(values, size=len(values), replace=True))

    low = float(np.quantile(boot_medians, alpha / 2))
    high = float(np.quantile(boot_medians, 1 - alpha / 2))
    return median, low, high


def aggregate_budget_rows(
    rows: list[dict[str, Any]],
    algorithms: Iterable[str],
    algorithm_labels: dict[str, str],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    profile_names = sorted({str(r["profile"]) for r in rows})
    budgets = sorted({int(r["budget"]) for r in rows})

    for profile in profile_names:
        for budget in budgets:
            for alg in algorithms:
                alg_name = algorithm_labels[alg]
                subset = [
                    r
                    for r in rows
                    if str(r["profile"]) == profile
                    and int(r["budget"]) == budget
                    and r["algorithm"] == alg_name
                ]
                if not subset:
                    continue

                abs_err = np.asarray([float(r["abs_error"]) for r in subset], dtype=float)
                nrmse = np.asarray([float(r["nrmse"]) for r in subset], dtype=float)
                time_budget = np.asarray(
                    [float(r.get("time_to_budget_seconds", np.nan)) for r in subset],
                    dtype=float,
                )
                kmax_budget = np.asarray(
                    [float(r.get("k_max_budget", np.nan)) for r in subset],
                    dtype=float,
                )
                mae_mean, mae_low, mae_high = bootstrap_mean_ci(abs_err)
                nrmse_mean, nrmse_low, nrmse_high = bootstrap_mean_ci(nrmse)
                nrmse_median, nrmse_median_low, nrmse_median_high = bootstrap_median_ci(nrmse)
                time_mean, time_low, time_high = bootstrap_mean_ci(time_budget)
                kmax_mean, kmax_low, kmax_high = bootstrap_mean_ci(kmax_budget)

                summary_rows.append(
                    {
                        "profile": profile,
                        "budget": int(budget),
                        "algorithm": alg_name,
                        "n_points": int(len(subset)),
                        "n_runs": int(len(subset)),
                        "abs_error_median": float(np.nanmedian(abs_err)),
                        "mae_mean": mae_mean,
                        "mae_ci_low": mae_low,
                        "mae_ci_high": mae_high,
                        "nrmse_median": nrmse_median,
                        "nrmse_median_ci_low": nrmse_median_low,
                        "nrmse_median_ci_high": nrmse_median_high,
                        "nrmse_mean": nrmse_mean,
                        "nrmse_ci_low": nrmse_low,
                        "nrmse_ci_high": nrmse_high,
                        "time_to_budget_seconds_median": float(np.nanmedian(time_budget)),
                        "time_mean": time_mean,
                        "time_ci_low": time_low,
                        "time_ci_high": time_high,
                        "k_max_budget_median": float(np.nanmedian(kmax_budget)),
                        "kmax_mean": kmax_mean,
                        "kmax_ci_low": kmax_low,
                        "kmax_ci_high": kmax_high,
                    }
                )
    return summary_rows


def print_budget_summary(summary_rows: list[dict[str, Any]], profile_names: Iterable[str]) -> None:
    print("\n=== Budget-aligned summary on noisy AE circuit ===")
    for profile_name in profile_names:
        print(f"profile={profile_name}")
        prof_rows = [r for r in summary_rows if str(r["profile"]) == profile_name]
        budgets = sorted({int(r["budget"]) for r in prof_rows})
        for budget in budgets:
            b_rows = [r for r in prof_rows if int(r["budget"]) == budget]
            b_rows.sort(key=lambda x: float(x["nrmse_median"]))
            print(f"  budget={budget}")
            for row in b_rows:
                time_med = float(row.get("time_to_budget_seconds_median", np.nan))
                kmax_med = float(row.get("k_max_budget_median", np.nan))
                time_part = f" | t_budget_med={time_med:.3f}s" if np.isfinite(time_med) else ""
                k_part = f" | Kmax_budget_med={int(np.rint(kmax_med))}" if np.isfinite(kmax_med) else ""
                print(
                    "    "
                    + f"{row['algorithm']:16s} "
                    + f"nRMSE_med={row['nrmse_median']:.3e} | "
                    + f"AbsErr_med={row['abs_error_median']:.3e} | "
                    + f"n={row['n_points']}"
                    + time_part
                    + k_part
                )


def print_final_summary(
    final_rows: list[dict[str, Any]],
    profile_names: Iterable[str],
    algorithms: Iterable[str],
    algorithm_labels: dict[str, str],
) -> None:
    print("\n=== Final-stop auxiliary summary ===")
    for profile_name in profile_names:
        print(f"profile={profile_name}")
        for alg in algorithms:
            alg_name = algorithm_labels[alg]
            subset = [
                r
                for r in final_rows
                if str(r["profile"]) == profile_name and r["algorithm"] == alg_name
            ]
            if not subset:
                continue

            q = np.asarray([float(r["final_queries"]) for r in subset], dtype=float)
            nr = np.asarray([float(r["final_nrmse"]) for r in subset], dtype=float)
            cov = np.asarray([float(r["coverage"]) for r in subset], dtype=float)
            kmax = np.asarray([float(r["k_max"]) for r in subset], dtype=float)

            runtime_part = ""
            if any("runtime_seconds" in r for r in subset):
                runtime = np.asarray([float(r.get("runtime_seconds", np.nan)) for r in subset], dtype=float)
                runtime_part = f"time_med={np.nanmedian(runtime):.3f}s | "

            print(
                "  "
                + f"{alg_name:16s} "
                + f"Q_med={int(np.nanmedian(q)):6d} | "
                + f"nRMSE_med={np.nanmedian(nr):.3e} | "
                + f"coverage={np.nanmean(cov):.2f} | "
                + runtime_part
                + f"K_med={np.nanmedian(kmax):.0f}"
            )


def plot_budget_panels(
    summary_rows: list[dict[str, Any]],
    output_path: str,
    profile_names: Iterable[str],
    algorithms: Iterable[str],
    algorithm_labels: dict[str, str],
    algorithm_styles: dict[str, dict[str, str]],
    budgets_reference: np.ndarray,
) -> None:
    profile_names = list(profile_names)
    fig, axes = plt.subplots(1, len(profile_names), figsize=(5.4 * len(profile_names), 4.8), squeeze=False)

    for j, profile in enumerate(profile_names):
        ax = axes[0, j]
        prof_rows = [r for r in summary_rows if str(r["profile"]) == profile]

        for alg in algorithms:
            alg_name = algorithm_labels[alg]
            subset = [r for r in prof_rows if r["algorithm"] == alg_name]
            if not subset:
                continue

            budgets = np.asarray([int(r["budget"]) for r in subset], dtype=float)
            nrmse = np.asarray([float(r["nrmse_median"]) for r in subset], dtype=float)
            kmax = np.asarray([float(r.get("k_max_budget_median", np.nan)) for r in subset], dtype=float)
            nrmse_low = np.asarray([float(r.get("nrmse_median_ci_low", np.nan)) for r in subset], dtype=float)
            nrmse_high = np.asarray([float(r.get("nrmse_median_ci_high", np.nan)) for r in subset], dtype=float)

            order = np.argsort(budgets)
            budgets = budgets[order]
            nrmse = nrmse[order]
            kmax = kmax[order]
            nrmse_low = nrmse_low[order]
            nrmse_high = nrmse_high[order]

            style = algorithm_styles[alg]
            valid_band = (
                np.isfinite(nrmse_low)
                & np.isfinite(nrmse_high)
                & (nrmse_low > 0.0)
                & (nrmse_high > 0.0)
            )
            if np.any(valid_band):
                ax.fill_between(
                    budgets[valid_band],
                    nrmse_low[valid_band],
                    nrmse_high[valid_band],
                    color=style["color"],
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=1,
                )

            ax.loglog(
                budgets,
                nrmse,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=4,
                label=alg_name,
            )

            for x_val, y_val, k_val in zip(budgets, nrmse, kmax):
                if np.isfinite(k_val):
                    ax.annotate(
                        f"K={int(np.rint(k_val))}",
                        xy=(x_val, y_val),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=style["color"],
                        alpha=0.9,
                    )

        reference_budgets = budgets_reference[
            np.isfinite(budgets_reference) & (budgets_reference > 0.0)
        ]
        if reference_budgets.size == 0:
            raise ValueError("budgets_reference must contain at least one positive finite value.")
        reference_anchor = float(reference_budgets[0])
        ax.loglog(reference_budgets, 1.0 / np.sqrt(reference_budgets), "--", color="gray", alpha=0.6)
        ax.loglog(reference_budgets, np.sqrt(reference_anchor) / reference_budgets, "-.", color="black", alpha=0.5)
        ax.set_xlabel("Common query budget")
        ax.set_ylabel("Median normalized RMSE")
        ax.grid(True, which="both", alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(labels), 4),
            frameon=False,
            columnspacing=1.6,
            handlelength=2.2,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output_path, dpi=250, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


__all__ = [
    "AerCountSampler",
    "BAE_KIND",
    "CalibrationResult",
    "DEFAULT_AE_REFERENCE_KS",
    "DEFAULT_ROUTING_METHOD",
    "INITIAL_LAYOUT",
    "NOISE_PROFILE_BASELINE",
    "NOISE_PROFILE_MILD",
    "NOISE_PROFILE_PROJECTED",
    "NOISE_PROFILE_REALISTIC",
    "PHYSICAL_BACKEND_NAME",
    "TRANSPILER_OPTIMIZATION_LEVEL",
    "TRANSPILER_SEED",
    "aggregate_budget_rows",
    "bootstrap_mean_ci",
    "bootstrap_median_ci",
    "build_ae_pass_manager",
    "build_estimation_problem",
    "build_large_problem",
    "build_large_state_preparation",
    "build_noise_model",
    "build_problem_with_true_amplitude",
    "build_solver",
    "build_state_preparation",
    "calibrate_effective_T",
    "choose_transpilation_plan",
    "circuit_cache_key",
    "construct_measured_circuit",
    "estimate_at_budget",
    "estimate_prob_from_sampler",
    "extract_trace",
    "ideal_good_probability",
    "kmax_at_budget",
    "plot_budget_panels",
    "print_budget_summary",
    "print_final_summary",
    "save_csv",
    "time_to_budget",
    "true_amplitude",
]
