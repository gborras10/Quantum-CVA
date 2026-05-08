from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_algorithms import EstimationProblem

matplotlib.use("Agg")

from quantum_cva.amplitude_estimation.experiments.circuits import (
    construct_measured_circuit,
    construct_metadata_query_circuit,
    patch_construct_circuit,
)
from quantum_cva.amplitude_estimation.experiments.cva import (
    build_6q_cva_problem_bundle,
)
from quantum_cva.amplitude_estimation.experiments.hardware import analyze_amplification
from quantum_cva.amplitude_estimation.experiments.io import (
    load_csv,
    load_json,
    save_csv,
    save_json,
)
from quantum_cva.amplitude_estimation.experiments.plotting import (
    bootstrap_ci_errorbar,
    plot_budget_summary,
    plot_final_runtime_scatter_from_budget_rows,
    power_fit_anchor_y0,
)
from quantum_cva.amplitude_estimation.experiments.problems import (
    bundle_from_problem,
    count_good_states,
    true_amplitude,
)
from quantum_cva.amplitude_estimation.experiments.samplers import (
    ContrastDecaySampler,
    FastIdealAmplificationSampler,
    apply_contrast_decay,
    ReplayCountSampler,
    count_good_from_counts,
    extract_result_counts,
    ideal_amplified_good_probability,
    ideal_good_probability_for_circuit,
)
from quantum_cva.amplitude_estimation.experiments.statistics import (
    aggregate_budget_summary,
)
from quantum_cva.amplitude_estimation.experiments.traces import (
    extract_trace,
    rows_at_budgets,
    trace_rows_from_result,
)
from quantum_cva.algorithms.third_party.standalone_bae_hardware import (
    SamplerCountsAdapter,
)
from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta
from quantum_cva.algorithms.proposed_algorithms.cabiae_known_t import CABIQAE
from quantum_cva.amplitude_estimation.experiments.solvers import build_solver


def _synthetic_bundle(*, post_scale: float = 1.0):
    state_preparation = QuantumCircuit(3, name="synthetic_A")
    state_preparation.ry(0.72, 0)
    state_preparation.ry(0.51, 1)
    state_preparation.ry(0.65, 2)
    state_preparation.cx(0, 2)
    state_preparation.ry(-0.21, 2)
    state_preparation.cx(0, 2)

    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=[0, 1, 2],
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "") == "111",
        post_processing=lambda amplitude: post_scale * float(amplitude),
    )
    problem.grover_operator = problem.grover_operator
    return bundle_from_problem(
        problem,
        target_name="synthetic",
        good_bitstring="111",
    )


def test_good_state_semantics_are_exact_111() -> None:
    bundle = _synthetic_bundle()

    assert bundle.good_bitstring == "111"
    assert bundle.problem.is_good_state("111")
    assert not bundle.problem.is_good_state("011")
    assert not bundle.problem.is_good_state("1")
    assert count_good_states({"111": 4, "011": 9}, problem=bundle.problem) == 4
    assert np.isclose(true_amplitude(bundle.problem, "111"), bundle.true_amplitude)


def test_construct_measured_circuit_multi_objective_metadata() -> None:
    bundle = _synthetic_bundle()

    circuit0 = construct_measured_circuit(bundle.problem, 0)
    circuit1 = construct_measured_circuit(bundle.problem, 1)

    assert circuit0.num_clbits == 3
    assert circuit1.num_clbits == 3
    assert circuit0.metadata["grover_power"] == 0
    assert circuit1.metadata["grover_power"] == 1
    assert circuit1.metadata["amplification_factor"] == 3


def test_patch_construct_circuit_caches_by_k_and_measurement() -> None:
    bundle = _synthetic_bundle()
    solver = type("DummySolver", (), {})()
    shared_cache = {}
    patch_construct_circuit(solver, source="test", circuit_cache=shared_cache)

    first = solver.construct_circuit(bundle.problem, 2, measurement=True)
    second = solver.construct_circuit(bundle.problem, 2, measurement=True)
    third = solver.construct_circuit(bundle.problem, 2, measurement=False)

    another_solver = type("DummySolver", (), {})()
    patch_construct_circuit(
        another_solver,
        source="test2",
        circuit_cache=shared_cache,
    )
    fourth = another_solver.construct_circuit(bundle.problem, 2, measurement=True)

    assert first is second
    assert first is fourth
    assert third is not first
    assert first.metadata["grover_power"] == 2
    assert third.metadata["grover_power"] == 2
    assert solver._construct_circuit_metrics["construct_circuit_cache_hits"] == 1
    assert solver._construct_circuit_metrics["construct_circuit_cache_misses"] == 2
    assert another_solver._construct_circuit_metrics["construct_circuit_cache_hits"] == 1
    assert another_solver._construct_circuit_metrics["construct_circuit_cache_misses"] == 0
    assert len(shared_cache) == 2


def test_metadata_only_construct_circuit_avoids_full_query_body() -> None:
    bundle = _synthetic_bundle()
    solver = type("DummySolver", (), {})()
    patch_construct_circuit(
        solver,
        source="test",
        construction_mode="metadata_only",
    )

    circuit = solver.construct_circuit(bundle.problem, 5, measurement=True)
    direct = construct_metadata_query_circuit(bundle.problem, 5, measurement=True)

    assert circuit.metadata["grover_power"] == 5
    assert circuit.metadata["construction_mode"] == "metadata_only"
    assert circuit.count_ops() == direct.count_ops()
    assert "ry" not in circuit.count_ops()
    assert "cx" not in circuit.count_ops()


def test_replay_sampler_counts_only_good_bitstring() -> None:
    bundle = _synthetic_bundle()
    circuit = construct_measured_circuit(bundle.problem, 0)

    sampler = ReplayCountSampler({0: 1.0}, bundle, seed=123)
    counts = extract_result_counts(sampler.run([circuit], shots=25).result(), 0)
    assert count_good_from_counts(counts, bundle) == 25

    sampler = ReplayCountSampler({0: 0.0}, bundle, seed=123)
    counts = extract_result_counts(sampler.run([circuit], shots=25).result(), 0)
    assert count_good_from_counts(counts, bundle) == 0
    assert "111" in counts


def test_contrast_decay_sampler_supports_three_bit_counts() -> None:
    bundle = _synthetic_bundle()
    circuit = construct_measured_circuit(bundle.problem, 0)

    sampler = ContrastDecaySampler(bundle, T=None, seed=123)
    counts = extract_result_counts(sampler.run([circuit], shots=31).result(), 0)

    assert sum(counts.values()) == 31
    assert all(len(bitstring) == 3 for bitstring in counts)


def test_fast_ideal_sampler_supports_exact_three_bit_good_state() -> None:
    state_preparation = QuantumCircuit(3, name="deterministic_good_A")
    state_preparation.x([0, 1, 2])
    problem = EstimationProblem(
        state_preparation=state_preparation,
        objective_qubits=[0, 1, 2],
        is_good_state=lambda bitstr: str(bitstr).replace(" ", "") == "111",
    )
    problem.grover_operator = problem.grover_operator
    bundle = bundle_from_problem(
        problem,
        target_name="synthetic",
        good_bitstring="111",
    )
    circuit = construct_measured_circuit(bundle.problem, 3)

    assert ideal_amplified_good_probability(bundle.true_amplitude, 3) == 1.0

    sampler = FastIdealAmplificationSampler(bundle, T=None, seed=123)
    counts = extract_result_counts(sampler.run([circuit], shots=25).result(), 0)

    assert count_good_from_counts(counts, bundle) == 25
    assert "111" in counts

    adapter = SamplerCountsAdapter(sampler)
    direct_counts = adapter.counts_for_grover_power(3, 25)
    assert count_good_from_counts(direct_counts, bundle) == 25


class _FakeIterativeResult:
    powers = [0, 1]
    estimate_intervals = [[0.0, 1.0], [0.2, 0.4], [0.3, 0.5]]
    estimation = 0.4
    confidence_interval = (0.25, 0.55)
    num_state_prep_calls = 40


class _FakeBAEResult:
    estimation = 0.31
    confidence_interval = (0.2, 0.5)
    num_state_prep_calls = 80
    powers = []
    circuit_depths = []
    history = {
        "queries": [20, 80],
        "estimations": [0.2, 0.31],
        "controls": [0, 1],
    }


class _FakeELFResult:
    elf_layers = [0, 1]
    estimate_intervals = [[0.0, 1.0], [0.1, 0.3], [0.2, 0.4]]
    estimation = 0.3
    confidence_interval = (0.2, 0.4)
    num_state_prep_calls = 4


def test_extract_trace_variants_and_processed_rows() -> None:
    bundle = _synthetic_bundle(post_scale=10.0)

    q, estimates, amp = extract_trace("biqae", _FakeIterativeResult(), 10)
    assert q.tolist() == [10.0, 40.0]
    assert estimates.tolist() == [0.30000000000000004, 0.4]
    assert amp.tolist() == [1.0, 3.0]

    q_bae, estimates_bae, amp_bae = extract_trace("bae", _FakeBAEResult(), 20)
    assert q_bae.tolist() == [20.0, 80.0]
    assert estimates_bae.tolist() == [0.2, 0.31]
    assert amp_bae.tolist() == [1.0, 3.0]

    q_elf, _, amp_elf = extract_trace("elf_qae", _FakeELFResult(), 1)
    assert q_elf.tolist() == [1.0, 4.0]
    assert amp_elf.tolist() == [1.0, 3.0]

    rows, final = trace_rows_from_result(
        _FakeIterativeResult(),
        bundle=bundle,
        algorithm="biqae",
        algorithm_labels={"biqae": "BIQAE"},
        repetition=2,
        n_shots=10,
        elapsed_wall_seconds=1.5,
        run_kind="ideal_simulation",
    )
    assert rows[0]["target_name"] == "synthetic"
    assert np.isclose(rows[0]["processed_estimate"], 3.0)
    assert np.isclose(final["processed_estimate"], 4.0)


def test_io_statistics_and_plot_smoke(tmp_path: Path) -> None:
    rows = [
        {
            "run_kind": "test",
            "repetition": idx,
            "algorithm": "BIQAE",
            "algorithm_key": "biqae",
            "budget": 10 * (idx + 1),
            "query_budget": 10 * (idx + 1),
            "query_budget_actual": 10 * (idx + 1),
            "estimate": 0.2,
            "abs_error": 0.1,
            "normalized_abs_error": 0.5 / (idx + 1),
            "normalized_sq_error": 0.25,
            "grover_power": idx,
            "k_max_budget": idx,
            "amplification_factor": 2 * idx + 1,
            "a_true": 0.4,
            "runtime_wall_seconds": 0.1 * (idx + 1),
            "time_to_budget_seconds": 0.1 * (idx + 1),
            "target_name": "amplitude",
            "processed_true_value": 0.4,
            "processed_estimate": 0.2,
            "processed_abs_error": 0.2,
            "processed_relative_error": 0.5,
        }
        for idx in range(3)
    ]
    json_path = tmp_path / "config.json"
    csv_path = tmp_path / "rows.csv"
    save_json({"rows": len(rows), "bad_float": float("nan")}, json_path)
    save_csv(rows, csv_path)

    assert load_json(json_path)["rows"] == 3
    assert len(load_csv(csv_path)) == 3

    budget_rows = rows_at_budgets(rows, [10, 20, 30], run_kind="test")
    summary = aggregate_budget_summary(
        budget_rows,
        total_repetitions=3,
        group_by_budget=True,
        bootstrap_samples=10,
    )
    assert summary
    assert "processed_relative_error_median_ci_low" in summary[0]
    assert "processed_relative_error_median_ci_high" in summary[0]

    plot_budget_summary(summary, output_path=tmp_path / "budget.png")
    plot_final_runtime_scatter_from_budget_rows(
        budget_rows,
        output_path=tmp_path / "scatter.png",
        x_kind="queries",
    )
    assert (tmp_path / "budget.png").exists()
    assert (tmp_path / "scatter.png").exists()


def test_query_scaling_guides_use_power_fit_anchor() -> None:
    x_values = np.asarray([1.0, 10.0, 100.0])
    y_values = np.asarray([1.0, 0.01, 0.01])

    y0 = power_fit_anchor_y0(x_values, y_values)
    fit_slope, fit_log_intercept = np.polyfit(np.log(x_values), np.log(y_values), deg=1)
    expected = float(np.exp(fit_log_intercept) * x_values[0] ** fit_slope)

    assert np.isclose(y0, expected)
    assert not np.isclose(y0, float(np.nanmedian(y_values)))


def test_bootstrap_ci_errorbar_matches_toy_convention() -> None:
    yerr = bootstrap_ci_errorbar(
        np.asarray([10.0, 10.0]),
        np.asarray([8.0, -5.0]),
        np.asarray([13.0, 12.0]),
    )

    assert np.allclose(yerr[:, 0], [2.0, 3.0])
    assert np.allclose(yerr[:, 1], [9.5, 2.0])


def test_noise_floor_defaults_to_legacy_half_and_can_be_overridden() -> None:
    assert np.isclose(apply_contrast_decay(0.9, 0, 10.0), 0.5 + np.exp(-0.1) * 0.4)
    assert np.isclose(
        apply_contrast_decay(0.9, 0, 10.0, baseline=0.125),
        0.125 + np.exp(-0.1) * (0.9 - 0.125),
    )

    latent = CABIQAELatentTheta(
        epsilon_target=0.1,
        alpha=0.05,
        noise_model="exponential_contrast",
        T_known=10.0,
        noise_floor=0.125,
    )
    known_t = CABIQAE(
        epsilon_target=0.1,
        alpha=0.05,
        noise_model="exponential_contrast",
        T_known=10.0,
        noise_floor=0.125,
    )

    assert np.isclose(latent.noise_floor, 0.125)
    assert np.isclose(known_t.noise_floor, 0.125)
    assert np.isclose(
        latent._ideal_to_obs_prob(0.9, 0),
        0.125 + np.exp(-0.1) * (0.9 - 0.125),
    )
    assert np.isclose(
        known_t._ideal_to_obs_prob(0.9, 0),
        0.125 + np.exp(-0.1) * (0.9 - 0.125),
    )
    assert np.isclose(latent._obs_to_ideal_prob(latent._ideal_to_obs_prob(0.9, 0), 0), 0.9)
    assert np.isclose(known_t._obs_to_ideal_prob(known_t._ideal_to_obs_prob(0.9, 0), 0), 0.9)


def test_amplification_calibration_can_fit_noise_floor_and_robust_k_visible() -> None:
    bundle = _synthetic_bundle()
    baseline = 0.18
    prefactor = 0.92
    t_eff = 11.0
    shots = 200_000
    count_rows = []
    for k in range(12):
        circuit = construct_measured_circuit(bundle.problem, k)
        p_ideal = ideal_good_probability_for_circuit(circuit, bundle)
        amplification_factor = 2 * k + 1
        p_observed = baseline + prefactor * np.exp(-amplification_factor / t_eff) * (
            p_ideal - baseline
        )
        count_rows.append(
            {
                "grover_power": k,
                "shots": shots,
                "good_counts": int(round(float(np.clip(p_observed, 0.0, 1.0)) * shots)),
            }
        )

    points, summary, _ = analyze_amplification(
        count_rows,
        bundle,
        {"readout_denom": 1.0},
        contrast_baseline="fit",
        min_ideal_offset=0.1,
        min_baseline_fit_points=4,
    )

    assert summary["contrast_baseline_mode"] == "fitted"
    assert np.isclose(summary["contrast_baseline"], baseline, atol=0.02)
    assert summary["k_visible"] <= summary["k_contrast_fit_max"]
    assert summary["k_signal_from_baseline"] >= summary["k_visible"]
    assert any(point["visible_by_contrast"] for point in points)


def test_cabiqae_can_disable_hard_k_cap_without_disabling_fisher_scheduler() -> None:
    solver, bayes = build_solver(
        "cabiqae_latentt",
        sampler=None,
        epsilon_target=0.1,
        alpha=0.05,
        t_eff=6.0,
        cap_kappa=1.0,
        disable_hard_k_cap=True,
    )

    assert bayes
    assert solver._noise_model == "exponential_contrast"
    assert solver._use_noise_cap is True
    assert solver._k_cap() > 10**20

    capped_solver, _ = build_solver(
        "cabiqae_latentt",
        sampler=None,
        epsilon_target=0.1,
        alpha=0.05,
        t_eff=6.0,
        cap_kappa=1.0,
        disable_hard_k_cap=False,
    )
    assert capped_solver._k_cap() == 2


def test_6q_cva_builder_smoke_if_artifacts_available() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "cva_pricing_pipeline"
        / "multi_asset"
        / "6q_instance"
        / "cva_pricing_multi_asset"
        / "quantum"
        / "full_cva_pipeline.py"
    )
    if not config_path.exists():
        pytest.skip("6q config module is not present.")

    spec = importlib.util.spec_from_file_location("full_cva_pipeline_6q", config_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"6q config module cannot be imported: {exc}")

    try:
        bundle = build_6q_cva_problem_bundle(module.CONFIG, repo_root=repo_root)
    except FileNotFoundError as exc:
        pytest.skip(f"6q artifacts are not available: {exc}")

    assert bundle.target_name == "cva"
    assert bundle.good_bitstring == "111"
    assert bundle.problem.objective_qubits == [
        bundle.metadata["total_state_qubits"],
        bundle.metadata["total_state_qubits"] + 1,
        bundle.metadata["total_state_qubits"] + 2,
    ]
