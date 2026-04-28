from __future__ import annotations

import pathlib
import sys


def _bootstrap_src_path() -> None:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from quantum_cva.multi_asset.pipeline_cfg.cfg_utilities import (
    AssetConfig,
    BackendNoiseConfig,
    ClassicalConfig,
    FinalCVAConfig,
    InstrumentConfig,
    MarketDataConfig,
    PipelineConfig,
    PipelinePathConfig,
    PositiveExposureTrainingConfig,
    QCBMTrainingConfig,
    QuantumProblemConfig,
    ScalarCRCTrainingConfig,
    TrainingOutputConfig,
    build_arg_parser,
    run_pipeline,
)


# ======================================================================
# Full 6q CVA pipeline configuration
# ======================================================================

CONFIG = PipelineConfig(
    market_data=MarketDataConfig(
        discount_curve_path="data/loaded_market_data/discount_curve.xlsx",
        credit_data_path="data/loaded_market_data/iberdrola_data.xlsx",
        historical_series_path="data/loaded_market_data/time_series.xlsx",
        atm_vol_surfaces_path="data/loaded_market_data/vol_surfaces.xlsx",
        valuation_date="2026-03-15",
        flat_interest_rate_override=None,
    ),
    assets=(
        AssetConfig(symbol=".STOXX50E", dividend_yield=0.0224722),
        AssetConfig(symbol=".SSMI", dividend_yield=0.0316306),
    ),
    instruments=(
        InstrumentConfig(
            kind="call",
            asset_symbol=".STOXX50E",
            quantity=1.0,
            multiplier=4.0,
            strike=4500.0,
            maturity_years=3.0 / 12.0,
        ),
        InstrumentConfig(
            kind="put",
            asset_symbol=".SSMI",
            quantity=-1.0,
            multiplier=2.0,
            strike=12500.0,
            maturity_years=6.0 / 12.0,
        ),
    ),
    classical=ClassicalConfig(
        m_time=2,
        n_paths=100_000,
        seed=105,
        antithetic=True,
        moment_match=True,
        replications=1,
        replication_seed=12345,
        fine_time_grid_size=50,
        grid_convergence_min_bits=1,
        grid_convergence_max_bits=6,
        grid_limit_bits=8,
        n_sigma=3.0,
        payoff_repr="left",
        flattening_order="time_major",
        recovery_rate_cva=0.415,
        cds_pay_freq=4,
    ),
    quantum=QuantumProblemConfig(
        n_bits_per_asset=(2, 2),
    ),
    backend_noise=BackendNoiseConfig(
        backend_name="ibm_basquecountry",
        runtime_channel="ibm_cloud",
        use_fractional_gates=True,
        noise_snapshot_iso_utc="2026-04-07T12:10:00+00:00",
        simulator_method="density_matrix",
        simulator_seed=20260407,
        transpilation_opt_level=3,
        seed_transpiler=1234,
        readout_quantile=0.95,
        local_2q_quantile=0.95,
        relax_if_needed=True,
        approximation_degree=1.0,
    ),
    qcbm_training=QCBMTrainingConfig(
        optimizer="SPSA",
        topology="qcbm_heavyhex6",
        entangler="rzz",
        n_layers=6,
        shots=60_000,
        n_iters=500,
        resamplings=3,
        dirichlet_alpha=1.0,
        eps_cost=1e-9,
        init_scale=0.01,
        theta_seed=42,
        cost_seed=None,
        spsa_trust_region=True,
        spsa_blocking=False,
        spsa_regularization=0.01,
        target_magnitude=None,
        kl_eval_runs=10,
        kl_eval_shots=100_000,
        kl_eval_seed_base=42,
        statevector_reference_path=(
            "data/multi_asset/6q_instance/quantum/training/qcbm/statevector/"
            "training_qcbm_heavyhex6_6lay.npz"
        ),
    ),
    crca_default_training=ScalarCRCTrainingConfig(
        optimizer="SPSA",
        topology="crca2",
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        m_time=2,
        n_price=0,
        n_layers=1,
        shots=60_000,
        n_iters=70,
        resamplings=3,
        init_scale=0.10,
        theta_seed=42,
        shot_seed=355,
        last_avg=25,
        second_order=True,
        blocking=True,
        trust_region=True,
        target_magnitude=None,
    ),
    crca_discount_training=ScalarCRCTrainingConfig(
        optimizer="SPSA",
        topology="crca2",
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        m_time=2,
        n_price=0,
        n_layers=1,
        shots=60_000,
        n_iters=70,
        resamplings=3,
        init_scale=0.10,
        theta_seed=42,
        shot_seed=355,
        last_avg=25,
        second_order=True,
        blocking=True,
        trust_region=True,
        target_magnitude=None,
    ),
    crca_exposure_training=PositiveExposureTrainingConfig(
        optimizer="SPSA",
        topology="heavy_hex_star",
        m_time=2,
        n_price=4,
        n_layers=2,
        shots=60_000,
        init_scale=0.01,
        theta_seed=12,
        shot_seed=355,
        repeat_seed_stride=10007,
        use_two_stage=False,
        stage1_mode="l2",
        stage2_mode="support_aware",
        stage1_maxiter=120,
        stage2_maxiter=150,
        single_stage_mode="l2",
        single_stage_maxiter=270,
        stage1_resamplings={0: 3, 40: 4, 90: 5},
        stage2_resamplings={0: 4, 120: 6, 280: 8},
        single_stage_resamplings={0: 3, 120: 4, 260: 5, 420: 6},
        stage1_eval_repeats=2,
        stage2_eval_repeats=3,
        single_stage_eval_repeats=2,
        stage1_target_magnitude=0.08,
        stage2_target_magnitude=0.05,
        single_stage_target_magnitude=0.08,
        stage1_second_order=True,
        stage2_second_order=False,
        single_stage_second_order=True,
        stage1_blocking=True,
        stage2_blocking=True,
        single_stage_blocking=True,
        stage1_trust_region=True,
        stage2_trust_region=True,
        single_stage_trust_region=True,
        stage1_regularization=0.02,
        stage2_regularization=0.0,
        single_stage_regularization=0.02,
        stage1_hessian_delay=40,
        stage2_hessian_delay=0,
        single_stage_hessian_delay=60,
        last_avg=40,
        init_selection_eval_repeats=5,
        postselect_top_k=12,
        postselect_eval_repeats=5,
        target_threshold=1e-10,
        relative_eps=1e-4,
        lambda_pos=10.0,
        lambda_zero=15.0,
        lambda_l2_mix=25.0,
        robust_rel_clip=2.5,
        robust_rel_huber_delta=0.6,
        robust_zero_huber_delta=0.02,
        thermal_relaxation_requested=True,
        use_statevector_warmstart=True,
        warmstart_path=(
            "data/multi_asset/6q_instance/quantum/training/crca/"
            "positive_exposure/training_heavy_hex_star.npz"
        ),
    ),
    final_cva=FinalCVAConfig(
        run_qae=True,
        run_iqae=True,
        qae_num_eval_qubits=6,
        iqae_epsilon_target=1e-3,
        iqae_alpha=0.05,
        classical_reference_grid_bits=2,
        statevector_backend_name="statevector",
    ),
    paths=PipelinePathConfig(
        instance_name="6q_instance",
        benchmark_relative_path=(
            "data/multi_asset/6q_instance/benchmark/three_asset_instance.npz"
        ),
        qcbm_training_relative_path=(
            "data/multi_asset/6q_instance/quantum/training/qcbm/shots/"
            "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz"
        ),
        crca_default_training_relative_path=(
            "data/multi_asset/6q_instance/quantum/training/crca/"
            "default_probabilities/training_crca2_shots_backend_noise_snapshot.npz"
        ),
        crca_discount_training_relative_path=(
            "data/multi_asset/6q_instance/quantum/training/crca/"
            "discount_factors/training_crca2_shots_backend_noise_snapshot.npz"
        ),
        crca_exposure_training_relative_path=(
            "data/multi_asset/6q_instance/quantum/training/crca/"
            "positive_exposure/training_heavy_hex_star_shots_backend_noise_snapshot.npz"
        ),
        results_dir_relative_path=(
            "data/multi_asset/6q_instance/quantum/pipeline_runs/beta_full_pipeline"
        ),
    ),
    output=TrainingOutputConfig(
        save_npz=True,
        save_qpy=True,
        save_plots=False,
        show_plots=False,
        checkpoint_tol=1e-15,
    ),
)


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(
        CONFIG,
        stage=args.stage,
        resume=args.resume,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()