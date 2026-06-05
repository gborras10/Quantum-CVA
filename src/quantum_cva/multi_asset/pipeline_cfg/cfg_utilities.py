from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

StageName = Literal[
    "all",
    "classical",
    "train_qcbm",
    "train_crca_default",
    "train_crca_discount",
    "train_crca_exposure",
    "final_statevector_cva",
]


@dataclass(frozen=True, slots=True)
class MarketDataConfig:
    discount_curve_path: str = "data/loaded_market_data/discount_curve.xlsx"
    credit_data_path: str = "data/loaded_market_data/iberdrola_data.xlsx"
    historical_series_path: str = "data/loaded_market_data/time_series.xlsx"
    atm_vol_surfaces_path: str = "data/loaded_market_data/vol_surfaces.xlsx"
    valuation_date: str = "2026-03-15"
    flat_interest_rate_override: float | None = None


@dataclass(frozen=True, slots=True)
class AssetConfig:
    symbol: str
    dividend_yield: float


@dataclass(frozen=True, slots=True)
class InstrumentConfig:
    kind: Literal["call", "put", "forward"]
    asset_symbol: str
    quantity: float
    multiplier: float
    strike: float
    maturity_years: float


@dataclass(frozen=True, slots=True)
class ClassicalConfig:
    m_time: int = 2
    n_paths: int = 100_000
    seed: int = 105
    antithetic: bool = True
    moment_match: bool = True
    replications: int = 1
    replication_seed: int = 12345
    fine_time_grid_size: int = 50
    grid_convergence_min_bits: int = 1
    grid_convergence_max_bits: int = 6
    grid_limit_bits: int = 8
    n_sigma: float = 3.0
    payoff_repr: Literal["left", "mid", "right"] = "left"
    flattening_order: Literal["time_major", "price_major"] = "time_major"
    recovery_rate_cva: float = 0.415
    cds_pay_freq: int = 4


@dataclass(frozen=True, slots=True)
class QuantumProblemConfig:
    n_bits_per_asset: tuple[int, ...] = (2, 2)

    @property
    def n_underlying_qubits(self) -> int:
        return int(sum(self.n_bits_per_asset))


@dataclass(frozen=True, slots=True)
class BackendNoiseConfig:
    backend_name: str = "ibm_basquecountry"
    runtime_channel: str = "ibm_cloud"
    use_fractional_gates: bool = True
    noise_snapshot_iso_utc: str = "2026-04-07T12:10:00+00:00"
    simulator_method: str = "density_matrix"
    simulator_seed: int = 20260407
    transpilation_opt_level: int = 3
    seed_transpiler: int = 1234
    readout_quantile: float = 0.95
    local_2q_quantile: float = 0.95
    relax_if_needed: bool = True
    approximation_degree: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingOutputConfig:
    save_npz: bool = True
    save_qpy: bool = True
    save_plots: bool = False
    show_plots: bool = False
    checkpoint_tol: float = 1e-15


@dataclass(frozen=True, slots=True)
class QCBMTrainingConfig:
    optimizer: Literal["SPSA"] = "SPSA"
    topology: str = "qcbm_heavyhex6"
    entangler: Literal["rzz", "rxx", "cz"] = "rzz"
    n_layers: int = 6
    shots: int = 60_000
    n_iters: int = 500
    resamplings: int | dict[int, int] = 3
    dirichlet_alpha: float = 1.0
    eps_cost: float = 1e-9
    init_scale: float = 0.01
    theta_seed: int = 42
    cost_seed: int | None = None
    spsa_trust_region: bool = True
    spsa_blocking: bool = False
    spsa_regularization: float = 0.01
    target_magnitude: float | None = None
    kl_eval_runs: int = 10
    kl_eval_shots: int = 100_000
    kl_eval_seed_base: int = 42
    statevector_reference_path: str = (
        "data/multi_asset/6q_instance/quantum/training/qcbm/statevector/"
        "training_qcbm_heavyhex6_6lay.npz"
    )


@dataclass(frozen=True, slots=True)
class ScalarCRCTrainingConfig:
    optimizer: Literal["SPSA"] = "SPSA"
    topology: str = "crca2"
    ansatz_type: str = "native_tree"
    native_1q_order: tuple[str, ...] = ("rx", "rz")
    m_time: int = 2
    n_price: int = 0
    n_layers: int = 1
    shots: int = 60_000
    n_iters: int = 70
    resamplings: int | dict[int, int] = 3
    init_scale: float = 0.10
    theta_seed: int = 42
    shot_seed: int = 355
    last_avg: int = 25
    second_order: bool = True
    blocking: bool = True
    trust_region: bool = True
    target_magnitude: float | None = None


@dataclass(frozen=True, slots=True)
class PositiveExposureTrainingConfig:
    optimizer: Literal["SPSA"] = "SPSA"
    topology: str = "heavy_hex_star"
    m_time: int = 2
    n_price: int = 4
    n_layers: int = 2
    shots: int = 60_000
    init_scale: float = 0.01
    theta_seed: int = 12
    shot_seed: int = 355
    repeat_seed_stride: int = 10007
    use_two_stage: bool = False
    stage1_mode: Literal["l2", "support_aware", "support_aware_robust"] = "l2"
    stage2_mode: Literal["l2", "support_aware", "support_aware_robust"] = (
        "support_aware"
    )
    stage1_maxiter: int = 120
    stage2_maxiter: int = 150
    single_stage_mode: Literal["l2", "support_aware", "support_aware_robust"] = "l2"
    single_stage_maxiter: int = 270
    stage1_resamplings: int | dict[int, int] = field(
        default_factory=lambda: {0: 3, 40: 4, 90: 5}
    )
    stage2_resamplings: int | dict[int, int] = field(
        default_factory=lambda: {0: 4, 120: 6, 280: 8}
    )
    single_stage_resamplings: int | dict[int, int] = field(
        default_factory=lambda: {0: 3, 120: 4, 260: 5, 420: 6}
    )
    stage1_eval_repeats: int = 2
    stage2_eval_repeats: int = 3
    single_stage_eval_repeats: int = 2
    stage1_target_magnitude: float | None = 0.08
    stage2_target_magnitude: float | None = 0.05
    single_stage_target_magnitude: float | None = 0.08
    stage1_second_order: bool = True
    stage2_second_order: bool = False
    single_stage_second_order: bool = True
    stage1_blocking: bool = True
    stage2_blocking: bool = True
    single_stage_blocking: bool = True
    stage1_trust_region: bool = True
    stage2_trust_region: bool = True
    single_stage_trust_region: bool = True
    stage1_regularization: float = 0.02
    stage2_regularization: float = 0.0
    single_stage_regularization: float = 0.02
    stage1_hessian_delay: int = 40
    stage2_hessian_delay: int = 0
    single_stage_hessian_delay: int = 60
    last_avg: int = 40
    init_selection_eval_repeats: int = 5
    postselect_top_k: int = 12
    postselect_eval_repeats: int = 5
    target_threshold: float = 1e-10
    relative_eps: float = 1e-4
    lambda_pos: float = 10.0
    lambda_zero: float = 15.0
    lambda_l2_mix: float = 25.0
    robust_rel_clip: float = 2.5
    robust_rel_huber_delta: float = 0.6
    robust_zero_huber_delta: float = 0.02
    thermal_relaxation_requested: bool = True
    use_statevector_warmstart: bool = True
    warmstart_path: str = (
        "data/multi_asset/6q_instance/quantum/training/crca/"
        "positive_exposure/training_heavy_hex_star.npz"
    )


@dataclass(frozen=True, slots=True)
class FinalCVAConfig:
    run_qae: bool = True
    run_iqae: bool = True
    qae_num_eval_qubits: int = 6
    iqae_epsilon_target: float = 1e-3
    iqae_alpha: float = 0.05
    classical_reference_grid_bits: int = 2
    statevector_backend_name: Literal["statevector"] = "statevector"


@dataclass(frozen=True, slots=True)
class PipelinePathConfig:
    instance_name: str = "6q_instance"
    benchmark_relative_path: str = (
        "data/multi_asset/6q_instance/benchmark/three_asset_instance.npz"
    )
    qcbm_training_relative_path: str = (
        "data/multi_asset/6q_instance/quantum/training/qcbm/shots/"
        "6LAY_training_qcbm_heavyhex6_shots_backend_noise_snapshot.npz"
    )
    crca_default_training_relative_path: str = (
        "data/multi_asset/6q_instance/quantum/training/crca/"
        "default_probabilities/training_crca2_shots_backend_noise_snapshot.npz"
    )
    crca_discount_training_relative_path: str = (
        "data/multi_asset/6q_instance/quantum/training/crca/"
        "discount_factors/training_crca2_shots_backend_noise_snapshot.npz"
    )
    crca_exposure_training_relative_path: str = (
        "data/multi_asset/6q_instance/quantum/training/crca/"
        "positive_exposure/training_heavy_hex_star_shots_backend_noise_snapshot.npz"
    )
    results_dir_relative_path: str = (
        "data/multi_asset/6q_instance/quantum/pipeline_runs/beta_full_pipeline"
    )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    market_data: MarketDataConfig
    assets: tuple[AssetConfig, ...]
    instruments: tuple[InstrumentConfig, ...]
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    quantum: QuantumProblemConfig = field(default_factory=QuantumProblemConfig)
    backend_noise: BackendNoiseConfig = field(default_factory=BackendNoiseConfig)
    qcbm_training: QCBMTrainingConfig = field(default_factory=QCBMTrainingConfig)
    crca_default_training: ScalarCRCTrainingConfig = field(
        default_factory=ScalarCRCTrainingConfig
    )
    crca_discount_training: ScalarCRCTrainingConfig = field(
        default_factory=ScalarCRCTrainingConfig
    )
    crca_exposure_training: PositiveExposureTrainingConfig = field(
        default_factory=PositiveExposureTrainingConfig
    )
    final_cva: FinalCVAConfig = field(default_factory=FinalCVAConfig)
    paths: PipelinePathConfig = field(default_factory=PipelinePathConfig)
    output: TrainingOutputConfig = field(default_factory=TrainingOutputConfig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 6q multi-asset CVA pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "classical",
            "train_qcbm",
            "train_crca_default",
            "train_crca_discount",
            "train_crca_exposure",
            "final_statevector_cva",
        ],
        default="all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose required output artifact already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute a stage even if its output artifact already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and stage order without executing.",
    )
    return parser


def run_pipeline(
    config: PipelineConfig,
    *,
    stage: StageName = "all",
    resume: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    runner = CVAPipelineRunner(config)
    return runner.run(stage=stage, resume=resume, force=force, dry_run=dry_run)


class CVAPipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.repo_root = find_repo_root()
        self.results_dir = self._resolve(config.paths.results_dir_relative_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        stage: StageName = "all",
        resume: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._validate_config()
        stages = self._stage_sequence(stage)
        config_path = self.results_dir / "pipeline_config.json"
        _write_json(config_path, _jsonable(self.config))

        print("============================================================")
        print("BETA full CVA pipeline")
        print("============================================================")
        print(f"repo_root={self.repo_root}")
        print(f"stage={stage}")
        print(f"stages={stages}")
        print(f"resume={resume}, force={force}, dry_run={dry_run}")
        print(f"config_snapshot={config_path}")

        if dry_run:
            return {"dry_run": True, "stages": stages}

        results: dict[str, Any] = {}
        for stage_name in stages:
            print("\n============================================================")
            print(f"Running stage: {stage_name}")
            print("============================================================")
            t0 = time.perf_counter()

            if stage_name == "classical":
                result = self.run_classical(resume=resume, force=force)
            elif stage_name == "train_qcbm":
                result = self.train_qcbm_noise_shots(resume=resume, force=force)
            elif stage_name == "train_crca_default":
                result = self.train_scalar_crca_noise_shots(
                    task="default_probabilities",
                    resume=resume,
                    force=force,
                )
            elif stage_name == "train_crca_discount":
                result = self.train_scalar_crca_noise_shots(
                    task="discount_factors",
                    resume=resume,
                    force=force,
                )
            elif stage_name == "train_crca_exposure":
                result = self.train_positive_exposure_noise_shots(
                    resume=resume,
                    force=force,
                )
            elif stage_name == "final_statevector_cva":
                result = self.run_final_statevector_cva()
            else:
                raise ValueError(f"Unknown stage: {stage_name}")

            elapsed = float(time.perf_counter() - t0)
            result = dict(result)
            result["stage_elapsed_s"] = elapsed
            results[stage_name] = result
            _write_json(self.results_dir / "pipeline_summary.json", _jsonable(results))
            print(f"[OK] stage={stage_name} elapsed_s={elapsed:.2f}")

        return results

    def run_classical(self, *, resume: bool, force: bool) -> dict[str, Any]:
        out_path = self._benchmark_path()
        if resume and out_path.exists() and not force:
            print(f"[SKIP] benchmark already exists: {out_path}")
            return {"skipped": True, "path": str(out_path)}

        from quantum_cva.multi_asset.classical.classical_cva.cva_auxiliar_functions import (
            build_survival_from_cds,
        )
        from quantum_cva.multi_asset.classical.classical_cva.classical_cont_cva import (
            ContinuousUnderlyingCvaEngine,
        )
        from quantum_cva.multi_asset.classical.classical_cva.classical_discrete_cva import (
            DiscreteUnderlyingCvaEngine,
        )
        from quantum_cva.multi_asset.classical.probability_and_underlying.multi_asset_dynamics_utils import (
            simulate_multi_asset_gbm,
        )
        from quantum_cva.multi_asset.classical.probability_and_underlying.piecewise_volatility_utils import (
            build_integrated_covariance_grid,
            build_piecewise_sigma_grid,
            build_residual_volatility_function_for_underlying,
        )
        from quantum_cva.multi_asset.instruments.derivatives import Call, Forward, Put
        from quantum_cva.multi_asset.instruments.market_data import MarketData

        cfg = self.config
        classical = cfg.classical
        assets = list(cfg.assets)
        underlyings = [asset.symbol for asset in assets]
        div_yields = [float(asset.dividend_yield) for asset in assets]
        d = len(assets)

        market_data = MarketData.load(
            discount_curve_path=str(self._resolve(cfg.market_data.discount_curve_path)),
            credit_data_path=str(self._resolve(cfg.market_data.credit_data_path)),
            historical_series_path=str(self._resolve(cfg.market_data.historical_series_path)),
            atm_vol_surfaces_path=str(self._resolve(cfg.market_data.atm_vol_surfaces_path)),
            valuation_date=cfg.market_data.valuation_date,
        )

        if cfg.market_data.flat_interest_rate_override is None:
            p0_func: Callable[[float], float] = lambda u: market_data.discount_factor(u)
            r = float(-np.log(p0_func(1.0)))
        else:
            r = float(cfg.market_data.flat_interest_rate_override)
            p0_func = lambda u, r=r: float(np.exp(-r * float(u)))

        s0_list = market_data.get_spot_vector(underlyings)
        atm_vol_curves = market_data.get_atm_vol_curves(underlyings=underlyings)
        rho = market_data.get_log_return_correlation(underlyings)
        mu_list = [r - dy for dy in div_yields]

        maturity_max = max(float(inst.maturity_years) for inst in cfg.instruments)
        m_time = int(classical.m_time)
        m_dates = 2**m_time
        t = np.linspace(0.0, maturity_max, m_dates + 1, dtype=float)[1:]

        rng = np.random.default_rng(int(classical.seed))
        z = rng.standard_normal(size=(int(classical.n_paths), m_dates, d))

        r_cva = float(classical.recovery_rate_cva)
        r_cds = float(market_data.recovery_rate)
        lgd = 1.0 - r_cva
        cds_tenors_years, cds_spreads = market_data.get_cds_curve()
        survival_breaks, survival_lambdas, survival_curve, q_interval = (
            build_survival_from_cds(
                P0=p0_func,
                tenors=cds_tenors_years,
                spreads=cds_spreads,
                R_cds=r_cds,
                pay_freq=int(classical.cds_pay_freq),
            )
        )

        sigma_grid = build_piecewise_sigma_grid(
            atm_vol_curves=atm_vol_curves,
            underlyings=underlyings,
            sim_times=t,
        )
        integrated_covariance_grid = build_integrated_covariance_grid(
            atm_vol_curves=atm_vol_curves,
            underlyings=underlyings,
            sim_times=t,
            rho=rho,
        )
        s_by_time = simulate_multi_asset_gbm(
            S0=s0_list,
            mu=mu_list,
            sigma=sigma_grid,
            rho=rho,
            t=t,
            Z=z,
            antithetic=bool(classical.antithetic),
            moment_match=bool(classical.moment_match),
            replications=int(classical.replications),
            replication_seed=int(classical.replication_seed),
            pathwise=True,
            integrated_covariances=integrated_covariance_grid,
        )

        sigma_funcs = {
            symbol: build_residual_volatility_function_for_underlying(
                atm_vol_curves=atm_vol_curves,
                underlying=symbol,
            )
            for symbol in underlyings
        }
        asset_index = {symbol: idx for idx, symbol in enumerate(underlyings)}
        instruments = []
        for inst_cfg in cfg.instruments:
            idx = asset_index[inst_cfg.asset_symbol]
            common = dict(
                asset_idx=idx,
                quantity=float(inst_cfg.quantity),
                multiplier=float(inst_cfg.multiplier),
                K=float(inst_cfg.strike),
                T=float(inst_cfg.maturity_years),
            )
            if inst_cfg.kind == "call":
                instruments.append(
                    Call(**common, sigma_func=sigma_funcs[inst_cfg.asset_symbol])
                )
            elif inst_cfg.kind == "put":
                instruments.append(
                    Put(**common, sigma_func=sigma_funcs[inst_cfg.asset_symbol])
                )
            elif inst_cfg.kind == "forward":
                instruments.append(Forward(**common))
            else:
                raise ValueError(f"Unsupported instrument kind: {inst_cfg.kind}")

        continuous_engine = ContinuousUnderlyingCvaEngine(
            instruments=instruments,
            P0_func=p0_func,
            q_interval=q_interval,
            LGD=lgd,
            r=r,
        )
        t0 = time.perf_counter()
        cva_mc_continuous, cva_std_err_mc_continuous = (
            continuous_engine.cva_from_paths(S_by_time=s_by_time, t=t)
        )
        continuous_elapsed_s = float(time.perf_counter() - t0)

        def make_engine(n_bits: int | Sequence[int]) -> Any:
            if isinstance(n_bits, int):
                n_bits_list = [int(n_bits)] * d
            else:
                n_bits_list = [int(x) for x in n_bits]
            return DiscreteUnderlyingCvaEngine(
                instruments=instruments,
                P0_func=p0_func,
                q_interval=q_interval,
                LGD=lgd,
                r=r,
                n_bits=n_bits_list,
                n_sigma=float(classical.n_sigma),
                payoff_repr=classical.payoff_repr,
                order=classical.flattening_order,
                time_weights=None,
            )

        cva_by_grid_size: dict[int, float] = {}
        for n_bits in range(
            int(classical.grid_convergence_min_bits),
            int(classical.grid_convergence_max_bits) + 1,
        ):
            cva_by_grid_size[n_bits] = float(
                make_engine(n_bits).cva_from_paths_discretized(
                    S_by_time=s_by_time,
                    t=t,
                    return_blocks=False,
                )
            )

        grid_sizes = np.array(sorted(cva_by_grid_size), dtype=int)
        cva_values = np.array([cva_by_grid_size[int(n)] for n in grid_sizes], dtype=float)

        limit_engine = make_engine(int(classical.grid_limit_bits))
        t_limit_0 = time.perf_counter()
        cva_limit, grid_inf, p_joint_t_inf, v_joint_t_inf, p_target_inf, w_t_inf = (
            limit_engine.cva_from_paths_discretized(
                S_by_time=s_by_time,
                t=t,
                return_blocks=True,
            )
        )
        limit_elapsed_s = float(time.perf_counter() - t_limit_0)

        t_fine = np.linspace(0.0, maturity_max, int(classical.fine_time_grid_size), dtype=float)[1:]
        z_fine = rng.standard_normal(size=(int(classical.n_paths), t_fine.size, d))
        sigma_grid_fine = build_piecewise_sigma_grid(
            atm_vol_curves=atm_vol_curves,
            underlyings=underlyings,
            sim_times=t_fine,
        )
        integrated_covariance_grid_fine = build_integrated_covariance_grid(
            atm_vol_curves=atm_vol_curves,
            underlyings=underlyings,
            sim_times=t_fine,
            rho=rho,
        )
        s_by_time_fine = simulate_multi_asset_gbm(
            S0=s0_list,
            mu=mu_list,
            sigma=sigma_grid_fine,
            rho=rho,
            t=t_fine,
            Z=z_fine,
            antithetic=bool(classical.antithetic),
            moment_match=bool(classical.moment_match),
            replications=int(classical.replications),
            replication_seed=int(classical.replication_seed),
            pathwise=True,
            integrated_covariances=integrated_covariance_grid_fine,
        )

        small_engine = make_engine(cfg.quantum.n_bits_per_asset)
        (
            cva_small,
            grid_small,
            p_joint_t_small,
            v_joint_t_small,
            p_target_small,
            w_t_small,
        ) = small_engine.cva_from_paths_discretized(
            S_by_time=s_by_time_fine,
            t=t_fine,
            t_output=t,
            return_blocks=True,
        )
        p_t_small = small_engine.discount_factors_on_grid(t)
        q_t_small = small_engine.default_increments_on_grid(t)

        c_p = float(np.max(p_t_small))
        c_q = float(np.max(q_t_small))
        c_v = float(np.max(v_joint_t_small))
        cva_small_scaled = small_engine.cva_from_discrete_blocks(
            P_joint_t=p_joint_t_small,
            v_joint_t=v_joint_t_small,
            t=t,
            C_p=c_p,
            C_q=c_q,
            C_v=c_v,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            t=t,
            p_t=p_t_small,
            q_t=q_t_small,
            edges_list=np.array(grid_small.edges_list, dtype=object),
            rep_list=np.array(grid_small.rep_list, dtype=object),
            n_bins=np.array(grid_small.n_bins, dtype=int),
            N_joint=np.array(grid_small.N_joint, dtype=int),
            P_joint_t=p_joint_t_small,
            v_joint_t=v_joint_t_small,
            p_target=p_target_small,
            w_t=w_t_small,
            C_p=np.array(c_p),
            C_q=np.array(c_q),
            C_v=np.array(c_v),
            cva_mc_continuous=np.array(cva_mc_continuous),
            cva_std_err_mc_continuous=np.array(cva_std_err_mc_continuous),
            cva_limit=np.array(cva_limit),
            cva_small=np.array(cva_small),
            cva_small_scaled=np.array(cva_small_scaled),
            grid_sizes=grid_sizes,
            cva_by_grid_size_values=cva_values,
            S0=np.array(s0_list, dtype=float),
            K=np.array([inst.strike for inst in cfg.instruments], dtype=float),
            sigma=np.array(sigma_grid, dtype=float),
            mu=np.array(mu_list, dtype=float),
            r=np.array(r),
            T=np.array(maturity_max),
            rho=np.array(rho, dtype=float),
            R_cva=np.array(r_cva),
            R_cds=np.array(r_cds),
            LGD=np.array(lgd),
            M=np.array(m_dates),
            n_bits_small=np.array(cfg.quantum.n_bits_per_asset, dtype=int),
            n_sigma=np.array(float(classical.n_sigma)),
            survival_breaks=np.array(survival_breaks, dtype=float),
            survival_lambdas=np.array(survival_lambdas, dtype=float),
            survival_on_exposure_grid=np.array(
                [survival_curve(float(ti)) for ti in t],
                dtype=float,
            ),
            pipeline_config=np.array(_jsonable(cfg), dtype=object),
        )

        result = {
            "path": str(out_path),
            "cva_mc_continuous": float(cva_mc_continuous),
            "cva_std_err_mc_continuous": float(cva_std_err_mc_continuous),
            "continuous_elapsed_s": continuous_elapsed_s,
            "cva_limit": float(cva_limit),
            "limit_elapsed_s": limit_elapsed_s,
            "cva_small": float(cva_small),
            "cva_small_scaled": float(cva_small_scaled),
            "C_p": c_p,
            "C_q": c_q,
            "C_v": c_v,
            "grid_sizes": grid_sizes.tolist(),
            "cva_by_grid_size_values": cva_values.tolist(),
        }
        print(json.dumps(_jsonable(result), indent=2))
        return result

    def train_qcbm_noise_shots(self, *, resume: bool, force: bool) -> dict[str, Any]:
        out_path = self._qcbm_path()
        if resume and out_path.exists() and not force:
            print(f"[SKIP] QCBM training already exists: {out_path}")
            return {"skipped": True, "path": str(out_path)}

        _require_optimizer(self.config.qcbm_training.optimizer)

        from qiskit import qpy
        from qiskit_aer import AerSimulator
        from qiskit_algorithms.optimizers import SPSA
        from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
            MLQcbmCircuit,
        )
        from quantum_cva.quantum_hardware_utilities.layout_utils import (
            select_best_layout,
            summarize_circuit,
        )

        cfg = self.config
        tr = cfg.qcbm_training
        backend_ctx = self._backend_context(thermal_relaxation=False)
        benchmark = _load_npz(self._benchmark_path())
        p_target = _as_1d_float(benchmark["p_target"])
        p_target = p_target / float(np.sum(p_target))
        n_qubits = _num_qubits_from_dim(p_target.size, "QCBM target")

        chosen_layout, layout_score, layout_meta = select_best_layout(
            backend_ctx.real_backend,
            topology=tr.topology,
            length=n_qubits,
            readout_quantile=cfg.backend_noise.readout_quantile,
            local_2q_quantile=cfg.backend_noise.local_2q_quantile,
            relax_if_needed=cfg.backend_noise.relax_if_needed,
        )
        effective_topology = layout_meta["selected_topology"]

        noisy_backend = AerSimulator(
            method=cfg.backend_noise.simulator_method,
            noise_model=backend_ctx.noise_model,
            coupling_map=backend_ctx.coupling_map,
            seed_simulator=cfg.backend_noise.simulator_seed,
        )
        qcbm = MLQcbmCircuit(
            n_qubits=n_qubits,
            n_layers=tr.n_layers,
            name="G_p_shots_backend_noise_snapshot_transpiled",
            entangler=tr.entangler,
            topology=effective_topology,
            backend=noisy_backend,
            transpile_backend=backend_ctx.real_backend,
            noise_model=backend_ctx.noise_model,
            basis_gates=list(backend_ctx.noise_model.basis_gates),
            simulation_method=cfg.backend_noise.simulator_method,
            optimization_level=cfg.backend_noise.transpilation_opt_level,
            initial_layout=[int(q) for q in chosen_layout],
            layout_method="trivial",
            routing_method="none",
            seed_transpiler=cfg.backend_noise.seed_transpiler,
        )
        summarize_circuit(qcbm._tqc, label="Transpiled QCBM Circuit")

        target_entropy = float(
            -np.sum(p_target * np.log(np.clip(p_target, tr.eps_cost, 1.0)))
        )
        theta_statevector = None
        kl_eval_values: list[float] = []
        statevector_reference_path = self._resolve(tr.statevector_reference_path)
        if statevector_reference_path.exists():
            sv_data = np.load(statevector_reference_path, allow_pickle=True)
            if "theta_star" in sv_data:
                theta_statevector = _as_1d_float(sv_data["theta_star"])
                _assert_size("statevector QCBM theta", theta_statevector, qcbm.n_params)
                for run_idx in range(int(tr.kl_eval_runs)):
                    seed = int(tr.kl_eval_seed_base + run_idx)
                    p_sv_noise = qcbm.probabilities(
                        theta_statevector,
                        shots=int(tr.kl_eval_shots),
                        seed=seed,
                    )
                    kl_eval_values.append(
                        float(qcbm.metrics(p_target, p_sv_noise, eps=tr.eps_cost)["kl"])
                    )

        cost = qcbm.cost_fn(
            p_target,
            eps=float(tr.eps_cost),
            shots=int(tr.shots),
            seed=tr.cost_seed,
            rescaled=True,
            smoothing="dirichlet",
            alpha=float(tr.dirichlet_alpha),
        )
        rng = np.random.default_rng(int(tr.theta_seed))
        theta0 = float(tr.init_scale) * rng.standard_normal(qcbm.n_params).astype(float)
        cost_history: list[float] = []
        theta_history: list[np.ndarray] = [theta0.copy()]
        best = {"fx": float("inf"), "x": theta0.copy()}
        iter_times: list[float] = []
        live_log: list[dict[str, float]] = []
        training_t0: float | None = None
        last_callback_t: float | None = None

        def callback(nfev, x, fx, stepsize, accepted):
            nonlocal training_t0, last_callback_t
            now = time.perf_counter()
            if training_t0 is None:
                training_t0 = now
            iter_dt = 0.0 if last_callback_t is None else now - last_callback_t
            last_callback_t = now
            fx = float(fx)
            x_arr = np.asarray(x, dtype=float).copy()
            cost_history.append(fx)
            theta_history.append(x_arr)
            iter_times.append(iter_dt)
            if fx < best["fx"]:
                best["fx"] = fx
                best["x"] = x_arr.copy()
            mean_dt = float(np.mean(iter_times[1:])) if len(iter_times) > 1 else 0.0
            live_log.append(
                {
                    "iter": float(len(cost_history)),
                    "nfev": float(nfev),
                    "fx": fx,
                    "best_fx": float(best["fx"]),
                    "iter_time_s": float(iter_dt),
                    "mean_iter_time_s": mean_dt,
                    "stepsize": float(stepsize),
                    "accepted": float(bool(accepted)),
                }
            )
            print(
                f"[QCBM iter {len(cost_history):5d}] fx={fx:.6e} "
                f"| best={best['fx']:.6e} | nfev={int(nfev):6d}"
            )

        print("Calibrating QCBM SPSA hyperparameters...")
        if tr.target_magnitude is None:
            learning_rate, perturbation = SPSA.calibrate(cost, theta0)
        else:
            learning_rate, perturbation = SPSA.calibrate(
                cost,
                theta0,
                target_magnitude=float(tr.target_magnitude),
            )

        opt = SPSA(
            maxiter=int(tr.n_iters),
            learning_rate=learning_rate,
            perturbation=perturbation,
            resamplings=tr.resamplings,
            blocking=bool(tr.spsa_blocking),
            callback=callback,
            trust_region=bool(tr.spsa_trust_region),
            regularization=float(tr.spsa_regularization),
        )
        t0 = time.perf_counter()
        res = opt.minimize(fun=cost, x0=theta0)
        elapsed_time = float(time.perf_counter() - t0)

        theta_last = _as_1d_float(res.x)
        theta_star = _as_1d_float(best["x"])
        p0 = qcbm.probabilities(theta0, shots=int(tr.shots), seed=None)
        p_last = qcbm.probabilities(theta_last, shots=int(tr.shots), seed=None)
        p_star = qcbm.probabilities(theta_star, shots=int(tr.shots), seed=None)
        cost_history_arr = np.asarray(cost_history, dtype=float)
        if cost_history_arr.size == 0:
            cost_history_arr = np.asarray([float(res.fun)], dtype=float)
        theta_history_arr = np.asarray(theta_history, dtype=float)
        best_so_far = np.minimum.accumulate(cost_history_arr)
        best_idx = np.flatnonzero(
            np.r_[
                True,
                best_so_far[1:]
                < best_so_far[:-1] - float(cfg.output.checkpoint_tol),
            ]
        )
        metrics_best = qcbm.metrics(p_target, p_star, eps=tr.eps_cost)
        metrics_last = qcbm.metrics(p_target, p_last, eps=tr.eps_cost)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.output.save_qpy:
            with open(
                out_path.with_name("trained_qcbm_circuit_shots_backend_noise_snapshot.qpy"),
                "wb",
            ) as f:
                qpy.dump(qcbm._tqc, f)

        t1_values = [
            _safe_property_value(backend_ctx.backend_props, int(q), "T1")
            for q in chosen_layout
        ]
        t2_values = [
            _safe_property_value(backend_ctx.backend_props, int(q), "T2")
            for q in chosen_layout
        ]
        readout_values = [
            _safe_property_value(backend_ctx.backend_props, int(q), "readout_error")
            for q in chosen_layout
        ]
        np.savez(
            out_path,
            theta_star=theta_star,
            theta_last=theta_last,
            theta_init=theta0,
            cost_history=cost_history_arr,
            kl_history=np.maximum(cost_history_arr, 1e-15),
            best_so_far=best_so_far,
            best_idx=best_idx,
            theta_history=theta_history_arr,
            p_target=p_target,
            p_init=p0,
            p_last=p_last,
            p_star=p_star,
            elapsed_time=np.float64(elapsed_time),
            best_cost=np.float64(best["fx"]),
            target_entropy=np.float64(target_entropy),
            kl_eval_values_statevector=np.asarray(kl_eval_values, dtype=float),
            kl_eval_mean_statevector=np.float64(
                np.mean(kl_eval_values) if kl_eval_values else np.nan
            ),
            kl_eval_std_statevector=np.float64(
                np.std(kl_eval_values, ddof=1) if len(kl_eval_values) > 1 else 0.0
            ),
            n_iters=np.int64(tr.n_iters),
            shots=np.int64(tr.shots),
            epsilon=np.float64(tr.eps_cost),
            theta_seed=np.int64(tr.theta_seed),
            init_scale=np.float64(tr.init_scale),
            resamplings=np.array(tr.resamplings, dtype=object),
            dirichlet_alpha=np.float64(tr.dirichlet_alpha),
            n_qubits=np.int64(n_qubits),
            n_layers=np.int64(tr.n_layers),
            backend_name=np.array(cfg.backend_noise.backend_name),
            requested_topology=np.array(tr.topology),
            effective_topology=np.array(effective_topology),
            chosen_layout=np.array(chosen_layout, dtype=int),
            layout_score=np.float64(layout_score),
            fallback_used=np.bool_(layout_meta["fallback_used"]),
            tried_layout_search=np.array(layout_meta["tried"], dtype=object),
            transpiled_depth=np.int64(qcbm._tqc.depth()),
            transpiled_size=np.int64(qcbm._tqc.size()),
            transpiled_ops=np.array(dict(qcbm._tqc.count_ops()), dtype=object),
            metrics_best=np.array(metrics_best, dtype=object),
            metrics_last=np.array(metrics_last, dtype=object),
            noise_snapshot_iso_utc=np.array(backend_ctx.snapshot_dt_utc.isoformat()),
            backend_props_last_update=np.array(
                str(getattr(backend_ctx.backend_props, "last_update_date", None))
            ),
            simulator_seed=np.int64(cfg.backend_noise.simulator_seed),
            simulator_method=np.array(cfg.backend_noise.simulator_method),
            noise_basis_gates=np.array(
                list(backend_ctx.noise_model.basis_gates),
                dtype=object,
            ),
            used_noise_fallback=np.bool_(backend_ctx.used_noise_fallback),
            snapshot_t1_chosen_layout=np.array(t1_values, dtype=object),
            snapshot_t2_chosen_layout=np.array(t2_values, dtype=object),
            snapshot_readout_error_chosen_layout=np.array(readout_values, dtype=object),
            pipeline_config=np.array(_jsonable(cfg), dtype=object),
        )
        return {
            "path": str(out_path),
            "elapsed_time": elapsed_time,
            "best_cost": float(best["fx"]),
            "metrics_best": metrics_best,
            "n_params": int(qcbm.n_params),
            "effective_topology": str(effective_topology),
            "chosen_layout": [int(q) for q in chosen_layout],
        }

    def train_scalar_crca_noise_shots(
        self,
        *,
        task: Literal["default_probabilities", "discount_factors"],
        resume: bool,
        force: bool,
    ) -> dict[str, Any]:
        if task == "default_probabilities":
            tr = self.config.crca_default_training
            out_path = self._crca_default_path()
            target_key = "q_t"
            scale_key = "C_q"
            task_label = "default_probability"
            qpy_name = "trained_crca_default_probabilities_shots_backend_noise_snapshot.qpy"
        else:
            tr = self.config.crca_discount_training
            out_path = self._crca_discount_path()
            target_key = "p_t"
            scale_key = "C_p"
            task_label = "discount_factor"
            qpy_name = "trained_crca_discount_factors_shots_backend_noise_snapshot.qpy"

        if resume and out_path.exists() and not force:
            print(f"[SKIP] CRCA {task} training already exists: {out_path}")
            return {"skipped": True, "path": str(out_path)}

        _require_optimizer(tr.optimizer)

        from qiskit import ClassicalRegister, qpy
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_algorithms.optimizers import SPSA
        from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
            CrcaCircuit,
        )
        from quantum_cva.quantum_hardware_utilities.layout_utils import (
            select_best_layout,
            summarize_circuit,
        )

        cfg = self.config
        benchmark = _load_npz(self._benchmark_path())
        f_target = _as_1d_float(benchmark[target_key]) / float(benchmark[scale_key])
        backend_ctx = self._backend_context(thermal_relaxation=False)

        crca = CrcaCircuit(
            m_time=tr.m_time,
            n_price=tr.n_price,
            n_layers=tr.n_layers,
            ansatz_type=tr.ansatz_type,
            native_1q_order=tr.native_1q_order,
            name=f"crca_{task_label}_shots_backend_noise_snapshot",
        )
        _assert_size(f"CRCA {task} target", f_target, crca.dim_controls)

        chosen_layout, layout_score, layout_meta = select_best_layout(
            backend_ctx.real_backend,
            topology=tr.topology,
            length=int(crca.qc.num_qubits),
            readout_quantile=cfg.backend_noise.readout_quantile,
            local_2q_quantile=cfg.backend_noise.local_2q_quantile,
            relax_if_needed=cfg.backend_noise.relax_if_needed,
        )
        effective_topology = layout_meta["selected_topology"]
        noisy_backend = AerSimulator(
            method=cfg.backend_noise.simulator_method,
            noise_model=backend_ctx.noise_model,
            coupling_map=backend_ctx.coupling_map,
            seed_simulator=cfg.backend_noise.simulator_seed,
        )
        pm = generate_preset_pass_manager(
            backend=backend_ctx.real_backend,
            optimization_level=cfg.backend_noise.transpilation_opt_level,
            initial_layout=[int(q) for q in chosen_layout],
            seed_transpiler=cfg.backend_noise.seed_transpiler,
            approximation_degree=cfg.backend_noise.approximation_degree,
        )
        tqc_ansatz = pm.run(crca.qc)
        tqc_eval = pm.run(crca.qc_eval)
        qc_meas = crca.qc_eval.copy()
        c_ctrl = ClassicalRegister(crca.n_controls, "c")
        c_a = ClassicalRegister(1, "ca")
        qc_meas.add_register(c_ctrl, c_a)
        qc_meas.measure(crca._control_qubit_indices, c_ctrl)
        qc_meas.measure([crca._ancilla_qubit_index], c_a)
        tqc_eval_meas = pm.run(qc_meas)

        crca._backend = noisy_backend
        crca._tqc_eval_meas = tqc_eval_meas
        crca._tqc_eval_meas_param_set = set(tqc_eval_meas.parameters)
        crca._n_clbits = len(tqc_eval_meas.clbits)
        crca._ctrl_clbit_indices, crca._a_clbit_index = crca._extract_clbit_indices(
            tqc_eval_meas
        )

        summarize_circuit(tqc_ansatz, label=f"CRCA {task} ansatz transpiled")
        summarize_circuit(tqc_eval, label=f"CRCA {task} eval transpiled")
        summarize_circuit(tqc_eval_meas, label=f"CRCA {task} eval+measure transpiled")

        cost = crca.cost_fn(f_target, shots=int(tr.shots), seed=int(tr.shot_seed))
        rng = np.random.default_rng(int(tr.theta_seed))
        theta0 = float(tr.init_scale) * rng.standard_normal(crca.n_params).astype(float)
        f0 = crca.function_values(theta0, shots=int(tr.shots), seed=int(tr.shot_seed))
        cost_history: list[float] = []
        theta_history: list[np.ndarray] = []

        def callback(nfev, x, fx, step, accepted):
            cost_history.append(float(fx))
            theta_history.append(np.asarray(x, dtype=float).copy())
            print(
                f"[CRCA {task} iter {len(cost_history):5d}] "
                f"fx={float(fx):.6e} | nfev={int(nfev):6d}"
            )

        print(f"Calibrating CRCA {task} SPSA hyperparameters...")
        if tr.target_magnitude is None:
            learning_rate, perturbation = SPSA.calibrate(cost, theta0)
        else:
            learning_rate, perturbation = SPSA.calibrate(
                cost,
                theta0,
                target_magnitude=float(tr.target_magnitude),
            )
        opt = SPSA(
            maxiter=int(tr.n_iters),
            learning_rate=learning_rate,
            perturbation=perturbation,
            resamplings=tr.resamplings,
            last_avg=int(tr.last_avg),
            second_order=bool(tr.second_order),
            blocking=bool(tr.blocking),
            trust_region=bool(tr.trust_region),
            callback=callback,
        )
        t0 = time.perf_counter()
        res = opt.minimize(fun=cost, x0=theta0)
        elapsed_time = float(time.perf_counter() - t0)
        cost_history_arr = np.asarray(cost_history, dtype=float)
        if cost_history_arr.size == 0:
            cost_history_arr = np.asarray([float(res.fun)], dtype=float)
            theta_best = _as_1d_float(res.x)
            best_fx = float(res.fun)
        else:
            best_pos = int(np.argmin(cost_history_arr))
            theta_best = _as_1d_float(theta_history[best_pos])
            best_fx = float(cost_history_arr[best_pos])

        theta_last = _as_1d_float(res.x)
        f_last = crca.function_values(theta_last, shots=int(tr.shots), seed=int(tr.shot_seed))
        f_star = crca.function_values(theta_best, shots=int(tr.shots), seed=int(tr.shot_seed))
        best_so_far = np.minimum.accumulate(cost_history_arr)
        best_idx = np.flatnonzero(
            np.r_[
                True,
                best_so_far[1:]
                < best_so_far[:-1] - float(cfg.output.checkpoint_tol),
            ]
        )
        metadata = {
            "model": "CRCA",
            "task": task_label,
            "ansatz_type": tr.ansatz_type,
            "native_1q_order": tuple(tr.native_1q_order),
            "m_time": tr.m_time,
            "n_price": tr.n_price,
            "n_controls": crca.n_controls,
            "n_layers": tr.n_layers,
            "n_parameters": crca.n_params,
            "optimizer": tr.optimizer,
            "maxiter": tr.n_iters,
            "resamplings": tr.resamplings,
            "shots": tr.shots,
            "shot_seed": tr.shot_seed,
            "backend_name": cfg.backend_noise.backend_name,
            "requested_topology": tr.topology,
            "effective_topology": effective_topology,
            "layout_score": float(layout_score),
            "fallback_used": bool(layout_meta["fallback_used"]),
            "noise_snapshot_iso_utc": backend_ctx.snapshot_dt_utc.isoformat(),
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.output.save_qpy:
            with open(out_path.with_name(qpy_name), "wb") as f:
                qpy.dump(tqc_eval_meas, f)

        t1_values = [
            _safe_property_value(backend_ctx.backend_props, int(q), "T1")
            for q in chosen_layout
        ]
        t2_values = [
            _safe_property_value(backend_ctx.backend_props, int(q), "T2")
            for q in chosen_layout
        ]
        readout_values = [
            _safe_property_value(backend_ctx.backend_props, int(q), "readout_error")
            for q in chosen_layout
        ]
        save_kwargs = {
            "theta_star": theta_best,
            "theta_last": theta_last,
            "theta_init": theta0,
            "cost_history": cost_history_arr,
            "best_so_far": best_so_far,
            "best_idx": best_idx,
            "f_target": f_target,
            "f_init_shots": f0,
            "f_last_shots": f_last,
            "f_star_shots": f_star,
            "elapsed_time": np.float64(elapsed_time),
            "best_cost": np.float64(best_fx),
            "final_cost": np.float64(np.mean((f_last - f_target) ** 2)),
            scale_key: np.float64(float(benchmark[scale_key])),
            "n_iters": np.int64(tr.n_iters),
            "shots": np.int64(tr.shots),
            "resamplings": np.array(tr.resamplings, dtype=object),
            "theta_seed": np.int64(tr.theta_seed),
            "shot_seed": np.int64(tr.shot_seed),
            "transpile_optimization_level": np.int64(
                cfg.backend_noise.transpilation_opt_level
            ),
            "seed_transpiler": np.int64(cfg.backend_noise.seed_transpiler),
            "backend_name": np.array(cfg.backend_noise.backend_name),
            "requested_topology": np.array(tr.topology),
            "effective_topology": np.array(effective_topology),
            "chosen_layout": np.array(chosen_layout, dtype=int),
            "layout_score": np.float64(layout_score),
            "fallback_used": np.bool_(layout_meta["fallback_used"]),
            "tried_layout_search": np.array(layout_meta["tried"], dtype=object),
            "transpiled_ansatz_depth": np.int64(tqc_ansatz.depth()),
            "transpiled_eval_depth": np.int64(tqc_eval.depth()),
            "transpiled_eval_meas_depth": np.int64(tqc_eval_meas.depth()),
            "noise_snapshot_iso_utc": np.array(backend_ctx.snapshot_dt_utc.isoformat()),
            "backend_props_last_update": np.array(
                str(getattr(backend_ctx.backend_props, "last_update_date", None))
            ),
            "simulator_seed": np.int64(cfg.backend_noise.simulator_seed),
            "simulator_method": np.array(cfg.backend_noise.simulator_method),
            "noise_basis_gates": np.array(
                list(backend_ctx.noise_model.basis_gates),
                dtype=object,
            ),
            "used_noise_fallback": np.bool_(backend_ctx.used_noise_fallback),
            "snapshot_t1_chosen_layout": np.array(t1_values, dtype=object),
            "snapshot_t2_chosen_layout": np.array(t2_values, dtype=object),
            "snapshot_readout_error_chosen_layout": np.array(readout_values, dtype=object),
            "metadata": np.array(metadata, dtype=object),
            "pipeline_config": np.array(_jsonable(cfg), dtype=object),
        }
        np.savez(out_path, **save_kwargs)
        return {
            "path": str(out_path),
            "elapsed_time": elapsed_time,
            "best_cost": best_fx,
            "final_l2": float(np.mean((f_last - f_target) ** 2)),
            "n_params": int(crca.n_params),
            "effective_topology": str(effective_topology),
            "chosen_layout": [int(q) for q in chosen_layout],
        }

    def train_positive_exposure_noise_shots(
        self,
        *,
        resume: bool,
        force: bool,
    ) -> dict[str, Any]:
        out_path = self._crca_exposure_path()
        if resume and out_path.exists() and not force:
            print(f"[SKIP] CRCA exposure training already exists: {out_path}")
            return {"skipped": True, "path": str(out_path)}

        tr = self.config.crca_exposure_training
        _require_optimizer(tr.optimizer)

        from qiskit import qpy
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
            CrcaCircuit,
        )
        from quantum_cva.quantum_hardware_utilities.layout_utils import (
            select_best_layout,
            summarize_circuit,
        )

        cfg = self.config
        benchmark = _load_npz(self._benchmark_path())
        c_v = float(benchmark["C_v"])
        f_target_2d = np.asarray(benchmark["v_joint_t"], dtype=float) / c_v
        f_target = _as_1d_float(f_target_2d)

        service, real_backend = _load_real_backend(cfg.backend_noise)
        snapshot_dt_utc = _parse_snapshot_datetime(
            cfg.backend_noise.noise_snapshot_iso_utc
        )
        backend_props = real_backend.properties(datetime=snapshot_dt_utc)
        if backend_props is None:
            raise RuntimeError(
                "Could not retrieve historical backend properties for "
                f"{cfg.backend_noise.noise_snapshot_iso_utc}."
            )

        logical_crca = CrcaCircuit(
            m_time=tr.m_time,
            n_price=tr.n_price,
            n_layers=tr.n_layers,
            ansatz_type=tr.topology,
        )
        _assert_size("CRCA positive exposure target", f_target, logical_crca.dim_controls)

        snapshot_view = _BackendSnapshotView(real_backend, backend_props)
        chosen_layout, layout_score, layout_meta = select_best_layout(
            snapshot_view,
            topology=tr.topology,
            length=int(logical_crca.qc.num_qubits),
            readout_quantile=cfg.backend_noise.readout_quantile,
            local_2q_quantile=cfg.backend_noise.local_2q_quantile,
            relax_if_needed=cfg.backend_noise.relax_if_needed,
        )
        effective_topology = layout_meta["selected_topology"]

        full_coupling = getattr(real_backend, "coupling_map", None)
        if full_coupling is None:
            full_coupling = real_backend.configuration().coupling_map
        full_edges = (
            list(full_coupling.get_edges())
            if hasattr(full_coupling, "get_edges")
            else list(full_coupling)
        )
        n_total_logical_qubits = int(logical_crca.qc.num_qubits)
        local_layout = list(range(n_total_logical_qubits))
        local_coupling_map = _build_local_coupling_map(full_edges, chosen_layout)
        if not local_coupling_map:
            raise RuntimeError("Local coupling map is empty for selected layout.")

        props_for_noise = backend_props
        injected_frequency_count = 0
        if tr.thermal_relaxation_requested:
            props_for_noise, injected_frequency_count = _inject_missing_frequencies(
                backend_props,
                real_backend,
            )

        subset_props = _subset_backend_properties(props_for_noise, chosen_layout)
        thermal_relaxation_effective = bool(tr.thermal_relaxation_requested)
        noise_model_build = "snapshot_backend_properties"
        try:
            noise_model = NoiseModel.from_backend_properties(
                subset_props,
                thermal_relaxation=bool(tr.thermal_relaxation_requested),
            )
        except Exception as exc:
            print("[WARNING] thermal_relaxation=True failed; falling back to False.")
            print(f"[WARNING] {exc}")
            thermal_relaxation_effective = False
            noise_model_build = "snapshot_backend_properties_without_thermal_relaxation"
            noise_model = NoiseModel.from_backend_properties(
                subset_props,
                thermal_relaxation=False,
            )

        noisy_backend = AerSimulator(
            method=cfg.backend_noise.simulator_method,
            noise_model=noise_model,
            coupling_map=local_coupling_map,
            seed_simulator=cfg.backend_noise.simulator_seed,
        )
        crca = CrcaCircuit(
            m_time=tr.m_time,
            n_price=tr.n_price,
            n_layers=tr.n_layers,
            ansatz_type=effective_topology,
            name="crca_positive_exposure_heavy_hex_star_shots_backend_noise_snapshot",
        )
        pm = generate_preset_pass_manager(
            backend=noisy_backend,
            optimization_level=cfg.backend_noise.transpilation_opt_level,
            initial_layout=local_layout,
            seed_transpiler=cfg.backend_noise.seed_transpiler,
            approximation_degree=cfg.backend_noise.approximation_degree,
        )
        tqc_ansatz = pm.run(crca.qc)
        tqc_eval = pm.run(crca.qc_eval)
        tqc_eval_meas = _build_transpiled_measured_eval_circuit(crca, pm)

        crca._backend = noisy_backend
        crca._tqc_eval_meas = tqc_eval_meas
        crca._tqc_eval_meas_param_set = set(tqc_eval_meas.parameters)
        crca._n_clbits = len(tqc_eval_meas.clbits)
        crca._ctrl_clbit_indices, crca._a_clbit_index = crca._extract_clbit_indices(
            tqc_eval_meas
        )
        crca._tqc = tqc_ansatz

        summarize_circuit(tqc_ansatz, label="CRCA exposure ansatz transpiled")
        summarize_circuit(tqc_eval, label="CRCA exposure eval transpiled")
        summarize_circuit(tqc_eval_meas, label="CRCA exposure eval+measure transpiled")

        init_select_objective = _build_positive_objective(
            crca,
            f_target,
            tr,
            mode="l2",
            shots=tr.shots,
            eval_repeats=tr.init_selection_eval_repeats,
        )
        rng = np.random.default_rng(int(tr.theta_seed))
        theta0_random = float(tr.init_scale) * rng.standard_normal(crca.n_params).astype(float)
        theta0_warm = _load_warmstart_theta(
            self.repo_root,
            path=tr.warmstart_path,
            n_params_expected=crca.n_params,
            enabled=tr.use_statevector_warmstart,
        )
        l2_random = float(init_select_objective(theta0_random))
        theta0 = theta0_random.copy()
        warmstart_used = False
        warmstart_l2 = float("nan")
        if theta0_warm is not None:
            warmstart_l2 = float(init_select_objective(theta0_warm))
            if warmstart_l2 <= l2_random:
                theta0 = theta0_warm.copy()
                warmstart_used = True

        print("\nPositive exposure optimization setup:")
        print(f"shots={tr.shots}")
        print(f"use_two_stage={tr.use_two_stage}")
        print(f"warmstart_used={warmstart_used}")
        print(f"l2_random={l2_random:.6e}, warmstart_l2={warmstart_l2:.6e}")

        if tr.use_two_stage:
            stage1 = _run_positive_spsa_stage(
                stage_name="stage1",
                objective_mode=tr.stage1_mode,
                crca=crca,
                f_target=f_target,
                x0=theta0,
                shots=tr.shots,
                calibration_shots=tr.shots,
                maxiter=tr.stage1_maxiter,
                resamplings=tr.stage1_resamplings,
                eval_repeats=tr.stage1_eval_repeats,
                second_order=tr.stage1_second_order,
                blocking=tr.stage1_blocking,
                trust_region=tr.stage1_trust_region,
                regularization=tr.stage1_regularization,
                hessian_delay=tr.stage1_hessian_delay,
                calibration_target_magnitude=tr.stage1_target_magnitude,
                training_config=tr,
            )
            stage2 = _run_positive_spsa_stage(
                stage_name="stage2",
                objective_mode=tr.stage2_mode,
                crca=crca,
                f_target=f_target,
                x0=stage1["theta_best_l2"],
                shots=tr.shots,
                calibration_shots=tr.shots,
                maxiter=tr.stage2_maxiter,
                resamplings=tr.stage2_resamplings,
                eval_repeats=tr.stage2_eval_repeats,
                second_order=tr.stage2_second_order,
                blocking=tr.stage2_blocking,
                trust_region=tr.stage2_trust_region,
                regularization=tr.stage2_regularization,
                hessian_delay=tr.stage2_hessian_delay,
                calibration_target_magnitude=tr.stage2_target_magnitude,
                training_config=tr,
            )
            theta_history_arr, cost_history_arr, l2_history_arr = _merge_stage_histories(
                stage1,
                stage2,
            )
            elapsed_time = float(stage1["elapsed_s"] + stage2["elapsed_s"])
            theta_last = _as_1d_float(stage2["theta_last"])
        else:
            stage1 = _run_positive_spsa_stage(
                stage_name="single_l2",
                objective_mode=tr.single_stage_mode,
                crca=crca,
                f_target=f_target,
                x0=theta0,
                shots=tr.shots,
                calibration_shots=tr.shots,
                maxiter=tr.single_stage_maxiter,
                resamplings=tr.single_stage_resamplings,
                eval_repeats=tr.single_stage_eval_repeats,
                second_order=tr.single_stage_second_order,
                blocking=tr.single_stage_blocking,
                trust_region=tr.single_stage_trust_region,
                regularization=tr.single_stage_regularization,
                hessian_delay=tr.single_stage_hessian_delay,
                calibration_target_magnitude=tr.single_stage_target_magnitude,
                training_config=tr,
            )
            stage2 = None
            theta_history_arr = np.asarray(stage1["theta_history"], dtype=float)
            cost_history_arr = np.asarray(stage1["obj_history"], dtype=float)
            l2_history_arr = np.asarray(stage1["l2_history"], dtype=float)
            elapsed_time = float(stage1["elapsed_s"])
            theta_last = _as_1d_float(stage1["theta_last"])

        idx_best_l2_raw = int(np.argmin(l2_history_arr))
        idx_best_obj_global = int(np.argmin(cost_history_arr))
        idx_best_l2_global, best_l2_rechecked = _select_best_theta_by_recheck(
            crca,
            f_target,
            theta_history_arr,
            l2_history_arr,
            shots=tr.shots,
            top_k=tr.postselect_top_k,
            eval_repeats=tr.postselect_eval_repeats,
            training_config=tr,
        )
        theta_star = theta_history_arr[idx_best_l2_global].copy()
        best_fx = float(cost_history_arr[idx_best_obj_global])

        f0 = _evaluate_crca_function_values(
            crca,
            theta0,
            shots=tr.shots,
            seed=tr.shot_seed,
        )
        f_last = _evaluate_crca_function_values(
            crca,
            theta_last,
            shots=tr.shots,
            seed=tr.shot_seed,
        )
        f_star = _evaluate_crca_function_values(
            crca,
            theta_star,
            shots=tr.shots,
            seed=tr.shot_seed,
        )
        best_so_far = np.minimum.accumulate(np.maximum(l2_history_arr, 1e-15))
        best_idx = np.flatnonzero(
            np.r_[
                True,
                best_so_far[1:]
                < best_so_far[:-1] - float(self.config.output.checkpoint_tol),
            ]
        )
        final_l2 = _mean_squared_error(f_last, f_target)
        best_l2 = _mean_squared_error(f_star, f_target)

        metadata = {
            "model": "CRCA",
            "task": "positive_exposure",
            "optimizer": tr.optimizer,
            "use_two_stage": tr.use_two_stage,
            "shots": tr.shots,
            "m_time": tr.m_time,
            "n_price": tr.n_price,
            "n_layers": tr.n_layers,
            "n_parameters": crca.n_params,
            "backend_name": cfg.backend_noise.backend_name,
            "requested_topology": tr.topology,
            "effective_topology": effective_topology,
            "layout_score": float(layout_score),
            "fallback_used": bool(layout_meta["fallback_used"]),
            "noise_snapshot_iso_utc": snapshot_dt_utc.isoformat(),
            "thermal_relaxation_requested": tr.thermal_relaxation_requested,
            "thermal_relaxation_effective": thermal_relaxation_effective,
            "injected_frequency_count": injected_frequency_count,
            "noise_model_build": noise_model_build,
            "warmstart_used": warmstart_used,
            "warmstart_l2": warmstart_l2,
            "random_start_l2": l2_random,
            "best_l2_rechecked": best_l2_rechecked,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.output.save_qpy:
            with open(
                out_path.with_name(
                    "trained_crca_positive_exposure_circuit_shots_backend_noise_snapshot.qpy"
                ),
                "wb",
            ) as f:
                qpy.dump(tqc_eval_meas, f)

        if stage2 is None:
            stage2_obj_history = np.asarray([], dtype=float)
            stage2_l2_history = np.asarray([], dtype=float)
        else:
            stage2_obj_history = np.asarray(stage2["obj_history"], dtype=float)
            stage2_l2_history = np.asarray(stage2["l2_history"], dtype=float)

        np.savez(
            out_path,
            theta_star=theta_star,
            theta_last=theta_last,
            theta_init=theta0,
            theta_history=theta_history_arr,
            cost_history=cost_history_arr,
            l2_history=l2_history_arr,
            best_so_far=best_so_far,
            best_idx=best_idx,
            f_target=f_target,
            f_target_2d=f_target_2d,
            f_init=f0,
            f_last=f_last,
            f_star=f_star,
            C_v=np.float64(c_v),
            elapsed_time=np.float64(elapsed_time),
            best_cost=np.float64(best_fx),
            final_l2=np.float64(final_l2),
            best_l2=np.float64(best_l2),
            best_l2_rechecked=np.float64(best_l2_rechecked),
            idx_best_l2_raw=np.int64(idx_best_l2_raw),
            idx_best_l2_rechecked=np.int64(idx_best_l2_global),
            n_iters=np.int64(
                tr.stage1_maxiter + tr.stage2_maxiter
                if tr.use_two_stage
                else tr.single_stage_maxiter
            ),
            stage1_obj_history=np.asarray(stage1["obj_history"], dtype=float),
            stage1_l2_history=np.asarray(stage1["l2_history"], dtype=float),
            stage2_obj_history=stage2_obj_history,
            stage2_l2_history=stage2_l2_history,
            shots=np.int64(tr.shots),
            theta_seed=np.int64(tr.theta_seed),
            shot_seed=np.int64(tr.shot_seed),
            simulator_seed=np.int64(cfg.backend_noise.simulator_seed),
            seed_transpiler=np.int64(cfg.backend_noise.seed_transpiler),
            transpile_optimization_level=np.int64(
                cfg.backend_noise.transpilation_opt_level
            ),
            requested_topology=np.array(tr.topology),
            effective_topology=np.array(effective_topology),
            chosen_layout=np.array(chosen_layout, dtype=int),
            chosen_layout_local=np.array(local_layout, dtype=int),
            local_coupling_map=np.array(local_coupling_map, dtype=int),
            layout_score=np.float64(layout_score),
            fallback_used=np.bool_(layout_meta["fallback_used"]),
            tried_layout_search=np.array(layout_meta["tried"], dtype=object),
            transpiled_ansatz_depth=np.int64(tqc_ansatz.depth()),
            transpiled_eval_depth=np.int64(tqc_eval.depth()),
            transpiled_eval_meas_depth=np.int64(tqc_eval_meas.depth()),
            noise_snapshot_iso_utc=np.array(snapshot_dt_utc.isoformat()),
            backend_props_last_update=np.array(
                str(getattr(backend_props, "last_update_date", None))
            ),
            thermal_relaxation_requested=np.bool_(tr.thermal_relaxation_requested),
            thermal_relaxation_effective=np.bool_(thermal_relaxation_effective),
            injected_frequency_count=np.int64(injected_frequency_count),
            noise_model_build=np.array(noise_model_build),
            noise_basis_gates=np.array(list(noise_model.basis_gates), dtype=object),
            metadata=np.array(metadata, dtype=object),
            pipeline_config=np.array(_jsonable(cfg), dtype=object),
        )
        return {
            "path": str(out_path),
            "elapsed_time": elapsed_time,
            "best_cost": best_fx,
            "best_l2": best_l2,
            "best_l2_rechecked": float(best_l2_rechecked),
            "final_l2": final_l2,
            "n_params": int(crca.n_params),
            "effective_topology": str(effective_topology),
            "chosen_layout": [int(q) for q in chosen_layout],
        }

    def run_final_statevector_cva(self) -> dict[str, Any]:
        from qiskit.primitives import StatevectorSampler
        from qiskit_algorithms import (
            AmplitudeEstimation,
            EstimationProblem,
            IterativeAmplitudeEstimation,
        )
        from qiskit_aer import AerSimulator
        from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (
            QuantumCVACircuit,
        )
        from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
            CrcaCircuit,
        )
        from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
            MLQcbmCircuit,
        )

        cfg = self.config
        benchmark = _load_npz(self._benchmark_path())
        qcbm_data = _load_npz(self._qcbm_path())
        default_data = _load_npz(self._crca_default_path())
        discount_data = _load_npz(self._crca_discount_path())
        exposure_data = _load_npz(self._crca_exposure_path())

        qcbm_theta = _as_1d_float(qcbm_data["theta_star"])
        default_theta = _as_1d_float(default_data["theta_star"])
        discount_theta = _as_1d_float(discount_data["theta_star"])
        exposure_theta = _as_1d_float(exposure_data["theta_star"])

        num_qubits_time = int(cfg.classical.m_time)
        num_qubits_underlying = int(cfg.quantum.n_underlying_qubits)
        total_state_qubits = num_qubits_time + num_qubits_underlying

        qcbm_topology = _npz_str(qcbm_data, "effective_topology", cfg.qcbm_training.topology)
        qcbm_n_layers = _npz_int(qcbm_data, "n_layers", cfg.qcbm_training.n_layers)
        qcbm = MLQcbmCircuit(
            n_qubits=total_state_qubits,
            n_layers=qcbm_n_layers,
            name="qcbm_state_prep_circuit_shots_noise_theta",
            entangler=cfg.qcbm_training.entangler,
            topology=qcbm_topology,
            backend=AerSimulator(method="statevector"),
            simulation_method="statevector",
            optimization_level=0,
        )

        exposure_meta = _metadata_dict(exposure_data)
        exposure_n_layers = int(
            exposure_meta.get(
                "n_layers",
                _npz_int(exposure_data, "n_layers", cfg.crca_exposure_training.n_layers),
            )
        )
        exposure_ansatz = _npz_str(
            exposure_data,
            "effective_topology",
            cfg.crca_exposure_training.topology,
        )
        crca_exposure = CrcaCircuit(
            m_time=num_qubits_time,
            n_price=num_qubits_underlying,
            n_layers=exposure_n_layers,
            ansatz_type=exposure_ansatz,
            name="crca_positive_exposure_circuit_shots_noise_theta",
        )
        crca_default = CrcaCircuit(
            m_time=num_qubits_time,
            n_price=0,
            n_layers=_npz_int(
                default_data,
                "n_layers",
                cfg.crca_default_training.n_layers,
            ),
            ansatz_type=cfg.crca_default_training.ansatz_type,
            native_1q_order=cfg.crca_default_training.native_1q_order,
            name="crca_default_probabilities_circuit_shots_noise_theta",
        )
        crca_discount = CrcaCircuit(
            m_time=num_qubits_time,
            n_price=0,
            n_layers=_npz_int(
                discount_data,
                "n_layers",
                cfg.crca_discount_training.n_layers,
            ),
            ansatz_type=cfg.crca_discount_training.ansatz_type,
            native_1q_order=cfg.crca_discount_training.native_1q_order,
            name="crca_discount_factors_circuit_shots_noise_theta",
        )

        _assert_size("QCBM theta", qcbm_theta, qcbm.n_params)
        _assert_size("CRCA exposure theta", exposure_theta, crca_exposure.n_params)
        _assert_size("CRCA default theta", default_theta, crca_default.n_params)
        _assert_size("CRCA discount theta", discount_theta, crca_discount.n_params)

        qcbm_target = _as_1d_float(qcbm_data["p_target"])
        default_target = _as_1d_float(benchmark["q_t"]) / float(benchmark["C_q"])
        discount_target = _as_1d_float(benchmark["p_t"]) / float(benchmark["C_p"])
        exposure_target = _as_1d_float(benchmark["v_joint_t"]) / float(benchmark["C_v"])
        _assert_size("QCBM target", qcbm_target, qcbm.dim)
        _assert_size("CRCA exposure target", exposure_target, crca_exposure.dim_controls)
        _assert_size("CRCA default target", default_target, crca_default.dim_controls)
        _assert_size("CRCA discount target", discount_target, crca_discount.dim_controls)

        qcbm_probs_sv = qcbm.probabilities(qcbm_theta, shots=None, seed=None)
        qcbm_kl_sv = float(qcbm.metrics(qcbm_target, qcbm_probs_sv, eps=1e-12)["kl"])
        default_l2 = float(
            np.linalg.norm(
                crca_default.function_values(default_theta, shots=None, seed=None)
                - default_target,
                ord=2,
            )
        )
        discount_l2 = float(
            np.linalg.norm(
                crca_discount.function_values(discount_theta, shots=None, seed=None)
                - discount_target,
                ord=2,
            )
        )
        exposure_l2 = float(
            np.linalg.norm(
                crca_exposure.function_values(exposure_theta, shots=None, seed=None)
                - exposure_target,
                ord=2,
            )
        )

        quantum_cva_circuit = QuantumCVACircuit(
            num_qubits_time=num_qubits_time,
            num_qubits_underlying=num_qubits_underlying,
            qcbm_circuit=qcbm,
            crca_circuit_exposure=crca_exposure,
            crca_circuit_default_prob=crca_default,
            crca_circuit_discount_factor=crca_discount,
            recovery_rate=float(benchmark["R_cva"]),
            C_v=float(benchmark["C_v"]),
            C_p=float(benchmark["C_p"]),
            C_q=float(benchmark["C_q"]),
            name="quantum_cva_circuit_shots_noise_theta",
            backend=cfg.final_cva.statevector_backend_name,
        )
        qc_cva = quantum_cva_circuit.build_cva_circuit(
            qcbm_params=qcbm_theta,
            crca_exposure_params=exposure_theta,
            crca_default_params=default_theta,
            crca_discount_params=discount_theta,
            measured=False,
        )
        p111 = quantum_cva_circuit.prob_111(
            qcbm_params=qcbm_theta,
            crca_exposure_params=exposure_theta,
            crca_default_params=default_theta,
            crca_discount_params=discount_theta,
        )
        cva_statevector = quantum_cva_circuit.cva_from_prob(p111)
        cva_classical = _classical_reference_cva(
            benchmark,
            grid_bits=int(cfg.final_cva.classical_reference_grid_bits),
        )

        result: dict[str, Any] = {
            "p111_statevector": float(p111),
            "cva_statevector": float(cva_statevector),
            "cva_classical_reference": float(cva_classical),
            "relative_error_statevector_pct": _percent_relative_error(
                cva_statevector,
                cva_classical,
            ),
            "qcbm_kl_statevector": qcbm_kl_sv,
            "default_l2_statevector": default_l2,
            "discount_l2_statevector": discount_l2,
            "exposure_l2_statevector": exposure_l2,
            "qcbm_n_params": int(qcbm.n_params),
            "exposure_n_params": int(crca_exposure.n_params),
            "default_n_params": int(crca_default.n_params),
            "discount_n_params": int(crca_discount.n_params),
        }

        ancilla_exposure_idx = total_state_qubits
        ancilla_default_idx = total_state_qubits + 1
        ancilla_discount_idx = total_state_qubits + 2
        if cfg.final_cva.run_qae or cfg.final_cva.run_iqae:
            problem = EstimationProblem(
                state_preparation=qc_cva,
                objective_qubits=[
                    ancilla_exposure_idx,
                    ancilla_default_idx,
                    ancilla_discount_idx,
                ],
                is_good_state=lambda bitstr: bitstr == "111",
                post_processing=quantum_cva_circuit.cva_from_prob,
            )
            sampler = StatevectorSampler()
            if cfg.final_cva.run_qae:
                ae = AmplitudeEstimation(
                    num_eval_qubits=int(cfg.final_cva.qae_num_eval_qubits),
                    sampler=sampler,
                )
                ae_result = ae.estimate(problem)
                result["qae_cva"] = float(ae_result.estimation_processed)
                result["qae_oracle_queries"] = int(ae_result.num_oracle_queries)
                result["relative_error_qae_pct"] = _percent_relative_error(
                    float(ae_result.estimation_processed),
                    cva_classical,
                )
            if cfg.final_cva.run_iqae:
                iae = IterativeAmplitudeEstimation(
                    epsilon_target=float(cfg.final_cva.iqae_epsilon_target),
                    alpha=float(cfg.final_cva.iqae_alpha),
                    sampler=sampler,
                )
                iae_result = iae.estimate(problem)
                result["iqae_cva"] = float(iae_result.estimation_processed)
                result["iqae_oracle_queries"] = int(iae_result.num_oracle_queries)
                result["relative_error_iqae_pct"] = _percent_relative_error(
                    float(iae_result.estimation_processed),
                    cva_classical,
                )

        out_path = self.results_dir / "final_cva_results.json"
        _write_json(out_path, _jsonable(result))
        print(json.dumps(_jsonable(result), indent=2))
        return result

    def _stage_sequence(self, stage: StageName) -> list[str]:
        full = [
            "classical",
            "train_qcbm",
            "train_crca_default",
            "train_crca_discount",
            "train_crca_exposure",
            "final_statevector_cva",
        ]
        if stage == "all":
            return full
        return [str(stage)]

    def _resolve(self, path_like: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path_like)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _benchmark_path(self) -> pathlib.Path:
        return self._resolve(self.config.paths.benchmark_relative_path)

    def _qcbm_path(self) -> pathlib.Path:
        return self._resolve(self.config.paths.qcbm_training_relative_path)

    def _crca_default_path(self) -> pathlib.Path:
        return self._resolve(self.config.paths.crca_default_training_relative_path)

    def _crca_discount_path(self) -> pathlib.Path:
        return self._resolve(self.config.paths.crca_discount_training_relative_path)

    def _crca_exposure_path(self) -> pathlib.Path:
        return self._resolve(self.config.paths.crca_exposure_training_relative_path)

    def _backend_context(self, *, thermal_relaxation: bool) -> BackendContext:
        from qiskit_aer.noise import NoiseModel

        backend_cfg = self.config.backend_noise
        service, real_backend = _load_real_backend(backend_cfg)
        snapshot_dt_utc = _parse_snapshot_datetime(backend_cfg.noise_snapshot_iso_utc)
        backend_props = real_backend.properties(datetime=snapshot_dt_utc)
        if backend_props is None:
            raise RuntimeError(
                "Could not retrieve historical backend properties for "
                f"{backend_cfg.noise_snapshot_iso_utc}."
            )
        used_noise_fallback = False
        try:
            noise_model = NoiseModel.from_backend_properties(
                backend_props,
                thermal_relaxation=thermal_relaxation,
            )
        except AttributeError:
            used_noise_fallback = True
            noise_model = NoiseModel.from_backend(real_backend)

        coupling_map = getattr(real_backend, "coupling_map", None)
        if coupling_map is None:
            try:
                coupling_map = real_backend.configuration().coupling_map
            except Exception:
                coupling_map = None

        return BackendContext(
            service=service,
            real_backend=real_backend,
            backend_props=backend_props,
            snapshot_dt_utc=snapshot_dt_utc,
            noise_model=noise_model,
            coupling_map=coupling_map,
            used_noise_fallback=used_noise_fallback,
        )

    def _validate_config(self) -> None:
        n_assets = len(self.config.assets)
        if n_assets < 1:
            raise ValueError("At least one asset is required.")
        if len(self.config.quantum.n_bits_per_asset) != n_assets:
            raise ValueError(
                "quantum.n_bits_per_asset must have one entry per asset."
            )
        if self.config.classical.m_time <= 0:
            raise ValueError("classical.m_time must be positive.")
        total_state_qubits = (
            int(self.config.classical.m_time)
            + int(sum(self.config.quantum.n_bits_per_asset))
        )
        if total_state_qubits != 6:
            raise ValueError(
                "This BETA pipeline is scoped to 6 state-preparation qubits. "
                f"Got {total_state_qubits} = m_time + sum(n_bits_per_asset)."
            )
        asset_symbols = {asset.symbol for asset in self.config.assets}
        for inst in self.config.instruments:
            if inst.asset_symbol not in asset_symbols:
                raise ValueError(
                    f"Instrument references unknown asset: {inst.asset_symbol}"
                )
            if inst.maturity_years <= 0.0:
                raise ValueError("Instrument maturity must be positive.")


@dataclass(frozen=True, slots=True)
class BackendContext:
    service: Any
    real_backend: Any
    backend_props: Any
    snapshot_dt_utc: datetime
    noise_model: Any
    coupling_map: Any
    used_noise_fallback: bool


class _BackendSnapshotView:
    def __init__(self, backend: Any, snapshot_props: Any) -> None:
        self._backend = backend
        self._snapshot_props = snapshot_props

    def properties(self, *args, **kwargs):
        return self._snapshot_props

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def find_repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    here = pathlib.Path(__file__).resolve() if start is None else pathlib.Path(start).resolve()
    candidates = [here, *here.parents]
    for parent in candidates:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repository root with pyproject.toml.")


def _load_real_backend(backend_cfg: BackendNoiseConfig) -> tuple[Any, Any]:
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel=backend_cfg.runtime_channel)
    real_backend = service.backend(
        backend_cfg.backend_name,
        use_fractional_gates=bool(backend_cfg.use_fractional_gates),
    )
    return service, real_backend


def _parse_snapshot_datetime(snapshot_iso_utc: str) -> datetime:
    dt = datetime.fromisoformat(snapshot_iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_property_value(props: Any, qubit: int, name: str) -> float | None:
    try:
        out = props.qubit_property(int(qubit))
        value = out.get(name, (None, None))[0]
        return None if value is None else float(value)
    except Exception:
        return None


def _inject_missing_frequencies(
    snapshot_props: Any,
    fallback_backend: Any,
    default_frequency_ghz: float = 5.0,
) -> tuple[Any, int]:
    props_dict = snapshot_props.to_dict()
    fallback_qprops = getattr(
        getattr(fallback_backend, "target", None),
        "qubit_properties",
        None,
    )
    injected = 0
    for qi, q_props in enumerate(props_dict.get("qubits", [])):
        if any(str(p.get("name", "")).lower() == "frequency" for p in q_props):
            continue
        freq_ghz = None
        if fallback_qprops is not None and qi < len(fallback_qprops):
            maybe_freq = getattr(fallback_qprops[qi], "frequency", None)
            if maybe_freq is not None and np.isfinite(maybe_freq):
                maybe_freq = float(maybe_freq)
                freq_ghz = maybe_freq / 1e9 if maybe_freq > 1e6 else maybe_freq
        q_props.append(
            {
                "date": props_dict.get("last_update_date"),
                "name": "frequency",
                "unit": "GHz",
                "value": float(
                    freq_ghz if freq_ghz is not None else default_frequency_ghz
                ),
            }
        )
        injected += 1
    return type(snapshot_props).from_dict(props_dict), injected


def _build_local_coupling_map(
    full_edges: Sequence[Sequence[int]],
    chosen_layout: Sequence[int],
) -> list[list[int]]:
    phys_to_local = {int(q): i for i, q in enumerate(chosen_layout)}
    local_edges: list[list[int]] = []
    for a, b in full_edges:
        if int(a) in phys_to_local and int(b) in phys_to_local:
            local_edges.append([phys_to_local[int(a)], phys_to_local[int(b)]])
    return local_edges


def _subset_backend_properties(snapshot_props: Any, chosen_layout: Sequence[int]) -> Any:
    props_dict = snapshot_props.to_dict()
    phys_to_local = {int(q): i for i, q in enumerate(chosen_layout)}
    subset_dict = {k: v for k, v in props_dict.items()}
    subset_dict["qubits"] = [props_dict["qubits"][int(q)] for q in chosen_layout]
    subset_gates = []
    for gate in props_dict.get("gates", []):
        qids = [int(q) for q in gate.get("qubits", [])]
        if all(q in phys_to_local for q in qids):
            new_gate = dict(gate)
            new_gate["qubits"] = [phys_to_local[q] for q in qids]
            subset_gates.append(new_gate)
    subset_dict["gates"] = subset_gates
    subset_dict["general_qlists"] = []
    subset_dict["backend_name"] = (
        f"{props_dict.get('backend_name', 'backend')}_subset_{len(chosen_layout)}q"
    )
    return type(snapshot_props).from_dict(subset_dict)


def _build_transpiled_measured_eval_circuit(crca: Any, pass_manager: Any) -> Any:
    from qiskit import ClassicalRegister

    qc_meas = crca.qc_eval.copy()
    c_ctrl = ClassicalRegister(crca.n_controls, "c")
    c_a = ClassicalRegister(1, "ca")
    qc_meas.add_register(c_ctrl, c_a)
    qc_meas.measure(crca._control_qubit_indices, c_ctrl)
    qc_meas.measure([crca._ancilla_qubit_index], c_a)
    return pass_manager.run(qc_meas)


def _evaluate_crca_function_values(
    crca: Any,
    theta: np.ndarray,
    *,
    shots: int,
    seed: int,
) -> np.ndarray:
    theta = _as_1d_float(theta)
    try:
        values = crca.function_values(theta, shots=int(shots), seed=int(seed))
    except TypeError:
        values = crca.function_values(theta, shots=int(shots))
    return _as_1d_float(values)


def _build_positive_objective(
    crca: Any,
    f_target: np.ndarray,
    training_config: PositiveExposureTrainingConfig,
    *,
    mode: Literal["l2", "support_aware", "support_aware_robust"],
    shots: int,
    eval_repeats: int,
) -> Callable[[np.ndarray], float]:
    f_target = _as_1d_float(f_target)
    eval_repeats = max(1, int(eval_repeats))
    pos_mask = f_target > float(training_config.target_threshold)
    zero_mask = ~pos_mask
    pos_denom = np.maximum(
        np.abs(f_target[pos_mask]),
        float(training_config.relative_eps),
    )
    if mode not in {"l2", "support_aware", "support_aware_robust"}:
        raise ValueError(f"Unsupported positive exposure objective mode: {mode}")

    def objective(theta: np.ndarray) -> float:
        theta_arr = _as_1d_float(theta)
        acc = 0.0
        for k in range(eval_repeats):
            seed_k = int(
                training_config.shot_seed + training_config.repeat_seed_stride * k
            )
            f_model = _evaluate_crca_function_values(
                crca,
                theta_arr,
                shots=int(shots),
                seed=seed_k,
            )
            if mode == "l2":
                acc += _mean_squared_error(f_model, f_target)
            elif mode == "support_aware":
                acc += _support_aware_cost(
                    f_model,
                    f_target,
                    training_config,
                    pos_mask=pos_mask,
                    zero_mask=zero_mask,
                    pos_denom=pos_denom,
                )
            else:
                acc += _support_aware_robust_cost(
                    f_model,
                    f_target,
                    training_config,
                    pos_mask=pos_mask,
                    zero_mask=zero_mask,
                    pos_denom=pos_denom,
                )
        return float(acc / eval_repeats)

    return objective


def _support_aware_cost(
    f_model: np.ndarray,
    f_target: np.ndarray,
    training_config: PositiveExposureTrainingConfig,
    *,
    pos_mask: np.ndarray,
    zero_mask: np.ndarray,
    pos_denom: np.ndarray,
) -> float:
    loss = 0.0
    if np.any(pos_mask):
        rel_err_sq = ((f_model[pos_mask] - f_target[pos_mask]) / pos_denom) ** 2
        loss += float(training_config.lambda_pos * np.mean(rel_err_sq))
    if np.any(zero_mask):
        loss += float(training_config.lambda_zero * np.mean(np.abs(f_model[zero_mask])))
    return float(loss)


def _support_aware_robust_cost(
    f_model: np.ndarray,
    f_target: np.ndarray,
    training_config: PositiveExposureTrainingConfig,
    *,
    pos_mask: np.ndarray,
    zero_mask: np.ndarray,
    pos_denom: np.ndarray,
) -> float:
    loss = float(training_config.lambda_l2_mix * _mean_squared_error(f_model, f_target))
    if np.any(pos_mask):
        rel_err = (f_model[pos_mask] - f_target[pos_mask]) / pos_denom
        rel_err = np.clip(
            rel_err,
            -float(training_config.robust_rel_clip),
            float(training_config.robust_rel_clip),
        )
        loss += float(
            training_config.lambda_pos
            * np.mean(_huber_loss(rel_err, training_config.robust_rel_huber_delta))
        )
    if np.any(zero_mask):
        zero_vals = np.clip(f_model[zero_mask], -1.0, 1.0)
        loss += float(
            training_config.lambda_zero
            * np.mean(
                _huber_loss(zero_vals, training_config.robust_zero_huber_delta)
            )
        )
    return float(loss)


def _huber_loss(x: np.ndarray, delta: float) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * abs_x**2, delta * (abs_x - 0.5 * delta))


def _run_positive_spsa_stage(
    *,
    stage_name: str,
    objective_mode: str,
    crca: Any,
    f_target: np.ndarray,
    x0: np.ndarray,
    shots: int,
    calibration_shots: int,
    maxiter: int,
    resamplings: int | dict[int, int],
    eval_repeats: int,
    second_order: bool,
    blocking: bool,
    trust_region: bool,
    regularization: float,
    hessian_delay: int,
    calibration_target_magnitude: float | None,
    training_config: PositiveExposureTrainingConfig,
) -> dict[str, Any]:
    from qiskit_algorithms.optimizers import SPSA

    objective = _build_positive_objective(
        crca,
        f_target,
        training_config,
        mode=objective_mode,
        shots=int(shots),
        eval_repeats=int(eval_repeats),
    )
    objective_calibration = _build_positive_objective(
        crca,
        f_target,
        training_config,
        mode=objective_mode,
        shots=int(calibration_shots),
        eval_repeats=int(eval_repeats),
    )
    l2_objective = _build_positive_objective(
        crca,
        f_target,
        training_config,
        mode="l2",
        shots=int(shots),
        eval_repeats=int(eval_repeats),
    )

    x0 = _as_1d_float(x0)
    obj_history: list[float] = [float(objective(x0))]
    l2_history: list[float] = [float(l2_objective(x0))]
    theta_history: list[np.ndarray] = [x0.copy()]

    print(
        f"\n[{stage_name}] mode={objective_mode} | shots={shots} "
        f"| maxiter={maxiter} | eval_repeats={eval_repeats}"
    )
    print(f"[{stage_name}] Calibrating SPSA hyperparameters...")
    lr, pert = SPSA.calibrate(
        objective_calibration,
        x0,
        target_magnitude=calibration_target_magnitude,
    )

    def callback(nfev, x, fx, step, accepted):
        x_arr = np.asarray(x, dtype=float).copy()
        fx_obj = float(fx)
        fx_l2 = float(l2_objective(x_arr))
        obj_history.append(fx_obj)
        l2_history.append(fx_l2)
        theta_history.append(x_arr)
        print(
            f"[{stage_name} iter {len(obj_history)-1:4d}] "
            f"obj={fx_obj:.6e} | l2={fx_l2:.6e} | nfev={int(nfev):6d}"
        )

    kwargs: dict[str, Any] = {
        "maxiter": int(maxiter),
        "learning_rate": lr,
        "perturbation": pert,
        "resamplings": resamplings,
        "last_avg": int(training_config.last_avg),
        "second_order": bool(second_order),
        "blocking": bool(blocking),
        "trust_region": bool(trust_region),
        "callback": callback,
    }
    if second_order:
        kwargs["regularization"] = float(regularization)
        kwargs["hessian_delay"] = int(hessian_delay)

    opt = SPSA(**kwargs)
    t0 = time.perf_counter()
    res = opt.minimize(fun=objective, x0=x0)
    elapsed_s = float(time.perf_counter() - t0)
    theta_last = _as_1d_float(res.x)
    if not np.allclose(theta_history[-1], theta_last, rtol=0.0, atol=1e-15):
        obj_history.append(float(objective(theta_last)))
        l2_history.append(float(l2_objective(theta_last)))
        theta_history.append(theta_last.copy())

    obj_arr = np.asarray(obj_history, dtype=float)
    l2_arr = np.asarray(l2_history, dtype=float)
    theta_arr = np.asarray(theta_history, dtype=float)
    idx_best_obj = int(np.argmin(obj_arr))
    idx_best_l2 = int(np.argmin(l2_arr))
    result = {
        "stage_name": stage_name,
        "objective_mode": objective_mode,
        "elapsed_s": elapsed_s,
        "theta_last": theta_last,
        "theta_best_obj": theta_arr[idx_best_obj].copy(),
        "theta_best_l2": theta_arr[idx_best_l2].copy(),
        "best_obj": float(obj_arr[idx_best_obj]),
        "best_l2": float(l2_arr[idx_best_l2]),
        "obj_history": obj_arr,
        "l2_history": l2_arr,
        "theta_history": theta_arr,
        "result_success": bool(getattr(res, "success", False)),
        "result_message": str(getattr(res, "message", "")),
    }
    print(
        f"[{stage_name}] done in {elapsed_s:.1f}s | "
        f"best_obj={result['best_obj']:.6e} | best_l2={result['best_l2']:.6e}"
    )
    return result


def _select_best_theta_by_recheck(
    crca: Any,
    f_target: np.ndarray,
    theta_history: np.ndarray,
    l2_history: np.ndarray,
    *,
    shots: int,
    top_k: int,
    eval_repeats: int,
    training_config: PositiveExposureTrainingConfig,
) -> tuple[int, float]:
    l2_history = np.asarray(l2_history, dtype=float).reshape(-1)
    theta_history = np.asarray(theta_history, dtype=float)
    if theta_history.shape[0] != l2_history.size:
        raise ValueError("theta_history and l2_history must have same length.")
    top_k = max(1, min(int(top_k), int(l2_history.size)))
    candidate_idx = np.argsort(l2_history)[:top_k]
    candidate_idx = np.unique(np.r_[candidate_idx, [l2_history.size - 1]])
    scorer = _build_positive_objective(
        crca,
        f_target,
        training_config,
        mode="l2",
        shots=int(shots),
        eval_repeats=int(eval_repeats),
    )
    best_idx = int(candidate_idx[0])
    best_l2 = float("inf")
    for idx in candidate_idx:
        l2_val = float(scorer(theta_history[int(idx)]))
        if l2_val < best_l2:
            best_l2 = l2_val
            best_idx = int(idx)
    return best_idx, best_l2


def _merge_stage_histories(
    stage1: dict[str, Any],
    stage2: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts_from_stage1_tail = np.allclose(
        stage2["theta_history"][0],
        stage1["theta_history"][-1],
        rtol=0.0,
        atol=1e-15,
    )
    if starts_from_stage1_tail:
        theta_history_arr = np.vstack(
            [stage1["theta_history"], stage2["theta_history"][1:]]
        )
        cost_history_arr = np.r_[stage1["obj_history"], stage2["obj_history"][1:]]
        l2_history_arr = np.r_[stage1["l2_history"], stage2["l2_history"][1:]]
    else:
        theta_history_arr = np.vstack(
            [stage1["theta_history"], stage2["theta_history"]]
        )
        cost_history_arr = np.r_[stage1["obj_history"], stage2["obj_history"]]
        l2_history_arr = np.r_[stage1["l2_history"], stage2["l2_history"]]
    return theta_history_arr, cost_history_arr, l2_history_arr


def _load_warmstart_theta(
    repo_root: pathlib.Path,
    *,
    path: str,
    n_params_expected: int,
    enabled: bool,
) -> np.ndarray | None:
    if not enabled:
        return None
    warmstart_path = pathlib.Path(path)
    if not warmstart_path.is_absolute():
        warmstart_path = repo_root / warmstart_path
    if not warmstart_path.exists():
        print(f"[INFO] warmstart file not found: {warmstart_path}")
        return None
    data = np.load(warmstart_path, allow_pickle=True)
    if "theta_star" not in data:
        print("[INFO] warmstart file has no theta_star; skipping.")
        return None
    theta = _as_1d_float(data["theta_star"])
    if theta.size != int(n_params_expected):
        print(
            "[WARNING] warmstart theta size mismatch; expected "
            f"{n_params_expected}, got {theta.size}; skipping."
        )
        return None
    return theta


def _load_npz(path: pathlib.Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact does not exist: {path}")
    return np.load(path, allow_pickle=True)


def _as_1d_float(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _num_qubits_from_dim(dim: int, label: str) -> int:
    n_qubits = int(np.log2(int(dim)))
    if 2**n_qubits != int(dim):
        raise ValueError(f"{label} dimension must be a power of 2; got {dim}.")
    return n_qubits


def _assert_size(label: str, values: Any, expected_size: int) -> None:
    actual = int(np.asarray(values).size)
    if actual != int(expected_size):
        raise ValueError(f"{label} size mismatch: expected {expected_size}, got {actual}.")


def _npz_int(npz_data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in npz_data:
        return int(default)
    return int(np.asarray(npz_data[key]).item())


def _npz_str(npz_data: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz_data:
        return str(default)
    return str(np.asarray(npz_data[key]).item())


def _metadata_dict(npz_data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in npz_data:
        return {}
    maybe_dict = npz_data["metadata"]
    if hasattr(maybe_dict, "item"):
        maybe_dict = maybe_dict.item()
    return maybe_dict if isinstance(maybe_dict, dict) else {}


def _classical_reference_cva(
    benchmark: np.lib.npyio.NpzFile,
    *,
    grid_bits: int,
) -> float:
    grid_sizes = np.asarray(benchmark["grid_sizes"], dtype=int).reshape(-1)
    values = np.asarray(benchmark["cva_by_grid_size_values"], dtype=float).reshape(-1)
    matches = np.flatnonzero(grid_sizes == int(grid_bits))
    if matches.size:
        return float(values[int(matches[0])])
    if "cva_small_scaled" in benchmark:
        return float(benchmark["cva_small_scaled"])
    return float(values[min(1, values.size - 1)])


def _percent_relative_error(estimate: float, reference: float) -> float:
    reference = float(reference)
    if reference == 0.0:
        return float("nan")
    return float(abs(float(estimate) - reference) / abs(reference) * 100.0)


def _mean_squared_error(values: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((np.asarray(values) - np.asarray(target)) ** 2))


def _require_optimizer(optimizer: str) -> None:
    if str(optimizer).upper() != "SPSA":
        raise NotImplementedError(
            "This pipeline currently implements SPSA for noise+shots training. "
            f"Requested optimizer={optimizer!r}."
        )


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "AssetConfig",
    "BackendNoiseConfig",
    "ClassicalConfig",
    "FinalCVAConfig",
    "InstrumentConfig",
    "MarketDataConfig",
    "PipelineConfig",
    "PipelinePathConfig",
    "PositiveExposureTrainingConfig",
    "QCBMTrainingConfig",
    "QuantumProblemConfig",
    "ScalarCRCTrainingConfig",
    "TrainingOutputConfig",
    "build_arg_parser",
    "run_pipeline",
]
