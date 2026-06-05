from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from scipy.optimize import minimize


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
INSTANCE_DIR = SCRIPT_DIR.parent


def _find_repo_root() -> pathlib.Path:
    return next(
        parent
        for parent in pathlib.Path(__file__).resolve().parents
        if (parent / "pyproject.toml").exists()
    )


REPO_ROOT = _find_repo_root()
for bootstrap_path in (REPO_ROOT / "src", INSTANCE_DIR):
    if str(bootstrap_path) not in sys.path:
        sys.path.insert(0, str(bootstrap_path))


from full_cva_pipeline import CONFIG as BASE_CONFIG
from qiskit_aer import AerSimulator
from quantum_cva.multi_asset.instruments.market_data import MarketData
from quantum_cva.multi_asset.quantum.amplitude_estimation.cva_circuit import (
    QuantumCVACircuit,
)
from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import (
    CrcaCircuit,
)
from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (
    MLQcbmCircuit,
)


QCBM_WARMSTART = REPO_ROOT / (
    "data/multi_asset/6q_instance/quantum/training/qcbm/statevector/"
    "training_qcbm_heavyhex6_6lay.npz"
)
DEFAULT_WARMSTART = REPO_ROOT / (
    "data/multi_asset/6q_instance/quantum/training/crca/default_probabilities/"
    "training_crca2.npz"
)
DISCOUNT_WARMSTART = REPO_ROOT / (
    "data/multi_asset/6q_instance/quantum/training/crca/discount_factors/"
    "training_crca2.npz"
)
EXPOSURE_WARMSTART = REPO_ROOT / (
    "data/multi_asset/6q_instance/quantum/training/crca/positive_exposure/"
    "training_heavy_hex_star.npz"
)
CANONICAL_BENCHMARK = REPO_ROOT / (
    "data/multi_asset/6q_instance/benchmark/three_asset_instance.npz"
)
CLASSICAL_BENCHMARK_METHODOLOGY = "legacy_right_endpoint_small_grid_n_bits_2_v1"
STATEVECTOR_EVALUATION_METHODOLOGY = "run_ideal_cva_logical_statevector_v1"
RELATIVE_ERROR_REFERENCE = "classical_small_grid_n_bits_2_per_scenario"
CANONICAL_BASE_CLASSICAL_SMALL_GRID_CVA = 0.5215900006813697

RESULT_FIELDS = [
    "case_id",
    "label",
    "family",
    "status",
    "error_message",
    "component_policy",
    "statevector_evaluation_methodology",
    "classical_benchmark_methodology",
    "relative_error_reference",
    "call_strike_scale",
    "put_strike_scale",
    "volatility_scale",
    "rate_shift_bps",
    "default_spread_scale",
    "cva_reference_grid_2",
    "cva_classical_small_grid_n_bits_2",
    "cva_encoded_target",
    "cva_mc_continuous",
    "cva_statevector",
    "p111_statevector",
    "absolute_error_vs_reference",
    "relative_error_vs_reference_pct",
    "absolute_relative_error_vs_reference_pct",
    "absolute_error_vs_encoded_target",
    "relative_error_vs_encoded_target_pct",
    "qcbm_target_changed",
    "default_target_changed",
    "discount_target_changed",
    "exposure_target_changed",
    "qcbm_retrained",
    "default_retrained",
    "discount_retrained",
    "exposure_retrained",
    "qcbm_kl",
    "qcbm_tv",
    "qcbm_linf",
    "default_mse",
    "default_linf",
    "discount_mse",
    "discount_linf",
    "exposure_mse",
    "exposure_linf",
    "C_q",
    "C_p",
    "C_v",
    "benchmark_path",
    "qcbm_artifact",
    "default_artifact",
    "discount_artifact",
    "exposure_artifact",
    "elapsed_s",
]


@dataclass(frozen=True, slots=True)
class Scenario:
    case_id: str
    label: str
    call_strike_scale: float = 1.0
    put_strike_scale: float = 1.0
    volatility_scale: float = 1.0
    rate_shift_bps: float = 0.0
    default_spread_scale: float = 1.0

    @property
    def family(self) -> str:
        active = []
        if not math.isclose(self.call_strike_scale, 1.0):
            active.append("call_strike")
        if not math.isclose(self.put_strike_scale, 1.0):
            active.append("put_strike")
        if not math.isclose(self.volatility_scale, 1.0):
            active.append("volatility")
        if not math.isclose(self.rate_shift_bps, 0.0):
            active.append("interest_rate")
        if not math.isclose(self.default_spread_scale, 1.0):
            active.append("default_spread")
        if not active:
            return "baseline"
        if len(active) == 1:
            return active[0]
        return "combined"


@dataclass(slots=True)
class TrainingArtifact:
    path: pathlib.Path
    theta: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    metrics: dict[str, float]
    loaded_from_cache: bool


@dataclass(frozen=True, slots=True)
class OptimizerBudget:
    cobyla_maxiter: int
    lbfgsb_maxiter: int


@dataclass(slots=True)
class OptimizerResult:
    theta_best: np.ndarray
    theta_last: np.ndarray
    objective_history: np.ndarray
    best_objective_history: np.ndarray
    elapsed_s: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: pathlib.Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return ""
            return f"{value:.8g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(cell(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def _write_markdown_table(
    path: pathlib.Path,
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_table(rows, fields), encoding="utf-8")


def _slug_number(value: float) -> str:
    return f"{value:+g}".replace("+", "p").replace("-", "m").replace(".", "p")


def focused_scenarios() -> list[Scenario]:
    return [
        Scenario("base", "Baseline"),
        Scenario("call_k_m10", "Call strike -10%", call_strike_scale=0.90),
        Scenario("call_k_p10", "Call strike +10%", call_strike_scale=1.10),
        Scenario("put_k_m10", "Put strike -10%", put_strike_scale=0.90),
        Scenario("put_k_p10", "Put strike +10%", put_strike_scale=1.10),
        Scenario("both_k_m10", "Both strikes -10%", call_strike_scale=0.90, put_strike_scale=0.90),
        Scenario("both_k_p10", "Both strikes +10%", call_strike_scale=1.10, put_strike_scale=1.10),
        Scenario("vol_m20", "Volatility -20%", volatility_scale=0.80),
        Scenario("vol_m10", "Volatility -10%", volatility_scale=0.90),
        Scenario("vol_p10", "Volatility +10%", volatility_scale=1.10),
        Scenario("vol_p20", "Volatility +20%", volatility_scale=1.20),
        Scenario("rate_m100", "Curve shift -100 bp", rate_shift_bps=-100.0),
        Scenario("rate_m50", "Curve shift -50 bp", rate_shift_bps=-50.0),
        Scenario("rate_p50", "Curve shift +50 bp", rate_shift_bps=50.0),
        Scenario("rate_p100", "Curve shift +100 bp", rate_shift_bps=100.0),
        Scenario("default_m25", "CDS spreads -25%", default_spread_scale=0.75),
        Scenario("default_m10", "CDS spreads -10%", default_spread_scale=0.90),
        Scenario("default_p10", "CDS spreads +10%", default_spread_scale=1.10),
        Scenario("default_p25", "CDS spreads +25%", default_spread_scale=1.25),
        Scenario(
            "risk_on",
            "Risk-on combined",
            call_strike_scale=0.95,
            put_strike_scale=1.05,
            volatility_scale=0.90,
            rate_shift_bps=50.0,
            default_spread_scale=0.90,
        ),
        Scenario(
            "risk_off",
            "Risk-off combined",
            call_strike_scale=1.05,
            put_strike_scale=0.95,
            volatility_scale=1.10,
            rate_shift_bps=-50.0,
            default_spread_scale=1.10,
        ),
        Scenario(
            "risk_off_extreme",
            "Risk-off combined extreme",
            call_strike_scale=1.10,
            put_strike_scale=0.90,
            volatility_scale=1.20,
            rate_shift_bps=-100.0,
            default_spread_scale=1.25,
        ),
    ]


def ultra_scenarios() -> list[Scenario]:
    scenarios: list[Scenario] = []
    for call_scale in (0.90, 1.00, 1.10):
        for put_scale in (0.90, 1.00, 1.10):
            for vol_scale in (0.90, 1.00, 1.10):
                for rate_shift in (-50.0, 0.0, 50.0):
                    for default_scale in (0.90, 1.00, 1.10):
                        is_base = all(
                            (
                                math.isclose(call_scale, 1.0),
                                math.isclose(put_scale, 1.0),
                                math.isclose(vol_scale, 1.0),
                                math.isclose(rate_shift, 0.0),
                                math.isclose(default_scale, 1.0),
                            )
                        )
                        case_id = (
                            "base"
                            if is_base
                            else (
                                f"ck_{_slug_number(call_scale)}"
                                f"__pk_{_slug_number(put_scale)}"
                                f"__vol_{_slug_number(vol_scale)}"
                                f"__rate_{_slug_number(rate_shift)}"
                                f"__cds_{_slug_number(default_scale)}"
                            )
                        )
                        scenarios.append(
                            Scenario(
                                case_id=case_id,
                                label=case_id,
                                call_strike_scale=call_scale,
                                put_strike_scale=put_scale,
                                volatility_scale=vol_scale,
                                rate_shift_bps=rate_shift,
                                default_spread_scale=default_scale,
                            )
                        )
    return sorted(scenarios, key=lambda scenario: scenario.case_id != "base")


def _scenarios_from_csv(path: pathlib.Path) -> list[Scenario]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scenarios = []
    for index, row in enumerate(rows):
        case_id = row.get("case_id") or f"custom_{index:04d}"
        scenarios.append(
            Scenario(
                case_id=case_id,
                label=row.get("label") or case_id,
                call_strike_scale=float(row.get("call_strike_scale") or 1.0),
                put_strike_scale=float(row.get("put_strike_scale") or 1.0),
                volatility_scale=float(row.get("volatility_scale") or 1.0),
                rate_shift_bps=float(row.get("rate_shift_bps") or 0.0),
                default_spread_scale=float(row.get("default_spread_scale") or 1.0),
            )
        )
    if not scenarios:
        raise ValueError(f"No scenarios found in {path}.")
    return sorted(scenarios, key=lambda scenario: scenario.case_id != "base")


def _validate_scenarios(scenarios: Sequence[Scenario]) -> None:
    ids = [scenario.case_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Scenario case_id values must be unique.")
    if not scenarios or scenarios[0].case_id != "base":
        raise ValueError("The first scenario must have case_id='base'.")
    for scenario in scenarios:
        if scenario.call_strike_scale <= 0.0 or scenario.put_strike_scale <= 0.0:
            raise ValueError(f"{scenario.case_id}: strike scales must be positive.")
        if scenario.volatility_scale <= 0.0:
            raise ValueError(f"{scenario.case_id}: volatility_scale must be positive.")
        if scenario.default_spread_scale <= 0.0:
            raise ValueError(f"{scenario.case_id}: default_spread_scale must be positive.")


def _catalog_rows(scenarios: Sequence[Scenario]) -> list[dict[str, Any]]:
    return [{**asdict(scenario), "family": scenario.family} for scenario in scenarios]


def write_scenario_catalog(output_dir: pathlib.Path, scenarios: Sequence[Scenario]) -> None:
    fields = [
        "case_id",
        "label",
        "family",
        "call_strike_scale",
        "put_strike_scale",
        "volatility_scale",
        "rate_shift_bps",
        "default_spread_scale",
    ]
    rows = _catalog_rows(scenarios)
    _write_csv(output_dir / "scenario_catalog.csv", rows, fields)
    _write_markdown_table(output_dir / "scenario_catalog.md", rows, fields)


def write_scenario_design_plot(output_dir: pathlib.Path, scenarios: Sequence[Scenario]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _catalog_rows(scenarios)
    values = np.array(
        [
            [
                row["call_strike_scale"] - 1.0,
                row["put_strike_scale"] - 1.0,
                row["volatility_scale"] - 1.0,
                row["rate_shift_bps"] / 100.0,
                row["default_spread_scale"] - 1.0,
            ]
            for row in rows
        ],
        dtype=float,
    )
    max_abs = max(float(np.max(np.abs(values))), 1e-12)
    fig_height = min(max(5.0, 0.22 * len(rows)), 22.0)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["call K", "put K", "vol", "rate / 100 bp", "CDS spread"])
    if len(rows) <= 60:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["case_id"] for row in rows], fontsize=7)
    else:
        ax.set_ylabel(f"{len(rows)} scenarios")
    ax.set_title("Ideal CVA robustness scenario design")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("Normalized perturbation")
    fig.tight_layout()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "scenario_design_matrix.png", dpi=220)
    fig.savefig(plots_dir / "scenario_design_matrix.pdf")
    plt.close(fig)


@contextmanager
def _market_data_stress(scenario: Scenario) -> Iterator[None]:
    original_vols = MarketData.get_atm_vol_curves
    original_discount = MarketData.discount_factor
    original_cds = MarketData.get_cds_curve
    rate_shift = float(scenario.rate_shift_bps) / 10_000.0

    def stressed_vols(self: MarketData, underlyings: Sequence[str] | None = None) -> Any:
        frame = original_vols(self, underlyings=underlyings).copy()
        frame.loc[:, "atm_vol"] = (
            frame["atm_vol"].astype(float) * float(scenario.volatility_scale)
        )
        return frame

    def stressed_discount(self: MarketData, maturity: float) -> float:
        return float(original_discount(self, maturity) * np.exp(-rate_shift * float(maturity)))

    def stressed_cds(self: MarketData) -> tuple[Any, Any]:
        tenors, spreads = original_cds(self)
        return tenors, np.asarray(spreads, dtype=float) * float(scenario.default_spread_scale)

    MarketData.get_atm_vol_curves = stressed_vols
    MarketData.discount_factor = stressed_discount
    MarketData.get_cds_curve = stressed_cds
    try:
        yield
    finally:
        MarketData.get_atm_vol_curves = original_vols
        MarketData.discount_factor = original_discount
        MarketData.get_cds_curve = original_cds


def _advance_standard_normal_rng(
    rng: np.random.Generator,
    *,
    total_values: int,
    chunk_size: int = 1_000_000,
) -> None:
    remaining = int(total_values)
    while remaining > 0:
        current = min(remaining, int(chunk_size))
        rng.standard_normal(size=current)
        remaining -= current


def _generate_stressed_classical_benchmark(
    scenario: Scenario,
    benchmark_path: pathlib.Path,
    args: argparse.Namespace,
) -> None:
    from quantum_cva.multi_asset.classical.classical_cva.classical_cont_cva import (
        ContinuousUnderlyingCvaEngine,
    )
    from quantum_cva.multi_asset.classical.classical_cva.classical_discrete_cva import (
        DiscreteUnderlyingCvaEngine,
    )
    from quantum_cva.multi_asset.classical.classical_cva.cva_auxiliar_functions import (
        build_survival_from_cds,
    )
    from quantum_cva.multi_asset.classical.probability_and_underlying.multi_asset_dynamics_utils import (
        simulate_multi_asset_gbm,
    )
    from quantum_cva.multi_asset.classical.probability_and_underlying.piecewise_volatility_utils import (
        build_piecewise_vol_curve_for_underlying,
        build_residual_volatility_function_for_underlying,
    )
    from quantum_cva.multi_asset.instruments.derivatives import Call, Forward, Put

    market_cfg = BASE_CONFIG.market_data
    market_data = MarketData.load(
        discount_curve_path=str(REPO_ROOT / market_cfg.discount_curve_path),
        credit_data_path=str(REPO_ROOT / market_cfg.credit_data_path),
        historical_series_path=str(REPO_ROOT / market_cfg.historical_series_path),
        atm_vol_surfaces_path=str(REPO_ROOT / market_cfg.atm_vol_surfaces_path),
        valuation_date=market_cfg.valuation_date,
    )
    underlyings = [asset.symbol for asset in BASE_CONFIG.assets]
    dividend_yields = [float(asset.dividend_yield) for asset in BASE_CONFIG.assets]
    spots = market_data.get_spot_vector(underlyings)
    atm_vol_curves = market_data.get_atm_vol_curves(underlyings=underlyings)
    correlation = market_data.get_log_return_correlation(underlyings)
    discount_factor = lambda maturity: market_data.discount_factor(maturity)
    interest_rate = float(-np.log(discount_factor(1.0)))
    drift = [interest_rate - dividend_yield for dividend_yield in dividend_yields]

    maturity_max = max(float(instrument.maturity_years) for instrument in BASE_CONFIG.instruments)
    m_time = int(BASE_CONFIG.classical.m_time)
    n_dates = 2**m_time
    exposure_times = np.linspace(0.0, maturity_max, n_dates + 1, dtype=float)[1:]
    n_paths = int(args.n_paths)
    n_assets = len(underlyings)
    rng = np.random.default_rng(int(BASE_CONFIG.classical.seed))
    normals = rng.standard_normal(size=(n_paths, n_dates, n_assets))

    def right_endpoint_sigma_grid(sim_times: np.ndarray) -> np.ndarray:
        sigma_columns = []
        for underlying in underlyings:
            maturities, sigma_piecewise = build_piecewise_vol_curve_for_underlying(
                atm_vol_curves=atm_vol_curves,
                underlying=underlying,
            )
            indices = np.searchsorted(maturities, sim_times, side="left")
            sigma_columns.append(sigma_piecewise[indices])
        return np.column_stack(sigma_columns)

    sigma_grid = right_endpoint_sigma_grid(exposure_times)
    paths = simulate_multi_asset_gbm(
        S0=spots,
        mu=drift,
        sigma=sigma_grid,
        rho=correlation,
        t=exposure_times,
        Z=normals,
        antithetic=bool(BASE_CONFIG.classical.antithetic),
        moment_match=bool(BASE_CONFIG.classical.moment_match),
        replications=int(BASE_CONFIG.classical.replications),
        replication_seed=int(BASE_CONFIG.classical.replication_seed),
        pathwise=True,
        integrated_covariances=None,
    )

    recovery_rate_cva = float(BASE_CONFIG.classical.recovery_rate_cva)
    cds_tenors, cds_spreads = market_data.get_cds_curve()
    _, _, _, default_increments = build_survival_from_cds(
        P0=discount_factor,
        tenors=cds_tenors,
        spreads=cds_spreads,
        R_cds=float(market_data.recovery_rate),
        pay_freq=int(BASE_CONFIG.classical.cds_pay_freq),
    )
    residual_volatility = {
        underlying: build_residual_volatility_function_for_underlying(
            atm_vol_curves=atm_vol_curves,
            underlying=underlying,
        )
        for underlying in underlyings
    }
    asset_index = {underlying: index for index, underlying in enumerate(underlyings)}
    instruments = []
    strikes = []
    for instrument in BASE_CONFIG.instruments:
        strike_scale = 1.0
        if instrument.kind == "call":
            strike_scale = float(scenario.call_strike_scale)
        elif instrument.kind == "put":
            strike_scale = float(scenario.put_strike_scale)
        strike = float(instrument.strike) * strike_scale
        strikes.append(strike)
        common = {
            "asset_idx": asset_index[instrument.asset_symbol],
            "quantity": float(instrument.quantity),
            "multiplier": float(instrument.multiplier),
            "K": strike,
            "T": float(instrument.maturity_years),
        }
        if instrument.kind == "call":
            instruments.append(Call(**common, sigma_func=residual_volatility[instrument.asset_symbol]))
        elif instrument.kind == "put":
            instruments.append(Put(**common, sigma_func=residual_volatility[instrument.asset_symbol]))
        elif instrument.kind == "forward":
            instruments.append(Forward(**common))
        else:
            raise ValueError(f"Unsupported instrument type: {instrument.kind}")

    continuous_engine = ContinuousUnderlyingCvaEngine(
        instruments=instruments,
        P0_func=discount_factor,
        q_interval=default_increments,
        LGD=1.0 - recovery_rate_cva,
        r=interest_rate,
    )
    cva_mc_continuous, cva_std_err_mc_continuous = continuous_engine.cva_from_paths(
        S_by_time=paths,
        t=exposure_times,
    )

    def make_engine(n_bits: int | Sequence[int]) -> Any:
        bits = [int(n_bits)] * n_assets if isinstance(n_bits, int) else [int(bit) for bit in n_bits]
        return DiscreteUnderlyingCvaEngine(
            instruments=instruments,
            P0_func=discount_factor,
            q_interval=default_increments,
            LGD=1.0 - recovery_rate_cva,
            r=interest_rate,
            n_bits=bits,
            n_sigma=float(BASE_CONFIG.classical.n_sigma),
            payoff_repr=BASE_CONFIG.classical.payoff_repr,
            order=BASE_CONFIG.classical.flattening_order,
            time_weights=None,
        )

    max_grid_bits = max(2, int(args.grid_convergence_max_bits))
    cva_by_grid_size = {
        n_bits: float(
            make_engine(n_bits).cva_from_paths_discretized(
                S_by_time=paths,
                t=exposure_times,
                return_blocks=False,
            )
        )
        for n_bits in range(1, max_grid_bits + 1)
    }
    grid_sizes = np.asarray(sorted(cva_by_grid_size), dtype=int)
    cva_values = np.asarray([cva_by_grid_size[int(n_bits)] for n_bits in grid_sizes], dtype=float)

    # Preserve the historical RNG sequence used by the canonical classical script.
    _advance_standard_normal_rng(
        rng,
        total_values=n_paths * 252 * n_assets,
    )
    fine_times = np.linspace(
        0.0,
        maturity_max,
        int(args.fine_time_grid_size),
        dtype=float,
    )[1:]
    fine_normals = rng.standard_normal(size=(n_paths, fine_times.size, n_assets))
    fine_paths = simulate_multi_asset_gbm(
        S0=spots,
        mu=drift,
        sigma=sigma_grid,
        sigma_times=exposure_times,
        rho=correlation,
        t=fine_times,
        Z=fine_normals,
        antithetic=bool(BASE_CONFIG.classical.antithetic),
        moment_match=bool(BASE_CONFIG.classical.moment_match),
        replications=int(BASE_CONFIG.classical.replications),
        replication_seed=int(BASE_CONFIG.classical.replication_seed),
        pathwise=True,
        integrated_covariances=None,
    )
    small_engine = make_engine(BASE_CONFIG.quantum.n_bits_per_asset)
    (
        cva_small,
        grid_small,
        p_joint_t_small,
        v_joint_t_small,
        p_target_small,
        w_t_small,
    ) = small_engine.cva_from_paths_discretized(
        S_by_time=fine_paths,
        t=fine_times,
        t_output=exposure_times,
        return_blocks=True,
    )
    p_t_small = small_engine.discount_factors_on_grid(exposure_times)
    q_t_small = small_engine.default_increments_on_grid(exposure_times)
    c_p = float(np.max(p_t_small))
    c_q = float(np.max(q_t_small))
    c_v = float(np.max(v_joint_t_small))
    cva_small_scaled = float(
        small_engine.cva_from_discrete_blocks(
            P_joint_t=p_joint_t_small,
            v_joint_t=v_joint_t_small,
            t=exposure_times,
            C_p=c_p,
            C_q=c_q,
            C_v=c_v,
        )
    )
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        benchmark_path,
        benchmark_methodology=np.array(CLASSICAL_BENCHMARK_METHODOLOGY),
        t=exposure_times,
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
        cva_small=np.array(cva_small),
        cva_small_scaled=np.array(cva_small_scaled),
        grid_sizes=grid_sizes,
        cva_by_grid_size_values=cva_values,
        S0=np.array(spots, dtype=float),
        K=np.array(strikes, dtype=float),
        sigma=np.array(sigma_grid, dtype=float),
        mu=np.array(drift, dtype=float),
        r=np.array(interest_rate),
        T=np.array(maturity_max),
        rho=np.array(correlation, dtype=float),
        R_cva=np.array(recovery_rate_cva),
        R_cds=np.array(float(market_data.recovery_rate)),
        LGD=np.array(1.0 - recovery_rate_cva),
        M=np.array(n_dates),
        n_bits_small=np.array(BASE_CONFIG.quantum.n_bits_per_asset, dtype=int),
        n_sigma=np.array(float(BASE_CONFIG.classical.n_sigma)),
    )


def generate_classical_benchmark(
    scenario: Scenario,
    case_dir: pathlib.Path,
    args: argparse.Namespace,
) -> pathlib.Path:
    if scenario.case_id == "base":
        benchmark = _load_npz(CANONICAL_BENCHMARK)
        reference = _classical_reference(benchmark)
        if not math.isclose(
            reference,
            CANONICAL_BASE_CLASSICAL_SMALL_GRID_CVA,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Canonical base small-grid CVA changed: "
                f"expected {CANONICAL_BASE_CLASSICAL_SMALL_GRID_CVA}, got {reference}."
            )
        return CANONICAL_BENCHMARK

    benchmark_path = case_dir / "benchmark" / "benchmark.npz"
    scenario_path = case_dir / "scenario.json"
    scenario_payload = asdict(scenario)
    if args.resume and not args.force and benchmark_path.exists():
        if not scenario_path.exists():
            raise ValueError(
                f"Cannot safely resume {scenario.case_id}: {scenario_path} is missing. "
                "Use --force to regenerate the benchmark."
            )
        previous_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        if previous_payload != scenario_payload:
            raise ValueError(
                f"Cannot safely resume {scenario.case_id}: its stress factors changed. "
                "Use --force to regenerate the benchmark."
            )
        benchmark = _load_npz(benchmark_path)
        methodology = str(np.asarray(benchmark.get("benchmark_methodology", "")).item())
        if methodology != CLASSICAL_BENCHMARK_METHODOLOGY:
            raise ValueError(
                f"Cannot safely resume {scenario.case_id}: benchmark methodology is "
                f"{methodology!r}, expected {CLASSICAL_BENCHMARK_METHODOLOGY!r}. "
                "Use --force to regenerate the benchmark."
            )
        return benchmark_path
    _write_json(scenario_path, scenario_payload)
    with _market_data_stress(scenario):
        _generate_stressed_classical_benchmark(scenario, benchmark_path, args)
    return benchmark_path


def _load_npz(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _load_theta(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required warm-start artifact does not exist: {path}")
    data = _load_npz(path)
    if "theta_star" not in data:
        raise KeyError(f"Artifact does not define theta_star: {path}")
    return np.asarray(data["theta_star"], dtype=float).ravel()


def _normalized(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).ravel()
    total = float(np.sum(out))
    if not np.all(np.isfinite(out)) or total <= 0.0:
        raise ValueError("Probability target must be finite with a positive sum.")
    return out / total


def benchmark_targets(benchmark: dict[str, Any]) -> dict[str, np.ndarray]:
    c_q = float(benchmark["C_q"])
    c_p = float(benchmark["C_p"])
    c_v = float(benchmark["C_v"])
    if min(c_q, c_p, c_v) <= 0.0:
        raise ValueError("Benchmark scaling constants must be positive.")
    return {
        "qcbm": _normalized(benchmark["p_target"]),
        "default": np.asarray(benchmark["q_t"], dtype=float).ravel() / c_q,
        "discount": np.asarray(benchmark["p_t"], dtype=float).ravel() / c_p,
        # The circuit convention is time-major, matching the benchmark pipeline.
        "exposure": np.asarray(benchmark["v_joint_t"], dtype=float).reshape(-1) / c_v,
    }


def _same_target(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=1e-10, atol=1e-12))


def _make_qcbm() -> MLQcbmCircuit:
    return MLQcbmCircuit(
        n_qubits=6,
        n_layers=6,
        name="ideal_robustness_qcbm",
        entangler="rzz",
        topology="qcbm_heavyhex6",
        backend=AerSimulator(method="statevector"),
        simulation_method="statevector",
    )


def _make_scalar_crca(name: str) -> CrcaCircuit:
    return CrcaCircuit(
        m_time=2,
        n_price=0,
        n_layers=1,
        ansatz_type="native_tree",
        native_1q_order=("rx", "rz"),
        name=name,
    )


def _make_exposure_crca() -> CrcaCircuit:
    return CrcaCircuit(
        m_time=2,
        n_price=4,
        n_layers=2,
        ansatz_type="heavy_hex_star",
        name="ideal_robustness_exposure",
    )


def _run_optimizer(
    objective: Callable[[np.ndarray], float],
    theta_init: np.ndarray,
    budget: OptimizerBudget,
    *,
    component: str,
    log_every: int,
) -> OptimizerResult:
    start = time.perf_counter()
    history: list[float] = []
    best_history: list[float] = []
    theta_best = np.asarray(theta_init, dtype=float).ravel().copy()
    theta_last = theta_best.copy()
    best_value = math.inf

    def tracked(theta: np.ndarray) -> float:
        nonlocal theta_best, theta_last, best_value
        theta_last = np.asarray(theta, dtype=float).ravel().copy()
        value = float(objective(theta_last))
        if not math.isfinite(value):
            raise ValueError(f"{component}: optimizer produced non-finite objective.")
        history.append(value)
        if value < best_value:
            best_value = value
            theta_best = theta_last.copy()
        best_history.append(best_value)
        if log_every > 0 and (len(history) == 1 or len(history) % log_every == 0):
            print(f"[{component}] evaluations={len(history)} best={best_value:.8g}")
        return value

    tracked(theta_best)
    if budget.cobyla_maxiter > 0:
        minimize(
            tracked,
            theta_best,
            method="COBYLA",
            options={"maxiter": int(budget.cobyla_maxiter), "tol": 1e-10},
        )
    if budget.lbfgsb_maxiter > 0:
        minimize(
            tracked,
            theta_best,
            method="L-BFGS-B",
            options={"maxiter": int(budget.lbfgsb_maxiter), "ftol": 1e-14},
        )
    return OptimizerResult(
        theta_best=theta_best,
        theta_last=theta_last,
        objective_history=np.asarray(history, dtype=float),
        best_objective_history=np.asarray(best_history, dtype=float),
        elapsed_s=float(time.perf_counter() - start),
    )


def _crca_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    diff = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        "mse": float(np.mean(diff * diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "l1_mean": float(np.mean(np.abs(diff))),
        "linf": float(np.max(np.abs(diff))),
    }


def _load_cached_training_artifact(
    path: pathlib.Path,
    target: np.ndarray,
) -> TrainingArtifact | None:
    if not path.exists():
        return None
    data = _load_npz(path)
    if "theta_star" not in data or "target" not in data or "prediction" not in data:
        return None
    cached_target = np.asarray(data["target"], dtype=float).ravel()
    if not _same_target(cached_target, target):
        return None
    metrics = json.loads(str(np.asarray(data["metrics_json"]).item()))
    return TrainingArtifact(
        path=path,
        theta=np.asarray(data["theta_star"], dtype=float).ravel(),
        target=cached_target,
        prediction=np.asarray(data["prediction"], dtype=float).ravel(),
        metrics={str(key): float(value) for key, value in metrics.items()},
        loaded_from_cache=True,
    )


def _save_training_artifact(
    path: pathlib.Path,
    *,
    component: str,
    theta_init: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    metrics: dict[str, float],
    result: OptimizerResult,
) -> TrainingArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        component=np.array(component),
        training_regime=np.array("ideal_statevector"),
        theta_init=np.asarray(theta_init, dtype=float),
        theta_star=np.asarray(result.theta_best, dtype=float),
        theta_last=np.asarray(result.theta_last, dtype=float),
        target=np.asarray(target, dtype=float),
        prediction=np.asarray(prediction, dtype=float),
        metrics_json=np.array(json.dumps(metrics, sort_keys=True)),
        objective_history=result.objective_history,
        best_objective_history=result.best_objective_history,
        elapsed_s=np.array(result.elapsed_s),
    )
    return TrainingArtifact(
        path=path,
        theta=result.theta_best,
        target=np.asarray(target, dtype=float),
        prediction=np.asarray(prediction, dtype=float),
        metrics=metrics,
        loaded_from_cache=False,
    )


def train_qcbm_exact(
    path: pathlib.Path,
    target: np.ndarray,
    theta_init: np.ndarray,
    budget: OptimizerBudget,
    args: argparse.Namespace,
) -> TrainingArtifact:
    if args.resume and not args.force:
        cached = _load_cached_training_artifact(path, target)
        if cached is not None:
            print(f"[SKIP] cached ideal QCBM: {path}")
            return cached
    qcbm = _make_qcbm()
    objective = qcbm.cost_fn(target, shots=None, eps=1e-12, rescaled=True)
    result = _run_optimizer(
        objective,
        theta_init,
        budget,
        component="qcbm",
        log_every=args.log_every,
    )
    prediction = qcbm.probabilities(result.theta_best, shots=None)
    metrics = qcbm.metrics(target, prediction)
    metrics["objective_final"] = float(objective(result.theta_best))
    return _save_training_artifact(
        path,
        component="qcbm",
        theta_init=theta_init,
        target=target,
        prediction=prediction,
        metrics=metrics,
        result=result,
    )


def train_crca_exact(
    path: pathlib.Path,
    component: str,
    target: np.ndarray,
    theta_init: np.ndarray,
    budget: OptimizerBudget,
    args: argparse.Namespace,
) -> TrainingArtifact:
    if args.resume and not args.force:
        cached = _load_cached_training_artifact(path, target)
        if cached is not None:
            print(f"[SKIP] cached ideal CRCA {component}: {path}")
            return cached
    circuit = _make_exposure_crca() if component == "exposure" else _make_scalar_crca(component)
    objective = circuit.cost_fn(target, shots=None)
    result = _run_optimizer(
        objective,
        theta_init,
        budget,
        component=component,
        log_every=args.log_every,
    )
    prediction = circuit.function_values(result.theta_best, shots=None)
    metrics = _crca_metrics(target, prediction)
    metrics["objective_final"] = float(objective(result.theta_best))
    return _save_training_artifact(
        path,
        component=component,
        theta_init=theta_init,
        target=target,
        prediction=prediction,
        metrics=metrics,
        result=result,
    )


def load_repository_training_artifact(
    component: str,
    path: pathlib.Path,
    target: np.ndarray,
) -> TrainingArtifact:
    theta = _load_theta(path)
    if component == "qcbm":
        circuit = _make_qcbm()
        prediction = circuit.probabilities(theta, shots=None)
        metrics = circuit.metrics(target, prediction)
    else:
        circuit = _make_exposure_crca() if component == "exposure" else _make_scalar_crca(component)
        prediction = circuit.function_values(theta, shots=None)
        metrics = _crca_metrics(target, prediction)
    return TrainingArtifact(
        path=path,
        theta=theta,
        target=np.asarray(target, dtype=float),
        prediction=np.asarray(prediction, dtype=float),
        metrics=metrics,
        loaded_from_cache=True,
    )


def _evaluate_statevector_cva(
    benchmark: dict[str, Any],
    qcbm: TrainingArtifact,
    default: TrainingArtifact,
    discount: TrainingArtifact,
    exposure: TrainingArtifact,
) -> tuple[float, float]:
    model = QuantumCVACircuit(
        num_qubits_time=2,
        num_qubits_underlying=4,
        qcbm_circuit=_make_qcbm(),
        crca_circuit_exposure=_make_exposure_crca(),
        crca_circuit_default_prob=_make_scalar_crca("default_probabilities"),
        crca_circuit_discount_factor=_make_scalar_crca("discount_factors"),
        recovery_rate=float(benchmark["R_cva"]),
        C_v=float(benchmark["C_v"]),
        C_p=float(benchmark["C_p"]),
        C_q=float(benchmark["C_q"]),
        name="ideal_robustness_cva",
        backend="statevector",
    )
    p111 = model.prob_111(
        qcbm_params=qcbm.theta,
        crca_exposure_params=exposure.theta,
        crca_default_params=default.theta,
        crca_discount_params=discount.theta,
    )
    return float(p111), float(model.cva_from_prob(p111))


def _classical_reference(benchmark: dict[str, Any]) -> float:
    """Match run_ideal_cva.py: use the classical n_bits=2 small-grid CVA."""
    grid_sizes = np.asarray(benchmark["grid_sizes"], dtype=int).ravel()
    values = np.asarray(benchmark["cva_by_grid_size_values"], dtype=float).ravel()
    matches = np.flatnonzero(grid_sizes == 2)
    if matches.size:
        return float(values[int(matches[0])])
    return float(benchmark["cva_small_scaled"])


def _encoded_target_reference(benchmark: dict[str, Any]) -> float:
    if "cva_small_scaled" not in benchmark:
        return math.nan
    return float(benchmark["cva_small_scaled"])


def _relative_error_pct(estimate: float, reference: float) -> float:
    if math.isclose(reference, 0.0, abs_tol=1e-15):
        return math.nan
    return 100.0 * (estimate - reference) / abs(reference)


def _component_artifact_path(case_dir: pathlib.Path, component: str) -> pathlib.Path:
    return case_dir / "training" / component / "training_ideal_statevector.npz"


def _component_row_fields(
    row: dict[str, Any],
    prefix: str,
    artifact: TrainingArtifact,
) -> None:
    for metric_name in ("kl", "tv", "linf", "mse"):
        if metric_name in artifact.metrics:
            row[f"{prefix}_{metric_name}"] = artifact.metrics[metric_name]


def _diagnose_for_target(
    component: str,
    artifact: TrainingArtifact,
    target: np.ndarray,
) -> TrainingArtifact:
    metrics = (
        MLQcbmCircuit.metrics(target, artifact.prediction)
        if component == "qcbm"
        else _crca_metrics(target, artifact.prediction)
    )
    return TrainingArtifact(
        path=artifact.path,
        theta=artifact.theta,
        target=np.asarray(target, dtype=float),
        prediction=artifact.prediction,
        metrics=metrics,
        loaded_from_cache=artifact.loaded_from_cache,
    )


def _write_results(output_dir: pathlib.Path, rows_by_id: dict[str, dict[str, Any]]) -> None:
    rows = list(rows_by_id.values())
    _write_csv(output_dir / "results" / "ideal_cva_robustness_results.csv", rows, RESULT_FIELDS)
    compact_fields = [
        "case_id",
        "family",
        "status",
        "cva_classical_small_grid_n_bits_2",
        "cva_statevector",
        "absolute_relative_error_vs_reference_pct",
        "qcbm_kl",
        "default_mse",
        "discount_mse",
        "exposure_mse",
    ]
    _write_markdown_table(
        output_dir / "tables" / "ideal_cva_robustness_results.md",
        rows,
        compact_fields,
    )


def _write_summary(output_dir: pathlib.Path, rows_by_id: dict[str, dict[str, Any]]) -> None:
    valid = [row for row in rows_by_id.values() if row.get("status") == "ok"]
    if not valid:
        return
    abs_errors = np.asarray([row["absolute_error_vs_reference"] for row in valid], dtype=float)
    rel_errors = np.asarray([row["relative_error_vs_reference_pct"] for row in valid], dtype=float)
    summary = {
        "successful_scenarios": len(valid),
        "failed_scenarios": len(rows_by_id) - len(valid),
        "max_absolute_error_vs_reference": float(np.max(abs_errors)),
        "mean_absolute_error_vs_reference": float(np.mean(abs_errors)),
        "max_absolute_relative_error_vs_reference_pct": float(np.nanmax(np.abs(rel_errors))),
        "mean_absolute_relative_error_vs_reference_pct": float(np.nanmean(np.abs(rel_errors))),
    }
    family_rows = []
    for family in sorted({row["family"] for row in valid}):
        family_values = [row for row in valid if row["family"] == family]
        family_rel = np.asarray(
            [row["relative_error_vs_reference_pct"] for row in family_values],
            dtype=float,
        )
        family_rows.append(
            {
                "family": family,
                "scenarios": len(family_values),
                "mean_abs_relative_error_pct": float(np.nanmean(np.abs(family_rel))),
                "max_abs_relative_error_pct": float(np.nanmax(np.abs(family_rel))),
            }
        )
    _write_json(output_dir / "results" / "ideal_cva_robustness_summary.json", summary)
    _write_csv(
        output_dir / "tables" / "ideal_cva_robustness_summary_by_family.csv",
        family_rows,
        ["family", "scenarios", "mean_abs_relative_error_pct", "max_abs_relative_error_pct"],
    )
    _write_markdown_table(
        output_dir / "tables" / "ideal_cva_robustness_summary_by_family.md",
        family_rows,
        ["family", "scenarios", "mean_abs_relative_error_pct", "max_abs_relative_error_pct"],
    )


def _existing_results(
    output_dir: pathlib.Path,
    overwrite: bool,
    allowed_case_ids: set[str],
) -> dict[str, dict[str, Any]]:
    path = output_dir / "results" / "ideal_cva_robustness_results.csv"
    if overwrite or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["case_id"]: dict(row)
            for row in csv.DictReader(handle)
            if row["case_id"] in allowed_case_ids
        }


def run_robustness(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir).resolve()
    scenarios = (
        _scenarios_from_csv(pathlib.Path(args.scenarios_csv).resolve())
        if args.scenarios_csv
        else (focused_scenarios() if args.profile == "focused" else ultra_scenarios())
    )
    if args.max_scenarios is not None:
        scenarios = scenarios[: int(args.max_scenarios)]
    _validate_scenarios(scenarios)
    write_scenario_catalog(output_dir, scenarios)
    write_scenario_design_plot(output_dir, scenarios)
    _write_json(
        output_dir / "run_configuration.json",
        {
            "arguments": vars(args),
            "scenario_count": len(scenarios),
            "repository_root": REPO_ROOT,
            "scientific_regime": "ideal logical statevector; no shots, no noise, no Runtime",
        },
    )
    print(f"scenario_count={len(scenarios)}")
    print(f"output_dir={output_dir}")
    if args.dry_run:
        print("[DRY RUN] Scenario catalog and design plots created. No benchmark or training executed.")
        return

    rows_by_id = _existing_results(
        output_dir,
        overwrite=bool(args.overwrite_results),
        allowed_case_ids={scenario.case_id for scenario in scenarios},
    )
    warmstarts = {
        "qcbm": _load_theta(QCBM_WARMSTART),
        "default": _load_theta(DEFAULT_WARMSTART),
        "discount": _load_theta(DISCOUNT_WARMSTART),
        "exposure": _load_theta(EXPOSURE_WARMSTART),
    }
    budgets = {
        "qcbm": OptimizerBudget(args.qcbm_cobyla_maxiter, args.qcbm_lbfgsb_maxiter),
        "scalar": OptimizerBudget(args.scalar_cobyla_maxiter, args.scalar_lbfgsb_maxiter),
        "exposure": OptimizerBudget(args.exposure_cobyla_maxiter, args.exposure_lbfgsb_maxiter),
    }
    base_targets: dict[str, np.ndarray] | None = None
    base_artifacts: dict[str, TrainingArtifact] | None = None

    for index, scenario in enumerate(scenarios, start=1):
        started = time.perf_counter()
        case_dir = output_dir / "cases" / scenario.case_id
        print("=" * 72)
        print(f"[{index}/{len(scenarios)}] {scenario.case_id}: {scenario.label}")
        print("=" * 72)
        row: dict[str, Any] = {
            **asdict(scenario),
            "family": scenario.family,
            "component_policy": args.component_policy,
            "statevector_evaluation_methodology": STATEVECTOR_EVALUATION_METHODOLOGY,
            "classical_benchmark_methodology": CLASSICAL_BENCHMARK_METHODOLOGY,
            "relative_error_reference": RELATIVE_ERROR_REFERENCE,
            "status": "running",
            "error_message": "",
        }
        try:
            benchmark_path = generate_classical_benchmark(scenario, case_dir, args)
            benchmark = _load_npz(benchmark_path)
            targets = benchmark_targets(benchmark)
            if base_targets is None:
                base_targets = targets

            changed = {
                component: not _same_target(targets[component], base_targets[component])
                for component in targets
            }
            is_base = scenario.case_id == "base"
            if is_base:
                qcbm = load_repository_training_artifact(
                    "qcbm",
                    QCBM_WARMSTART,
                    targets["qcbm"],
                )
                default = load_repository_training_artifact(
                    "default_probabilities",
                    DEFAULT_WARMSTART,
                    targets["default"],
                )
                discount = load_repository_training_artifact(
                    "discount_factors",
                    DISCOUNT_WARMSTART,
                    targets["discount"],
                )
                exposure = load_repository_training_artifact(
                    "exposure",
                    EXPOSURE_WARMSTART,
                    targets["exposure"],
                )
            else:
                if base_artifacts is None:
                    raise RuntimeError("Base artifacts were not initialized.")
                coherent = args.component_policy == "coherent"
                qcbm = (
                    train_qcbm_exact(
                        _component_artifact_path(case_dir, "qcbm"),
                        targets["qcbm"],
                        base_artifacts["qcbm"].theta,
                        budgets["qcbm"],
                        args,
                    )
                    if coherent and changed["qcbm"]
                    else base_artifacts["qcbm"]
                )
                default = (
                    train_crca_exact(
                        _component_artifact_path(case_dir, "default_probabilities"),
                        "default_probabilities",
                        targets["default"],
                        base_artifacts["default"].theta,
                        budgets["scalar"],
                        args,
                    )
                    if coherent and changed["default"]
                    else base_artifacts["default"]
                )
                discount = (
                    train_crca_exact(
                        _component_artifact_path(case_dir, "discount_factors"),
                        "discount_factors",
                        targets["discount"],
                        base_artifacts["discount"].theta,
                        budgets["scalar"],
                        args,
                    )
                    if coherent and changed["discount"]
                    else base_artifacts["discount"]
                )

                exposure = train_crca_exact(
                    _component_artifact_path(case_dir, "positive_exposure"),
                    "exposure",
                    targets["exposure"],
                    base_artifacts["exposure"].theta,
                    budgets["exposure"],
                    args,
                )
            if is_base:
                base_artifacts = {
                    "qcbm": qcbm,
                    "default": default,
                    "discount": discount,
                    "exposure": exposure,
                }

            p111, cva_statevector = _evaluate_statevector_cva(
                benchmark,
                qcbm=qcbm,
                default=default,
                discount=discount,
                exposure=exposure,
            )
            cva_reference = _classical_reference(benchmark)
            cva_encoded_target = _encoded_target_reference(benchmark)
            relative_error_vs_reference_pct = _relative_error_pct(cva_statevector, cva_reference)
            row.update(
                {
                    "status": "ok",
                    "cva_reference_grid_2": cva_reference,
                    "cva_classical_small_grid_n_bits_2": cva_reference,
                    "cva_encoded_target": cva_encoded_target,
                    "cva_mc_continuous": float(benchmark["cva_mc_continuous"]),
                    "cva_statevector": cva_statevector,
                    "p111_statevector": p111,
                    "absolute_error_vs_reference": abs(cva_statevector - cva_reference),
                    "relative_error_vs_reference_pct": relative_error_vs_reference_pct,
                    "absolute_relative_error_vs_reference_pct": abs(relative_error_vs_reference_pct),
                    "absolute_error_vs_encoded_target": abs(cva_statevector - cva_encoded_target),
                    "relative_error_vs_encoded_target_pct": _relative_error_pct(cva_statevector, cva_encoded_target),
                    "qcbm_target_changed": changed["qcbm"],
                    "default_target_changed": changed["default"],
                    "discount_target_changed": changed["discount"],
                    "exposure_target_changed": changed["exposure"],
                    "qcbm_retrained": (not is_base) and args.component_policy == "coherent" and changed["qcbm"],
                    "default_retrained": (not is_base) and args.component_policy == "coherent" and changed["default"],
                    "discount_retrained": (not is_base) and args.component_policy == "coherent" and changed["discount"],
                    "exposure_retrained": not is_base,
                    "C_q": float(benchmark["C_q"]),
                    "C_p": float(benchmark["C_p"]),
                    "C_v": float(benchmark["C_v"]),
                    "benchmark_path": str(benchmark_path),
                    "qcbm_artifact": str(qcbm.path),
                    "default_artifact": str(default.path),
                    "discount_artifact": str(discount.path),
                    "exposure_artifact": str(exposure.path),
                }
            )
            _component_row_fields(row, "qcbm", _diagnose_for_target("qcbm", qcbm, targets["qcbm"]))
            _component_row_fields(
                row,
                "default",
                _diagnose_for_target("default", default, targets["default"]),
            )
            _component_row_fields(
                row,
                "discount",
                _diagnose_for_target("discount", discount, targets["discount"]),
            )
            _component_row_fields(
                row,
                "exposure",
                _diagnose_for_target("exposure", exposure, targets["exposure"]),
            )
            _write_json(case_dir / "final_statevector_cva.json", row)
        except Exception as exc:
            row.update({"status": "error", "error_message": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {row['error_message']}")
            if not args.keep_going:
                row["elapsed_s"] = float(time.perf_counter() - started)
                rows_by_id[scenario.case_id] = row
                _write_results(output_dir, rows_by_id)
                raise
        row["elapsed_s"] = float(time.perf_counter() - started)
        rows_by_id[scenario.case_id] = row
        _write_results(output_dir, rows_by_id)
        _write_summary(output_dir, rows_by_id)

    if not args.skip_plots:
        from plot_ideal_cva_robustness import generate_analysis_outputs

        generate_analysis_outputs(output_dir)
    print(f"[OK] Ideal CVA robustness analysis completed: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated ideal-statevector CVA robustness analysis.",
    )
    parser.add_argument("--profile", choices=("focused", "ultra"), default="focused")
    parser.add_argument("--scenarios-csv", help="Optional custom scenario CSV. Must include a base row first.")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "analysis_output"))
    parser.add_argument("--component-policy", choices=("coherent", "exposure_only"), default="coherent")
    parser.add_argument("--n-paths", type=int, default=BASE_CONFIG.classical.n_paths)
    parser.add_argument("--fine-time-grid-size", type=int, default=BASE_CONFIG.classical.fine_time_grid_size)
    parser.add_argument("--grid-convergence-max-bits", type=int, default=BASE_CONFIG.classical.grid_convergence_max_bits)
    parser.add_argument("--qcbm-cobyla-maxiter", type=int, default=250)
    parser.add_argument("--qcbm-lbfgsb-maxiter", type=int, default=120)
    parser.add_argument("--scalar-cobyla-maxiter", type=int, default=100)
    parser.add_argument("--scalar-lbfgsb-maxiter", type=int, default=120)
    parser.add_argument("--exposure-cobyla-maxiter", type=int, default=500)
    parser.add_argument("--exposure-lbfgsb-maxiter", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one deliberately cheap scenario to verify the end-to-end workflow.",
    )
    return parser


def _apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.max_scenarios = 1
    args.n_paths = min(args.n_paths, 500)
    args.fine_time_grid_size = min(args.fine_time_grid_size, 12)
    args.grid_convergence_max_bits = min(args.grid_convergence_max_bits, 2)
    args.qcbm_cobyla_maxiter = min(args.qcbm_cobyla_maxiter, 2)
    args.qcbm_lbfgsb_maxiter = 0
    args.scalar_cobyla_maxiter = min(args.scalar_cobyla_maxiter, 2)
    args.scalar_lbfgsb_maxiter = 0
    args.exposure_cobyla_maxiter = min(args.exposure_cobyla_maxiter, 2)
    args.exposure_lbfgsb_maxiter = 0


def main() -> None:
    args = build_arg_parser().parse_args()
    _apply_smoke_overrides(args)
    run_robustness(args)


if __name__ == "__main__":
    main()
