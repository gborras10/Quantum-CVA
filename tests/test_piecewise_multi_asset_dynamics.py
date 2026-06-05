import numpy as np
import pandas as pd

from quantum_cva.multi_asset.classical.probability_and_underlying.multi_asset_dynamics_utils import (
    simulate_multi_asset_gbm,
)
from quantum_cva.multi_asset.classical.probability_and_underlying.piecewise_volatility_utils import (
    build_integrated_covariance_grid,
    map_piecewise_vol_to_sim_grid,
)


def test_map_piecewise_vol_uses_integrated_variance_across_buckets() -> None:
    sigma_grid = map_piecewise_vol_to_sim_grid(
        market_maturities=np.array([0.25, 0.50]),
        sigma_pw=np.array([0.20, 0.40]),
        sim_times=np.array([0.50]),
    )

    expected_variance = 0.20**2 * 0.25 + 0.40**2 * 0.25
    np.testing.assert_allclose(
        sigma_grid,
        [np.sqrt(expected_variance / 0.50)],
    )


def test_integrated_covariance_includes_each_assets_buckets() -> None:
    atm_vol_a_at_half_year = np.sqrt(
        (0.20**2 * 0.25 + 0.40**2 * 0.25) / 0.50
    )
    atm_vol_curves = pd.DataFrame(
        {
            "underlying": ["A", "A", "B"],
            "t": [0.25, 0.50, 0.50],
            "atm_vol": [0.20, atm_vol_a_at_half_year, 0.30],
        }
    )

    covariance_grid = build_integrated_covariance_grid(
        atm_vol_curves=atm_vol_curves,
        underlyings=["A", "B"],
        sim_times=np.array([0.50]),
        rho=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )

    expected = np.array(
        [
            [
                [0.20**2 * 0.25 + 0.40**2 * 0.25, 0.0225],
                [0.0225, 0.30**2 * 0.50],
            ]
        ]
    )
    np.testing.assert_allclose(covariance_grid, expected)


def test_simulation_uses_integrated_covariance_for_one_step() -> None:
    covariance = np.array([[[0.05, 0.0225], [0.0225, 0.045]]])
    z = np.array([[[1.0, -0.5]]])

    simulated = simulate_multi_asset_gbm(
        S0=[100.0, 80.0],
        mu=[0.01, 0.02],
        sigma=np.array([[np.sqrt(0.05 / 0.5), np.sqrt(0.045 / 0.5)]]),
        rho=np.eye(2),
        t=np.array([0.5]),
        Z=z,
        antithetic=False,
        moment_match=False,
        pathwise=True,
        integrated_covariances=covariance,
    )

    expected_log_increment = (
        np.array([0.01, 0.02]) * 0.5
        - 0.5 * np.diagonal(covariance[0])
        + z[0, 0] @ np.linalg.cholesky(covariance[0]).T
    )
    expected = np.array([[100.0, 80.0]]) * np.exp(expected_log_increment)
    np.testing.assert_allclose(simulated[0], expected)
