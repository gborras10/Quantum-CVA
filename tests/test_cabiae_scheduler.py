from __future__ import annotations

import numpy as np
import pytest

from quantum_cva.algorithms.proposed_algorithms.cabiae import CABIQAELatentTheta


def _legacy_uniform_expected_fisher(
    estimator: CABIQAELatentTheta,
    k: int,
    theta_interval: tuple[float, float],
) -> float:
    K = 2 * k + 1
    grid = estimator._build_theta_grid(theta_interval, num_points=estimator._scheduler_grid_size)
    p = np.asarray(estimator._theta_to_obs_prob(grid, k), dtype=float)
    dp = np.asarray(estimator._theta_to_obs_prob_derivative(grid, k), dtype=float)
    fisher_theta = dp**2 / np.clip(p * (1.0 - p), 1e-12, None)
    angular_alignment = np.mean(np.abs(np.sin(4.0 * np.pi * K * grid)))
    return float((np.mean(fisher_theta) / max(K, 1)) * angular_alignment)


def test_scheduler_posterior_weight_is_validated() -> None:
    with pytest.raises(ValueError, match="scheduler_posterior_weight"):
        CABIQAELatentTheta(
            epsilon_target=0.1,
            alpha=0.05,
            scheduler_posterior_weight=1.1,
        )


def test_expected_fisher_weight_zero_keeps_legacy_uniform_score(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = CABIQAELatentTheta(
        epsilon_target=0.1,
        alpha=0.05,
        confint_method="beta",
        noise_model="exponential_contrast",
        T_known=10.0,
        scheduler_mode="expected_fisher",
        scheduler_posterior_weight=0.0,
        scheduler_grid_size=65,
    )
    theta_interval = (0.03, 0.12)
    expected = _legacy_uniform_expected_fisher(estimator, k=2, theta_interval=theta_interval)

    def _unexpected_density(*args: object, **kwargs: object) -> object:
        raise AssertionError("Posterior density path should not be used when weight is zero.")

    monkeypatch.setattr(estimator, "_theta_density_from_beta_posterior", _unexpected_density)

    score = estimator._expected_fisher_score(
        2,
        theta_interval,
        posterior=(4.0, 7.0),
        posterior_k=1,
    )

    assert np.isclose(score, expected)


def test_expected_fisher_weight_one_accepts_beta_posterior(monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = CABIQAELatentTheta(
        epsilon_target=0.1,
        alpha=0.05,
        confint_method="beta",
        noise_model="exponential_contrast",
        T_known=10.0,
        scheduler_mode="expected_fisher",
        scheduler_posterior_weight=1.0,
        scheduler_grid_size=65,
    )
    theta_interval = (0.03, 0.12)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original = estimator._theta_density_from_beta_posterior

    def _wrapped_density(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator, "_theta_density_from_beta_posterior", _wrapped_density)

    score = estimator._expected_fisher_score(
        2,
        theta_interval,
        posterior=(4.0, 7.0),
        posterior_k=1,
    )

    assert calls
    assert calls[0][0][:3] == (4.0, 7.0, 1)
    assert np.isfinite(score)
