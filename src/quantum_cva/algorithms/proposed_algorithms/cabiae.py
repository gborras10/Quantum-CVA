# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Latent-theta CABIQAE.
"""

from __future__ import annotations

from collections.abc import Callable
import time
import warnings

import numpy as np
import scipy.stats as stats

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_algorithms import (
    AlgorithmError,
    AmplitudeEstimator,
    AmplitudeEstimatorResult,
    EstimationProblem,
)
from qiskit_ibm_runtime import SamplerV2


class CABIQAELatentTheta(AmplitudeEstimator):
    r"""Noisy-aware Bayesian Iterative Quantum Amplitude Estimation.

    This latent-theta variant preserves the observable-versus-ideal separation
    in the likelihood model, but performs stage-to-stage Bayesian transport in
    the latent angle ``theta`` rather than directly in the observed success
    probability. The result is a numerically more stable update rule under
    contrast-compression noise, together with a scheduler that remains faithful
    to IQAE identifiability constraints.
    """

    def __init__(
        self,
        epsilon_target: float,
        alpha: float,
        confint_method: str = "beta",
        min_ratio: float = 2,
        sampler: SamplerV2 | None = None,
        noise_model: str | None = "ideal",
        T_known: float | None = None,
        cap_kappa: float = 1.0,
        use_noise_cap: bool = True,
        max_shots_same_k: int | None = None,
        latent_grid_size: int = 2049,
        latent_resampling_size: int = 2000,
        scheduler_grid_size: int = 129,
        scheduler_mode: str = "expected_fisher",
        random_seed: int | None = None,
    ) -> None:
        r"""Initialize the latent-theta CABIQAE estimator.

        Args:
            epsilon_target: Target half-width for the final amplitude interval.
            alpha: Failure probability of the final confidence statement.
            confint_method: Stagewise interval construction rule. Supported
                values are ``"chernoff"`` and ``"beta"``.
            min_ratio: Minimum admissible stage-growth ratio for the IQAE
                ``FindNextK`` rule.
            sampler: Sampler primitive used to execute measurement circuits.
            noise_model: Observation model used to map ideal probabilities to
                observed probabilities.
            T_known: Effective decoherence scale used by the exponential
                contrast model.
            cap_kappa: Multiplicative constant defining the hard Grover-depth
                cap under the noise model.
            use_noise_cap: Whether the scheduler should enforce the hard
                noise-aware depth cap.
            max_shots_same_k: Optional limit on the cumulative number of shots
                spent at a fixed Grover depth.
            latent_grid_size: Number of grid points used to reconstruct latent
                theta densities.
            latent_resampling_size: Number of latent-theta samples used when
                transporting the posterior to the next stage prior.
            scheduler_grid_size: Number of theta grid points used by the
                scheduler score.
            scheduler_mode: Scoring rule used to rank admissible Grover depths.
            random_seed: Seed for the internal random number generator used in
                latent resampling.

        Raises:
            ValueError: If any scalar hyperparameter is outside its admissible
                range or if the selected noise or scheduler mode is unsupported.
        """
        valid_noise_models = {None, "ideal", "exponential_contrast"}
        valid_scheduler_modes = {"expected_fisher", "contrast_linear", "contrast_times_slope"}

        if not 0 < epsilon_target <= 0.5:
            raise ValueError(f"The target epsilon must be in (0, 0.5], but is {epsilon_target}.")
        
        if not 0 < alpha < 1:
            raise ValueError(f"The confidence level alpha must be in (0, 1), but is {alpha}")
        
        if not 0 < cap_kappa:
            raise ValueError(f"The cap_kappa parameter must be positive, but is {cap_kappa}.")
        
        if noise_model not in valid_noise_models:
            raise ValueError(
                f"noise_model must be one of {valid_noise_models}, but is {noise_model}."
            )
        
        if noise_model == "exponential_contrast" and T_known is None:
            raise ValueError("T_known must be provided when noise_model='exponential_contrast'.")
        
        if T_known is not None and T_known <= 0:
            raise ValueError(f"T_known must be positive, but is {T_known}.")
        
        if confint_method not in {"chernoff", "beta"}:
            raise ValueError(
                f"The interval estimation method must be `chernoff` or `beta`, but is {confint_method}."
            )
        
        if latent_grid_size < 33 or latent_grid_size % 2 == 0:
            raise ValueError("latent_grid_size must be an odd integer >= 33.")
        
        if latent_resampling_size < 100:
            raise ValueError("latent_resampling_size must be at least 100.")
        
        if scheduler_grid_size < 17:
            raise ValueError("scheduler_grid_size must be at least 17.")
        
        if scheduler_mode not in valid_scheduler_modes:
            raise ValueError(
                f"scheduler_mode must be one of {valid_scheduler_modes}, but is {scheduler_mode}."
            )

        super().__init__()

        self._epsilon = epsilon_target
        self._alpha = alpha
        self._min_ratio = min_ratio
        self._confint_method = confint_method
        self._sampler = sampler
        self._noise_model = "ideal" if noise_model is None else noise_model
        self._T_known = float(T_known) if T_known is not None else None
        self._cap_kappa = cap_kappa
        self._use_noise_cap = use_noise_cap
        self._max_shots_same_k = max_shots_same_k
        self._latent_grid_size = latent_grid_size
        self._latent_resampling_size = latent_resampling_size
        self._scheduler_grid_size = scheduler_grid_size
        self._scheduler_mode = scheduler_mode
        self._rng = np.random.default_rng(random_seed)

    @property
    def noise_model(self) -> str:
        """Return the currently configured observation-noise model."""
        return self._noise_model

    @property
    def is_noise_aware(self) -> bool:
        """Return whether the estimator is running with explicit noise awareness."""
        return self._noise_model != "ideal"

    @property
    def sampler(self) -> SamplerV2 | None:
        """Return the sampler primitive used to evaluate measurement circuits."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: SamplerV2) -> None:
        """Store the sampler primitive used to execute the estimator."""
        self._sampler = sampler

    @property
    def epsilon_target(self) -> float:
        """Return the target half-width for the final amplitude interval."""
        return self._epsilon

    @epsilon_target.setter
    def epsilon_target(self, epsilon: float) -> None:
        """Set the target half-width for the final amplitude interval."""
        self._epsilon = epsilon

    def _contrast(self, k: int) -> float:
        r"""Return the contrast factor at Grover depth ``k``.

        Args:
            k: Number of Grover applications in the queried circuit.

        Returns:
            The multiplicative contrast applied to the ideal success
            probability at depth ``k``.

        Raises:
            ValueError: If ``k`` is negative.
            RuntimeError: If the configured noise model is unsupported.
        """
        if k < 0:
            raise ValueError("k must be a non-negative integer.")
        if self._noise_model == "ideal":
            return 1.0
        if self._noise_model == "exponential_contrast":
            c = float(np.exp(-(2 * k + 1) / self._T_known))
            return min(max(c, 1e-15), 1.0)
        raise RuntimeError(f"Unsupported noise model: {self._noise_model}")

    def _theta_to_ideal_prob(self, theta: np.ndarray | float, k: int) -> np.ndarray | float:
        r"""Map latent angles to ideal success probabilities.

        Args:
            theta: Latent angle or array of latent angles, scaled to the unit
                interval.
            k: Number of Grover applications.

        Returns:
            The ideal good-state probability after amplification depth ``k``.
        """
        theta_arr = np.asarray(theta, dtype=float)
        q = np.sin((2 * k + 1) * 2 * np.pi * theta_arr) ** 2
        q = np.clip(q, 0.0, 1.0)
        if np.ndim(theta) == 0:
            return float(q)
        return q

    def _theta_to_obs_prob(self, theta: np.ndarray | float, k: int) -> np.ndarray | float:
        r"""Map latent angles to observed success probabilities.

        Args:
            theta: Latent angle or array of latent angles.
            k: Number of Grover applications.

        Returns:
            The noisy observed success probability induced by ``theta`` at
            amplification depth ``k``.
        """
        q = self._theta_to_ideal_prob(theta, k)
        c = self._contrast(k)
        p = 0.5 + c * (np.asarray(q, dtype=float) - 0.5)
        p = np.clip(p, 0.0, 1.0)
        if np.ndim(theta) == 0:
            return float(p)
        return p

    def _theta_to_obs_prob_derivative(self, theta: np.ndarray | float, k: int) -> np.ndarray | float:
        r"""Differentiate the observed success probability with respect to ``theta``.

        Args:
            theta: Latent angle or array of latent angles.
            k: Number of Grover applications.

        Returns:
            The derivative of the observed success probability with respect to
            the latent angle.
        """
        theta_arr = np.asarray(theta, dtype=float)
        u = (2 * k + 1) * 2 * np.pi * theta_arr
        deriv = self._contrast(k) * (2 * k + 1) * 2 * np.pi * np.sin(2 * u)
        if np.ndim(theta) == 0:
            return float(deriv)
        return deriv

    def _obs_to_ideal_prob(self, p_obs: float, k: int) -> float:
        r"""Invert the observation model at fixed Grover depth.

        Args:
            p_obs: Observed good-state probability.
            k: Number of Grover applications.

        Returns:
            The corresponding ideal success probability after undoing the
            configured contrast model.

        Raises:
            ValueError: If ``p_obs`` is not a valid probability.
        """
        if not (0.0 <= p_obs <= 1.0):
            raise ValueError("The observed probability p_obs must be in [0, 1].")
        c = self._contrast(k)
        q = (p_obs - 0.5) / c + 0.5
        return float(np.clip(q, 0.0, 1.0))

    def _ideal_to_obs_prob(self, q: float, k: int) -> float:
        r"""Push an ideal success probability through the observation model.

        Args:
            q: Ideal good-state probability.
            k: Number of Grover applications.

        Returns:
            The corresponding observed success probability at depth ``k``.

        Raises:
            ValueError: If ``q`` is not a valid probability.
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError("The ideal probability q must be in [0, 1].")
        c = self._contrast(k)
        p_obs = c * q + (1.0 - c) * 0.5
        return float(np.clip(p_obs, 0.0, 1.0))

    def _obs_interval_to_ideal_interval(
        self,
        p_l: float,
        p_u: float,
        k: int,
    ) -> tuple[float, float]:
        r"""Transport an observed probability interval into ideal-probability space.

        Args:
            p_l: Lower endpoint of the observed interval.
            p_u: Upper endpoint of the observed interval.
            k: Number of Grover applications.

        Returns:
            The corresponding clipped interval in ideal-probability space.

        Raises:
            ValueError: If the interval is malformed or lies outside
                ``[0, 1]``.
        """
        if not (0.0 <= p_l <= 1.0 and 0.0 <= p_u <= 1.0):
            raise ValueError("Observed probability interval must lie inside [0, 1].")
        if p_l > p_u:
            raise ValueError("Lower bound must not exceed upper bound.")
        c = self._contrast(k)
        q_l = (p_l - 0.5) / c + 0.5
        q_u = (p_u - 0.5) / c + 0.5
        return float(np.clip(q_l, 0.0, 1.0)), float(np.clip(q_u, 0.0, 1.0))

    def _build_theta_grid(
        self,
        theta_interval: tuple[float, float] | list[float],
        num_points: int | None = None,
    ) -> np.ndarray:
        """Construct a uniform grid over the current identifiable theta interval.

        Args:
            theta_interval: Interval supporting the latent angle.
            num_points: Optional override for the number of grid points.

        Returns:
            A one-dimensional theta grid covering the supplied interval.
        """
        theta_l, theta_u = float(theta_interval[0]), float(theta_interval[1])
        num = self._latent_grid_size if num_points is None else int(num_points)
        if theta_u <= theta_l:
            return np.array([theta_l], dtype=float)
        return np.linspace(theta_l, theta_u, num, dtype=float)

    def _normalize_pdf_on_grid(self, grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
        """Normalize a nonnegative density sampled on a fixed theta grid.

        Args:
            grid: Support points of the latent-angle grid.
            pdf: Unnormalized density values on ``grid``.

        Returns:
            A numerically stabilized density whose trapezoidal integral is one,
            or a uniform fallback if normalization fails.
        """
        pdf = np.nan_to_num(np.asarray(pdf, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        pdf = np.clip(pdf, 0.0, None)
        if grid.size <= 1:
            return np.array([1.0], dtype=float)
        area = float(np.trapezoid(pdf, grid))
        if not np.isfinite(area) or area <= 0.0:
            return np.full_like(grid, 1.0 / max(grid.size, 1), dtype=float)
        return pdf / area

    def _grid_cdf(self, grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
        """Build a monotone CDF from density samples on a grid.

        Args:
            grid: Support points of the latent-angle grid.
            pdf: Normalized or unnormalized density samples on ``grid``.

        Returns:
            A cumulative distribution function aligned with ``grid``.
        """
        if grid.size == 1:
            return np.array([1.0], dtype=float)
        dx = np.diff(grid)
        mass = 0.5 * (pdf[:-1] + pdf[1:]) * dx
        cdf = np.concatenate(([0.0], np.cumsum(mass)))
        total = float(cdf[-1])
        if total <= 0.0 or not np.isfinite(total):
            return np.linspace(0.0, 1.0, grid.size, dtype=float)
        cdf = cdf / total
        cdf[-1] = 1.0
        return cdf

    def _theta_density_from_beta_posterior(
        self,
        alpha: float,
        beta: float,
        k: int,
        theta_interval: tuple[float, float] | list[float],
        num_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Reconstruct a latent-theta density from a Beta posterior in observed space.

        Args:
            alpha: Alpha parameter of the Beta posterior.
            beta: Beta parameter of the Beta posterior.
            k: Grover depth at which the posterior was obtained.
            theta_interval: Current identifiable theta interval.
            num_points: Optional override for the latent grid resolution.

        Returns:
            A tuple ``(grid, pdf_theta, cdf_theta)`` describing the latent-theta
            posterior on the supplied interval.
        """
        grid = self._build_theta_grid(theta_interval, num_points=num_points)
        p_obs = np.asarray(self._theta_to_obs_prob(grid, k), dtype=float)
        dp_dtheta = np.abs(np.asarray(self._theta_to_obs_prob_derivative(grid, k), dtype=float))
        beta_pdf = stats.beta.pdf(np.clip(p_obs, 1e-12, 1.0 - 1e-12), alpha, beta)
        pdf_theta = beta_pdf * np.clip(dp_dtheta, 1e-14, None)
        pdf_theta = self._normalize_pdf_on_grid(grid, pdf_theta)
        cdf_theta = self._grid_cdf(grid, pdf_theta)
        return grid, pdf_theta, cdf_theta

    def _theta_credible_interval_from_beta_posterior(
        self,
        post: tuple[float, float],
        k: int,
        theta_interval: tuple[float, float] | list[float],
        alpha: float,
    ) -> tuple[float, float]:
        r"""Compute an equal-tailed credible interval for the latent angle.

        Args:
            post: Beta posterior parameters in observed-probability space.
            k: Grover depth at which the posterior was formed.
            theta_interval: Current identifiable theta interval.
            alpha: Tail probability assigned to the returned interval.

        Returns:
            The equal-tailed credible interval for ``theta`` inside the current
            identifiable branch.
        """
        grid, pdf_theta, cdf_theta = self._theta_density_from_beta_posterior(
            post[0],
            post[1],
            k,
            theta_interval,
        )
        if grid.size == 1 or np.allclose(pdf_theta.sum(), 0.0):
            return float(theta_interval[0]), float(theta_interval[1])
        q_l = float(np.interp(alpha / 2.0, cdf_theta, grid))
        q_u = float(np.interp(1.0 - alpha / 2.0, cdf_theta, grid))
        return q_l, q_u

    def _sample_theta_from_beta_posterior(
        self,
        alpha: float,
        beta: float,
        k: int,
        theta_interval: tuple[float, float] | list[float],
        num_samples: int,
    ) -> np.ndarray:
        r"""Sample latent angles from the reconstructed stage posterior.

        Args:
            alpha: Alpha parameter of the Beta posterior.
            beta: Beta parameter of the Beta posterior.
            k: Grover depth at which the posterior was formed.
            theta_interval: Current identifiable theta interval.
            num_samples: Number of latent-angle samples to draw.

        Returns:
            Samples from the latent-theta posterior obtained by inverse-CDF
            sampling on the reconstructed grid density.
        """
        grid, _pdf_theta, cdf_theta = self._theta_density_from_beta_posterior(
            alpha,
            beta,
            k,
            theta_interval,
        )
        u = self._rng.random(num_samples)
        return np.interp(u, cdf_theta, grid)

    def _fit_beta_moments(
            self, 
            data: np.ndarray,
            phi_cap: float | None = None
        ) -> np.ndarray:
        """Fit a Beta distribution to data by moment matching.

        Args:
            data: Probability samples to approximate with a Beta law.
            phi_cap: Optional upper bound on the Beta concentration parameter.

        Returns:
            A two-entry array containing the fitted ``(alpha, beta)``
            parameters.
        """
        x = np.asarray(data, dtype=float)
        x = np.clip(x, 1e-9, 1.0 - 1e-9)
        mu = float(np.mean(x))
        var = float(np.var(x, ddof=0))
        ceiling = mu * (1.0 - mu)
        if not np.isfinite(var) or var <= 1e-12 or ceiling <= var:
            return np.array([0.5, 0.5], dtype=float)
        phi = ceiling / var - 1.0
        if phi_cap is not None:
            phi = min(float(phi), phi_cap)
        alpha = max(mu * phi, 0.05)
        beta = max((1.0 - mu) * phi, 0.05)
        return np.array([alpha, beta], dtype=float)

    def _k_cap(self) -> int:
        r"""Return the hard Grover-depth cap implied by the noise model.

        Returns:
            The largest admissible Grover depth under the configured
            contrast-decay cap.

        Raises:
            RuntimeError: If the configured noise model is unsupported.
        """
        if self._noise_model == "ideal":
            return np.iinfo(np.int64).max
        if self._noise_model == "exponential_contrast":
            k_cap = int(np.floor((self._cap_kappa * self._T_known - 1.0) / 2.0))
            return max(k_cap, 0)
        raise RuntimeError(f"Unsupported noise model: {self._noise_model}")

    def _interval_fits_half_circle(
        self,
        k: int,
        theta_interval: tuple[float, float],
    ) -> tuple[bool, bool]:
        r"""Check whether an amplified theta interval remains identifiable.

        Args:
            k: Candidate Grover depth.
            theta_interval: Current identifiable theta interval.

        Returns:
            A pair ``(fits, upper_half_circle)`` indicating whether the scaled
            interval stays within a single half-circle and, if so, which branch
            it occupies.
        """
        theta_l, theta_u = theta_interval
        scaling = 4 * k + 2
        theta_min = (scaling * theta_l) % 1
        theta_max = (scaling * theta_u) % 1

        if theta_min <= theta_max:
            if theta_max <= 0.5:
                return True, True
            if theta_min >= 0.5:
                return True, False
        return False, True

    def _expected_fisher_score(self, k: int, theta_interval: tuple[float, float]) -> float:
        r"""Score a candidate Grover depth for the noise-aware scheduler.

        Args:
            k: Candidate Grover depth.
            theta_interval: Current identifiable theta interval.

        Returns:
            A scalar utility score balancing expected information gain and
            oracle cost under the configured scheduler mode.
        """
        K = 2 * k + 1

        if self._scheduler_mode == "contrast_linear":
            return float((self._contrast(k) ** 2) * K)

        if self._scheduler_mode == "contrast_times_slope":
            grid = self._build_theta_grid(theta_interval, num_points=self._scheduler_grid_size)
            slope_factor = np.abs(np.sin(4.0 * np.pi * K * grid))
            return float(self._contrast(k) * K * np.mean(slope_factor))

        grid = self._build_theta_grid(theta_interval, num_points=self._scheduler_grid_size)
        p = np.asarray(self._theta_to_obs_prob(grid, k), dtype=float)
        dp = np.asarray(self._theta_to_obs_prob_derivative(grid, k), dtype=float)
        fisher_theta = dp**2 / np.clip(p * (1.0 - p), 1e-12, None)

        angular_alignment = np.mean(np.abs(np.sin(4.0 * np.pi * K * grid)))

        return float((np.mean(fisher_theta) / max(K, 1)) * angular_alignment)

    def _find_next_k(
        self,
        k: int,
        upper_half_circle: bool,
        theta_interval: tuple[float, float],
        min_ratio: float = 2.0,
    ) -> tuple[int, bool]:
        r"""Apply the IQAE identifiability rule to choose the next Grover depth.

        Args:
            k: Current Grover depth.
            upper_half_circle: Current identifiable branch indicator.
            theta_interval: Current identifiable theta interval.
            min_ratio: Minimum admissible growth ratio between consecutive
                stages.

        Returns:
            The next Grover depth together with the branch flag of the
            amplified interval.

        Raises:
            AlgorithmError: If ``min_ratio`` is not strictly larger than one.
        """
        if min_ratio <= 1:
            raise AlgorithmError("min_ratio must be larger than 1 to ensure convergence")

        theta_l, theta_u = theta_interval
        old_scaling = 4 * k + 2
        max_scaling = int(1 / (2 * (theta_u - theta_l)))
        scaling = max_scaling - (max_scaling - 2) % 4
        decrement = max(4, (old_scaling // 10) - (old_scaling // 10) % 4)

        while scaling >= min_ratio * old_scaling:
            theta_min = (scaling * theta_l) % 1
            theta_max = (scaling * theta_u) % 1
            if theta_min <= theta_max:
                if theta_max <= 0.5:
                    return (scaling - 2) // 4, True
                if theta_min >= 0.5:
                    return (scaling - 2) // 4, False
            scaling -= decrement
        return k, upper_half_circle

    def _find_next_k_noise_aware(
        self,
        k: int,
        upper_half_circle: bool,
        theta_interval: tuple[float, float],
        min_ratio: float = 2.0,
    ) -> tuple[int, bool]:
        r"""Choose the next Grover depth under identifiability and noise constraints.

        The method starts from the ideal IQAE proposal, optionally truncates the
        admissible range with the hard noise cap, and then ranks feasible
        candidates with the configured information score.

        Args:
            k: Current Grover depth.
            upper_half_circle: Current identifiable branch indicator.
            theta_interval: Current identifiable theta interval.
            min_ratio: Minimum admissible growth ratio between consecutive
                stages.

        Returns:
            The selected Grover depth together with the associated branch flag.

        Raises:
            AlgorithmError: If ``min_ratio`` is not strictly larger than one.
        """
        if min_ratio <= 1:
            raise AlgorithmError("min_ratio must be larger than 1 to ensure convergence")

        k_id, upper_half_circle_id = self._find_next_k(
            k=k,
            upper_half_circle=upper_half_circle,
            theta_interval=theta_interval,
            min_ratio=min_ratio,
        )
        if not self._use_noise_cap or self._noise_model == "ideal":
            return k_id, upper_half_circle_id

        k_cap = self._k_cap()
        k_max = min(k_id, k_cap)
        old_scaling = 4 * k + 2
        candidates: list[tuple[float, int, bool]] = []

        fits_current, uhc_current = self._interval_fits_half_circle(k, theta_interval)
        if fits_current:
            candidates.append((self._expected_fisher_score(k, theta_interval), k, uhc_current))
        else:
            candidates.append((self._expected_fisher_score(k, theta_interval), k, upper_half_circle))

        for k_try in range(k + 1, k_max + 1):
            scaling_try = 4 * k_try + 2
            if scaling_try < min_ratio * old_scaling:
                continue
            fits, uhc_try = self._interval_fits_half_circle(k_try, theta_interval)
            if not fits:
                continue
            score = self._expected_fisher_score(k_try, theta_interval)
            candidates.append((score, k_try, uhc_try))

        best_score, best_k, best_uhc = max(candidates, key=lambda x: (x[0], x[1]))
        _ = best_score
        return best_k, best_uhc

    def _get_prior(
        self,
        alpha: float,
        beta: float,
        k_next: int,
        k: int,
        theta_interval: list[float],
        upper_half_circle: bool,
        num_samples: int | None = None,
    ) -> np.ndarray:
        """Prepare the next-stage Beta prior via latent-theta forward propagation.

        A density on theta is reconstructed over the current identifiable
        interval from the current-stage posterior in observed-probability space.
        Samples from this latent density are transported through the next-stage
        noisy likelihood, and a Beta prior is fitted to the resulting observed
        probabilities.

        Args:
            alpha: Alpha parameter of the current-stage Beta posterior.
            beta: Beta parameter of the current-stage Beta posterior.
            k_next: Grover depth proposed for the next stage.
            k: Grover depth used in the current stage.
            theta_interval: Current identifiable theta interval.
            upper_half_circle: Unused legacy branch flag retained for interface
                compatibility.
            num_samples: Optional override for the number of latent-theta
                samples used in the transport step.

        Returns:
            The Beta parameters of the next-stage prior in observed-probability
            space.
        """
        del upper_half_circle  # handled by the identifiable theta interval itself
        n_samples = self._latent_resampling_size if num_samples is None else int(num_samples)
        theta_samples = self._sample_theta_from_beta_posterior(
            alpha,
            beta,
            k,
            theta_interval,
            n_samples,
        )
        p_next = np.asarray(self._theta_to_obs_prob(theta_samples, k_next), dtype=float)
        
        # Contrast-compression correction
        c_next = float(self._contrast(k_next))
        phi_cap = max(c_next ** 2 * float(n_samples), 1.0)
        return self._fit_beta_moments(p_next, phi_cap=phi_cap)

    def _compute_confidence_interval(
        self,
        prob: float,
        stage_shots: int,
        stage_one_counts: int,
        prior: list[float],
        max_stages: int,
        upper_half_circle: bool,
        k: int,
        theta_interval: tuple[float, float],
    ) -> tuple[float, float, tuple[float, float] | None]:
        r"""Compute the next theta interval from the accumulated stage data.

        Args:
            prob: Empirical observed success probability of the latest batch.
            stage_shots: Total number of shots accumulated at the current
                Grover depth.
            stage_one_counts: Total number of observed good-state outcomes at
                the current Grover depth.
            prior: Prior parameters for the observed-probability model.
            max_stages: Upper bound used for stagewise confidence allocation.
            upper_half_circle: Unused legacy branch flag retained for interface
                compatibility.
            k: Current Grover depth.
            theta_interval: Current identifiable theta interval.

        Returns:
            A tuple ``(theta_l, theta_u, post)`` containing the updated latent
            interval and, when available, the Beta posterior parameters used to
            generate it.
        """
        del upper_half_circle
        if self._confint_method == "chernoff":
            p_obs_min, p_obs_max = _chernoff_confint(prob, stage_shots, max_stages, self._alpha)
            post = None
            q_min, q_max = self._obs_interval_to_ideal_interval(p_obs_min, p_obs_max, k)
            angle_min = np.arccos(1 - 2 * q_min) / (2 * np.pi)
            angle_max = np.arccos(1 - 2 * q_max) / (2 * np.pi)
            scaling = 4 * k + 2
            base = int(scaling * theta_interval[0])
            theta_l = float((base + angle_min) / scaling)
            theta_u = float((base + angle_max) / scaling)
            return theta_l, theta_u, post

        post = (
            stage_one_counts + prior[0],
            stage_shots - stage_one_counts + prior[1],
        )
        theta_l, theta_u = self._theta_credible_interval_from_beta_posterior(
            post,
            k,
            theta_interval,
            alpha=self._alpha / max_stages,
        )
        return theta_l, theta_u, post

    def construct_circuit(
        self,
        estimation_problem: EstimationProblem,
        k: int = 0,
        measurement: bool = False,
    ) -> QuantumCircuit:
        """Construct the amplified circuit evaluated at Grover depth ``k``.

        Args:
            estimation_problem: Problem instance defining the state
                preparation, Grover operator, and objective qubits.
            k: Number of Grover applications appended after state preparation.
            measurement: Whether to add measurement operations on the objective
                register.

        Returns:
            The circuit implementing ``Q^k A |0>`` with optional measurement.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")
        circuit.compose(estimation_problem.state_preparation, inplace=True)
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k).decompose(), inplace=True)
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits), "c0")
            circuit.add_register(c)
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])
        return circuit

    def _good_state_probability(
        self,
        problem: EstimationProblem,
        counts_dict: dict[str, int],
    ) -> tuple[int, float]:
        """Compute the empirical good-state probability from sampler counts.

        Args:
            problem: Estimation problem used to classify bitstrings.
            counts_dict: Counts returned by the sampler for one circuit.

        Returns:
            A pair containing the total number of good outcomes and their
            empirical relative frequency.
        """
        one_counts = 0
        for state, counts in counts_dict.items():
            if problem.is_good_state(state):
                one_counts += counts
        return int(one_counts), one_counts / sum(counts_dict.values())

    def estimate(
        self,
        estimation_problem: EstimationProblem,
        show_details: bool = False,
        bayes: bool = True,
        n_shots: int = 10,
    ) -> "BayesianIQAEResult":
        r"""Run the complete latent-theta CABIQAE estimation loop.

        The routine alternates between schedule selection, circuit execution,
        latent-theta posterior transport, and interval refinement until the
        amplitude interval width is no larger than ``2 * epsilon_target``.

        Args:
            estimation_problem: Amplitude-estimation instance to solve.
            show_details: Whether to print iteration-level diagnostics.
            bayes: Whether to propagate stage information with Beta priors and
                posteriors.
            n_shots: Number of shots used per circuit batch.

        Returns:
            A result object containing the final estimate, interval histories,
            and diagnostic counters.

        Raises:
            ValueError: If Bayesian mode is requested with a non-Beta interval
                rule.
            AlgorithmError: If sampler execution fails.
        """
        if show_details:
            print("\n" + "=" * 80)
            print(f"{'LATENT-THETA CABIQAE':^80}")
            print("=" * 80)
            print(f"Target epsilon: {self._epsilon:.6f}")
            print(f"Confidence level: {1 - self._alpha:.6f}")
            print(f"Confidence interval method: {self._confint_method}")
            print(f"Bayesian mode: {'Enabled' if bayes else 'Disabled'}")
            print(f"Scheduler mode: {self._scheduler_mode}")
            print("-" * 80)

        if bayes and self._confint_method != "beta":
            raise ValueError("Bayesian mode requires Beta credible intervals.")

        if self._sampler is None:
            warnings.warn("No sampler provided, defaulting to SamplerV2 from qiskit_ibm_runtime")
            from qiskit_aer import AerSimulator

            self._sampler = SamplerV2(mode=AerSimulator())

        powers = [0]
        ratios: list[float] = []
        theta_intervals = [[0.0, 0.25]]
        a_intervals = [[0.0, 1.0]]
        terminated_early = False
        termination_reason = None

        num_state_prep_calls = 0
        num_circuit_evaluations = 0
        num_grover_applications = 0
        num_one_shots: list[int] = []
        circuit_depths: list[int] = []
        elapsed_times: list[float] = []
        posterior_params: list[tuple[float, float] | tuple[None, None]] = []

        max_stages = int(np.log(self._min_ratio * np.pi / (8 * self._epsilon)) / np.log(self._min_ratio)) + 1
        if show_details:
            print(f"Maximum number of stages: {max_stages}")
            print(f"Number of shots taken in each iteration: {n_shots}")

        num_iterations = 0
        num_stages = 0
        upper_half_circle = True
        prior = [0.5, 0.5]
        post: tuple[float, float] | None = tuple(prior) if bayes else None

        while a_intervals[-1][1] - a_intervals[-1][0] > 2 * self._epsilon:
            if show_details:
                print("-" * 80)
                print(f"Stage: {num_stages}    Iteration: {num_iterations}")
                print(f"Current theta interval: {theta_intervals[-1]}")

            num_iterations += 1
            upper_half_circle_pre = upper_half_circle

            if show_details:
                start_time = time.time()
            k, upper_half_circle = self._find_next_k_noise_aware(
                powers[-1],
                upper_half_circle,
                tuple(theta_intervals[-1]),
                min_ratio=self._min_ratio,
            )
            if show_details:
                end_time = time.time()
                print(f"Found k={k}, running time: {end_time - start_time:.4f} seconds")

            if k != powers[-1]:
                num_stages += 1
                if bayes and post is not None:
                    if show_details:
                        start_time = time.time()
                    prior = self._get_prior(
                        post[0],
                        post[1],
                        k,
                        powers[-1],
                        theta_intervals[-1],
                        upper_half_circle_pre,
                    ).tolist()
                    if show_details:
                        end_time = time.time()
                        print(f"Updated the prior, running time: {end_time - start_time:.4f} seconds")

            powers.append(k)
            ratios.append((2 * k + 1) / (2 * powers[-2] + 1))

            if show_details:
                start_time = time.time()
            circuit = self.construct_circuit(estimation_problem, k, measurement=True)
            circuit_depths.append(circuit.depth())
            if show_details:
                end_time = time.time()
                print(
                    f"Circuit constructed with {k} Q operators. "
                    f"Depth: {circuit_depths[-1]}. "
                    f"Construction time: {end_time - start_time:.4f} seconds"
                )

            if show_details:
                start_time = time.time()
            try:
                job = self._sampler.run([circuit], shots=n_shots)
                ret = job.result()
                if show_details:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    elapsed_times.append(elapsed_time)
                    print(f"Sampled {n_shots} shots, running time: {elapsed_time:.2f} seconds")
            except Exception as exc:
                raise AlgorithmError("The job was not completed successfully.") from exc

            counts = ret[0].data.c0.get_counts()
            one_counts, prob = self._good_state_probability(estimation_problem, counts)
            num_one_shots.append(one_counts)

            num_state_prep_calls += n_shots * (2 * k + 1)
            num_circuit_evaluations += n_shots
            num_grover_applications += n_shots * k

            if show_details:
                print(f"Accumulated state-preparation calls: {num_state_prep_calls}")
                print(f"Accumulated circuit evaluations: {num_circuit_evaluations}")
                print(f"Accumulated Grover applications: {num_grover_applications}")

            j = 1
            stage_shots = n_shots
            stage_one_counts = one_counts
            if num_iterations > 1:
                while num_iterations >= j + 1 and powers[num_iterations - j] == powers[num_iterations]:
                    j += 1
                    stage_shots += n_shots
                    stage_one_counts += num_one_shots[-j]

            if self._max_shots_same_k is not None and stage_shots > self._max_shots_same_k:
                terminated_early = True
                termination_reason = (
                    f"Exceeded maximum shots for the same K: {stage_shots} > {self._max_shots_same_k}"
                )
                break

            theta_l, theta_u, post = self._compute_confidence_interval(
                prob,
                stage_shots,
                stage_one_counts,
                prior,
                max_stages,
                upper_half_circle,
                k,
                tuple(theta_intervals[-1]),
            )

            posterior_params.append(post if post is not None else (None, None))
            theta_intervals.append([theta_l, theta_u])

            a_l = float(np.sin(2 * np.pi * theta_l) ** 2)
            a_u = float(np.sin(2 * np.pi * theta_u) ** 2)
            a_intervals.append([a_l, a_u])

        confidence_interval = tuple(a_intervals[-1])
        estimation = float(np.mean(confidence_interval))

        result = BayesianIQAEResult()
        result.alpha = self._alpha
        result.post_processing = estimation_problem.post_processing
        result.num_oracle_queries = num_state_prep_calls
        result.num_state_prep_calls = num_state_prep_calls
        result.num_circuit_evaluations = num_circuit_evaluations
        result.num_grover_applications = num_grover_applications
        result.estimation = estimation
        result.epsilon_estimated = (confidence_interval[1] - confidence_interval[0]) / 2
        result.confidence_interval = confidence_interval
        result.estimation_processed = estimation_problem.post_processing(estimation)
        confidence_interval_processed = tuple(
            estimation_problem.post_processing(x) for x in confidence_interval
        )
        result.confidence_interval_processed = confidence_interval_processed
        result.epsilon_estimated_processed = (
            confidence_interval_processed[1] - confidence_interval_processed[0]
        ) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals
        result.powers = powers[1:]
        result.ratios = ratios
        result.circuit_depths = circuit_depths
        result.elapsed_times = elapsed_times
        result.posterior_params = posterior_params
        result.terminated_early = terminated_early
        result.termination_reason = termination_reason
        return result


class BayesianIQAEResult(AmplitudeEstimatorResult):
    """Result container produced by a latent-theta CABIQAE run.

    Besides the final estimate, the object stores the full interval history,
    posterior trace, Grover schedule, runtime diagnostics, and quantum-cost
    counters required for reproducible benchmark analysis.
    """

    def __init__(self) -> None:
        """Initialize an empty result container."""
        super().__init__()
        self._alpha: float | None = None
        self._epsilon_target: float | None = None
        self._epsilon_estimated: float | None = None
        self._epsilon_estimated_processed: float | None = None
        self._estimate_intervals: list[list[float]] | None = None
        self._theta_intervals: list[list[float]] | None = None
        self._powers: list[int] | None = None
        self._ratios: list[float] | None = None
        self._confidence_interval_processed: tuple[float, float] | None = None
        self._circuit_depths: list[int] | None = None
        self._elapsed_times: list[float] | None = None
        self._posterior_params: list[tuple[float, float] | tuple[None, None]] | None = None
        self._num_state_prep_calls: int | None = None
        self._num_circuit_evaluations: int | None = None
        self._num_grover_applications: int | None = None
        self._terminated_early: bool | None = None
        self._termination_reason: str | None = None

    @property
    def alpha(self) -> float:
        """Return the nominal tail probability used in the run."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Store the nominal tail probability used in the run."""
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the requested terminal half-width."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        """Store the requested terminal half-width."""
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the achieved half-width in amplitude space."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        """Store the achieved half-width in amplitude space."""
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the achieved half-width after post-processing."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        """Store the achieved half-width after post-processing."""
        self._epsilon_estimated_processed = value

    @property
    def estimate_intervals(self) -> list[list[float]]:
        """Return the amplitude intervals generated throughout the run."""
        return self._estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: list[list[float]]) -> None:
        """Store the amplitude intervals generated throughout the run."""
        self._estimate_intervals = value

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the latent-theta intervals generated throughout the run."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Store the latent-theta intervals generated throughout the run."""
        self._theta_intervals = value

    @property
    def powers(self) -> list[int]:
        """Return the Grover depths selected across stages."""
        return self._powers

    @powers.setter
    def powers(self, value: list[int]) -> None:
        """Store the Grover depths selected across stages."""
        self._powers = value

    @property
    def ratios(self) -> list[float]:
        """Return the stage-to-stage amplification ratios."""
        return self._ratios

    @ratios.setter
    def ratios(self, value: list[float]) -> None:
        """Store the stage-to-stage amplification ratios."""
        self._ratios = value

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the final confidence interval after post-processing."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, value: tuple[float, float]) -> None:
        """Store the final confidence interval after post-processing."""
        self._confidence_interval_processed = value

    @property
    def circuit_depths(self) -> list[int]:
        """Return the circuit depth recorded at each executed stage."""
        return self._circuit_depths

    @circuit_depths.setter
    def circuit_depths(self, value: list[int]) -> None:
        """Store the circuit depth recorded at each executed stage."""
        self._circuit_depths = value

    @property
    def elapsed_times(self) -> list[float]:
        """Return the measured execution times for sampler calls."""
        return self._elapsed_times

    @elapsed_times.setter
    def elapsed_times(self, value: list[float]) -> None:
        """Store the measured execution times for sampler calls."""
        self._elapsed_times = value

    @property
    def posterior_params(self) -> list[tuple[float, float] | tuple[None, None]]:
        """Return the posterior Beta parameters recorded at each stage."""
        return self._posterior_params

    @posterior_params.setter
    def posterior_params(self, value: list[tuple[float, float] | tuple[None, None]]) -> None:
        """Store the posterior Beta parameters recorded at each stage."""
        self._posterior_params = value

    @property
    def num_state_prep_calls(self) -> int:
        """Return total effective calls to the state-preparation operator."""
        return self._num_state_prep_calls

    @num_state_prep_calls.setter
    def num_state_prep_calls(self, value: int) -> None:
        """Store total effective calls to the state-preparation operator."""
        self._num_state_prep_calls = value

    @property
    def num_circuit_evaluations(self) -> int:
        """Return the total number of measured circuit executions."""
        return self._num_circuit_evaluations

    @num_circuit_evaluations.setter
    def num_circuit_evaluations(self, value: int) -> None:
        """Store the total number of measured circuit executions."""
        self._num_circuit_evaluations = value

    @property
    def num_grover_applications(self) -> int:
        """Return the total number of Grover iterates applied over all shots."""
        return self._num_grover_applications

    @num_grover_applications.setter
    def num_grover_applications(self, value: int) -> None:
        """Store the total number of Grover iterates applied over all shots."""
        self._num_grover_applications = value

    @property
    def terminated_early(self) -> bool | None:
        """Return whether the run stopped before meeting the target precision."""
        return self._terminated_early

    @terminated_early.setter
    def terminated_early(self, value: bool | None) -> None:
        """Store whether the run stopped before meeting the target precision."""
        self._terminated_early = value

    @property
    def termination_reason(self) -> str | None:
        """Return the reason for early termination, if one exists."""
        return self._termination_reason

    @termination_reason.setter
    def termination_reason(self, value: str | None) -> None:
        """Store the reason for early termination, if one exists."""
        self._termination_reason = value

def _chernoff_confint(
    value: float,
    shots: int,
    max_stages: int,
    alpha: float,
) -> tuple[float, float]:
    r"""Compute a Chernoff interval for a Bernoulli mean.

    Args:
        value: Empirical Bernoulli mean.
        shots: Number of Bernoulli samples.
        max_stages: Maximum number of stages entering the union bound.
        alpha: Global failure probability.

    Returns:
        The clipped Chernoff confidence interval.
    """
    eps = np.sqrt(3 * np.log(2 * max_stages / alpha) / shots)
    lower = np.maximum(0.0, value - eps)
    upper = np.minimum(1.0, value + eps)
    return float(lower), float(upper)