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

# Modified by Qilin Li in 2024
# Modifications include:
# - Extension of the original IQAE algorithm to a Bayesian version with Beta priors

"""The Bayesian Iterative Quantum Amplitude Estimation Algorithm."""

from __future__ import annotations
from typing import cast, Callable, Tuple
import warnings
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_ibm_runtime import SamplerV2
from qiskit_algorithms import AmplitudeEstimator, AmplitudeEstimatorResult, EstimationProblem, AlgorithmError

import time


class CABIQAE(AmplitudeEstimator):
    r"""Noisy-aware Bayesian Iterative Quantum Amplitude Estimation.
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
        noise_floor: float = 0.5,
    ) -> None:
        r"""
        Args:
            epsilon_target: Target precision for estimation target `a`, has values between 0 and 0.5
            alpha: Confidence level, the target probability is 1 - alpha, has values between 0 and 1
            confint_method: Statistical method used to compute the interval estimates in
                each iteration, can be 'chernoff' for the Chernoff intervals or 'beta' for the
                Beta credible intervals (default)
            min_ratio: Minimal q-ratio (:math:`K_{i+1} / K_i`) for FindNextK
            sampler: A sampler primitive to evaluate the circuits.
            T_known: The known value of T, the decoherence time of the noise model.
            cap_kappa: The kappa parameter for the noise cap.
            use_noise_cap: Whether to use the noise cap in the algorithm.
            max_shots_same_k: The maximum number of shots to use for each value of k.
            noise_floor: Asymptotic observed success probability when contrast
                has fully decayed. The default ``0.5`` preserves the legacy
                two-outcome noise model.
        Raises:
            AlgorithmError: if the method to compute the interval estimates is not supported
            ValueError: If the target epsilon is not in (0, 0.5]
            ValueError: If alpha is not in (0, 1)
            ValueError: If confint_method is not supported
        """
        # validate ranges of input arguments
        valid_noise_models = {None, "ideal", "exponential_contrast"}

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
            raise ValueError(
                "T_known must be provided when noise_model='exponential_contrast'."
            )

        if T_known is not None and T_known <= 0:
            raise ValueError(f"T_known must be positive, but is {T_known}.")

        if not np.isfinite(float(noise_floor)) or not 0.0 <= float(noise_floor) <= 1.0:
            raise ValueError(
                f"noise_floor must be a finite probability in [0, 1], got {noise_floor}."
            )

        if confint_method not in {"chernoff", "beta"}:
            raise ValueError(
                f"The interval estimation method must be `chernoff` or `beta`, but is {confint_method}."
            )
        
        super().__init__()

        # store parameters
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
        self._noise_floor = float(noise_floor)

    @property
    def noise_model(self) -> str:
        """Return the active noise model."""
        return self._noise_model
    
    @property   
    def is_noise_aware(self) -> bool:
        """Whether the algorithm uses an explicit noise model."""
        return self._noise_model != "ideal"

    @property
    def noise_floor(self) -> float:
        """Return the asymptotic success probability under full contrast loss."""
        return self._noise_floor

    @property
    def sampler(self) -> SamplerV2 | None:
        """Get the sampler primitive.

        Returns:
            The sampler primitive to evaluate the circuits.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: SamplerV2) -> None:
        """Set sampler primitive.

        Args:
            sampler: A sampler primitive to evaluate the circuits.
        """
        self._sampler = sampler

    @property
    def epsilon_target(self) -> float:
        """Returns the target precision ``epsilon_target`` of the algorithm.

        Returns:
            The target precision (which is half the width of the confidence interval).
        """
        return self._epsilon

    @epsilon_target.setter
    def epsilon_target(self, epsilon: float) -> None:
        """Set the target precision of the algorithm.

        Args:
            epsilon: Target precision for estimation target `a`.
        """
        self._epsilon = epsilon

    #! New method in the CABIQAE implementation: _contrast()
    def _contrast(
            self,
            k: int,
    ) -> float:
        """
        Exponential contrast factor c(k, T) for the noisy-aware likelihood.

        Args:
            k: Number of Grover operator applications used in the circuit.

        Returns:
            The contrast factor c = exp(-(2k + 1) / T_known), clipped to (0, 1].
        """
        if k < 0:
            raise ValueError("k must be a non-negative integer.")
        
        if self._noise_model == "ideal":
            return 1.0

        if self._noise_model == "exponential_contrast":
            c = float(np.exp(-(2 * k + 1) / self._T_known))
            return min(max(c, 1e-15), 1.0)

        raise RuntimeError(f"Unsupported noise model: {self._noise_model}")
    
    #! New method in the CABIQAE implementation: _obs_to_ideal_prob()
    def _obs_to_ideal_prob(
            self,
            p_obs: float,
            k: int,
    ) -> float:
        """
        Convert observed noisy probability into ideal probability using the
        exponential contrast model.
        Since this internal method will be called 

        Args:
            p_obs: Observed (empirical) probability of measuring '1' (i.e. the good state).
            k: Number of Grover operator applications.

        Returns:
            Ideal probability q corresponding to noiseless amplification

        Notes
        
            This function is applied to Beta posterior quantiles rather than point
            estimates. When the contrast factor c(k) is small, the inversion can map
            valid observed probabilities outside the interval [0, 1]. Therefore the
            result is explicitly clipped to [0, 1] to ensure numerical stability of
            subsequent interval propagation.
        """
        if not (0.0 <= p_obs <= 1.0):
            raise ValueError("The observed probability p_obs must be in [0, 1].")

        c = self._contrast(k)

        # Invert noisy observation model
        b = self._noise_floor
        q = (p_obs - b) / c + b

        # Clip to valid probability range
        return float(np.clip(q, 0.0, 1.0))
    
    #! New method in the CABIQAE implementation: _ideal_to_obs_prob()
    def _ideal_to_obs_prob(
            self,
            q:float, 
            k:int,
    ) -> float:
        """
        Convert ideal probability into observed noisy probability using the
        exponential contrast model.

        Args:
            q: Ideal probability of measuring '1' (i.e. the good state) in the noiseless case.
            k: Number of Grover operator applications.

        Returns:
            Observed probability p_obs corresponding to the noisy amplification.
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError("The ideal probability q must be in [0, 1].")
        
        c = self._contrast(k)
        b = self._noise_floor
        p_obs = c * q + (1.0 - c) * b

        return float(np.clip(p_obs, 0.0, 1.0))


    #! New method in the CABIQAE implementation: _obs_interval_to_ideal_interval()
    def _obs_interval_to_ideal_interval(
            self,
            p_l: float,
            p_u: float,
            k: int,
    ) -> tuple[float, float]:
        r"""
        Map an observed probability interval to the corresponding ideal 
        (noiseless) probability interval using the exponential 
        contrast model.

        Args:
            p_l: Lower bound of observed probability interval.
            p_u: Upper bound of observed probability interval.
            k: Number of Grover operator applications.

        Returns:
            Interval (q_l, q_u) corresponding to the noiseless probability.

        """
        if not (0.0 <= p_l <= 1.0 and 0.0 <= p_u <= 1.0):
            raise ValueError("Observed probability interval must lie inside [0, 1]")
        
        if p_l > p_u:
            raise ValueError("Lower bound must not exceed upper bound.")
        
        c = self._contrast(k)

        b = self._noise_floor
        q_l = (p_l - b) / c + b
        q_u = (p_u - b) / c + b

        # Clip to probability domain
        q_l = float(np.clip(q_l, 0.0, 1.0))
        q_u = float(np.clip(q_u, 0.0, 1.0))

        return q_l, q_u
    



    def _find_next_k(
        self,
        k: int,
        upper_half_circle: bool,
        theta_interval: tuple[float, float],
        min_ratio: float = 2.0,
    ) -> tuple[int, bool]:
        """Find the largest integer k_next, such that the interval (4 * k_next + 2)*theta_interval
        lies completely in [0, pi] or [pi, 2pi], for theta_interval = (theta_lower, theta_upper).

        Args:
            k: The current power of the Q operator.
            upper_half_circle: Boolean flag of whether theta_interval lies in the
                upper half-circle [0, pi] or in the lower one [pi, 2pi].
            theta_interval: The current confidence interval for the angle theta,
                i.e. (theta_lower, theta_upper).
            min_ratio: Minimal ratio K/K_next allowed in the algorithm.

        Returns:
            The next power k, and boolean flag for the extrapolated interval.

        Raises:
            AlgorithmError: if min_ratio is smaller or equal to 1
        """
        if min_ratio <= 1:
            raise AlgorithmError("min_ratio must be larger than 1 to ensure convergence")

        theta_l, theta_u = theta_interval
        old_scaling = 4 * k + 2  # current scaling factor, called K := (4k + 2)

        # Calculate the maximal scaling factor K, limited by the precision of the current interval
        max_scaling = int(1 / (2 * (theta_u - theta_l)))
        scaling = max_scaling - (max_scaling - 2) % 4  # bring into the form 4 * k_max + 2

        # calculate the decrement amount as 10% of the current scaling, rounded down to the nearest multiple of 4
        decrement = max(4, (old_scaling // 10) - (old_scaling // 10) % 4)

        # find the largest feasible scaling factor K_next, and thus k_next
        while scaling >= min_ratio * old_scaling:
            theta_min = (scaling * theta_l) % 1
            theta_max = (scaling * theta_u) % 1

            # Check if the scaled interval fits within the half-circle boundaries
            if theta_min <= theta_max:
                if theta_max <= 0.5:
                    # Interval is within the upper half-circle
                    return (scaling - 2) // 4, True
                elif theta_min >= 0.5:
                    # Interval is within the lower half-circle
                    return (scaling - 2) // 4, False

            scaling -= decrement

        return k, upper_half_circle
    
    #! New internal method in CABIQAE: _k_cap()
    def _k_cap(
            self,
    ) -> int:
        """
        Maximum admissible Grover depth under the hard noise-aware cap.
       
        Convention
        -------
        T_known is expressed in units of oracle-A execution time. 
        A circuit at Gorver depth k has oracle time
                    N_A(k) = 2k + 1
        The cap enforces
                    N_A(k) <= cap_kappa * T_known

        Returns
        -------
        int
            Maximum allowed k, interpreted as the number of Grover iterations m,
            such that K = 2k + 1 <= cap_kappa * T_known.

        """
        if self._noise_model == "ideal":
            return np.iinfo(np.int64).max #integer infinity
        
        if self._noise_model == "exponential_contrast":
            k_cap = int(np.floor((self._cap_kappa * self._T_known - 1.0) / 2.0))
            return max(k_cap, 0)

        raise RuntimeError(f"Unsupported noise model: {self._noise_model}")
    
    #! New helper in CABIQAE
    def _interval_fits_half_circle(
        self,
        k: int,
        theta_interval: tuple[float, float],
    ) -> tuple[bool, bool]:
        """
        Check whether the scaled interval for a given k lies entirely in the
        upper or lower half-circle.

        Returns
        -------
        (fits, upper_half_circle_flag)
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
    
    #! New internal method for choosing the next amplification noise-aware in CABIQAE:_find_next_k_noise_aware()
    def _find_next_k_noise_aware(
            self,
            k: int,
            upper_half_circle: bool,
            theta_interval: tuple[float, float],
            min_ratio: float = 2.0,
    ) -> tuple[int, bool]:
        """
        Noise-aware wrapper around the BIQAE FindNextK rule.

        It first computes the ideal BIQAE proposal, then enforces the hard
        noise-aware cap. If the cap becomes active, it searches downward for the
        largest k that is still identifiable for the current theta interval and
        still satisfies the stage-growth requirement. If none exists, the
        algorithm stays at the current k.
        """
        k_id, upper_half_circle_id = self._find_next_k(
            k=k,
            upper_half_circle=upper_half_circle,
            theta_interval=theta_interval,
            min_ratio=min_ratio,
        )

        if not self._use_noise_cap or self._noise_model == "ideal":
            return k_id, upper_half_circle_id

        k_cap = self._k_cap()

        if k_id <= k_cap:
            return k_id, upper_half_circle_id

        old_scaling = 4 * k + 2

        for k_try in range(k_cap, k - 1, -1):
            scaling_try = 4 * k_try + 2
            if scaling_try < min_ratio * old_scaling:
                continue

            fits, upper_half_circle_try = self._interval_fits_half_circle(
                k_try,
                theta_interval,
            )
            if fits:
                return k_try, upper_half_circle_try

        return k, upper_half_circle


    def _get_prior(
        self, 
        alpha: float, 
        beta: float, 
        k_next: int, 
        k: int, 
        theta_interval: list[float], 
        upper_half_circle: bool, 
        num_samples: int = 1000
    ) -> np.ndarray:
        """
        Calculate the prior distribution of prob(1) for the next stage.

        Args:
            alpha: Alpha parameter of the posterior Beta distribution at the current stage.
            beta: Beta parameter of the posterior Beta distribution at the current stage.
            k_next: The number of Grover iterations to be applied in the next stage.
            k: The current number of Grover iterations.
            theta_interval: The current interval for theta.
            upper_half_circle: Boolean indicating if the amplified angle is in the upper half-circle.
            num_samples: Number of samples to generate for fitting (default: 1000).

        Returns:
            A numpy array [alpha, beta] representing the parameters of the Beta prior for the next stage.
        """
        # Generate samples from the current posterior Beta distribution of prob(1)
        samples = np.random.beta(alpha, beta, num_samples)

        # Apply the transformation to all samples
        transformed_samples = np.array([
            self._transform_probability(x, k_next, k, theta_interval, upper_half_circle) 
            for x in samples
        ])

        # Fit a new Beta distribution to the transformed data
        result = minimize(
            self._neg_log_likelihood,
            x0=[1.0, 1.0],  # Initial guess for alpha and beta
            args=(transformed_samples,),
            method='L-BFGS-B',
            bounds=[(0.01, None), (0.01, None)]  # Ensure alpha and beta are positive
        )

        return result.x  # Return the optimized alpha and beta parameters for the new Beta distribution

    #! Modified internal method in CABIQAE: _transform_probability()
    def _transform_probability(
        self, 
        prob: float, 
        k_next: int, 
        k: int, 
        theta_interval: list[float], 
        upper_half_circle: bool
    ) -> float:
        """
        Transform the observed (success) probability from the current stage to the next stage
        under the noise-aware CABIQAE model.
        
        Args:
            prob: Probability of measuring '1' at the current stage.
            k_next: The number of Grover iterations to be applied in the next stage.
            k: The current number of Grover iterations.
            theta_interval: The current interval for theta.
            upper_half_circle: Boolean indicating if the amplified angle is in the upper half-circle.
        
        Returns:
            Probability of measuring '1' at the next stage.
        """
        # Step 1: observed -> ideal probability at current stage
        q_current = self._obs_to_ideal_prob(prob, k)

        # Step 2: ideal probability -> amplified angle in the current stage
        angle = (
            np.arccos(1 - 2 * q_current) / (2 * np.pi) if upper_half_circle 
            else 1 - np.arccos(1 - 2 * q_current) / (2 * np.pi)
        )

        # Step 3: recover theta from the current stage scaling
        scaling = 4 * k + 2  # Calculate the scaling factor based on the current k
        theta = (int(scaling * theta_interval[0]) + angle) / scaling  # compute theta from the angle

        # Step 4: propagate theta ideally to the next stage
        q_next = np.sin((2 * k_next + 1) * 2 * np.pi * theta) ** 2
        q_next = float(np.clip(q_next, 0.0, 1.0))

        # Step 5: ideal -> observed probability at next stage
        return self._ideal_to_obs_prob(q_next, k_next)

    def _neg_log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """
        Compute the negative log-likelihood function for the Beta distribution.
        
        Args:
            params: The parameters [alpha, beta] of the Beta distribution.
            data: The observed data points.
        
        Returns:
            The negative log-likelihood value.
        """
        return -np.sum(stats.beta.logpdf(data, *params))

    #! Modified internal method in the CABIQAE implementation
    def _compute_confidence_interval(
        self, 
        prob: float, 
        stage_shots: int, 
        stage_one_counts: int, 
        prior: list[float], 
        max_stages: int, 
        upper_half_circle: bool,
        k: int, #! new in CABIQAE
    ) -> tuple[float, float, tuple[float, float] | None]:
        """
        Compute the confidence interval for the angle and amplitude.
        
        Args:
            prob: Probability of measuring '1'.
            stage_shots: Total number of shots in the current stage.
            stage_one_counts: Number of '1' counts in the current stage.
            prior: Prior parameters [alpha, beta] for the Beta distribution.
            max_stages: Maximum number of stages.
            upper_half_circle: Boolean indicating if the angle is in the upper half-circle.
        
        Returns:
            A tuple containing (theta_min, theta_max, post), where post is the posterior parameters.
        """
        if self._confint_method == "chernoff":
            p_obs_min, p_obs_max = _chernoff_confint(
                prob, stage_shots, max_stages, self._alpha
            )
            post = None
        else:  # 'beta'
            post = (
                stage_one_counts + prior[0], 
                stage_shots - stage_one_counts + prior[1],
            )
            p_obs_min, p_obs_max = _beta_confint(
                self._alpha / max_stages, 
                post=post,
            )

        # Contrast correction: observed interval -> ideal interval
        q_min, q_max = self._obs_interval_to_ideal_interval(
            p_obs_min, p_obs_max, k
        )

        # Map ideal probability interval to amplified-angle interval
        if upper_half_circle:
            theta_min_i = np.arccos(1 - 2 * q_min) / 2 / np.pi
            theta_max_i = np.arccos(1 - 2 * q_max) / 2 / np.pi
        else:
            theta_min_i = 1 - np.arccos(1 - 2 * q_max) / 2 / np.pi
            theta_max_i = 1 - np.arccos(1 - 2 * q_min) / 2 / np.pi
            
        return theta_min_i, theta_max_i, post

    def construct_circuit(
        self, estimation_problem: EstimationProblem, k: int = 0, measurement: bool = False
    ) -> QuantumCircuit:
        r"""Construct the circuit :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.

        The A operator is the unitary specifying the QAE problem and Q the associated Grover
        operator.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            k: The power of the Q operator.
            measurement: Boolean flag to indicate if measurements should be included in the
                circuits.

        Returns:
            The circuit implementing :math:`\mathcal{Q}^k \mathcal{A} |0\rangle`.
        """
        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        circuit = QuantumCircuit(num_qubits, name="circuit")

        # add A operator
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        # add Q^k
        if k != 0:
            circuit.compose(estimation_problem.grover_operator.power(k).decompose(), inplace=True)
        

        # add optional measurement
        if measurement:
            # add classical register if needed
            c = ClassicalRegister(len(estimation_problem.objective_qubits), "c0")
            circuit.add_register(c)
            # real hardware can currently not handle operations after measurements, which might
            # happen if the circuit gets transpiled, hence we're adding a safeguard-barrier
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, c[:])

        return circuit

    def _good_state_probability(
        self,
        problem: EstimationProblem,
        counts_dict: dict[str, int],
    ) -> tuple[int, float]:
        """Get the probability to measure '1' in the last qubit.

        Args:
            problem: The estimation problem, used to obtain the number of objective qubits and
                the ``is_good_state`` function.
            counts_dict: A counts-dictionary (with one measured qubit only!)

        Returns:
            #one-counts, #one-counts/#all-counts
        """
        one_counts = 0
        for state, counts in counts_dict.items():
            if problem.is_good_state(state):
                one_counts += counts

        return int(one_counts), one_counts / sum(counts_dict.values())

    def estimate(
        self, estimation_problem: EstimationProblem, show_details = False, bayes = True, n_shots = 10
    ) -> "BayesianIQAEResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.
            show_details: True for printing details for each iteration.
            bayes: True for Bayesian IQAE, False for the original IQAE
            n_shots: number of shots for each iteration

        Returns:
            An amplitude estimation results object.

        Raises:
            ValueError: A Sampler must be provided.
            AlgorithmError: Sampler job run error.
            ValueError: Bayesian mode requires Beta credible intervals.
        """
        if show_details:
            print("\n" + "=" * 80)
            print(f"{'BAYESIAN ITERATIVE QUANTUM AMPLITUDE ESTIMATION':^80}")
            print("=" * 80)
            print(f"Target epsilon: {self._epsilon:.6f}")
            print(f"Confidence level: {1 - self._alpha:.6f}")
            print(f"Confidence interval method: {self._confint_method}")
            print(f"Bayesian mode: {'Enabled' if bayes else 'Disabled'}")
            print("-" * 80)
        
        if bayes and self._confint_method != "beta":
            raise ValueError("Bayesian mode requires Beta credible intervals.")

        if self._sampler is None:
            warnings.warn("No sampler provided, defaulting to SamplerV2 from qiskit_ibm_runtime")
            from qiskit_aer import AerSimulator
            self._sampler = SamplerV2(mode = AerSimulator())

        # Initialize memory variables
        powers = [0]  # List of powers k: Q^k (called 'k' in paper)
        ratios = []  # List of multiplication factors (called 'q' in paper)
        theta_intervals = [[0, 1/4]]  # Valid range of theta / (2*pi)
        a_intervals = [[0.0, 1.0]]  # Valid range of the target amplitude
        terminated_early = False
        termination_reason = None

        #! Modified convention for quantum complexity
        num_state_prep_calls = 0 # counts total effective uses of the state-preparation
        num_circuit_evaluations = 0 # counts raw circuit executions / samples #!cleanest quantity to compare with MC
        num_grover_applications = 0 # counts total Grover iterates only

        num_one_shots = [] # counts of '1' in each iteration

        circuit_depths = []  # Circuit depth in each iteration
        elapsed_times = []  # Running time of each iteration
        posterior_params = []  # Beta posterior parameters for each iteration

        # Calculate maximum number of stages
        max_stages = int(np.log(self._min_ratio * np.pi / (8 * self._epsilon)) / np.log(self._min_ratio)) + 1
        
        if show_details:
            print(f"Maximum number of stages: {max_stages}")
            print(f"Number of shots taken in each iteration: {n_shots}")
        
        # Initialize iteration variables
        num_iterations = 0
        num_stages = 0
        upper_half_circle = True  # Initially, theta is in the upper half-circle
        
        # Set initial prior (Jeffreys prior)
        prior = [0.5, 0.5]
        post = prior if bayes else None
        
        # do while loop, keep in mind that we scaled theta mod 2pi such that it lies in [0,1]
        while a_intervals[-1][1] - a_intervals[-1][0] > 2 * self._epsilon:
            if show_details:
                print("-" * 80)
                print(f"Stage: {num_stages}    Iteration: {num_iterations}")
                print(f"Current theta interval: {theta_intervals[-1]}")
            
            num_iterations += 1
            
            # Determine the next k and update upper_half_circle
            upper_half_circle_pre = upper_half_circle  # Store current upper_half_circle for prior computation
            if show_details:
                start_time = time.time()
            k, upper_half_circle = self._find_next_k_noise_aware(
                powers[-1],
                upper_half_circle,
                theta_intervals[-1],  # type: ignore
                min_ratio=self._min_ratio,
            )
            if show_details:
                end_time = time.time()
                print(f"Found k={k}, running time: {end_time - start_time:.4f} seconds")

            # Update Bayesian prior if necessary
            if k != powers[-1]:
                num_stages += 1
                if bayes:
                    if show_details:
                        start_time = time.time()
                    prior = self._get_prior(post[0], post[1], k, powers[-1], theta_intervals[-1], upper_half_circle_pre)
                    if show_details:
                        end_time = time.time()
                        print(f"Updated the prior, running time: {end_time - start_time:.4f} seconds")

            # store the variables
            powers.append(k)
            ratios.append((2 * k + 1) / (2 * powers[-2] + 1))

            # run measurements for Q^k A|0> circuit
            # construct the circuit
            if show_details:
                start_time = time.time()
            circuit = self.construct_circuit(estimation_problem, k, measurement=True)
            circuit_depths.append(circuit.depth())
            if show_details:
                end_time = time.time()
                print(f"Circuit constructed with {k} Q operators. Depth: {circuit_depths[-1]}. Construction time: {end_time - start_time:.4f} seconds")

            # Run the circuit
            if show_details:
                start_time = time.time()

            try:
                job = self._sampler.run([circuit], shots = n_shots)
                ret = job.result()
                if show_details:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    elapsed_times.append(elapsed_time)
                    print(f"Sampled {n_shots} shots, running time: {elapsed_time:.2f} seconds")
            except Exception as exc:
                raise AlgorithmError("The job was not completed successfully.") from exc

            
            # Extract shots and counts from `ret`
            counts = ret[0].data.c0.get_counts()

            # calculate the probability of measuring '1', 'prob' is a_i in the paper
            one_counts, prob = self._good_state_probability(estimation_problem, counts)
            num_one_shots.append(one_counts)

            # track number of Q-oracle calls
            num_state_prep_calls += n_shots * (2 * k + 1)
            num_circuit_evaluations += n_shots
            num_grover_applications += n_shots * k

            if show_details:
                print(f"Accumulated state-preparation calls: {num_state_prep_calls}")
                print(f"Accumulated circuit evaluations: {num_circuit_evaluations}")
                print(f"Accumulated Grover applications: {num_grover_applications}")    

            # if on the previous iterations we have K_{i-1} == K_i, we sum these samples up
            j = 1  # number of times we stayed fixed at the same K
            stage_shots = n_shots
            stage_one_counts = one_counts
            if num_iterations > 1:
                while num_iterations >= j + 1 and powers[num_iterations - j] == powers[num_iterations]:
                    j += 1
                    stage_shots += n_shots
                    stage_one_counts += num_one_shots[-j]
            if self._max_shots_same_k is not None and stage_shots > self._max_shots_same_k:
                terminated_early = True
                termination_reason = f"Exceeded maximum shots for the same K: {stage_shots} > {self._max_shots_same_k}"
                break
            # compute confidence intervals
            theta_min_i, theta_max_i, post = self._compute_confidence_interval(
                prob, stage_shots, stage_one_counts, prior, max_stages, upper_half_circle, k #! k -> new argument in CABIQAE
            )
            
            # Store posterior parameters if available
            if post is not None:
                posterior_params.append(post)
            else:
                posterior_params.append((None, None))

            # compute theta_u, theta_l of this iteration by adding the base to the angle and scaling
            scaling = 4 * k + 2  # current K_i factor
            theta_u = (int(scaling * theta_intervals[-1][0]) + theta_max_i) / scaling
            theta_l = (int(scaling * theta_intervals[-1][0]) + theta_min_i) / scaling
            theta_intervals.append([theta_l, theta_u])

            # compute a_u_i, a_l_i
            a_u = np.sin(2 * np.pi * theta_u) ** 2
            a_l = np.sin(2 * np.pi * theta_l) ** 2
            a_u = cast(float, a_u)
            a_l = cast(float, a_l)
            a_intervals.append([a_l, a_u])

        # get the latest confidence interval for the estimate of a
        confidence_interval = cast(Tuple[float, float], a_intervals[-1])

        # the final estimate is the mean of the confidence interval
        estimation = np.mean(confidence_interval)

        
        # Construct the result object   
        result = BayesianIQAEResult()
        result.alpha = self._alpha
        result.post_processing = cast(Callable[[float], float], estimation_problem.post_processing)
        result.num_oracle_queries = num_state_prep_calls

        #! Additional counters for diagnostics and benchmarking
        result.num_state_prep_calls = num_state_prep_calls
        result.num_circuit_evaluations = num_circuit_evaluations
        result.num_grover_applications = num_grover_applications

        result.estimation = float(estimation)
        result.epsilon_estimated = (confidence_interval[1] - confidence_interval[0]) / 2
        result.confidence_interval = confidence_interval

        result.estimation_processed = estimation_problem.post_processing(
            estimation  # type: ignore[arg-type,assignment]
        )
        confidence_interval = tuple(
            estimation_problem.post_processing(x)  # type: ignore[arg-type,assignment]
            for x in confidence_interval
        )

        result.confidence_interval_processed = confidence_interval
        result.epsilon_estimated_processed = (confidence_interval[1] - confidence_interval[0]) / 2
        result.estimate_intervals = a_intervals
        result.theta_intervals = theta_intervals
        result.powers = powers[1:]
        result.ratios = ratios
        
        result.circuit_depths = circuit_depths
        result.elapsed_times = elapsed_times  
        result.posterior_params = posterior_params  # Add posterior parameters to result

        result.terminated_early = terminated_early
        result.termination_reason = termination_reason

        return result

class BayesianIQAEResult(AmplitudeEstimatorResult):
    """The ``BayesianIQAEResult`` result object."""

    def __init__(self) -> None:
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
        self._posterior_params: list[tuple[float, float]] | None = None
        #! new counters
        self._num_state_prep_calls: int | None = None
        self._num_circuit_evaluations: int | None = None
        self._num_grover_applications: int | None = None
        self._terminated_early: bool | None = None
        self._termination_reason: str | None = None

    @property
    def alpha(self) -> float:
        r"""Return the confidence level :math:`\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        r"""Set the confidence level :math:`\alpha`."""
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the target half-width of the confidence interval."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        """Set the target half-width of the confidence interval."""
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the estimated half-width of the confidence interval."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        """Set the estimated half-width of the confidence interval."""
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the post-processed estimated half-width of the confidence interval."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        """Set the post-processed estimated half-width of the confidence interval."""
        self._epsilon_estimated_processed = value

    @property
    def estimate_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the estimate in each iteration."""
        return self._estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the estimate in each iteration."""
        self._estimate_intervals = value

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the confidence intervals for the angles in each iteration."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        """Set the confidence intervals for the angles in each iteration."""
        self._theta_intervals = value

    @property
    def powers(self) -> list[int]:
        """Return the powers of the Grover operator in each iteration."""
        return self._powers

    @powers.setter
    def powers(self, value: list[int]) -> None:
        """Set the powers of the Grover operator in each iteration."""
        self._powers = value

    @property
    def ratios(self) -> list[float]:
        r"""Return the ratios :math:`K_{i+1}/K_{i}` for each iteration :math:`i`."""
        return self._ratios

    @ratios.setter
    def ratios(self, value: list[float]) -> None:
        r"""Set the ratios :math:`K_{i+1}/K_{i}` for each iteration :math:`i`."""
        self._ratios = value

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the post-processed confidence interval."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, value: tuple[float, float]) -> None:
        """Set the post-processed confidence interval."""
        self._confidence_interval_processed = value

    @property
    def circuit_depths(self) -> list[int]:
        """Return the circuit depths for each iteration."""
        return self._circuit_depths

    @circuit_depths.setter
    def circuit_depths(self, value: list[int]) -> None:
        """Set the circuit depths for each iteration."""
        self._circuit_depths = value

    @property
    def elapsed_times(self) -> list[float]:
        """Return the elapsed times for each iteration."""
        return self._elapsed_times

    @elapsed_times.setter
    def elapsed_times(self, value: list[float]) -> None:
        """Set the elapsed times for each iteration."""
        self._elapsed_times = value

    @property
    def posterior_params(self) -> list[tuple[float, float]]:
        """Return the Beta posterior parameters (alpha, beta) for each iteration."""
        return self._posterior_params

    @posterior_params.setter
    def posterior_params(self, value: list[tuple[float, float]]) -> None:
        """Set the Beta posterior parameters."""
        self._posterior_params = value

    #! new properties
    @property
    def num_state_prep_calls(self) -> int:
        """
        Return total effective uses of the state-preparation oracle A.

        Notes
        -----
        One circuit execution at Grover depth k contributes 2k + 1 calls,
        counting A and A^{-1} at equal unit cost.
        """
        return self._num_state_prep_calls

    @num_state_prep_calls.setter
    def num_state_prep_calls(self, value: int) -> None:
        """Set total effective uses of the state-preparation oracle A."""
        self._num_state_prep_calls = value

    @property
    def num_circuit_evaluations(self) -> int:
        """
        Return total number of measured circuit executions.

        Notes
        -----
        This is the cleanest counter to compare against the number of Monte Carlo
        samples / paths, since each circuit execution produces one Bernoulli sample.
        """
        return self._num_circuit_evaluations

    @num_circuit_evaluations.setter
    def num_circuit_evaluations(self, value: int) -> None:
        """Set total number of measured circuit executions."""
        self._num_circuit_evaluations = value

    @property
    def num_grover_applications(self) -> int:
        """
        Return total number of Grover iterates applied across all shots.

        Notes
        -----
        This is a scheduler diagnostic only. It is not the primary quantum
        complexity metric used for CABIQAE benchmarking.
        """
        return self._num_grover_applications

    @num_grover_applications.setter
    def num_grover_applications(self, value: int) -> None:
        """Set total number of Grover iterates applied across all shots."""
        self._num_grover_applications = value

    @property
    def terminated_early(self) -> bool | None:
        return self._terminated_early

    @terminated_early.setter
    def terminated_early(self, value: bool | None) -> None:
        self._terminated_early = value

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    @termination_reason.setter
    def termination_reason(self, value: str | None) -> None:
        self._termination_reason = value


def _chernoff_confint(
    value: float, shots: int, max_stages: int, alpha: float
) -> tuple[float, float]:
    """Compute the Chernoff confidence interval for `shots` i.i.d. Bernoulli trials.

    The confidence interval is

        [value - eps, value + eps], where eps = sqrt(3 * log(2 * max_stages/ alpha) / shots)

    but at most [0, 1].

    Args:
        value: The current estimate.
        shots: The number of shots.
        max_stages: The maximum number of stages, used to compute epsilon_a.
        alpha: The confidence level, used to compute epsilon_a.

    Returns:
        The Chernoff confidence interval.
    """
    eps = np.sqrt(3 * np.log(2 * max_stages / alpha) / shots)
    lower = np.maximum(0, value - eps)
    upper = np.minimum(1, value + eps)
    return lower, upper


def _beta_confint(alpha: float, post: tuple[float, float]) -> tuple[float, float]:
    """Compute the Beta confidence interval for `shots` i.i.d. Bernoulli trials.

    Args:
        alpha: The confidence level for the confidence interval.
        post: A tuple containing the parameters of the posterior Beta distribution.

    Returns:
        The Beta confidence interval.
    """
    lower, upper = 0, 1

    # if counts == 0, the beta quantile returns nan
    lower = stats.beta.ppf(alpha / 2, post[0], post[1])

    # if counts == shots, the beta quantile returns nan
    upper = stats.beta.ppf(1 - alpha / 2, post[0], post[1])

    return lower, upper
