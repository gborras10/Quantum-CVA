"""ELF amplitude estimation following Alcazar et al., Appendix E.

This module implements the engineered likelihood function (ELF) amplitude
estimator used by Alcazar et al. for quantum CVA. The estimator keeps a
Gaussian belief over ``theta`` where ``eta = cos(theta) = <A|O|A>`` and
``O = 2 Pi - I``. It uses the ancilla-free ELF circuit family

    Q(x)|A> = V(x_2L) U(x_2L-1) ... V(x_2) U(x_1)|A>,

with ``U(x) = exp(i x Pi)`` and
``V(y) = A exp(i y |0><0|) A^dagger``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast
import time
import warnings

import numpy as np
from scipy import optimize, stats

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit_algorithms import (
    AlgorithmError,
    AmplitudeEstimator,
    AmplitudeEstimatorResult,
    EstimationProblem,
)


class ELFQAE(AmplitudeEstimator):
    r"""Engineered likelihood function QAE from Alcazar et al.

    The posterior approximation is Gaussian throughout the algorithm. A small
    local stencil is only used to fit the sinusoidal approximation to the ELF
    bias, as described in the source ELF method referenced by Alcazar et al.;
    it is not used as a posterior representation.
    """

    _I2 = np.eye(2, dtype=complex)
    _X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    _Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    _KET0 = np.array([1.0, 0.0], dtype=complex)
    _KET0_PROJECTOR = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    def __init__(
        self,
        epsilon_target: float,
        alpha: float,
        sampler: Any | None = None,
        layers: int = 1,
        max_layers: int | None = None,
        layer_selection: str = "fixed",
        circuit_fidelity: float | None = None,
        spam_fidelity: float = 1.0,
        layer_fidelity: float = 1.0,
        initial_eta_mean: float = 0.0,
        initial_eta_std: float = 0.5,
        initial_theta_mean: float | None = None,
        initial_theta_std: float | None = None,
        max_rounds: int = 10_000,
        max_state_prep_calls: int | None = None,
        optimizer_restarts: int = 8,
        local_fit_points: int = 9,
        local_fit_width_sigma: float = 2.0,
        finite_difference_step: float = 1e-6,
        min_theta_std: float = 1e-9,
        random_seed: int | None = None,
    ) -> None:
        r"""Initialize the Alcazar ELF estimator.

        Args:
            epsilon_target: Target half-width in amplitude space.
            alpha: Tail probability for Gaussian credible intervals.
            sampler: Sampler primitive used to execute ELF circuits.
            layers: Fixed number of ELF layers when ``layer_selection`` is
                ``"fixed"``.
            max_layers: Maximum layer count when adaptive layer selection is
                enabled. Defaults to ``layers``.
            layer_selection: Either ``"fixed"`` or ``"fisher_per_cost"``.
            circuit_fidelity: Full circuit fidelity. If provided, this is used
                directly in the likelihood.
            spam_fidelity: SPAM fidelity factor used when
                ``circuit_fidelity`` is not provided.
            layer_fidelity: Per-layer fidelity used when
                ``circuit_fidelity`` is not provided.
            initial_eta_mean: Mean of the initial eta belief.
            initial_eta_std: Standard deviation of the initial eta belief.
            initial_theta_mean: Optional direct initial theta mean.
            initial_theta_std: Optional direct initial theta standard
                deviation.
            max_rounds: Maximum number of one-shot ELF rounds.
            max_state_prep_calls: Optional cap on effective state-preparation
                calls.
            optimizer_restarts: Number of random optimizer restarts.
            local_fit_points: Odd number of local points for the sinusoid fit.
            local_fit_width_sigma: Local fit half-width in prior sigmas.
            finite_difference_step: Step used for theta derivatives.
            min_theta_std: Lower numerical bound for theta standard deviation.
            random_seed: Seed for the internal random number generator.
        """
        if not 0 < epsilon_target <= 0.5:
            raise ValueError("epsilon_target must be in (0, 0.5].")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if layers < 1:
            raise ValueError("layers must be a positive integer.")
        if max_layers is not None and max_layers < 1:
            raise ValueError("max_layers must be a positive integer.")
        if layer_selection not in {"fixed", "fisher_per_cost"}:
            raise ValueError("layer_selection must be 'fixed' or 'fisher_per_cost'.")
        for name, value in {
            "spam_fidelity": spam_fidelity,
            "layer_fidelity": layer_fidelity,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        if circuit_fidelity is not None and not 0 <= circuit_fidelity <= 1:
            raise ValueError("circuit_fidelity must be in [0, 1].")
        if not -1 <= initial_eta_mean <= 1:
            raise ValueError("initial_eta_mean must be in [-1, 1].")
        if initial_eta_std <= 0:
            raise ValueError("initial_eta_std must be positive.")
        if initial_theta_std is not None and initial_theta_std <= 0:
            raise ValueError("initial_theta_std must be positive.")
        if max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer.")
        if max_state_prep_calls is not None and max_state_prep_calls < 1:
            raise ValueError("max_state_prep_calls must be positive.")
        if optimizer_restarts < 0:
            raise ValueError("optimizer_restarts must be non-negative.")
        if local_fit_points < 3 or local_fit_points % 2 == 0:
            raise ValueError("local_fit_points must be an odd integer >= 3.")
        if local_fit_width_sigma <= 0:
            raise ValueError("local_fit_width_sigma must be positive.")
        if finite_difference_step <= 0:
            raise ValueError("finite_difference_step must be positive.")
        if min_theta_std <= 0:
            raise ValueError("min_theta_std must be positive.")

        super().__init__()

        self._epsilon = float(epsilon_target)
        self._alpha = float(alpha)
        self._sampler = sampler
        self._layers = int(layers)
        self._max_layers = int(max_layers) if max_layers is not None else int(layers)
        self._layer_selection = layer_selection
        self._circuit_fidelity = circuit_fidelity
        self._spam_fidelity = float(spam_fidelity)
        self._layer_fidelity = float(layer_fidelity)
        self._initial_eta_mean = float(initial_eta_mean)
        self._initial_eta_std = float(initial_eta_std)
        self._initial_theta_mean = initial_theta_mean
        self._initial_theta_std = initial_theta_std
        self._max_rounds = int(max_rounds)
        self._max_state_prep_calls = max_state_prep_calls
        self._optimizer_restarts = int(optimizer_restarts)
        self._local_fit_points = int(local_fit_points)
        self._local_fit_width_sigma = float(local_fit_width_sigma)
        self._finite_difference_step = float(finite_difference_step)
        self._min_theta_std = float(min_theta_std)
        self._rng = np.random.default_rng(random_seed)

    @property
    def sampler(self) -> Any | None:
        """Return the sampler primitive used by the estimator."""
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: Any) -> None:
        """Store the sampler primitive used by the estimator."""
        self._sampler = sampler

    @property
    def epsilon_target(self) -> float:
        """Return the requested amplitude half-width."""
        return self._epsilon

    @property
    def alpha(self) -> float:
        """Return the tail probability used for Gaussian intervals."""
        return self._alpha

    @property
    def layers(self) -> int:
        """Return the fixed ELF layer count."""
        return self._layers

    @property
    def max_layers(self) -> int:
        """Return the maximum allowed ELF layer count."""
        return self._max_layers

    def _initial_theta_distribution(self) -> tuple[float, float]:
        """Return the Gaussian theta belief used at the start of inference."""
        if self._initial_theta_mean is not None:
            if self._initial_theta_std is None:
                raise ValueError(
                    "initial_theta_std must be set with initial_theta_mean."
                )
            return (
                self._clip_theta(float(self._initial_theta_mean)),
                max(float(self._initial_theta_std), self._min_theta_std),
            )

        eta = float(np.clip(self._initial_eta_mean, -1.0 + 1e-12, 1.0 - 1e-12))
        theta = float(np.arccos(eta))
        jacobian = 1.0 / max(np.sqrt(1.0 - eta * eta), 1e-12)
        theta_std = max(self._initial_eta_std * jacobian, self._min_theta_std)
        return self._clip_theta(theta), float(theta_std)

    @staticmethod
    def _clip_probability(value: float) -> float:
        """Clip a probability away from exact zero and one."""
        return float(np.clip(value, 1e-15, 1.0 - 1e-15))

    @staticmethod
    def _clip_eta(value: float) -> float:
        """Clip eta to the physically valid interval."""
        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _clip_theta(value: float) -> float:
        """Clip theta to the physical interval used by eta = cos(theta)."""
        return float(np.clip(value, 0.0, np.pi))

    @staticmethod
    def eta_to_amplitude(eta: float | np.ndarray) -> float | np.ndarray:
        """Map Alcazar's eta expectation to the comparable amplitude."""
        return (np.asarray(eta, dtype=float) + 1.0) / 2.0

    @staticmethod
    def amplitude_to_eta(amplitude: float | np.ndarray) -> float | np.ndarray:
        """Map an amplitude for Pi to Alcazar's eta expectation."""
        return 2.0 * np.asarray(amplitude, dtype=float) - 1.0

    def _fidelity(self, layers: int) -> float:
        """Return the likelihood bias rescaling factor for a layer count."""
        if self._circuit_fidelity is not None:
            return float(self._circuit_fidelity)
        fidelity = self._spam_fidelity * self._layer_fidelity**layers
        return float(np.clip(fidelity, 0.0, 1.0))

    def _observable(self, theta: float) -> np.ndarray:
        r"""Return the logical observable O(theta).

        In the virtual basis ``|A> == |0bar>`` and
        ``|A_perp> == |1bar>``, this is
        ``O(theta) = cos(theta) Z + sin(theta) X``.
        """
        return np.cos(theta) * self._Z + np.sin(theta) * self._X

    def _projector(self, theta: float) -> np.ndarray:
        """Return the logical projector Pi(theta) = (I + O(theta)) / 2."""
        return 0.5 * (self._I2 + self._observable(theta))

    def _logical_u(self, theta: float, x: float) -> np.ndarray:
        """Return U(x) = exp(i x Pi(theta)) in the logical subspace."""
        projector = self._projector(theta)
        return self._I2 + (np.exp(1j * x) - 1.0) * projector

    def _logical_v(self, y: float) -> np.ndarray:
        """Return V(y) = exp(i y |0bar><0bar|) in the logical subspace."""
        return self._I2 + (np.exp(1j * y) - 1.0) * self._KET0_PROJECTOR

    def _q_matrix(self, theta: float, phase_controls: Sequence[float]) -> np.ndarray:
        """Return the logical ELF unitary Q(x)."""
        phases = np.asarray(phase_controls, dtype=float)
        if phases.ndim != 1 or len(phases) % 2 != 0 or len(phases) == 0:
            raise ValueError("phase_controls must have length 2L with L >= 1.")

        q_matrix = self._I2.copy()
        for u_angle, v_angle in phases.reshape((-1, 2)):
            q_matrix = (
                self._logical_v(float(v_angle))
                @ self._logical_u(theta, float(u_angle))
                @ q_matrix
            )
        return q_matrix

    def _bias(
        self,
        theta: float | np.ndarray,
        phase_controls: Sequence[float],
    ) -> float | np.ndarray:
        r"""Return Delta(theta; x) = <A|Q^dag O Q|A>."""
        theta_arr = np.asarray(theta, dtype=float)
        flat = theta_arr.reshape(-1)
        values = []
        for theta_value in flat:
            q_matrix = self._q_matrix(float(theta_value), phase_controls)
            value = (
                self._KET0.conj()
                @ (q_matrix.conj().T @ self._observable(float(theta_value)) @ q_matrix)
                @ self._KET0
            )
            values.append(float(np.real_if_close(value).real))

        out = np.asarray(values, dtype=float).reshape(theta_arr.shape)
        out = np.clip(out, -1.0, 1.0)
        if np.ndim(theta) == 0:
            return float(out)
        return out

    def _bias_derivative(
        self,
        theta: float,
        phase_controls: Sequence[float],
    ) -> float:
        """Return the theta derivative of the logical ELF bias."""
        step = self._finite_difference_step
        return float(
            (
                self._bias(theta + step, phase_controls)
                - self._bias(theta - step, phase_controls)
            )
            / (2.0 * step)
        )

    def _likelihood_probability(
        self,
        outcome: int,
        theta: float,
        fidelity: float,
        phase_controls: Sequence[float],
    ) -> float:
        """Return P(d | theta; f, x) for d in {0, 1}."""
        sign = 1.0 if outcome == 0 else -1.0
        bias = float(self._bias(theta, phase_controls))
        return self._clip_probability(0.5 * (1.0 + sign * fidelity * bias))

    def _fisher_information(
        self,
        theta: float,
        fidelity: float,
        phase_controls: Sequence[float],
    ) -> float:
        r"""Return J(theta; f, x) for the two-outcome ELF likelihood."""
        bias = float(self._bias(theta, phase_controls))
        derivative = self._bias_derivative(theta, phase_controls)
        numerator = fidelity * fidelity * derivative * derivative
        denominator = max(1.0 - fidelity * fidelity * bias * bias, 1e-15)
        return float(numerator / denominator)

    def _optimize_phase_controls(
        self,
        mu: float,
        sigma: float,
        fidelity: float,
        layers: int,
    ) -> np.ndarray:
        """Choose ELF phases by maximizing Fisher information at theta=mu."""
        del sigma
        if layers < 1:
            raise ValueError("layers must be positive.")

        bounds = [(0.0, 2.0 * np.pi)] * (2 * layers)
        seeds = [np.full(2 * layers, np.pi)]
        for _ in range(self._optimizer_restarts):
            seeds.append(self._rng.uniform(0.0, 2.0 * np.pi, size=2 * layers))

        def objective(phases: np.ndarray) -> float:
            return -self._fisher_information(mu, fidelity, phases)

        best_phases = seeds[0]
        best_score = objective(best_phases)
        for seed in seeds:
            result = optimize.minimize(
                objective,
                seed,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-12},
            )
            score = float(result.fun) if result.success else objective(seed)
            phases = np.asarray(result.x if result.success else seed, dtype=float)
            if score < best_score:
                best_score = score
                best_phases = phases

        return np.mod(best_phases, 2.0 * np.pi)

    def _select_layer_count(self, mu: float, sigma: float) -> tuple[int, np.ndarray, float]:
        """Select the layer count and phases for the next ELF round."""
        if self._layer_selection == "fixed":
            layers = self._layers
            fidelity = self._fidelity(layers)
            phases = self._optimize_phase_controls(mu, sigma, fidelity, layers)
            return layers, phases, fidelity

        best_layers = 1
        best_fidelity = self._fidelity(1)
        best_phases = self._optimize_phase_controls(mu, sigma, best_fidelity, 1)
        best_score = self._fisher_information(mu, best_fidelity, best_phases) / 3.0
        for layers in range(2, self._max_layers + 1):
            fidelity = self._fidelity(layers)
            phases = self._optimize_phase_controls(mu, sigma, fidelity, layers)
            score = self._fisher_information(mu, fidelity, phases) / (2 * layers + 1)
            if score > best_score:
                best_layers = layers
                best_fidelity = fidelity
                best_phases = phases
                best_score = score
        return best_layers, best_phases, best_fidelity

    def _fit_local_sinusoid(
        self,
        mu: float,
        sigma: float,
        phase_controls: Sequence[float],
    ) -> tuple[float, float]:
        r"""Fit Delta(theta; x) ~= sin(r theta + b) near the prior mean."""
        half_width = max(
            self._local_fit_width_sigma * sigma,
            10.0 * self._finite_difference_step,
        )
        theta_values = np.linspace(
            mu - half_width,
            mu + half_width,
            self._local_fit_points,
        )
        biases = np.asarray(self._bias(theta_values, phase_controls), dtype=float)
        z_values = np.unwrap(np.arcsin(np.clip(biases, -1.0, 1.0)))
        design = np.column_stack([theta_values, np.ones_like(theta_values)])
        slope, intercept = np.linalg.lstsq(design, z_values, rcond=None)[0]
        return float(slope), float(intercept)

    def _gaussian_update(
        self,
        mu: float,
        sigma: float,
        outcome: int,
        fidelity: float,
        r: float,
        b: float,
    ) -> tuple[float, float]:
        r"""Project the sinusoidal-likelihood posterior back to a Gaussian."""
        if outcome not in {0, 1}:
            raise ValueError("outcome must be 0 or 1.")

        sign = 1.0 if outcome == 0 else -1.0
        variance = sigma * sigma
        phase = r * mu + b
        damping = np.exp(-0.5 * r * r * variance)
        mean_sin = damping * np.sin(phase)
        mean_cos = damping * np.cos(phase)

        evidence = 0.5 * (1.0 + sign * fidelity * mean_sin)
        evidence = max(float(evidence), 1e-15)

        first_sin = mu * mean_sin + variance * r * mean_cos
        second_sin = (
            (mu * mu + variance - variance * variance * r * r) * mean_sin
            + 2.0 * mu * variance * r * mean_cos
        )

        first_moment = 0.5 * (mu + sign * fidelity * first_sin) / evidence
        second_moment = (
            0.5
            * (mu * mu + variance + sign * fidelity * second_sin)
            / evidence
        )
        new_variance = max(second_moment - first_moment * first_moment, 0.0)
        new_sigma = max(float(np.sqrt(new_variance)), self._min_theta_std)
        return self._clip_theta(float(first_moment)), new_sigma

    def _theta_interval(self, mu: float, sigma: float) -> tuple[float, float]:
        """Return a clipped equal-tail Gaussian theta interval."""
        z_value = float(stats.norm.ppf(1.0 - self._alpha / 2.0))
        lower = self._clip_theta(mu - z_value * sigma)
        upper = self._clip_theta(mu + z_value * sigma)
        return min(lower, upper), max(lower, upper)

    @staticmethod
    def _theta_interval_to_eta(
        theta_interval: tuple[float, float],
    ) -> tuple[float, float]:
        """Map a theta interval to the corresponding eta interval."""
        lower, upper = theta_interval
        eta_low = float(np.cos(upper))
        eta_high = float(np.cos(lower))
        return min(eta_low, eta_high), max(eta_low, eta_high)

    @staticmethod
    def _eta_interval_to_amplitude(
        eta_interval: tuple[float, float],
    ) -> tuple[float, float]:
        """Map an eta interval to the corresponding amplitude interval."""
        lower, upper = eta_interval
        return (lower + 1.0) / 2.0, (upper + 1.0) / 2.0

    @staticmethod
    def _theta_gaussian_to_eta_mean(mu: float, sigma: float) -> float:
        """Return E[cos(theta)] for theta distributed as N(mu, sigma^2)."""
        return float(np.exp(-0.5 * sigma * sigma) * np.cos(mu))

    def construct_circuit(
        self,
        estimation_problem: EstimationProblem,
        phase_controls: Sequence[float],
        measurement: bool = True,
    ) -> QuantumCircuit:
        """Construct a Qiskit circuit for one Alcazar ELF query.

        The implementation supports projectors induced by
        ``EstimationProblem.is_good_state`` on the objective qubits. The phase
        convention is ``U(x)=exp(i x Pi)`` and
        ``V(y)=A exp(i y |0><0|) A^dagger``.
        """
        phases = np.asarray(phase_controls, dtype=float)
        if len(phases) % 2 != 0 or len(phases) == 0:
            raise ValueError("phase_controls must have length 2L with L >= 1.")

        num_qubits = estimation_problem.state_preparation.num_qubits
        circuit = QuantumCircuit(num_qubits, name="elf_qae")
        circuit.compose(estimation_problem.state_preparation, inplace=True)

        for u_angle, v_angle in phases.reshape((-1, 2)):
            self._append_projector_phase(circuit, estimation_problem, float(u_angle))
            circuit.compose(estimation_problem.state_preparation.inverse(), inplace=True)
            self._append_zero_state_phase(circuit, float(v_angle))
            circuit.compose(estimation_problem.state_preparation, inplace=True)

        circuit.metadata = {
            "algorithm": "elf_qae",
            "phase_controls": [float(x) for x in phases],
            "elf_layers": int(len(phases) // 2),
            "state_prep_equivalent_calls": int(2 * (len(phases) // 2) + 1),
        }

        if measurement:
            creg = ClassicalRegister(len(estimation_problem.objective_qubits), "c0")
            circuit.add_register(creg)
            circuit.barrier()
            circuit.measure(estimation_problem.objective_qubits, creg[:])
        return circuit

    @staticmethod
    def _all_bitstrings(num_bits: int) -> list[str]:
        """Return bitstrings in lexicographic binary order."""
        return [format(index, f"0{num_bits}b") for index in range(2**num_bits)]

    def _append_projector_phase(
        self,
        circuit: QuantumCircuit,
        estimation_problem: EstimationProblem,
        angle: float,
    ) -> None:
        """Append exp(i angle Pi) for the problem's good-state projector."""
        objective_qubits = list(estimation_problem.objective_qubits)
        if not objective_qubits:
            raise ValueError("estimation_problem.objective_qubits cannot be empty.")

        for bitstring in self._all_bitstrings(len(objective_qubits)):
            if estimation_problem.is_good_state(bitstring):
                self._append_basis_state_phase(circuit, objective_qubits, bitstring, angle)

    def _append_zero_state_phase(self, circuit: QuantumCircuit, angle: float) -> None:
        """Append exp(i angle |0...0><0...0|)."""
        qubits = list(range(circuit.num_qubits))
        self._append_basis_state_phase(circuit, qubits, "0" * len(qubits), angle)

    @staticmethod
    def _append_basis_state_phase(
        circuit: QuantumCircuit,
        qubits: Sequence[int],
        bitstring: str,
        angle: float,
    ) -> None:
        """Append a phase to one computational basis pattern."""
        from qiskit.circuit.library import MCPhaseGate

        if len(qubits) != len(bitstring):
            raise ValueError("qubits and bitstring must have the same length.")

        for qubit, bit in zip(qubits, bitstring):
            if bit == "0":
                circuit.x(qubit)

        if len(qubits) == 1:
            circuit.p(angle, qubits[0])
        else:
            gate = MCPhaseGate(angle, num_ctrl_qubits=len(qubits) - 1)
            circuit.append(gate, list(qubits))

        for qubit, bit in reversed(list(zip(qubits, bitstring))):
            if bit == "0":
                circuit.x(qubit)

    def _good_counts_to_outcome(
        self,
        estimation_problem: EstimationProblem,
        counts_dict: dict[str, int],
    ) -> int:
        """Convert a one-shot count dictionary to Alcazar outcome d."""
        total = sum(counts_dict.values())
        if total != 1:
            raise ValueError("Alcazar ELF rounds require exactly one shot.")

        positive_outcomes = [state for state, count in counts_dict.items() if count > 0]
        if len(positive_outcomes) != 1:
            raise ValueError("Could not identify the unique ELF one-shot outcome.")

        bitstring = positive_outcomes[0]
        return 0 if estimation_problem.is_good_state(bitstring) else 1

    def estimate(
        self,
        estimation_problem: EstimationProblem,
        show_details: bool = False,
        n_shots: int = 1,
    ) -> "ELFQAEResult":
        """Run the one-outcome-per-round Alcazar ELF QAE algorithm."""
        if n_shots != 1:
            raise ValueError(
                "The original Alcazar ELF loop receives one Bernoulli outcome "
                "per adaptive round; call estimate with n_shots=1."
            )

        if self._sampler is None:
            warnings.warn(
                "No sampler provided, defaulting to SamplerV2 from qiskit_aer.",
                stacklevel=2,
            )
            from qiskit_aer import AerSimulator
            from qiskit_ibm_runtime import SamplerV2

            self._sampler = SamplerV2(mode=AerSimulator())

        mu, sigma = self._initial_theta_distribution()

        theta_intervals: list[list[float]] = []
        eta_intervals: list[list[float]] = []
        amplitude_intervals: list[list[float]] = []
        phase_controls_trace: list[list[float]] = []
        elf_layers: list[int] = []
        fidelities: list[float] = []
        posterior_params: list[tuple[float, float]] = []
        circuit_depths: list[int] = []
        elapsed_times: list[float] = []
        outcomes: list[int] = []

        num_state_prep_calls = 0
        num_circuit_evaluations = 0
        termination_reason = "max_rounds"
        terminated_early = True

        for round_index in range(self._max_rounds):
            theta_interval = self._theta_interval(mu, sigma)
            eta_interval = self._theta_interval_to_eta(theta_interval)
            amplitude_interval = self._eta_interval_to_amplitude(eta_interval)

            theta_intervals.append([theta_interval[0], theta_interval[1]])
            eta_intervals.append([eta_interval[0], eta_interval[1]])
            amplitude_intervals.append([amplitude_interval[0], amplitude_interval[1]])
            posterior_params.append((mu, sigma))

            if amplitude_interval[1] - amplitude_interval[0] <= 2.0 * self._epsilon:
                termination_reason = "epsilon_target"
                terminated_early = False
                break
            if (
                self._max_state_prep_calls is not None
                and num_state_prep_calls >= self._max_state_prep_calls
            ):
                termination_reason = "max_state_prep_calls"
                break

            layers, phases, fidelity = self._select_layer_count(mu, sigma)
            circuit = self.construct_circuit(estimation_problem, phases, measurement=True)
            circuit_depths.append(circuit.depth())

            start_time = time.time()
            try:
                job = self._sampler.run([circuit], shots=1)
                ret = job.result()
            except Exception as exc:
                raise AlgorithmError("The ELF sampler job failed.") from exc
            elapsed_times.append(time.time() - start_time)

            counts = ret[0].data.c0.get_counts()
            outcome = self._good_counts_to_outcome(estimation_problem, counts)
            r_value, b_value = self._fit_local_sinusoid(mu, sigma, phases)
            mu, sigma = self._gaussian_update(
                mu,
                sigma,
                outcome,
                fidelity,
                r_value,
                b_value,
            )

            outcomes.append(outcome)
            phase_controls_trace.append([float(x) for x in phases])
            elf_layers.append(layers)
            fidelities.append(fidelity)
            num_circuit_evaluations += 1
            num_state_prep_calls += 2 * layers + 1

            if show_details:
                print(
                    "ELF round "
                    f"{round_index + 1}: d={outcome}, L={layers}, "
                    f"f={fidelity:.6g}, mu={mu:.8f}, sigma={sigma:.8f}"
                )

        final_theta_interval = self._theta_interval(mu, sigma)
        final_eta_interval = self._theta_interval_to_eta(final_theta_interval)
        final_amplitude_interval = self._eta_interval_to_amplitude(final_eta_interval)
        eta_estimation = self._clip_eta(self._theta_gaussian_to_eta_mean(mu, sigma))
        amplitude_estimation = float(self.eta_to_amplitude(eta_estimation))

        result = ELFQAEResult()
        result.alpha = self._alpha
        result.epsilon_target = self._epsilon
        result.post_processing = cast(
            Callable[[float], float],
            estimation_problem.post_processing,
        )
        result.theta_mean = mu
        result.theta_std = sigma
        result.eta_estimation = eta_estimation
        result.estimation = amplitude_estimation
        result.estimation_processed = estimation_problem.post_processing(
            amplitude_estimation
        )
        result.theta_intervals = theta_intervals + [
            [final_theta_interval[0], final_theta_interval[1]]
        ]
        result.eta_intervals = eta_intervals + [
            [final_eta_interval[0], final_eta_interval[1]]
        ]
        result.estimate_intervals = amplitude_intervals + [
            [final_amplitude_interval[0], final_amplitude_interval[1]]
        ]
        result.eta_confidence_interval = final_eta_interval
        result.confidence_interval = final_amplitude_interval
        result.confidence_interval_processed = tuple(
            estimation_problem.post_processing(x) for x in final_amplitude_interval
        )
        result.epsilon_estimated = (
            final_amplitude_interval[1] - final_amplitude_interval[0]
        ) / 2.0
        result.epsilon_estimated_processed = (
            result.confidence_interval_processed[1]
            - result.confidence_interval_processed[0]
        ) / 2.0
        result.phase_controls = phase_controls_trace
        result.elf_layers = elf_layers
        result.fidelities = fidelities
        result.posterior_params = posterior_params
        result.outcomes = outcomes
        result.num_state_prep_calls = num_state_prep_calls
        result.num_oracle_queries = num_state_prep_calls
        result.num_circuit_evaluations = num_circuit_evaluations
        result.circuit_depths = circuit_depths
        result.elapsed_times = elapsed_times
        result.terminated_early = terminated_early
        result.termination_reason = termination_reason
        return result

    def run_internal_self_checks(self, atol: float = 1e-8) -> None:
        """Run lightweight mathematical checks for the logical ELF core."""
        for layers in range(1, min(self._max_layers, 3) + 1):
            phases = np.full(2 * layers, np.pi)
            for theta in np.linspace(0.1, np.pi - 0.1, 9):
                observed = self._bias(theta, phases)
                expected = np.cos((2 * layers + 1) * theta)
                if abs(observed - expected) > atol:
                    raise AssertionError(
                        "Chebyshev ELF self-check failed for "
                        f"L={layers}, theta={theta}: {observed} != {expected}"
                    )

        eta = np.array([-1.0, 0.0, 1.0])
        amplitude = self.eta_to_amplitude(eta)
        if not np.allclose(self.amplitude_to_eta(amplitude), eta):
            raise AssertionError("eta/amplitude mapping self-check failed.")

        mu = np.pi / 2.0
        sigma = 0.1
        phases = np.full(2, np.pi)
        r_value, b_value = self._fit_local_sinusoid(mu, sigma, phases)
        _, new_sigma = self._gaussian_update(mu, sigma, 0, 0.95, r_value, b_value)
        if not new_sigma < sigma:
            raise AssertionError("Gaussian update did not reduce theta std.")


class ELFQAEResult(AmplitudeEstimatorResult):
    """Result container for the Alcazar ELF QAE estimator."""

    def __init__(self) -> None:
        """Initialize an empty ELF QAE result."""
        super().__init__()
        self._alpha: float | None = None
        self._epsilon_target: float | None = None
        self._epsilon_estimated: float | None = None
        self._epsilon_estimated_processed: float | None = None
        self._theta_mean: float | None = None
        self._theta_std: float | None = None
        self._eta_estimation: float | None = None
        self._eta_confidence_interval: tuple[float, float] | None = None
        self._confidence_interval_processed: tuple[float, float] | None = None
        self._estimate_intervals: list[list[float]] | None = None
        self._eta_intervals: list[list[float]] | None = None
        self._theta_intervals: list[list[float]] | None = None
        self._phase_controls: list[list[float]] | None = None
        self._elf_layers: list[int] | None = None
        self._fidelities: list[float] | None = None
        self._posterior_params: list[tuple[float, float]] | None = None
        self._outcomes: list[int] | None = None
        self._num_state_prep_calls: int | None = None
        self._num_circuit_evaluations: int | None = None
        self._circuit_depths: list[int] | None = None
        self._elapsed_times: list[float] | None = None
        self._terminated_early: bool | None = None
        self._termination_reason: str | None = None

    @property
    def alpha(self) -> float:
        """Return the nominal tail probability."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = value

    @property
    def epsilon_target(self) -> float:
        """Return the requested amplitude half-width."""
        return self._epsilon_target

    @epsilon_target.setter
    def epsilon_target(self, value: float) -> None:
        self._epsilon_target = value

    @property
    def epsilon_estimated(self) -> float:
        """Return the achieved amplitude half-width."""
        return self._epsilon_estimated

    @epsilon_estimated.setter
    def epsilon_estimated(self, value: float) -> None:
        self._epsilon_estimated = value

    @property
    def epsilon_estimated_processed(self) -> float:
        """Return the processed achieved half-width."""
        return self._epsilon_estimated_processed

    @epsilon_estimated_processed.setter
    def epsilon_estimated_processed(self, value: float) -> None:
        self._epsilon_estimated_processed = value

    @property
    def theta_mean(self) -> float:
        """Return the final Gaussian theta mean."""
        return self._theta_mean

    @theta_mean.setter
    def theta_mean(self, value: float) -> None:
        self._theta_mean = value

    @property
    def theta_std(self) -> float:
        """Return the final Gaussian theta standard deviation."""
        return self._theta_std

    @theta_std.setter
    def theta_std(self, value: float) -> None:
        self._theta_std = value

    @property
    def eta_estimation(self) -> float:
        """Return the final estimate of eta = <A|O|A>."""
        return self._eta_estimation

    @eta_estimation.setter
    def eta_estimation(self, value: float) -> None:
        self._eta_estimation = value

    @property
    def eta_confidence_interval(self) -> tuple[float, float]:
        """Return the final eta interval."""
        return self._eta_confidence_interval

    @eta_confidence_interval.setter
    def eta_confidence_interval(self, value: tuple[float, float]) -> None:
        self._eta_confidence_interval = value

    @property
    def confidence_interval_processed(self) -> tuple[float, float]:
        """Return the processed final amplitude interval."""
        return self._confidence_interval_processed

    @confidence_interval_processed.setter
    def confidence_interval_processed(self, value: tuple[float, float]) -> None:
        self._confidence_interval_processed = value

    @property
    def estimate_intervals(self) -> list[list[float]]:
        """Return the amplitude interval trace."""
        return self._estimate_intervals

    @estimate_intervals.setter
    def estimate_intervals(self, value: list[list[float]]) -> None:
        self._estimate_intervals = value

    @property
    def eta_intervals(self) -> list[list[float]]:
        """Return the eta interval trace."""
        return self._eta_intervals

    @eta_intervals.setter
    def eta_intervals(self, value: list[list[float]]) -> None:
        self._eta_intervals = value

    @property
    def theta_intervals(self) -> list[list[float]]:
        """Return the theta interval trace."""
        return self._theta_intervals

    @theta_intervals.setter
    def theta_intervals(self, value: list[list[float]]) -> None:
        self._theta_intervals = value

    @property
    def phase_controls(self) -> list[list[float]]:
        """Return optimized ELF phase controls per executed round."""
        return self._phase_controls

    @phase_controls.setter
    def phase_controls(self, value: list[list[float]]) -> None:
        self._phase_controls = value

    @property
    def elf_layers(self) -> list[int]:
        """Return ELF layer counts per executed round."""
        return self._elf_layers

    @elf_layers.setter
    def elf_layers(self, value: list[int]) -> None:
        self._elf_layers = value

    @property
    def fidelities(self) -> list[float]:
        """Return likelihood fidelity factors per executed round."""
        return self._fidelities

    @fidelities.setter
    def fidelities(self, value: list[float]) -> None:
        self._fidelities = value

    @property
    def posterior_params(self) -> list[tuple[float, float]]:
        """Return Gaussian posterior parameters per round."""
        return self._posterior_params

    @posterior_params.setter
    def posterior_params(self, value: list[tuple[float, float]]) -> None:
        self._posterior_params = value

    @property
    def outcomes(self) -> list[int]:
        """Return observed Alcazar outcomes per executed round."""
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: list[int]) -> None:
        self._outcomes = value

    @property
    def num_state_prep_calls(self) -> int:
        """Return total effective state-preparation calls."""
        return self._num_state_prep_calls

    @num_state_prep_calls.setter
    def num_state_prep_calls(self, value: int) -> None:
        self._num_state_prep_calls = value

    @property
    def num_circuit_evaluations(self) -> int:
        """Return total measured circuit executions."""
        return self._num_circuit_evaluations

    @num_circuit_evaluations.setter
    def num_circuit_evaluations(self, value: int) -> None:
        self._num_circuit_evaluations = value

    @property
    def circuit_depths(self) -> list[int]:
        """Return circuit depths per executed round."""
        return self._circuit_depths

    @circuit_depths.setter
    def circuit_depths(self, value: list[int]) -> None:
        self._circuit_depths = value

    @property
    def elapsed_times(self) -> list[float]:
        """Return sampler elapsed times per executed round."""
        return self._elapsed_times

    @elapsed_times.setter
    def elapsed_times(self, value: list[float]) -> None:
        self._elapsed_times = value

    @property
    def terminated_early(self) -> bool | None:
        """Return whether the run stopped before reaching epsilon."""
        return self._terminated_early

    @terminated_early.setter
    def terminated_early(self, value: bool | None) -> None:
        self._terminated_early = value

    @property
    def termination_reason(self) -> str | None:
        """Return the termination reason."""
        return self._termination_reason

    @termination_reason.setter
    def termination_reason(self, value: str | None) -> None:
        self._termination_reason = value
