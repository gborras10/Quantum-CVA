README — Alcázar-style ELF QAE implementation
=============================================

File reviewed
-------------

    elf_qae.py

Reference
---------

    Alcázar et al. (2022), "Quantum algorithm for credit valuation adjustments",
    Appendix E: Engineered Likelihood Functions (ELF) for Quantum Amplitude
    Estimation.

Purpose
-------

This README documents the relationship between the implementation in
``elf_qae.py`` and the ELF QAE framework described in Appendix E of Alcázar
et al. The implementation reproduces the main mathematical structure of the
paper, but it also makes several concrete numerical choices that are not fully
specified in the article.

For this reason, the implementation should be described as an
``Alcázar-style ELF QAE implementation following Appendix E`` rather than as an
exact reproduction of the original numerical implementation used by the authors.


1. Overall assessment
---------------------

The implementation correctly follows the central ELF QAE construction used by
Alcázar et al. for quantum CVA. In particular, it uses the expectation-value
formulation

    eta = cos(theta) = <A|O|A>,

with

    O = 2 Pi - I,

and interprets the CVA-relevant amplitude as

    <Pi> = (1 + eta) / 2.

It also implements the parametrized ELF circuit family

    Q(x)|A> = V(x_2L) U(x_2L-1) ... V(x_2) U(x_1)|A>,

where

    U(x) = exp(i x Pi),
    V(y) = A exp(i y |0><0|) A^dagger.

The measurement model is a two-outcome Bernoulli likelihood of the form

    P(d | x, theta) = [1 + (-1)^d <A|Q(x)^dagger O Q(x)|A>] / 2,

with an optional fidelity contraction

    P(d | f, x, theta) = [1 + (-1)^d f <A|Q(x)^dagger O Q(x)|A>] / 2.

The implementation maintains a Gaussian belief over theta and performs an
approximate Bayesian update after each one-shot ELF measurement.

The main discrepancy is not in the high-level mathematical structure, but in
the implementation details. Appendix E describes the ELF framework and refers
to previous ELF references for some internal numerical choices. The code fixes
specific choices for phase optimization, Gaussian projection, local sinusoidal
fitting, fidelity modelling, layer selection, and stopping criteria. These
choices are reasonable, but they are not uniquely determined by Alcázar et al.


2. Components aligned with Alcázar et al.
-----------------------------------------

2.1. Expectation-value parametrization

The paper formulates the amplitude estimation task in terms of

    eta = <A|O|A> = cos(theta),

where

    O = 2 Pi - I.

The implementation follows this parametrization and provides the affine maps

    eta_to_amplitude(eta) = (eta + 1) / 2,
    amplitude_to_eta(a)  = 2a - 1.

This is mathematically consistent because

    <O> = <2Pi - I> = 2<Pi> - 1.

Thus, estimating eta is equivalent to estimating the probability <Pi>, but in a
symmetric [-1,1] representation.


2.2. Logical observable

In the two-dimensional logical subspace spanned by |A> and its orthogonal
component, the implementation uses

    O(theta) = cos(theta) Z + sin(theta) X.

Since |A> is represented as the logical |0> state, this gives

    <A|O(theta)|A> = cos(theta) = eta.

This is consistent with the angular representation used by ELF QAE.


2.3. Projector representation

The implementation defines

    Pi(theta) = (I + O(theta)) / 2,

which is exactly equivalent to

    O = 2Pi - I.


2.4. ELF circuit family

The paper defines

    U(x) = exp(i x Pi),
    V(y) = A exp(i y |0><0|) A^dagger,

and combines them as

    Q(x) = V(x_2L) U(x_2L-1) ... V(x_2) U(x_1).

The implementation reproduces this both in the logical two-dimensional model
and in the Qiskit circuit construction:

    _logical_u(theta, x)
    _logical_v(y)
    _q_matrix(theta, phase_controls)
    construct_circuit(...)

For each layer, the Qiskit circuit applies

    1. exp(i u_angle Pi),
    2. A^dagger,
    3. exp(i v_angle |0...0><0...0|),
    4. A.

This realizes the operator product

    V(v_angle) U(u_angle)

on the state prepared by A.


2.5. Bernoulli likelihood

The implementation uses

    P(d | theta; f, x) = 0.5 * (1 + sign * f * bias),

where

    bias = <A|Q(x)^dagger O Q(x)|A>,
    sign = +1 for d = 0,
    sign = -1 for d = 1.

This matches the ELF likelihood convention when d = 0 is associated with the
projector Pi and d = 1 with its complement.


2.6. Fidelity-contracted bias

The paper introduces a scalar fidelity parameter f that contracts the ideal
bias. The implementation supports this directly via

    circuit_fidelity = f.

When this parameter is used, the likelihood matches the fidelity-contracted
ELF likelihood described in Appendix E.


3. Implementation choices beyond the paper
------------------------------------------

3.1. Phase optimization criterion

Appendix E states that the ELF phases are chosen to maximize the information
obtained from the next measurement. The implementation makes this concrete by
maximizing the Fisher information at the current posterior mean:

    x_t = argmax_x J(mu_t; f, x).

This is a local Fisher-information heuristic. It is compatible with the ELF
idea, but it is not necessarily identical to maximizing a fully Bayesian
expected utility such as

    x_t = argmax_x E_{theta ~ posterior_t}[J(theta; f, x)].

Therefore, the phase-selection rule is an implementation-specific realization
of the ELF design principle.


3.2. Layer selection

The paper defines ELF circuits with L layers, but it does not specify an
adaptive rule for selecting L. The implementation supports two modes:

    layer_selection = "fixed"
    layer_selection = "fisher_per_cost"

The second mode selects the layer count by maximizing Fisher information per
estimated cost. This is an extension of the high-level framework, not a literal
step specified by Alcázar et al.

A closer Appendix-E-style configuration uses

    layer_selection = "fixed"
    layers = L
    max_layers = L

with L chosen externally.


3.3. Fidelity model

Alcázar et al. write the noisy likelihood using a single effective circuit
fidelity f. The implementation supports this directly through

    circuit_fidelity = f.

It also supports a decomposed model

    f = spam_fidelity * layer_fidelity**layers.

This decomposed fidelity model is an additional modelling choice. It is useful
for experimentation, but it is not the scalar-fidelity model written explicitly
in Appendix E.


3.4. Approximate Bayesian update

The implementation keeps the posterior over theta Gaussian throughout the run.
After each one-shot measurement, it approximates the local ELF bias by

    Delta(theta; x) ~= sin(r theta + b),

computes posterior moments analytically, and projects the resulting posterior
back to a Gaussian.

This follows the spirit of the ELF method, where the likelihood is treated as
locally sinusoidal in the relevant region, but it is a concrete numerical
choice. The paper does not provide a line-by-line prescription for this update.


3.5. Local sinusoidal fit

The implementation estimates the local sinusoidal approximation through

    z_values = unwrap(arcsin(biases)),

followed by a linear least-squares fit

    z ~= r theta + b.

This can work well when the posterior is sufficiently concentrated and the
local interval does not cross a turning point of the sinusoid. If the local
window crosses a maximum or minimum, the arcsin transformation can fold the
phase and produce an unstable fit.

More robust alternatives include:

    - reducing the local fitting window;
    - monitoring the local fit residual;
    - fitting sin(r theta + b) directly by nonlinear least squares;
    - fitting a local model A sin(r theta) + B cos(r theta).


3.6. Initial transformation from eta to theta

The implementation maps the initial eta distribution to theta using a
first-order delta-method approximation:

    mu_theta = arccos(mu_eta),
    sigma_theta ~= sigma_eta / sqrt(1 - mu_eta^2).

This is accurate when the initial uncertainty in eta is small and eta is not
close to +/-1. It is not the exact pushforward of a distribution under
arccos(.).

A more faithful moment-based transformation would sample eta, truncate it to
[-1,1], transform each sample using theta = arccos(eta), and fit a Gaussian to
the resulting theta samples.


3.7. Physical support of theta

The implementation clips theta to the physical interval

    theta in [0, pi].

This is physically correct because eta = cos(theta). However, the posterior is
still treated analytically as a non-truncated Gaussian. Near theta = 0 or
theta = pi, this may introduce bias. A more rigorous implementation would use a
truncated Gaussian or another posterior representation with explicit support on
[0, pi].


3.8. Qiskit bit-order convention

For the CVA projector |111><111|, the Qiskit count-string ordering does not
create ambiguity because all objective bits are equal. For general good states,
however, the bitstring returned by Qiskit may not match the logical order of
``estimation_problem.objective_qubits``.

The current implementation should therefore be used carefully for good states
other than symmetric strings such as "111". A robust general version should
normalize the measured bitstring to the logical objective-qubit order before
calling ``is_good_state``.


3.9. Stopping rule

Appendix E describes iteration until convergence, but does not define a unique
operational stopping criterion. The implementation stops when the final
amplitude interval has width at most

    2 * epsilon_target.

This is a reasonable practical criterion, but it is an implementation choice.


4. Strict Appendix-E-style operating mode
----------------------------------------

The configuration closest to the Alcázar Appendix E description is:

    ELFQAE(
        epsilon_target=epsilon,
        alpha=alpha,
        sampler=sampler,
        layers=L,
        max_layers=L,
        layer_selection="fixed",
        circuit_fidelity=f,
        ...
    )

This mode fixes the number of ELF layers and uses a single scalar fidelity
parameter. It avoids the additional adaptive layer-selection rule and the
decomposed SPAM/layer fidelity model.


5. Suggested wording for the module docstring
---------------------------------------------

A precise description of the module is:

    "This module implements an Alcázar-style engineered likelihood function
    (ELF) amplitude estimator following the high-level description in Appendix E
    of Alcázar et al. for quantum CVA. Several implementation details, such as
    phase optimization, Gaussian projection, local sinusoidal fitting, fidelity
    modelling, and stopping criteria, are concrete choices made in this
    implementation."

This wording avoids implying that the code is a line-by-line reproduction of
the authors' original implementation.


6. Suggested helper constructor
-------------------------------

A convenience constructor can make the strict Appendix-E-style configuration
explicit:

    @classmethod
    def alcazar_style(
        cls,
        epsilon_target,
        alpha,
        sampler,
        layers,
        fidelity,
        **kwargs,
    ):
        return cls(
            epsilon_target=epsilon_target,
            alpha=alpha,
            sampler=sampler,
            layers=layers,
            max_layers=layers,
            layer_selection="fixed",
            circuit_fidelity=fidelity,
            **kwargs,
        )

This prevents accidental use of implementation extensions when the objective is
to run the closest version to the Appendix E description.


7. Suggested code comment
-------------------------

    # Alcázar-style ELF QAE implementation. The code reproduces the high-level
    # Appendix E construction: eta = cos(theta), O = 2Pi - I, alternating U/V
    # ELF circuits, Bernoulli engineered likelihoods, and a fidelity-contracted
    # bias model. Details such as local Fisher optimization, sinusoidal fitting,
    # Gaussian posterior projection, fidelity modelling, and the stopping rule
    # are implementation choices rather than fully specified steps in Alcázar
    # et al.


8. Summary
----------

The implementation is mathematically aligned with the ELF QAE framework used by
Alcázar et al. for quantum CVA. It correctly implements the expectation-value
encoding, the alternating U/V circuit structure, the engineered Bernoulli
likelihood, and the scalar fidelity contraction.

The implementation is not an exact reproduction of the authors' original code,
because the paper does not fully specify every internal numerical detail. The
main additional choices are:

    - local Fisher-information phase optimization at the posterior mean;
    - optional Fisher-information-per-cost layer selection;
    - optional decomposed fidelity model;
    - Gaussian posterior representation;
    - local sinusoidal fitting of the ELF bias;
    - Gaussian moment projection after each update;
    - practical stopping rule based on the amplitude interval;
    - Qiskit-specific circuit and measurement handling.

The most accurate description is therefore:

    "Alcázar-style ELF QAE implementation following Appendix E."

not:

    "Exact reproduction of Alcázar et al.'s ELF QAE implementation."