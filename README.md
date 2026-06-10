# Quantum-CVA

Quantum-CVA is a research repository for quantum Credit Valuation Adjustment
(CVA). It combines classical multi-asset CVA pricing, market-data-driven
discretisation, quantum state preparation, controlled function encoding,
amplitude estimation, noisy simulation, and selected hardware-oriented
experiments.

The central software contribution is the reusable source package in
`src/quantum_cva`. The repository is not a collection of isolated notebooks: it
contains a public end-to-end quantum CVA pipeline that builds the classical
benchmark, trains the quantum state and function encoders, composes the CVA
circuit, and runs amplitude-estimation experiments with explicit accounting of
queries, noise, calibration, and hardware constraints.

This README describes the components that are part of the public project tree:
source code, experiment scripts, selected data artifacts, references, and final
plotting utilities.

## Repository Structure

- `src/quantum_cva/`: reusable source code. This is the core of the repository.
- `cva_pricing_pipeline/`: executable CVA pipeline instances. The important
  thesis instance is `cva_pricing_pipeline/multi_asset/6q_instance`.
- `toys/`: focused experiments for studying amplitude estimation,
  discretisation, hardware effects, and modelling choices.
- `plots/`: final AE/CVA figure-generation scripts and curated figures.
- `data/`: market inputs, benchmarks, trained parameters, and selected
  reproducibility artifacts.
- `references/`: literature used during the development of the project.
- `requirements.txt`: dependency snapshot used for the experiments.

## Single Asset And Multi Asset

`cva_pricing_pipeline` is split into `single_asset` and `multi_asset`.

`single_asset` should be read as legacy code associated with the
single-underlying line of work of Alcazar et al. It is useful historically
because it records the starting point from which the project moved toward a
more general pipeline, but it is not the main contribution of this repository.

`multi_asset` contains the relevant CVA work. The key directory is:

```text
cva_pricing_pipeline/multi_asset/6q_instance
```

This is the instance developed for the thesis. It combines two time qubits and
four underlying-state qubits, giving a compact multi-asset discretisation that
is still compatible with statevector validation, noisy simulation, and
hardware-aware amplitude-estimation experiments.

Inside `6q_instance`:

- `full_cva_pipeline.py` defines the full 6q workflow configuration: market
  inputs, portfolio instruments, classical benchmark, quantum register sizes,
  backend-noise assumptions, QCBM training, CRCA training, and final CVA
  computation.
- `training_multi_asset/` contains the training scripts for the quantum
  components: QCBM state preparation, CRCA function encoders,
  statevector-vs-shots comparisons, layer studies, and ansatz/entangler
  comparisons.
- `cva_pricing_multi_asset/classical/` contains scripts for the classical
  benchmark and discretisation diagnostics.
- `cva_pricing_multi_asset/quantum/sv_cva/` contains ideal and noisy
  statevector-style CVA executions, plus diagnostics for the quantum CVA
  operator.
- `cva_pricing_multi_asset/quantum/ae_cva/` contains amplitude-estimation
  experiments on the real 6q CVA circuit: noiseless simulation, noisy
  simulation, and hardware-oriented execution/replay.
- `cva_robustness_test/` and `cva_robustness_test_ideal/` contain robustness
  studies for perturbations of the 6q instance.

The other multi-asset cases, such as `8q_instance` and `10q_instance`, are not
explained in the thesis. They are additional reproducibility and scaling tests
used to study the feasibility of different qubit counts and to support the
choice of the 6q instance.

## Source Code: `src/quantum_cva`

The package `src/quantum_cva` is the reusable implementation layer. Experiment
scripts should be understood as entrypoints that call this package, not as
parallel implementations of the method.

### Classical Multi-Asset CVA

`src/quantum_cva/multi_asset/classical` implements the classical side of the
pipeline:

- multi-asset GBM simulation with piecewise volatility and correlation;
- construction of tensor-product price grids;
- conversion of simulated paths into conditional discrete distributions;
- construction of the QCBM target distribution over time and multi-asset
  states;
- continuous and discrete CVA engines;
- explicit validation of shapes, probability normalisation, and grid
  compatibility.

`src/quantum_cva/multi_asset/instruments` contains the portfolio primitives:
forwards, calls, puts, and market-data utilities. This separation allows the
same pricing and mark-to-market conventions to be reused by both the classical
benchmark and the quantum training targets.

`src/quantum_cva/multi_asset/pipeline_cfg/cfg_utilities.py` defines the
configuration dataclasses and the `CVAPipelineRunner`. This runner orchestrates
the 6q instance: it resolves market data, runs the classical benchmark, trains
or loads quantum artifacts, and records a structured pipeline summary.

### Quantum Training Pipeline

The quantum CVA construction requires two families of trained components.

The first is state preparation. `state_prep_qcbm/qcbm_circuit.py` implements a
multi-layer Quantum Circuit Born Machine (QCBM). It supports several entangling
topologies, including heavy-hex-compatible layouts, exact statevector
evaluation, shot-based sampling, clipped cross-entropy/negative-log-likelihood
objectives, Dirichlet smoothing, and distance metrics between the trained and
target distributions. In the 6q instance, the QCBM prepares the joint
distribution over exposure date and discretised multi-asset state.

The second is function encoding. The CRCA implementation in
`functional_encoding_crca/crca` builds controlled-rotation ansatzes for the
quantities entering CVA:

- positive exposure;
- default probability increments;
- discount factors.

Each CRCA encodes a classical function on the relevant control register into an
ancilla success probability. The training scripts compare ideal and shot-based
regimes, hardware-aware topologies, native decompositions, and depth choices.

`state_prep_mps/mps.py` provides an alternative tensor-network/MPS state
preparation route. It acts as a structured reference for distribution loading
and for comparing expressivity and resource cost against the QCBM.

### Quantum CVA Circuit

`src/quantum_cva/multi_asset/quantum/amplitude_estimation/cva_circuit.py`
combines the trained QCBM and CRCA blocks into the final CVA circuit. The
register layout is:

```text
time qubits | state qubits | exposure ancilla | default ancilla | discount ancilla
```

The CVA quantity is encoded in the probability that the three objective
ancillas are simultaneously in state `|111>`. The class provides:

- construction of the parameter-bound CVA circuit;
- exact statevector evaluation of the `|111>` probability;
- shot-based estimation of the same probability when using Aer;
- deterministic post-processing from estimated probability to CVA through the
  benchmark scaling constants.

For the 6q thesis instance, the amplitude-estimation problem is built in
`src/quantum_cva/amplitude_estimation/experiments/cva.py`. The resulting
`EstimationProblem` uses objective qubits `[6, 7, 8]`, good bitstring `111`,
and the CVA post-processing function defined by `QuantumCVACircuit`.

### Amplitude Estimation And CABIQAE

`src/quantum_cva/amplitude_estimation` contains the reusable amplitude
estimation layer:

- adapters for algorithm construction and trace extraction;
- ideal, noisy, replay, Aer, Runtime, and Q-CTRL-oriented sampler paths;
- hardware calibration utilities;
- budget aggregation and plotting utilities;
- CVA-specific hardware experiment runners.

The repository includes several amplitude-estimation algorithms for comparison,
but the algorithmic contribution that should be highlighted is CABIQAE. In the
code it appears primarily as `CABIQAELatentTheta` in
`src/quantum_cva/algorithms/proposed_algorithms/cabiae.py`, and it is used in
the runners with the key `cabiqae_latentt`.

CABIQAE is a Bayesian iterative amplitude-estimation method with explicit noise
awareness. Its main design choices are:

- it maintains a posterior over a latent angle variable, rather than updating
  only in observed probability space;
- it separates the ideal amplified probability from the observed probability
  under noise;
- it supports an exponential contrast model in which deeper Grover powers lose
  contrast toward a configurable noise floor;
- it transports Bayesian information between stages in the latent variable;
- it chooses Grover powers with a scheduler that respects IQAE identifiability
  constraints and prioritises information criteria among admissible candidates;
- it can use an effective hardware contrast scale, `T_eff`, and a depth cap
  when required by the noisy regime.

This structure is important for CVA on NISQ devices: the theoretically
attractive regime of large Grover powers can stop being useful once
depth-induced contrast loss dominates. CABIQAE is therefore evaluated not only
by final error, but also by actual query cost, selected Grover depths, coverage,
runtime, and robustness under simulated or calibrated noise.

## 6q CVA AE Experiments

The most important amplitude-estimation experiments for CVA are in:

```text
cva_pricing_pipeline/multi_asset/6q_instance/cva_pricing_multi_asset/quantum/ae_cva
```

`noiseless_simulation/` runs the real 6q CVA `EstimationProblem` in an ideal
amplitude-estimation regime. It can use the closed-form ideal amplification law
instead of repeatedly reconstructing full `A Q^k` circuits, while preserving the
same query accounting and CVA post-processing.

`noisy_simulation/` runs the same 6q target under simulated noise. It includes
contrast calibration, readout correction, transpilation preflight, query-budget
summaries, and CVA-specific aliases such as `cva_true`, `cva_estimate`,
`cva_abs_error`, and `cva_relative_error`.

`hardware/` contains the hardware-oriented route for the 6q instance. The
reusable implementation lives in
`src/quantum_cva/amplitude_estimation/experiments/cva_hwd_experiments`. The
workflow is:

1. build the 6q CVA `EstimationProblem` from the pipeline configuration and the
   trained artifacts;
2. select or load the backend;
3. run ISA preflight diagnostics for reference Grover powers;
4. collect readout and amplification calibration data;
5. fit or select a contrast model;
6. run direct AE on hardware or replay algorithms from measured probabilities;
7. persist traces, budgets, calibration summaries, QASM snapshots, and plot
   inputs.

The semantic invariant is the same as in simulation: the target is the `|111>`
probability of the objective register, and all reported CVA estimates are
obtained by post-processing that amplitude.

## Toys

`toys/` contains exploratory and methodological experiments. Most toy folders
are supporting material for understanding discretisation, state preparation,
transpilation, and hardware effects. The main area is:

```text
toys/amplitude_estimation_experiments
```

### Ideal Regime

`toys/amplitude_estimation_experiments/ideal_regime` studies AE behaviour before
introducing hardware noise. The main benchmark compares BAE, BIQAE, and CABIQAE
on a canonical low-qubit amplitude-estimation problem. It sweeps a set of
objective rotation offsets, repeats the experiment many times, extracts full
algorithm traces, and saves:

- per-step trace rows;
- final-estimator rows;
- budget-aligned summaries;
- error rows;
- paper-style plots.

This directory isolates the statistical and query-complexity behaviour of the
algorithms from the engineering complications of hardware execution.

### Noise-Aware 3-Qubit Hardware Toy

The most important hardware toy is:

```text
toys/amplitude_estimation_experiments/noise_aware_regime/3qubit_toy/hardware
```

This toy is intentionally smaller than the full CVA circuit, but it exercises
the same hardware-aware AE logic. The workflow is:

1. construct a canonical 3-qubit AE problem and its measured `A Q^k` circuits;
2. run transpilation preflight on the selected IBM or fake backend and reject
   circuits exceeding configured depth or two-qubit-gate limits;
3. run readout calibration;
4. run an amplification scan over Grover powers and save raw counts;
5. mitigate measured probabilities and compare them with the ideal Grover
   oscillation;
6. fit an empirical contrast model and derive an effective noise scale;
7. replay BAE, BIQAE, and CABIQAE from the measured probability table;
8. aggregate errors by actual query cost and generate hardware-replay plots.

The methodological point is to separate expensive hardware sampling from
algorithmic post-processing. Hardware is used to obtain calibrated probabilities
by Grover depth; replay then enables many statistically independent AE
executions without resubmitting every adaptive trajectory to the device.

The remaining toys, including multi-asset demos, single-asset demos, and
quantum hardware demos, should be read as supporting notebooks and experiments.
They document how specific modelling, discretisation, training, or
transpilation decisions were understood, but they are not the main
implementation of the CVA pipeline.

## Plots

The important AE plots are in `plots/`. This directory contains the final
figure-generation layer for paper/thesis outputs. The scripts read existing CSV
summaries and apply consistent styles, labels, and aggregations; they are not
new experiment runners.

Main scripts:

- `plots/make_error_budget.py`: assembles AE and CVA error-budget curves from
  summaries of the ideal toy, hardware toy, noiseless CVA, and hardware CVA
  experiments.
- `plots/make_ae_final_error_density_grid.py`: builds final-error density grids
  for AE algorithms and the classical direct-sampling baseline.
- `plots/make_ae_cva_hardware_noiseless_final_error_grid.py`: compares final
  CVA error distributions between noiseless and hardware-oriented 6q runs.
- `plots/make_hardware_amplification_contrast_grid.py`: visualises hardware
  amplification curves and the fitted contrast-decay model.

Other useful plots are kept close to the experiment that generates them. In the
6q training tree, the layer-comparison and ansatz-comparison folders contain
figures for QCBM training quality, resource scaling, two-qubit depth, and
training trajectories. The robustness folders contain CVA summaries, scenario
histograms, and training-quality diagnostics.

## Data And References

`data/` contains the selected inputs and artifacts needed to reproduce the
visible experiments: market data, classical CVA benchmarks, multi-asset 6q
training artifacts, selected 8q/10q artifacts, and single-asset legacy data.
The most important public path is `data/multi_asset/6q_instance`, which
contains the benchmark and trained quantum artifacts used by the thesis
instance.

`references/` contains the bibliography used in the project: quantum CVA,
quantum option pricing, amplitude estimation, near-term/noise-aware AE, QCBM,
MPS, tensor networks, and controlled-rotation circuits.

## Execution Notes

Most scripts are intended to be run from the repository root. Lightweight
experiments and plot builders can be executed locally after installing the
dependencies in `requirements.txt`. Hardware modes require configured IBM
Quantum or Qiskit Runtime credentials and should not be launched as smoke tests.

For quick validation, use few repetitions, few shots, and the available
`dry-run` or `replay-only` modes. Full 6q training and hardware executions are
costly reproducibility workflows, not ordinary unit tests.
