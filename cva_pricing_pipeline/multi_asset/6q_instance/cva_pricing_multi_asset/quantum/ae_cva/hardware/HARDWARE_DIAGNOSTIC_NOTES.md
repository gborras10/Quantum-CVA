# Hardware CVA AE Diagnostic Notes

These notes capture the current interpretation of the `ibm_aachen` hardware run
stored under:

`cva_pricing_pipeline/multi_asset/6q_instance/cva_pricing_multi_asset/quantum/ae_cva/hardware/results/hardware_cva_ae_ibm_aachen_runtime`

## Main Diagnosis

The hardware data show a severe loss of coherent Grover-amplification contrast.
The loss does not currently look like a clean exponential contrast decay with a
single effective time `T_eff`.

This means the present data can support measured-`k` replay, but they should not
be used to extrapolate probabilities to unmeasured Grover powers unless a future
calibration obtains an identifiable `T_eff`.

## Q-CTRL Performance Management Usage

The Q-CTRL Fire Opal Performance Management path must be treated differently
from the IBM Runtime Sampler path.

For IBM Runtime, this pipeline submits ISA circuits produced by the local
Qiskit pass manager. For Q-CTRL, the pipeline now submits the measured logical
`QuantumCircuit` directly through:

`perf_mgmt.run(primitive="sampler", pubs=[(circuit, None, shots)], backend_name=..., options={"session_id": session_id})`

This follows Q-CTRL/IBM guidance: Performance Management accepts abstract
circuits and should run its own Fire Opal logical transpilation, hardware
mapping, gate replacement, crosstalk handling, and error-suppression pipeline.
The local preflight transpilation is still useful as a diagnostic and as a
conservative Grover-power cap, but those ISA circuits are not sent to Q-CTRL.
Live Q-CTRL executions open a Qiskit Runtime `Session` and pass its identifier
to Performance Management, so related Q-CTRL jobs remain in the same scheduler
session. The amplification scan submits all scheduled `(k, repeat)` circuits as
PUBs of one Q-CTRL job; readout calibration remains a separate two-PUB job.

New Q-CTRL runs mark this explicitly in the artifacts:

- `config.json`: `qctrl_submission.submitted_circuit_kind = "abstract_logical"`
- `backend_snapshot.json`: `qctrl_submission_strategy = "abstract_logical_circuits"`
- `session_details.json`: `submitted_circuit_kind = "abstract_logical"`
- `session_details.json`: `session_id` is the shared Runtime session identifier
- `runtime_jobs.csv`: `submitted_circuit_kind`, `qctrl_transpilation_policy`,
  `local_pass_manager_applied`, `session_id`, and logical circuit
  depth/2q-count columns

The saved `hardware_cva_ae_ibm_aachen_runtime` run did not use Q-CTRL. It used
the IBM Runtime Sampler path, so its poor contrast diagnosis is about the
standard Runtime execution, not a Q-CTRL Performance Management execution.

For a minimal live Q-CTRL Sampler check without amplitude-estimation algorithms,
run:

`python cva_pricing_pipeline/multi_asset/6q_instance/cva_pricing_multi_asset/quantum/ae_cva/hardware/qctrl_toys/run_qctrl_q0_smoke.py --backend-name ibm_aachen --shots 128`

This submits only the `Q^0 A |0>` CVA state-preparation circuit, measures
objective qubits `[6, 7, 8]`, counts `good_bitstring='111'`, and saves
`summary.json`, `summary.csv`, and `qctrl_jobs.csv` under
`hardware/qctrl_toys/results/qctrl_q0_smoke_<timestamp>/`.

The matching standard IBM Runtime Sampler smoke test is:

`python cva_pricing_pipeline/multi_asset/6q_instance/cva_pricing_multi_asset/quantum/ae_cva/hardware/qctrl_toys/run_runtime_q0_smoke.py --backend-name ibm_aachen --shots 128 --show-counts`

That path transpiles the same `Q^0` circuit locally to ISA before submission,
so its results are the direct comparison against the Q-CTRL managed path.
The Runtime smoke test intentionally disables `use_fractional_gates` by default.
With fractional-gate targets enabled, Qiskit can emit `rzz` gates with angles
outside the range accepted by Runtime validation, for example negative angles,
which causes submission to fail before the job reaches the backend.

## Evidence Against a Circuit-Construction Bug

The logical/statevector circuit appears correct for the measured Grover powers.
For `k = 0..4`, the statevector probability of the good state `111` on objective
qubits `[6, 7, 8]` matches the ideal Grover law

`p_k = sin^2((2k + 1) theta)`

to numerical precision:

| k | K = 2k + 1 | statevector probability |
|---|------------|-------------------------|
| 0 | 1 | 0.132873094295 |
| 1 | 3 | 0.809666170855 |
| 2 | 5 | 0.915594194425 |
| 3 | 7 | 0.255380404157 |
| 4 | 9 | 0.046119258015 |

An Aer ideal sampling check with the same measured circuits and the same
`count_good_from_counts` path also matched the ideal probabilities within
finite-shot error.

The readout calibration is also consistent with the intended good bitstring:

| prepared objective bits | observed good probability |
|-------------------------|---------------------------|
| `000` | 0.000000 |
| `111` | 0.970947 |

This strongly argues against a simple qubit-ordering, classical-bit-ordering, or
good-state counting bug.

## Hardware vs Ideal

Aggregated over the saved hardware counts:

| k | K | ideal p_good | hardware mitigated p_good | ISA depth | ISA 2q gates |
|---|---|--------------|---------------------------|-----------|--------------|
| 0 | 1 | 0.132873 | 0.269148 | 322 | 117 |
| 1 | 3 | 0.809666 | 0.076138 | 2691 | 1122 |
| 2 | 5 | 0.915594 | 0.146191 | 5082 | 2093 |
| 3 | 7 | 0.255380 | 0.114257 | 7611 | 3101 |
| 4 | 9 | 0.046119 | 0.074227 | 9815 | 4091 |

The deviations from the ideal probabilities are far larger than shot noise
would explain. With 20,480 shots per Grover power, the differences are many
standard errors away from the ideal values.

## Why `T_eff` Was Not Identifiable

The contrast model assumes something like:

`p_hw(k) = b + C(K) * (p_ideal(k) - b)`

with

`C(K) ~= exp(-K / T_eff)`

where `b` is the noise-floor baseline and `K = 2k + 1`.

For a valid `T_eff` fit, the inferred contrast values should be positive,
bounded, and generally decrease with increasing `K`. In the saved run this does
not happen:

- Some points fall on the wrong side of the baseline, making the inferred
  contrast negative.
- The usable points do not form a monotonic exponential decay.
- In particular, the apparent contrast at deeper circuits can look larger than
  at shallower ones, which is incompatible with a simple decay model.

Therefore `T_eff` should not be manually invented for extrapolation.
