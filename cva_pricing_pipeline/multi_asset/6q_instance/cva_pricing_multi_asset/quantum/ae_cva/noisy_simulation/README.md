# Noisy 6q CVA AE Pipeline

This folder runs simulated-noise amplitude-estimation experiments on the real
6q CVA `EstimationProblem`. It reuses the shared experiment code in
`src/quantum_cva/amplitude_estimation/experiments` for:

- building the 6q CVA problem bundle;
- Aer noisy count sampling;
- projected/realistic toy-style noise models;
- fixed-layout/preflight transpilation reports;
- readout and contrast calibration;
- AE solver construction and trace extraction;
- budget aggregation and plotting.

## Run

From the repo root:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noisy_simulation\run_noisy_cva_ae.py
```

Useful smoke run:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noisy_simulation\run_noisy_cva_ae.py --repetitions 1 --algorithms cabiqae_latentt --budgets 128,256 --max-queries 512 --max-grover-power 1 --scan-shots 128 --readout-shots 128 --make-plots false
```

For the 6q CVA target, the good state is one bitstring, `111`, over three
objective qubits. To use the uniform objective-register floor `1/8` in both
contrast calibration and noise-aware CABIQAE/BAE likelihoods, add:

```powershell
--noise-floor uniform_objective
```

The default is still `--noise-floor 0.5` to preserve the legacy two-outcome toy
convention unless a different floor is explicitly requested.

To estimate the asymptotic floor from the amplification scan instead of fixing
it, use:

```powershell
--noise-floor fit
```

This fits the baseline before computing the contrast curve, stores it as
`contrast_baseline`, and passes the fitted value to the noise-aware likelihoods.
The fit requires calibration; it cannot be combined with `--calibrate false` or
`--t-eff`.

`--max-grover-power` controls preflight, QASM export, and the calibration scan.
It does not cap the adaptive AE execution. This lets a run calibrated up to, for
example, `k=4` still execute and record the estimate if BAE or BIQAE selects
`k=7`. Add `--execution-max-grover-power K` only when you explicitly want a hard
execution cap. Rows produced above the calibrated range are marked with
`grover_power_exceeds_calibration`; final rows use `k_max_exceeds_calibration`.

The calibration summary now reports `k_visible` using a contrast-based criterion,
not the older distance-from-floor criterion. A point is visible only when it has
a valid contrast estimate, is separated from the fitted/fixed floor by
`--min-ideal-offset`, and satisfies `contrast / contrast_se >=
--min-visible-contrast-z`. The older diagnostic is kept as
`k_signal_from_baseline`.

CABIQAE keeps its noise-aware Fisher scheduler active, but the runner disables
the hard `cap_kappa` depth cap by default. That means CABIQAE may search the
identifiable candidate range and select the candidate with the largest internal
information score instead of being clipped at
`floor((cap_kappa * T_eff - 1) / 2)`. Use `--cabiqae-hard-k-cap` only when you
want to restore that hard cap.

Use an IBM backend only as a transpilation target:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noisy_simulation\run_noisy_cva_ae.py --transpile-backend-name ibm_basquecountry
```

If `--transpile-backend-name` is omitted, the script uses a local
`AerSimulator` pass manager. The noisy execution is always local Aer simulation.
When an IBM backend is selected, the backend target is requested with
`use_fractional_gates=True` by default. Pass `--no-use-fractional-gates` only if
you explicitly need the non-fractional target.

## Outputs

Default outputs are written to `experiment_results` in this folder:

- `config.json`
- `backend_snapshot.json`
- `transpilation_report.csv`
- `readout_calibration.csv`
- `amplification_counts.csv`
- `amplification_points.csv`
- `calibration_summary.json`
- `runtime_jobs.csv`
- `direct_trace_rows.csv`
- `direct_final_rows.csv`
- `budget_rows.csv`
- `replay_budget_rows.csv` compatibility copy
- `budget_summary.csv`
- `actual_query_summary.csv`
- `errors.csv`
- `trace_bundle.npz`
- `plots/*.png` and `plots/*.pdf`

The CSVs keep the amplitude-estimation columns and add CVA aliases:
`cva_true`, `cva_estimate`, `cva_abs_error`, and `cva_relative_error`.
The selected contrast floor is stored as `noise_floor` in `config.json` and as
`contrast_baseline` in the calibration outputs.

## Budget And Plot Convention

The main budget comparison uses fixed budgets. For each run and each configured
budget, the runner selects the last adaptive trace point satisfying
`N_q <= Budget`. These rows are saved in `budget_rows.csv`, and
`budget_summary.csv` aggregates them by exact budget using median errors.

The script also saves `actual_query_summary.csv`, which follows the toy plotting
convention for actual adaptive query costs: observed `N_q` values are grouped
into logarithmic bins, and each plotted point uses the median query cost and
median error in that bin.

BAE is capped by `max(budgets, --max-queries)`. This runner does not use the toy
v2 phase-2 policy where BAE budgets are matched to the median query count of the
iterative algorithms.

## Plot Existing Outputs

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noisy_simulation\plot_noisy_cva_ae.py
```
