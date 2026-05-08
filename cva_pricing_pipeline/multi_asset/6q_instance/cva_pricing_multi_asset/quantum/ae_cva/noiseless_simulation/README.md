# Noiseless 6q CVA AE Pipeline

This folder runs the real 6q CVA `EstimationProblem` through the reusable
amplitude-estimation experiment utilities in `src/quantum_cva/amplitude_estimation`.
It does not modify the toy experiments.

## Run

From the repo root:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noseless_simulation\run_noseless_cva_ae.py
```

Useful quick smoke run:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noseless_simulation\run_noseless_cva_ae.py --repetitions 1 --algorithms cabiqae_latentt,biqae --budgets 64,128,256 --max-queries 512
```

By default the runner uses the fast ideal amplification formula instead of
simulating each `A Q^k` statevector:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noseless_simulation\run_noseless_cva_ae.py --sampler false
```

With `--sampler false`, the runner also uses metadata-only circuits by default:
the algorithms receive a minimal circuit carrying `grover_power = k`, and the
sampler evaluates the noiseless law
`sin^2((2k + 1) asin(sqrt(a_true)))`. This avoids building `A Q^k` in each
iteration.

To keep the fast formula sampler but still build real `A Q^k` circuits:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noseless_simulation\run_noseless_cva_ae.py --sampler false --fast-circuits false
```

To force the slower circuit/statevector sampler for validation:

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noseless_simulation\run_noseless_cva_ae.py --sampler true
```

## Outputs

Default outputs are written to `experiment_results` in this folder:

- `direct_trace_rows.csv`
- `direct_final_rows.csv`
- `replay_budget_rows.csv`
- `budget_summary.csv`
- `errors.csv`
- `trace_bundle.npz`
- `plots/*.png` and `plots/*.pdf`

The CSVs keep amplitude-estimation columns and add CVA aliases:
`cva_true`, `cva_estimate`, `cva_abs_error`, and `cva_relative_error`.

For BIQAE, IQAE, and CABIQAE variants, `construct_circuit()` uses a shared
run-level cache keyed by problem identity, `grover_power`, measurement mode, and
construction mode. In metadata-only mode this cache stores minimal query stubs,
not full CVA circuits. The `runtime_wall_seconds` column is the
amplitude-estimation runtime after subtracting `construct_circuit()` time. Cache
and construction timings are saved separately in:

- `construct_circuit_wall_seconds`
- `construct_circuit_cache_hits`
- `construct_circuit_cache_misses`
- `construct_circuit_cache_size`

## Plot Existing Outputs

```powershell
.venv311\Scripts\python.exe cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\ae_cva\noseless_simulation\plot_noseless_cva_ae.py
```

The plots include median amplitude error versus query budget and the same
views for processed CVA relative error.
