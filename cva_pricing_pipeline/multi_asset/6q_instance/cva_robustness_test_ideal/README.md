# Ideal CVA robustness analysis

This directory contains an automated robustness pipeline for the 6-qubit
multi-asset CVA instance. It evaluates the exact logical statevector regime:
there are no finite shots, no backend-noise model, no transpilation snapshot and
no IBM Runtime dependency.

## Scientific policy

`Scenario 1` is the canonical baseline used by
`cva_pricing_multi_asset/quantum/sv_cva/run_ideal_cva.py`. It loads the existing
noiseless QCBM and CRCA `theta_star` artifacts without retraining them. This
reproduces the baseline methodology exactly:

```text
CVA_classical_small_grid(n_bits=2) = 0.5215900006813697
CVA_exact_ideal_statevector        = 0.6704615214666005
```

Every stressed scenario regenerates its classical benchmark and retrains the
positive-exposure CRCA with exact statevector probabilities. The local
benchmark generator preserves the historical right-endpoint volatility-bucket
semantics and RNG sequence used by the canonical classical script.

The default `--component-policy coherent` also retrains each ideal component
whose target changes:

- QCBM when the joint time-price distribution changes.
- Default-probability CRCA when normalized default increments change.
- Discount-factor CRCA when normalized discount factors change.
- Positive-exposure CRCA in every stressed scenario, unconditionally.

This matters for volatility and rate stresses. Retraining exposure alone would
leave some state-preparation or scalar-encoding blocks calibrated to the base
market. The optional `--component-policy exposure_only` mode is retained as an
explicit ablation study, not as the primary robustness estimate.

The default-probability stress is implemented by scaling CDS spreads before the
survival curve is bootstrapped. Interest-rate stresses are parallel shifts of
the input discount curve. This preserves market-consistent benchmark
construction instead of editing already-normalized quantum targets.

The exposure target keeps the benchmark pipeline's `time_major` flattening
order. Existing ideal artifacts anchor `Scenario 1` and are used as warm starts
for stressed scenarios.

## Execute

Create the scenario catalog and design graph without training:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\run_ideal_cva_robustness.py --profile focused --dry-run
```

Run the focused 22-scenario analysis:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\run_ideal_cva_robustness.py --profile focused --force --overwrite-results
```

Run the exhaustive 243-scenario Cartesian stress grid:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\run_ideal_cva_robustness.py --profile ultra --overwrite-results
```

The exhaustive profile retrains many exact variational circuits and is
intentionally compute-intensive. Interrupted runs can continue from their
artifacts:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\run_ideal_cva_robustness.py --profile ultra --resume
```

Verify the complete workflow cheaply before a long run:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\run_ideal_cva_robustness.py --smoke --overwrite-results --output-dir .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\smoke_validation
```

Use `example_custom_scenarios.csv` as a template for a custom scenario catalog:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\run_ideal_cva_robustness.py --scenarios-csv .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\example_custom_scenarios.csv --overwrite-results
```

## Outputs

By default outputs are written below `analysis_output`:

- `scenario_catalog.csv` and `scenario_catalog.md`
- `plots/scenario_design_matrix.{png,pdf}`
- `cases/<case_id>/benchmark/benchmark.npz`
- `cases/<case_id>/training/.../training_ideal_statevector.npz`
- `results/ideal_cva_robustness_results.csv`
- `results/ideal_cva_robustness_summary.json`
- `tables/*.csv`, `tables/*.md` and `tables/*.tex`
- `plots/ideal_cva_absolute_relative_error_by_scenario.{png,pdf}`
- `tables/ideal_cva_scenario_definitions.{csv,md,tex}`
- `tables/ideal_cva_absolute_relative_error_caption.md`

Regenerate result plots and ranked tables without rerunning training:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\plot_ideal_cva_robustness.py
```

This reads the existing `analysis_output/results/ideal_cva_robustness_results.csv`
file. It does not regenerate classical benchmarks or rerun any variational
training.

## Reading relative errors

Use `plots/ideal_cva_absolute_relative_error_by_scenario.png` as the primary
robustness figure. It reports one academic-style curve: the absolute relative
error of the exact ideal-statevector CVA estimate with respect to the
scenario-specific classical small-grid CVA with `n_bits=2`. The x-axis uses
`Scenario 1` to `Scenario 22`; the definitions are reported in
`tables/ideal_cva_scenario_definitions.md`.

For scenario \(s\), the plotted metric is:

```text
absolute_relative_error_pct(s)
    = 100 * abs(CVA_statevector(s) - CVA_classical_small_grid(s))
          / abs(CVA_classical_small_grid(s))
```

For `Scenario 1`, `CVA_classical_small_grid = 0.5215900006813697`.

Suggested figure caption:

> Absolute relative error of the exact ideal-statevector CVA estimate with
> respect to the scenario-specific classical small-grid CVA across the 22 focused
> robustness scenarios. Scenario definitions are reported in the accompanying
> scenario-definition table.

Generate the additional diagnostic plots only when they are needed:

```powershell
python .\cva_pricing_pipeline\multi_asset\6q_instance\cva_robustness_test_ideal\plot_ideal_cva_robustness.py --include-diagnostics
```

## Focused scenario definitions

| Scenario | Definition |
| --- | --- |
| Scenario 1 | Baseline market configuration. |
| Scenario 2 | Call-option strike decreased by 10%. |
| Scenario 3 | Call-option strike increased by 10%. |
| Scenario 4 | Put-option strike decreased by 10%. |
| Scenario 5 | Put-option strike increased by 10%. |
| Scenario 6 | Both option strikes decreased by 10%. |
| Scenario 7 | Both option strikes increased by 10%. |
| Scenario 8 | Volatilities decreased by 20%. |
| Scenario 9 | Volatilities decreased by 10%. |
| Scenario 10 | Volatilities increased by 10%. |
| Scenario 11 | Volatilities increased by 20%. |
| Scenario 12 | Parallel interest-rate curve shift of -100 bp. |
| Scenario 13 | Parallel interest-rate curve shift of -50 bp. |
| Scenario 14 | Parallel interest-rate curve shift of +50 bp. |
| Scenario 15 | Parallel interest-rate curve shift of +100 bp. |
| Scenario 16 | CDS spreads decreased by 25%. |
| Scenario 17 | CDS spreads decreased by 10%. |
| Scenario 18 | CDS spreads increased by 10%. |
| Scenario 19 | CDS spreads increased by 25%. |
| Scenario 20 | Combined moderate risk-on stress. |
| Scenario 21 | Combined moderate risk-off stress. |
| Scenario 22 | Combined extreme risk-off stress. |
