# Noise-vs-Ideal Training Comparison

This folder contains a reproducible post-processing script for comparing two
CVA-pipeline training regimes:

- `ideal_statevector`: train QCBM/CRCA components in ideal statevector mode.
- `shots_backend_noise`: train QCBM/CRCA components with finite shots and the
  frozen backend-noise snapshot.

The comparison is intentionally component-level and aggregate-level:

- QCBM: KL(target || p_theta)
- CRCA default probabilities: MSE(f_target, f_theta)
- CRCA discount factors: MSE(f_target, f_theta)
- CRCA positive exposure: MSE(f_target, f_theta)
- Aggregate CVA: CVA value and relative error versus the classical CVA reference

## Generate fast paper outputs

```powershell
python cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\sv_cva\noise_vs_ideal_training\make_paper_results.py
```

This mode uses stored training/evaluation artifacts and recomputes exact
statevector CVA for both parameter sets. It does not contact IBM Runtime.

Outputs are written to:

```text
cva_pricing_pipeline/multi_asset/6q_instance/cva_pricing_multi_asset/quantum/sv_cva/noise_vs_ideal_training/paper_results
```

## Add full noisy CVA evaluation

```powershell
python cva_pricing_pipeline\multi_asset\6q_instance\cva_pricing_multi_asset\quantum\sv_cva\noise_vs_ideal_training\make_paper_results.py --run-noisy-cva
```

This additionally simulates the full measured CVA circuit under a backend-noise
model built from the saved snapshot metadata. It requires IBM Runtime backend
metadata access and can be slower.

Useful options:

```powershell
--cva-shots 100000 --cva-repetitions 10 --seed-base 42
```

## Main outputs

- `tables/subblock_training_regime_metrics.{csv,md,tex}`
- `tables/cva_training_regime_metrics.{csv,md,tex}`
- `tables/noise_eval_comparison_summary.{csv,md,tex}`
- `figures/fig_subblock_noise_eval_comparison.{png,pdf,svg}`
- `figures/fig_noise_training_ratio.{png,pdf,svg}`
- `figures/fig_cva_training_regime_comparison.{png,pdf,svg}`
- `manifest.json`
