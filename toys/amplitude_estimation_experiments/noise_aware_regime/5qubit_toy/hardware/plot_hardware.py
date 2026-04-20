from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
RUN_DIR = Path(
    r"c:\Users\guilb\Desktop\Quantum-CVA\toys\amplitude_estimation_experiments\noise_aware_regime\5qubit_toy\hardware\hardware_bae_vs_cabiqae_20260420_110033"
)

TRACE_CSV = RUN_DIR / "trace_rows.csv"
FINAL_CSV = RUN_DIR / "final_rows.csv"
OUT_DIR = RUN_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------
trace_df = pd.read_csv(TRACE_CSV)
final_df = pd.read_csv(FINAL_CSV)

trace_df = trace_df[trace_df["algorithm"] == "cabiqae"].copy()
final_df = final_df[final_df["algorithm"] == "cabiqae"].copy()

if trace_df.empty:
    raise RuntimeError("No CABIQAE rows found in trace_rows.csv")

if final_df.empty:
    raise RuntimeError("No CABIQAE rows found in final_rows.csv")


# -----------------------------------------------------------------------------
# BASIC VALUES
# -----------------------------------------------------------------------------
a_true = float(trace_df["a_true"].iloc[0])
final_row = final_df.iloc[0]

queries = trace_df["query_budget"].to_numpy(dtype=float)
estimates = trace_df["estimate"].to_numpy(dtype=float)
abs_error = trace_df["abs_error"].to_numpy(dtype=float)
nrmse = trace_df["nrmse"].to_numpy(dtype=float)
k_values = trace_df["k_value"].to_numpy(dtype=int)

final_est = float(final_row["final_estimate"])
final_queries = float(final_row["final_queries"])
final_nrmse = float(final_row["final_nrmse"])
final_runtime = float(final_row["runtime_seconds"])
k_max = int(final_row["k_max"])


# -----------------------------------------------------------------------------
# PLOT 1: estimate vs queries
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(queries, estimates, marker="o")
ax.axhline(a_true, linestyle="--")
ax.set_xlabel("Queries")
ax.set_ylabel("Estimate")
ax.set_title("CABIQAE: estimate vs queries")
ax.grid(True, alpha=0.25)

text = (
    f"a_true = {a_true:.6f}\n"
    f"final_est = {final_est:.6f}\n"
    f"final_queries = {int(final_queries)}\n"
    f"final_nRMSE = {final_nrmse:.3e}\n"
    f"runtime = {final_runtime:.2f}s\n"
    f"Kmax = {k_max}"
)
ax.text(
    0.02,
    0.98,
    text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", alpha=0.15),
)

fig.tight_layout()
fig.savefig(OUT_DIR / "cabiqae_estimate_vs_queries.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------------------------
# PLOT 2: abs error vs queries
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(queries, abs_error, marker="o")
ax.set_xlabel("Queries")
ax.set_ylabel("Absolute error")
ax.set_title("CABIQAE: absolute error vs queries")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "cabiqae_abs_error_vs_queries.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------------------------
# PLOT 3: nRMSE vs queries
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(queries, nrmse, marker="o")
ax.set_xlabel("Queries")
ax.set_ylabel("Normalized RMSE")
ax.set_title("CABIQAE: nRMSE vs queries")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "cabiqae_nrmse_vs_queries.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------------------------
# PLOT 4: k (actually K=2k+1 in your trace naming) vs queries
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.step(queries, k_values, where="post", marker="o")
ax.set_xlabel("Queries")
ax.set_ylabel("Recorded k_value")
ax.set_title("CABIQAE: k_value vs queries")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "cabiqae_kvalue_vs_queries.png", dpi=200)
plt.close(fig)


# -----------------------------------------------------------------------------
# OPTIONAL: combined panel
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

axes[0, 0].plot(queries, estimates, marker="o")
axes[0, 0].axhline(a_true, linestyle="--")
axes[0, 0].set_xlabel("Queries")
axes[0, 0].set_ylabel("Estimate")
axes[0, 0].set_title("Estimate vs queries")
axes[0, 0].grid(True, alpha=0.25)

axes[0, 1].semilogy(queries, abs_error, marker="o")
axes[0, 1].set_xlabel("Queries")
axes[0, 1].set_ylabel("Absolute error")
axes[0, 1].set_title("Abs error vs queries")
axes[0, 1].grid(True, which="both", alpha=0.25)

axes[1, 0].semilogy(queries, nrmse, marker="o")
axes[1, 0].set_xlabel("Queries")
axes[1, 0].set_ylabel("nRMSE")
axes[1, 0].set_title("nRMSE vs queries")
axes[1, 0].grid(True, which="both", alpha=0.25)

axes[1, 1].step(queries, k_values, where="post", marker="o")
axes[1, 1].set_xlabel("Queries")
axes[1, 1].set_ylabel("Recorded k_value")
axes[1, 1].set_title("k_value vs queries")
axes[1, 1].grid(True, alpha=0.25)

fig.suptitle(
    f"CABIQAE hardware run | a_true={a_true:.6f} | final_est={final_est:.6f} | "
    f"Q={int(final_queries)} | nRMSE={final_nrmse:.3e} | Kmax={k_max}",
    y=0.98,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_DIR / "cabiqae_summary_panel.png", dpi=220)
plt.close(fig)

print(f"Plots saved in: {OUT_DIR}")