# src/quantum_cva/mc_benchmark/data_saving_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .scaling_constants import DiscreteCvaTables


def save_classical_tables(
    *,
    tables: DiscreteCvaTables,
    t: np.ndarray,
    p_target: np.ndarray,                 # <-- NUEVO
    filename_stem: str,
    outdir: str | Path = "data/classical_cva_tables",
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """
    Save classical CVA grids and quantum-ready target distribution.

    Conventions:
    - M time points
    - N = 2^n price points
    - p_target is flattened (time_major) and sums to 1
    """

    # -------------------------------------------------
    # Anchor to project root
    # -------------------------------------------------
    project_root = Path(__file__).resolve().parents[3]
    outdir = project_root / "data" / "classical_cva_tables"
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Core arrays (NO slicing, already M-based)
    # -------------------------------------------------
    t_M = np.asarray(t, dtype=float)                  # (M,)
    p_t = np.asarray(tables.p_t, dtype=float)         # (M,)
    q_t = np.asarray(tables.q_t, dtype=float)         # (M,)
    s_grid = np.asarray(tables.s_rep, dtype=float)    # (N,)
    P_s_t = np.asarray(tables.P_s_t, dtype=float)     # (M, N)
    v_s_t = np.asarray(tables.v_s_t, dtype=float)     # (M, N)
    p_target = np.asarray(p_target, dtype=float)      # (M*N,)

    M, N = P_s_t.shape

    # -------------------------------------------------
    # CHECKS (críticos)
    # -------------------------------------------------
    print("=== CHECK: dimensiones guardadas ===")
    print("M (time)        :", M)
    print("N (price bins)  :", N)
    print("t               :", t_M.shape)
    print("s_grid          :", s_grid.shape)
    print("P_s_t           :", P_s_t.shape)
    print("p_target        :", p_target.shape)
    print("sum(p_target)   :", p_target.sum())
    print("===================================")

    if p_target.shape != (M * N,):
        raise ValueError("p_target must have shape (M*N,)")

    if not np.isclose(p_target.sum(), 1.0, atol=1e-12):
        raise ValueError("p_target must be normalized (sum = 1)")

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    npz_path = outdir / f"{filename_stem}.npz"
    json_path = outdir / f"{filename_stem}.json"

    # -------------------------------------------------
    # NPZ: fuente de verdad numérica
    # -------------------------------------------------
    np.savez_compressed(
        npz_path,
        t=t_M,
        n=np.array(int(tables.n), dtype=int),
        s_grid=s_grid,
        p_t=p_t,
        q_t=q_t,
        P_s_t=P_s_t,
        v_s_t=v_s_t,
        p_target=p_target,              # <-- CLAVE
        C_p=np.array(float(tables.C_p)),
        C_q=np.array(float(tables.C_q)),
        C_v=np.array(float(tables.C_v)),
    )

    # -------------------------------------------------
    # JSON: legible + quantum-ready
    # -------------------------------------------------
    meta_out: dict[str, Any] = {
        "schema": "classical_cva_tables_v3_quantum_ready",
        "n": int(tables.n),
        "M": int(M),
        "N": int(N),
        "order": "time_major",
        "grids": {
            "t": t_M.tolist(),
            "s_grid": s_grid.tolist(),
            "p_t": p_t.tolist(),
            "q_t": q_t.tolist(),
        },
        "quantum": {
            "p_target": p_target.tolist(),   # <-- LISTA PLANA
        },
        "scalings": {
            "C_p": float(tables.C_p),
            "C_q": float(tables.C_q),
            "C_v": float(tables.C_v),
        },
    }

    if metadata is not None:
        meta_out["metadata"] = metadata

    json_path.write_text(json.dumps(meta_out, indent=2))

    return npz_path, json_path