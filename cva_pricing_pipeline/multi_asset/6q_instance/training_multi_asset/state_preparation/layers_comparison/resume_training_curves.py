from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


def _bootstrap_src_path() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    repo_root = next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return repo_root


REPO_ROOT = _bootstrap_src_path()

from quantum_cva.multi_asset.quantum.training.state_prep_qcbm.qcbm_circuit import (  # noqa: E402
    MLQcbmCircuit,
)


EPS_COST = 1e-9
RESULTS_RELATIVE_DIR = (
    "cva_pricing_pipeline/multi_asset/6q_instance/training_multi_asset/"
    "state_preparation/layers_comparison/results/ideal"
)
CONTINUATION_STEM = "training_curve_continuation"


def _save_continuation(
    path: pathlib.Path,
    *,
    layer: int,
    base_history_points: int,
    best_kl_history: np.ndarray,
    continuation_kl_history: list[float],
    theta_star: np.ndarray,
    elapsed_s: float,
) -> None:
    metadata = {
        "layer": int(layer),
        "base_history_points": int(base_history_points),
        "total_history_points": int(best_kl_history.size),
        "last_callback_index": int(best_kl_history.size - 1),
        "elapsed_s": float(elapsed_s),
        "purpose": "plot-only ideal optimization continuation",
        "optimizer": "L-BFGS-B",
        "optimizer_options": {
            "ftol": 0.0,
            "gtol": 1e-14,
            "eps": 1e-7,
            "maxls": 50,
            "maxcor": 20,
        },
    }
    np.savez(
        path,
        best_kl_history=np.asarray(best_kl_history, dtype=float),
        continuation_kl_history=np.asarray(continuation_kl_history, dtype=float),
        theta_star=np.asarray(theta_star, dtype=float),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )


def resume_layer(
    *,
    layer: int,
    output_dir: pathlib.Path,
    target_iteration: int,
    chunk_size: int,
) -> None:
    layer_dir = output_dir / f"L{layer:02d}"
    result_path = layer_dir / f"qcbm_ideal_L{layer:02d}.npz"
    continuation_path = layer_dir / f"{CONTINUATION_STEM}_L{layer:02d}.npz"
    with np.load(result_path, allow_pickle=True) as result:
        original_history = np.asarray(result["best_kl_history"], dtype=float)
        theta_star = np.asarray(result["theta_star"], dtype=float)
        ptg = np.asarray(result["p_target"], dtype=float)

    base_history_points = int(original_history.size)
    best_kl_history = original_history.copy()
    continuation_kl_history: list[float] = []
    elapsed_s = 0.0
    if continuation_path.exists():
        with np.load(continuation_path, allow_pickle=False) as continuation:
            best_kl_history = np.asarray(
                continuation["best_kl_history"], dtype=float
            )
            continuation_kl_history = list(
                np.asarray(continuation["continuation_kl_history"], dtype=float)
            )
            theta_star = np.asarray(continuation["theta_star"], dtype=float)
            metadata = json.loads(str(continuation["metadata_json"]))
            elapsed_s = float(metadata["elapsed_s"])

    target_points = int(target_iteration) + 1
    if best_kl_history.size >= target_points:
        print(
            f"L={layer:02d}: already has {best_kl_history.size} real history "
            f"points; target is {target_points}."
        )
        return

    qcbm = MLQcbmCircuit(
        n_qubits=int(round(math.log2(ptg.size))),
        n_layers=int(layer),
        name=f"G_p_ideal_L{layer:02d}_continuation",
        entangler="rzz",
        topology="qcbm_heavyhex6",
        backend=AerSimulator(method="statevector"),
        noise_model=None,
        simulation_method="statevector",
    )
    cost_fn = qcbm.cost_fn(ptg, eps=EPS_COST)
    target_h = qcbm.entropy(ptg, eps=EPS_COST)

    while best_kl_history.size < target_points:
        maxiter = min(int(chunk_size), target_points - best_kl_history.size)
        chunk_kl_history: list[float] = []

        def callback(xk: np.ndarray) -> None:
            kl = float(cost_fn(np.asarray(xk, dtype=float)) - target_h)
            chunk_kl_history.append(kl)

        start = time.perf_counter()
        result = minimize(
            cost_fn,
            x0=theta_star,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": int(maxiter),
                "maxfun": 5_000_000,
                "ftol": 0.0,
                "gtol": 1e-14,
                "eps": 1e-7,
                "maxls": 50,
                "maxcor": 20,
                "disp": False,
            },
        )
        elapsed_s += time.perf_counter() - start
        theta_star = np.asarray(result.x, dtype=float)
        if not chunk_kl_history:
            print(
                f"L={layer:02d}: optimizer stopped before producing another "
                f"callback: {result.message}"
            )
            break

        continuation_kl_history.extend(chunk_kl_history)
        best_kl_history = np.r_[
            best_kl_history,
            np.minimum.accumulate(
                np.r_[best_kl_history[-1], chunk_kl_history]
            )[1:],
        ]
        _save_continuation(
            continuation_path,
            layer=layer,
            base_history_points=base_history_points,
            best_kl_history=best_kl_history,
            continuation_kl_history=continuation_kl_history,
            theta_star=theta_star,
            elapsed_s=elapsed_s,
        )
        print(
            f"L={layer:02d}: callbacks={best_kl_history.size - 1}/"
            f"{target_iteration}, best_KL={best_kl_history[-1]:.12e}, "
            f"last_chunk_nit={result.nit}, elapsed={elapsed_s:.1f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue ideal QCBM trajectories for plotting only."
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--target-iteration", type=int, default=2500)
    parser.add_argument("--chunk-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / RESULTS_RELATIVE_DIR
    for layer in args.layers:
        resume_layer(
            layer=int(layer),
            output_dir=output_dir,
            target_iteration=int(args.target_iteration),
            chunk_size=int(args.chunk_size),
        )


if __name__ == "__main__":
    main()
