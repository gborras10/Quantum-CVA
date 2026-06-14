from quantum_cva.multi_asset.quantum.training.functional_encoding_crca.crca.crca_circuit import CrcaCircuit
import numpy as np


def build_support_aware_cost(
    TARGET_THRESHOLD: float,
    RELATIVE_EPS: float,
    LAMBDA_POS: float,
    LAMBDA_ZERO: float,
    crca: CrcaCircuit, 
    f_target: np.ndarray,
) -> tuple[callable, np.ndarray, np.ndarray]:
    pos_mask = f_target > TARGET_THRESHOLD
    zero_mask = ~pos_mask

    def cost_fn(x: np.ndarray) -> float:
        fx = np.asarray(
            crca.function_values(np.asarray(x, dtype=float), shots=None),
            dtype=float,
        ).reshape(-1)

        pos_term = 0.0
        zero_term = 0.0

        if np.any(pos_mask):
            rel_diff = (fx[pos_mask] - f_target[pos_mask]) / (f_target[pos_mask] + RELATIVE_EPS)
            pos_term = float(np.mean(rel_diff * rel_diff))

        if np.any(zero_mask):
            zero_term = float(np.mean(np.abs(fx[zero_mask])))

        return LAMBDA_POS * pos_term + LAMBDA_ZERO * zero_term

    return cost_fn, pos_mask, zero_mask