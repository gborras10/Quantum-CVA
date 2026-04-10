import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def minimize_with_cost_history(
        cost_fn, 
        *, 
        x0, 
        minimize_fn, 
        method, 
        options
    ):
    x0 = np.asarray(x0, dtype=float)
    f0 = float(cost_fn(x0))

    cost_history: list[float] = [f0]
    theta_history: list[np.ndarray] = [x0.copy()]

    def wrapped(x):
        x = np.asarray(x, dtype=float)
        return float(cost_fn(x))

    def callback(xk):
        xk = np.asarray(xk, dtype=float)
        fk = float(cost_fn(xk))
        cost_history.append(fk)
        theta_history.append(xk.copy())

    res = minimize_fn(
        wrapped,
        x0=x0,
        method=method,
        callback=callback,
        options=options,
    )

    x_final = np.asarray(res.x, dtype=float)
    f_final = float(res.fun)

    same_theta = np.allclose(theta_history[-1], x_final, rtol=0.0, atol=1e-15)
    same_cost = abs(cost_history[-1] - f_final) <= 1e-15

    if not (same_theta and same_cost):
        cost_history.append(f_final)
        theta_history.append(x_final.copy())

    return (
        res,
        np.asarray(cost_history, dtype=float),
        np.vstack(theta_history),
    )

def run_stage1(
        x0: np.ndarray, 
        *, 
        maxiter: int, 
        rhobeg: float, 
        cost_fn, 
        qcbm, 
        tol: float,
        target_entropy: float
    ) -> dict[str, object]:

    t0 = time.perf_counter()

    result, cost_history, theta_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        minimize_fn=minimize,
        method="COBYLA",
        options={
            "maxiter": int(maxiter),
            "rhobeg": float(rhobeg),
            "disp": True,
            "tol": float(tol),
        },
    )

    elapsed = time.perf_counter() - t0
    theta_star = np.asarray(result.x, dtype=float)
    p_star = qcbm.probabilities(theta_star)
    ce_final = float(result.fun)
    kl_final = float(ce_final - target_entropy)

    return {
        "result": result,
        "theta_star": theta_star,
        "p_star": p_star,
        "cost_history": np.asarray(cost_history, dtype=float),
        "theta_history": np.asarray(theta_history, dtype=float),
        "elapsed_time": elapsed,
        "ce_final": ce_final,
        "kl_final": kl_final,
    }

def run_stage2(
        x0: np.ndarray, 
        *, 
        maxiter: int, 
        cost_fn, 
        qcbm, 
        target_entropy: float, 
        maxfun: int = 300000
    ) -> dict[str, object]:
    t0 = time.perf_counter()

    result, cost_history, theta_history = minimize_with_cost_history(
        cost_fn,
        x0=x0,
        minimize_fn=minimize,
        method="L-BFGS-B",
        options={
                    "maxiter": int(maxiter),
                    "maxfun": int(maxfun),
                    "ftol": 1e-12,   
                    "gtol": 1e-10,      
                    "eps": 1e-6,       
                    "maxls": 50,       
                    "maxcor": 20,       
                    "disp": True
                },
    )

    elapsed = time.perf_counter() - t0
    theta_star = np.asarray(result.x, dtype=float)
    p_star = qcbm.probabilities(theta_star)
    ce_final = float(result.fun)
    kl_final = float(ce_final - target_entropy)

    return {
        "result": result,
        "theta_star": theta_star,
        "p_star": p_star,
        "cost_history": np.asarray(cost_history, dtype=float),
        "theta_history": np.asarray(theta_history, dtype=float),
        "elapsed_time": elapsed,
        "ce_final": ce_final,
        "kl_final": kl_final,
    }

def select_checkpoint_indices(cost_history: np.ndarray, *, every: int = 25, top_k: int = 20, n_recent: int = 10) -> np.ndarray:
    cost_history = np.asarray(cost_history, dtype=float).ravel()
    n = cost_history.size
    if n == 0:
        return np.empty(0, dtype=int)

    idx = {0, n - 1}
    every = max(1, int(every))
    idx.update(range(0, n, every))

    top_k = min(int(top_k), n)
    idx.update(np.argsort(cost_history)[:top_k].tolist())

    n_recent = min(int(n_recent), n)
    idx.update(range(n - n_recent, n))

    return np.array(sorted(idx), dtype=int)

def save_experiment(saving_path, checkpoint_path, main_data_dict, checkpoint_data_dict):
    np.savez(saving_path, **main_data_dict)
    np.savez(checkpoint_path, **checkpoint_data_dict)
    print(f"\nResultados guardados en: {saving_path}")
    print(f"Checkpoints guardados en: {checkpoint_path}")
    