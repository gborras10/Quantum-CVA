import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def minimize_with_cost_history(cost_fn, *, x0, minimize_fn, method, options):
    x0 = np.asarray(x0, dtype=float)
    f0 = float(cost_fn(x0))

    # Históricos alineados: coste[i] corresponde a theta[i]
    cost_history: list[float] = [f0]
    theta_history: list[np.ndarray] = [x0.copy()]

    def wrapped(x):
        x = np.asarray(x, dtype=float)
        return float(cost_fn(x))

    def callback(xk):
        # Iterado aceptado por el optimizador
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

    # Asegura que el último punto guardado coincide con el resultado final
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

def run_stage1(x0: np.ndarray, *, maxiter: int, rhobeg: float, cost_fn, qcbm, target_entropy: float) -> dict[str, object]:
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

def run_stage2(x0: np.ndarray, *, maxiter: int, cost_fn, qcbm, target_entropy: float, maxfun: int = 300000) -> dict[str, object]:
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
    """
    Selección híbrida de checkpoints.
    """
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

def plot_training_dynamics(cost_history, target_entropy, n_stage1_iters):
    """
    Genera y muestra el gráfico de la evolución del entrenamiento.
    """
    rescaled_plot = np.maximum(cost_history - target_entropy, 1e-15)
    best_so_far = np.minimum.accumulate(rescaled_plot)
    iters = np.arange(1, len(rescaled_plot) + 1)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    ax.plot(iters, rescaled_plot, lw=1.0, color="gray", alpha=0.5, zorder=1)
    ax.scatter(iters[:n_stage1_iters], rescaled_plot[:n_stage1_iters], color="blue", s=15, alpha=0.7, label="Stage 1 (COBYLA)", zorder=2)
    ax.scatter(iters[n_stage1_iters:], rescaled_plot[n_stage1_iters:], color="red", s=15, alpha=0.7, label="Stage 2 (L-BFGS-B)", zorder=2)
    ax.axvline(x=n_stage1_iters, color="black", linestyle="--", linewidth=1.5, label="Transition Stage 1 -> 2")
    ax.plot(iters, best_so_far, lw=2.0, color="green", linestyle=":", label="Best KL so far", zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(r"$KL(p_{\text{target}} \parallel p_{\theta})$")
    ax.set_title("KL evolution during training (Stage 1 vs Stage 2)")
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

def save_experiment(saving_path, checkpoint_path, main_data_dict, checkpoint_data_dict):
    """
    Guarda los diccionarios de datos en archivos .npz.
    """
    np.savez(saving_path, **main_data_dict)
    np.savez(checkpoint_path, **checkpoint_data_dict)
    print(f"\nResultados guardados en: {saving_path}")
    print(f"Checkpoints guardados en: {checkpoint_path}")
    