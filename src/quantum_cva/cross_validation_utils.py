from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from qiskit_algorithms.optimizers import SPSA


@dataclass(frozen=True)
class SPSACVResult:
    best_cost: float
    best_lr: float
    best_pert: float
    best_eps: float | None
    coarse_topk: list[tuple]
    refined_topk: list[tuple]
    all_coarse: list[tuple]


class SPSAHyperparamCV:
    """
    Two-stage SPSA hyperparameter selection.

    Supports:
      - LR, PERT only  (e.g., CRCA)
      - LR, PERT, EPS  (e.g., QCBM)

    Improvement vs previous version:
      - Instead of evaluating only the final theta from SPSA,
        you can evaluate an *aggregate* of the last `tail_window` iterates
        to reduce sensitivity to outliers (shot noise).
    """

    def __init__(
        self,
        *,
        x0: np.ndarray,
        lr_grid: Iterable[float],
        pert_grid: Iterable[float],
        eps_grid: Iterable[float] | None = None,
        shots_train: int = 2000,
        shots_eval: int = 8000,
        cv_iters: int = 250,
        top_k: int = 5,
        cv_iters_fine: int = 400,
        refine_factors: tuple[float, ...] = (0.7, 1.0, 1.3),
        resamplings: int = 1,
        blocking: bool = False,
        trust_region: bool = True,
        verbose: bool = True,
        # --- new ---
        tail_window: int = 20,
        tail_mode: Literal["mean_theta", "best_eval_in_tail"] = "mean_theta",
        tail_eval_shots: int | None = None,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.lr_grid = list(lr_grid)
        self.pert_grid = list(pert_grid)
        self.eps_grid = list(eps_grid) if eps_grid is not None else None

        self.shots_train = int(shots_train)
        self.shots_eval = int(shots_eval)
        self.cv_iters = int(cv_iters)
        self.top_k = int(top_k)
        self.cv_iters_fine = int(cv_iters_fine)

        self.refine_factors = tuple(refine_factors)
        self.resamplings = int(resamplings)
        self.blocking = bool(blocking)
        self.trust_region = bool(trust_region)
        self.verbose = bool(verbose)

        self.tail_window = int(tail_window)
        self.tail_mode = tail_mode
        self.tail_eval_shots = int(tail_eval_shots) if tail_eval_shots is not None else None

    def _refine(self, val: float) -> list[float]:
        return sorted({float(val * f) for f in self.refine_factors if (val * f) > 0.0})

    def _run_spsa_and_pick_theta(
        self,
        *,
        fun_train: Callable[[np.ndarray], float],
        fun_eval: Callable[[np.ndarray], float],
        iters: int,
        lr: float,
        pert: float,
    ) -> np.ndarray:
        """
        Run SPSA and return a representative theta for evaluation:
          - "mean_theta": average of last `tail_window` iterates
          - "best_eval_in_tail": pick argmin(fun_eval) among last window
        """
        tail: list[np.ndarray] = []

        def cb(nfev, x, fx, stepsize, accepted):
            # store copies of the iterates; keep only last W
            tail.append(np.asarray(x, dtype=float).copy())
            if len(tail) > self.tail_window:
                tail.pop(0)

        opt = SPSA(
            maxiter=int(iters),
            learning_rate=float(lr),
            perturbation=float(pert),
            resamplings=int(self.resamplings),
            blocking=bool(self.blocking),
            trust_region=bool(self.trust_region),
            callback=cb,
        )
        _ = opt.minimize(fun=fun_train, x0=self.x0)

        if not tail:
            return self.x0.copy()

        if self.tail_mode == "mean_theta":
            return np.mean(np.stack(tail, axis=0), axis=0)

        # tail_mode == "best_eval_in_tail"
        vals = [float(fun_eval(th)) for th in tail]
        return tail[int(np.argmin(vals))]

    def fit(
        self,
        *,
        make_cost_train: Callable[[int, float | None], Callable[[np.ndarray], float]],
        make_cost_eval: Callable[[int, float | None], Callable[[np.ndarray], float]],
    ) -> SPSACVResult:
        use_eps = self.eps_grid is not None

        # ---------- coarse ----------
        all_coarse: list[tuple] = []

        def eval_theta(cost_eval, theta) -> float:
            return float(cost_eval(theta))

        if use_eps:
            for lr, pert, eps in product(self.lr_grid, self.pert_grid, self.eps_grid):  # type: ignore[arg-type]
                epsf = float(eps)
                cost_train = make_cost_train(self.shots_train, epsf)

                # evaluation objective for *ranking*
                shots_eval_main = self.shots_eval
                cost_eval_main = make_cost_eval(shots_eval_main, epsf)

                # optional separate shots for choosing inside tail
                if self.tail_eval_shots is not None:
                    cost_eval_tail = make_cost_eval(self.tail_eval_shots, epsf)
                else:
                    cost_eval_tail = cost_eval_main

                theta_rep = self._run_spsa_and_pick_theta(
                    fun_train=cost_train,
                    fun_eval=cost_eval_tail,
                    iters=self.cv_iters,
                    lr=lr,
                    pert=pert,
                )
                val = eval_theta(cost_eval_main, theta_rep)
                all_coarse.append((val, float(lr), float(pert), epsf))

                if self.verbose:
                    print(f"[coarse] lr={lr:.4f}, pert={pert:.4f}, eps={epsf:.1e} -> eval={val:.3e}")
        else:
            for lr, pert in product(self.lr_grid, self.pert_grid):
                cost_train = make_cost_train(self.shots_train, None)

                shots_eval_main = self.shots_eval
                cost_eval_main = make_cost_eval(shots_eval_main, None)

                if self.tail_eval_shots is not None:
                    cost_eval_tail = make_cost_eval(self.tail_eval_shots, None)
                else:
                    cost_eval_tail = cost_eval_main

                theta_rep = self._run_spsa_and_pick_theta(
                    fun_train=cost_train,
                    fun_eval=cost_eval_tail,
                    iters=self.cv_iters,
                    lr=lr,
                    pert=pert,
                )
                val = eval_theta(cost_eval_main, theta_rep)
                all_coarse.append((val, float(lr), float(pert)))

                if self.verbose:
                    print(f"[coarse] lr={lr:.4f}, pert={pert:.4f} -> eval={val:.3e}")

        all_coarse.sort(key=lambda t: t[0])
        coarse_topk = all_coarse[: max(1, self.top_k)]

        if self.verbose:
            print("\nTOP (coarse):")
            for r in coarse_topk:
                print(r)

        # ---------- refine ----------
        refined_all: list[tuple] = []

        if use_eps:
            for _, lr0, pert0, eps0 in coarse_topk:
                lr_f = self._refine(float(lr0))
                pe_f = self._refine(float(pert0))
                ep_f = self._refine(float(eps0))

                for lr, pert, eps in product(lr_f, pe_f, ep_f):
                    epsf = float(eps)
                    cost_train = make_cost_train(self.shots_train, epsf)

                    cost_eval_main = make_cost_eval(self.shots_eval, epsf)
                    if self.tail_eval_shots is not None:
                        cost_eval_tail = make_cost_eval(self.tail_eval_shots, epsf)
                    else:
                        cost_eval_tail = cost_eval_main

                    theta_rep = self._run_spsa_and_pick_theta(
                        fun_train=cost_train,
                        fun_eval=cost_eval_tail,
                        iters=self.cv_iters_fine,
                        lr=lr,
                        pert=pert,
                    )
                    val = float(cost_eval_main(theta_rep))
                    refined_all.append((val, float(lr), float(pert), epsf))

                    if self.verbose:
                        print(f"[fine]   lr={lr:.6f}, pert={pert:.6f}, eps={epsf:.1e} -> eval={val:.3e}")
        else:
            for _, lr0, pert0 in coarse_topk:
                lr_f = self._refine(float(lr0))
                pe_f = self._refine(float(pert0))

                for lr, pert in product(lr_f, pe_f):
                    cost_train = make_cost_train(self.shots_train, None)

                    cost_eval_main = make_cost_eval(self.shots_eval, None)
                    if self.tail_eval_shots is not None:
                        cost_eval_tail = make_cost_eval(self.tail_eval_shots, None)
                    else:
                        cost_eval_tail = cost_eval_main

                    theta_rep = self._run_spsa_and_pick_theta(
                        fun_train=cost_train,
                        fun_eval=cost_eval_tail,
                        iters=self.cv_iters_fine,
                        lr=lr,
                        pert=pert,
                    )
                    val = float(cost_eval_main(theta_rep))
                    refined_all.append((val, float(lr), float(pert)))

                    if self.verbose:
                        print(f"[fine]   lr={lr:.6f}, pert={pert:.6f} -> eval={val:.3e}")

        refined_all.sort(key=lambda t: t[0])
        best = refined_all[0]

        if use_eps:
            best_cost, best_lr, best_pert, best_eps = best
        else:
            best_cost, best_lr, best_pert = best
            best_eps = None

        if self.verbose:
            print("\nBEST (final):")
            print(f"  LR   = {best_lr}")
            print(f"  PERT = {best_pert}")
            if use_eps:
                print(f"  EPS  = {best_eps}")
            print(f"  Eval = {best_cost:.6e}")
            print(f"  tail_mode={self.tail_mode}, tail_window={self.tail_window}, tail_eval_shots={self.tail_eval_shots}")

        return SPSACVResult(
            best_cost=float(best_cost),
            best_lr=float(best_lr),
            best_pert=float(best_pert),
            best_eps=None if best_eps is None else float(best_eps),
            coarse_topk=coarse_topk,
            refined_topk=refined_all[: max(10, self.top_k)],
            all_coarse=all_coarse,
        )