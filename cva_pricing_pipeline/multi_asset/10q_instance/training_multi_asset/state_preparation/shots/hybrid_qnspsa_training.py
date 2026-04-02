from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from hybrid_qnspsa_utils import (
    TrainingConfig,
    build_paths,
    load_target,
    build_qcbm,
    run_statevector_pretraining,
    evaluate_sampled_reference,
    build_shots_warm_start,
    run_local_shots_refinement,
    compute_final_metrics,
    plot_distributions,
    plot_exact_kl_history,
    save_results,
)


def main() -> None:
    cfg = TrainingConfig()
    paths = build_paths(__file__)

    p_target, n_qubits = load_target(paths.target_path)
    qcbm = build_qcbm(n_qubits=n_qubits, n_layers=cfg.n_layers)

    sv = run_statevector_pretraining(
        qcbm=qcbm,
        p_target=p_target,
        cfg=cfg,
    )

    sampled_ref = evaluate_sampled_reference(
        qcbm=qcbm,
        p_target=p_target,
        theta_sv=sv["theta_sv"],
        cfg=cfg,
    )

    warm = build_shots_warm_start(
        qcbm=qcbm,
        p_target=p_target,
        theta_sv=sv["theta_sv"],
        cfg=cfg,
    )

    print(
        f"Initial exact KL at warm start: {warm['metrics_p0_exact']['kl']:.6e} | "
        f"reference statevector KL: {sv['kl_sv_final']:.6e}"
    )

    shots = run_local_shots_refinement(
        qcbm=qcbm,
        p_target=p_target,
        theta_sv=sv["theta_sv"],
        x0_shots=warm["x0_shots"],
        cfg=cfg,
    )

    final = compute_final_metrics(
        qcbm=qcbm,
        p_target=p_target,
        theta_star=shots["theta_star"],
        cfg=cfg,
    )

    print("\n=== FINAL SUMMARY ===")
    print(f"Best exact KL found during shots refinement: {shots['best_exact_kl']:.6e}")
    print(f"Final exact KL(theta_star):   {final['metrics_final_exact']['kl']:.6e}")
    print(f"Final sampled KL(theta_star): {final['metrics_final_sampled']['kl']:.6e}")
    print(f"Shots refinement elapsed: {shots['shots_elapsed_time']:.2f}s")

    _ = qcbm.qc.draw(output="mpl", fold=40)
    plt.show()

    plot_distributions(
        target=p_target,
        before=warm["p0_exact"],
        after=final["p_star_exact"],
        title_before=(
            "Before shots refinement (warm start from statevector) | "
            f"exact KL = {warm['metrics_p0_exact']['kl']:.3e}"
        ),
        title_after=(
            "After shots refinement (best exact-KL checkpoint) | "
            f"exact KL = {final['metrics_final_exact']['kl']:.3e}"
        ),
    )

    kl_best_so_far = np.minimum.accumulate(shots["kl_history_exact"])
    kl_best_idx = np.flatnonzero(
        np.r_[True, kl_best_so_far[1:] < kl_best_so_far[:-1] - 1e-15]
    )

    plot_exact_kl_history(
        kl_eval_iters=shots["kl_eval_iters"],
        kl_history=shots["kl_history_exact"],
        kl_best_so_far=kl_best_so_far,
        kl_best_idx=kl_best_idx,
        stage_boundaries=shots["stage_boundaries"],
        stage_labels=shots["stage_labels"],
        kl_statevector_reference=sv["kl_sv_final"],
    )

    save_results(
        out_path=paths.out_path,
        cfg=cfg,
        p_target=p_target,
        sv=sv,
        sampled_ref=sampled_ref,
        warm=warm,
        shots=shots,
        final=final,
    )


if __name__ == "__main__":
    main()