"""PINN 训练与 FD 对比评估.

Usage:
    python experiments/pinn/train.py
    python experiments/pinn/train.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np
import torch

from sol_pinn.physics.regimes import from_name
from sol_pinn.physics.params import SolverConfig
from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.pinn.pinn_solver import PINNSolver
from sol_pinn.utils.sampling import target_refined, torch_collocation
from sol_pinn.utils.plotting import plot_comparison, plot_loss_curves
from sol_pinn.utils.metrics import relative_L2_error, max_error


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Regime: {regime}  Device: {device}")

    config = from_name(regime)

    # FD baseline
    print("Generating FD baseline...")
    sc = SolverConfig(n_points=500, max_iter=2000, tol=1e-10)
    fd_result = FDSolver(config, sc).solve()
    T_t = fd_result["T"][-1]
    print(f"  FD: T_t={T_t:.2f} eV (T_t/T_up={T_t/config.T_up:.3f}), converged={fd_result['converged']}")

    # PINN
    print("\nBuilding PINN solver...")
    solver = PINNSolver(
        config,
        use_fourier=True,
        fourier_kwargs={"mapping_size": 64, "sigma": 1.0},
        network_kwargs={"layer_sizes": [128, 128, 128, 128, 128],
                        "activation": "tanh",
                        "output_bias": config.T_up / 2},
        causal=False,
        device=device,
    )

    # Sampling
    s_np = target_refined(200, 100, config.L, boundary_ratio=0.1)
    s_colloc = torch_collocation(s_np)

    # Train
    print("\nTraining PINN...")
    solver.train(s_colloc, n_adam=5000, n_lbfgs=300)

    # Evaluate
    print("\nEvaluating PINN...")
    s_eval = np.linspace(0, config.L, 500)
    T_pred = solver.predict(s_eval)
    T_ref = np.interp(s_eval, fd_result["s"], fd_result["T"])

    rel_l2 = relative_L2_error(T_pred, T_ref)
    max_err = max_error(T_pred, T_ref)
    print(f"  Relative L2 error: {rel_l2:.4e}")
    print(f"  Max error: {max_err:.4e}")

    # Plots
    plot_comparison(
        s_eval, T_ref, T_pred,
        title=f"PINN vs FD ({regime}, T_up={config.T_up} eV, rel_L2={rel_l2:.4f})",
        save_path=f"figures/pinn/pinn_vs_fd_comparison_{regime}.png",
    )
    plot_loss_curves(
        solver.trainer.history,
        save_path=f"figures/pinn/loss_curves_{regime}.png",
    )
    print(f"Figures saved to figures/pinn/ ({regime})")


if __name__ == "__main__":
    main()
