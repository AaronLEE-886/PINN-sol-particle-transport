"""PINN vs FD 详细对比 (FD 验证).

Usage:
    python experiments/validation/compare_pinn_fd.py
    python experiments/validation/compare_pinn_fd.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from sol_pinn.physics.params import SOLConfig, SolverConfig
from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.pinn.pinn_solver import PINNSolver
from sol_pinn.physics.regimes import from_name
from sol_pinn.utils.sampling import target_refined, torch_collocation
from sol_pinn.utils.metrics import relative_L2_error, max_error, physics_consistency_metrics
from sol_pinn.utils.plotting import plot_comparison


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    print("=" * 60)
    print(f"PINN vs FD: Detailed Comparison ({regime})")
    print("=" * 60)

    config = from_name(regime)
    device = "cpu"

    # FD reference
    print("\nGenerating FD reference solution...")
    fd = FDSolver(config, SolverConfig(n_points=2000, max_iter=2000, tol=1e-10))
    fd_result = fd.solve()
    print(f"  T_target = {fd_result['T'][-1]:.4f} eV")

    # PINN
    print("\nTraining PINN...")
    solver = PINNSolver(
        config,
        use_fourier=True,
        network_kwargs={"layer_sizes": [128, 128, 128, 128, 128]},
        device=device,
    )
    s_np = target_refined(300, 150, config.L, boundary_ratio=0.1)
    s_colloc = torch_collocation(s_np)
    solver.train(s_colloc, n_adam=10000, n_lbfgs=500)

    # Evaluate on fine grid
    s_eval = np.linspace(0, config.L, 1000)
    T_pred = solver.predict(s_eval)
    T_ref = np.interp(s_eval, fd_result["s"], fd_result["T"])

    rel_l2 = relative_L2_error(T_pred, T_ref)
    max_err = max_error(T_pred, T_ref)
    physics = physics_consistency_metrics(s_eval, T_pred, config)
    print(f"\nResults:")
    print(f"  Relative L2 error: {rel_l2:.4e}")
    print(f"  Max error: {max_err:.4e}")
    print(f"  PDE residual RMS: {physics['pde_rms']:.4e}")
    print(f"  Upstream BC abs: {physics['upstream_bc_abs']:.4e}")
    print(f"  Sheath BC abs: {physics['sheath_bc_abs']:.4e}")

    # Plot comparison
    plot_comparison(
        s_eval, T_ref, T_pred,
        title=f"PINN vs FD Comparison ({regime}, rel_L2={rel_l2:.4e})",
        save_path=f"figures/pinn/pinn_vs_fd_detailed_{regime}.png",
    )

    # Error distribution
    error = T_pred - T_ref
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(error, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Error [eV]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Error Distribution ({regime}, rel_L2={rel_l2:.4e})", fontsize=13)
    ax.axvline(0, color="r", linestyle="--")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/pinn/error_distribution_{regime}.png", dpi=150, bbox_inches="tight")
    print(f"Figures saved to figures/pinn/ ({regime})")

    print(f"\nSheath BC check:")
    print(f"  q_pred   = {physics['q_t']:.6e}")
    print(f"  q_sheath = {physics['q_sheath']:.6e}")
    print(f"  diff     = {abs(physics['q_t'] - physics['q_sheath']):.6e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
