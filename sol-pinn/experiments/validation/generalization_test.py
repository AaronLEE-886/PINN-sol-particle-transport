"""参数泛化测试 (FD 验证).

Usage:
    python experiments/validation/generalization_test.py
    python experiments/validation/generalization_test.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sol_pinn.physics.params import SOLConfig, SolverConfig
from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.pinn.network import ParameterizedPINN
from sol_pinn.physics.regimes import from_name
from sol_pinn.utils.metrics import relative_L2_error


def train_parameterized_pinn(regime, T_up_range=(50, 200), n_iter=20000,
                              n_colloc=100, n_bc=20,
                              fourier_kwargs=None, lr=1e-3,
                              device="cpu"):
    """Train a ParameterizedPINN on a range of T_up values.

    Args:
        regime: Regime name for config
        T_up_range: (min, max) T_up range
        n_iter: Number of training iterations
        n_colloc: Collocation points per batch
        n_bc: Boundary condition points per batch
        fourier_kwargs: Fourier encoding parameters
        lr: Learning rate
        device: Device

    Returns:
        model: Trained ParameterizedPINN
        history: Loss history
    """
    config = from_name(regime)
    model = ParameterizedPINN(
        fourier_kwargs=fourier_kwargs,
        L=config.L, T_up_ref=80.0,
    ).to(device)

    # 热身 + 余弦退火
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iter, eta_min=lr * 0.01,
    )

    T_up_min, T_up_max = T_up_range
    history = {"total": [], "pde": [], "up": [], "sheath": []}

    alpha = config.alpha  # sheath constant (independent of T_up)
    kappa = config.kappa_parallel

    pbar = tqdm(range(n_iter), desc="Param-PINN")
    for step in pbar:
        # Sample collocation points
        s_colloc = torch.rand(n_colloc, 1, device=device) * config.L
        s_colloc.requires_grad_(True)
        T_up_colloc = torch.empty(n_colloc, 1, device=device).uniform_(
            T_up_min, T_up_max,
        )

        # Sample BC points
        s_up = torch.zeros(n_bc, 1, device=device)
        T_up_bc = torch.empty(n_bc, 1, device=device).uniform_(
            T_up_min, T_up_max,
        )

        s_sheath = torch.full((n_bc, 1), config.L, device=device)
        s_sheath.requires_grad_(True)
        T_up_sheath = torch.empty(n_bc, 1, device=device).uniform_(
            T_up_min, T_up_max,
        )

        optimizer.zero_grad()

        # === PDE loss (properly normalized) ===
        # PDE: d/ds(κ·T^(5/2)·dT/ds) = 0
        # Normalized: dq_ds / (κ·T_up^3.5 / L²)
        T_colloc = model(s_colloc, T_up_colloc)
        dT_ds = torch.autograd.grad(
            T_colloc, s_colloc,
            grad_outputs=torch.ones_like(T_colloc),
            create_graph=True,
        )[0]

        T_safe = T_colloc + 1e-8
        kappa_eff = kappa * T_safe ** 2.5
        q = -kappa_eff * dT_ds

        dq_ds = torch.autograd.grad(
            q, s_colloc,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
        )[0]

        # Per-sample normalization (each sample has its own T_up)
        scale = kappa * (T_up_colloc.detach() ** 3.5) / (config.L ** 2)
        dq_ds_normalized = dq_ds / (scale + 1e-30)
        L_pde = torch.mean(dq_ds_normalized ** 2)

        # === Upstream BC loss (normalized by T_up²) ===
        T_0 = model(s_up, T_up_bc)
        L_up = torch.mean((T_0 - T_up_bc) ** 2 / (T_up_bc.detach() ** 2 + 1e-30))

        # === Sheath BC loss (normalized by α·T_up^(1/2)) ===
        T_L = model(s_sheath, T_up_sheath)
        dT_ds_L = torch.autograd.grad(
            T_L, s_sheath,
            grad_outputs=torch.ones_like(T_L),
            create_graph=True,
        )[0]

        kappa_eff_L = kappa * (T_L + 1e-8) ** 2.5
        sheath_res = kappa_eff_L * dT_ds_L + alpha * (T_L + 1e-8) ** 0.5
        q_char = alpha * (T_up_sheath.detach() ** 0.5)
        L_sheath = torch.mean(sheath_res ** 2 / (q_char ** 2 + 1e-30))

        # === Total loss ===
        total = L_pde + L_up + L_sheath
        total.backward()
        optimizer.step()
        scheduler.step()

        history["total"].append(total.item())
        history["pde"].append(L_pde.item())
        history["up"].append(L_up.item())
        history["sheath"].append(L_sheath.item())

        if step % 1000 == 0:
            pbar.set_postfix({
                "total": f"{total.item():.2e}",
                "pde": f"{L_pde.item():.2e}",
                "up": f"{L_up.item():.2e}",
                "sheath": f"{L_sheath.item():.2e}",
            })

    return model, history


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    print("=" * 60)
    print(f"Generalization Test: Parameterized PINN ({regime})")
    print("=" * 60)

    config = from_name(regime)

    # ===== Train parameterized PINN =====
    print(f"\nTraining ParameterizedPINN on T_up ∈ [60, 200] ({regime})...")
    model, history = train_parameterized_pinn(
        regime=regime,
        T_up_range=(60, 200),
        n_iter=40000,
        n_colloc=200,
        fourier_kwargs={"mapping_size": 64, "sigma": 1.0},
    )

    # ===== Evaluate on T_up range =====
    T_up_values = np.linspace(60, 200, 15)
    s_eval = np.linspace(0, config.L, 500)
    errors_param = []
    errors_linear = []  # Reference: linear scaling from T_up=80

    # Train a reference single-T_up PINN for comparison
    print("\nTraining reference PINN (single T_up=80)...")
    from sol_pinn.pinn.pinn_solver import PINNSolver
    from sol_pinn.utils.sampling import target_refined, torch_collocation
    ref_config = from_name(regime, T_up=80.0)
    ref_solver = PINNSolver(ref_config, use_fourier=True)
    ref_s_np = target_refined(200, 100, ref_config.L, boundary_ratio=0.1)
    ref_solver.train(torch_collocation(ref_s_np), n_adam=5000, n_lbfgs=300)
    T_ref_100 = ref_solver.predict(s_eval)

    print("\nEvaluating across T_up range...")
    for T_up_val in T_up_values:
        test_config = from_name(regime, T_up=T_up_val)
        fd = FDSolver(test_config, SolverConfig(n_points=500, max_iter=2000, tol=1e-10))
        fd_result = fd.solve()
        T_fd = np.interp(s_eval, fd_result["s"], fd_result["T"])

        # Parameterized PINN prediction
        s_tensor = torch.tensor(s_eval.reshape(-1, 1), dtype=torch.float32)
        T_up_tensor = torch.full((len(s_eval), 1), T_up_val, dtype=torch.float32)
        T_param = model(s_tensor, T_up_tensor).detach().numpy().flatten()
        err_param = relative_L2_error(T_param, T_fd)
        errors_param.append(err_param)

        # Linear scaling baseline
        T_linear = T_ref_100 * (T_up_val / 80.0)
        err_linear = relative_L2_error(T_linear, T_fd)
        errors_linear.append(err_linear)

        print(f"  T_up={T_up_val:.0f} eV: "
              f"param_L2={err_param:.4e}  linear_L2={err_linear:.4e}")

    # ===== Plot =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error comparison
    ax = axes[0, 0]
    ax.semilogy(T_up_values, errors_param, "o-", linewidth=2, label="Parameterized PINN")
    ax.semilogy(T_up_values, errors_linear, "s--", linewidth=2,
                label="Linear scaling from T_up=80")
    ax.axvline(x=80.0, color="r", linestyle=":", alpha=0.5, label="T_up=80")
    ax.set_xlabel("$T_{up}$ [eV]", fontsize=12)
    ax.set_ylabel("Relative L2 Error", fontsize=12)
    ax.set_title(f"Generalization: Parameterized vs Linear Scaling ({regime})", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Profile comparison (selected T_up values)
    ax = axes[0, 1]
    for T_up_show in [70, 100, 150, 200]:
        s_tensor = torch.tensor(s_eval.reshape(-1, 1), dtype=torch.float32)
        T_up_tensor = torch.full((len(s_eval), 1), T_up_show, dtype=torch.float32)
        T_p = model(s_tensor, T_up_tensor).detach().numpy().flatten()

        test_config = from_name(regime, T_up=T_up_show)
        fd = FDSolver(test_config, SolverConfig(n_points=500))
        fd_result = fd.solve()
        T_fd = np.interp(s_eval, fd_result["s"], fd_result["T"])

        ax.plot(s_eval, T_fd, "-", linewidth=2, label=f"FD T_up={T_up_show}")
        ax.plot(s_eval, T_p, "--", linewidth=1.5, label=f"PINN T_up={T_up_show}")

    ax.set_xlabel("s [m]", fontsize=12)
    ax.set_ylabel("$T_e$ [eV]", fontsize=12)
    ax.set_title(f"Profile Comparison ({regime})", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Loss curve
    ax = axes[1, 0]
    ax.semilogy(history["total"], linewidth=1, label="Total")
    ax.semilogy(history["pde"], linewidth=1, label="PDE")
    ax.semilogy(history["up"], linewidth=1, label="Upstream")
    ax.semilogy(history["sheath"], linewidth=1, label="Sheath")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error histogram
    ax = axes[1, 1]
    ax.bar(T_up_values - 2, errors_param, width=3, alpha=0.7, label="Parameterized")
    ax.bar(T_up_values + 2, errors_linear, width=3, alpha=0.7, label="Linear scaling")
    ax.set_xlabel("$T_{up}$ [eV]", fontsize=12)
    ax.set_ylabel("Relative L2 Error", fontsize=12)
    ax.set_title(f"Error Comparison ({regime})", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path("figures/pinn").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"figures/pinn/generalization_test_{regime}.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to figures/pinn/generalization_test_{regime}.png")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Parameterized PINN mean error: {np.mean(errors_param):.4e}")
    print(f"  Linear scaling mean error:    {np.mean(errors_linear):.4e}")
    print(f"  Parameterized max error:      {max(errors_param):.4e}")
    print(f"  Linear scaling max error:     {max(errors_linear):.4e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
