"""T_up 参数扫描: 传导限制区多组 T_up 的 PINN vs FD 对比.

涵盖从强非线性 (T_up=40 eV) 到接近线性 (T_up=200 eV) 的范围,
确保清晰展示 T(s) 剖面的形状变化.

Usage:
    python experiments/pinn/scan_tup.py
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from sol_pinn.physics.regimes import from_name
from sol_pinn.physics.params import SolverConfig
from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.pinn.pinn_solver import PINNSolver
from sol_pinn.utils.sampling import target_refined, torch_collocation
from sol_pinn.utils.metrics import relative_L2_error, max_error


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -- Conduction-limited regime params --
    T_up_values = [40, 50, 60, 70, 80, 90, 100, 150, 200]
    regime = "conduction-limited"
    base_config = from_name(regime)

    results = []

    for i, T_up in enumerate(T_up_values):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(T_up_values)}] T_up = {T_up} eV")
        print(f"{'='*60}")

        config = base_config.clone_with(T_up=T_up)

        # -- FD reference --
        print("  FD reference...")
        sc = SolverConfig(n_points=500, max_iter=2000, tol=1e-10)
        fd = FDSolver(config, sc).solve()
        T_t_fd = fd["T"][-1]
        ratio = T_t_fd / T_up
        print(f"    T_t = {T_t_fd:.2f} eV  (T_t/T_up = {ratio:.4f})")

        if not fd["converged"]:
            print(f"    WARNING: FD did not converge, skipping PINN training")
            results.append({"T_up": T_up, "T_t": T_t_fd, "ratio": ratio,
                            "rel_L2": None, "max_err": None, "converged": False})
            continue

        # -- PINN training --
        print("  Building PINN...")
        solver = PINNSolver(
            config,
            use_fourier=True,
            fourier_kwargs={"mapping_size": 64, "sigma": 1.0},
            network_kwargs={"layer_sizes": [128, 128, 128, 128, 128],
                            "activation": "tanh",
                            "output_bias": T_up / 2},
            causal=False,
            device=device,
        )

        s_np = target_refined(200, 100, config.L, boundary_ratio=0.1)
        s_colloc = torch_collocation(s_np)

        print("  Training PINN...")
        solver.train(s_colloc, n_adam=3000, n_lbfgs=200)

        # -- Evaluation --
        s_eval = np.linspace(0, config.L, 500)
        T_pred = solver.predict(s_eval)
        T_ref = np.interp(s_eval, fd["s"], fd["T"])

        rel_l2 = relative_L2_error(T_pred, T_ref)
        max_err = max_error(T_pred, T_ref)
        print(f"  Results: rel_L2 = {rel_l2:.4e}, max_err = {max_err:.4e}")

        results.append({
            "T_up": T_up,
            "T_t": T_t_fd,
            "ratio": ratio,
            "rel_L2": rel_l2,
            "max_err": max_err,
            "converged": True,
            "s_eval": s_eval,
            "T_pred": T_pred,
            "T_ref": T_ref,
        })

    # -- Console summary --
    print(f"\n{'='*70}")
    print(f"{'T_up':>6} | {'T_t':>8} | {'T_t/T_up':>8} | {'rel_L2':>10} | {'max_err':>10}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    for r in results:
        if r["rel_L2"] is not None:
            print(f"{r['T_up']:6.0f} | {r['T_t']:8.2f} | {r['ratio']:8.4f} | "
                  f"{r['rel_L2']:10.4e} | {r['max_err']:10.4e}")
        else:
            print(f"{r['T_up']:6.0f} | {r['T_t']:8.2f} | {r['ratio']:8.4f} | "
                  f"{'FAIL':>10} | {'FAIL':>10}")
    print(f"{'='*70}")

    # -- Plotting --
    out_dir = Path("figures/pinn")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 3x3 grid of T(s) profiles
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        if r["T_pred"] is None:
            ax.text(0.5, 0.5, "FAILED", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"T_up = {r['T_up']:.0f} eV", fontsize=11)
            continue

        ax.plot(r["s_eval"], r["T_ref"], "b-", linewidth=2, label="FD")
        ax.plot(r["s_eval"], r["T_pred"], "r--", linewidth=1.5, label="PINN")
        ax.set_xlabel("s [m]", fontsize=10)
        ax.set_ylabel("$T_e$ [eV]", fontsize=10)
        ax.set_title(f"T_up={r['T_up']:.0f} eV  (T_t/T_up={r['ratio']:.3f})", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("PINN vs FD: T(s) Profiles Across T_up (Conduction-Limited Regime)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "scan_tup_profiles.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved figures/pinn/scan_tup_profiles.png")

    # Figure 2: Summary — T_t/T_up ratio and PINN error
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    tup_vals = np.array([r["T_up"] for r in results])
    ratio_vals = np.array([r["ratio"] for r in results])
    err_vals = np.array([r["rel_L2"] if r["rel_L2"] is not None else np.nan for r in results])

    # T_t/T_up vs T_up
    ax1.plot(tup_vals, ratio_vals, "bo-", linewidth=2, markersize=6)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="T_t = T_up")
    ax1.set_xlabel("$T_{up}$ [eV]", fontsize=12)
    ax1.set_ylabel("$T_t / T_{up}$", fontsize=12)
    ax1.set_title("Target-to-Upstream Temperature Ratio", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Relative L2 error vs T_up
    valid = ~np.isnan(err_vals)
    ax2.semilogy(tup_vals[valid], err_vals[valid], "ro-", linewidth=2, markersize=6)
    ax2.set_xlabel("$T_{up}$ [eV]", fontsize=12)
    ax2.set_ylabel("Relative L2 Error", fontsize=12)
    ax2.set_title("PINN Error vs $T_{up}$", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig2.suptitle("Multi-T_up Scan Summary (Conduction-Limited Regime)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "scan_tup_summary.png", dpi=150, bbox_inches="tight")
    print(f"Saved figures/pinn/scan_tup_summary.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
