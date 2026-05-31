"""PINN 消融实验: Fourier 特征与因果训练对比.

Usage:
    python experiments/pinn/ablation_study.py
    python experiments/pinn/ablation_study.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np

from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.physics.params import SOLConfig, SolverConfig
from sol_pinn.physics.regimes import from_name
from sol_pinn.pinn.pinn_solver import PINNSolver
from sol_pinn.utils.metrics import relative_L2_error
from sol_pinn.utils.sampling import target_refined, torch_collocation


def run_ablation(regime, config_name, use_fourier, causal, n_adam=2000):
    """Run one ablation configuration and return its relative L2 error."""
    config = from_name(regime)
    device = "cpu"

    solver = PINNSolver(
        config,
        use_fourier=use_fourier,
        causal=causal,
        network_kwargs={"layer_sizes": [128, 128, 128, 128, 128]},
        device=device,
    )
    s_np = target_refined(200, 100, config.L, boundary_ratio=0.1)
    s_colloc = torch_collocation(s_np)

    print(f"\n  Training {config_name}...")
    solver.train(s_colloc, n_adam=n_adam, n_lbfgs=100)

    s_eval = np.linspace(0, config.L, 500)
    fd = FDSolver(config, SolverConfig(n_points=500, max_iter=2000, tol=1e-10))
    fd_result = fd.solve()
    T_ref = np.interp(s_eval, fd_result["s"], fd_result["T"])
    T_pred = solver.predict(s_eval)

    rel_l2 = relative_L2_error(T_pred, T_ref)
    return {"config": config_name, "rel_L2": rel_l2, "T_pred": T_pred, "T_ref": T_ref}


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    print("=" * 60)
    print(f"Ablation Study: PINN Configuration Comparison ({regime})")
    print("=" * 60)

    configs = [
        ("Baseline (no Fourier)", False, False),
        ("+Fourier", True, False),
        ("+Causal", False, True),
        ("Full (Fourier+Causal)", True, True),
    ]

    results = []
    for name, use_fourier, causal in configs:
        result = run_ablation(regime, name, use_fourier, causal, n_adam=2000)
        results.append(result)
        print(f"  -> {name}: rel_L2 = {result['rel_L2']:.4e}")

    print("\n" + "=" * 60)
    print("Summary:")
    for r in results:
        print(f"  {r['config']:30s}: rel_L2 = {r['rel_L2']:.4e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
