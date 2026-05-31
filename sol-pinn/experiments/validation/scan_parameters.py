"""FD 参数扫描: T_up 与 gamma 参数影响.

Usage:
    python experiments/validation/scan_parameters.py
    python experiments/validation/scan_parameters.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np

from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.physics.params import SOLConfig, SolverConfig
from sol_pinn.physics.regimes import from_name
from sol_pinn.utils.io import save_solution


def scan_T_up(regime):
    """Scan upstream temperature over ``[60, 200]`` eV."""
    config = from_name(regime)
    sc = SolverConfig(n_points=500, max_iter=1000, tol=1e-10)

    T_up_values = np.linspace(60, 200, 8)
    results = []

    for T_up in T_up_values:
        cfg = config.clone_with(T_up=T_up)
        solver = FDSolver(cfg, sc)
        result = solver.solve()
        results.append({
            "T_up": T_up,
            "T_target": float(result["T"][-1]),
            "q_target": float(result["q"][-1]),
            "converged": result["converged"],
            "n_iter": result["n_iter"],
        })
        save_solution(
            f"data/baseline/Tup_{T_up:.0f}.npz",
            result["s"],
            result["T"],
            metadata={"T_up": T_up, "L": config.L},
        )
        status = "OK" if result["converged"] else "FAIL"
        print(
            f"  T_up={T_up:.0f} eV -> T_t={result['T'][-1]:.1f} eV "
            f"q_t={result['q'][-1]:.2e} W/m^2 [{status}]"
        )

    return results


def scan_gamma(regime):
    """Scan sheath coefficient gamma over ``[5, 10]``."""
    config = from_name(regime)
    sc = SolverConfig(n_points=500, max_iter=1000, tol=1e-10)

    gamma_values = np.linspace(5, 10, 6)
    results = []

    for gamma in gamma_values:
        cfg = config.clone_with(gamma_sheath=gamma)
        solver = FDSolver(cfg, sc)
        result = solver.solve()
        results.append({
            "gamma": gamma,
            "T_target": float(result["T"][-1]),
            "converged": result["converged"],
        })
        print(f"  gamma={gamma:.1f} -> T_t={result['T'][-1]:.1f} eV")

    return results


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    print("=" * 60)
    print(f"Phase 1: Parameter Scan ({regime})")
    print("=" * 60)

    print(f"\n1. Scanning T_up over [50, 200] eV ({regime})...")
    T_up_results = scan_T_up(regime)

    print(f"\n2. Scanning gamma over [5, 10] ({regime})...")
    gamma_results = scan_gamma(regime)

    print("\n" + "=" * 60)
    print("Summary:")
    T_targets = [r["T_target"] for r in T_up_results]
    print(f"  T_up range: {T_up_results[0]['T_up']:.0f} - {T_up_results[-1]['T_up']:.0f} eV")
    print(f"  T_target range: {min(T_targets):.1f} - {max(T_targets):.1f} eV")
    print(f"  Gamma scan count: {len(gamma_results)}")
    print("Done!")


if __name__ == "__main__":
    main()
