"""FD 参考求解器 (独立运行).

Usage:
    python experiments/validation/run_fd_solver.py
    python experiments/validation/run_fd_solver.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.physics.regimes import from_name
from sol_pinn.physics.params import SolverConfig
from sol_pinn.utils.io import save_solution
from sol_pinn.utils.plotting import plot_heat_flux, plot_temperature_profile


def main():
    # Parse optional --regime argument
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    config = from_name(regime)
    solver_config = SolverConfig(n_points=500, max_iter=2000, tol=1e-10)

    print(f"Regime: {regime}")
    print(f"  T_up={config.T_up} eV, L={config.L} m, kappa={config.kappa_parallel}, "
          f"p0={config.p0:.0e}, gamma={config.gamma_sheath}")

    solver = FDSolver(config, solver_config)
    result = solver.solve()

    if result["converged"]:
        T_t = result["T"][-1]
        print(f"Converged in {result['n_iter']} iterations")
        print(f"  T_t = {T_t:.2f} eV  (T_t/T_up = {T_t/config.T_up:.4f})")
        print(f"  q_t = {result['q'][-1]:.2e} W/m^2")
    else:
        print(f"WARNING: Did not converge after {result['n_iter']} iterations")

    save_solution(
        f"data/baseline/reference_{regime}.npz",
        result["s"],
        result["T"],
        metadata={
            "regime": regime,
            "T_up": config.T_up,
            "L": config.L,
            "kappa": config.kappa_parallel,
            "p0": config.p0,
            "gamma": config.gamma_sheath,
            "n_iter": result["n_iter"],
            "T_target": float(result["T"][-1]),
            "q_target": float(result["q"][-1]),
        },
    )
    print(f"Saved to data/baseline/reference_{regime}.npz")

    plot_temperature_profile(
        result["s"],
        result["T"],
        title=f"SOL Temperature Profile ({regime}, T_up={config.T_up} eV)",
        save_path=f"figures/baseline/temperature_profile_{regime}.png",
    )
    plot_heat_flux(
        result["s"],
        result["T"],
        result["dT_ds"],
        config.kappa_parallel,
        title=f"Parallel Heat Flux q(s) - {regime}",
        save_path=f"figures/baseline/heat_flux_{regime}.png",
    )
    print(f"Figures saved to figures/baseline/ ({regime})")


if __name__ == "__main__":
    main()
