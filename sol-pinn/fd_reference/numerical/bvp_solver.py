"""Boundary-value solver wrapper used to cross-check the FD implementation."""

import numpy as np
from scipy.integrate import solve_bvp

from sol_pinn.physics.constants import T_EPSILON
from sol_pinn.physics.params import SOLConfig, SolverConfig


class BVPSolver:
    """Wrap ``scipy.integrate.solve_bvp`` for the SOL transport equation."""

    def __init__(self, config: SOLConfig, solver_config: SolverConfig):
        self.config = config
        self.sc = solver_config

    def _ode_system(self, s, y):
        """Return the first-order ODE system for ``T`` and ``dT/ds``."""
        T, dT_ds = y
        T_safe = np.maximum(T, T_EPSILON)
        S_val = np.array([self.config.S_E(si) for si in s])

        kappa = self.config.kappa_parallel
        d2T_ds2 = -2.5 * dT_ds ** 2 / T_safe - S_val / (kappa * T_safe ** 2.5)
        return np.vstack([dT_ds, d2T_ds2])

    def _bc(self, ya, yb):
        """Return upstream and sheath boundary residuals."""
        T_a, _ = ya
        T_b, dT_ds_b = yb
        T_b_safe = max(T_b, T_EPSILON)

        bc_up = T_a - self.config.T_up
        bc_target = (
            self.config.kappa_parallel * T_b_safe ** 2.5 * dT_ds_b
            + self.config.alpha * np.sqrt(T_b_safe)
        )
        return np.array([bc_up, bc_target])

    def solve(self, T_init=None):
        """Solve the BVP and return the reconstructed fields."""
        N = self.sc.n_points
        s = np.linspace(0, self.config.L, N)

        if T_init is not None:
            T_guess = T_init
        else:
            T_guess = self.config.T_up * (1.0 - 0.8 * s / self.config.L)
        T_guess = np.maximum(T_guess, T_EPSILON)
        dT_ds_guess = np.gradient(T_guess, s)
        y_init = np.vstack([T_guess, dT_ds_guess])

        solution = solve_bvp(
            self._ode_system,
            self._bc,
            s,
            y_init,
            max_nodes=10 * N,
            tol=self.sc.tol,
        )

        if not solution.success:
            print(f"BVP solver warning: {solution.message}")

        T = solution.sol(s)[0]
        dT_ds = solution.sol(s)[1]
        T_safe = np.maximum(T, T_EPSILON)
        q = -self.config.kappa_parallel * T_safe ** 2.5 * dT_ds

        return {
            "s": s,
            "T": T,
            "dT_ds": dT_ds,
            "q": q,
            "success": solution.success,
            "solution_obj": solution,
        }
