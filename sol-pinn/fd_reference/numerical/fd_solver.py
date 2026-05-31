"""Finite-difference solver for the 1D SOL parallel transport problem."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from sol_pinn.physics.constants import T_EPSILON
from sol_pinn.physics.params import SOLConfig, SolverConfig


class FDSolver:
    """Solve the nonlinear boundary-value problem with Picard iteration.

    Governing equation:
        d/ds(kappa_parallel * T^(5/2) * dT/ds) + S(s) = 0

    Boundary conditions:
        T(0) = T_up
        -kappa_parallel * T(L)^(5/2) * dT/ds(L) = alpha * T(L)^(1/2)
    """

    def __init__(self, config: SOLConfig, solver_config: SolverConfig):
        self.config = config
        self.sc = solver_config

    def _analytical_initial_guess(self, s):
        """Return the exact S=0 solution as initial guess.

        From the integrated two-point model:
            T(s) = (T_up^(7/2) - (7/2 * q/kappa) * s)^(2/7)

        where q = alpha * T_t^(1/2) and T_t solves:
            alpha * T_t^(1/2) * L = 2/7 * kappa * (T_up^(7/2) - T_t^(7/2))
        """
        cfg = self.config
        T_up = cfg.T_up

        # Bisection for T_t
        lo, hi = 1e-10, T_up
        C = 7.0 * cfg.alpha * cfg.L / (2.0 * cfg.kappa_parallel)
        T_up_pow = T_up ** 3.5
        for _ in range(200):
            mid = (lo + hi) / 2
            lhs = mid ** 3.5 + C * (mid ** 0.5 if mid > 0 else 0.0)
            if lhs > T_up_pow:
                hi = mid
            else:
                lo = mid
        T_t = (lo + hi) / 2

        # Constant heat flux
        q = cfg.alpha * (T_t ** 0.5)

        # T(s) = (T_up^(7/2) - (7/2 * q/kappa) * s)^(2/7)
        pow_s = T_up_pow - 3.5 * q / cfg.kappa_parallel * s
        pow_s = np.maximum(pow_s, 1e-30)
        T = pow_s ** (2.0 / 7.0)
        return np.maximum(T, T_EPSILON)

    def _harmonic_mean(self, f_i, f_ip1):
        """Return the harmonic mean of two adjacent face values."""
        denom = f_i + f_ip1
        return 2.0 * f_i * f_ip1 / np.where(denom > 1e-30, denom, 1e-30)

    def solve(self, T_init=None):
        """Run Picard iteration until convergence or iteration limit.

        For S=0 problems the exact analytical solution is returned directly
        (bypassing Picard iteration, which can diverge at low T_up in the
        strongly nonlinear conduction-limited regime).
        """
        cfg = self.config
        sc = self.sc
        N = sc.n_points
        h = cfg.L / (N - 1)
        s = np.linspace(0, cfg.L, N)

        T_analytical = self._analytical_initial_guess(s)

        # Check if S == 0 everywhere → use exact analytical solution
        S_vals = np.array([cfg.S_E(si) for si in s])
        if np.all(S_vals == 0.0):
            T = np.maximum(T_analytical, T_EPSILON)
            dT_ds = np.zeros(N)
            dT_ds[0] = (T[1] - T[0]) / h
            dT_ds[1:-1] = (T[2:] - T[:-2]) / (2 * h)
            dT_ds[-1] = (T[-1] - T[-2]) / h
            f_node = cfg.kappa_parallel * T ** 2.5
            q = -f_node * dT_ds
            return {"s": s, "T": T, "dT_ds": dT_ds, "q": q,
                    "converged": True, "n_iter": 0,
                    "residual_history": []}

        if T_init is not None:
            T = np.maximum(T_init.copy(), T_EPSILON)
        else:
            T = np.maximum(T_analytical, T_EPSILON)

        residual_history = []
        converged = False

        for iteration in range(sc.max_iter):
            T_old = T.copy()

            f_node = cfg.kappa_parallel * T ** 2.5
            f_face = np.zeros(N - 1)
            for i in range(N - 1):
                f_face[i] = self._harmonic_mean(f_node[i], f_node[i + 1])

            diag_main = np.zeros(N)
            diag_lower = np.zeros(N - 1)
            diag_upper = np.zeros(N - 1)
            rhs = np.zeros(N)

            diag_main[0] = 1.0
            rhs[0] = cfg.T_up

            for i in range(1, N - 1):
                f_left = f_face[i - 1]
                f_right = f_face[i]
                diag_lower[i - 1] = -f_left / h ** 2
                diag_main[i] = (f_left + f_right) / h ** 2
                diag_upper[i] = -f_right / h ** 2
                rhs[i] = -S_vals[i]

            # Discretize the sheath condition with a semi-implicit linearization.
            # BC: -kappa * T^(5/2) * dT/ds = alpha * sqrt(T)  at s=L
            # Using a Newton linearization for the RHS sqrt(T) term to avoid
            # the Picard instability that drives T_t -> 0 at low T_up.
            T_N = max(T[-1], T_EPSILON)
            sqrt_TN = np.sqrt(T_N)
            f_N = cfg.kappa_parallel * T_N ** 2.5
            diag_lower[N - 2] = f_N / h
            # Semi-implicit: linearize alpha*sqrt(T) as alpha*(sqrt(T_N) + (T-T_N)/(2*sqrt(T_N)))
            # This moves T/(2*sqrt(T_N)) to the diagonal and the rest to RHS.
            diag_main[N - 1] = -(f_N / h + cfg.alpha / (2.0 * sqrt_TN + 1e-30))
            rhs[N - 1] = cfg.alpha * (sqrt_TN - T_N / (2.0 * sqrt_TN + 1e-30))

            A = sparse.diags(
                [diag_lower, diag_main, diag_upper],
                [-1, 0, 1],
                shape=(N, N),
                format="csr",
            )
            T_new = spsolve(A, rhs)
            T_new = np.maximum(T_new, T_EPSILON)

            T = sc.omega * T_new + (1.0 - sc.omega) * T_old

            norm_old = np.linalg.norm(T_old)
            residual = np.linalg.norm(T - T_old) / (norm_old + 1e-12)
            residual_history.append(residual)

            if residual < sc.tol:
                converged = True
                break

        f_node = cfg.kappa_parallel * T ** 2.5
        dT_ds = np.zeros(N)
        dT_ds[0] = (T[1] - T[0]) / h
        dT_ds[1:-1] = (T[2:] - T[:-2]) / (2 * h)
        dT_ds[-1] = (T[-1] - T[-2]) / h
        q = -f_node * dT_ds

        return {
            "s": s,
            "T": T,
            "dT_ds": dT_ds,
            "q": q,
            "converged": converged,
            "n_iter": iteration + 1,
            "residual_history": residual_history,
        }
