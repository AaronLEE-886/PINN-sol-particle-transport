"""Least-squares inversion baseline for comparison with PINN-based inversion."""

import numpy as np
from scipy.optimize import curve_fit

from fd_reference.numerical.fd_solver import FDSolver
from ..physics.params import SOLConfig, SolverConfig


def solve_forward_for_gamma(gamma, config_template, s_eval):
    """Solve the forward problem for a candidate gamma and interpolate to ``s_eval``."""
    cfg = config_template.clone_with(gamma_sheath=gamma)
    sc = SolverConfig(n_points=500, max_iter=2000, tol=1e-10)
    result = FDSolver(cfg, sc).solve()
    return np.interp(s_eval, result["s"], result["T"])


def least_squares_inversion(s_obs, T_obs, config_template, gamma_guess=(5.0, 10.0)):
    """Estimate gamma from observations with ``scipy.optimize.curve_fit``."""

    def model(s, gamma):
        return solve_forward_for_gamma(gamma, config_template, s)

    try:
        popt, pcov = curve_fit(
            model,
            s_obs,
            T_obs,
            p0=gamma_guess[0],
            bounds=(gamma_guess[0] * 0.1, gamma_guess[1] * 10.0),
            maxfev=100,
        )
        gamma_opt = popt[0]
        gamma_std = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else np.inf
    except (RuntimeError, ValueError) as e:
        print(f"  LS inversion failed: {e}")
        gamma_opt = gamma_guess[0]
        gamma_std = np.inf

    return gamma_opt, gamma_std
