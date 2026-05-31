"""Boundary-condition residuals for upstream and sheath constraints."""

import torch

from .constants import T_EPSILON
from .params import SOLConfig


def upstream_bc_residual(T_pred, T_up: float) -> torch.Tensor:
    """Return the squared residual of the upstream Dirichlet condition."""
    return (T_pred - T_up) ** 2


def sheath_bc_residual(T_t, dT_ds_t, config: SOLConfig) -> torch.Tensor:
    """Return the squared residual of the target sheath boundary condition.

    The physical condition is

        kappa_parallel * T(L)^(5/2) * dT/ds(L) + alpha * T(L)^(1/2) = 0.
    """
    T_safe = T_t + T_EPSILON
    kappa_eff = config.kappa_parallel * T_safe ** 2.5
    alpha = config.alpha
    residual = kappa_eff * dT_ds_t + alpha * torch.sqrt(T_safe)
    return residual ** 2


