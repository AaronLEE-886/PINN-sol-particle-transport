"""PDE residual evaluation for the 1D SOL heat transport equation."""

import torch

from .constants import T_EPSILON
from .params import SOLConfig


def compute_kappa_eff(T, config: SOLConfig) -> torch.Tensor:
    """Compute the effective conductivity ``kappa_parallel * T^(5/2)``."""
    T_safe = T + T_EPSILON
    return config.kappa_parallel * T_safe ** 2.5


def compute_pde_residual(model, s, config: SOLConfig, source_fn=None,
                         normalized=True):
    """Compute the residual of ``d/ds(kappa*T^(5/2)*dT/ds) + S(s) = 0``.

    The residual is obtained by automatic differentiation:
    1. Predict ``T(s)`` from the model.
    2. Compute ``dT/ds``.
    3. Form the conductive heat flux ``q = -kappa*T^(5/2)*dT/ds``.
    4. Differentiate again to obtain ``dq/ds``.

    When ``normalized=True``, the residual is divided by the characteristic
    scale ``kappa_parallel * T_up^(7/2) / L^2`` so that its magnitude is closer
    to ``O(1)`` during training.
    """
    s = s.clone().detach().requires_grad_(True)
    T = model(s)

    dT_ds = torch.autograd.grad(
        T, s,
        grad_outputs=torch.ones_like(T),
        create_graph=True,
    )[0]

    kappa_eff = compute_kappa_eff(T, config)
    q = -kappa_eff * dT_ds

    dq_ds = torch.autograd.grad(
        q, s,
        grad_outputs=torch.ones_like(q),
        create_graph=True,
    )[0]

    S = source_fn(s) if source_fn is not None else torch.zeros_like(s)
    residual = dq_ds - S

    if normalized:
        scale = config.kappa_parallel * config.T_up ** 3.5 / config.L ** 2
        residual = residual / scale

    aux = {
        "T": T,
        "dT_ds": dT_ds,
        "q": q,
        "dq_ds": dq_ds,
        "kappa_eff": kappa_eff,
    }
    return residual, aux
