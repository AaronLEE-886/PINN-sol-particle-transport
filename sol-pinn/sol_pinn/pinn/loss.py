"""Loss functions for PDE residuals, boundary conditions, and data fitting."""

import torch

from ..physics.boundary import sheath_bc_residual, upstream_bc_residual
from ..physics.params import SOLConfig
from ..physics.pde import compute_pde_residual


def pde_loss(model, s_colloc, config: SOLConfig, source_fn=None) -> torch.Tensor:
    """Return the mean-squared PDE residual over collocation points."""
    residual, _ = compute_pde_residual(model, s_colloc, config, source_fn)
    return torch.mean(residual ** 2)


def upstream_loss(model, config: SOLConfig) -> torch.Tensor:
    """Return the normalized upstream Dirichlet loss."""
    s_up = torch.zeros((1, 1), dtype=torch.float32)
    T_pred = model(s_up)
    return upstream_bc_residual(T_pred, config.T_up) / config.T_up ** 2


def sheath_loss(model, config: SOLConfig) -> torch.Tensor:
    """Return the normalized sheath boundary loss."""
    s_L = torch.tensor([[config.L]], dtype=torch.float32, requires_grad=True)
    T_t = model(s_L)

    dT_ds_t = torch.autograd.grad(
        T_t, s_L,
        grad_outputs=torch.ones_like(T_t),
        create_graph=True,
    )[0]

    q_char = config.alpha * (config.T_up ** 0.5)
    return sheath_bc_residual(T_t, dT_ds_t, config) / q_char ** 2


def data_loss(model, s_data, T_data, loss_type="mse", delta=1.0) -> torch.Tensor:
    """Return data misfit on observed temperature samples."""
    T_pred = model(s_data)
    diff = T_pred - T_data
    if loss_type == "mse":
        return torch.mean(diff ** 2)
    if loss_type == "huber":
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=abs_diff.device))
        linear = abs_diff - quadratic
        return torch.mean(0.5 * quadratic ** 2 + delta * linear)
    raise ValueError("loss_type must be 'mse' or 'huber'.")


def combined_loss(model, s_colloc, config: SOLConfig, loss_weights=None,
                  source_fn=None, s_data=None, T_data=None,
                  data_loss_type="mse", data_delta=1.0) -> dict:
    """Return all active loss terms and their weighted sum."""
    if loss_weights is None:
        loss_weights = {"pde": 1.0, "up": 1.0, "sheath": 1.0}

    L_pde = pde_loss(model, s_colloc, config, source_fn)
    L_up = upstream_loss(model, config)
    L_sheath = sheath_loss(model, config)

    losses = {
        "pde": L_pde,
        "up": L_up,
        "sheath": L_sheath,
    }

    total = (
        loss_weights.get("pde", 1.0) * L_pde
        + loss_weights.get("up", 1.0) * L_up
        + loss_weights.get("sheath", 1.0) * L_sheath
    )

    if s_data is not None and T_data is not None:
        L_data = data_loss(model, s_data, T_data, data_loss_type, data_delta)
        losses["data"] = L_data
        total += loss_weights.get("data", 1.0) * L_data

    losses["total"] = total
    return losses
