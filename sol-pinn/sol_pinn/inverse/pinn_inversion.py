"""Sheath-parameter inversion based on a trained PINN temperature field."""

import numpy as np
import torch
import torch.nn as nn

from ..physics.constants import E_CHARGE, M_I
from ..physics.params import SOLConfig
from ..pinn.loss import combined_loss
from ..pinn.pinn_solver import PINNSolver
from ..utils.metrics import physics_consistency_metrics, relative_L2_error


class GammaParameter(nn.Module):
    """Legacy trainable gamma parameter retained for compatibility."""

    def __init__(self, gamma_init=7.0):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma_init)))

    @property
    def gamma(self):
        return torch.exp(self.log_gamma)

    def forward(self):
        return self.gamma


class InversePINNSolver(PINNSolver):
    """PINN-based inversion using post-training sheath diagnosis.

    Instead of jointly optimizing the neural network and ``gamma``, this solver:
    1. trains a PINN for ``T(s)``,
    2. evaluates ``T(L)`` and ``dT/ds(L)``,
    3. recovers ``gamma`` from the analytic sheath condition.
    """

    def __init__(self, config: SOLConfig, gamma_init=5.0, **kwargs):
        config_with_gamma = config.clone_with(gamma_sheath=gamma_init)
        super().__init__(config_with_gamma, **kwargs)
        self.gamma_init = gamma_init
        self._diagnosed_gamma = None

    @property
    def current_gamma(self):
        """Return the diagnosed gamma when available, otherwise the initial guess."""
        return self._diagnosed_gamma if self._diagnosed_gamma is not None else self.gamma_init

    def diagnose_gamma_from_sheath(self):
        """Recover gamma from the target sheath boundary condition.

        The sheath boundary condition is:

            -kappa_parallel * T(L)^(5/2) * dT/ds(L) = alpha * T(L)^(1/2)

        where alpha = gamma * e^(3/2) * p0 / (2 * sqrt(m_i)).
        Defining ``C = e^(3/2) * p0 / (2 * sqrt(m_i))``, we have
        alpha = gamma * C and therefore:

            gamma = -kappa_parallel * T(L)^2 * dT/ds(L) / C.
        """
        s_L = torch.tensor(
            [[self.config.L]],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        with torch.enable_grad():
            T_L = self.model(s_L)
        dT_ds_L = torch.autograd.grad(
            T_L, s_L, grad_outputs=torch.ones_like(T_L), create_graph=True,
        )[0]

        # C = e^(3/2) * p0 / (2 * sqrt(m_i))  — factor 2 comes from alpha formula
        C = E_CHARGE ** 1.5 * self.config.p0 / (2.0 * np.sqrt(M_I))
        gamma = -self.config.kappa_parallel * T_L.item() ** 2 * dT_ds_L.item() / C

        return max(gamma, 0.1)

    def train_with_inversion(self, s_colloc, s_obs, T_obs, n_adam=5000,
                             n_lbfgs=300, gamma_lr=1e-3,
                             sheath_weight=0.0):
        """Train the PINN and then diagnose gamma from the sheath boundary.

        The default choice ``sheath_weight=0`` avoids biasing the temperature field
        with an incorrect initial guess for ``gamma`` during training.
        """
        del gamma_lr

        s_colloc = s_colloc.to(self.device)
        s_obs = s_obs.to(self.device)
        T_obs = T_obs.to(self.device)

        self.trainer.loss_weights["data"] = 1.0
        self.trainer.loss_weights["sheath"] = sheath_weight

        from ..pinn.loss import data_loss, pde_loss, upstream_loss
        from tqdm import tqdm

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.trainer.lr)

        for step in tqdm(range(n_adam), desc="Inversion-Adam"):
            optimizer.zero_grad()
            L_pde = pde_loss(self.model, s_colloc, self.config)
            L_up = upstream_loss(self.model, self.config)
            L_data = data_loss(self.model, s_obs, T_obs)

            w = self.trainer.loss_weights
            total = (
                w.get("pde", 1.0) * L_pde
                + w.get("up", 1.0) * L_up
                + w.get("data", 1.0) * L_data
            )

            if sheath_weight > 0:
                from ..pinn.loss import sheath_loss
                total = total + sheath_weight * sheath_loss(self.model, self.config)

            total.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"  [{step}] data={L_data.item():.2e} pde={L_pde.item():.2e}")

        self._train_lbfgs_with_data(
            s_colloc, s_obs, T_obs, n_lbfgs, sheath_weight=sheath_weight,
        )

        self._diagnosed_gamma = self.diagnose_gamma_from_sheath()
        print(f"  Diagnosed gamma: {self._diagnosed_gamma:.4f} (init={self.gamma_init})")

    def _train_lbfgs_with_data(self, s_colloc, s_obs, T_obs, n_iter,
                               sheath_weight=0.0):
        """Refine the inversion solution with L-BFGS."""
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=n_iter,
            max_eval=n_iter * 2,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        old_weights = dict(self.trainer.loss_weights)
        self.trainer.loss_weights["sheath"] = sheath_weight

        def closure():
            optimizer.zero_grad()
            losses = combined_loss(
                self.model,
                s_colloc,
                self.config,
                self.trainer.loss_weights,
                None,
                s_data=s_obs,
                T_data=T_obs,
            )
            losses["total"].backward()
            return losses["total"]

        optimizer.step(closure)
        self.trainer.loss_weights.update(old_weights)

    def get_gamma_history(self):
        return getattr(self, "_gamma_history", [])

    def compute_inversion_metrics(self, s_eval, T_ref=None):
        """Return physics-consistency and optional reference-error metrics."""
        T_pred = self.predict(s_eval)
        metrics = physics_consistency_metrics(s_eval, T_pred, self.config)
        metrics["gamma_diagnosed"] = float(self.current_gamma)
        if T_ref is not None:
            metrics["temperature_rel_l2"] = float(relative_L2_error(T_pred, T_ref))
        return metrics


def run_inversion_test(gamma_true=7.0, n_obs=20, noise_level=0.0, n_adam=3000,
                       n_lbfgs=200, gamma_lr=1e-3, sheath_weight=0.0,
                       config=None):
    """Run a synthetic inversion test and return recovered gamma and error."""
    del gamma_lr

    from fd_reference.numerical.fd_solver import FDSolver
    from ..physics.params import SOLConfig, SolverConfig
    from ..utils.sampling import torch_collocation, uniform_grid

    if config is None:
        config = SOLConfig(T_up=80.0, gamma_sheath=gamma_true)
    else:
        config = config.clone_with(gamma_sheath=gamma_true)
    sc = SolverConfig(n_points=2000, max_iter=2000, tol=1e-10)
    fd_result = FDSolver(config, sc).solve()

    s_obs_np = uniform_grid(n_obs, config.L)
    T_obs_np = np.interp(s_obs_np, fd_result["s"], fd_result["T"])
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * config.T_up, size=n_obs)
        T_obs_np = np.maximum(T_obs_np + noise, 1e-6)

    inv_solver = InversePINNSolver(config, gamma_init=5.0, use_fourier=True)
    inv_solver.trainer.loss_weights["sheath"] = sheath_weight

    from ..utils.sampling import target_refined
    s_colloc_np = target_refined(200, 100, config.L, boundary_ratio=0.1)
    s_colloc = torch_collocation(s_colloc_np)
    s_obs = torch_collocation(s_obs_np)
    T_obs = torch.tensor(T_obs_np, dtype=torch.float32).reshape(-1, 1)

    inv_solver.train_with_inversion(
        s_colloc,
        s_obs,
        T_obs,
        n_adam=n_adam,
        n_lbfgs=n_lbfgs,
        sheath_weight=sheath_weight,
    )

    gamma_recovered = inv_solver.current_gamma
    gamma_error = abs(gamma_recovered - gamma_true) / gamma_true
    return gamma_recovered, gamma_error
