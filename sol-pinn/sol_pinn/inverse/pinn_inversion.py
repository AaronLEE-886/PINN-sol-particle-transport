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
        self.gamma_parameter = GammaParameter(gamma_init).to(self.device)

    @property
    def current_gamma(self):
        """Return the diagnosed gamma when available, otherwise the initial guess."""
        return self._diagnosed_gamma if self._diagnosed_gamma is not None else self.gamma_init

    def initialize_gamma_from_least_squares(self, s_obs, T_obs,
                                            gamma_guess=(5.0, 10.0)):
        """Initialize trainable gamma from the FD least-squares baseline."""
        from .least_squares import least_squares_inversion

        if isinstance(s_obs, torch.Tensor):
            s_obs = s_obs.detach().cpu().numpy().reshape(-1)
        else:
            s_obs = np.asarray(s_obs, dtype=float).reshape(-1)
        if isinstance(T_obs, torch.Tensor):
            T_obs = T_obs.detach().cpu().numpy().reshape(-1)
        else:
            T_obs = np.asarray(T_obs, dtype=float).reshape(-1)

        gamma, _ = least_squares_inversion(
            s_obs,
            T_obs,
            self.config,
            gamma_guess=gamma_guess,
        )
        gamma = max(float(gamma), 0.1)

        with torch.no_grad():
            self.gamma_parameter.log_gamma.copy_(
                torch.log(torch.tensor(gamma, dtype=torch.float32, device=self.device))
            )
        self.gamma_init = gamma
        self.config = self.config.clone_with(gamma_sheath=gamma)
        self.trainer.config = self.config
        self._diagnosed_gamma = gamma
        return gamma

    def trainable_sheath_loss(self, log_gamma, n_points=1,
                              window_ratio=0.0) -> torch.Tensor:
        """Return a differentiable sheath loss for a trainable gamma.

        This loss lets inversion optimize ``gamma`` jointly with the
        temperature network instead of estimating it from a post-training
        derivative.  A small target-near window can be used to stabilize cases
        with steep target gradients.
        """
        gamma = torch.exp(log_gamma)
        if n_points < 1:
            raise ValueError("n_points must be at least 1.")
        if window_ratio < 0.0 or window_ratio > 1.0:
            raise ValueError("window_ratio must be in [0, 1].")

        if n_points == 1:
            s = torch.tensor(
                [[self.config.L]],
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )
        else:
            start = self.config.L * (1.0 - window_ratio)
            s = torch.linspace(
                start,
                self.config.L,
                n_points,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1, 1)
            s.requires_grad_(True)

        T = self.model(s)
        dT_ds = torch.autograd.grad(
            T, s, grad_outputs=torch.ones_like(T), create_graph=True,
        )[0]

        T_safe = torch.clamp(T, min=1e-8)
        C = E_CHARGE ** 1.5 * self.config.p0 / (2.0 * np.sqrt(M_I))
        residual = (
            self.config.kappa_parallel * T_safe ** 2.5 * dT_ds
            + gamma * C * torch.sqrt(T_safe)
        )
        scale = C * np.sqrt(self.config.T_up)
        return torch.mean(residual ** 2) / (scale ** 2 + 1e-30)

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

    def diagnose_gamma_from_window(self, window_ratio=0.1, n_points=32,
                                   reduction="median"):
        """Recover gamma from a target-near window instead of one derivative.

        The single-point sheath diagnosis is sensitive to noise in
        ``dT/ds(L)``.  For zero-source test cases the conductive heat flux is
        constant, so the target sheath relation can be evaluated from several
        near-target flux estimates:

            gamma = q / (C * sqrt(T(L)))

        where ``q = -kappa_parallel * T^(5/2) * dT/ds`` and
        ``C = e^(3/2) * p0 / (2 * sqrt(m_i))``.
        """
        if not (0.0 < window_ratio <= 1.0):
            raise ValueError("window_ratio must be in (0, 1].")
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")

        start = self.config.L * (1.0 - window_ratio)
        s_window = torch.linspace(
            start,
            self.config.L,
            n_points,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        s_window.requires_grad_(True)

        with torch.enable_grad():
            T = self.model(s_window)
        dT_ds = torch.autograd.grad(
            T, s_window, grad_outputs=torch.ones_like(T), create_graph=True,
        )[0]

        q = -self.config.kappa_parallel * torch.clamp(T, min=1e-8) ** 2.5 * dT_ds

        s_L = torch.tensor([[self.config.L]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            T_L = torch.clamp(self.model(s_L), min=1e-8)

        C = E_CHARGE ** 1.5 * self.config.p0 / (2.0 * np.sqrt(M_I))
        gamma_values = q / (C * torch.sqrt(T_L))
        gamma_values = gamma_values.detach().cpu().numpy().reshape(-1)
        gamma_values = gamma_values[np.isfinite(gamma_values)]
        if gamma_values.size == 0:
            return 0.1

        if reduction == "mean":
            gamma = float(np.mean(gamma_values))
        elif reduction == "median":
            gamma = float(np.median(gamma_values))
        else:
            raise ValueError("reduction must be 'median' or 'mean'.")

        return max(gamma, 0.1)

    def train_with_inversion(self, s_colloc, s_obs, T_obs, n_adam=5000,
                             n_lbfgs=300, gamma_lr=1e-3,
                             sheath_weight=0.0,
                             diagnosis="window",
                             diagnosis_window_ratio=0.1,
                             data_loss_type="huber",
                             data_delta=None):
        """Train the PINN and then diagnose gamma from the sheath boundary.

        The default choice ``sheath_weight=0`` avoids biasing the temperature field
        with an incorrect initial guess for ``gamma`` during training.
        """
        del gamma_lr

        s_colloc = s_colloc.to(self.device)
        s_obs = s_obs.to(self.device)
        T_obs = T_obs.to(self.device)
        if data_delta is None:
            data_delta = 0.02 * self.config.T_up

        self.trainer.loss_weights["data"] = 1.0
        self.trainer.loss_weights["sheath"] = sheath_weight

        from ..pinn.loss import data_loss, pde_loss, upstream_loss
        from tqdm import tqdm

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.trainer.lr)

        for step in tqdm(range(n_adam), desc="Inversion-Adam"):
            optimizer.zero_grad()
            L_pde = pde_loss(self.model, s_colloc, self.config)
            L_up = upstream_loss(self.model, self.config)
            L_data = data_loss(
                self.model,
                s_obs,
                T_obs,
                loss_type=data_loss_type,
                delta=data_delta,
            )

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
            s_colloc,
            s_obs,
            T_obs,
            n_lbfgs,
            sheath_weight=sheath_weight,
            data_loss_type=data_loss_type,
            data_delta=data_delta,
        )

        if diagnosis == "window":
            self._diagnosed_gamma = self.diagnose_gamma_from_window(
                window_ratio=diagnosis_window_ratio,
            )
        elif diagnosis == "sheath":
            self._diagnosed_gamma = self.diagnose_gamma_from_sheath()
        else:
            raise ValueError("diagnosis must be 'window' or 'sheath'.")
        print(f"  Diagnosed gamma: {self._diagnosed_gamma:.4f} (init={self.gamma_init})")

    def train_with_joint_inversion(self, s_colloc, s_obs, T_obs,
                                   n_adam=5000, n_lbfgs=300,
                                   gamma_lr=5e-3,
                                   joint_sheath_weight=1.0,
                                   data_loss_type="huber",
                                   data_delta=None,
                                   sheath_window_points=8,
                                   sheath_window_ratio=0.05,
                                   gamma_prior=None,
                                   gamma_prior_weight=0.0):
        """Jointly optimize the PINN temperature field and sheath gamma.

        This is intended for noisy or steep-gradient target cases where
        post-training gradient diagnosis is fragile.
        """
        s_colloc = s_colloc.to(self.device)
        s_obs = s_obs.to(self.device)
        T_obs = T_obs.to(self.device)
        if data_delta is None:
            data_delta = 0.02 * self.config.T_up

        from ..pinn.loss import data_loss, pde_loss, upstream_loss
        from tqdm import tqdm

        model_params = list(self.model.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": model_params, "lr": self.trainer.lr},
                {"params": self.gamma_parameter.parameters(), "lr": gamma_lr},
            ],
        )

        self._gamma_history = []
        if gamma_prior is not None:
            log_gamma_prior = torch.log(
                torch.tensor(float(gamma_prior), dtype=torch.float32, device=self.device)
            )
        else:
            log_gamma_prior = None

        for step in tqdm(range(n_adam), desc="Joint-Inversion-Adam"):
            optimizer.zero_grad()
            L_pde = pde_loss(self.model, s_colloc, self.config)
            L_up = upstream_loss(self.model, self.config)
            L_data = data_loss(
                self.model,
                s_obs,
                T_obs,
                loss_type=data_loss_type,
                delta=data_delta,
            )
            L_sheath = self.trainable_sheath_loss(
                self.gamma_parameter.log_gamma,
                n_points=sheath_window_points,
                window_ratio=sheath_window_ratio,
            )
            total = L_pde + L_up + L_data + joint_sheath_weight * L_sheath
            if log_gamma_prior is not None and gamma_prior_weight > 0:
                total = total + gamma_prior_weight * (
                    self.gamma_parameter.log_gamma - log_gamma_prior
                ) ** 2
            total.backward()
            optimizer.step()

            if step % 500 == 0:
                gamma_val = float(self.gamma_parameter.gamma.detach().cpu())
                self._gamma_history.append(gamma_val)
                print(
                    f"  [{step}] gamma={gamma_val:.4f} "
                    f"data={L_data.item():.2e} sheath={L_sheath.item():.2e}"
                )

        self._train_lbfgs_joint(
            s_colloc,
            s_obs,
            T_obs,
            n_lbfgs,
            joint_sheath_weight=joint_sheath_weight,
            data_loss_type=data_loss_type,
            data_delta=data_delta,
            sheath_window_points=sheath_window_points,
            sheath_window_ratio=sheath_window_ratio,
            gamma_prior=gamma_prior,
            gamma_prior_weight=gamma_prior_weight,
        )
        self._diagnosed_gamma = float(self.gamma_parameter.gamma.detach().cpu())
        self._gamma_history.append(self._diagnosed_gamma)
        print(f"  Joint gamma: {self._diagnosed_gamma:.4f} (init={self.gamma_init})")

    def _train_lbfgs_joint(self, s_colloc, s_obs, T_obs, n_iter,
                           joint_sheath_weight=1.0,
                           data_loss_type="huber",
                           data_delta=1.0,
                           sheath_window_points=8,
                           sheath_window_ratio=0.05,
                           gamma_prior=None,
                           gamma_prior_weight=0.0):
        """Refine joint inversion with L-BFGS."""
        if n_iter <= 0:
            return

        from ..pinn.loss import data_loss, pde_loss, upstream_loss

        params = list(self.model.parameters()) + list(self.gamma_parameter.parameters())
        if gamma_prior is not None:
            log_gamma_prior = torch.log(
                torch.tensor(float(gamma_prior), dtype=torch.float32, device=self.device)
            )
        else:
            log_gamma_prior = None

        optimizer = torch.optim.LBFGS(
            params,
            lr=0.1,
            max_iter=n_iter,
            max_eval=n_iter * 2,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            total = (
                pde_loss(self.model, s_colloc, self.config)
                + upstream_loss(self.model, self.config)
                + data_loss(
                    self.model,
                    s_obs,
                    T_obs,
                    loss_type=data_loss_type,
                    delta=data_delta,
                )
                + joint_sheath_weight * self.trainable_sheath_loss(
                    self.gamma_parameter.log_gamma,
                    n_points=sheath_window_points,
                    window_ratio=sheath_window_ratio,
                )
            )
            if log_gamma_prior is not None and gamma_prior_weight > 0:
                total = total + gamma_prior_weight * (
                    self.gamma_parameter.log_gamma - log_gamma_prior
                ) ** 2
            total.backward()
            return total

        optimizer.step(closure)

    def train_with_hybrid_inversion(self, s_colloc, s_obs, T_obs,
                                    gamma_guess=(5.0, 10.0),
                                    gamma_prior_weight=20.0,
                                    **kwargs):
        """Use LS to initialize gamma, then refine with joint PINN inversion."""
        gamma_prior = self.initialize_gamma_from_least_squares(
            s_obs,
            T_obs,
            gamma_guess=gamma_guess,
        )
        kwargs.setdefault("gamma_lr", 1e-3)
        kwargs.setdefault("joint_sheath_weight", 0.1)
        kwargs.setdefault("n_lbfgs", 0)
        self._diagnosed_gamma = None
        self.train_with_joint_inversion(
            s_colloc,
            s_obs,
            T_obs,
            gamma_prior=gamma_prior,
            gamma_prior_weight=gamma_prior_weight,
            **kwargs,
        )

    def _train_lbfgs_with_data(self, s_colloc, s_obs, T_obs, n_iter,
                               sheath_weight=0.0,
                               data_loss_type="mse",
                               data_delta=1.0):
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
                data_loss_type=data_loss_type,
                data_delta=data_delta,
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
