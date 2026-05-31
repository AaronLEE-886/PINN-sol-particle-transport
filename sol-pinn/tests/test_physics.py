"""Physics layer tests: PDE residual, BC residual."""

import numpy as np
import torch

from sol_pinn.physics.constants import T_EPSILON
from sol_pinn.physics.params import SOLConfig
from sol_pinn.physics.pde import compute_pde_residual, compute_kappa_eff
from sol_pinn.physics.boundary import upstream_bc_residual, sheath_bc_residual


class TestPARAMS:
    def test_alpha_calculation(self):
        """Test sheath alpha coefficient calculation."""
        config = SOLConfig(p0=1e21, gamma_sheath=7.0)
        assert config.alpha > 0
        assert np.isfinite(config.alpha)

    def test_clone_with(self):
        """Test config cloning."""
        config = SOLConfig(T_up=100.0)
        cloned = config.clone_with(T_up=150.0)
        assert cloned.T_up == 150.0
        assert cloned.L == config.L  # unchanged


class TestPDE:
    def test_kappa_eff(self):
        """Test effective conductivity computation."""
        config = SOLConfig(kappa_parallel=2000.0)
        T = torch.tensor([[100.0]])
        kappa = compute_kappa_eff(T, config)
        expected = 2000.0 * 100.0 ** 2.5
        assert torch.isclose(kappa, torch.tensor(expected), rtol=1e-4)

    def test_kappa_eff_with_epsilon(self):
        """Test kappa_eff at near-zero temperature (should not crash)."""
        config = SOLConfig()
        T = torch.tensor([[T_EPSILON]])
        kappa = compute_kappa_eff(T, config)
        assert torch.isfinite(kappa).all()


class TestBoundary:
    def test_upstream_bc(self):
        """Test upstream BC residual."""
        T_pred = torch.tensor([[100.0]])
        residual = upstream_bc_residual(T_pred, 100.0)
        assert torch.isclose(residual, torch.tensor(0.0), atol=1e-6)

    def test_upstream_bc_mismatch(self):
        """Test upstream BC residual with mismatch."""
        T_pred = torch.tensor([[90.0]])
        residual = upstream_bc_residual(T_pred, 100.0)
        assert torch.isclose(residual, torch.tensor(100.0))

    def test_sheath_bc(self):
        """Test sheath BC residual is finite."""
        config = SOLConfig()
        T_t = torch.tensor([[10.0]])
        dT_ds_t = torch.tensor([[-500.0]])
        residual = sheath_bc_residual(T_t, dT_ds_t, config)
        assert torch.isfinite(residual)
        assert residual.item() >= 0
