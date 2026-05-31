"""Inverse problem tests: gamma diagnostics and recovery.

Tests the diagnosis-based inversion approach:
1. diagnose_gamma_from_sheath() formula correctness
2. Full inversion pipeline with clean data
3. Noise robustness
"""

import torch
import pytest
import numpy as np
from sol_pinn.physics.params import SOLConfig
from sol_pinn.physics.constants import E_CHARGE, M_I
from sol_pinn.pinn.network import PINN
from sol_pinn.inverse.pinn_inversion import InversePINNSolver
from sol_pinn.inverse.least_squares import solve_forward_for_gamma
from sol_pinn.utils.sampling import torch_collocation


class TestDiagnoseGamma:
    """Test the sheath BC gamma diagnostic formula."""

    def _make_mock_model(self, T_L_val, dT_ds_L_val):
        """Create a mock model with known T(L) and dT/ds(L)."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.T_L = torch.nn.Parameter(torch.tensor([T_L_val]))
                self.dT_ds = torch.nn.Parameter(torch.tensor([dT_ds_L_val]))

            def forward(self, s):
                """Linear profile: T(s) = T_up + (T_L - T_up) * s/L,
                giving dT/ds = (T_L - T_up)/L."""
                return self.T_L + self.dT_ds * (s - 10.0)

            def __call__(self, s):
                return self.forward(s)

        return MockModel()

    def _compute_gamma_formula(self, T_L, dT_ds_L, config):
        """Direct computation of gamma from sheath BC formula.

        alpha = gamma * e^(3/2) * p0 / (2 * sqrt(m_i))
        so gamma = -kappa * T^2 * dT/ds / (e^1.5 * p0 / (2 * sqrt(m_i)))
        """
        C = E_CHARGE ** 1.5 * config.p0 / (2.0 * np.sqrt(M_I))
        return -config.kappa_parallel * T_L ** 2 * dT_ds_L / C

    def test_diagnose_formula_known_values(self):
        """Test that the diagnostic formula gives correct gamma for known T(L), dT/ds(L).

        From FD reference (conduction-limited): T(L), dT/ds(L) should give gamma ≈ 7.0
        """
        config = SOLConfig(T_up=80.0, gamma_sheath=7.0)  # conduction-limited defaults
        # Use the analytical solution to get T(L) and dT/ds(L)
        from fd_reference.numerical.fd_solver import FDSolver
        from sol_pinn.physics.params import SolverConfig
        fd = FDSolver(config, SolverConfig(n_points=2000, max_iter=2000, tol=1e-10))
        result = fd.solve()
        T_L = float(result["T"][-1])
        dT_ds_L = float(result["dT_ds"][-1])

        gamma_computed = self._compute_gamma_formula(T_L, dT_ds_L, config)
        assert abs(gamma_computed - 7.0) < 0.1, \
            f"Expected gamma≈7.0, got {gamma_computed:.4f} (T_L={T_L:.2f}, dT_ds={dT_ds_L:.4f})"

    def test_diagnose_gamma_linear(self):
        """Test diagnose_gamma_from_sheath with a linear model."""
        config = SOLConfig(T_up=80.0)  # conduction-limited defaults for rest
        model = PINN(output_bias=50.0)
        solver = InversePINNSolver(config, gamma_init=5.0, use_fourier=False)
        solver.model = model

        # For a fresh untrained model, diagnosis should return something finite
        gamma = solver.diagnose_gamma_from_sheath()
        assert np.isfinite(gamma)
        assert gamma > 0

    def test_window_diagnosis_recovers_exact_gamma(self):
        """Window diagnosis should recover gamma from an exact S=0 profile."""
        config = SOLConfig(T_up=80.0, gamma_sheath=7.0)
        from fd_reference.numerical.fd_solver import FDSolver
        from sol_pinn.physics.params import SolverConfig

        fd = FDSolver(config, SolverConfig(n_points=2000, max_iter=2000, tol=1e-10))
        result = fd.solve()
        T_L = float(result["T"][-1])
        q = config.alpha * np.sqrt(T_L)

        class ExactProfile(torch.nn.Module):
            def forward(self, s):
                T_power = config.T_up ** 3.5 - 3.5 * q / config.kappa_parallel * s
                return torch.pow(T_power, 2.0 / 7.0)

        solver = InversePINNSolver(config, gamma_init=5.0, use_fourier=False)
        solver.model = ExactProfile()

        gamma = solver.diagnose_gamma_from_window(window_ratio=0.1, n_points=32)

        assert abs(gamma - 7.0) < 0.1

    def test_trainable_sheath_loss_prefers_true_gamma(self):
        """The differentiable sheath loss should be minimized near true gamma."""
        config = SOLConfig(T_up=80.0, gamma_sheath=7.0)
        from fd_reference.numerical.fd_solver import FDSolver
        from sol_pinn.physics.params import SolverConfig

        fd = FDSolver(config, SolverConfig(n_points=2000, max_iter=2000, tol=1e-10))
        result = fd.solve()
        T_L = float(result["T"][-1])
        q = config.alpha * np.sqrt(T_L)

        class ExactProfile(torch.nn.Module):
            def forward(self, s):
                T_power = config.T_up ** 3.5 - 3.5 * q / config.kappa_parallel * s
                return torch.pow(T_power, 2.0 / 7.0)

        solver = InversePINNSolver(config, gamma_init=5.0, use_fourier=False)
        solver.model = ExactProfile()

        true_loss = solver.trainable_sheath_loss(torch.log(torch.tensor(7.0)))
        wrong_loss = solver.trainable_sheath_loss(torch.log(torch.tensor(5.0)))

        assert true_loss < wrong_loss


class TestInversionPipeline:
    """Test the full inversion pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = SOLConfig(T_up=80.0)  # conduction-limited defaults for rest

    def _run_inversion_quick(self, n_obs=10, n_adam=2000, noise_level=0.0):
        """Run a quick inversion test for unit testing."""
        from sol_pinn.physics.params import SolverConfig
        from fd_reference.numerical.fd_solver import FDSolver
        from sol_pinn.utils.sampling import uniform_grid, torch_collocation

        # Reference solution
        fd = FDSolver(self.config, SolverConfig(
            n_points=2000, max_iter=2000, tol=1e-10,
        ))
        fd_result = fd.solve()

        # Observations
        s_obs_np = uniform_grid(n_obs, self.config.L)
        T_obs_np = np.interp(s_obs_np, fd_result["s"], fd_result["T"])
        if noise_level > 0:
            T_obs_np += np.random.normal(0, noise_level * self.config.T_up, size=n_obs)
            T_obs_np = np.maximum(T_obs_np, 1e-6)

        # Inverse solver
        solver = InversePINNSolver(self.config, gamma_init=5.0, use_fourier=True)
        solver.trainer.loss_weights["sheath"] = 0.0

        s_colloc = torch_collocation(np.linspace(0, self.config.L, 200))
        s_obs = torch_collocation(s_obs_np)
        T_obs = torch.tensor(T_obs_np.reshape(-1, 1), dtype=torch.float32)

        solver.train_with_inversion(
            s_colloc, s_obs, T_obs, n_adam=n_adam, n_lbfgs=100,
        )

        return solver.current_gamma

    @pytest.mark.slow
    def test_inversion_clean_10obs(self):
        """PINN recovers gamma from clean data with 10 observations."""
        gamma = self._run_inversion_quick(n_obs=10, n_adam=3000)
        error = abs(gamma - 7.0) / 7.0 * 100
        assert error < 1.0, \
            f"Gamma error too high: {gamma:.4f} ({error:.2f}%)"

    @pytest.mark.slow
    def test_inversion_clean_20obs(self):
        """PINN recovers gamma from clean data with 20 observations."""
        gamma = self._run_inversion_quick(n_obs=20, n_adam=3000)
        error = abs(gamma - 7.0) / 7.0 * 100
        assert error < 0.5, \
            f"Gamma error too high: {gamma:.4f} ({error:.2f}%)"

    @pytest.mark.slow
    def test_inversion_with_noise(self):
        """PINN inversion handles mild noise gracefully."""
        np.random.seed(42)
        gamma = self._run_inversion_quick(
            n_obs=30, n_adam=3000, noise_level=0.01,
        )
        error = abs(gamma - 7.0) / 7.0 * 100
        # With 1% noise, expect < 10% error
        assert error < 10.0, \
            f"Gamma error too high with noise: {gamma:.4f} ({error:.2f}%)"


class TestLeastSquaresBaseline:
    def test_forward_solver_imports_fd_reference(self):
        """Least-squares baseline should use the packaged FD reference solver."""
        config = SOLConfig(T_up=80.0)
        s_eval = np.linspace(0.0, config.L, 8)

        T = solve_forward_for_gamma(7.0, config, s_eval)

        assert T.shape == s_eval.shape
        assert np.all(np.isfinite(T))

    def test_least_squares_initialization_sets_gamma_near_truth(self):
        config = SOLConfig(T_up=80.0, gamma_sheath=7.0)
        from fd_reference.numerical.fd_solver import FDSolver
        from sol_pinn.physics.params import SolverConfig
        from sol_pinn.utils.sampling import uniform_grid

        fd = FDSolver(config, SolverConfig(n_points=1000, max_iter=2000, tol=1e-10))
        result = fd.solve()
        s_obs = uniform_grid(10, config.L)
        T_obs = np.interp(s_obs, result["s"], result["T"])

        solver = InversePINNSolver(config, gamma_init=5.0, use_fourier=False)
        gamma = solver.initialize_gamma_from_least_squares(s_obs, T_obs)

        assert abs(gamma - 7.0) < 0.1
        assert abs(solver.current_gamma - gamma) < 1e-6
