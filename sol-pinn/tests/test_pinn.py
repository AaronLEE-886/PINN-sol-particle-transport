"""PINN model tests: network, losses, and causal training helpers."""

import numpy as np
import torch

from sol_pinn.physics.params import SOLConfig
from sol_pinn.pinn.loss import combined_loss, data_loss, pde_loss, sheath_loss, upstream_loss
from sol_pinn.pinn.network import FourierFeatureEncoding, PINN, PiecewisePINN, TransformedPINN
from sol_pinn.pinn.pinn_solver import PINNSolver
from sol_pinn.pinn.trainer import CausalTrainer
from sol_pinn.utils.sampling import torch_collocation


class TestNetwork:
    def test_pinn_output_shape(self):
        model = PINN()
        s = torch.randn(100, 1)
        T = model(s)
        assert T.shape == (100, 1)
        assert (T > 0).all()

    def test_pinn_with_fourier(self):
        fourier = FourierFeatureEncoding(mapping_size=32, sigma=1.0)
        model = PINN(fourier_encoding=fourier)
        s = torch.randn(50, 1)
        T = model(s)
        assert T.shape == (50, 1)
        assert torch.isfinite(T).all()

    def test_fourier_encoding_shape(self):
        fourier = FourierFeatureEncoding(mapping_size=64)
        s = torch.randn(10, 1)
        encoded = fourier(s)
        assert encoded.shape == (10, 128)

    def test_pinn_differentiable(self):
        model = PINN()
        s = torch.tensor([[0.5]], requires_grad=True)
        T = model(s)
        dT_ds = torch.autograd.grad(T, s, torch.ones_like(T), create_graph=True)[0]
        assert dT_ds.shape == (1, 1)
        assert torch.isfinite(dT_ds).all()

    def test_transformed_pinn_output_shape(self):
        model = TransformedPINN(T_up=50.0, L=20.0)
        s = torch.randn(100, 1)
        T = model(s)
        assert T.shape == (100, 1)
        assert (T > 0).all()
        assert torch.isfinite(T).all()

    def test_transformed_pinn_differentiable(self):
        model = TransformedPINN(T_up=50.0, L=20.0)
        s = torch.tensor([[0.5]], requires_grad=True)
        T = model(s)
        dT_ds = torch.autograd.grad(T, s, torch.ones_like(T), create_graph=True)[0]
        assert dT_ds.shape == (1, 1)
        assert torch.isfinite(dT_ds).all()

    def test_pinn_solver_transformed_model_type(self):
        config = SOLConfig(T_up=50.0, L=20.0)
        solver = PINNSolver(config, use_fourier=False, model_type="transformed_temperature")
        s_eval = np.linspace(0, config.L, 16)
        T_pred = solver.predict(s_eval)
        assert T_pred.shape == (16,)
        assert np.all(T_pred > 0)

    def test_piecewise_pinn_output_shape(self):
        model = PiecewisePINN(T_up=50.0, L=20.0, layer_sizes=[32, 32], target_layer_sizes=[32, 32])
        s = torch.linspace(0.0, 20.0, 64).reshape(-1, 1)
        T = model(s)
        assert T.shape == (64, 1)
        assert torch.isfinite(T).all()
        assert (T > 0).all()

    def test_piecewise_pinn_differentiable(self):
        model = PiecewisePINN(T_up=50.0, L=20.0, layer_sizes=[32, 32], target_layer_sizes=[32, 32])
        s = torch.tensor([[18.0]], requires_grad=True)
        T = model(s)
        dT_ds = torch.autograd.grad(T, s, torch.ones_like(T), create_graph=True)[0]
        assert dT_ds.shape == (1, 1)
        assert torch.isfinite(dT_ds).all()

    def test_pinn_solver_piecewise_model_type(self):
        config = SOLConfig(T_up=50.0, L=20.0)
        solver = PINNSolver(
            config,
            use_fourier=False,
            model_type="piecewise_temperature",
            network_kwargs={"layer_sizes": [16, 16], "target_layer_sizes": [16, 16]},
        )
        s_eval = np.linspace(0, config.L, 16)
        T_pred = solver.predict(s_eval)
        assert T_pred.shape == (16,)
        assert np.all(T_pred > 0)

    def test_pinn_solver_clone_for_config(self):
        config = SOLConfig(T_up=80.0, L=20.0)
        solver = PINNSolver(
            config,
            use_fourier=False,
            network_kwargs={"layer_sizes": [16, 16], "output_bias": 40.0},
        )
        new_config = config.clone_with(T_up=60.0)
        cloned = solver.clone_for_config(new_config)
        s_eval = np.linspace(0, new_config.L, 8)
        T_pred = cloned.predict(s_eval)
        assert cloned.config.T_up == 60.0
        assert cloned.model_type == solver.model_type
        assert T_pred.shape == (8,)
        assert np.all(T_pred > 0)


class TestLoss:
    def test_pde_loss_finite(self):
        config = SOLConfig(T_up=100.0, L=10.0)
        model = PINN()
        s = torch_collocation(np.linspace(0, config.L, 100))
        loss = pde_loss(model, s, config)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_upstream_loss(self):
        config = SOLConfig(T_up=100.0, L=10.0)
        model = PINN(output_bias=100.0)
        loss = upstream_loss(model, config)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_sheath_loss(self):
        config = SOLConfig(T_up=100.0, L=10.0)
        model = PINN()
        loss = sheath_loss(model, config)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_combined_loss(self):
        config = SOLConfig(T_up=100.0, L=10.0)
        model = PINN()
        s = torch_collocation(np.linspace(0, config.L, 50))
        losses = combined_loss(model, s, config)
        for key in ["total", "pde", "up", "sheath"]:
            assert key in losses
            assert torch.isfinite(losses[key])

    def test_loss_weights_effect(self):
        config = SOLConfig(T_up=100.0, L=10.0)
        model = PINN()
        s = torch_collocation(np.linspace(0, config.L, 50))

        losses_high = combined_loss(
            model,
            s,
            config,
            loss_weights={"pde": 1.0, "up": 1e6, "sheath": 1.0},
        )
        losses_low = combined_loss(
            model,
            s,
            config,
            loss_weights={"pde": 1.0, "up": 1e-6, "sheath": 1.0},
        )

        assert losses_high["total"].item() != losses_low["total"].item()

    def test_huber_data_loss_reduces_outlier_penalty(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, s):
                return torch.zeros_like(s)

        s_data = torch.tensor([[0.0], [1.0]])
        T_data = torch.tensor([[0.1], [10.0]])

        mse = data_loss(ConstantModel(), s_data, T_data, loss_type="mse")
        huber = data_loss(
            ConstantModel(),
            s_data,
            T_data,
            loss_type="huber",
            delta=1.0,
        )

        assert huber < mse


class TestCausalTrainer:
    def test_interval_slice_covers_expected_ranges(self):
        trainer = CausalTrainer(PINN(), SOLConfig(), n_intervals=4)
        n_points = 10
        n_per_interval = max(1, n_points // trainer.n_intervals)

        slices = [
            trainer._interval_slice(k, n_points, n_per_interval)
            for k in range(trainer.n_intervals)
        ]

        assert slices == [(0, 2), (2, 4), (4, 6), (6, 8)]

    def test_causal_activation_progresses(self):
        torch.manual_seed(0)
        config = SOLConfig(T_up=100.0, L=10.0)
        trainer = CausalTrainer(
            PINN(),
            config,
            n_intervals=3,
            interval_activation_steps=1,
        )
        s_colloc = torch_collocation(np.linspace(0, config.L, 9))

        trainer.train_adam(s_colloc, n_iter=2, log_every=10)

        assert trainer.active_intervals == 3
