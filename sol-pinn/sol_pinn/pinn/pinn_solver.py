"""High-level interface for training and evaluating SOL PINN models."""

import copy
import numpy as np
import torch

from ..physics.params import SOLConfig, SolverConfig
from ..utils.metrics import physics_consistency_metrics
from ..utils.sampling import target_refined, torch_collocation
from .network import FourierFeatureEncoding, PINN, PiecewisePINN, TransformedPINN
from .trainer import CausalTrainer, Trainer


class PINNSolver:
    """Unified wrapper around model construction, training, and inference."""

    def __init__(self, config: SOLConfig, solver_config: SolverConfig = None,
                 use_fourier=True, fourier_kwargs=None,
                 network_kwargs=None, causal=False, device="cpu",
                 loss_weights=None, model_type="temperature"):
        self.config = config
        self.solver_config = solver_config or SolverConfig()
        self.causal = causal
        self.device = device
        self.model_type = model_type

        fourier_encoding = None
        if use_fourier:
            kwargs = {"mapping_size": 64, "sigma": 1.0}
            if fourier_kwargs:
                kwargs.update(fourier_kwargs)
            fourier_encoding = FourierFeatureEncoding(**kwargs)

        net_kwargs = {
            "layer_sizes": [128, 128, 128, 128, 128],
            "activation": "tanh",
            "output_bias": config.T_up / 2,
        }
        if network_kwargs:
            net_kwargs.update(network_kwargs)

        model_map = {
            "temperature": PINN,
            "transformed_temperature": TransformedPINN,
            "piecewise_temperature": PiecewisePINN,
        }
        model_cls = model_map.get(model_type, PINN)
        self.model = model_cls(
            fourier_encoding=fourier_encoding,
            L=config.L,
            T_up=config.T_up,
            **net_kwargs,
        )
        self.network_kwargs = copy.deepcopy(net_kwargs)
        self.fourier_kwargs = copy.deepcopy(fourier_kwargs) if fourier_kwargs else None

        if causal:
            self.trainer = CausalTrainer(self.model, config, loss_weights=loss_weights, device=device)
        else:
            self.trainer = Trainer(self.model, config, loss_weights=loss_weights, device=device)

    def train(self, s_colloc=None, n_adam=20000, n_lbfgs=500,
              source_fn=None):
        """Train the PINN with default sheath-refined collocation if needed."""
        if s_colloc is None:
            s_np = target_refined(200, 100, self.config.L, boundary_ratio=0.1)
            s_colloc = torch_collocation(s_np)

        self.trainer.train(
            s_colloc,
            n_adam=n_adam,
            n_lbfgs=n_lbfgs,
            source_fn=source_fn,
        )

    @torch.no_grad()
    def predict(self, s_eval):
        """Predict temperature on evaluation points."""
        if isinstance(s_eval, np.ndarray):
            if s_eval.ndim == 1:
                s_eval = s_eval.reshape(-1, 1)
            s_tensor = torch.tensor(s_eval, dtype=torch.float32, device=self.device)
        else:
            s_tensor = s_eval.to(self.device)

        self.model.eval()
        T_pred = self.model(s_tensor)
        return T_pred.cpu().numpy().flatten()

    def compute_error(self, s_ref, T_ref):
        """Compute relative L2 error against a reference solution."""
        T_pred = self.predict(s_ref)
        from ..utils.metrics import relative_L2_error
        return relative_L2_error(T_pred, T_ref)

    def compute_physics_metrics(self, s_eval):
        """Evaluate unified physics-consistency diagnostics on a grid."""
        T_pred = self.predict(s_eval)
        return physics_consistency_metrics(s_eval, T_pred, self.config)

    def clone_for_config(self, new_config: SOLConfig, reset_output_bias=True):
        """Create a same-architecture solver for a new physical configuration.

        The model weights are copied over so the new solver can be fine-tuned
        as part of a curriculum-learning workflow.
        """
        network_kwargs = copy.deepcopy(self.network_kwargs)
        if reset_output_bias and "output_bias" in network_kwargs:
            network_kwargs["output_bias"] = new_config.T_up / 2

        cloned = PINNSolver(
            new_config,
            solver_config=self.solver_config,
            use_fourier=self.fourier_kwargs is not None,
            fourier_kwargs=copy.deepcopy(self.fourier_kwargs) if self.fourier_kwargs else None,
            network_kwargs=network_kwargs,
            causal=self.causal,
            device=self.device,
            loss_weights=copy.deepcopy(self.trainer.loss_weights),
            model_type=self.model_type,
        )
        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=False)
        return cloned
