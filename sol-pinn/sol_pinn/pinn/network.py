"""Neural network architectures used by the SOL PINN workflows."""

import numpy as np
import torch
import torch.nn as nn


class FourierFeatureEncoding(nn.Module):
    """Random Fourier feature mapping for low-dimensional coordinates."""

    def __init__(self, input_dim: int = 1, mapping_size: int = 64,
                 sigma: float = 1.0, learnable: bool = False):
        super().__init__()
        self.mapping_size = mapping_size
        self.sigma = sigma

        B = torch.randn(input_dim, mapping_size) * sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x):
        """Map inputs to concatenated sine and cosine features."""
        x_proj = 2.0 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def extra_repr(self):
        return f"mapping_size={self.mapping_size}, sigma={self.sigma}"


class PINN(nn.Module):
    """Feed-forward PINN for a single SOL temperature profile ``T(s)``."""

    def __init__(self, fourier_encoding=None, layer_sizes=None,
                 activation="tanh", output_bias=50.0, L=10.0, T_up=100.0):
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [128, 128, 128, 128, 128]

        self.fourier_encoding = fourier_encoding
        self.L = L
        self.T_up = T_up

        in_dim = (2 * fourier_encoding.mapping_size) if fourier_encoding else 1

        layers = []
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sin":
                layers.append(torch.sin)
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

        if output_bias is not None:
            nn.init.constant_(self.output_layer.bias, output_bias / T_up)
            nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)

    def forward(self, s):
        """Evaluate the temperature profile in physical units of eV."""
        if s.dim() == 1:
            s = s.unsqueeze(-1)

        s_norm = s / self.L
        x = self.fourier_encoding(s_norm) if self.fourier_encoding is not None else s_norm
        x = self.hidden(x)
        T_norm = self.output_layer(x)
        T = torch.nn.functional.softplus(T_norm) * self.T_up
        return T


class TransformedPINN(nn.Module):
    """PINN variant that predicts a transformed state ``u = T^(7/2)``.

    This representation is useful for hard cases where directly predicting
    ``T`` leads to severe stiffness near the target boundary.
    """

    def __init__(self, fourier_encoding=None, layer_sizes=None,
                 activation="tanh", output_bias=None, L=10.0, T_up=100.0):
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [128, 128, 128, 128, 128]

        self.fourier_encoding = fourier_encoding
        self.L = L
        self.T_up = T_up
        self.u_up = T_up ** 3.5

        in_dim = (2 * fourier_encoding.mapping_size) if fourier_encoding else 1

        layers = []
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sin":
                layers.append(torch.sin)
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())
            in_dim = out_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

        bias_value = self.u_up if output_bias is None else output_bias
        nn.init.constant_(self.output_layer.bias, bias_value / self.u_up)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)

    def forward(self, s):
        """Evaluate temperature by predicting ``u=T^(7/2)`` then inverting."""
        if s.dim() == 1:
            s = s.unsqueeze(-1)

        s_norm = s / self.L
        x = self.fourier_encoding(s_norm) if self.fourier_encoding is not None else s_norm
        x = self.hidden(x)
        u_norm = self.output_layer(x)
        u = torch.nn.functional.softplus(u_norm) * self.u_up
        T = torch.pow(u + 1e-12, 2.0 / 7.0)
        return T


class PiecewisePINN(nn.Module):
    """Two-branch PINN with smooth blending near the target boundary.

    The global branch models the whole profile, while the target branch is
    given extra capacity to correct the sharp boundary-layer structure.
    """

    def __init__(self, fourier_encoding=None, layer_sizes=None,
                 target_layer_sizes=None, activation="tanh",
                 output_bias=50.0, L=10.0, T_up=100.0,
                 blend_center=0.85, blend_sharpness=18.0):
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [128, 128, 128, 128, 128]
        if target_layer_sizes is None:
            target_layer_sizes = layer_sizes

        self.fourier_encoding = fourier_encoding
        self.L = L
        self.T_up = T_up
        self.blend_center = blend_center
        self.blend_sharpness = blend_sharpness

        in_dim = (2 * fourier_encoding.mapping_size) if fourier_encoding else 1

        def build_stack(sizes):
            layers = []
            cur_dim = in_dim
            for out_dim in sizes:
                layers.append(nn.Linear(cur_dim, out_dim))
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sin":
                    layers.append(torch.sin)
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.Tanh())
                cur_dim = out_dim
            return nn.Sequential(*layers), cur_dim

        self.global_hidden, global_dim = build_stack(layer_sizes)
        self.target_hidden, target_dim = build_stack(target_layer_sizes)
        self.global_output = nn.Linear(global_dim, 1)
        self.target_output = nn.Linear(target_dim, 1)

        if output_bias is not None:
            bias_value = output_bias / T_up
            nn.init.constant_(self.global_output.bias, bias_value)
            nn.init.constant_(self.target_output.bias, bias_value)
            nn.init.xavier_normal_(self.global_output.weight, gain=0.1)
            nn.init.xavier_normal_(self.target_output.weight, gain=0.1)

    def _encode(self, s):
        s_norm = s / self.L
        return self.fourier_encoding(s_norm) if self.fourier_encoding is not None else s_norm

    def forward(self, s):
        """Blend global and target-focused predictions with a smooth gate."""
        if s.dim() == 1:
            s = s.unsqueeze(-1)

        s_norm = s / self.L
        x = self._encode(s)

        global_norm = self.global_output(self.global_hidden(x))
        target_norm = self.target_output(self.target_hidden(x))

        w = torch.sigmoid(self.blend_sharpness * (s_norm - self.blend_center))
        blended_norm = (1.0 - w) * global_norm + w * target_norm
        T = torch.nn.functional.softplus(blended_norm) * self.T_up
        return T


class ParameterizedPINN(nn.Module):
    """Parameterized PINN that learns the family ``T(s; T_up)``."""

    def __init__(self, fourier_kwargs=None, layer_sizes=None,
                 activation="tanh", L=10.0, T_up_ref=100.0):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [128, 128, 128, 128, 128]

        input_dim = 2

        if fourier_kwargs:
            self.fourier = FourierFeatureEncoding(input_dim=2, **fourier_kwargs)
            in_features = 2 * fourier_kwargs.get("mapping_size", 64)
        else:
            self.fourier = None
            in_features = input_dim

        layers = []
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_features, out_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sin":
                layers.append(torch.sin)
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())
            in_features = out_dim

        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_features, 1)
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)

        self.L = L
        self.T_up_ref = T_up_ref

    def forward(self, s, T_up=None):
        """Evaluate ``T(s; T_up)`` in physical units of eV."""
        if s.dim() == 1:
            s = s.unsqueeze(-1)

        if T_up is None:
            T_up = torch.full_like(s, self.T_up_ref)
        elif not isinstance(T_up, torch.Tensor):
            T_up = torch.tensor(T_up, dtype=torch.float32).reshape(-1, 1)
        elif T_up.dim() == 0 or (T_up.dim() == 1 and T_up.shape[0] == 1):
            T_up = T_up.reshape(1, 1).expand(s.shape[0], 1)
        elif T_up.shape[0] != s.shape[0]:
            T_up = T_up.reshape(1, 1).expand(s.shape[0], 1)

        s_norm = s / self.L
        T_up_norm = T_up / self.T_up_ref
        x = torch.cat([s_norm, T_up_norm], dim=-1)

        if self.fourier is not None:
            x = self.fourier(x)

        x = self.hidden(x)
        T_norm = self.output_layer(x)
        T = torch.nn.functional.softplus(T_norm) * self.T_up_ref
        return T
