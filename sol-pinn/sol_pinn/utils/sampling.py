"""Sampling utilities for FD grids, PINN collocation, and adaptive refinement."""

import numpy as np
import torch


def uniform_grid(n_points: int, L: float, start: float = 0.0) -> np.ndarray:
    """Generate a uniform grid on ``[start, L]``."""
    return np.linspace(start, L, n_points)


def target_refined(n_core: int, n_boundary: int, L: float,
                   boundary_ratio: float = 0.1) -> np.ndarray:
    """Generate collocation points with extra density near the target boundary."""
    core = uniform_grid(n_core, L)
    boundary_start = (1.0 - boundary_ratio) * L
    boundary = uniform_grid(n_boundary, L, start=boundary_start)
    combined = np.unique(np.sort(np.concatenate([core, boundary])))
    return combined


def residual_adaptive(existing_points, residual_fn, n_new: int,
                      strategy="residual") -> np.ndarray:
    """Add new points in regions with large residuals."""
    residuals = residual_fn(existing_points)
    residuals = np.abs(residuals).flatten()

    if strategy == "residual":
        probs = residuals / (residuals.sum() + 1e-12)
        indices = np.random.choice(len(existing_points), size=n_new, p=probs)
        new_points = existing_points[indices]
        noise = np.random.uniform(-1, 1, size=n_new) * (existing_points[1] - existing_points[0])
        new_points = np.clip(new_points + noise, existing_points[0], existing_points[-1])
    else:
        threshold = np.percentile(residuals, 75)
        high_residual_region = existing_points[residuals > threshold]
        if len(high_residual_region) > 1:
            low, high = high_residual_region.min(), high_residual_region.max()
            new_points = np.random.uniform(low, high, size=n_new)
        else:
            new_points = np.random.uniform(existing_points[0], existing_points[-1], size=n_new)

    return np.sort(new_points)


def torch_collocation(s_np: np.ndarray) -> torch.Tensor:
    """Convert a NumPy coordinate array to a ``(N, 1)`` float tensor."""
    return torch.tensor(s_np, dtype=torch.float32).reshape(-1, 1)
