"""Error metrics and boundary-condition diagnostics."""

import numpy as np
import torch


def _to_numpy(value):
    """Convert tensors to NumPy arrays without changing ndarray inputs."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def relative_L2_error(T_pred, T_ref) -> float:
    """Compute relative L2 error ``||pred-ref|| / ||ref||``."""
    T_pred = _to_numpy(T_pred)
    T_ref = _to_numpy(T_ref)
    diff = T_pred - T_ref
    return np.linalg.norm(diff) / (np.linalg.norm(T_ref) + 1e-12)


def max_error(T_pred, T_ref) -> float:
    """Compute the maximum absolute error."""
    T_pred = _to_numpy(T_pred)
    T_ref = _to_numpy(T_ref)
    return np.max(np.abs(T_pred - T_ref))


def boundary_error(T_pred, T_ref, s, boundary_ratio: float = 0.1) -> float:
    """Compute relative L2 error near the target boundary."""
    T_pred = _to_numpy(T_pred)
    T_ref = _to_numpy(T_ref)
    s = _to_numpy(s)

    mask = s >= (1.0 - boundary_ratio) * s.max()
    T_pred_b = T_pred[mask]
    T_ref_b = T_ref[mask]
    return np.linalg.norm(T_pred_b - T_ref_b) / (np.linalg.norm(T_ref_b) + 1e-12)


def sheath_satisfaction(T_t, dT_ds_t, alpha, kappa_parallel) -> float:
    """Measure the absolute residual of the sheath boundary condition."""
    T_t = _to_numpy(T_t)
    dT_ds_t = _to_numpy(dT_ds_t)
    T_safe = np.maximum(T_t, 1e-8)
    return np.abs(kappa_parallel * T_safe ** 2.5 * dT_ds_t + alpha * np.sqrt(T_safe))


def pde_residual_stats(residual) -> dict:
    """Summarize PDE residual magnitude with a compact set of statistics."""
    residual = np.asarray(_to_numpy(residual), dtype=float).reshape(-1)
    abs_res = np.abs(residual)
    return {
        "mean_abs": float(np.mean(abs_res)),
        "max_abs": float(np.max(abs_res)),
        "rms": float(np.sqrt(np.mean(residual ** 2))),
    }


def profile_target_metrics(s, T) -> dict:
    """Return target-end temperature and gradient diagnostics from a profile."""
    s = np.asarray(_to_numpy(s), dtype=float).reshape(-1)
    T = np.asarray(_to_numpy(T), dtype=float).reshape(-1)
    dT_ds = np.gradient(T, s)
    return {
        "T_t": float(T[-1]),
        "dT_ds_t": float(dT_ds[-1]),
        "q_t": None,
    }


def physics_consistency_metrics(s, T, config, source_values=None) -> dict:
    """Compute unified physics-consistency diagnostics for a 1D profile."""
    s = np.asarray(_to_numpy(s), dtype=float).reshape(-1)
    T = np.asarray(_to_numpy(T), dtype=float).reshape(-1)

    dT_ds = np.gradient(T, s)
    T_safe = np.maximum(T, 1e-8)
    q = -config.kappa_parallel * T_safe ** 2.5 * dT_ds
    dq_ds = np.gradient(q, s)

    if source_values is None:
        source_values = np.array([config.S_E(si) for si in s], dtype=float)
    else:
        source_values = np.asarray(_to_numpy(source_values), dtype=float).reshape(-1)

    residual = dq_ds - source_values
    pde_stats = pde_residual_stats(residual)

    up_residual = float(abs(T[0] - config.T_up))
    sheath_abs = float(sheath_satisfaction(T[-1], dT_ds[-1], config.alpha, config.kappa_parallel))
    q_t = float(q[-1])
    q_sheath = float(config.alpha * np.sqrt(T_safe[-1]))

    return {
        "T_t": float(T[-1]),
        "dT_ds_t": float(dT_ds[-1]),
        "q_t": q_t,
        "q_sheath": q_sheath,
        "upstream_bc_abs": up_residual,
        "sheath_bc_abs": sheath_abs,
        "pde_mean_abs": pde_stats["mean_abs"],
        "pde_max_abs": pde_stats["max_abs"],
        "pde_rms": pde_stats["rms"],
    }
