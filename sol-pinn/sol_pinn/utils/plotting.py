"""Plotting helpers for profiles, comparisons, and training curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_temperature_profile(s, T, title=None, save_path=None,
                             ax=None, label=None, color=None, linestyle="-"):
    """Plot a temperature profile ``T(s)``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, T, label=label, color=color, linestyle=linestyle, linewidth=2)
    ax.set_xlabel("s [m]", fontsize=12)
    ax.set_ylabel("$T_e$ [eV]", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    if label:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_heat_flux(s, T, dT_ds, kappa_parallel, title=None, save_path=None):
    """Plot the conductive heat flux ``q = -kappa*T^(5/2)*dT/ds``."""
    T_safe = np.maximum(T, 1e-8)
    q = -kappa_parallel * T_safe ** 2.5 * dT_ds
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, q, linewidth=2)
    ax.set_xlabel("s [m]", fontsize=12)
    ax.set_ylabel("$q_\\parallel$ [W/m$^2$]", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_comparison(s, T_ref, T_pred, title=None, save_path=None,
                    labels=("Reference", "PINN")):
    """Plot a reference solution, prediction, and pointwise error."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(s, T_ref, "b-", label=labels[0], linewidth=2)
    ax1.plot(s, T_pred, "r--", label=labels[1], linewidth=2)
    ax1.set_ylabel("$T_e$ [eV]", fontsize=12)
    if title:
        ax1.set_title(title, fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    error = T_pred - T_ref
    ax2.plot(s, error, "k-", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("s [m]", fontsize=12)
    ax2.set_ylabel("Error [eV]", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_loss_curves(loss_history, save_path=None):
    """Plot training loss histories on a semilog scale."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in loss_history:
        if loss_history[key]:
            ax.semilogy(loss_history[key], label=key, linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss History", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax


def plot_parameter_scan(params, values, errors, xlabel, ylabel,
                        title=None, save_path=None):
    """Plot the result of a one-parameter scan with optional error bands."""
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(params, values, "o-", linewidth=2)
    if errors is not None:
        ax.fill_between(
            params,
            np.array(values) - np.array(errors),
            np.array(values) + np.array(errors),
            alpha=0.2,
        )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return ax
