"""Systematic analysis for the conduction-limited T_up=50 eV hard case.

Usage:
    python -m experiments.validation.hardcase_analysis
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path
from sol_pinn.physics.params import SolverConfig
from sol_pinn.physics.regimes import from_name
from sol_pinn.pinn.pinn_solver import PINNSolver
from sol_pinn.utils.io import save_solution, save_table_csv
from sol_pinn.utils.metrics import (
    max_error,
    physics_consistency_metrics,
    relative_L2_error,
)
from sol_pinn.utils.sampling import target_refined, torch_collocation

ensure_repo_root_on_path(__file__)


def _run_case(name, config, fourier_kwargs, network_kwargs, collocation_fn,
              n_adam, n_lbfgs, loss_weights=None, causal=False, device="cpu",
              model_type="temperature"):
    """Train one hard-case configuration and return unified diagnostics."""
    solver = PINNSolver(
        config,
        use_fourier=True,
        fourier_kwargs=fourier_kwargs,
        network_kwargs=network_kwargs,
        loss_weights=loss_weights,
        causal=causal,
        device=device,
        model_type=model_type,
    )

    s_colloc = torch_collocation(collocation_fn(config.L))
    t0 = time.time()
    solver.train(s_colloc, n_adam=n_adam, n_lbfgs=n_lbfgs)
    elapsed = time.time() - t0

    s_eval = np.linspace(0.0, config.L, 500)
    T_pred = solver.predict(s_eval)

    metrics = physics_consistency_metrics(s_eval, T_pred, config)
    return {
        "name": name,
        "solver": solver,
        "time_s": elapsed,
        "s_eval": s_eval,
        "T_pred": T_pred,
        **metrics,
    }


def _run_curriculum_case(name, regime_name, curriculum_tups, final_tup,
                         fourier_kwargs, network_kwargs, collocation_fn,
                         n_adam, n_lbfgs, loss_weights=None, causal=False,
                         device="cpu", model_type="temperature"):
    """Train via staged fine-tuning from easier to harder upstream temperatures."""
    assert curriculum_tups, "curriculum_tups must not be empty"

    solver = None
    total_time = 0.0
    stage_logs = []

    for stage_tup in curriculum_tups:
        stage_config = from_name(regime_name, T_up=stage_tup)
        if solver is None:
            solver = PINNSolver(
                stage_config,
                use_fourier=True,
                fourier_kwargs=fourier_kwargs,
                network_kwargs=network_kwargs,
                loss_weights=loss_weights,
                causal=causal,
                device=device,
                model_type=model_type,
            )
        else:
            solver = solver.clone_for_config(stage_config)

        s_colloc = torch_collocation(collocation_fn(stage_config.L))
        t0 = time.time()
        solver.train(s_colloc, n_adam=n_adam, n_lbfgs=n_lbfgs)
        stage_time = time.time() - t0
        total_time += stage_time
        stage_logs.append((stage_tup, stage_time))

    final_config = from_name(regime_name, T_up=final_tup)
    solver = solver.clone_for_config(final_config)
    s_colloc = torch_collocation(collocation_fn(final_config.L))
    t0 = time.time()
    solver.train(s_colloc, n_adam=n_adam, n_lbfgs=n_lbfgs)
    stage_time = time.time() - t0
    total_time += stage_time
    stage_logs.append((final_tup, stage_time))

    s_eval = np.linspace(0.0, final_config.L, 500)
    T_pred = solver.predict(s_eval)
    metrics = physics_consistency_metrics(s_eval, T_pred, final_config)
    return {
        "name": name,
        "solver": solver,
        "time_s": total_time,
        "stage_logs": stage_logs,
        "s_eval": s_eval,
        "T_pred": T_pred,
        **metrics,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = from_name("conduction-limited", T_up=50.0)
    out_dir = Path("figures/pinn")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Hard Case Analysis: conduction-limited T_up=50 eV")
    print("=" * 72)

    fd = FDSolver(config, SolverConfig(n_points=2000, max_iter=2000, tol=1e-10)).solve()
    s_eval = np.linspace(0.0, config.L, 500)
    T_fd = np.interp(s_eval, fd["s"], fd["T"])
    fd_metrics = physics_consistency_metrics(s_eval, T_fd, config)

    colloc_default = lambda L: target_refined(200, 100, L, boundary_ratio=0.10)
    colloc_heavy = lambda L: target_refined(300, 200, L, boundary_ratio=0.05)

    cases = [
        {
            "name": "baseline_best_known",
            "fourier_kwargs": {"mapping_size": 64, "sigma": 5.0},
            "network_kwargs": {"layer_sizes": [64, 64, 64], "activation": "tanh", "output_bias": 25.0},
            "collocation_fn": colloc_default,
            "n_adam": 3000,
            "n_lbfgs": 200,
            "loss_weights": None,
            "causal": False,
        },
        {
            "name": "longer_training",
            "fourier_kwargs": {"mapping_size": 64, "sigma": 5.0},
            "network_kwargs": {"layer_sizes": [64, 64, 64], "activation": "tanh", "output_bias": 25.0},
            "collocation_fn": colloc_default,
            "n_adam": 8000,
            "n_lbfgs": 300,
            "loss_weights": None,
            "causal": False,
        },
        {
            "name": "heavier_boundary_sampling",
            "fourier_kwargs": {"mapping_size": 64, "sigma": 5.0},
            "network_kwargs": {"layer_sizes": [64, 64, 64], "activation": "tanh", "output_bias": 25.0},
            "collocation_fn": colloc_heavy,
            "n_adam": 3000,
            "n_lbfgs": 200,
            "loss_weights": None,
            "causal": False,
        },
        {
            "name": "sheath_weighted",
            "fourier_kwargs": {"mapping_size": 64, "sigma": 5.0},
            "network_kwargs": {"layer_sizes": [64, 64, 64], "activation": "tanh", "output_bias": 25.0},
            "collocation_fn": colloc_default,
            "n_adam": 3000,
            "n_lbfgs": 200,
            "loss_weights": {"pde": 1.0, "up": 1.0, "sheath": 20.0},
            "causal": False,
        },
        {
            "name": "transformed_state",
            "fourier_kwargs": {"mapping_size": 64, "sigma": 5.0},
            "network_kwargs": {"layer_sizes": [64, 64, 64], "activation": "tanh"},
            "collocation_fn": colloc_default,
            "n_adam": 3000,
            "n_lbfgs": 200,
            "loss_weights": None,
            "causal": False,
            "model_type": "transformed_temperature",
        },
        {
            "name": "piecewise_target_branch",
            "fourier_kwargs": {"mapping_size": 64, "sigma": 5.0},
            "network_kwargs": {
                "layer_sizes": [64, 64, 64],
                "target_layer_sizes": [96, 96, 96],
                "activation": "tanh",
                "output_bias": 25.0,
                "blend_center": 0.82,
                "blend_sharpness": 20.0,
            },
            "collocation_fn": colloc_heavy,
            "n_adam": 3000,
            "n_lbfgs": 200,
            "loss_weights": None,
            "causal": False,
            "model_type": "piecewise_temperature",
        },
    ]

    results = []
    for case in cases:
        print(f"\nRunning case: {case['name']}")
        result = _run_case(device=device, config=config, **case)
        result["temperature_rel_l2"] = relative_L2_error(result["T_pred"], T_fd)
        result["temperature_max_abs"] = max_error(result["T_pred"], T_fd)
        result["T_t_fd"] = float(T_fd[-1])
        result["q_t_fd"] = float(fd_metrics["q_t"])
        print(
            f"  rel_L2={result['temperature_rel_l2']:.4e}  "
            f"T_t={result['T_t']:.4f}  sheath_abs={result['sheath_bc_abs']:.4e}  "
            f"pde_rms={result['pde_rms']:.4e}"
        )
        results.append(result)

    print("\nRunning case: curriculum_80_60_50")
    curriculum_result = _run_curriculum_case(
        name="curriculum_80_60_50",
        regime_name="conduction-limited",
        curriculum_tups=[80.0, 60.0],
        final_tup=50.0,
        fourier_kwargs={"mapping_size": 64, "sigma": 5.0},
        network_kwargs={"layer_sizes": [64, 64, 64], "activation": "tanh", "output_bias": 25.0},
        collocation_fn=colloc_default,
        n_adam=2000,
        n_lbfgs=100,
        loss_weights=None,
        causal=False,
        device=device,
        model_type="temperature",
    )
    curriculum_result["temperature_rel_l2"] = relative_L2_error(curriculum_result["T_pred"], T_fd)
    curriculum_result["temperature_max_abs"] = max_error(curriculum_result["T_pred"], T_fd)
    curriculum_result["T_t_fd"] = float(T_fd[-1])
    curriculum_result["q_t_fd"] = float(fd_metrics["q_t"])
    print(
        f"  rel_L2={curriculum_result['temperature_rel_l2']:.4e}  "
        f"T_t={curriculum_result['T_t']:.4f}  sheath_abs={curriculum_result['sheath_bc_abs']:.4e}  "
        f"pde_rms={curriculum_result['pde_rms']:.4e}"
    )
    print(f"  stages={curriculum_result['stage_logs']}")
    results.append(curriculum_result)

    # Composite ranking: balance temperature error and physics consistency
    for r in results:
        r["composite_score"] = (
            np.log10(r["temperature_rel_l2"] + 1e-16)
            + 0.5 * np.log10(r["sheath_bc_abs"] + 1e-16)
            + 0.5 * np.log10(r["pde_rms"] + 1e-16)
        )

    print("\nSummary")
    print("-" * 72)
    print(
        f"{'case':<24} | {'rel_L2':>10} | {'T_t':>8} | {'T_t_fd':>8} | "
        f"{'sheath_bc':>11} | {'pde_rms':>10} | {'score':>9} | {'time':>6}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<24} | {r['temperature_rel_l2']:10.4e} | {r['T_t']:8.4f} | "
            f"{r['T_t_fd']:8.4f} | {r['sheath_bc_abs']:11.4e} | {r['pde_rms']:10.4e} | "
            f"{r['composite_score']:9.4f} | "
            f"{r['time_s']:6.0f}s"
        )

    best = min(results, key=lambda item: item["temperature_rel_l2"])
    best_composite = min(results, key=lambda item: item["composite_score"])

    # Save tabular summary for later reporting
    summary_rows = []
    for r in results:
        summary_rows.append({
            "case": r["name"],
            "temperature_rel_l2": r["temperature_rel_l2"],
            "temperature_max_abs": r["temperature_max_abs"],
            "T_t": r["T_t"],
            "T_t_fd": r["T_t_fd"],
            "q_t": r["q_t"],
            "q_t_fd": r["q_t_fd"],
            "upstream_bc_abs": r["upstream_bc_abs"],
            "sheath_bc_abs": r["sheath_bc_abs"],
            "pde_rms": r["pde_rms"],
            "pde_max_abs": r["pde_max_abs"],
            "composite_score": r["composite_score"],
            "time_s": r["time_s"],
        })
        save_solution(
            out_dir / f"hardcase_{r['name']}.npz",
            r["s_eval"],
            r["T_pred"],
            metadata={
                "case": r["name"],
                "temperature_rel_l2": r["temperature_rel_l2"],
                "sheath_bc_abs": r["sheath_bc_abs"],
                "pde_rms": r["pde_rms"],
                "composite_score": r["composite_score"],
            },
        )

    save_table_csv(
        out_dir / "hardcase_analysis_summary.csv",
        summary_rows,
        fieldnames=[
            "case",
            "temperature_rel_l2",
            "temperature_max_abs",
            "T_t",
            "T_t_fd",
            "q_t",
            "q_t_fd",
            "upstream_bc_abs",
            "sheath_bc_abs",
            "pde_rms",
            "pde_max_abs",
            "composite_score",
            "time_s",
        ],
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(s_eval, T_fd, "k-", linewidth=2.5, label="FD")
    for r in results:
        axes[0].plot(s_eval, r["T_pred"], "--", linewidth=1.5, label=r["name"])
    axes[0].set_xlabel("s [m]")
    axes[0].set_ylabel("T [eV]")
    axes[0].set_title("Hard Case Temperature Profiles")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    target_mask = s_eval >= 0.8 * config.L
    axes[1].plot(s_eval[target_mask], T_fd[target_mask], "k-", linewidth=2.5, label="FD")
    for r in results:
        axes[1].plot(s_eval[target_mask], r["T_pred"][target_mask], "--", linewidth=1.5, label=r["name"])
    axes[1].set_xlabel("s [m]")
    axes[1].set_ylabel("T [eV]")
    axes[1].set_title("Target-Region Zoom")
    axes[1].grid(True, alpha=0.3)

    names = [r["name"] for r in results]
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, [r["temperature_rel_l2"] for r in results], width=0.4, label="rel_L2")
    axes[2].bar(x + 0.2, [r["sheath_bc_abs"] for r in results], width=0.4, label="sheath_bc_abs")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_yscale("log")
    axes[2].set_title("Error vs Physics Consistency")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig_path = out_dir / "hardcase_analysis_Tup50.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {fig_path}")
    print(f"Saved {out_dir / 'hardcase_analysis_summary.csv'}")
    print(f"Best configuration by rel_L2: {best['name']}")
    print(f"Best configuration by composite score: {best_composite['name']}")


if __name__ == "__main__":
    main()
