"""基线对比图生成.

Usage:
    python experiments/validation/plot_baseline.py
    python experiments/validation/plot_baseline.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sol_pinn.physics.regimes import from_name
from sol_pinn.physics.params import SolverConfig
from fd_reference.numerical.fd_solver import FDSolver
from sol_pinn.utils.plotting import (
    plot_parameter_scan,
    plot_temperature_profile,
)


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    config = from_name(regime)
    sc = SolverConfig(n_points=500, max_iter=2000, tol=1e-10)
    tag = regime

    # ── Multi T_up profiles ──
    fig, ax = plt.subplots(figsize=(8, 5))
    T_up_values = [60, 80, 100, 120, 150] if "conduction" in regime else [50, 80, 100, 150, 200]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(T_up_values)))
    for T_up, color in zip(T_up_values, colors):
        cfg = config.clone_with(T_up=T_up)
        result = FDSolver(cfg, sc).solve()
        T_t = result["T"][-1]
        ax.plot(result["s"], result["T"],
                label=f"$T_{{up}}$={T_up} eV, $T_t$={T_t:.1f} eV",
                color=color, linewidth=2)
    ax.set_xlabel("s [m]", fontsize=12)
    ax.set_ylabel("$T_e$ [eV]", fontsize=12)
    ax.set_title(f"Temperature Profiles ({tag})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    Path("figures/baseline").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"figures/baseline/multi_Tup_profiles_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/baseline/multi_Tup_profiles_{tag}.png")

    # ── T_target vs T_up ──
    T_up_scan = np.linspace(60, 200, 15) if "conduction" in regime else np.linspace(50, 200, 16)
    T_targets, q_targets = [], []
    for T_up in T_up_scan:
        result = FDSolver(config.clone_with(T_up=T_up), sc).solve()
        T_targets.append(result["T"][-1])
        q_targets.append(result["q"][-1])

    plot_parameter_scan(
        T_up_scan, T_targets, None,
        xlabel="$T_{up}$ [eV]", ylabel="$T_{target}$ [eV]",
        title=f"Target Temperature vs Upstream ({tag})",
        save_path=f"figures/baseline/Ttarget_vs_Tup_{tag}.png",
    )
    plot_parameter_scan(
        T_up_scan, q_targets, None,
        xlabel="$T_{up}$ [eV]", ylabel="$q_{target}$ [W/m$^2$]",
        title=f"Target Heat Flux vs Upstream ({tag})",
        save_path=f"figures/baseline/qtarget_vs_Tup_{tag}.png",
    )
    print(f"All {tag} baseline figures generated.")


if __name__ == "__main__":
    main()
