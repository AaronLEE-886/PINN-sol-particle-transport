"""Phase 3: study the effect of observation noise on gamma recovery.

Usage:
    python experiments/phase3_inverse/noise_study.py
    python experiments/phase3_inverse/noise_study.py --regime sheath-limited
"""

import sys
from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sol_pinn.physics.regimes import from_name
from sol_pinn.inverse.pinn_inversion import run_inversion_test


def main():
    regime = "conduction-limited"
    if "--regime" in sys.argv:
        idx = sys.argv.index("--regime")
        if idx + 1 < len(sys.argv):
            regime = sys.argv[idx + 1]

    config = from_name(regime)

    print("=" * 60)
    print(f"Noise Study: Impact of Noise on Gamma Recovery ({regime})")
    print("Method: diagnose gamma from sheath BC after PINN training")
    print("=" * 60)

    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]
    n_trials = 3
    n_obs = 20

    results = {nl: {"errors": [], "recovered": []} for nl in noise_levels}

    for nl in noise_levels:
        print(f"\nNoise level: {nl*100:.0f}%")
        for trial in range(n_trials):
            gamma_rec, gamma_err = run_inversion_test(
                gamma_true=7.0,
                n_obs=n_obs,
                noise_level=nl,
                n_adam=5000,
                n_lbfgs=200,
                config=config,
            )
            results[nl]["recovered"].append(gamma_rec)
            results[nl]["errors"].append(gamma_err)
            print(f"  Trial {trial+1}: gamma={gamma_rec:.4f}, error={gamma_err*100:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    mean_errors = [np.mean(results[nl]["errors"]) * 100 for nl in noise_levels]
    std_errors = [np.std(results[nl]["errors"]) * 100 for nl in noise_levels]

    ax1.errorbar(noise_levels, mean_errors, yerr=std_errors, fmt="o-", capsize=5, linewidth=2)
    ax1.set_xlabel("Noise Level (fraction of $T_{up}$)", fontsize=12)
    ax1.set_ylabel("Gamma Recovery Error [%]", fontsize=12)
    ax1.set_title(f"Gamma Error vs Noise Level ({regime})", fontsize=13)
    ax1.grid(True, alpha=0.3)

    for nl in noise_levels:
        ax2.scatter([nl] * len(results[nl]["recovered"]), results[nl]["recovered"], alpha=0.6, s=50)
    ax2.axhline(y=7.0, color="r", linestyle="--", label="True gamma=7.0")
    ax2.set_xlabel("Noise Level", fontsize=12)
    ax2.set_ylabel("Recovered $\gamma$", fontsize=12)
    ax2.set_title(f"Recovered Gamma Values ({regime})", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path("figures/inverse").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"figures/inverse/noise_study_{regime}.png", dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to figures/inverse/noise_study_{regime}.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
