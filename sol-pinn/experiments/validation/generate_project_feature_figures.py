"""Generate a compact figure set highlighting the project's main features.

Usage:
    python -m experiments.validation.generate_project_feature_figures
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sol_pinn.experiments.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)


def _load_hardcase_summary(base_dir: Path):
    csv_path = base_dir / "hardcase_analysis_summary.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main():
    fig_dir = Path("figures")
    pinn_dir = fig_dir / "pinn"
    feature_dir = fig_dir / "feature_report"
    feature_dir.mkdir(parents=True, exist_ok=True)

    hardcase_df = _load_hardcase_summary(pinn_dir)

    if hardcase_df is not None and not hardcase_df.empty:
        hardcase_df = hardcase_df.sort_values("composite_score", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Hard case: error and physics consistency together
        x = np.arange(len(hardcase_df))
        labels = hardcase_df["case"].tolist()
        axes[0].bar(x - 0.2, hardcase_df["temperature_rel_l2"], width=0.4, label="rel_L2")
        axes[0].bar(x + 0.2, hardcase_df["sheath_bc_abs"], width=0.4, label="sheath BC abs")
        axes[0].set_yscale("log")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=25, ha="right")
        axes[0].set_title("Hard Case: Error and Physics Consistency")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Composite score ranking
        axes[1].barh(labels, hardcase_df["composite_score"], color="#c44e52")
        axes[1].invert_yaxis()
        axes[1].set_title("Hard Case Composite Ranking")
        axes[1].set_xlabel("Lower is better")
        axes[1].grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(feature_dir / "hardcase_feature_summary.png", dpi=150, bbox_inches="tight")

    # Project-level feature board using existing generated figures
    existing_refs = [
        ("Forward", pinn_dir / "pinn_vs_fd_comparison_conduction-limited.png"),
        ("Hard Case", pinn_dir / "hardcase_analysis_Tup50.png"),
        ("Parameter Scan", pinn_dir / "scan_tup_summary.png"),
        ("Heat Flux", pinn_dir / "heatflux_summary.png"),
    ]

    available = [(title, path) for title, path in existing_refs if path.exists()]
    if available:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for ax, (title, path) in zip(axes, available):
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        for ax in axes[len(available):]:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(feature_dir / "project_feature_board.png", dpi=150, bbox_inches="tight")

    print(f"Feature figures saved to {feature_dir}")


if __name__ == "__main__":
    main()
