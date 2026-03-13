import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
FIG_DIR = os.path.join(PROJECT_DIR, "experiments", "paper_figures_v2")
MECH_DIR = os.path.join(PROJECT_DIR, "experiments", "mechanism_exploration_v1")
XGB_DIR = os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_explainability_v1")
HP_DIR = os.path.join(PROJECT_DIR, "experiments", "outputs_hierarchical_positive_inverter_explainability_v1")


def panel(ax, image_path, title, label):
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(0.01, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=(0.1, 0.2, 0.3, 0.75), edgecolor="none"))


def generate_mechanism_xgb_summary():
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.45], height_ratios=[1.0, 1.0], wspace=0.08, hspace=0.12)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    panel(ax_a, os.path.join(MECH_DIR, "Fig_M2_average_heatmaps.png"), "Positive vs. negative mean tactile maps", "A")
    panel(ax_b, os.path.join(MECH_DIR, "Fig_M4_condition_trends.png"), "Condition-level trends across size and depth", "B")
    panel(ax_c, os.path.join(XGB_DIR, "Fig_XGB_depth_explainability_tbme.png"), "TBME-style XGBoost depth explainability figure", "C")

    fig.suptitle("Mechanism analysis and XGBoost depth explainability summary", fontsize=18, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "Fig_R12_mechanism_xgboost_summary.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_hierarchical_explain_summary():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    panel(axes[0, 0], os.path.join(HP_DIR, "hierarchical_probe_family_comparison.png"), "Latent probe recovery by feature family", "A")
    panel(axes[0, 1], os.path.join(HP_DIR, "hierarchical_branch_ablation.png"), "Branch ablation of the unified inverter", "B")
    panel(axes[1, 0], os.path.join(HP_DIR, "hierarchical_phase_occlusion.png"), "Phase occlusion analysis", "C")
    panel(axes[1, 1], os.path.join(HP_DIR, "hierarchical_hard_pair_examples.png"), "Hard-pair examples: deep-large vs shallow-small", "D")
    fig.suptitle("Explainability summary of the hierarchical positive inverter", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, "Fig_R13_hierarchical_explainability_summary.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    generate_mechanism_xgb_summary()
    generate_hierarchical_explain_summary()
    print("Generated Fig_R12_mechanism_xgboost_summary.png and Fig_R13_hierarchical_explainability_summary.png")


if __name__ == "__main__":
    main()
