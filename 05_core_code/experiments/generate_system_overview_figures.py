import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
APP_DIR = os.path.join(PROJECT_DIR, "app", "replay_snapshot_output")
FIG_DIR = os.path.join(PROJECT_DIR, "experiments", "paper_figures_v2")
HPINV_SUMMARY = os.path.join(
    PROJECT_DIR,
    "experiments",
    "outputs_hierarchical_positive_inverter_run1",
    "paper_hierarchical_positive_inverter_summary.json",
)
STAGE1_SUMMARY = os.path.join(
    PROJECT_DIR,
    "experiments",
    "outputs_stage1_dualstream_mstcn_detection_raw_delta",
    "paper_stage1_dualstream_mstcn_summary.json",
)
XGB_SUMMARY = os.path.join(
    PROJECT_DIR,
    "experiments",
    "outputs_xgboost_baselines_v1",
    "xgboost_baseline_summary.json",
)
GRID_IMG = os.path.join(APP_DIR, "testset_best_grid_hierarchical.png")
GUI_IMG = os.path.join(APP_DIR, "replay_snapshot_hierarchical.png")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_card(ax, xywh, title, lines, facecolor="#F7F9FC", edgecolor="#355C7D", title_color="#17324D"):
    x, y, w, h = xywh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.05, title, fontsize=13, fontweight="bold", color=title_color, va="top")
    top = y + h - 0.10
    for idx, line in enumerate(lines):
        ax.text(x + 0.02, top - idx * 0.055, line, fontsize=10.5, color="#243746", va="top")
    return patch


def add_arrow(ax, start, end, color="#355C7D", lw=2.2):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=lw, color=color)
    ax.add_patch(arrow)


def draw_overview():
    os.makedirs(FIG_DIR, exist_ok=True)
    hpinv = load_json(HPINV_SUMMARY)
    stage1 = load_json(STAGE1_SUMMARY)
    xgb = load_json(XGB_SUMMARY)

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.96, "Flexible Tactile Sensing and Hierarchical Neural Inversion System", fontsize=20, fontweight="bold", color="#13293D", va="top")
    ax.text(0.03, 0.925, "From dynamic pressing signals to interpretable intraoperative nodule detection, size inversion, and coarse depth prediction", fontsize=11.5, color="#4C6272", va="top")

    add_card(
        ax,
        (0.03, 0.59, 0.26, 0.27),
        "1. Intraoperative tactile acquisition",
        [
            "Flexible array: 12x8 tactile grid",
            "Dynamic pressing sequence: T = 10, stride = 2",
            "42 size-depth conditions x 3 repetitions",
            "Ex vivo porcine lung with embedded nodule phantoms",
        ],
        facecolor="#F7FBFF",
        edgecolor="#3E7CB1",
    )
    add_card(
        ax,
        (0.36, 0.59, 0.28, 0.27),
        "2. Hierarchical learning pipeline",
        [
            f"Stage 1 detector: test AUC = {stage1['test_auc']:.4f}",
            f"Unified inverter: size top1 = {hpinv['test_metrics']['size_top1']:.4f}",
            f"Unified inverter: hard-route depth bAcc = {hpinv['test_metrics']['depth_hard_balanced_accuracy']:.4f}",
            "Route-aware optimization: GT / hard / soft / top2 route",
        ],
        facecolor="#F8FFF8",
        edgecolor="#4C9A2A",
    )
    add_card(
        ax,
        (0.71, 0.59, 0.26, 0.27),
        "3. Interpretable deployment",
        [
            f"XGBoost depth baseline bAcc = {xgb['size_depth_xgboost']['gt_positive_test_metrics']['depth_balanced_accuracy']:.4f}",
            "Latent probe + branch ablation + hard-pair",
            "Phase occlusion validates temporal strategy",
            "Real-time GUI: probability, size, depth, confidence",
        ],
        facecolor="#FFF8F3",
        edgecolor="#C97A40",
    )

    add_arrow(ax, (0.29, 0.72), (0.36, 0.72))
    add_arrow(ax, (0.64, 0.72), (0.71, 0.72))

    grid = mpimg.imread(GRID_IMG)
    gui = mpimg.imread(GUI_IMG)

    ax_grid = fig.add_axes([0.05, 0.10, 0.46, 0.37])
    ax_grid.imshow(grid)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.set_title("Test-set 6x7 condition matrix (best predicted frame per condition)", fontsize=11, pad=8)

    ax_gui = fig.add_axes([0.56, 0.10, 0.39, 0.37])
    ax_gui.imshow(gui)
    ax_gui.set_xticks([])
    ax_gui.set_yticks([])
    ax_gui.set_title("Real-time interface with detector + hierarchical inverter", fontsize=11, pad=8)

    ax.text(0.05, 0.49, "Clinical-to-algorithm overview", fontsize=12.5, fontweight="bold", color="#17324D")
    ax.text(0.56, 0.49, "System outputs for paper and deployment", fontsize=12.5, fontweight="bold", color="#17324D")

    fig.savefig(os.path.join(FIG_DIR, "Fig_R10_system_overview.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_pipeline():
    os.makedirs(FIG_DIR, exist_ok=True)
    hpinv = load_json(HPINV_SUMMARY)
    stage1 = load_json(STAGE1_SUMMARY)
    xgb = load_json(XGB_SUMMARY)

    fig = plt.figure(figsize=(16, 8.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Training and Inference Workflow of the Final Hierarchical System", fontsize=20, fontweight="bold", color="#13293D", va="top")

    ax.text(0.03, 0.86, "Training path", fontsize=14, fontweight="bold", color="#17324D")
    y = 0.72
    train_boxes = [
        (0.03, y, 0.16, 0.12, "Data protocol", ["File1 train", "File2 val", "File3 test", "Positive windows for inversion"]),
        (0.23, y, 0.18, 0.12, "Mechanism & XGBoost", [f"Depth baseline bAcc {xgb['size_depth_xgboost']['gt_positive_test_metrics']['depth_balanced_accuracy']:.4f}", "Spread / shape / deformation cues", "Provides interpretable reference"]),
        (0.45, y, 0.16, 0.12, "Stage 1 detector", [f"AUC {stage1['test_auc']:.4f}", "Raw + delta", "Detection-first design"]),
        (0.65, y, 0.15, 0.12, "Size learning", [f"Size top1 {hpinv['test_metrics']['size_top1']:.4f}", f"Size MAE {hpinv['test_metrics']['size_mae']:.4f}", "Route supervision"]),
        (0.83, y, 0.14, 0.12, "Unified inverter", [f"Hard-route depth bAcc {hpinv['test_metrics']['depth_hard_balanced_accuracy']:.4f}", "GT / hard / soft / top2 route", "Route-consistency regularization"]),
    ]
    for x, yy, w, h, title, lines in train_boxes:
        add_card(ax, (x, yy, w, h), title, lines, facecolor="#F8FBFD", edgecolor="#4F6D7A")
    for i in range(len(train_boxes) - 1):
        add_arrow(ax, (train_boxes[i][0] + train_boxes[i][2], y + 0.06), (train_boxes[i + 1][0], y + 0.06))

    ax.text(0.03, 0.49, "Inference path", fontsize=14, fontweight="bold", color="#17324D")
    y2 = 0.33
    infer_boxes = [
        (0.03, y2, 0.16, 0.12, "Tactile sequence", ["Raw amplitude", "Normalized shape", "Delta dynamics"]),
        (0.23, y2, 0.15, 0.12, "Detection gate", ["Always-visible probability", "Positive gate for inversion"]),
        (0.42, y2, 0.18, 0.12, "Hierarchical inverter", ["Size classification + regression", "Per-size depth experts", "Confidence-aware route"]),
        (0.64, y2, 0.15, 0.12, "Interpretability", ["Latent probe", "Branch ablation", "Hard-pair / phase occlusion"]),
        (0.83, y2, 0.14, 0.12, "Surgical UI", ["Probability curve", "Size / depth confidence", "Best-frame overview"]),
    ]
    for x, yy, w, h, title, lines in infer_boxes:
        add_card(ax, (x, yy, w, h), title, lines, facecolor="#FFFDF7", edgecolor="#B07D3C")
    for i in range(len(infer_boxes) - 1):
        add_arrow(ax, (infer_boxes[i][0] + infer_boxes[i][2], y2 + 0.06), (infer_boxes[i + 1][0], y2 + 0.06), color="#8A5A20")

    ax.text(0.03, 0.18, "Final system-level message", fontsize=14, fontweight="bold", color="#17324D")
    summary_patch = FancyBboxPatch(
        (0.03, 0.06),
        0.94,
        0.09,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=2.0,
        facecolor="#F4F7FA",
        edgecolor="#355C7D",
    )
    ax.add_patch(summary_patch)
    ax.text(
        0.05,
        0.105,
        "The final contribution is not a single best isolated metric. The system-level advance is that the hierarchical neural pipeline "
        "stably exceeds the structured XGBoost baseline on routed coarse depth under realistic deployment constraints while remaining interpretable.",
        fontsize=11.5,
        color="#243746",
        va="center",
    )

    fig.savefig(os.path.join(FIG_DIR, "Fig_R11_final_pipeline.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    draw_overview()
    draw_pipeline()
    print("Generated Fig_R10_system_overview.png and Fig_R11_final_pipeline.png")


if __name__ == "__main__":
    main()
