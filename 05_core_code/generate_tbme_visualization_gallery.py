import json
from pathlib import Path
from string import ascii_uppercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "tbme_submission_release_v1"
OUTPUT_DIR = RELEASE_DIR / "03_results_core" / "visualization_gallery_v1"

LATENT_FEATURES_PATH = ROOT / "experiments" / "depth_model_explainability_v1" / "depth_model_latent_features.npy"
LATENT_META_PATH = ROOT / "experiments" / "depth_model_explainability_v1" / "depth_model_positive_windows.csv"
HIER_PRED_PATH = ROOT / "experiments" / "outputs_hierarchical_positive_inverter_explainability_v1" / "hierarchical_test_window_predictions.csv"
HIER_SUMMARY_PATH = ROOT / "experiments" / "outputs_hierarchical_positive_inverter_explainability_v1" / "hierarchical_explainability_summary.json"

DEPTH_ORDER = ["shallow", "middle", "deep"]
DEPTH_COLORS = {
    "shallow": "#2a9d8f",
    "middle": "#e9c46a",
    "deep": "#e76f51",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_output() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def panel_labels(axes):
    flat_axes = np.ravel(axes)
    for idx, ax in enumerate(flat_axes):
        ax.text(
            -0.14,
            1.06,
            ascii_uppercase[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def save_dual(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", dpi=260, bbox_inches="tight")
    plt.close(fig)


def load_latent_inputs():
    features = np.load(LATENT_FEATURES_PATH)
    meta = pd.read_csv(LATENT_META_PATH)
    if len(meta) != features.shape[0]:
        raise ValueError("Latent metadata and feature array length mismatch.")
    return features, meta


def load_hier_inputs():
    hier = pd.read_csv(HIER_PRED_PATH)
    summary = json.loads(HIER_SUMMARY_PATH.read_text(encoding="utf-8"))
    return hier, summary


def build_embeddings(features: np.ndarray):
    z = StandardScaler().fit_transform(features)
    z_pca20 = PCA(n_components=min(20, z.shape[1], z.shape[0] - 1), random_state=2026).fit_transform(z)
    z_pca2 = PCA(n_components=2, random_state=2026).fit_transform(z)
    perplexity = max(20, min(35, z.shape[0] // 30))
    z_tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=2026,
    ).fit_transform(z_pca20)
    return z_tsne, z_pca2


def plot_embedding_gallery(features: np.ndarray, meta: pd.DataFrame) -> None:
    z_tsne, z_pca2 = build_embeddings(features)
    meta = meta.copy()
    meta["tsne_x"] = z_tsne[:, 0]
    meta["tsne_y"] = z_tsne[:, 1]
    meta["pca_x"] = z_pca2[:, 0]
    meta["pca_y"] = z_pca2[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.5))
    depth_palette = [DEPTH_COLORS[d] for d in DEPTH_ORDER]

    sns.scatterplot(
        data=meta,
        x="tsne_x",
        y="tsne_y",
        hue="coarse_depth",
        hue_order=DEPTH_ORDER,
        palette=depth_palette,
        s=24,
        linewidth=0.0,
        alpha=0.78,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("t-SNE colored by coarse depth")
    axes[0, 0].legend(title="Depth", frameon=False, loc="best")

    sc1 = axes[0, 1].scatter(
        meta["tsne_x"],
        meta["tsne_y"],
        c=meta["size_cm"],
        cmap="viridis",
        s=24,
        alpha=0.80,
        linewidths=0.0,
    )
    axes[0, 1].set_title("t-SNE colored by nodule size")
    cbar1 = fig.colorbar(sc1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label("Size (cm)")

    sns.scatterplot(
        data=meta,
        x="pca_x",
        y="pca_y",
        hue="coarse_depth",
        hue_order=DEPTH_ORDER,
        palette=depth_palette,
        s=24,
        linewidth=0.0,
        alpha=0.78,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("PCA colored by coarse depth")
    axes[1, 0].legend(title="Depth", frameon=False, loc="best")

    sc2 = axes[1, 1].scatter(
        meta["pca_x"],
        meta["pca_y"],
        c=meta["size_cm"],
        cmap="viridis",
        s=24,
        alpha=0.80,
        linewidths=0.0,
    )
    axes[1, 1].set_title("PCA colored by nodule size")
    cbar2 = fig.colorbar(sc2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar2.set_label("Size (cm)")

    for depth_name in DEPTH_ORDER:
        subset = meta[meta["coarse_depth"] == depth_name]
        if subset.empty:
            continue
        axes[0, 0].text(subset["tsne_x"].mean(), subset["tsne_y"].mean(), depth_name, fontsize=9, fontweight="bold")
        axes[1, 0].text(subset["pca_x"].mean(), subset["pca_y"].mean(), depth_name, fontsize=9, fontweight="bold")

    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(alpha=0.18)

    panel_labels(axes)
    fig.suptitle("G1. Candidate embedding views for the raw-input scientific model", y=1.02, fontweight="bold")
    save_dual(fig, "G1_embedding_gallery_tsne_pca")


def plot_feature_distributions(hier: pd.DataFrame) -> None:
    df = hier.copy()
    df["true_label"] = pd.Categorical(df["true_label"], categories=DEPTH_ORDER, ordered=True)
    features = [
        ("raw_max_max", "Peak amplitude"),
        ("spatial_entropy_max", "Spatial entropy"),
        ("hotspot_radius_max", "Hotspot radius"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    for ax, (col, title) in zip(axes, features):
        sns.boxplot(
            data=df,
            x="true_label",
            y=col,
            hue="true_label",
            order=DEPTH_ORDER,
            palette=DEPTH_COLORS,
            linewidth=1.2,
            fliersize=0,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=df.sample(min(len(df), 350), random_state=2026),
            x="true_label",
            y=col,
            order=DEPTH_ORDER,
            color="black",
            size=2.6,
            alpha=0.28,
            jitter=0.22,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(alpha=0.18)

    panel_labels(axes)
    fig.suptitle("G2. Physically interpretable feature distributions by coarse depth", y=1.04, fontweight="bold")
    save_dual(fig, "G2_feature_distributions_by_depth")


def plot_route_heatmaps(hier: pd.DataFrame) -> None:
    df = hier.copy()
    df["size_cm"] = df["size_cm"].round(2)
    df["depth_cm"] = df["depth_cm"].round(1)
    acc = (
        df.groupby(["depth_cm", "size_cm"], as_index=False)["correct"]
        .mean()
        .pivot(index="depth_cm", columns="size_cm", values="correct")
        .sort_index(ascending=True)
    )
    conf = (
        df.groupby(["depth_cm", "size_cm"], as_index=False)["depth_conf"]
        .mean()
        .pivot(index="depth_cm", columns="size_cm", values="depth_conf")
        .sort_index(ascending=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3))
    sns.heatmap(acc, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=axes[0], linewidths=0.4, linecolor="#f0f0f0")
    axes[0].set_title("Predicted-route correctness")
    axes[0].set_xlabel("Size (cm)")
    axes[0].set_ylabel("Depth (cm)")

    sns.heatmap(conf, annot=True, fmt=".2f", cmap="rocket_r", cbar=True, ax=axes[1], linewidths=0.4, linecolor="#f0f0f0")
    axes[1].set_title("Mean depth confidence")
    axes[1].set_xlabel("Size (cm)")
    axes[1].set_ylabel("Depth (cm)")

    panel_labels(axes)
    fig.suptitle("G3. Size-depth landscape views of the hierarchical deployment model", y=1.04, fontweight="bold")
    save_dual(fig, "G3_route_heatmaps")


def plot_probability_structure_views(hier: pd.DataFrame) -> None:
    df = hier.copy()
    df["true_label"] = pd.Categorical(df["true_label"], categories=DEPTH_ORDER, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 5.2))

    sns.violinplot(
        data=df,
        x="true_label",
        y="p_deep",
        hue="true_label",
        order=DEPTH_ORDER,
        palette=DEPTH_COLORS,
        inner="quartile",
        cut=0,
        linewidth=1.0,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Distribution of $p_{deep}$ by true depth")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("$p_{deep}$")

    scatter = axes[1].scatter(
        df["raw_max_max"],
        df["spatial_entropy_max"],
        c=df["depth_cm"],
        cmap="viridis",
        s=22 + df["size_cm"] * 28,
        alpha=0.68,
        linewidths=0.25,
        edgecolors="white",
    )
    axes[1].set_title("Amplitude-versus-spread structure view")
    axes[1].set_xlabel("Peak amplitude")
    axes[1].set_ylabel("Spatial entropy")
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Depth (cm)")

    panel_labels(axes)
    fig.suptitle("G4. Candidate data views for depth separability and overlap", y=1.04, fontweight="bold")
    save_dual(fig, "G4_probability_structure_views")


def plot_probe_phase_summary(summary: dict) -> None:
    probe = summary["probe_summary"]
    phase_rows = pd.DataFrame(summary["phase_occlusion_summary"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    probe_names = ["Hybrid latent", "Raw trunk", "Size only"]
    probe_vals = [
        probe["mean_hybrid_r2"],
        probe["mean_trunk_r2"],
        probe["mean_size_only_r2"],
    ]
    sns.barplot(
        x=probe_names,
        y=probe_vals,
        hue=probe_names,
        palette=["#264653", "#2a9d8f", "#e9c46a"],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Frozen-feature linear probes")
    axes[0].set_ylabel("Mean test $R^2$")
    axes[0].set_xlabel("")
    for idx, val in enumerate(probe_vals):
        axes[0].text(idx, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    sns.barplot(
        data=phase_rows,
        x="phase",
        y="mean_drop",
        hue="phase",
        palette=["#8ecae6", "#219ebc", "#ffb703", "#fb8500"],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Phase occlusion summary")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean probability drop")
    axes[1].tick_params(axis="x", rotation=18)
    for idx, val in enumerate(phase_rows["mean_drop"]):
        axes[1].text(idx, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    panel_labels(axes)
    fig.suptitle("G5. Compact probe and phase-summary views", y=1.04, fontweight="bold")
    save_dual(fig, "G5_probe_phase_summary")


def write_gallery_doc() -> None:
    lines = [
        "# 可视化画廊（TBME 候选版）",
        "",
        "这套画廊不是主文最终图，而是帮助选择 `TBME` 风格数据图的候选视图。",
        "组织思路参考了 HMIL 的数据图习惯：",
        "- 先给结构清楚的多面板图",
        "- 再给嵌入空间图",
        "- 再给分布图和热图",
        "- 每张图只回答一个明确问题",
        "",
        "## 当前生成的候选图",
        "",
        "1. `G1_embedding_gallery_tsne_pca`",
        "   - 内容：raw scientific model 的 t-SNE / PCA 嵌入视图",
        "   - 用途：适合补充材料，服务“网络确实学到结构”这一句",
        "",
        "2. `G2_feature_distributions_by_depth`",
        "   - 内容：峰值、空间熵、热点半径在不同 coarse depth 下的分布",
        "   - 用途：适合主文或补充材料，作为机制与解释的桥梁",
        "",
        "3. `G3_route_heatmaps`",
        "   - 内容：按 `size x depth` 组织的正确率和深度置信度热图",
        "   - 用途：很适合主文 Fig.8 或补充材料，用于展示部署增强后的系统景观",
        "",
        "4. `G4_probability_structure_views`",
        "   - 内容：`p_deep` 的分布以及幅值-扩散二维结构视图",
        "   - 用途：适合作为 Supplement 的漂亮数据图",
        "",
        "5. `G5_probe_phase_summary`",
        "   - 内容：probe 与 phase occlusion 的紧凑汇总图",
        "   - 用途：适合替换现有 explainability 图中过于拥挤的子图",
        "",
        "## 使用建议",
        "",
        "- `G1` 更像 HMIL Fig.5 的角色：展示表征空间，而不是主结果。",
        "- `G2` 和 `G3` 更适合主文，因为它们兼具物理可解释性和工程说服力。",
        "- `G4` 和 `G5` 更适合作为补充材料或 explainability 备选图。",
    ]
    (OUTPUT_DIR / "VISUALIZATION_GALLERY_CN.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "output_dir": str(OUTPUT_DIR),
        "figures": [
            "G1_embedding_gallery_tsne_pca",
            "G2_feature_distributions_by_depth",
            "G3_route_heatmaps",
            "G4_probability_structure_views",
            "G5_probe_phase_summary",
        ],
        "inputs": {
            "latent_features": str(LATENT_FEATURES_PATH),
            "latent_metadata": str(LATENT_META_PATH),
            "hier_predictions": str(HIER_PRED_PATH),
            "hier_summary": str(HIER_SUMMARY_PATH),
        },
    }
    (OUTPUT_DIR / "visualization_gallery_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    set_style()
    ensure_output()
    latent_features, latent_meta = load_latent_inputs()
    hier, summary = load_hier_inputs()
    plot_embedding_gallery(latent_features, latent_meta)
    plot_feature_distributions(hier)
    plot_route_heatmaps(hier)
    plot_probability_structure_views(hier)
    plot_probe_phase_summary(summary)
    write_gallery_doc()
    write_manifest()
    print(f"Visualization gallery written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
