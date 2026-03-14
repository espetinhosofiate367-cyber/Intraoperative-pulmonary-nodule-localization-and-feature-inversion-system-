from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LATENT_PATH = (
    ROOT.parent
    / "experiments"
    / "depth_model_explainability_v1"
    / "depth_model_latent_features.npy"
)
META_PATH = (
    ROOT.parent
    / "experiments"
    / "depth_model_explainability_v1"
    / "depth_model_positive_windows.csv"
)
OUT_DIR = ROOT / "03_results_core" / "visualization_gallery_v1"

DEPTH_ORDER = ["shallow", "middle", "deep"]
DEPTH_COLORS = {
    "shallow": "#2a9d8f",
    "middle": "#e9c46a",
    "deep": "#e76f51",
}


def set_style():
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def balanced_subset(df: pd.DataFrame, max_per_class: int = 220) -> pd.DataFrame:
    chunks = []
    for depth_name in DEPTH_ORDER:
        part = df[df["coarse_depth"] == depth_name].copy()
        if len(part) > max_per_class:
            part = part.sample(max_per_class, random_state=2026)
        chunks.append(part)
    return pd.concat(chunks, axis=0).sort_values("sample_idx").reset_index(drop=True)


def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features = np.load(LATENT_PATH)
    meta = pd.read_csv(META_PATH)
    meta["sample_idx"] = np.arange(len(meta))
    meta = balanced_subset(meta, max_per_class=220)
    z = features[meta["sample_idx"].to_numpy()]

    scaler = StandardScaler()
    z_std = scaler.fit_transform(z)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.20,
        metric="euclidean",
        random_state=2026,
    )
    emb = reducer.fit_transform(z_std)
    meta["umap_x"] = emb[:, 0]
    meta["umap_y"] = emb[:, 1]

    y = pd.Categorical(meta["coarse_depth"], categories=DEPTH_ORDER, ordered=True).codes
    clf = LogisticRegression(max_iter=3000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    pred = cross_val_predict(clf, z_std, y, cv=cv, method="predict")
    meta["probe_pred"] = pred
    meta["correct"] = pred == y
    probe_acc = float(np.mean(meta["correct"]))

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2))

    for depth_name in DEPTH_ORDER:
        part = meta[meta["coarse_depth"] == depth_name]
        axes[0].scatter(
            part["umap_x"],
            part["umap_y"],
            s=26,
            alpha=0.76,
            color=DEPTH_COLORS[depth_name],
            linewidths=0.0,
            label=depth_name,
        )
        axes[0].text(
            float(part["umap_x"].mean()),
            float(part["umap_y"].mean()),
            depth_name,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
        )
    axes[0].set_title("UMAP colored by true coarse depth")
    axes[0].legend(frameon=False, loc="best", title="Depth")

    sc = axes[1].scatter(
        meta["umap_x"],
        meta["umap_y"],
        c=meta["size_cm"],
        cmap="viridis",
        s=26,
        alpha=0.80,
        linewidths=0.0,
    )
    axes[1].set_title("UMAP colored by nodule size")
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.048, pad=0.04)
    cbar.set_label("Size (cm)")

    correct = meta[meta["correct"]]
    wrong = meta[~meta["correct"]]
    axes[2].scatter(
        correct["umap_x"],
        correct["umap_y"],
        c="#cfd8dc",
        s=22,
        alpha=0.55,
        linewidths=0.0,
        label="correct",
    )
    axes[2].scatter(
        wrong["umap_x"],
        wrong["umap_y"],
        facecolors="none",
        edgecolors="#111111",
        s=52,
        alpha=0.95,
        linewidths=0.8,
        label="misclassified by OOF probe",
    )
    axes[2].set_title(f"UMAP with OOF probe errors (acc = {probe_acc:.3f})")
    axes[2].legend(frameon=False, loc="best")

    for idx, ax in enumerate(axes):
        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle(
        "Clean UMAP triptych for the raw scientific model latent representation",
        y=1.03,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "G1B_clean_umap_triptych_raw_scientific.png", dpi=280, bbox_inches="tight")
    fig.savefig(OUT_DIR / "G1B_clean_umap_triptych_raw_scientific.pdf", dpi=280, bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        "# Clean UMAP Triptych Summary",
        "",
        f"- Sample count: `{len(meta)}`",
        f"- Balanced per-depth subset used for plotting.",
        f"- Out-of-fold linear probe accuracy on the plotted subset: `{probe_acc:.4f}`",
        "- Panel A: true coarse depth",
        "- Panel B: true size",
        "- Panel C: misclassified samples highlighted by out-of-fold probe predictions",
        "",
        "Interpretation boundary:",
        "- This figure supports that the latent space is not random and contains class-related structure.",
        "- It should not be used as the sole proof of classification performance.",
    ]
    (OUT_DIR / "G1B_clean_umap_triptych_raw_scientific.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    selected_sizes = [0.25, 1.0, 1.75]
    fig2, axes2 = plt.subplots(1, 3, figsize=(16.6, 5.2))
    panel_summaries = []
    for ax, size_cm in zip(axes2, selected_sizes):
        part = meta[np.isclose(meta["size_cm"], size_cm)].copy()
        z_part = StandardScaler().fit_transform(features[part["sample_idx"].to_numpy()])
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=24,
            min_dist=0.18,
            metric="euclidean",
            random_state=2026,
        )
        emb_part = reducer.fit_transform(z_part)
        part["ux"] = emb_part[:, 0]
        part["uy"] = emb_part[:, 1]
        y_part = pd.Categorical(part["coarse_depth"], categories=DEPTH_ORDER, ordered=True).codes
        pred_part = cross_val_predict(
            LogisticRegression(max_iter=3000),
            z_part,
            y_part,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2026),
            method="predict",
        )
        acc_part = float(np.mean(pred_part == y_part))
        panel_summaries.append((size_cm, len(part), acc_part))
        part["correct"] = pred_part == y_part
        for depth_name in DEPTH_ORDER:
            cls = part[part["coarse_depth"] == depth_name]
            ax.scatter(
                cls["ux"],
                cls["uy"],
                s=28,
                alpha=0.78,
                color=DEPTH_COLORS[depth_name],
                linewidths=0.0,
                label=depth_name,
            )
        wrong = part[~part["correct"]]
        if not wrong.empty:
            ax.scatter(
                wrong["ux"],
                wrong["uy"],
                s=64,
                facecolors="none",
                edgecolors="#111111",
                linewidths=0.85,
                alpha=0.95,
            )
        ax.set_title(f"Size = {size_cm:.2f} cm\nOOF probe acc = {acc_part:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

    axes2[0].legend(frameon=False, loc="best", title="Depth")
    for idx, ax in enumerate(axes2):
        ax.text(
            -0.12,
            1.04,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig2.suptitle(
        "Size-controlled UMAP views reveal clearer depth structure than the global embedding",
        y=1.03,
        fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "G1C_size_controlled_umap_depth.png", dpi=280, bbox_inches="tight")
    fig2.savefig(OUT_DIR / "G1C_size_controlled_umap_depth.pdf", dpi=280, bbox_inches="tight")
    plt.close(fig2)

    size_lines = [
        "# Size-Controlled UMAP Summary",
        "",
        "Global UMAP was intentionally demoted because the full latent space mixes strong size and weaker depth effects.",
        "The figure below controls the size condition and visualizes depth structure within selected size bands.",
        "",
    ]
    for size_cm, n_count, acc_part in panel_summaries:
        size_lines.append(f"- Size `{size_cm:.2f} cm`: `n={n_count}`, out-of-fold probe accuracy `{acc_part:.4f}`")
    size_lines += [
        "",
        "Interpretation boundary:",
        "- This figure is more defensible than the global UMAP because it respects the known size-depth coupling.",
        "- It still serves as representation evidence rather than a replacement for confusion matrices or balanced accuracy.",
    ]
    (OUT_DIR / "G1C_size_controlled_umap_depth.md").write_text(
        "\n".join(size_lines), encoding="utf-8"
    )

    print("Saved:", OUT_DIR / "G1B_clean_umap_triptych_raw_scientific.png")


if __name__ == "__main__":
    main()
