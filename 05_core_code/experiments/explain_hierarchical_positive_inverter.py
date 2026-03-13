import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)
REPO_ROOT = os.path.dirname(CODE_ARCHIVE_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)
if CODE_ARCHIVE_DIR not in sys.path:
    sys.path.insert(0, CODE_ARCHIVE_DIR)

from depth_analysis_utils import PHASE_ORDER, assign_pressing_phases, frame_physics_features, window_temporal_features
from hierarchical_positive_inverter import HierarchicalPositiveInverter
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM, WINDOW_STRIDE
from train_hierarchical_positive_inverter import PositiveInverterDataset, enrich_positive_samples_with_features
from train_stage3_raw_size_conditioned_depth import (
    balanced_accuracy_from_cm,
    build_positive_depth_samples_for_file,
    compute_raw_scale,
    confusion_matrix_counts,
)
from train_xgboost_baselines import build_records_and_samples_for_file
from triplet_repeat_classifier.train_triplet_repeat_classifier import load_json


FAMILY_MAP = {
    "raw_": "amplitude_response",
    "window_raw_": "amplitude_response",
    "meanframe_norm_raw_": "amplitude_response",
    "maxframe_raw_raw_": "amplitude_response",
    "hotspot_": "spread_extent",
    "second_moment_": "spread_extent",
    "spatial_entropy": "distribution_complexity",
    "center_border_contrast": "shape_contrast",
    "center_mean": "shape_contrast",
    "anisotropy_ratio": "shape_irregularity",
    "peak_count": "shape_irregularity",
    "centroid_": "deformation_position",
    "rise_time_to_peak": "temporal_phase",
    "decay_after_peak": "temporal_phase",
    "peak_persistence_ratio": "temporal_phase",
    "deltaframe_": "temporal_phase",
}

HARD_PAIR_MAX_RAW = 170.0
HARD_PAIR_MAX_ENTROPY = 0.97


def infer_family(feature_name: str) -> str:
    for prefix, family in FAMILY_MAP.items():
        if feature_name.startswith(prefix) or prefix in feature_name:
            return family
    return "other"


def np_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ridge_probe_metrics(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, test_y: np.ndarray) -> Dict[str, float]:
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return {
        "r2": float(r2_score(test_y, pred)),
        "pearson_r": np_corr(pred.reshape(-1), test_y.reshape(-1)),
    }


def ridge_probe_predict(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, test_y: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    model = Ridge(alpha=1.0)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    metrics = {
        "r2": float(r2_score(test_y, pred)),
        "pearson_r": np_corr(pred.reshape(-1), test_y.reshape(-1)),
    }
    return metrics, np.asarray(pred, dtype=np.float32)


def size_norm_to_cm(size_norm: np.ndarray) -> np.ndarray:
    size_norm = np.asarray(size_norm, dtype=np.float32)
    lo = float(min(SIZE_VALUES_CM))
    hi = float(max(SIZE_VALUES_CM))
    return lo + np.clip(size_norm, 0.0, 1.0) * (hi - lo)


def compute_window_summary(raw_window: np.ndarray) -> Dict[str, float]:
    frame_rows = [frame_physics_features(frame.astype(np.float32)) for frame in raw_window]
    temporal = window_temporal_features(frame_rows)
    return {
        "raw_max_max": float(max(fr["raw_max"] for fr in frame_rows)),
        "spatial_entropy_max": float(max(fr["spatial_entropy"] for fr in frame_rows)),
        "hotspot_radius_max": float(max(fr["hotspot_radius"] for fr in frame_rows)),
        "center_border_contrast_max": float(max(fr["center_border_contrast"] for fr in frame_rows)),
        "rise_time_to_peak": float(temporal["rise_time_to_peak"]),
        "decay_after_peak": float(temporal["decay_after_peak"]),
        "window_raw_sum_gain": float(temporal["window_raw_sum_gain"]),
        "peak_persistence_ratio": float(temporal["peak_persistence_ratio"]),
    }


def normalize_frame_for_display(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    lo = float(np.percentile(frame, 1.0))
    hi = float(np.percentile(frame, 99.0))
    if hi - lo < 1e-6:
        lo = float(np.min(frame))
        hi = float(np.max(frame))
    if hi - lo < 1e-6:
        return np.zeros_like(frame, dtype=np.float32)
    norm = (frame - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def frame_display_quality(frame: np.ndarray) -> float:
    norm = normalize_frame_for_display(frame)
    center = float(norm[3:9, 2:6].mean())
    border_pixels = np.concatenate(
        [
            norm[:2, :].reshape(-1),
            norm[-2:, :].reshape(-1),
            norm[2:-2, :2].reshape(-1),
            norm[2:-2, -2:].reshape(-1),
        ]
    )
    border = float(border_pixels.mean()) if border_pixels.size else 0.0
    hot_area = float((norm >= 0.82).mean())
    compact_term = -abs(hot_area - 0.18)
    return center - 0.65 * border + 0.35 * float(norm.max()) + 0.45 * compact_term


def choose_display_frame(raw_window: np.ndarray) -> Tuple[np.ndarray, float]:
    best_idx = 0
    best_score = float("-inf")
    best_frame = normalize_frame_for_display(raw_window[0])
    for idx, frame in enumerate(raw_window):
        score = frame_display_quality(frame)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_frame = normalize_frame_for_display(frame)
    return best_frame, float(best_score)


def depth_metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = np.argmax(probs, axis=1).astype(np.int32)
    cm = confusion_matrix_counts(labels.astype(np.int32), pred, len(COARSE_DEPTH_ORDER))
    return {
        "accuracy": float(np.mean(pred == labels)),
        "balanced_accuracy": float(balanced_accuracy_from_cm(cm)),
        "confusion_matrix": cm.tolist(),
        "pred": pred,
    }


def plot_probe_top_features(df: pd.DataFrame, output_path: str, top_k: int = 12):
    use_df = df.sort_values("hybrid_r2", ascending=False).head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(use_df))
    ax.barh(y + 0.22, use_df["hybrid_r2"], height=0.22, label="hybrid latent", color="#1f77b4")
    ax.barh(y, use_df["trunk_r2"], height=0.22, label="raw trunk", color="#ff7f0e")
    ax.barh(y - 0.22, use_df["size_only_r2"], height=0.22, label="size-only", color="#7f7f7f")
    ax.set_yticks(y)
    ax.set_yticklabels(use_df["feature"])
    ax.set_xlabel("Test $R^2$")
    ax.set_title("Hierarchical Inverter Latent Probe Recovery")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_probe_family(df: pd.DataFrame, output_path: str):
    fam = (
        df.groupby("family", as_index=False)[["hybrid_r2", "trunk_r2", "size_only_r2"]]
        .mean()
        .sort_values("hybrid_r2", ascending=False)
    )
    x = np.arange(len(fam))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.22, fam["hybrid_r2"], width=0.22, label="hybrid latent", color="#1f77b4")
    ax.bar(x, fam["trunk_r2"], width=0.22, label="raw trunk", color="#ff7f0e")
    ax.bar(x + 0.22, fam["size_only_r2"], width=0.22, label="size-only", color="#7f7f7f")
    ax.set_xticks(x)
    ax.set_xticklabels(fam["family"], rotation=20)
    ax.set_ylabel("Mean test $R^2$")
    ax.set_title("Probe Performance by Feature Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_probe_scatter(rows: List[Tuple[str, np.ndarray, np.ndarray]], output_path: str):
    n = min(len(rows), 4)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7))
    axes = axes.ravel()
    for ax, (name, truth, pred) in zip(axes[:n], rows[:n]):
        ax.scatter(truth, pred, s=12, alpha=0.55, color="#1f77b4")
        lo = float(min(np.min(truth), np.min(pred)))
        hi = float(max(np.max(truth), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], "--", color="#d62728", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("True")
        ax.set_ylabel("Probe")
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def forward_variant(
    model: HierarchicalPositiveInverter,
    raw_x: torch.Tensor,
    norm_x: torch.Tensor,
    feat_x: torch.Tensor,
    variant: str = "full",
) -> Dict[str, torch.Tensor]:
    raw_seq = model.amplitude_encoder(raw_x)
    shape_seq = model.shape_encoder(norm_x)
    delta_seq = model.delta_encoder(model.compute_delta(norm_x))

    if variant == "no_raw":
        raw_seq = torch.zeros_like(raw_seq)
    elif variant == "no_shape":
        shape_seq = torch.zeros_like(shape_seq)
    elif variant == "no_delta":
        delta_seq = torch.zeros_like(delta_seq)

    seq = torch.cat([raw_seq, shape_seq, delta_seq], dim=-1).transpose(1, 2)
    seq = model.temporal_input(seq)
    for block in model.temporal_blocks:
        seq = block(seq)
    temporal_seq = seq.transpose(1, 2)
    global_feat, attn_weights = model.global_pooling(temporal_seq)
    phase_feat, phase_masks = model.phase_pooling(temporal_seq, raw_x)
    fused = torch.cat([global_feat, phase_feat], dim=1)
    trunk_feat = model.trunk(fused)

    tabular_feat = model.tabular_branch(feat_x)
    tabular_proj = model.tabular_to_trunk(tabular_feat)
    if variant == "no_tabular":
        tabular_proj = torch.zeros_like(tabular_proj)
        tabular_feat = torch.zeros_like(tabular_feat)

    hybrid_feat = model.fusion(torch.cat([trunk_feat, tabular_proj, trunk_feat * tabular_proj], dim=1))
    depth_feat = model.depth_adapter(hybrid_feat)
    size_logits = model.size_cls_head(hybrid_feat)
    size_ord_logits = model.size_ord_head(hybrid_feat)
    size_probs = torch.softmax(size_logits, dim=1)
    size_values = torch.linspace(0.0, 1.0, steps=model.num_size_classes, device=size_probs.device, dtype=size_probs.dtype)
    expected_norm = torch.sum(size_probs * size_values.view(1, -1), dim=1, keepdim=True)
    residual = 0.35 * torch.tanh(model.size_residual_head(hybrid_feat))
    size_reg_norm = torch.clamp(expected_norm + residual, 0.0, 1.0)
    depth_logits_gt = model.route_depth_logits(depth_feat, torch.argmax(size_probs, dim=1))
    depth_logits_hard = model.route_depth_logits(depth_feat, torch.argmax(size_probs, dim=1))
    depth_probs_hard = torch.softmax(depth_logits_hard, dim=1)
    return {
        "trunk_feat": trunk_feat,
        "hybrid_feat": hybrid_feat,
        "size_logits": size_logits,
        "size_ord_logits": size_ord_logits,
        "size_probs": size_probs,
        "size_reg_norm": size_reg_norm,
        "depth_logits_hard": depth_logits_hard,
        "depth_probs_hard": depth_probs_hard,
        "attn_weights": attn_weights,
        "phase_masks": phase_masks,
    }


def collect_latents_and_predictions(
    model: HierarchicalPositiveInverter,
    dataset: PositiveInverterDataset,
    device: torch.device,
    variant: str = "full",
) -> Dict[str, object]:
    trunk_rows, hybrid_rows = [], []
    size_idx_rows, depth_idx_rows = [], []
    feature_rows, meta_rows = [], []
    depth_prob_rows, size_prob_rows = [], []
    size_reg_rows = []
    raw_window_rows = []
    for idx, sample in enumerate(dataset.samples):
        raw_x, norm_x, feat_x, size_idx_t, size_cm_t, _size_norm, depth_idx_t, _phase_weight = dataset[idx]
        raw_x_b = raw_x.unsqueeze(0).to(device)
        norm_x_b = norm_x.unsqueeze(0).to(device)
        feat_x_b = feat_x.unsqueeze(0).to(device)
        with torch.no_grad():
            out = forward_variant(model, raw_x_b, norm_x_b, feat_x_b, variant=variant)
        trunk_rows.append(out["trunk_feat"][0].cpu().numpy().astype(np.float32))
        hybrid_rows.append(out["hybrid_feat"][0].cpu().numpy().astype(np.float32))
        size_idx_rows.append(int(size_idx_t.item()))
        depth_idx_rows.append(int(depth_idx_t.item()))
        feature_rows.append(np.asarray(sample["selected_features"], dtype=np.float32))
        size_prob_rows.append(out["size_probs"][0].cpu().numpy().astype(np.float32))
        depth_prob_rows.append(out["depth_probs_hard"][0].cpu().numpy().astype(np.float32))
        size_reg_rows.append(float(size_norm_to_cm(out["size_reg_norm"][0, 0].cpu().numpy())))
        raw_window = raw_x.numpy()[:, 0].astype(np.float32) * float(dataset.raw_scale)
        raw_window_rows.append(raw_window)
        meta_rows.append(
            {
                "sample_index": idx,
                "group_key": sample["group_key"],
                "base_group": sample["base_group"],
                "size_class_index": int(sample["size_class_index"]),
                "size_cm": float(size_cm_t.item()),
                "depth_coarse_index": int(depth_idx_t.item()),
                "depth_cm": float(dataset.records_by_key[sample["group_key"]]["depth_cm"]),
            }
        )
    return {
        "trunk_feat": np.asarray(trunk_rows, dtype=np.float32),
        "hybrid_feat": np.asarray(hybrid_rows, dtype=np.float32),
        "size_idx": np.asarray(size_idx_rows, dtype=np.int32),
        "depth_idx": np.asarray(depth_idx_rows, dtype=np.int32),
        "selected_features": np.asarray(feature_rows, dtype=np.float32),
        "size_probs": np.asarray(size_prob_rows, dtype=np.float32),
        "size_reg_cm": np.asarray(size_reg_rows, dtype=np.float32),
        "depth_probs": np.asarray(depth_prob_rows, dtype=np.float32),
        "raw_windows": raw_window_rows,
        "meta": pd.DataFrame(meta_rows),
    }


def plot_branch_ablation(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    use = df.copy()
    x = np.arange(len(use))
    axes[0].bar(x - 0.18, use["size_top1"], width=0.36, label="size top1", color="#1f77b4")
    axes[0].bar(x + 0.18, use["depth_hard_balanced_accuracy"], width=0.36, label="depth hard bAcc", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(use["variant"], rotation=20)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Branch ablation performance")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - 0.18, use["size_mae"], width=0.36, label="size MAE", color="#2ca02c")
    axes[1].bar(x + 0.18, use["depth_accuracy"], width=0.36, label="depth acc", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(use["variant"], rotation=20)
    axes[1].set_title("Secondary metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_hard_pairs(pair_rows: List[Dict[str, object]], output_path: str):
    if not pair_rows:
        return
    n = min(len(pair_rows), 6)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.asarray([axes])
    for row_idx, pair in enumerate(pair_rows[:n]):
        for col_idx, side in enumerate(["deep", "shallow"]):
            ax = axes[row_idx, col_idx]
            ax.imshow(pair[f"{side}_peak_frame"], cmap="turbo", interpolation="bicubic", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{side}: S={pair[f'{side}_size_cm']:.2f} D={pair[f'{side}_depth_cm']:.1f}\n"
                f"pred={pair[f'{side}_pred_label']} p_deep={pair[f'{side}_p_deep']:.2f}\n"
                f"max={pair[f'{side}_raw_max_max']:.1f} ent={pair[f'{side}_spatial_entropy_max']:.2f}"
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_phase_occlusion(phase_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    overall = phase_df.groupby("phase", as_index=False)["true_prob_drop"].mean()
    axes[0].bar(overall["phase"], overall["true_prob_drop"], color="#4C78A8")
    axes[0].set_title("Mean true-class probability drop")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)

    by_depth = phase_df.groupby(["depth_label", "phase"], as_index=False)["true_prob_drop"].mean()
    x = np.arange(len(PHASE_ORDER))
    width = 0.22
    for idx, depth_label in enumerate(COARSE_DEPTH_ORDER):
        sub = by_depth[by_depth["depth_label"] == depth_label].set_index("phase").reindex(PHASE_ORDER).reset_index()
        axes[1].bar(x + (idx - 1) * width, sub["true_prob_drop"], width=width, label=depth_label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(PHASE_ORDER, rotation=20)
    axes[1].set_title("Phase occlusion by true depth")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

@dataclass
class Config:
    checkpoint_path: str = os.environ.get(
        "PAPERHPINV_EXPLAIN_CKPT",
        os.path.join(
            PROJECT_DIR,
            "experiments",
            "outputs_hierarchical_positive_inverter_run1",
            "paper_hierarchical_positive_inverter_best.pth",
        ),
    )
    output_dir: str = os.environ.get(
        "PAPERHPINV_EXPLAIN_OUT",
        os.path.join(
            PROJECT_DIR,
            "experiments",
            "outputs_hierarchical_positive_inverter_explainability_v1",
        ),
    )
    data_root: str = os.environ.get(
        "PAPERHPINV_EXPLAIN_DATA_ROOT",
        os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"),
    )
    file1_labels: str = os.environ.get(
        "PAPERHPINV_EXPLAIN_FILE1_LABELS",
        os.path.join(REPO_ROOT, "manual_keyframe_labels.json"),
    )
    file2_labels: str = os.environ.get(
        "PAPERHPINV_EXPLAIN_FILE2_LABELS",
        os.path.join(data_root, "manual_keyframe_labels_file2.json"),
    )
    file3_labels: str = os.environ.get(
        "PAPERHPINV_EXPLAIN_FILE3_LABELS",
        os.path.join(data_root, "manual_keyframe_labels_file3.json"),
    )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    pos_rec1, pos_samples1 = build_positive_depth_samples_for_file(file1_all, "1.CSV", cfg.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6)
    pos_rec2, pos_samples2 = build_positive_depth_samples_for_file(file2_all, "2.CSV", cfg.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6)
    pos_rec3, pos_samples3 = build_positive_depth_samples_for_file(file3_all, "3.CSV", cfg.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6)

    xgb_rec1, _ = build_records_and_samples_for_file(file1_all, "1.CSV", cfg.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6)
    xgb_rec2, _ = build_records_and_samples_for_file(file2_all, "2.CSV", cfg.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6)
    xgb_rec3, _ = build_records_and_samples_for_file(file3_all, "3.CSV", cfg.data_root, INPUT_SEQ_LEN, WINDOW_STRIDE, 6)

    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in pos_rec1.values())
            & set(v["base_group"] for v in pos_rec2.values())
            & set(v["base_group"] for v in pos_rec3.values())
        )
    )
    common_set = set(common_base_groups)
    train_records = {k: v for k, v in pos_rec1.items() if v["base_group"] in common_set}
    val_records = {k: v for k, v in pos_rec2.items() if v["base_group"] in common_set}
    test_records = {k: v for k, v in pos_rec3.items() if v["base_group"] in common_set}
    train_samples = [s for s in pos_samples1 if s["base_group"] in common_set]
    val_samples = [s for s in pos_samples2 if s["base_group"] in common_set]
    test_samples = [s for s in pos_samples3 if s["base_group"] in common_set]

    selected_features = list(ckpt.get("selected_features", []))
    if not selected_features:
        raise ValueError("No selected_features found in hierarchical inverter checkpoint.")
    train_samples = enrich_positive_samples_with_features(train_records, train_samples, xgb_rec1, selected_features, "train")
    val_samples = enrich_positive_samples_with_features(val_records, val_samples, xgb_rec2, selected_features, "val")
    test_samples = enrich_positive_samples_with_features(test_records, test_samples, xgb_rec3, selected_features, "test")

    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    raw_scale = float(ckpt.get("raw_scale", compute_raw_scale(train_records)))

    ds_train = PositiveInverterDataset(train_records, train_samples, raw_scale, feature_mean, feature_std, False)
    ds_val = PositiveInverterDataset(val_records, val_samples, raw_scale, feature_mean, feature_std, False)
    ds_test = PositiveInverterDataset(test_records, test_samples, raw_scale, feature_mean, feature_std, False)

    model_cfg = ckpt.get("model_config", {})
    model = HierarchicalPositiveInverter(
        seq_len=int(model_cfg.get("seq_len", INPUT_SEQ_LEN)),
        frame_feature_dim=int(model_cfg.get("frame_feature_dim", 24)),
        temporal_channels=int(model_cfg.get("temporal_channels", 48)),
        temporal_blocks=int(model_cfg.get("temporal_blocks", 3)),
        dropout=float(model_cfg.get("dropout", 0.28)),
        num_size_classes=int(model_cfg.get("num_size_classes", len(SIZE_VALUES_CM))),
        num_depth_classes=int(model_cfg.get("num_depth_classes", len(COARSE_DEPTH_ORDER))),
        num_tabular_features=int(model_cfg.get("num_tabular_features", len(selected_features))),
        tabular_hidden_dim=int(model_cfg.get("tabular_hidden_dim", 64)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_pack = collect_latents_and_predictions(model, ds_train, device, variant="full")
    test_pack = collect_latents_and_predictions(model, ds_test, device, variant="full")

    train_size_onehot = np.eye(len(SIZE_VALUES_CM), dtype=np.float32)[train_pack["size_idx"]]
    test_size_onehot = np.eye(len(SIZE_VALUES_CM), dtype=np.float32)[test_pack["size_idx"]]
    probe_rows = []
    scatter_rows = []
    for feat_idx, feat_name in enumerate(selected_features):
        y_train = train_pack["selected_features"][:, feat_idx].astype(np.float32)
        y_test = test_pack["selected_features"][:, feat_idx].astype(np.float32)
        hybrid_metrics, hybrid_pred = ridge_probe_predict(train_pack["hybrid_feat"], test_pack["hybrid_feat"], y_train, y_test)
        trunk_metrics, trunk_pred = ridge_probe_predict(train_pack["trunk_feat"], test_pack["trunk_feat"], y_train, y_test)
        size_metrics, _ = ridge_probe_predict(train_size_onehot, test_size_onehot, y_train, y_test)
        probe_rows.append(
            {
                "feature": feat_name,
                "family": infer_family(feat_name),
                "hybrid_r2": hybrid_metrics["r2"],
                "hybrid_pearson_r": hybrid_metrics["pearson_r"],
                "trunk_r2": trunk_metrics["r2"],
                "trunk_pearson_r": trunk_metrics["pearson_r"],
                "size_only_r2": size_metrics["r2"],
                "size_only_pearson_r": size_metrics["pearson_r"],
            }
        )
        scatter_rows.append((feat_name, y_test, trunk_pred))

    probe_df = pd.DataFrame(probe_rows).sort_values("hybrid_r2", ascending=False).reset_index(drop=True)
    probe_df.to_csv(os.path.join(cfg.output_dir, "hierarchical_probe_metrics.csv"), index=False, encoding="utf-8-sig")
    plot_probe_top_features(probe_df, os.path.join(cfg.output_dir, "hierarchical_probe_hybrid_top_features.png"))
    plot_probe_family(probe_df, os.path.join(cfg.output_dir, "hierarchical_probe_family_comparison.png"))
    top_scatter_features = probe_df.sort_values("trunk_r2", ascending=False)["feature"].head(4).tolist()
    scatter_payload = []
    for feat_name in top_scatter_features:
        for name, truth, pred in scatter_rows:
            if name == feat_name:
                scatter_payload.append((feat_name, truth, pred))
                break
    plot_probe_scatter(scatter_payload, os.path.join(cfg.output_dir, "hierarchical_probe_scatter_top4.png"))

    ablation_rows = []
    for variant in ["full", "no_raw", "no_shape", "no_delta", "no_tabular"]:
        pack = test_pack if variant == "full" else collect_latents_and_predictions(model, ds_test, device, variant=variant)
        size_pred = np.argmax(pack["size_probs"], axis=1).astype(np.int32)
        depth_metrics = depth_metrics_from_probs(pack["depth_probs"], pack["depth_idx"])
        ablation_rows.append(
            {
                "variant": variant,
                "size_top1": float(np.mean(size_pred == pack["size_idx"])),
                "size_mae": float(np.mean(np.abs(pack["size_reg_cm"] - pack["meta"]["size_cm"].to_numpy(dtype=np.float32)))),
                "depth_accuracy": float(depth_metrics["accuracy"]),
                "depth_hard_balanced_accuracy": float(depth_metrics["balanced_accuracy"]),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(cfg.output_dir, "hierarchical_branch_ablation.csv"), index=False, encoding="utf-8-sig")
    plot_branch_ablation(ablation_df, os.path.join(cfg.output_dir, "hierarchical_branch_ablation.png"))

    sample_df = test_pack["meta"].copy()
    sample_df["pred_label"] = [COARSE_DEPTH_ORDER[i] for i in np.argmax(test_pack["depth_probs"], axis=1)]
    sample_df["p_shallow"] = test_pack["depth_probs"][:, 0]
    sample_df["p_middle"] = test_pack["depth_probs"][:, 1]
    sample_df["p_deep"] = test_pack["depth_probs"][:, 2]
    sample_df["size_pred_idx"] = np.argmax(test_pack["size_probs"], axis=1).astype(np.int32)
    sample_df["size_pred_cm"] = [SIZE_VALUES_CM[i] for i in sample_df["size_pred_idx"]]
    sample_df["size_conf"] = np.max(test_pack["size_probs"], axis=1)
    summaries = [compute_window_summary(w) for w in test_pack["raw_windows"]]
    for key in summaries[0].keys():
        sample_df[key] = [row[key] for row in summaries]
    display_frames = []
    display_scores = []
    for raw_window in test_pack["raw_windows"]:
        frame, score = choose_display_frame(raw_window)
        display_frames.append(frame)
        display_scores.append(score)
    sample_df["peak_frame"] = display_frames
    sample_df["display_score"] = display_scores
    sample_df["depth_conf"] = np.max(test_pack["depth_probs"], axis=1)
    sample_df["true_label"] = [COARSE_DEPTH_ORDER[i] for i in sample_df["depth_coarse_index"]]
    sample_df["correct"] = sample_df["pred_label"] == sample_df["true_label"]
    sample_df["quality_ok"] = (
        (sample_df["raw_max_max"] < HARD_PAIR_MAX_RAW)
        & (sample_df["spatial_entropy_max"] < HARD_PAIR_MAX_ENTROPY)
        & (sample_df["center_border_contrast_max"] > 0.0)
    )
    reps = (
        sample_df.sort_values(
            ["quality_ok", "correct", "depth_conf", "display_score"],
            ascending=[False, False, False, False],
        )
        .groupby("base_group", as_index=False)
        .first()
    )
    deep_reps = (
        reps[(reps["depth_coarse_index"] == 2) & (reps["correct"]) & (reps["quality_ok"])]
        .sort_values(["size_cm", "depth_conf"], ascending=[False, False])
        .reset_index(drop=True)
    )
    shallow_reps = (
        reps[(reps["depth_coarse_index"] == 0) & (reps["correct"]) & (reps["quality_ok"])]
        .sort_values(["size_cm", "depth_conf"], ascending=[True, False])
        .reset_index(drop=True)
    )
    hard_pairs = []
    for _, deep_row in deep_reps.iterrows():
        candidates = shallow_reps[shallow_reps["size_cm"] < deep_row["size_cm"]].copy()
        if len(candidates) == 0:
            continue
        candidates["raw_gap"] = np.abs(candidates["raw_max_max"] - deep_row["raw_max_max"])
        candidates["size_gap"] = deep_row["size_cm"] - candidates["size_cm"]
        candidates["entropy_gap"] = np.abs(candidates["spatial_entropy_max"] - deep_row["spatial_entropy_max"])
        candidates["radius_gap"] = np.abs(candidates["hotspot_radius_max"] - deep_row["hotspot_radius_max"])
        candidates["contrast_gap_abs"] = np.abs(
            candidates["center_border_contrast_max"] - deep_row["center_border_contrast_max"]
        )
        candidates["pair_score"] = (
            1.0 * candidates["raw_gap"]
            + 15.0 * candidates["size_gap"]
            + 20.0 * candidates["entropy_gap"]
            + 4.0 * candidates["radius_gap"]
        )
        best = candidates.sort_values(
            ["pair_score", "raw_gap", "size_gap", "display_score"],
            ascending=[True, True, True, False],
        ).iloc[0]
        hard_pairs.append(
            {
                "deep_group": deep_row["base_group"],
                "shallow_group": best["base_group"],
                "deep_size_cm": float(deep_row["size_cm"]),
                "shallow_size_cm": float(best["size_cm"]),
                "deep_depth_cm": float(deep_row["depth_cm"]),
                "shallow_depth_cm": float(best["depth_cm"]),
                "deep_pred_label": deep_row["pred_label"],
                "shallow_pred_label": best["pred_label"],
                "deep_p_deep": float(deep_row["p_deep"]),
                "shallow_p_deep": float(best["p_deep"]),
                "deep_raw_max_max": float(deep_row["raw_max_max"]),
                "shallow_raw_max_max": float(best["raw_max_max"]),
                "deep_spatial_entropy_max": float(deep_row["spatial_entropy_max"]),
                "shallow_spatial_entropy_max": float(best["spatial_entropy_max"]),
                "deep_hotspot_radius_max": float(deep_row["hotspot_radius_max"]),
                "shallow_hotspot_radius_max": float(best["hotspot_radius_max"]),
                "deep_center_border_contrast_max": float(deep_row["center_border_contrast_max"]),
                "shallow_center_border_contrast_max": float(best["center_border_contrast_max"]),
                "pair_score_raw_max_gap": float(abs(deep_row["raw_max_max"] - best["raw_max_max"])),
                "pair_score_total": float(best["pair_score"]),
                "model_prefers_deeper_on_pdeep": bool(float(deep_row["p_deep"]) > float(best["p_deep"])),
                "entropy_gap": float(deep_row["spatial_entropy_max"] - best["spatial_entropy_max"]),
                "radius_gap": float(deep_row["hotspot_radius_max"] - best["hotspot_radius_max"]),
                "contrast_gap": float(deep_row["center_border_contrast_max"] - best["center_border_contrast_max"]),
                "deep_peak_frame": deep_row["peak_frame"],
                "shallow_peak_frame": best["peak_frame"],
            }
        )
    hard_pairs = sorted(
        hard_pairs,
        key=lambda row: (
            row["pair_score_total"],
            -float(row["model_prefers_deeper_on_pdeep"]),
            row["pair_score_raw_max_gap"],
        ),
    )
    display_pairs = [
        row for row in hard_pairs if row["deep_p_deep"] >= 0.90 and row["shallow_p_deep"] <= 0.40
    ]
    if len(display_pairs) < 6:
        used_keys = {(row["deep_group"], row["shallow_group"]) for row in display_pairs}
        for row in hard_pairs:
            key = (row["deep_group"], row["shallow_group"])
            if key in used_keys:
                continue
            display_pairs.append(row)
            used_keys.add(key)
            if len(display_pairs) >= 6:
                break
    pd.DataFrame(hard_pairs).to_csv(
        os.path.join(cfg.output_dir, "hierarchical_hard_pairs.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    plot_hard_pairs(display_pairs, os.path.join(cfg.output_dir, "hierarchical_hard_pair_examples.png"))

    phase_rows = []
    for idx, sample in enumerate(ds_test.samples):
        raw_x, norm_x, feat_x, _size_idx, _size_cm, _size_norm, depth_idx_t, _phase_weight = ds_test[idx]
        raw_b = raw_x.unsqueeze(0).to(device)
        norm_b = norm_x.unsqueeze(0).to(device)
        feat_b = feat_x.unsqueeze(0).to(device)
        depth_idx = int(depth_idx_t.item())
        with torch.no_grad():
            full_out = forward_variant(model, raw_b, norm_b, feat_b, variant="full")
        base_true_prob = float(full_out["depth_probs_hard"][0, depth_idx].item())
        raw_window = test_pack["raw_windows"][idx]
        phase_names = assign_pressing_phases(raw_window.sum(axis=(1, 2)))["phase_name"]
        for phase in PHASE_ORDER:
            phase_mask = (phase_names == phase).astype(np.float32).reshape(-1, 1, 1)
            if phase_mask.sum() <= 0:
                continue
            phase_mask_t = torch.from_numpy(phase_mask).to(device).unsqueeze(0).unsqueeze(2)
            with torch.no_grad():
                occ_out = forward_variant(model, raw_b * (1.0 - phase_mask_t), norm_b * (1.0 - phase_mask_t), feat_b, variant="full")
            true_prob_occ = float(occ_out["depth_probs_hard"][0, depth_idx].item())
            phase_rows.append(
                {
                    "sample_index": idx,
                    "depth_label": COARSE_DEPTH_ORDER[depth_idx],
                    "phase": phase,
                    "true_prob_base": base_true_prob,
                    "true_prob_occ": true_prob_occ,
                    "true_prob_drop": base_true_prob - true_prob_occ,
                }
            )
    phase_df = pd.DataFrame(phase_rows)
    phase_df.to_csv(os.path.join(cfg.output_dir, "hierarchical_phase_occlusion_table.csv"), index=False, encoding="utf-8-sig")
    plot_phase_occlusion(phase_df, os.path.join(cfg.output_dir, "hierarchical_phase_occlusion.png"))

    sample_df.drop(columns=["peak_frame"]).to_csv(
        os.path.join(cfg.output_dir, "hierarchical_test_window_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    hard_pair_summary = {
        "pair_count": int(len(hard_pairs)),
        "pairwise_pdeep_success_rate": float(np.mean([p["model_prefers_deeper_on_pdeep"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_raw_max_gap": float(np.mean([p["pair_score_raw_max_gap"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_total_pair_score": float(np.mean([p["pair_score_total"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_entropy_gap": float(np.mean([p["entropy_gap"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_radius_gap": float(np.mean([p["radius_gap"] for p in hard_pairs])) if hard_pairs else float("nan"),
    }
    phase_summary = (
        phase_df.groupby("phase", as_index=False)["true_prob_drop"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_drop", "median": "median_drop"})
    )
    probe_summary = {
        "mean_hybrid_r2": float(probe_df["hybrid_r2"].mean()),
        "mean_trunk_r2": float(probe_df["trunk_r2"].mean()),
        "mean_size_only_r2": float(probe_df["size_only_r2"].mean()),
        "mean_hybrid_pearson_r": float(probe_df["hybrid_pearson_r"].mean()),
        "mean_trunk_pearson_r": float(probe_df["trunk_pearson_r"].mean()),
        "mean_size_only_pearson_r": float(probe_df["size_only_pearson_r"].mean()),
    }

    summary = {
        "checkpoint": cfg.checkpoint_path,
        "selected_features": selected_features,
        "probe_summary": probe_summary,
        "branch_ablation": ablation_df.to_dict(orient="records"),
        "hard_pair_summary": hard_pair_summary,
        "phase_occlusion_summary": phase_summary.to_dict(orient="records"),
    }
    with open(os.path.join(cfg.output_dir, "hierarchical_explainability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Hierarchical Positive Inverter Explainability Report")
    lines.append("")
    lines.append("## Probe Summary")
    lines.append(f"- Mean hybrid latent test R2: `{probe_summary['mean_hybrid_r2']:.4f}`")
    lines.append(f"- Mean raw trunk test R2: `{probe_summary['mean_trunk_r2']:.4f}`")
    lines.append(f"- Mean size-only test R2: `{probe_summary['mean_size_only_r2']:.4f}`")
    lines.append("")
    lines.append("## Branch Ablation")
    for row in ablation_rows:
        lines.append(
            f"- `{row['variant']}`: size_top1={row['size_top1']:.4f}, "
            f"size_mae={row['size_mae']:.4f}, depth_bAcc={row['depth_hard_balanced_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("## Hard-Pair Summary")
    lines.append(
        f"- Clean pair filters: `raw_max_max < {HARD_PAIR_MAX_RAW:.0f}`, "
        f"`spatial_entropy_max < {HARD_PAIR_MAX_ENTROPY:.2f}`, and both sides correctly classified."
    )
    lines.append(f"- Pair count: `{hard_pair_summary['pair_count']}`")
    lines.append(f"- Deeper sample higher p_deep rate: `{hard_pair_summary['pairwise_pdeep_success_rate']:.4f}`")
    lines.append(f"- Mean raw_max gap: `{hard_pair_summary['mean_raw_max_gap']:.4f}`")
    lines.append("")
    lines.append("## Phase Occlusion")
    top_phase = phase_summary.sort_values("mean_drop", ascending=False).iloc[0]
    lines.append(
        f"- Largest true-class probability drop came from `{top_phase['phase']}` "
        f"with mean drop `{float(top_phase['mean_drop']):.4f}`."
    )
    with open(os.path.join(cfg.output_dir, "HIERARCHICAL_EXPLAINABILITY_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Hierarchical inverter explainability complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
