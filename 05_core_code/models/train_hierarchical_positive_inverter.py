import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_DIR, "experiments")
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)
REPO_ROOT = os.path.dirname(CODE_ARCHIVE_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
if CODE_ARCHIVE_DIR not in sys.path:
    sys.path.insert(0, CODE_ARCHIVE_DIR)

from hierarchical_positive_inverter import HierarchicalPositiveInverter
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, INPUT_SHAPE, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
from train_stage2_raw_hybrid_positive_size import (
    gaussian_neighbor_targets,
    load_selected_feature_names,
    ordinal_loss,
    size_cm_to_norm,
    size_norm_to_cm,
)
from train_stage3_raw_size_conditioned_depth import (
    balanced_accuracy_from_cm,
    build_positive_depth_samples_for_file,
    compute_raw_scale,
    confusion_matrix_counts,
)
from train_xgboost_baselines import build_feature_table, build_records_and_samples_for_file, regression_metrics, topk_accuracy
from triplet_repeat_classifier.train_triplet_repeat_classifier import load_json, set_seed


def sample_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    loss = F.cross_entropy(
        logits,
        labels.long(),
        weight=class_weight,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    if sample_weight is not None:
        loss = loss * sample_weight.view(-1)
        return loss.sum() / torch.clamp(sample_weight.sum(), min=1.0)
    return loss.mean()


def truncate_topk_probs(size_probs: torch.Tensor, k: int = 2) -> torch.Tensor:
    k = int(max(1, min(k, size_probs.shape[1])))
    if k >= size_probs.shape[1]:
        return size_probs
    top_idx = torch.argsort(size_probs, dim=1, descending=True)[:, :k]
    truncated = torch.zeros_like(size_probs)
    truncated.scatter_(1, top_idx, size_probs.gather(1, top_idx))
    truncated = truncated / torch.clamp(truncated.sum(dim=1, keepdim=True), min=1e-8)
    return truncated


def depth_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    pred = np.argmax(logits, axis=1).astype(np.int32)
    cm = confusion_matrix_counts(labels.astype(np.int32), pred, len(COARSE_DEPTH_ORDER))
    return {
        "accuracy": float(np.mean(pred == labels)),
        "balanced_accuracy": float(balanced_accuracy_from_cm(cm)),
        "confusion_matrix": cm.tolist(),
        "pred": pred,
    }


class PositiveInverterDataset(Dataset):
    def __init__(
        self,
        records_by_key: Dict[str, dict],
        sample_records: List[dict],
        raw_scale: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        is_train: bool = False,
        aug_noise_std: float = 0.0,
        aug_scale_jitter: float = 0.0,
        aug_frame_dropout: float = 0.0,
        feat_noise_std: float = 0.0,
    ):
        self.records_by_key = records_by_key
        self.samples = sample_records
        self.raw_scale = float(max(raw_scale, 1e-6))
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = np.maximum(feature_std.astype(np.float32), 1e-6)
        self.is_train = bool(is_train)
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_scale_jitter = float(max(0.0, aug_scale_jitter))
        self.aug_frame_dropout = float(min(max(0.0, aug_frame_dropout), 0.5))
        self.feat_noise_std = float(max(0.0, feat_noise_std))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rec = self.records_by_key[s["group_key"]]
        end_row = int(s["end_row"])
        seq_len = int(rec["seq_len"])
        st = end_row - seq_len + 1
        raw_x = rec["raw_frames"][st : end_row + 1].astype(np.float32) / self.raw_scale
        raw_x = np.clip(raw_x, 0.0, 3.0)
        norm_x = rec["norm_frames"][st : end_row + 1].astype(np.float32)
        raw_x = np.expand_dims(raw_x, axis=1)
        norm_x = np.expand_dims(norm_x, axis=1)
        feat_x = ((np.asarray(s["selected_features"], dtype=np.float32) - self.feature_mean) / self.feature_std).astype(np.float32)

        if self.is_train:
            if self.aug_scale_jitter > 0.0:
                scale = 1.0 + float(np.random.uniform(-self.aug_scale_jitter, self.aug_scale_jitter))
                raw_x = raw_x * scale
            if self.aug_noise_std > 0.0:
                raw_x = raw_x + np.random.normal(0.0, self.aug_noise_std, size=raw_x.shape).astype(np.float32)
                norm_x = norm_x + np.random.normal(0.0, self.aug_noise_std, size=norm_x.shape).astype(np.float32)
            if self.aug_frame_dropout > 0.0:
                keep = (np.random.rand(raw_x.shape[0], 1, 1, 1) >= self.aug_frame_dropout).astype(np.float32)
                raw_x = raw_x * keep
                norm_x = norm_x * keep
            if self.feat_noise_std > 0.0:
                feat_x = feat_x + np.random.normal(0.0, self.feat_noise_std, size=feat_x.shape).astype(np.float32)
            raw_x = np.clip(raw_x, 0.0, 3.0)
            norm_x = np.clip(norm_x, 0.0, 1.0)

        size_idx = int(s["size_class_index"])
        size_cm = float(s["size_cm"])
        depth_idx = int(s["depth_coarse_index"])
        phase_weight = float(s.get("sample_weight", 1.0))
        return (
            torch.from_numpy(raw_x),
            torch.from_numpy(norm_x),
            torch.from_numpy(feat_x),
            torch.tensor(size_idx, dtype=torch.long),
            torch.tensor(size_cm, dtype=torch.float32),
            torch.tensor(float(size_cm_to_norm([size_cm])[0]), dtype=torch.float32),
            torch.tensor(depth_idx, dtype=torch.long),
            torch.tensor(phase_weight, dtype=torch.float32),
        )


def enrich_positive_samples_with_features(
    depth_records: Dict[str, dict],
    depth_samples: List[dict],
    xgb_records: Dict[str, dict],
    selected_features: List[str],
    split_name: str,
) -> List[dict]:
    feature_records = {}
    for key, rec in depth_records.items():
        merged = dict(rec)
        if key not in xgb_records:
            raise KeyError(f"Missing XGBoost-style record for key: {key}")
        merged["frame_rows"] = xgb_records[key]["frame_rows"]
        feature_records[key] = merged

    feature_samples = []
    for s in depth_samples:
        rec = feature_records[s["group_key"]]
        row = dict(s)
        row["label"] = 1
        row["size_cm"] = float(rec["size_cm"])
        row["depth_cm"] = float(rec["depth_cm"])
        row["size_text"] = str(rec["size_text"])
        row["depth_text"] = str(rec["depth_text"])
        feature_samples.append(row)

    feature_df = build_feature_table(feature_records, feature_samples, split_name)
    missing = [col for col in selected_features if col not in feature_df.columns]
    if missing:
        raise KeyError(f"Selected features missing from feature table: {missing}")
    rows = feature_df[selected_features].to_numpy(dtype=np.float32)

    enriched = []
    for sample, row in zip(feature_samples, rows):
        copied = dict(sample)
        copied["selected_features"] = row.astype(np.float32).tolist()
        enriched.append(copied)
    return enriched


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    size_class_weight: torch.Tensor,
    depth_class_weight: torch.Tensor,
    cfg: "Config",
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    size_logits_all, size_reg_all = [], []
    size_true_all, size_cm_all = [], []
    depth_gt_all, depth_soft_all, depth_hard_all = [], [], []
    depth_true_all = []

    with torch.no_grad():
        for raw_x, norm_x, feat_x, size_idx, size_cm, size_norm, depth_idx, phase_weight in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            feat_x = feat_x.to(device)
            size_idx = size_idx.to(device)
            size_cm = size_cm.to(device)
            size_norm = size_norm.to(device)
            depth_idx = depth_idx.to(device)
            phase_weight = phase_weight.to(device)

            size_logits, size_ord_logits, size_reg_norm, size_probs, feats = model(raw_x, norm_x, feat_x, return_features=True)
            depth_logits_gt = model.route_depth_logits(feats["depth_feat"], size_idx)
            depth_logits_soft = model.route_depth_logits_soft(feats["depth_feat"], size_probs)
            depth_pred_idx = torch.argmax(size_probs, dim=1)
            depth_logits_hard = model.route_depth_logits(feats["depth_feat"], depth_pred_idx)

            size_loss = sample_weighted_cross_entropy(size_logits, size_idx, size_class_weight, None, cfg.label_smoothing)
            size_ord_loss = ordinal_loss(size_ord_logits, size_idx)
            size_reg_loss = nn.functional.smooth_l1_loss(size_reg_norm.squeeze(1), size_norm)
            depth_gt_loss = sample_weighted_cross_entropy(
                depth_logits_gt, depth_idx, depth_class_weight, phase_weight, cfg.depth_label_smoothing
            )
            depth_soft_loss = sample_weighted_cross_entropy(
                depth_logits_soft, depth_idx, depth_class_weight, phase_weight, cfg.depth_label_smoothing
            )
            depth_hard_loss = sample_weighted_cross_entropy(
                depth_logits_hard, depth_idx, depth_class_weight, phase_weight, cfg.depth_label_smoothing
            )
            depth_kl = F.kl_div(F.log_softmax(depth_logits_soft, dim=1), F.softmax(depth_logits_gt, dim=1), reduction="batchmean")
            loss = (
                size_loss
                + float(cfg.ordinal_loss_weight) * size_ord_loss
                + float(cfg.reg_loss_weight) * size_reg_loss
                + float(cfg.depth_gt_loss_weight) * depth_gt_loss
                + float(cfg.depth_soft_loss_weight) * depth_soft_loss
                + float(cfg.depth_hard_loss_weight) * depth_hard_loss
                + float(cfg.depth_route_kl_weight) * depth_kl
            )
            total_loss += float(loss.item()) * int(size_idx.shape[0])
            total_n += int(size_idx.shape[0])

            size_logits_all.append(size_logits.cpu().numpy().astype(np.float64))
            size_reg_all.append(size_norm_to_cm(size_reg_norm.squeeze(1).cpu().numpy().astype(np.float32)).astype(np.float64))
            size_true_all.append(size_idx.cpu().numpy().astype(np.int32))
            size_cm_all.append(size_cm.cpu().numpy().astype(np.float64))
            depth_true_all.append(depth_idx.cpu().numpy().astype(np.int32))
            depth_gt_all.append(depth_logits_gt.cpu().numpy().astype(np.float64))
            depth_soft_all.append(depth_logits_soft.cpu().numpy().astype(np.float64))
            depth_hard_all.append(depth_logits_hard.cpu().numpy().astype(np.float64))

    size_logits_np = np.concatenate(size_logits_all, axis=0)
    size_reg_np = np.concatenate(size_reg_all, axis=0)
    size_true_np = np.concatenate(size_true_all, axis=0)
    size_cm_np = np.concatenate(size_cm_all, axis=0)
    depth_true_np = np.concatenate(depth_true_all, axis=0)
    depth_gt_logits_np = np.concatenate(depth_gt_all, axis=0)
    depth_soft_logits_np = np.concatenate(depth_soft_all, axis=0)
    depth_hard_logits_np = np.concatenate(depth_hard_all, axis=0)

    size_reg_metrics = regression_metrics(size_reg_np, size_cm_np)
    depth_gt_metrics = depth_metrics_from_logits(depth_gt_logits_np, depth_true_np)
    depth_soft_metrics = depth_metrics_from_logits(depth_soft_logits_np, depth_true_np)
    depth_hard_metrics = depth_metrics_from_logits(depth_hard_logits_np, depth_true_np)

    return {
        "loss": float(total_loss / max(total_n, 1)),
        "count": int(total_n),
        "size_top1": float(topk_accuracy(size_logits_np, size_true_np, 1)),
        "size_top2": float(topk_accuracy(size_logits_np, size_true_np, 2)),
        "size_mae": float(size_reg_metrics["mae"]),
        "size_median_ae": float(size_reg_metrics["median_ae"]),
        "depth_gt_accuracy": float(depth_gt_metrics["accuracy"]),
        "depth_gt_balanced_accuracy": float(depth_gt_metrics["balanced_accuracy"]),
        "depth_gt_confusion_matrix": depth_gt_metrics["confusion_matrix"],
        "depth_soft_accuracy": float(depth_soft_metrics["accuracy"]),
        "depth_soft_balanced_accuracy": float(depth_soft_metrics["balanced_accuracy"]),
        "depth_soft_confusion_matrix": depth_soft_metrics["confusion_matrix"],
        "depth_hard_accuracy": float(depth_hard_metrics["accuracy"]),
        "depth_hard_balanced_accuracy": float(depth_hard_metrics["balanced_accuracy"]),
        "depth_hard_confusion_matrix": depth_hard_metrics["confusion_matrix"],
    }


def plot_curves(history: Dict[str, List[float]], output_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_size_top1"], label="val top1")
    axes[1].plot(epochs, history["test_size_top1"], label="test top1")
    axes[1].set_title("Size Top1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_size_mae"], label="val mae")
    axes[2].plot(epochs, history["test_size_mae"], label="test mae")
    axes[2].set_title("Size MAE")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, history["val_depth_gt_bal"], label="val gt-route")
    axes[3].plot(epochs, history["test_depth_gt_bal"], label="test gt-route")
    axes[3].set_title("Depth GT-Route Bal Acc")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    axes[4].plot(epochs, history["val_depth_hard_bal"], label="val hard-route")
    axes[4].plot(epochs, history["test_depth_hard_bal"], label="test hard-route")
    axes[4].set_title("Depth Hard-Route Bal Acc")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    axes[5].plot(epochs, history["val_depth_soft_bal"], label="val soft-route")
    axes[5].plot(epochs, history["test_depth_soft_bal"], label="test soft-route")
    axes[5].set_title("Depth Soft-Route Bal Acc")
    axes[5].legend()
    axes[5].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERHPINV_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERHPINV_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERHPINV_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERHPINV_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERHPINV_EPOCHS", "120"))
    batch_size: int = int(os.environ.get("PAPERHPINV_BATCH_SIZE", "48"))
    eval_batch_size: int = int(os.environ.get("PAPERHPINV_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERHPINV_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("PAPERHPINV_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("PAPERHPINV_DROPOUT", "0.28"))
    frame_feature_dim: int = int(os.environ.get("PAPERHPINV_FRAME_DIM", "24"))
    temporal_channels: int = int(os.environ.get("PAPERHPINV_TEMPORAL_CHANNELS", "48"))
    temporal_blocks: int = int(os.environ.get("PAPERHPINV_TEMPORAL_BLOCKS", "3"))
    tabular_hidden_dim: int = int(os.environ.get("PAPERHPINV_TAB_HIDDEN", "64"))
    ordinal_loss_weight: float = float(os.environ.get("PAPERHPINV_ORD_LOSS_WEIGHT", "0.15"))
    reg_loss_weight: float = float(os.environ.get("PAPERHPINV_REG_LOSS_WEIGHT", "0.20"))
    neighbor_soft_weight: float = float(os.environ.get("PAPERHPINV_NEIGHBOR_SOFT_WEIGHT", "0.20"))
    neighbor_sigma_classes: float = float(os.environ.get("PAPERHPINV_NEIGHBOR_SIGMA", "0.75"))
    depth_gt_loss_weight: float = float(os.environ.get("PAPERHPINV_DEPTH_GT_WEIGHT", "0.60"))
    depth_soft_loss_weight: float = float(os.environ.get("PAPERHPINV_DEPTH_SOFT_WEIGHT", "0.90"))
    depth_hard_loss_weight: float = float(os.environ.get("PAPERHPINV_DEPTH_HARD_WEIGHT", "0.25"))
    depth_route_kl_weight: float = float(os.environ.get("PAPERHPINV_DEPTH_KL_WEIGHT", "0.10"))
    patience: int = int(os.environ.get("PAPERHPINV_PATIENCE", "24"))
    grad_clip: float = float(os.environ.get("PAPERHPINV_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("PAPERHPINV_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERHPINV_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERHPINV_AUG_FRAME_DROPOUT", "0.02"))
    feat_noise_std: float = float(os.environ.get("PAPERHPINV_FEAT_NOISE_STD", "0.01"))
    label_smoothing: float = float(os.environ.get("PAPERHPINV_LABEL_SMOOTH", "0.03"))
    depth_label_smoothing: float = float(os.environ.get("PAPERHPINV_DEPTH_LABEL_SMOOTH", "0.02"))
    num_workers: int = int(os.environ.get("PAPERHPINV_NUM_WORKERS", "0"))

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        self.data_root = os.environ.get("PAPERHPINV_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERHPINV_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERHPINV_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERHPINV_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERHPINV_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_hierarchical_positive_inverter_run1"),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    pos_rec1, pos_samples1 = build_positive_depth_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    pos_rec2, pos_samples2 = build_positive_depth_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    pos_rec3, pos_samples3 = build_positive_depth_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

    xgb_rec1, _ = build_records_and_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    xgb_rec2, _ = build_records_and_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    xgb_rec3, _ = build_records_and_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

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

    selected_features = load_selected_feature_names()
    train_samples = enrich_positive_samples_with_features(train_records, train_samples, xgb_rec1, selected_features, "train")
    val_samples = enrich_positive_samples_with_features(val_records, val_samples, xgb_rec2, selected_features, "val")
    test_samples = enrich_positive_samples_with_features(test_records, test_samples, xgb_rec3, selected_features, "test")

    train_feat = np.asarray([s["selected_features"] for s in train_samples], dtype=np.float32)
    feature_mean = train_feat.mean(axis=0)
    feature_std = np.maximum(train_feat.std(axis=0), 1e-6)
    raw_scale = compute_raw_scale(train_records)

    ds_train = PositiveInverterDataset(
        train_records,
        train_samples,
        raw_scale,
        feature_mean,
        feature_std,
        True,
        cfg.aug_noise_std,
        cfg.aug_scale_jitter,
        cfg.aug_frame_dropout,
        cfg.feat_noise_std,
    )
    ds_val = PositiveInverterDataset(val_records, val_samples, raw_scale, feature_mean, feature_std, False)
    ds_test = PositiveInverterDataset(test_records, test_samples, raw_scale, feature_mean, feature_std, False)

    train_size = np.array([int(s["size_class_index"]) for s in train_samples], dtype=np.int32)
    train_depth = np.array([int(s["depth_coarse_index"]) for s in train_samples], dtype=np.int32)
    train_phase = np.array([float(s.get("sample_weight", 1.0)) for s in train_samples], dtype=np.float32)
    size_counts = np.bincount(train_size, minlength=len(SIZE_VALUES_CM)).astype(np.float32)
    depth_counts = np.bincount(train_depth, minlength=len(COARSE_DEPTH_ORDER)).astype(np.float32)
    size_class_weight_np = size_counts.sum() / np.maximum(size_counts * len(SIZE_VALUES_CM), 1.0)
    depth_class_weight_np = depth_counts.sum() / np.maximum(depth_counts * len(COARSE_DEPTH_ORDER), 1.0)
    sampler_weight_np = size_class_weight_np[train_size] * depth_class_weight_np[train_depth] * train_phase

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=WeightedRandomSampler(torch.tensor(sampler_weight_np, dtype=torch.float32), num_samples=len(train_samples), replacement=True),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(ds_val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(ds_test, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size_class_weight_t = torch.tensor(size_class_weight_np, dtype=torch.float32, device=device)
    depth_class_weight_t = torch.tensor(depth_class_weight_np, dtype=torch.float32, device=device)

    model = HierarchicalPositiveInverter(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        num_size_classes=len(SIZE_VALUES_CM),
        num_depth_classes=len(COARSE_DEPTH_ORDER),
        num_tabular_features=len(selected_features),
        tabular_hidden_dim=cfg.tabular_hidden_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_size_top1": [],
        "test_size_top1": [],
        "val_size_mae": [],
        "test_size_mae": [],
        "val_depth_gt_bal": [],
        "test_depth_gt_bal": [],
        "val_depth_hard_bal": [],
        "test_depth_hard_bal": [],
        "val_depth_soft_bal": [],
        "test_depth_soft_bal": [],
    }
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for raw_x, norm_x, feat_x, size_idx, _size_cm, size_norm, depth_idx, phase_weight in train_loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            feat_x = feat_x.to(device)
            size_idx = size_idx.to(device)
            size_norm = size_norm.to(device)
            depth_idx = depth_idx.to(device)
            phase_weight = phase_weight.to(device)

            optimizer.zero_grad()
            size_logits, size_ord_logits, size_reg_norm, size_probs, feats = model(raw_x, norm_x, feat_x, return_features=True)
            depth_logits_gt = model.route_depth_logits(feats["depth_feat"], size_idx)
            depth_logits_soft = model.route_depth_logits_soft(feats["depth_feat"], size_probs)
            pred_size_idx = torch.argmax(size_probs.detach(), dim=1)
            depth_logits_hard = model.route_depth_logits(feats["depth_feat"], pred_size_idx)
            depth_logits_top2 = model.route_depth_logits_soft(feats["depth_feat"], truncate_topk_probs(size_probs, 2))

            loss = sample_weighted_cross_entropy(size_logits, size_idx, size_class_weight_t, None, cfg.label_smoothing)
            loss = loss + float(cfg.ordinal_loss_weight) * ordinal_loss(size_ord_logits, size_idx)
            loss = loss + float(cfg.reg_loss_weight) * F.smooth_l1_loss(size_reg_norm.squeeze(1), size_norm)
            if float(cfg.neighbor_soft_weight) > 0.0:
                soft_target = gaussian_neighbor_targets(size_idx, len(SIZE_VALUES_CM), cfg.neighbor_sigma_classes)
                loss = loss + float(cfg.neighbor_soft_weight) * F.kl_div(
                    F.log_softmax(size_logits, dim=1),
                    soft_target,
                    reduction="batchmean",
                )
            loss = loss + float(cfg.depth_gt_loss_weight) * sample_weighted_cross_entropy(
                depth_logits_gt, depth_idx, depth_class_weight_t, phase_weight, cfg.depth_label_smoothing
            )
            loss = loss + float(cfg.depth_soft_loss_weight) * sample_weighted_cross_entropy(
                depth_logits_soft, depth_idx, depth_class_weight_t, phase_weight, cfg.depth_label_smoothing
            )
            loss = loss + float(cfg.depth_hard_loss_weight) * sample_weighted_cross_entropy(
                depth_logits_hard, depth_idx, depth_class_weight_t, phase_weight, cfg.depth_label_smoothing
            )
            loss = loss + 0.15 * sample_weighted_cross_entropy(
                depth_logits_top2, depth_idx, depth_class_weight_t, phase_weight, cfg.depth_label_smoothing
            )
            loss = loss + float(cfg.depth_route_kl_weight) * F.kl_div(
                F.log_softmax(depth_logits_soft, dim=1),
                F.softmax(depth_logits_gt.detach(), dim=1),
                reduction="batchmean",
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

            total_loss += float(loss.item()) * int(size_idx.shape[0])
            total_n += int(size_idx.shape[0])

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, size_class_weight_t, depth_class_weight_t, cfg)
        test_metrics = evaluate_model(model, test_loader, device, size_class_weight_t, depth_class_weight_t, cfg)

        history["train_loss"].append(float(total_loss / max(total_n, 1)))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_size_top1"].append(float(val_metrics["size_top1"]))
        history["test_size_top1"].append(float(test_metrics["size_top1"]))
        history["val_size_mae"].append(float(val_metrics["size_mae"]))
        history["test_size_mae"].append(float(test_metrics["size_mae"]))
        history["val_depth_gt_bal"].append(float(val_metrics["depth_gt_balanced_accuracy"]))
        history["test_depth_gt_bal"].append(float(test_metrics["depth_gt_balanced_accuracy"]))
        history["val_depth_hard_bal"].append(float(val_metrics["depth_hard_balanced_accuracy"]))
        history["test_depth_hard_bal"].append(float(test_metrics["depth_hard_balanced_accuracy"]))
        history["val_depth_soft_bal"].append(float(val_metrics["depth_soft_balanced_accuracy"]))
        history["test_depth_soft_bal"].append(float(test_metrics["depth_soft_balanced_accuracy"]))

        score = (
            float(val_metrics["depth_hard_balanced_accuracy"]),
            float(val_metrics["depth_soft_balanced_accuracy"]),
            float(val_metrics["size_top1"]),
            -float(val_metrics["size_mae"]),
            float(val_metrics["depth_gt_balanced_accuracy"]),
        )
        if best is None or score > best["score"]:
            best = {
                "epoch": epoch,
                "score": score,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            patience_left = int(cfg.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best is None:
        raise RuntimeError("Training failed to produce a best checkpoint.")

    ckpt_path = os.path.join(cfg.output_dir, "paper_hierarchical_positive_inverter_best.pth")
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "input_shape": list(INPUT_SHAPE),
            "raw_scale": float(raw_scale),
            "router_model_name": "HierarchicalPositiveInverter",
            "selected_features": list(selected_features),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
            "size_class_weight": size_class_weight_np.tolist(),
            "depth_class_weight": depth_class_weight_np.tolist(),
            "model_config": {
                "seq_len": cfg.seq_len,
                "frame_feature_dim": cfg.frame_feature_dim,
                "temporal_channels": cfg.temporal_channels,
                "temporal_blocks": cfg.temporal_blocks,
                "dropout": cfg.dropout,
                "num_size_classes": len(SIZE_VALUES_CM),
                "num_depth_classes": len(COARSE_DEPTH_ORDER),
                "num_tabular_features": len(selected_features),
                "tabular_hidden_dim": cfg.tabular_hidden_dim,
            },
        },
        ckpt_path,
    )

    plot_curves(history, os.path.join(cfg.output_dir, "paper_hierarchical_positive_inverter_curves.png"))
    xgb_ref_path = os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_baselines_v1", "xgboost_baseline_summary.json")
    xgb_reference = None
    if os.path.exists(xgb_ref_path):
        with open(xgb_ref_path, "r", encoding="utf-8") as f:
            xgb_summary = json.load(f)
        xgb_reference = xgb_summary.get("size_depth_xgboost", {}).get("gt_positive_test_metrics")

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "HierarchicalPositiveInverter",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "best_epoch": int(best["epoch"]),
        "selected_features": list(selected_features),
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "xgboost_reference_gt_positive_test": xgb_reference,
        "curve_path": os.path.join(cfg.output_dir, "paper_hierarchical_positive_inverter_curves.png"),
    }
    with open(os.path.join(cfg.output_dir, "paper_hierarchical_positive_inverter_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_rows = []
    for split_name, split_samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for sample in split_samples:
            manifest_rows.append(
                {
                    "split": split_name,
                    "group_key": sample["group_key"],
                    "base_group": sample["base_group"],
                    "file_name": sample["file_name"],
                    "size_text": sample["size_text"],
                    "depth_text": sample["depth_text"],
                    "center_row": int(sample["center_row"]),
                    "end_row": int(sample["end_row"]),
                    "size_class_index": int(sample["size_class_index"]),
                    "depth_coarse_index": int(sample["depth_coarse_index"]),
                    "size_cm": float(sample["size_cm"]),
                    "sample_weight": float(sample["sample_weight"]),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(cfg.output_dir, "paper_hierarchical_positive_inverter_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
