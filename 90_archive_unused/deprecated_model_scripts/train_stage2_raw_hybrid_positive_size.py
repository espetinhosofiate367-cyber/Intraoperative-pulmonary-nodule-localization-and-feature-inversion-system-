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

from raw_hybrid_positive_size_model import RawHybridPositiveSizeModel
from task_protocol_v1 import INPUT_SEQ_LEN, INPUT_SHAPE, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
from train_stage3_raw_size_conditioned_depth import compute_raw_scale
from train_xgboost_baselines import build_feature_table, build_records_and_samples_for_file
from triplet_repeat_classifier.train_triplet_repeat_classifier import load_json, set_seed


DEFAULT_SIZE_FEATURES = [
    "center_border_contrast_max",
    "meanframe_norm_center_border_contrast",
    "raw_max_max",
    "anisotropy_ratio_min",
    "deltaframe_centroid_col",
    "maxframe_raw_center_border_contrast",
    "spatial_entropy_max",
    "window_norm_global_std",
    "centroid_row_last",
    "centroid_row_min",
    "maxframe_raw_centroid_col",
    "hotspot_radius_min",
    "window_raw_global_std",
    "raw_p95_max",
    "meanframe_norm_raw_p95",
    "maxframe_raw_hotspot_radius",
    "maxframe_raw_raw_p95",
    "deltaframe_hotspot_radius",
    "deltaframe_spatial_entropy",
    "meanframe_norm_center_mean",
]


def size_cm_to_norm(size_cm: np.ndarray) -> np.ndarray:
    size_cm = np.asarray(size_cm, dtype=np.float32)
    lo = float(min(SIZE_VALUES_CM))
    hi = float(max(SIZE_VALUES_CM))
    return np.clip((size_cm - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def size_norm_to_cm(size_norm: np.ndarray) -> np.ndarray:
    size_norm = np.asarray(size_norm, dtype=np.float32)
    lo = float(min(SIZE_VALUES_CM))
    hi = float(max(SIZE_VALUES_CM))
    return lo + np.clip(size_norm, 0.0, 1.0) * (hi - lo)


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(1, int(num_classes), device=labels.device).view(1, -1)
    return (labels.view(-1, 1) >= thresholds).float()


def gaussian_neighbor_targets(labels: torch.Tensor, num_classes: int, sigma_classes: float) -> torch.Tensor:
    cls_idx = torch.arange(int(num_classes), device=labels.device, dtype=torch.float32).view(1, -1)
    labels_f = labels.view(-1, 1).float()
    dist2 = (cls_idx - labels_f) ** 2
    targets = torch.exp(-0.5 * dist2 / max(float(sigma_classes) ** 2, 1e-8))
    return targets / torch.clamp(targets.sum(dim=1, keepdim=True), min=1e-8)


def ordinal_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    targets = ordinal_targets(labels, logits.shape[1] + 1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    if sample_weight is not None:
        loss = loss * sample_weight.view(-1)
        return loss.sum() / torch.clamp(sample_weight.sum(), min=1.0)
    return loss.mean()


class HybridPositiveSizeDataset(Dataset):
    def __init__(
        self,
        records_by_key: Dict[str, dict],
        samples: List[dict],
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
        self.samples = samples
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
        return (
            torch.from_numpy(raw_x),
            torch.from_numpy(norm_x),
            torch.from_numpy(feat_x),
            torch.tensor(size_idx, dtype=torch.long),
            torch.tensor(size_cm, dtype=torch.float32),
            torch.tensor(float(size_cm_to_norm([size_cm])[0]), dtype=torch.float32),
        )


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = np.argmax(logits, axis=1)
    top1 = float(np.mean(pred == labels))
    top2 = float(np.mean(np.any(np.argsort(-logits, axis=1)[:, :2] == labels[:, None], axis=1)))
    return {"top1": top1, "top2": top2}


def regression_metrics(pred_cm: np.ndarray, target_cm: np.ndarray) -> Dict[str, float]:
    ae = np.abs(pred_cm.astype(np.float64) - target_cm.astype(np.float64))
    return {"mae": float(np.mean(ae)), "median_ae": float(np.median(ae))}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion_cls: nn.Module,
    ordinal_loss_weight: float,
    reg_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    logits_all, reg_all, cls_all, reg_true_all = [], [], [], []
    with torch.no_grad():
        for raw_x, norm_x, feat_x, size_idx, size_cm, size_norm in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            feat_x = feat_x.to(device)
            size_idx = size_idx.to(device)
            size_cm = size_cm.to(device)
            size_norm = size_norm.to(device)
            size_logits, size_ord_logits, size_reg_norm, _size_probs = model(raw_x, norm_x, feat_x)
            loss_cls = criterion_cls(size_logits, size_idx)
            loss_ord = ordinal_loss(size_ord_logits, size_idx)
            loss_reg = nn.functional.smooth_l1_loss(size_reg_norm.squeeze(1), size_norm)
            loss = loss_cls + float(ordinal_loss_weight) * loss_ord + float(reg_loss_weight) * loss_reg
            total_loss += float(loss.item()) * int(size_idx.shape[0])
            total_n += int(size_idx.shape[0])
            logits_all.append(size_logits.cpu().numpy().astype(np.float64))
            reg_all.append(size_norm_to_cm(size_reg_norm.squeeze(1).cpu().numpy().astype(np.float32)).astype(np.float64))
            cls_all.append(size_idx.cpu().numpy().astype(np.int32))
            reg_true_all.append(size_cm.cpu().numpy().astype(np.float64))
    logits_np = np.concatenate(logits_all, axis=0)
    reg_np = np.concatenate(reg_all, axis=0)
    cls_np = np.concatenate(cls_all, axis=0)
    reg_true_np = np.concatenate(reg_true_all, axis=0)
    cls_metrics = classification_metrics(logits_np, cls_np)
    reg_metrics = regression_metrics(reg_np, reg_true_np)
    return {
        "loss": float(total_loss / max(total_n, 1)),
        "top1": cls_metrics["top1"],
        "top2": cls_metrics["top2"],
        "mae": reg_metrics["mae"],
        "median_ae": reg_metrics["median_ae"],
        "count": int(total_n),
    }


def plot_curves(history: Dict[str, List[float]], output_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_top1"], label="val top1")
    axes[1].plot(epochs, history["test_top1"], label="test top1")
    axes[1].set_title("Top1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_top2"], label="val top2")
    axes[2].plot(epochs, history["test_top2"], label="test top2")
    axes[2].set_title("Top2")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, history["val_mae"], label="val mae")
    axes[3].plot(epochs, history["test_mae"], label="test mae")
    axes[3].set_title("MAE")
    axes[3].legend()
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_selected_feature_names() -> List[str]:
    summary_path = os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_explainability_v1", "xgboost_explainability_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        names = []
        names.extend(summary.get("top_size_features", []))
        names.extend(summary.get("top_size_regression_features", []))
        merged = []
        seen = set()
        for name in names:
            if isinstance(name, str) and name and name not in seen:
                seen.add(name)
                merged.append(name)
        if merged:
            return merged
    return list(DEFAULT_SIZE_FEATURES)


def attach_selected_features(records_by_key: Dict[str, dict], samples: List[dict], selected_features: List[str]) -> List[dict]:
    feature_df = build_feature_table(records_by_key, samples, "tmp")
    missing = [col for col in selected_features if col not in feature_df.columns]
    if missing:
        raise KeyError(f"Selected features missing from feature table: {missing}")
    rows = feature_df[selected_features].to_numpy(dtype=np.float32)
    enriched = []
    for sample, row in zip(samples, rows):
        copied = dict(sample)
        copied["selected_features"] = row.astype(np.float32).tolist()
        enriched.append(copied)
    return enriched


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERHSIZE_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERHSIZE_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERHSIZE_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERHSIZE_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERHSIZE_EPOCHS", "120"))
    batch_size: int = int(os.environ.get("PAPERHSIZE_BATCH_SIZE", "48"))
    eval_batch_size: int = int(os.environ.get("PAPERHSIZE_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERHSIZE_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("PAPERHSIZE_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("PAPERHSIZE_DROPOUT", "0.28"))
    frame_feature_dim: int = int(os.environ.get("PAPERHSIZE_FRAME_DIM", "24"))
    temporal_channels: int = int(os.environ.get("PAPERHSIZE_TEMPORAL_CHANNELS", "48"))
    temporal_blocks: int = int(os.environ.get("PAPERHSIZE_TEMPORAL_BLOCKS", "3"))
    tabular_hidden_dim: int = int(os.environ.get("PAPERHSIZE_TAB_HIDDEN", "64"))
    ordinal_loss_weight: float = float(os.environ.get("PAPERHSIZE_ORD_LOSS_WEIGHT", "0.30"))
    reg_loss_weight: float = float(os.environ.get("PAPERHSIZE_REG_LOSS_WEIGHT", "0.50"))
    neighbor_soft_weight: float = float(os.environ.get("PAPERHSIZE_NEIGHBOR_SOFT_WEIGHT", "0.00"))
    neighbor_sigma_classes: float = float(os.environ.get("PAPERHSIZE_NEIGHBOR_SIGMA", "0.75"))
    patience: int = int(os.environ.get("PAPERHSIZE_PATIENCE", "22"))
    grad_clip: float = float(os.environ.get("PAPERHSIZE_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("PAPERHSIZE_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERHSIZE_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERHSIZE_AUG_FRAME_DROPOUT", "0.02"))
    feat_noise_std: float = float(os.environ.get("PAPERHSIZE_FEAT_NOISE_STD", "0.01"))
    label_smoothing: float = float(os.environ.get("PAPERHSIZE_LABEL_SMOOTH", "0.03"))
    num_workers: int = int(os.environ.get("PAPERHSIZE_NUM_WORKERS", "0"))

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        self.data_root = os.environ.get("PAPERHSIZE_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERHSIZE_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERHSIZE_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERHSIZE_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERHSIZE_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage2_raw_hybrid_positive_size"),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    rec1, samples1_all = build_records_and_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, samples2_all = build_records_and_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3_all = build_records_and_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        )
    )
    common_set = set(common_base_groups)

    train_records = {k: v for k, v in rec1.items() if v["base_group"] in common_set}
    val_records = {k: v for k, v in rec2.items() if v["base_group"] in common_set}
    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_set}
    train_samples = [s for s in samples1_all if s["base_group"] in common_set and int(s["label"]) == 1]
    val_samples = [s for s in samples2_all if s["base_group"] in common_set and int(s["label"]) == 1]
    test_samples = [s for s in samples3_all if s["base_group"] in common_set and int(s["label"]) == 1]

    selected_features = load_selected_feature_names()
    train_samples = attach_selected_features(train_records, train_samples, selected_features)
    val_samples = attach_selected_features(val_records, val_samples, selected_features)
    test_samples = attach_selected_features(test_records, test_samples, selected_features)

    train_feat = np.asarray([s["selected_features"] for s in train_samples], dtype=np.float32)
    feature_mean = train_feat.mean(axis=0)
    feature_std = np.maximum(train_feat.std(axis=0), 1e-6)
    raw_scale = compute_raw_scale(train_records)

    ds_train = HybridPositiveSizeDataset(
        train_records,
        train_samples,
        raw_scale=raw_scale,
        feature_mean=feature_mean,
        feature_std=feature_std,
        is_train=True,
        aug_noise_std=cfg.aug_noise_std,
        aug_scale_jitter=cfg.aug_scale_jitter,
        aug_frame_dropout=cfg.aug_frame_dropout,
        feat_noise_std=cfg.feat_noise_std,
    )
    ds_val = HybridPositiveSizeDataset(val_records, val_samples, raw_scale, feature_mean, feature_std, is_train=False)
    ds_test = HybridPositiveSizeDataset(test_records, test_samples, raw_scale, feature_mean, feature_std, is_train=False)

    train_size = np.array([int(s["size_class_index"]) for s in train_samples], dtype=np.int32)
    counts = np.bincount(train_size, minlength=len(SIZE_VALUES_CM)).astype(np.float32)
    class_weight_np = counts.sum() / np.maximum(counts * len(SIZE_VALUES_CM), 1.0)
    sampler_weight_np = class_weight_np[train_size]

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
    class_weight_t = torch.tensor(class_weight_np, dtype=torch.float32, device=device)
    model = RawHybridPositiveSizeModel(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        num_size_classes=len(SIZE_VALUES_CM),
        num_tabular_features=len(selected_features),
        tabular_hidden_dim=cfg.tabular_hidden_dim,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "val_top1": [], "test_top1": [], "val_top2": [], "test_top2": [], "val_mae": [], "test_mae": []}
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for raw_x, norm_x, feat_x, size_idx, _size_cm, size_norm in train_loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            feat_x = feat_x.to(device)
            size_idx = size_idx.to(device)
            size_norm = size_norm.to(device)

            optimizer.zero_grad()
            size_logits, size_ord_logits, size_reg_norm, _size_probs = model(raw_x, norm_x, feat_x)
            loss_cls = criterion_cls(size_logits, size_idx)
            loss_ord = ordinal_loss(size_ord_logits, size_idx)
            loss_reg = nn.functional.smooth_l1_loss(size_reg_norm.squeeze(1), size_norm)
            loss = loss_cls + float(cfg.ordinal_loss_weight) * loss_ord + float(cfg.reg_loss_weight) * loss_reg
            if float(cfg.neighbor_soft_weight) > 0.0:
                soft_target = gaussian_neighbor_targets(size_idx, len(SIZE_VALUES_CM), cfg.neighbor_sigma_classes)
                soft_loss = nn.functional.kl_div(
                    nn.functional.log_softmax(size_logits, dim=1),
                    soft_target,
                    reduction="batchmean",
                )
                loss = loss + float(cfg.neighbor_soft_weight) * soft_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()

            total_loss += float(loss.item()) * int(size_idx.shape[0])
            total_n += int(size_idx.shape[0])

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, criterion_cls, cfg.ordinal_loss_weight, cfg.reg_loss_weight)
        test_metrics = evaluate_model(model, test_loader, device, criterion_cls, cfg.ordinal_loss_weight, cfg.reg_loss_weight)
        history["train_loss"].append(float(total_loss / max(total_n, 1)))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_top1"].append(float(val_metrics["top1"]))
        history["test_top1"].append(float(test_metrics["top1"]))
        history["val_top2"].append(float(val_metrics["top2"]))
        history["test_top2"].append(float(test_metrics["top2"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        history["test_mae"].append(float(test_metrics["mae"]))

        score = (float(val_metrics["top1"]), float(val_metrics["top2"]), -float(val_metrics["mae"]), -float(val_metrics["loss"]))
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

    ckpt_path = os.path.join(cfg.output_dir, "paper_stage2_raw_hybrid_positive_size_best.pth")
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "input_shape": list(INPUT_SHAPE),
            "raw_scale": float(raw_scale),
            "router_model_name": "RawHybridPositiveSizeModel",
            "selected_features": list(selected_features),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
            "class_weight": class_weight_np.tolist(),
            "model_config": {
                "seq_len": cfg.seq_len,
                "frame_feature_dim": cfg.frame_feature_dim,
                "temporal_channels": cfg.temporal_channels,
                "temporal_blocks": cfg.temporal_blocks,
                "dropout": cfg.dropout,
                "num_size_classes": len(SIZE_VALUES_CM),
                "num_tabular_features": len(selected_features),
                "tabular_hidden_dim": cfg.tabular_hidden_dim,
            },
        },
        ckpt_path,
    )

    plot_curves(history, os.path.join(cfg.output_dir, "paper_stage2_raw_hybrid_positive_size_curves.png"))
    xgb_ref_path = os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_baselines_v1", "xgboost_baseline_summary.json")
    xgb_reference = None
    if os.path.exists(xgb_ref_path):
        with open(xgb_ref_path, "r", encoding="utf-8") as f:
            xgb_summary = json.load(f)
        xgb_reference = xgb_summary.get("size_depth_xgboost", {}).get("gt_positive_test_metrics")

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "RawHybridPositiveSizeModel",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "best_epoch": int(best["epoch"]),
        "selected_features": list(selected_features),
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "xgboost_reference_gt_positive_test": xgb_reference,
        "curve_path": os.path.join(cfg.output_dir, "paper_stage2_raw_hybrid_positive_size_curves.png"),
    }
    with open(os.path.join(cfg.output_dir, "paper_stage2_raw_hybrid_positive_size_summary.json"), "w", encoding="utf-8") as f:
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
                    "size_cm": float(sample["size_cm"]),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(cfg.output_dir, "paper_stage2_raw_hybrid_positive_size_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
