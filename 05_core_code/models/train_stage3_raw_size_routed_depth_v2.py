import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)
REPO_ROOT = os.path.dirname(CODE_ARCHIVE_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if CODE_ARCHIVE_DIR not in sys.path:
    sys.path.insert(0, CODE_ARCHIVE_DIR)

from raw_positive_size_model import RawPositiveSizeModel
from raw_positive_size_model_v2 import RawPositiveSizeModelV2
from raw_size_routed_depth_model import RawSizeRoutedDepthModel
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, INPUT_SHAPE, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
from train_stage2_raw_hybrid_positive_size import size_norm_to_cm
from train_stage3_raw_size_conditioned_depth import (
    PositiveDepthDataset,
    balanced_accuracy_from_cm,
    build_positive_depth_samples_for_file,
    compute_concept_stats,
    compute_raw_scale,
    confusion_matrix_counts,
    depth_majority_baseline,
)
from triplet_repeat_classifier.train_triplet_repeat_classifier import load_json, set_seed


def plot_curves(history: Dict[str, List[float]], output_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_gt_bal_acc"], label="val gt")
    axes[1].plot(epochs, history["val_hard_bal_acc"], label="val hard")
    axes[1].set_title("Val Balanced Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["test_gt_bal_acc"], label="test gt")
    axes[2].plot(epochs, history["test_hard_bal_acc"], label="test hard")
    axes[2].set_title("Test Balanced Accuracy")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].plot(epochs, history["val_soft_bal_acc"], label="val soft")
    axes[3].plot(epochs, history["test_soft_bal_acc"], label="test soft")
    axes[3].set_title("Soft Route Balanced Accuracy")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    axes[4].plot(epochs, history["val_size_top1"], label="val size top1")
    axes[4].plot(epochs, history["test_size_top1"], label="test size top1")
    axes[4].set_title("Router Top1")
    axes[4].legend()
    axes[4].grid(alpha=0.3)

    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_depth(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    cm = confusion_matrix_counts(y_true, y_pred, len(COARSE_DEPTH_ORDER))
    return {
        "count": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_from_cm(cm)),
        "confusion_matrix": cm.tolist(),
    }


def to_jsonable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        clean[key] = value
    return clean


def build_top2_probs(size_probs: torch.Tensor) -> torch.Tensor:
    top2_probs = size_probs.clone()
    zero_idx = torch.argsort(top2_probs, dim=1, descending=True)[:, 2:]
    top2_probs.scatter_(1, zero_idx, 0.0)
    top2_probs = top2_probs / torch.clamp(top2_probs.sum(dim=1, keepdim=True), min=1e-8)
    return top2_probs


def load_frozen_size_router(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    router_name = str(ckpt.get("router_model_name", ckpt.get("model_name", "")))
    if router_name == "RawPositiveSizeModel":
        model = RawPositiveSizeModel(
            seq_len=INPUT_SEQ_LEN,
            frame_feature_dim=24,
            temporal_channels=48,
            temporal_blocks=3,
            dropout=0.25,
            num_size_classes=len(SIZE_VALUES_CM),
        ).to(device)
        mode = "raw_positive_size"
    elif router_name == "RawPositiveSizeModelV2":
        model_cfg = ckpt.get("model_config", {})
        model = RawPositiveSizeModelV2(
            seq_len=int(model_cfg.get("seq_len", INPUT_SEQ_LEN)),
            frame_feature_dim=int(model_cfg.get("frame_feature_dim", 32)),
            temporal_channels=int(model_cfg.get("temporal_channels", 64)),
            temporal_blocks=int(model_cfg.get("temporal_blocks", 4)),
            dropout=float(model_cfg.get("dropout", 0.22)),
            num_size_classes=int(model_cfg.get("num_size_classes", len(SIZE_VALUES_CM))),
            residual_scale=float(model_cfg.get("residual_scale", 0.35)),
        ).to(device)
        mode = "raw_positive_size_v2"
    else:
        raise RuntimeError(f"Unsupported stage2 router for raw-only route-aware depth training: {router_name}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, mode


def router_forward(router: nn.Module, router_mode: str, raw_x: torch.Tensor, norm_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if router_mode == "raw_positive_size":
        size_logits, size_reg_pred, size_probs = router(raw_x, norm_x)
        size_reg_cm = size_reg_pred.squeeze(1)
    elif router_mode == "raw_positive_size_v2":
        size_logits, _size_ord_logits, size_reg_pred, size_probs = router(raw_x, norm_x)
        size_reg_cm = torch.tensor(
            size_norm_to_cm(size_reg_pred.squeeze(1).detach().cpu().numpy()),
            dtype=raw_x.dtype,
            device=raw_x.device,
        )
    else:
        raise RuntimeError(f"Unsupported router_mode: {router_mode}")
    return size_logits, size_probs, size_reg_cm


def evaluate_model(
    model: nn.Module,
    router: nn.Module,
    router_mode: str,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    gt_weight: float,
    hard_weight: float,
    soft_weight: float,
    top2_weight: float,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_n = 0

    y_true = []
    y_gt = []
    y_hard = []
    y_soft = []
    y_top2 = []
    gt_size = []
    pred_size = []
    size_probs_rows = []

    with torch.no_grad():
        for raw_x, norm_x, size_idx, depth_idx, _concept_target, _sample_weight in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)

            _size_logits, size_probs, _size_reg_cm = router_forward(router, router_mode, raw_x, norm_x)
            size_pred_idx = torch.argmax(size_probs, dim=1)
            top2_probs = build_top2_probs(size_probs)

            gt_logits, _ = model(raw_x, norm_x, size_idx)
            hard_logits, _ = model(raw_x, norm_x, size_pred_idx)
            soft_logits, _ = model.forward_soft(raw_x, norm_x, size_probs)
            top2_logits, _ = model.forward_soft(raw_x, norm_x, top2_probs)

            loss_gt = criterion(gt_logits, depth_idx)
            loss_hard = criterion(hard_logits, depth_idx)
            loss_soft = criterion(soft_logits, depth_idx)
            loss_top2 = criterion(top2_logits, depth_idx)
            loss = (
                float(gt_weight) * loss_gt
                + float(hard_weight) * loss_hard
                + float(soft_weight) * loss_soft
                + float(top2_weight) * loss_top2
            )

            batch_n = int(depth_idx.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_n += batch_n

            y_true.append(depth_idx.cpu().numpy().astype(np.int32))
            y_gt.append(torch.argmax(gt_logits, dim=1).cpu().numpy().astype(np.int32))
            y_hard.append(torch.argmax(hard_logits, dim=1).cpu().numpy().astype(np.int32))
            y_soft.append(torch.argmax(soft_logits, dim=1).cpu().numpy().astype(np.int32))
            y_top2.append(torch.argmax(top2_logits, dim=1).cpu().numpy().astype(np.int32))
            gt_size.append(size_idx.cpu().numpy().astype(np.int32))
            pred_size.append(size_pred_idx.cpu().numpy().astype(np.int32))
            size_probs_rows.append(size_probs.cpu().numpy().astype(np.float32))

    y_true_np = np.concatenate(y_true)
    y_gt_np = np.concatenate(y_gt)
    y_hard_np = np.concatenate(y_hard)
    y_soft_np = np.concatenate(y_soft)
    y_top2_np = np.concatenate(y_top2)
    gt_size_np = np.concatenate(gt_size)
    pred_size_np = np.concatenate(pred_size)
    size_probs_np = np.concatenate(size_probs_rows, axis=0)
    route_match = pred_size_np == gt_size_np
    top2_idx = np.argsort(-size_probs_np, axis=1)[:, :2]
    size_top2 = float(np.mean(np.any(top2_idx == gt_size_np[:, None], axis=1)))

    metrics = {
        "loss": float(total_loss / max(total_n, 1)),
        "count": int(total_n),
        "size_top1": float(np.mean(route_match)),
        "size_top2": size_top2,
        "route_match_rate": float(np.mean(route_match)),
        "gt_route": summarize_depth(y_true_np, y_gt_np),
        "hard_route": summarize_depth(y_true_np, y_hard_np),
        "soft_route": summarize_depth(y_true_np, y_soft_np),
        "top2_soft_route": summarize_depth(y_true_np, y_top2_np),
        "hard_route_when_size_correct": summarize_depth(y_true_np[route_match], y_hard_np[route_match]) if np.any(route_match) else None,
        "hard_route_when_size_wrong": summarize_depth(y_true_np[~route_match], y_hard_np[~route_match]) if np.any(~route_match) else None,
    }
    return metrics


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERD3ROUTEV2_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERD3ROUTEV2_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERD3ROUTEV2_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERD3ROUTEV2_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERD3ROUTEV2_EPOCHS", "90"))
    batch_size: int = int(os.environ.get("PAPERD3ROUTEV2_BATCH_SIZE", "48"))
    eval_batch_size: int = int(os.environ.get("PAPERD3ROUTEV2_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERD3ROUTEV2_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("PAPERD3ROUTEV2_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("PAPERD3ROUTEV2_DROPOUT", "0.25"))
    frame_feature_dim: int = int(os.environ.get("PAPERD3ROUTEV2_FRAME_DIM", "24"))
    temporal_channels: int = int(os.environ.get("PAPERD3ROUTEV2_TEMPORAL_CHANNELS", "48"))
    temporal_blocks: int = int(os.environ.get("PAPERD3ROUTEV2_TEMPORAL_BLOCKS", "3"))
    patience: int = int(os.environ.get("PAPERD3ROUTEV2_PATIENCE", "18"))
    grad_clip: float = float(os.environ.get("PAPERD3ROUTEV2_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("PAPERD3ROUTEV2_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERD3ROUTEV2_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERD3ROUTEV2_AUG_FRAME_DROPOUT", "0.02"))
    label_smoothing: float = float(os.environ.get("PAPERD3ROUTEV2_LABEL_SMOOTH", "0.03"))
    gt_loss_weight: float = float(os.environ.get("PAPERD3ROUTEV2_GT_LOSS_WEIGHT", "1.0"))
    hard_loss_weight: float = float(os.environ.get("PAPERD3ROUTEV2_HARD_LOSS_WEIGHT", "0.85"))
    soft_loss_weight: float = float(os.environ.get("PAPERD3ROUTEV2_SOFT_LOSS_WEIGHT", "0.35"))
    top2_loss_weight: float = float(os.environ.get("PAPERD3ROUTEV2_TOP2_LOSS_WEIGHT", "0.15"))
    consistency_kl_weight: float = float(os.environ.get("PAPERD3ROUTEV2_KL_WEIGHT", "0.05"))
    num_workers: int = int(os.environ.get("PAPERD3ROUTEV2_NUM_WORKERS", "0"))

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        self.data_root = os.environ.get("PAPERD3ROUTEV2_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERD3ROUTEV2_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERD3ROUTEV2_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERD3ROUTEV2_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERD3ROUTEV2_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_raw_size_routed_depth_v2"),
        )
        self.stage2_ckpt = os.environ.get(
            "PAPERD3ROUTEV2_STAGE2_CKPT",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage2_raw_positive_size_v2", "paper_stage2_raw_positive_size_v2_best.pth"),
        )
        self.warm_start_ckpt = os.environ.get(
            "PAPERD3ROUTEV2_WARM_START_CKPT",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_raw_size_routed_depth_run1", "paper_stage3_raw_size_routed_depth_best.pth"),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    rec1, samples1 = build_positive_depth_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, samples2 = build_positive_depth_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3 = build_positive_depth_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

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
    train_samples = [s for s in samples1 if s["base_group"] in common_set]
    val_samples = [s for s in samples2 if s["base_group"] in common_set]
    test_samples = [s for s in samples3 if s["base_group"] in common_set]

    if len(train_samples) == 0 or len(val_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("Empty train, val, or test samples for stage3 raw routed depth v2 training.")

    raw_scale = compute_raw_scale(train_records)
    concept_mean, concept_std = compute_concept_stats(train_samples)

    ds_train = PositiveDepthDataset(
        train_records,
        train_samples,
        raw_scale=raw_scale,
        concept_mean=concept_mean,
        concept_std=concept_std,
        is_train=True,
        aug_noise_std=cfg.aug_noise_std,
        aug_scale_jitter=cfg.aug_scale_jitter,
        aug_frame_dropout=cfg.aug_frame_dropout,
    )
    ds_val = PositiveDepthDataset(val_records, val_samples, raw_scale, concept_mean, concept_std, is_train=False)
    ds_test = PositiveDepthDataset(test_records, test_samples, raw_scale, concept_mean, concept_std, is_train=False)

    train_depth = np.array([int(s["depth_coarse_index"]) for s in train_samples], dtype=np.int32)
    train_phase_weight = np.array([float(s["sample_weight"]) for s in train_samples], dtype=np.float32)
    counts = np.bincount(train_depth, minlength=len(COARSE_DEPTH_ORDER)).astype(np.float32)
    class_weight_np = counts.sum() / np.maximum(counts * len(COARSE_DEPTH_ORDER), 1.0)
    sampler_weight_np = class_weight_np[train_depth] * train_phase_weight

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=WeightedRandomSampler(torch.tensor(sampler_weight_np, dtype=torch.float32), num_samples=len(train_samples), replacement=True),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight_t = torch.tensor(class_weight_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=cfg.label_smoothing)

    router, router_mode = load_frozen_size_router(cfg.stage2_ckpt, device)
    model = RawSizeRoutedDepthModel(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        num_size_classes=len(SIZE_VALUES_CM),
        num_depth_classes=len(COARSE_DEPTH_ORDER),
    ).to(device)

    if cfg.warm_start_ckpt and os.path.exists(cfg.warm_start_ckpt):
        warm_ckpt = torch.load(cfg.warm_start_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(warm_ckpt["model_state_dict"], strict=True)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_gt_bal_acc": [],
        "val_hard_bal_acc": [],
        "val_soft_bal_acc": [],
        "val_size_top1": [],
        "test_gt_bal_acc": [],
        "test_hard_bal_acc": [],
        "test_soft_bal_acc": [],
        "test_size_top1": [],
    }
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for raw_x, norm_x, size_idx, depth_idx, _concept_target, _sample_weight in train_loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)

            with torch.no_grad():
                _size_logits, size_probs, _size_reg_cm = router_forward(router, router_mode, raw_x, norm_x)
                size_pred_idx = torch.argmax(size_probs, dim=1)
                top2_probs = build_top2_probs(size_probs)

            optimizer.zero_grad()
            gt_logits, _ = model(raw_x, norm_x, size_idx)
            hard_logits, _ = model(raw_x, norm_x, size_pred_idx)
            soft_logits, _ = model.forward_soft(raw_x, norm_x, size_probs)
            top2_logits, _ = model.forward_soft(raw_x, norm_x, top2_probs)

            loss_gt = criterion(gt_logits, depth_idx)
            loss_hard = criterion(hard_logits, depth_idx)
            loss_soft = criterion(soft_logits, depth_idx)
            loss_top2 = criterion(top2_logits, depth_idx)
            consistency = nn.functional.kl_div(
                nn.functional.log_softmax(soft_logits, dim=1),
                nn.functional.softmax(gt_logits.detach(), dim=1),
                reduction="batchmean",
            )
            loss = (
                float(cfg.gt_loss_weight) * loss_gt
                + float(cfg.hard_loss_weight) * loss_hard
                + float(cfg.soft_loss_weight) * loss_soft
                + float(cfg.top2_loss_weight) * loss_top2
                + float(cfg.consistency_kl_weight) * consistency
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()
            total_loss += float(loss.item()) * int(depth_idx.shape[0])
            total_n += int(depth_idx.shape[0])
        scheduler.step()

        val_metrics = evaluate_model(
            model,
            router,
            router_mode,
            val_loader,
            device,
            criterion,
            cfg.gt_loss_weight,
            cfg.hard_loss_weight,
            cfg.soft_loss_weight,
            cfg.top2_loss_weight,
        )
        test_metrics = evaluate_model(
            model,
            router,
            router_mode,
            test_loader,
            device,
            criterion,
            cfg.gt_loss_weight,
            cfg.hard_loss_weight,
            cfg.soft_loss_weight,
            cfg.top2_loss_weight,
        )

        history["train_loss"].append(float(total_loss / max(total_n, 1)))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_gt_bal_acc"].append(float(val_metrics["gt_route"]["balanced_accuracy"]))
        history["val_hard_bal_acc"].append(float(val_metrics["hard_route"]["balanced_accuracy"]))
        history["val_soft_bal_acc"].append(float(val_metrics["soft_route"]["balanced_accuracy"]))
        history["val_size_top1"].append(float(val_metrics["size_top1"]))
        history["test_gt_bal_acc"].append(float(test_metrics["gt_route"]["balanced_accuracy"]))
        history["test_hard_bal_acc"].append(float(test_metrics["hard_route"]["balanced_accuracy"]))
        history["test_soft_bal_acc"].append(float(test_metrics["soft_route"]["balanced_accuracy"]))
        history["test_size_top1"].append(float(test_metrics["size_top1"]))

        score = (
            float(val_metrics["hard_route"]["balanced_accuracy"]),
            float(val_metrics["soft_route"]["balanced_accuracy"]),
            float(val_metrics["gt_route"]["balanced_accuracy"]),
            -float(val_metrics["loss"]),
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

    best_path = os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_v2_best.pth")
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "input_shape": list(INPUT_SHAPE),
            "raw_scale": float(raw_scale),
            "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
            "class_weight": class_weight_np.tolist(),
            "size_routed": True,
            "training_mode": "route_aware_with_frozen_raw_stage2_router",
            "stage2_ckpt": cfg.stage2_ckpt,
            "stage2_mode": router_mode,
        },
        best_path,
    )

    plot_curves(history, os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_v2_curves.png"))
    majority_baseline = depth_majority_baseline(train_samples, test_samples)

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "RawSizeRoutedDepthModelV2",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "training_mode": "route_aware_with_frozen_raw_stage2_router",
        "stage2_ckpt": cfg.stage2_ckpt,
        "stage2_mode": router_mode,
        "size_routed": True,
        "raw_scale": float(raw_scale),
        "class_weight": class_weight_np.tolist(),
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "test_sample_count": len(test_samples),
        "test_majority_baseline": majority_baseline,
        "best_epoch": int(best["epoch"]),
        "val_metrics": to_jsonable_metrics(best["val_metrics"]),
        "test_metrics": to_jsonable_metrics(best["test_metrics"]),
        "history_tail": {key: [float(x) for x in values[-10:]] for key, values in history.items()},
    }
    with open(os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
