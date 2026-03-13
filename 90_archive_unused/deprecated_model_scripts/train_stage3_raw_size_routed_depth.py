import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

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

from raw_size_routed_depth_model import RawSizeRoutedDepthModel
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, INPUT_SHAPE, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for raw_x, norm_x, size_idx, depth_idx, _concept_target, _sample_weight in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            logits, _probs = model(raw_x, norm_x, size_idx)
            loss = criterion(logits, depth_idx)
            total_loss += float(loss.item()) * int(depth_idx.shape[0])
            total_n += int(depth_idx.shape[0])
            y_true.append(depth_idx.cpu().numpy().astype(np.int32))
            y_pred.append(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32))
    y_true_np = np.concatenate(y_true).astype(np.int32)
    y_pred_np = np.concatenate(y_pred).astype(np.int32)
    cm = confusion_matrix_counts(y_true_np, y_pred_np, len(COARSE_DEPTH_ORDER))
    return {
        "loss": float(total_loss / max(total_n, 1)),
        "accuracy": float(np.mean(y_true_np == y_pred_np)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "confusion_matrix": cm.tolist(),
        "count": int(total_n),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def to_jsonable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        clean[key] = value
    return clean


def plot_curves(history: Dict[str, List[float]], output_path: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["test_acc"], label="test acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_bal_acc"], label="val bal acc")
    axes[2].plot(epochs, history["test_bal_acc"], label="test bal acc")
    axes[2].set_title("Balanced Accuracy")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERD3ROUTE_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERD3ROUTE_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERD3ROUTE_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERD3ROUTE_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERD3ROUTE_EPOCHS", "90"))
    batch_size: int = int(os.environ.get("PAPERD3ROUTE_BATCH_SIZE", "48"))
    eval_batch_size: int = int(os.environ.get("PAPERD3ROUTE_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERD3ROUTE_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("PAPERD3ROUTE_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("PAPERD3ROUTE_DROPOUT", "0.25"))
    frame_feature_dim: int = int(os.environ.get("PAPERD3ROUTE_FRAME_DIM", "24"))
    temporal_channels: int = int(os.environ.get("PAPERD3ROUTE_TEMPORAL_CHANNELS", "48"))
    temporal_blocks: int = int(os.environ.get("PAPERD3ROUTE_TEMPORAL_BLOCKS", "3"))
    patience: int = int(os.environ.get("PAPERD3ROUTE_PATIENCE", "18"))
    grad_clip: float = float(os.environ.get("PAPERD3ROUTE_GRAD_CLIP", "1.0"))
    aug_noise_std: float = float(os.environ.get("PAPERD3ROUTE_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERD3ROUTE_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERD3ROUTE_AUG_FRAME_DROPOUT", "0.02"))
    label_smoothing: float = float(os.environ.get("PAPERD3ROUTE_LABEL_SMOOTH", "0.03"))
    num_workers: int = int(os.environ.get("PAPERD3ROUTE_NUM_WORKERS", "0"))

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        self.data_root = os.environ.get("PAPERD3ROUTE_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERD3ROUTE_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERD3ROUTE_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERD3ROUTE_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERD3ROUTE_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_raw_size_routed_depth"),
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
        raise RuntimeError("Empty train, val, or test samples for stage3 raw routed depth training.")

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
    model = RawSizeRoutedDepthModel(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        num_size_classes=len(SIZE_VALUES_CM),
        num_depth_classes=len(COARSE_DEPTH_ORDER),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "test_acc": [], "test_bal_acc": []}
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
            optimizer.zero_grad()
            logits, _probs = model(raw_x, norm_x, size_idx)
            loss = criterion(logits, depth_idx)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optimizer.step()
            total_loss += float(loss.item()) * int(depth_idx.shape[0])
            total_n += int(depth_idx.shape[0])
        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, device, criterion)
        test_metrics = evaluate_model(model, test_loader, device, criterion)

        history["train_loss"].append(float(total_loss / max(total_n, 1)))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_bal_acc"].append(float(val_metrics["balanced_accuracy"]))
        history["test_acc"].append(float(test_metrics["accuracy"]))
        history["test_bal_acc"].append(float(test_metrics["balanced_accuracy"]))

        score = (
            float(val_metrics["balanced_accuracy"]),
            float(val_metrics["accuracy"]),
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

    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "input_shape": list(INPUT_SHAPE),
            "raw_scale": float(raw_scale),
            "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
            "class_weight": class_weight_np.tolist(),
            "size_routed": True,
        },
        os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_best.pth"),
    )

    plot_curves(history, os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_curves.png"))
    majority_baseline = depth_majority_baseline(train_samples, test_samples)

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "RawSizeRoutedDepthModel",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
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
    with open(os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_summary.json"), "w", encoding="utf-8") as f:
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
                    "end_row": int(sample["end_row"]),
                    "center_row": int(sample["center_row"]),
                    "size_class_index": int(sample["size_class_index"]),
                    "depth_coarse_index": int(sample["depth_coarse_index"]),
                    "phase_band": sample["phase_band"],
                    "sample_weight": float(sample["sample_weight"]),
                }
            )
    with open(os.path.join(cfg.output_dir, "paper_stage3_raw_size_routed_depth_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
