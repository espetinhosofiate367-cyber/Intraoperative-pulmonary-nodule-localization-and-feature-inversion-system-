import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

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
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from task_protocol_v1 import COARSE_DEPTH_ORDER, SIZE_VALUES_CM, protocol_summary


SELECTED_FEATURES = [
    "maxframe_raw_centroid_row",
    "centroid_row_min",
    "deltaframe_centroid_row",
    "maxframe_raw_centroid_col",
    "hotspot_radius_max",
    "meanframe_raw_hotspot_radius",
    "hotspot_radius_min",
    "second_moment_spread_max",
    "anisotropy_ratio_min",
    "anisotropy_ratio_last",
    "peak_count_min",
    "center_border_contrast_max",
    "window_norm_global_std",
    "raw_max_max",
    "spatial_entropy_max",
    "window_rise_time_to_peak",
    "window_peak_persistence_ratio",
    "window_decay_after_peak",
    "window_window_raw_sum_gain",
]


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true.astype(np.int32), y_pred.astype(np.int32)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    recalls = []
    for idx in range(cm.shape[0]):
        total = int(np.sum(cm[idx]))
        recalls.append(float(cm[idx, idx] / total) if total > 0 else np.nan)
    recalls = np.asarray(recalls, dtype=np.float64)
    if np.all(~np.isfinite(recalls)):
        return float("nan")
    return float(np.nanmean(recalls))


class PositiveFeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, size_idx: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.size_idx = torch.tensor(size_idx, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.size_idx[idx], self.y[idx]


class SizeConditionedDepthMLP(nn.Module):
    def __init__(self, input_dim: int, num_size_classes: int = 7, size_embed_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.size_embedding = nn.Embedding(int(num_size_classes), int(size_embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(input_dim) + int(size_embed_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim // 2), len(COARSE_DEPTH_ORDER)),
        )

    def forward(self, x: torch.Tensor, size_idx: torch.Tensor) -> torch.Tensor:
        emb = self.size_embedding(size_idx)
        fused = torch.cat([x, emb], dim=1)
        return self.mlp(fused)


def build_size_conditioned_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, object]:
    train_depth = train_df["depth_coarse_index"].to_numpy(dtype=np.int32)
    overall_majority = int(np.bincount(train_depth, minlength=len(COARSE_DEPTH_ORDER)).argmax())
    size_to_majority = {}
    for size_idx in range(len(SIZE_VALUES_CM)):
        subset = train_df[train_df["size_class_index"] == size_idx]["depth_coarse_index"].to_numpy(dtype=np.int32)
        if subset.size == 0:
            size_to_majority[size_idx] = overall_majority
        else:
            size_to_majority[size_idx] = int(np.bincount(subset, minlength=len(COARSE_DEPTH_ORDER)).argmax())
    y_true = eval_df["depth_coarse_index"].to_numpy(dtype=np.int32)
    y_pred = np.array([size_to_majority[int(v)] for v in eval_df["size_class_index"].to_numpy(dtype=np.int32)], dtype=np.int32)
    cm = confusion_matrix_counts(y_true, y_pred, len(COARSE_DEPTH_ORDER))
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, size_idx, y in loader:
            x = x.to(device)
            size_idx = size_idx.to(device)
            y = y.to(device)
            logits = model(x, size_idx)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * int(y.shape[0])
            total_n += int(y.shape[0])
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_true_np = np.concatenate(y_true).astype(np.int32)
    y_pred_np = np.concatenate(y_pred).astype(np.int32)
    cm = confusion_matrix_counts(y_true_np, y_pred_np, len(COARSE_DEPTH_ORDER))
    return {
        "loss": float(total_loss / max(total_n, 1)),
        "accuracy": float(np.mean(y_true_np == y_pred_np)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "confusion_matrix": cm.tolist(),
        "count": int(total_n),
    }


def plot_curves(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].plot(epochs, history["test_acc"], label="test")
    axes[1].set_title("Accuracy")
    axes[2].plot(epochs, history["val_bal_acc"], label="val")
    axes[2].plot(epochs, history["test_bal_acc"], label="test")
    axes[2].set_title("Balanced Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, output_path: str) -> None:
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERD3MLP_SEED", "2026"))
    epochs: int = int(os.environ.get("PAPERD3MLP_EPOCHS", "80"))
    batch_size: int = int(os.environ.get("PAPERD3MLP_BATCH_SIZE", "64"))
    lr: float = float(os.environ.get("PAPERD3MLP_LR", "8e-4"))
    weight_decay: float = float(os.environ.get("PAPERD3MLP_WEIGHT_DECAY", "5e-4"))
    patience: int = int(os.environ.get("PAPERD3MLP_PATIENCE", "16"))
    hidden_dim: int = int(os.environ.get("PAPERD3MLP_HIDDEN", "64"))
    size_embed_dim: int = int(os.environ.get("PAPERD3MLP_SIZE_EMBED", "8"))
    dropout: float = float(os.environ.get("PAPERD3MLP_DROPOUT", "0.15"))
    label_smoothing: float = float(os.environ.get("PAPERD3MLP_LABEL_SMOOTH", "0.03"))

    def __post_init__(self):
        self.feature_table = os.environ.get(
            "PAPERD3MLP_FEATURE_TABLE",
            os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_baselines_v1", "xgboost_window_feature_table.csv"),
        )
        self.output_dir = os.environ.get(
            "PAPERD3MLP_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_size_conditioned_mlp_depth"),
        )


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = pd.read_csv(cfg.feature_table)
    missing = [col for col in SELECTED_FEATURES if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing selected features: {missing}")

    pos_train = df[(df["split"] == "train_all") & (df["label"] == 1)].reset_index(drop=True)
    pos_val = df[(df["split"] == "val") & (df["label"] == 1)].reset_index(drop=True)
    pos_test = df[(df["split"] == "test") & (df["label"] == 1)].reset_index(drop=True)

    x_train = pos_train[SELECTED_FEATURES].to_numpy(dtype=np.float32)
    x_val = pos_val[SELECTED_FEATURES].to_numpy(dtype=np.float32)
    x_test = pos_test[SELECTED_FEATURES].to_numpy(dtype=np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    size_train = pos_train["size_class_index"].to_numpy(dtype=np.int64)
    size_val = pos_val["size_class_index"].to_numpy(dtype=np.int64)
    size_test = pos_test["size_class_index"].to_numpy(dtype=np.int64)
    y_train = pos_train["depth_coarse_index"].to_numpy(dtype=np.int64)
    y_val = pos_val["depth_coarse_index"].to_numpy(dtype=np.int64)
    y_test = pos_test["depth_coarse_index"].to_numpy(dtype=np.int64)

    counts = np.bincount(y_train, minlength=len(COARSE_DEPTH_ORDER)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts * len(COARSE_DEPTH_ORDER), 1.0)
    sample_weights = class_weights[y_train]

    train_ds = PositiveFeatureDataset(x_train, size_train, y_train)
    val_ds = PositiveFeatureDataset(x_val, size_val, y_val)
    test_ds = PositiveFeatureDataset(x_test, size_test, y_test)

    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float32), num_samples=len(y_train), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SizeConditionedDepthMLP(
        input_dim=len(SELECTED_FEATURES),
        num_size_classes=len(SIZE_VALUES_CM),
        size_embed_dim=cfg.size_embed_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device), label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "test_acc": [], "test_bal_acc": []}
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        for x, size_idx, y in train_loader:
            x = x.to(device)
            size_idx = size_idx.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x, size_idx)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * int(y.shape[0])
            running_n += int(y.shape[0])
        scheduler.step()

        train_loss = float(running_loss / max(running_n, 1))
        val_metrics = evaluate(model, val_loader, device, criterion)
        test_metrics = evaluate(model, test_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_bal_acc"].append(float(val_metrics["balanced_accuracy"]))
        history["test_acc"].append(float(test_metrics["accuracy"]))
        history["test_bal_acc"].append(float(test_metrics["balanced_accuracy"]))

        score = (float(val_metrics["balanced_accuracy"]), float(val_metrics["accuracy"]), -float(val_metrics["loss"]))
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
        raise RuntimeError("No best checkpoint found.")

    baseline = build_size_conditioned_baseline(pos_train, pos_test)
    plot_curves(history, os.path.join(cfg.output_dir, "paper_stage3_size_conditioned_mlp_curves.png"))
    plot_confusion_matrix(
        np.asarray(best["test_metrics"]["confusion_matrix"], dtype=np.int32),
        ["shallow", "middle", "deep"],
        "Size-conditioned MLP depth confusion",
        os.path.join(cfg.output_dir, "paper_stage3_size_conditioned_mlp_confusion.png"),
    )

    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "selected_features": list(SELECTED_FEATURES),
            "feature_mean": mean.reshape(-1).tolist(),
            "feature_std": std.reshape(-1).tolist(),
            "size_conditioning": "gt_size_class",
        },
        os.path.join(cfg.output_dir, "paper_stage3_size_conditioned_mlp_best.pth"),
    )

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "SizeConditionedDepthMLP",
        "selected_features": list(SELECTED_FEATURES),
        "train_sample_count": int(len(pos_train)),
        "val_sample_count": int(len(pos_val)),
        "test_sample_count": int(len(pos_test)),
        "class_weights": class_weights.tolist(),
        "best_epoch": int(best["epoch"]),
        "val_metrics": best["val_metrics"],
        "test_metrics": best["test_metrics"],
        "size_conditioned_majority_baseline": baseline,
        "history_tail": {key: [float(v) for v in vals[-10:]] for key, vals in history.items()},
    }
    with open(os.path.join(cfg.output_dir, "paper_stage3_size_conditioned_mlp_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
