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

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from task_protocol_v1 import COARSE_DEPTH_ORDER, SIZE_VALUES_CM, protocol_summary
from train_stage3_size_conditioned_mlp_depth import (
    SELECTED_FEATURES,
    balanced_accuracy_from_cm,
    confusion_matrix_counts,
    plot_confusion_matrix,
)


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class DepthSpecialistMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), len(COARSE_DEPTH_ORDER)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
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
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def size_majority_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, object]:
    y_train = train_df["depth_coarse_index"].to_numpy(dtype=np.int32)
    y_true = eval_df["depth_coarse_index"].to_numpy(dtype=np.int32)
    majority = int(np.bincount(y_train, minlength=len(COARSE_DEPTH_ORDER)).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)
    cm = confusion_matrix_counts(y_true, y_pred, len(COARSE_DEPTH_ORDER))
    return {
        "majority_class": majority,
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "confusion_matrix": cm.tolist(),
    }


def plot_curves(curves: Dict[int, Dict[str, List[float]]], output_path: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for size_idx, ax in enumerate(axes[: len(SIZE_VALUES_CM)]):
        hist = curves[size_idx]
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="train")
        ax.plot(epochs, hist["val_loss"], label="val")
        ax2 = ax.twinx()
        ax2.plot(epochs, hist["val_bal_acc"], label="val bal acc", color="#c44e52", alpha=0.75)
        ax.set_title(f"size={SIZE_VALUES_CM[size_idx]:g}cm")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    axes[7].axis("off")
    axes[8].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERD3SPEC_SEED", "2026"))
    epochs: int = int(os.environ.get("PAPERD3SPEC_EPOCHS", "120"))
    batch_size: int = int(os.environ.get("PAPERD3SPEC_BATCH_SIZE", "48"))
    lr: float = float(os.environ.get("PAPERD3SPEC_LR", "8e-4"))
    weight_decay: float = float(os.environ.get("PAPERD3SPEC_WEIGHT_DECAY", "3e-4"))
    patience: int = int(os.environ.get("PAPERD3SPEC_PATIENCE", "20"))
    hidden_dim: int = int(os.environ.get("PAPERD3SPEC_HIDDEN", "48"))
    dropout: float = float(os.environ.get("PAPERD3SPEC_DROPOUT", "0.10"))
    label_smoothing: float = float(os.environ.get("PAPERD3SPEC_LABEL_SMOOTH", "0.02"))

    def __post_init__(self):
        self.feature_table = os.environ.get(
            "PAPERD3SPEC_FEATURE_TABLE",
            os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_baselines_v1", "xgboost_window_feature_table.csv"),
        )
        self.output_dir = os.environ.get(
            "PAPERD3SPEC_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_size_specialist_depth"),
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

    pos = df[df["label"] == 1].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_val_true, all_val_pred = [], []
    all_test_true, all_test_pred = [], []
    per_size_summary = []
    histories = {}
    ckpt_dir = os.path.join(cfg.output_dir, "size_specialist_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    for size_idx, size_cm in enumerate(SIZE_VALUES_CM):
        size_df = pos[pos["size_class_index"] == size_idx].reset_index(drop=True)
        train_df = size_df[size_df["file_name"] == "1.CSV"].reset_index(drop=True)
        val_df = size_df[size_df["file_name"] == "2.CSV"].reset_index(drop=True)
        test_df = size_df[size_df["file_name"] == "3.CSV"].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            continue

        x_train = train_df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        x_val = val_df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        x_test = test_df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        mean = x_train.mean(axis=0, keepdims=True)
        std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std

        y_train = train_df["depth_coarse_index"].to_numpy(dtype=np.int64)
        y_val = val_df["depth_coarse_index"].to_numpy(dtype=np.int64)
        y_test = test_df["depth_coarse_index"].to_numpy(dtype=np.int64)

        counts = np.bincount(y_train, minlength=len(COARSE_DEPTH_ORDER)).astype(np.float32)
        class_weights = counts.sum() / np.maximum(counts * len(COARSE_DEPTH_ORDER), 1.0)
        sample_weights = class_weights[y_train]

        train_ds = FeatureDataset(x_train, y_train)
        val_ds = FeatureDataset(x_val, y_val)
        test_ds = FeatureDataset(x_test, y_test)
        sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.float32), num_samples=len(y_train), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = DepthSpecialistMLP(input_dim=len(SELECTED_FEATURES), hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=cfg.label_smoothing,
        )
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

        hist = {"train_loss": [], "val_loss": [], "val_bal_acc": []}
        best = None
        patience_left = int(cfg.patience)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            running_n = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += float(loss.item()) * int(y.shape[0])
                running_n += int(y.shape[0])
            scheduler.step()

            val_metrics = evaluate(model, val_loader, device, criterion)
            hist["train_loss"].append(float(running_loss / max(running_n, 1)))
            hist["val_loss"].append(float(val_metrics["loss"]))
            hist["val_bal_acc"].append(float(val_metrics["balanced_accuracy"]))

            score = (float(val_metrics["balanced_accuracy"]), float(val_metrics["accuracy"]), -float(val_metrics["loss"]))
            if best is None or score > best["score"]:
                best = {
                    "epoch": epoch,
                    "score": score,
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                }
                patience_left = int(cfg.patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        model.load_state_dict(best["state_dict"])
        val_metrics = evaluate(model, val_loader, device, criterion)
        test_metrics = evaluate(model, test_loader, device, criterion)
        baseline = size_majority_baseline(train_df, test_df)
        histories[size_idx] = hist

        torch.save(
            {
                "model_state_dict": best["state_dict"],
                "size_cm": float(size_cm),
                "selected_features": list(SELECTED_FEATURES),
                "feature_mean": mean.reshape(-1).tolist(),
                "feature_std": std.reshape(-1).tolist(),
            },
            os.path.join(ckpt_dir, f"size_{size_cm:g}cm_depth_specialist.pth"),
        )

        all_val_true.append(val_metrics["y_true"])
        all_val_pred.append(val_metrics["y_pred"])
        all_test_true.append(test_metrics["y_true"])
        all_test_pred.append(test_metrics["y_pred"])

        per_size_summary.append(
            {
                "size_cm": float(size_cm),
                "best_epoch": int(best["epoch"]),
                "train_count": int(len(train_df)),
                "val_count": int(len(val_df)),
                "test_count": int(len(test_df)),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
                "test_majority_accuracy": float(baseline["accuracy"]),
                "test_majority_balanced_accuracy": float(baseline["balanced_accuracy"]),
                "test_confusion_matrix": test_metrics["confusion_matrix"],
            }
        )

    val_true = np.concatenate(all_val_true).astype(np.int32)
    val_pred = np.concatenate(all_val_pred).astype(np.int32)
    test_true = np.concatenate(all_test_true).astype(np.int32)
    test_pred = np.concatenate(all_test_pred).astype(np.int32)
    val_cm = confusion_matrix_counts(val_true, val_pred, len(COARSE_DEPTH_ORDER))
    test_cm = confusion_matrix_counts(test_true, test_pred, len(COARSE_DEPTH_ORDER))

    plot_curves(histories, os.path.join(cfg.output_dir, "paper_stage3_size_specialist_curves.png"))
    plot_confusion_matrix(val_cm, ["shallow", "middle", "deep"], "Size-specialist depth validation confusion", os.path.join(cfg.output_dir, "paper_stage3_size_specialist_val_confusion.png"))
    plot_confusion_matrix(test_cm, ["shallow", "middle", "deep"], "Size-specialist depth test confusion", os.path.join(cfg.output_dir, "paper_stage3_size_specialist_test_confusion.png"))

    overall_train = pos[pos["file_name"] == "1.CSV"].reset_index(drop=True)
    overall_test = pos[pos["file_name"] == "3.CSV"].reset_index(drop=True)
    overall_baseline = size_majority_baseline(overall_train, overall_test)

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "SizeSpecialistDepthMLP",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "selected_features": list(SELECTED_FEATURES),
        "per_size_summary": per_size_summary,
        "aggregate_val": {
            "accuracy": float(np.mean(val_true == val_pred)),
            "balanced_accuracy": float(balanced_accuracy_from_cm(val_cm)),
            "confusion_matrix": val_cm.tolist(),
            "count": int(len(val_true)),
        },
        "aggregate_test": {
            "accuracy": float(np.mean(test_true == test_pred)),
            "balanced_accuracy": float(balanced_accuracy_from_cm(test_cm)),
            "confusion_matrix": test_cm.tolist(),
            "count": int(len(test_true)),
        },
        "size_conditioned_majority_baseline_test": overall_baseline,
    }
    with open(os.path.join(cfg.output_dir, "paper_stage3_size_specialist_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
