import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)
REPO_ROOT = os.path.dirname(CODE_ARCHIVE_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if CODE_ARCHIVE_DIR not in sys.path:
    sys.path.insert(0, CODE_ARCHIVE_DIR)

from concept_guided_depth_model import CONCEPT_NAMES, ConceptGuidedDepthModel
from task_protocol_v1 import (
    COARSE_DEPTH_ORDER,
    INPUT_SEQ_LEN,
    INPUT_SHAPE,
    SIZE_VALUES_CM,
    WINDOW_STRIDE,
    protocol_summary,
)
from train_stage1_detection import is_center_positive
from triplet_repeat_classifier.train_triplet_repeat_classifier import (
    compress_samples_by_gap,
    env_bool,
    filter_labels_for_file,
    load_json,
    normalize_frames,
    parse_float_from_cm_text,
    parse_size_depth_from_group,
    read_csv_data,
    sanitize_segments,
    set_seed,
)


def compute_center_border_contrast(frame: np.ndarray) -> float:
    center = frame[4:8, 2:6]
    border_mask = np.ones_like(frame, dtype=bool)
    border_mask[4:8, 2:6] = False
    border = frame[border_mask]
    return float(np.mean(center) - np.mean(border))


def compute_second_moment_spread(frame: np.ndarray) -> float:
    weights = np.clip(frame.astype(np.float64), 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-8:
        return 0.0
    rr, cc = np.meshgrid(
        np.linspace(-1.0, 1.0, frame.shape[0]),
        np.linspace(-1.0, 1.0, frame.shape[1]),
        indexing="ij",
    )
    r0 = float(np.sum(rr * weights) / total)
    c0 = float(np.sum(cc * weights) / total)
    spread = np.sqrt(np.sum(weights * ((rr - r0) ** 2 + (cc - c0) ** 2)) / total)
    return float(spread)


def compute_phase_proxy(frame_energy: np.ndarray) -> Tuple[float, float]:
    peak_idx = int(np.argmax(frame_energy))
    peak_value = float(frame_energy[peak_idx]) if frame_energy.size else 0.0
    if peak_value <= 1e-8:
        return 0.0, 0.0
    early = float(np.mean(frame_energy[:peak_idx])) / peak_value if peak_idx > 0 else 0.0
    release = float(np.mean(frame_energy[peak_idx + 1 :])) / peak_value if peak_idx + 1 < len(frame_energy) else 0.0
    return early, release


def classify_window_phase(frame_energy: np.ndarray, center_local_idx: int) -> str:
    peak_idx = int(np.argmax(frame_energy))
    delta = peak_idx - int(center_local_idx)
    if delta >= 2:
        return "loading_early"
    if delta == 1:
        return "loading_late"
    if abs(delta) <= 1:
        return "peak_neighborhood"
    return "release"


def compute_concept_targets(raw_window: np.ndarray, norm_window: np.ndarray) -> np.ndarray:
    frame_energy = raw_window.sum(axis=(1, 2)).astype(np.float64)
    peak_idx = int(np.argmax(frame_energy))
    peak_raw = raw_window[peak_idx].astype(np.float64)
    peak_norm = norm_window[peak_idx].astype(np.float64)
    peak_strength = float(np.max(peak_raw))
    integrated_response = float(np.sum(peak_raw))
    spread_extent = compute_second_moment_spread(peak_norm)
    shape_contrast = compute_center_border_contrast(peak_norm)
    temporal_prominence = float((frame_energy[peak_idx] - np.mean(frame_energy)) / (np.std(frame_energy) + 1e-6))
    phase_early, phase_release = compute_phase_proxy(frame_energy)
    return np.array(
        [
            peak_strength,
            integrated_response,
            spread_extent,
            shape_contrast,
            temporal_prominence,
            phase_early,
            phase_release,
        ],
        dtype=np.float32,
    )


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(1, int(num_classes), device=labels.device).view(1, -1)
    return (labels.view(-1, 1) >= thresholds).float()


def ordinal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = ordinal_targets(labels, logits.shape[1] + 1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
    if class_weight is not None:
        loss = loss * class_weight[labels.long()]
    if sample_weight is not None:
        loss = loss * sample_weight.view(-1)
        denom = torch.clamp(sample_weight.sum(), min=1.0)
        return loss.sum() / denom
    return loss.mean()


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
    recalls = np.array(recalls, dtype=np.float64)
    if np.all(~np.isfinite(recalls)):
        return float("nan")
    return float(np.nanmean(recalls))


class PositiveDepthDataset(Dataset):
    def __init__(
        self,
        records_by_key: Dict[str, dict],
        sample_records: List[dict],
        raw_scale: float,
        concept_mean: np.ndarray,
        concept_std: np.ndarray,
        is_train: bool = False,
        aug_noise_std: float = 0.0,
        aug_scale_jitter: float = 0.0,
        aug_frame_dropout: float = 0.0,
    ):
        self.records_by_key = records_by_key
        self.samples = sample_records
        self.raw_scale = float(max(raw_scale, 1e-6))
        self.concept_mean = concept_mean.astype(np.float32)
        self.concept_std = np.maximum(concept_std.astype(np.float32), 1e-6)
        self.is_train = bool(is_train)
        self.aug_noise_std = float(max(0.0, aug_noise_std))
        self.aug_scale_jitter = float(max(0.0, aug_scale_jitter))
        self.aug_frame_dropout = float(min(max(0.0, aug_frame_dropout), 0.5))

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
            raw_x = np.clip(raw_x, 0.0, 3.0)
            norm_x = np.clip(norm_x, 0.0, 1.0)

        concept_target = ((np.asarray(s["concept_targets"], dtype=np.float32) - self.concept_mean) / self.concept_std).astype(
            np.float32
        )
        return (
            torch.from_numpy(raw_x),
            torch.from_numpy(norm_x),
            torch.tensor(int(s["size_class_index"]), dtype=torch.long),
            torch.tensor(int(s["depth_coarse_index"]), dtype=torch.long),
            torch.from_numpy(concept_target),
            torch.tensor(float(s["sample_weight"]), dtype=torch.float32),
        )


def build_positive_depth_samples_for_file(
    label_map: Dict,
    target_file: str,
    data_root: str,
    seq_len: int,
    stride: int,
    dedup_gap: int,
) -> Tuple[Dict[str, dict], List[dict]]:
    filtered = filter_labels_for_file(label_map, target_file)
    records_by_key: Dict[str, dict] = {}
    sample_records: List[dict] = []
    phase_weights = {
        "loading_early": 1.0,
        "loading_late": 0.85,
        "peak_neighborhood": 0.70,
        "release": 1.0,
    }

    for base_group in sorted(filtered.keys()):
        size_text, depth_text = parse_size_depth_from_group(base_group)
        file_path = os.path.join(data_root, size_text, depth_text, target_file)
        if not os.path.exists(file_path):
            continue

        raw_96 = read_csv_data(file_path)
        n_frames = len(raw_96)
        if n_frames < seq_len:
            continue

        segments = sanitize_segments(filtered[base_group]["segments"], n_frames)
        if not segments:
            continue

        raw_maps = raw_96.reshape(n_frames, 12, 8).astype(np.float32)
        norm_maps = normalize_frames(raw_96[:n_frames]).astype(np.float32)
        group_key = f"{base_group}|{target_file}"
        size_cm = parse_float_from_cm_text(size_text)
        depth_cm = parse_float_from_cm_text(depth_text)
        records_by_key[group_key] = {
            "group_key": group_key,
            "base_group": base_group,
            "size_text": size_text,
            "depth_text": depth_text,
            "size_cm": float(size_cm),
            "depth_cm": float(depth_cm),
            "size_class_index": int(np.argmin(np.abs(np.array(SIZE_VALUES_CM, dtype=np.float32) - size_cm))),
            "depth_coarse_index": int(0 if depth_cm <= 1.0 else 1 if depth_cm <= 2.0 else 2),
            "file_name": target_file,
            "raw_frames": raw_maps,
            "norm_frames": norm_maps,
            "seq_len": int(seq_len),
        }

        candidate = []
        for end_row in range(seq_len - 1, n_frames, stride):
            center_row = end_row - (seq_len // 2)
            if not is_center_positive(center_row, segments):
                continue
            st = end_row - seq_len + 1
            raw_window = raw_maps[st : end_row + 1]
            norm_window = norm_maps[st : end_row + 1]
            frame_energy = raw_window.sum(axis=(1, 2))
            phase_band = classify_window_phase(frame_energy, center_row - st)
            concept_targets = compute_concept_targets(raw_window, norm_window).tolist()
            candidate.append(
                {
                    "group_key": group_key,
                    "base_group": base_group,
                    "file_name": target_file,
                    "label": 1,
                    "end_row": int(end_row),
                    "center_row": int(center_row),
                    "size_class_index": records_by_key[group_key]["size_class_index"],
                    "depth_coarse_index": records_by_key[group_key]["depth_coarse_index"],
                    "phase_band": phase_band,
                    "sample_weight": float(phase_weights.get(phase_band, 1.0)),
                    "concept_targets": concept_targets,
                }
            )

        compressed = compress_samples_by_gap(candidate, dedup_gap)
        sample_records.extend(compressed)

    return records_by_key, sample_records


def compute_raw_scale(records_by_key: Dict[str, dict]) -> float:
    maxima = []
    for rec in records_by_key.values():
        maxima.append(float(np.percentile(rec["raw_frames"], 99.5)))
    if not maxima:
        return 1.0
    return float(max(np.median(maxima), 1.0))


def compute_concept_stats(samples: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray([s["concept_targets"] for s in samples], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(len(CONCEPT_NAMES), dtype=np.float32), np.ones(len(CONCEPT_NAMES), dtype=np.float32)
    return arr.mean(axis=0), np.maximum(arr.std(axis=0), 1e-6)


def depth_majority_baseline(train_samples: List[dict], eval_samples: List[dict]) -> Dict[str, object]:
    y_train = np.array([int(s["depth_coarse_index"]) for s in train_samples], dtype=np.int32)
    y_eval = np.array([int(s["depth_coarse_index"]) for s in eval_samples], dtype=np.int32)
    majority = int(np.bincount(y_train, minlength=len(COARSE_DEPTH_ORDER)).argmax())
    pred = np.full_like(y_eval, fill_value=majority)
    cm = confusion_matrix_counts(y_eval, pred, len(COARSE_DEPTH_ORDER))
    return {
        "majority_class": majority,
        "accuracy": float(np.mean(y_eval == pred)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "confusion_matrix": cm.tolist(),
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

    axes[3].plot(epochs, history["val_concept_mae"], label="val concept mae")
    axes[3].plot(epochs, history["test_concept_mae"], label="test concept mae")
    axes[3].set_title("Concept MAE")
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    concept_loss_weight: float,
    grad_clip: float,
    class_weight: Optional[torch.Tensor],
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for raw_x, norm_x, size_idx, depth_idx, concept_target, sample_weight in loader:
        raw_x = raw_x.to(device)
        norm_x = norm_x.to(device)
        size_idx = size_idx.to(device)
        depth_idx = depth_idx.to(device)
        concept_target = concept_target.to(device)
        sample_weight = sample_weight.to(device)

        optimizer.zero_grad()
        depth_logits, _depth_probs, concept_pred = model(raw_x, norm_x, size_idx)
        depth_loss = ordinal_loss(depth_logits, depth_idx, sample_weight=sample_weight, class_weight=class_weight)
        concept_loss = torch.abs(concept_pred - concept_target).mean(dim=1)
        if class_weight is not None:
            concept_loss = concept_loss * class_weight[depth_idx.long()]
        concept_loss = (concept_loss * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1.0)
        loss = depth_loss + float(concept_loss_weight) * concept_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        batch_weight = float(sample_weight.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
    return float(total_loss / max(total_weight, 1.0))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    concept_mean: np.ndarray,
    concept_std: np.ndarray,
    class_weight: Optional[torch.Tensor],
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    y_true, y_pred = [], []
    concept_mae_running = []

    mean_t = torch.tensor(concept_mean, dtype=torch.float32, device=device).view(1, -1)
    std_t = torch.tensor(concept_std, dtype=torch.float32, device=device).view(1, -1)

    with torch.no_grad():
        for raw_x, norm_x, size_idx, depth_idx, concept_target, sample_weight in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)
            concept_target = concept_target.to(device)
            sample_weight = sample_weight.to(device)

            depth_logits, _depth_probs, concept_pred = model(raw_x, norm_x, size_idx)
            depth_loss = ordinal_loss(depth_logits, depth_idx, sample_weight=sample_weight, class_weight=class_weight)
            concept_loss = torch.abs(concept_pred - concept_target).mean(dim=1)
            if class_weight is not None:
                concept_loss = concept_loss * class_weight[depth_idx.long()]
            concept_loss = (concept_loss * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1.0)
            loss = depth_loss + concept_loss

            pred_depth = ConceptGuidedDepthModel.ordinal_logits_to_class(depth_logits).cpu().numpy().astype(np.int32)
            y_true.append(depth_idx.cpu().numpy().astype(np.int32))
            y_pred.append(pred_depth)

            concept_pred_raw = concept_pred * std_t + mean_t
            concept_true_raw = concept_target * std_t + mean_t
            concept_mae = torch.abs(concept_pred_raw - concept_true_raw).mean(dim=1)
            concept_mae_running.append(concept_mae.cpu().numpy())

            batch_weight = float(sample_weight.sum().item())
            total_loss += float(loss.item()) * batch_weight
            total_weight += batch_weight

    y_true_np = np.concatenate(y_true).astype(np.int32)
    y_pred_np = np.concatenate(y_pred).astype(np.int32)
    cm = confusion_matrix_counts(y_true_np, y_pred_np, len(COARSE_DEPTH_ORDER))
    concept_mae_np = np.concatenate(concept_mae_running).astype(np.float32)

    return {
        "loss": float(total_loss / max(total_weight, 1.0)),
        "accuracy": float(np.mean(y_true_np == y_pred_np)),
        "balanced_accuracy": balanced_accuracy_from_cm(cm),
        "confusion_matrix": cm.tolist(),
        "count": int(len(y_true_np)),
        "concept_mae_mean": float(np.mean(concept_mae_np)),
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


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERD3RAW_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERD3RAW_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERD3RAW_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERD3RAW_DEDUP_GAP", "6"))
    epochs: int = int(os.environ.get("PAPERD3RAW_EPOCHS", "80"))
    batch_size: int = int(os.environ.get("PAPERD3RAW_BATCH_SIZE", "48"))
    eval_batch_size: int = int(os.environ.get("PAPERD3RAW_EVAL_BATCH_SIZE", "128"))
    lr: float = float(os.environ.get("PAPERD3RAW_LR", "2e-4"))
    weight_decay: float = float(os.environ.get("PAPERD3RAW_WEIGHT_DECAY", "1e-3"))
    dropout: float = float(os.environ.get("PAPERD3RAW_DROPOUT", "0.30"))
    frame_feature_dim: int = int(os.environ.get("PAPERD3RAW_FRAME_DIM", "24"))
    temporal_channels: int = int(os.environ.get("PAPERD3RAW_TEMPORAL_CHANNELS", "48"))
    temporal_blocks: int = int(os.environ.get("PAPERD3RAW_TEMPORAL_BLOCKS", "3"))
    size_embedding_dim: int = int(os.environ.get("PAPERD3RAW_SIZE_EMBED_DIM", "12"))
    patience: int = int(os.environ.get("PAPERD3RAW_PATIENCE", "16"))
    grad_clip: float = float(os.environ.get("PAPERD3RAW_GRAD_CLIP", "1.0"))
    concept_loss_weight: float = float(os.environ.get("PAPERD3RAW_CONCEPT_LOSS_WEIGHT", "0.35"))
    aug_noise_std: float = float(os.environ.get("PAPERD3RAW_AUG_NOISE_STD", "0.01"))
    aug_scale_jitter: float = float(os.environ.get("PAPERD3RAW_AUG_SCALE_JITTER", "0.06"))
    aug_frame_dropout: float = float(os.environ.get("PAPERD3RAW_AUG_FRAME_DROPOUT", "0.02"))
    num_workers: int = int(os.environ.get("PAPERD3RAW_NUM_WORKERS", "0"))
    save_plotly_html: bool = env_bool("PAPERD3RAW_SAVE_PLOTLY_HTML", False)

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        self.data_root = os.environ.get("PAPERD3RAW_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
        self.file1_labels = os.environ.get("PAPERD3RAW_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
        self.file2_labels = os.environ.get("PAPERD3RAW_FILE2_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file2.json"))
        self.file3_labels = os.environ.get("PAPERD3RAW_FILE3_LABELS", os.path.join(self.data_root, "manual_keyframe_labels_file3.json"))
        self.output_dir = os.environ.get(
            "PAPERD3RAW_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_raw_size_conditioned_depth"),
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
        raise RuntimeError("Empty train, val, or test samples for stage3 raw depth training.")

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
    sampler = WeightedRandomSampler(
        torch.tensor(sampler_weight_np, dtype=torch.float32),
        num_samples=len(train_samples),
        replacement=True,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight_t = torch.tensor(class_weight_np, dtype=torch.float32, device=device)
    model = ConceptGuidedDepthModel(
        seq_len=cfg.seq_len,
        frame_feature_dim=cfg.frame_feature_dim,
        temporal_channels=cfg.temporal_channels,
        temporal_blocks=cfg.temporal_blocks,
        dropout=cfg.dropout,
        size_embedding_dim=cfg.size_embedding_dim,
        num_size_classes=len(SIZE_VALUES_CM),
        num_concepts=len(CONCEPT_NAMES),
        num_depth_classes=len(COARSE_DEPTH_ORDER),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_bal_acc": [], "test_acc": [], "test_bal_acc": [], "val_concept_mae": [], "test_concept_mae": []}
    best = None
    patience_left = int(cfg.patience)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(
            model,
            loader_train,
            optimizer,
            device,
            cfg.concept_loss_weight,
            cfg.grad_clip,
            class_weight_t,
        )
        val_metrics = evaluate_model(model, loader_val, device, concept_mean, concept_std, class_weight_t)
        test_metrics = evaluate_model(model, loader_test, device, concept_mean, concept_std, class_weight_t)
        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_bal_acc"].append(float(val_metrics["balanced_accuracy"]))
        history["test_acc"].append(float(test_metrics["accuracy"]))
        history["test_bal_acc"].append(float(test_metrics["balanced_accuracy"]))
        history["val_concept_mae"].append(float(val_metrics["concept_mae_mean"]))
        history["test_concept_mae"].append(float(test_metrics["concept_mae_mean"]))

        score = (
            float(val_metrics["balanced_accuracy"]),
            float(val_metrics["accuracy"]),
            -float(val_metrics["concept_mae_mean"]),
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

    best_path = os.path.join(cfg.output_dir, "paper_stage3_raw_size_conditioned_depth_best.pth")
    torch.save(
        {
            "model_state_dict": best["state_dict"],
            "protocol_v1": protocol_summary(),
            "input_shape": list(INPUT_SHAPE),
            "raw_scale": float(raw_scale),
            "concept_names": list(CONCEPT_NAMES),
            "concept_mean": concept_mean.tolist(),
            "concept_std": concept_std.tolist(),
            "size_conditioning": "gt_size_class",
            "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
            "class_weight": class_weight_np.tolist(),
        },
        best_path,
    )

    majority_baseline = depth_majority_baseline(train_samples, test_samples)
    plot_curves(history, os.path.join(cfg.output_dir, "paper_stage3_raw_size_conditioned_depth_curves.png"))

    summary = {
        "protocol_v1": protocol_summary(),
        "model_name": "RawSizeConditionedConceptDepthModel",
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "size_conditioning": "gt_size_class",
        "concept_names": list(CONCEPT_NAMES),
        "raw_scale": float(raw_scale),
        "concept_mean": concept_mean.tolist(),
        "concept_std": concept_std.tolist(),
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
    with open(os.path.join(cfg.output_dir, "paper_stage3_raw_size_conditioned_depth_summary.json"), "w", encoding="utf-8") as f:
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
    with open(os.path.join(cfg.output_dir, "paper_stage3_raw_size_conditioned_depth_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
