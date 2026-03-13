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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)
REPO_ROOT = os.path.dirname(CODE_ARCHIVE_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if os.path.join(PROJECT_DIR, "models") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_DIR, "models"))
if CODE_ARCHIVE_DIR not in sys.path:
    sys.path.insert(0, CODE_ARCHIVE_DIR)

from depth_analysis_utils import frame_physics_features, window_temporal_features
from task_protocol_v1 import (
    COARSE_DEPTH_ORDER,
    INPUT_SEQ_LEN,
    WINDOW_STRIDE,
    depth_to_coarse_index,
    protocol_summary,
    size_to_class_index,
)
from train_stage1_detection import is_center_positive
from triplet_repeat_classifier.train_file12_holdout_file3 import (
    downsample_negatives,
    split_base_groups_train_val_balanced,
)
from triplet_repeat_classifier.train_triplet_repeat_classifier import (
    compress_samples_by_gap,
    filter_labels_for_file,
    load_json,
    normalize_frames,
    parse_float_from_cm_text,
    parse_size_depth_from_group,
    read_csv_data,
    sanitize_segments,
    select_best_f1_threshold,
    set_seed,
)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def cls_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_pred = (y_score >= float(threshold)).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    accuracy = float((tp + tn) / max(len(y_true), 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def topk_accuracy(logits_or_probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    if logits_or_probs.size == 0 or labels.size == 0:
        return float("nan")
    k = int(max(1, min(k, logits_or_probs.shape[1])))
    topk = np.argsort(-logits_or_probs, axis=1)[:, :k]
    hits = np.any(topk == labels.reshape(-1, 1), axis=1)
    return float(np.mean(hits))


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if pred.size == 0 or target.size == 0:
        return {"mae": float("nan"), "median_ae": float("nan")}
    ae = np.abs(pred.astype(np.float64) - target.astype(np.float64))
    return {"mae": float(np.mean(ae)), "median_ae": float(np.median(ae))}


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


def subset_multitask_metrics(
    size_probs: np.ndarray,
    size_reg_pred: np.ndarray,
    depth_probs: np.ndarray,
    size_true: np.ndarray,
    size_reg_true: np.ndarray,
    depth_true: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, object]:
    mask = mask.astype(bool)
    count = int(np.sum(mask))
    if count <= 0:
        return {
            "count": 0,
            "size_top1": float("nan"),
            "size_top2": float("nan"),
            "size_mae": float("nan"),
            "size_median_ae": float("nan"),
            "depth_accuracy": float("nan"),
            "depth_balanced_accuracy": float("nan"),
            "depth_confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        }

    sub_size_probs = size_probs[mask]
    sub_size_reg_pred = size_reg_pred[mask]
    sub_depth_probs = depth_probs[mask]
    sub_size_true = size_true[mask]
    sub_size_reg_true = size_reg_true[mask]
    sub_depth_true = depth_true[mask]
    depth_pred = np.argmax(sub_depth_probs, axis=1).astype(np.int32)
    depth_cm = confusion_matrix_counts(sub_depth_true, depth_pred, len(COARSE_DEPTH_ORDER))
    reg = regression_metrics(sub_size_reg_pred, sub_size_reg_true)
    return {
        "count": count,
        "size_top1": topk_accuracy(sub_size_probs, sub_size_true, 1),
        "size_top2": topk_accuracy(sub_size_probs, sub_size_true, 2),
        "size_mae": reg["mae"],
        "size_median_ae": reg["median_ae"],
        "depth_accuracy": float(np.mean(depth_pred == sub_depth_true)),
        "depth_balanced_accuracy": balanced_accuracy_from_cm(depth_cm),
        "depth_confusion_matrix": depth_cm.tolist(),
    }


def inverse_frequency_sample_weight(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.int32)
    classes, counts = np.unique(y, return_counts=True)
    weights = {int(cls): float(len(y) / max(len(classes) * cnt, 1)) for cls, cnt in zip(classes, counts)}
    return np.asarray([weights[int(v)] for v in y], dtype=np.float32)


def summarize_series(out: Dict[str, float], prefix: str, values: List[float]) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        out[f"{prefix}_mean"] = 0.0
        out[f"{prefix}_std"] = 0.0
        out[f"{prefix}_min"] = 0.0
        out[f"{prefix}_max"] = 0.0
        out[f"{prefix}_first"] = 0.0
        out[f"{prefix}_center"] = 0.0
        out[f"{prefix}_last"] = 0.0
        out[f"{prefix}_delta"] = 0.0
        out[f"{prefix}_slope"] = 0.0
        return
    x = np.arange(arr.size, dtype=np.float32)
    x_mean = float(x.mean())
    y_mean = float(arr.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (arr - y_mean)) / denom) if denom > 1e-8 else 0.0
    center_idx = int(arr.size // 2)
    out[f"{prefix}_mean"] = float(arr.mean())
    out[f"{prefix}_std"] = float(arr.std())
    out[f"{prefix}_min"] = float(arr.min())
    out[f"{prefix}_max"] = float(arr.max())
    out[f"{prefix}_first"] = float(arr[0])
    out[f"{prefix}_center"] = float(arr[center_idx])
    out[f"{prefix}_last"] = float(arr[-1])
    out[f"{prefix}_delta"] = float(arr[-1] - arr[0])
    out[f"{prefix}_slope"] = slope


def build_records_and_samples_for_file(
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
        raw_frames = raw_96.reshape(n_frames, 12, 8).astype(np.float32)
        norm_frames = normalize_frames(raw_96)
        frame_rows = [frame_physics_features(raw_frames[idx]) for idx in range(n_frames)]
        group_key = f"{base_group}|{target_file}"
        size_cm = float(parse_float_from_cm_text(size_text))
        depth_cm = float(parse_float_from_cm_text(depth_text))

        records_by_key[group_key] = {
            "raw_frames": raw_frames,
            "norm_frames": norm_frames,
            "frame_rows": frame_rows,
            "seq_len": int(seq_len),
            "base_group": base_group,
            "file_name": target_file,
            "size_text": size_text,
            "depth_text": depth_text,
            "size_cm": size_cm,
            "depth_cm": depth_cm,
            "segments": segments,
        }

        for end_row in range(seq_len - 1, n_frames, stride):
            start_row = end_row - seq_len + 1
            raw_window_96 = raw_96[start_row : end_row + 1]
            if np.isnan(raw_window_96).any() or np.all(raw_window_96 == 0):
                continue
            center_row = start_row + (seq_len // 2)
            label = int(is_center_positive(center_row, segments))
            sample_records.append(
                {
                    "group_key": group_key,
                    "base_group": base_group,
                    "file_name": target_file,
                    "end_row": int(end_row),
                    "center_row": int(center_row),
                    "label": label,
                    "size_text": size_text,
                    "depth_text": depth_text,
                    "size_cm": size_cm,
                    "depth_cm": depth_cm,
                    "size_class_index": int(size_to_class_index(size_cm)),
                    "depth_coarse_index": int(depth_to_coarse_index(depth_cm)),
                }
            )

    sample_records = compress_samples_by_gap(sample_records, min_gap=dedup_gap)
    return records_by_key, sample_records


def window_feature_row(records_by_key: Dict[str, dict], sample: dict) -> Dict[str, float]:
    rec = records_by_key[sample["group_key"]]
    end_row = int(sample["end_row"])
    seq_len = int(rec["seq_len"])
    start_row = end_row - seq_len + 1
    raw_window = rec["raw_frames"][start_row : end_row + 1]
    norm_window = rec["norm_frames"][start_row : end_row + 1]
    frame_rows = rec["frame_rows"][start_row : end_row + 1]

    out: Dict[str, float] = {
        "label": int(sample["label"]),
        "size_cm": float(sample["size_cm"]),
        "depth_cm": float(sample["depth_cm"]),
        "size_class_index": int(sample["size_class_index"]),
        "depth_coarse_index": int(sample["depth_coarse_index"]),
        "center_row": int(sample["center_row"]),
        "end_row": int(sample["end_row"]),
    }

    frame_keys = list(frame_rows[0].keys()) if frame_rows else []
    for key in frame_keys:
        summarize_series(out, key, [float(row[key]) for row in frame_rows])

    for key, value in window_temporal_features(frame_rows).items():
        out[f"window_{key}"] = float(value)

    mean_frame_raw = raw_window.mean(axis=0)
    max_frame_raw = raw_window.max(axis=0)
    center_frame_raw = raw_window[len(raw_window) // 2]
    mean_frame_norm = norm_window.mean(axis=0)

    for prefix, frame in {
        "meanframe_raw": mean_frame_raw,
        "maxframe_raw": max_frame_raw,
        "centerframe_raw": center_frame_raw,
        "meanframe_norm": mean_frame_norm,
    }.items():
        for key, value in frame_physics_features(frame).items():
            out[f"{prefix}_{key}"] = float(value)

    if len(raw_window) > 1:
        delta = np.diff(raw_window, axis=0)
        abs_delta = np.abs(delta)
        out["delta_abs_mean"] = float(abs_delta.mean())
        out["delta_abs_std"] = float(abs_delta.std())
        out["delta_abs_max"] = float(abs_delta.max())
        delta_mean_frame = abs_delta.mean(axis=0)
        for key, value in frame_physics_features(delta_mean_frame).items():
            out[f"deltaframe_{key}"] = float(value)
    else:
        out["delta_abs_mean"] = 0.0
        out["delta_abs_std"] = 0.0
        out["delta_abs_max"] = 0.0
        for key, value in frame_physics_features(np.zeros((12, 8), dtype=np.float32)).items():
            out[f"deltaframe_{key}"] = float(value)

    out["window_raw_global_mean"] = float(raw_window.mean())
    out["window_raw_global_std"] = float(raw_window.std())
    out["window_raw_global_max"] = float(raw_window.max())
    out["window_raw_global_p95"] = float(np.percentile(raw_window.reshape(-1), 95))
    out["window_norm_global_mean"] = float(norm_window.mean())
    out["window_norm_global_std"] = float(norm_window.std())
    return out


def build_feature_table(records_by_key: Dict[str, dict], sample_records: List[dict], split_name: str) -> pd.DataFrame:
    rows: List[dict] = []
    for sample in sample_records:
        row = window_feature_row(records_by_key, sample)
        row["split"] = split_name
        row["group_key"] = sample["group_key"]
        row["base_group"] = sample["base_group"]
        row["file_name"] = sample["file_name"]
        row["size_text"] = sample["size_text"]
        row["depth_text"] = sample["depth_text"]
        rows.append(row)
    return pd.DataFrame(rows)


def detection_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    out = cls_metrics_at_threshold(y_true, y_score, threshold)
    out["auc"] = safe_auc(y_true, y_score)
    out["ap"] = safe_ap(y_true, y_score)
    return out


def plot_top_feature_importance(model, feature_names: List[str], output_path: str, title: str, top_n: int = 20) -> None:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return
    importance = np.asarray(importance, dtype=np.float64)
    if importance.size != len(feature_names):
        return
    order = np.argsort(importance)[::-1][:top_n]
    top_importance = importance[order][::-1]
    top_names = [feature_names[idx] for idx in order][::-1]
    plt.figure(figsize=(11, 8))
    plt.barh(np.arange(len(top_names)), top_importance, color="#2b6cb0")
    plt.yticks(np.arange(len(top_names)), top_names, fontsize=9)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_evals_result_plot(model, metric_key: str, output_path: str, title: str) -> None:
    evals = getattr(model, "evals_result_", None)
    if not evals:
        return
    if "validation_0" not in evals or metric_key not in evals["validation_0"]:
        return
    series0 = evals["validation_0"][metric_key]
    series1 = evals.get("validation_1", {}).get(metric_key, None)
    plt.figure(figsize=(8, 5))
    plt.plot(series0, label=f"train_{metric_key}")
    if series1 is not None:
        plt.plot(series1, label=f"val_{metric_key}")
    plt.title(title)
    plt.xlabel("Boosting Round")
    plt.ylabel(metric_key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def dummy_probabilities(model: DummyClassifier, X: np.ndarray, positive_class: int = 1) -> np.ndarray:
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if positive_class not in classes:
        return np.zeros((len(X),), dtype=np.float64)
    return probs[:, classes.index(positive_class)]


@dataclass
class Config:
    seed: int = int(os.environ.get("PAPERXGB_SEED", "2026"))
    seq_len: int = int(os.environ.get("PAPERXGB_SEQ_LEN", str(INPUT_SEQ_LEN)))
    stride: int = int(os.environ.get("PAPERXGB_STRIDE", str(WINDOW_STRIDE)))
    dedup_gap: int = int(os.environ.get("PAPERXGB_DEDUP_GAP", "6"))
    max_neg_pos_ratio: float = float(os.environ.get("PAPERXGB_MAX_NEG_POS_RATIO", "2.5"))
    n_estimators: int = int(os.environ.get("PAPERXGB_N_ESTIMATORS", "700"))
    learning_rate: float = float(os.environ.get("PAPERXGB_LR", "0.03"))
    max_depth: int = int(os.environ.get("PAPERXGB_MAX_DEPTH", "4"))
    subsample: float = float(os.environ.get("PAPERXGB_SUBSAMPLE", "0.85"))
    colsample_bytree: float = float(os.environ.get("PAPERXGB_COLSAMPLE", "0.85"))
    early_stopping_rounds: int = int(os.environ.get("PAPERXGB_EARLY_STOP", "35"))
    n_jobs: int = int(os.environ.get("PAPERXGB_N_JOBS", "8"))

    def __post_init__(self):
        if int(self.seq_len) != int(INPUT_SEQ_LEN):
            raise ValueError(f"Locked protocol requires seq_len={INPUT_SEQ_LEN}, got {self.seq_len}.")
        if int(self.stride) != int(WINDOW_STRIDE):
            raise ValueError(f"Locked protocol requires stride={WINDOW_STRIDE}, got {self.stride}.")
        self.data_root = os.environ.get(
            "PAPERXGB_DATA_ROOT",
            os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"),
        )
        self.file1_labels = os.environ.get(
            "PAPERXGB_FILE1_LABELS",
            os.path.join(REPO_ROOT, "manual_keyframe_labels.json"),
        )
        self.file2_labels = os.environ.get(
            "PAPERXGB_FILE2_LABELS",
            os.path.join(self.data_root, "manual_keyframe_labels_file2.json"),
        )
        self.file3_labels = os.environ.get(
            "PAPERXGB_FILE3_LABELS",
            os.path.join(self.data_root, "manual_keyframe_labels_file3.json"),
        )
        self.output_dir = os.environ.get(
            "PAPERXGB_OUTPUT_DIR",
            os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_baselines_v1"),
        )


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    rec1, samples1 = build_records_and_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, samples2 = build_records_and_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3 = build_records_and_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        )
    )
    common_base_group_set = set(common_base_groups)
    train_base_groups, val_base_groups = split_base_groups_train_val_balanced(common_base_groups)
    train_base_group_set = set(train_base_groups)
    val_base_group_set = set(val_base_groups)

    train_records: Dict[str, dict] = {}
    train_records.update({k: v for k, v in rec1.items() if v["base_group"] in train_base_group_set})
    train_records.update({k: v for k, v in rec2.items() if v["base_group"] in train_base_group_set})

    val_records: Dict[str, dict] = {}
    val_records.update({k: v for k, v in rec1.items() if v["base_group"] in val_base_group_set})
    val_records.update({k: v for k, v in rec2.items() if v["base_group"] in val_base_group_set})

    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_base_group_set}

    train_samples_all = [s for s in (samples1 + samples2) if s["base_group"] in train_base_group_set]
    train_samples_det = downsample_negatives(train_samples_all, cfg.max_neg_pos_ratio, cfg.seed)
    val_samples = [s for s in (samples1 + samples2) if s["base_group"] in val_base_group_set]
    test_samples = [s for s in samples3 if s["base_group"] in common_base_group_set]

    feature_train_det = build_feature_table(train_records, train_samples_det, "train_det")
    feature_train_all = build_feature_table(train_records, train_samples_all, "train_all")
    feature_val = build_feature_table(val_records, val_samples, "val")
    feature_test = build_feature_table(test_records, test_samples, "test")

    feature_table = pd.concat([feature_train_det, feature_train_all, feature_val, feature_test], ignore_index=True)
    feature_table.to_csv(os.path.join(cfg.output_dir, "xgboost_window_feature_table.csv"), index=False, encoding="utf-8-sig")

    metadata_cols = {
        "split",
        "group_key",
        "base_group",
        "file_name",
        "size_text",
        "depth_text",
        "label",
        "size_cm",
        "depth_cm",
        "size_class_index",
        "depth_coarse_index",
        "center_row",
        "end_row",
    }
    feature_cols = [c for c in feature_train_det.columns if c not in metadata_cols]

    X_train_det = feature_train_det[feature_cols].to_numpy(dtype=np.float32)
    y_train_det = feature_train_det["label"].to_numpy(dtype=np.int32)
    X_val = feature_val[feature_cols].to_numpy(dtype=np.float32)
    y_val_det = feature_val["label"].to_numpy(dtype=np.int32)
    X_test = feature_test[feature_cols].to_numpy(dtype=np.float32)
    y_test_det = feature_test["label"].to_numpy(dtype=np.int32)

    det_model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr"],
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    det_model.fit(
        X_train_det,
        y_train_det,
        eval_set=[(X_train_det, y_train_det), (X_val, y_val_det)],
        verbose=False,
    )
    y_val_det_score = det_model.predict_proba(X_val)[:, 1]
    y_test_det_score = det_model.predict_proba(X_test)[:, 1]
    det_val_best = select_best_f1_threshold(y_val_det, y_val_det_score)
    det_threshold = float(det_val_best["threshold"])
    det_val_metrics = detection_metrics(y_val_det, y_val_det_score, det_threshold)
    det_test_metrics = detection_metrics(y_test_det, y_test_det_score, det_threshold)

    dummy_det = DummyClassifier(strategy="prior")
    dummy_det.fit(X_train_det, y_train_det)
    dummy_val_score = dummy_probabilities(dummy_det, X_val)
    dummy_test_score = dummy_probabilities(dummy_det, X_test)
    dummy_det_threshold = 0.5
    dummy_det_val_metrics = detection_metrics(y_val_det, dummy_val_score, dummy_det_threshold)
    dummy_det_test_metrics = detection_metrics(y_test_det, dummy_test_score, dummy_det_threshold)

    train_pos = feature_train_all[feature_train_all["label"] == 1].reset_index(drop=True)
    val_pos = feature_val[feature_val["label"] == 1].reset_index(drop=True)
    test_pos = feature_test[feature_test["label"] == 1].reset_index(drop=True)

    X_train_pos = train_pos[feature_cols].to_numpy(dtype=np.float32)
    X_val_pos = val_pos[feature_cols].to_numpy(dtype=np.float32)
    X_test_pos = test_pos[feature_cols].to_numpy(dtype=np.float32)

    y_train_size_cls = train_pos["size_class_index"].to_numpy(dtype=np.int32)
    y_val_size_cls = val_pos["size_class_index"].to_numpy(dtype=np.int32)
    y_test_size_cls = test_pos["size_class_index"].to_numpy(dtype=np.int32)

    y_train_size_reg = train_pos["size_cm"].to_numpy(dtype=np.float32)
    y_val_size_reg = val_pos["size_cm"].to_numpy(dtype=np.float32)
    y_test_size_reg = test_pos["size_cm"].to_numpy(dtype=np.float32)

    y_train_depth = train_pos["depth_coarse_index"].to_numpy(dtype=np.int32)
    y_val_depth = val_pos["depth_coarse_index"].to_numpy(dtype=np.int32)
    y_test_depth = test_pos["depth_coarse_index"].to_numpy(dtype=np.int32)

    size_cls_model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    size_cls_model.fit(
        X_train_pos,
        y_train_size_cls,
        sample_weight=inverse_frequency_sample_weight(y_train_size_cls),
        eval_set=[(X_train_pos, y_train_size_cls), (X_val_pos, y_val_size_cls)],
        verbose=False,
    )

    size_reg_model = XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    size_reg_model.fit(
        X_train_pos,
        y_train_size_reg,
        eval_set=[(X_train_pos, y_train_size_reg), (X_val_pos, y_val_size_reg)],
        verbose=False,
    )

    depth_model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=cfg.n_jobs,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    depth_model.fit(
        X_train_pos,
        y_train_depth,
        sample_weight=inverse_frequency_sample_weight(y_train_depth),
        eval_set=[(X_train_pos, y_train_depth), (X_val_pos, y_val_depth)],
        verbose=False,
    )

    val_size_probs = size_cls_model.predict_proba(X_val_pos)
    test_size_probs = size_cls_model.predict_proba(X_test_pos)
    val_size_reg_pred = size_reg_model.predict(X_val_pos)
    test_size_reg_pred = size_reg_model.predict(X_test_pos)
    val_depth_probs = depth_model.predict_proba(X_val_pos)
    test_depth_probs = depth_model.predict_proba(X_test_pos)

    gt_val_metrics = subset_multitask_metrics(
        val_size_probs,
        val_size_reg_pred,
        val_depth_probs,
        y_val_size_cls,
        y_val_size_reg,
        y_val_depth,
        np.ones(len(y_val_size_cls), dtype=bool),
    )
    gt_test_metrics = subset_multitask_metrics(
        test_size_probs,
        test_size_reg_pred,
        test_depth_probs,
        y_test_size_cls,
        y_test_size_reg,
        y_test_depth,
        np.ones(len(y_test_size_cls), dtype=bool),
    )

    dummy_size_cls = DummyClassifier(strategy="prior")
    dummy_size_cls.fit(X_train_pos, y_train_size_cls)
    dummy_size_reg = DummyRegressor(strategy="mean")
    dummy_size_reg.fit(X_train_pos, y_train_size_reg)
    dummy_depth = DummyClassifier(strategy="prior")
    dummy_depth.fit(X_train_pos, y_train_depth)

    dummy_test_size_probs = dummy_size_cls.predict_proba(X_test_pos)
    dummy_test_size_reg = dummy_size_reg.predict(X_test_pos)
    dummy_test_depth_probs = dummy_depth.predict_proba(X_test_pos)
    dummy_gt_test_metrics = subset_multitask_metrics(
        dummy_test_size_probs,
        dummy_test_size_reg,
        dummy_test_depth_probs,
        y_test_size_cls,
        y_test_size_reg,
        y_test_depth,
        np.ones(len(y_test_size_cls), dtype=bool),
    )

    all_size_probs_test = size_cls_model.predict_proba(X_test)
    all_size_reg_pred_test = size_reg_model.predict(X_test)
    all_depth_probs_test = depth_model.predict_proba(X_test)
    gated_test_mask = y_test_det_score >= det_threshold
    gated_test_metrics = subset_multitask_metrics(
        all_size_probs_test,
        all_size_reg_pred_test,
        all_depth_probs_test,
        feature_test["size_class_index"].to_numpy(dtype=np.int32),
        feature_test["size_cm"].to_numpy(dtype=np.float32),
        feature_test["depth_coarse_index"].to_numpy(dtype=np.int32),
        gated_test_mask,
    )

    feature_importance_dir = os.path.join(cfg.output_dir, "feature_importance")
    os.makedirs(feature_importance_dir, exist_ok=True)
    plot_top_feature_importance(
        det_model,
        feature_cols,
        os.path.join(feature_importance_dir, "detection_top_features.png"),
        "XGBoost Detection Top Features",
    )
    plot_top_feature_importance(
        size_cls_model,
        feature_cols,
        os.path.join(feature_importance_dir, "size_classification_top_features.png"),
        "XGBoost Size Classification Top Features",
    )
    plot_top_feature_importance(
        depth_model,
        feature_cols,
        os.path.join(feature_importance_dir, "depth_coarse_top_features.png"),
        "XGBoost Coarse Depth Top Features",
    )
    save_evals_result_plot(
        det_model,
        "logloss",
        os.path.join(cfg.output_dir, "detection_logloss_curve.png"),
        "XGBoost Detection Logloss",
    )
    save_evals_result_plot(
        det_model,
        "aucpr",
        os.path.join(cfg.output_dir, "detection_aucpr_curve.png"),
        "XGBoost Detection AUCPR",
    )
    save_evals_result_plot(
        size_cls_model,
        "mlogloss",
        os.path.join(cfg.output_dir, "size_classification_mlogloss_curve.png"),
        "XGBoost Size Classification mlogloss",
    )
    save_evals_result_plot(
        size_reg_model,
        "rmse",
        os.path.join(cfg.output_dir, "size_regression_rmse_curve.png"),
        "XGBoost Size Regression RMSE",
    )
    save_evals_result_plot(
        depth_model,
        "mlogloss",
        os.path.join(cfg.output_dir, "depth_classification_mlogloss_curve.png"),
        "XGBoost Coarse Depth mlogloss",
    )

    condition_rows = []
    for group in common_base_groups:
        size_text, depth_text = parse_size_depth_from_group(group)
        condition_rows.append(
            {
                "group": group,
                "size_text": size_text,
                "depth_text": depth_text,
                "size_cm": float(parse_float_from_cm_text(size_text)),
                "depth_cm": float(parse_float_from_cm_text(depth_text)),
                "train_det_samples": int(np.sum(feature_train_det["base_group"] == group)),
                "train_all_samples": int(np.sum(feature_train_all["base_group"] == group)),
                "val_samples": int(np.sum(feature_val["base_group"] == group)),
                "test_samples": int(np.sum(feature_test["base_group"] == group)),
                "train_positive": int(np.sum((feature_train_all["base_group"] == group) & (feature_train_all["label"] == 1))),
                "val_positive": int(np.sum((feature_val["base_group"] == group) & (feature_val["label"] == 1))),
                "test_positive": int(np.sum((feature_test["base_group"] == group) & (feature_test["label"] == 1))),
            }
        )
    pd.DataFrame(condition_rows).to_csv(
        os.path.join(cfg.output_dir, "xgboost_condition_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "protocol_v1": protocol_summary(),
        "config": {
            "seed": cfg.seed,
            "seq_len": cfg.seq_len,
            "stride": cfg.stride,
            "dedup_gap": cfg.dedup_gap,
            "max_neg_pos_ratio": cfg.max_neg_pos_ratio,
            "n_estimators": cfg.n_estimators,
            "learning_rate": cfg.learning_rate,
            "max_depth": cfg.max_depth,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "early_stopping_rounds": cfg.early_stopping_rounds,
        },
        "splits": {
            "common_base_groups": len(common_base_groups),
            "train_base_groups": len(train_base_groups),
            "val_base_groups": len(val_base_groups),
            "test_base_groups": len(common_base_groups),
            "train_det_samples": len(feature_train_det),
            "train_all_samples": len(feature_train_all),
            "train_positive_samples": int(np.sum(feature_train_all["label"] == 1)),
            "val_samples": len(feature_val),
            "val_positive_samples": int(np.sum(feature_val["label"] == 1)),
            "test_samples": len(feature_test),
            "test_positive_samples": int(np.sum(feature_test["label"] == 1)),
        },
        "detection_xgboost": {
            "best_iteration": int(getattr(det_model, "best_iteration", -1)),
            "threshold": det_threshold,
            "val_metrics": det_val_metrics,
            "test_metrics": det_test_metrics,
        },
        "detection_dummy_prior": {
            "threshold": dummy_det_threshold,
            "val_metrics": dummy_det_val_metrics,
            "test_metrics": dummy_det_test_metrics,
        },
        "size_depth_xgboost": {
            "size_cls_best_iteration": int(getattr(size_cls_model, "best_iteration", -1)),
            "size_reg_best_iteration": int(getattr(size_reg_model, "best_iteration", -1)),
            "depth_best_iteration": int(getattr(depth_model, "best_iteration", -1)),
            "gt_positive_val_metrics": gt_val_metrics,
            "gt_positive_test_metrics": gt_test_metrics,
            "gated_test_metrics_at_det_threshold": gated_test_metrics,
        },
        "size_depth_dummy": {
            "gt_positive_test_metrics": dummy_gt_test_metrics,
        },
        "artifacts": {
            "feature_table_csv": "xgboost_window_feature_table.csv",
            "condition_manifest_csv": "xgboost_condition_manifest.csv",
            "feature_importance_dir": "feature_importance",
        },
    }

    with open(os.path.join(cfg.output_dir, "xgboost_baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# XGBoost Baseline Report",
        "",
        "## Protocol",
        f"- Input window: `{cfg.seq_len} x 1 x 12 x 8`",
        f"- Stride: `{cfg.stride}`",
        "- Split: `1.CSV + 2.CSV` development, `3.CSV` final test",
        "- Detection label: center frame inside positive segment",
        "",
        "## Detection",
        f"- XGBoost test AUC: `{det_test_metrics['auc']:.4f}`",
        f"- XGBoost test AP: `{det_test_metrics['ap']:.4f}`",
        f"- XGBoost test F1@val threshold: `{det_test_metrics['f1']:.4f}`",
        f"- Dummy prior test AUC: `{dummy_det_test_metrics['auc']:.4f}`",
        f"- Dummy prior test AP: `{dummy_det_test_metrics['ap']:.4f}`",
        "",
        "## Size / Depth (GT-positive test windows)",
        f"- XGBoost size top1: `{gt_test_metrics['size_top1']:.4f}`",
        f"- XGBoost size top2: `{gt_test_metrics['size_top2']:.4f}`",
        f"- XGBoost size MAE: `{gt_test_metrics['size_mae']:.4f} cm`",
        f"- XGBoost depth coarse acc: `{gt_test_metrics['depth_accuracy']:.4f}`",
        f"- Dummy size top1: `{dummy_gt_test_metrics['size_top1']:.4f}`",
        f"- Dummy size MAE: `{dummy_gt_test_metrics['size_mae']:.4f} cm`",
        f"- Dummy depth acc: `{dummy_gt_test_metrics['depth_accuracy']:.4f}`",
        "",
        "## Notes",
        "- This baseline uses only handcrafted window features plus XGBoost.",
        "- It is intended to anchor the later neural-network comparison.",
    ]
    with open(os.path.join(cfg.output_dir, "XGBOOST_BASELINE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
