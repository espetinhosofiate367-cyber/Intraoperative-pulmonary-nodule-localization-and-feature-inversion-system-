import json
import os
import sys
from dataclasses import dataclass
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

from dual_stream_mstcn_multitask import DualStreamMSTCNMultiTask
from raw_hybrid_positive_size_model import RawHybridPositiveSizeModel
from raw_positive_size_model import RawPositiveSizeModel
from raw_positive_size_model_v2 import RawPositiveSizeModelV2
from raw_size_routed_depth_model import RawSizeRoutedDepthModel
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM, WINDOW_STRIDE, protocol_summary
from train_stage2_raw_hybrid_positive_size import size_norm_to_cm
from train_stage3_raw_size_conditioned_depth import (
    PositiveDepthDataset,
    balanced_accuracy_from_cm,
    build_positive_depth_samples_for_file,
    compute_concept_stats,
    compute_raw_scale,
    confusion_matrix_counts,
)
from train_xgboost_baselines import build_feature_table, build_records_and_samples_for_file
from triplet_repeat_classifier.train_triplet_repeat_classifier import load_json


def plot_confusion(cm: np.ndarray, title: str, output_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(COARSE_DEPTH_ORDER)))
    ax.set_yticks(np.arange(len(COARSE_DEPTH_ORDER)))
    ax.set_xticklabels(COARSE_DEPTH_ORDER)
    ax.set_yticklabels(COARSE_DEPTH_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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


def gaussian_size_weights(pred_size_cm: torch.Tensor, sigma_cm: float) -> torch.Tensor:
    size_values = torch.tensor(SIZE_VALUES_CM, dtype=pred_size_cm.dtype, device=pred_size_cm.device).view(1, -1)
    dist2 = (pred_size_cm.view(-1, 1) - size_values) ** 2
    weights = torch.exp(-0.5 * dist2 / max(float(sigma_cm) ** 2, 1e-8))
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
    return weights


@dataclass
class Config:
    stage2_ckpt: str = os.environ.get(
        "PAPERD3PRED_STAGE2_CKPT",
        os.path.join(
            PROJECT_DIR,
            "experiments",
            "outputs_stage2_dualstream_mstcn_multitask_raw",
            "paper_stage2_dualstream_mstcn_best.pth",
        ),
    )
    stage3_ckpt: str = os.environ.get(
        "PAPERD3PRED_STAGE3_CKPT",
        os.path.join(
            PROJECT_DIR,
            "experiments",
            "outputs_stage3_raw_size_routed_depth_run1",
            "paper_stage3_raw_size_routed_depth_best.pth",
        ),
    )
    output_dir: str = os.environ.get(
        "PAPERD3PRED_OUT",
        os.path.join(PROJECT_DIR, "experiments", "outputs_stage3_predicted_size_routing_v1"),
    )
    data_root: str = os.environ.get("PAPERD3PRED_DATA_ROOT", os.path.join(REPO_ROOT, "整理好的数据集", "建表数据"))
    file1_labels: str = os.environ.get("PAPERD3PRED_FILE1_LABELS", os.path.join(REPO_ROOT, "manual_keyframe_labels.json"))
    file2_labels: str = os.environ.get("PAPERD3PRED_FILE2_LABELS", os.path.join(data_root, "manual_keyframe_labels_file2.json"))
    file3_labels: str = os.environ.get("PAPERD3PRED_FILE3_LABELS", os.path.join(data_root, "manual_keyframe_labels_file3.json"))
    seq_len: int = INPUT_SEQ_LEN
    stride: int = WINDOW_STRIDE
    dedup_gap: int = 6
    batch_size: int = 128


def main():
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    file1_all = load_json(cfg.file1_labels)
    file2_all = load_json(cfg.file2_labels)
    file3_all = load_json(cfg.file3_labels)

    rec1, _samples1 = build_positive_depth_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, _samples2 = build_positive_depth_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3 = build_positive_depth_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)

    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        )
    )
    common_set = set(common_base_groups)
    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_set}
    test_samples = [s for s in samples3 if s["base_group"] in common_set]

    raw_scale = compute_raw_scale(test_records)
    concept_mean, concept_std = compute_concept_stats(test_samples)
    ds_test = PositiveDepthDataset(test_records, test_samples, raw_scale, concept_mean, concept_std, is_train=False)
    loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage2_ckpt = torch.load(cfg.stage2_ckpt, map_location="cpu", weights_only=True)
    stage2_router_name = str(stage2_ckpt.get("router_model_name", ""))
    feature_matrix_test = None
    if stage2_router_name == "RawPositiveSizeModel":
        stage2_model = RawPositiveSizeModel(
            seq_len=INPUT_SEQ_LEN,
            frame_feature_dim=24,
            temporal_channels=48,
            temporal_blocks=3,
            dropout=0.25,
            num_size_classes=len(SIZE_VALUES_CM),
        ).to(device)
        stage2_model.load_state_dict(stage2_ckpt["model_state_dict"], strict=True)
        stage2_mode = "raw_positive_size"
    elif stage2_router_name == "RawPositiveSizeModelV2":
        model_cfg = stage2_ckpt.get("model_config", {})
        stage2_model = RawPositiveSizeModelV2(
            seq_len=int(model_cfg.get("seq_len", INPUT_SEQ_LEN)),
            frame_feature_dim=int(model_cfg.get("frame_feature_dim", 32)),
            temporal_channels=int(model_cfg.get("temporal_channels", 64)),
            temporal_blocks=int(model_cfg.get("temporal_blocks", 4)),
            dropout=float(model_cfg.get("dropout", 0.22)),
            num_size_classes=int(model_cfg.get("num_size_classes", len(SIZE_VALUES_CM))),
            residual_scale=float(model_cfg.get("residual_scale", 0.35)),
        ).to(device)
        stage2_model.load_state_dict(stage2_ckpt["model_state_dict"], strict=True)
        stage2_mode = "raw_positive_size_v2"
    elif stage2_router_name == "RawHybridPositiveSizeModel":
        model_cfg = stage2_ckpt.get("model_config", {})
        stage2_model = RawHybridPositiveSizeModel(
            seq_len=int(model_cfg.get("seq_len", INPUT_SEQ_LEN)),
            frame_feature_dim=int(model_cfg.get("frame_feature_dim", 24)),
            temporal_channels=int(model_cfg.get("temporal_channels", 48)),
            temporal_blocks=int(model_cfg.get("temporal_blocks", 3)),
            dropout=float(model_cfg.get("dropout", 0.28)),
            num_size_classes=int(model_cfg.get("num_size_classes", len(SIZE_VALUES_CM))),
            num_tabular_features=int(model_cfg.get("num_tabular_features", len(stage2_ckpt.get("selected_features", [])))),
            tabular_hidden_dim=int(model_cfg.get("tabular_hidden_dim", 64)),
        ).to(device)
        stage2_model.load_state_dict(stage2_ckpt["model_state_dict"], strict=True)
        selected_features = [str(x) for x in stage2_ckpt.get("selected_features", [])]
        feature_mean = np.asarray(stage2_ckpt.get("feature_mean", []), dtype=np.float32)
        feature_std = np.asarray(stage2_ckpt.get("feature_std", []), dtype=np.float32)
        if len(selected_features) == 0:
            raise RuntimeError("Hybrid size router checkpoint is missing selected_features.")
        rec3_xgb, _samples3_xgb = build_records_and_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
        enriched_records = {}
        for key, rec in test_records.items():
            merged = dict(rec)
            if key not in rec3_xgb:
                raise KeyError(f"Missing XGBoost-style record for key: {key}")
            merged["frame_rows"] = rec3_xgb[key]["frame_rows"]
            merged["size_text"] = rec3_xgb[key]["size_text"]
            merged["depth_text"] = rec3_xgb[key]["depth_text"]
            merged["size_cm"] = rec3_xgb[key]["size_cm"]
            merged["depth_cm"] = rec3_xgb[key]["depth_cm"]
            enriched_records[key] = merged
        feature_samples = []
        for s in test_samples:
            rec = enriched_records[s["group_key"]]
            row = dict(s)
            row["size_cm"] = float(rec["size_cm"])
            row["depth_cm"] = float(rec["depth_cm"])
            row["size_text"] = str(rec["size_text"])
            row["depth_text"] = str(rec["depth_text"])
            feature_samples.append(row)
        feature_df = build_feature_table(enriched_records, feature_samples, "test")
        feature_matrix_test = feature_df[selected_features].to_numpy(dtype=np.float32)
        feature_matrix_test = (feature_matrix_test - feature_mean.reshape(1, -1)) / np.maximum(feature_std.reshape(1, -1), 1e-6)
        stage2_mode = "raw_hybrid_positive_size"
    else:
        stage2_cfg = stage2_ckpt.get("config", {})
        stage2_model = DualStreamMSTCNMultiTask(
            seq_len=int(stage2_cfg.get("seq_len", INPUT_SEQ_LEN)),
            frame_feature_dim=32,
            temporal_channels=64,
            temporal_blocks=3,
            dropout=float(stage2_cfg.get("dropout", 0.35)),
            use_delta_branch=bool(stage2_cfg.get("use_delta_branch", False)),
            num_size_classes=len(SIZE_VALUES_CM),
            num_depth_classes=len(COARSE_DEPTH_ORDER),
        ).to(device)
        stage2_model.load_state_dict(stage2_ckpt["model_state_dict"], strict=True)
        stage2_mode = "stage2_multitask"
    stage2_model.eval()

    stage3_ckpt = torch.load(cfg.stage3_ckpt, map_location="cpu", weights_only=True)
    stage3_model = RawSizeRoutedDepthModel(
        seq_len=INPUT_SEQ_LEN,
        frame_feature_dim=24,
        temporal_channels=48,
        temporal_blocks=3,
        dropout=0.25,
        num_size_classes=len(SIZE_VALUES_CM),
        num_depth_classes=len(COARSE_DEPTH_ORDER),
    ).to(device)
    stage3_model.load_state_dict(stage3_ckpt["model_state_dict"], strict=True)
    stage3_model.eval()

    y_true = []
    gt_size = []
    pred_size = []
    size_prob_rows = []
    pred_gt_route = []
    pred_hard_route = []
    pred_soft_route = []
    pred_reg_sigma015_route = []
    pred_reg_sigma020_route = []
    pred_reg_sigma025_route = []
    pred_top2_soft_route = []
    pred_temp05_soft_route = []
    pred_temp07_soft_route = []
    pred_temp03_soft_route = []
    cursor = 0

    with torch.no_grad():
        for raw_x, norm_x, size_idx, depth_idx, _concept_target, _sample_weight in loader:
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            size_idx = size_idx.to(device)
            depth_idx = depth_idx.to(device)

            if stage2_mode == "raw_positive_size":
                size_logits, size_reg_pred = stage2_model(raw_x, norm_x)[:2]
                size_reg_cm = size_reg_pred.squeeze(1)
            elif stage2_mode == "raw_positive_size_v2":
                size_logits, _size_ord_logits, size_reg_pred, _size_probs_model = stage2_model(raw_x, norm_x)
                size_reg_cm = torch.tensor(
                    size_norm_to_cm(size_reg_pred.squeeze(1).cpu().numpy()),
                    dtype=raw_x.dtype,
                    device=device,
                )
            elif stage2_mode == "raw_hybrid_positive_size":
                batch_size = int(raw_x.shape[0])
                feat_x = torch.from_numpy(feature_matrix_test[cursor : cursor + batch_size]).to(device)
                cursor += batch_size
                size_logits, _size_ord_logits, size_reg_pred, _size_probs_model = stage2_model(raw_x, norm_x, feat_x)
                size_reg_cm = torch.tensor(size_norm_to_cm(size_reg_pred.squeeze(1).cpu().numpy()), dtype=raw_x.dtype, device=device)
            else:
                _det_logits, size_logits, size_reg_pred, _depth_logits = stage2_model(norm_x)
                size_reg_cm = size_reg_pred.squeeze(1)
            size_probs = torch.softmax(size_logits, dim=1)
            size_pred_idx = torch.argmax(size_probs, dim=1)

            gt_logits, _ = stage3_model(raw_x, norm_x, size_idx)
            hard_logits, _ = stage3_model(raw_x, norm_x, size_pred_idx)
            soft_logits, _ = stage3_model.forward_soft(raw_x, norm_x, size_probs)
            reg015_logits, _ = stage3_model.forward_soft(raw_x, norm_x, gaussian_size_weights(size_reg_cm, 0.15))
            reg020_logits, _ = stage3_model.forward_soft(raw_x, norm_x, gaussian_size_weights(size_reg_cm, 0.20))
            reg025_logits, _ = stage3_model.forward_soft(raw_x, norm_x, gaussian_size_weights(size_reg_cm, 0.25))

            top2_probs = size_probs.clone()
            zero_idx = torch.argsort(top2_probs, dim=1, descending=True)[:, 2:]
            top2_probs.scatter_(1, zero_idx, 0.0)
            top2_probs = top2_probs / torch.clamp(top2_probs.sum(dim=1, keepdim=True), min=1e-8)
            top2_soft_logits, _ = stage3_model.forward_soft(raw_x, norm_x, top2_probs)

            temp05_probs = size_probs.pow(1.0 / 0.5)
            temp05_probs = temp05_probs / torch.clamp(temp05_probs.sum(dim=1, keepdim=True), min=1e-8)
            temp05_soft_logits, _ = stage3_model.forward_soft(raw_x, norm_x, temp05_probs)

            temp07_probs = size_probs.pow(1.0 / 0.7)
            temp07_probs = temp07_probs / torch.clamp(temp07_probs.sum(dim=1, keepdim=True), min=1e-8)
            temp07_soft_logits, _ = stage3_model.forward_soft(raw_x, norm_x, temp07_probs)

            temp03_probs = size_probs.pow(1.0 / 0.3)
            temp03_probs = temp03_probs / torch.clamp(temp03_probs.sum(dim=1, keepdim=True), min=1e-8)
            temp03_soft_logits, _ = stage3_model.forward_soft(raw_x, norm_x, temp03_probs)

            y_true.append(depth_idx.cpu().numpy().astype(np.int32))
            gt_size.append(size_idx.cpu().numpy().astype(np.int32))
            pred_size.append(size_pred_idx.cpu().numpy().astype(np.int32))
            size_prob_rows.append(size_probs.cpu().numpy().astype(np.float32))
            pred_gt_route.append(torch.argmax(gt_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_hard_route.append(torch.argmax(hard_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_soft_route.append(torch.argmax(soft_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_reg_sigma015_route.append(torch.argmax(reg015_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_reg_sigma020_route.append(torch.argmax(reg020_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_reg_sigma025_route.append(torch.argmax(reg025_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_top2_soft_route.append(torch.argmax(top2_soft_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_temp05_soft_route.append(torch.argmax(temp05_soft_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_temp07_soft_route.append(torch.argmax(temp07_soft_logits, dim=1).cpu().numpy().astype(np.int32))
            pred_temp03_soft_route.append(torch.argmax(temp03_soft_logits, dim=1).cpu().numpy().astype(np.int32))

    y_true = np.concatenate(y_true)
    gt_size = np.concatenate(gt_size)
    pred_size = np.concatenate(pred_size)
    size_prob_rows = np.concatenate(size_prob_rows, axis=0)
    pred_gt_route = np.concatenate(pred_gt_route)
    pred_hard_route = np.concatenate(pred_hard_route)
    pred_soft_route = np.concatenate(pred_soft_route)
    pred_reg_sigma015_route = np.concatenate(pred_reg_sigma015_route)
    pred_reg_sigma020_route = np.concatenate(pred_reg_sigma020_route)
    pred_reg_sigma025_route = np.concatenate(pred_reg_sigma025_route)
    pred_top2_soft_route = np.concatenate(pred_top2_soft_route)
    pred_temp05_soft_route = np.concatenate(pred_temp05_soft_route)
    pred_temp07_soft_route = np.concatenate(pred_temp07_soft_route)
    pred_temp03_soft_route = np.concatenate(pred_temp03_soft_route)

    size_top1 = float(np.mean(pred_size == gt_size))
    top2_idx = np.argsort(-size_prob_rows, axis=1)[:, :2]
    size_top2 = float(np.mean(np.any(top2_idx == gt_size[:, None], axis=1)))

    route_match = pred_size == gt_size
    gt_metrics = summarize_depth(y_true, pred_gt_route)
    hard_metrics = summarize_depth(y_true, pred_hard_route)
    soft_metrics = summarize_depth(y_true, pred_soft_route)
    reg015_metrics = summarize_depth(y_true, pred_reg_sigma015_route)
    reg020_metrics = summarize_depth(y_true, pred_reg_sigma020_route)
    reg025_metrics = summarize_depth(y_true, pred_reg_sigma025_route)
    top2_soft_metrics = summarize_depth(y_true, pred_top2_soft_route)
    temp05_soft_metrics = summarize_depth(y_true, pred_temp05_soft_route)
    temp07_soft_metrics = summarize_depth(y_true, pred_temp07_soft_route)
    temp03_soft_metrics = summarize_depth(y_true, pred_temp03_soft_route)
    hard_when_size_correct = summarize_depth(y_true[route_match], pred_hard_route[route_match]) if np.any(route_match) else None
    hard_when_size_wrong = summarize_depth(y_true[~route_match], pred_hard_route[~route_match]) if np.any(~route_match) else None

    gt_cm = np.asarray(gt_metrics["confusion_matrix"], dtype=np.int32)
    hard_cm = np.asarray(hard_metrics["confusion_matrix"], dtype=np.int32)
    soft_cm = np.asarray(soft_metrics["confusion_matrix"], dtype=np.int32)
    plot_confusion(gt_cm, "Depth with GT Size Routing", os.path.join(cfg.output_dir, "cm_gt_route.png"))
    plot_confusion(hard_cm, "Depth with Hard Predicted-Size Routing", os.path.join(cfg.output_dir, "cm_hard_pred_route.png"))
    plot_confusion(soft_cm, "Depth with Soft Predicted-Size Routing", os.path.join(cfg.output_dir, "cm_soft_pred_route.png"))
    plot_confusion(np.asarray(reg020_metrics["confusion_matrix"], dtype=np.int32), "Depth with Regression-Gaussian Routing", os.path.join(cfg.output_dir, "cm_regression_gaussian_route.png"))

    summary = {
        "protocol_v1": protocol_summary(),
        "validation_protocol": "file1_train_file2_val_file3_test_positive_windows",
        "stage2_ckpt": cfg.stage2_ckpt,
        "stage2_mode": stage2_mode,
        "stage3_ckpt": cfg.stage3_ckpt,
        "sample_count": int(len(y_true)),
        "size_prediction_on_stage3_test_windows": {
            "top1": size_top1,
            "top2": size_top2,
            "route_match_count": int(np.sum(route_match)),
            "route_match_rate": float(np.mean(route_match)),
        },
        "depth_metrics": {
            "gt_size_route": gt_metrics,
            "hard_predicted_size_route": hard_metrics,
            "soft_predicted_size_route": soft_metrics,
            "regression_sigma015_route": reg015_metrics,
            "regression_sigma020_route": reg020_metrics,
            "regression_sigma025_route": reg025_metrics,
            "top2_soft_predicted_size_route": top2_soft_metrics,
            "temp05_soft_predicted_size_route": temp05_soft_metrics,
            "temp07_soft_predicted_size_route": temp07_soft_metrics,
            "temp03_soft_predicted_size_route": temp03_soft_metrics,
            "hard_route_when_size_correct": hard_when_size_correct,
            "hard_route_when_size_wrong": hard_when_size_wrong,
        },
    }

    with open(os.path.join(cfg.output_dir, "stage3_predicted_size_routing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Stage3 Predicted-Size Routing Evaluation")
    lines.append("")
    lines.append("## Size Routing Quality")
    lines.append("")
    lines.append(f"- Size top-1 on Stage3 test windows: `{size_top1:.4f}`")
    lines.append(f"- Size top-2 on Stage3 test windows: `{size_top2:.4f}`")
    lines.append(f"- Exact route match rate: `{float(np.mean(route_match)):.4f}`")
    lines.append("")
    lines.append("## Depth Performance")
    lines.append("")
    lines.append(f"- GT size route balanced accuracy: `{gt_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Hard predicted-size route balanced accuracy: `{hard_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Soft predicted-size route balanced accuracy: `{soft_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Regression-Gaussian route (sigma=0.15) balanced accuracy: `{reg015_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Regression-Gaussian route (sigma=0.20) balanced accuracy: `{reg020_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Regression-Gaussian route (sigma=0.25) balanced accuracy: `{reg025_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Top2-soft predicted-size route balanced accuracy: `{top2_soft_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Temperature-0.5 soft route balanced accuracy: `{temp05_soft_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Temperature-0.7 soft route balanced accuracy: `{temp07_soft_metrics['balanced_accuracy']:.4f}`")
    lines.append(f"- Temperature-0.3 soft route balanced accuracy: `{temp03_soft_metrics['balanced_accuracy']:.4f}`")
    if hard_when_size_correct is not None:
        lines.append(f"- Hard route balanced accuracy when size route is correct: `{hard_when_size_correct['balanced_accuracy']:.4f}`")
    if hard_when_size_wrong is not None:
        lines.append(f"- Hard route balanced accuracy when size route is wrong: `{hard_when_size_wrong['balanced_accuracy']:.4f}`")
    with open(os.path.join(cfg.output_dir, "STAGE3_PREDICTED_SIZE_ROUTING_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
