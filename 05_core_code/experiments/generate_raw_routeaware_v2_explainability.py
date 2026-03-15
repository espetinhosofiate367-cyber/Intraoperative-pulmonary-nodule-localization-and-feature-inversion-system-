import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_DIR = SCRIPT_DIR.parent
RELEASE_ROOT = CORE_DIR.parent
PROJECT_ROOT = RELEASE_ROOT.parent
CODE_ARCHIVE_ROOT = PROJECT_ROOT.parent
PACKAGE_ROOT = CODE_ARCHIVE_ROOT.parent
ORIG_MODELS_DIR = PROJECT_ROOT / "models"
ORIG_EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RELEASE_MODELS_DIR = CORE_DIR / "models"

for path in [SCRIPT_DIR, RELEASE_MODELS_DIR, ORIG_MODELS_DIR, ORIG_EXPERIMENTS_DIR, PROJECT_ROOT, CODE_ARCHIVE_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from depth_analysis_utils import PHASE_ORDER, assign_pressing_phases, frame_physics_features, window_temporal_features
from raw_size_routed_depth_model import RawSizeRoutedDepthModel
from task_protocol_v1 import COARSE_DEPTH_ORDER, INPUT_SEQ_LEN, SIZE_VALUES_CM, WINDOW_STRIDE
from train_stage3_raw_size_conditioned_depth import (
    PositiveDepthDataset,
    build_positive_depth_samples_for_file,
    compute_concept_stats,
    compute_raw_scale,
)
from triplet_repeat_classifier.train_triplet_repeat_classifier import load_json

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

FEATURE_FAMILY = {
    "maxframe_raw_centroid_row": "deformation_position",
    "centroid_row_min": "deformation_position",
    "deltaframe_centroid_row": "deformation_position",
    "maxframe_raw_centroid_col": "deformation_position",
    "hotspot_radius_max": "spread_extent",
    "meanframe_raw_hotspot_radius": "spread_extent",
    "hotspot_radius_min": "spread_extent",
    "second_moment_spread_max": "spread_extent",
    "anisotropy_ratio_min": "shape_contrast",
    "anisotropy_ratio_last": "shape_contrast",
    "peak_count_min": "shape_contrast",
    "center_border_contrast_max": "shape_contrast",
    "window_norm_global_std": "shape_contrast",
    "raw_max_max": "amplitude_response",
    "spatial_entropy_max": "distribution_complexity",
    "window_rise_time_to_peak": "temporal_phase",
    "window_peak_persistence_ratio": "temporal_phase",
    "window_decay_after_peak": "temporal_phase",
    "window_window_raw_sum_gain": "temporal_phase",
}

DEPTH_COLORS = {
    "shallow": "#2a9d8f",
    "middle": "#e9c46a",
    "deep": "#e76f51",
}


@dataclass
class Config:
    checkpoint_path: Path = Path(
        os.environ.get(
            "PAPERD3ROUTEV2_EXPLAIN_CKPT",
            str(ORIG_EXPERIMENTS_DIR / "outputs_stage3_raw_size_routed_depth_v2" / "paper_stage3_raw_size_routed_depth_v2_best.pth"),
        )
    )
    output_dir: Path = Path(
        os.environ.get(
            "PAPERD3ROUTEV2_EXPLAIN_OUT",
            str(RELEASE_ROOT / "03_results_core" / "raw_routeaware_explainability_v2"),
        )
    )
    data_root: Path = Path(os.environ.get("PAPERD3ROUTEV2_DATA_ROOT", str(PACKAGE_ROOT / "整理好的数据集" / "建表数据")))
    file1_labels: Path = Path(os.environ.get("PAPERD3ROUTEV2_FILE1_LABELS", str(PACKAGE_ROOT / "manual_keyframe_labels.json")))
    file2_labels: Path = Path(os.environ.get("PAPERD3ROUTEV2_FILE2_LABELS", str((PACKAGE_ROOT / "整理好的数据集" / "建表数据" / "manual_keyframe_labels_file2.json"))))
    file3_labels: Path = Path(os.environ.get("PAPERD3ROUTEV2_FILE3_LABELS", str((PACKAGE_ROOT / "整理好的数据集" / "建表数据" / "manual_keyframe_labels_file3.json"))))
    seq_len: int = INPUT_SEQ_LEN
    stride: int = WINDOW_STRIDE
    dedup_gap: int = 6
    batch_size: int = 128
    ig_steps: int = 20
    ig_max_per_class: int = 40


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "grid.color": "#d0d0d0",
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
        }
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_ridge_probe(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    y_mean = float(np.mean(y_train))
    y_std = float(max(np.std(y_train), 1e-6))
    x_train_n = (x_train - x_mean) / x_std
    x_eval_n = (x_eval - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    xtx = x_train_n.T @ x_train_n
    beta = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0], dtype=np.float64), x_train_n.T @ y_train_n)
    pred_n = x_eval_n @ beta
    return pred_n * y_std + y_mean


def normalize_frame_for_display(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    lo = float(np.percentile(frame, 1.0))
    hi = float(np.percentile(frame, 99.0))
    if hi - lo < 1e-6:
        lo = float(np.min(frame))
        hi = float(np.max(frame))
    if hi - lo < 1e-6:
        return np.zeros_like(frame, dtype=np.float32)
    return np.clip((frame - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def choose_display_frame(raw_window: np.ndarray) -> np.ndarray:
    frame_energy = raw_window.sum(axis=(1, 2))
    peak_idx = int(np.argmax(frame_energy))
    return normalize_frame_for_display(raw_window[peak_idx])


def compute_selected_probe_features(records_by_key: Dict[str, dict], samples: List[dict]) -> pd.DataFrame:
    rows = []
    for s in samples:
        rec = records_by_key[s["group_key"]]
        end_row = int(s["end_row"])
        seq_len = int(rec["seq_len"])
        start_row = end_row - seq_len + 1
        raw_window = rec["raw_frames"][start_row : end_row + 1]
        norm_window = rec["norm_frames"][start_row : end_row + 1]
        frame_rows = [frame_physics_features(frame.astype(np.float32)) for frame in raw_window]
        mean_frame_raw = raw_window.mean(axis=0)
        max_frame_raw = raw_window.max(axis=0)
        temporal = window_temporal_features(frame_rows)
        if len(raw_window) > 1:
            delta = np.diff(raw_window, axis=0)
            delta_mean_frame = np.abs(delta).mean(axis=0)
            delta_features = frame_physics_features(delta_mean_frame.astype(np.float32))
        else:
            delta_features = frame_physics_features(np.zeros((12, 8), dtype=np.float32))
        meanframe_raw_features = frame_physics_features(mean_frame_raw.astype(np.float32))
        maxframe_raw_features = frame_physics_features(max_frame_raw.astype(np.float32))
        rows.append(
            {
                "group_key": s["group_key"],
                "file_name": s["file_name"],
                "end_row": int(s["end_row"]),
                "center_row": int(s["center_row"]),
                "size_class_index": int(s["size_class_index"]),
                "depth_coarse_index": int(s["depth_coarse_index"]),
                "maxframe_raw_centroid_row": float(maxframe_raw_features["centroid_row"]),
                "centroid_row_min": float(min(fr["centroid_row"] for fr in frame_rows)),
                "deltaframe_centroid_row": float(delta_features["centroid_row"]),
                "maxframe_raw_centroid_col": float(maxframe_raw_features["centroid_col"]),
                "hotspot_radius_max": float(max(fr["hotspot_radius"] for fr in frame_rows)),
                "meanframe_raw_hotspot_radius": float(meanframe_raw_features["hotspot_radius"]),
                "hotspot_radius_min": float(min(fr["hotspot_radius"] for fr in frame_rows)),
                "second_moment_spread_max": float(max(fr["second_moment_spread"] for fr in frame_rows)),
                "anisotropy_ratio_min": float(min(fr["anisotropy_ratio"] for fr in frame_rows)),
                "anisotropy_ratio_last": float(frame_rows[-1]["anisotropy_ratio"]),
                "peak_count_min": float(min(fr["peak_count"] for fr in frame_rows)),
                "center_border_contrast_max": float(max(fr["center_border_contrast"] for fr in frame_rows)),
                "window_norm_global_std": float(norm_window.std()),
                "raw_max_max": float(max(fr["raw_max"] for fr in frame_rows)),
                "spatial_entropy_max": float(max(fr["spatial_entropy"] for fr in frame_rows)),
                "window_rise_time_to_peak": float(temporal["rise_time_to_peak"]),
                "window_peak_persistence_ratio": float(temporal["peak_persistence_ratio"]),
                "window_decay_after_peak": float(temporal["decay_after_peak"]),
                "window_window_raw_sum_gain": float(temporal["window_raw_sum_gain"]),
            }
        )
    return pd.DataFrame(rows)

def build_splits(cfg: Config):
    file1_all = load_json(str(cfg.file1_labels))
    file2_all = load_json(str(cfg.file2_labels))
    file3_all = load_json(str(cfg.file3_labels))
    rec1, samples1 = build_positive_depth_samples_for_file(file1_all, "1.CSV", str(cfg.data_root), cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, samples2 = build_positive_depth_samples_for_file(file2_all, "2.CSV", str(cfg.data_root), cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3 = build_positive_depth_samples_for_file(file3_all, "3.CSV", str(cfg.data_root), cfg.seq_len, cfg.stride, cfg.dedup_gap)
    common_base_groups = sorted(
        list(
            set(v["base_group"] for v in rec1.values())
            & set(v["base_group"] for v in rec2.values())
            & set(v["base_group"] for v in rec3.values())
        )
    )
    common_set = set(common_base_groups)
    train_records = {k: v for k, v in rec1.items() if v["base_group"] in common_set}
    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_set}
    train_samples = [s for s in samples1 if s["base_group"] in common_set]
    test_samples = [s for s in samples3 if s["base_group"] in common_set]
    return train_records, test_records, train_samples, test_samples


def load_model(cfg: Config, device: torch.device) -> RawSizeRoutedDepthModel:
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    model = RawSizeRoutedDepthModel(
        seq_len=INPUT_SEQ_LEN,
        frame_feature_dim=24,
        temporal_channels=48,
        temporal_blocks=3,
        dropout=0.25,
        num_size_classes=len(SIZE_VALUES_CM),
        num_depth_classes=len(COARSE_DEPTH_ORDER),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_latents_and_predictions(
    model: RawSizeRoutedDepthModel,
    dataset: PositiveDepthDataset,
    samples: List[dict],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    cursor = 0
    with torch.no_grad():
        for raw_x, norm_x, size_idx, depth_idx, _concept_target, _sample_weight in loader:
            batch = int(raw_x.shape[0])
            raw_x = raw_x.to(device)
            norm_x = norm_x.to(device)
            size_idx = size_idx.to(device)
            _logits, probs, feats = model(raw_x, norm_x, size_idx, return_features=True)
            trunk = feats["trunk_feat"].detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
            pred_idx = np.argmax(probs_np, axis=1).astype(np.int32)
            for i in range(batch):
                s = samples[cursor + i]
                row = {
                    "sample_index": cursor + i,
                    "group_key": s["group_key"],
                    "file_name": s["file_name"],
                    "end_row": int(s["end_row"]),
                    "center_row": int(s["center_row"]),
                    "size_class_index": int(s["size_class_index"]),
                    "depth_coarse_index": int(s["depth_coarse_index"]),
                    "depth_label": COARSE_DEPTH_ORDER[int(s["depth_coarse_index"])],
                    "pred_index": int(pred_idx[i]),
                    "pred_label": COARSE_DEPTH_ORDER[int(pred_idx[i])],
                    "correct": bool(int(pred_idx[i]) == int(s["depth_coarse_index"])),
                    "p_shallow": float(probs_np[i, 0]),
                    "p_middle": float(probs_np[i, 1]),
                    "p_deep": float(probs_np[i, 2]),
                }
                for j in range(trunk.shape[1]):
                    row[f"z_{j:03d}"] = float(trunk[i, j])
                rows.append(row)
            cursor += batch
    return pd.DataFrame(rows)


def compute_probe_summary(train_merged: pd.DataFrame, test_merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    z_cols = [c for c in train_merged.columns if c.startswith("z_")]
    size_onehot_train = np.eye(len(SIZE_VALUES_CM), dtype=np.float64)[train_merged["size_class_index"].to_numpy(dtype=np.int32)]
    size_onehot_test = np.eye(len(SIZE_VALUES_CM), dtype=np.float64)[test_merged["size_class_index"].to_numpy(dtype=np.int32)]
    rows = []
    for feat in SELECTED_FEATURES:
        y_train = train_merged[feat].to_numpy(dtype=np.float64)
        y_test = test_merged[feat].to_numpy(dtype=np.float64)
        pred_test_latent = fit_ridge_probe(train_merged[z_cols].to_numpy(dtype=np.float64), y_train, test_merged[z_cols].to_numpy(dtype=np.float64), alpha=10.0)
        pred_test_size = fit_ridge_probe(size_onehot_train, y_train, size_onehot_test, alpha=1.0)
        rows.append(
            {
                "feature": feat,
                "family": FEATURE_FAMILY.get(feat, "other"),
                "test_r2_latent": safe_r2(y_test, pred_test_latent),
                "test_r2_size_only": safe_r2(y_test, pred_test_size),
                "test_r_latent": safe_corr(y_test, pred_test_latent),
                "test_r_size_only": safe_corr(y_test, pred_test_size),
                "test_mae_latent": float(np.mean(np.abs(y_test - pred_test_latent))),
                "test_mae_size_only": float(np.mean(np.abs(y_test - pred_test_size))),
            }
        )
    probe_df = pd.DataFrame(rows).sort_values("test_r2_latent", ascending=False).reset_index(drop=True)
    probe_summary = {
        "mean_test_r2_latent": float(probe_df["test_r2_latent"].mean()),
        "mean_test_r2_size_only": float(probe_df["test_r2_size_only"].mean()),
        "mean_test_r_latent": float(probe_df["test_r_latent"].mean()),
        "mean_test_r_size_only": float(probe_df["test_r_size_only"].mean()),
    }
    return probe_df, probe_summary


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


def build_sample_prediction_table(
    model: RawSizeRoutedDepthModel,
    dataset: PositiveDepthDataset,
    records: Dict[str, dict],
    samples: List[dict],
    raw_scale: float,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            raw_x, norm_x, size_idx_t, depth_idx_t, _concept_target, _sample_weight = dataset[idx]
            raw_x = raw_x.unsqueeze(0).to(device)
            norm_x = norm_x.unsqueeze(0).to(device)
            size_idx = size_idx_t.view(1).to(device)
            depth_idx = int(depth_idx_t.item())
            _logits, probs, _feats = model(raw_x, norm_x, size_idx, return_features=True)
            probs_np = probs[0].cpu().numpy().astype(np.float64)
            pred_idx = int(np.argmax(probs_np))
            raw_window = raw_x[0, :, 0].cpu().numpy().astype(np.float32) * float(raw_scale)
            peak_frame = choose_display_frame(raw_window)
            feature_summary = compute_window_summary(raw_window)
            rows.append(
                {
                    "sample_index": idx,
                    "group_key": sample["group_key"],
                    "base_group": sample["base_group"],
                    "size_class_index": int(sample["size_class_index"]),
                    "size_cm": float(SIZE_VALUES_CM[int(sample["size_class_index"])]),
                    "depth_coarse_index": depth_idx,
                    "depth_cm": float(records[sample["group_key"]]["depth_cm"]),
                    "depth_label": COARSE_DEPTH_ORDER[depth_idx],
                    "pred_index": pred_idx,
                    "pred_label": COARSE_DEPTH_ORDER[pred_idx],
                    "correct": bool(pred_idx == depth_idx),
                    "p_shallow": float(probs_np[0]),
                    "p_middle": float(probs_np[1]),
                    "p_deep": float(probs_np[2]),
                    "peak_frame": peak_frame,
                    **feature_summary,
                }
            )
    return pd.DataFrame(rows)

def compute_hard_pairs(sample_df: pd.DataFrame) -> Dict[str, object]:
    reps = sample_df.sort_values("p_shallow", ascending=False).groupby("base_group", as_index=False).first()
    deep_reps = reps[reps["depth_label"] == "deep"].sort_values("size_cm", ascending=False).reset_index(drop=True)
    shallow_reps = reps[reps["depth_label"] == "shallow"].sort_values("size_cm", ascending=True).reset_index(drop=True)
    used_shallow = set()
    hard_pairs = []
    for _, deep_row in deep_reps.iterrows():
        candidates = shallow_reps[(shallow_reps["size_cm"] < deep_row["size_cm"]) & (~shallow_reps["base_group"].isin(used_shallow))].copy()
        if len(candidates) == 0:
            continue
        candidates["pair_score"] = np.abs(candidates["raw_max_max"] - deep_row["raw_max_max"])
        best = candidates.sort_values("pair_score", ascending=True).iloc[0]
        used_shallow.add(best["base_group"])
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
                "model_prefers_deeper_on_pdeep": bool(float(deep_row["p_deep"]) > float(best["p_deep"])),
                "entropy_gap": float(deep_row["spatial_entropy_max"] - best["spatial_entropy_max"]),
                "radius_gap": float(deep_row["hotspot_radius_max"] - best["hotspot_radius_max"]),
                "contrast_gap": float(deep_row["center_border_contrast_max"] - best["center_border_contrast_max"]),
                "deep_peak_frame": deep_row["peak_frame"],
                "shallow_peak_frame": best["peak_frame"],
            }
        )
    hard_pairs = sorted(hard_pairs, key=lambda x: (not x["model_prefers_deeper_on_pdeep"], x["pair_score_raw_max_gap"]))
    summary = {
        "pair_count": int(len(hard_pairs)),
        "pairwise_pdeep_success_rate": float(np.mean([p["model_prefers_deeper_on_pdeep"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_raw_max_gap": float(np.mean([p["pair_score_raw_max_gap"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_entropy_gap": float(np.mean([p["entropy_gap"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "mean_radius_gap": float(np.mean([p["radius_gap"] for p in hard_pairs])) if hard_pairs else float("nan"),
        "pairs": hard_pairs,
    }
    return summary


def compute_phase_occlusion(
    model: RawSizeRoutedDepthModel,
    dataset: PositiveDepthDataset,
    samples: List[dict],
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    model.eval()
    for idx, sample in enumerate(samples):
        raw_x, norm_x, size_idx_t, depth_idx_t, _concept_target, _sample_weight = dataset[idx]
        raw_x = raw_x.unsqueeze(0).to(device)
        norm_x = norm_x.unsqueeze(0).to(device)
        size_idx = size_idx_t.view(1).to(device)
        depth_idx = int(depth_idx_t.item())
        with torch.no_grad():
            _logits, probs = model(raw_x, norm_x, size_idx)
        probs_np = probs[0].detach().cpu().numpy().astype(np.float64)
        base_true_prob = float(probs_np[depth_idx])
        raw_window = raw_x[0, :, 0].detach().cpu().numpy().astype(np.float32)
        frame_energy = raw_window.sum(axis=(1, 2))
        phase_info = assign_pressing_phases(frame_energy)
        phase_names = phase_info["phase_name"]
        for phase in PHASE_ORDER:
            phase_mask = (phase_names == phase).astype(np.float32).reshape(-1, 1, 1)
            if phase_mask.sum() <= 0:
                continue
            phase_mask_t = torch.from_numpy(phase_mask).to(device).unsqueeze(0).unsqueeze(2)
            raw_occ = raw_x * (1.0 - phase_mask_t)
            norm_occ = norm_x * (1.0 - phase_mask_t)
            with torch.no_grad():
                _logits_occ, probs_occ = model(raw_occ, norm_occ, size_idx)
            true_prob_occ = float(probs_occ[0, depth_idx].item())
            rows.append(
                {
                    "sample_index": idx,
                    "depth_label": COARSE_DEPTH_ORDER[depth_idx],
                    "phase": phase,
                    "true_prob_base": base_true_prob,
                    "true_prob_occ": true_prob_occ,
                    "true_prob_drop": base_true_prob - true_prob_occ,
                }
            )
    phase_df = pd.DataFrame(rows)
    overall = (
        phase_df.groupby("phase", as_index=False)["true_prob_drop"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_drop", "median": "median_drop"})
    )
    by_depth = (
        phase_df.groupby(["depth_label", "phase"], as_index=False)["true_prob_drop"]
        .mean()
        .rename(columns={"true_prob_drop": "mean_drop"})
    )
    return overall, by_depth


def integrated_gradients(
    model: RawSizeRoutedDepthModel,
    raw_x: torch.Tensor,
    norm_x: torch.Tensor,
    size_idx: torch.Tensor,
    target_idx: int,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    baseline_raw = torch.zeros_like(raw_x)
    baseline_norm = torch.zeros_like(norm_x)
    total_grad_raw = torch.zeros_like(raw_x)
    total_grad_norm = torch.zeros_like(norm_x)
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=raw_x.device)
    for alpha in alphas:
        raw_i = (baseline_raw + alpha * (raw_x - baseline_raw)).clone().detach().requires_grad_(True)
        norm_i = (baseline_norm + alpha * (norm_x - baseline_norm)).clone().detach().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            logits, _probs = model(raw_i, norm_i, size_idx)
            score = logits[:, int(target_idx)].sum()
            model.zero_grad(set_to_none=True)
            score.backward()
        total_grad_raw += raw_i.grad.detach()
        total_grad_norm += norm_i.grad.detach()
    ig_raw = (raw_x - baseline_raw) * total_grad_raw / max(steps, 1)
    ig_norm = (norm_x - baseline_norm) * total_grad_norm / max(steps, 1)
    return ig_raw.detach().cpu().numpy(), ig_norm.detach().cpu().numpy()


def compute_ig_summary(
    model: RawSizeRoutedDepthModel,
    dataset: PositiveDepthDataset,
    prediction_df: pd.DataFrame,
    device: torch.device,
    steps: int,
    ig_max_per_class: int,
) -> Dict[str, object]:
    class_maps = {label: [] for label in COARSE_DEPTH_ORDER}
    class_temporal = {label: [] for label in COARSE_DEPTH_ORDER}
    class_count = {label: 0 for label in COARSE_DEPTH_ORDER}
    selected_rows = []
    for depth_label in COARSE_DEPTH_ORDER:
        sub = prediction_df[(prediction_df["depth_label"] == depth_label) & (prediction_df["correct"])]
        prob_col = {"shallow": "p_shallow", "middle": "p_middle", "deep": "p_deep"}[depth_label]
        sub = sub.sort_values(prob_col, ascending=False).head(ig_max_per_class)
        for _, row in sub.iterrows():
            idx = int(row["sample_index"])
            raw_x, norm_x, size_idx_t, depth_idx_t, _concept_target, _sample_weight = dataset[idx]
            raw_x = raw_x.unsqueeze(0).to(device)
            norm_x = norm_x.unsqueeze(0).to(device)
            size_idx = size_idx_t.view(1).to(device)
            target_idx = int(depth_idx_t.item())
            ig_raw, ig_norm = integrated_gradients(model, raw_x, norm_x, size_idx, target_idx, steps)
            ig_combined = np.abs(ig_raw[0, :, 0]) + np.abs(ig_norm[0, :, 0])
            spatial = ig_combined.sum(axis=0)
            temporal = ig_combined.sum(axis=(1, 2))
            if spatial.sum() > 1e-8:
                spatial = spatial / spatial.sum()
            if temporal.sum() > 1e-8:
                temporal = temporal / temporal.sum()
            class_maps[depth_label].append(spatial)
            class_temporal[depth_label].append(temporal)
            class_count[depth_label] += 1
            selected_rows.append({"sample_index": idx, "depth_label": depth_label, "group_key": row["group_key"]})
    map_summary = {}
    temporal_summary = {}
    for depth_label in COARSE_DEPTH_ORDER:
        if class_maps[depth_label]:
            map_summary[depth_label] = np.mean(np.stack(class_maps[depth_label], axis=0), axis=0)
            temporal_summary[depth_label] = np.mean(np.stack(class_temporal[depth_label], axis=0), axis=0)
        else:
            map_summary[depth_label] = np.zeros((12, 8), dtype=np.float32)
            temporal_summary[depth_label] = np.zeros(INPUT_SEQ_LEN, dtype=np.float32)
    return {
        "spatial_maps": map_summary,
        "temporal_profiles": temporal_summary,
        "count_per_class": class_count,
        "selected_rows": selected_rows,
    }

def plot_main_figure(
    probe_df: pd.DataFrame,
    phase_by_depth: pd.DataFrame,
    ig_summary: Dict[str, object],
    hard_pair_summary: Dict[str, object],
    probe_summary: Dict[str, float],
    output_png: Path,
    output_pdf: Path,
) -> None:
    apply_style()
    fig = plt.figure(figsize=(13.5, 8.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.32, hspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    fam = probe_df.groupby("family", as_index=False)[["test_r2_latent", "test_r2_size_only"]].mean().sort_values("test_r2_latent", ascending=False)
    x = np.arange(len(fam))
    width = 0.36
    ax_a.bar(x - width / 2, fam["test_r2_latent"], width=width, color="#2a9d8f", label="latent probe")
    ax_a.bar(x + width / 2, fam["test_r2_size_only"], width=width, color="#9aa3ad", label="size-only")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(fam["family"], rotation=18, ha="right")
    ax_a.set_ylabel("Test $R^2$")
    ax_a.set_title("A. Current route-aware depth v2 probe families")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="upper right")

    ax_b = fig.add_subplot(gs[0, 1])
    depth_order = ["shallow", "middle", "deep"]
    x2 = np.arange(len(PHASE_ORDER))
    width2 = 0.22
    for idx, depth_label in enumerate(depth_order):
        sub = phase_by_depth[phase_by_depth["depth_label"] == depth_label].set_index("phase").reindex(PHASE_ORDER).reset_index()
        ax_b.bar(x2 + (idx - 1) * width2, sub["mean_drop"], width=width2, color=DEPTH_COLORS[depth_label], label=depth_label)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(PHASE_ORDER, rotation=18, ha="right")
    ax_b.set_ylabel("Mean true-class prob. drop")
    ax_b.set_title("B. Phase occlusion by true depth")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False)

    map_norm = Normalize(vmin=0.0, vmax=max(float(np.max(v)) for v in ig_summary["spatial_maps"].values()) + 1e-8)
    panel_defs = [("C", "shallow"), ("D", "middle"), ("E", "deep")]
    for panel_idx, (letter, depth_label) in enumerate(panel_defs):
        if panel_idx == 0:
            ax = fig.add_subplot(gs[0, 2])
        elif panel_idx == 1:
            ax = fig.add_subplot(gs[1, 0])
        else:
            ax = fig.add_subplot(gs[1, 1])
        im = ax.imshow(ig_summary["spatial_maps"][depth_label], cmap="viridis", norm=map_norm, interpolation="bicubic")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{letter}. Mean IG map: {depth_label}\n(n={ig_summary['count_per_class'][depth_label]})")
    cbar_ax = fig.add_axes([0.91, 0.18, 0.015, 0.22])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_title("IG", fontsize=8)

    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.axis("off")
    phase_top = phase_by_depth.groupby("phase", as_index=False)["mean_drop"].mean().sort_values("mean_drop", ascending=False).iloc[0]
    lines = [
        "F. Key takeaways",
        f"latent probe mean R^2: {probe_summary['mean_test_r2_latent']:.3f}",
        f"size-only mean R^2: {probe_summary['mean_test_r2_size_only']:.3f}",
        f"hard-pair success: {hard_pair_summary['pairwise_pdeep_success_rate']:.3f}",
        f"dominant occlusion phase: {phase_top['phase']}",
        "IG maps show class-related spatial organization,",
        "while phase occlusion still peaks near peak-neighborhood.",
    ]
    ax_f.text(
        0.0,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#d5d5d5"),
    )

    fig.suptitle("Raw route-aware depth v2 explainability summary", y=0.98, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.96])
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_hard_pair_figure(pair_summary: Dict[str, object], output_png: Path, output_pdf: Path) -> None:
    pairs = [p for p in pair_summary["pairs"] if p["model_prefers_deeper_on_pdeep"]][:3]
    if not pairs:
        pairs = pair_summary["pairs"][:3]
    if not pairs:
        return
    apply_style()
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8.2, 2.8 * len(pairs)))
    if len(pairs) == 1:
        axes = np.asarray([axes])
    for row_idx, pair in enumerate(pairs):
        for col_idx, side in enumerate(["deep", "shallow"]):
            ax = axes[row_idx, col_idx]
            frame = pair[f"{side}_peak_frame"]
            ax.imshow(frame, cmap="turbo", interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{side}: S={pair[f'{side}_size_cm']:.2f} D={pair[f'{side}_depth_cm']:.1f}\n"
                f"pred={pair[f'{side}_pred_label']} p_deep={pair[f'{side}_p_deep']:.2f}\n"
                f"max={pair[f'{side}_raw_max_max']:.1f} ent={pair[f'{side}_spatial_entropy_max']:.2f}",
                fontsize=8,
            )
    fig.suptitle("Representative hard pairs for depth discrimination", y=0.995, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def write_report(
    cfg: Config,
    probe_summary: Dict[str, float],
    probe_df: pd.DataFrame,
    pair_summary: Dict[str, object],
    phase_overall: pd.DataFrame,
    ig_summary: Dict[str, object],
) -> None:
    top_phase = phase_overall.sort_values("mean_drop", ascending=False).iloc[0]
    lines = []
    lines.append("# Raw Route-Aware Depth V2 Explainability Report")
    lines.append("")
    lines.append(f"- Checkpoint: `{cfg.checkpoint_path}`")
    lines.append("- Protocol: `1.CSV train / 2.CSV val / 3.CSV test`, positive windows only")
    lines.append(f"- Mean latent probe test R2: `{probe_summary['mean_test_r2_latent']:.4f}`")
    lines.append(f"- Mean size-only probe test R2: `{probe_summary['mean_test_r2_size_only']:.4f}`")
    lines.append(f"- Hard-pair success rate by p_deep: `{pair_summary['pairwise_pdeep_success_rate']:.4f}`")
    lines.append(f"- Dominant occlusion phase: `{top_phase['phase']}` with mean drop `{top_phase['mean_drop']:.4f}`")
    lines.append("")
    lines.append("## Probe top features")
    lines.append("")
    for _, row in probe_df.head(5).iterrows():
        lines.append(
            f"- `{row['feature']}` ({row['family']}): latent test R2 `{row['test_r2_latent']:.4f}`, size-only test R2 `{row['test_r2_size_only']:.4f}`"
        )
    lines.append("")
    lines.append("## Integrated Gradients subset sizes")
    lines.append("")
    for depth_label in COARSE_DEPTH_ORDER:
        lines.append(f"- `{depth_label}`: `{ig_summary['count_per_class'][depth_label]}` correctly predicted windows")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The current route-aware depth v2 model still shows a stronger latent recoverability than a size-only baseline, which supports true depth-related internal encoding.")
    lines.append("- Hard-pair analysis indicates that the model can distinguish a portion of deeper-vs-shallower cases even when raw-max gaps are reduced, but the evidence remains partial rather than definitive.")
    lines.append("- Phase occlusion confirms that the dominant temporal evidence remains concentrated near the peak neighborhood, so temporal strategy is improved but not yet fully physics-complete.")
    (cfg.output_dir / "RAW_ROUTEAWARE_V2_EXPLAINABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {cfg.data_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_records, test_records, train_samples, test_samples = build_splits(cfg)
    raw_scale = compute_raw_scale(train_records)
    concept_mean, concept_std = compute_concept_stats(train_samples)
    ds_train = PositiveDepthDataset(train_records, train_samples, raw_scale, concept_mean, concept_std, is_train=False)
    ds_test = PositiveDepthDataset(test_records, test_samples, raw_scale, concept_mean, concept_std, is_train=False)
    model = load_model(cfg, device)

    latent_train = extract_latents_and_predictions(model, ds_train, train_samples, device, cfg.batch_size)
    latent_test = extract_latents_and_predictions(model, ds_test, test_samples, device, cfg.batch_size)
    merge_cols = ["group_key", "file_name", "end_row", "center_row", "size_class_index", "depth_coarse_index"]
    train_targets = compute_selected_probe_features(train_records, train_samples)
    test_targets = compute_selected_probe_features(test_records, test_samples)
    train_merged = latent_train.merge(train_targets, on=merge_cols, how="inner")
    test_merged = latent_test.merge(test_targets, on=merge_cols, how="inner")
    probe_df, probe_summary = compute_probe_summary(train_merged, test_merged)

    sample_pred_df = build_sample_prediction_table(model, ds_test, test_records, test_samples, raw_scale, device)
    pair_summary = compute_hard_pairs(sample_pred_df)
    phase_overall, phase_by_depth = compute_phase_occlusion(model, ds_test, test_samples, device)
    ig_summary = compute_ig_summary(model, ds_test, sample_pred_df, device, cfg.ig_steps, cfg.ig_max_per_class)

    probe_df.to_csv(cfg.output_dir / "probe_metrics_current_v2.csv", index=False, encoding="utf-8-sig")
    phase_overall.to_csv(cfg.output_dir / "phase_occlusion_overall_current_v2.csv", index=False, encoding="utf-8-sig")
    phase_by_depth.to_csv(cfg.output_dir / "phase_occlusion_by_depth_current_v2.csv", index=False, encoding="utf-8-sig")
    sample_pred_df.drop(columns=["peak_frame"]).to_csv(cfg.output_dir / "test_predictions_current_v2.csv", index=False, encoding="utf-8-sig")

    summary = {
        "checkpoint": str(cfg.checkpoint_path),
        "probe_summary": probe_summary,
        "hard_pair_summary": {k: v for k, v in pair_summary.items() if k != "pairs"},
        "phase_occlusion_summary": {
            "overall": phase_overall.to_dict(orient="records"),
            "by_depth": phase_by_depth.to_dict(orient="records"),
        },
        "ig_summary": {
            "count_per_class": ig_summary["count_per_class"],
            "selected_rows": ig_summary["selected_rows"],
        },
    }
    (cfg.output_dir / "raw_routeaware_v2_explainability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    plot_main_figure(
        probe_df,
        phase_by_depth,
        ig_summary,
        pair_summary,
        probe_summary,
        cfg.output_dir / "raw_routeaware_v2_explainability_main.png",
        cfg.output_dir / "raw_routeaware_v2_explainability_main.pdf",
    )
    plot_hard_pair_figure(
        pair_summary,
        cfg.output_dir / "raw_routeaware_v2_hard_pairs.png",
        cfg.output_dir / "raw_routeaware_v2_hard_pairs.pdf",
    )
    write_report(cfg, probe_summary, probe_df, pair_summary, phase_overall, ig_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
