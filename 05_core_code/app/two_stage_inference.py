from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from dual_stream_mstcn_detection import DualStreamMSTCNDetector
from dual_stream_mstcn_multitask import DualStreamMSTCNMultiTask
from hierarchical_positive_inverter import HierarchicalPositiveInverter
from task_protocol_v1 import (
    INPUT_SEQ_LEN,
    SIZE_VALUES_CM,
    class_index_to_size,
    coarse_index_to_name,
    format_runtime_payload,
)
from depth_analysis_utils import frame_physics_features, window_temporal_features
from train_xgboost_baselines import window_feature_row


COARSE_DEPTH_DISPLAY = {
    "shallow": "浅层 (0.5-1.0 cm)",
    "middle": "中层 (1.5-2.0 cm)",
    "deep": "深层 (2.5-3.0 cm)",
}


def _torch_load_compat(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary_threshold(summary: Dict, fallback: float = 0.62) -> float:
    best_record = summary.get("best_record", {}) if isinstance(summary, dict) else {}
    threshold = best_record.get("val_best_threshold")
    if threshold is None:
        threshold = summary.get("stage1_reference_metrics", {}).get("stage1_val_best_threshold")
    try:
        return float(threshold)
    except Exception:
        return float(fallback)


def _coerce_frame_to_matrix(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.shape == (12, 8):
        return arr.astype(np.float32, copy=False)
    flat = arr.reshape(-1)
    if flat.size < 96:
        raise ValueError(f"Frame has {flat.size} values, expected at least 96.")
    if flat.size > 96:
        flat = flat[-96:]
    return flat.reshape(12, 8).astype(np.float32, copy=False)


def _normalize_sequence(seq_raw: np.ndarray) -> np.ndarray:
    seq_raw = np.asarray(seq_raw, dtype=np.float32)
    if seq_raw.shape != (INPUT_SEQ_LEN, 12, 8):
        raise ValueError(f"Expected sequence shape {(INPUT_SEQ_LEN, 12, 8)}, got {tuple(seq_raw.shape)}")
    seq_norm = np.zeros((INPUT_SEQ_LEN, 1, 12, 8), dtype=np.float32)
    for i in range(INPUT_SEQ_LEN):
        frame = seq_raw[i]
        mn = float(frame.min())
        mx = float(frame.max())
        if mx - mn > 1e-6:
            frame = (frame - mn) / (mx - mn)
        else:
            frame = frame - mn
        seq_norm[i, 0] = frame
    return seq_norm


def _size_norm_to_cm(size_norm: float) -> float:
    lo = float(min(SIZE_VALUES_CM))
    hi = float(max(SIZE_VALUES_CM))
    return float(lo + np.clip(float(size_norm), 0.0, 1.0) * (hi - lo))


def _compute_runtime_feature_vector(
    seq_raw: np.ndarray,
    seq_norm: np.ndarray,
    selected_features: list,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    frame_rows = [frame_physics_features(frame) for frame in seq_raw]
    records = {
        "runtime_window": {
            "raw_frames": seq_raw.astype(np.float32),
            "norm_frames": seq_norm[:, 0].astype(np.float32),
            "frame_rows": frame_rows,
            "seq_len": int(seq_raw.shape[0]),
        }
    }
    sample = {
        "group_key": "runtime_window",
        "label": 1,
        "size_cm": 1.0,
        "depth_cm": 1.5,
        "size_class_index": 3,
        "depth_coarse_index": 1,
        "center_row": int(seq_raw.shape[0] // 2),
        "end_row": int(seq_raw.shape[0] - 1),
    }
    row = window_feature_row(records, sample)
    feat = np.asarray([float(row[name]) for name in selected_features], dtype=np.float32)
    feat = (feat - feature_mean.astype(np.float32)) / np.maximum(feature_std.astype(np.float32), 1e-6)
    return feat


class TwoStageNoduleInference:
    def __init__(
        self,
        detector_ckpt: Optional[str] = None,
        detector_summary: Optional[str] = None,
        inverter_ckpt: Optional[str] = None,
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        default_stage1_dir = PROJECT_DIR / "experiments" / "outputs_stage1_dualstream_mstcn_detection_raw_delta"
        default_stage2_dir = PROJECT_DIR / "experiments" / "outputs_hierarchical_positive_inverter_run1"

        detector_ckpt = detector_ckpt or os.environ.get(
            "PAPER_GUI_STAGE1_CKPT",
            str(default_stage1_dir / "paper_stage1_dualstream_mstcn_best.pth"),
        )
        detector_summary = detector_summary or os.environ.get(
            "PAPER_GUI_STAGE1_SUMMARY",
            str(default_stage1_dir / "paper_stage1_dualstream_mstcn_summary.json"),
        )
        inverter_ckpt = inverter_ckpt or os.environ.get(
            "PAPER_GUI_STAGE2_CKPT",
            str(default_stage2_dir / "paper_hierarchical_positive_inverter_best.pth"),
        )

        self.detector_ckpt = Path(detector_ckpt)
        self.detector_summary_path = Path(detector_summary)
        self.inverter_ckpt = Path(inverter_ckpt)

        if not self.detector_ckpt.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {self.detector_ckpt}")
        if not self.inverter_ckpt.exists():
            raise FileNotFoundError(f"Inverter checkpoint not found: {self.inverter_ckpt}")

        detector_summary_data = _load_json(self.detector_summary_path) if self.detector_summary_path.exists() else {}
        self.threshold = float(threshold) if threshold is not None else _summary_threshold(detector_summary_data)
        self.inverter_kind = "legacy_multitask"
        self.selected_features = None
        self.feature_mean = None
        self.feature_std = None
        self.raw_scale = 1.0

        self.detector = self._load_detector(self.detector_ckpt)
        self.inverter = self._load_inverter(self.inverter_ckpt)

    def _load_detector(self, ckpt_path: Path) -> DualStreamMSTCNDetector:
        ckpt = _torch_load_compat(str(ckpt_path), map_location=self.device)
        config = ckpt.get("config", {})
        model = DualStreamMSTCNDetector(
            seq_len=int(config.get("seq_len", INPUT_SEQ_LEN)),
            frame_feature_dim=int(config.get("frame_feature_dim", 32)),
            temporal_channels=int(config.get("temporal_channels", 64)),
            temporal_blocks=int(config.get("temporal_blocks", 3)),
            dropout=float(config.get("dropout", 0.35)),
            use_delta_branch=bool(config.get("use_delta_branch", True)),
        ).to(self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_inverter(self, ckpt_path: Path) -> DualStreamMSTCNMultiTask:
        ckpt = _torch_load_compat(str(ckpt_path), map_location=self.device)
        router_name = str(ckpt.get("router_model_name", ckpt.get("model_name", "")))
        if router_name == "HierarchicalPositiveInverter":
            config = ckpt.get("model_config", {})
            model = HierarchicalPositiveInverter(
                seq_len=int(config.get("seq_len", INPUT_SEQ_LEN)),
                frame_feature_dim=int(config.get("frame_feature_dim", 24)),
                temporal_channels=int(config.get("temporal_channels", 48)),
                temporal_blocks=int(config.get("temporal_blocks", 3)),
                dropout=float(config.get("dropout", 0.28)),
                num_size_classes=int(config.get("num_size_classes", len(SIZE_VALUES_CM))),
                num_depth_classes=int(config.get("num_depth_classes", 3)),
                num_tabular_features=int(config.get("num_tabular_features", len(ckpt.get("selected_features", [])))),
                tabular_hidden_dim=int(config.get("tabular_hidden_dim", 64)),
            ).to(self.device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            self.inverter_kind = "hierarchical_positive_inverter"
            self.selected_features = [str(x) for x in ckpt.get("selected_features", [])]
            self.feature_mean = np.asarray(ckpt.get("feature_mean", []), dtype=np.float32)
            self.feature_std = np.asarray(ckpt.get("feature_std", []), dtype=np.float32)
            self.raw_scale = float(ckpt.get("raw_scale", 1.0))
            return model

        config = ckpt.get("config", {})
        model = DualStreamMSTCNMultiTask(
            seq_len=int(config.get("seq_len", INPUT_SEQ_LEN)),
            dropout=float(config.get("dropout", 0.35)),
            use_delta_branch=bool(config.get("use_delta_branch", False)),
            num_size_classes=7,
            num_depth_classes=3,
        ).to(self.device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        self.inverter_kind = "legacy_multitask"
        return model

    def predict_from_frames(self, frames: Iterable[np.ndarray]) -> Dict[str, object]:
        seq_raw = np.stack([_coerce_frame_to_matrix(frame) for frame in frames], axis=0).astype(np.float32)
        seq_norm = _normalize_sequence(seq_raw)
        input_tensor = torch.from_numpy(seq_norm).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            det_logit = self.detector(input_tensor)
            det_prob = float(torch.sigmoid(det_logit).item())

            payload = format_runtime_payload(det_prob=det_prob, threshold=self.threshold)
            size_logits = None
            depth_logits = None

            if payload["gate_open"]:
                if self.inverter_kind == "hierarchical_positive_inverter":
                    raw_input = np.clip(seq_raw / max(float(self.raw_scale), 1e-6), 0.0, 3.0).astype(np.float32)
                    raw_tensor = torch.from_numpy(raw_input[:, None, :, :]).unsqueeze(0).to(self.device)
                    feat_np = _compute_runtime_feature_vector(
                        seq_raw,
                        seq_norm,
                        self.selected_features or [],
                        self.feature_mean if self.feature_mean is not None else np.zeros((0,), dtype=np.float32),
                        self.feature_std if self.feature_std is not None else np.ones((0,), dtype=np.float32),
                    )
                    feat_tensor = torch.from_numpy(feat_np[None, :]).to(self.device)
                    size_logits_tensor, _size_ord_tensor, size_reg_norm_tensor, size_probs_tensor, feats = self.inverter(
                        raw_tensor, input_tensor, feat_tensor, return_features=True
                    )
                    depth_logits_tensor = self.inverter.route_depth_logits(
                        feats["depth_feat"], torch.argmax(size_probs_tensor, dim=1)
                    )
                    size_logits = size_probs_tensor.detach().cpu().numpy()[0]
                    depth_logits = torch.softmax(depth_logits_tensor, dim=1).detach().cpu().numpy()[0]
                    size_reg_cm = _size_norm_to_cm(float(size_reg_norm_tensor.item()))
                else:
                    _det2, size_logits_tensor, size_reg_tensor, depth_logits_tensor = self.inverter(input_tensor)
                    size_logits = torch.softmax(size_logits_tensor, dim=1).detach().cpu().numpy()[0]
                    depth_logits = torch.softmax(depth_logits_tensor, dim=1).detach().cpu().numpy()[0]
                    size_reg_cm = float(size_reg_tensor.item())

                size_index = int(np.argmax(size_logits))
                depth_index = int(np.argmax(depth_logits))
                size_class_cm = float(class_index_to_size(size_index))
                depth_name = coarse_index_to_name(depth_index)
                payload = format_runtime_payload(
                    det_prob=det_prob,
                    threshold=self.threshold,
                    size_class=f"{size_class_cm:g}cm",
                    size_reg_cm=size_reg_cm,
                    depth_coarse=depth_name,
                )
                payload.update(
                    {
                        "size_class_index": size_index,
                        "size_class_cm": size_class_cm,
                        "size_probs": size_logits.tolist(),
                        "depth_coarse_index": depth_index,
                        "depth_coarse_probs": depth_logits.tolist(),
                        "depth_coarse_display": COARSE_DEPTH_DISPLAY.get(depth_name, depth_name),
                        "inverter_kind": self.inverter_kind,
                    }
                )

        latency_ms = (time.perf_counter() - start) * 1000.0
        payload.setdefault("size_class_index", None)
        payload.setdefault("size_class_cm", None)
        payload.setdefault("size_probs", None)
        payload.setdefault("depth_coarse_index", None)
        payload.setdefault("depth_coarse_probs", None)
        payload.setdefault("depth_coarse_display", None)
        payload["latency_ms"] = float(latency_ms)
        payload["ready"] = True
        payload["threshold"] = float(self.threshold)
        return payload
