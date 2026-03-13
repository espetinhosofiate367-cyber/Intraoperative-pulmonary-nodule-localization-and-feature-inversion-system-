from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
CODE_ARCHIVE_DIR = PROJECT_DIR.parent
REPO_ROOT = CODE_ARCHIVE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
if str(CODE_ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_ARCHIVE_DIR))

from generate_replay_snapshots import compute_timeline
from task_protocol_v1 import DEPTH_VALUES_CM, SIZE_VALUES_CM
from triplet_repeat_classifier.train_triplet_repeat_classifier import read_csv_data
from two_stage_inference import TwoStageNoduleInference


def _format_size_dir(size_cm: float) -> str:
    return f"{size_cm:g}cm大"


def _format_depth_dir(depth_cm: float) -> str:
    return f"{depth_cm:g}cm深"


def _find_condition_csv(data_root: Path, size_cm: float, depth_cm: float) -> Path:
    size_candidates = [
        data_root / f"{size_cm:g}cm大",
        data_root / f"{size_cm:.1f}cm大",
    ]
    depth_candidates = [
        f"{depth_cm:g}cm深",
        f"{depth_cm:.1f}cm深",
    ]
    for size_dir in size_candidates:
        if not size_dir.exists():
            continue
        for depth_name in depth_candidates:
            csv_path = size_dir / depth_name / "3.CSV"
            if csv_path.exists():
                return csv_path
    return size_candidates[0] / depth_candidates[0] / "3.CSV"


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    mn = float(frame.min())
    mx = float(frame.max())
    if mx - mn > 1e-6:
        return (frame - mn) / (mx - mn)
    return frame - mn


def _load_default_data_root() -> Path:
    summary_path = PROJECT_DIR / "experiments" / "outputs_stage1_dualstream_mstcn_detection_raw_delta" / "paper_stage1_dualstream_mstcn_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        data_root = summary.get("config", {}).get("data_root")
        if data_root:
            return Path(data_root)
    return REPO_ROOT / "整理好的数据集" / "建表数据"


def _variant_to_ckpt(variant: str) -> Path:
    if variant == "hierarchical":
        return PROJECT_DIR / "experiments" / "outputs_hierarchical_positive_inverter_run1" / "paper_hierarchical_positive_inverter_best.pth"
    if variant == "raw+delta":
        return PROJECT_DIR / "experiments" / "outputs_stage2_dualstream_mstcn_multitask_raw_delta" / "paper_stage2_dualstream_mstcn_best.pth"
    return PROJECT_DIR / "experiments" / "outputs_stage2_dualstream_mstcn_multitask_raw" / "paper_stage2_dualstream_mstcn_best.pth"


def build_testset_best_grid(
    data_root: Path,
    inverter_variant: str,
    out_png: Path,
    out_json: Path,
) -> Dict[str, object]:
    inverter_ckpt = _variant_to_ckpt(inverter_variant)
    pipe = TwoStageNoduleInference(inverter_ckpt=str(inverter_ckpt))

    rows = len(DEPTH_VALUES_CM)
    cols = len(SIZE_VALUES_CM)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), facecolor="white")
    if rows == 1:
        axes = np.array([axes])

    records: List[Dict[str, object]] = []
    best_probs = []

    for r, depth_cm in enumerate(DEPTH_VALUES_CM):
        for c, size_cm in enumerate(SIZE_VALUES_CM):
            ax = axes[r, c]
            csv_path = _find_condition_csv(data_root, size_cm, depth_cm)
            if not csv_path.exists():
                ax.axis("off")
                records.append(
                    {
                        "size_cm": float(size_cm),
                        "depth_cm": float(depth_cm),
                        "csv_path": str(csv_path),
                        "exists": False,
                    }
                )
                continue

            raw = read_csv_data(str(csv_path)).astype(np.float32)
            frames = raw.reshape(len(raw), 12, 8)
            timeline = compute_timeline(pipe, frames)
            best_idx = int(np.argmax(timeline["smooth_probs"]))
            best_payload = timeline["outputs"][best_idx]
            best_prob = float(timeline["smooth_probs"][best_idx])
            best_probs.append(best_prob)

            frame = _normalize_frame(frames[best_idx])
            ax.imshow(frame, cmap="turbo", vmin=0.0, vmax=1.0, interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            pred_size = best_payload.get("size_class") if best_payload.get("gate_open", False) else "--"
            pred_depth = best_payload.get("depth_coarse") if best_payload.get("gate_open", False) else "--"
            ax.set_title(
                f"S:{size_cm:g} D:{depth_cm:g}\nP:{best_prob:.2f} | {pred_size} | {pred_depth}",
                fontsize=8,
            )
            records.append(
                {
                    "size_cm": float(size_cm),
                    "depth_cm": float(depth_cm),
                    "csv_path": str(csv_path),
                    "exists": True,
                    "frame_count": int(len(frames)),
                    "best_frame_index": best_idx,
                    "best_probability_smoothed": best_prob,
                    "best_probability_raw": float(timeline["raw_probs"][best_idx]),
                    "gate_open": bool(best_payload.get("gate_open", False)),
                    "pred_size_class": best_payload.get("size_class"),
                    "pred_size_reg_cm": best_payload.get("size_reg_cm"),
                    "pred_depth_coarse": best_payload.get("depth_coarse"),
                    "pred_depth_display": best_payload.get("depth_coarse_display"),
                }
            )

    mean_prob = float(np.mean(best_probs)) if best_probs else 0.0
    med_prob = float(np.median(best_probs)) if best_probs else 0.0
    min_prob = float(np.min(best_probs)) if best_probs else 0.0
    max_prob = float(np.max(best_probs)) if best_probs else 0.0

    fig.suptitle(
        f"Test Set File3 Best-Prediction Frames ({inverter_variant})",
        fontsize=18,
        y=0.995,
    )
    fig.text(
        0.5,
        0.973,
        f"Stage1 detector: raw+delta | Stage2 inverter: {inverter_variant} | "
        f"Mean {mean_prob:.3f} | Median {med_prob:.3f} | Min {min_prob:.3f} | Max {max_prob:.3f}",
        ha="center",
        va="top",
        fontsize=11,
    )
    plt.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=0.02, wspace=0.06, hspace=0.34)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "data_root": str(data_root),
        "stage1_detector": "raw+delta",
        "stage2_inverter": inverter_variant,
        "stage2_ckpt": str(inverter_ckpt),
        "threshold": float(pipe.threshold),
        "grid_shape": [rows, cols],
        "mean_best_probability": mean_prob,
        "median_best_probability": med_prob,
        "min_best_probability": min_prob,
        "max_best_probability": max_prob,
        "records": records,
        "output_png": str(out_png),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    data_root = Path(os.environ.get("PAPER_TESTSET_GRID_DATA_ROOT", str(_load_default_data_root())))
    inverter_variant = os.environ.get("PAPER_TESTSET_GRID_VARIANT", "hierarchical").strip().lower()
    if inverter_variant not in {"hierarchical", "raw", "raw+delta"}:
        inverter_variant = "hierarchical"

    output_dir = PROJECT_DIR / "app" / "replay_snapshot_output"
    out_png = output_dir / f"testset_best_grid_{inverter_variant}.png"
    out_json = output_dir / f"testset_best_grid_{inverter_variant}.json"
    summary = build_testset_best_grid(data_root, inverter_variant, out_png, out_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
