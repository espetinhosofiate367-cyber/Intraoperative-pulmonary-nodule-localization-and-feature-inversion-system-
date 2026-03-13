from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
CODE_ARCHIVE_DIR = PROJECT_DIR.parent
REPO_ROOT = CODE_ARCHIVE_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(CODE_ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_ARCHIVE_DIR))

from two_stage_inference import TwoStageNoduleInference
from triplet_repeat_classifier.train_triplet_repeat_classifier import read_csv_data

EN_DEPTH_DISPLAY = {
    "shallow": "Shallow (0.5-1.0 cm)",
    "middle": "Middle (1.5-2.0 cm)",
    "deep": "Deep (2.5-3.0 cm)",
}


def find_default_csv() -> Path:
    candidates = list(REPO_ROOT.rglob("3.CSV"))
    for path in candidates:
        text = str(path)
        if "0.75cm" in text and "1.5cm" in text and "clean_data" in text:
            return path
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No replay CSV found.")


def compute_timeline(pipe: TwoStageNoduleInference, frames: np.ndarray, alpha: float = 0.35, low_margin: float = 0.08):
    raw_probs = []
    smooth_probs = []
    gate_states = []
    outputs = []
    smoothed = None
    gate_open = False
    last_payload = None
    threshold_high = float(pipe.threshold)
    threshold_low = max(0.0, threshold_high - float(low_margin))

    for i in range(len(frames)):
        if i >= 9:
            seq = frames[i - 9 : i + 1]
        else:
            pad = np.repeat(frames[[0]], 10 - (i + 1), axis=0)
            seq = np.concatenate([pad, frames[: i + 1]], axis=0)

        out = pipe.predict_from_frames(seq)
        raw_prob = float(out["p_det"])
        smoothed = raw_prob if smoothed is None else alpha * raw_prob + (1.0 - alpha) * smoothed
        if gate_open:
            gate_open = bool(smoothed >= threshold_low)
        else:
            gate_open = bool(smoothed >= threshold_high)
        if bool(out.get("gate_open", False)):
            last_payload = out
        if not gate_open:
            last_payload = None

        raw_probs.append(raw_prob)
        smooth_probs.append(float(smoothed))
        gate_states.append(bool(gate_open))
        outputs.append(last_payload if last_payload is not None else out)

    return {
        "raw_probs": np.array(raw_probs, dtype=np.float32),
        "smooth_probs": np.array(smooth_probs, dtype=np.float32),
        "gate_states": np.array(gate_states, dtype=bool),
        "outputs": outputs,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
    }


def make_snapshot(csv_path: Path, inverter_name: str, inverter_ckpt: str, out_path: Path):
    pipe = TwoStageNoduleInference(inverter_ckpt=inverter_ckpt)
    raw = read_csv_data(str(csv_path)).astype(np.float32)
    frames = raw.reshape(len(raw), 12, 8)
    timeline = compute_timeline(pipe, frames)

    best_idx = int(np.argmax(timeline["smooth_probs"]))
    best_payload = timeline["outputs"][best_idx]
    frame = frames[best_idx]

    left = max(0, best_idx - 2)
    right = min(len(frames), best_idx + 3)
    strip = frames[left:right]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.8, 1.4], height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.25)

    ax_heat = fig.add_subplot(gs[:, 0])
    im = ax_heat.imshow(frame, cmap="turbo", interpolation="bicubic")
    ax_heat.set_title(f"Peak Heatmap\nFrame {best_idx}", fontsize=13)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    ax_prob = fig.add_subplot(gs[0, 1:])
    x = np.arange(len(frames))
    ax_prob.plot(x, timeline["raw_probs"], color="#A0A4AA", linewidth=1.0, alpha=0.65, label="Raw probability")
    ax_prob.plot(x, timeline["smooth_probs"], color="#1565C0", linewidth=2.0, label="Smoothed probability")
    ax_prob.axhline(timeline["threshold_high"], color="#C62828", linestyle="--", alpha=0.75, label=f"Enter {timeline['threshold_high']:.3f}")
    ax_prob.axhline(timeline["threshold_low"], color="#EF6C00", linestyle="--", alpha=0.65, label=f"Exit {timeline['threshold_low']:.3f}")
    ax_prob.axvline(best_idx, color="#2E7D32", linestyle=":", linewidth=1.5)
    ax_prob.plot(best_idx, timeline["smooth_probs"][best_idx], "o", color="#2E7D32", markersize=7)
    ax_prob.set_title(f"Replay Probability Trace ({inverter_name})", fontsize=13)
    ax_prob.set_xlabel("Frame")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.grid(True, alpha=0.25, linestyle="--")
    ax_prob.legend(fontsize=9, loc="upper right")

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    text_lines = [
        "Two-Stage Replay Snapshot",
        f"CSV: {csv_path.name}",
        f"Inverter: {inverter_name}",
        f"Peak smoothed probability: {timeline['smooth_probs'][best_idx]:.2%}",
        f"Peak raw probability: {timeline['raw_probs'][best_idx]:.2%}",
        f"Gate open frames: {int(timeline['gate_states'].sum())} / {len(frames)}",
        f"Predicted size class: {best_payload.get('size_class') or '--'}",
        f"Predicted size reg: {best_payload.get('size_reg_cm') if best_payload.get('size_reg_cm') is not None else '--'}",
        f"Predicted depth: {EN_DEPTH_DISPLAY.get(best_payload.get('depth_coarse'), '--')}",
    ]
    ax_text.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
        family="DejaVu Sans",
    )

    ax_strip = fig.add_subplot(gs[1, 2])
    ax_strip.axis("off")
    ax_strip.set_title("Neighboring Frames", fontsize=12)
    ncols = strip.shape[0]
    for idx in range(ncols):
        inner = ax_strip.inset_axes([idx / max(ncols, 1), 0.02, 0.95 / max(ncols, 1), 0.9])
        inner.imshow(strip[idx], cmap="turbo", interpolation="bicubic")
        inner.set_title(f"{left + idx}", fontsize=9, pad=2)
        inner.set_xticks([])
        inner.set_yticks([])

    fig.suptitle("Flexible Tactile Lung Nodule Localization Replay Demo", fontsize=16, y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_large_best_overview(csv_path: Path, inverter_name: str, inverter_ckpt: str, out_path: Path):
    pipe = TwoStageNoduleInference(inverter_ckpt=inverter_ckpt)
    raw = read_csv_data(str(csv_path)).astype(np.float32)
    frames = raw.reshape(len(raw), 12, 8)
    timeline = compute_timeline(pipe, frames)

    best_idx = int(np.argmax(timeline["smooth_probs"]))
    best_payload = timeline["outputs"][best_idx]
    peak_frame = frames[best_idx]
    baseline_frame = frames[0]
    diff_frame = peak_frame - baseline_frame

    left = max(0, best_idx - 3)
    right = min(len(frames), best_idx + 4)
    strip = frames[left:right]

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.15, 1.15, 1.6, 1.25],
        height_ratios=[0.95, 1.0, 1.0],
        wspace=0.22,
        hspace=0.28,
    )

    ax_curve = fig.add_subplot(gs[0, :])
    x = np.arange(len(frames))
    ax_curve.plot(x, timeline["raw_probs"], color="#B0B5BB", linewidth=1.0, alpha=0.7, label="Raw probability")
    ax_curve.plot(x, timeline["smooth_probs"], color="#0D47A1", linewidth=2.4, label="Smoothed probability")
    ax_curve.axhline(timeline["threshold_high"], color="#C62828", linestyle="--", linewidth=1.5, label=f"Enter threshold {timeline['threshold_high']:.3f}")
    ax_curve.axhline(timeline["threshold_low"], color="#EF6C00", linestyle="--", linewidth=1.3, label=f"Exit threshold {timeline['threshold_low']:.3f}")
    ax_curve.axvline(best_idx, color="#1B5E20", linestyle=":", linewidth=1.6)
    ax_curve.plot(best_idx, timeline["smooth_probs"][best_idx], "o", color="#1B5E20", markersize=8)
    ax_curve.text(
        best_idx,
        min(0.98, timeline["smooth_probs"][best_idx] + 0.06),
        f"Best frame {best_idx}\n{timeline['smooth_probs'][best_idx]:.2%}",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#1B5E20",
        fontweight="bold",
    )
    ax_curve.set_title(f"Best Prediction Overview ({inverter_name})", fontsize=16)
    ax_curve.set_xlabel("Frame")
    ax_curve.set_ylabel("Probability")
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.grid(True, alpha=0.25, linestyle="--")
    ax_curve.legend(ncol=4, fontsize=9, loc="upper right")

    ax_peak = fig.add_subplot(gs[1, 0])
    im_peak = ax_peak.imshow(peak_frame, cmap="turbo", interpolation="bicubic")
    ax_peak.set_title(f"Peak Heatmap\nFrame {best_idx}", fontsize=13)
    ax_peak.set_xticks([])
    ax_peak.set_yticks([])
    plt.colorbar(im_peak, ax=ax_peak, fraction=0.046, pad=0.04)

    ax_base = fig.add_subplot(gs[1, 1])
    im_base = ax_base.imshow(baseline_frame, cmap="turbo", interpolation="bicubic")
    ax_base.set_title("Baseline Heatmap\nFrame 0", fontsize=13)
    ax_base.set_xticks([])
    ax_base.set_yticks([])
    plt.colorbar(im_base, ax=ax_base, fraction=0.046, pad=0.04)

    ax_diff = fig.add_subplot(gs[1, 2])
    vmax = float(np.max(np.abs(diff_frame))) if np.max(np.abs(diff_frame)) > 0 else 1.0
    im_diff = ax_diff.imshow(diff_frame, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="bicubic")
    ax_diff.set_title("Peak - Baseline Difference", fontsize=13)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis("off")
    text_lines = [
        "Inference Summary",
        f"CSV: {csv_path.name}",
        f"Inverter: {inverter_name}",
        f"Total frames: {len(frames)}",
        f"Gate-open frames: {int(timeline['gate_states'].sum())}",
        f"Best frame: {best_idx}",
        f"Peak smoothed prob: {timeline['smooth_probs'][best_idx]:.2%}",
        f"Peak raw prob: {timeline['raw_probs'][best_idx]:.2%}",
        "",
        f"Size class: {best_payload.get('size_class') or '--'}",
        f"Size regression: {best_payload.get('size_reg_cm'):.3f} cm" if best_payload.get('size_reg_cm') is not None else "Size regression: --",
        f"Depth coarse: {EN_DEPTH_DISPLAY.get(best_payload.get('depth_coarse'), '--')}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", fontsize=11, linespacing=1.5)

    strip_title_ax = fig.add_subplot(gs[2, :])
    strip_title_ax.axis("off")
    strip_title_ax.set_title("Neighboring Frames Around Peak", fontsize=14, pad=10)
    ncols = max(strip.shape[0], 1)
    for idx in range(strip.shape[0]):
        inner = strip_title_ax.inset_axes([0.01 + idx * (0.98 / ncols), 0.02, 0.95 / ncols, 0.92])
        inner.imshow(strip[idx], cmap="turbo", interpolation="bicubic")
        frame_id = left + idx
        label = f"{frame_id}"
        if frame_id == best_idx:
            label += " *"
        inner.set_title(label, fontsize=10, pad=4)
        inner.set_xticks([])
        inner.set_yticks([])

    fig.suptitle("Flexible Tactile Lung Nodule Localization System", fontsize=18, y=0.985)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    csv_path = Path(os.environ.get("PAPER_GUI_SNAPSHOT_CSV", str(find_default_csv())))
    output_dir = PROJECT_DIR / "app" / "replay_snapshot_output"
    variants = {
        "hierarchical": PROJECT_DIR / "experiments" / "outputs_hierarchical_positive_inverter_run1" / "paper_hierarchical_positive_inverter_best.pth",
        "raw": PROJECT_DIR / "experiments" / "outputs_stage2_dualstream_mstcn_multitask_raw" / "paper_stage2_dualstream_mstcn_best.pth",
        "raw+delta": PROJECT_DIR / "experiments" / "outputs_stage2_dualstream_mstcn_multitask_raw_delta" / "paper_stage2_dualstream_mstcn_best.pth",
    }
    for name, ckpt in variants.items():
        make_snapshot(csv_path, name, str(ckpt), output_dir / f"replay_snapshot_{name}.png")
        make_large_best_overview(csv_path, name, str(ckpt), output_dir / f"best_prediction_overview_{name}.png")
    print(f"Replay snapshots saved to: {output_dir}")


if __name__ == "__main__":
    main()
