import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
CODE_ARCHIVE_DIR = os.path.dirname(PROJECT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from train_xgboost_baselines import (
    Config,
    balanced_accuracy_from_cm,
    build_records_and_samples_for_file,
    cls_metrics_at_threshold,
    confusion_matrix_counts,
    inverse_frequency_sample_weight,
    subset_multitask_metrics,
)


META_COLS = {
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

TBME_BLUE = "#1f5aa6"
TBME_ORANGE = "#d97a19"
TBME_GREEN = "#2d8b57"
TBME_RED = "#b73a3a"
TBME_PURPLE = "#6b53b3"
TBME_GREY = "#666666"


def apply_tbme_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
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


def feature_family(name: str) -> str:
    if any(key in name for key in ("hotspot_area", "hotspot_radius", "second_moment_spread", "spatial_entropy")):
        return "spread_extent"
    if any(key in name for key in ("center_border_contrast", "center_mean", "border_mean", "anisotropy_ratio", "peak_count")):
        return "shape_contrast"
    if any(key in name for key in ("centroid_row", "centroid_col")):
        return "deformation_position"
    if any(
        key in name
        for key in (
            "rise_time_to_peak",
            "peak_persistence_ratio",
            "decay_after_peak",
            "temporal_raw_sum_slope",
            "temporal_raw_max_slope",
            "window_raw_sum_gain",
            "delta_abs",
            "deltaframe_",
        )
    ):
        return "temporal_phase"
    if any(key in name for key in ("raw_mean", "raw_max", "raw_sum", "raw_p95", "window_raw_global", "window_norm_global")):
        return "amplitude_response"
    return "other"


def feature_concept(name: str) -> str:
    concepts = [
        "center_border_contrast",
        "second_moment_spread",
        "peak_persistence_ratio",
        "rise_time_to_peak",
        "decay_after_peak",
        "temporal_raw_sum_slope",
        "temporal_raw_max_slope",
        "window_raw_sum_gain",
        "hotspot_radius",
        "spatial_entropy",
        "hotspot_area",
        "anisotropy_ratio",
        "center_mean",
        "border_mean",
        "peak_count",
        "centroid_row",
        "centroid_col",
        "raw_mean",
        "raw_max",
        "raw_sum",
        "raw_p95",
        "delta_abs_mean",
        "delta_abs_std",
        "delta_abs_max",
        "window_raw_global_mean",
        "window_raw_global_std",
        "window_raw_global_max",
        "window_raw_global_p95",
        "window_norm_global_mean",
        "window_norm_global_std",
    ]
    for concept in concepts:
        if concept in name:
            return concept
    return name


def load_feature_table(path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df, feature_cols


def build_models(cfg: Config):
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
    return det_model, size_cls_model, size_reg_model, depth_model


def fit_models_from_feature_table(df: pd.DataFrame, feature_cols: List[str], cfg: Config):
    train_det = df[df["split"] == "train_det"].reset_index(drop=True)
    train_all = df[df["split"] == "train_all"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train_det = train_det[feature_cols]
    y_train_det = train_det["label"].to_numpy(dtype=np.int32)
    X_val = val_df[feature_cols]
    y_val_det = val_df["label"].to_numpy(dtype=np.int32)

    pos_train = train_all[train_all["label"] == 1].reset_index(drop=True)
    pos_val = val_df[val_df["label"] == 1].reset_index(drop=True)
    pos_test = test_df[test_df["label"] == 1].reset_index(drop=True)

    X_train_pos = pos_train[feature_cols]
    y_train_size_cls = pos_train["size_class_index"].to_numpy(dtype=np.int32)
    y_train_size_reg = pos_train["size_cm"].to_numpy(dtype=np.float32)
    y_train_depth = pos_train["depth_coarse_index"].to_numpy(dtype=np.int32)

    X_val_pos = pos_val[feature_cols]
    y_val_size_cls = pos_val["size_class_index"].to_numpy(dtype=np.int32)
    y_val_size_reg = pos_val["size_cm"].to_numpy(dtype=np.float32)
    y_val_depth = pos_val["depth_coarse_index"].to_numpy(dtype=np.int32)

    det_model, size_cls_model, size_reg_model, depth_model = build_models(cfg)
    det_model.fit(
        X_train_det,
        y_train_det,
        sample_weight=inverse_frequency_sample_weight(y_train_det),
        eval_set=[(X_train_det, y_train_det), (X_val, y_val_det)],
        verbose=False,
    )
    size_cls_model.fit(
        X_train_pos,
        y_train_size_cls,
        sample_weight=inverse_frequency_sample_weight(y_train_size_cls),
        eval_set=[(X_train_pos, y_train_size_cls), (X_val_pos, y_val_size_cls)],
        verbose=False,
    )
    size_reg_model.fit(
        X_train_pos,
        y_train_size_reg,
        eval_set=[(X_train_pos, y_train_size_reg), (X_val_pos, y_val_size_reg)],
        verbose=False,
    )
    depth_model.fit(
        X_train_pos,
        y_train_depth,
        sample_weight=inverse_frequency_sample_weight(y_train_depth),
        eval_set=[(X_train_pos, y_train_depth), (X_val_pos, y_val_depth)],
        verbose=False,
    )

    return {
        "det_model": det_model,
        "size_cls_model": size_cls_model,
        "size_reg_model": size_reg_model,
        "depth_model": depth_model,
        "train_det": train_det,
        "train_all": train_all,
        "val_df": val_df,
        "test_df": test_df,
        "pos_train": pos_train,
        "pos_val": pos_val,
        "pos_test": pos_test,
        "feature_cols": feature_cols,
    }


def get_best_iteration(model) -> int:
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        return 0
    return int(max(best_iter, 0))


def predict_contribs(model, X: pd.DataFrame) -> np.ndarray:
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    booster = model.get_booster()
    best_iter = get_best_iteration(model)
    kwargs = {
        "pred_contribs": True,
        "validate_features": False,
    }
    if best_iter > 0:
        kwargs["iteration_range"] = (0, best_iter + 1)
    return booster.predict(dmat, **kwargs)


def feature_gain_importance(model, feature_names: Sequence[str]) -> pd.Series:
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    values = {name: float(gain_dict.get(name, 0.0)) for name in feature_names}
    return pd.Series(values, name="gain_importance", dtype=np.float64)


def contrib_dataframe(feature_names: Sequence[str], contrib: np.ndarray, gain: pd.Series, task_name: str, class_name: str = "overall") -> pd.DataFrame:
    if contrib.ndim == 2:
        values = np.mean(np.abs(contrib[:, :-1]), axis=0)
    else:
        values = np.mean(np.abs(contrib[:, :, :-1]), axis=(0, 1))
    df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_contrib": values,
            "gain_importance": [float(gain.get(name, 0.0)) for name in feature_names],
        }
    )
    df["family"] = df["feature"].map(feature_family)
    df["concept"] = df["feature"].map(feature_concept)
    df["task"] = task_name
    df["class_name"] = class_name
    return df.sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)


def classwise_contrib_dataframes(feature_names: Sequence[str], contrib: np.ndarray, gain: pd.Series, task_name: str, class_names: Sequence[str]):
    outputs = {}
    for class_idx, class_name in enumerate(class_names):
        class_values = np.mean(np.abs(contrib[:, class_idx, :-1]), axis=0)
        df = pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_abs_contrib": class_values,
                "gain_importance": [float(gain.get(name, 0.0)) for name in feature_names],
            }
        )
        df["family"] = df["feature"].map(feature_family)
        df["concept"] = df["feature"].map(feature_concept)
        df["task"] = task_name
        df["class_name"] = class_name
        outputs[class_name] = df.sort_values("mean_abs_contrib", ascending=False).reset_index(drop=True)
    return outputs


def aggregate_importance(df: pd.DataFrame, key: str) -> pd.DataFrame:
    out = df.groupby(key, as_index=False)["mean_abs_contrib"].sum().sort_values("mean_abs_contrib", ascending=False)
    total = float(out["mean_abs_contrib"].sum())
    out["share"] = out["mean_abs_contrib"] / max(total, 1e-12)
    return out


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: str, top_n: int = 12) -> None:
    top = df.head(int(top_n)).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.42 * len(top) + 1)))
    plt.barh(top[y_col], top[x_col], color="#2f6db0")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str, output_path: str) -> None:
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(np.arange(len(labels)), labels, rotation=20)
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


def plot_local_bar(ax, feature_names: Sequence[str], contrib_values: np.ndarray, title: str, top_k: int = 8):
    order = np.argsort(np.abs(contrib_values))[-int(top_k):]
    vals = contrib_values[order]
    names = [feature_names[idx] for idx in order]
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in vals]
    ax.barh(range(len(order)), vals, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)


def plot_depth_local_explanations(pos_test: pd.DataFrame, depth_probs: np.ndarray, depth_contribs: np.ndarray, feature_names: Sequence[str], output_path: str) -> List[Dict[str, object]]:
    y_true = pos_test["depth_coarse_index"].to_numpy(dtype=np.int32)
    y_pred = np.argmax(depth_probs, axis=1).astype(np.int32)
    labels = ["shallow", "middle", "deep"]
    example_indices = []
    metadata = []
    for class_idx, class_name in enumerate(labels):
        mask = (y_true == class_idx) & (y_pred == class_idx)
        if np.any(mask):
            local_idx = np.argmax(depth_probs[mask, class_idx])
            global_idx = np.where(mask)[0][local_idx]
            example_indices.append(global_idx)
            metadata.append({"kind": "correct", "focus_class": class_name})
    wrong_mask = y_true != y_pred
    if np.any(wrong_mask):
        confidence = np.max(depth_probs[wrong_mask], axis=1)
        local_idx = np.argmax(confidence)
        global_idx = np.where(wrong_mask)[0][local_idx]
        example_indices.append(global_idx)
        metadata.append({"kind": "misclassified", "focus_class": labels[int(y_pred[global_idx])]})

    if not example_indices:
        return []

    fig, axes = plt.subplots(len(example_indices), 1, figsize=(10, max(4.0, 3.5 * len(example_indices))))
    if len(example_indices) == 1:
        axes = [axes]
    selected = []
    for ax, idx, meta in zip(axes, example_indices, metadata):
        pred_cls = int(y_pred[idx])
        true_cls = int(y_true[idx])
        contrib = depth_contribs[idx, pred_cls, :-1]
        row = pos_test.iloc[int(idx)]
        title = (
            f"{meta['kind']} | true={labels[true_cls]} pred={labels[pred_cls]} | "
            f"S={float(row['size_cm']):.2f}cm D={float(row['depth_cm']):.1f}cm | P={float(np.max(depth_probs[idx])):.3f}"
        )
        plot_local_bar(ax, feature_names, contrib, title)
        selected.append(
            {
                "kind": meta["kind"],
                "true_depth": labels[true_cls],
                "pred_depth": labels[pred_cls],
                "size_text": str(row["size_text"]),
                "depth_text": str(row["depth_text"]),
                "group_key": str(row["group_key"]),
                "center_row": int(row["center_row"]),
                "confidence": float(np.max(depth_probs[idx])),
            }
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return selected


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
    raw_window = np.asarray(raw_window, dtype=np.float32)
    scores = []
    frames = []
    for frame in raw_window:
        disp = normalize_frame_for_display(frame)
        center = float(disp[3:9, 2:6].mean())
        edge = float(
            np.concatenate(
                [
                    disp[:2, :].reshape(-1),
                    disp[-2:, :].reshape(-1),
                    disp[2:-2, :2].reshape(-1),
                    disp[2:-2, -2:].reshape(-1),
                ]
            ).mean()
        )
        hot_area = float((disp >= 0.82).mean())
        score = center - 0.6 * edge - 0.4 * abs(hot_area - 0.18)
        scores.append(score)
        frames.append(disp)
    return frames[int(np.argmax(scores))]


def top_feature_order_for_heatmap(depth_global: pd.DataFrame, top_n: int = 12) -> List[str]:
    return depth_global.sort_values("mean_abs_contrib", ascending=False)["feature"].head(top_n).tolist()


def plot_classwise_concept_heatmap(depth_classwise: Dict[str, pd.DataFrame], depth_global: pd.DataFrame, output_path: str, top_n: int = 12) -> pd.DataFrame:
    feature_order = top_feature_order_for_heatmap(depth_global, top_n=top_n)
    label_order = ["shallow", "middle", "deep"]
    matrix_rows = []
    for label in label_order:
        df = depth_classwise[label].set_index("feature")
        for feat in feature_order:
            matrix_rows.append(
                {
                    "depth_label": label,
                    "feature": feat,
                    "concept": str(df.loc[feat, "concept"]) if feat in df.index else feat,
                    "mean_abs_contrib": float(df.loc[feat, "mean_abs_contrib"]) if feat in df.index else 0.0,
                }
            )
    heat_df = pd.DataFrame(matrix_rows)
    pivot = heat_df.pivot(index="depth_label", columns="feature", values="mean_abs_contrib").reindex(index=label_order)
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    im = ax.imshow(pivot.to_numpy(dtype=np.float32), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(label_order)))
    ax.set_yticklabels(label_order)
    ax.set_title("Classwise depth contribution heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("mean |contribution|", rotation=90)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return heat_df


def plot_depth_family_share_tbme(depth_family: pd.DataFrame, output_path: str) -> None:
    use = depth_family.sort_values("share", ascending=True)
    family_colors = {
        "deformation_position": TBME_BLUE,
        "shape_contrast": TBME_ORANGE,
        "spread_extent": TBME_GREEN,
        "amplitude_response": TBME_RED,
        "temporal_phase": TBME_PURPLE,
        "other": TBME_GREY,
    }
    colors = [family_colors.get(name, TBME_GREY) for name in use["family"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.barh(use["family"], use["share"], color=colors)
    for idx, row in enumerate(use.itertuples(index=False)):
        ax.text(float(row.share) + 0.004, idx, f"{row.share * 100:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0.0, max(0.36, float(use["share"].max()) + 0.05))
    ax.set_xlabel("Contribution share")
    ax.set_title("Depth cue families used by XGBoost")
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_test_records_for_visualization(cfg: Config) -> Tuple[Dict[str, dict], Dict[Tuple[str, int], dict]]:
    file1_all = json.load(open(cfg.file1_labels, "r", encoding="utf-8"))
    file2_all = json.load(open(cfg.file2_labels, "r", encoding="utf-8"))
    file3_all = json.load(open(cfg.file3_labels, "r", encoding="utf-8"))
    rec1, _ = build_records_and_samples_for_file(file1_all, "1.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec2, _ = build_records_and_samples_for_file(file2_all, "2.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    rec3, samples3 = build_records_and_samples_for_file(file3_all, "3.CSV", cfg.data_root, cfg.seq_len, cfg.stride, cfg.dedup_gap)
    common_base_groups = (
        set(v["base_group"] for v in rec1.values())
        & set(v["base_group"] for v in rec2.values())
        & set(v["base_group"] for v in rec3.values())
    )
    test_records = {k: v for k, v in rec3.items() if v["base_group"] in common_base_groups}
    lookup = {}
    for sample in samples3:
        if sample["base_group"] not in common_base_groups:
            continue
        lookup[(sample["group_key"], int(sample["end_row"]))] = sample
    return test_records, lookup


def get_window_from_sample(records_by_key: Dict[str, dict], row: pd.Series) -> np.ndarray:
    rec = records_by_key[str(row["group_key"])]
    end_row = int(row["end_row"])
    seq_len = int(rec["seq_len"])
    start_row = end_row - seq_len + 1
    return np.asarray(rec["raw_frames"][start_row : end_row + 1], dtype=np.float32)


def select_depth_representative_indices(pos_test: pd.DataFrame, depth_probs: np.ndarray) -> List[int]:
    y_true = pos_test["depth_coarse_index"].to_numpy(dtype=np.int32)
    y_pred = np.argmax(depth_probs, axis=1).astype(np.int32)
    labels = [0, 1, 2]
    chosen = []
    used = set()
    for class_idx in labels:
        mask = (y_true == class_idx) & (y_pred == class_idx)
        if np.any(mask):
            local_idx = int(np.argmax(depth_probs[mask, class_idx]))
            global_idx = int(np.where(mask)[0][local_idx])
            chosen.append(global_idx)
            used.add(global_idx)
    wrong_mask = y_true != y_pred
    if np.any(wrong_mask):
        confidence = np.max(depth_probs[wrong_mask], axis=1)
        local_idx = int(np.argmax(confidence))
        global_idx = int(np.where(wrong_mask)[0][local_idx])
        if global_idx not in used:
            chosen.append(global_idx)
    return chosen


def plot_depth_representative_cases_tbme(
    pos_test: pd.DataFrame,
    depth_probs: np.ndarray,
    depth_contribs: np.ndarray,
    feature_names: Sequence[str],
    records_by_key: Dict[str, dict],
    output_path: str,
    top_k: int = 7,
) -> List[Dict[str, object]]:
    example_indices = select_depth_representative_indices(pos_test, depth_probs)
    if not example_indices:
        return []
    labels = ["shallow", "middle", "deep"]
    fig = plt.figure(figsize=(10.5, max(5.2, 2.5 * len(example_indices))))
    gs = GridSpec(len(example_indices), 2, width_ratios=[1.05, 1.7], hspace=0.55, wspace=0.28)
    selected = []
    for row_idx, idx in enumerate(example_indices):
        row = pos_test.iloc[int(idx)]
        true_cls = int(row["depth_coarse_index"])
        pred_cls = int(np.argmax(depth_probs[idx]))
        disp = choose_display_frame(get_window_from_sample(records_by_key, row))
        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.imshow(disp, cmap="turbo", interpolation="bicubic", vmin=0.0, vmax=1.0)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(
            f"{'correct' if true_cls == pred_cls else 'error'} | "
            f"S={float(row['size_cm']):.2f} D={float(row['depth_cm']):.1f}\n"
            f"true={labels[true_cls]} pred={labels[pred_cls]} P={float(np.max(depth_probs[idx])):.2f}",
            fontsize=9,
        )
        contrib = depth_contribs[idx, pred_cls, :-1]
        order = np.argsort(np.abs(contrib))[-int(top_k):]
        vals = contrib[order]
        names = [feature_names[j] for j in order]
        colors = [TBME_RED if v > 0 else TBME_BLUE for v in vals]
        ax_bar = fig.add_subplot(gs[row_idx, 1])
        ax_bar.barh(np.arange(len(order)), vals, color=colors)
        ax_bar.axvline(0.0, color="#222222", linewidth=0.8)
        ax_bar.set_yticks(np.arange(len(order)))
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_title("Local additive contributions", fontsize=9)
        ax_bar.grid(axis="x")
        selected.append(
            {
                "group_key": str(row["group_key"]),
                "true_depth": labels[true_cls],
                "pred_depth": labels[pred_cls],
                "size_cm": float(row["size_cm"]),
                "depth_cm": float(row["depth_cm"]),
                "confidence": float(np.max(depth_probs[idx])),
            }
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return selected


def plot_depth_tbme_summary(
    depth_family: pd.DataFrame,
    depth_classwise: Dict[str, pd.DataFrame],
    depth_global: pd.DataFrame,
    depth_cm: np.ndarray,
    depth_model,
    X_ref: pd.DataFrame,
    output_path: str,
) -> None:
    feature_order = top_feature_order_for_heatmap(depth_global, top_n=12)
    label_order = ["shallow", "middle", "deep"]
    heat = []
    for label in label_order:
        df = depth_classwise[label].set_index("feature")
        heat.append([float(df.loc[f, "mean_abs_contrib"]) if f in df.index else 0.0 for f in feature_order])
    heat = np.asarray(heat, dtype=np.float32)

    pdp_features = depth_global["feature"].head(4).tolist()
    fig = plt.figure(figsize=(11.2, 8.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.15], height_ratios=[0.95, 1.05], hspace=0.36, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    fam_use = depth_family.sort_values("share", ascending=True)
    fam_colors = {
        "deformation_position": TBME_BLUE,
        "shape_contrast": TBME_ORANGE,
        "spread_extent": TBME_GREEN,
        "amplitude_response": TBME_RED,
        "temporal_phase": TBME_PURPLE,
        "other": TBME_GREY,
    }
    ax_a.barh(fam_use["family"], fam_use["share"], color=[fam_colors.get(x, TBME_GREY) for x in fam_use["family"]])
    ax_a.set_title("A  Feature-family contribution share", loc="left")
    ax_a.set_xlabel("Share")
    ax_a.grid(axis="x")

    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(heat, cmap="YlOrRd", aspect="auto")
    ax_b.set_title("B  Classwise contribution reshape", loc="left")
    ax_b.set_xticks(np.arange(len(feature_order)))
    ax_b.set_xticklabels(feature_order, rotation=30, ha="right")
    ax_b.set_yticks(np.arange(len(label_order)))
    ax_b.set_yticklabels(label_order)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.04, pad=0.02)
    cbar.ax.set_ylabel("mean |contribution|", rotation=90)

    cm_spec = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], width_ratios=[1.0, 0.06], wspace=0.08)
    ax_c = fig.add_subplot(cm_spec[0, 0])
    im2 = ax_c.imshow(depth_cm, cmap="Blues")
    ax_c.set_title("C  Depth confusion matrix", loc="left")
    ax_c.set_xticks(np.arange(3))
    ax_c.set_xticklabels(label_order)
    ax_c.set_yticks(np.arange(3))
    ax_c.set_yticklabels(label_order)
    ax_c.set_xlabel("Predicted")
    ax_c.set_ylabel("True")
    for i in range(depth_cm.shape[0]):
        for j in range(depth_cm.shape[1]):
            ax_c.text(j, i, int(depth_cm[i, j]), ha="center", va="center", fontsize=8)
    ax_cb = fig.add_subplot(cm_spec[0, 1])
    fig.colorbar(im2, cax=ax_cb)

    pdp_spec = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 1], hspace=0.42, wspace=0.30)
    colors = [TBME_BLUE, TBME_GREEN, TBME_RED]
    for idx, feat in enumerate(pdp_features):
        ax = fig.add_subplot(pdp_spec[idx // 2, idx % 2])
        grid, curves = partial_dependence_multiclass(depth_model, X_ref, feat)
        for class_idx, class_name in enumerate(label_order):
            ax.plot(grid, curves[:, class_idx], label=class_name, color=colors[class_idx], linewidth=1.5)
        ax.set_title(feat, fontsize=8.5)
        ax.grid(axis="both")
        if idx // 2 == 1:
            ax.set_xlabel(feat, fontsize=8)
        if idx % 2 == 0:
            ax.set_ylabel("Class prob.", fontsize=8)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7, loc="best")
    big_ax = fig.add_subplot(gs[1, 1], frame_on=False)
    big_ax.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    big_ax.set_title("D  Partial dependence of top depth cues", loc="left", pad=10)

    fig.suptitle("TBME-style XGBoost depth explainability summary", fontsize=13, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def partial_dependence_binary(model, X_ref: pd.DataFrame, feature_name: str, num_points: int = 40):
    values = X_ref[feature_name].to_numpy(dtype=np.float32)
    q_low, q_high = np.percentile(values, [2, 98])
    grid = np.linspace(q_low, q_high, int(num_points), dtype=np.float32)
    curves = []
    for value in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature_name] = value
        curves.append(model.predict_proba(X_tmp)[:, 1].mean())
    return grid, np.asarray(curves, dtype=np.float32)


def partial_dependence_regression(model, X_ref: pd.DataFrame, feature_name: str, num_points: int = 40):
    values = X_ref[feature_name].to_numpy(dtype=np.float32)
    q_low, q_high = np.percentile(values, [2, 98])
    grid = np.linspace(q_low, q_high, int(num_points), dtype=np.float32)
    curves = []
    for value in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature_name] = value
        curves.append(model.predict(X_tmp).mean())
    return grid, np.asarray(curves, dtype=np.float32)


def partial_dependence_multiclass(model, X_ref: pd.DataFrame, feature_name: str, num_points: int = 40):
    values = X_ref[feature_name].to_numpy(dtype=np.float32)
    q_low, q_high = np.percentile(values, [2, 98])
    grid = np.linspace(q_low, q_high, int(num_points), dtype=np.float32)
    curves = []
    for value in grid:
        X_tmp = X_ref.copy()
        X_tmp[feature_name] = value
        curves.append(model.predict_proba(X_tmp).mean(axis=0))
    return grid, np.asarray(curves, dtype=np.float32)


def plot_detection_partial_dependence(model, X_ref: pd.DataFrame, feature_names: Sequence[str], output_path: str) -> None:
    top = list(feature_names)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, top):
        grid, curve = partial_dependence_binary(model, X_ref, name)
        ax.plot(grid, curve, color="#2f6db0")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("mean P(det=1)")
        ax.grid(alpha=0.25)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_size_partial_dependence(model, X_ref: pd.DataFrame, feature_names: Sequence[str], output_path: str) -> None:
    top = list(feature_names)[:4]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, top):
        grid, curve = partial_dependence_regression(model, X_ref, name)
        ax.plot(grid, curve, color="#3c8d2f")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("mean predicted size (cm)")
        ax.grid(alpha=0.25)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_depth_partial_dependence(model, X_ref: pd.DataFrame, feature_names: Sequence[str], output_path: str) -> None:
    top = list(feature_names)[:4]
    labels = ["shallow", "middle", "deep"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, top):
        grid, curves = partial_dependence_multiclass(model, X_ref, name)
        for class_idx, class_name in enumerate(labels):
            ax.plot(grid, curves[:, class_idx], label=class_name, color=colors[class_idx])
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("mean class probability")
        ax.grid(alpha=0.25)
    for ax in axes[len(top):]:
        ax.axis("off")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def model_summary_lines(task_name: str, df: pd.DataFrame, family_df: pd.DataFrame, top_n: int = 8) -> List[str]:
    top_features = ", ".join(df["feature"].head(top_n).tolist())
    top_families = ", ".join([f"{row[0]} ({row[1] * 100:.1f}%)" for row in family_df.head(4)[["family", "share"]].to_numpy()])
    return [
        f"### {task_name}",
        f"- Top features by mean |contribution|: {top_features}",
        f"- Dominant concept families: {top_families}",
        "",
    ]


def main() -> None:
    apply_tbme_style()
    cfg = Config()
    base_output_dir = os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_baselines_v1")
    explain_dir = os.environ.get(
        "PAPERXGB_EXPLAIN_OUTPUT_DIR",
        os.path.join(PROJECT_DIR, "experiments", "outputs_xgboost_explainability_v1"),
    )
    os.makedirs(explain_dir, exist_ok=True)
    os.makedirs(os.path.join(explain_dir, "models"), exist_ok=True)

    feature_table_path = os.path.join(base_output_dir, "xgboost_window_feature_table.csv")
    df, feature_cols = load_feature_table(feature_table_path)
    artifacts = fit_models_from_feature_table(df, feature_cols, cfg)

    det_model = artifacts["det_model"]
    size_cls_model = artifacts["size_cls_model"]
    size_reg_model = artifacts["size_reg_model"]
    depth_model = artifacts["depth_model"]
    val_df = artifacts["val_df"]
    test_df = artifacts["test_df"]
    pos_test = artifacts["pos_test"]

    det_model.save_model(os.path.join(explain_dir, "models", "detection_xgb.json"))
    size_cls_model.save_model(os.path.join(explain_dir, "models", "size_classification_xgb.json"))
    size_reg_model.save_model(os.path.join(explain_dir, "models", "size_regression_xgb.json"))
    depth_model.save_model(os.path.join(explain_dir, "models", "depth_coarse_xgb.json"))

    val_det_scores = det_model.predict_proba(val_df[feature_cols])[:, 1]
    test_det_scores = det_model.predict_proba(test_df[feature_cols])[:, 1]
    thresholds = np.linspace(0.0, 1.0, 1001)
    threshold_metrics = [cls_metrics_at_threshold(val_df["label"].to_numpy(dtype=np.int32), val_det_scores, thr) for thr in thresholds]
    det_threshold = float(max(threshold_metrics, key=lambda row: row["f1"])["threshold"])
    det_test_metrics = cls_metrics_at_threshold(test_df["label"].to_numpy(dtype=np.int32), test_det_scores, det_threshold)

    size_probs_test = size_cls_model.predict_proba(pos_test[feature_cols])
    size_reg_test = size_reg_model.predict(pos_test[feature_cols])
    depth_probs_test = depth_model.predict_proba(pos_test[feature_cols])
    gt_metrics = subset_multitask_metrics(
        size_probs_test,
        size_reg_test,
        depth_probs_test,
        pos_test["size_class_index"].to_numpy(dtype=np.int32),
        pos_test["size_cm"].to_numpy(dtype=np.float32),
        pos_test["depth_coarse_index"].to_numpy(dtype=np.int32),
        np.ones(len(pos_test), dtype=bool),
    )

    det_contrib = predict_contribs(det_model, test_df[feature_cols])
    size_cls_contrib = predict_contribs(size_cls_model, pos_test[feature_cols])
    size_reg_contrib = predict_contribs(size_reg_model, pos_test[feature_cols])
    depth_contrib = predict_contribs(depth_model, pos_test[feature_cols])

    det_gain = feature_gain_importance(det_model, feature_cols)
    size_cls_gain = feature_gain_importance(size_cls_model, feature_cols)
    size_reg_gain = feature_gain_importance(size_reg_model, feature_cols)
    depth_gain = feature_gain_importance(depth_model, feature_cols)

    det_global = contrib_dataframe(feature_cols, det_contrib, det_gain, "detection")
    size_global = contrib_dataframe(feature_cols, size_cls_contrib, size_cls_gain, "size_classification")
    size_reg_global = contrib_dataframe(feature_cols, size_reg_contrib, size_reg_gain, "size_regression")
    depth_global = contrib_dataframe(feature_cols, depth_contrib, depth_gain, "depth_coarse")
    depth_classwise = classwise_contrib_dataframes(feature_cols, depth_contrib, depth_gain, "depth_coarse", ["shallow", "middle", "deep"])

    det_family = aggregate_importance(det_global, "family")
    size_family = aggregate_importance(size_global, "family")
    size_reg_family = aggregate_importance(size_reg_global, "family")
    depth_family = aggregate_importance(depth_global, "family")
    depth_concept = aggregate_importance(depth_global, "concept")

    det_global.to_csv(os.path.join(explain_dir, "detection_global_contrib.csv"), index=False, encoding="utf-8-sig")
    size_global.to_csv(os.path.join(explain_dir, "size_classification_global_contrib.csv"), index=False, encoding="utf-8-sig")
    size_reg_global.to_csv(os.path.join(explain_dir, "size_regression_global_contrib.csv"), index=False, encoding="utf-8-sig")
    depth_global.to_csv(os.path.join(explain_dir, "depth_global_contrib.csv"), index=False, encoding="utf-8-sig")
    det_family.to_csv(os.path.join(explain_dir, "detection_family_contrib.csv"), index=False, encoding="utf-8-sig")
    size_family.to_csv(os.path.join(explain_dir, "size_classification_family_contrib.csv"), index=False, encoding="utf-8-sig")
    size_reg_family.to_csv(os.path.join(explain_dir, "size_regression_family_contrib.csv"), index=False, encoding="utf-8-sig")
    depth_family.to_csv(os.path.join(explain_dir, "depth_family_contrib.csv"), index=False, encoding="utf-8-sig")
    depth_concept.to_csv(os.path.join(explain_dir, "depth_concept_contrib.csv"), index=False, encoding="utf-8-sig")
    for class_name, class_df in depth_classwise.items():
        class_df.to_csv(os.path.join(explain_dir, f"depth_global_contrib_{class_name}.csv"), index=False, encoding="utf-8-sig")

    plot_bar(det_global, "mean_abs_contrib", "feature", "Detection: mean |contribution|", os.path.join(explain_dir, "detection_top_contrib.png"), top_n=18)
    plot_bar(size_global, "mean_abs_contrib", "feature", "Size Classification: mean |contribution|", os.path.join(explain_dir, "size_top_contrib.png"), top_n=18)
    plot_bar(size_reg_global, "mean_abs_contrib", "feature", "Size Regression: mean |contribution|", os.path.join(explain_dir, "size_regression_top_contrib.png"), top_n=18)
    plot_bar(depth_global, "mean_abs_contrib", "feature", "Depth Coarse: mean |contribution|", os.path.join(explain_dir, "depth_top_contrib.png"), top_n=18)
    plot_bar(det_family, "share", "family", "Detection: concept-family contribution share", os.path.join(explain_dir, "detection_family_share.png"), top_n=8)
    plot_bar(size_family, "share", "family", "Size Classification: concept-family contribution share", os.path.join(explain_dir, "size_family_share.png"), top_n=8)
    plot_bar(size_reg_family, "share", "family", "Size Regression: concept-family contribution share", os.path.join(explain_dir, "size_regression_family_share.png"), top_n=8)
    plot_bar(depth_family, "share", "family", "Depth Coarse: concept-family contribution share", os.path.join(explain_dir, "depth_family_share.png"), top_n=8)
    plot_bar(depth_concept, "share", "concept", "Depth Coarse: aggregated concept contribution share", os.path.join(explain_dir, "depth_concept_share.png"), top_n=16)

    depth_true = pos_test["depth_coarse_index"].to_numpy(dtype=np.int32)
    depth_pred = np.argmax(depth_probs_test, axis=1).astype(np.int32)
    depth_cm = confusion_matrix_counts(depth_true, depth_pred, num_classes=3)
    plot_confusion_matrix(depth_cm, ["shallow", "middle", "deep"], "Depth coarse confusion matrix", os.path.join(explain_dir, "depth_confusion_matrix.png"))

    size_true = pos_test["size_class_index"].to_numpy(dtype=np.int32)
    size_pred = np.argmax(size_probs_test, axis=1).astype(np.int32)
    size_cm = confusion_matrix_counts(size_true, size_pred, num_classes=7)
    plot_confusion_matrix(
        size_cm,
        ["0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "1.75"],
        "Size classification confusion matrix",
        os.path.join(explain_dir, "size_confusion_matrix.png"),
    )

    plot_detection_partial_dependence(det_model, test_df[feature_cols], det_global["feature"].head(4).tolist(), os.path.join(explain_dir, "detection_partial_dependence_top4.png"))
    plot_size_partial_dependence(size_reg_model, pos_test[feature_cols], size_reg_global["feature"].head(4).tolist(), os.path.join(explain_dir, "size_regression_partial_dependence_top4.png"))
    plot_depth_partial_dependence(depth_model, pos_test[feature_cols], depth_global["feature"].head(4).tolist(), os.path.join(explain_dir, "depth_partial_dependence_top4.png"))

    local_examples = plot_depth_local_explanations(pos_test, depth_probs_test, depth_contrib, feature_cols, os.path.join(explain_dir, "depth_local_explanations.png"))

    test_records_by_key, _ = build_test_records_for_visualization(cfg)
    depth_heat_df = plot_classwise_concept_heatmap(
        depth_classwise,
        depth_global,
        os.path.join(explain_dir, "depth_classwise_concept_heatmap_tbme.png"),
        top_n=12,
    )
    plot_depth_family_share_tbme(depth_family, os.path.join(explain_dir, "depth_family_share_tbme.png"))
    tbme_local_examples = plot_depth_representative_cases_tbme(
        pos_test,
        depth_probs_test,
        depth_contrib,
        feature_cols,
        test_records_by_key,
        os.path.join(explain_dir, "depth_representative_cases_tbme.png"),
    )
    plot_depth_tbme_summary(
        depth_family,
        depth_classwise,
        depth_global,
        depth_cm,
        depth_model,
        pos_test[feature_cols],
        os.path.join(explain_dir, "Fig_XGB_depth_explainability_tbme.png"),
    )

    summary = {
        "source_feature_table": feature_table_path,
        "retrained_with_same_protocol": True,
        "detection_threshold": det_threshold,
        "detection_test_metrics": det_test_metrics,
        "gt_positive_test_metrics": gt_metrics,
        "depth_balanced_accuracy": float(balanced_accuracy_from_cm(depth_cm)),
        "top_detection_features": det_global["feature"].head(12).tolist(),
        "top_size_features": size_global["feature"].head(12).tolist(),
        "top_size_regression_features": size_reg_global["feature"].head(12).tolist(),
        "top_depth_features": depth_global["feature"].head(12).tolist(),
        "top_depth_families": depth_family.head(6).to_dict(orient="records"),
        "top_depth_concepts": depth_concept.head(10).to_dict(orient="records"),
        "depth_local_examples": local_examples,
        "depth_local_examples_tbme": tbme_local_examples,
    }
    with open(os.path.join(explain_dir, "xgboost_explainability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "# XGBoost Explainability Report",
        "",
        "## 1. Baseline definition",
        "- The XGBoost baseline uses only handcrafted window-level tactile physics features.",
        "- Detection is trained on `train_det` windows; size and depth are trained on ground-truth positive windows from `train_all`.",
        "- The split protocol remains unchanged: `1.CSV + 2.CSV` for development and `3.CSV` for final testing.",
        "",
        "## 2. Why this baseline is interpretable",
        "- The model input is not a raw image tensor but a structured feature table with explicit physical meaning.",
        "- Tree models permit global interpretation through gain importance and additive contribution analysis (`pred_contribs`).",
        "- Therefore we can analyze both what the model uses globally and why a specific sample is classified as shallow, middle, or deep.",
        "",
        "## 2.1 What this explanation can and cannot claim",
        "- It can explain which engineered features the fitted tree model relied on.",
        "- It can explain which feature families pushed one specific test sample toward shallow, middle, or deep.",
        "- It cannot by itself prove a causal tissue-mechanics law.",
        "- It should therefore be interpreted together with the earlier non-deep-learning mechanism analysis.",
        "",
    ]
    lines.extend(model_summary_lines("Detection", det_global, det_family))
    lines.extend(model_summary_lines("Size classification", size_global, size_family))
    lines.extend(model_summary_lines("Size regression", size_reg_global, size_reg_family))
    lines.extend(model_summary_lines("Depth coarse", depth_global, depth_family))
    lines.extend(
        [
            "## 3. Depth-specific reading",
            f"- Depth coarse test accuracy: `{gt_metrics['depth_accuracy']:.4f}`",
            f"- Depth coarse balanced accuracy: `{gt_metrics['depth_balanced_accuracy']:.4f}`",
            f"- Top depth concepts: {', '.join(depth_concept['concept'].head(8).tolist())}",
            "- In the current XGBoost model, depth is dominated by deformation/position, shape/contrast, and spread features, while pure amplitude contributes less.",
            "- This is consistent with the earlier conclusion that depth is not just a peak-strength effect.",
            "- Temporal/phase features are present but not dominant in the current handcrafted baseline, which suggests either that the current temporal features are still too weak or that depth information is encoded more strongly in spatial morphology than in our present temporal summary.",
            "- If amplitude had dominated depth too strongly, the model would have been more vulnerable to confusing large-deep with small-shallow cases.",
            "",
            "## 4. Local explanation usage",
            "- `depth_local_explanations.png` shows feature contributions for representative correct and misclassified depth samples.",
            "- These panels explain the model's decision toward the predicted class, not the causal ground truth itself.",
            "",
            "## 5. Limits",
            "- XGBoost explanations are model-usage explanations, not direct physical proof.",
            "- Highly correlated handcrafted features may share or redistribute credit across one another.",
            "- Therefore these explanations should be read together with the earlier mechanism analysis, not in isolation.",
            "",
            "## 6. Implication for the next neural network",
            "- The next neural model should not try to learn depth from a single shared black-box feature.",
            "- It should explicitly preserve at least these concept families: amplitude, spread, shape contrast, and temporal/phase response.",
            "- A concept-guided or size-conditioned depth branch is therefore justified by this baseline.",
        ]
    )
    with open(os.path.join(explain_dir, "XGBOOST_EXPLAINABILITY_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
