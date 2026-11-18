"""
train_models.py
===============
Amrithaa – KNN model training and evaluation for map-matching.

Task: given a GPS observation and K candidate road arcs, predict which
candidate is the correct match (the arc the vehicle was actually on).

Label definition
----------------
The label column is  `is_correct_arc`  (1 = this candidate is the ground-truth
arc, 0 = not).  It is created during training-data construction by joining the
candidate table with Arun's validated route arc IDs.

If the label column is not present (e.g. for prediction on test data) the
script loads a saved model and outputs probabilities.

Trip-level train/val split
--------------------------
GPS observations from the same trip must NOT appear in both train and val.
We split at the trip level to prevent leakage (as specified in the project
guidelines).

Models trained here
-------------------
  KNN  –  Amrithaa's assigned model
           (other models in Architha's / Arun's / Harshitha's scripts)

Usage
-----
    # Train
    python src/train_models.py --features results/features/all_features.csv \
                               --labels   results/labels/all_labels.csv    \
                               --out_dir  results/models/knn

    # Predict on new data
    python src/train_models.py --predict  results/features/test_features.csv \
                               --model    results/models/knn/knn_model.pkl   \
                               --out_dir  results/predictions/knn
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns used by KNN
# ─────────────────────────────────────────────────────────────────────────────

KNN_FEATURES = [
    "perp_dist_m",
    "norm_perp_dist",
    "heading_diff_deg",
    "norm_heading_diff",
    "length_m",
    "log_length_m",
    "log_perp_dist_m",
    "in_degree_src",
    "out_degree_src",
    "total_degree_src",
    "in_degree_tgt",
    "out_degree_tgt",
    "total_degree_tgt",
    "is_bidirectional",
    "turn_angle_src_deg",
    "turn_angle_tgt_deg",
    "connectivity_ratio_src",
    "connectivity_ratio_tgt",
    "speed_arc_compat",
    "is_prev_arc_adj",
    "norm_rank",
    "candidate_rank",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(features_csv: Path, labels_csv: Path) -> pd.DataFrame:
    """
    Merge feature table with labels.

    labels_csv must have columns: trip_id, gps_idx, orig_arc_id, is_correct_arc
    """
    feats = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)

    merged = feats.merge(
        labels[["trip_id", "gps_idx", "orig_arc_id", "is_correct_arc"]],
        on=["trip_id", "gps_idx", "orig_arc_id"],
        how="inner",
    )
    print(f"Merged rows: {len(merged)}  "
          f"(positive: {merged['is_correct_arc'].sum()}, "
          f"negative: {(~merged['is_correct_arc'].astype(bool)).sum()})")
    return merged


def trip_level_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split at the trip level so no trip appears in both train and val.

    Returns (train_df, val_df).
    """
    rng = np.random.default_rng(random_seed)
    trips = df["trip_id"].unique()
    rng.shuffle(trips)

    n_val = max(1, int(len(trips) * val_fraction))
    val_trips = set(trips[:n_val])
    train_trips = set(trips[n_val:])

    print(f"Train trips: {len(train_trips)}   Val trips: {len(val_trips)}")
    return df[df["trip_id"].isin(train_trips)], df[df["trip_id"].isin(val_trips)]


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays, filling missing feature columns with 0."""
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing feature cols (filled with 0): {missing}")
        for c in missing:
            df = df.copy()
            df[c] = 0.0

    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df["is_correct_arc"].values.astype(int)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# KNN training
# ─────────────────────────────────────────────────────────────────────────────

def train_knn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    n_neighbors_list: list[int] = [3, 5, 7, 11],
) -> Pipeline:
    """
    Train KNN with StandardScaler, evaluate on val set, pick best k,
    save model and all metrics.

    Returns the best fitted Pipeline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val,   y_val   = prepare_xy(val_df,   feature_cols)

    best_f1 = -1.0
    best_pipeline = None
    best_k = None
    tuning_rows: list[dict] = []

    for k in n_neighbors_list:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn",    KNeighborsClassifier(
                n_neighbors=k,
                metric="euclidean",
                weights="distance",
                n_jobs=-1,
            )),
        ])

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred     = pipe.predict(X_val)
        y_prob     = pipe.predict_proba(X_val)[:, 1]

        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        f1   = f1_score(y_val, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = float("nan")

        print(f"  k={k:2d}  acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}  "
              f"train_time={train_time:.2f}s")

        tuning_rows.append({
            "k": k, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc_roc": auc,
            "train_time_s": train_time,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe
            best_k = k

    print(f"\nBest k={best_k}  F1={best_f1:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = out_dir / "knn_model.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Saved model → {model_path}")

    # ── Detailed evaluation on val ────────────────────────────────────────────
    y_pred_best = best_pipeline.predict(X_val)
    y_prob_best = best_pipeline.predict_proba(X_val)[:, 1]

    _save_metrics(
        y_val, y_pred_best, y_prob_best,
        best_k, out_dir,
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
    )

    # ── Tuning summary ────────────────────────────────────────────────────────
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(out_dir / "knn_tuning_results.csv", index=False)
    _plot_tuning(tuning_df, out_dir)

    return best_pipeline


def _save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    best_k: int,
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Save metrics JSON, classification report, and confusion matrix plot."""

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics = {
        "model": "KNN",
        "best_k": best_k,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_train_trips": int(train_df["trip_id"].nunique()),
        "n_val_trips": int(val_df["trip_id"].nunique()),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "auc_roc": round(auc, 6) if not np.isnan(auc) else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "split": "trip-level (no leakage)",
    }

    metrics_path = out_dir / "knn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics  → {metrics_path}")
    print(f"\n{report}")

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not match", "Match"],
        yticklabels=["Not match", "Match"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"KNN (k={best_k}) – Confusion Matrix")
    plt.tight_layout()
    cm_path = out_dir / "knn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {cm_path}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
        "value":  [acc, prec, rec, f1, auc],
    })
    results_df.to_csv(out_dir / "knn_results.csv", index=False)


def _plot_tuning(tuning_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot F1, precision, recall vs k."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(tuning_df["k"], tuning_df["f1"],       "o-", label="F1")
    axes[0].plot(tuning_df["k"], tuning_df["precision"], "s--", label="Precision")
    axes[0].plot(tuning_df["k"], tuning_df["recall"],    "^--", label="Recall")
    axes[0].set_xlabel("k (neighbours)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("KNN – F1 / Precision / Recall vs k")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(tuning_df["k"], tuning_df["accuracy"], "o-", color="steelblue", label="Accuracy")
    ax2 = axes[1].twinx()
    ax2.plot(tuning_df["k"], tuning_df["train_time_s"], "s--", color="tomato", label="Train time (s)")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Accuracy", color="steelblue")
    ax2.set_ylabel("Train time (s)", color="tomato")
    axes[1].set_title("KNN – Accuracy and Training Time vs k")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "knn_tuning_curves.png", dpi=150)
    plt.close()
    print(f"Saved tuning curves → {out_dir / 'knn_tuning_curves.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction on new data
# ─────────────────────────────────────────────────────────────────────────────

def predict_knn(
    features_csv: Path,
    model_path: Path,
    out_dir: Path,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Load a saved KNN model and predict on new feature data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(features_csv)
    pipe  = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    X, _ = prepare_xy(feats, feature_cols)  # no labels for test
    probs = pipe.predict_proba(X)[:, 1]
    preds = pipe.predict(X)

    feats = feats.copy()
    feats["knn_pred"]     = preds
    feats["knn_prob_arc"] = probs

    out_path = out_dir / "knn_predictions.csv"
    feats.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train or run KNN map-matching model")
    p.add_argument("--features", type=Path,
                   default=Path("results/features/all_features.csv"),
                   help="Feature CSV (output of feature_engineering.py)")
    p.add_argument("--labels",   type=Path,
                   default=Path("results/labels/all_labels.csv"),
                   help="Label CSV with is_correct_arc column")
    p.add_argument("--out_dir",  type=Path,
                   default=Path("results/models/knn"),
                   help="Directory to save model and metrics")
    p.add_argument("--predict",  type=Path, default=None,
                   help="Run prediction on this feature CSV instead of training")
    p.add_argument("--model",    type=Path, default=None,
                   help="Path to saved model (for --predict mode)")
    p.add_argument("--val_frac", type=float, default=0.2,
                   help="Fraction of trips for validation (default 0.2)")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.predict is not None:
        # ── Prediction mode ───────────────────────────────────────────────────
        model_path = args.model or (args.out_dir / "knn_model.pkl")
        predict_knn(
            features_csv=args.predict,
            model_path=model_path,
            out_dir=args.out_dir,
            feature_cols=KNN_FEATURES,
        )
    else:
        # ── Training mode ─────────────────────────────────────────────────────
        if not args.features.exists():
            raise FileNotFoundError(
                f"Features file not found: {args.features}\n"
                "Run feature_engineering.py first."
            )
        if not args.labels.exists():
            raise FileNotFoundError(
                f"Labels file not found: {args.labels}\n"
                "Labels are created by joining candidates with Arun's route output."
            )

        df = load_and_merge(args.features, args.labels)
        train_df, val_df = trip_level_split(
            df, val_fraction=args.val_frac, random_seed=args.seed
        )

        print(f"\nTraining KNN  –  {len(train_df)} train rows, {len(val_df)} val rows")
        best_model = train_knn(
            train_df=train_df,
            val_df=val_df,
            feature_cols=KNN_FEATURES,
            out_dir=args.out_dir,
        )
        print("\n── KNN training complete ──")
