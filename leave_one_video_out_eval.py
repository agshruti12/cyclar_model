from __future__ import annotations

import argparse
import re
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


LABEL_MAP_DEFAULT = {"low": 0, "medium": 1, "high": 2}


def _coerce_labels(y: pd.Series, label_map: Dict[str, int]) -> np.ndarray:
    if y.dtype == object:
        ys = y.astype(str).str.strip().str.lower()
        unknown = sorted(set(ys.unique()) - set(label_map.keys()))
        if unknown:
            raise ValueError(
                f"Unknown labels: {unknown}. Expected {sorted(label_map.keys())}."
            )
        return ys.map(label_map).to_numpy(dtype=int)

    return pd.to_numeric(y, errors="raise").astype(int).to_numpy()


def _extract_video_name(frame_id: str) -> str:
    match = re.match(r"^(.*)_frame\d+_(Left|Right)$", str(frame_id))
    if not match:
        raise ValueError(
            f"Could not parse video name from id '{frame_id}'. "
            "Expected format like 'cyclar_data_6_frame000000_Left'."
        )
    return match.group(1)


def _build_feature_matrix(
    df: pd.DataFrame,
    label_col: str,
    id_cols: Tuple[str, ...],
    label_map: Dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    if "id" not in df.columns:
        raise ValueError("Expected an 'id' column in the master CSV.")

    if df[label_col].isna().any():
        missing = int(df[label_col].isna().sum())
        raise ValueError(f"Found {missing} rows with missing labels in '{label_col}'.")

    y = _coerce_labels(df[label_col], label_map)

    drop_cols = [label_col] + [c for c in id_cols if c in df.columns]
    X_df = (
        df.drop(columns=drop_cols, errors="ignore")
        .apply(pd.to_numeric, errors="raise")
        .fillna(0.0)
    )

    if X_df.shape[1] == 0:
        raise ValueError("No feature columns remain after dropping label/id columns.")

    return X_df, y


def evaluate_leave_one_video_out(
    csv_path: str,
    label_col: str = "danger_label",
    id_cols: Tuple[str, ...] = (
        "id",
        "frame_id",
        "timestamp",
        "video",
        "video_name",
        "frame",
        "fps",
        "side",
    ),
    class_weight: Optional[str] = "balanced",
    C: float = 1.0,
    max_iter: int = 2000,
    out_csv: str = "leave_one_video_out_results.csv",
    label_map: Optional[Dict[str, int]] = None,
) -> None:
    if label_map is None:
        label_map = LABEL_MAP_DEFAULT

    df = pd.read_csv(csv_path)

    if "id" not in df.columns:
        raise ValueError(
            "The master CSV must include an 'id' column to derive video names."
        )

    df = df.copy()
    df["video_name"] = df["id"].apply(_extract_video_name)

    videos = [str(v) for v in sorted(df["video_name"].dropna().unique())]
    if len(videos) < 2:
        raise ValueError("Need at least 2 unique videos for leave-one-video-out evaluation.")

    results = []

    print(f"Found {len(videos)} unique videos:")
    for v in videos:
        print(f"  - {v}")

    for held_out_video in videos:
        train_df = df[df["video_name"] != held_out_video].copy()
        test_df = df[df["video_name"] == held_out_video].copy()

        if train_df.empty:
            raise ValueError(f"No training rows remain when holding out '{held_out_video}'.")
        if test_df.empty:
            raise ValueError(f"No test rows found for held-out video '{held_out_video}'.")

        X_train_df, y_train = _build_feature_matrix(train_df, label_col, id_cols, label_map)
        X_test_df, y_test = _build_feature_matrix(test_df, label_col, id_cols, label_map)

        missing_in_test = [c for c in X_train_df.columns if c not in X_test_df.columns]
        extra_in_test = [c for c in X_test_df.columns if c not in X_train_df.columns]
        if missing_in_test or extra_in_test:
            raise ValueError(
                f"Feature mismatch for held-out video '{held_out_video}'. "
                f"Missing in test: {missing_in_test}. Extra in test: {extra_in_test}."
            )

        X_test_df = X_test_df[X_train_df.columns]

        X_train = X_train_df.to_numpy(dtype=float)
        X_test = X_test_df.to_numpy(dtype=float)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="lbfgs",
            class_weight=class_weight,
            C=C,
            max_iter=max_iter,
        )
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)

        all_labels = sorted(label_map.values())   # [0, 1, 2]


        report = classification_report(y_test, y_pred, labels=all_labels, output_dict=True, zero_division=0)

        macro_f1 = f1_score(y_test, y_pred, average="macro", labels=all_labels, zero_division=0)
    
        medium_recall = report["1"]["recall"]   # assuming MEDIUM = 1
        medium_precision = report["1"]["precision"]
        high_recall = report["2"]["recall"]   # assuming HIGH = 2
        high_precision = report["2"]["precision"]

        print(f"\nHeld-out video: {held_out_video}")
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        results.append(
            {
                "held_out_video": held_out_video,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "medium_recall": round(float(medium_recall), 4),
                "medium_precision": round(float(medium_precision), 4),
                "high_recall": round(float(high_recall), 4),
                "high_precision": round(float(high_precision), 4),
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="master_features_labelled.csv",
        help="Merged labelled master CSV",
    )
    parser.add_argument(
        "--label_col",
        default="danger_label",
        help="Name of the label column",
    )
    parser.add_argument(
        "--out_csv",
        default="leave_one_video_out_results.csv",
        help="Where to save the evaluation summary",
    )

    args = parser.parse_args()

    evaluate_leave_one_video_out(
        csv_path=args.csv,
        label_col=args.label_col,
        out_csv=args.out_csv,
    )
