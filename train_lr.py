from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


LABEL_MAP_DEFAULT = {"low": 0, "medium": 1, "high": 2}


@dataclass
class LRArtifact:
    feature_order: List[str]
    index_to_label: Dict[int, str]
    scaler_mean: List[float]
    scaler_std: List[float]
    coef: List[List[float]]     # [K, F]
    intercept: List[float]      # [K]


def _coerce_labels(y: pd.Series, label_map: Dict[str, int]) -> np.ndarray:
    if y.dtype == object:
        ys = y.astype(str).str.strip().str.lower()
        unknown = sorted(set(ys.unique()) - set(label_map.keys()))
        if unknown:
            raise ValueError(f"Unknown labels: {unknown}. Expected {sorted(label_map.keys())}.")
        return ys.map(label_map).to_numpy(dtype=int)
    return pd.to_numeric(y, errors="raise").astype(int).to_numpy()


def train_and_export(
    features_csv: str,
    out_artifact_json: str = "risk_lr_artifact.json",
    label_col: str = "label",
    id_cols: Tuple[str, ...] = ("id", "frame_id", "timestamp"),
    test_size: float = 0.2,
    random_state: int = 42,
    class_weight: str | None = "balanced",
    C: float = 1.0,
    max_iter: int = 2000,
    label_map: Optional[Dict[str, int]] = None,
) -> None:
    if label_map is None:
        label_map = LABEL_MAP_DEFAULT

    df = pd.read_csv(features_csv)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {features_csv}")

    y = _coerce_labels(df[label_col], label_map)

    drop_cols = [label_col] + [c for c in id_cols if c in df.columns]
    X_df = df.drop(columns=drop_cols, errors="ignore").apply(pd.to_numeric, errors="raise").fillna(0.0)

    feature_order = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight=class_weight,
        C=C,
        max_iter=max_iter,
    )
    clf.fit(X_tr_s, y_tr)

    # quick sanity eval
    y_pred = clf.predict(X_te_s)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, zero_division=0)
    print("Confusion matrix:\n", cm)
    print("Report:\n", report)

    index_to_label = {v: k for k, v in label_map.items()}
    # ensure indices in order 0..K-1 exist
    for idx in clf.classes_:
        if int(idx) not in index_to_label:
            index_to_label[int(idx)] = str(int(idx))

    artifact = LRArtifact(
        feature_order=feature_order,
        index_to_label={int(k): v for k, v in index_to_label.items()},
        scaler_mean=scaler.mean_.tolist(),
        scaler_std=scaler.scale_.tolist(),
        coef=clf.coef_.tolist(),
        intercept=clf.intercept_.tolist(),
    )

    with open(out_artifact_json, "w", encoding="utf-8") as f:
        json.dump(asdict(artifact), f, indent=2)

    print(f"Saved artifact -> {out_artifact_json}")


if __name__ == "__main__":
    train_and_export("features.csv", "risk_lr_artifact.json")
