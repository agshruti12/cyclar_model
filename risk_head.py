from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)  # stability
    exps = np.exp(logits)
    return exps / np.sum(exps)


class RiskHead:
    """
    Lightweight multinomial logistic regression inference:
    - Loads feature order, scaler mean/std, weights, intercept.
    - predict_from_dict() takes an in-memory feature dict each frame.
    """

    def __init__(self, artifact_path: str):
        with open(artifact_path, "r", encoding="utf-8") as f:
            art = json.load(f)

        self.feature_order: List[str] = art["feature_order"]
        self.index_to_label: Dict[int, str] = {int(k): v for k, v in art["index_to_label"].items()}

        self.mean = np.array(art["scaler_mean"], dtype=np.float32)
        self.std = np.array(art["scaler_std"], dtype=np.float32)
        self.std = np.where(self.std == 0.0, 1.0, self.std).astype(np.float32)

        self.W = np.array(art["coef"], dtype=np.float32)       # [K, F]
        self.b = np.array(art["intercept"], dtype=np.float32)  # [K]

        # Pre-allocate a vector to reduce per-frame allocations
        self._x = np.zeros((len(self.feature_order),), dtype=np.float32)

    def predict_from_dict(self, feat: Dict[str, float]) -> Tuple[str, np.ndarray]:
        """
        feat: {feature_name: value} for current frame.
        Missing features default to 0.0.
        Returns: (label_str, probs[K])
        """
        # Fill feature vector in the correct order
        # (fast path: reuse the same array each frame)
        for i, name in enumerate(self.feature_order):
            self._x[i] = float(feat.get(name, 0.0))

        x_s = (self._x - self.mean) / self.std
        logits = self.W @ x_s + self.b
        probs = _softmax(logits)

        pred_idx = int(np.argmax(probs))
        pred_label = self.index_to_label.get(pred_idx, str(pred_idx))
        return pred_label, probs

    def predict_from_array(self, x: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        If you already compute a feature vector array in the correct feature_order.
        """
        x = x.astype(np.float32, copy=False)
        x_s = (x - self.mean) / self.std
        probs = _softmax(self.W @ x_s + self.b)
        pred_idx = int(np.argmax(probs))
        return self.index_to_label.get(pred_idx, str(pred_idx)), probs
