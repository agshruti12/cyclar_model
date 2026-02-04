# Blind-Spot Risk Classification (YOLO + Logistic Regression)

This project implements a **two-stage risk classification pipeline** for bicyclist blind-spot safety:

1. **Perception**: a vision model (e.g., YOLO) detects nearby vehicles.
2. **Decision / Risk Head**: a lightweight logistic regression model classifies the scene as `low`, `medium`, or `high` danger based on engineered features.

The design intentionally separates **training (offline, on a laptop)** from **inference (real-time, on a Raspberry Pi 5)**.

---

## High-level idea

- YOLO (or another detector) answers: _what objects are present and where?_
- Feature engineering answers: _how risky does this configuration look?_
- Logistic regression answers: _given these features, what is the danger level?_

The logistic regression model is trained once on labeled data, exported as a small JSON artifact, and then reused for real-time inference without retraining.

---

## File overview

### `features.csv`

A tabular dataset used **only during training**. Each row corresponds to one video frame (or image) and contains:

- engineered numeric features derived from detections (e.g., bounding box areas in a blind-spot ROI),
- a single ground-truth label: `low`, `medium`, or `high`.

---

### `train_lr.py`

This script trains the **logistic regression risk head** using `features.csv`.

What it does:

- loads features and labels,
- standardizes features,
- trains a multinomial logistic regression model,
- exports everything needed for inference into a single JSON file (`risk_lr_artifact.json`).

This script is run on a laptop or development machine, not on the Pi.

---

### `risk_lr_artifact.json`

The trained model artifact produced by `train_lr.py`.

It contains:

- the feature ordering,
- scaler mean and standard deviation,
- learned weights and intercepts,
- the mapping from class index to label.

This file is copied to the Raspberry Pi and loaded once at runtime.

---

### `risk_head.py`

A lightweight **inference-only module** for the logistic regression model.

Key points:

- Uses only NumPy (no scikit-learn, no PyTorch).
- Loads `risk_lr_artifact.json` once at initialization.
- Exposes `predict_from_dict(...)`, which takes a single in-memory feature dictionary and returns:
  - the predicted label,
  - class probabilities.

This is the core component used during real-time inference.

---

### `test_prediction.py`

A simple test script that demonstrates how to run predictions on **manually defined feature vectors** without a camera or YOLO.

This is useful for:

- sanity-checking the trained model,
- understanding how feature values affect predictions,
- debugging before deploying to hardware.

---

### `run_pi_live.py`

A skeleton for **real-time inference on the Raspberry Pi**.

What it shows:

- how to load the trained risk head once,
- how to process frames in a loop,
- where YOLO detections should be plugged in,
- how to build a feature dictionary per frame,
- how to call the risk head for each frame.

The YOLO detector itself is intentionally left as a placeholder so it can be swapped for different backends (Ultralytics, ONNX Runtime, OpenCV DNN, etc.).

---

## Typical workflow

### 1. Offline training (laptop)

1. Generate or collect labeled data.
2. Store engineered features in `features.csv`.
3. Run:
   ```bash
   python train_lr.py
   ```
