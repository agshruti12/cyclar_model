import time
import cv2
from risk_head import RiskHead

# --- TODO: replace this with your YOLO detector wrapper ---
def yolo_detect(frame):
    """
    Return detections as list of dicts:
    [{"cls":"car","conf":0.9,"x1":...,"y1":...,"x2":...,"y2":...}, ...]
    """
    return []

# --- Example ROI: normalized polygon points for blind-spot region ---
ROI_NORM = [(0.55, 0.35), (1.0, 0.35), (1.0, 1.0), (0.55, 1.0)]  # right side

def polygon_to_pixels(poly_norm, w, h):
    return [(int(x*w), int(y*h)) for x, y in poly_norm]

def box_area(x1, y1, x2, y2):
    return max(0, x2 - x1) * max(0, y2 - y1)

def build_features(dets, roi_poly_px, frame_w, frame_h):
    """
    Minimal example features. IMPORTANT: keys must match feature_order used in training.
    """
    # For simplicity, we’ll approximate ROI overlap by checking if box center is inside ROI bounding rect.
    # Replace with true polygon intersection if you want (recommended).
    xs = [p[0] for p in roi_poly_px]
    ys = [p[1] for p in roi_poly_px]
    rx1, rx2 = min(xs), max(xs)
    ry1, ry2 = min(ys), max(ys)

    sum_area_in_roi_all = 0.0
    max_area_in_roi_all = 0.0
    count_in_roi_all = 0
    max_bottom_y_in_roi_all = 0.0

    for d in dets:
        if d["cls"] not in ("car", "truck", "bus", "motorcycle"):
            continue
        x1, y1, x2, y2 = map(float, (d["x1"], d["y1"], d["x2"], d["y2"]))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        in_roi = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
        if not in_roi:
            continue

        a = box_area(x1, y1, x2, y2) / float(frame_w * frame_h)  # normalized
        sum_area_in_roi_all += a
        max_area_in_roi_all = max(max_area_in_roi_all, a)
        count_in_roi_all += 1
        max_bottom_y_in_roi_all = max(max_bottom_y_in_roi_all, y2 / frame_h)

    return {
        "sum_area_in_roi_all": sum_area_in_roi_all,
        "max_area_in_roi_all": max_area_in_roi_all,
        "count_in_roi_all": float(count_in_roi_all),
        "max_bottom_y_in_roi_all": max_bottom_y_in_roi_all,
    }

def main():
    # Load LR head (JSON) once
    risk = RiskHead("risk_lr_artifact.json")

    # Camera (0 = default USB camera; for Pi cam you may need a different pipeline)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try a different index or pipeline.")

    last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        roi_px = polygon_to_pixels(ROI_NORM, w, h)

        # 1) YOLO detections
        dets = yolo_detect(frame)

        # 2) Build in-memory feature vector (dict)
        feat = build_features(dets, roi_px, w, h)

        # 3) Predict risk
        label, probs = risk.predict_from_dict(feat)

        # FPS
        now = time.time()
        dt = now - last
        last = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        # Overlay
        cv2.polylines(frame, [cv2.UMat(np.array(roi_px, dtype=int))], isClosed=True, color=(255, 255, 255), thickness=2)
        cv2.putText(frame, f"risk={label} probs={probs.round(2)} fps={fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Pi Live Risk", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
