# src/features_eng/build_features.py

from typing import Dict, List, Tuple

# Rear-helmet ROI fractions
ROI_X1_FRAC = 0.20
ROI_X2_FRAC = 0.80
ROI_Y1_FRAC = 0.20
ROI_Y2_FRAC = 0.95

def box_area(xyxy: List[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_center(xyxy: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def compute_frame_features(
    frame_idx: int,
    dets: List[Dict],
    frame_w: int,
    frame_h: int,
    video: str = "",
) -> Dict:
    """
    Frame-by-frame ONLY.
    Adds ROI aggregates for rear-facing helmet camera.
    dets: [{"cls": str, "conf": float, "xyxy": [x1,y1,x2,y2]}, ...]
    """

    # ROI pixel bounds
    roi_x1 = ROI_X1_FRAC * frame_w
    roi_x2 = ROI_X2_FRAC * frame_w
    roi_y1 = ROI_Y1_FRAC * frame_h
    roi_y2 = ROI_Y2_FRAC * frame_h

    def in_roi(det: Dict) -> bool:
        cx, cy = box_center(det["xyxy"])
        return (roi_x1 <= cx <= roi_x2) and (roi_y1 <= cy <= roi_y2)

    feat = {
        "video": video,
        "frame": frame_idx,

        # global minimal set
        "count_total": 0,
        "area_frac": 0.0,
        "bottom_y_norm": 0.0,
        "center_x_dist": 0.0,

        # ROI features to add
        "count_in_roi_all": 0,
        "max_area_in_roi_all": 0.0,
        "max_bottom_y_in_roi_all": 0.0,
    }

    # ---------- Global features (max-area box in full frame) ----------
    if dets:
        feat["count_total"] = len(dets)

        best = max(dets, key=lambda d: box_area(d["xyxy"]))
        x1, y1, x2, y2 = best["xyxy"]

        a = box_area(best["xyxy"])
        feat["area_frac"] = a / float(frame_w * frame_h)
        feat["bottom_y_norm"] = float(y2) / float(frame_h)

        cx = (x1 + x2) / 2.0
        feat["center_x_dist"] = abs(cx - (frame_w / 2.0)) / (frame_w / 2.0)

    # ---------- ROI features (aggregate over boxes whose CENTER is in ROI) ----------
    dets_roi = [d for d in dets if in_roi(d)]
    feat["count_in_roi_all"] = len(dets_roi)

    if dets_roi:
        best_roi = max(dets_roi, key=lambda d: box_area(d["xyxy"]))
        feat["max_area_in_roi_all"] = box_area(best_roi["xyxy"]) / float(frame_w * frame_h)
        feat["max_bottom_y_in_roi_all"] = max(d["xyxy"][3] for d in dets_roi) / float(frame_h)

    return feat
