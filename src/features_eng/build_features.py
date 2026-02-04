# src/features_eng/build_features.py

from typing import Dict, List, Optional, Tuple

def compute_frame_features(
    frame_idx: int,
    dets: List[Dict],
    frame_w: int,
    frame_h: int,
    video: str = "",
) -> Dict:
    """
    Frame-by-frame ONLY. No previous frame, no categorical fields.
    dets: [{"cls": str, "conf": float, "xyxy": [x1,y1,x2,y2]}, ...]
    """
    feat = {
        "video": video,
        "frame": frame_idx,
        "count_total": 0,
        "area_frac": 0.0,
        "bottom_y_norm": 0.0,
        "center_x_dist": 0.0,
    }

    if dets:
        feat["count_total"] = len(dets)

        def box_area(xyxy):
            x1, y1, x2, y2 = xyxy
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        best = max(dets, key=lambda d: box_area(d["xyxy"]))
        x1, y1, x2, y2 = best["xyxy"]
        a = box_area(best["xyxy"])

        feat["area_frac"] = a / float(frame_w * frame_h)
        feat["bottom_y_norm"] = float(y2) / float(frame_h)

        cx = (x1 + x2) / 2.0
        feat["center_x_dist"] = abs(cx - (frame_w / 2.0)) / (frame_w / 2.0)

    return feat
