# src/features_eng/build_features.py

from typing import Dict, List, Tuple


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
    Minimalist frame-level features for a jittery rear helmet camera.

    Design choices:
    - no hard-coded ROI
    - no hard-coded danger thresholds
    - keep feature count small to reduce overfitting risk
    - explicitly include both:
        1) largest box  -> strongest visual presence
        2) closest box  -> strongest immediate proximity cue

    Expected input:
      dets = [{"cls": str, "conf": float, "xyxy": [x1,y1,x2,y2], ...}, ...]

    Returns a flat dict of numeric features.
    """
    frame_area = float(max(frame_w * frame_h, 1))

    feat: Dict = {

        # scene-level
        "count_total": 0,

        # largest box
        "largest_area_frac": 0.0,
        "largest_bottom_y_norm": 0.0,
        "largest_conf": 0.0,

        # closest box
        "closest_area_frac": 0.0,
        "closest_bottom_y_norm": 0.0,
        "closest_conf": 0.0,

        # light aggregate context
        "sum_area_frac": 0.0,
        "mean_conf": 0.0,
    }

    if not dets:
        return feat

    feat["count_total"] = len(dets)

    area_fracs = []
    confs = []
    bottoms = []

    for d in dets:
        area_frac = box_area(d["xyxy"]) / frame_area
        conf = float(d.get("conf", 0.0))
        bottom_norm = d["xyxy"][3] / float(frame_h)

        area_fracs.append(area_frac)
        confs.append(conf)
        bottoms.append(bottom_norm)

    feat["sum_area_frac"] = float(sum(area_fracs))
    feat["mean_conf"] = float(sum(confs) / len(confs))

    # largest box = max area
    largest_idx = max(range(len(dets)), key=lambda i: area_fracs[i])
    feat["largest_area_frac"] = float(area_fracs[largest_idx])
    feat["largest_bottom_y_norm"] = float(bottoms[largest_idx])
    feat["largest_conf"] = float(confs[largest_idx])

    # closest box = largest bottom edge (lowest in frame)
    closest_idx = max(range(len(dets)), key=lambda i: bottoms[i])
    feat["closest_area_frac"] = float(area_fracs[closest_idx])
    feat["closest_bottom_y_norm"] = float(bottoms[closest_idx])
    feat["closest_conf"] = float(confs[closest_idx])

    return feat
