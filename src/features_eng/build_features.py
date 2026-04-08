# src/features_eng/build_features.py

from typing import Dict, List, Tuple
import numpy as np
import cmath

def box_area(xyxy: List[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_center(xyxy: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

# Danger zone thresholds (tunable)
DANGER_AREA_THRESH   = 0.08   # bbox covers >8% of frame
DANGER_BOTTOM_THRESH = 0.50   # bbox bottom edge in lower 25% of frame

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
    Frame-level features for rear-facing helmet camera danger detection.
    The entire frame is the region of interest.
    dets: [{"cls": str, "conf": float, "xyxy": [x1,y1,x2,y2], "track_id": int|None}]
    """
    frame_area = float(frame_w * frame_h)

    feat: Dict = {
        "video": video,
        "frame": frame_idx,

        # ── global count ───────────────────────────────────────────────────────
        "count_total":           0,

        # ── largest box (strongest presence) ──────────────────────────────────
        "area_frac":             0.0,  # bbox area / frame area
        "bottom_y_norm":         0.0,  # normalised y2 of largest box
        "max_conf":              0.0,  # YOLO confidence of most prominent detection

        # ── closest box (lowest in frame = physically nearest) ────────────────
        "closest_area_frac":     0.0,
        "closest_bottom_y_norm": 0.0,

        # ── danger zone ────────────────────────────────────────────────────────
        "danger_zone_hit":       0,    # binary: large + low object present

        # ── spatial distribution ───────────────────────────────────────────────
        "lateral_spread":        0.0,  # normalised x-spread across all detections
        "count_far":             0,    # objects in top third (distant, approaching)
        "count_mid":             0,    # objects in middle third
        "count_near":            0,    # objects in bottom third (imminent)
    }

    if not dets:
        return feat

    feat["count_total"] = len(dets)

    # ── largest box ────────────────────────────────────────────────────────────
    largest = max(dets, key=lambda d: box_area(d["xyxy"]))
    feat["area_frac"]     = box_area(largest["xyxy"]) / frame_area
    feat["bottom_y_norm"] = largest["xyxy"][3] / frame_h
    feat["max_conf"]      = max(d["conf"] for d in dets)

    # ── closest box (highest y2 = lowest in frame) ────────────────────────────
    closest = max(dets, key=lambda d: d["xyxy"][3])
    feat["closest_area_frac"]     = box_area(closest["xyxy"]) / frame_area
    feat["closest_bottom_y_norm"] = closest["xyxy"][3] / frame_h

    # ── danger zone binary ─────────────────────────────────────────────────────
    feat["danger_zone_hit"] = int(any(
        box_area(d["xyxy"]) / frame_area > DANGER_AREA_THRESH
        and d["xyxy"][3] / frame_h       > DANGER_BOTTOM_THRESH
        for d in dets
    ))

    # ── lateral spread ─────────────────────────────────────────────────────────
    if len(dets) > 1:
        centers_x = [box_center(d["xyxy"])[0] / frame_w for d in dets]
        feat["lateral_spread"] = float(max(centers_x) - min(centers_x))

    # ── vertical zone counts ───────────────────────────────────────────────────
    for d in dets:
        cy_norm = box_center(d["xyxy"])[1] / frame_h
        if cy_norm < 0.33:
            feat["count_far"]  += 1
        elif cy_norm < 0.66:
            feat["count_mid"]  += 1
        else:
            feat["count_near"] += 1

    return feat

def compute_temporal_features(
    frame_idx: int,
    detections: List[Dict],
    cache: List[List[Dict]],
    frame_w: int,
    frame_h: int,
) -> Dict:
    """
    Computes temporal features for blind-spot danger prediction using track_id
    to match objects across the LIFO cache (up to 10 frames).

    Each detection must contain: 'track_id', 'xyxy', 'cls', 'conf'.

    Key signals for danger:
      - Closing speed / TTC:  object rapidly growing in frame → imminent collision
      - Lateral drift:        object drifting toward cyclist lane
      - Sustained accel:      object consistently speeding up
      - New entries:          sudden appearance of vehicles in blind spot
    """
    temporal_features = {
        # ── count ──────────────────────────────────────────────────────────────
        "temporal_count_avg":               0.0,  # avg objects over cache window
        # ── speed / approach ───────────────────────────────────────────────────
        "avg_speed_avg":                    0.0,  # avg pixel speed across tracked objects
        "avg_speed_max":                    0.0,  # max pixel speed (fastest object)
        "closing_speed_avg":                0.0,  # avg bbox area growth rate (norm by frame)
        "closing_speed_max":                0.0,  # max closing speed — strongest danger signal
        "ttc_min":                          999.0, # min time-to-collision in frames (lower = more danger)
        # ── acceleration ───────────────────────────────────────────────────────
        "avg_acceleration_avg":             0.0,
        "avg_acceleration_max":             0.0,
        "accel_trend_avg":                  0.0,  # slope of speed history (sustained accel)
        "accel_trend_max":                  0.0,
        # ── direction / lateral ────────────────────────────────────────────────
        "circular_directional_consistency": 0.0,  # 1=all same dir, 0=random (circular mean)
        "x_drift_rate_avg":                 0.0,  # avg lateral drift slope (toward cyclist)
        "x_drift_rate_max":                 0.0,  # max lateral drift (most intrusive object)
        # ── area / proximity ───────────────────────────────────────────────────
        "area_change_avg":                  0.0,
        "area_change_max":                  0.0,
        # ── tracking quality ───────────────────────────────────────────────────
        "tracking_consistency_avg":         0.0,  # fraction of cache frames each ID was seen
        "new_entry_count":                  0,    # IDs in current frame absent from entire cache
        # ── clustering ─────────────────────────────────────────────────────────
        "clustering_density_avg":           0.0,  # avg pairwise dist between current objects
    }

    if not cache or not detections:
        return temporal_features

    frame_area = frame_w * frame_h

    # ── index current detections by track_id ───────────────────────────────────
    current_by_id: Dict[int, Dict] = {
        d["track_id"]: d
        for d in detections
        if d.get("track_id") is not None
    }

    # ── build per-ID speed history walking cache oldest → newest ───────────────
    # speed_history[track_id] = [speed_oldest, ..., speed_most_recent]
    speed_history: Dict[int, List[float]] = {tid: [] for tid in current_by_id}

    # ── collect per-match metrics ───────────────────────────────────────────────
    object_speeds:          List[float] = []
    object_accelerations:   List[float] = []
    movement_directions:    List[float] = []
    area_changes:           List[float] = []
    closing_speeds:         List[float] = []
    ttc_values:             List[float] = []
    accel_trends:           List[float] = []
    x_drift_rates:          List[float] = []

    # track how many cache frames each ID appeared in (for consistency score)
    id_frame_hits: Dict[int, int] = {tid: 0 for tid in current_by_id}

    for prev_frame_dets in cache:  # oldest → newest
        prev_by_id = {
            d["track_id"]: d
            for d in prev_frame_dets
            if d.get("track_id") is not None
        }

        for track_id, current_det in current_by_id.items():
            if track_id not in prev_by_id:
                continue  # object not seen in this past frame

            id_frame_hits[track_id] += 1
            prev_det = prev_by_id[track_id]

            prev_center    = np.array(box_center(prev_det["xyxy"]))
            current_center = np.array(box_center(current_det["xyxy"]))
            delta          = current_center - prev_center

            # ── pixel speed ────────────────────────────────────────────────────
            speed = float(np.linalg.norm(delta))
            object_speeds.append(speed)

            # ── acceleration (Δ vs previous recorded speed for this track) ─────
            if speed_history[track_id]:
                object_accelerations.append(speed - speed_history[track_id][-1])
            speed_history[track_id].append(speed)

            # ── movement direction (circular) ──────────────────────────────────
            movement_directions.append(float(np.arctan2(delta[1], delta[0])))

            # ── area change + closing speed ────────────────────────────────────
            prev_area    = box_area(prev_det["xyxy"])
            current_area = box_area(current_det["xyxy"])
            area_delta   = current_area - prev_area
            area_changes.append(area_delta)

            # closing speed: normalised bbox growth rate per frame
            if frame_area > 0:
                closing_speeds.append(area_delta / frame_area)

            # ── TTC estimate (frames until bbox fills 30% of frame) ────────────
            if prev_area > 0 and current_area > prev_area:
                growth_rate = current_area / prev_area
                target_fill = 0.30 * frame_area
                if growth_rate > 1.0 and current_area < target_fill:
                    ttc = np.log(target_fill / current_area) / np.log(growth_rate)
                    ttc_values.append(float(ttc))

    # ── per-track derived features (need full speed_history) ───────────────────
    for track_id in current_by_id:
        hist = speed_history[track_id]

        # sustained acceleration trend (linear slope over speed history)
        if len(hist) >= 3:
            slope = float(np.polyfit(range(len(hist)), hist, 1)[0])
            accel_trends.append(slope)

        # lateral drift rate (linear slope of normalised x-position over cache)
        x_positions = []
        for prev_frame_dets in cache:
            prev_by_id = {
                d["track_id"]: d
                for d in prev_frame_dets
                if d.get("track_id") is not None
            }
            if track_id in prev_by_id:
                cx = box_center(prev_by_id[track_id]["xyxy"])[0] / frame_w
                x_positions.append(cx)

        # append current position last so slope reads oldest→newest
        current_cx = box_center(current_by_id[track_id]["xyxy"])[0] / frame_w
        x_positions.append(current_cx)

        if len(x_positions) >= 3:
            x_drift_rates.append(
                float(np.polyfit(range(len(x_positions)), x_positions, 1)[0])
            )

    # ── new entries: IDs in current frame not seen anywhere in cache ────────────
    all_cached_ids = {
        d["track_id"]
        for frame in cache
        for d in frame
        if d.get("track_id") is not None
    }
    new_entry_count = sum(
        1 for tid in current_by_id if tid not in all_cached_ids
    )

    # ── clustering: pairwise distances between current objects (deduped) ────────
    current_list = list(current_by_id.values())
    clustering_dists = [
        float(np.linalg.norm(
            np.array(box_center(current_list[i]["xyxy"])) -
            np.array(box_center(current_list[j]["xyxy"]))
        ))
        for i in range(len(current_list))
        for j in range(i + 1, len(current_list))
    ]

    # ── temporal count (avg objects per cached frame) ──────────────────────────
    temporal_features["temporal_count_avg"] = float(np.mean([len(f) for f in cache]))

    # ── aggregate collected metrics ────────────────────────────────────────────
    if object_speeds:
        temporal_features["avg_speed_avg"] = float(np.mean(object_speeds))
        temporal_features["avg_speed_max"] = float(np.max(object_speeds))

    if closing_speeds:
        temporal_features["closing_speed_avg"] = float(np.mean(closing_speeds))
        temporal_features["closing_speed_max"] = float(np.max(closing_speeds))

    if ttc_values:
        temporal_features["ttc_min"] = float(np.min(ttc_values))  # most urgent object

    if object_accelerations:
        temporal_features["avg_acceleration_avg"] = float(np.mean(object_accelerations))
        temporal_features["avg_acceleration_max"] = float(np.max(np.abs(object_accelerations)))

    if accel_trends:
        temporal_features["accel_trend_avg"] = float(np.mean(accel_trends))
        temporal_features["accel_trend_max"] = float(np.max(accel_trends))

    if movement_directions:
        # circular mean resultant length: 1.0 = perfectly consistent, 0.0 = random
        temporal_features["circular_directional_consistency"] = float(
            abs(np.mean([cmath.exp(1j * d) for d in movement_directions]))
        )

    if x_drift_rates:
        temporal_features["x_drift_rate_avg"] = float(np.mean(x_drift_rates))
        temporal_features["x_drift_rate_max"] = float(np.max(np.abs(x_drift_rates)))

    if area_changes:
        temporal_features["area_change_avg"] = float(np.mean(area_changes))
        temporal_features["area_change_max"] = float(np.max(area_changes))

    if id_frame_hits:
        # fraction of cache frames each tracked object was consistently seen in
        consistency_scores = [hits / len(cache) for hits in id_frame_hits.values()]
        temporal_features["tracking_consistency_avg"] = float(np.mean(consistency_scores))

    temporal_features["new_entry_count"] = new_entry_count

    if clustering_dists:
        temporal_features["clustering_density_avg"] = float(np.mean(clustering_dists))

    return temporal_features