from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cmath

CACHE_SIZE = 10  # number of frames to keep in rolling history for temporal features

# ── per-object incremental state ───────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id:     int
    cls:          str
    last_center:  Tuple[float, float]  = (0.0, 0.0)
    last_area:    float                = 0.0
    speed_history: List[float]         = field(default_factory=list)
    x_history:    List[float]          = field(default_factory=list)  # normalised
    frames_seen:  int                  = 0
    # pre-aggregated scalars updated incrementally
    last_speed:         float = 0.0
    last_closing_speed: float = 0.0
    last_direction:     float = 0.0
    last_area_change:   float = 0.0

    def update(self, xyxy: List[float], frame_w: int, frame_h: int) -> None:
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
        frame_area = frame_w * frame_h

        if self.frames_seen > 0:
            dx = cx - self.last_center[0]
            dy = cy - self.last_center[1]
            self.last_speed         = float(np.hypot(dx, dy))
            self.last_closing_speed = (area - self.last_area) / frame_area
            self.last_direction     = float(np.arctan2(dy, dx))
            self.last_area_change   = area - self.last_area

            # cap speed history to cache size to bound memory
            self.speed_history.append(self.last_speed)
            if len(self.speed_history) > CACHE_SIZE:
                self.speed_history.pop(0)

        self.x_history.append(cx / frame_w)
        if len(self.x_history) > CACHE_SIZE:
            self.x_history.pop(0)

        self.last_center = (cx, cy)
        self.last_area   = area
        self.frames_seen += 1


# ── per-side incremental state ──────────────────────────────────────────────────

@dataclass
class SideState:
    """All mutable state for one side (left or right)."""
    # per-object tracker: track_id → TrackedObject
    objects: Dict[int, TrackedObject] = field(default_factory=dict)
    # rolling deque of per-frame count (for temporal_count_avg)
    count_history: deque = field(default_factory=lambda: deque(maxlen=CACHE_SIZE))
    # frame index of last seen frame per track_id (for consistency scoring)
    last_seen: Dict[int, int] = field(default_factory=dict)

    def update(self, frame_idx: int, dets: List[Dict], frame_w: int, frame_h: int) -> None:
        active_ids = set()
        for d in dets:
            tid = d.get("track_id")
            if tid is None:
                continue
            if tid not in self.objects:
                self.objects[tid] = TrackedObject(track_id=tid, cls=d["cls"])
            self.objects[tid].update(d["xyxy"], frame_w, frame_h)
            self.last_seen[tid] = frame_idx
            active_ids.add(tid)

        self.count_history.append(len(active_ids))

        # evict objects not seen for more than CACHE_SIZE frames
        stale = [tid for tid, last in self.last_seen.items()
                 if frame_idx - last > CACHE_SIZE]
        for tid in stale:
            self.objects.pop(tid, None)
            self.last_seen.pop(tid, None)


# ── feature computation ─────────────────────────────────────────────────────────

def compute_temporal_features_fast(
    frame_idx: int,
    dets: List[Dict],
    state: SideState,
    frame_w: int,
    frame_h: int,
) -> Dict:
    """
    O(N) in number of current detections — reads pre-aggregated per-object
    state rather than re-walking raw detection history.
    """
    temporal_features = {
        "temporal_count_avg":               0.0,
        "avg_speed_avg":                    0.0,
        "avg_speed_max":                    0.0,
        "closing_speed_avg":                0.0,
        "closing_speed_max":                0.0,
        "ttc_min":                          999.0,
        "avg_acceleration_avg":             0.0,
        "avg_acceleration_max":             0.0,
        "accel_trend_avg":                  0.0,
        "accel_trend_max":                  0.0,
        "circular_directional_consistency": 0.0,
        "x_drift_rate_avg":                 0.0,
        "x_drift_rate_max":                 0.0,
        "area_change_avg":                  0.0,
        "area_change_max":                  0.0,
        "tracking_consistency_avg":         0.0,
        "new_entry_count":                  0,
        "clustering_density_avg":           0.0,
    }

    if state.count_history:
        temporal_features["temporal_count_avg"] = float(np.mean(state.count_history))

    active_ids = [d["track_id"] for d in dets if d.get("track_id") is not None]
    if not active_ids:
        return temporal_features

    frame_area = frame_w * frame_h
    speeds, closing_speeds, directions = [], [], []
    area_changes, accelerations, accel_trends = [], [], []
    x_drift_rates, consistency_scores = [], []
    ttc_values = []

    for tid in active_ids:
        obj = state.objects.get(tid)
        if obj is None or obj.frames_seen < 2:
            continue  # no history yet for this object

        # ── read pre-aggregated scalars directly (O(1) each) ──────────────────
        speeds.append(obj.last_speed)
        closing_speeds.append(obj.last_closing_speed)
        directions.append(obj.last_direction)
        area_changes.append(obj.last_area_change)

        # ── acceleration from speed history (already capped to CACHE_SIZE) ────
        hist = obj.speed_history
        if len(hist) >= 2:
            accelerations.append(hist[-1] - hist[-2])
        if len(hist) >= 3:
            accel_trends.append(
                float(np.polyfit(range(len(hist)), hist, 1)[0])
            )

        # ── x drift rate from x_history (already capped to CACHE_SIZE) ───────
        xh = obj.x_history
        if len(xh) >= 3:
            x_drift_rates.append(
                float(np.polyfit(range(len(xh)), xh, 1)[0])
            )

        # ── TTC from closing speed (O(1), no log needed for near-zero) ────────
        if obj.last_closing_speed > 0 and obj.last_area > 0:
            target_fill = 0.30 * frame_area
            if obj.last_area < target_fill:
                growth_rate = (obj.last_area + obj.last_closing_speed * frame_area) / obj.last_area
                if growth_rate > 1.0:
                    ttc = np.log(target_fill / obj.last_area) / np.log(growth_rate)
                    ttc_values.append(float(ttc))

        # ── tracking consistency: fraction of recent frames this ID was seen ──
        frames_in_window = min(obj.frames_seen, CACHE_SIZE)
        consistency_scores.append(frames_in_window / CACHE_SIZE)

    # ── new entries: active IDs with only 1 frame of history ──────────────────
    temporal_features["new_entry_count"] = sum(
        1 for tid in active_ids
        if state.objects.get(tid) and state.objects[tid].frames_seen == 1
    )

    # ── clustering: pairwise distances O(N²) but N is small in practice ───────
    centers = [state.objects[tid].last_center for tid in active_ids if tid in state.objects]
    if len(centers) > 1:
        dists = [
            float(np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
            for i in range(len(centers))
            for j in range(i + 1, len(centers))
        ]
        temporal_features["clustering_density_avg"] = float(np.mean(dists))

    # ── aggregate ─────────────────────────────────────────────────────────────
    if speeds:
        temporal_features["avg_speed_avg"] = float(np.mean(speeds))
        temporal_features["avg_speed_max"] = float(np.max(speeds))
    if closing_speeds:
        temporal_features["closing_speed_avg"] = float(np.mean(closing_speeds))
        temporal_features["closing_speed_max"] = float(np.max(closing_speeds))
    if ttc_values:
        temporal_features["ttc_min"] = float(np.min(ttc_values))
    if accelerations:
        temporal_features["avg_acceleration_avg"] = float(np.mean(accelerations))
        temporal_features["avg_acceleration_max"] = float(np.max(np.abs(accelerations)))
    if accel_trends:
        temporal_features["accel_trend_avg"] = float(np.mean(accel_trends))
        temporal_features["accel_trend_max"] = float(np.max(accel_trends))
    if directions:
        temporal_features["circular_directional_consistency"] = float(
            abs(np.mean([cmath.exp(1j * d) for d in directions]))
        )
    if x_drift_rates:
        temporal_features["x_drift_rate_avg"] = float(np.mean(x_drift_rates))
        temporal_features["x_drift_rate_max"] = float(np.max(np.abs(x_drift_rates)))
    if area_changes:
        temporal_features["area_change_avg"] = float(np.mean(area_changes))
        temporal_features["area_change_max"] = float(np.max(area_changes))
    if consistency_scores:
        temporal_features["tracking_consistency_avg"] = float(np.mean(consistency_scores))

    return temporal_features