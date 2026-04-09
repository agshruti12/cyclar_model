import argparse, yaml, csv
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from src.features_eng.build_features import compute_frame_features, box_center, compute_temporal_features
from src.features_eng.build_temporal_features import (
    compute_temporal_features_fast,
    TrackedObject,
    SideState,
)

CACHE_SIZE = 10

# replace left_cache / right_cache with
left_state  = SideState()
right_state = SideState()


# One row per side (left/right) per frame
FIELDNAMES = [
    # ── identifiers ────────────────────────────────────────────────────────────
    "video", "frame", "fps", "side",

    # ── frame-level ────────────────────────────────────────────────────────────
    "count_total",
    "area_frac",
    "bottom_y_norm",
    "max_conf",
    "closest_area_frac",
    "closest_bottom_y_norm",
    "danger_zone_hit",
    "lateral_spread",
    "count_far",
    "count_mid",
    "count_near",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",       required=True)
    ap.add_argument("--out",         required=True)
    ap.add_argument("--config",      default="configs/yolo.yaml")
    ap.add_argument("--yolo_stride", type=int, default=1,
                    help="Run YOLO every k frames, reuse last dets in-between.")
    args = ap.parse_args()

    cfg           = yaml.safe_load(open(args.config, "r"))
    model         = YOLO(cfg["model"])
    vehicle_names = set(cfg["vehicle_classes"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_name = str(Path(args.video).name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    left_cache:  list = []
    right_cache: list = []
    last_dets:   list = []

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for frame_idx in tqdm(range(nframes), desc="YOLO->CSV"):
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.yolo_stride == 0:
                r = model.track(
                    frame,
                    imgsz=cfg["imgsz"],
                    conf=cfg["conf"],
                    persist=True,
                    verbose=False,
                )[0]
                names = r.names

                dets = []
                if r.boxes is not None:
                    track_ids = (
                        r.boxes.id.int().cpu().tolist()
                        if r.boxes.id is not None
                        else [None] * len(r.boxes)
                    )
                    for b, track_id in zip(r.boxes, track_ids):
                        cls_id   = int(b.cls.item())
                        cls_name = names.get(cls_id, str(cls_id))
                        if cls_name not in vehicle_names:
                            continue
                        dets.append({
                            "cls":      cls_name,
                            "conf":     float(b.conf.item()),
                            "xyxy":     [float(x) for x in b.xyxy.squeeze(0).tolist()],
                            "track_id": track_id,
                        })
                last_dets = dets
            else:
                dets = last_dets

            # ── split by horizontal centre ──────────────────────────────────
            left_dets  = [d for d in dets if box_center(d["xyxy"])[0] <  frame_w / 2]
            right_dets = [d for d in dets if box_center(d["xyxy"])[0] >= frame_w / 2]

            # ── update per-side caches ──────────────────────────────────────
            # if len(left_cache) >= CACHE_SIZE:
            #     left_cache.pop(0)
            # left_cache.append(left_dets)

            # if len(right_cache) >= CACHE_SIZE:
            #     right_cache.pop(0)
            # right_cache.append(right_dets)

            # inside the frame loop, replace the cache update + temporal feature calls with:
            left_state.update(frame_idx, left_dets,  frame_w, frame_h)
            right_state.update(frame_idx, right_dets, frame_w, frame_h)

            # ── compute and write one row per side ──────────────────────────
            for side, side_dets, side_cache in (
                ("left",  left_dets,  left_state),
                ("right", right_dets, right_state),
            ):
                frame_feats    = compute_frame_features(frame_idx, side_dets, frame_w, frame_h, video=video_name)
                temporal_feats = compute_temporal_features_fast(frame_idx, side_dets, side_cache, frame_w, frame_h)

                writer.writerow({
                    **frame_feats,
                    **temporal_feats,
                    "fps":  float(fps),
                    "side": side,
                })

    cap.release()
    print(f"Saved features -> {out_path}")

if __name__ == "__main__":
    main()