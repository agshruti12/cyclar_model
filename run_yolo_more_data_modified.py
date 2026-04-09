import argparse
import csv
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO

from src.features_eng.build_features import compute_frame_features, box_center


# One row per side (Left/Right) per frame
FIELDNAMES = [
    # identifiers
    "id",

    # frame-level features
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
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/yolo.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    model = YOLO(cfg["model"])
    vehicle_names = set(cfg["vehicle_classes"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_stem = Path(args.video).stem
    video_name = Path(args.video).name

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for frame_idx in tqdm(range(nframes), desc="YOLO->CSV"):
            ok, frame = cap.read()
            if not ok:
                break

            # Run detection independently on each frame.
            r = model.predict(
                frame,
                imgsz=cfg["imgsz"],
                conf=cfg["conf"],
                verbose=False,
            )[0]
            names = r.names

            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name not in vehicle_names:
                        continue
                    dets.append({
                        "cls": cls_name,
                        "conf": float(b.conf.item()),
                        "xyxy": [float(x) for x in b.xyxy.squeeze(0).tolist()],
                    })

            # Split by horizontal centre.
            left_dets = [d for d in dets if box_center(d["xyxy"])[0] < frame_w / 2]
            right_dets = [d for d in dets if box_center(d["xyxy"])[0] >= frame_w / 2]

            # Write exactly two rows per frame: one Left and one Right.
            for side_label, side_dets in (("Left", left_dets), ("Right", right_dets)):
                frame_feats = compute_frame_features(
                    frame_idx,
                    side_dets,
                    frame_w,
                    frame_h,
                    video=video_name,
                )

                row_id = f"{video_stem}_frame{frame_idx:06d}_{side_label}"

                writer.writerow({
                    "id": row_id,
                    **frame_feats,
                })

    cap.release()
    print(f"Saved frame-level features -> {out_path}")


if __name__ == "__main__":
    main()
