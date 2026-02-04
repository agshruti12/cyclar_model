import argparse, yaml, csv
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

from src.features_eng.build_features import compute_frame_features

# Example:
# python src/perception/run_yolo.py \
#   --video data_raw/myvideo.mp4 \
#   --out features/myvideo.csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)          # e.g. features/myvideo.csv
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--yolo_stride", type=int, default=1, help="Run YOLO every k frames, reuse last dets in-between.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    model = YOLO(cfg["model"])
    vehicle_names = set(cfg["vehicle_classes"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video",
        "frame",
        "count_total",
        "area_frac",
        "bottom_y_norm",
        "center_x_dist",
        # optional but handy:
        "fps",
    ]

    last_dets = []

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx in tqdm(range(nframes), desc="YOLO->CSV"):
            ok, frame = cap.read()
            if not ok:
                break

            # YOLO every yolo_stride frames (optional speedup)
            if frame_idx % args.yolo_stride == 0:
                r = model.predict(frame, imgsz=cfg["imgsz"], conf=cfg["conf"], verbose=False)[0]
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
                last_dets = dets
            else:
                dets = last_dets

            feat = compute_frame_features(
                frame_idx=frame_idx,
                dets=dets,
                frame_w=frame_w,
                frame_h=frame_h,
                video=str(Path(args.video).name),
            )
            feat["fps"] = float(fps)

            writer.writerow(feat)

    cap.release()
    print(f"Saved features -> {out_path}")

if __name__ == "__main__":
    main()
