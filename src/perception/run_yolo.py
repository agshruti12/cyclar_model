import argparse, json, yaml
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# -- Example Script for video data_raw/myvideo.mp4 --
# python src/perception/run_yolo.py \
#   --video data_raw/myvideo.mp4 \
#   --out detections/myvideo.jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)          # e.g. detections/myvideo.jsonl
    ap.add_argument("--config", default="configs/yolo.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    model = YOLO(cfg["model"])
    vehicle_names = set(cfg["vehicle_classes"])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for frame_idx in tqdm(range(nframes), desc="YOLO"):
            ok, frame = cap.read()
            if not ok:
                break

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

            f.write(json.dumps({
                "frame": frame_idx,
                "fps": float(fps),
                "dets": dets
            }) + "\n")

    cap.release()
    print(f"Saved detections -> {out_path}")

if __name__ == "__main__":
    main()
