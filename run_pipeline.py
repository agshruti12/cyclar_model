import time
import argparse
from pathlib import Path
import yaml
import cv2

from ultralytics import YOLO

from src.features_eng.build_features import compute_frame_features 
from risk_head import RiskHead 


def load_cfg(cfg_path: str) -> dict:
    return yaml.safe_load(open(cfg_path, "r"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolo.yaml", help="YOLO config yaml")
    ap.add_argument("--risk_artifact", default="risk_lr_artifact.json", help="LR JSON artifact")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--yolo_stride", type=int, default=1, help="Run YOLO every k frames")
    ap.add_argument("--show", action="store_true", help="Show OpenCV window")
    ap.add_argument("--print_every", type=int, default=10, help="Print every N frames")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    vehicle_names = set(cfg["vehicle_classes"])  # from yolo.yaml

    # 1) Load models once
    yolo = YOLO(cfg["model"])  # yolo11n.pt by default in yolo.yaml 
    risk = RiskHead(args.risk_artifact)

    # 2) Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try --camera 0/1 or a different pipeline.")

    last_dets = []
    frame_idx = 0
    last_t = time.time()
    ema_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # 3) YOLO detections (optionally strided for speed)
        if frame_idx % args.yolo_stride == 0:
            r = yolo.predict(frame, imgsz=cfg["imgsz"], conf=cfg["conf"], verbose=False)[0]  # :contentReference[oaicite:8]{index=8}
            names = r.names

            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    cls_name = names.get(cls_id, str(cls_id))
                    if cls_name not in vehicle_names:
                        continue

                    dets.append(
                        {
                            "cls": cls_name,
                            "conf": float(b.conf.item()),
                            "xyxy": [float(x) for x in b.xyxy.squeeze(0).tolist()],
                        }
                    )
            last_dets = dets
        else:
            dets = last_dets

        # 4) Feature engineering
        feat = compute_frame_features(
            frame_idx=frame_idx,
            dets=dets,
            frame_w=w,
            frame_h=h,
            video="LIVE",
        )

        # 5) Risk prediction
        label, probs = risk.predict_from_dict(feat)

        # 6) Output (console + optional overlay)
        now = time.time()
        dt = now - last_t
        last_t = now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        ema_fps = 0.9 * ema_fps + 0.1 * inst_fps

        if frame_idx % args.print_every == 0:
            probs_pretty = ", ".join(
                f"{risk.index_to_label[i]}={float(probs[i]):.3f}" for i in range(len(probs))
            )
            print(f"[frame {frame_idx}] risk={label} | {probs_pretty} | fps={ema_fps:.1f}")

        if args.show:
            cv2.putText(
                frame,
                f"risk={label} probs={probs.round(2)} fps={ema_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Live Risk", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
