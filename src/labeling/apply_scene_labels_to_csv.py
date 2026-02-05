# What it does:
# reads features/<video>.csv
# reads labels/<video>_scene_labels.json
# creates a risk_label per frame (0/1/2 or empty if skipped)
# writes features_labeled/<video>.csv

import argparse
import json
from pathlib import Path

import pandas as pd

def build_frame_label_array(label_json_path: str):
    payload = json.load(open(label_json_path, "r"))
    frame_count = int(payload["frame_count"])
    labels = [-1] * frame_count  # -1 = unlabeled/skip

    for seg in payload["segments"]:
        lab = seg["label"]
        if lab is None:
            continue
        start_f = int(seg["start_frame"])
        end_f = int(seg["end_frame"])
        for f in range(start_f, end_f + 1):
            if 0 <= f < frame_count:
                labels[f] = int(lab)
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.features_csv)

    frame_labels = build_frame_label_array(args.labels_json)

    # safety: ensure CSV frame indices align
    max_frame = int(df["frame"].max())
    if max_frame >= len(frame_labels):
        raise ValueError(
            f"CSV has frame {max_frame} but labels only cover {len(frame_labels)} frames. "
            "Are you labeling the same video as the CSV?"
        )

    df["risk_label"] = df["frame"].apply(lambda f: frame_labels[int(f)])

    # ---- reorder columns so risk_label comes after frame ----
    cols = list(df.columns)

    cols.remove("risk_label")
    frame_idx = cols.index("frame")
    cols.insert(frame_idx + 1, "risk_label")

    df = df[cols]
    
    # optional: also add readable strings
    # map_str = {0:"LOW",1:"MED",2:"HIGH",-1:"UNLABELED"}
    # df["risk_label_str"] = df["risk_label"].map(map_str)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote labeled CSV -> {args.out_csv}")

if __name__ == "__main__":
    main()
