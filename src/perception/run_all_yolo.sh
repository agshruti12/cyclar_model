#!/usr/bin/env bash
set -euo pipefail

mkdir -p features

for v in data_raw/*.mp4; do
  [ -e "$v" ] || continue  # if no mp4 files, don't error
  base="$(basename "$v" .mp4)"

  python src/perception/run_yolo.py \
    --video "$v" \
    --out "features/${base}.csv" \
    --config configs/yolo.yaml \
    --yolo_stride 1
done
