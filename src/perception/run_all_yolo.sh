#!/usr/bin/env bash
set -euo pipefail

mkdir -p features

for v in data_raw/*.mp4; do
  [ -e "$v" ] || { echo "No .mp4 files found in data_raw/"; exit 0; }

  base="$(basename "$v" .mp4)"
  out="features/${base}.csv"

  echo "=== Processing: $v -> $out ==="

  PYTHONPATH=. python src/perception/run_yolo.py \
    --video "$v" \
    --out "$out" \
    --config configs/yolo.yaml \
    --yolo_stride 1
done

echo "Done. Wrote per-video CSVs to ./features/"
