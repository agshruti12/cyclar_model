#!/usr/bin/env bash
set -euo pipefail

mkdir -p features_labeled

for feat in features/*.csv; do
  base="$(basename "$feat" .csv)"
  label_json="labels/${base}_scene_labels.json"
  out_csv="features_labeled/${base}_labeled.csv"

  # Skip if labels don't exist yet
  if [ ! -f "$label_json" ]; then
    echo "Skipping $base (no labels found)"
    continue
  fi

  echo "Labeling $feat -> $out_csv"

  PYTHONPATH=. python src/labeling/apply_scene_labels_to_csv.py \
    --features_csv "$feat" \
    --labels_json "$label_json" \
    --out_csv "$out_csv"
done

echo "Done applying labels to all videos."