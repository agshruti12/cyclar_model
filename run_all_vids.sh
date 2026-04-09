for vid in final_data/*.mp4; do
  name=$(basename "$vid" .mp4)
  python run_yolo_more_data_modified.py \
    --video "$vid" \
    --out "features_minimalist/${name}.csv" \
    --config configs/yolo.yaml
done