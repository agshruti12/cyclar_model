To generate frame-level features for a given video:

python run_yolo_more_data_modified.py \
  --video path/to/video.mp4 \
  --out outputs/video_features.csv \
  --config configs/yolo.yaml

To generate frame-level features for a given folder:

./run_all_vids but change the filepath location of the videos as needed


Data Cleaning:

1. Clean frame ID on the labels (entire folder)

  python data_cleaning/fix_id_format_inplace.py \
  --folder final_labels/arushi_vid_labels \
  --column frame_id

2. Merge labels with features (entire folder)

python data_cleaning/merge_all_features_labels.py \
  --features_dir features_minimalist/ \
  --labels_dir final_labels/ \
  --output_dir merged_output_minimalist/

3. Merge all CSVs into one master CSVs

python data_cleaning/merge_master_csv.py --input_dir merged_output_minimalist

4. Training (checkout to training branch)

python train_lr_master.py --csv master_features_labelled.csv
python leave_one_video_out_eval.py --csv master_features_labelled.csv



ONCE features are done making:

1. merge labels
2. merge into large CSV
3. python leave_one_video_out_eval.py --csv master_features_labelled_minimalist.csv
see if it's giving better results this time around, if it is, then train final model as listed below
4. python train_lr_master.py --csv master_features_labelled_minimalist.csv


to copy into pi:
1. copy the build_features.py function
2. copy the risk_lr_artifact.json