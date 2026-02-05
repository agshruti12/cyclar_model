# cyclar_model


### Global Frame Features

These summarize the **most dominant detected vehicle** in the full frame.

* **`count_total`**

  Total number of detected vehicles in the frame.

  *Captures traffic density behind the cyclist.*
* **`area_frac`**

  Area of the largest bounding box divided by total frame area.

  *Acts as a proxy for object proximity (larger → closer).*
* **`bottom_y_norm`**

  Bottom y-coordinate of the largest bounding box, normalized by frame height.

  *Encodes how low (and typically how close) the dominant vehicle is in the frame.*
* **`center_x_dist`**

  Normalized horizontal distance between the frame center and the center of the largest bounding box.

  *Measures alignment with the cyclist (centered vehicles are more directly behind).*
* **`fps`**

  Video frames per second.

  *Used to normalize temporal models and ensure consistency across videos.*

---

### Region-of-Interest (ROI) Features

*(Rear-Facing Helmet ROI: x ∈ [0.20W, 0.80W], y ∈ [0.20H, 0.95H])*

These aggregate information about vehicles in the **primary danger zone** behind the cyclist.

* **`count_in_roi_all`**

  Number of detected vehicles whose bounding box centers fall within the ROI.

  *Captures crowding in the most relevant spatial region.*
* **`max_area_in_roi_all`**

  Largest bounding box area within the ROI, normalized by frame area.

  *Measures the closest or most threatening vehicle in the danger zone.*
* **`max_bottom_y_in_roi_all`**

  Maximum normalized bottom y-coordinate among ROI detections.

  *Indicates how close the nearest vehicle in the ROI is to the cyclist.*


Commands:

* Requrements installation: pip install -r requirements.txt
* Run Yolo on all and save to features (csv) - bash src/perception/run_all_yolo.sh
* Label video at a time

PYTHONPATH=. python src/labeling/label_video_scene.py
  --video data_raw/flipped_part_007.mp4
  --out labels/flipped_part_007_scene_labels.json
  --segment-seconds 1.0

* Merge with its csv

PYTHONPATH=. python src/labeling/apply_scene_labels_to_csv.py
  --features_csv features/flipped_part_007.csv
  --labels_json labels/flipped_part_007_scene_labels.json
  --out_csv features_labeled/flipped_part_007_labeled.csv

* Merge all csvs

PYTHONPATH=. python src/utils/merge_feature_csvs.py
