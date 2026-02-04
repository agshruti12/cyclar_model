import json
from risk_head import RiskHead

def main():
    risk = RiskHead("risk_lr_artifact.json")  # load ONCE

    # --- Sample feature vectors (dicts) ---
    # Tip: You don't need to include every feature. Missing ones default to 0.0.
    samples = [
        {
            "sum_area_in_roi_all": 0.02,
            "max_area_in_roi_all": 0.02,
            "count_in_roi_all": 0,
            "max_bottom_y_in_roi_all": 0.25,
            "max_conf_in_roi_all": 0.70,
            "roi_depth": 0.20,
        },
        {
            "sum_area_in_roi_all": 0.18,
            "max_area_in_roi_all": 0.12,
            "count_in_roi_all": 2,
            "max_bottom_y_in_roi_all": 0.70,
            "max_conf_in_roi_all": 0.82,
            "roi_depth": 0.65,
            "max_area_in_roi_truck": 0.00,
            "sum_area_in_roi_truck": 0.00,
            "count_in_roi_truck": 0,
        },
        {
            "sum_area_in_roi_all": 0.45,
            "max_area_in_roi_all": 0.30,
            "count_in_roi_all": 3,
            "max_bottom_y_in_roi_all": 0.92,
            "max_conf_in_roi_all": 0.90,
            "roi_depth": 0.90,
            "max_area_in_roi_truck": 0.22,
            "sum_area_in_roi_truck": 0.28,
            "count_in_roi_truck": 1,
        },
    ]

    # --- Run predictions ---
    for i, feat in enumerate(samples):
        label, probs = risk.predict_from_dict(feat)

        # Print probs in a stable order (0,1,2)
        probs_by_label = {risk.index_to_label[j]: float(probs[j]) for j in range(len(probs))}
        probs_pretty = ", ".join([f"{k}={v:.3f}" for k, v in probs_by_label.items()])

        print(f"Sample {i}: pred={label} | {probs_pretty}")

if __name__ == "__main__":
    main()