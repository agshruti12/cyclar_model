import argparse
from pathlib import Path
import pandas as pd

def merge_all(features_dir, labels_dir, output_dir):
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_files = list(features_dir.glob("*.csv"))

    if not feature_files:
        print("No feature CSVs found.")
        return

    for feat_path in feature_files:
        base_name = feat_path.stem  # e.g. cyclar_data_5
        label_name = f"{base_name}_labels.csv"
        label_path = labels_dir / label_name

        if not label_path.exists():
            print(f"Skipping {base_name}: no matching label file ({label_name})")
            continue

        print(f"Merging {base_name}...")

        df_feat = pd.read_csv(feat_path)
        df_lab = pd.read_csv(label_path)

        # Validate columns
        if "id" not in df_feat.columns:
            raise ValueError(f"{feat_path.name}: missing 'id' column")

        if "frame_id" not in df_lab.columns:
            raise ValueError(f"{label_path.name}: missing 'frame_id' column")

        merged = df_feat.merge(
            df_lab,
            left_on="id",
            right_on="frame_id",
            how="left"
        )

        # Drop redundant column
        merged = merged.drop(columns=["frame_id"], errors="ignore")

        # 1. Check for missing labels
        label_cols = [col for col in df_lab.columns if col != "frame_id"]

        for col in label_cols:
            if merged[col].isna().any():
                missing_ids = merged.loc[merged[col].isna(), "id"].tolist()[:5]
                raise ValueError(
                    f"{base_name}: Missing labels in column '{col}'.\n"
                    f"Example missing IDs: {missing_ids}"
                )

        # 2. Check for duplicate label IDs (VERY important)
        if df_lab["frame_id"].duplicated().any():
            dupes = df_lab[df_lab["frame_id"].duplicated()]["frame_id"].tolist()[:5]
            raise ValueError(
                f"{base_name}: Duplicate frame_id values in labels.\n"
                f"Examples: {dupes}"
            )

        # 3. (Optional but powerful) ensure row count didn’t change
        if len(merged) != len(df_feat):
            raise ValueError(
                f"{base_name}: Row count mismatch after merge.\n"
                f"Features: {len(df_feat)}, Merged: {len(merged)}"
            )

        out_path = output_dir / f"{base_name}.csv"
        merged.to_csv(out_path, index=False)

    print(f"\n✅ Done. Files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="features_csv/")
    parser.add_argument("--labels_dir", default="labels_csv/")
    parser.add_argument("--output_dir", default="final_csv/")

    args = parser.parse_args()

    merge_all(args.features_dir, args.labels_dir, args.output_dir)