import glob
import os
import pandas as pd

def main():
    input_glob = "features_labeled/*.csv"
    output_csv = "features.csv"

    csv_paths = sorted(glob.glob(input_glob))
    if not csv_paths:
        raise RuntimeError(f"No CSVs found matching {input_glob}")

    print("Merging the following files:")
    for p in csv_paths:
        print("  ", p)

    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # Optional: drop unlabeled frames if you want
    # merged = merged[merged["risk_label"] != -1]

    merged.to_csv(output_csv, index=False)
    print(f"\nWrote merged CSV -> {output_csv}")
    print(f"Total rows: {len(merged)}")
    print(f"Videos: {merged['video'].nunique()}")

if __name__ == "__main__":
    main()
