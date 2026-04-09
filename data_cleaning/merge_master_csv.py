import argparse
from pathlib import Path
import pandas as pd


def merge_all_csvs(input_dir: str, output_csv: str = "master_features_labelled.csv") -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_dir}")

    dfs = []
    base_columns = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if base_columns is None:
            base_columns = list(df.columns)
        elif list(df.columns) != base_columns:
            raise ValueError(
                f"Column mismatch in {csv_file.name}.\n"
                f"Expected: {base_columns}\n"
                f"Found:    {list(df.columns)}"
            )

        dfs.append(df)

    master_df = pd.concat(dfs, axis=0, ignore_index=True)
    master_df.to_csv(output_csv, index=False)

    print(f"Merged {len(csv_files)} CSV files into {output_csv}")
    print(f"Total rows: {len(master_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="final_csv",
        help="Folder containing per-video labelled feature CSVs"
    )
    parser.add_argument(
        "--output_csv",
        default="master_features_labelled.csv",
        help='Path for merged output CSV'
    )
    args = parser.parse_args()

    merge_all_csvs(args.input_dir, args.output_csv)
