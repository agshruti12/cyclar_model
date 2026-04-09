import argparse
import csv
import re
import tempfile
import os
from pathlib import Path

def fix_id_format_inplace(csv_path, column_name):
    pattern = re.compile(r"(.*)_sec_?\d+_frame(\d+_.*)")

    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    with open(csv_path, "r", newline="") as infile, open(temp_path, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            original = row.get(column_name, "")

            match = pattern.match(original)
            if match:
                row[column_name] = f"{match.group(1)}_frame{match.group(2)}"

            writer.writerow(row)

    # Replace original file
    os.replace(temp_path, csv_path)


def process_folder(folder_path, column_name):
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        fix_id_format_inplace(csv_file, column_name)

    print(f"\n✅ Done. Updated {len(csv_files)} files in-place.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Folder containing CSVs")
    parser.add_argument("--column", required=True, help="Column to fix (e.g., id)")

    args = parser.parse_args()

    process_folder(args.folder, args.column)