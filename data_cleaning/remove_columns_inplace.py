import argparse
import csv
import tempfile
import os

def remove_columns_inplace(csv_path, columns_to_remove):
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)

    with open(csv_path, "r", newline="") as infile, open(temp_path, "w", newline="") as outfile:
        reader = csv.DictReader(infile)

        remaining_fields = [f for f in reader.fieldnames if f not in columns_to_remove]

        writer = csv.DictWriter(outfile, fieldnames=remaining_fields)
        writer.writeheader()

        for row in reader:
            filtered_row = {k: v for k, v in row.items() if k in remaining_fields}
            writer.writerow(filtered_row)

    # Replace original file
    os.replace(temp_path, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV file to modify")
    parser.add_argument("--remove", nargs="+", required=True,
                        help="Columns to remove")

    args = parser.parse_args()
    remove_columns_inplace(args.input, args.remove)