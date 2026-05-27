import csv
from pathlib import Path


INPUT_CSV = "generated_ts5_output.csv"
OUTPUT_CSV = "Gemma_ts5_repaired.csv"
REMOVED_CSV = "removed_cases.csv"


REQUIRED_COLUMNS = ["Case ID", "Activity", "Timestamp", "Group", "Lifecycle"]


def count_events(value: str) -> int:

    if value is None:
        return 0

    value = str(value).strip()

    if value == "":
        return 0

    return len(value.split("/"))


def repair_csv(input_path: str, output_path: str, removed_path: str):
    kept_rows = []
    removed_rows = []

    with open(input_path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        fieldnames = reader.fieldnames

        for row_number, row in enumerate(reader, start=2):
            counts = {
                "Activity": count_events(row["Activity"]),
                "Timestamp": count_events(row["Timestamp"]),
                "Group": count_events(row["Group"]),
                "Lifecycle": count_events(row["Lifecycle"]),
            }

            if len(set(counts.values())) == 1:
                kept_rows.append(row)
            else:
                row["_row_number"] = row_number
                row["_activity_count"] = counts["Activity"]
                row["_timestamp_count"] = counts["Timestamp"]
                row["_group_count"] = counts["Group"]
                row["_lifecycle_count"] = counts["Lifecycle"]
                removed_rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(kept_rows)

    removed_fieldnames = fieldnames + [
        "_row_number",
        "_activity_count",
        "_timestamp_count",
        "_group_count",
        "_lifecycle_count",
    ]

    with open(removed_path, "w", encoding="utf-8", newline="") as removed_file:
        writer = csv.DictWriter(removed_file, fieldnames=removed_fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(removed_rows)

    print("Repair complete.")
    print(f"Input file:   {input_path}")
    print(f"Output file:  {output_path}")
    print(f"Removed file: {removed_path}")
    print(f"Kept cases:   {len(kept_rows)}")
    print(f"Removed cases:{len(removed_rows)}")


if __name__ == "__main__":
    repair_csv(INPUT_CSV, OUTPUT_CSV, REMOVED_CSV)