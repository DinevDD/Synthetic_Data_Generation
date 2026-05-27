import os
import pandas as pd
import matplotlib.pyplot as plt


## =========================
# 1. Load CSV event log
# =========================

def load_csv_event_log_dataframe(path: str) -> pd.DataFrame:
    """
    Loads synthetic CSV logs in either format:

    TS3/event-level format, preferred:
        Case ID, Activity, Timestamp, Group, Lifecycle
        1, ER Registration, 2014-02-26T10:15:03+00:00, A, complete
        1, ER Triage,       2014-02-26T10:22:43+00:00, C, complete

    TS1/case-level format, still supported:
        Case ID, Activity, Timestamp, Group, Lifecycle
        1, A/B/C, 2014-02-26T10:15:03+00:00, A, complete

    For TS3, timestamps/groups/lifecycles are preserved per event.
    For TS1, activities are expanded and synthetic +1 second offsets are used.
    """
    df = pd.read_csv(path)

    print(f"    Raw rows  : {len(df)}")
    print(f"    Columns   : {list(df.columns)}")

    required_cols = ["Case ID", "Activity", "Timestamp"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "Group" not in df.columns:
        df["Group"] = None
    if "Lifecycle" not in df.columns:
        df["Lifecycle"] = "complete"

    event_rows = []

    for row_idx, row in df.iterrows():
        case_id = str(row["Case ID"]).strip()

        activities = [
            part.strip()
            for part in str(row["Activity"]).split("/")
            if part.strip()
        ]

        timestamp_parts = [
            part.strip()
            for part in str(row["Timestamp"]).split("/")
            if part.strip()
        ]

        group_parts = [
            part.strip()
            for part in str(row.get("Group", "")).split("/")
            if part.strip()
        ]

        lifecycle_parts = [
            part.strip()
            for part in str(row.get("Lifecycle", "complete")).split("/")
            if part.strip()
        ]

        if not activities:
            raise ValueError(f"Row {row_idx} has no activity: {row.to_dict()}")

        if not timestamp_parts:
            raise ValueError(f"Row {row_idx} has no timestamp: {row.to_dict()}")

        for idx, activity in enumerate(activities):
            if len(timestamp_parts) == len(activities):
                timestamp_value = timestamp_parts[idx]
            elif len(timestamp_parts) == 1:
                timestamp_value = pd.to_datetime(
                    timestamp_parts[0],
                    utc=True,
                    errors="coerce"
                ) + pd.Timedelta(seconds=idx)
            else:
                raise ValueError(
                    f"Row {row_idx} has {len(activities)} activities but "
                    f"{len(timestamp_parts)} timestamps. Use either one timestamp "
                    "or one timestamp per activity."
                )

            if len(group_parts) == len(activities):
                group_value = group_parts[idx]
            elif len(group_parts) == 1:
                group_value = group_parts[0]
            else:
                group_value = None

            if len(lifecycle_parts) == len(activities):
                lifecycle_value = lifecycle_parts[idx]
            elif len(lifecycle_parts) == 1:
                lifecycle_value = lifecycle_parts[0]
            else:
                lifecycle_value = "complete"

            event_rows.append({
                "case:concept:name": case_id,
                "concept:name": activity,
                "time:timestamp": timestamp_value,
                "org:group": group_value,
                "lifecycle:transition": lifecycle_value,
            })

    event_df = pd.DataFrame(event_rows)

    event_df["time:timestamp"] = pd.to_datetime(
        event_df["time:timestamp"],
        utc=True,
        errors="coerce"
    )

    if event_df["time:timestamp"].isna().any():
        bad_rows = event_df[event_df["time:timestamp"].isna()]
        raise ValueError(f"Some timestamps could not be parsed:\n{bad_rows}")

    event_df = event_df.sort_values(
        ["case:concept:name", "time:timestamp"]
    ).reset_index(drop=True)

    print(f"    Event rows : {len(event_df)}")
    print(f"    Traces     : {event_df['case:concept:name'].nunique()}")
    print(f"    Events     : {len(event_df)}")
    print(f"    Activities ({event_df['concept:name'].nunique()}): "
          f"{sorted(event_df['concept:name'].dropna().unique())}")

    return event_df


df = load_csv_event_log_dataframe("Data/data.csv")


# =========================
# 2. Create output folder
# =========================

output_folder = "visualizations_synthetic"
os.makedirs(output_folder, exist_ok=True)


# =========================
# 3. Case-level statistics
# =========================

case_stats = (
    df
    .groupby("case:concept:name")
    .agg(
        start_time=("time:timestamp", "min"),
        end_time=("time:timestamp", "max"),
        events_per_case=("concept:name", "count")
    )
)

case_stats["case_duration"] = case_stats["end_time"] - case_stats["start_time"]
case_stats["case_duration_hours"] = case_stats["case_duration"].dt.total_seconds() / 3600
case_stats["case_duration_days"] = case_stats["case_duration_hours"] / 24


# =========================
# 4. Summary statistics
# =========================

summary_stats = pd.DataFrame({
    "Metric": [
        "Number of events",
        "Number of cases",
        "Number of unique activities",
        "Average events per case",
        "Median events per case",
        "Min events per case",
        "Max events per case",
        "Average case duration in hours",
        "Median case duration in hours",
        "Min case duration in hours",
        "Max case duration in hours"
    ],
    "Value": [
        len(df),
        df["case:concept:name"].nunique(),
        df["concept:name"].nunique(),
        case_stats["events_per_case"].mean(),
        case_stats["events_per_case"].median(),
        case_stats["events_per_case"].min(),
        case_stats["events_per_case"].max(),
        case_stats["case_duration_hours"].mean(),
        case_stats["case_duration_hours"].median(),
        case_stats["case_duration_hours"].min(),
        case_stats["case_duration_hours"].max()
    ]
})

summary_stats.to_csv(
    os.path.join(output_folder, "summary_statistics.csv"),
    index=False
)

print("\nSummary statistics:")
print(summary_stats.to_string(index=False))


# =========================
# 5. Helper function for saving plots
# =========================

def save_plot(filename):
    path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =========================
# 6. Activity frequency
# =========================

activity_counts = df["concept:name"].value_counts().head(20)

plt.figure(figsize=(10, 6))
activity_counts.sort_values().plot(kind="barh")
plt.title("Events Distribution")
plt.xlabel("Frequency")
plt.ylabel("Activity")
save_plot("top_20_most_common_events.png")


# =========================
# 7. Box plot of case durations
# =========================

plt.figure(figsize=(8, 5))
plt.boxplot(case_stats["case_duration_hours"], vert=False)
plt.title("Box Plot of Case Durations")
plt.xlabel("Case Duration in Hours")
save_plot("boxplot_case_durations.png")


# =========================
# 8. Box plot of events per case
# =========================

plt.figure(figsize=(8, 5))
plt.boxplot(case_stats["events_per_case"], vert=False)
plt.title("Box Plot of Events per Case")
plt.xlabel("Number of Events per Case")
save_plot("boxplot_events_per_case.png")


# =========================
# 9. Histogram of case durations
# =========================

plt.figure(figsize=(8, 5))
plt.hist(case_stats["case_duration_hours"], bins=30)
plt.title("Distribution of Case Durations")
plt.xlabel("Case Duration in Hours")
plt.ylabel("Number of Cases")
save_plot("histogram_case_durations.png")


# =========================
# 10. Histogram of events per case
# =========================

plt.figure(figsize=(8, 5))
plt.hist(case_stats["events_per_case"], bins=30)
plt.title("Distribution of Events per Case")
plt.xlabel("Number of Events per Case")
plt.ylabel("Number of Cases")
save_plot("histogram_events_per_case.png")


# =========================
# 11. Start activity distribution
# =========================

start_activities = (
    df
    .groupby("case:concept:name")
    .first()["concept:name"]
    .value_counts()
)

plt.figure(figsize=(10, 6))
start_activities.sort_values().plot(kind="barh")
plt.title("Start Activity Distribution")
plt.xlabel("Number of Cases")
plt.ylabel("Start Activity")
save_plot("start_activity_distribution.png")


# =========================
# 12. End activity distribution
# =========================

end_activities = (
    df
    .groupby("case:concept:name")
    .last()["concept:name"]
    .value_counts()
)

plt.figure(figsize=(10, 6))
end_activities.sort_values().plot(kind="barh")
plt.title("End Activity Distribution")
plt.xlabel("Number of Cases")
plt.ylabel("End Activity")
save_plot("end_activity_distribution.png")


# =========================
# 13. Top 10 variants
# =========================

variants = (
    df
    .groupby("case:concept:name")["concept:name"]
    .apply(lambda x: " → ".join(x))
)

variant_counts = variants.value_counts().head(10)

plt.figure(figsize=(12, 6))
variant_counts.sort_values().plot(kind="barh")
plt.title("Top 10 Most Common Variants")
plt.xlabel("Number of Cases")
plt.ylabel("Variant")
save_plot("top_10_most_common_variants.png")


# =========================
# 14. Duration percentiles
# =========================

duration_percentiles = case_stats["case_duration_hours"].quantile([
    0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00
])

duration_percentiles.to_csv(
    os.path.join(output_folder, "case_duration_percentiles.csv"),
    header=["case_duration_hours"]
)


# =========================
# 15. Event-count percentiles
# =========================

event_count_percentiles = case_stats["events_per_case"].quantile([
    0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00
])

event_count_percentiles.to_csv(
    os.path.join(output_folder, "events_per_case_percentiles.csv"),
    header=["events_per_case"]
)


# =========================
# 16. Activity percentages
# =========================

activity_percentage = (
    df["concept:name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

activity_percentage.to_csv(
    os.path.join(output_folder, "activity_frequency_percentage.csv"),
    header=["percentage"]
)


print(f"\nAll diagrams and statistics saved in folder: {output_folder}")