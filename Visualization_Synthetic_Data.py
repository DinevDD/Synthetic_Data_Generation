import os
import pandas as pd
import matplotlib.pyplot as plt


## =========================
# 1. Load CSV event log
# =========================

raw_df = pd.read_csv("Data/data.csv")

print(f"Raw rows: {len(raw_df)}")
print(f"Columns: {list(raw_df.columns)}")

required_columns = [
    "Case ID",
    "Activity",
    "Timestamp"
]

missing_columns = [
    col for col in required_columns
    if col not in raw_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in Data/data.csv: {missing_columns}\n"
        f"Available columns are: {list(raw_df.columns)}"
    )

raw_df["Timestamp"] = pd.to_datetime(
    raw_df["Timestamp"],
    errors="coerce"
)

if raw_df["Timestamp"].isna().any():
    bad_rows = raw_df[raw_df["Timestamp"].isna()]
    raise ValueError(
        f"Some timestamps could not be parsed:\n{bad_rows}"
    )

event_rows = []

for _, row in raw_df.iterrows():
    case_id = row["Case ID"]
    base_timestamp = row["Timestamp"]
    group = row.get("Group", None)
    lifecycle = row.get("Lifecycle", "complete")

    activities = [
        activity.strip()
        for activity in str(row["Activity"]).split("/")
        if activity.strip()
    ]

    for idx, activity in enumerate(activities):
        event_rows.append({
            "case:concept:name": case_id,
            "concept:name": activity,
            "time:timestamp": base_timestamp + pd.Timedelta(seconds=idx),
            "org:group": group,
            "lifecycle:transition": lifecycle
        })

df = pd.DataFrame(event_rows)

df["time:timestamp"] = pd.to_datetime(
    df["time:timestamp"],
    utc=True,
    errors="coerce"
)

df = df.sort_values(
    ["case:concept:name", "time:timestamp"]
)

print(f"Expanded rows: {len(df)}")
print(f"Traces: {df['case:concept:name'].nunique()}")
print(f"Events: {len(df)}")
print(f"Activities: {sorted(df['concept:name'].unique())}")


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