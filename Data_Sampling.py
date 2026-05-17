import pm4py
import pandas as pd
import numpy as np

INPUT_LOG_PATH = "Data/filtered_log.xes.gz"
OUTPUT_LOG_PATH = "Data/sampled_10_percent_representative.xes.gz"
SAMPLE_PERCENTAGE = 0.10
RANDOM_SEED = 42
CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
TIMESTAMP_COLUMN = "time:timestamp"


log = pm4py.read_xes(INPUT_LOG_PATH)
df = pm4py.convert_to_dataframe(log)
df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
df = df.sort_values([CASE_ID_COLUMN, TIMESTAMP_COLUMN])

case_summary = (
    df.groupby(CASE_ID_COLUMN)
    .agg(
        trace_length=(ACTIVITY_COLUMN, "count"),
        start_time=(TIMESTAMP_COLUMN, "min"),
        end_time=(TIMESTAMP_COLUMN, "max"),
        start_activity=(ACTIVITY_COLUMN, "first"),
        end_activity=(ACTIVITY_COLUMN, "last"),
        variant=(ACTIVITY_COLUMN, lambda x: " -> ".join(x))
    )
    .reset_index()
)

case_summary["duration_seconds"] = (
    case_summary["end_time"] - case_summary["start_time"]
).dt.total_seconds()

total_cases = len(case_summary)
target_sample_size = round(total_cases * SAMPLE_PERCENTAGE)

print(f"Total cases: {total_cases}")
print(f"Target sampled cases: {target_sample_size}")

case_summary["length_bin"] = pd.qcut(
    case_summary["trace_length"],
    q=4,
    duplicates="drop"
)

case_summary["duration_bin"] = pd.qcut(
    case_summary["duration_seconds"],
    q=4,
    duplicates="drop"
)


variant_counts = case_summary["variant"].value_counts()
case_summary["variant_group"] = case_summary["variant"].where(
    case_summary["variant"].map(variant_counts) >= 3,
    "RARE_VARIANT"
)
case_summary["stratum"] = (
    case_summary["variant_group"].astype(str)
    + " | length=" + case_summary["length_bin"].astype(str)
    + " | duration=" + case_summary["duration_bin"].astype(str)
)


rng = np.random.default_rng(RANDOM_SEED)
sampled_case_ids = []
stratum_counts = case_summary["stratum"].value_counts()
for stratum, stratum_size in stratum_counts.items():
    stratum_cases = case_summary[
        case_summary["stratum"] == stratum
    ][CASE_ID_COLUMN].tolist()
    n_to_sample = round(stratum_size * SAMPLE_PERCENTAGE)
    if n_to_sample == 0 and stratum_size > 0:
        n_to_sample = 1
    n_to_sample = min(n_to_sample, stratum_size)
    sampled_cases = rng.choice(
        stratum_cases,
        size=n_to_sample,
        replace=False
    ).tolist()
    sampled_case_ids.extend(sampled_cases)


sampled_case_ids = list(set(sampled_case_ids))
if len(sampled_case_ids) > target_sample_size:
    sampled_case_ids = rng.choice(
        sampled_case_ids,
        size=target_sample_size,
        replace=False
    ).tolist()
elif len(sampled_case_ids) < target_sample_size:
    remaining_case_ids = list(
        set(case_summary[CASE_ID_COLUMN]) - set(sampled_case_ids)
    )
    extra_needed = target_sample_size - len(sampled_case_ids)
    extra_cases = rng.choice(
        remaining_case_ids,
        size=extra_needed,
        replace=False
    ).tolist()
    sampled_case_ids.extend(extra_cases)

sampled_df = df[df[CASE_ID_COLUMN].isin(sampled_case_ids)].copy()
sampled_df = sampled_df.sort_values(
    [CASE_ID_COLUMN, TIMESTAMP_COLUMN]
)
sampled_log = pm4py.convert_to_event_log(sampled_df)
pm4py.write_xes(sampled_log, OUTPUT_LOG_PATH)



sampled_case_summary = case_summary[
    case_summary[CASE_ID_COLUMN].isin(sampled_case_ids)
].copy()

print("\nSampling complete.")
print(f"Original cases: {len(case_summary)}")
print(f"Sampled cases: {len(sampled_case_summary)}")
print(f"Original events: {len(df)}")
print(f"Sampled events: {len(sampled_df)}")
print(f"Saved to: {OUTPUT_LOG_PATH}")

print("\nOriginal average trace length:")
print(case_summary["trace_length"].mean())

print("\nSampled average trace length:")
print(sampled_case_summary["trace_length"].mean())

print("\nOriginal average duration in seconds:")
print(case_summary["duration_seconds"].mean())

print("\nSampled average duration in seconds:")
print(sampled_case_summary["duration_seconds"].mean())

print("\nOriginal top variants:")
print(case_summary["variant"].value_counts(normalize=True).head(10))

print("\nSampled top variants:")
print(sampled_case_summary["variant"].value_counts(normalize=True).head(10))

print("\nOriginal start activity distribution:")
print(case_summary["start_activity"].value_counts(normalize=True).head(10))

print("\nSampled start activity distribution:")
print(sampled_case_summary["start_activity"].value_counts(normalize=True).head(10))

print("\nOriginal end activity distribution:")
print(case_summary["end_activity"].value_counts(normalize=True).head(10))

print("\nSampled end activity distribution:")
print(sampled_case_summary["end_activity"].value_counts(normalize=True).head(10))