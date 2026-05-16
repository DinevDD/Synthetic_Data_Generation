import os
import pm4py
import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. File paths
# =========================

original_path = "Data/Sepsis Cases - Event Log.xes.gz"
synthetic_path = "Data/data.csv"

output_folder = "comparison_results"
os.makedirs(output_folder, exist_ok=True)


# =========================
# 2. Load original XES.GZ log
# =========================

def load_original_xes(path):
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    return df


# =========================
# 3. Load synthetic CSV log
# =========================

def load_synthetic_csv(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Case ID": "case:concept:name",
        "Activity": "concept:name",
        "Timestamp": "time:timestamp",
        "Group": "org:group",
        "Lifecycle": "lifecycle:transition"
    })

    required_columns = [
        "case:concept:name",
        "concept:name",
        "time:timestamp"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path}: {missing_columns}\n"
            f"Available columns are: {list(df.columns)}"
        )

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    return df


original_df = load_original_xes(original_path)
synthetic_df = load_synthetic_csv(synthetic_path)


# =========================
# 4. Activity distribution
# =========================

def get_activity_distribution(df):
    return df["concept:name"].value_counts(normalize=True)


original_activity_dist = get_activity_distribution(original_df)
synthetic_activity_dist = get_activity_distribution(synthetic_df)

all_activities = sorted(
    set(original_activity_dist.index)
    | set(synthetic_activity_dist.index)
)

original_activity_vector = (
    original_activity_dist
    .reindex(all_activities, fill_value=0)
    .values
)

synthetic_activity_vector = (
    synthetic_activity_dist
    .reindex(all_activities, fill_value=0)
    .values
)


# =========================
# 5. Cosine similarity for event distribution
# =========================

activity_cosine_similarity = cosine_similarity(
    [original_activity_vector],
    [synthetic_activity_vector]
)[0][0]


# =========================
# 6. Jensen-Shannon divergence for event distribution
# =========================

js_distance = jensenshannon(
    original_activity_vector,
    synthetic_activity_vector,
    base=2
)

js_divergence = js_distance ** 2


# =========================
# 7. Case duration Wasserstein distance
# =========================

def get_case_durations_hours(df):
    case_stats = (
        df
        .groupby("case:concept:name")
        .agg(
            start_time=("time:timestamp", "min"),
            end_time=("time:timestamp", "max")
        )
    )

    case_durations = (
        case_stats["end_time"] - case_stats["start_time"]
    ).dt.total_seconds() / 3600

    return case_durations


original_case_durations = get_case_durations_hours(original_df)
synthetic_case_durations = get_case_durations_hours(synthetic_df)

case_duration_wasserstein = wasserstein_distance(
    original_case_durations,
    synthetic_case_durations
)


# =========================
# 8. DFG distribution
# =========================

def get_dfg_distribution(df):
    dfg_edges = []

    for _, case in df.groupby("case:concept:name"):
        case = case.sort_values("time:timestamp")
        activities = case["concept:name"].tolist()

        for i in range(len(activities) - 1):
            edge = (activities[i], activities[i + 1])
            dfg_edges.append(edge)

    if len(dfg_edges) == 0:
        return pd.Series(dtype=float)

    return pd.Series(dfg_edges).value_counts(normalize=True)


original_dfg_dist = get_dfg_distribution(original_df)
synthetic_dfg_dist = get_dfg_distribution(synthetic_df)

all_dfg_edges = sorted(
    set(original_dfg_dist.index)
    | set(synthetic_dfg_dist.index)
)

original_dfg_vector = (
    original_dfg_dist
    .reindex(all_dfg_edges, fill_value=0)
    .values
)

synthetic_dfg_vector = (
    synthetic_dfg_dist
    .reindex(all_dfg_edges, fill_value=0)
    .values
)

dfg_cosine_similarity = cosine_similarity(
    [original_dfg_vector],
    [synthetic_dfg_vector]
)[0][0]


# =========================
# 9. Extra DFG overlap metrics
# =========================

original_dfg_edges = set(original_dfg_dist.index)
synthetic_dfg_edges = set(synthetic_dfg_dist.index)

common_dfg_edges = original_dfg_edges & synthetic_dfg_edges
all_edges = original_dfg_edges | synthetic_dfg_edges

dfg_edge_jaccard_similarity = (
    len(common_dfg_edges) / len(all_edges)
    if len(all_edges) > 0 else 0
)

dfg_edge_coverage_original = (
    len(common_dfg_edges) / len(original_dfg_edges)
    if len(original_dfg_edges) > 0 else 0
)

dfg_edge_extra_synthetic_ratio = (
    len(synthetic_dfg_edges - original_dfg_edges) / len(synthetic_dfg_edges)
    if len(synthetic_dfg_edges) > 0 else 0
)


# =========================
# 10. Results table
# =========================

results = pd.DataFrame({
    "Metric": [
        "Cosine similarity of event distribution",
        "DFG cosine similarity",
        "DFG edge Jaccard similarity",
        "DFG edge coverage of original",
        "Extra synthetic DFG edge ratio",
        "Wasserstein distance of case durations in hours",
        "Jensen-Shannon divergence of event distribution"
    ],
    "Value": [
        activity_cosine_similarity,
        dfg_cosine_similarity,
        dfg_edge_jaccard_similarity,
        dfg_edge_coverage_original,
        dfg_edge_extra_synthetic_ratio,
        case_duration_wasserstein,
        js_divergence
    ]
})

results.to_csv(
    os.path.join(output_folder, "comparison_metrics.csv"),
    index=False
)

print("\nComparison metrics:")
print(results.to_string(index=False))


# =========================
# 11. Save detailed activity comparison
# =========================

activity_comparison = pd.DataFrame({
    "activity": all_activities,
    "original_percentage": original_activity_vector * 100,
    "synthetic_percentage": synthetic_activity_vector * 100,
    "absolute_difference": np.abs(
        original_activity_vector - synthetic_activity_vector
    ) * 100
})

activity_comparison = activity_comparison.sort_values(
    "absolute_difference",
    ascending=False
)

activity_comparison.to_csv(
    os.path.join(output_folder, "activity_distribution_comparison.csv"),
    index=False
)


# =========================
# 12. Save detailed DFG comparison
# =========================

dfg_comparison = pd.DataFrame({
    "dfg_edge": [f"{edge[0]} -> {edge[1]}" for edge in all_dfg_edges],
    "original_percentage": original_dfg_vector * 100,
    "synthetic_percentage": synthetic_dfg_vector * 100,
    "absolute_difference": np.abs(
        original_dfg_vector - synthetic_dfg_vector
    ) * 100
})

dfg_comparison = dfg_comparison.sort_values(
    "absolute_difference",
    ascending=False
)

dfg_comparison.to_csv(
    os.path.join(output_folder, "dfg_distribution_comparison.csv"),
    index=False
)


# =========================
# 13. Save case duration comparison
# =========================

duration_comparison = pd.DataFrame({
    "original_case_duration_hours": pd.Series(original_case_durations.values),
    "synthetic_case_duration_hours": pd.Series(synthetic_case_durations.values)
})

duration_comparison.to_csv(
    os.path.join(output_folder, "case_duration_comparison.csv"),
    index=False
)


print(f"\nAll comparison results saved in folder: {output_folder}")