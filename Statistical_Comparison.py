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

    print(f"\nLoading synthetic CSV: {path}")
    print(f"Raw rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    required_columns = [
        "Case ID",
        "Activity",
        "Timestamp"
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

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    if df["Timestamp"].isna().any():
        bad_rows = df[df["Timestamp"].isna()]
        raise ValueError(
            f"Some timestamps could not be parsed:\n{bad_rows}"
        )

    event_rows = []

    for _, row in df.iterrows():
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

    expanded_df = pd.DataFrame(event_rows)

    expanded_df["time:timestamp"] = pd.to_datetime(
        expanded_df["time:timestamp"],
        utc=True,
        errors="coerce"
    )

    expanded_df = expanded_df.sort_values(
        ["case:concept:name", "time:timestamp"]
    )

    print(f"Expanded rows: {len(expanded_df)}")
    print(f"Traces: {expanded_df['case:concept:name'].nunique()}")
    print(f"Events: {len(expanded_df)}")
    print(f"Activities: {sorted(expanded_df['concept:name'].unique())}")

    return expanded_df


original_df = load_original_xes(original_path)
synthetic_df = load_synthetic_csv(synthetic_path)

# =========================
# 4. Load already mined models
# =========================

# IMPORTANT:
# DFG should be saved as .dfg, not .xml
original_dfg_path = "pm4py_outputs_inductive/discovery/dfg.dfg"
synthetic_dfg_path = "pm4py_outputs_inductive/discovery_synthetic/dfg.dfg"

original_petri_path = "pm4py_outputs_inductive/discovery/petri_net.pnml"
synthetic_petri_path = "pm4py_outputs_inductive/discovery_synthetic/petri_net.pnml"

# Process tree should be saved as .ptml, not .txt
original_tree_path = "pm4py_outputs_inductive/discovery/process_tree.ptml"
synthetic_tree_path = "pm4py_outputs_inductive/discovery_synthetic/process_tree.ptml"

original_bpmn_path = "pm4py_outputs_inductive/discovery/process_model.bpmn"
synthetic_bpmn_path = "pm4py_outputs_inductive/discovery_synthetic/process_model.bpmn"


# Load DFGs
original_dfg, original_start_activities, original_end_activities = pm4py.read_dfg(
    original_dfg_path
)

synthetic_dfg, synthetic_start_activities, synthetic_end_activities = pm4py.read_dfg(
    synthetic_dfg_path
)


# Load Petri nets
original_net, original_im, original_fm = pm4py.read_pnml(
    original_petri_path
)

synthetic_net, synthetic_im, synthetic_fm = pm4py.read_pnml(
    synthetic_petri_path
)


# Load process trees
original_tree = pm4py.read_ptml(
    original_tree_path
)

synthetic_tree = pm4py.read_ptml(
    synthetic_tree_path
)


# Load BPMN models
original_bpmn = pm4py.read_bpmn(
    original_bpmn_path
)

synthetic_bpmn = pm4py.read_bpmn(
    synthetic_bpmn_path
)

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

# =========================
# DFG similarity using already mined DFGs
# =========================




def normalize_dfg(dfg):
    """
    Converts a mined DFG into a normalized frequency distribution.

    Expected DFG format:
    {
        ("Activity A", "Activity B"): count,
        ("Activity B", "Activity C"): count,
        ...
    }
    """

    if len(dfg) == 0:
        return pd.Series(dtype=float)

    dfg_series = pd.Series(dfg, dtype=float)

    total = dfg_series.sum()

    if total == 0:
        return pd.Series(dtype=float)

    return dfg_series / total


original_dfg_dist = normalize_dfg(original_dfg)
synthetic_dfg_dist = normalize_dfg(synthetic_dfg)

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
# Petri net structural comparison
# =========================

def get_petri_net_structure(net):
    places = {p.name for p in net.places}

    transitions = {
        t.label for t in net.transitions
        if t.label is not None
    }

    silent_transitions = {
        t.name for t in net.transitions
        if t.label is None
    }

    arcs = {
        (arc.source.name, arc.target.name)
        for arc in net.arcs
    }

    return places, transitions, silent_transitions, arcs


def jaccard_similarity(set_a, set_b):
    union = set_a | set_b

    if len(union) == 0:
        return 1.0

    return len(set_a & set_b) / len(union)


original_places, original_transitions, original_silent, original_arcs = get_petri_net_structure(original_net)
synthetic_places, synthetic_transitions, synthetic_silent, synthetic_arcs = get_petri_net_structure(synthetic_net)

petri_transition_jaccard = jaccard_similarity(
    original_transitions,
    synthetic_transitions
)

petri_arc_jaccard = jaccard_similarity(
    original_arcs,
    synthetic_arcs
)

petri_place_count_difference = abs(
    len(original_places) - len(synthetic_places)
)

petri_transition_count_difference = abs(
    len(original_transitions) - len(synthetic_transitions)
)

petri_arc_count_difference = abs(
    len(original_arcs) - len(synthetic_arcs)
)

# =========================
# Process tree structural comparison
# =========================

def get_process_tree_labels_and_operators(tree):
    labels = []
    operators = []

    def visit(node):
        if node.label is not None:
            labels.append(node.label)

        if node.operator is not None:
            operators.append(str(node.operator))

        for child in node.children:
            visit(child)

    visit(tree)

    return labels, operators


original_tree_labels, original_tree_operators = get_process_tree_labels_and_operators(original_tree)
synthetic_tree_labels, synthetic_tree_operators = get_process_tree_labels_and_operators(synthetic_tree)

process_tree_label_jaccard = jaccard_similarity(
    set(original_tree_labels),
    set(synthetic_tree_labels)
)

process_tree_operator_jaccard = jaccard_similarity(
    set(original_tree_operators),
    set(synthetic_tree_operators)
)

process_tree_size_difference = abs(
    len(original_tree_labels) + len(original_tree_operators)
    - len(synthetic_tree_labels) - len(synthetic_tree_operators)
)

# =========================
# BPMN structural comparison
# =========================

def get_bpmn_structure(bpmn_graph):
    nodes = set()
    activities = set()
    flows = set()

    for node in bpmn_graph.get_nodes():
        nodes.add(str(node))

        if hasattr(node, "get_name"):
            name = node.get_name()
            if name is not None:
                activities.add(name)

    for flow in bpmn_graph.get_flows():
        source = str(flow.get_source())
        target = str(flow.get_target())
        flows.add((source, target))

    return nodes, activities, flows


original_bpmn_nodes, original_bpmn_activities, original_bpmn_flows = get_bpmn_structure(original_bpmn)
synthetic_bpmn_nodes, synthetic_bpmn_activities, synthetic_bpmn_flows = get_bpmn_structure(synthetic_bpmn)

bpmn_activity_jaccard = jaccard_similarity(
    original_bpmn_activities,
    synthetic_bpmn_activities
)

bpmn_flow_jaccard = jaccard_similarity(
    original_bpmn_flows,
    synthetic_bpmn_flows
)

bpmn_node_count_difference = abs(
    len(original_bpmn_nodes) - len(synthetic_bpmn_nodes)
)

bpmn_flow_count_difference = abs(
    len(original_bpmn_flows) - len(synthetic_bpmn_flows)
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
        "Jensen-Shannon divergence of event distribution",

        "Petri net transition Jaccard similarity",
        "Petri net arc Jaccard similarity",
        "Petri net place count difference",
        "Petri net transition count difference",
        "Petri net arc count difference",

        "Process tree activity-label Jaccard similarity",
        "Process tree operator Jaccard similarity",
        "Process tree size difference",

        "BPMN activity Jaccard similarity",
        "BPMN flow Jaccard similarity",
        "BPMN node count difference",
        "BPMN flow count difference"
    ],
    "Value": [
        activity_cosine_similarity,
        dfg_cosine_similarity,
        dfg_edge_jaccard_similarity,
        dfg_edge_coverage_original,
        dfg_edge_extra_synthetic_ratio,
        case_duration_wasserstein,
        js_divergence,

        petri_transition_jaccard,
        petri_arc_jaccard,
        petri_place_count_difference,
        petri_transition_count_difference,
        petri_arc_count_difference,

        process_tree_label_jaccard,
        process_tree_operator_jaccard,
        process_tree_size_difference,

        bpmn_activity_jaccard,
        bpmn_flow_jaccard,
        bpmn_node_count_difference,
        bpmn_flow_count_difference
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