"""
Conformance Checking Pipeline
==============================
Prerequisite: run discovery.py first so the Petri Net (PNML) exists on disk.

Steps:
  1. Load Petri Net from PNML  (no rediscovery)
  2. Import CSV event log
  3. Token-Based Replay
  4. Alignment-Based Conformance
  5. Print all metrics and save conformance_report.json
"""

import os
import json
import pandas as pd

import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

# --- Import: Petri Net
from pm4py.objects.petri_net.importer import importer as pnml_importer

# --- Conformance: Token-Based Replay
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

# --- Conformance: Alignments
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

# --- Metrics / Evaluation helpers
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CSV_LOG_PATH = "Data/data.csv"

OUTPUT_DIR   = "pm4py_outputs_inductive/conformance"   # inductive miner
PNML_PATH = os.path.join("pm4py_outputs_inductive/discovery", "petri_net.pnml") # inductive miner

# CSV column mapping (adjust to your column names)
CSV_CASE_ID_COL   = "case:concept:name"
CSV_ACTIVITY_COL  = "concept:name"
CSV_TIMESTAMP_COL = "time:timestamp"


# ══════════════════════════════════════════════
# PART 1 – LOAD PETRI NET FROM DISK
# ══════════════════════════════════════════════
def load_petri_net(path: str):
    print(f"\n{'='*60}")
    print(f"[1] Loading Petri Net from: {path}")
    print('='*60)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PNML file not found: {path}\n"
            "  → Please run discovery.py first."
        )

    net, initial_marking, final_marking = pnml_importer.apply(path)
    print(f"    Places      : {len(net.places)}")
    print(f"    Transitions : {len(net.transitions)}")
    print(f"    Arcs        : {len(net.arcs)}")
    return net, initial_marking, final_marking


# ══════════════════════════════════════════════
# PART 2 – IMPORT CSV LOG
# ══════════════════════════════════════════════
def import_csv_log(path: str):
    print(f"\n{'='*60}")
    print(f"[2] Importing CSV log: {path}")
    print('='*60)

    df = pd.read_csv(path)
    print(f"    Raw rows  : {len(df)}")
    print(f"    Columns   : {list(df.columns)}")

    # ── Auto-detect the three mandatory columns ───────────────────
    CASE_ALIASES = [CSV_CASE_ID_COL,
                    "case:concept:name", "case_id", "caseid",
                    "Case ID", "CaseID", "trace", "Trace", "CaseId"]
    ACT_ALIASES  = [CSV_ACTIVITY_COL,
                    "concept:name", "activity", "Activity",
                    "ActivityID", "task", "Task", "event", "Event",
                    "action", "Action"]
    TIME_ALIASES = [CSV_TIMESTAMP_COL,
                    "time:timestamp", "timestamp", "Timestamp",
                    "time", "Time", "datetime", "date", "Date",
                    "start_time", "StartTime", "completeTime",
                    "Complete Timestamp", "start timestamp"]

    def pick_col(aliases, df_cols, label):
        cols_lower = {c.lower(): c for c in df_cols}
        for alias in aliases:
            if alias in df_cols:
                return alias
            if alias.lower() in cols_lower:
                return cols_lower[alias.lower()]
        raise KeyError(
            f"\nCould not auto-detect the {label} column.\n"
            f"  Available columns : {list(df_cols)}\n"
            f"  Set the matching name in the CONFIG section at the top of the script."
        )

    case_col = pick_col(CASE_ALIASES, df.columns, "Case ID")
    act_col  = pick_col(ACT_ALIASES,  df.columns, "Activity")
    time_col = pick_col(TIME_ALIASES, df.columns, "Timestamp")
    print(f"    Mapped    : case='{case_col}'  activity='{act_col}'  time='{time_col}'")

    # Rename to PM4Py standard names
    rename_map = {}
    if case_col != "case:concept:name": rename_map[case_col] = "case:concept:name"
    if act_col  != "concept:name":      rename_map[act_col]  = "concept:name"
    if time_col != "time:timestamp":    rename_map[time_col] = "time:timestamp"
    if rename_map:
        df = df.rename(columns=rename_map)

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["case:concept:name", "time:timestamp"])

    log = log_converter.apply(
        df,
        parameters={log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY:
                    "case:concept:name"},
    )
    print(f"    Traces    : {len(log)}")
    print(f"    Events    : {sum(len(t) for t in log)}")
    activities = set(e["concept:name"] for t in log for e in t)
    print(f"    Activities: {sorted(activities)}")
    return log


# ══════════════════════════════════════════════
# PART 3 – TOKEN-BASED REPLAY
# ══════════════════════════════════════════════
def run_token_replay(log, net, initial_marking, final_marking):
    print(f"\n{'='*60}")
    print("[3] Token-Based Replay …")
    print('='*60)

    replayed = token_replay.apply(log, net, initial_marking, final_marking)

    print("\n    Per-trace summary (first 10 traces):")
    print(f"    {'Trace':>6}  {'Fit':>6}  {'Consumed':>9}  "
          f"{'Produced':>9}  {'Missing':>8}  {'Remaining':>10}")
    print("    " + "-" * 58)
    for i, r in enumerate(replayed[:10]):
        print(f"    {i:>6}  {r['trace_fitness']:>6.3f}  "
              f"{r['consumed_tokens']:>9}  {r['produced_tokens']:>9}  "
              f"{r['missing_tokens']:>8}  {r['remaining_tokens']:>10}")

    fitness_tbr = pm4py.fitness_token_based_replay(
        log, net, initial_marking, final_marking
    )
    print(f"\n    [Token-Based Fitness]")
    for k, v in fitness_tbr.items():
        print(f"      {k:<35}: {v:.4f}" if isinstance(v, float)
              else f"      {k:<35}: {v}")

    return replayed, fitness_tbr


# ══════════════════════════════════════════════
# PART 4 – ALIGNMENT-BASED CONFORMANCE
# ══════════════════════════════════════════════
def run_alignments(log, net, initial_marking, final_marking):
    print(f"\n{'='*60}")
    print("[4] Alignment-Based Conformance …")
    print('='*60)

    aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)

    print("\n    Per-trace summary (first 10 traces):")
    print(f"    {'Trace':>6}  {'Cost':>8}  {'BWC':>8}  {'Fitness':>8}  "
          "Alignment (sync moves shown as activity name)")
    print("    " + "-" * 80)
    for i, a in enumerate(aligned_traces[:10]):
        moves = []
        for (log_mv, model_mv) in a["alignment"]:
            if log_mv == model_mv:
                moves.append(str(log_mv))
            elif model_mv == ">>":
                moves.append(f"LOG:{log_mv}")
            else:
                moves.append(f"MDL:{model_mv}")
        alignment_str = " → ".join(moves)
        if len(alignment_str) > 50:
            alignment_str = alignment_str[:50] + "…"
        bwc     = a.get("bwc", "?")
        fitness = a.get("fitness", "?")
        fit_str = f"{fitness:.4f}" if isinstance(fitness, float) else str(fitness)
        print(f"    {i:>6}  {a['cost']:>8}  {str(bwc):>8}  {fit_str:>8}  {alignment_str}")

    fitness_align = pm4py.fitness_alignments(
        log, net, initial_marking, final_marking
    )
    print(f"\n    [Alignment-Based Fitness]")
    for k, v in fitness_align.items():
        print(f"      {k:<35}: {v:.4f}" if isinstance(v, float)
              else f"      {k:<35}: {v}")

    return aligned_traces, fitness_align


# ══════════════════════════════════════════════
# PART 5 – ALL CONFORMANCE METRICS
# ══════════════════════════════════════════════
def compute_all_metrics(log, net, initial_marking, final_marking,
                        replayed_traces, aligned_traces,
                        fitness_tbr, fitness_align):
    print(f"\n{'='*60}")
    print("[5] Full Conformance Metrics Report")
    print('='*60)

    precision     = precision_evaluator.apply(
        log, net, initial_marking, final_marking,
        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
    )
    generalization = generalization_evaluator.apply(
        log, net, initial_marking, final_marking
    )
    simplicity = simplicity_evaluator.apply(net)

    # Raw token counts from per-trace replay
    n_traces        = len(replayed_traces)
    total_missing   = sum(r["missing_tokens"]   for r in replayed_traces)
    total_remaining = sum(r["remaining_tokens"] for r in replayed_traces)
    total_consumed  = sum(r["consumed_tokens"]  for r in replayed_traces)
    total_produced  = sum(r["produced_tokens"]  for r in replayed_traces)
    n_fit_traces    = sum(
        1 for r in replayed_traces
        if r.get("missing_tokens", 1) == 0 and r.get("remaining_tokens", 1) == 0
    )
    avg_align_cost = sum(a["cost"] for a in aligned_traces) / len(aligned_traces)

    f_tbr   = fitness_tbr.get("log_fitness",
              fitness_tbr.get("average_trace_fitness", 0.0))
    f_align = fitness_align.get("log_fitness",
              fitness_align.get("averageFitness",
              fitness_align.get("average_trace_fitness", 0.0)))

    f1_tbr   = (2*f_tbr  *precision)/(f_tbr  +precision) if (f_tbr  +precision)>0 else 0.0
    f1_align = (2*f_align*precision)/(f_align+precision) if (f_align+precision)>0 else 0.0

    report = {
        **{f"tbr_{k}":   v for k, v in fitness_tbr.items()},
        **{f"align_{k}": v for k, v in fitness_align.items()},
        "tbr_total_traces"           : n_traces,
        "tbr_fitting_traces"         : n_fit_traces,
        "tbr_pct_fitting_traces"     : round(n_fit_traces / n_traces * 100, 2),
        "tbr_total_missing_tokens"   : total_missing,
        "tbr_total_remaining_tokens" : total_remaining,
        "tbr_total_consumed_tokens"  : total_consumed,
        "tbr_total_produced_tokens"  : total_produced,
        "align_avg_cost_per_trace"   : avg_align_cost,
        "precision"                  : precision,
        "generalization"             : generalization,
        "simplicity"                 : simplicity,
        "f1_tbr_fitness_x_precision" : f1_tbr,
        "f1_align_fitness_x_precision": f1_align,
    }

    max_key = max(len(k) for k in report)
    print(f"\n    {'METRIC':<{max_key}}  VALUE")
    print("    " + "-" * (max_key + 12))
    for k, v in report.items():
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:<{max_key}}  {val_str}")

    report_path = os.path.join(OUTPUT_DIR, "conformance_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n    [✓] Full report saved → {report_path}")

    return report


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█" * 60)
    print("  Conformance Checking Pipeline")
    print("█" * 60)

    # 1. Load pre-built model (no rediscovery)
    net, im, fm = load_petri_net(PNML_PATH)

    # 2. Import CSV event log
    log = import_csv_log(CSV_LOG_PATH)

    # 3. Token-based replay
    replayed_traces, fitness_tbr = run_token_replay(log, net, im, fm)

    # 4. Alignments
    aligned_traces, fitness_align = run_alignments(log, net, im, fm)

    # 5. All metrics
    compute_all_metrics(log, net, im, fm,
                        replayed_traces, aligned_traces,
                        fitness_tbr, fitness_align)

    print("\n" + "█" * 60)
    print(f"  Conformance complete.  Report → {OUTPUT_DIR}/conformance_report.json")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
