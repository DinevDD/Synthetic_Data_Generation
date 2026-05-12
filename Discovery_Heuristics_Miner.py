"""
Process Discovery Pipeline  (run once)
=======================================
Steps:
  1. Import XES log
  2. Discover model with Heuristics Miner  →  Petri Net + Process Tree
  3. Export Petri Net (PNML), BPMN (XML), DFG (JSON + XML)
  4. Save PNG visualisations
"""

import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

# --- Discovery
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

# --- Export: Petri Net
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

# --- Export: BPMN
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter

# --- Export: DFG
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.start_activities.log import get as start_act_get
from pm4py.statistics.end_activities.log import get as end_act_get

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
XES_LOG_PATH = "Data/Sepsis Cases - Event Log.xes.gz"
OUTPUT_DIR   = "pm4py_outputs_heuristics/discovery"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════
# PART 1 – IMPORT XES LOG
# ══════════════════════════════════════════════
def import_xes_log(path: str):
    print(f"\n{'='*60}")
    print(f"[1] Importing XES log: {path}")
    print('='*60)
    log = xes_importer.apply(path)
    print(f"    Traces     : {len(log)}")
    print(f"    Events     : {sum(len(t) for t in log)}")
    activities = set(e["concept:name"] for t in log for e in t)
    print(f"    Activities ({len(activities)}): {sorted(activities)}")
    return log


# ══════════════════════════════════════════════
# PART 2 – HEURISTICS MINER → PETRI NET
# ══════════════════════════════════════════════
def discover_model(log):
    print(f"\n{'=' * 60}")
    print("[2] Running Heuristics Miner …")
    print('=' * 60)

    # The Heuristics Miner in pm4py directly returns a Petri Net tuple
    # (PetriNet, initial_marking, final_marking)
    net, initial_marking, final_marking = heuristics_miner.apply(log)

    print(f"    Petri Net — places      : {len(net.places)}")
    print(f"    Petri Net — transitions : {len(net.transitions)}")
    print(f"    Petri Net — arcs        : {len(net.arcs)}")

    # We return None for the process_tree so it still unpacks correctly in your main()
    return None, net, initial_marking, final_marking


# ══════════════════════════════════════════════
# PART 3 – EXPORT MODELS
# ══════════════════════════════════════════════
def export_models(log, net, initial_marking, final_marking):
    print(f"\n{'='*60}")
    print("[3] Exporting models & visualisations …")
    print('='*60)

    # ── 3a. Petri Net → PNML & PNG ────────────
    pnml_path = os.path.join(OUTPUT_DIR, "petri_net.pnml")
    pnml_exporter.apply(net, initial_marking, pnml_path,
                        final_marking=final_marking)
    print(f"    [✓] Petri Net (PNML) saved → {pnml_path}")

    pn_img_path = os.path.join(OUTPUT_DIR, "petri_net.png")
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, pn_img_path)
    print(f"    [✓] Petri Net (PNG)  saved → {pn_img_path}")

    # ── 3b. BPMN → XML & PNG ──────────────────
    bpmn_path = os.path.join(OUTPUT_DIR, "process_model.bpmn")
    bpmn_graph = pm4py.convert_to_bpmn(net, initial_marking, final_marking)
    bpmn_exporter.apply(bpmn_graph, bpmn_path)
    print(f"    [✓] BPMN (XML)       saved → {bpmn_path}")

    bpmn_img_path = os.path.join(OUTPUT_DIR, "process_model.png")
    pm4py.save_vis_bpmn(bpmn_graph, bpmn_img_path)
    print(f"    [✓] BPMN (PNG)       saved → {bpmn_img_path}")

    # ── 3c. DFG → JSON, XML & PNG ─────────────
    dfg        = dfg_discovery.apply(log)
    activities = pm4py.get_event_attribute_values(log, "concept:name")
    start_acts = start_act_get.get_start_activities(log)
    end_acts   = end_act_get.get_end_activities(log)

    dfg_img_path = os.path.join(OUTPUT_DIR, "dfg.png")
    pm4py.save_vis_dfg(dfg, start_acts, end_acts, dfg_img_path)
    print(f"    [✓] DFG (PNG)        saved → {dfg_img_path}")

    # JSON
    dfg_json_path = os.path.join(OUTPUT_DIR, "dfg.json")
    dfg_payload = {
        "activities"      : {act: cnt for act, cnt in activities.items()},
        "start_activities": start_acts,
        "end_activities"  : end_acts,
        "edges"           : {f"{src}||{tgt}": cnt
                             for (src, tgt), cnt in dfg.items()},
    }
    with open(dfg_json_path, "w", encoding="utf-8") as fh:
        json.dump(dfg_payload, fh, indent=2, ensure_ascii=False)
    print(f"    [✓] DFG (JSON)       saved → {dfg_json_path}")

    # XML
    root     = ET.Element("directlyFollowsGraph")
    acts_el  = ET.SubElement(root, "activities")
    for act, cnt in sorted(activities.items()):
        a = ET.SubElement(acts_el, "activity",
                          count=str(cnt),
                          isStart=str(act in start_acts).lower(),
                          isEnd=str(act in end_acts).lower())
        a.text = act
    edges_el = ET.SubElement(root, "edges")
    for (src, tgt), cnt in sorted(dfg.items(), key=lambda x: -x[1]):
        ET.SubElement(edges_el, "edge",
                      source=src, target=tgt, count=str(cnt))
    raw_xml = ET.tostring(root, encoding="unicode")
    pretty  = minidom.parseString(raw_xml).toprettyxml(indent="  ")
    pretty  = "\n".join(pretty.split("\n")[1:])          # drop XML declaration
    dfg_xml_path = os.path.join(OUTPUT_DIR, "dfg.xml")
    with open(dfg_xml_path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="utf-8"?>\n')
        fh.write(pretty)
    print(f"    [✓] DFG (XML)        saved → {dfg_xml_path}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    print("\n" + "█" * 60)
    print("  Process Discovery Pipeline")
    print("█" * 60)

    log = import_xes_log(XES_LOG_PATH)
    _, net, im, fm = discover_model(log)
    export_models(log, net, im, fm)

    print("\n" + "█" * 60)
    print(f"  Discovery complete.  All outputs → {OUTPUT_DIR}/")
    print("  Run conformance.py next.")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
