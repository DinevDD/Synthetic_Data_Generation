#!/usr/bin/env python3
"""
run_gemini_pipeline.py
======================
Orchestrates the full process-mining pipeline in the correct order,
redirecting every input and output to the Gemini folder tree.

Execution order
---------------
  1. Discovery_Synthetic_Model.py
  2. Visualization_Synthetic_Data.py
  3. Conformance_Checking.py
  4. Conformance_Synthetic.py
  5. Statistical_Comparison.py

Gemini folder layout
--------------------
  Gemini/
  ├── TS1.csv              ← synthetic input log  (CSV, stages 1 2 3 5)
  ├── filtered_log.xes.gz  ← original/final log   (XES, stages 4 & 5)
  ├── Discovery/            ← discovery outputs
  ├── Visualizations/       ← visualisation outputs
  ├── Conformance/          ← conformance outputs
  └── Statistics/           ← statistical-comparison outputs

Usage
-----
  python run_gemini_pipeline.py

  Run a single stage (1-based index):
  python run_gemini_pipeline.py --stage 3

  Skip a stage:
  python run_gemini_pipeline.py --skip 2
"""

import argparse
import os
import sys
import textwrap
import time
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Anchor every path to the directory that contains THIS script, so the
# pipeline works correctly regardless of what PyCharm or the terminal sets
# as the current working directory.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
GEMINI_ROOT = os.path.join(SCRIPT_DIR, "Gemini")

# Output sub-directories
OUT_DISCOVERY      = os.path.join(GEMINI_ROOT, "Discovery")
OUT_VISUALIZATIONS = os.path.join(GEMINI_ROOT, "Visualizations")
OUT_CONFORMANCE    = os.path.join(GEMINI_ROOT, "Conformance")
OUT_STATISTICS     = os.path.join(GEMINI_ROOT, "Statistics")

# Input files (must be placed here by the user before running)
IN_CSV          = os.path.join(GEMINI_ROOT, "TS3.csv")            # synthetic log
IN_XES_FILTERED = os.path.join(GEMINI_ROOT, "filtered_log.xes.gz")  # original/final log

# ─────────────────────────────────────────────────────────────────────────────
# PATH-PATCH MAPS
# Each map is an ordered list of (old_string, new_string) tuples.
# Replacements happen in order – put more-specific strings first.
# ─────────────────────────────────────────────────────────────────────────────

# HOW PATCHES WORK
# -----------------
# Each entry is (find_string, replace_string).
# The find_string is EXACTLY what appears hardcoded in the original script.
# It is never used as a path — it is only searched for so it can be swapped
# out for the correct Gemini path on the right-hand side.

PATCHES = {
    # ── 1. Discovery_Synthetic_Model ──────────────────────────────────────────
    # "Data/data.csv" is the hardcoded string inside the original script;
    # it gets replaced with IN_CSV (Gemini/TS1.csv).
    # DFG_GUARD skips the visualisation when the DFG has no edges
    # (e.g. TS1.csv has one event per case → zero DFG transitions).
    "discovery": [
        ("Data/data.csv",                               IN_CSV),
        ("pm4py_outputs_inductive/discovery_synthetic", OUT_DISCOVERY),
        ("pm4py.save_vis_dfg(dfg, start_acts, end_acts, dfg_img_path)",
         "if dfg:\n        pm4py.save_vis_dfg(dfg, start_acts, end_acts, dfg_img_path)"
         "\n    else:\n        print('    [!] DFG empty — skipping DFG visualisation')"),
    ],

    # ── 2. Visualization_Synthetic_Data ──────────────────────────────────────
    "visualization": [
        ("Data/data.csv",            IN_CSV),
        ("visualizations_synthetic", OUT_VISUALIZATIONS),
    ],

    # ── 3. Conformance_Checking ───────────────────────────────────────────────
    # The script uses TS1.csv (synthetic) to verify it fits its own model.
    # "pm4py_outputs_inductive/discovery" must be replaced BEFORE
    # "pm4py_outputs_inductive/conformance" to avoid a partial match.
    "conformance": [
        ("Data/data.csv",                        IN_CSV),
        ("pm4py_outputs_inductive/discovery",    OUT_DISCOVERY),
        ("pm4py_outputs_inductive/conformance",  OUT_CONFORMANCE),
    ],

    # ── 4. Conformance_Synthetic ──────────────────────────────────────────────
    # Checks the original log (filtered_log.xes.gz) against the synthetic model.
    "conformance_synthetic": [
        ("Data/filtered_log.xes.gz",                       IN_XES_FILTERED),
        ("pm4py_outputs_inductive/discovery_synthetic",    OUT_DISCOVERY),
        ("pm4py_outputs_inductive/conformance_synhtetic",  OUT_CONFORMANCE),
    ],

    # ── 5. Statistical_Comparison ─────────────────────────────────────────────
    # "Data/Sepsis Cases…" is the hardcoded original-log path in the script;
    # it is replaced with filtered_log.xes.gz (the actual original log).
    # "Data/data.csv" is the hardcoded synthetic path; replaced with TS1.csv.
    # discovery_synthetic must come before discovery to avoid partial match.
    "statistics": [
        ("Data/Sepsis Cases - Event Log.xes.gz",           IN_XES_FILTERED),
        ("Data/data.csv",                                  IN_CSV),
        ("comparison_results",                             OUT_STATISTICS),
        ("pm4py_outputs_inductive/discovery_synthetic",    OUT_DISCOVERY),
        ("pm4py_outputs_inductive/discovery",              OUT_DISCOVERY),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT SOURCE PATHS
# Adjust these if your scripts live somewhere other than the current directory.
# ─────────────────────────────────────────────────────────────────────────────

# Script paths are resolved relative to THIS file so they are found
# regardless of the working directory when the runner is launched.
SCRIPT_FILES = {
    "discovery":             os.path.join(SCRIPT_DIR, "Discovery_Synthetic_Model.py"),
    "visualization":         os.path.join(SCRIPT_DIR, "Visualization_Synthetic_Data.py"),
    "conformance":           os.path.join(SCRIPT_DIR, "Conformance_Checking.py"),
    "conformance_synthetic": os.path.join(SCRIPT_DIR, "Conformance_Synthetic.py"),
    "statistics":            os.path.join(SCRIPT_DIR, "Statistical_Comparison.py"),
}

# Ordered pipeline stages
PIPELINE = [
    ("discovery",             "Discovery_Synthetic_Model"),
    ("visualization",         "Visualization_Synthetic_Data"),
    ("conformance",           "Conformance_Checking"),
    ("conformance_synthetic", "Conformance_Synthetic"),
    ("statistics",            "Statistical_Comparison"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "═", width: int = 70) -> None:
    line = char * width
    print(f"\n{line}")
    for row in textwrap.wrap(text, width - 4):
        print(f"  {row}")
    print(f"{line}")


def make_output_dirs() -> None:
    for d in [OUT_DISCOVERY, OUT_VISUALIZATIONS, OUT_CONFORMANCE, OUT_STATISTICS]:
        os.makedirs(d, exist_ok=True)
        print(f"  [dir] {d}")


def check_inputs() -> list[str]:
    """Return a list of missing input files (warnings only)."""
    missing = []
    required = {
        "stages 1, 2, 3 & 5": IN_CSV,
        "stages 4 & 5":      IN_XES_FILTERED,
    }
    for label, path in required.items():
        if not os.path.exists(path):
            missing.append(f"  [MISSING] {path}  (needed for {label})")
    return missing


def apply_patches(source: str, patch_list: list[tuple[str, str]]) -> str:
    # Replace each old string with its new value inside source code.
    # Backslashes in the replacement are converted to forward slashes so
    # that Windows absolute paths do not cause a SyntaxError when injected
    # into a Python string literal (backslash starts an escape sequence).
    # Forward slashes are valid path separators on Windows.
    for old, new in patch_list:
        new_safe = new.replace(chr(92), chr(47))
        source = source.replace(old, new_safe)
    return source

def run_stage(key: str, label: str, script_path: str) -> bool:
    """
    Load the script, patch all hardcoded paths, then exec() it.
    Returns True on success, False on failure.
    """
    if not os.path.exists(script_path):
        print(f"\n  [ERROR] Script not found: {script_path}")
        return False

    with open(script_path, "r", encoding="utf-8") as fh:
        source = fh.read()

    patched_source = apply_patches(source, PATCHES[key])

    # Provide a fresh __main__ namespace so module-level code is isolated
    ns = {
        "__name__":    "__main__",
        "__file__":    script_path,
        "__builtins__": __builtins__,
    }

    t0 = time.perf_counter()
    try:
        exec(compile(patched_source, script_path, "exec"), ns)   # noqa: S102
    except SystemExit:
        pass   # some scripts call sys.exit(0) on clean completion
    except Exception:
        print(f"\n  [ERROR] Stage '{label}' raised an exception:")
        traceback.print_exc()
        return False

    elapsed = time.perf_counter() - t0
    print(f"\n  [✓] Stage '{label}' completed in {elapsed:.1f}s")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Gemini process-mining pipeline."
    )
    parser.add_argument(
        "--stage", type=int, default=None,
        help="Run only this stage (1-based). Default: run all stages."
    )
    parser.add_argument(
        "--skip", type=int, default=None,
        help="Skip this stage (1-based). Default: skip nothing."
    )
    args = parser.parse_args()

    banner("Gemini Process-Mining Pipeline Runner", char="█")

    # ── Output directories ────────────────────────────────────────────────────
    print("\n[Setup] Creating output directories …")
    make_output_dirs()

    # ── Input file warnings ───────────────────────────────────────────────────
    missing = check_inputs()
    if missing:
        print("\n[Warning] The following input files were not found.")
        print("  Stages that depend on them will fail unless files exist at run time.")
        for m in missing:
            print(m)

    # ── Stage selection ───────────────────────────────────────────────────────
    stages_to_run = []
    for idx, (key, label) in enumerate(PIPELINE, start=1):
        if args.stage is not None and idx != args.stage:
            continue
        if args.skip is not None and idx == args.skip:
            print(f"\n  [SKIP] Stage {idx}: {label}")
            continue
        stages_to_run.append((idx, key, label))

    if not stages_to_run:
        print("\n  No stages selected. Exiting.")
        sys.exit(0)

    # ── Run ───────────────────────────────────────────────────────────────────
    results = {}
    overall_start = time.perf_counter()

    for idx, key, label in stages_to_run:
        script_path = SCRIPT_FILES[key]
        banner(f"Stage {idx}/5 — {label}")
        print(f"  Script : {script_path}")
        print(f"  Patches applied:")
        for old, new in PATCHES[key]:
            print(f"    {old!r:55s} → {new!r}")
        print()

        ok = run_stage(key, label, script_path)
        results[label] = "✓ OK" if ok else "✗ FAILED"

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - overall_start
    banner("Pipeline Summary", char="═")
    for label, status in results.items():
        print(f"  {status}  {label}")
    print(f"\n  Total wall-clock time : {total_elapsed:.1f}s")
    print(f"\n  Output locations:")
    print(f"    Discovery      → {OUT_DISCOVERY}")
    print(f"    Visualizations → {OUT_VISUALIZATIONS}")
    print(f"    Conformance    → {OUT_CONFORMANCE}")
    print(f"    Statistics     → {OUT_STATISTICS}")

    if any("FAILED" in s for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()