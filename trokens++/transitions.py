#!/usr/bin/env python3
"""
Compute and visualize behavior transition probabilities from BORIS-style CSV/TSV files.

For each annotated POINT event A, every POINT event B in the same video whose
timestamp falls in (t_A, t_A + WINDOW] is counted as an A -> B transition.
Probabilities are row-normalized: P(B | A) = count(A->B) / sum_over_B(count(A->B)).

Outputs: transition_probs_<WINDOW>sec.png
Requires: pandas, matplotlib, seaborn, numpy
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — edit these
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "/fs/vulcan-projects/fsh_track/raw_data"   # folder with CSV/TSV files
OUTPUT_FOLDER = "/fs/vulcan-projects/fsh_track/bhargav/results/transition_probs"  # where the PNG goes

TRANSITION_WINDOW = 2.0   # seconds — look this far ahead for the next behavior

# ─────────────────────────────────────────────────────────────────────────────
# Column name settings (mirrors original script)
# ─────────────────────────────────────────────────────────────────────────────
 
TIME_COLUMN_NAMES     = ['Time']
BEHAVIOR_COLUMN_NAMES = ['Behavior']
STATUS_COLUMN_NAMES   = ['Status', 'Behavior type']
 
# ─────────────────────────────────────────────────────────────────────────────
# Behavior normalization (mirrors original script)
# ─────────────────────────────────────────────────────────────────────────────
 
BEHAVIOR_NORMALIZATION = {
    "Peck":                 "Peck",
    "Quiver-m":             "Quiver",
    "Quiver (Male)":        "Quiver",
    "Bite (Male)":          "Bite",
    "Tilt (Specify Sex)":   "Tilt",
    "Peck (Specify Sex)":   "Peck",
    "Lead (Male)":          "Lead",
    "peck":                 "Peck",
    "Lead-m":               "Lead",
    "pot entry":            "Enter Pot",
    "Chase/Charge (Male)":  "Chase/Charge",
    "Run/Flee (Female)":    "Run/Flee",
    "male quiver":          "Quiver",
    "exit plot":            "Exit Plot",
    "male lead":            "Lead",
    "egg retrieval":        "Egg Retrieval",
    "circling":             "Circling",
    "female follow":        "Follow",
    "Follow-f":             "Follow",
    "Spawning-f":           "Spawning",
    "Follow (Female)":      "Follow",
}
 
for _canon in (
    "Peck", "Quiver", "Bite", "Tilt", "Lead", "Follow",
    "Chase/Charge", "Run/Flee", "Enter Pot", "Exit Plot",
    "Circling", "Egg Retrieval", "Spawning",
):
    BEHAVIOR_NORMALIZATION.setdefault(_canon, _canon)
 
_BEHAVIOR_NORMALIZATION_CI = {
    k.strip().lower(): v for k, v in BEHAVIOR_NORMALIZATION.items()
}
 
 
def normalize_behavior_label(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "", ""
    og = str(raw).strip()
    if not og:
        return "", ""
    canon = BEHAVIOR_NORMALIZATION.get(og)
    if canon is None:
        canon = _BEHAVIOR_NORMALIZATION_CI.get(og.lower())
    if canon is None:
        return og, og
    return canon, og
 
 
# ─────────────────────────────────────────────────────────────────────────────
# File discovery & CSV reading (mirrors original script)
# ─────────────────────────────────────────────────────────────────────────────
 
def find_csv_files(data_path):
    """Return all CSV and TSV files found recursively under data_path."""
    csv_files = glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True)
    tsv_files = glob.glob(os.path.join(data_path, "**", "*.tsv"), recursive=True)
    return csv_files + tsv_files
 
 
def read_timestamps(data_path,
                    time_column_names=None,
                    status_column_names=None,
                    behavior_column_names=None):
    """
    Read (timestamp, behavior_raw) pairs from a BORIS-style CSV/TSV.
    Returns list of (float_timestamp, str_behavior_raw) for POINT rows only.
    """
    if time_column_names     is None: time_column_names     = TIME_COLUMN_NAMES
    if status_column_names   is None: status_column_names   = STATUS_COLUMN_NAMES
    if behavior_column_names is None: behavior_column_names = BEHAVIOR_COLUMN_NAMES
 
    is_tsv    = os.path.splitext(data_path)[1].lower() == '.tsv'
    separator = '\t' if is_tsv else ','
 
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
 
        header_line = None
        for i, line in enumerate(lines):
            has_time     = any(col in line for col in time_column_names)
            has_status   = any(col in line for col in status_column_names)
            has_behavior = any(col in line for col in behavior_column_names)
            if has_time and has_status and has_behavior:
                header_line = i
                break
 
        if header_line is None:
            print(f"  [skip] could not find header row in {os.path.basename(data_path)}")
            return []
 
        df = pd.read_csv(data_path, skiprows=header_line, sep=separator)
 
        time_col = next((c for c in time_column_names     if c in df.columns), None)
        stat_col = next((c for c in status_column_names   if c in df.columns), None)
        beh_col  = next((c for c in behavior_column_names if c in df.columns), None)
 
        if not (time_col and stat_col and beh_col):
            print(f"  [skip] missing required columns in {os.path.basename(data_path)}: "
                  f"found {list(df.columns)}")
            return []
 
        point_rows = df[df[stat_col] == 'POINT']
        result = []
        for _, row in point_rows.iterrows():
            if pd.notna(row[time_col]) and pd.notna(row[beh_col]):
                try:
                    result.append((float(row[time_col]), row[beh_col]))
                except (TypeError, ValueError):
                    pass
        return result
 
    except Exception as e:
        print(f"  [error] reading {os.path.basename(data_path)}: {e}")
        return []
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Transition counting
# ─────────────────────────────────────────────────────────────────────────────
 
def count_transitions(all_events_by_file, window):
    """
    all_events_by_file: dict of filename -> sorted list of (timestamp, canonical_behavior)
    window: float, seconds
 
    Returns:
        counts       - dict-of-dicts: counts[behavior_A][behavior_B] = int
        none_counts  - dict: none_counts[behavior_A] = number of A occurrences with
                       nothing following within the window
        occurrences  - dict: occurrences[behavior_A] = total number of A events
    """
    counts      = defaultdict(lambda: defaultdict(int))
    none_counts = defaultdict(int)
    occurrences = defaultdict(int)
 
    for fname, events in all_events_by_file.items():
        n = len(events)
        for i, (t_a, beh_a) in enumerate(events):
            occurrences[beh_a] += 1
            # Only look at the very next event in time
            if i + 1 < n:
                t_b, beh_b = events[i + 1]
                if t_b - t_a <= window:
                    counts[beh_a][beh_b] += 1
                else:
                    none_counts[beh_a] += 1  # next event exists but is outside window
            else:
                none_counts[beh_a] += 1      # no next event at all
 
    return counts, none_counts, occurrences
 
 
def build_probability_matrix(counts, none_counts, occurrences):
    """
    Convert raw counts to a probability DataFrame normalized by total occurrences.
 
    Rows = source behavior (A), columns = destination behavior (B) + "None".
    P(B | A)    = count(A->B)  / occurrences[A]   (can exceed 1 across all B if
                                                    multiple events follow one A)
    P(None | A) = none_count[A] / occurrences[A]   (fraction of A events with
                                                    nothing following in window)
    """
    all_behaviors = sorted(
        set(occurrences.keys()) | set(b for inner in counts.values() for b in inner)
    )
    columns = all_behaviors + ["None"]
 
    mat = pd.DataFrame(0.0, index=all_behaviors, columns=columns)
 
    for beh_a in all_behaviors:
        n = occurrences.get(beh_a, 0)
        if n == 0:
            continue
        for beh_b, cnt in counts[beh_a].items():
            mat.loc[beh_a, beh_b] = cnt / n
        mat.loc[beh_a, "None"] = none_counts.get(beh_a, 0) / n
 
    return mat
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_heatmap(prob_matrix, window, output_path):
    n = len(prob_matrix)
    fig_size = max(8, n * 0.7)
 
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
 
    sns.heatmap(
        prob_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Transition probability", "shrink": 0.8},
    )
 
    ax.set_title(
        f"Behavior Transition Probabilities  (window = {window} s)\n"
        f"P(column behavior | row behavior occurred)",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel("Behavior B  (follows within window)", fontsize=11)
    ax.set_ylabel("Behavior A  (trigger)", fontsize=11)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    print("Behavior Transition Probability Analyzer")
    print("=" * 50)
    print(f"Data path    : {DATA_PATH}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Window       : {TRANSITION_WINDOW} s")
    print()
 
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: DATA_PATH not found: {DATA_PATH}")
        sys.exit(1)
 
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
 
    # ── Collect events ────────────────────────────────────────────────────────
    data_files = find_csv_files(DATA_PATH)
    print(f"Found {len(data_files)} data file(s).\n")
 
    all_events_by_file = {}
 
    for fpath in data_files:
        fname = os.path.basename(fpath)
        print(f"Reading: {fname}")
        raw_events = read_timestamps(fpath)
 
        if not raw_events:
            print(f"  → no POINT events found, skipping.\n")
            continue
 
        canonical_events = []
        for (ts, raw_beh) in raw_events:
            canon, og = normalize_behavior_label(raw_beh)
            if canon:
                canonical_events.append((ts, canon))
 
        # Sort by timestamp (BORIS exports are usually sorted, but be safe)
        canonical_events.sort(key=lambda x: x[0])
        all_events_by_file[fname] = canonical_events
        print(f"  → {len(canonical_events)} POINT events across "
              f"{len(set(b for _, b in canonical_events))} behavior(s).\n")
 
    if not all_events_by_file:
        print("No events found across any file. Exiting.")
        sys.exit(1)
 
    # ── Transition counts ─────────────────────────────────────────────────────
    print("Computing transitions …")
    counts, none_counts, occurrences = count_transitions(all_events_by_file, TRANSITION_WINDOW)
 
    if not occurrences:
        print("No events found (window may be too small, or only one event per file).")
        sys.exit(1)
 
    prob_matrix = build_probability_matrix(counts, none_counts, occurrences)
 
    # Print raw counts summary
    total_transitions = sum(sum(v.values()) for v in counts.values())
    print(f"Total (A→B) pairs counted: {total_transitions}")
    print(f"Behaviors detected: {list(prob_matrix.index)}\n")
 
    # ── Plot ──────────────────────────────────────────────────────────────────
    out_fname = f"transition_probs_{TRANSITION_WINDOW}sec.png"
    out_path  = os.path.join(OUTPUT_FOLDER, out_fname)
    plot_heatmap(prob_matrix, TRANSITION_WINDOW, out_path)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 
