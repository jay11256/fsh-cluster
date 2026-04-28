#!/usr/bin/env python3
#python ../bhargav/fsh-cluster/trokens++/replace_broken_pkls.py ./sam3pklds7/ --sub_directory ./cotrackpklds7/cotracker3_bip_fr_32_fps_10/fshdata/feat_dump/

import os
import shutil
import pickle
import argparse


def is_pred_tracks_empty(obj):
    """
    Returns True if pred_tracks exists and is empty.
    """
    if not isinstance(obj, dict):
        return False

    if "pred_tracks" not in obj:
        return False

    tensor = obj["pred_tracks"]

    # Ensure it's a torch tensor-like object
    try:
        return tensor.numel() == 0
    except AttributeError:
        return False


def replace_empty_files(flagged, input_dir, sub_directory):
    """
    For each flagged file, look for a matching file in sub_directory and
    overwrite the empty file in input_dir with it.
    """
    replaced = []
    missing = []

    for fname in flagged:
        src = os.path.join(sub_directory, fname)
        dst = os.path.join(input_dir, fname)

        if os.path.isfile(src):
            shutil.copy2(src, dst)
            replaced.append(fname)
            print(f"[REPLACED] {fname}")
        else:
            missing.append(fname)
            print(f"[MISSING]  {fname} not found in sub_directory — skipped")

    return replaced, missing


def main():
    parser = argparse.ArgumentParser(
        description="Flag PKL files with empty pred_tracks tensors, and optionally replace them."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing .pkl files to check",
    )
    parser.add_argument(
        "--sub_directory",
        default=None,
        help=(
            "Directory containing replacement .pkl files "
            "(must share the same filenames as the flagged files). "
            "When provided, flagged files in input_dir are overwritten."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file for flagged PKL names "
            "(default: flagged_files.txt inside input_dir)"
        ),
    )
    args = parser.parse_args()

    # Default output sits alongside the files being checked
    if args.output is None:
        args.output = os.path.join(args.input_dir, "flagged_files.txt")

    # Validate sub_directory if provided
    if args.sub_directory is not None and not os.path.isdir(args.sub_directory):
        parser.error(f"--sub_directory '{args.sub_directory}' is not a valid directory.")

    flagged = []
    total = 0

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".pkl"):
            continue

        total += 1
        fpath = os.path.join(args.input_dir, fname)

        try:
            with open(fpath, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            print(f"[ERROR]   Failed to load {fname}: {e}")
            continue

        if is_pred_tracks_empty(obj):
            print(f"[FLAGGED] {fname}")
            flagged.append(fname)

    # Write flagged filenames before any replacements
    with open(args.output, "w") as out:
        for f in flagged:
            out.write(f + "\n")

    # --- Replacement pass ---
    replaced, missing = [], []
    if args.sub_directory and flagged:
        print()
        replaced, missing = replace_empty_files(flagged, args.input_dir, args.sub_directory)

    # --- Summary ---
    print("\nSummary:")
    print(f"  Total PKL files checked : {total}")
    print(f"  Flagged (empty pred_tracks): {len(flagged)}")
    if args.sub_directory:
        print(f"  Replaced from sub_directory : {len(replaced)}")
        if missing:
            print(f"  Could not replace (missing in sub_directory): {len(missing)}")
            for m in missing:
                print(f"    - {m}")
    print(f"  Flagged list saved to   : {args.output}")


if __name__ == "__main__":
    main()