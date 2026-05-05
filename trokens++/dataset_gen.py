#!/usr/bin/env python3
"""
Extract per-annotation behavior clips from BORIS-style CSV/TSV (POINT rows):
for each timestamp, cut DURATION_BEFORE + DURATION_AFTER in the master video
(clamped to file bounds), then write dataset7.csv in the output folder.
Requires: ffmpeg, ffprobe, pandas
"""

import pandas as pd
import subprocess
import os
import sys
import hashlib
from collections import defaultdict
from pathlib import Path
import glob

# Configuration variables - modify these paths as needed
DATA_PATH = "/fs/vulcan-projects/fsh_track/raw_data"        # Folder containing CSV files and video files
OUTPUT_FOLDER = "/fs/vulcan-projects/fsh_track/processed_data/dataset7"       # Folder where output videos will be saved

# Duration settings for behavior clips
DURATION_BEFORE = 2.0  # Seconds before the timestamp
DURATION_AFTER = 2.0   # Seconds after the timestamp

# Column name settings - add alternative column names as needed
TIME_COLUMN_NAMES = ['Time']  # List of possible time column names
BEHAVIOR_COLUMN_NAMES = ['Behavior']  # List of possible behavior column names
STATUS_COLUMN_NAMES = ['Status', 'Behavior type']  # List of possible status column names

# Raw annotation label -> canonical behavior name (CSV "behavior" column).
# Keys match BORIS exports; lookup also tries case-insensitive match on stripped text.
BEHAVIOR_NORMALIZATION = {
    "Peck": "Peck",
    "Quiver-m": "Quiver",
    "Quiver (Male)": "Quiver",
    "Bite (Male)": "Bite",
    "Tilt (Specify Sex)": "Tilt",
    "Peck (Specify Sex)": "Peck",
    "Lead (Male)": "Lead",
    "peck": "Peck",
    "Lead-m": "Lead",
    "pot entry": "Enter Pot",
    "Chase/Charge (Male)": "Chase/Charge",
    "Run/Flee (Female)": "Run/Flee",
    "male quiver": "Quiver",
    "exit plot": "Exit Plot",
    "male lead": "Lead",
    "egg retrieval": "Egg Retrieval",
    "circling": "Circling",
    "female follow": "Follow",
    "Follow-f": "Follow",
    "Spawning-f": "Spawning",
    "Follow (Female)": "Follow",
}

for _canon in (
    "Peck",
    "Quiver",
    "Bite",
    "Tilt",
    "Lead",
    "Follow",
    "Chase/Charge",
    "Run/Flee",
    "Enter Pot",
    "Exit Plot",
    "Circling",
    "Egg Retrevial",
    "Spawning",
):
    BEHAVIOR_NORMALIZATION.setdefault(_canon, _canon)

_BEHAVIOR_NORMALIZATION_CI = {
    k.strip().lower(): v for k, v in BEHAVIOR_NORMALIZATION.items()
}


def normalize_behavior_label(raw):
    """
    Map a raw Behavior cell to (canonical_behavior, og_behavior).
    og_behavior is the trimmed string from the sheet; canonical uses BEHAVIOR_NORMALIZATION
    when matched, otherwise the trimmed original (same as og).
    """
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


def validate_paths():
    """Validate that data path exists and create output folder."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data folder not found: {DATA_PATH}")
    
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"✓ Data folder: {DATA_PATH}")
    print(f"✓ Output folder: {OUTPUT_FOLDER}")

def find_csv_video_pairs(data_path):
    """
    Find all CSV/TSV files and match them with corresponding video files.
    Handles subfolders: if a subfolder has exactly one video and one data file, 
    they are paired regardless of names. Otherwise, uses name-based matching.
    
    Args:
        data_path (str): Path to folder containing CSV/TSV and video files (may contain subfolders)
    
    Returns:
        list: List of tuples (data_file_path, video_path) for matched pairs
    """
    pairs = []
    
    # Get all subdirectories (including the root data_path itself)
    subdirs = [data_path]  # Start with root directory
    for root, dirs, files in os.walk(data_path):
        if root != data_path:  # Don't add root twice
            subdirs.append(root)
    
    print(f"Scanning {len(subdirs)} directories (including subfolders)")
    
    for subdir in subdirs:
        print(f"\nProcessing directory: {subdir}")
        
        # Find CSV/TSV files in this directory
        csv_files = glob.glob(os.path.join(subdir, "*.csv"))
        tsv_files = glob.glob(os.path.join(subdir, "*.tsv"))
        data_files = csv_files + tsv_files
        
        # Find video files in this directory
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(subdir, ext)))
        
        print(f"  Found {len(csv_files)} CSV files, {len(tsv_files)} TSV files, and {len(video_files)} video files")
        
        # Special case: exactly one video and exactly one data file - pair them regardless of names
        if len(video_files) == 1 and len(data_files) == 1:
            pairs.append((data_files[0], video_files[0]))
            print(f"  ✓ Paired (exact match): {os.path.basename(data_files[0])} <-> {os.path.basename(video_files[0])}")
        else:
            # Use name-based matching (original behavior)
            data_basenames = {os.path.splitext(os.path.basename(data_file))[0]: data_file for data_file in data_files}
            video_basenames = {os.path.splitext(os.path.basename(video))[0]: video for video in video_files}
            
            # Find matches
            for base_name, data_file_path in data_basenames.items():
                if base_name in video_basenames:
                    pairs.append((data_file_path, video_basenames[base_name]))
                    print(f"  ✓ Matched by name: {os.path.basename(data_file_path)} <-> {os.path.basename(video_basenames[base_name])}")
                else:
                    print(f"  Warning: No matching video found for {os.path.basename(data_file_path)}")
    
    print(f"\nTotal pairs found: {len(pairs)}")
    return pairs

def read_timestamps(data_path, time_column_names=None, status_column_names=None, behavior_column_names=None):
    """
    Read timestamps from CSV or TSV file, filtering for POINT status.
    Handles data files with metadata at the top.
    
    Args:
        data_path (str): Path to CSV or TSV file
        time_column_names (list): List of possible time column names (defaults to global TIME_COLUMN_NAMES)
        status_column_names (list): List of possible status column names (defaults to global STATUS_COLUMN_NAMES)
        behavior_column_names (list): List of possible behavior column names (defaults to global BEHAVIOR_COLUMN_NAMES)
    
    Returns:
        list: List of tuples containing (timestamp, behavior) for POINT status rows
    """
    # Use global variables if not provided
    if time_column_names is None:
        time_column_names = TIME_COLUMN_NAMES
    if status_column_names is None:
        status_column_names = STATUS_COLUMN_NAMES
    if behavior_column_names is None:
        behavior_column_names = BEHAVIOR_COLUMN_NAMES
    
    # Detect if file is TSV or CSV based on extension (before try block for exception handling)
    file_ext = os.path.splitext(data_path)[1].lower()
    is_tsv = file_ext == '.tsv'
    
    try:
        separator = '\t' if is_tsv else ','
        
        # First, read the file to find where the actual data starts
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        # Find the line with the actual column headers
        # Check if any time column name, any status column name, and any behavior column name are present
        header_line = None
        for i, line in enumerate(lines):
            has_time = any(time_col in line for time_col in time_column_names)
            has_status = any(status_col in line for status_col in status_column_names)
            has_behavior = any(behavior_col in line for behavior_col in behavior_column_names)
            if has_time and has_status and has_behavior:
                header_line = i
                break
        
        if header_line is None:
            file_type = "TSV" if is_tsv else "CSV"
            print(f"Could not find header row with required columns")
            print(f"  Looking for time columns: {time_column_names}")
            print(f"  Looking for status columns: {status_column_names}")
            print(f"  Looking for behavior columns: {behavior_column_names}")
            print(f"First few lines of {file_type}:")
            for i, line in enumerate(lines[:10]):
                print(f"Line {i+1}: {line.strip()}")
            raise ValueError(f"Required columns not found in {file_type} file")
        
        print(f"Found header row at line {header_line + 1}")
        
        # Read file starting from the header line with appropriate separator
        df = pd.read_csv(data_path, skiprows=header_line, sep=separator)
        
        # Find the first matching time column name
        time_column = None
        for time_col in time_column_names:
            if time_col in df.columns:
                time_column = time_col
                break
        
        # Find the first matching behavior column name
        behavior_column = None
        for behavior_col in behavior_column_names:
            if behavior_col in df.columns:
                behavior_column = behavior_col
                break
        
        # Find the first matching status column name
        status_column = None
        for status_col in status_column_names:
            if status_col in df.columns:
                status_column = status_col
                break
        
        # Check if required columns exist
        missing_columns = []
        if time_column is None:
            missing_columns.append(f"Time column (tried: {time_column_names})")
        if status_column is None:
            missing_columns.append(f"Status column (tried: {status_column_names})")
        if behavior_column is None:
            missing_columns.append(f"Behavior column (tried: {behavior_column_names})")
        
        if missing_columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Required columns not found: {missing_columns}")
        
        print(f"Using time column: '{time_column}', status column: '{status_column}', behavior column: '{behavior_column}'")
        
        # Filter for rows with POINT status
        point_rows = df[df[status_column] == 'POINT']
        
        if len(point_rows) == 0:
            print(f"No rows found with status 'POINT'")
            print(f"Available status values: {df[status_column].unique()}")
            return []
        
        # Get timestamps and behaviors, remove any NaN values
        timestamps_behaviors = []
        for _, row in point_rows.iterrows():
            if pd.notna(row[time_column]) and pd.notna(row[behavior_column]):
                timestamps_behaviors.append((row[time_column], row[behavior_column]))
        
        file_type = "TSV" if is_tsv else "CSV"
        print(f"Found {len(timestamps_behaviors)} timestamps with POINT status in column '{time_column}'")
        return timestamps_behaviors
        
    except Exception as e:
        file_type = "TSV" if is_tsv else "CSV"
        print(f"Error reading {file_type} file: {e}")
        sys.exit(1)


def get_video_duration_seconds(video_path):
    """Return container duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"ffprobe failed for {video_path}: {e}") from e


def extract_clip_ffmpeg(
    master_path,
    clip_start,
    duration,
    output_path,
):
    """
    Cut [clip_start, clip_start + duration) from master_path into output_path.
    Seeks after input for frame-accurate cuts (slower but matches timestamps).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(clip_start),
        "-i",
        master_path,
        "-t",
        str(duration),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p2",
        "-cq",
        "18",
        "-an",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def assign_train_test_per_behavior(rows):
    """
    Assign split per behavior: ~80% train, ~20% test, deterministic.
    rows: list of dicts, each must include 'behavior' and a stable 'split_key' string.
    Mutates rows in place with 'split' = 'train' or 'test'.
    """
    by_behavior = defaultdict(list)
    for i, row in enumerate(rows):
        by_behavior[row["behavior"]].append(i)

    for behavior, idxs in by_behavior.items():
        n = len(idxs)
        if n == 0:
            continue

        def sort_key(i):
            h = hashlib.sha256(rows[i]["split_key"].encode("utf-8")).hexdigest()
            return h

        ordered = sorted(idxs, key=sort_key)

        if n == 1:
            train_positions = set(ordered)
        else:
            n_train = (4 * n) // 5
            if n_train <= 0:
                n_train = 1
            if n_train >= n:
                n_train = n - 1
            train_positions = set(ordered[:n_train])

        for i in ordered:
            rows[i]["split"] = "train" if i in train_positions else "test"

    return rows


def main():
    """Main function to orchestrate the video processing."""
    print("Video Clip Extractor - Batch Processing")
    print("=" * 50)
    
    # Validate paths
    try:
        validate_paths()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Find all CSV/video pairs
    print(f"\nFinding CSV and video files in: {DATA_PATH}")
    csv_video_pairs = find_csv_video_pairs(DATA_PATH)
    
    if not csv_video_pairs:
        print("No matching CSV/video pairs found.")
        sys.exit(1)
    
    # Process all pairs
    all_clip_info = []
    clip_counter = 0
    
    for csv_path, video_path in csv_video_pairs:
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(csv_path)} <-> {os.path.basename(video_path)}")
        print(f"{'='*50}")
        
        # Read timestamps from CSV/TSV
        data_file_name = os.path.basename(csv_path)
        print(f"\nReading timestamps from: {data_file_name}")
        timestamps_behaviors = read_timestamps(csv_path)
        
        if not timestamps_behaviors:
            print(f"No timestamps found with POINT status in {data_file_name}. Skipping.")
            continue
        
        # Process video clips
        master_video_name = os.path.basename(video_path)
        master_stem = Path(video_path).stem

        try:
            master_duration = get_video_duration_seconds(video_path)
        except RuntimeError as e:
            print(f"  Skipping pair: {e}")
            continue

        for timestamp, behavior_raw in timestamps_behaviors:
            try:
                ts = float(timestamp)
            except (TypeError, ValueError):
                print(f"  Skipping non-numeric timestamp: {timestamp!r}")
                continue

            behavior, og_behavior = normalize_behavior_label(behavior_raw)
            if not og_behavior:
                print(f"  Skipping empty behavior at t={ts}")
                continue

            ideal_start = ts - DURATION_BEFORE
            ideal_end = ts + DURATION_AFTER
            clip_start = max(0.0, ideal_start)
            clip_end = min(master_duration, ideal_end)
            seg_duration = clip_end - clip_start

            if seg_duration <= 0:
                print(
                    f"  Skipping behavior {og_behavior!r} at t={ts}: "
                    f"non-positive segment [{clip_start}, {clip_end}]"
                )
                continue

            clip_counter += 1
            out_name = f"{master_stem}_clip{clip_counter}.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)

            clip_already = os.path.isfile(out_path) and os.path.getsize(out_path) > 0

            label_note = f"{behavior}" if behavior == og_behavior else f"{behavior} (og: {og_behavior})"
            print(
                f"  Clip {clip_counter}: {label_note} @ master [{clip_start:.3f}, {clip_end:.3f}] "
                f"({seg_duration:.3f}s) -> {out_name}"
                + (" (exists, skipping ffmpeg)" if clip_already else "")
            )

            if not clip_already:
                try:
                    extract_clip_ffmpeg(video_path, clip_start, seg_duration, out_path)
                except subprocess.CalledProcessError as e:
                    print(f"  ffmpeg failed for clip {clip_counter}: {e}")
                    if os.path.isfile(out_path):
                        try:
                            os.remove(out_path)
                        except OSError:
                            pass
                    continue

            split_key = (
                f"{master_video_name}|{clip_start:.6f}|{clip_end:.6f}|{behavior}|{og_behavior}"
            )

            all_clip_info.append(
                {
                    "video_path": out_path,
                    "behavior": behavior,
                    "og_behavior": og_behavior,
                    "start_time": clip_start,
                    "end_time": clip_end,
                    "master_video": master_video_name,
                    "duration": seg_duration,
                    "dataset": "fshdata",
                    "split_key": split_key,
                }
            )

    assign_train_test_per_behavior(all_clip_info)

    for row in all_clip_info:
        row.pop("split_key", None)

    summary_path = os.path.join(OUTPUT_FOLDER, "dataset7.csv")
    if all_clip_info:
        summary_df = pd.DataFrame(all_clip_info)
        summary_df = summary_df[
            [
                "video_path",
                "behavior",
                "og_behavior",
                "start_time",
                "end_time",
                "master_video",
                "duration",
                "dataset",
                "split",
            ]
        ]
        summary_df.to_csv(summary_path, index=False)
        print(f"\nWrote summary CSV: {summary_path}")
    else:
        print("\nNo clips were produced; not writing dataset7.csv.")

    print(f"\n{'='*50}")
    print("Batch processing complete!")
    print(f"Total clips created: {len(all_clip_info)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()