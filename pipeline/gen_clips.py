"""Full video prediction pipeline — splits video into clips and runs SAM3 on each.
Pipelined: clip N+1 is being encoded by ffmpeg while SAM3 processes clip N.
"""

import os
import pickle
import subprocess
import threading
import queue
import torch
import argparse
import numpy as np
from tqdm import tqdm
import time
import sys 

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Split video and run SAM3 on each clip")
parser.add_argument("video_path",               type=str,   help="Path to the input video")
parser.add_argument("--prompt",                 type=str,   default="fish")
parser.add_argument("--max_objects",            type=int,   default=2)
parser.add_argument("--pkl_dir",                type=str,   default="./pkls")
parser.add_argument("--clip_dir",               type=str,   default="./clips")
parser.add_argument("--logs_dir",               type=str,   default="./logs",   help="Directory for per-clip .out log files")
parser.add_argument("--window_len",             type=int,   default=4)
parser.add_argument("--overlap_len",            type=int,   default=2)
parser.add_argument("--inference_frame_start",  type=int,   default=0)
# SLURM array sharding — set automatically by the .sh launcher
parser.add_argument("--shard_id",    type=int, default=0,  help="Which shard this job processes (SLURM_ARRAY_TASK_ID)")
parser.add_argument("--num_shards",  type=int, default=1,  help="Total number of shards (SLURM_ARRAY_TASK_COUNT)")

args = parser.parse_args()

video_path            = args.video_path
prompt                = args.prompt
MAX_OBJECTS           = args.max_objects
OUTPUT_DIR            = args.pkl_dir
TEMP_CLIP_DIR         = args.clip_dir
LOGS_DIR              = args.logs_dir
WINDOW_LEN            = args.window_len
OVERLAP_LEN           = args.overlap_len
inference_frame_start = args.inference_frame_start
SHARD_ID              = args.shard_id
NUM_SHARDS            = args.num_shards

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_CLIP_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Initialize SAM3 (once)
# ---------------------------------------------------------------------------

import sam3
gpus_to_use = range(torch.cuda.device_count())
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization

predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
print(f"SAM3 initialized on {torch.cuda.device_count()} GPU(s)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_POINTS  = 9   # 9 points per object (3x3 grid)
NUM_OBJECTS = 2   # always exactly 2 objects per pkl

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    return int(float(result.stdout.strip()))


def get_clip_schedule(video_path, window_len, overlap_len):
    """Return list of (idx, start_sec) for every clip in the full video."""
    duration = get_video_duration(video_path)
    return [
        (idx, start)
        for idx, start in enumerate(range(0, duration - window_len, window_len - overlap_len))
    ]


def encode_clip(video_path, start, window_len, out_path):
    """Blocking ffmpeg call — runs in the producer thread."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start), "-t", str(window_len),
        "-reset_timestamps", "1",
        out_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def get_uniform_points(mask, n_points=9):
    """Extract exactly n_points uniformly distributed across a binary mask using
    a sqrt(n) x sqrt(n) grid (3x3 for 9 points). Always returns exactly n_points
    — cells with no mask pixels fall back to the nearest mask pixel.

    Args:
        mask:     2D binary numpy array
        n_points: number of points (must be a perfect square, default 9)

    Returns:
        List of exactly n_points (x, y) tuples, or None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    grid  = int(round(n_points ** 0.5))
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    mask_coords  = np.stack([xs, ys], axis=1).astype(np.float32)

    points = []
    for row in range(grid):
        for col in range(grid):
            x_lo = x_min + col       * (x_max - x_min + 1) / grid
            x_hi = x_min + (col + 1) * (x_max - x_min + 1) / grid
            y_lo = y_min + row       * (y_max - y_min + 1) / grid
            y_hi = y_min + (row + 1) * (y_max - y_min + 1) / grid

            in_cell = (
                (mask_coords[:, 0] >= x_lo) & (mask_coords[:, 0] < x_hi) &
                (mask_coords[:, 1] >= y_lo) & (mask_coords[:, 1] < y_hi)
            )
            cell_coords = mask_coords[in_cell]

            if len(cell_coords) > 0:
                px, py = float(cell_coords[:, 0].mean()), float(cell_coords[:, 1].mean())
            else:
                # Fall back to nearest mask pixel to cell centre
                gx, gy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
                dists  = np.sqrt((mask_coords[:, 0] - gx)**2 + (mask_coords[:, 1] - gy)**2)
                idx_   = np.argmin(dists)
                px, py = float(mask_coords[idx_, 0]), float(mask_coords[idx_, 1])

            points.append((px, py))

    assert len(points) == n_points, f"Expected {n_points} points, got {len(points)}"
    return points


def process_clip(predictor, clip_path, output_dir, logs_dir, prompt, max_objects, inference_frame_start):
    """Run SAM3 on one clip, save a .pkl, and write a .out log. Returns output path."""
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
    pkl_path  = os.path.join(output_dir, f"{clip_name}.pkl")
    out_path  = os.path.join(logs_dir,   f"{clip_name}.out")

    if os.path.exists(pkl_path):
        print(f"  [skip] {pkl_path} already exists")
        return pkl_path

    with open(out_path, "w") as log:
        def logprint(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            log.write(msg + "\n")
            log.flush()

        logprint(f"Clip: {clip_name}")
        logprint(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        response   = predictor.handle_request(dict(type="start_session", resource_path=clip_path))
        session_id = response["session_id"]
        predictor.handle_request(dict(type="reset_session", session_id=session_id))
        predictor.handle_request(dict(
            type="add_prompt", session_id=session_id,
            frame_index=inference_frame_start, text=prompt,
        ))

        outputs_per_frame = propagate_in_video(predictor, session_id)
        vid_seg           = prepare_masks_for_visualization(outputs_per_frame)
        predictor.handle_request(dict(type="close_session", session_id=session_id))
        torch.cuda.empty_cache()

        centroids, points_9 = {}, {}
        for frame_idx, objs in vid_seg.items():
            centroids[frame_idx]  = {}
            points_9[frame_idx]   = {}
            for obj_id, mask in objs.items():
                if mask is None:
                    continue
                mask = np.asarray(mask)
                if mask.ndim == 3:
                    mask = mask[0]
                if mask.ndim != 2:
                    continue
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                centroids[frame_idx][obj_id] = (float(xs.mean()), float(ys.mean()))
                pts = get_uniform_points(mask)
                if pts is not None:
                    points_9[frame_idx][obj_id] = pts

        all_obj_ids_orig = sorted({oid for fd in centroids.values() for oid in fd})
        logprint(f"Found {len(all_obj_ids_orig)} objects: {all_obj_ids_orig}")

        # Always select exactly NUM_OBJECTS=2, ranked by appearance count then earliest frame
        trajectories = {
            oid: [(fi, centroids[fi].get(oid)) for fi in sorted(centroids)]
            for oid in all_obj_ids_orig
        }
        scores = {
            oid: (
                sum(1 for _, p in traj if p is not None),
                -next((i for i, (_, p) in enumerate(traj) if p is not None), float("inf")),
                oid,
            )
            for oid, traj in trajectories.items()
        }
        primary = [oid for oid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:NUM_OBJECTS]]

        # Pad with a placeholder (-1) if fewer than NUM_OBJECTS were detected
        while len(primary) < NUM_OBJECTS:
            primary.append(-(len(primary) + 1))  # sentinel id: -1, -2, ...
            logprint(f"Warning: only {len(all_obj_ids_orig)} object(s) detected; padding to {NUM_OBJECTS}")

        processed_centroids = {
            fi: {oid: pos for oid, pos in fd.items() if oid in primary}
            for fi, fd in centroids.items()
        }

        all_obj_ids   = primary  # fixed order, always length NUM_OBJECTS
        num_objects   = NUM_OBJECTS
        num_frames    = max(processed_centroids.keys()) + 1
        obj_id_to_idx = {oid: i for i, oid in enumerate(all_obj_ids)}

        # Shape: (frames, NUM_OBJECTS * NUM_POINTS, 2) — always 2*9=18 columns
        arr = np.full((num_frames, num_objects * NUM_POINTS, 2), -1, dtype=np.float32)
        for fi, objs in points_9.items():
            for oid, pts in objs.items():
                if oid not in obj_id_to_idx:
                    continue
                oi = obj_id_to_idx[oid]
                for pi, (px, py) in enumerate(pts):
                    arr[fi, oi * NUM_POINTS + pi] = [px, py]

        tracks     = torch.from_numpy(arr)
        visibility = torch.from_numpy(arr[:, :, 0] != -10)
        ids        = torch.tensor([oid for oid in all_obj_ids for _ in range(NUM_POINTS)], dtype=torch.long)

        queries = np.full((num_objects * NUM_POINTS,), -1, dtype=np.int64)
        for fi, objs in points_9.items():
            for oid, pts in objs.items():
                if oid not in obj_id_to_idx:
                    continue
                oi = obj_id_to_idx[oid]
                for pi in range(len(pts)):
                    col = oi * NUM_POINTS + pi
                    if queries[col] == -1:
                        queries[col] = fi
        queries = torch.from_numpy(queries)

        # Sanity check: must always be exactly NUM_OBJECTS*NUM_POINTS=18 columns
        expected_cols = NUM_OBJECTS * NUM_POINTS
        assert tracks.shape[1] == expected_cols,     f"tracks col mismatch: {tracks.shape[1]} != {expected_cols}"
        assert visibility.shape[1] == expected_cols, f"visibility col mismatch: {visibility.shape[1]} != {expected_cols}"
        assert ids.shape[0] == expected_cols,        f"ids length mismatch: {ids.shape[0]} != {expected_cols}"
        assert queries.shape[0] == expected_cols,    f"queries length mismatch: {queries.shape[0]} != {expected_cols}"

        with open(pkl_path, "wb") as f:
            pickle.dump({
                "pred_tracks":     tracks.half(),
                "pred_visibility": visibility.bool(),
                "obj_ids":         ids,
                "point_queries":   queries,
            }, f)

        logprint(f"num_objects: {num_objects} (fixed)  points_per_object: {NUM_POINTS}  total_cols: {num_objects * NUM_POINTS}")
        logprint(f"pred_tracks shape:     {tuple(tracks.shape)}")
        logprint(f"pred_visibility shape: {tuple(visibility.shape)}")
        logprint(f"Saved pkl: {pkl_path}")
        logprint(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return pkl_path


# ---------------------------------------------------------------------------
# Producer thread — encodes clips and pushes paths onto a queue
# ---------------------------------------------------------------------------

SENTINEL = None  # signals consumer that production is done

def producer(clip_schedule, video_path, temp_clip_dir, window_len, q):
    """Encodes clips one by one and puts the finished path on the queue."""
    for idx, start in clip_schedule:
        clip_path = os.path.join(temp_clip_dir, f"clip_{idx:05d}.mp4")
        print(f"[encode] clip {idx:05d}  (t={start}s)")
        encode_clip(video_path, start, window_len, clip_path)
        q.put(clip_path)
        print(f"[encode] clip {idx:05d} ready")
    q.put(SENTINEL)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

full_schedule = get_clip_schedule(video_path, WINDOW_LEN, OVERLAP_LEN)
total_clips   = len(full_schedule)

def clip_is_complete(idx):
    """Return True only if both the pkl and a finished .out log exist."""
    pkl = os.path.join(OUTPUT_DIR, f"clip_{idx:05d}.pkl")
    log = os.path.join(LOGS_DIR,   f"clip_{idx:05d}.out")
    if not os.path.exists(pkl):
        return False
    if not os.path.exists(log):
        return False
    with open(log) as f:
        return any(line.startswith("End:") for line in f)

clip_schedule = [
    entry for i, entry in enumerate(full_schedule)
    if i % NUM_SHARDS == SHARD_ID
    and not clip_is_complete(entry[0])
]

done_count = sum(
    1 for i, entry in enumerate(full_schedule)
    if i % NUM_SHARDS == SHARD_ID
    and clip_is_complete(entry[0])
)
print(f"Shard {SHARD_ID}/{NUM_SHARDS}: {done_count} already done, processing {len(clip_schedule)}/{total_clips} clips")
print(f"Points per object: {NUM_POINTS} (3x3 grid)  |  Objects per clip: {NUM_OBJECTS} (fixed)  |  Total cols: {NUM_OBJECTS * NUM_POINTS}")

clip_queue = queue.Queue(maxsize=2)
enc_thread = threading.Thread(
    target=producer,
    args=(clip_schedule, video_path, TEMP_CLIP_DIR, WINDOW_LEN, clip_queue),
    daemon=True,
)
enc_thread.start()

pipeline_start = time.time()

with tqdm(total=len(clip_schedule), desc=f"SAM3 shard {SHARD_ID}", file=sys.stdout) as pbar:
    while True:
        clip_path = clip_queue.get()
        if clip_path is SENTINEL:
            break

        clip_start = time.time()
        process_clip(predictor, clip_path, OUTPUT_DIR, LOGS_DIR, prompt, MAX_OBJECTS, inference_frame_start)
        clip_elapsed = time.time() - clip_start

        clip_name = os.path.splitext(os.path.basename(clip_path))[0]
        print(f"  {clip_name}  {clip_elapsed:.1f}s", flush=True)
        pbar.update(1)

total_elapsed = time.time() - pipeline_start
hours, rem    = divmod(int(total_elapsed), 3600)
mins, secs    = divmod(rem, 60)
print(f"\nDone. Total time: {hours:02d}:{mins:02d}:{secs:02d}")

enc_thread.join()
predictor.shutdown()
torch.cuda.empty_cache()