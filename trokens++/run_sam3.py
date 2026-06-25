import os
import sys
import pickle
import torch
import argparse

# Parse command-line arguments
#bhargav's args: python ../../fsh-cluster/run_sam3.py 60min_clip547.mp4 --create_output_vid --output_dir ./pklfiles/
parser = argparse.ArgumentParser(description="SAM3 video segmentation with object tracking")
parser.add_argument("video_path", type=str, help="Path to the video file")
parser.add_argument("--prompt", type=str, default="fish", help="Text prompt for segmentation")
parser.add_argument("--max_objects", type=int, default=2, help="Maximum number of objects to track (default: 2)")
parser.add_argument("--target_fps", type=int, default=25, help="Target FPS for video processing")
parser.add_argument("--output_dir", type=str, default="./", help="Output directory")
parser.add_argument("--create_output_vid", action="store_true", help="Create output video")
parser.add_argument("--inference_frame_start", type=int, default=0, help="Frame to start inference")

args = parser.parse_args()

TARGET_FPS            = args.target_fps
video_path            = args.video_path
prompt                = args.prompt
MAX_OBJECTS           = args.max_objects
inference_frame_start = args.inference_frame_start
CLIP_NAME             = os.path.splitext(os.path.basename(video_path))[0]
OUTPUT_DIR            = args.output_dir
CREATE_OUTPUT_VID     = args.create_output_vid

# ---------------------------------------------------------------------------
# Constants — must match gen_clips.py exactly
# ---------------------------------------------------------------------------
NUM_POINTS  = 9   # 9 points per object (3x3 grid)
NUM_OBJECTS = 2   # always exactly 2 objects — pad with sentinels if needed

print(f"Fixed objects: {NUM_OBJECTS}  |  Points per object: {NUM_POINTS}  |  Total columns: {NUM_OBJECTS * NUM_POINTS}")

# ---------------------------------------------------------------------------
# Setup SAM3
# ---------------------------------------------------------------------------
import sam3
import torch
sam3_root  = os.path.join(os.path.dirname(sam3.__file__), "..")
gpus_to_use = range(torch.cuda.device_count())
from sam3.model_builder import build_sam3_video_predictor
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12


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


# ---------------------------------------------------------------------------
# Loading Video
# ---------------------------------------------------------------------------
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    TARGET_FPS = cap.get(cv2.CAP_PROP_FPS)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()

# ---------------------------------------------------------------------------
# SAM3 Inference
# ---------------------------------------------------------------------------
response   = predictor.handle_request(dict(type="start_session", resource_path=video_path))
session_id = response["session_id"]

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

_ = predictor.handle_request(dict(type="reset_session", session_id=session_id))

response = predictor.handle_request(dict(
    type="add_prompt",
    session_id=session_id,
    frame_index=inference_frame_start,
    text=prompt,
))

out               = response["outputs"]
outputs_per_frame = propagate_in_video(predictor, session_id)
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

# ---------------------------------------------------------------------------
# Output path setup
# ---------------------------------------------------------------------------
from tqdm import tqdm
if CREATE_OUTPUT_VID:
    os.makedirs(f"{OUTPUT_DIR}/{CLIP_NAME}_output", exist_ok=True)
video_name = os.path.splitext(os.path.basename(video_path))[0]
vid_seg    = outputs_per_frame

if CREATE_OUTPUT_VID:
    CENTROID_PATH = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", "centroids.pkl")
else:
    CENTROID_PATH = os.path.join(OUTPUT_DIR, f"{CLIP_NAME}.pkl")

# ---------------------------------------------------------------------------
# Extract centroids and 9-point grids per frame
# ---------------------------------------------------------------------------
centroids = {}  # used only for object selection ranking
points_9  = {}  # 9 uniform points per object per frame — goes into PKL

for frame_idx, objs in vid_seg.items():
    centroids[frame_idx] = {}
    points_9[frame_idx]  = {}

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

        pts = get_uniform_points(mask, n_points=NUM_POINTS)
        if pts is not None:
            points_9[frame_idx][obj_id] = pts

# ---------------------------------------------------------------------------
# Select exactly NUM_OBJECTS=2, ranked by appearance count then earliest frame.
# Pad with sentinel IDs (-1, -2, ...) if fewer than NUM_OBJECTS were detected.
#
# Detection count behaviour:
#   0 objects → sentinels -1 and -2 fill all 18 cols; all points = -1, all vis = False
#   1 object  → real obj in cols 0-8, sentinel -1 in cols 9-17; cols 9-17 all -1 / False
#   2 objects → both real objs fill cols 0-8 and 9-17 normally; no padding needed
#   3+ objects → top 2 by (appearance count, earliest frame) kept; rest discarded
# ---------------------------------------------------------------------------
print(f"Selecting exactly {NUM_OBJECTS} objects from SAM3 output...")

all_obj_ids_original = sorted({
    oid for fd in centroids.values() for oid in fd.keys()
})
print(f"Found {len(all_obj_ids_original)} unique objects: {all_obj_ids_original}")

trajectories = {
    oid: [(fi, centroids[fi].get(oid)) for fi in sorted(centroids)]
    for oid in all_obj_ids_original
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

# Pad with sentinel IDs if needed
while len(primary) < NUM_OBJECTS:
    primary.append(-(len(primary) + 1))
    print(f"Warning: only {len(all_obj_ids_original)} object(s) detected; padding to {NUM_OBJECTS}")

processed_centroids = {
    fi: {oid: pos for oid, pos in fd.items() if oid in primary}
    for fi, fd in centroids.items()
}

all_obj_ids   = primary        # fixed order, always length NUM_OBJECTS
num_objects   = NUM_OBJECTS    # always 2
num_frames    = max(processed_centroids.keys()) + 1
obj_id_to_idx = {oid: i for i, oid in enumerate(all_obj_ids)}

print(f"Primary objects: {all_obj_ids}")
print(f"num_objects: {num_objects} (fixed)  points_per_object: {NUM_POINTS}  total_cols: {num_objects * NUM_POINTS}")

# ---------------------------------------------------------------------------
# Build tracks array — shape (frames, NUM_OBJECTS * NUM_POINTS, 2)
# Sentinel/undetected points stay at -1; visibility = (arr[:,:,0] != -1)
# ---------------------------------------------------------------------------
arr = np.full((num_frames, num_objects * NUM_POINTS, 2), -1, dtype=np.float32)

for frame_idx in sorted(points_9.keys()):
    for obj_id, pts in points_9[frame_idx].items():
        if obj_id not in obj_id_to_idx:
            continue
        oi = obj_id_to_idx[obj_id]
        for pi, (px, py) in enumerate(pts):
            arr[frame_idx, oi * NUM_POINTS + pi] = [px, py]

tracks     = torch.from_numpy(arr)
visibility = torch.from_numpy(arr[:, :, 0] != -1)  # False where points are -1 (not detected)
ids        = torch.tensor(
    [oid for oid in all_obj_ids for _ in range(NUM_POINTS)],
    dtype=torch.long,
)

queries = np.full((num_objects * NUM_POINTS,), -1, dtype=np.int64)
for frame_idx in sorted(points_9.keys()):
    for obj_id, pts in points_9[frame_idx].items():
        if obj_id not in obj_id_to_idx:
            continue
        oi = obj_id_to_idx[obj_id]
        for pi in range(len(pts)):
            col = oi * NUM_POINTS + pi
            if queries[col] == -1:
                queries[col] = frame_idx
queries = torch.from_numpy(queries)

# Sanity checks — must always be exactly NUM_OBJECTS*NUM_POINTS=18 columns
expected_cols = NUM_OBJECTS * NUM_POINTS
assert tracks.shape[1]     == expected_cols, f"tracks col mismatch: {tracks.shape[1]} != {expected_cols}"
assert visibility.shape[1] == expected_cols, f"visibility col mismatch: {visibility.shape[1]} != {expected_cols}"
assert ids.shape[0]        == expected_cols, f"ids length mismatch: {ids.shape[0]} != {expected_cols}"
assert queries.shape[0]    == expected_cols, f"queries length mismatch: {queries.shape[0]} != {expected_cols}"

# ---------------------------------------------------------------------------
# Save pickle
# ---------------------------------------------------------------------------
output = {
    "pred_tracks":     tracks.half(),        # (frames, 18, 2)
    "pred_visibility": visibility.bool(),    # (frames, 18)
    "obj_ids":         ids,                  # (18,)
    "point_queries":   queries,              # (18,)
}
with open(CENTROID_PATH, "wb") as f:
    pickle.dump(output, f)

print(f"Saved to {CENTROID_PATH}")
print("pred_tracks shape:    ", tracks.shape)
print("pred_visibility shape:", visibility.shape)
print("obj_ids shape:        ", ids.shape)
print("point_queries shape:  ", queries.shape)

# ---------------------------------------------------------------------------
# Optional output video
# ---------------------------------------------------------------------------
if CREATE_OUTPUT_VID:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps         = TARGET_FPS or cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    out_path    = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", f"{video_name}_pred.mp4")
    out         = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    object_colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0),
    ]
    alpha         = .95
    obj_color_map = {}
    frame_idx     = 0

    with tqdm(total=frame_count, desc=f"Rendering {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in vid_seg:
                overlay = frame.copy()
                for obj_id, mask in vid_seg[frame_idx].items():
                    if mask is None:
                        continue
                    mask = np.asarray(mask)
                    if mask.ndim == 3:
                        mask = mask[0]
                    if mask.ndim != 2:
                        continue
                    h, w = mask.shape
                    if h == 0 or w == 0:
                        continue
                    if (h, w) != frame.shape[:2]:
                        mask = cv2.resize(
                            mask, (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask = (mask > 0).astype(np.uint8)
                    if obj_id not in obj_color_map:
                        obj_color_map[obj_id] = len(obj_color_map) % len(object_colors)
                    color       = object_colors[obj_color_map[obj_id]]
                    color_layer = np.zeros_like(frame, dtype=np.uint8)
                    color_layer[mask == 1] = color
                    overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame = overlay

            if frame_idx in points_9:
                for obj_id, pts in points_9[frame_idx].items():
                    if obj_id not in obj_color_map:
                        obj_color_map[obj_id] = len(obj_color_map) % len(object_colors)
                    color = object_colors[obj_color_map[obj_id]]
                    for i, (px, py) in enumerate(pts):
                        px, py = int(px), int(py)
                        radius = 7 if i == 4 else 4

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    import subprocess
    fixed_path = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", f"{video_name}_fixed.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", out_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        fixed_path,
    ], check=True)

    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"Deleted {out_path}")
    else:
        print(f"File not found: {out_path}")

    print(f"Output video saved to {fixed_path}")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
_ = predictor.handle_request(dict(type="close_session", session_id=session_id))
predictor.shutdown()
torch.cuda.empty_cache()