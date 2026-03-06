import os
import sys
import pickle
import torch
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="SAM3 video segmentation with object tracking")
parser.add_argument("video_path", type=str, help="Path to the video file")
parser.add_argument("--prompt", type=str, default="fish", help="Text prompt for segmentation")
parser.add_argument("--max_objects", type=int, default=2, help="Maximum number of objects to track (default: 2)")
parser.add_argument("--target_fps", type=int, default=25, help="Target FPS for video processing")
parser.add_argument("--output_dir", type=str, default="./", help="Output directory")
parser.add_argument("--create_output_vid", action="store_true", help="Create output video")
parser.add_argument("--inference_frame_start", type=int, default=0, help="Frame to start inference")

args = parser.parse_args()

TARGET_FPS = args.target_fps
video_path = args.video_path
prompt = args.prompt
MAX_OBJECTS = args.max_objects  # <-- NEW PARAMETER
inference_frame_start = args.inference_frame_start
CLIP_NAME = os.path.splitext(os.path.basename(video_path))[0]
OUTPUT_DIR = args.output_dir
CREATE_OUTPUT_VID = args.create_output_vid

print(f"Maximum objects to track: {MAX_OBJECTS}")

# region Setup
import sam3
import torch
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
gpus_to_use = range(torch.cuda.device_count())
from sam3.model_builder import build_sam3_video_predictor
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
# endregion

# region Helper Functions #
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
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")


def get_uniform_points(mask, n_points=9):
    """Extract n_points uniformly distributed across a binary mask using a 3x3 grid.

    Divides the mask bounding box into a grid and samples the mean of mask
    pixels within each cell. Falls back to the nearest mask pixel for empty cells.

    Args:
        mask: 2D binary numpy array
        n_points: number of points (must be a perfect square, e.g. 9)

    Returns:
        List of (x, y) tuples of length n_points, or None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    grid = int(round(n_points ** 0.5))  # 3 for 9 points

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    mask_coords = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)

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
                px = float(cell_coords[:, 0].mean())
                py = float(cell_coords[:, 1].mean())
            else:
                # Nearest mask pixel to cell centre
                gx = (x_lo + x_hi) / 2
                gy = (y_lo + y_hi) / 2
                dists = np.sqrt((mask_coords[:, 0] - gx) ** 2 + (mask_coords[:, 1] - gy) ** 2)
                nearest = np.argmin(dists)
                px = float(mask_coords[nearest, 0])
                py = float(mask_coords[nearest, 1])

            points.append((px, py))

    return points  # 9 x (x, y)
# endregion

# region Loading Video
# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
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
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()
# endregion

# region SAM3 Inference
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)
prompt_text_str = prompt
frame_idx = inference_frame_start  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text_str,
    )
)
out = response["outputs"]
outputs_per_frame = propagate_in_video(predictor, session_id)
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)
# endregion

# region Initializing video library stuff
from tqdm import tqdm
if CREATE_OUTPUT_VID:
    os.makedirs(f"{OUTPUT_DIR}/{CLIP_NAME}_output", exist_ok=True)
video_name = os.path.splitext(os.path.basename(video_path))[0]
vid_seg = outputs_per_frame
# endregion

# region Centroids (pred_tracks) — PKL shape is unchanged (centroid per object)
if CREATE_OUTPUT_VID:
    CENTROID_PATH = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", "centroids.pkl")
else:
    CENTROID_PATH = os.path.join(OUTPUT_DIR, f"{CLIP_NAME}.pkl")

centroids = {}
# Parallel structure holding 9 points per object — used only for video rendering
points_9 = {}

for frame_idx, objs in vid_seg.items():
    centroids[frame_idx] = {}
    points_9[frame_idx] = {}

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

        # Centroid — unchanged, still goes into PKL
        cx = xs.mean()
        cy = ys.mean()
        centroids[frame_idx][obj_id] = (float(cx), float(cy))

        # 9 uniform points — video rendering only
        pts = get_uniform_points(mask, n_points=9)
        if pts is not None:
            points_9[frame_idx][obj_id] = pts

# Post-process: Enforce max objects with temporal consistency
print(f"Post-processing centroids to enforce max {MAX_OBJECTS} objects...")

# Step 1: Collect all unique object IDs across all frames
all_obj_ids_original = sorted({
    obj_id
    for frame_data in centroids.values()
    for obj_id in frame_data.keys()
})

print(f"Found {len(all_obj_ids_original)} unique objects: {all_obj_ids_original}")

# If we already have MAX_OBJECTS or fewer, no need to post-process
if len(all_obj_ids_original) <= MAX_OBJECTS:
    print(f"Already have {MAX_OBJECTS} or fewer objects, skipping post-processing")
    processed_centroids = centroids
else:
    # Step 2: Track object trajectories to maintain temporal consistency
    # Build trajectory for each object
    trajectories = {}
    for obj_id in all_obj_ids_original:
        trajectory = []
        for frame_idx in sorted(centroids.keys()):
            if obj_id in centroids[frame_idx]:
                trajectory.append((frame_idx, centroids[frame_idx][obj_id]))
            else:
                trajectory.append((frame_idx, None))
        trajectories[obj_id] = trajectory
    
    # Step 3: Assign objects to slots based on first appearance and consistency
    # Use greedy assignment: prioritize objects that appear earliest and most frequently
    object_scores = {}
    for obj_id in all_obj_ids_original:
        # Score = number of frames with detection
        appearances = sum(1 for frame_idx, pos in trajectories[obj_id] if pos is not None)
        # Early appearance bonus
        first_appearance = next((i for i, (_, pos) in enumerate(trajectories[obj_id]) if pos is not None), float('inf'))
        object_scores[obj_id] = (appearances, -first_appearance, obj_id)  # negative first_appearance for sorting
    
    # Sort by score and assign first MAX_OBJECTS to slots
    sorted_objects = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
    primary_obj_ids = [obj_id for obj_id, _ in sorted_objects[:MAX_OBJECTS]]
    extra_obj_ids = [obj_id for obj_id, _ in sorted_objects[MAX_OBJECTS:]]
    
    print(f"Primary objects (will be tracked): {primary_obj_ids}")
    print(f"Extra objects (will be ignored): {extra_obj_ids}")
    
    # Step 4: Build processed centroids with ONLY primary objects
    # Extra objects are completely ignored
    processed_centroids = {}
    for frame_idx in sorted(centroids.keys()):
        processed_centroids[frame_idx] = {}
        
        # Only include primary objects
        for obj_id in primary_obj_ids:
            if obj_id in centroids[frame_idx]:
                processed_centroids[frame_idx][obj_id] = centroids[frame_idx][obj_id]

# Sort frames
frame_indices = sorted(processed_centroids.keys())
num_frames = max(frame_indices) + 1

# Collect unique object IDs (should now be exactly MAX_OBJECTS or fewer)
all_obj_ids = sorted({
    obj_id
    for frame_data in processed_centroids.values()
    for obj_id in frame_data.keys()
})

num_objects = len(all_obj_ids)
print(f"After post-processing: {num_objects} objects in pkl file")

# Map object ID → tensor index
obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(all_obj_ids)}

# Initialize array with -1s (for missing detections)
arr = np.full((num_frames, num_objects, 2), -1, dtype=np.float32)

# Fill array
for frame_idx, frame_data in processed_centroids.items():
    for obj_id, (cx, cy) in frame_data.items():
        obj_idx = obj_id_to_idx[obj_id]
        arr[frame_idx, obj_idx] = [cx, cy]

# If coordinate is unknown, use the last known coordinate (forward filling)
for obj_idx in range(num_objects):
    last_value = None
    for t in range(num_frames):
        if arr[t, obj_idx, 0] != -1:
            last_value = arr[t, obj_idx]
        elif last_value is not None:
            arr[t, obj_idx] = last_value

# Convert to torch tensor
tracks = torch.from_numpy(arr)
# endregion

# region Visibility (pred_visibility)
visibility = torch.full((num_frames, num_objects), True)
# Mark as False where coordinates are -1 (never appeared)
for obj_idx in range(num_objects):
    for t in range(num_frames):
        if arr[t, obj_idx, 0] == -1:
            visibility[t, obj_idx] = False
# endregion

# region IDs (obj_ids)
ids = torch.tensor(all_obj_ids)
# endregion

# region Queries (point_queries)
valid = arr[:, :, 0] != -1
first_appearance = np.argmax(valid, axis = 0)
never_appeared = ~valid.any(axis = 0)
first_appearance[never_appeared] = -1
queries = torch.from_numpy(first_appearance)
# endregion

# Saving pickle — shape unchanged (centroid-based, as before)
output = {
    "pred_tracks": tracks,
    "pred_visibility": visibility,
    "obj_ids": ids,
    "point_queries": queries
}
with open(CENTROID_PATH, "wb") as f:
    pickle.dump(output, f)
print(f"Centroids saved to {CENTROID_PATH}")
print("Tensor shape:", tracks.shape)

## Creating an output video ##
if CREATE_OUTPUT_VID:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = TARGET_FPS or cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", f"{video_name}_pred.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open")
    object_colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0),
    ]
    alpha = 0.4
    obj_color_map = {}
    frame_idx = 0
    with tqdm(total=frame_count, desc=f"Rendering {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in vid_seg:
                overlay = frame.copy()

                for obj_id, mask in vid_seg[frame_idx].items():

                    # ---- Convert to numpy ----
                    if mask is None:
                        continue

                    mask = np.asarray(mask)

                    # ---- Reduce to 2D ----
                    if mask.ndim == 3:
                        mask = mask[0]

                    # ---- Validate mask ----
                    if mask.ndim != 2:
                        continue

                    h, w = mask.shape
                    if h == 0 or w == 0:
                        continue

                    # ---- Resize if needed ----
                    if (h, w) != frame.shape[:2]:
                        mask = cv2.resize(
                            mask,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                    # ---- Binarize ----
                    mask = (mask > 0).astype(np.uint8)

                    # ---- Stable color assignment ----
                    if obj_id not in obj_color_map:
                        obj_color_map[obj_id] = len(obj_color_map) % len(object_colors)
                    color = object_colors[obj_color_map[obj_id]]

                    # ---- Apply overlay ----
                    color_layer = np.zeros_like(frame, dtype=np.uint8)
                    color_layer[mask == 1] = color

                    overlay = cv2.addWeighted(
                        overlay, 1.0,
                        color_layer, alpha,
                        0
                    )

                frame = overlay

            # Draw 9 uniform points per object (replaces single centroid dot)
            if frame_idx in points_9:
                for obj_id, pts in points_9[frame_idx].items():

                    if obj_id not in obj_color_map:
                        obj_color_map[obj_id] = len(obj_color_map) % len(object_colors)
                    color = object_colors[obj_color_map[obj_id]]

                    for i, (px, py) in enumerate(pts):
                        px, py = int(px), int(py)
                        # Centre point (index 4 in a 3x3 grid) slightly larger
                        radius = 7 if i == 4 else 4
                        cv2.circle(frame, (px, py), radius, color, -1)
                        cv2.circle(frame, (px, py), radius + 1, (0, 0, 0), 1)  # thin black outline

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    # Converting video from mp4v to avc1
    import subprocess
    fixed_path = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", f"{video_name}_fixed.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", f"{out_path}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        f"{fixed_path}"
    ], check=True)

    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"Deleted {out_path}")
    else:
        print(f"File not found: {out_path}")

    print(f"Output video saved to {out_path}")

## End and Clean ##
_ = predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
predictor.shutdown()
torch.cuda.empty_cache()