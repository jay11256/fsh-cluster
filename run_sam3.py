import os
import sys

TARGET_FPS = 25
video_path = sys.argv[1]
prompt = "fish"
inference_frame_start = 0
CLIP_NAME = os.path.splitext(os.path.basename(video_path))[0]
OUTPUT_DIR = f"jason/outputs"
CREATE_OUTPUT_VID = False

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

## Initializing video library stuff ##
from tqdm import tqdm
if CREATE_OUTPUT_VID:
    os.makedirs(f"{OUTPUT_DIR}/{CLIP_NAME}_output", exist_ok=True)
video_name = os.path.splitext(os.path.basename(video_path))[0]
vid_seg = outputs_per_frame

# Processing Masks

import pickle
import torch

# ========================
# region Centroids (pred_tracks)
# ========================
if CREATE_OUTPUT_VID:
    CENTROID_PATH = os.path.join(f"{OUTPUT_DIR}/{CLIP_NAME}_output", "centroids.pkl")
else:
    CENTROID_PATH = os.path.join(OUTPUT_DIR, f"{CLIP_NAME}.pkl")
centroids = {}
for frame_idx, objs in vid_seg.items():
    centroids[frame_idx] = {}

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
        cx = xs.mean()
        cy = ys.mean()
        centroids[frame_idx][obj_id] = (float(cx), float(cy))

# Convert to tensor maybe
# Sort frames
frame_indices = sorted(centroids.keys())
num_frames = max(frame_indices) + 1

# Collect unique object IDs
all_obj_ids = sorted({
    obj_id
    for frame_data in centroids.values()
    for obj_id in frame_data.keys()
})

num_objects = len(all_obj_ids)

# Map object ID → tensor index
obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(all_obj_ids)}

# Initialize array with NaNs (for missing detections)
arr = np.full((num_objects, num_frames, 2), np.nan, dtype=np.float32)

# Fill array
for frame_idx, frame_data in centroids.items():
    for obj_id, (cx, cy) in frame_data.items():
        obj_idx = obj_id_to_idx[obj_id]
        arr[obj_idx, frame_idx] = [cx, cy]

# Convert to torch tensor
tracks = torch.from_numpy(arr)
# ========================
# endregion
# ========================

# ========================
# region Visibility (pred_visibility)
# ========================

# ========================
# endregion
# ========================

# Saving pickle
output = {
    "pred_tracks": tracks,
    "pred_visibility": 1,
    "obj_ids": 1,
    "point_queries": 1
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

            # Draw Centroids
            if frame_idx in centroids:
                for obj_id, (cx, cy) in centroids[frame_idx].items():

                    cx = int(cx)
                    cy = int(cy)

                    if obj_id not in obj_color_map:
                        obj_color_map[obj_id] = len(obj_color_map) % len(object_colors)
                    color = object_colors[obj_color_map[obj_id]]

                    cv2.circle(frame, (cx, cy), 6, color, -1)

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