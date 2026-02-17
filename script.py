# Global Variables
TSV_PATH = "./jason/fsh-cluster/selected_points.tsv"
DATA_DIR = "./jason/data/"
OUTPUT_DIR = "./jason/outputs/"
TEMP_DIR = "./jason/temp/"

# Video Info
target_fps = 16

# Model Selection
sam2_checkpoint = "./programs/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

##################################################################################################################################

# Processing TSV

fish_array = []
class FishData:
    def __init__(self, filename="", objectID=-1, points=[], labels=[]):
        self.filename = filename
        self.objectID = objectID
        self.points = points
        self.labels = labels
    
    def add_point(self, xPos, yPos, type):
        self.points.append([xPos, yPos])
        self.labels.append(type)

    def __str__(self):
        return f"{self.filename}:{self.objectID} has {len(self.points)} points"

import csv

with open(TSV_PATH, "r", newline='') as file:
    reader = csv.reader(file, delimiter="\t")
    next(reader)
    for row in reader:
        filepath = DATA_DIR + row[0]
        objectID = row[1]
        coords = []
        types = []
        for i in range(2, len(row), 3):
            x = row[i]
            y = row[i+1]
            coords.append([x, y])
            # t = row[i+2]
            t = 1
            types.append(t)
        fish = FishData(filepath, objectID, coords, types)
        fish_array.append(fish)
        print(fish)

##################################################################################################################################

# Imports

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import cv2
from tqdm import tqdm
import subprocess

##################################################################################################################################

# Device Check

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.float16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
# subprocess.run(["nvidia-smi"], check=True)

##################################################################################################################################

# Processing and predicting on each line

from collections import defaultdict

# --- Group fish by video filename ---
videos_dict = defaultdict(list)
for fish in fish_array:
    videos_dict[fish.filename].append(fish)

##################################################################################################################################
# Process each video
for video_path, fish_list in videos_dict.items():
    print(f"\nProcessing video: {video_path} ({len(fish_list)} objects)")

    # Convert video into JPEGs
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-filter:v", f"fps={target_fps}",  # use n to downsample, e.g., n=2 gives 15 fps
        "-vsync", "cfr",
        f"{TEMP_DIR}/%05d.jpg"
    ], check=True)

    # Collect frame names
    frame_names = sorted(
        [p for p in os.listdir(TEMP_DIR) if p.lower().endswith((".jpg", ".jpeg"))],
        key=lambda p: int(os.path.splitext(p)[0])
    )

    # Initialize predictor once
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(
        video_path=TEMP_DIR,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    # Add points for all fish objects
    def add_point(frame, obj, points_arr, labels_arr):
        points = np.array(points_arr, dtype=np.float32)
        labels = np.array(labels_arr, np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame,
            obj_id=int(obj),
            points=points,
            labels=labels,
        )

    for fish in fish_list:
        for i in range(len(fish.points)):
            add_point(0, fish.objectID, [fish.points[i]], [fish.labels[i]])

    # Propagate all masks once
    def propagate():
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    vid_seg = propagate()

    ##################################################################################################################################
    # Assemble a combined output video

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    fps = target_fps

    first_frame = next(iter(vid_seg))
    first_obj = next(iter(vid_seg[first_frame]))
    mask_shape = vid_seg[first_frame][first_obj].shape[-2:]
    height, width = mask_shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = f"{OUTPUT_DIR}/{video_name}_combined3.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)

    # Assign distinct colors per object ID
    object_colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0)
    ]
    alpha = 0.4

    for frame_idx in tqdm(sorted(vid_seg.keys()), desc=f"Rendering {video_name}"):
        frame_path = os.path.join(TEMP_DIR, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        color_mask = np.zeros_like(frame)
        for i, (obj_id, mask) in enumerate(vid_seg[frame_idx].items()):
            mask_bin = mask.squeeze().astype(np.uint8)
            color = object_colors[i % len(object_colors)]
            color_mask[mask_bin > 0] = color

        blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
        out.write(blended)

    out.release()
    print(f"✅ Combined video saved to {out_path}")

    ##################################################################################################################################
    # Cleanup
    del predictor, inference_state, vid_seg
    torch.cuda.empty_cache()
