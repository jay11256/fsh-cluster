# Global Variables
TSV_PATH = "./selected_points.tsv"
DATA_DIR = "../../data/autumn/"
OUTPUT_DIR = "../outputs"
TEMP_DIR = "../temp"

# Model Selection
sam2_checkpoint = "../../programs/sam2/checkpoints/sam2.1_hiera_large.pt"
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
            t = row[i+2]
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

for fish in fish_array:

    # Converting video into jpegs 
    # Select every nth frame
    n = 10

    subprocess.run(["ffmpeg",
            "-i",
            fish.filename,
            "-vf",
            'select=not(mod(n\\,' + str(n) + '))',
            "-vsync",
            "vfr",
            f"{TEMP_DIR}/%05d.jpg"], check=True)

    # Scanning each jpeg
    frame_names = [
        p for p in os.listdir(TEMP_DIR)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # Initializing predictor
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(
        video_path=TEMP_DIR,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )

    # Helper functions
    def add_point(frame, obj, points_arr, labels_arr, display):
        points = np.array(points_arr, dtype=np.float32)
        labels = np.array(labels_arr, np.int32)
        # prompts[obj] = points, labels
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state = inference_state,
            frame_idx = frame,
            obj_id = obj,
            points = points,
            labels = labels,
        )
    def propagate(vis_frame_stride, display):
    # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments
    
    disp = False
    for i in range(0, len(fish.points)):
        add_point(0, fish.objectID, [fish.points[i]], [fish.labels[i]], disp)
    vid_seg = propagate(40, disp)

##################################################################################################################################

    # Assembling an output video

    fish.filename = os.path.splitext(os.path.basename(fish.filename))[0]

    fps = 30
    first_frame = next(iter(vid_seg))
    first_obj = next(iter(vid_seg[first_frame]))
    mask_shape = vid_seg[first_frame][first_obj].shape[-2:]
    height, width = mask_shape

    # # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(f"{OUTPUT_DIR}/{fish.filename}_{fish.objectID}_bw.mp4", fourcc, fps, (width, height), isColor=True)

    # # Iterate over frames in order
    # for frame_idx in sorted(vid_seg.keys()):
    #     # Combine masks for all objects (e.g., OR them together)
    #     combined_mask = np.zeros((height, width), dtype=np.uint8)
    #     for obj_id, mask in vid_seg[frame_idx].items(): # Can change this to iterate over specific object id(s)
    #         # mask shape: (1, H, W) or (H, W)
    #         mask_bin = mask.squeeze().astype(np.uint8) * 255
    #         combined_mask = np.maximum(combined_mask, mask_bin)
    #     # Convert to 3-channel image for video
    #     mask_rgb = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    #     out.write(mask_rgb)

    # out.release()
    # print(f"Video saved to {OUTPUT_DIR}")

    # Get frame size and fps from previous variables
    frame_size = (width, height)

    # Create VideoWriter for the final video
    final_out = cv2.VideoWriter(f"{OUTPUT_DIR}/{fish.filename}_{fish.objectID}.mp4", fourcc, fps, frame_size, isColor=True)

    # Define a list of distinct colors for up to 10 objects (B, G, R)
    object_colors = [
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 128, 0),    # Olive
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (255, 128, 0),    # Orange
    ]

    # Alpha blending factor for mask overlay
    alpha = 0.4

    for frame_idx in tqdm(sorted(vid_seg.keys()), desc="Creating final video"):
        # Load original frame
        frame_path = os.path.join(TEMP_DIR, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Create a color mask for all objects
        color_mask = np.zeros_like(frame)
        for i, (obj_id, mask) in enumerate(vid_seg[frame_idx].items()):
            mask_bin = mask.squeeze().astype(np.uint8)
            color = object_colors[i % len(object_colors)]
            color_mask[mask_bin > 0] = color

        # Blend mask with original frame
        blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)

        final_out.write(blended)

    final_out.release()
    print(f"Final video saved to {OUTPUT_DIR}")

##################################################################################################################################

    # Cleaning out memory
    del predictor
    del inference_state
    del vid_seg
    torch.cuda.empty_cache()