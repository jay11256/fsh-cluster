# Global Variables
TSV_PATH = "../temp_data.tsv"
OUTPUT_DIR = ""
TEMP_DIR = ""

# Model Selection
sam2_checkpoint = "../../programs/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

##################################################################################################################################

# Processing TSV

fish_array = []
class FishData:
    def __init__(self, filename="", objectID=-1, points=[]):
        self.filename = filename
        self.objectID = objectID
        self.points = points
    
    def add_point(self, xPos, yPos, type):
        self.points.append((xPos, yPos, type))

    def __str__(self):
        return f"{self.filename}:{self.objectID} has {len(self.points)} points"

import csv

with open(TSV_PATH, "r", newline='') as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        filepath = row[0]
        objectID = row[1]
        triplets = []
        for i in range(2, len(row), 3):
            x = row[i]
            y = row[i+1]
            t = row[i+2]
            triplets.append((x, y, t))
        fish = FishData(filepath, objectID, triplets)
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
# subprocess.run(["nvidia-smi"])

##################################################################################################################################

# for fish in fish_array: