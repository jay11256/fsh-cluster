#!/usr/bin/env python3
"""Launch clip prediction via torchrun."""

import csv
import os
import random
import subprocess
import sys
import torch
import numpy as np

#Data
DATA_DIR = "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/clipping2/clips"
DATA_CSV_PATH = None #Leave none unless csv made in advance
#Points
POINT_INFO_ENABLE = True
TROKENS_PT_DATA = (
    "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/clipping2/pkls"
)
NUM_POINTS_TO_SAMPLE = 18
#Model
CHECKPOINT_FILE = (
    "/fs/vulcan-projects/fsh_track/models/ds6_BCEL/BCEL-5_way-1_shot-sam3-both/checkpoints/checkpoint_best.pyth"
)
#Output
BASE_OUTPUT_DIR = "/fs/vulcan-projects/fsh_track/will/will_files/pipeline_tests/demo6"

#Dont need changing
TORCH_HOME = (
    "/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home"
)
CONFIG_TO_USE = "fshdata"
NUM_CLASSES = 6
FILTER_ONE = True
POINT_INFO_NAME = "cotracker3_bip_fr_32"
N_WAY = 5
K_SHOT = 3
NUM_GPUS = 1
NUM_WORKERS = 4
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

def _ensure_data_csv(data_dir, output_dir, data_csv_path):
    """Build a CSV from videos in data_dir when no CSV path is provided."""
    if data_csv_path is not None:
        return data_csv_path

    video_paths = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, name))
        and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise ValueError(f"No video files found in {data_dir}")

    csv_path = os.path.join(output_dir, "predict_clips.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "behavior","split"])
        writer.writeheader()
        writer.writerows(
            {"video_path": path, "behavior": "Lead", "split": "test"} for path in video_paths
        )

    return csv_path


def main():
    master_port = f"{random.randint(1250, 9999):04d}"

    exp_name = os.environ.get("EXP_NAME", "")
    secondary_exp_name = os.environ.get("SECONDARY_EXP_NAME", "")
    output_dir = os.path.join(BASE_OUTPUT_DIR, exp_name, secondary_exp_name)

    os.environ["TORCH_HOME"] = TORCH_HOME
    os.makedirs(output_dir, exist_ok=True)

    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(pipeline_dir)
    #trokens_root = _find_trokens_root(pipeline_dir)
    trokens_root = os.path.join(par_dir, "trokens")
    os.chdir(trokens_root)

    data_csv_path = _ensure_data_csv(DATA_DIR, output_dir, DATA_CSV_PATH)

    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        f"--master_port={master_port}",
        "tools/get_preds.py",
        "--init_method", "env://",
        "--new_dist_init",
        "--cfg", f"configs/trokens/{CONFIG_TO_USE}.yaml",
        "OUTPUT_DIR", output_dir,
        "NUM_GPUS", str(NUM_GPUS),
        "DATA_LOADER.NUM_WORKERS", str(NUM_WORKERS),
        "DATA.PATH_TO_DATA_DIR", DATA_DIR,
        "DATA.PATH_TO_TROKEN_PT_DATA", TROKENS_PT_DATA,
        "FEW_SHOT.N_WAY", str(N_WAY),
        "FEW_SHOT.K_SHOT", str(K_SHOT),
        "POINT_INFO.ENABLE", str(POINT_INFO_ENABLE),
        "POINT_INFO.NAME", POINT_INFO_NAME,
        "POINT_INFO.NUM_POINTS_TO_SAMPLE", str(NUM_POINTS_TO_SAMPLE),
        "MODEL.DINO_CONFIG", "dinov2_vitb14",
        "MODEL.MOTION_MODULE.USE_CROSS_MOTION_MODULE", "True",
        "MODEL.MOTION_MODULE.USE_HOD_MOTION_MODULE", "True",
        "DATA_LOADER.FILTER_ONE", str(FILTER_ONE),
        "DATA_LOADER.DATA_CSV_PATH", data_csv_path,
        "MODEL.NUM_CLASSES", str(NUM_CLASSES),
        "TEST.CHECKPOINT_FILE_PATH", CHECKPOINT_FILE,
    ]

    # result = subprocess.run(cmd, check=False)

    preds = torch.from_numpy(np.load("/fs/vulcan-projects/fsh_track/will/clipping3_preds/preds.npy"))

    preds = torch.softmax(preds, dim=1)
    # one_hot = torch.zeros_like(preds)
    # preds = one_hot.scatter_(1, preds.argmax(dim=1, keepdim=True), 1)
    # preds = torch.sigmoid(preds)
    print(preds)


    #DEFAULT_LABELS = ["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"]
    from visualize_matrix import visualize_matrix
    visualize_matrix(
        ground_truth_path = "/fs/vulcan-projects/fsh_track/jason/pipeline_testing/all_behaviors.tsv",
        pred_matrix=preds.T,
        threshold=.95,
        thresholds=[.1,.2,.3,.4,.5,.6],
        window_len=4,
        overlap_len=2,
        video_window=(850, 1230),
        save_path=output_dir,
    )
    # sys.exit(result.returncode)


if __name__ == "__main__":
    main()
