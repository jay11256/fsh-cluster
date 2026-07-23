#!/usr/bin/env python3
"""Launch clip prediction via torchrun, then localize with ASM-Loc post-processing.

Identical to predict_clips.py through inference; the ONLY difference is the final
localization step: instead of visualize_matrix (threshold -> merge adjacent clips),
this uses ASM-Loc's multi-threshold Outer-Inner-Contrast proposal generation +
temporal NMS (see asmloc_postprocess.py). The metrics file it writes has the same
format as predict_clips.py's, so you can diff the two `metrics` files to see the
before/after mAP@IoU.
"""

import csv
import os
import random
import subprocess
import sys
import torch
import numpy as np

#Data
DATA_DIR = "/fs/vulcan-projects/fsh_track/bhargav/data/60min_0522/clips"
DATA_CSV_PATH = None #Leave none unless csv made in advance
#Points
POINT_INFO_ENABLE = False
TROKENS_PT_DATA = (
    "/fs/vulcan-projects/fsh_track/bhargav/data/60min_0522/pkls"
)
NUM_POINTS_TO_SAMPLE = 256
#Model
CHECKPOINT_FILE = (
    "/fs/vulcan-projects/fsh_track/models/ds11/2_way-1_shot-none-both/checkpoints/checkpoint_best.pyth"
)
#Output
BASE_OUTPUT_DIR = "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/ds11_check_postproc"
#Ground truth / windowing (must match how the clips were generated)
GROUND_TRUTH_PATH = "/fs/vulcan-projects/fsh_track/raw_data/box/CirclingAssayPGF2a_NoseBlockVsShamVsCNG_202505-202509_CGP/2025-05-22_Pi14_TankA1-1_Run1_ShamNoseblockFemalePGF2a_Circling/BORISAnnotations.tsv"
WINDOW_LEN = 4
OVERLAP_LEN = 2
THRESHOLD = 0.5           # only used for the clip-level P/R columns; ASM-Loc sweeps its own
VIDEO_WINDOW = None       # (start, end) seconds, or None for the whole recording

#ASM-Loc post-processing knobs (tune on your data; defaults are a sensible start)
UPSCALE = 20              # temporal up-sampling factor for sub-clip boundaries
CAS_THRESH_LIST = np.round(np.arange(0.10, 0.85, 0.05), 3)  # multi-threshold sweep
OIC_LAMBDA = 0.2          # outer (context) margin as a fraction of proposal length
OIC_GAMMA = 0.0           # weight on the recording-level class score (0 = OIC only)
NMS_IOU = 0.45            # temporal NMS IoU threshold

#Dont need changing
TORCH_HOME = (
    "/fs/vulcan-projects/fsh_track/programs/trokens_workspace/trokens/torch_home"
)
CONFIG_TO_USE = "fshdata"
NUM_CLASSES = 7
FILTER_ONE = True
FILTER_TWO = True
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
        writer = csv.DictWriter(f, fieldnames=["video_path", "behavior", "split"])
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
        "DATA_LOADER.FILTER_ONE", str(FILTER_TWO),
        "DATA_LOADER.DATA_CSV_PATH", data_csv_path,
        "MODEL.NUM_CLASSES", str(NUM_CLASSES),
        "TEST.CHECKPOINT_FILE_PATH", CHECKPOINT_FILE,
    ]

    result = subprocess.run(cmd, check=False)

    preds = torch.from_numpy(np.load(os.path.join(output_dir, "preds.npy")))
    preds = torch.sigmoid(preds)
    print(preds)

    # ── ASM-Loc post-processing (replaces visualize_matrix's threshold-merge) ──
    # pipeline_dir is on sys.path so we can import the sibling module.
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    from asmloc_postprocess import visualize_matrix_asmloc

    visualize_matrix_asmloc(
        ground_truth_path=GROUND_TRUTH_PATH,
        pred_matrix=preds.T,
        threshold=THRESHOLD,
        window_len=WINDOW_LEN,
        overlap_len=OVERLAP_LEN,
        video_window=VIDEO_WINDOW,
        save_path=output_dir,
        # ASM-Loc knobs
        upscale=UPSCALE,
        cas_thresh_list=CAS_THRESH_LIST,
        oic_lambda=OIC_LAMBDA,
        oic_gamma=OIC_GAMMA,
        nms_iou=NMS_IOU,
    )
    # sys.exit(result.returncode)


if __name__ == "__main__":
    main()
