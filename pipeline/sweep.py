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
    "/fs/vulcan-projects/fsh_track/models/ds6/5_way-3_shot-sam3-both/checkpoints/checkpoint_best.pyth"
)
#Output
BASE_OUTPUT_DIR = "/fs/vulcan-projects/fsh_track/jason/pipeline_testing/sweep_bcel"

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

import os
import csv
import numpy as np
import torch
from visualize_matrix import visualize_matrix, compute_detection_report, _load_ground_truth

def sweep_hyperparameters(
    raw_preds,           # torch tensor, shape (num_clips, num_behaviors) - pre-transform logits
    ground_truth_path,
    output_dir,
    window_len=4,
    overlap_len=2,
    video_window=(850, 1230),
    behavior_names=None,
    top_k=20,
):
    os.makedirs(output_dir, exist_ok=True)

    num_clips, num_behaviors = raw_preds.shape
    stride = window_len - overlap_len
    clip_starts = np.arange(num_clips) * stride
    clip_ends = clip_starts + window_len

    gt_times, gt_behaviors = _load_ground_truth(ground_truth_path, video_window)

    if behavior_names is None:
        from visualize_matrix import DEFAULT_LABELS
        behavior_names = DEFAULT_LABELS

    duration = float(video_window[1] - video_window[0])

    iou_thresholds = (.1, .2, .3, .4, .5, .6, .7, .8, .9)

    transform_configs = []

    # Softmax transform (row-wise / per-clip across behaviors), various thresholds
    softmax_preds = torch.softmax(raw_preds, dim=1).numpy()
    for thresh in np.arange(0.1, 1.0, 0.01):
        transform_configs.append({
            "transform": "softmax",
            "threshold": round(float(thresh), 4),
            "pred_matrix": softmax_preds.T,
        })

    # Sigmoid transform, various thresholds
    sigmoid_preds = torch.sigmoid(raw_preds).numpy()
    for thresh in np.arange(0.1, 1.0, 0.01):
        transform_configs.append({
            "transform": "sigmoid",
            "threshold": round(float(thresh), 4),
            "pred_matrix": sigmoid_preds.T,
        })

    results = []

    for cfg in transform_configs:
        pred_matrix = cfg["pred_matrix"]
        threshold = cfg["threshold"]

        print(f"\n=== Running transform={cfg['transform']!r}, threshold={threshold:.3f} ===")

        report = compute_detection_report(
            gt_times=gt_times,
            gt_behaviors=gt_behaviors,
            pred_matrix=pred_matrix,
            labels=behavior_names,
            clip_starts=clip_starts,
            clip_ends=clip_ends,
            threshold=threshold,
            iou_thresholds=iou_thresholds,
            gt_duration=window_len,
            eval_start=0.0,
            eval_end=duration,
        )

        map_values = list(report["map_by_iou"].values())
        avg_map = float(np.nanmean(map_values))

        precisions = [m["precision"] for m in report["class_metrics"].values()]
        recalls = [m["recall"] for m in report["class_metrics"].values()]
        f1s = [m["f1"] for m in report["class_metrics"].values()]

        results.append({
            "transform": cfg["transform"],
            "threshold": threshold,
            "avg_map_iou": avg_map,
            "mean_precision": float(np.nanmean(precisions)),
            "mean_recall": float(np.nanmean(recalls)),
            "mean_f1": float(np.nanmean(f1s)),
            "pred_matrix": pred_matrix,  # keep around for plotting top-k
        })

    # Save CSV (without pred_matrix)
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "transform", "threshold", "avg_map_iou",
            "mean_precision", "mean_recall", "mean_f1",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "transform": r["transform"],
                "threshold": r["threshold"],
                "avg_map_iou": r["avg_map_iou"],
                "mean_precision": r["mean_precision"],
                "mean_recall": r["mean_recall"],
                "mean_f1": r["mean_f1"],
            })

    print(f"Saved sweep results → {csv_path}")

    # Sort by avg_map_iou descending, handle nan as -inf
    results_sorted = sorted(
        results,
        key=lambda r: (r["avg_map_iou"] if not np.isnan(r["avg_map_iou"]) else -np.inf),
        reverse=True,
    )

    top_results = results_sorted[:top_k]

    for r in top_results:
        fname = f"{r['transform']}_{r['threshold']:.3f}.png".replace(" ", "")
        save_path = os.path.join(output_dir, fname)

        visualize_matrix(
            ground_truth_path=ground_truth_path,
            pred_matrix=r["pred_matrix"],
            threshold=r["threshold"],
            window_len=window_len,
            overlap_len=overlap_len,
            video_window=video_window,
            behavior_names=behavior_names,
            save_path=save_path,
        )

    print(f"Saved top {len(top_results)} plots to {output_dir}")
    return results_sorted

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

    # preds = torch.softmax(preds, dim=1) # Softmax
    # one_hot = torch.zeros_like(preds) # One hot
    # preds = one_hot.scatter_(1, preds.argmax(dim=1, keepdim=True), 1) # One hot
    # preds = torch.sigmoid(preds) # Sigmoid
    # print(preds)

    sweep_hyperparameters(
        raw_preds=preds,  # raw logits, shape (num_clips, num_behaviors)
        ground_truth_path="/fs/vulcan-projects/fsh_track/jason/pipeline_testing/all_behaviors.tsv",
        output_dir=BASE_OUTPUT_DIR,
        window_len=4,
        overlap_len=2,
        video_window=(850, 1230),
        behavior_names=["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"],  # or your own list matching preds.shape[1]
        top_k=20,
    )


if __name__ == "__main__":
    main()
