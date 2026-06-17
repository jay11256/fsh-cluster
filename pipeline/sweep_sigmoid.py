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
BASE_OUTPUT_DIR = "/fs/vulcan-projects/fsh_track/jason/pipeline_testing/sweep_sigmoids2"

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
    default_threshold=1.0,
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
    thresh_values = np.round(np.arange(0.1, 1.0, 0.01), 4).tolist()

    sigmoid_preds = torch.sigmoid(raw_preds).numpy()  # (num_clips, num_behaviors)
    pred_matrix_T = sigmoid_preds.T                   # (num_behaviors, num_clips)

    # ── Per-class threshold sweep ─────────────────────────────────────────────
    # For each class i, sweep its threshold while all other classes hold at
    # default_threshold. We record the per-class AP for class i at each value.

    per_class_results = []   # rows for CSV
    best_thresh_per_class = [default_threshold] * num_behaviors

    for class_idx, class_name in enumerate(behavior_names):
        best_ap = -np.inf
        best_thresh = default_threshold

        for thresh in thresh_values:
            thresholds = [default_threshold] * num_behaviors
            thresholds[class_idx] = thresh

            report = compute_detection_report(
                gt_times=gt_times,
                gt_behaviors=gt_behaviors,
                pred_matrix=pred_matrix_T,
                labels=behavior_names,
                clip_starts=clip_starts,
                clip_ends=clip_ends,
                thresholds=thresholds,
                iou_thresholds=iou_thresholds,
                gt_duration=window_len,
                eval_start=0.0,
                eval_end=duration,
            )

            cm = report["class_metrics"][class_name]
            ap_values = list(cm["ap_by_iou"].values())
            avg_ap = float(np.nanmean(ap_values))

            per_class_results.append({
                "class": class_name,
                "threshold": thresh,
                "avg_ap_iou": avg_ap,
                "precision": cm["precision"],
                "recall": cm["recall"],
                "f1": cm["f1"],
                "thresholds_used": thresholds,  # kept for plotting, not written to CSV
            })

            if not np.isnan(avg_ap) and avg_ap > best_ap:
                best_ap = avg_ap
                best_thresh = thresh

        best_thresh_per_class[class_idx] = best_thresh
        print(f"Class '{class_name}': best threshold = {best_thresh:.3f}  (avg AP = {best_ap:.4f})")

    # ── Save per-class sweep CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "sweep_results_per_class.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class", "threshold", "avg_ap_iou", "precision", "recall", "f1",
        ])
        writer.writeheader()
        for r in per_class_results:
            writer.writerow({
                "class": r["class"],
                "threshold": r["threshold"],
                "avg_ap_iou": r["avg_ap_iou"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
            })
    print(f"Saved per-class sweep results → {csv_path}")

    # ── Best-combo joint evaluation ───────────────────────────────────────────
    print(f"\nBest per-class thresholds: { {n: t for n, t in zip(behavior_names, best_thresh_per_class)} }")

    combo_report = compute_detection_report(
        gt_times=gt_times,
        gt_behaviors=gt_behaviors,
        pred_matrix=pred_matrix_T,
        labels=behavior_names,
        clip_starts=clip_starts,
        clip_ends=clip_ends,
        thresholds=best_thresh_per_class,
        iou_thresholds=iou_thresholds,
        gt_duration=window_len,
        eval_start=0.0,
        eval_end=duration,
    )
    combo_map_values = list(combo_report["map_by_iou"].values())
    combo_avg_map = float(np.nanmean(combo_map_values))
    print(f"Best-combo avg mAP@IoU = {combo_avg_map:.4f}")

    # Save combo summary CSV
    combo_csv_path = os.path.join(output_dir, "sweep_results_best_combo.csv")
    with open(combo_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class", "best_threshold", "avg_ap_iou", "precision", "recall", "f1",
        ])
        writer.writeheader()
        for class_name in behavior_names:
            cm = combo_report["class_metrics"][class_name]
            ap_values = list(cm["ap_by_iou"].values())
            writer.writerow({
                "class": class_name,
                "best_threshold": best_thresh_per_class[behavior_names.index(class_name)],
                "avg_ap_iou": float(np.nanmean(ap_values)),
                "precision": cm["precision"],
                "recall": cm["recall"],
                "f1": cm["f1"],
            })
    print(f"Saved best-combo results → {combo_csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1) Best-combo plot (uses all best per-class thresholds together)
    combo_save_path = os.path.join(output_dir, "best_combo.png")
    visualize_matrix(
        ground_truth_path=ground_truth_path,
        pred_matrix=pred_matrix_T,
        thresholds=best_thresh_per_class,
        window_len=window_len,
        overlap_len=overlap_len,
        video_window=video_window,
        save_path=combo_save_path,
    )
    print(f"Saved best-combo plot → {combo_save_path}")

    # 2) Top-5 per class by avg_ap_iou, one plot each.
    #    Each plot shows that class at its sweep threshold; others at default.
    from collections import defaultdict
    plotted_per_class = defaultdict(int)
    total_plotted = 0

    results_sorted = sorted(
        per_class_results,
        key=lambda r: (r["avg_ap_iou"] if not np.isnan(r["avg_ap_iou"]) else -np.inf),
        reverse=True,
    )

    for r in results_sorted:
        if plotted_per_class[r["class"]] >= top_k:
            continue

        fname = (
            f"cls_{r['class']}_thresh_{r['threshold']:.3f}.png"
            .replace(" ", "_")
            .replace("/", "-")
        )
        visualize_matrix(
            ground_truth_path=ground_truth_path,
            pred_matrix=pred_matrix_T,
            thresholds=r["thresholds_used"],
            window_len=window_len,
            overlap_len=overlap_len,
            video_window=video_window,
            save_path=os.path.join(output_dir, fname),
        )
        plotted_per_class[r["class"]] += 1
        total_plotted += 1

    print(f"Saved top {top_k} plots per class ({total_plotted} total) to {output_dir}")
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

    sweep_hyperparameters(
        raw_preds=preds,  # raw logits, shape (num_clips, num_behaviors)
        ground_truth_path="/fs/vulcan-projects/fsh_track/jason/pipeline_testing/all_behaviors.tsv",
        output_dir=BASE_OUTPUT_DIR,
        window_len=4,
        overlap_len=2,
        video_window=(850, 1230),
        behavior_names=["Bite", "Lead", "Peck", "Quiver", "Run/Flee", "Tilt"],
        top_k=5,
    )


if __name__ == "__main__":
    main()