"""Feature-bank access and BORIS annotation loading.

Provides the recording-level inputs FishFormer trains on; windowing and target
assignment live in `train_former.py` (`FormerWindows`, `assign_targets`).

Feature timeline: one 768-d vector per feature step, extracted from a frozen
Trokens backbone. `FEATS_ROOT` maps a feature-mode name to its dump directory
and `STRIDE` gives that mode's seconds-per-step -- the modes differ in backbone
checkpoint, frame rate, and how the backbone itself was trained. The one used
for the published results is `ds12_06_5fold_fold{0-4}`: true 5-fold
cross-validation over all 17 recordings, where each fold's backbone was
retrained holding its own recordings out entirely, so evaluating fold F on
fold F's recordings is genuinely leave-out at both the backbone and the
FishFormer level.

Annotations are BORIS *point* events -- a bare timestamp per behavior, with no
start or end. Nothing here invents a duration for them: `load_recording`
returns the raw `(time_seconds, class_index)` list, and any proxy box is built
downstream at supervision or evaluation time, where its width is an explicit,
swept parameter rather than a property of the data.
"""
import os
import sys
import glob
import random

import numpy as np

# data11make (BORIS annotation parsing) lives outside this repo, in the dataset
# generation tree. Override with FSH_DATASET_GEN if it sits elsewhere.
sys.path.insert(0, os.environ.get(
    "FSH_DATASET_GEN",
    "/fs/vulcan-projects/fsh_track/will/will_files/dataset_gen"))
import data11make as d11   # noqa: E402

# Annotation TSVs, one directory per recording. Override with FSH_PAIRS.
PAIRS = os.environ.get(
    "FSH_PAIRS", "/fs/vulcan-projects/fsh_track/raw_data/processed_ofure/pairs")
FEATS_ROOT = {
    "patchx": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats_patchx",
    "clsx": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats",
    # patchx features + the MIL model's per-clip predictions concatenated. preds
    # are per 2s clip while patchx runs at 0.5s, so each prediction row is
    # repeated 4x to align. Hands FishTAL the MIL classifier's output -- including
    # P(NoBehavior) -- as explicit input alongside the raw embedding.
    "patchx_mil": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats_patchx",
    # patchx features + an explicit frame-to-frame motion signal (finite
    # difference between consecutive 0.5s steps), computed on the fly from the
    # same cached feats.npy -- no new dump needed. Trokens' own MIL model spends
    # 6.4M params on cross_motion_module + hod_motion_module giving it explicit
    # motion information; FishFormer has no equivalent and relies on the
    # temporal trunk to infer motion from a static per-step embedding alone.
    # This is the cheapest possible test of whether that gap matters.
    "patchx_motion": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats_patchx",
    # coarse (4x4=16-region) DINO patch grid per 0.5s step instead of a single
    # mean-pooled vector -- see dump_feats_patchx_spatial.py. feats.npy here is
    # (T, 16, 768), not (T, 768); FishFormer's spatial_pool=True learns an
    # attention-weighted pool over the 16 regions instead of a fixed mean.
    "patchx_spatial": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats_patchx_spatial",
    # raw DINOv2 patch-token mean, extracted directly (dump_dino_raw.py) --
    # bypasses Trokens' Pointformer wrapper entirely, so no NUM_FRAMES=8 limit
    # (that count is baked into the frozen MIL checkpoint's time_pos_embed, not
    # a property of DINOv2 itself) and no cross_motion_module/hod_motion_module
    # pre-processing (trained for few-shot classification, not localization).
    # 8fps vs patchx's 2fps -- see EXPERIMENTS.md "Independent of the MIL backbone".
    "dino_raw_8fps": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats_dino_raw/8fps",
    # The "no MIL at all" ablation arm. Same raw-DINOv2 bypass as
    # dino_raw_8fps, but dumped at 4fps so the stride (0.25s) matches every
    # ds12_06_5fold bank -- that makes the ABSENCE OF THE MIL BACKBONE the only
    # difference from those arms, where 8fps would have confounded it with
    # double the temporal resolution. Also covers all 17 recordings, unlike the
    # 8fps bank's 14 (run_dump_dino_raw.py iterates the older patchx list).
    #
    # Deliberately NOT per-fold: raw DINO involves no training, so there is
    # nothing to hold out at the feature level and one bank correctly serves
    # all 5 folds. This arm therefore carries strictly less leakage risk than
    # the MIL arms, not more.
    "dino_raw_4fps": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds11_baseline/feats_dino_raw/4fps",
    # Experiment 1's ablated arm: the SAME few-shot MIL pipeline, trained with
    # POINT_INFO.ENABLE=False so the backbone sees a uniform 16x16 DINO patch
    # grid instead of 18 SAM3-tracked keypoints. Per-fold like the sam3 banks,
    # so the leave-out property is preserved; the only difference from
    # ds12_06_5fold_fold{F} is the absence of keypoint conditioning.
    # Point-DENSITY ablation. Same SAM3 masks and the same few-shot MIL recipe,
    # but the backbone was trained on fewer sampled points per fish: p8 keeps the
    # four 3x3 grid corners, p2 the grid centre (see
    # trokens/tools/subsample_sam3_points.py). Per-fold, so leave-out holds.
    "ds12_06_5fold_sam3p8_fold0": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p8_fold0",
    "ds12_06_5fold_sam3p8_fold1": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p8_fold1",
    "ds12_06_5fold_sam3p8_fold2": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p8_fold2",
    "ds12_06_5fold_sam3p8_fold3": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p8_fold3",
    "ds12_06_5fold_sam3p8_fold4": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p8_fold4",
    "ds12_06_5fold_sam3p2_fold0": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p2_fold0",
    "ds12_06_5fold_sam3p2_fold1": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p2_fold1",
    "ds12_06_5fold_sam3p2_fold2": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p2_fold2",
    "ds12_06_5fold_sam3p2_fold3": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p2_fold3",
    "ds12_06_5fold_sam3p2_fold4": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3p2_fold4",
    # APPEARANCE ablation: same few-shot MIL recipe and the same real 18-point
    # SAM3 tracks as the baseline, but trained AND dumped with BLACK_FRAMES=1 so
    # every clip yields an identical feature map. The only thing distinguishing
    # clips is their point trajectories, making this a geometry-only backbone --
    # the exact complement of the nokp arm, which keeps pixels and flattens
    # trajectory.
    "ds12_06_5fold_sam3black_fold0": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3black_fold0",
    "ds12_06_5fold_sam3black_fold1": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3black_fold1",
    "ds12_06_5fold_sam3black_fold2": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3black_fold2",
    "ds12_06_5fold_sam3black_fold3": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3black_fold3",
    "ds12_06_5fold_sam3black_fold4": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_sam3black_fold4",
    "ds12_06_5fold_nokp_fold0": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_nokp_fold0",
    "ds12_06_5fold_nokp_fold1": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_nokp_fold1",
    "ds12_06_5fold_nokp_fold2": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_nokp_fold2",
    "ds12_06_5fold_nokp_fold3": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_nokp_fold3",
    "ds12_06_5fold_nokp_fold4": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_nokp_fold4",
    # NEW Trokens backbones (ds12_04_sweep_frames): retrained with the actual
    # few-shot training pipeline at NUM_FRAMES in {8,16,32} (2/4/8fps), AND
    # with POINT_INFO.ENABLE=True (real CoTracker point-tracked features, 18
    # points, not a uniform 256-patch grid) -- unlike ds11 (patchx's source),
    # which was ENABLE=False. Motion modules (cross_motion_module/
    # hod_motion_module) are intact and trained on dataset12_04, unlike
    # dino_raw_8fps's bypass. See EXPERIMENTS.md "New Trokens backbones".
    "nf8": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_nf8",
    "nf16": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_nf16",
    "nf32": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_nf32",
    # MIL backbone retrained holding out 3 recordings entirely (never seen during
    # MIL pretraining, not just held out for the downstream head) -- see
    # EXPERIMENTS.md "Leave-3 held-out MIL backbone". Same NUM_FRAMES/stride
    # convention as nf8/16/32, different checkpoint + feats dir only.
    "leave3_nf8": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_leave3_nf8",
    "leave3_nf16": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_leave3_nf16",
    "leave3_nf32": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_leave3_nf32",
    # ds12_leaveout_wlamo: a different training recipe on the SAME leave-3 split
    # (see EXPERIMENTS.md "Leave-3 held-out MIL backbone"), NUM_FRAMES=16 fixed,
    # K_SHOT in {3,5} instead of a frame-count sweep.
    "wlamo_3shot": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_wlamo_3shot",
    "wlamo_5shot": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_wlamo_5shot",
    # models/ds12/ds12_{02,04,06,max}: sweep over the proportion of
    # NoBehavior-labeled clips in the backbone's training data (realized
    # ratios ~0.20/0.39/0.47/0.47), NUM_FRAMES=8 fixed, full dataset (NOT
    # leave-3 retrained -- these backbones saw all recordings during their
    # own training, unlike leave3_nf8/wlamo_3shot/wlamo_5shot above).
    "ds12_02": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_02",
    "ds12_04": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_04",
    "ds12_06": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06",
    "ds12_max": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_max",
    # models/ds12_06_fewshot: combines the winning none-ratio backbone data
    # (dataset12_06, ~47% NoBehavior clips) with a few-shot (5-way, K_SHOT in
    # {3,5}) training recipe, NUM_FRAMES=16, full dataset (NOT leave-3 --
    # same non-leave-out caveat as ds12_02/04/06/max above).
    "ds12_06_fewshot_3shot": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_fewshot_3shot",
    "ds12_06_fewshot_5shot": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_fewshot_5shot",
    # models/ds12_06_5fold: TRUE 5-fold cross-validation over all 17 source
    # videos (dataset12_06 data, 5-way-3-shot recipe) -- each fold's MIL
    # checkpoint was trained holding its own ~3-4 videos out entirely, so
    # (unlike every backbone above) evaluating fold F on fold F's held-out
    # videos is genuinely leave-out at both the MIL and FishFormer level.
    # See trokens_folds.sh / dataset12_06_folds/ for the fold definitions.
    "ds12_06_5fold_fold0": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_fold0",
    "ds12_06_5fold_fold1": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_fold1",
    "ds12_06_5fold_fold2": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_fold2",
    "ds12_06_5fold_fold3": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_fold3",
    "ds12_06_5fold_fold4": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_fold4",
    # ds12_06_5fold_neural: same 5-fold leave-out split/data as ds12_06_5fold_fold{F}
    # above, but the MIL backbone was trained WITHOUT the few-shot episodic
    # recipe (FEW_SHOT.DISABLE=True at train time -- a plain 7-way classifier
    # head instead) -- see trokens_sweep.sh jobs 7148707-7148711 and
    # EXPERIMENTS.md "neural vs few-shot backbone". Same NUM_FRAMES=16 (4fps,
    # stride 0.25s) as the few-shot variant.
    "ds12_06_5fold_neural_fold0": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_neural_fold0",
    "ds12_06_5fold_neural_fold1": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_neural_fold1",
    "ds12_06_5fold_neural_fold2": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_neural_fold2",
    "ds12_06_5fold_neural_fold3": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_neural_fold3",
    "ds12_06_5fold_neural_fold4": "/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/feats_ds12_06_5fold_neural_fold4",
}
STRIDE = {"patchx": 0.5, "clsx": 2.0, "patchx_mil": 0.5, "patchx_motion": 0.5,
          "patchx_spatial": 0.5, "dino_raw_8fps": 0.125,
          "nf8": 0.5, "nf16": 0.25, "nf32": 0.125,
          "leave3_nf8": 0.5, "leave3_nf16": 0.25, "leave3_nf32": 0.125,
          "wlamo_3shot": 0.25, "wlamo_5shot": 0.25,
          "ds12_02": 0.5, "ds12_04": 0.5, "ds12_06": 0.5, "ds12_max": 0.5,
          "ds12_06_fewshot_3shot": 0.25, "ds12_06_fewshot_5shot": 0.25,
          "ds12_06_5fold_fold0": 0.25, "ds12_06_5fold_fold1": 0.25,
          "ds12_06_5fold_fold2": 0.25, "ds12_06_5fold_fold3": 0.25,
          "ds12_06_5fold_fold4": 0.25,
          "ds12_06_5fold_neural_fold0": 0.25, "ds12_06_5fold_neural_fold1": 0.25,
          "ds12_06_5fold_neural_fold2": 0.25, "ds12_06_5fold_neural_fold3": 0.25,
          "ds12_06_5fold_neural_fold4": 0.25,
          "dino_raw_4fps": 0.25,
          "ds12_06_5fold_sam3black_fold0": 0.25, "ds12_06_5fold_sam3black_fold1": 0.25, "ds12_06_5fold_sam3black_fold2": 0.25, "ds12_06_5fold_sam3black_fold3": 0.25, "ds12_06_5fold_sam3black_fold4": 0.25,
          "ds12_06_5fold_sam3p8_fold0": 0.25, "ds12_06_5fold_sam3p8_fold1": 0.25, "ds12_06_5fold_sam3p8_fold2": 0.25, "ds12_06_5fold_sam3p8_fold3": 0.25, "ds12_06_5fold_sam3p8_fold4": 0.25, "ds12_06_5fold_sam3p2_fold0": 0.25, "ds12_06_5fold_sam3p2_fold1": 0.25, "ds12_06_5fold_sam3p2_fold2": 0.25, "ds12_06_5fold_sam3p2_fold3": 0.25, "ds12_06_5fold_sam3p2_fold4": 0.25,
          "ds12_06_5fold_nokp_fold0": 0.25, "ds12_06_5fold_nokp_fold1": 0.25, "ds12_06_5fold_nokp_fold2": 0.25, "ds12_06_5fold_nokp_fold3": 0.25, "ds12_06_5fold_nokp_fold4": 0.25}

# 6 foreground behaviors + background. Background is index 6 (last) so foreground
# ids line up with FG_CLASSES for the eval harness.
FG_CLASSES = ["Bite", "Chase/Charge", "Lead", "Peck", "Quiver", "Tilt"]
BG_INDEX = len(FG_CLASSES)
NUM_CLASSES = len(FG_CLASSES) + 1

# These 3 recordings were excluded everywhere for months as "broken GT". They are
# not broken: BORIS's aggregated-events export has a 'Behavior type' column
# (always "POINT") that visualize_matrix._load_ground_truth preferred over the
# real 'Behavior' column, so they parsed to zero events. data11make -- which this
# module and the ASM-Loc data builder use -- always read them correctly, so the
# exclusion was never needed on the training path at all. Parser fixed
# 2026-07-17; that restores 1771 annotated events (369 + 997 + 405) and takes the
# LOO set from 11 to 14 recordings.
# Set FISHTAL_LEGACY_11=1 to reproduce the old 11-recording split for
# apples-to-apples comparison against results recorded before the fix.
BROKEN_GT = ({"25-08-14-Run1-Sham-Cir", "25-08-15-Run1-Vetbond-NoCir",
              "25-08-26-Run1-Vetbond-Cir"}
             if os.environ.get("FISHTAL_LEGACY_11") == "1" else set())
# clsx dumps for these two live under their old 60min_* names; patchx dumps use
# the real recording name.
CLSX_ALIAS = {"25-05-22-Run1-Sham-Cir": "60min_0522",
              "25-07-21-Run1-Vetbond-Cir": "60min_0721"}


def list_recordings(feature_mode="patchx"):
    root = FEATS_ROOT[feature_mode]
    out = []
    for name in sorted(os.listdir(PAIRS)):
        if name.startswith("Missing") or name in BROKEN_GT:
            continue
        if not os.path.isdir(os.path.join(PAIRS, name)):
            continue
        feats_name = CLSX_ALIAS.get(name, name) if feature_mode == "clsx" else name
        if feature_mode == "patchx_mil" and not os.path.isfile(
                os.path.join(root, feats_name, "preds.npy")):
            continue
        if not os.path.isfile(os.path.join(root, feats_name, "feats.npy")):
            continue
        if not glob.glob(os.path.join(PAIRS, name, "*.tsv")):
            continue
        out.append(name)
    return out


def load_recording(name, feature_mode="patchx"):
    """-> feats (T, D) float32, events [(time_s, class_idx)]"""
    root = FEATS_ROOT[feature_mode]
    feats_name = CLSX_ALIAS.get(name, name) if feature_mode == "clsx" else name
    feats = np.load(os.path.join(root, feats_name, "feats.npy")).astype(np.float32)
    if feature_mode == "patchx_mil":
        logits = np.load(os.path.join(root, feats_name, "preds.npy")).astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))                  # (T/4, 7)
        probs = np.repeat(probs, 4, axis=0)                     # -> 0.5s grid
        n = min(len(feats), len(probs))
        feats = np.concatenate([feats[:n], probs[:n]], axis=1)  # (T, 775)
    elif feature_mode == "patchx_motion":
        delta = np.diff(feats, axis=0, prepend=feats[:1])       # (T, 768)
        feats = np.concatenate([feats, delta], axis=1)          # (T, 1536)
    tsv = sorted(glob.glob(os.path.join(PAIRS, name, "*.tsv")))[0]
    anns = d11.parse_annotations(d11.read_timestamps(tsv))
    events = [(a["time"], FG_CLASSES.index(a["behavior"]))
              for a in anns if a["behavior"] in FG_CLASSES]
    return feats, events


def gap_none_windows(name, master_duration, none_ratio=0.6, seed=0):
    """Recover the same None-clip (hard-negative, pre-annotation-gap) 4s
    windows data11make.py's generate_clips_for_pair/generate_none_clips_in_gap
    would carve out of this recording for the MIL backbone's own clip-
    classification training set -- same algorithm (walk annotations in order,
    top up None clips in each gap toward `none_ratio` of the running clip
    total via none_clips_remaining/sample_none_clip_starts, skip a candidate
    if collect_labels_in_window finds it isn't actually label-free), just
    without any of data11make's file I/O (no ffmpeg, no clip_rows/CSV).
    Foreground clips are walked through too (to advance cursor/clip_counts
    identically) but not returned -- FishFormer already gets foreground
    supervision from `events` via assign_targets, this is only for sampling
    extra background-heavy training windows.

    Returns [(start_s, end_s), ...] None-clip windows, time-sorted.
    """
    tsv = sorted(glob.glob(os.path.join(PAIRS, name, "*.tsv")))[0]
    annotations = d11.parse_annotations(d11.read_timestamps(tsv))
    rng = random.Random(seed)
    cd = d11.CLIP_DURATION
    none_windows = []
    clip_counts = {"none": 0, "total": 0}
    cursor = 0.0
    ann_idx = 0

    while ann_idx < len(annotations):
        while ann_idx < len(annotations) and annotations[ann_idx]["time"] < cursor:
            ann_idx += 1
        if ann_idx >= len(annotations):
            break

        next_ann_time = annotations[ann_idx]["time"]
        gap = next_ann_time - cursor
        if gap > 2 * cd:
            remaining = d11.none_clips_remaining(clip_counts["none"], clip_counts["total"], none_ratio)
            if remaining > 0:
                slot_size = int(2 * cd)
                max_in_gap = int(gap // slot_size)
                num_to_create = min(remaining, max_in_gap)
                if num_to_create > 0:
                    starts = d11.sample_none_clip_starts(rng, cursor, next_ann_time, num_to_create)
                    for clip_start in sorted(starts):
                        clip_end = min(clip_start + cd, master_duration)
                        if clip_end - clip_start <= 0:
                            continue
                        behavior, _, _ = d11.collect_labels_in_window(annotations, clip_start, clip_end)
                        if behavior:
                            continue
                        none_windows.append((clip_start, clip_end))
                        clip_counts["total"] += 1
                        clip_counts["none"] += 1
                        cursor = max(cursor, clip_end)

        anchor = annotations[ann_idx]
        clip_start, clip_end = d11.sample_clip_window(rng, anchor["time"], cursor, master_duration)
        if clip_end - clip_start <= 0:
            ann_idx += 1
            continue
        behavior, _, _ = d11.collect_labels_in_window(annotations, clip_start, clip_end)
        if not behavior:
            ann_idx += 1
            continue
        clip_counts["total"] += 1
        cursor = clip_end

    return none_windows

