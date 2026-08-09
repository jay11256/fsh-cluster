"""Box-timeline visualization for FishFormer predictions, reusing
visualize_matrix.py (the same script used throughout this project for
MIL/ASM-Loc box visualizations) unmodified.

visualize_matrix() expects a dense per-clip score matrix (num_behaviors,
num_clips), since MIL/ASM-Loc score every clip on a fixed stride. FishFormer
instead emits final (start, end, score) spans directly (after NMS), so there
is no native "matrix" to hand it. The adapter here rasterizes those spans
onto a fine per-step grid at the model's own feature stride (0.25s for
leave3_nf16/wlamo_3shot) and thresholds at 0.5 -- since the grid is binary,
visualize_matrix's own span-merging (_build_pred_spans) reconstructs exactly
the FishFormer boxes (to within one grid step), so the plot shows the real
predicted boxes, not an approximation.

Usage:
    python plot_former_boxes.py span_dumps/Leave3_NF16_Former2__25-05-22-Run1-Sham-Cir.json
"""
import os
import sys
import glob
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: data dirs + fishformer package
sys.path.insert(0, ROOT)
from fishformer.data import FG_CLASSES, PAIRS                                    # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline"))
from visualize_matrix import visualize_matrix                         # noqa: E402

OUT_DIR = f"{ROOT}/box_viz"
os.makedirs(OUT_DIR, exist_ok=True)


def _gt_tsv(name):
    hits = sorted(glob.glob(os.path.join(PAIRS, name, "*.tsv")))
    if not hits:
        raise FileNotFoundError(f"no GT tsv found for {name} under {PAIRS}")
    return hits[0]


def plot_dump(dump_path):
    d = json.load(open(dump_path))
    rec, stride, duration = d["recording"], d["stride"], d["duration"]
    label = d["label"]

    n_clips = int(np.ceil(duration / stride))
    pred_matrix = np.zeros((len(FG_CLASSES), n_clips), dtype=np.float32)
    clip_starts = np.arange(n_clips) * stride
    clip_ends = clip_starts + stride
    clip_centers = (clip_starts + clip_ends) / 2

    for ci, cls in enumerate(FG_CLASSES):
        for s, e, _score in d["spans"].get(cls, []):
            hit = (clip_centers >= s) & (clip_centers < e)
            pred_matrix[ci, hit] = 1.0

    gt_path = _gt_tsv(rec)
    save_dir = f"{OUT_DIR}/{label}__{rec}"
    print(f"[{label}] {rec}: {sum(len(v) for v in d['spans'].values())} predicted spans, "
          f"{len(d['gt_events'])} GT events, duration={duration:.1f}s, stride={stride}s "
          f"-> {save_dir}", flush=True)

    visualize_matrix(
        ground_truth_path=gt_path,
        pred_matrix=pred_matrix,
        threshold=0.5,
        window_len=stride,
        overlap_len=0.0,
        behavior_names=FG_CLASSES,
        save_path=save_dir,
        video_window=(0.0, duration),
    )
    print(f"  wrote {save_dir}/visualize_predictions.png", flush=True)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        plot_dump(path)
