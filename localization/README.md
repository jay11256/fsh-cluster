# FishFormer

Anchor-free temporal action localization for cichlid social behavior.

FishFormer takes a Trokens feature stream over a full ~60-minute recording and
predicts *timestamped behavior spans* — it regresses span boundaries directly
rather than deriving them by thresholding an actionness curve. A dilated-conv +
transformer trunk feeds three 1-D CNN heads emitting per-timestep class logits,
centerness, and binned distance-to-boundary offsets (FCOS/ActionFormer-style
detection with TriDet-style distributional regression).

Behaviors are annotated in BORIS as *point* events — a single timestamp, not a
start/end — so the true extent is unknown even to the annotator. Predicting a
distribution over each offset rather than a point estimate is therefore matched
to what the annotation actually supports, and is the design choice the model is
built around.

**The model is single-scale, deliberately.** Earlier versions carried the
4-level feature pyramid (strides 1/2/4/8) that ActionFormer and TriDet use, on
the reasoning that behaviors of differing duration should be owned by different
scales. That does not apply here: supervision is a BORIS point expanded to a
fixed `span_s` box, so every ground-truth segment is exactly the same width and
only the finest level ever receives a positive assignment. Measured on 64 real
training windows, level 0 got 7,323 positives and levels 1/2/3 got 0/0/0 — 5.3M
parameters, 25% of the model, trained solely to emit background, whose
untrained predictions then had to be suppressed by NMS. Dropping them is a
strict simplification: the set of supervised positions is bit-for-bit
unchanged. A pyramid would be worth revisiting if the annotations ever carried
real, varying durations.

## Layout

```
fishformer/                 the model and its training/eval code
├── former.py               FishFormer, FishFormerLoss, CrossMotionLite
├── blocks.py               TemporalBlock, DilatedTemporalConv (Trokens-parameterised)
├── nms.py                  temporal NMS over scored spans
├── data.py                 feature-bank loading, BORIS parsing, fold definitions
└── train_former.py         window dataset, target assignment, decode, mAP eval, CLI

analysis/                   post-hoc studies, all run on saved checkpoints
├── dump_former_spans.py            span dumps from a fresh training run
├── dump_former_spans_from_ckpt.py  span dumps from saved per-fold checkpoints
├── decode_variants.py              class-agnostic vs class-specific NMS; multilabel vs argmax
├── sweep_cand_thresh.py            candidate-generation threshold sweep
├── nested_threshold_sweep.py       non-leaky CV selection of the operating threshold
├── measure_span_len.py             predicted-span duration distribution
├── confusion_matrix_fishformer.py  per-class confusion, single- and multi-label
└── plot_former_boxes.py            prediction-vs-ground-truth timeline figures

slurm/                      one launcher per experiment
```

Source only. Checkpoints, span dumps, figures, logs and result JSONs are
generated at run time into the repository root (`checkpoints/`, `span_dumps/`,
`box_viz/`, `logs/`) and are not tracked — see the repo `.gitignore`.

## Running

Everything is launched from this directory, with SLURM scripts referenced by
path so the working directory is unambiguous:

```bash
mkdir -p logs
sbatch slurm/run_former_5fold_ckpt.sbatch     # the main result: array 0-4, one fold per task
```

That trains leave-one-recording-out within each of 5 folds and appends per-fold
results to `fishtal_results.json`. Ablations follow the same pattern:

| launcher | ablation |
|---|---|
| `run_window_sweep.sbatch`, `run_window_sweep_small.sbatch` | context window ∈ {8, 15, 30, 45, 180}s against the default 90s |
| `run_former_5fold_neural_ckpt.sbatch` | neural rather than few-shot Trokens backbone |
| `run_former_5fold_none_ckpt.sbatch` | with curated hard-negative windows |

The analyses reuse saved checkpoints and retrain nothing
(`run_dump_from_ckpt.sbatch`, `run_decode_variants.sbatch`,
`run_sweep_cand_thresh.sbatch`, `run_confusion_matrix.sbatch`).
`analysis/nested_threshold_sweep.py` is CPU-only and reads span dumps directly.

To train outside SLURM, the package is a module:

```bash
python -m fishformer.train_former --fold <recording> --feature-mode ds12_06_5fold_fold0 \
    --window-s 90 --span-s 4 --reg-bins 16 --epochs 100
```

## Dependencies

Python with PyTorch, NumPy, scikit-learn and Matplotlib, plus three modules
outside this directory:

| module | location | needed by |
|---|---|---|
| `trokens.models.{common,attention}` | `../trokens/` (this repo) | `blocks.py` — `Attention`, `Mlp`, `DropPath` |
| `visualize_matrix` | `../pipeline/` (this repo) | mAP helpers in training and every decode script |
| `data11make` | dataset-generation tree, outside this repo | `data.py` — BORIS annotation parsing |

The two in-repo dependencies resolve relative to this file, so a clone works
from any checkout path. The external ones, and the data locations, are
environment-overridable:

| variable | default | meaning |
|---|---|---|
| `FSH_TROKENS_ROOT` | `../trokens` | Trokens package root |
| `FSH_DATASET_GEN` | absolute path to the dataset-generation tree | where `data11make` lives |
| `FSH_PAIRS` | absolute path under `raw_data/processed_ofure` | BORIS annotation TSVs, one directory per recording |

Feature banks (`feats.npy`, 768-d Trokens features at a 0.25 s stride, one
directory per fold) are named in `FEATS_ROOT` in `fishformer/data.py` and must
be repointed to wherever they have been staged.
