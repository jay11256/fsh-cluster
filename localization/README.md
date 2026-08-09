# localization

Temporal action localization for cichlid social behavior — the models that turn
the Trokens feature stream into timestamped behavior detections.

Three models share this directory, in the order they were built:

| model | files | what it does |
|---|---|---|
| **FishTAL** | `model.py` (`FishTAL`), `train.py` | dense per-timestep classification + an actionness curve; spans come from thresholding that curve |
| **FishPoint** | `point_model.py`, `train_point.py` | predicts event *points* with a sub-stride offset, scored by point-AP instead of tIoU |
| **FishFormer** | `former.py`, `train_former.py` | anchor-free detector that *regresses* span boundaries (FCOS/ActionFormer-style, TriDet-style binned offsets) — the main model |

`model.py` also holds `TemporalBlock` and `DilatedTemporalConv`, which
`former.py` and `point_model.py` both import, so the three are one module graph
rather than three independent codebases.

## Layout

```
former.py                     FishFormer: pyramid, shared 1-D CNN heads, DFL boundary regression
train_former.py               training loop, window dataset, target assignment, decode + mAP eval
model.py                      FishTAL + the shared TemporalBlock / DilatedTemporalConv
train.py                      FishTAL training; also _nms, reused by every decode script
point_model.py                FishPoint model + loss
train_point.py                FishPoint training and point-AP evaluation
data.py                       feature-bank loading, BORIS annotation parsing, fold definitions

confusion_matrix_fishformer.py  per-class confusion (single-label and multi-label)
decode_variants.py              class-agnostic vs class-specific NMS, multilabel vs argmax
dump_former_spans.py            span dumps from a fresh training run
dump_former_spans_from_ckpt.py  span dumps from saved per-recording fold checkpoints
measure_span_len.py             predicted-span duration distribution
nested_threshold_sweep.py       non-leaky cross-validated operating-threshold selection
sweep_cand_thresh.py            candidate-generation threshold sweep
sweep_inference.py              FishTAL decode sweep
sweep_point_decode.py           FishPoint decode sweep
plot_former_boxes.py            prediction-vs-GT timeline figures

run_*.sbatch                  SLURM launchers; one per experiment (see below)
```

Source only — the result records these scripts read and write (`fishtal_results.json`,
`point_results.json`, the per-sweep JSONs, and the `winsweep/` and `nopyr/`
per-fold outputs) are not tracked here. Scripts create them on first run.

## Running

Everything is launched through SLURM. The main 5-fold FishFormer run is:

```bash
sbatch run_former_5fold_ckpt.sbatch     # array 0-4, one fold per task, --save-ckpt
```

which trains leave-one-recording-out within each fold and appends per-fold
results to `fishtal_results.json`. Ablations follow the same pattern:

| script | ablation |
|---|---|
| `run_former_5fold_nopyramid.sbatch`, `run_nopyramid_5fold.sbatch` | `--n-levels 1` (single scale) |
| `run_window_sweep.sbatch`, `run_window_sweep_small.sbatch` | context window ∈ {8, 15, 30, 45, 180}s vs the default 90s |
| `run_former_5fold_neural_ckpt.sbatch` | neural (vs few-shot) Trokens backbone |
| `run_former_5fold_none_ckpt.sbatch` | with curated hard-negative windows |

Post-hoc analyses run on the saved checkpoints without retraining
(`run_dump_from_ckpt.sbatch`, `run_decode_variants.sbatch`,
`run_sweep_cand_thresh.sbatch`, `run_confusion_matrix.sbatch`).

`nested_threshold_sweep.py` is CPU-only and reads the span dumps directly.

## External dependencies

These are imported by absolute path and are **not** in this directory:

| module | location | needed by |
|---|---|---|
| `visualize_matrix` | `../pipeline/` (this repo) | every mAP/decode script |
| `trokens.models.{common,attention}` | `../trokens/` (this repo) | `model.py` (`Attention`, `Mlp`, `DropPath`) |
| `data11make` | `/fs/vulcan-projects/fsh_track/will/will_files/dataset_gen` | `data.py` (BORIS annotation parsing) |
| `point_ap` | `/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training` | `train_point.py` only |

Feature banks (`feats.npy`, 768-d Trokens features at 0.25 s stride, one
directory per fold) live under
`/fs/vulcan-projects/fsh_track/bhargav/sandboxes/asmloc_training/ds12_sweep/`
and are referenced by `FEATS_ROOT` in `data.py`. Annotations come from
`/fs/vulcan-projects/fsh_track/raw_data/processed_ofure/pairs`.

## Note on paths

These files are a verbatim copy of the scripts as they were run, so they still
contain absolute paths into the sandbox they ran from
(`/fs/vulcan-projects/fsh_track/bhargav/sandboxes/fishtal`). Nothing has been
rewritten, so what is published is exactly what produced the paper's results.
To run from a clone, three things need repointing:

1. `cd <path>` at the top of every `run_*.sbatch`
2. `HERE` in `measure_span_len.py` (the only `.py` with the sandbox path hardcoded rather than derived from `__file__`)
3. `FEATS_ROOT` / `PAIRS` in `data.py`, plus the two `sys.path.insert` lines above

Model checkpoints (~13 GB), span dumps, box visualizations and SLURM logs are
deliberately not tracked — see the repo `.gitignore`.
