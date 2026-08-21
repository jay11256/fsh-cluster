#!/usr/bin/env python3
"""Build the point pkls for the NO-KEYPOINT ablation (PT_DATA=none) on ds12_06.

WHY THIS EXISTS. Setting POINT_INFO.ENABLE=False is not sufficient to run
Trokens without keypoints on this dataset, for two reasons found by reading the
code rather than the config:

  1. base_ds.py:160 does `pickle.load(open(feat_path,'rb'))` UNCONDITIONALLY --
     there is no ENABLE guard -- so every clip still needs a pkl to exist.
     trokens_exp.sh's `none` branch points at the ds6 cotracker dump, which
     holds 2,515 pkls all named 080225_spawn_B1-5_clipNNN.pkl and not one
     ds12_06 clip, so that path is an immediate FileNotFoundError here.
  2. With ENABLE=False, pointformer.py:423 takes its else-branch and
     `sampled_feat` becomes the full 16x16=256 DINO patch grid. But lines
     492-499 then ADD the motion features to it, and those are computed from
     the pkl's own points (hod_feat comes from pred_tracks). So the pkl's point
     count must equal the patch count, 256. The ds6 dump has 192, which would
     not have matched either. And line 116 asserts at least one motion module
     stays enabled, so they cannot simply be switched off.

WHAT THIS WRITES. A static 16x16 grid of 256 points, uniformly spanning the
1280x720 clip frame, constant across frames. That satisfies both constraints
and gives the intended semantics: the model never grid-samples at the points
(ENABLE=False), and because the grid does not move, the HOD and cross-motion
modules see zero displacement and contribute no trajectory information. The
result is Trokens as a pure appearance model over a uniform patch grid.

The content is identical for every clip -- a fixed grid does not depend on the
video -- so this writes ONE real pkl and symlinks every clip name to it, which
costs kilobytes instead of ~2GB.

    python3 tools/make_grid_pkls.py            # write
    python3 tools/make_grid_pkls.py --check    # verify only
"""
import os
import glob
import pickle
import argparse

import torch

SAM3_DIR = "/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06"
OUT_DIR = "/fs/vulcan-projects/fsh_track/processed_data/gridpklds12_06"
# Second naming scheme. The BACKBONE trains off dataset12_06, whose clips are
# named "<recording>_clipNNNNN.mp4"; the FEATURE DUMP walks the ofure tree,
# whose clips are "clip_NNNNN.mp4" per recording. Same static grid either way,
# so the ofure-named directory is the identical canonical pkl under the names
# dump_feats_new_backbone.py will actually look up. Without it the dump dies
# with FileNotFoundError on clip_00000.pkl.
OFURE_DIR = "/fs/vulcan-projects/fsh_track/processed_data/gridpkls_ofure"
OFURE_N = 1900        # > max clips in any recording (1801); extras are inert
CANON = "_uniform_grid_16x16.pkl"

GRID = 16                      # 16x16 = 256, matching (224/14)^2 DINO patches
N_FRAMES = 100                 # >= any clip's frame count; indexed independently
W, H = 1280, 720               # native clip resolution, the space sam3 coords use


def build_canonical():
    """A 16x16 grid at pixel centres, repeated over frames (i.e. never moving)."""
    xs = (torch.arange(GRID, dtype=torch.float32) + 0.5) * (W / GRID)
    ys = (torch.arange(GRID, dtype=torch.float32) + 0.5) * (H / GRID)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)      # (256, 2)
    tracks = pts.unsqueeze(0).repeat(N_FRAMES, 1, 1)                 # (T, 256, 2)
    return {
        "pred_tracks": tracks,
        "pred_visibility": torch.ones(N_FRAMES, GRID * GRID, dtype=torch.bool),
        # No object grounding: every point belongs to the same nominal "object".
        "obj_ids": torch.zeros(GRID * GRID, dtype=torch.int64),
        "point_queries": torch.zeros(GRID * GRID, dtype=torch.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ofure", action="store_true",
                    help="also emit clip_NNNNN.pkl names for the dump")
    args = ap.parse_args()

    names = [os.path.basename(p) for p in glob.glob(os.path.join(SAM3_DIR, "*.pkl"))]
    if not names:
        raise SystemExit(f"no sam3 pkls found in {SAM3_DIR}")

    if args.check:
        made = glob.glob(os.path.join(OUT_DIR, "*.pkl"))
        print(f"{len(made)} pkls in {OUT_DIR} (sam3 has {len(names)})")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    canon_path = os.path.join(OUT_DIR, CANON)
    with open(canon_path, "wb") as f:
        pickle.dump(build_canonical(), f)

    n_new = 0
    for name in names:
        if name == CANON:
            continue
        dst = os.path.join(OUT_DIR, name)
        if not os.path.lexists(dst):
            os.symlink(CANON, dst)      # relative: survives moving the directory
            n_new += 1
    print(f"canonical grid: {canon_path}")
    print(f"{n_new} new symlinks; {len(names)} clips covered in {OUT_DIR}")

    if args.ofure:
        os.makedirs(OFURE_DIR, exist_ok=True)
        src = os.path.join(OFURE_DIR, CANON)
        with open(src, "wb") as f:
            pickle.dump(build_canonical(), f)
        made = 0
        for i in range(OFURE_N):
            dst = os.path.join(OFURE_DIR, f"clip_{i:05d}.pkl")
            if not os.path.lexists(dst):
                os.symlink(CANON, dst)
                made += 1
        print(f"{made} ofure-named symlinks in {OFURE_DIR}")


if __name__ == "__main__":
    main()
