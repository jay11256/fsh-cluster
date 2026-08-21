#!/usr/bin/env python3
"""Write reduced-point SAM3 pkls by subsampling the existing 18-point ones.

WHY SUBSAMPLE RATHER THAN RE-RUN SAM3. run_sam3.py consumes each frame's mask
in memory via get_uniform_points() and persists only the sampled coordinates --
pred_tracks/pred_visibility/obj_ids/point_queries -- so the masks are gone and
a different point count normally means re-running segmentation over all 9,184
clips. Subsampling the points that were already kept avoids that entirely. It
can only go DOWN in count, which is the informative direction anyway: SAM3
grounding is worth 9.5 mAP, and the open question is how much of it is needed.

LAYOUT. Each pkl holds 2 objects x 9 points = 18 columns, object-major
(columns 0-8 are the first fish, 9-17 the second), where the 9 points per fish
came from a 3x3 grid over that fish's mask in row-major order:

    0 1 2
    3 4 5
    6 7 8

  --points 8  keeps the four grid CORNERS per fish (0, 2, 6, 8), i.e. a 2x2
              subsampling that preserves the spatial extent of the animal.
  --points 2  keeps the grid CENTRE per fish (4), the closest thing to a
              centroid, reducing each animal to a single tracked location.

Both preserve the object-major ordering and the per-object balance, so obj_ids
stays meaningful and the model still sees both animals.

Note the model factorizes the point count into a 2D layout as
[grid, n/grid] with grid the middle divisor (pointformer.get_point_grid_size):
18 -> 6x3, 8 -> 4x2, 2 -> 2x1. All three are valid.

    python3 tools/subsample_sam3_points.py --points 8
    python3 tools/subsample_sam3_points.py --points 2 --check
"""
import os
import glob
import pickle
import argparse

import torch

SRC = "/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06"
DST_TMPL = "/fs/vulcan-projects/fsh_track/processed_data/sam3pklds12_06_p{n}"

N_OBJ, N_PT = 2, 9
KEEP = {
    8: [0, 2, 6, 8],   # 3x3 grid corners -> 2x2
    2: [4],            # grid centre
}


def columns_for(keep):
    """Object-major column indices, so both fish keep the same grid positions."""
    return [o * N_PT + k for o in range(N_OBJ) for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, required=True, choices=sorted(KEEP))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ofure", action="store_true",
                    help="also write per-recording ofure trees "
                         "(<rec>/pkls_p{n}/clip_NNNNN.pkl) for the feature dump")
    args = ap.parse_args()

    dst = DST_TMPL.format(n=args.points)
    cols = columns_for(KEEP[args.points])
    assert len(cols) == args.points, f"{len(cols)} != {args.points}"

    src_files = sorted(glob.glob(os.path.join(SRC, "*.pkl")))
    if not src_files:
        raise SystemExit(f"no pkls in {SRC}")

    if args.check:
        made = glob.glob(os.path.join(dst, "*.pkl"))
        print(f"{len(made)} / {len(src_files)} pkls in {dst}")
        if made:
            d = pickle.load(open(made[0], "rb"))
            for k, v in d.items():
                print(f"  {k}: {tuple(v.shape)}")
        return

    os.makedirs(dst, exist_ok=True)
    idx = torch.tensor(cols, dtype=torch.long)
    n_done = 0
    for p in src_files:
        out = os.path.join(dst, os.path.basename(p))
        if os.path.exists(out):
            continue
        d = pickle.load(open(p, "rb"))
        red = {
            "pred_tracks": d["pred_tracks"][:, idx],        # (frames, n, 2)
            "pred_visibility": d["pred_visibility"][:, idx],  # (frames, n)
            "obj_ids": d["obj_ids"][idx],                    # (n,)
            "point_queries": d["point_queries"][idx],        # (n,)
        }
        with open(out, "wb") as f:
            pickle.dump(red, f)
        n_done += 1
        if n_done % 2000 == 0:
            print(f"  {n_done} written", flush=True)

    print(f"wrote {n_done} pkls ({len(src_files)} total) to {dst}")
    d = pickle.load(open(os.path.join(dst, os.path.basename(src_files[0])), "rb"))
    print("  sample shapes: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in d.items()))
    print("  obj_ids: " + str(d["obj_ids"].tolist()))

    if args.ofure:
        write_ofure(args.points, KEEP[args.points])


def write_ofure(points, keep):
    """Per-recording reduced pkls under OFURE_ROOT/<rec>/pkls_p{n}/.

    The feature dump walks the ofure tree, whose point files are named
    clip_NNNNN.pkl per recording -- a different naming AND a different set of
    files from the dataset12_06-named pkls the backbone trains on. Unlike the
    static uniform grid, reduced SAM3 points differ per clip, so these cannot be
    symlinks to one canonical file and must be subsampled individually.
    """
    # Two recordings (25-05-22-Run1-Sham-Cir, 25-07-21-Run1-Vetbond-Cir) have
    # EMPTY clips dirs under ofure -- their data lives in these standalone
    # 60min_* trees under the CLSX_ALIAS names, and dump_missing_two.py reaches
    # them directly. They use the same clip_NNNNN.pkl naming, so they need the
    # same reduced subdirectory or the finish step cannot find points for them.
    roots = ["/fs/vulcan-projects/fsh_track/bhargav/data/ofure"]
    extra = sorted(glob.glob("/fs/vulcan-projects/fsh_track/bhargav/data/60min_*"))
    cols = torch.tensor(columns_for(keep), dtype=torch.long)
    n_rec = n_file = 0
    dirs = [(r, os.path.join(root, r)) for root in roots for r in sorted(os.listdir(root))]
    dirs += [(os.path.basename(d), d) for d in extra]
    for rec, base in dirs:
        src = os.path.join(base, "pkls")
        if not os.path.isdir(src):
            continue
        dst = os.path.join(base, f"pkls_p{points}")
        os.makedirs(dst, exist_ok=True)
        n_rec += 1
        for f in sorted(os.listdir(src)):
            if not f.endswith(".pkl"):
                continue
            out = os.path.join(dst, f)
            if os.path.exists(out):
                continue
            d = pickle.load(open(os.path.join(src, f), "rb"))
            pickle.dump({"pred_tracks": d["pred_tracks"][:, cols],
                         "pred_visibility": d["pred_visibility"][:, cols],
                         "obj_ids": d["obj_ids"][cols],
                         "point_queries": d["point_queries"][cols]},
                        open(out, "wb"))
            n_file += 1
    print(f"ofure trees: {n_rec} recordings, {n_file} new pkls written")


if __name__ == "__main__":
    main()
