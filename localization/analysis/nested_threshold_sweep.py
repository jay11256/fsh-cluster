"""Nested (non-leaky) cross-validated selection of FishFormer's reported
operating threshold, reproducing the appendix's 0.18 / F1 48.6% claim.

The paper documents this analysis but the script that produced it is not in
the repo (searched; not found), so this reimplements it from the persisted
per-recording span dumps (span_dumps/Ds12_06_5fold_Former2__*.json, written
by dump_former_spans_from_ckpt.py from each recording's own fold checkpoint).

PROTOCOL (the "nested"/non-leaky part). For each fold f:
  1. Take the OTHER four folds' recordings as the selection set.
  2. Sweep the score threshold over a grid; pick t_f maximizing F1 there.
  3. Apply that single fixed t_f to fold f's held-out recordings and record
     the resulting recall/precision.
Fold f's own recordings never influence the threshold applied to them, which
is what distinguishes this from the earlier "oracle" sweep (threshold picked
on the same fold it is scored on) that the appendix says it supersedes.

Note on what is and isn't held out: each recording's SPANS already come from
a model that never trained on it (its own fold checkpoint). This script adds
the second level -- the threshold, a hyperparameter shared across folds, is
also chosen without seeing the recordings it is applied to.

METRIC matches train_former.evaluate(): per recording, point-recall and
point-precision are averaged over classes; those per-recording values are
averaged across recordings; F1 is then 2rp/(r+p) of those means ("F1 of
means", the same convention slides 8/11-13 use).

CPU-only, reads the dumps -- no GPU or model loading needed.
"""
import os
import sys
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: data dirs + fishformer package
DUMPS = f"{ROOT}/span_dumps"
OUT_PATH = f"{ROOT}/nested_threshold_results.json"
LABEL = "Ds12_06_5fold_Former2"

# The ds12_06_5fold partition, identical to run_former_5fold_ckpt.sbatch's
# FOLD_TEST array (kept in sync by hand -- verified against that file).
FOLDS = [
    ["25-05-22-Run1-Sham-Cir", "25-06-26-Run1-VetBond-NoCir",
     "25-07-21-Run1-Vetbond-Cir", "25-07-21-Run2-Sham-NoCir"],
    ["25-06-26-Run2-VetBond-NoCir", "25-07-18-Run1-Sham-Cir",
     "25-08-07-Run1-Vetbond-Cir", "25-07-24-Run2-Sham-NoCir"],
    ["25-07-18-Run2-Vetbond-NoCir", "25-08-06-Run1-Sham-Cir",
     "25-08-08-Run1-Vetbond-Cir"],
    ["25-08-13-Run1-Sham-Cir", "25-08-26-Run1-Vetbond-Cir",
     "25-08-01-Run1-Vetbond-NoCir"],
    ["25-05-21-Run1-Sham-NoCir", "25-08-14-Run1-Sham-Cir",
     "25-08-15-Run1-Vetbond-NoCir"],
]

GRID = np.round(np.arange(0.01, 0.91, 0.01), 2)


def load_dumps():
    out = {}
    for path in sorted(glob.glob(f"{DUMPS}/{LABEL}__*.json")):
        d = json.load(open(path))
        gt = {}
        for t, c in d["gt_events"]:
            gt.setdefault(c, []).append(float(t))
        out[d["recording"]] = {"spans": d["spans"], "gt": gt}
    return out


def rec_metrics(rec, thresh):
    """Per-recording (point_recall, point_precision), averaged over classes."""
    rl, pl = [], []
    for c, spans in rec["spans"].items():
        kept = [(s, e) for s, e, sc in spans if sc >= thresh]
        pts = rec["gt"].get(c, [])
        if pts:
            rl.append(float(np.mean([any(s <= t <= e for s, e in kept) for t in pts])))
        if kept:
            pl.append(float(np.mean([any(s <= t <= e for t in pts) for s, e in kept])))
    return (float(np.mean(rl)) if rl else np.nan,
            float(np.mean(pl)) if pl else np.nan)


def agg_f1(dumps, recs, thresh):
    rs, ps = [], []
    for name in recs:
        r, p = rec_metrics(dumps[name], thresh)
        if not np.isnan(r):
            rs.append(r)
        if not np.isnan(p):
            ps.append(p)
    r = float(np.mean(rs)) if rs else 0.0
    p = float(np.mean(ps)) if ps else 0.0
    f1 = float(2 * r * p / (r + p)) if (r + p) > 0 else 0.0
    return r, p, f1


def main():
    dumps = load_dumps()
    print(f"loaded {len(dumps)} recording dumps from {DUMPS}")
    missing = [r for f in FOLDS for r in f if r not in dumps]
    if missing:
        print(f"!! missing dumps for: {missing}")
        sys.exit(1)

    # ---- oracle (leaky) reference: one global threshold picked on ALL 17 ----
    all_recs = [r for f in FOLDS for r in f]
    oracle_curve = [(t,) + agg_f1(dumps, all_recs, t) for t in GRID]
    o_t, o_r, o_p, o_f1 = max(oracle_curve, key=lambda x: x[3])
    print(f"\n[oracle / leaky, for contrast] best global t={o_t:.2f} -> "
          f"R={100*o_r:.1f}% P={100*o_p:.1f}% F1={100*o_f1:.1f}%")

    # ---- nested: per fold, select on the OTHER four folds only ----
    print("\nnested (non-leaky) selection:")
    per_fold, sel_ts = [], []
    for f, test_recs in enumerate(FOLDS):
        sel_recs = [r for g, fold in enumerate(FOLDS) if g != f for r in fold]
        curve = [(t,) + agg_f1(dumps, sel_recs, t) for t in GRID]
        t_f = max(curve, key=lambda x: x[3])[0]
        r, p, f1 = agg_f1(dumps, test_recs, t_f)
        sel_ts.append(float(t_f))
        per_fold.append({"fold": f, "threshold": float(t_f), "n_test": len(test_recs),
                         "recall": r, "precision": p, "f1": f1})
        print(f"  fold {f}: selected t={t_f:.2f} on {len(sel_recs)} recs -> "
              f"held-out {len(test_recs)} recs: R={100*r:.1f}% P={100*p:.1f}% F1={100*f1:.1f}%")

    # Aggregate the way the paper reports it: pool all held-out recordings,
    # each scored at ITS OWN fold's selected threshold, then one r/p/F1.
    rs, ps = [], []
    for f, test_recs in enumerate(FOLDS):
        for name in test_recs:
            r, p = rec_metrics(dumps[name], sel_ts[f])
            if not np.isnan(r):
                rs.append(r)
            if not np.isnan(p):
                ps.append(p)
    R, P = float(np.mean(rs)), float(np.mean(ps))
    F1 = float(2 * R * P / (R + P))
    print(f"\nNESTED RESULT (17 recs, each at its own fold's threshold):")
    print(f"  thresholds selected: {sel_ts}  (stable={len(set(sel_ts)) == 1})")
    print(f"  recall={100*R:.1f}%  precision={100*P:.1f}%  F1={100*F1:.1f}%")
    print(f"\n  vs oracle F1 {100*o_f1:.1f}% -> optimism from leaky selection: "
          f"{100*(o_f1-F1):+.1f} pts")

    json.dump({"grid": GRID.tolist(), "per_fold": per_fold,
               "selected_thresholds": sel_ts,
               "nested": {"recall": R, "precision": P, "f1": F1},
               "oracle": {"threshold": float(o_t), "recall": o_r,
                          "precision": o_p, "f1": o_f1}},
              open(OUT_PATH, "w"), indent=1)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
