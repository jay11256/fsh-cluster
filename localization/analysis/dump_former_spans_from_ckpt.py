"""Dump predicted spans for all 17 recordings from the ALREADY-TRAINED
Ds12_06_5fold_fold{0-4}_Former2 checkpoints (the few-shot backbone, the
paper's primary/reported FishFormer numbers -- see checkpoints/*.pt's own
`config`/`result`, which reproduce fishtal_results.json's reported avg mAP
to 3 decimal places).

Unlike dump_former_spans.py (which retrains a fresh model per recording,
excluding only that one recording from ALL 17 -- correct for the old
"Leave3" leave-one-out protocol, but NOT the actual 5-fold protocol these
checkpoints were trained under, where each recording's model excludes its
entire fold, ~13-14 recordings' training pool, not 16), this script loads
each recording's own already-trained fold checkpoint directly and only runs
inference -- no retraining, so it uses the EXACT weights behind the numbers
already reported in the paper (checkpoints/Ds12_06_5fold_fold{F}_Former2__
<recording>.pt, one per recording, saved by run_former_5fold_ckpt.sbatch).

Writes span_dumps/Ds12_06_5fold_Former2__<recording>.json, one per
recording, 17 total. Schema matches the existing span_dumps/*.json files
exactly (label, recording, epoch, stride, n_steps, duration, gt_events,
spans, metrics) except "epoch" is null -- the checkpoint only stores the
best-epoch model weights, not which epoch number that was.
"""
import os
import sys
import glob
import json
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: data dirs + fishformer package
sys.path.insert(0, ROOT)
from fishformer.former import FishFormer                                  # noqa: E402
from fishformer.data import NUM_CLASSES, FG_CLASSES, load_recording       # noqa: E402
from fishformer.train_former import predict_spans, evaluate               # noqa: E402

CKPT_DIR = f"{ROOT}/checkpoints"
OUT_DIR = f"{ROOT}/span_dumps"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_LABEL = "Ds12_06_5fold_Former2"


def dump_one(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    args = SimpleNamespace(**cfg)
    test_rec = ckpt["test_rec"]
    feat_dim = ckpt["feat_dim"]

    feats, _ = load_recording(test_rec, args.feature_mode)
    spatial_pool = feats.ndim == 3

    model = FishFormer(feat_dim=feat_dim, num_classes=NUM_CLASSES, hidden=args.hidden,
                       depth=args.depth, num_heads=args.num_heads, drop=args.drop,
                       drop_path=args.drop_path,
                       reg_bins=args.reg_bins, spatial_pool=spatial_pool,
                       use_motion=args.use_motion).to(device)
    model.load_state_dict(ckpt["model"])

    res = evaluate(model, test_rec, args, device)
    props, events, stride, n = predict_spans(model, test_rec, args, device)

    dump = {
        "label": OUT_LABEL, "recording": test_rec, "epoch": None,
        "stride": stride, "n_steps": n, "duration": n * stride,
        "gt_events": [[float(t), FG_CLASSES[c]] for t, c in events],
        "spans": {FG_CLASSES[c]: [[float(s), float(e), float(sc)] for s, e, sc in v]
                  for c, v in props.items()},
        "metrics": {k: v for k, v in res.items() if k in
                    ("avg_map", "point_recall", "point_precision")},
    }
    out_path = f"{OUT_DIR}/{OUT_LABEL}__{test_rec}.json"
    json.dump(dump, open(out_path, "w"), indent=1)
    print(f"{test_rec}: avg_mAP={res['avg_map']:.4f} "
          f"(ckpt result was {ckpt['result']['avg_map']:.4f}) -> {out_path}", flush=True)
    return res["avg_map"], ckpt["result"]["avg_map"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/Ds12_06_5fold_fold*_Former2__*.pt"))
    ckpts = [c for c in ckpts if "_Neural" not in c and "_None" not in c
             and "_NoPyramid" not in c]
    print(f"{len(ckpts)} checkpoints | device={device}", flush=True)
    mismatches = []
    for c in ckpts:
        fresh, stored = dump_one(c, device)
        if abs(fresh - stored) > 1e-4:
            mismatches.append((os.path.basename(c), fresh, stored))
    if mismatches:
        print("\n!! avg_mAP mismatch vs stored checkpoint result (possible nondeterminism):")
        for name, fresh, stored in mismatches:
            print(f"   {name}: fresh={fresh:.4f} stored={stored:.4f}")
    else:
        print(f"\nAll {len(ckpts)} recordings' freshly-computed avg_mAP matched their "
              f"checkpoint's stored result exactly (within 1e-4).")


if __name__ == "__main__":
    main()
