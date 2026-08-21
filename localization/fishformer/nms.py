"""Temporal NMS over scored (start, end) spans.

Extracted verbatim from the FishTAL trainer, where it was first written; every
FishFormer decode path (training-time eval, span dumps, the decode-variant and
candidate-threshold sweeps) uses this one implementation so that proposal
suppression is identical across them.
"""


def _nms(spans, iou_thr):
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: -x[2])
    keep = []
    for s, e, sc in spans:
        ok = True
        for ks, ke, _ in keep:
            inter = max(0.0, min(e, ke) - max(s, ks))
            union = (e - s) + (ke - ks) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append((s, e, sc))
    return keep
