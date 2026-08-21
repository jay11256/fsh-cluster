"""FishFormer -- anchor-free temporal action localization for cichlid behavior.

Predicts behavior spans directly from a Trokens feature stream: a dilated-conv
+ transformer trunk and three 1-D CNN heads emitting per-timestep class logits,
centerness, and binned distance-to-boundary regression (FCOS/ActionFormer-style
with TriDet-style distributional offsets).

Single-scale by design: because supervision comes from BORIS points expanded to
a fixed-width box, every ground-truth segment has the same duration and a
feature pyramid's coarser levels receive no positive assignments at all. See
`former.py`'s module docstring for the measurement.
"""
from .former import FishFormer, FishFormerLoss, CrossMotionLite
from .blocks import TemporalBlock, DilatedTemporalConv
from .nms import _nms

__all__ = ["FishFormer", "FishFormerLoss", "CrossMotionLite",
           "TemporalBlock", "DilatedTemporalConv", "_nms"]
