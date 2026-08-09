"""Temporal building blocks shared by FishFormer's trunk and pyramid.

Both classes are lifted unchanged from the earlier FishTAL model, which is why
they are parameterised from Trokens' own `Attention`/`Mlp`/`DropPath` rather
than torch.nn's: the temporal head was built to sit directly on top of a frozen
Trokens backbone and reuse its parameterisation.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Trokens' attention/MLP primitives live in the sibling `trokens/` package of
# this repo. Resolved relative to this file so a clone works from any checkout
# path; override with the FSH_TROKENS_ROOT environment variable if the Trokens
# tree lives elsewhere.
TROKENS_ROOT = os.environ.get(
    "FSH_TROKENS_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "trokens"))
if TROKENS_ROOT not in sys.path:
    sys.path.insert(0, TROKENS_ROOT)
from trokens.models.common import DropPath, Mlp          # noqa: E402
from trokens.models.attention import Attention           # noqa: E402


class TemporalBlock(nn.Module):
    """Pre-norm self-attention + MLP over the time axis.

    Trokens' own `Block` is written for space-time token grids; this is the same
    pre-norm residual pattern specialised to a plain 1-D temporal sequence, built
    from Trokens' `Attention` and `Mlp` so the parameterisation matches.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DilatedTemporalConv(nn.Module):
    """Dilated depthwise-separable conv stack, run before attention.

    Behaviors here are short (a bite is well under a second) but their context is
    long, and attention alone over a 180-step window starts from no locality
    prior at all. The dilation ladder gives an explicitly multi-scale receptive
    field cheaply, which matters because our per-class event counts are far too
    small to learn that structure from data.
    """

    def __init__(self, dim, dilations=(1, 2, 4, 8), drop=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=d, dilation=d, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout(drop),
            ))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                      # (B, T, C)
        h = x.transpose(1, 2)                  # (B, C, T)
        for layer in self.layers:
            h = h + layer(h)                   # residual per dilation
        return self.norm(h.transpose(1, 2))
