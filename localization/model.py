"""FishTAL -- a Trokens-based joint classification + localization model for
cichlid behavior.

Built on Trokens (/fs/vulcan-projects/fsh_track/bhargav/fsh-cluster/trokens):
reuses its `Attention`/`Block`/`Mlp` modules and consumes the same DINOv2-backed
point-tracked features the MIL model uses. The frozen Trokens backbone stays
frozen; this is the temporal head that sits on top of it.

Why a new architecture instead of more ASM-Loc tuning (see
asmloc_training/EXPERIMENTS.md for the evidence behind each point):

1. **Exact-point supervision.** Our GT is BORIS *point* clicks -- a timestamp, not
   a measured start/end. ASM-Loc (and the MIL eval) turn each point into a fixed
   +/-2s proxy box, which mathematically caps mAP: a perfect detector of a 1s
   behavior scores IoU 0.25 against a 4s box. FishTAL never builds a box at
   training time -- the localization head regresses a Gaussian bump centered on
   the true timestamp, so supervision carries the information we actually have.

2. **Explicit background.** ASM-Loc's actionness branch gets no dense negative
   supervision; it must infer foreground/background from a video-level label
   alone. The MIL model has a NoBehavior class trained on every clip, and that is
   the single clearest difference between them (MIL point-recall 0.478 vs
   ASM-Loc's 0.263). FishTAL keeps NoBehavior as a real class in a dense
   per-timestep loss.

3. **Joint heads, shared trunk.** Classification and localization are trained
   together over one temporal encoder, so the actionness head is shaped by the
   same features that have to support class discrimination, rather than being a
   separate post-hoc stage over frozen scores.

4. **Class imbalance.** Chase/Charge and Tilt appear single-digit times per hour
   in some recordings while Bite/Quiver appear hundreds of times. The dense
   classification loss is class-weighted, which the ASM-Loc top-k ranking has no
   mechanism to do.
"""
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse Trokens' own attention/MLP building blocks.
TROKENS_ROOT = "/fs/vulcan-projects/fsh_track/bhargav/fsh-cluster/trokens"
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


class FishTAL(nn.Module):
    """Joint per-timestep classification + actionness localization head.

    Input  : (B, T, D) Trokens features (patch_x pooled per frame, 0.5 s/step).
    Outputs: class logits (B, T, C) including a background class, and an
             actionness logit (B, T) trained against Gaussian bumps at the exact
             BORIS timestamps.
    """

    def __init__(self, feat_dim=768, num_classes=7, hidden=256, depth=4,
                 num_heads=8, mlp_ratio=4.0, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, max_len=4096, dilations=(1, 2, 4, 8),
                 use_conv=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_conv = use_conv

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop)

        self.conv = DilatedTemporalConv(hidden, dilations, drop) if use_conv else None

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          drop=drop, attn_drop=attn_drop, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden)

        self.cls_head = nn.Linear(hidden, num_classes)
        self.act_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feats):
        """feats: (B, T, D) -> cls_logits (B, T, C), act_logits (B, T)"""
        _, t, _ = feats.shape
        x = self.input_proj(feats)
        x = self.pos_drop(x + self.pos_embed[:, :t])
        if self.conv is not None:
            x = self.conv(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.cls_head(x), self.act_head(x).squeeze(-1)


class FishTALLoss(nn.Module):
    """Dense classification + point-localization, with a weak video-level term.

    - `cls`: class-weighted cross-entropy over every timestep, background
      included. This is the dense supervision ASM-Loc never gets.
    - `act`: BCE against Gaussian bumps centered on the exact BORIS timestamps --
      no proxy box, so no box-width ceiling baked into the target.
    - `mil`: top-k pooled class scores vs. the video-level label, i.e. the weak
      signal ASM-Loc trains on, kept as a regulariser so the model still has to
      explain the clip as a whole.
    """

    def __init__(self, class_weights=None, lamb_cls=1.0, lamb_act=1.0,
                 lamb_mil=0.5, topk_frac=0.125, focal_gamma=2.0):
        super().__init__()
        self.lamb_cls = lamb_cls
        self.lamb_act = lamb_act
        self.lamb_mil = lamb_mil
        self.topk_frac = topk_frac
        self.focal_gamma = focal_gamma
        self.register_buffer(
            "class_weights",
            torch.ones(1) if class_weights is None else torch.as_tensor(class_weights,
                                                                        dtype=torch.float),
        )

    def forward(self, cls_logits, act_logits, cls_target, act_target, vid_label):
        b, t, c = cls_logits.shape
        w = self.class_weights if self.class_weights.numel() == c else None
        cls_loss = F.cross_entropy(cls_logits.reshape(b * t, c),
                                   cls_target.reshape(b * t), weight=w)

        # Focal BCE: the Gaussian targets are overwhelmingly near-zero, so plain
        # BCE is dominated by easy background steps.
        p = torch.sigmoid(act_logits)
        bce = F.binary_cross_entropy_with_logits(act_logits, act_target, reduction="none")
        p_t = p * act_target + (1 - p) * (1 - act_target)
        act_loss = ((1 - p_t) ** self.focal_gamma * bce).mean()

        k = max(1, int(round(t * self.topk_frac)))
        topk = cls_logits.topk(k, dim=1).values.mean(dim=1)        # (B, C)
        mil_loss = F.binary_cross_entropy_with_logits(topk, vid_label)

        loss = self.lamb_cls * cls_loss + self.lamb_act * act_loss + self.lamb_mil * mil_loss
        return loss, {"cls_loss": cls_loss.item(), "act_loss": act_loss.item(),
                      "mil_loss": mil_loss.item()}


def build_fishtal(cfg):
    """cfg: plain dict/argparse namespace (no fvcore registry dependency)."""
    get = (lambda k, d: getattr(cfg, k, d)) if not isinstance(cfg, dict) else cfg.get
    return FishTAL(
        feat_dim=get("feat_dim", 768),
        num_classes=get("num_classes", 7),
        hidden=get("hidden", 256),
        depth=get("depth", 4),
        num_heads=get("num_heads", 8),
        drop=get("drop", 0.1),
        attn_drop=get("attn_drop", 0.1),
        drop_path=get("drop_path", 0.1),
        dilations=tuple(get("dilations", (1, 2, 4, 8))),
        use_conv=get("use_conv", True),
    )
