"""FishPoint -- pure point-localization model. No box regression anywhere.

FishTAL and FishFormer both still need *some* nominal width to train a
boundary-regression head against (FishFormer's `span_s`), even though the only
thing BORIS actually gives us is a timestamp -- that width is a hyperparameter
of the method, not a fact about the data (see FishFormer's own docstring in
train_former.py). This model removes the regression head entirely: it outputs
a per-timestep class score and an "actionness" (event-likelihood) score, and
optionally a small sub-stride time offset -- never a span. Detections are
POINTS (time, score), decoded by 1-D peak-picking (distance-based NMS, the
point analogue of IoU-based NMS over spans), and scored with point_ap.py's
confidence-ranked point-containment AP against a `--tolerance` window instead
of tIoU-vs-a-proxy-box mAP.

Two knobs, tried as separate variants (see train_point.py):
  n_levels   : 1 = the same single-scale trunk as FishTAL (no pyramid -- there
               is no duration to assign across scales anymore, so the pyramid's
               original justification doesn't automatically carry over; kept as
               an ablation to see whether the extra receptive field/context
               still helps confidence calibration even without duration
               specialization).
  use_offset : adds a small head regressing the sub-stride fractional offset
               from the nearest feature step to the true timestamp (CenterNet-
               style local offset), only supervised at the single closest step
               per event -- not a window, so it never becomes a proxy box.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import TemporalBlock, DilatedTemporalConv   # noqa: E402  (Trokens-derived)


class FishPointModel(nn.Module):
    def __init__(self, feat_dim=768, num_classes=7, hidden=256, depth=4,
                 num_heads=8, drop=0.1, attn_drop=0.1, drop_path=0.1,
                 n_levels=1, use_offset=False, max_len=4096,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        self.num_classes = num_classes
        self.n_levels = n_levels
        self.use_offset = use_offset

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(drop))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.conv = DilatedTemporalConv(hidden, dilations, drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.trunk = nn.ModuleList([
            TemporalBlock(hidden, num_heads=num_heads, drop=drop,
                          attn_drop=attn_drop, drop_path=dpr[i]) for i in range(depth)])
        self.level_blocks = nn.ModuleList([
            TemporalBlock(hidden, num_heads=num_heads, drop=drop,
                          attn_drop=attn_drop, drop_path=drop_path)
            for _ in range(max(0, n_levels - 1))])
        self.norm = nn.LayerNorm(hidden)

        def head(out_dim):
            return nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv1d(hidden, out_dim, 1))

        self.cls_head = head(num_classes)
        self.act_head = head(1)
        self.offset_head = head(1) if use_offset else None
        self.apply(self._init)
        # background-biased init, same reasoning as FishFormer: foreground is a
        # few percent of timesteps, uninitialized cls collapses to it otherwise.
        nn.init.constant_(self.cls_head[-1].bias, -4.0)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, feats):
        """feats (B,T,D) -> list per level of dicts with cls/act[/offset]/stride."""
        x = self.input_proj(feats)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.conv(x)
        for blk in self.trunk:
            x = blk(x)

        outs, cur, stride = [], x, 1
        for lvl in range(self.n_levels):
            if lvl > 0:
                cur = F.max_pool1d(cur.transpose(1, 2), 2, 2).transpose(1, 2)
                cur = self.level_blocks[lvl - 1](cur)
                stride *= 2
            h = self.norm(cur).transpose(1, 2)                          # (B,C,T)
            out = {
                "cls": self.cls_head(h).permute(0, 2, 1),               # (B,T,C)
                "act": self.act_head(h).squeeze(1),                     # (B,T) logits
                "stride": stride,
            }
            if self.use_offset:
                # bounded to +/-0.5 steps -- a *correction*, not a span; if the
                # true point is further than half a step from the nearest
                # feature step, that's what the next step's peak is for.
                out["offset"] = torch.tanh(self.offset_head(h).squeeze(1)) * 0.5
            outs.append(out)
        return outs


class FishPointLoss(nn.Module):
    """Class-weighted CE cls + focal BCE actionness (both dense, same pattern as
    FishTALLoss) + optional smooth-L1 offset loss at foreground steps only."""

    def __init__(self, num_classes=7, class_weights=None, lamb_cls=1.0,
                 lamb_act=1.0, lamb_off=1.0, focal_gamma=2.0):
        super().__init__()
        self.lamb_cls, self.lamb_act, self.lamb_off = lamb_cls, lamb_act, lamb_off
        self.focal_gamma = focal_gamma
        self.register_buffer(
            "class_weights",
            torch.ones(num_classes) if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float))

    def forward(self, outs, targets):
        """targets: per level, dict of cls (B,T) long, act (B,T) float,
        offset (B,T) float, offset_mask (B,T) bool."""
        total_cls = total_act = total_off = 0.0
        n_off_all = 0
        for out, tgt in zip(outs, targets):
            b, t, c = out["cls"].shape
            w = self.class_weights if self.class_weights.numel() == c else None
            total_cls = total_cls + F.cross_entropy(
                out["cls"].reshape(b * t, c), tgt["cls"].reshape(b * t), weight=w)

            p = torch.sigmoid(out["act"])
            bce = F.binary_cross_entropy_with_logits(out["act"], tgt["act"], reduction="none")
            p_t = p * tgt["act"] + (1 - p) * (1 - tgt["act"])
            total_act = total_act + ((1 - p_t) ** self.focal_gamma * bce).mean()

            if "offset" in out:
                m = tgt["offset_mask"]
                n_off = int(m.sum().item())
                n_off_all += n_off
                if n_off > 0:
                    total_off = total_off + F.smooth_l1_loss(out["offset"][m], tgt["offset"][m])

        n_lvl = max(1, len(outs))
        loss = (self.lamb_cls * total_cls / n_lvl + self.lamb_act * total_act / n_lvl
                + (self.lamb_off * total_off / n_lvl if n_off_all else 0.0))

        def _item(v):
            return float(v.detach()) if torch.is_tensor(v) else float(v)

        return loss, {"cls_loss": _item(total_cls / n_lvl), "act_loss": _item(total_act / n_lvl),
                      "off_loss": _item(total_off / n_lvl) if n_off_all else 0.0}
