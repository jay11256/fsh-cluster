"""FishFormer -- single-scale anchor-free localizer with boundary regression.

Predicts span boundaries directly rather than deriving them by thresholding an
actionness curve, so boundary quality is optimised rather than being a side
effect of where a curve crosses a threshold.

**Boundaries in our data are genuinely uncertain.** BORIS gives a timestamp, not
a start/end, so the "true" extent is unknown even to the annotator. Predicting a
*distribution* over each offset (TriDet-style) rather than a point estimate is
therefore not just a refinement -- it matches what the annotation actually
supports. `--reg-bins 0` falls back to plain scalar regression for ablation.

WHY THERE IS NO FEATURE PYRAMID. Earlier versions followed
ActionFormer/TriDet in stacking a 4-level pyramid (strides 1/2/4/8), on the
usual reasoning that behaviors of different duration should be owned by
different scales. That reasoning does not survive contact with this dataset.
Supervision is built from BORIS *points* expanded to a fixed `span_s` box, so
every ground-truth segment has the same width, and each level only accepts
segments whose max(left, right) distance falls in its own range. With
span_s=4.0 at a 0.25s stride the accept ranges are [0,4] / [8,16] / [32,64] /
[128,inf] seconds while a 4s box always yields max(left, right) in [2,4]s --
so levels 1-3 receive *exactly zero* positive assignments. Measured on 64 real
training windows: level 0 got 7,323 positives, levels 1/2/3 got 0/0/0.

Those three levels were 5.3M parameters (25% of the model) trained only to emit
background, and their untrained predictions still had to be suppressed by NMS
at inference. Removing them is a strict simplification: no positive was ever
assigned to them, so nothing supervised is lost. Note this is a consequence of
fixed-width point-derived supervision -- a pyramid would be justified again if
the annotations carried real, varying durations.

Head layout, per timestep t (stride 1, one scale):
    cls   (C,)  class logits, background included -- dense supervision
    reg   (2,)  distance to (start, end), either scalar or expectation over a
                softmax'd distance distribution
    ctr   (1,)  centerness: down-weights timesteps far from an event center, so
                sloppy off-center predictions do not dominate NMS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import TemporalBlock, DilatedTemporalConv


class CrossMotionLite(nn.Module):
    """Learned local temporal-difference operator -- a small depthwise conv
    that lets the model discover its own motion signal, trained end-to-end
    against the localization loss.

    Structurally in the same spot Trokens' own cross_motion_module/
    hod_motion_module occupy (a dedicated motion block before the main
    temporal trunk), but trained for OUR objective instead of inherited from
    Trokens' few-shot classification objective, which has no reason to
    preserve within-clip timing. Also unlike the earlier hand-crafted
    `feats[t] - feats[t-1]` diff (rejected -- see EXPERIMENTS.md "Motion-delta
    feature", a 20x regression on a matched smoke test): that diff subtracted
    an already spatially-pooled, slowly-varying vector and was mostly noise.
    A learned depthwise conv over several neighbouring *raw* per-frame steps
    can weight which offsets and channels actually carry motion information,
    rather than assuming a literal one-step subtraction is meaningful.
    """
    def __init__(self, dim, kernel_size=5, drop=0.1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):                       # (B,T,C)
        h = x.transpose(1, 2)                    # (B,C,T)
        h = self.drop(self.proj(F.gelu(self.conv(h)))).transpose(1, 2)
        return self.norm(x + h)                  # residual: add motion cues on top of appearance


class FishFormer(nn.Module):
    def __init__(self, feat_dim=768, num_classes=7, hidden=256, depth=4,
                 num_heads=8, drop=0.1, attn_drop=0.1, drop_path=0.1,
                 reg_bins=16, reg_max=64.0, max_len=4096,
                 dilations=(1, 2, 4, 8), spatial_pool=False, use_motion=False,
                 mlp_ratio=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.reg_bins = reg_bins
        self.reg_max = reg_max
        self.spatial_pool = spatial_pool
        self.use_motion = use_motion

        # When feats come in as (B,T,P,D) -- a coarse DINO patch grid per frame,
        # P regions instead of one pre-pooled vector -- learn WHERE to look
        # instead of averaging every region uniformly (dump_feats_patchx_spatial.py
        # keeps a 4x4=16-region grid rather than mean-pooling all 256 DINO patch
        # tokens to a single vector). A single-head attention score per region,
        # trained jointly with the localization loss, replaces that fixed mean.
        if spatial_pool:
            self.spatial_score = nn.Linear(feat_dim, 1)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden), nn.GELU(), nn.Dropout(drop))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # same spot Trokens' own motion modules occupy: right after the input
        # projection, before the main temporal trunk gets to reason over it.
        self.motion = CrossMotionLite(hidden, drop=drop) if use_motion else None
        self.conv = DilatedTemporalConv(hidden, dilations, drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.trunk = nn.ModuleList([
            TemporalBlock(hidden, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          drop=drop, attn_drop=attn_drop,
                          drop_path=dpr[i]) for i in range(depth)])

        self.norm = nn.LayerNorm(hidden)

        def head(out_dim):
            return nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv1d(hidden, out_dim, 1))

        self.cls_head = head(num_classes)
        self.reg_head = head(2 * reg_bins if reg_bins > 0 else 2)
        self.ctr_head = head(1)
        # Learned scale on the regression output, letting the head calibrate the
        # bin expectation to the distance range actually observed. Retained from
        # the pyramid version, where it was one scale per level.
        self.scale = nn.Parameter(torch.ones(()))
        self.apply(self._init)
        # Bias cls toward background at init -- foreground is a few percent of
        # timesteps and without this the first epochs are pure background collapse.
        nn.init.constant_(self.cls_head[-1].bias, -4.0)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _decode_reg(self, raw):
        """(B, out, T) -> (B, T, 2) distances, in timesteps."""
        b, _, t = raw.shape
        scale = F.softplus(self.scale)
        if self.reg_bins <= 0:
            return F.relu(raw.permute(0, 2, 1) * scale)
        # Expectation over a softmax'd distance distribution (TriDet-style):
        # boundaries our annotation never pinned down are better modelled as a
        # spread than as one number.
        logits = raw.view(b, 2, self.reg_bins, t).permute(0, 3, 1, 2)   # (B,T,2,bins)
        probs = logits.softmax(dim=-1)
        centers = torch.linspace(0, self.reg_max, self.reg_bins, device=raw.device)
        return (probs * centers).sum(-1) * scale                        # (B,T,2)

    def forward(self, feats):
        """feats (B,T,D), or (B,T,P,D) if spatial_pool -> a single-element list
        of dicts with cls/reg/ctr and stride.

        The list is kept (rather than returning the dict bare) so the loss,
        target assignment and decode paths stay written against a sequence of
        output levels; there is now exactly one, at stride 1.
        """
        if self.spatial_pool:
            assert feats.dim() == 4, f"spatial_pool=True expects (B,T,P,D), got {feats.shape}"
            w = self.spatial_score(feats).softmax(dim=2)   # (B,T,P,1)
            feats = (feats * w).sum(dim=2)                  # (B,T,D)
        x = self.input_proj(feats)
        x = x + self.pos_embed[:, :x.shape[1]]
        if self.motion is not None:
            x = self.motion(x)
        x = self.conv(x)
        for blk in self.trunk:
            x = blk(x)

        h = self.norm(x).transpose(1, 2)                                # (B,C,T)
        return [{
            "cls": self.cls_head(h).permute(0, 2, 1),                   # (B,T,C)
            "reg": self._decode_reg(self.reg_head(h)),                  # (B,T,2)
            "ctr": self.ctr_head(h).squeeze(1),                         # (B,T)
            "stride": 1,
        }]


class FishFormerLoss(nn.Module):
    """Focal cls + DIoU-style temporal regression + centerness BCE.

    Assignment is anchor-free: a timestep is positive for the event whose
    supervision window contains it, and only levels whose regression range can
    represent that duration take responsibility for it.
    """

    def __init__(self, num_classes=7, bg_index=6, class_weights=None,
                 lamb_cls=1.0, lamb_reg=1.0, lamb_ctr=0.5, focal_gamma=2.0,
                 focal_alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.bg_index = bg_index
        self.lamb_cls, self.lamb_reg, self.lamb_ctr = lamb_cls, lamb_reg, lamb_ctr
        self.gamma, self.alpha = focal_gamma, focal_alpha
        self.register_buffer(
            "class_weights",
            torch.ones(num_classes) if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float))

    @staticmethod
    def _tiou_loss(pred, target, eps=1e-7):
        """1 - IoU on 1-D (left, right) distance pairs, plus a centre penalty."""
        pl, pr = pred[..., 0], pred[..., 1]
        tl, tr = target[..., 0], target[..., 1]
        inter = torch.min(pl, tl) + torch.min(pr, tr)
        union = (pl + pr) + (tl + tr) - inter
        iou = inter.clamp(min=0) / union.clamp(min=eps)
        enclose = torch.max(pl, tl) + torch.max(pr, tr)
        centre = ((pr - pl) - (tr - tl)).abs() / 2.0
        return (1 - iou + centre / enclose.clamp(min=eps)).mean()

    def forward(self, outs, targets):
        """targets: per level, dict of cls (B,T,C) multi-hot float, reg (B,T,2),
        ctr (B,T), pos (B,T) bool.

        `cls` arrives multi-hot from assign_targets -- a timestep covered by two
        behaviors is positive for both -- which is what the per-class sigmoid
        focal below expects. Background is a channel like any other, hot only
        where nothing covers.
        """
        total_cls = total_reg = total_ctr = 0.0
        n_pos_all = 0
        for out, tgt in zip(outs, targets):
            cls_logits, reg, ctr = out["cls"], out["reg"], out["ctr"]
            cls_t, reg_t, ctr_t, pos = tgt["cls"], tgt["reg"], tgt["ctr"], tgt["pos"]

            tgt_c = cls_t.float()
            p = torch.sigmoid(cls_logits)
            ce = F.binary_cross_entropy_with_logits(cls_logits, tgt_c, reduction="none")
            p_t = p * tgt_c + (1 - p) * (1 - tgt_c)
            a_t = self.alpha * tgt_c + (1 - self.alpha) * (1 - tgt_c)
            focal = a_t * (1 - p_t) ** self.gamma * ce
            total_cls = total_cls + (focal * self.class_weights).sum() / max(1, pos.sum().item())

            n_pos = int(pos.sum().item())
            n_pos_all += n_pos
            if n_pos > 0:
                total_reg = total_reg + self._tiou_loss(reg[pos], reg_t[pos])
                total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                    ctr[pos], ctr_t[pos])

        n_lvl = max(1, len(outs))
        loss = (self.lamb_cls * total_cls / n_lvl
                + self.lamb_reg * (total_reg / n_lvl if n_pos_all else 0.0)
                + self.lamb_ctr * (total_ctr / n_lvl if n_pos_all else 0.0))
        def _item(v):
            return float(v.detach()) if torch.is_tensor(v) else float(v)

        return loss, {
            "cls_loss": _item(total_cls / n_lvl),
            "reg_loss": _item(total_reg / n_lvl) if n_pos_all else 0.0,
            "ctr_loss": _item(total_ctr / n_lvl) if n_pos_all else 0.0,
            "n_pos": n_pos_all,
        }
