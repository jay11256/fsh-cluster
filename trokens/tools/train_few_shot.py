#!/usr/bin/env python3

"""Train a multilabel few-shot video classification model."""
import os
import pprint
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fvcore.common.config import CfgNode
import wandb
import trokens.models.optimizer as optim
import trokens.utils.checkpoint as cu
import trokens.utils.distributed as du
import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc
from trokens.datasets import loader
from trokens.datasets.mixup import MixUp
from trokens.models import build_model
from trokens.utils.meters import EpochTimer, TrainMeter, ValMeter
from trokens.utils.multigrid import MultigridSchedule

warnings.filterwarnings('ignore')


def wandb_init_dict(cfg_node):
    """Convert a config node to dictionary.
    """
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = wandb_init_dict(v)
        return cfg_dict

logger = logging.get_logger(__name__)


# pylint: disable=line-too-long
def process_patch_tokens(cfg, support_tokens, query_tokens):
    """
    Process the patch tokens for few shot learning.
    Ref: https://github.com/alibaba-mmai-research/MoLo/blob/f7f73b6dd8cba446b414b1c47652ab26033bc88e/models/base/few_shot.py#L2552
    args:
        cfg: config
        support_tokens: (num_support, temp_len, num_patches, embed_dim)
        query_tokens: (num_query, temp_len, num_patches, embed_dim)
    """
    support_tokens = F.relu(support_tokens)
    query_tokens = F.relu(query_tokens)

    num_supports = support_tokens.shape[0]
    num_querries = query_tokens.shape[0]
    if not cfg.MODEL.USE_EXTRA_ENCODER:
        if cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'temporal':
            support_tokens = support_tokens.mean(dim=1)
            query_tokens = query_tokens.mean(dim=1)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'spatial':
            support_tokens = support_tokens.mean(dim=2)
            query_tokens = query_tokens.mean(dim=2)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'no_agg':
            support_tokens = rearrange(support_tokens, 'b t p e -> b (t p) e')
            query_tokens = rearrange(query_tokens, 'b t p e -> b (t p) e')
        else:
            raise NotImplementedError(
                f"Aggregation method {cfg.FEW_SHOT.PATCH_TOKENS_AGG} not implemented")

    support_tokens = rearrange(support_tokens, 'b p e -> (b p) e')
    query_tokens = rearrange(query_tokens, 'b p e -> (b p) e')
    sim_matrix = cos_sim(query_tokens, support_tokens)
    dist_matrix = 1 - sim_matrix

    dist_rearranged = rearrange(dist_matrix, '(q qt) (s st) -> q s qt st',
                                q=num_querries, s=num_supports)
    # Take the minimum distance for each query token
    dist_logits = dist_rearranged.min(3)[0].sum(2) + dist_rearranged.min(2)[0].sum(2)
    if cfg.FEW_SHOT.DIST_NORM == 'max_div':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = dist_logits / max_dist
    elif cfg.FEW_SHOT.DIST_NORM == 'max_sub':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = max_dist - dist_logits

    return - dist_logits


def cos_sim(x, y, epsilon=0.01):
    """Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1, -2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1, -2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def support_query_split(preds, labels, metadata):
    """
    Split preds/labels into support and query, then build one prototype per
    episode class.

    For multilabel supports, every support whose multi-hot is positive for an
    episode class contributes its embedding to that class prototype (not only
    the supports that were sampled under that class's slot).

    Args:
        preds (torch.Tensor): patch tokens / embeddings for the batch
        labels (torch.Tensor): multi-hot labels [B, C] (or scalar labels)
        metadata (dict): must contain sample_type, batch_label, episode_classes
    """
    device = preds.device
    sample_info = np.array(metadata['sample_type'])
    support_condition = sample_info == 'support'
    support_preds = preds[support_condition]
    support_labels = labels[support_condition]
    query_preds = preds[~support_condition]
    query_labels = labels[~support_condition]

    # All samples in an episode share the same episode_classes row.
    episode_classes = metadata['episode_classes']
    if episode_classes.dim() == 2:
        episode_classes = episode_classes[0]
    episode_classes = episode_classes.to(device=device, dtype=torch.long)

    support_to_take = []
    for global_cls in episode_classes.tolist():
        if support_labels.dim() == 1:
            # Single-label fallback: match scalar labels to global class id
            pos_mask = support_labels == global_cls
        else:
            pos_mask = support_labels[:, global_cls] > 0

        if not bool(pos_mask.any()):
            raise RuntimeError(
                f"No support examples positive for episode class {global_cls}"
            )
        support_to_take.append(
            support_preds[pos_mask].mean(dim=0, keepdim=True)
        )

    support_preds = torch.cat(support_to_take, dim=0)
    support_batch_labels = torch.arange(
        len(episode_classes), device=device, dtype=torch.long
    )

    if query_labels.dim() == 1:
        # Map global class ids -> episode slots for single-label CE.
        class_to_slot = {
            int(c): slot for slot, c in enumerate(episode_classes.tolist())
        }
        query_batch_labels = torch.tensor(
            [class_to_slot[int(c)] for c in query_labels.tolist()],
            device=device,
            dtype=torch.long,
        )
        query_episode_targets = None
    else:
        query_episode_targets = query_labels[:, episode_classes].float()
        # Keep a primary slot for logging / optional CE fallback.
        query_batch_labels = metadata['batch_label'][~support_condition]
        if not torch.is_tensor(query_batch_labels):
            query_batch_labels = torch.as_tensor(
                query_batch_labels, device=device, dtype=torch.long
            )
        else:
            query_batch_labels = query_batch_labels.to(device=device, dtype=torch.long)

    return {
        'query_labels': query_labels,
        'query_batch_labels': query_batch_labels,
        'query_episode_targets': query_episode_targets,
        'support_labels': support_labels,
        'support_batch_labels': support_batch_labels,
        'support_preds': support_preds,
        'query_preds': query_preds,
        'episode_classes': episode_classes,
    }


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    wandb_run=None
):
    """
    Perform the video training for one epoch.
    """
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    epoch_cls_loss = []
    epoch_q2s_loss = []
    epoch_f1 = []
    epoch_q2s_f1 = []

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=misc.get_num_classes(cfg)
        )

    cls_loss_fun = nn.BCEWithLogitsLoss(reduction="mean")
    q2s_loss_fun = nn.BCEWithLogitsLoss(reduction="mean")
    lr = optim.get_epoch_lr(cur_epoch, cfg)
    optim.set_lr(optimizer, lr, log=True)

    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(train_loader):
        if cur_iter > len(train_loader):
            break
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])
        labels = labels.float()

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples

        with torch.amp.autocast('cuda', enabled=cfg.TRAIN.MIXED_PRECISION):
            input_dict = {'video': inputs, 'metadata': meta}
            model_out = model(input_dict)
            if isinstance(model_out, tuple):
                preds, patch_tokens = model_out
            else:
                preds = model_out
                patch_tokens = None

            if isinstance(preds, tuple):
                preds, _ = preds

            preds = preds / cfg.SOLVER.TEMPRATURE
            classfication_loss = cls_loss_fun(preds, labels)
            loss_dict = {'classfication_loss': classfication_loss}

            if patch_tokens is None:
                raise RuntimeError(
                    "Few-shot training requires patch tokens from the model. "
                    "Ensure cfg.TASK == 'few_shot'."
                )

            patch_support_query_dict = support_query_split(
                patch_tokens, labels, meta
            )
            patch_q2s_logits = process_patch_tokens(
                cfg,
                patch_support_query_dict['support_preds'],
                patch_support_query_dict['query_preds'],
            )
            patch_q2s_logits = patch_q2s_logits / cfg.SOLVER.TEMPRATURE
            q2s_targets = patch_support_query_dict['query_episode_targets']
            q2s_loss = q2s_loss_fun(patch_q2s_logits, q2s_targets)
            loss_dict['q2s_loss'] = q2s_loss

        loss = (
            cfg.FEW_SHOT.CLASS_LOSS_LAMBDA * classfication_loss
            + cfg.FEW_SHOT.Q2S_LOSS_LAMBDA * q2s_loss
        )

        misc.check_nan_losses(loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )

        scaler.step(optimizer)
        scaler.update()

        classification_loss = loss_dict['classfication_loss']
        q2s_loss_val = loss_dict['q2s_loss']
        iter_f1 = metrics.multilabel_f1(preds, labels)
        iter_q2s_f1 = metrics.multilabel_f1(patch_q2s_logits, q2s_targets)

        if cfg.NUM_GPUS > 1:
            classification_loss, q2s_loss_val, iter_f1, iter_q2s_f1 = du.all_reduce(
                [classification_loss, q2s_loss_val, iter_f1, iter_q2s_f1]
            )

        classification_loss = classification_loss.item()
        q2s_loss_val = q2s_loss_val.item()
        iter_f1_val = iter_f1.item()
        iter_q2s_f1_val = iter_q2s_f1.item()

        epoch_cls_loss.append(classification_loss)
        epoch_q2s_loss.append(q2s_loss_val)
        epoch_f1.append(iter_f1_val)
        epoch_q2s_f1.append(iter_q2s_f1_val)

        global_iter = data_size * cur_epoch + cur_iter
        wandb_iter_dict = {
            'iter_cls_loss': classification_loss,
            'iter_q2s_loss': q2s_loss_val,
            'iter_f1': iter_f1_val,
            'iter_q2s_f1': iter_q2s_f1_val,
            'iteration': global_iter,
        }
        if wandb_run:
            wandb_run.log(wandb_iter_dict)

        train_meter.update_stats(
            None,
            None,
            loss_dict,
            lr,
            inputs[0].size(0) * max(cfg.NUM_GPUS, 1),
        )
        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    wandb_epoch_dict = {
        'train_cls_loss': np.mean(epoch_cls_loss),
        'train_q2s_loss': np.mean(epoch_q2s_loss),
        'train_f1': np.mean(epoch_f1),
        'train_q2s_f1': np.mean(epoch_q2s_f1),
        'epoch': cur_epoch,
    }
    if wandb_run:
        wandb_run.log(wandb_epoch_dict)


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_run=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
        wandb_run (wandb.run): wandb run object
    """

    model.eval()
    all_preds = []
    all_labels = []

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])
        labels = labels.float()

        input_dict = {'video': inputs, 'metadata': meta}
        model_out = model(input_dict)
        if isinstance(model_out, tuple):
            preds, _ = model_out
        else:
            preds = model_out

        iter_f1 = metrics.multilabel_f1(preds, labels)
        iter_dict = {
            'iteration': cur_iter,
            'eval_iter_f1': iter_f1.item(),
        }
        if wandb_run:
            wandb_run.log(iter_dict)

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if cfg.NUM_GPUS > 1:
        all_preds, all_labels = du.all_gather([all_preds, all_labels])

    total_f1 = metrics.multilabel_f1(all_preds, all_labels).item()
    log_dict = {
        'eval_f1': total_f1,
        'epoch': cur_epoch,
    }
    if wandb_run:
        wandb_run.log(log_dict)

    return total_f1


def train_few_shot(cfg, args, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    """
    if not args.new_dist_init:
        du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    logging.setup_logging(cfg.OUTPUT_DIR)

    if du.get_rank() == 0:
        wandb_config_dict = wandb_init_dict(cfg)
        wandb_config_dict['slurm_id'] = os.environ.get('SLURM_JOB_ID')
        wandb_run = wandb.init(
            project=cfg.WANDB.PROJECT,
            config=wandb_config_dict,
            entity=cfg.WANDB.ENTITY,
            name=cfg.WANDB.EXP_NAME,
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("iteration")
        wandb_run.define_metric("iter*", step_metric="iteration")
        wandb_run.define_metric("train*", step_metric="epoch")
        wandb_run.define_metric("val*", step_metric="epoch")
        wandb_run.define_metric("train_cls_loss", summary="min")
        wandb_run.define_metric("train_q2s_loss", summary="min")
        wandb_run.define_metric("train_f1", summary="max")
        wandb_run.define_metric("val_q2s_f1", summary="max")
    else:
        wandb_run = None

    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)

    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    model = build_model(cfg)
    if cfg.NUM_GPUS > 1:
        cfg['num_patches'] = model.module.num_patches
    else:
        cfg['num_patches'] = model.num_patches

    optimizer = optim.construct_optimizer(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.TRAIN.MIXED_PRECISION)

    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "test", less_iters=True)
    if du.is_master_proc():
        model_info = misc.log_model_info(model, cfg, train_loader)
        wandb_run.summary.update(model_info)

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    logger.info("Start epoch: %s", start_epoch + 1)

    epoch_timer = EpochTimer()
    best_val_f1 = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        epoch_timer.epoch_tic()
        if not cfg.TRAIN.VAL_ONLY:
            train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                wandb_run
            )
        epoch_timer.epoch_toc()
        logger.info(
            "Epoch %s takes %.2fs. Epochs "
            "from %s to %s take "
            "%.2fs in average and "
            "%.2fs in median.",
            cur_epoch, epoch_timer.last_epoch_time(),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time(),
            epoch_timer.median_epoch_time()
        )
        logger.info(
            "For epoch %s, each iteraction takes "
            "%.2fs in average. "
            "From epoch %s to %s, each iteraction takes "
            "%.2fs in average.",
            cur_epoch, epoch_timer.last_epoch_time() / len(train_loader),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time() / len(train_loader)
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        cfg_to_save = cfg.clone()
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg_to_save,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        if is_eval_epoch:
            val_f1 = eval_epoch(
                val_loader, model, val_meter, cur_epoch, cfg, wandb_run
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                cu.save_checkpoint(
                    cfg.OUTPUT_DIR,
                    model,
                    optimizer,
                    cur_epoch,
                    cfg_to_save,
                    scaler if cfg.TRAIN.MIXED_PRECISION else None,
                    best=True
                )
        if cfg.TRAIN.VAL_ONLY:
            break

    return wandb_run
