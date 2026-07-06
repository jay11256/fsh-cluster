#!/usr/bin/env python3

"""Test a few shot classification model."""
# pylint: disable=wrong-import-position,import-error,wrong-import-order
import os
import sys
import pprint
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from einops import rearrange
import trokens.utils.checkpoint as cu
import trokens.utils.distributed as du
import trokens.utils.logging as logging
import trokens.utils.metrics as metrics
import trokens.utils.misc as misc
from trokens.datasets import loader
from trokens.utils.meters import ValMeter
from trokens.models import build_model
from fvcore.common.config import CfgNode
from fvcore.nn.precise_bn import update_bn_stats

import trokens.models.losses as losses

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
# pylint: disable=line-too-long

def topks_correct_with_misclassified_video_names(preds, labels, ks, video_names):
    """
    Same counts as `topks_correct`, plus the subset of `video_names` for which
    the **top-1** predicted class does not match `labels`.

    Args:
        preds, labels, ks: same as `topks_correct`.
        video_names: one entry per row (same length as batch size), aligned with
            rows of `preds` / `labels`.

    Returns:
        topks_correct: list of per-k correct counts (same as `topks_correct`).
        misclassified_video_names: names for samples wrong at top-1 (empty if none).
    """
    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    n = preds.size(0)
    if len(video_names) != n:
        raise ValueError(
            f"video_names length ({len(video_names)}) must match batch size ({n})"
        )

    counts = metrics.topks_correct(preds, labels, ks)

    top1_pred = preds.argmax(dim=1)
    wrong = ~top1_pred.eq(labels)
    misclassified = [
        str(video_names[i])
        for i in range(n)
        if bool(wrong[i].item())
    ]
    #print(f"misclassified list from tpks modified: {misclassified}")
    return counts, misclassified

def video_names_from_meta(meta):
    """
    Extract one string name per batch row from collated `metadata` (as returned
    in the 4th slot of the dataset / loader).

    Handles collated batches where `meta['video_name']` is a list of strings,
    a length-1 batch str, or a tensor.
    """
    if "video_name" not in meta:
        raise KeyError("metadata missing key 'video_name'")

    names = meta["video_name"]
    if isinstance(names, str):
        return [names]
    if isinstance(names, (list, tuple)):
        return [str(x) for x in names]
    if isinstance(names, torch.Tensor):
        return [str(x) for x in names.detach().cpu().reshape(-1).tolist()]
    raise TypeError(f"Unsupported video_name type: {type(names)}")

def conv_fp16(var):
    """Convert to float16.
    """
    return np.float16(np.around(var, 4))

@torch.no_grad()
def test_epoch(val_loader, model, cur_epoch, cfg):
    model.eval()
    all_preds, all_labels, all_df = [], [], []

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])
        labels = labels.float()

        model_out = model({'video': inputs, 'metadata': meta})
        preds = model_out[0] if isinstance(model_out, tuple) else model_out

        if cfg['wandb']:
            cfg['wandb'].log({
                'iteration': cur_iter,
                'test_iter_f1': metrics.multilabel_f1(preds, labels).item(),
                'test_iter_hamming': metrics.multilabel_hamming_score(preds, labels).item(),
                'test_iter_exact_match': metrics.multilabel_exact_match(preds, labels).item(),
            })

        all_preds.append(preds)
        all_labels.append(labels)

        #vid_names = video_names_from_meta(meta)
        #print(preds[0]); print(preds[1]); print(preds[2])

        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        pred_bin = (1 / (1 + np.exp(-preds_np)) >= 0.5).astype(np.int64)
        all_df.append(pd.DataFrame({
            'y_true': [",".join(map(str, np.where(r > 0)[0])) for r in labels_np],
            'y_preds': [",".join(map(str, np.where(r > 0)[0])) for r in pred_bin],
        }))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    if cfg.NUM_GPUS > 1:
        all_preds, all_labels = du.all_gather([all_preds, all_labels])

    behavior_to_label = getattr(val_loader.dataset, "behavior_to_label", {})
    label_to_name = {v: k for k, v in behavior_to_label.items()}
    class_names = [label_to_name.get(i, str(i)) for i in range(all_labels.shape[1])]

    per_class = metrics.multilabel_per_class_stats(all_preds, all_labels, class_names=class_names)
    total_f1 = metrics.multilabel_f1(all_preds, all_labels).item()
    total_hamming = metrics.multilabel_hamming_score(all_preds, all_labels).item()
    total_exact = metrics.multilabel_exact_match(all_preds, all_labels).item()

    if cfg['wandb']:
        cfg['wandb'].log({
            'test_f1': total_f1,
            'test_hamming': total_hamming,
            'test_exact_match': total_exact,
            'epoch': cur_epoch,
        })

    pd.concat(all_df, ignore_index=True).to_csv(
        os.path.join(cfg.OUTPUT_DIR, cfg['csv_dump_name']), index=False)

    with open(os.path.join(cfg.OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"macro_f1: {total_f1:.4f}\n")
        f.write(f"hamming_score: {total_hamming:.4f}\n")
        f.write(f"exact_match: {total_exact:.4f}\n\n")
        f.write(metrics.format_multilabel_metrics_report(per_class))

# pylint: disable=redefined-outer-name
def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i, _ in enumerate(inputs):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

def test_few_shot(cfg, args, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
    """
    # Set up environment.
    if not args.new_dist_init:
        du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if wandb_run is not None:
        wandb_instance = wandb_run
        wandb_instance.define_metric("test*", step_metric="epoch")
        wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
    else:
        if du.get_rank() == 0:
            wandb_config_dict = wandb_init_dict(cfg)
            wandb_instance = wandb.init(project=cfg.WANDB.PROJECT,config=wandb_config_dict,
                                        entity=cfg.WANDB.ENTITY)
            wandb_instance.define_metric("epoch")
            wandb_instance.define_metric("iteration")

            wandb_instance.define_metric("iter*", step_metric="iteration")

            wandb_instance.define_metric("train*", step_metric="epoch")
            wandb_instance.define_metric("val*", step_metric="epoch")
            wandb_instance.define_metric("test*", step_metric="epoch")

            wandb_instance.define_metric("test_f1", summary="max")
            wandb_instance.define_metric("test_hamming", summary="max")
            wandb_instance.define_metric("test_exact_match", summary="max")
        else:
            wandb_instance = None
    cfg['wandb'] = wandb_instance
    cfg['csv_dump_name'] = 'preds_dump.csv'

    logger = logging.get_logger(__name__)
    
    # Init multigrid.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    cur_epoch = cu.load_test_checkpoint(cfg, model)
    val_loader = loader.construct_loader(cfg, "test") # MOLO uses test set for validation
    # val_meter = ValMeter(len(val_loader), cfg)

    test_epoch(val_loader, model, cur_epoch, cfg)
    # Close wandb logging
    if wandb_instance is not None:
        wandb_instance.finish()

    # Exit
    sys.exit()