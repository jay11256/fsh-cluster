#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Use a model on a dataset to get predictions."""
# pylint: disable=wrong-import-position,import-error,wrong-import-order
import os
import sys
sys.path = [x for x  in sys.path if not (os.path.isdir(x) and 'trokens' in os.listdir(x))]
sys.path.append(os.getcwd())
from dist_utils import init_distributed_mode
import trokens
assert trokens.__file__.startswith(os.getcwd()), (f"sys.path: {sys.path}, "
                                                  f"trokens.__file__: {trokens.__file__}")

from trokens.config.defaults import assert_and_infer_cfg
from trokens.utils.misc import launch_job
from trokens.utils.parser import load_config, parse_args

import pprint
import torch
import torch.nn.functional as F
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

def main():
    """
    Main function to spawn the prediction process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    if args.new_dist_init:
        args = init_distributed_mode(args)
    else:
        os.environ["MASTER_PORT"] = str(cfg.MASTER_PORT)
    if cfg.DEBUG:
        os.environ["WANDB_MODE"] = "offline"
        os.environ['DEBUG'] = 'True'
    else:
        os.environ['DEBUG'] = 'False'
    if '$SCRATCH_DIR' in  cfg.DATA.PATH_TO_DATA_DIR:
        cfg.DATA.PATH_TO_DATA_DIR = cfg.DATA.PATH_TO_DATA_DIR.replace(
                                        '$SCRATCH_DIR', os.environ['SCRATCH_DIR'])
    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES

    # Set up environment.
    if not args.new_dist_init:
        du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    # Build the video model
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    val_loader = loader.construct_loader(cfg, "test")

    get_preds(val_loader, model, cfg)

    # Exit
    sys.exit()

@torch.no_grad()
def get_preds(val_loader, model, cfg):
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
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    all_preds = []

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        input_dict = {'video':inputs, 'metadata':meta}

        preds, _ = model(input_dict) #get predictions
        
        if cfg.NUM_GPUS > 1:
            preds, labels = du.all_gather([preds, labels])

        preds = preds.cpu().numpy()
        all_preds.extend(preds.tolist())

    np.save(os.path.join(cfg.OUTPUT_DIR,'preds'), all_preds)

if __name__ == "__main__":
    main()