#!/usr/bin/env python3

"""Train a multi-label video classification model."""
import os
import pprint
import warnings
import numpy as np
import torch
import torch.nn as nn
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
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        wandb_run (wandb.run): wandb run object
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    epoch_loss = []
    epoch_f1 = []

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=misc.get_num_classes(cfg)
        )
    loss_fun = nn.BCEWithLogitsLoss(reduction="mean")
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
                preds, _ = model_out
            else:
                preds = model_out

            preds = preds / cfg.SOLVER.TEMPRATURE
            loss = loss_fun(preds, labels)
            loss_dict = {'loss': loss}

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

        iter_f1 = metrics.multilabel_f1(preds, labels)
        if cfg.NUM_GPUS > 1:
            loss, iter_f1 = du.all_reduce([loss, iter_f1])

        loss_val = loss.item()
        iter_f1_val = iter_f1.item()
        epoch_loss.append(loss_val)
        epoch_f1.append(iter_f1_val)

        global_iter = data_size * cur_epoch + cur_iter
        wandb_iter_dict = {
            'iter_loss': loss_val,
            'iter_f1': iter_f1_val,
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

    wandb_iter_dict = {
        'train_loss': np.mean(epoch_loss),
        'train_f1': np.mean(epoch_f1),
        'epoch': cur_epoch,
    }
    if wandb_run:
        wandb_run.log(wandb_iter_dict)


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
    Args:
        cfg (CfgNode): configs. Details can be found in
            trokens/config/defaults.py
        args (argparse.Namespace): arguments
        wandb_run (wandb.run): wandb run object
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

    if du.get_rank() == 0:
        wandb_config_dict = wandb_init_dict(cfg)
        wandb_config_dict['slurm_id'] = os.environ.get('SLURM_JOB_ID')
        wandb_run = wandb.init(project=cfg.WANDB.PROJECT,config=wandb_config_dict,
                                    entity=cfg.WANDB.ENTITY, name=cfg.WANDB.EXP_NAME)
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("iteration")

        wandb_run.define_metric("iter*", step_metric="iteration")

        wandb_run.define_metric("train*", step_metric="epoch")
        wandb_run.define_metric("val*", step_metric="epoch")
        wandb_run.define_metric("train_loss", summary="min")
        wandb_run.define_metric("train_f1", summary="max")
        wandb_run.define_metric("eval_f1", step_metric="epoch", summary="max")
    else:
        wandb_run = None


    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.NUM_GPUS>1:
        cfg['num_patches'] = model.module.num_patches
    else:
        cfg['num_patches'] = model.num_patches

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
         cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
     )

    #start_epoch = 0
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    # MOLO uses test set for validation
    val_loader = loader.construct_loader(cfg, "test", less_iters=True)
    if du.is_master_proc():
        model_info = misc.log_model_info(model, cfg, train_loader)
        # log in wandb as a summary
        wandb_run.summary.update(model_info)

    # Create meters.


    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)



    # Perform the training loop.
    logger.info("Start epoch: %s", start_epoch + 1)

    epoch_timer = EpochTimer()
    best_val_f1 = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Train for one epoch.
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
            cur_epoch, epoch_timer.last_epoch_time()/len(train_loader),
            start_epoch, cur_epoch,
            epoch_timer.avg_epoch_time()/len(train_loader)
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )


        # Save a checkpoint.
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
        # Evaluate the model on validation set.1
        if is_eval_epoch:
            val_f1 = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_run)
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
