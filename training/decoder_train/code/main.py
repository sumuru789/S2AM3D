import logging
import os
import shutil
import sys
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import models
from param import parse_args
from train import Trainer
from train_dataset import make as make_train_dataset
from utils.logger import setup_logging
from utils.misc import dump_config, load_config


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["NCCL_TIMEOUT"] = "7200"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=2)
    )


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, cli_args, extras):
    setup(rank, world_size)

    config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
    config.trial_name = config.get("trial_name") + datetime.now().strftime("@%Y%m%d-%H%M%S")
    config.ckpt_dir = config.get("ckpt_dir") or os.path.join(
        config.exp_dir, config.trial_name, "ckpt"
    )
    config.code_dir = config.get("code_dir") or os.path.join(
        config.exp_dir, config.trial_name, "code"
    )

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        os.makedirs(config.code_dir, exist_ok=True)
        for file in os.listdir("."):
            if file.endswith(".py") and not file.startswith("test_"):
                shutil.copy2(file, config.code_dir)
        for dir_name in ["models", "utils", "configs"]:
            if os.path.exists(dir_name):
                shutil.copytree(dir_name, os.path.join(config.code_dir, dir_name))

    config.device = "cuda:{0}".format(rank)

    if rank == 0:
        config.log_path = config.get("log_path") or os.path.join(
            config.exp_dir, config.trial_name, "log.txt"
        )
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, "config.yaml"), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))

    point_feature_enhancer = models.make_PointFeatureEnhancer(config).to(config.device)
    decoder = models.make_decoder(config).to(config.device)
    seg_head = models.make_seg_head(config).to(config.device)

    point_feature_enhancer = DDP(
        point_feature_enhancer,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )
    decoder = DDP(
        decoder, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )
    seg_head = DDP(
        seg_head, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )

    point_feature_enhancer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
        point_feature_enhancer
    )
    decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
    seg_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(seg_head)

    train_loader = make_train_dataset(config, rank, world_size)
    if rank == 0 and train_loader is not None:
        logging.info("Train iterations: {}".format(len(train_loader)))

    params = (
        list(point_feature_enhancer.parameters())
        + list(decoder.parameters())
        + list(seg_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.lr_decay_step,
        gamma=config.training.lr_decay_rate,
    )

    trainer = Trainer(
        rank,
        config,
        point_feature_enhancer,
        decoder,
        seg_head,
        optimizer,
        scheduler,
        train_loader,
        config.device,
    )
    if config.resume is not None:
        trainer.load_from_checkpoint(config.resume)
    trainer.train()
    cleanup()


if __name__ == "__main__":
    cli_args, extras = parse_args(sys.argv[1:])
    world_size = cli_args.ngpu
    mp.spawn(main, args=(world_size, cli_args, extras), nprocs=world_size)
