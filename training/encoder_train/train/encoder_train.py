#!/usr/bin/env python3
import os
import sys

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODER_ROOT = os.path.abspath(os.path.join(TRAIN_DIR, ".."))
for path in (ENCODER_ROOT, TRAIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import random
from datetime import datetime

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from encoder_train_model import EncoderTrainModel
from partfield.config import default_argument_parser, setup
from partfield.dataloader import PartNetDataset


def train(cfg):
    seed_everything(cfg.seed)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = cfg.result_name
    if not os.path.isabs(result_root):
        result_root = os.path.join(ENCODER_ROOT, result_root)
    save_dir = os.path.join(result_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 60)
    print("Encoder Train")
    print("=" * 60)
    print(f"Save dir: {save_dir}")

    def _resolve(p):
        if not p or not isinstance(p, str):
            return p
        if os.path.isabs(p):
            return p
        return os.path.join(ENCODER_ROOT, p)

    train_datasets = []
    train_paths = getattr(cfg.dataset, "train_metadata_paths", None) or []
    if not train_paths:
        train_paths = [os.path.join(ENCODER_ROOT, "meta", "train_meta.example.json")]

    num_pts = getattr(cfg, "dataset_num_points", 10000)
    batch_size = getattr(cfg, "batch_size", 4)

    print("\nConfig:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - dataset_num_points: {num_pts}")
    print(f"  - contrast_num_points: {getattr(cfg, 'contrast_num_points', 500)}")
    print(f"  - temperature: {getattr(cfg, 'temperature', 0.07)}")
    print(f"  - use_color: {getattr(cfg, 'use_color', False)}")
    print(f"  - learning_rate: {cfg.optimizer.lr}")

    print("\nTrain metadata:")
    for p in train_paths:
        p = _resolve(p)
        print(f"  - {p}")
        if os.path.isfile(p):
            train_datasets.append(
                PartNetDataset(metadata_path=p, num_points=num_pts, is_train=True)
            )
        else:
            print("    [missing] path not found")

    if not train_datasets:
        raise FileNotFoundError("No valid train metadata found")
    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else torch.utils.data.ConcatDataset(train_datasets)
    )
    print(f"Train samples: {len(train_dataset)}")

    num_workers = getattr(cfg.dataset, "train_num_workers", 4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    print(f"Steps per epoch: {len(train_loader)} (batch_size={batch_size})")
    print("=" * 60)

    model = EncoderTrainModel(cfg, save_dir=save_dir)

    checkpoint_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints"),
            filename="epoch{epoch:03d}-loss{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, "checkpoints_all"),
            filename="epoch{epoch:03d}",
            save_top_k=-1,
            every_n_epochs=1,
        ),
    ]

    trainer = Trainer(
        devices=-1,
        accelerator="gpu",
        precision="16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=cfg.training_epochs,
        log_every_n_steps=10,
        callbacks=checkpoint_callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloaders=train_loader)


def main():
    parser = default_argument_parser()
    args = parser.parse_args()

    if not args.config_file:
        args.config_file = os.path.join(TRAIN_DIR, "encoder_train.yaml")

    cfg = setup(args)

    if getattr(cfg, "continue_ckpt", None) and not os.path.isabs(cfg.continue_ckpt):
        cfg.defrost()
        cfg.continue_ckpt = os.path.join(ENCODER_ROOT, cfg.continue_ckpt)
        cfg.freeze()

    train(cfg)


if __name__ == "__main__":
    main()
