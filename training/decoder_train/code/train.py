import logging
import os

import torch
from tqdm import tqdm

import loss


class Trainer(object):
    def __init__(
        self,
        rank,
        config,
        PointFeatureEnhancer,
        decoder,
        seg_head,
        optimizer,
        scheduler,
        train_loader,
        device,
    ):
        self.rank = rank
        self.config = config
        self.pointfeatureenhancer = PointFeatureEnhancer
        self.decoder = decoder
        self.seg_head = seg_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.epoch = 0
        self.step = 0
        self.device = device
        self.loss = loss.AdaptiveLoss()
        self.enhancefeat_dim = config.enhancer.enhancefeat_dim

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.pointfeatureenhancer.load_state_dict(
            checkpoint["point_feature_enhancer_state_dict"], strict=False
        )
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.seg_head.load_state_dict(checkpoint["seg_head_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            logging.warning("Failed to load optimizer state: %s", e)
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as e:
            logging.warning("Failed to load scheduler state: %s", e)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        logging.info("Loaded checkpoint from %s (epoch=%s step=%s)", path, self.epoch, self.step)

    def save_model(self, name):
        torch.save(
            {
                "point_feature_enhancer_state_dict": self.pointfeatureenhancer.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "seg_head_state_dict": self.seg_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
            },
            os.path.join(self.config.ckpt_dir, "{}.pt".format(name)),
        )

    def get_scale_dropout_rate(self, epoch):
        return float(self.config.get("scale_dropout_rate", 0.1))

    def train_one_epoch(self):
        self.pointfeatureenhancer.train()
        self.decoder.train()
        self.seg_head.train()

        current_dropout_rate = self.get_scale_dropout_rate(self.epoch)
        if hasattr(self.pointfeatureenhancer.module, "scale_dropout_rate"):
            self.pointfeatureenhancer.module.scale_dropout_rate = current_dropout_rate

        data_iter = tqdm(
            self.train_loader,
            desc="Train epoch {}".format(self.epoch),
            disable=(self.rank != 0),
        )
        for data in data_iter:
            self.step += 1
            self.optimizer.zero_grad()

            point_feat = data["feat"].to(self.device)
            prompt_indices = data["prompt_indices"].to(self.device)
            point_coords = data["coord"].to(self.device)

            points_per_batch = self.config.dataset.num_points
            batch_size = self.config.dataset.train_batch_size
            point_feat = point_feat.view(batch_size, points_per_batch, -1)
            point_coords = point_coords.view(batch_size, points_per_batch, 3)

            continuous_scales = None
            if self.config.get("use_continuous_scale", True):
                continuous_scales = data.get("continuous_scales", None)
                if continuous_scales is not None:
                    continuous_scales = continuous_scales.to(self.device)

            enhance_feat = self.pointfeatureenhancer(
                point_feat, point_coords, None, continuous_scales
            )
            enhance_feat = enhance_feat.view(
                batch_size * points_per_batch, self.enhancefeat_dim
            )
            prompt_feat = enhance_feat.index_select(0, prompt_indices)
            prompt_feat = prompt_feat.view(batch_size, 1, self.enhancefeat_dim)

            enhance_feat = enhance_feat.view(batch_size, points_per_batch, self.enhancefeat_dim)
            decoder_output = self.decoder(enhance_feat, prompt_feat)
            seg_pred = self.seg_head(decoder_output)

            labels = data["label"].to(self.device).float()
            loss_val = self.loss(seg_pred, labels)
            loss_val.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.rank == 0:
                try:
                    data_iter.set_postfix({"loss": "{:.4f}".format(loss_val.item())})
                except Exception:
                    pass

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if self.rank == 0:
                self.save_model("latest")
                if self.epoch % self.config.training.save_freq == 0:
                    self.save_model("epoch_{}".format(self.epoch))
