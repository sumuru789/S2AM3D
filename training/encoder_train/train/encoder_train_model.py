import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from partfield.model.PVCNN.encoder_pc import sample_triplane_feat
from partfield.model_trainer_pvcnn_only_demo import Model


class LightweightFeatureNet(nn.Module):
    def __init__(self, use_color=True):
        super().__init__()
        self.use_color = use_color
        if use_color:
            self.mlp = nn.Sequential(
                nn.Linear(6, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, xyz, color=None):
        if self.use_color and color is not None:
            xyz = torch.clamp(xyz, -1.0, 1.0)
            color = torch.clamp(color, -1.0, 1.0)
            features = torch.cat([xyz, color], dim=-1)
            output = self.mlp(features)
            return torch.clamp(output, -1.0, 1.0)
        return xyz


def single_object_contrastive_loss(features, labels, num_sample_points=500, temperature=0.07):
    N = features.shape[0]
    device = features.device

    if N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if N > num_sample_points:
        indices = torch.randperm(N, device=device)[:num_sample_points]
        features = features[indices]
        labels = labels[indices]
        N = num_sample_points

    labels = labels.view(-1)
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.t()) / temperature
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_self = ~torch.eye(N, dtype=torch.bool, device=device)
    pos_mask_matrix = label_matrix & mask_self

    neg_inf_diag = torch.diag(torch.full((N,), float("-inf"), device=device))
    masked_sim = similarity_matrix + neg_inf_diag
    log_sum_exp_all = torch.logsumexp(masked_sim, dim=1)

    pos_sim_masked = similarity_matrix.clone()
    pos_sim_masked[~pos_mask_matrix] = float("-inf")
    log_sum_exp_pos = torch.logsumexp(pos_sim_masked, dim=1)

    has_pos = pos_mask_matrix.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_per_anchor = log_sum_exp_all - log_sum_exp_pos
    return loss_per_anchor[has_pos].mean()


class EncoderTrainModel(pl.LightningModule):
    def __init__(self, cfg, save_dir=None):
        super().__init__()
        self.cfg = cfg
        self.model = Model(cfg)
        self.save_dir = save_dir
        self.contrast_num_points = getattr(cfg, "contrast_num_points", 500)
        self.temperature = getattr(cfg, "temperature", 0.07)
        self.use_color = getattr(cfg, "use_color", False)
        self.feature_net = LightweightFeatureNet(use_color=self.use_color)

        if hasattr(cfg, "continue_ckpt") and cfg.continue_ckpt:
            print(f"Loading checkpoint: {cfg.continue_ckpt}")
            checkpoint = torch.load(cfg.continue_ckpt, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            if not any(k.startswith("model.") for k in state_dict.keys()):
                new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict

            current_state_dict = self.state_dict()
            filtered_state_dict = {}
            for k, v in new_state_dict.items():
                if k in current_state_dict and current_state_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                elif k in current_state_dict:
                    print(f"Shape mismatch, skip: {k}")
            self.load_state_dict(filtered_state_dict, strict=False)
            print("Checkpoint loaded.")
        else:
            print(
                "No continue_ckpt; training from scratch. "
                "Pass --opts continue_ckpt /path/to/ckpt "
                "(see https://github.com/nv-tlabs/PartField)."
            )

        if getattr(cfg, "freeze_backbone", False):
            for param in self.model.pvcnn.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        pc = batch["pc"]
        labels = batch["label"]
        color = batch.get("color", None)
        B, N, _ = pc.shape

        pc_flat = pc.reshape(-1, 3)
        color_flat = color.reshape(-1, 3) if color is not None else None
        features = self.feature_net(pc_flat, color_flat).reshape(B, N, 3)

        pc_feat = self.model.pvcnn(pc, features)
        planes = self.model.triplane_transformer(pc_feat)
        _, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)
        point_feat = sample_triplane_feat(part_planes, pc)

        if torch.isnan(point_feat).any():
            print("Warning: NaN detected in point features!")
            point_feat = torch.nan_to_num(point_feat, nan=0.0)

        total_loss = 0.0
        valid_objects = 0
        for b in range(B):
            loss_b = single_object_contrastive_loss(
                point_feat[b],
                labels[b],
                num_sample_points=self.contrast_num_points,
                temperature=self.temperature,
            )
            if not torch.isnan(loss_b):
                total_loss += loss_b
                valid_objects += 1

        if valid_objects > 0:
            loss = total_loss / valid_objects
        else:
            loss = torch.tensor(0.0, device=pc.device, requires_grad=True)

        if torch.isnan(loss):
            print("Warning: NaN in final loss!")
            loss = torch.tensor(0.0, device=pc.device, requires_grad=True)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("batch_size", float(B), prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
