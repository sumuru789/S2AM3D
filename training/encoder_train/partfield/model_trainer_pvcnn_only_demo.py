import torch
import lightning.pytorch as pl
from .dataloader import Demo_Dataset, Demo_Remesh_Dataset, Correspondence_Demo_Dataset
from torch.utils.data import DataLoader
from partfield.model.UNet.model import ResidualUNet3D
from partfield.model.triplane import TriplaneTransformer, get_grid_coord
from partfield.model.model_utils import VanillaMLP
import torch.nn.functional as F
import torch.nn as nn
import os
import trimesh
import skimage
import numpy as np
import h5py
import torch.distributed as dist
from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
import json
import gc
import time
from plyfile import PlyData, PlyElement


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.automatic_optimization = False

        self.processed_files = 0
        self.skipped_files = 0
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1,
                                      out_activation="tanh",
                                      n_neurons=64,
                                      n_hidden_layers=6)
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device="cuda",
                shape_min=-1,
                shape_length=2,
                use_2d_feat=self.use_2d_feat)
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192,
                                out_activation="GELU",
                                n_neurons=64,
                                n_hidden_layers=6)

    def forward(self, pc):

        pc_feat = self.pvcnn(pc, pc)
        planes = self.triplane_transformer(pc_feat)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        point_feat = sample_triplane_feat(part_planes, pc)
        global_feat = point_feat.mean(dim=1)
        return global_feat

    def predict_dataloader(self):
        if self.cfg.remesh_demo:
            dataset = Demo_Remesh_Dataset(self.cfg)
        elif self.cfg.correspondence_demo:
            dataset = Correspondence_Demo_Dataset(self.cfg)
        else:
            dataset = Demo_Dataset(self.cfg)

        dataloader = DataLoader(dataset,
                            num_workers=self.cfg.dataset.val_num_workers,
                            batch_size=self.cfg.dataset.val_batch_size,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

        return dataloader


    @torch.no_grad()
    def predict_step(self, batch, batch_idx):

        if hasattr(self.cfg, 'output_feature_dir') and isinstance(self.cfg.output_feature_dir, str) and len(self.cfg.output_feature_dir) > 0:
            save_dir = self.cfg.output_feature_dir
        else:
            save_dir = f"exp_results/{self.cfg.result_name}"
        os.makedirs(save_dir, exist_ok=True)

        uid = batch['uid'][0]
        view_id = 0
        starttime = time.time()

        if uid == "car" or uid == "complex_car":

            print("Skipping this for now.")
            print(uid)
            return


        if self.cfg.is_pc:
            feature_file_path = f'{save_dir}/part_feat_{uid}_{view_id}.npy'
        else:

            feature_file_path = f'{save_dir}/part_feat_{uid}_{view_id}.npy'
            feature_file_path_batch = f'{save_dir}/part_feat_{uid}_{view_id}_batch.npy'


            if os.path.exists(feature_file_path) or os.path.exists(feature_file_path_batch):
                self.skipped_files += 1
                print(f"Skip existing: part_feat_{uid}_{view_id}.npy (skipped={self.skipped_files})")
                return


        if self.cfg.is_pc and os.path.exists(feature_file_path):
            self.skipped_files += 1
            print(f"Skip existing: part_feat_{uid}_{view_id}.npy (skipped={self.skipped_files})")
            return

        N = batch['pc'].shape[0]
        assert N == 1

        if self.use_2d_feat:
            print("ERROR. Dataloader not implemented with input 2d feat.")
            exit()
        else:
            pc_feat = self.pvcnn(batch['pc'], batch['pc'])

        planes = pc_feat
        planes = self.triplane_transformer(planes)
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        if self.cfg.is_pc:
            tensor_vertices = batch['pc'].reshape(1, -1, 3).cuda().to(torch.float16)
            point_feat = sample_triplane_feat(part_planes, tensor_vertices)
            point_feat = point_feat.cpu().detach().numpy().reshape(-1, 448)

            feature_file_path = f'{save_dir}/part_feat_{uid}_{view_id}.npy'
            np.save(feature_file_path, point_feat)
            self.processed_files += 1
            print(f"Exported part_feat_{uid}_{view_id}.npy (processed={self.processed_files})")


            """"""
        print("Time elapsed: " + str(time.time()-starttime))
        print(f"Done. processed={self.processed_files}, skipped={self.skipped_files}")

        return