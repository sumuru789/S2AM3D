# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from ast import Dict
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean #, scatter_max

from .unet_3daware import setup_unet #UNetTriplane3dAware
from .conv_pointnet import ConvPointnet

from .pc_encoder import PVCNNEncoder #PointNet

import einops

from .dnnlib_util import ScopedTorchProfiler, printarr

def generate_plane_features(p, c, resolution, plane='xz'):
    """
    Args:
        p: (B,3,n_p)
        c: (B,C,n_p)
    """
    padding = 0.
    c_dim = c.size(1)
    # acquire indices of features in plane
    xy = normalize_coordinate(p.clone(), plane=plane, padding=padding) # normalize to the range of (0, 1)
    index = coordinate2index(xy, resolution)

    # scatter plane features from points
    fea_plane = c.new_zeros(p.size(0), c_dim, resolution**2)
    fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
    fea_plane = fea_plane.reshape(p.size(0), c_dim, resolution, resolution) # sparce matrix (B x 512 x reso x reso)
    return fea_plane

def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # if there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def coordinate2index(x, resolution):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * resolution).long()
    index = x[:, :, 0] + resolution * x[:, :, 1]
    index = index[:, None, :]
    return index

def softclip(x, min, max, hardness=5):
    # Soft clipping for the logsigma
    x = min + F.softplus(hardness*(x - min))/hardness
    x = max - F.softplus(-hardness*(x - max))/hardness
    return x


def sample_triplane_feat(feature_triplane, normalized_pos):
    '''
        normalized_pos [-1, 1]
    '''
    tri_plane = torch.unbind(feature_triplane, dim=1)

    x_feat = F.grid_sample(
        tri_plane[0],
        torch.cat(
            [normalized_pos[:, :, 0:1], normalized_pos[:, :, 1:2]],
            dim=-1).unsqueeze(dim=1), padding_mode='border',
        align_corners=True)
    y_feat = F.grid_sample(
        tri_plane[1],
        torch.cat(
            [normalized_pos[:, :, 1:2], normalized_pos[:, :, 2:3]],
            dim=-1).unsqueeze(dim=1), padding_mode='border',
        align_corners=True)

    z_feat = F.grid_sample(
        tri_plane[2],
        torch.cat(
            [normalized_pos[:, :, 0:1], normalized_pos[:, :, 2:3]],
            dim=-1).unsqueeze(dim=1), padding_mode='border',
        align_corners=True)
    final_feat = (x_feat + y_feat + z_feat)
    final_feat = final_feat.squeeze(dim=2).permute(0, 2, 1)  # 32dimension
    return final_feat


# @persistence.persistent_class
class TriPlanePC2Encoder(torch.nn.Module):
    # Encoder that encode point cloud to triplane feature vector similar to ConvOccNet
    def __init__(
            self,
            cfg,
            device='cuda',
            shape_min=-1.0,
            shape_length=2.0,
            use_2d_feat=False,
            # point_encoder='pvcnn',
            # use_point_scatter=False
    ):
        """
        Outputs latent triplane from PC input
        Configs:
            max_logsigma: (float) Soft clip upper range for logsigm
            min_logsigma: (float)
            point_encoder_type: (str) one of ['pvcnn', 'pointnet']
            pvcnn_flatten_voxels: (bool) for pvcnn whether to reduce voxel 
                features (instead of scattering point features)
            unet_cfg: (dict)
            z_triplane_channels: (int) output latent triplane
            z_triplane_resolution: (int)
        Args:

        """
        # assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.device = device

        self.cfg = cfg

        self.shape_min = shape_min
        self.shape_length = shape_length

        self.z_triplane_resolution = cfg.z_triplane_resolution
        z_triplane_channels = cfg.z_triplane_channels

        point_encoder_out_dim = z_triplane_channels #* 2

        in_channels = 6
        # self.resample_filter=[1, 3, 3, 1]
        if cfg.point_encoder_type == 'pvcnn':
            self.pc_encoder = PVCNNEncoder(point_encoder_out_dim, 
            device=self.device, in_channels=in_channels, use_2d_feat=use_2d_feat)  # Encode it to a volume vector.
        elif cfg.point_encoder_type == 'pointnet':
            # TODO the pointnet was buggy, investigate
            self.pc_encoder = ConvPointnet(c_dim=point_encoder_out_dim, 
                                           dim=in_channels, hidden_dim=32, 
                                           plane_resolution=self.z_triplane_resolution, 
                                           padding=0)
        else:
            raise NotImplementedError(f"Point encoder {cfg.point_encoder_type} not implemented")

        if cfg.unet_cfg.enabled:
            self.unet_encoder = setup_unet(
                output_channels=point_encoder_out_dim, 
                input_channels=point_encoder_out_dim, 
                unet_cfg=cfg.unet_cfg)
        else:
            self.unet_encoder = None

    # @ScopedTorchProfiler('encode')
    def encode(self, point_cloud_xyz, point_cloud_feature, mv_feat=None, pc2pc_idx=None) -> Dict:
        # output = AttrDict()
        point_cloud_xyz = (point_cloud_xyz - self.shape_min) / self.shape_length # [0, 1]
        point_cloud_xyz = point_cloud_xyz - 0.5 # [-0.5, 0.5]
        point_cloud = torch.cat([point_cloud_xyz, point_cloud_feature], dim=-1)

        if self.cfg.point_encoder_type == 'pvcnn':
            if mv_feat is not None:
                pc_feat, points_feat = self.pc_encoder(point_cloud, mv_feat, pc2pc_idx)
            else:
                pc_feat, points_feat = self.pc_encoder(point_cloud)  # 3D feature volume: BxDx32x32x32
            if self.cfg.use_point_scatter:
                # Scattering from PVCNN point features
                points_feat_ = points_feat[0]
                # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
                pc_feat_1 = generate_plane_features(point_cloud_xyz, points_feat_, 
                                                    resolution=self.z_triplane_resolution, plane='xy') 
                pc_feat_2 = generate_plane_features(point_cloud_xyz, points_feat_,
                                                    resolution=self.z_triplane_resolution, plane='yz') 
                pc_feat_3 = generate_plane_features(point_cloud_xyz, points_feat_,
                                                    resolution=self.z_triplane_resolution, plane='xz') 
                pc_feat = pc_feat[0]

            else:
                pc_feat = pc_feat[0]
                sf = self.z_triplane_resolution//32 # 32 is PVCNN's voxel dim

                pc_feat_1 = torch.mean(pc_feat, dim=-1) #xy_plane, normalize in z plane
                pc_feat_2 = torch.mean(pc_feat, dim=-3) #yz_plane, normalize in x plane
                pc_feat_3 = torch.mean(pc_feat, dim=-2) #xz_plane, normalize in y plane

                # nearest upsample
                pc_feat_1 = einops.repeat(pc_feat_1, 'b c h w -> b c (h hm ) (w wm)', hm = sf, wm = sf)
                pc_feat_2 = einops.repeat(pc_feat_2, 'b c h w -> b c (h hm) (w wm)', hm = sf, wm = sf)
                pc_feat_3 = einops.repeat(pc_feat_3, 'b c h w -> b c (h hm) (w wm)', hm = sf, wm = sf)
        elif self.cfg.point_encoder_type == 'pointnet':
            assert self.cfg.use_point_scatter
            # Run ConvPointnet
            pc_feat = self.pc_encoder(point_cloud)
            pc_feat_1 = pc_feat['xy'] #
            pc_feat_2 = pc_feat['yz']
            pc_feat_3 = pc_feat['xz']
        else:
            raise NotImplementedError()

        if self.unet_encoder is not None:
            # TODO eval adding a skip connection
            # Unet expects B, 3, C, H, W
            pc_feat_tri_plane_stack_pre = torch.stack([pc_feat_1, pc_feat_2, pc_feat_3], dim=1)
            # dpc_feat_tri_plane_stack = self.unet_encoder(pc_feat_tri_plane_stack_pre)
            # pc_feat_tri_plane_stack = pc_feat_tri_plane_stack_pre + dpc_feat_tri_plane_stack
            pc_feat_tri_plane_stack = self.unet_encoder(pc_feat_tri_plane_stack_pre)
            pc_feat_1, pc_feat_2, pc_feat_3 = torch.unbind(pc_feat_tri_plane_stack, dim=1)

        return torch.stack([pc_feat_1, pc_feat_2, pc_feat_3], dim=1)
        
    def forward(self, point_cloud_xyz, point_cloud_feature=None, mv_feat=None, pc2pc_idx=None):
        return self.encode(point_cloud_xyz, point_cloud_feature=point_cloud_feature, mv_feat=mv_feat, pc2pc_idx=pc2pc_idx)