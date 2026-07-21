# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
                                                                       
                                                                      
                                                                            
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from ast import Dict
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean               

from .unet_3daware import setup_unet                     
from .conv_pointnet import ConvPointnet

from .pc_encoder import PVCNNEncoder          

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
                                          
    xy = normalize_coordinate(p.clone(), plane=plane, padding=padding)                                   
    index = coordinate2index(xy, resolution)

                                        
    fea_plane = c.new_zeros(p.size(0), c_dim, resolution**2)
    fea_plane = scatter_mean(c, index, out=fea_plane)                   
    fea_plane = fea_plane.reshape(p.size(0), c_dim, resolution, resolution)                                        
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

    xy_new = xy / (1 + padding + 10e-6)              
    xy_new = xy_new + 0.5               

                                            
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
    final_feat = final_feat.squeeze(dim=2).permute(0, 2, 1)               
    return final_feat


                               
class TriPlanePC2Encoder(torch.nn.Module):
                                                                                      
    def __init__(
            self,
            cfg,
            device='cuda',
            shape_min=-1.0,
            shape_length=2.0,
            use_2d_feat=False,
                                    
                                     
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
                                                                                   
        super().__init__()
        self.device = device

        self.cfg = cfg

        self.shape_min = shape_min
        self.shape_length = shape_length

        self.z_triplane_resolution = cfg.z_triplane_resolution
        z_triplane_channels = cfg.z_triplane_channels

        point_encoder_out_dim = z_triplane_channels     

        in_channels = 6
                                           
        if cfg.point_encoder_type == 'pvcnn':
            self.pc_encoder = PVCNNEncoder(point_encoder_out_dim, 
            device=self.device, in_channels=in_channels, use_2d_feat=use_2d_feat)                                 
        elif cfg.point_encoder_type == 'pointnet':
                                                      
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

                                    
    def encode(self, point_cloud_xyz, point_cloud_feature, mv_feat=None, pc2pc_idx=None) -> Dict:
                             
        point_cloud_xyz = (point_cloud_xyz - self.shape_min) / self.shape_length         
        point_cloud_xyz = point_cloud_xyz - 0.5              
        point_cloud = torch.cat([point_cloud_xyz, point_cloud_feature], dim=-1)

        if self.cfg.point_encoder_type == 'pvcnn':
            if mv_feat is not None:
                pc_feat, points_feat = self.pc_encoder(point_cloud, mv_feat, pc2pc_idx)
            else:
                pc_feat, points_feat = self.pc_encoder(point_cloud)                                   
            if self.cfg.use_point_scatter:
                                                      
                points_feat_ = points_feat[0]
                                                                                          
                pc_feat_1 = generate_plane_features(point_cloud_xyz, points_feat_, 
                                                    resolution=self.z_triplane_resolution, plane='xy') 
                pc_feat_2 = generate_plane_features(point_cloud_xyz, points_feat_,
                                                    resolution=self.z_triplane_resolution, plane='yz') 
                pc_feat_3 = generate_plane_features(point_cloud_xyz, points_feat_,
                                                    resolution=self.z_triplane_resolution, plane='xz') 
                pc_feat = pc_feat[0]

            else:
                pc_feat = pc_feat[0]
                sf = self.z_triplane_resolution//32                          

                pc_feat_1 = torch.mean(pc_feat, dim=-1)                                
                pc_feat_2 = torch.mean(pc_feat, dim=-3)                                
                pc_feat_3 = torch.mean(pc_feat, dim=-2)                                

                                  
                pc_feat_1 = einops.repeat(pc_feat_1, 'b c h w -> b c (h hm ) (w wm)', hm = sf, wm = sf)
                pc_feat_2 = einops.repeat(pc_feat_2, 'b c h w -> b c (h hm) (w wm)', hm = sf, wm = sf)
                pc_feat_3 = einops.repeat(pc_feat_3, 'b c h w -> b c (h hm) (w wm)', hm = sf, wm = sf)
        elif self.cfg.point_encoder_type == 'pointnet':
            assert self.cfg.use_point_scatter
                              
            pc_feat = self.pc_encoder(point_cloud)
            pc_feat_1 = pc_feat['xy']  
            pc_feat_2 = pc_feat['yz']
            pc_feat_3 = pc_feat['xz']
        else:
            raise NotImplementedError()

        if self.unet_encoder is not None:
                                                
                                        
            pc_feat_tri_plane_stack_pre = torch.stack([pc_feat_1, pc_feat_2, pc_feat_3], dim=1)
                                                                                       
                                                                                              
            pc_feat_tri_plane_stack = self.unet_encoder(pc_feat_tri_plane_stack_pre)
            pc_feat_1, pc_feat_2, pc_feat_3 = torch.unbind(pc_feat_tri_plane_stack, dim=1)

        return torch.stack([pc_feat_1, pc_feat_2, pc_feat_3], dim=1)
        
    def forward(self, point_cloud_xyz, point_cloud_feature=None, mv_feat=None, pc2pc_idx=None):
        return self.encode(point_cloud_xyz, point_cloud_feature=point_cloud_feature, mv_feat=mv_feat, pc2pc_idx=pc2pc_idx)