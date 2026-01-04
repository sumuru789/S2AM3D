import torch.nn as nn

from . import functional as F
from .voxelization import Voxelization
from .shared_mlp import SharedMLP
import torch

__all__ = ['PVConv']


class PVConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0, scale_pvcnn=False,
            device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps, scale_pvcnn=scale_pvcnn)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, device=device),
            nn.InstanceNorm3d(out_channels, eps=1e-4, device=device),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, device=device),
            nn.InstanceNorm3d(out_channels, eps=1e-4, device=device),
            nn.LeakyReLU(0.1, True),
        ]
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels, device=device)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        devoxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = devoxel_features + self.point_features(features)
        #fused_features = devoxel_features + self.point_features(features)+promptfeatures
        return fused_features, coords, voxel_features
