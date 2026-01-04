import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .pv_module import SharedMLP, PVConv 

def create_pointnet_components(
        blocks, in_channels, with_se=False, normalize=True, eps=0,
        width_multiplier=1, voxel_resolution_multiplier=1, scale_pvcnn=False, device='cuda'):
    r, vr = width_multiplier, voxel_resolution_multiplier
    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = functools.partial(SharedMLP, device=device)
        else:
            block = functools.partial(
                PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                with_se=with_se, normalize=normalize, eps=eps, scale_pvcnn=scale_pvcnn, device=device)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels

class PCMerger(nn.Module):
# merge surface sampled PC and rendering backprojected PC (w/ 2D features):
    def __init__(self, in_channels=204, device="cuda"):
        super(PCMerger, self).__init__()
        self.mlp_normal = SharedMLP(3, [128, 128], device=device)
        self.mlp_rgb = SharedMLP(3, [128, 128], device=device)
        self.mlp_sam = SharedMLP(204 - 6, [128, 128], device=device)

    def forward(self, feat, mv_feat, pc2pc_idx):
        mv_feat_normal = self.mlp_normal(mv_feat[:, :3, :])
        mv_feat_rgb = self.mlp_rgb(mv_feat[:, 3:6, :])
        mv_feat_sam = self.mlp_sam(mv_feat[:, 6:, :])

        mv_feat_normal = mv_feat_normal.permute(0, 2, 1)
        mv_feat_rgb = mv_feat_rgb.permute(0, 2, 1)
        mv_feat_sam = mv_feat_sam.permute(0, 2, 1)
        feat = feat.permute(0, 2, 1)
        
        for i in range(mv_feat.shape[0]):
            mask = (pc2pc_idx[i] != -1).reshape(-1)
            idx = pc2pc_idx[i][mask].reshape(-1)
            feat[i][mask] += mv_feat_normal[i][idx] + mv_feat_rgb[i][idx] + mv_feat_sam[i][idx]
            
        return feat.permute(0, 2, 1)


class PVCNNEncoder(nn.Module):
    def __init__(self, pvcnn_feat_dim, device='cuda', in_channels=3, use_2d_feat=False):
        super(PVCNNEncoder, self).__init__()
        self.device = device
        self.blocks = ((pvcnn_feat_dim, 1, 32), (128, 2, 16), (256, 1, 8))
        self.use_2d_feat=use_2d_feat
        if in_channels == 6:
            self.append_channel = 2
        elif in_channels == 3:
            self.append_channel = 1
        else:
            raise NotImplementedError
        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks, in_channels=in_channels + self.append_channel, with_se=False, normalize=False,
            width_multiplier=1, voxel_resolution_multiplier=1, scale_pvcnn=True,
            device=device
        )
        self.encoder = nn.ModuleList(layers)#.to(self.device)
        if self.use_2d_feat:
            self.merger = PCMerger()

        

    def forward(self, input_pc, mv_feat=None, pc2pc_idx=None):
        features = input_pc.permute(0, 2, 1) * 2  # make point cloud [-1, 1]
        coords = features[:, :3, :]
        out_features_list = []
        voxel_feature_list = []
        zero_padding = torch.zeros(features.shape[0], self.append_channel, features.shape[-1], device=features.device, dtype=torch.float)
        features = torch.cat([features, zero_padding], dim=1)##################

        for i in range(len(self.encoder)):
            features, _, voxel_feature = self.encoder[i]((features, coords))
            if i == 0 and mv_feat is not None:
               features = self.merger(features, mv_feat.permute(0, 2, 1), pc2pc_idx)
            out_features_list.append(features)
            voxel_feature_list.append(voxel_feature)
        return voxel_feature_list, out_features_list