import torch
import torch.nn as nn

from . import functional as F

__all__ = ['Voxelization']


def my_voxelization(features, coords, resolution):
    b, c, _ = features.shape
    result = torch.zeros(b, c + 1, resolution * resolution * resolution, device=features.device, dtype=torch.float)
    r = resolution
    r2 = resolution * resolution
    indices = coords[:, 0] * r2 + coords[:, 1] * r + coords[:, 2]
    indices = indices.unsqueeze(dim=1).expand(-1, result.shape[1], -1)
    features = torch.cat([features, torch.ones(features.shape[0], 1, features.shape[2], device=features.device, dtype=features.dtype)], dim=1)
    out_feature = result.scatter_reduce_(index=indices.long(), src=features, dim=2, reduce='sum')
    cnt = out_feature[:, -1:, :]
    zero_mask = (cnt == 0).float()
    cnt = cnt * (1 - zero_mask) + zero_mask * 1e-5
    vox_feature = out_feature[:, :-1, :] / cnt
    return vox_feature.view(b, c, resolution, resolution, resolution)

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0, scale_pvcnn=False):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
        self.scale_pvcnn = scale_pvcnn
        assert not normalize

    def forward(self, features, coords):
        with torch.no_grad():
            coords = coords.detach()

            if self.normalize:
                norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
            else:
                if self.scale_pvcnn:
                    norm_coords = (coords + 1) / 2.0 # [0, 1]
                else:
                    norm_coords = (norm_coords + 1) / 2.0
            norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
            vox_coords = torch.round(norm_coords)
        new_vox_feat = my_voxelization(features, vox_coords, self.r)
        return new_vox_feat, norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
