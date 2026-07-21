from torch.autograd import Function
import torch
import torch.nn.functional as F

__all__ = ['trilinear_devoxelize']

def trilinear_devoxelize(c, coords, r, training=None):
    coords = (coords * 2 + 1.0) / r - 1.0
    coords = coords.permute(0, 2, 1).reshape(c.shape[0], 1, 1, -1, 3)
    f = F.grid_sample(input=c, grid=coords, padding_mode='border', align_corners=False)
    f = f.squeeze(dim=2).squeeze(dim=2)
    return f
