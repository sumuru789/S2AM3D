#https://github.com/3DTopia/OpenLRM/blob/main/openlrm/models/modeling_lrm.py
# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from functools import partial

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    plane_axes = torch.tensor([[[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],
                                [[1, 0, 0],
                                [0, 0, 1],
                                [0, 1, 0]],
                                [[0, 0, 1],
                                [0, 1, 0],
                                [1, 0, 0]]], dtype=torch.float32).cuda()
    
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def get_grid_coord(grid_size = 256, align_corners=False):
    if align_corners == False:
        coords = torch.linspace(-1 + 1/(grid_size), 1 - 1/(grid_size), steps=grid_size)
    else:
        coords = torch.linspace(-1, 1, steps=grid_size)
    i, j, k = torch.meshgrid(coords, coords, coords, indexing='ij')
    coordinates = torch.stack((i, j, k), dim=-1).reshape(-1, 3)
    return coordinates

class BasicBlock(nn.Module):
    """
    Transformer block that is in its simplest form.
    Designed for PF-LRM architecture.
    """
    # Block contains a self-attention layer and an MLP
    def __init__(self, inner_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x):
        # x: [N, L, D]
        before_sa = self.norm1(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x
        
class ConditionBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond, need_weights=False)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, block_type: str,
                 num_layers: int, num_heads: int,
                 inner_dim: int, cond_dim: int = None,
                 eps: float = 1e-6):
        super().__init__()
        self.block_type = block_type
        self.layers = nn.ModuleList([
            self._block_fn(inner_dim, cond_dim)(
                num_heads=num_heads,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    @property
    def block_type(self):
        return self._block_type

    @block_type.setter
    def block_type(self, block_type):
        assert block_type in ['cond', 'basic'], \
            f"Unsupported block type: {block_type}"
        self._block_type = block_type

    def _block_fn(self, inner_dim, cond_dim):
        assert inner_dim is not None, f"inner_dim must always be specified"
        if self.block_type == 'basic':
            return partial(BasicBlock, inner_dim=inner_dim)
        elif self.block_type == 'cond':
            assert cond_dim is not None, f"Condition dimension must be specified for ConditionBlock"
            return partial(ConditionBlock, inner_dim=inner_dim, cond_dim=cond_dim)
        else:
            raise ValueError(f"Unsupported block type during runtime: {self.block_type}")


    def forward_layer(self, layer: nn.Module, x: torch.Tensor, cond: torch.Tensor,):
        if self.block_type == 'basic':
            return layer(x)
        elif self.block_type == 'cond':
            return layer(x, cond)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond] or None
        for layer in self.layers:
            x = self.forward_layer(layer, x, cond)
        x = self.norm(x)
        return x

class Voxel2Triplane(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int, voxel_feat_dim: int, normalize_vox_feat=False, voxel_dim=16):
        super().__init__()
        
        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim
        self.voxel_feat_dim = voxel_feat_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type='cond',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim, cond_dim=voxel_feat_dim
        )
        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=8, stride=8, padding=0)

        self.normalize_vox_feat = normalize_vox_feat
        if normalize_vox_feat:
            self.vox_norm = nn.LayerNorm(voxel_feat_dim, eps=1e-6)
            self.vox_pos_embed = nn.Parameter(torch.randn(1, voxel_dim * voxel_dim * voxel_dim, voxel_feat_dim) * (1. / voxel_feat_dim) ** 0.5)

    def forward_transformer(self, voxel_feats):
        N = voxel_feats.shape[0]
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        if self.normalize_vox_feat:
            vox_pos_embed = self.vox_pos_embed.repeat(N, 1, 1)  # [N, L, D]
            voxel_feats = self.vox_norm(voxel_feats + vox_pos_embed)
        x = self.transformer(
            x,
            cond=voxel_feats
        )
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    def forward(self, voxel_feats):
        N = voxel_feats.shape[0]

        # encode image
        assert voxel_feats.shape[-1] == self.voxel_feat_dim, \
            f"Feature dimension mismatch: {voxel_feats.shape[-1]} vs {self.voxel_feat_dim}"

        # transformer generating planes
        tokens = self.forward_transformer(voxel_feats)
        planes = self.reshape_upsample(tokens)
        assert planes.shape[0] == N, "Batch size mismatch for planes"
        assert planes.shape[1] == 3, "Planes should have 3 channels"

        return planes


class TriplaneTransformer(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self, input_dim: int, transformer_dim: int, transformer_layers: int, transformer_heads: int,
                 triplane_low_res: int, triplane_high_res: int, triplane_dim: int):
        super().__init__()
        
        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, transformer_dim) * (1. / transformer_dim) ** 0.5)
        self.transformer = TransformerDecoder(
            block_type='basic',
            num_layers=transformer_layers, num_heads=transformer_heads,
            inner_dim=transformer_dim,
        )
      
        self.downsampler = nn.Sequential(
            nn.Conv2d(input_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size from 128x128 to 64x64
            
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces size from 64x64 to 32x32
        )

        self.upsampler = nn.ConvTranspose2d(transformer_dim, triplane_dim, kernel_size=4, stride=4, padding=0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, triplane_dim),
            nn.ReLU(),
            nn.Linear(triplane_dim, triplane_dim)
        )

    def forward_transformer(self, triplanes):
        N = triplanes.shape[0]
        tokens = torch.einsum('nidhw->nihwd', triplanes).reshape(N, self.pos_embed.shape[1], -1) # [N, L, D]
        x = self.pos_embed.repeat(N, 1, 1) + tokens # [N, L, D]
        x = self.transformer(x)
        return x

    def reshape_downsample(self, triplanes):
        N = triplanes.shape[0]
        H = W = self.triplane_high_res
        x = triplanes.view(N, 3, -1, H, W)
        x = torch.einsum('nidhw->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.downsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x

    def reshape_upsample(self, tokens):
        N = tokens.shape[0]
        H = W = self.triplane_low_res
        x = tokens.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.upsampler(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()
        return x
    
    def forward(self, triplanes):
        downsampled_triplanes = self.reshape_downsample(triplanes)
        tokens = self.forward_transformer(downsampled_triplanes)
        residual = self.reshape_upsample(tokens)
        
        triplanes = triplanes.permute(0, 1, 3, 4, 2).contiguous()
        triplanes = self.mlp(triplanes)
        triplanes = triplanes.permute(0, 1, 4, 2, 3).contiguous()
        planes = triplanes + residual
        return planes
