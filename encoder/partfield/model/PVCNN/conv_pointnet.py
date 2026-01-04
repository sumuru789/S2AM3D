"""
Taken from gensdf
https://github.com/princeton-computational-imaging/gensdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from dnnlib.util import printarr
try:
    from torch_scatter import scatter_mean, scatter_max
except:
    pass
# from .unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ConvPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                #  unet=False, unet_kwargs=None, 
                 plane_resolution=None, plane_type=['xz', 'xy', 'yz'], padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # if unet:
        #     self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        # else:
        #     self.unet = None

        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean


    # takes in "p": point cloud and "query": sdf_xyz 
    # sample plane features for unlabeled_query as well 
    def forward(self, p):#, query2):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        plane_feat_sum = 0
        #second_sum = 0
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz') # shape: batch, latent size, resolution, resolution (e.g. 16, 256, 64, 64)
            # plane_feat_sum += self.sample_plane_feature(query, fea['xz'], 'xz')
            #second_sum += self.sample_plane_feature(query2, fea['xz'], 'xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            # plane_feat_sum += self.sample_plane_feature(query, fea['xy'], 'xy')
            #second_sum += self.sample_plane_feature(query2, fea['xy'], 'xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
            # plane_feat_sum += self.sample_plane_feature(query, fea['yz'], 'yz')
            #second_sum += self.sample_plane_feature(query2, fea['yz'], 'yz')
        return fea

        # return plane_feat_sum.transpose(2,1)#, second_sum.transpose(2,1)


    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
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

        # f there are outliers out of the range
        if xy_new.max() >= 1:
            xy_new[xy_new >= 1] = 1 - 10e-6
        if xy_new.min() < 0:
            xy_new[xy_new < 0] = 0.0
        return xy_new


    def coordinate2index(self, x, reso):
        ''' Normalize coordinate to [0, 1] for unit cube experiments.
            Corresponds to our 3D model

        Args:
            x (tensor): coordinate
            reso (int): defined resolution
            coord_type (str): coordinate type
        '''
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        index = index[:, None, :]
        return index


    # xy is the normalized coordinates of the point cloud of each plane 
    # I'm pretty sure the keys of xy are the same as those of index, so xy isn't needed here as input 
    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = self.coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)
        
        # printarr(fea_plane, c, p, xy, index)
        # import pdb; pdb.set_trace()

        # process the plane features with UNet
        # if self.unet is not None:
        #     fea_plane = self.unet(fea_plane)

        return fea_plane


    # sample_plane_feature function copied from /src/conv_onet/models/decoder.py
    # uses values from plane_feature and pixel locations from vgrid to interpolate feature
    def sample_plane_feature(self, query, plane_feature, plane):
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        sampled_feat = F.grid_sample(plane_feature, vgrid, padding_mode='border', align_corners=True, mode='bilinear').squeeze(-1)
        return sampled_feat


    