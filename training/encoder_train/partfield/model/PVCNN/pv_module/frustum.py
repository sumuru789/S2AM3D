import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import functional as PF

__all__ = ['FrustumPointNetLoss', 'get_box_corners_3d']


class FrustumPointNetLoss(nn.Module):
    def __init__(
            self, num_heading_angle_bins, num_size_templates, size_templates, box_loss_weight=1.0,
            corners_loss_weight=10.0, heading_residual_loss_weight=20.0, size_residual_loss_weight=20.0):
        super().__init__()
        self.box_loss_weight = box_loss_weight
        self.corners_loss_weight = corners_loss_weight
        self.heading_residual_loss_weight = heading_residual_loss_weight
        self.size_residual_loss_weight = size_residual_loss_weight

        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.register_buffer('size_templates', size_templates.view(self.num_size_templates, 3))
        self.register_buffer(
            'heading_angle_bin_centers', torch.arange(0, 2 * np.pi, 2 * np.pi / self.num_heading_angle_bins)
        )

    def forward(self, inputs, targets):
        mask_logits = inputs['mask_logits']             
        center_reg = inputs['center_reg']          
        center = inputs['center']          
        heading_scores = inputs['heading_scores']           
        heading_residuals_normalized = inputs['heading_residuals_normalized']           
        heading_residuals = inputs['heading_residuals']           
        size_scores = inputs['size_scores']           
        size_residuals_normalized = inputs['size_residuals_normalized']              
        size_residuals = inputs['size_residuals']              

        mask_logits_target = targets['mask_logits']          
        center_target = targets['center']          
        heading_bin_id_target = targets['heading_bin_id']         
        heading_residual_target = targets['heading_residual']         
        size_template_id_target = targets['size_template_id']         
        size_residual_target = targets['size_residual']          

        batch_size = center.size(0)
        batch_id = torch.arange(batch_size, device=center.device)

                                                    
        mask_loss = F.cross_entropy(mask_logits, mask_logits_target)
        heading_loss = F.cross_entropy(heading_scores, heading_bin_id_target)
        size_loss = F.cross_entropy(size_scores, size_template_id_target)
        center_loss = PF.huber_loss(torch.norm(center_target - center, dim=-1), delta=2.0)
        center_reg_loss = PF.huber_loss(torch.norm(center_target - center_reg, dim=-1), delta=1.0)

                                            
        heading_residuals_normalized = heading_residuals_normalized[batch_id, heading_bin_id_target]         
        heading_residual_normalized_target = heading_residual_target / (np.pi / self.num_heading_angle_bins)
        heading_residual_normalized_loss = PF.huber_loss(
            heading_residuals_normalized - heading_residual_normalized_target, delta=1.0
        )
        size_residuals_normalized = size_residuals_normalized[batch_id, size_template_id_target]          
        size_residual_normalized_target = size_residual_target / self.size_templates[size_template_id_target]
        size_residual_normalized_loss = PF.huber_loss(
            torch.norm(size_residual_normalized_target - size_residuals_normalized, dim=-1), delta=1.0
        )

                             
        heading = (heading_residuals[batch_id, heading_bin_id_target]
                   + self.heading_angle_bin_centers[heading_bin_id_target])         
                                                                                                                     
        size = (size_residuals[batch_id, size_template_id_target]
                + self.size_templates[size_template_id_target])          
        corners = get_box_corners_3d(centers=center, headings=heading, sizes=size, with_flip=False)             
        heading_target = self.heading_angle_bin_centers[heading_bin_id_target] + heading_residual_target         
        size_target = self.size_templates[size_template_id_target] + size_residual_target          
        corners_target, corners_target_flip = get_box_corners_3d(
            centers=center_target, headings=heading_target,
            sizes=size_target, with_flip=True)             
        corners_loss = PF.huber_loss(
            torch.min(
                torch.norm(corners - corners_target, dim=1), torch.norm(corners - corners_target_flip, dim=1)
            ), delta=1.0)
                    
        loss = mask_loss + self.box_loss_weight * (
                center_loss + center_reg_loss + heading_loss + size_loss
                + self.heading_residual_loss_weight * heading_residual_normalized_loss
                + self.size_residual_loss_weight * size_residual_normalized_loss
                + self.corners_loss_weight * corners_loss
        )

        return loss


def get_box_corners_3d(centers, headings, sizes, with_flip=False):
    """
    :param centers: coords of box centers, FloatTensor[N, 3]
    :param headings: heading angles, FloatTensor[N, ]
    :param sizes: box sizes, FloatTensor[N, 3]
    :param with_flip: bool, whether to return flipped box (headings + np.pi)
    :return:
        coords of box corners, FloatTensor[N, 3, 8]
        NOTE: corner points are in counter clockwise order, e.g.,
          2--1
        3--0 5
        7--4
    """
    l = sizes[:, 0]        
    w = sizes[:, 1]        
    h = sizes[:, 2]        
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)          
    y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)          
    z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)          

    c = torch.cos(headings)        
    s = torch.sin(headings)        
    o = torch.ones_like(headings)        
    z = torch.zeros_like(headings)        

    centers = centers.unsqueeze(-1)             
    corners = torch.stack([x_corners, y_corners, z_corners], dim=1)             
    R = torch.stack([c, z, s, z, o, z, -s, z, c], dim=1).view(-1, 3, 3)                          
    if with_flip:
        R_flip = torch.stack([-c, z, -s, z, o, z, s, z, -c], dim=1).view(-1, 3, 3)
        return torch.matmul(R, corners) + centers, torch.matmul(R_flip, corners) + centers
    else:
        return torch.matmul(R, corners) + centers

                                                 
                                                                                   
                                                                                       
                   
                                                                                                  
                                                                                                           
           
                                                                 

                                                                                  
                                                                                      
                                                                            
                                                    
