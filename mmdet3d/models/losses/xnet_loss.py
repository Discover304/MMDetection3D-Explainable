# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

def calculate_entropy(feat_in, feat_out):
    # print("计算Entropy") # TODO
    return torch.var(feat_out)-torch.var(feat_in)

@weighted_loss
def layer_entropy_loss(value, target):
    loss = value - target
    return loss

@LOSSES.register_module()
class EntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.target = 0

    def forward(self, net_info):
        delta_entropy = []
        for i in range(len(net_info)-1):
            delta_entropy.append(calculate_entropy(net_info[i], net_info[i+1]))
        delta_entropy = torch.stack(delta_entropy)
        var = torch.var(delta_entropy)
        net_loss = self.loss_weight * layer_entropy_loss(var, self.target)
        return {"loss_net": [net_loss]}
