# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES

import pickle
import time

@BACKBONES.register_module()
class SECOND_INFO(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SECOND_INFO, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_filters[i],
                        out_channels[i],
                        3,
                        stride=layer_strides[i],
                        padding=1),
                    build_norm_layer(norm_cfg, out_channels[i])[1],
                    nn.ReLU(inplace=True))]
            for j in range(layer_num):
                block.append(
                    nn.Sequential(
                        build_conv_layer(
                            conv_cfg,
                            out_channels[i],
                            out_channels[i],
                            3,
                            padding=1),
                        build_norm_layer(norm_cfg, out_channels[i])[1],
                        nn.ReLU(inplace=True)))
            block = nn.ModuleList(block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        info = []
        for block in self.blocks:
            for layer in block:
                x = layer(x)
                info.append(x)
            outs.append(x)
        # 计算每一个x之间的信息量差异
        # 计算信息差的增量的方差，作为损失函数。
        # file = f"/home/yanghaobo/MMDetection3D-Explainable/work_dirs/result_evaluate_folder/net_info/noise-base_1middle_layer{time.time()}.pickle"
        # with open(file, "wb") as f:
        #     pickle.dump(tuple(info),f)
        return tuple(outs), tuple(info)
