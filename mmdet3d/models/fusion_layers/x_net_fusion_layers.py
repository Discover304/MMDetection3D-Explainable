# 多模态融合部分的代码
import torch
from mmcv.runner import BaseModule

from ..builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class NoneFusion(BaseModule):
    def __init__(self,
                 init_cfg=None):
        super(NoneFusion, self).__init__(init_cfg=init_cfg)

    def forward(self, img_feats, pts_feats):
        return img_feats, pts_feats

# 信道编码
@FUSION_LAYERS.register_module()
class XNetFusion(BaseModule):
    def __init__(self,
                 img_channels=None,
                 pts_channels=None,
                 out_channels=None,
                 init_cfg=None):
        super(XNetFusion, self).__init__(init_cfg=init_cfg)
        self.linear = torch.nn.Linear(img_channels + pts_channels, out_channels)

    def forward(self, pts_feats, img_feats):
        # x = torch.cat((pts_feats, img_feats))
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        return img_feats, pts_feats
