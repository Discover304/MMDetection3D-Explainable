# 多模态融合部分的代码
from ntpath import join
import torch
from torch import nn as nn
from mmcv.runner import BaseModule


from ..builder import FUSION_LAYERS

# 信道编码
@FUSION_LAYERS.register_module()
class CatFusion(BaseModule):
    def __init__(self,
                img_channels=None,
                pts_channels=None,
                init_cfg=None):
        super(CatFusion, self).__init__(init_cfg=init_cfg)
    

    def forward(self, img_feats, pts_feats):
        pts_feat = pts_feats[0]
        feat_shape = list(pts_feat.size())
        feat_shape[-1] = -1
        
        pts_feat= pts_feat.view(pts_feat.size(0), pts_feat.size(1),-1)
        re_shape = list(pts_feat.size())
        
        for feat in img_feats: 
            feat = feat.view(pts_feat.size(0), pts_feat.size(1), -1)
            pts_feat = torch.cat((pts_feat, feat), dim=2)
        
        factor = pts_feat.size()[-1] // re_shape[-1] + 1
        if factor == 2:
            re_shape[-1] *= factor
            pad = torch.ones([re_shape[0], re_shape[1], re_shape[-1] - pts_feat.size()[-1]],device=pts_feat.device)
            pts_feat = torch.cat((pts_feat, pad), dim=2)
        else:
            re_shape[-1] *= 2
            pts_feat = torch.split(pts_feat, re_shape[-1], 2)[0]
    
        joint_feats = (pts_feat.reshape(*feat_shape),)
        return img_feats, joint_feats


@FUSION_LAYERS.register_module()
class XNetFusion(BaseModule):
    def __init__(self,
                 img_channels=None,
                 pts_channels=None,
                 out_channels=None,
                 init_cfg=None):
        super(XNetFusion, self).__init__(init_cfg=init_cfg)
        self.linear = torch.nn.Linear(img_channels + pts_channels, out_channels)

    def forward(self, img_feats, pts_feats):
        # x = torch.cat((pts_feats, img_feats))
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        return img_feats, pts_feats