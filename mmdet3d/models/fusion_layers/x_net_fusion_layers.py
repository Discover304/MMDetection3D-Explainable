# 多模态融合部分的代码
from ntpath import join
import torch
from torch import nn as nn
from mmcv.runner import BaseModule
from torch.nn import functional as F



from ..builder import FUSION_LAYERS

# 信道编码
@FUSION_LAYERS.register_module()
class CatFusion(BaseModule):
    """Concate features of pts and img

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (list[int] | int): Channels of point features.
            It could be a list if the input is multi-scale image features.
        out_channels (int): Channels of output fused features
        dropout_ratio (int, float, optional): Dropout ratio defaults to 0.2.
    
    Returns:
        tuple[torch.Tensor]: Corresponding features of image and points.
    """
        
    def __init__(self,
                img_channel=256,
                img_levels=5,
                pts_channel=256,
                pts_levels=1,
                out_channel=512,
                dropout_ratio=0.2,
                init_cfg=None):
        super(CatFusion, self).__init__(init_cfg=init_cfg)
        self.img_levels = img_levels
        self.pts_levels = pts_levels
        self.img_channel = img_channel
        self.img_channels = [img_channel for _ in range(img_levels)]
        self.pts_channel = pts_channel
        self.pts_channels = [pts_channel for _ in range(pts_levels)]
        self.dropout_ratio = dropout_ratio
        self.out_channel = out_channel
        
        self.img_transform = nn.Sequential(
            nn.Linear(img_channel, out_channel),
            nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01),
        )
        
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channel, out_channel),
            nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01),
        )

        self.fuse_conv = nn.Sequential(
            nn.Linear(out_channel, out_channel),
            nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=False))
    

    def forward(self, img_feats, pts_feats):
        N, C, H, W = pts_feats[0].size()
        
        img_feats = self.obtain_mlvl_feats(img_feats, self.img_channel, self.img_levels)
        img_pre_fuse = [self.img_transform(img_feat.t()).t() for img_feat in img_feats]
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = [F.dropout(x, self.dropout_ratio) for x in img_pre_fuse]
        img_pre_fuse = torch.stack(img_pre_fuse,dim=0)
        print(f"fusion_layer/img_pre_fuse: {img_pre_fuse.size()}")
        _, _, img_F = img_pre_fuse.size()
        
        pts_feats = self.obtain_mlvl_feats(pts_feats, self.pts_channel, self.pts_levels)
        pts_pre_fuse = [self.pts_transform(pts_feat.t()).t() for pts_feat in pts_feats]
        if self.training and self.dropout_ratio > 0:
            pts_pre_fuse = [F.dropout(x, self.dropout_ratio) for x in pts_pre_fuse]
        pts_pre_fuse = torch.stack(pts_pre_fuse,dim=0)
        print(f"fusion_layer/pts_pre_fuse: {pts_pre_fuse.size()}")
        _, _, pts_F = pts_pre_fuse.size()
        
        pad_h = 1000
        pad = torch.randn([N, self.out_channel, pad_h*((pts_F+img_F)//pad_h+1)-(pts_F+img_F)], device=next(self.parameters()).device)
        concate_out = torch.cat([img_pre_fuse, pts_pre_fuse, pad], dim=-1)
        print(f"fusion_layer/concate_out: {concate_out.size()}")
        
        fuse_pre_out = [self.fuse_conv(x.t()).t().view(self.out_channel, pad_h, -1) for x in concate_out]
        fuse_out = torch.stack(fuse_pre_out,dim=0)
        print(f"fusion_layer/fuse_out: {fuse_out.size()}")
        return fuse_out
    
    
    def obtain_mlvl_feats(self, feats, channel, levels):
        """Obtain multi-level features for point features.

        Args:
            feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            img_levels (int): Number of FPN levels.

        Returns:
            torch.Tensor: features.
        """
        mlvl_feats_list = []
        # Sample multi-level features
        for i in range(feats[0].size()[0]):
            mlvl_feats = []
            # torch.cuda.empty_cache() 
            for level in range(levels):
                feat = feats[level][i:i + 1]
                mlvl_feats.append(feat.view(channel, -1))
            mlvl_feats_list.append(torch.cat(mlvl_feats, dim=-1))
        mlvl_feats = torch.stack(mlvl_feats_list, dim=0)
        print(f"fusion_layer/mlvl_feats: {mlvl_feats.size()}")
        return mlvl_feats


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
    
    
