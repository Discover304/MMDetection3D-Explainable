# 多模态融合部分的代码
from ntpath import join
import torch
from torch import nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn import functional as F



from ..builder import FUSION_LAYERS

# 信道编码
@FUSION_LAYERS.register_module()
class CatFusion(BaseModule):
    """Concate features of pts and img

    Args:
        img_channels_list (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels_list (list[int] | int): Channels of point features.
            It could be a list if the input is multi-scale image features.
        out_channels_list (int): Channels of output fused features
        dropout_ratio (int, float, optional): Dropout ratio defaults to 0.2.
    
    Returns:
        tuple[torch.Tensor]: Corresponding features of image and points.
    """
        
    def __init__(self,
                img_channels=256,
                img_levels=5,
                pts_channels=256,
                pts_levels=1,
                hide_channels=512,
                out_channels=256,
                dropout_ratio=0.2,
                init_cfg=None):
        super(CatFusion, self).__init__(init_cfg=init_cfg)
        self.img_levels = img_levels
        self.pts_levels = pts_levels
        self.img_channels = img_channels
        self.img_channels_list = [img_channels for _ in range(img_levels)]
        self.pts_channels = pts_channels
        self.pts_channels_list = [pts_channels for _ in range(pts_levels)]
        self.dropout_ratio = dropout_ratio
        self.hide_channels = hide_channels
        self.out_channels = out_channels
        
        self.img_transform = ConvModule(
                                in_channels=img_channels,
                                out_channels=hide_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv1d'),
                                norm_cfg=dict(type='BN1d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)
        
        self.pts_transform = ConvModule(
                                in_channels=pts_channels,
                                out_channels=hide_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv1d'),
                                norm_cfg=dict(type='BN1d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)

        self.fuse_conv = ConvModule(
                                in_channels=hide_channels+hide_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv1d'),
                                norm_cfg=dict(type='BN1d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)


    def forward(self, img_feats, pts_feats):    
        N, C, H, W = pts_feats[0].size() 
           
        pts_pre_fuse = self.pts_transform(pts_feats[0].view(N, C, -1))
        if self.training and self.dropout_ratio > 0:
            pts_pre_fuse = F.dropout(pts_pre_fuse, self.dropout_ratio)
        # print(f"fusion_layer/pts_pre_fuse: {pts_pre_fuse.size()}")
         
        img_feats = self.obtain_img_mlvl_feats_of_pts(img_feats, self.img_levels, H, W)
        img_pre_fuse = self.img_transform(img_feats.view(N, C, -1))
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        # print(f"fusion_layer/img_pre_fuse: {img_pre_fuse.size()}")
        
        concate_out = torch.cat((img_pre_fuse, pts_pre_fuse), dim=1)
        # print(f"fusion_layer/concate_out: {concate_out.size()}")
        
        fuse_out = self.fuse_conv(concate_out).view(N, self.out_channels, H, W)
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")
        
        return fuse_out
    
    
    def obtain_img_mlvl_feats_of_pts(self, feats, levels, pts_H, pts_W):
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
            mlvl_feat = None
            # torch.cuda.empty_cache() 
            for level in range(levels):
                feat = feats[level][i:i + 1]
                # print(f"fusion_layer/img_fpn_feat: {feat.size()}")
                # sum and pad to pts structure
                N, C, H, W = feat.size()
                diff_H = pts_H-H
                diff_W = pts_W-W
                if mlvl_feat!=None:
                    mlvl_feat += F.pad(feat, (diff_W//2, diff_W-diff_W//2, diff_H//2, diff_H-diff_H//2), 
                                        mode="replicate")
                else:
                    mlvl_feat = F.pad(feat, (diff_W//2, diff_W-diff_W//2, diff_H//2, diff_H-diff_H//2), 
                                        mode="replicate")
            mlvl_feats_list.append(mlvl_feat)
        mlvl_feats = torch.cat(mlvl_feats_list, dim=0)
        # print(f"fusion_layer/mlvl_feats: {mlvl_feats.size()}")
        return mlvl_feats


@FUSION_LAYERS.register_module()
class XNetFusion(BaseModule):
    def __init__(self,
                 img_channels_list=None,
                 pts_channels_list=None,
                 out_channels_list=None,
                 init_cfg=None):
        super(XNetFusion, self).__init__(init_cfg=init_cfg)
        self.linear = torch.nn.Linear(img_channels_list + pts_channels_list, out_channels_list)

    def forward(self, img_feats, pts_feats):
        # x = torch.cat((pts_feats, img_feats))
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        return img_feats, pts_feats
    
    
