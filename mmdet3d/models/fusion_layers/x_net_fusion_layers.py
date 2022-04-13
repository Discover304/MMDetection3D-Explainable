# 多模态融合部分的代码
from ntpath import join

from pkg_resources import require
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from .notears.nonlinear import NotearsMLP, notears_nonlinear

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dense_to_sparse

from .. import builder
from ..builder import FUSION_LAYERS

# 信道编码
@FUSION_LAYERS.register_module()
class XnetFusion(BaseModule):  
    """_summary_

    Args:
        pre_fusion (_type_, optional): _description_. Defaults to None.
        get_graph (_type_, optional): _description_. Defaults to None.
        fusion (_type_, optional): _description_. Defaults to None.
        fusion_neck (_type_, optional): _description_. Defaults to None.
        init_cfg (_type_, optional): _description_. Defaults to None.
    
    Returns:
        torch.Tensor: fused feature
    """ 
          
    def __init__(self,
                pre_fusion=None,  
                get_graph=None,
                fusion=None,
                fusion_neck=None,
                init_cfg=None):
        super(XnetFusion, self).__init__(init_cfg=init_cfg)
        
        if pre_fusion:
            self.pre_fusion_layer = builder.build_fusion_layer(
                pre_fusion)
        if get_graph:
            self.get_graph_layer = builder.build_fusion_layer(
                get_graph)
        if fusion:
            self.fusion_layer = builder.build_fusion_layer(
                fusion)
        if fusion_neck:
            self.fusion_neck_layer = builder.build_fusion_layer(
                fusion_neck)

    @property
    def with_get_graph_layer(self):
        """bool: Whether the fusion layer need graph."""
        return hasattr(self,
                       'get_graph_layer') and self.get_graph_layer is not None

    @property
    def with_fusion_neck_layer(self):
        """bool: Whether the fusion layer need graph."""
        return hasattr(self,
                       'fusion_neck_layer') and self.fusion_neck_layer is not None
        
    def forward(self, img_feats, pts_feats):
        feats = self.pre_fusion_layer(img_feats, pts_feats)
        
        if self.with_get_graph_layer:
            adj_matrix = self.get_graph_layer(feats)
        else:
            adj_matrix = None
            
        fuse_out = self.fusion_layer(feats, adj_matrix)
        
        if self.with_fusion_neck_layer:
            fuse_out = self.fusion_neck_layer(fuse_out)
            
        return fuse_out


@FUSION_LAYERS.register_module()
class PreFusionCat(BaseModule):     
    """Concate features of pts and img
    Args:
        dropout_ratio (int, float, optional): Dropout ratio defaults to 0.2.
    
    Returns:
        tuple[torch.Tensor]: Corresponding features of image and points.
    """  
    def __init__(self,
                img_channels=512,
                img_out_channels=512,
                pts_channels=512,
                pts_out_channels=512,
                img_levels=5,
                pts_levels=1,
                dropout_ratio=0.2,
                init_cfg=None):
        super(PreFusionCat, self).__init__(init_cfg=init_cfg)
        self.img_levels = img_levels
        self.pts_levels = pts_levels
        
        self.img_transform = ConvModule(
                                in_channels=img_channels,
                                out_channels=img_out_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)
        
        self.pts_transform = ConvModule(
                                in_channels=pts_channels,
                                out_channels=pts_out_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)
        
        self.dropout_ratio = dropout_ratio


    def forward(self, img_feats, pts_feats):    
        N, C, H, W = pts_feats[0].size() 
           
        pts_feats = self.obtain_mlvl_feats(img_feats, self.img_levels, H, W)
        pts_pre_fuse = self.pts_transform(pts_feats)
        if self.dropout_ratio > 0:
            pts_pre_fuse = F.dropout(pts_pre_fuse, self.dropout_ratio, training=self.training)
        # print(f"fusion_layer/pts_pre_fuse: {pts_pre_fuse.size()}")
         
        img_feats = self.obtain_mlvl_feats(img_feats, self.img_levels, H, W)
        img_pre_fuse = self.img_transform(img_feats)
        if self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio, training=self.training)
        # print(f"fusion_layer/img_pre_fuse: {img_pre_fuse.size()}")
        
        concate_out = torch.cat((img_pre_fuse, pts_pre_fuse), dim=1)
        # print(f"fusion_layer/concate_out: {concate_out.size()}")

        return concate_out
    
    
    def obtain_mlvl_feats(self, feats, levels, target_H, target_W):
        """Obtain multi-level features for point features.

        Args:
            feats (list(torch.Tensor)): Multi-scale features in shape (N, C, H, W).
            levels (int): Number of FPN levels.

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
                if level>0:
                    feat = nn.Upsample(scale_factor=i+1, mode='nearest')(feat)
                N, C, H, W = feat.size()
                diff_H = target_H-H
                diff_W = target_W-W
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
class GetGraphPearson(BaseModule):
    """get adj matirx via pearson 

    Args:
        correlation_limit (float, optional): The threshold of edge values, Defaults to 0.8.
        
    Return:
        List[torch.Tensor]: if set correlation limit a value (less than 1) it will return a adj matrix with 0s and 1s, if set to none it will return a list of correlation matirx for each sample.
        
    """
    def __init__(self,
                correlation_limit=0.8,
                init_cfg=None):
        super(GetGraphPearson, self).__init__(init_cfg=init_cfg)
        self.correlation_limit=correlation_limit

    def forward(self, cat_feats):    
        correlations = []
        for sample in cat_feats:
            C,_,_ = sample.size()
            sample = torch.nan_to_num(sample, nan=0, posinf=0)
            # print(f"fusion_layer/sample: {sample}")
            correlation = torch.corrcoef(sample.view(C,-1))
            correlation = torch.nan_to_num(correlation, nan=0, posinf=0)
            if self.correlation_limit:
                correlation[torch.abs(correlation)<=self.correlation_limit] = 0
                correlation[correlation>self.correlation_limit] = 1
                correlation[correlation<-1*self.correlation_limit] = 1
            # print(f"fusion_layer/correlation: {correlation}")
            correlations.append(correlation)
        return correlations


@FUSION_LAYERS.register_module()
class GetGraphNoTear(BaseModule):
    """get adj matirx via no tear 

    Return:
        List[torch.Tensor]: if set correlation limit a value (less than 1) it will return a adj matrix with 0s and 1s, if set to none it will return a list of correlation matirx for each sample.
        
    """
    def __init__(self,
                init_cfg=None):
        super(GetGraphNoTear, self).__init__(init_cfg=init_cfg)

    def forward(self, cat_feats):    
        N, C, H, W = cat_feats.size()
        model = NotearsMLP(dims=[C, H*W, 1], bias=True)
        W_est = notears_nonlinear(model, cat_feats.view(N,C,-1,1), lambda1=0.01, lambda2=0.01)
        print(f"fusion_layer/notear_w: {W_est}")
        return W_est

@FUSION_LAYERS.register_module()
class FusionNN(BaseModule):
    """fuse input feature with no adj matrix

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                out_channels,
                init_cfg=None):
        super(FusionNN, self).__init__(init_cfg=init_cfg)

        self.fuse_conv = ConvModule(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)
        
    def forward(self, feats, adj_matrix):        
        fuse_out = self.fuse_conv(feats)
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")
        
        return fuse_out


@FUSION_LAYERS.register_module()
class FusionSummation(BaseModule):
    """fuse input feature by summarise most related feature channels.

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                init_cfg=None):
        super(FusionSummation, self).__init__(init_cfg=init_cfg)
        
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, feats, adj_matrix):  
        fuse_out = [] 
        for i in range(len(adj_matrix)):
            C,H,W = feats[i].size()
            sample = torch.mm(adj_matrix[i], feats[i].view(C,-1))
            fuse_out.append(sample.view(C,H,W))
        fuse_out = torch.stack(fuse_out, dim=0)  
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")  
           
        fuse_out=self.bn(fuse_out)
        return fuse_out


@FUSION_LAYERS.register_module()
class FusionGNN(BaseModule):
    """fuse input feature by GNN

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                out_channels,
                init_cfg=None):
        super(FusionGNN, self).__init__(init_cfg=init_cfg)
                
        self.fuse_gconv = GCNConv(
                                in_channels = in_channels,
                                out_channels = out_channels,
                                bias = True, 
                                normalize=True)
        
    def forward(self, feats, adj_matrix):  
        fuse_out = [] 
        for i in range(len(adj_matrix)):
            edge_index, edge_weights = dense_to_sparse(adj_matrix[i])
            # print(f"fusion_layer/edge_index: {edge_index}")
            
            C,H,W = feats[i].size()
            sample = self.fuse_gconv(feats[i].view(C,-1).T, edge_index, edge_weights)
            fuse_out.append(sample.T.view(C,H,W))
        fuse_out = torch.stack(fuse_out, dim=0)  
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")  
           
        return fuse_out


@FUSION_LAYERS.register_module()
class FusionNeck(BaseModule):
    """process the fused feature to desired shape of down stream process

    Args:
        in_channels (int): in channels, 
        out_channels (int): out channels
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels=512,
                out_channels=256,
                init_cfg=None):
        super(FusionNeck, self).__init__(init_cfg=init_cfg)

        self.fuse_conv = ConvModule(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=dict(type='Conv2d'),
                                norm_cfg=dict(type='BN2d'),
                                act_cfg=dict(type='ReLU'),
                                bias=True,
                                inplace=True)
        
    def forward(self, feats):        
        fuse_out = self.fuse_conv(feats)
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")
        
        return fuse_out

