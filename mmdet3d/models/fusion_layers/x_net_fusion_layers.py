from ntpath import join
from xml.etree.ElementInclude import include

from pkg_resources import require
# from mmdet3d.models.fusion_layers.notears.nonlinear import notears_nonlinear, NotearsMLP
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision

from mmcv.cnn import ConvModule
from .notears.nonlinear_gpu_ver import notears_nonlinear
# from .notears.linear import notears_linear
from ..builder import FUSION_LAYERS
from .notears.utils import set_random_seed, simulate_dag


"""Pre Fusion"""
@FUSION_LAYERS.register_module()
class PreFusionCat(nn.Module):     
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
                dropout_ratio=0.2):
        super(PreFusionCat, self).__init__()
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
        _, _, H, W = pts_feats[0].size() 
           
        pts_feats = self.obtain_mlvl_feats(pts_feats, self.pts_levels, H, W)
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
                _, _, H, W = feat.size()
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


"""Get Graph"""
@FUSION_LAYERS.register_module()
class GetGraphRandom(nn.Module):
    """get adj matirx in random

    Return:
        List[torch.Tensor]: if set correlation limit a value (less than 1) it will return a adj matrix with 0s and 1s, if set to none it will return a list of correlation matirx for each sample.
        
    """
    def __init__(self,
                vertex_ratio = 0.3):
        super(GetGraphRandom, self).__init__()
        dropout_ratio = 1-vertex_ratio
        self.dropout = nn.Dropout2d(p=dropout_ratio)
        

    def forward(self, cat_feats):    
        N, C, _, _ = cat_feats.size()    
        random_adj = self.dropout(torch.ones(C,C, device=cat_feats.get_device()))
        # print(f"fusion_layer/random_adj: {random_adj}")

        random_adjs = random_adj.repeat(N,1,1)
        return random_adjs


@FUSION_LAYERS.register_module()
class GetGraphNN(nn.Module):
    """get adj matirx via NN 

    Return:
        List[torch.Tensor]: if set correlation limit a value (less than 1) it will return a adj matrix with 0s and 1s, if set to none it will return a list of correlation matirx for each sample.
        
    """
    def __init__(self,
                in_channel):
        super(GetGraphNN, self).__init__()
        
        resnet18 = torchvision.models.resnet18()
        resnet18.conv1= nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,bias=False)
        
        self.model = nn.Sequential(
                resnet18,
                nn.Linear(1000, in_channel**2),
                nn.ReLU(inplace=True)
            )
        
        # # 如果不训练，那就加上这一部分。
        # for param in self.parameters():
        #     param.requires_grad = False
        
    def forward(self, cat_feats):
        N, C, _, _ = cat_feats.size()
        W = self.model(cat_feats).view(N,C,C)
        mask = torch.ones(C,C, device = W.device) - torch.eye(C, device = W.device)
        W = torch.where(mask==1, W, mask)
        # print(W, W.size(), torch.count_nonzero(W))
        return W


@FUSION_LAYERS.register_module()
class GetGraphPearson(nn.Module):
    """get adj matirx via pearson 

    Args:
        correlation_limit (float, optional): The threshold of edge values, Defaults to 0.8.
        
    Return:
        List[torch.Tensor]: if set correlation limit a value (less than 1) it will return a adj matrix with 0s and 1s, if set to none it will return a list of correlation matirx for each sample.
        
    """
    def __init__(self,
                 in_channel):
        super(GetGraphPearson, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cat_feats):   
        N, C, H, W = cat_feats.size()
        cat_feats = cat_feats.view(N*C,1,H,W)
        feats = self.model(cat_feats).view(N,C,1000)
        mask = torch.ones(C,C, device = feats.device) - torch.eye(C, device = feats.device)

        W_est = [] 
        for feat in feats:
            # print(f"fusion_layer/feat: {feat}")
            W = torch.corrcoef(feat)
            # print(f"fusion_layer/correlation: {W}")
            # 确保相关性矩阵没有问题。
            W = torch.where(mask==1, W, mask)
            W = self.relu(W)
            # print(W, W.size(), torch.count_nonzero(W))
            W_est.append(W)
        return W_est


@FUSION_LAYERS.register_module()
class GetGraphDAG(nn.Module):
    """get adj matirx via no tear 

    Return:
        List[torch.Tensor]: if set correlation limit a value (less than 1) it will return a adj matrix with 0s and 1s, if set to none it will return a list of correlation matirx for each sample.
        
    """
    def __init__(self, in_channel):
        super(GetGraphDAG, self).__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.linear = nn.Linear(1000, in_channel)
        # self.W = simulate_dag(in_channel, (in_channel*in_channel)//4, 'ER')
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, cat_feats):   
        N, C, H, W = cat_feats.size()
        cat_feats = cat_feats.view(N*C,1,H,W)
        feats = self.model(cat_feats).view(N,C,1000)
        # out = self.linear(feats.view(N*C,1000)).view(N,C,C)
        # mask = torch.ones(C,C, device = feats.device) - torch.eye(C, device = feats.device)

        W_est = [] 
        device = cat_feats.device
        for i in range(N):
            # print(f"fusion_layer/feat: {feat}")
            # W = self.W
            # W = notears_nonlinear(NotearsMLP(dims=[C, 20, 1], bias=True), feats[i].T.cpu(), lambda1=0.01, lambda2=0.01)
            # print("input feats: ",feats[i], feats[i].size(), torch.count_nonzero(feats[i]))
            W = notears_nonlinear(feats[i].T.cpu().detach().numpy(), feats[i].T, use_gpu=device, max_iter=100)
            # W = notears_linear(feats[i].T.cpu().detach().numpy(), lambda1=0.1, loss_type='logistic')
            W = torch.tensor(W, device=device).float()
            # print(f"fusion_layer/correlation: {W}")
            # 确保相关性矩阵没有问题。
            # W = torch.where(mask==1, W, mask)
            # W = self.relu(W)
            # print("output W: ", W, W.size(), torch.count_nonzero(W))
            # W = torch.where(W>(1000/(C*C)), out[i], torch.zeros(C,C,device=device))
            W_est.append(W)
            # torch.where(W>0.8, out[i], torch.zeros(C,C))
        return W_est


"""Fusion"""
@FUSION_LAYERS.register_module()
class FusionNN(nn.Module):
    """fuse input feature with no adj matrix

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                out_channels):
        super(FusionNN, self).__init__()

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
        return self.fuse_conv(feats)


@FUSION_LAYERS.register_module()
class FusionSummation(nn.Module):
    """fuse input feature by summarise most related feature channels.

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                out_channels):
        super(FusionSummation, self).__init__()
        
    def forward(self, feats, adj_matrix):  
        N,C,H,W = feats.size()
        fuse_out = [] 
        for i in range(len(adj_matrix)):
            sample = torch.mm(adj_matrix[i] + torch.eye(C, device=adj_matrix.device), feats[i].view(C,-1))/(torch.sum(adj_matrix[i])/C)
            fuse_out.append(torch.cat((feats[i], sample.view(C,H,W)), dim=0))
        fuse_out = torch.stack(fuse_out, dim=0)
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")  

        return fuse_out


@FUSION_LAYERS.register_module()
class FusionGNN(nn.Module):
    """fuse input feature by GNN

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                out_channels):
        super(FusionGNN, self).__init__()

        self.layers=nn.ModuleList([
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU'),
                bias=True,
                inplace=True),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU'),
                bias=True,
                inplace=True),
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU'),
                bias=True,
                inplace=True)])

    def forward(self, feats, adj_matrix):  
        N,C,H,W = feats.size()
        fuse_out = []
        for layer in self.layers:
            feats = layer(feats)
            layer_out = []
            for i in range(len(adj_matrix)):
                m = adj_matrix[i]
                x = torch.sparse.mm(m.to_sparse(), feats[i].view(C,-1))
                layer_out.append(x.view(C,H,W))
            feats = torch.stack(layer_out, dim=0)  
            fuse_out.append(feats)
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")  
           
        return fuse_out


@FUSION_LAYERS.register_module()
class FusionMarkovGNN(nn.Module):
    """solve each graph with fuse input feature by deep GNN

    Args:
        in_channels (int): in channels, 
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels,
                out_channels,
                slice = 20):
        super(FusionMarkovGNN, self).__init__()
        self.slice = slice
        self.layers=nn.ModuleList([
            ConvModule(
                in_channels=in_channels*slice,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU'),
                bias=True,
                inplace=True),
            ConvModule(
                in_channels=in_channels*slice,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU'),
                bias=True,
                inplace=True),
            ConvModule(
                in_channels=in_channels*slice,
                out_channels=in_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU'),
                bias=True,
                inplace=True)])

    def get_markov_sub(self, m, slice = 20):
        C,_  = m.size()
        # print([m, m.size(), torch.count_nonzero(m)])
        # i = torch.eye(C, device = m.device)
        # m = m+i
        ms = []
        for i in range(C):
            ms.append(torch.where(m[i]>0, m, torch.zeros(C,C,device = m.device)))
        mss = []
        slice_size = C//slice
        for i in range(slice):
            mss.append(torch.max(torch.stack(ms[slice_size*i:slice_size*(i+1)]), dim = 0).values)
        # print([[m, m.size(), torch.count_nonzero(m)] for m in mss])
        return torch.stack(mss)
                
    def forward(self, feats, adj_matrix):  
        N,C,H,W = feats.size()
        # C, C = adj_matrix.size()
        S = self.slice
        mss = []
        for i in range(N):
            m = adj_matrix[i]
            ms = self.get_markov_sub(m, slice = S)
            mss.append(ms)
        mss = torch.stack(mss) # (N,S,C,C)
           
        fuse_out = []
        for layer in self.layers:
            layer_out = []
            for i in range(len(adj_matrix)):
                ms = mss[i].view(S*C,C) # (S*C,C)
                feat = feats[i].view(C,H*W)
                x = torch.sparse.mm(ms.to_sparse(), feat) # (S*C,H*W)
                # x = torch.max(x, dim=0).values
                layer_out.append(x.view(S,C,H,W))
            feats = torch.stack(layer_out, dim=0) # (N,S,C,H,W)
            feats = layer(feats.view(N,S*C,H,W)) # (N,C,H,W)
            fuse_out.append(feats)
        
        
        # mss = [] 
        # for i in range(N):
        #     m = adj_matrix[i]
        #     ms = self.get_markov_sub(m, slice = S)
        #     mss.append(ms)
        # mss = torch.stack(mss) # (N,S,C,C)
           
        # fuse_out = []
        # for layer in self.layers:
        #     layer_out = []
        #     for i in range(len(adj_matrix)):
        #         ms = mss[i].view(S*C,C) # (S*C,C)
        #         feat = feats[i].view(C,H*W)
        #         x = torch.sparse.mm(ms.to_sparse(), feat) # (S*C,H*W)
        #         # x = torch.max(x, dim=0).values
        #         layer_out.append(x.view(S,C,H,W))
        #     feats = torch.stack(layer_out, dim=0) # (N,S,C,H,W)
        #     feats = layer(feats.view(N,S*C,H,W)) # (N,C,H,W)
        #     fuse_out.append(feats)
        # print(f"fusion_layer/fuse_out: {fuse_out.size()}")  
           
        return fuse_out


"""Fusion Neck"""
@FUSION_LAYERS.register_module()
class FusionNeckNN(nn.Module):
    """process the fused feature to desired shape of down stream process

    Args:
        in_channels (int): in channels, 
        out_channels (int): out channels
        
    Return:
        torch.Tensor: features.
    """
    def __init__(self,
                in_channels=512,
                out_channels=256):
        super(FusionNeckNN, self).__init__()

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
        return self.fuse_conv(feats)

