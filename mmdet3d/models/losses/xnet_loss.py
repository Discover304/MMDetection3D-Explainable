# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist
# ————————————————
# 版权声明：本文为CSDN博主「我想静静，」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_42764932/article/details/112998284

def PCA_svd(X, k, center=True):
    device = X.device
    n = X.size()[0]
    ones = torch.ones(n, device=device).view(n,1)
    h = ((1/n) * torch.sparse.mm(ones.to_sparse(), ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n, device=device) - h
    X_center =  torch.sparse.mm(H.double().to_sparse(), X.double())
    u, s, v = torch.svd(X_center)
    # print(f"xnet_loss/PCA_svd/k: {k}")
    # print(f"xnet_loss/PCA_svd/v: {v.size()}")
    components  = v[:k].t()
    # print(f"xnet_loss/PCA_svd/components: {components.size()}")
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components
# ————————————————
# 版权声明：本文为CSDN博主「一点也不可爱的王同学」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_43844219/article/details/105188844

def calculate_entropy(feats, k=1):
    _,C,H,W = feats.size()
    k = C//10 # 采用最合适的 k 值
    feats = feats.view(-1,C,H*W)
    H_mean = 0
    
    # dga = torch.special.digamma
    log = torch.log   
        
    N = C
    D = H*W
    
    # c_D = torch.pi**(D/2)/torch.e**torch.lgamma(1+D/2)
    # c_D = (torch.pi/torch.e)**(D/2)*torch.e**(D/2-torch.lgamma(1+D/2))
    # c_D = 0
        
    for feat in feats:  # (C,H*W)
        dist = euclidean_dist(feat, feat) # (C, C)
        order = torch.argsort(dist, dim=1)
        
        H = 0
        for n in range(N):
            # ball V
            r_ball = dist[n][order[n][k]]
            
            # # volume correction
            # correction = 1
            # pca_feat = PCA_svd(torch.stack([feat[order[n][i]] for i in range(k+1)]), D)
            # centre = torch.mean(pca_feat, dim=0) # 中心坐标 (1, C)
            # centre_dist = pca_feat-centre # (C, H*W)
            # rs = torch.max(centre_dist, dim=0).values # PCA主成分分析后得到的主轴半径列表。
            # rs, _ = torch.sort(rs, descending =True)
            # for d in range(10): # suppose to be D
            #     correction*=rs[d]/rs[0]
            
            # reference volume
            # 暂时不加
        
            # H_ec_knn = dga(torch.tensor(N)) - dga(torch.tensor(k)) + (log(c_D) + D*log(r_ball)) +log(correction)
            # H_ec_knn = log(torch.pi**(D/2)/torch.e**torch.lgamma(1+D/2)*r_ball**D*correction)
            # H_ec_knn = log(r_ball) - log(correction)
            # print(f"xnet_loss/calculate_entropy/r_ball: {r_ball}")
            # print(f"xnet_loss/calculate_entropy/log(correction+1): {log(correction+1)}")
            H_ec_knn = r_ball
            H += H_ec_knn
        H_mean += H
    H_mean = torch.mean(H_mean)
    return log(H_mean+1)

@weighted_loss
def layer_entropy_loss(value, target):
    return value

@LOSSES.register_module()
class EntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.target = 0

    def forward(self, net_info):
        delta_entropy = []
        for i in range(1, len(net_info)//2-1):
            # print("--------------------block01------------------------------------")
            # print(f"xnet_loss/calculate_entropy/delta: {calculate_entropy(net_info[i+1])-calculate_entropy(net_info[i])}")
            delta_entropy.append(calculate_entropy(net_info[i+1])-calculate_entropy(net_info[i]))
        delta_entropy = torch.stack(delta_entropy)
        var = torch.var(delta_entropy)
        # mean = -torch.mean(torch.sqrt(delta_entropy**2))
        
        delta_entropy = []
        for i in range(len(net_info)//2-1+1):
            # print("--------------------block02-------------------------------------")
            delta_entropy.append(calculate_entropy(net_info[len(net_info)//2-1+i+1])-calculate_entropy(net_info[len(net_info)//2-1+i]))
        delta_entropy = torch.stack(delta_entropy)
        var += torch.var(delta_entropy)
        # mean -= torch.mean(torch.sqrt(delta_entropy**2))
        
        # net_loss_mean = self.loss_weight * layer_entropy_loss(mean, self.target)
        net_loss_var = self.loss_weight * layer_entropy_loss(var, self.target)
        return {"loss_net_var": [net_loss_var]} # "loss_net_mean": [net_loss_mean], 
