# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp02-SECOND_ResNet_PearsonGraph_GNNFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
    ]

# 因为代码存在bug，返回的是除去对角线的全一矩阵，所以这个实验可以作为特征金字塔的验证实验。

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6
    )

load_from = "work_dirs/xnet_exp00-kitti_3d_3class-cyclic_40e/epoch_4.pth"