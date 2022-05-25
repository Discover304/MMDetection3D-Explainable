# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp00-SECOND_ResNet_NoGraph_CatFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_60e.py', 
    '../_base_/default_runtime.py'
    ]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
    )