# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp02-SECOND_ResNet_PearsonGraph_GNNFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
    ]


# 对pearson方法debug之后，再次尝试
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
    )

load_from = "work_dirs/xnet_exp02-kitti_3d_3class-cyclic_20e/epoch_11.pth"