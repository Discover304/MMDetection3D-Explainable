# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp05-SECOND_ResNet_PearsonGraph_GNNFusion_NoDecoder_EntropyLoss-3class copy.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
    ]


# 对pearson方法debug之后，再次尝试
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3
    )

load_from = "work_dirs/xnet_exp02-kitti_3d_3class-cyclic_20e/epoch_11.pth"