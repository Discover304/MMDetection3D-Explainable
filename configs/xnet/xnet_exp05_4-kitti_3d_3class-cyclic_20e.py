# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp05-SECOND_ResNet_PearsonGraph_GNNFusion_NoDecoder_EntropyLoss-3class copy.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py'
    ]


# entropy further test - debug
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3
    )

# load_from = "work_dirs/xnet_exp05_1-kitti_3d_3class-cyclic_20e/epoch_20.pth"
# load_from = "work_dirs/xnet_exp05_4-kitti_3d_3class-cyclic_20e/epoch_3.pth"
load_from = "work_dirs/0504_xnet_exp05_4-kitti_3d_3class-cyclic_20e/epoch_20.pth"