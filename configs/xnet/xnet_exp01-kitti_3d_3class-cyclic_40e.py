# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp01-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_40e.py',
    '../_base_/default_runtime.py'
    ]

# load_from = "work_dirs/storage/xnet_base_baseline-kitti_3d_3class-cyclic_40e/epoch_60.pth"