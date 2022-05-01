# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp00-SECOND_ResNet_NoGraph_CatFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_40e.py', 
    '../_base_/default_runtime.py'
    ]

load_from = "work_dirs/xnet_base-kitti_3d_3class-cyclic_40e/epoch_40.pth"
