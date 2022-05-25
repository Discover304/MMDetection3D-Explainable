# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_base_1-SECOND_NoImg_NoGraph_NoFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_40e.py', 
    '../_base_/default_runtime.py'
    ]

# load_from = "work_dirs/xnet_exp00_1-kitti_3d_3class-cyclic_20e/epoch_10.pth"
# load_from = "work_dirs/xnet_base-kitti_3d_3class-cyclic_40e/epoch_10.pth"
load_from = "work_dirs/0504_xnet_base_1-kitti_3d_3class-cyclic_20e/epoch_20.pth"
