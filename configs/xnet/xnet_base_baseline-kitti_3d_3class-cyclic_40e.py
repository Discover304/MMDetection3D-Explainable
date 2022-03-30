# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_base-SECOND_ResNet_CatFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_40e.py', 
    '../_base_/default_runtime.py'
    ]