# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp04-SECOND_ResNet_DAGraph_MarkovGNNFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
    ]

# 粒度20
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

load_from = "work_dirs/xnet_exp01_4-kitti_3d_3class-cyclic_20e/epoch_2.pth"
