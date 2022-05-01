# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp01_4-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
    ]

# 测试使用relu后的邻接矩阵，模拟dag图

load_from = "work_dirs/xnet_exp00-kitti_3d_3class-cyclic_40e/epoch_4.pth"
# 保留图神经网络之前的输入特征通道，进行堆叠

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6
    )

# runner = dict(
#     max_epochs=10
#     )
