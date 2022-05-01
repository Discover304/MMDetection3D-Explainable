# 需要修改的配置文件
_base_ = [
    './datasets/kitti_3d_3class-multimodal.py',
    './model/xnet_exp01-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
    ]

# 固定图预测部分网络，再进行学习
load_from = "work_dirs/xnet_exp01-kitti_3d_3class-cyclic_40e/epoch_17.pth"

# 使用
# 如果不训练，那就加上这一部分。
# for param in self.parameters():
#     param.requires_grad = False