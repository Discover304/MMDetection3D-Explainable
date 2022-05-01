_base_ = [
    './xnet_exp02-SECOND_ResNet_PearsonGraph_GNNFusion_NoDecoder-3class.py'
    ]

model = dict(
    pts_backbone=dict(
        type='SECOND_INFO',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    
    net_loss=dict(
        _delete_=True,
        type = 'EntropyLoss')
)