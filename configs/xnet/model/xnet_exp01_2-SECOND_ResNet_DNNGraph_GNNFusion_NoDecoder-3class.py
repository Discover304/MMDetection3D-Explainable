_base_ = [
    './xnet_exp01-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py'
    ]

model = dict( 
    fusion_neck=dict(
        in_channels=2*(64+64))
)