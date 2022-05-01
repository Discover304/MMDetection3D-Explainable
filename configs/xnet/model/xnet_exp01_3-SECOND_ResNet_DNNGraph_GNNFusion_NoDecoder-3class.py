_base_ = [
    './xnet_exp01-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py'
    ]

model = dict( 
    fusion_neck=dict(
        in_channels=4*(64+64),
        out_channels=512),
    
    pts_bbox_head=dict(
        feat_channels=512)   
)

