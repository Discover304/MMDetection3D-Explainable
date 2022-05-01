_base_ = [
    # './xnet_exp01-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py'
    # './xnet_exp02-SECOND_ResNet_PearsonGraph_GNNFusion_NoDecoder-3class.py'
    './xnet_exp03-SECOND_ResNet_DAGraph_GNNFusion_NoDecoder-3class.py'
    ]

model = dict( 
    fusion=dict(
        _delete_=True,
        type='FusionMarkovGNN',
        in_channels=64+64,
        out_channels=4*(64+64),
        slice = 2),
    
    fusion_neck=dict(
        in_channels=4*(64+64),
        out_channels=512),
    
    pts_bbox_head=dict(
        feat_channels=512)  
)
