_base_ = [
    './xnet_exp00-SECOND_ResNet_CatFusion_NoDecoder-3class.py'
    ]

model = dict(
    # 特征级别融合网络 Feature Fusion
    pre_fusion=dict(
        img_out_channels=64,
        pts_out_channels=64),
    # get_graph=dict(
    #     _delete_=True,
    #     type='GetGraphRandom'),
    get_graph=dict(
        _delete_=True,
        type='GetGraphPearson'),        
    # fusion=dict(
    #     _delete_=True,
    #     type='FusionSummation',
    #     in_channels=64+64,
    #     out_channels=(64+64)*2),
    # fusion_neck=dict(
    #     in_channels=(64+64)*2)
    fusion=dict(
        _delete_=True,
        type='FusionGNN',
        in_channels=64+64,
        out_channels=64+64),
    fusion_neck=dict(
        in_channels=64+64)
)