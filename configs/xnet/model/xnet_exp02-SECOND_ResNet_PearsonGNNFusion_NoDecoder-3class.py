_base_ = [
    './xnet_exp00-SECOND_ResNet_CatFusion_NoDecoder-3class.py'
    ]

model = dict(
    # 特征级别融合网络 Feature Fusion
    fusion_layer=dict(
        pre_fusion=dict(
            img_out_channels=32,
            pts_out_channels=32),
        get_graph=dict(
            _delete_=True,
            type='GetGraphPearson',
            correlation_limit=None),
        fusion=dict(
            _delete_=True,
            type='FusionGNN',
            in_channels=32+32,
            out_channels=64),
        fusion_neck=dict(
            in_channels=64)),
)