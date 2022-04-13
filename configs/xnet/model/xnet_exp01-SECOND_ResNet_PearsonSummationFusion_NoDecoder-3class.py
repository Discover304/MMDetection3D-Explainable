_base_ = [
    './xnet_exp00-SECOND_ResNet_CatFusion_NoDecoder-3class.py'
    ]

model = dict(
    # 特征级别融合网络 Feature Fusion
    fusion_layer=dict(
        pre_fusion=dict(
            img_out_channels=128,
            pts_out_channels=128),
        get_graph=dict(
            _delete_=True,
            type='GetGraphPearson',
            correlation_limit=0.8),
        fusion=dict(
            _delete_=True,
            type='FusionSummation',
            in_channels=128+128),
        fusion_neck=dict(
            in_channels=128+128)),
)
