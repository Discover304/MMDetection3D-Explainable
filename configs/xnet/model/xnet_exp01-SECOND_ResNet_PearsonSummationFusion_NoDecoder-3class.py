_base_ = [
    './xnet_exp00-SECOND_ResNet_CatFusion_NoDecoder-3class.py'
    ]

model = dict(
    # 特征级别融合网络 Feature Fusion
    fusion_layer=dict(
        get_graph=dict(
            type='GetGraphPearson',
            correlation_limit=0.8),
        fusion=dict(
            type='FusionSummation',
            in_channels=512+512
        ))
)