_base_ = [
    './xnet_exp01-SECOND_ResNet_PearsonSummationFusion_NoDecoder-3class.py'
    ]

model = dict(
    # 特征级别融合网络 Feature Fusion
    fusion_layer=dict(
        get_graph=dict(
            type='GetGraphPearson',
            correlation_limit=None),
        fusion=dict(
            type='FusionGNN',
            in_channels=512+512
        ))
)