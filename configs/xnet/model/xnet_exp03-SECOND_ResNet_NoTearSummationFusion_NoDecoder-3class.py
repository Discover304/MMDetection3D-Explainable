_base_ = [
    './xnet_exp02-SECOND_ResNet_PearsonGNNFusion_NoDecoder-3class.py'
    ]

model = dict(
    # 特征级别融合网络 Feature Fusion
    fusion_layer=dict(
        get_graph=dict(
            _delete_=True,
            type='GetGraphNoTear'))
)
