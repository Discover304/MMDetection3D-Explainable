_base_ = [
    './xnet_exp01-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py'
    ]

model = dict(
    get_graph=dict(
        _delete_=True,
        type='GetGraphNoTearLinear')
)
